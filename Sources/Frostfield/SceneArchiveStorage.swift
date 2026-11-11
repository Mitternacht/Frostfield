import CryptoKit
import Foundation
import Metal
import simd

/// A byte range of the archive's binary sidecar.
struct ArchiveView: Codable, Equatable {
    var offset: Int
    var length: Int
}

/// Appends 16-byte-aligned, little-endian views to the sidecar and shares identical bytes.
///
/// Views are found by a SHA-256 digest and confirmed by comparing against the bytes
/// already in the sidecar, so each block is held in memory once.
struct ArchiveBinaryWriter {
    private(set) var data = Data()
    private(set) var views: [ArchiveView] = []
    private var existing: [SHA256.Digest: [Int]] = [:]

    mutating func append(_ bytes: Data) -> Int {
        let digest = SHA256.hash(data: bytes)
        for index in existing[digest] ?? [] {
            let view = views[index]
            if view.length == bytes.count, data[view.offset..<(view.offset + view.length)] == bytes { return index }
        }
        let padding = (16 - data.count % 16) % 16
        data.append(Data(count: padding))
        views.append(ArchiveView(offset: data.count, length: bytes.count))
        data.append(bytes)
        existing[digest, default: []].append(views.count - 1)
        return views.count - 1
    }

    mutating func append<T>(_ values: [T]) -> Int {
        append(values.withUnsafeBytes { Data($0) })
    }
}

/// Reads typed arrays from validated views of the sidecar.
struct ArchiveBinaryReader {
    let data: Data
    let views: [ArchiveView]

    init(data: Data, views: [ArchiveView]) throws {
        for (index, view) in views.enumerated() {
            guard view.offset >= 0, view.length >= 0, view.offset <= data.count, view.length <= data.count - view.offset else {
                throw SceneArchive.Failure.invalidView(index)
            }
        }
        self.data = data
        self.views = views
    }

    func bytes(_ index: Int) throws -> Data {
        guard views.indices.contains(index) else { throw SceneArchive.Failure.invalidView(index) }
        let view = views[index]
        return data.subdata(in: (data.startIndex + view.offset)..<(data.startIndex + view.offset + view.length))
    }

    /// `count × factor` for a manifest count, throwing `invalidView` when a count is
    /// negative or the product overflows instead of trapping.
    func product(_ count: Int, _ factor: Int, view: Int) throws -> Int {
        let (result, overflow) = count.multipliedReportingOverflow(by: factor)
        guard count >= 0, factor >= 0, !overflow else { throw SceneArchive.Failure.invalidView(view) }
        return result
    }

    /// Reads `count` elements of `T`, which must be exactly the view's length.
    func array<T>(_ index: Int, of _: T.Type, count: Int) throws -> [T] {
        let bytes = try self.bytes(index)
        guard bytes.count == (try product(count, MemoryLayout<T>.stride, view: index)) else {
            throw SceneArchive.Failure.invalidView(index)
        }
        return [T](unsafeUninitializedCapacity: count) { buffer, initialized in
            bytes.withUnsafeBytes { raw in
                let target = UnsafeMutableRawBufferPointer(buffer)
                if !raw.isEmpty { target.copyMemory(from: raw) }
            }
            initialized = count
        }
    }

    /// Reads a view whose length must be a whole number of `T`.
    func array<T>(_ index: Int, of type: T.Type) throws -> [T] {
        let length = try bytes(index).count
        guard length % MemoryLayout<T>.stride == 0 else { throw SceneArchive.Failure.invalidView(index) }
        return try array(index, of: type, count: length / MemoryLayout<T>.stride)
    }
}

/// One texture's descriptor and the view holding every level and slice, tightly packed.
struct ArchiveTextureRecord: Codable, Equatable {
    var label: String?
    var pixelFormat: UInt
    var textureType: UInt
    var width: Int
    var height: Int
    var depth: Int
    var arrayLength: Int
    var mipmapLevelCount: Int
    var usage: UInt
    var storageMode: UInt
    var view: Int
}

/// Reads and recreates textures with their exact stored bits.
///
/// Levels are stored in order, each holding every slice (cube faces are slices),
/// each slice holding its rows of pixels or compressed blocks with no padding.
enum ArchiveTextureCodec {
    struct Layout {
        let blockWidth: Int
        let blockHeight: Int
        let bytesPerBlock: Int
        var family: KTX2Format.Family = .uncompressed

        func bytesPerRow(width: Int) -> Int { (width + blockWidth - 1) / blockWidth * bytesPerBlock }
        func bytesPerImage(width: Int, height: Int) -> Int { bytesPerRow(width: width) * ((height + blockHeight - 1) / blockHeight) }
    }

    static func layout(_ format: MTLPixelFormat) -> Layout? {
        if let size = uncompressedSizes[format] { return Layout(blockWidth: 1, blockHeight: 1, bytesPerBlock: size) }
        return compressedLayouts[format]
    }

    private static let uncompressedSizes: [MTLPixelFormat: Int] = {
        var sizes: [MTLPixelFormat: Int] = [:]
        for format: MTLPixelFormat in [.a8Unorm, .r8Unorm, .r8Unorm_srgb, .r8Snorm, .r8Uint, .r8Sint] { sizes[format] = 1 }
        for format: MTLPixelFormat in [
            .r16Unorm, .r16Snorm, .r16Uint, .r16Sint, .r16Float, .rg8Unorm, .rg8Unorm_srgb, .rg8Snorm, .rg8Uint, .rg8Sint,
            .b5g6r5Unorm, .a1bgr5Unorm, .abgr4Unorm, .bgr5A1Unorm,
        ] { sizes[format] = 2 }
        for format: MTLPixelFormat in [
            .r32Uint, .r32Sint, .r32Float, .rg16Unorm, .rg16Snorm, .rg16Uint, .rg16Sint, .rg16Float,
            .rgba8Unorm, .rgba8Unorm_srgb, .rgba8Snorm, .rgba8Uint, .rgba8Sint, .bgra8Unorm, .bgra8Unorm_srgb,
            .rgb10a2Unorm, .rgb10a2Uint, .rg11b10Float, .rgb9e5Float, .bgr10a2Unorm,
        ] { sizes[format] = 4 }
        for format: MTLPixelFormat in [
            .rg32Uint, .rg32Sint, .rg32Float, .rgba16Unorm, .rgba16Snorm, .rgba16Uint, .rgba16Sint, .rgba16Float,
        ] { sizes[format] = 8 }
        for format: MTLPixelFormat in [.rgba32Uint, .rgba32Sint, .rgba32Float] { sizes[format] = 16 }
        return sizes
    }()

    private static let compressedLayouts: [MTLPixelFormat: Layout] = {
        var layouts: [MTLPixelFormat: Layout] = [:]
        for vkFormat in UInt32(133)...184 {
            guard let format = KTX2Format(vkFormat: vkFormat), format.isCompressed else { continue }
            let layout = Layout(
                blockWidth: format.blockWidth, blockHeight: format.blockHeight, bytesPerBlock: format.bytesPerBlock,
                family: format.family
            )
            layouts[format.linear] = layout
            if let sRGB = format.sRGB { layouts[sRGB] = layout }
        }
        let bc8 = Layout(blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, family: .bc)
        let bc16 = Layout(blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, family: .bc)
        for format: MTLPixelFormat in [.bc4_rUnorm, .bc4_rSnorm] { layouts[format] = bc8 }
        for format: MTLPixelFormat in [.bc2_rgba, .bc2_rgba_srgb, .bc5_rgUnorm, .bc5_rgSnorm, .bc6H_rgbFloat, .bc6H_rgbuFloat] {
            layouts[format] = bc16
        }
        let eacRG = Layout(blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, family: .appleBlock)
        layouts[.eac_rg11Unorm] = eacRG
        layouts[.eac_rg11Snorm] = eacRG
        return layouts
    }()

    private static func sliceCount(type: MTLTextureType, arrayLength: Int) -> Int? {
        switch type {
        case .type2D, .type3D: 1
        case .type2DArray: arrayLength
        case .typeCube: 6
        case .typeCubeArray: 6 * arrayLength
        default: nil
        }
    }

    private static func validatePlatformMips(_ format: MTLPixelFormat, count: Int) throws {
        #if targetEnvironment(simulator)
        let astc = MTLPixelFormat.astc_4x4_srgb.rawValue...MTLPixelFormat.astc_12x12_hdr.rawValue
        // Metal simulator readback can return level-zero blocks for smaller ASTC mips.
        if count > 1, astc.contains(format.rawValue) {
            throw SceneArchive.Failure.simulatorCompressedMipChain
        }
        #endif
    }

    /// Copies every level and slice into one buffer and waits for the copy.
    static func read(_ texture: any MTLTexture, queue: any MTLCommandQueue) throws -> (ArchiveTextureRecord, Data) {
        guard texture.device === queue.device else { throw SceneArchive.Failure.wrongDevice }
        guard texture.sampleCount == 1, !texture.isFramebufferOnly, texture.storageMode != .memoryless,
              let slices = sliceCount(type: texture.textureType, arrayLength: texture.arrayLength) else {
            throw SceneArchive.Failure.unsupportedTexture(texture.label ?? "unnamed")
        }
        guard let layout = layout(texture.pixelFormat) else {
            throw SceneArchive.Failure.unsupportedTextureFormat(texture.pixelFormat.rawValue)
        }
        try validatePlatformMips(texture.pixelFormat, count: texture.mipmapLevelCount)
        let images = images(
            type: texture.textureType, width: texture.width, height: texture.height, depth: texture.depth,
            levels: texture.mipmapLevelCount, slices: slices, layout: layout
        )
        let total = images.reduce(0) { $0 + $1.length }
        guard let staging = queue.device.makeBuffer(length: max(total, 16), options: .storageModeShared),
              let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
            throw SceneArchive.Failure.resource("texture readback")
        }
        for image in images {
            blit.copy(
                from: texture, sourceSlice: image.slice, sourceLevel: image.level, sourceOrigin: MTLOrigin(),
                sourceSize: image.size, to: staging, destinationOffset: image.offset,
                destinationBytesPerRow: image.bytesPerRow, destinationBytesPerImage: image.size.depth > 1 ? image.bytesPerImage : 0
            )
        }
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw SceneArchive.Failure.gpu(command.error?.localizedDescription ?? "texture readback did not complete")
        }
        let record = ArchiveTextureRecord(
            label: texture.label, pixelFormat: texture.pixelFormat.rawValue, textureType: texture.textureType.rawValue,
            width: texture.width, height: texture.height, depth: texture.depth, arrayLength: texture.arrayLength,
            mipmapLevelCount: texture.mipmapLevelCount, usage: texture.usage.rawValue,
            storageMode: (texture.storageMode == .shared ? MTLStorageMode.shared : .private).rawValue, view: -1
        )
        return (record, Data(bytes: staging.contents(), count: total))
    }

    /// Metal's limits for the texture types archives store, checked before creating a
    /// texture because an invalid descriptor stops the process instead of returning nil.
    private static let maximumDimension = 16_384
    private static let maximumVolumeDimension = 2_048
    private static let maximumArrayLength = 2_048

    /// Whether a record describes a texture Metal can create: dimensions that fit its type,
    /// a square cube, and no more mip levels than a full chain.
    static func isValid(_ record: ArchiveTextureRecord, type: MTLTextureType) -> Bool {
        let (width, height, depth, layers) = (record.width, record.height, record.depth, record.arrayLength)
        guard width > 0, height > 0, depth > 0, layers > 0, record.mipmapLevelCount > 0 else { return false }
        switch type {
        case .type2D:
            guard width <= maximumDimension, height <= maximumDimension, depth == 1, layers == 1 else { return false }
        case .type2DArray:
            guard width <= maximumDimension, height <= maximumDimension, depth == 1, layers <= maximumArrayLength else { return false }
        case .typeCube, .typeCubeArray:
            guard width == height, width <= maximumDimension, depth == 1,
                  type == .typeCubeArray ? layers <= maximumArrayLength : layers == 1 else { return false }
        case .type3D:
            guard [width, height, depth].allSatisfy({ $0 <= maximumVolumeDimension }), layers == 1 else { return false }
        default:
            return false
        }
        let largest = type == .type3D ? max(width, height, depth) : max(width, height)
        let fullChain = Int.bitWidth - largest.leadingZeroBitCount
        return record.mipmapLevelCount <= fullChain
    }

    /// Creates a texture with the record's descriptor and uploads its bits, waiting for completion.
    ///
    /// The record is checked against Metal's limits and the device's compressed-format
    /// support, and its byte count against the view, before anything is allocated.
    static func make(_ record: ArchiveTextureRecord, bytes: Data, queue: any MTLCommandQueue) throws -> any MTLTexture {
        let device = queue.device
        let label = record.label ?? "unnamed"
        guard let type = MTLTextureType(rawValue: record.textureType),
              isValid(record, type: type), let slices = sliceCount(type: type, arrayLength: record.arrayLength) else {
            throw SceneArchive.Failure.invalidTexture(label)
        }
        let format = MTLPixelFormat(rawValue: record.pixelFormat) ?? .invalid
        guard let layout = layout(format) else { throw SceneArchive.Failure.unsupportedTextureFormat(record.pixelFormat) }
        let supported = switch layout.family {
        case .uncompressed: true
        case .bc: device.supportsBCTextureCompression
        case .appleBlock: device.supportsFamily(.apple2)
        }
        guard supported else { throw SceneArchive.Failure.unsupportedTextureFormat(record.pixelFormat) }
        try validatePlatformMips(format, count: record.mipmapLevelCount)
        let images = images(
            type: type, width: record.width, height: record.height, depth: record.depth,
            levels: record.mipmapLevelCount, slices: slices, layout: layout
        )
        guard images.reduce(0, { $0 + $1.length }) == bytes.count else { throw SceneArchive.Failure.invalidTexture(label) }
        // Keep only usages Metal defines; block-compressed formats cannot be written or rendered to.
        var usage = MTLTextureUsage(rawValue: record.usage).intersection([
            .shaderRead, .shaderWrite, .renderTarget, .pixelFormatView, .shaderAtomic,
        ])
        if layout.family != .uncompressed { usage.subtract([.shaderWrite, .renderTarget, .shaderAtomic]) }
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = type
        descriptor.pixelFormat = format
        descriptor.width = record.width
        descriptor.height = record.height
        descriptor.depth = record.depth
        descriptor.arrayLength = record.arrayLength
        descriptor.mipmapLevelCount = record.mipmapLevelCount
        descriptor.usage = usage
        let storage = MTLStorageMode(rawValue: record.storageMode) ?? .private
        descriptor.storageMode = storage == .shared ? .shared : .private
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw SceneArchive.Failure.invalidTexture(label) }
        texture.label = record.label
        guard let staging = bytes.withUnsafeBytes({ raw in
            raw.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: max(bytes.count, 1), options: .storageModeShared) }
        }), let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
            throw SceneArchive.Failure.resource("texture upload")
        }
        for image in images {
            blit.copy(
                from: staging, sourceOffset: image.offset, sourceBytesPerRow: image.bytesPerRow,
                sourceBytesPerImage: image.size.depth > 1 ? image.bytesPerImage : 0, sourceSize: image.size,
                to: texture, destinationSlice: image.slice, destinationLevel: image.level, destinationOrigin: MTLOrigin()
            )
        }
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw SceneArchive.Failure.gpu(command.error?.localizedDescription ?? "texture upload did not complete")
        }
        return texture
    }

    private struct Image {
        let level: Int
        let slice: Int
        let size: MTLSize
        let bytesPerRow: Int
        let bytesPerImage: Int
        let offset: Int
        var length: Int { bytesPerImage * size.depth }
    }

    private static func images(
        type: MTLTextureType, width: Int, height: Int, depth: Int, levels: Int, slices: Int, layout: Layout
    ) -> [Image] {
        var images: [Image] = []
        var offset = 0
        for level in 0..<levels {
            let levelWidth = max(1, width >> level)
            let levelHeight = max(1, height >> level)
            let levelDepth = type == .type3D ? max(1, depth >> level) : 1
            let bytesPerRow = layout.bytesPerRow(width: levelWidth)
            let bytesPerImage = layout.bytesPerImage(width: levelWidth, height: levelHeight)
            for slice in 0..<slices {
                let image = Image(
                    level: level, slice: slice, size: MTLSize(width: levelWidth, height: levelHeight, depth: levelDepth),
                    bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage, offset: offset
                )
                images.append(image)
                offset += image.length
            }
        }
        return images
    }
}

/// Converts between `simd_float4x4` arrays and the 64-byte column-major records stored in views.
enum ArchiveMatrices {
    static func pack(_ matrices: [simd_float4x4]) -> [SIMD4<Float>] {
        matrices.flatMap { [$0.columns.0, $0.columns.1, $0.columns.2, $0.columns.3] }
    }

    static func unpack(_ columns: [SIMD4<Float>]) -> [simd_float4x4] {
        stride(from: 0, to: columns.count - columns.count % 4, by: 4).map {
            simd_float4x4(columns: (columns[$0], columns[$0 + 1], columns[$0 + 2], columns[$0 + 3]))
        }
    }
}
