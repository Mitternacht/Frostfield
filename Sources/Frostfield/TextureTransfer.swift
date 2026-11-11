import Foundation
import IOSurface
import Metal

/// Bit-preserving uploads, copies and readback for tracked Metal textures.
public enum TextureTransfer {
    /// Maximum bytes returned by one read. Read large textures in regions.
    public static let maximumReadbackBytes = 256 * 1024 * 1024
    /// Maximum stored bytes staged by one upload.
    public static let maximumUploadBytes = 256 * 1024 * 1024

    public struct Readback: Sendable {
        public let pixelFormat: MTLPixelFormat
        public let size: MTLSize
        /// Tight pitch in stored bytes. Compressed formats count rows of blocks.
        public let bytesPerRow: Int
        public let bytesPerImage: Int
        /// Stored texels or blocks in Metal row order, with no color or alpha conversion.
        public let bytes: Data
    }

    /// Reads one mip and slice, or a 3D subvolume with slice zero. A nil region
    /// reads the whole selected mip. Cube slices follow Metal face order.
    /// Commit producer work on this queue before calling. Synchronize producers
    /// on other queues before reading. Cancellation after submission waits for
    /// Metal completion before releasing the GPU resources.
    public static func read(
        from texture: any MTLTexture, mipLevel: Int = 0, slice: Int = 0,
        region: MTLRegion? = nil, commandQueue: any MTLCommandQueue
    ) async throws -> Readback {
        try await readOnCaller(
            from: texture, mipLevel: mipLevel, slice: slice, region: region, commandQueue: commandQueue
        )
    }

    // Keep the read on the caller's isolation when it owns a non-Sendable texture.
    static nonisolated(nonsending) func readOnCaller(
        from texture: any MTLTexture, mipLevel: Int = 0, slice: Int = 0,
        region: MTLRegion? = nil, commandQueue: any MTLCommandQueue
    ) async throws -> Readback {
        try Task.checkCancellation()
        guard texture.device === commandQueue.device else { throw Failure.wrongDevice }
        let source = try Subresource(texture, level: mipLevel, slice: slice)
        let region = region ?? MTLRegion(origin: MTLOrigin(), size: source.size)
        try source.validate(region)
        let row = source.layout.bytesPerRow(width: region.size.width)
        let image = source.layout.bytesPerImage(width: region.size.width, height: region.size.height)
        let (length, overflow) = image.multipliedReportingOverflow(by: region.size.depth)
        guard !overflow, length <= min(maximumReadbackBytes, commandQueue.device.maxBufferLength) else {
            throw Failure.readbackSize
        }
        guard let staging = commandQueue.device.makeBuffer(length: length, options: .storageModeShared),
              let command = commandQueue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
            throw Failure.resource
        }
        command.label = "Frostfield texture readback"
        blit.copy(
            from: texture, sourceSlice: slice, sourceLevel: mipLevel, sourceOrigin: region.origin,
            sourceSize: region.size, to: staging, destinationOffset: 0, destinationBytesPerRow: row,
            destinationBytesPerImage: source.dimensions == 3 ? image : 0
        )
        blit.endEncoding()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            command.addCompletedHandler { completed in
                if completed.status == .completed {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: Failure.gpu(completed.error?.localizedDescription ?? "Metal readback failed."))
                }
            }
            command.commit()
        }
        try Task.checkCancellation()
        return Readback(
            pixelFormat: texture.pixelFormat, size: region.size, bytesPerRow: row, bytesPerImage: image,
            bytes: Data(bytes: staging.contents(), count: length)
        )
    }

    /// Encodes a copy between matching pixel formats and texture dimensions.
    /// For arrays and cubes, select one slice per call. For volumes use slice
    /// zero and specify depth in the source region. A nil region copies the
    /// whole source mip. The caller submits the command buffer and must close
    /// any active encoder before calling. Metal orders tracked writes on the
    /// same queue. Use a retaining command buffer.
    public static func encodeCopy(
        from source: any MTLTexture, sourceMipLevel: Int = 0, sourceSlice: Int = 0, sourceRegion: MTLRegion? = nil,
        to destination: any MTLTexture, destinationMipLevel: Int = 0, destinationSlice: Int = 0,
        destinationOrigin: MTLOrigin = MTLOrigin(), commandBuffer: any MTLCommandBuffer
    ) throws {
        guard source.device === commandBuffer.device, destination.device === commandBuffer.device else {
            throw Failure.wrongDevice
        }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        let from = try Subresource(source, level: sourceMipLevel, slice: sourceSlice)
        let to = try Subresource(destination, level: destinationMipLevel, slice: destinationSlice)
        guard from.dimensions == to.dimensions, source.pixelFormat == destination.pixelFormat else {
            throw Failure.incompatibleTextures
        }
        let region = sourceRegion ?? MTLRegion(origin: MTLOrigin(), size: from.size)
        try from.validate(region)
        try to.validate(MTLRegion(origin: destinationOrigin, size: region.size))
        if source === destination {
            if sourceMipLevel == destinationMipLevel, sourceSlice == destinationSlice,
               overlaps(region, destinationOrigin) { throw Failure.overlappingCopy }
        } else if sharesStorage(from.backing, to.backing) {
            throw Failure.overlappingCopy
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.resource }
        blit.copy(
            from: source, sourceSlice: sourceSlice, sourceLevel: sourceMipLevel, sourceOrigin: region.origin,
            sourceSize: region.size, to: destination, destinationSlice: destinationSlice,
            destinationLevel: destinationMipLevel, destinationOrigin: destinationOrigin
        )
        blit.endEncoding()
    }

    /// Uploads tightly packed texels or compressed blocks to one mip, array
    /// layer, cube face, or volume region. A nil region writes the full selected
    /// subresource. Untouched texels and other mips retain their stored values.
    ///
    /// The bytes are copied before this call returns, with no color conversion
    /// or mip generation. Close any active encoder and use an uncommitted,
    /// retaining command buffer. Later tracked work on the same queue sees the write.
    public static func encodeUpload(
        _ bytes: Data, to texture: any MTLTexture, mipLevel: Int = 0, slice: Int = 0,
        region: MTLRegion? = nil, commandBuffer: any MTLCommandBuffer
    ) throws {
        let device = commandBuffer.device
        guard texture.device === device else { throw Failure.wrongDevice }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        let target = try Subresource(texture, level: mipLevel, slice: slice)
        let region = region ?? MTLRegion(origin: MTLOrigin(), size: target.size)
        try target.validate(region)
        let row = target.layout.bytesPerRow(width: region.size.width)
        let image = target.layout.bytesPerImage(width: region.size.width, height: region.size.height)
        let (length, overflow) = image.multipliedReportingOverflow(by: region.size.depth)
        guard !overflow, length <= min(maximumUploadBytes, device.maxBufferLength) else { throw Failure.uploadSize }
        guard bytes.count == length else { throw Failure.uploadByteCount(expected: length) }
        let staging = bytes.withUnsafeBytes { source in
            source.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: length, options: .storageModeShared) }
        }
        guard let staging, let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.resource }
        staging.label = "Frostfield texture upload"
        blit.copy(
            from: staging, sourceOffset: 0, sourceBytesPerRow: row,
            sourceBytesPerImage: target.dimensions == 3 ? image : 0, sourceSize: region.size,
            to: texture, destinationSlice: slice, destinationLevel: mipLevel, destinationOrigin: region.origin
        )
        blit.endEncoding()
    }

    public enum Failure: LocalizedError, Equatable {
        case wrongDevice
        case unsupportedTexture
        case unsupportedPixelFormat(MTLPixelFormat)
        case untrackedTexture
        case invalidSubresource
        case invalidRegion
        case incompatibleTextures
        case overlappingCopy
        case commandBuffer
        case readbackSize
        case uploadSize
        case uploadByteCount(expected: Int)
        case resource
        case gpu(String)

        public var errorDescription: String? {
            switch self {
            case .wrongDevice: "All Metal resources must belong to the same device."
            case .unsupportedTexture: "Use a single-sample texture with persistent storage and blit access."
            case .unsupportedPixelFormat(let format): "Texture transfer does not support pixel format \(format.rawValue)."
            case .untrackedTexture: "Texture transfer requires Metal hazard tracking."
            case .invalidSubresource: "The mip or slice does not exist."
            case .invalidRegion: "Use a positive region inside the selected mip and align compressed blocks to its edges."
            case .incompatibleTextures: "The copy needs matching pixel formats and texture dimensions."
            case .overlappingCopy: "The copy overlaps a subresource or uses different views of shared texture storage."
            case .commandBuffer: "Use a retaining command buffer that has not been committed."
            case .readbackSize: "Read a smaller texture region."
            case .uploadSize: "Upload a smaller texture region."
            case .uploadByteCount(let expected): "Supply exactly \(expected) tightly packed texture bytes."
            case .resource: "Metal could not create the texture transfer resources."
            case .gpu(let reason): reason
            }
        }
    }

    private struct Subresource {
        let dimensions: Int
        let size: MTLSize
        let layout: ArchiveTextureCodec.Layout
        let backing: any MTLTexture

        init(_ texture: any MTLTexture, level: Int, slice: Int) throws {
            let backing = root(texture)
            let allocation: any MTLResource = backing.buffer.map { $0 as any MTLResource } ?? backing
            guard texture.sampleCount == 1, !texture.isFramebufferOnly, texture.storageMode != .memoryless,
                  backing.isSparse != true, !allocation.isAliasable() else { throw Failure.unsupportedTexture }
            if let heap = allocation.heap {
                guard heap.hazardTrackingMode == .tracked else { throw Failure.untrackedTexture }
            } else if allocation.hazardTrackingMode == .untracked {
                throw Failure.untrackedTexture
            }
            self.backing = backing
            let slices: Int
            switch texture.textureType {
            case .type1D: (dimensions, slices) = (1, 1)
            case .type1DArray: (dimensions, slices) = (1, texture.arrayLength)
            case .type2D: (dimensions, slices) = (2, 1)
            case .type2DArray: (dimensions, slices) = (2, texture.arrayLength)
            case .typeCube: (dimensions, slices) = (2, 6)
            case .typeCubeArray: (dimensions, slices) = (2, texture.arrayLength * 6)
            case .type3D: (dimensions, slices) = (3, 1)
            default: throw Failure.unsupportedTexture
            }
            guard level >= 0, level < texture.mipmapLevelCount, slice >= 0, slice < slices else {
                throw Failure.invalidSubresource
            }
            let format = texture.pixelFormat
            let layout: ArchiveTextureCodec.Layout
            switch format {
            case .depth16Unorm: layout = .init(blockWidth: 1, blockHeight: 1, bytesPerBlock: 2)
            case .depth32Float: layout = .init(blockWidth: 1, blockHeight: 1, bytesPerBlock: 4)
            default:
                guard let stored = ArchiveTextureCodec.layout(format) else { throw Failure.unsupportedPixelFormat(format) }
                layout = stored
            }
            #if targetEnvironment(simulator)
            // Some Simulator ASTC readbacks silently return blocks from level zero.
            if (level > 0 || backing !== texture),
               (MTLPixelFormat.astc_4x4_srgb.rawValue...MTLPixelFormat.astc_12x12_hdr.rawValue).contains(format.rawValue) {
                throw Failure.unsupportedPixelFormat(format)
            }
            #endif
            self.layout = layout
            size = MTLSize(
                width: max(1, texture.width >> level), height: dimensions >= 2 ? max(1, texture.height >> level) : 1,
                depth: dimensions == 3 ? max(1, texture.depth >> level) : 1
            )
        }

        func validate(_ region: MTLRegion) throws {
            func fits(_ origin: Int, _ length: Int, _ total: Int, block: Int = 1) -> Bool {
                origin >= 0 && length > 0 && origin <= total && length <= total - origin && origin.isMultiple(of: block)
                    && (length.isMultiple(of: block) || length == total - origin)
            }
            guard fits(region.origin.x, region.size.width, size.width, block: layout.blockWidth),
                  fits(region.origin.y, region.size.height, size.height, block: layout.blockHeight),
                  fits(region.origin.z, region.size.depth, size.depth) else { throw Failure.invalidRegion }
        }
    }

    private static func root(_ texture: any MTLTexture) -> any MTLTexture {
        var root = texture
        while let parent = root.parent { root = parent }
        return root
    }

    static func sharesStorage(_ first: any MTLTexture, _ second: any MTLTexture) -> Bool {
        let first = root(first), second = root(second)
        let firstAllocation: any MTLResource = first.buffer.map { $0 as any MTLResource } ?? first
        let secondAllocation: any MTLResource = second.buffer.map { $0 as any MTLResource } ?? second
        return firstAllocation === secondAllocation || sharePlacement(firstAllocation, secondAllocation)
            || shareSurface(first, second)
    }

    private static func sharePlacement(_ first: any MTLResource, _ second: any MTLResource) -> Bool {
        guard let heap = first.heap, heap.type == .placement, heap === second.heap else { return false }
        let (lower, upper) = first.heapOffset <= second.heapOffset ? (first, second) : (second, first)
        return upper.heapOffset - lower.heapOffset < lower.allocatedSize
    }

    private static func shareSurface(_ first: any MTLTexture, _ second: any MTLTexture) -> Bool {
        guard let firstSurface = first.iosurface, let secondSurface = second.iosurface else { return false }
        return first.iosurfacePlane == second.iosurfacePlane && IOSurfaceGetID(firstSurface) == IOSurfaceGetID(secondSurface)
    }

    private static func overlaps(_ source: MTLRegion, _ destination: MTLOrigin) -> Bool {
        abs(source.origin.x - destination.x) < source.size.width
            && abs(source.origin.y - destination.y) < source.size.height
            && abs(source.origin.z - destination.z) < source.size.depth
    }
}
