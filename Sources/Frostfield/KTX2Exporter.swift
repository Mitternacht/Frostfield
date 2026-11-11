import Foundation
import Metal

/// Encodes a Metal texture's stored texels and mips as an uncompressed KTX 2.0 file.
public enum KTX2Exporter {
    public enum ColorPrimaries: UInt8, Sendable {
        case unspecified = 0
        /// The primaries shared by sRGB and linear sRGB.
        case sRGB = 1
    }

    public enum AlphaMode: UInt8, Sendable {
        case straight = 0
        case premultiplied = 1
    }

    public struct Options: Sendable {
        /// Nil tags sRGB pixel formats with sRGB primaries and leaves linear
        /// formats unspecified. Set `.sRGB` for linear-sRGB color textures.
        public var colorPrimaries: ColorPrimaries?
        /// Alpha association metadata for four-channel textures.
        public var alphaMode: AlphaMode

        public init(colorPrimaries: ColorPrimaries? = nil, alphaMode: AlphaMode = .straight) {
            self.colorPrimaries = colorPrimaries
            self.alphaMode = alphaMode
        }
    }

    /// Exports every allocated mip of a single-sample 2D texture, 2D array, cube, cube
    /// array, or 3D texture.
    /// Supports R, RG, RGBA and BGRA 8-bit UNORM or sRGB, and R, RG and RGBA Float16
    /// or Float32. The source format determines the transfer function. No pixel,
    /// alpha, or color-space conversion takes place. The file stores top-left
    /// rows, array layers and cube faces in Metal order. A cube array writes
    /// `faceCount` 6 and one `layerCount` entry per cube, ordered cube by cube with
    /// the faces of each cube in +X, −X, +Y, −Y, +Z, −Z order, which is Metal's
    /// slice order (`6 × cube + face`). Cube arrays need a GPU that supports them
    /// (Apple4 or Mac2 family); the iOS Simulator does not. Complete cubes carry
    /// no `KTXcubemapIncomplete` entry. Block-compressed and Basis Universal cube
    /// arrays are not written. Use the explicit `TextureLoader.loadKTX2` entry
    /// point to restore a 2D array; it does not import cube arrays.
    ///
    /// Populate every mip and commit pending writes before calling. Supply the
    /// texture's producer queue to order the readback after tracked writes.
    /// Finish writes first when using an untracked texture or a different queue.
    /// The method waits for the copy and uses the loader's 64 MiB file budget.
    public static func data(
        from texture: any MTLTexture, commandQueue: (any MTLCommandQueue)? = nil,
        options: Options = .init()
    ) throws -> Data {
        if let commandQueue, commandQueue.device !== texture.device { throw Failure.wrongDevice }
        guard texture.sampleCount == 1, !texture.isFramebufferOnly, texture.storageMode != .memoryless else {
            throw Failure.textureType
        }
        let depth: Int
        let faces: Int
        let layerCount: Int
        switch texture.textureType {
        case .type2D: (depth, faces, layerCount) = (0, 1, 0)
        case .type2DArray: (depth, faces, layerCount) = (0, 1, texture.arrayLength)
        case .typeCube: (depth, faces, layerCount) = (0, 6, 0)
        // Metal counts cubes, not faces, in a cube array's length; KTX2 does the same.
        case .typeCubeArray: (depth, faces, layerCount) = (0, 6, texture.arrayLength)
        case .type3D: (depth, faces, layerCount) = (texture.depth, 1, 0)
        default: throw Failure.textureType
        }
        guard let format = KTX2Format(uncompressedPixelFormat: texture.pixelFormat),
              format.isSupported(by: texture.device, samplingAs: texture.pixelFormat) else {
            throw Failure.pixelFormat(texture.pixelFormat)
        }
        let layout = try plan(
            format: format, width: texture.width, height: texture.height, depth: depth, faces: faces,
            layerCount: layerCount, levelCount: texture.mipmapLevelCount, options: options
        )

        guard let queue = commandQueue ?? texture.device.makeCommandQueue() else { throw Failure.queue }
        let pixels: Data
        do {
            (_, pixels) = try ArchiveTextureCodec.read(texture, queue: queue)
        } catch {
            throw Failure.readback(error.localizedDescription)
        }
        return try encode(layout, pixels: pixels)
    }

    public enum Failure: LocalizedError, Equatable {
        case wrongDevice
        case textureType
        case pixelFormat(MTLPixelFormat)
        case size
        /// A cube or cube array whose faces are not square.
        case cubeFaceSize(width: Int, height: Int)
        /// A 2D array or cube array with no layers or more than `KTX2Container.maximumArrayLayers`.
        case arrayLength(Int)
        case queue
        case readback(String)

        public var errorDescription: String? {
            switch self {
            case .wrongDevice: "Use a queue on the source texture's Metal device."
            case .textureType:
                "Use a readable, single-sample 2D texture, 2D array, cube, cube array, or 3D texture."
            case .pixelFormat(let format): "KTX2 export does not support Metal pixel format \(format.rawValue)."
            case .size: "The KTX2 texture exceeds the supported dimensions, texel count, or 64 MiB file size."
            case .cubeFaceSize(let width, let height):
                "Cube faces must be square, but this texture is \(width) × \(height)."
            case .arrayLength(let count):
                "KTX2 export supports 1 through \(KTX2Container.maximumArrayLayers) array layers or cubes, not \(count)."
            case .queue: "Metal could not create a KTX2 readback queue."
            case .readback(let detail): "Could not read KTX2 texels. \(detail)"
            }
        }
    }

    struct Level: Equatable {
        /// Where the level starts in the readback bytes.
        let sourceOffset: Int
        /// Bytes of the level: every layer, face and depth slice.
        let length: Int
        var fileOffset = 0
    }

    /// The file structure derived from a texture's shape, fixed before any texel is read.
    struct Plan {
        let format: KTX2Format
        let width: Int
        let height: Int
        let depth: Int
        let faces: Int
        let layerCount: Int
        let descriptor: Data
        let keyValues: Data
        /// Levels from the base level to the smallest.
        let levels: [Level]
        /// Total bytes of all levels in readback order.
        let sourceLength: Int
        let fileLength: Int
    }

    /// Validates a texture shape and lays out the KTX2 file for it.
    ///
    /// `layerCount` is zero for a texture that is not an array and otherwise counts
    /// array elements, which are whole cubes when `faces` is 6. Readback bytes and
    /// the file both hold each level as layers, then faces, then depth slices,
    /// matching KTX 2.0 section 3.13.1.
    static func plan(
        format: KTX2Format, width: Int, height: Int, depth: Int, faces: Int, layerCount: Int, levelCount: Int,
        options: Options
    ) throws -> Plan {
        guard layerCount == 0 || (1...KTX2Container.maximumArrayLayers).contains(layerCount) else {
            throw Failure.arrayLength(layerCount)
        }
        guard faces == 1 || width == height else {
            throw Failure.cubeFaceSize(width: width, height: height)
        }
        let layers = max(1, layerCount)
        guard KTX2Container.dimensionsFit(width: width, height: height, depth: depth, faces: faces, layers: layers) else {
            throw Failure.size
        }

        let dfd = descriptor(format, options: options)
        let kvd = metadata(volume: depth > 0)
        let dfdOffset = 80 + 24 * levelCount
        let kvdOffset = dfdOffset + dfd.count
        let alignment = max(4, format.bytesPerBlock)
        var sourceOffset = 0
        var levels: [Level] = []
        for index in 0..<levelCount {
            let length = format.byteCount(width: max(1, width >> index), height: max(1, height >> index))
                * max(1, depth >> index) * faces * layers
            levels.append(Level(sourceOffset: sourceOffset, length: length))
            sourceOffset += length
        }
        var fileLength = kvdOffset + kvd.count
        for index in levels.indices.reversed() {
            // Supported texel sizes are powers of two, so max(4, texel size) is their LCM.
            let offset = (fileLength + alignment - 1) & ~(alignment - 1)
            levels[index].fileOffset = offset
            fileLength = offset + levels[index].length
        }
        guard fileLength <= TextureLoader.maxEncodedBytes else { throw Failure.size }
        return Plan(
            format: format, width: width, height: height, depth: depth, faces: faces, layerCount: layerCount,
            descriptor: dfd, keyValues: kvd, levels: levels, sourceLength: sourceOffset, fileLength: fileLength
        )
    }

    /// Writes the planned file from readback bytes ordered by level, layer, face and depth slice.
    static func encode(_ plan: Plan, pixels: Data) throws -> Data {
        guard pixels.count == plan.sourceLength else { throw Failure.readback("Unexpected texture byte count.") }
        let dfdOffset = 80 + 24 * plan.levels.count
        let kvdOffset = dfdOffset + plan.descriptor.count
        var result = Data(KTX2Container.identifier)
        result.reserveCapacity(plan.fileLength)
        for word in [
            plan.format.vkFormat, plan.format.typeSize, UInt32(plan.width), UInt32(plan.height), UInt32(plan.depth),
            UInt32(plan.layerCount), UInt32(plan.faces), UInt32(plan.levels.count), 0, // No supercompression.
            UInt32(dfdOffset), UInt32(plan.descriptor.count), UInt32(kvdOffset), UInt32(plan.keyValues.count),
        ] {
            append(word, to: &result)
        }
        append(UInt64(0), to: &result) // No supercompression global data.
        append(UInt64(0), to: &result)
        // The index names the base level first, while image bytes put the smallest first.
        for level in plan.levels {
            append(UInt64(level.fileOffset), to: &result)
            append(UInt64(level.length), to: &result)
            append(UInt64(level.length), to: &result)
        }
        result.append(plan.descriptor)
        result.append(plan.keyValues)
        for level in plan.levels.reversed() {
            result.append(contentsOf: repeatElement(0, count: level.fileOffset - result.count))
            result.append(pixels[(pixels.startIndex + level.sourceOffset)..<(pixels.startIndex + level.sourceOffset + level.length)])
        }
        return result
    }

    private static func descriptor(_ format: KTX2Format, options: Options) -> Data {
        let channelCount = format.bytesPerBlock / Int(format.typeSize)
        let channels: [UInt8] = format.linear == .bgra8Unorm ? [2, 1, 0, 15] : [0, 1, 2, 15]
        let bits = UInt16(format.typeSize * 8)
        let float = format.typeSize > 1
        let primaries = options.colorPrimaries ?? (format.isSRGB ? .sRGB : .unspecified)
        let alpha = channelCount == 4 ? options.alphaMode.rawValue : 0
        let blockSize = UInt16(24 + channelCount * 16)

        var data = Data()
        append(UInt32(blockSize) + 4, to: &data)
        append(UInt32(0), to: &data) // Khronos vendor and basic descriptor.
        append(UInt16(2), to: &data) // KHR_DF_VERSIONNUMBER_1_3.
        append(blockSize, to: &data)
        data.append(contentsOf: [1, primaries.rawValue, format.isSRGB ? 2 : 1, alpha]) // RGBSDA, primaries, transfer, flags.
        append(UInt32(0), to: &data) // 1 × 1 × 1 × 1 texel block.
        append(UInt32(format.bytesPerBlock), to: &data) // Only plane zero has bytes.
        append(UInt32(0), to: &data)
        for index in 0..<channelCount {
            var channel = channels[index]
            if float { channel |= 0xC0 } // Floating-point and signed qualifiers.
            if channel == 15, format.isSRGB { channel |= 0x10 } // Alpha stays linear.
            append(UInt16(index) * bits, to: &data)
            data.append(UInt8(bits - 1))
            data.append(channel)
            append(UInt32(0), to: &data) // Sample position in the texel.
            append(float ? Float(-1).bitPattern : 0, to: &data)
            append(float ? Float(1).bitPattern : 255, to: &data)
        }
        return data
    }

    private static func metadata(volume: Bool) -> Data {
        var data = Data()
        // KTX requires lexicographically sorted keys and 4-byte entry alignment.
        for (key, value) in [("KTXorientation", volume ? "rdi" : "rd"), ("KTXwriter", "Frostfield")] {
            let bytes = Data("\(key)\0\(value)\0".utf8)
            append(UInt32(bytes.count), to: &data)
            data.append(bytes)
            data.append(contentsOf: repeatElement(0, count: (4 - bytes.count % 4) % 4))
        }
        return data
    }

    private static func append<T: FixedWidthInteger>(_ word: T, to data: inout Data) {
        withUnsafeBytes(of: word.littleEndian) { data.append(contentsOf: $0) }
    }
}
