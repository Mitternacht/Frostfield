import Compression
import Foundation
import Metal

/// A validated KTX 2.0 container in a format Metal can sample
/// without transcoding. [KTX 2.0 specification](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html)
struct KTX2Container {
    typealias Failure = TextureLoader.Failure

    static let maximumArrayLayers = 2_048
    static let identifier: [UInt8] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]

    let format: KTX2Format
    let width: Int
    let height: Int
    let depth: Int
    let faceCount: Int
    let layerCount: Int
    let textureType: MTLTextureType
    var sliceCount: Int { faceCount * max(1, layerCount) }
    /// Byte ranges of the stored levels, starting with the base level.
    let levels: [Level]
    /// True when `levelCount` is zero, which asks the loader to generate a mip chain.
    let requestsGeneratedMipmaps: Bool
    private let compressedWithZlib: Bool
    private let data: Data

    struct Level {
        let offset: Int
        let length: Int
        let uncompressedLength: Int
    }

    static func hasIdentifier(_ data: Data) -> Bool {
        data.count >= identifier.count && data.prefix(identifier.count).elementsEqual(identifier)
    }

    /// Validates the header, level index, and metadata. Level payloads decompress on demand.
    init(_ source: Data, allowingSpatialTextures: Bool = false) throws {
        let data = Data(source)
        guard Self.hasIdentifier(data), data.count >= 80 else { throw Failure.invalidKTX2("missing KTX2 identifier or header") }
        func word(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        func long(_ offset: Int) -> UInt64 {
            data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) }
        }
        /// Converts an untrusted range to integers and checks it lies within the file.
        func range(_ offset: UInt64, _ length: UInt64, _ what: String) throws -> (Int, Int) {
            guard offset <= UInt64(data.count), length <= UInt64(data.count) - offset else {
                throw Failure.invalidKTX2("\(what) lies outside the file")
            }
            return (Int(offset), Int(length))
        }

        let vkFormat = word(12)
        let typeSize = word(16)
        let pixelWidth = Int(word(20)), pixelHeight = Int(word(24)), pixelDepth = Int(word(28))
        let layerCount = Int(word(32)), faceCount = Int(word(36)), levelCount = Int(word(40))
        let scheme = word(44)
        let (dfdOffset, dfdLength) = try range(UInt64(word(48)), UInt64(word(52)), "the data format descriptor")
        let (kvdOffset, kvdLength) = try range(UInt64(word(56)), UInt64(word(60)), "the key/value data")
        let (sgdOffset, sgdLength) = try range(long(64), long(72), "the supercompression global data")

        try Self.checkMetadata(data.subdata(in: kvdOffset..<(kvdOffset + kvdLength)), volume: pixelDepth > 0)

        // Basis Universal payloads are recognized before format checks so the error names them.
        if scheme == 1 { throw Failure.ktx2TranscodingRequired("BasisLZ/ETC1S") }
        let colorModel = dfdLength >= 16 ? data[data.startIndex + dfdOffset + 12] : nil
        switch (vkFormat, colorModel) {
        case (1_000_066_000, 167): throw Failure.ktx2TranscodingRequired("UASTC HDR 4×4")
        case (1_000_066_004, 162): throw Failure.ktx2TranscodingRequired("ASTC HDR 6×6")
        default: break
        }
        if vkFormat == 0 {
            guard let colorModel else { throw Failure.invalidKTX2("the data format descriptor is truncated") }
            switch colorModel {
            case 163: throw Failure.ktx2TranscodingRequired("BasisLZ/ETC1S")
            case 166: throw Failure.ktx2TranscodingRequired("UASTC")
            case 168: throw Failure.ktx2TranscodingRequired("UASTC HDR 6×6 intermediate")
            case let model: throw Failure.unsupportedKTX2("VK_FORMAT_UNDEFINED with color model \(model)")
            }
        }
        switch scheme {
        case 0, 3: break
        case 2: throw Failure.unsupportedKTX2("Zstandard supercompression")
        default: throw Failure.unsupportedKTX2("supercompression scheme \(scheme)")
        }
        guard let format = KTX2Format(vkFormat: vkFormat) else { throw Failure.unsupportedKTX2("vkFormat \(vkFormat)") }
        guard typeSize == format.typeSize else {
            throw Failure.invalidKTX2("typeSize \(typeSize) does not match vkFormat \(vkFormat)")
        }
        guard pixelHeight > 0 else { throw Failure.unsupportedKTX2("1D textures") }
        if !allowingSpatialTextures {
            guard pixelDepth == 0 else { throw Failure.unsupportedKTX2("3D textures") }
            guard faceCount == 1 else { throw Failure.unsupportedKTX2("cube maps") }
            // Keep the image path compatible with files declaring a singleton layer.
            guard layerCount <= 1 else { throw Failure.unsupportedKTX2("texture arrays") }
        }
        guard faceCount == 1 || faceCount == 6 else { throw Failure.invalidKTX2("faceCount must be one or six") }
        guard faceCount == 1 || (pixelWidth == pixelHeight && pixelDepth == 0) else {
            throw Failure.invalidKTX2("cube faces must be square and have no depth")
        }
        let textureType: MTLTextureType
        if faceCount == 6 {
            guard layerCount == 0 else { throw Failure.unsupportedKTX2("cube map arrays") }
            textureType = .typeCube
        } else if pixelDepth > 0 {
            guard layerCount == 0 else { throw Failure.unsupportedKTX2("3D texture arrays") }
            guard !format.isCompressed else { throw Failure.unsupportedKTX2("block-compressed 3D textures") }
            textureType = .type3D
        } else if allowingSpatialTextures, layerCount > 0 {
            guard layerCount <= Self.maximumArrayLayers else {
                throw Failure.unsupportedKTX2("more than \(Self.maximumArrayLayers) array layers")
            }
            textureType = .type2DArray
        } else {
            textureType = .type2D
        }
        let spatial = textureType != .type2D
        if spatial, format.isCompressed, levelCount == 0 {
            throw Failure.invalidKTX2("block-compressed spatial textures require a stored mip level")
        }
        guard pixelWidth > 0 else { throw Failure.invalidKTX2("pixelWidth must be positive") }
        let layers = max(1, layerCount)
        // Bound aggregate texels before deriving the per-level byte ranges.
        guard Self.dimensionsFit(width: pixelWidth, height: pixelHeight, depth: pixelDepth,
                                 faces: faceCount, layers: layers) else {
            throw Failure.imageDimensions(width: pixelWidth, height: pixelHeight)
        }
        let maximumLevels = Int.bitWidth - max(pixelWidth, pixelHeight, pixelDepth).leadingZeroBitCount
        guard levelCount <= maximumLevels else {
            throw Failure.invalidKTX2("levelCount \(levelCount) exceeds the \(maximumLevels) levels of a full chain")
        }
        let stored = max(1, levelCount)
        let indexEnd = 80 + stored * 24
        guard data.count >= indexEnd else { throw Failure.invalidKTX2("the level index is truncated") }
        let metadataEnd = max(indexEnd, dfdOffset + dfdLength,
                              kvdLength == 0 ? 0 : kvdOffset + kvdLength, sgdLength == 0 ? 0 : sgdOffset + sgdLength)

        var levels: [Level] = []
        var precedingLevel = data.count
        for level in 0..<stored {
            let entry = 80 + level * 24
            let (offset, length) = try range(long(entry), long(entry + 8), "level \(level)")
            let expected = format.byteCount(
                width: max(1, pixelWidth >> level), height: max(1, pixelHeight >> level)
            ) * max(1, pixelDepth >> level) * faceCount * layers
            guard long(entry + 16) == UInt64(expected) else {
                throw Failure.invalidKTX2("level \(level) needs \(expected) uncompressed bytes")
            }
            guard scheme == 3 ? length > 6 : length == expected else {
                throw Failure.invalidKTX2("level \(level) has an invalid byte length")
            }
            if spatial {
                // Indexed from largest to smallest, stored from smallest to largest.
                guard offset >= metadataEnd, offset + length <= precedingLevel else {
                    throw Failure.invalidKTX2("level \(level) is out of storage order or overlaps other data")
                }
                if scheme == 0, !offset.isMultiple(of: max(4, format.bytesPerBlock)) {
                    throw Failure.invalidKTX2("level \(level) has an unaligned byte offset")
                }
                precedingLevel = offset
            }
            levels.append(Level(offset: offset, length: length, uncompressedLength: expected))
        }

        self.format = format
        width = pixelWidth
        height = pixelHeight
        depth = pixelDepth
        self.faceCount = faceCount
        self.layerCount = layerCount
        self.textureType = textureType
        self.levels = levels
        requestsGeneratedMipmaps = levelCount == 0
        compressedWithZlib = scheme == 3
        self.data = data
    }

    /// Returns a level's texel blocks, inflating and checking a zlib stream when present.
    func bytes(level index: Int) throws -> Data {
        let level = levels[index]
        let stored = data.subdata(in: level.offset..<(level.offset + level.length))
        guard compressedWithZlib else { return stored }
        return try Self.inflate(stored, expected: level.uncompressedLength, level: index)
    }

    static func dimensionsFit(width: Int, height: Int, depth: Int, faces: Int, layers: Int = 1) -> Bool {
        let maxSide = depth > 0 ? 2_048 : 8_192
        return (1...maxSide).contains(width) && (1...maxSide).contains(height) && (0...2_048).contains(depth)
            && width * height * max(1, depth) * faces * layers <= 16_777_216
    }

    /// Accepts top-left orientation and identity swizzle, the layouts uploaded unchanged.
    private static func checkMetadata(_ data: Data, volume: Bool) throws {
        var offset = 0
        while data.count - offset >= 4 {
            let length = Int(data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) })
            offset += 4
            guard length <= data.count - offset else { throw Failure.invalidKTX2("a key/value entry is truncated") }
            let entry = data.subdata(in: offset..<(offset + length))
            if let separator = entry.firstIndex(of: 0) {
                let key = String(decoding: entry[..<separator], as: UTF8.self)
                let value = String(decoding: entry[(separator + 1)...].prefix { $0 != 0 }, as: UTF8.self)
                let orientation = volume ? "rdi" : "rd"
                if key == "KTXorientation", value != orientation {
                    throw Failure.unsupportedKTX2("orientation \(value); only \(orientation) is supported")
                }
                if key == "KTXcubemapIncomplete" {
                    throw Failure.unsupportedKTX2("incomplete cube maps")
                }
                if key == "KTXswizzle", value != "rgba" {
                    throw Failure.unsupportedKTX2("swizzle \(value); only rgba is supported")
                }
            }
            offset += (length + 3) & ~3
        }
    }

    /// Inflates an RFC 1950 zlib stream and verifies its Adler-32 checksum.
    private static func inflate(_ stream: Data, expected: Int, level: Int) throws -> Data {
        func invalid(_ reason: String) -> Failure { .invalidKTX2("level \(level) zlib data \(reason)") }
        let header = UInt16(stream[stream.startIndex]) << 8 | UInt16(stream[stream.startIndex + 1])
        guard header & 0x0F00 == 0x0800, header >> 12 <= 7, header % 31 == 0, header & 0x20 == 0 else {
            throw invalid("has an invalid header")
        }
        let deflate = stream.subdata(in: (stream.startIndex + 2)..<(stream.endIndex - 4))
        var output = Data(count: expected + 1)
        let written = output.withUnsafeMutableBytes { destination in
            deflate.withUnsafeBytes { source in
                compression_decode_buffer(
                    destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                    source.bindMemory(to: UInt8.self).baseAddress!, source.count, nil, COMPRESSION_ZLIB
                )
            }
        }
        guard written == expected else { throw invalid("does not inflate to \(expected) bytes") }
        output.removeLast()
        let checksum = stream.suffix(4).reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
        guard adler32(output) == checksum else { throw invalid("fails its Adler-32 check") }
        return output
    }

    static func adler32(_ data: Data) -> UInt32 {
        var a: UInt32 = 1, b: UInt32 = 0
        for chunk in stride(from: 0, to: data.count, by: 5_552) {
            for byte in data[(data.startIndex + chunk)..<(data.startIndex + min(chunk + 5_552, data.count))] {
                a += UInt32(byte)
                b += a
            }
            a %= 65_521
            b %= 65_521
        }
        return b << 16 | a
    }
}

/// A KTX2 `vkFormat` that maps to a Metal pixel format without transcoding.
struct KTX2Format: Equatable {
    let vkFormat: UInt32
    let name: String
    /// The non-sRGB Metal format, used for linear data.
    let linear: MTLPixelFormat
    /// The sRGB Metal format, when the encoding has one.
    let sRGB: MTLPixelFormat?
    /// True when the container declares sRGB-encoded color.
    let isSRGB: Bool
    let blockWidth: Int
    let blockHeight: Int
    let bytesPerBlock: Int
    let typeSize: UInt32
    let family: Family

    enum Family { case uncompressed, bc, appleBlock }

    var isCompressed: Bool { family != .uncompressed }
    var storedPixelFormat: MTLPixelFormat { isSRGB ? (sRGB ?? linear) : linear }

    func byteCount(width: Int, height: Int) -> Int {
        ((width + blockWidth - 1) / blockWidth) * ((height + blockHeight - 1) / blockHeight) * bytesPerBlock
    }

    func bytesPerRow(width: Int) -> Int {
        ((width + blockWidth - 1) / blockWidth) * bytesPerBlock
    }

    func isSupported(by device: any MTLDevice, samplingAs pixelFormat: MTLPixelFormat) -> Bool {
        switch family {
        case .uncompressed:
            #if targetEnvironment(simulator)
            // The Simulator Metal device rejects both narrow sRGB formats.
            return pixelFormat != .r8Unorm_srgb && pixelFormat != .rg8Unorm_srgb
            #else
            return true
            #endif
        case .bc: return device.supportsBCTextureCompression
        case .appleBlock: return device.supportsFamily(.apple2)
        }
    }

    init?(uncompressedPixelFormat: MTLPixelFormat) {
        let formats: [UInt32] = [9, 15, 16, 22, 37, 43, 44, 50, 76, 83, 97, 100, 103, 109]
        guard let match = formats.lazy.compactMap({ KTX2Format(vkFormat: $0) })
            .first(where: { $0.storedPixelFormat == uncompressedPixelFormat }) else { return nil }
        self = match
    }

    init?(vkFormat: UInt32) {
        switch vkFormat {
        case 9, 15:
            self.init(vkFormat, "R8", .r8Unorm, .r8Unorm_srgb, vkFormat == 15, 1, 1, 1, 1, .uncompressed)
        case 16, 22:
            self.init(vkFormat, "RG8", .rg8Unorm, .rg8Unorm_srgb, vkFormat == 22, 1, 1, 2, 1, .uncompressed)
        case 37, 43:
            self.init(vkFormat, "RGBA8", .rgba8Unorm, .rgba8Unorm_srgb, vkFormat == 43, 1, 1, 4, 1, .uncompressed)
        case 44, 50:
            self.init(vkFormat, "BGRA8", .bgra8Unorm, .bgra8Unorm_srgb, vkFormat == 50, 1, 1, 4, 1, .uncompressed)
        case 76:
            self.init(vkFormat, "R16 float", .r16Float, nil, false, 1, 1, 2, 2, .uncompressed)
        case 83:
            self.init(vkFormat, "RG16 float", .rg16Float, nil, false, 1, 1, 4, 2, .uncompressed)
        case 97:
            self.init(vkFormat, "RGBA16 float", .rgba16Float, nil, false, 1, 1, 8, 2, .uncompressed)
        case 100:
            self.init(vkFormat, "R32 float", .r32Float, nil, false, 1, 1, 4, 4, .uncompressed)
        case 103:
            self.init(vkFormat, "RG32 float", .rg32Float, nil, false, 1, 1, 8, 4, .uncompressed)
        case 109:
            self.init(vkFormat, "RGBA32 float", .rgba32Float, nil, false, 1, 1, 16, 4, .uncompressed)
        case 133, 134:
            self.init(vkFormat, "BC1 RGBA", .bc1_rgba, .bc1_rgba_srgb, vkFormat == 134, 4, 4, 8, 1, .bc)
        case 137, 138:
            self.init(vkFormat, "BC3 RGBA", .bc3_rgba, .bc3_rgba_srgb, vkFormat == 138, 4, 4, 16, 1, .bc)
        case 145, 146:
            self.init(vkFormat, "BC7 RGBA", .bc7_rgbaUnorm, .bc7_rgbaUnorm_srgb, vkFormat == 146, 4, 4, 16, 1, .bc)
        case 147, 148:
            self.init(vkFormat, "ETC2 RGB8", .etc2_rgb8, .etc2_rgb8_srgb, vkFormat == 148, 4, 4, 8, 1, .appleBlock)
        case 149, 150:
            self.init(vkFormat, "ETC2 RGB8A1", .etc2_rgb8a1, .etc2_rgb8a1_srgb, vkFormat == 150, 4, 4, 8, 1, .appleBlock)
        case 151, 152:
            self.init(vkFormat, "ETC2 RGBA8", .eac_rgba8, .eac_rgba8_srgb, vkFormat == 152, 4, 4, 16, 1, .appleBlock)
        case 157...184:
            let index = Int(vkFormat - 157) / 2
            let (width, height, linear, sRGB, _) = Self.astc[index]
            self.init(
                vkFormat, "ASTC \(width)×\(height)", linear, sRGB, vkFormat.isMultiple(of: 2), width, height, 16, 1,
                .appleBlock
            )
        default:
            return nil
        }
    }

    private init(
        _ vkFormat: UInt32, _ name: String, _ linear: MTLPixelFormat, _ sRGB: MTLPixelFormat?, _ isSRGB: Bool,
        _ blockWidth: Int, _ blockHeight: Int, _ bytesPerBlock: Int, _ typeSize: UInt32, _ family: Family
    ) {
        self.vkFormat = vkFormat
        self.name = isSRGB ? "\(name) sRGB" : name
        self.linear = linear
        self.sRGB = sRGB
        self.isSRGB = isSRGB
        self.blockWidth = blockWidth
        self.blockHeight = blockHeight
        self.bytesPerBlock = bytesPerBlock
        self.typeSize = typeSize
        self.family = family
    }

    /// ASTC block sizes in KTX2 LDR vkFormat order, with matching Metal HDR formats.
    static let astc: [(width: Int, height: Int, linear: MTLPixelFormat, sRGB: MTLPixelFormat, hdr: MTLPixelFormat)] = [
        (4, 4, .astc_4x4_ldr, .astc_4x4_srgb, .astc_4x4_hdr),
        (5, 4, .astc_5x4_ldr, .astc_5x4_srgb, .astc_5x4_hdr),
        (5, 5, .astc_5x5_ldr, .astc_5x5_srgb, .astc_5x5_hdr),
        (6, 5, .astc_6x5_ldr, .astc_6x5_srgb, .astc_6x5_hdr),
        (6, 6, .astc_6x6_ldr, .astc_6x6_srgb, .astc_6x6_hdr),
        (8, 5, .astc_8x5_ldr, .astc_8x5_srgb, .astc_8x5_hdr),
        (8, 6, .astc_8x6_ldr, .astc_8x6_srgb, .astc_8x6_hdr),
        (8, 8, .astc_8x8_ldr, .astc_8x8_srgb, .astc_8x8_hdr),
        (10, 5, .astc_10x5_ldr, .astc_10x5_srgb, .astc_10x5_hdr),
        (10, 6, .astc_10x6_ldr, .astc_10x6_srgb, .astc_10x6_hdr),
        (10, 8, .astc_10x8_ldr, .astc_10x8_srgb, .astc_10x8_hdr),
        (10, 10, .astc_10x10_ldr, .astc_10x10_srgb, .astc_10x10_hdr),
        (12, 10, .astc_12x10_ldr, .astc_12x10_srgb, .astc_12x10_hdr),
        (12, 12, .astc_12x12_ldr, .astc_12x12_srgb, .astc_12x12_hdr),
    ]
}

/// Classifies four-channel normalized formats that sample as float RGBA, so material
/// validation can accept block-compressed maps with the same color or data semantics.
enum MaterialTextureFormat {
    /// 32-bit float formats, which only some GPUs filter linearly (`supports32BitFloatFiltering`).
    static func isFloat32(_ format: MTLPixelFormat) -> Bool {
        format == .r32Float || format == .rg32Float || format == .rgba32Float
    }

    /// True for formats that decode sRGB on sample.
    static func compressedEncoding(_ format: MTLPixelFormat) -> (isCompressed: Bool, isSRGB: Bool) {
        if let sRGB = compressedFormats[format] { return (true, sRGB) }
        return (false, false)
    }

    private static let compressedFormats: [MTLPixelFormat: Bool] = {
        var result: [MTLPixelFormat: Bool] = [:]
        for vkFormat in UInt32(133)...184 {
            guard let format = KTX2Format(vkFormat: vkFormat), format.isCompressed else { continue }
            result[format.linear] = false
            if let sRGB = format.sRGB { result[sRGB] = true }
        }
        return result
    }()
}
