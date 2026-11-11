import Foundation

/// PVR v2/v3 with one PVRTC1 image or a complete cube.
struct PVRContainer {
    typealias Failure = TextureLoader.Failure

    private static let v3: UInt32 = 0x03525650
    private static let v2: UInt32 = 0x21525650
    private static let headerSize = 52
    private static let cubeFaces = Array("XxYyZz".utf8)

    static func hasIdentifier(_ data: Data) -> Bool {
        data.withUnsafeBytes { bytes in
            guard bytes.count >= 4 else { return false }
            let first = UInt32(littleEndian: bytes.loadUnaligned(as: UInt32.self))
            return first == v3 || first == v3.byteSwapped ||
                (first == UInt32(headerSize) && bytes.count >= headerSize &&
                 UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: 44, as: UInt32.self)) == v2)
        }
    }

    let data: Data
    let format: PVRTC1Format
    let isSRGB: Bool
    let faceCount: Int
    let levels: [StoredTextureLevel]

    init(_ source: Data, allowingCube: Bool) throws {
        guard source.count >= Self.headerSize else { throw Failure.invalidPVR("the header is truncated") }
        let data = Data(source)
        let marker = data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) }
        let bigEndian = marker == Self.v3.byteSwapped
        func word(_ offset: Int) -> UInt32 {
            let value = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
            return bigEndian ? UInt32(bigEndian: value) : UInt32(littleEndian: value)
        }

        let width: Int, height: Int, mipCount: Int, faceCount: Int, payloadStart: Int
        let format: PVRTC1Format, isSRGB: Bool
        var faceOrder = [0, 1, 2, 3, 4, 5]
        var faceMajor = false
        var legacyPayloadSize: Int?
        if word(0) == Self.v3 {
            let flags = word(4)
            guard flags == 0 else { throw Failure.unsupportedPVR("v3 flags \(flags)") }
            let pixelFormat = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self) }
            let formatID = bigEndian ? UInt64(bigEndian: pixelFormat) : UInt64(littleEndian: pixelFormat)
            guard let id = UInt32(exactly: formatID), let compressed = PVRTC1Format(rawValue: id) else {
                throw Failure.unsupportedPVR("v3 pixel formats other than PVRTC1")
            }
            guard word(16) <= 1, word(20) == 0 else {
                throw Failure.unsupportedPVR("v3 color space or channel type")
            }
            guard word(32) == 1, word(36) == 1 else { throw Failure.unsupportedPVR("arrays and volumes") }
            width = Int(word(28))
            height = Int(word(24))
            mipCount = Int(word(44))
            faceCount = Int(word(40))
            format = compressed
            isSRGB = word(16) == 1
            let metadataSize = Int(word(48))
            guard metadataSize <= data.count - Self.headerSize else {
                throw Failure.invalidPVR("the metadata length exceeds the file")
            }
            payloadStart = Self.headerSize + metadataSize
            var cursor = Self.headerSize
            var seenFaceOrder = false
            while cursor < payloadStart {
                guard payloadStart - cursor >= 12 else { throw Failure.invalidPVR("a metadata header is truncated") }
                let key = word(cursor + 4)
                let count = Int(word(cursor + 8))
                guard count <= payloadStart - cursor - 12 else {
                    throw Failure.invalidPVR("a metadata block exceeds the declared metadata")
                }
                let start = cursor + 12
                let bytes = data[start..<(start + count)]
                // FourCC is specified as bytes. Also accept its integer
                // representation when the surrounding header is big-endian.
                if word(cursor) == Self.v3 || data[cursor..<(cursor + 4)].elementsEqual([0x50, 0x56, 0x52, 0x03]) {
                    switch key {
                    case 2:
                        guard !seenFaceOrder, faceCount == 6, count == 6,
                              Set(bytes) == Set(Self.cubeFaces) else {
                            throw Failure.invalidPVR("the cube face order is invalid")
                        }
                        faceOrder = Self.cubeFaces.compactMap { axis in bytes.firstIndex(of: axis).map { $0 - start } }
                        seenFaceOrder = true
                    case 3:
                        guard count == 3 else { throw Failure.invalidPVR("the orientation length is invalid") }
                        guard bytes.allSatisfy({ $0 == 0 }) else { throw Failure.unsupportedPVR("alternate orientation") }
                    case 4:
                        guard count == 12 else { throw Failure.invalidPVR("the border length is invalid") }
                        guard bytes.allSatisfy({ $0 == 0 }) else { throw Failure.unsupportedPVR("texture borders") }
                    case 6, 7:
                        throw Failure.unsupportedPVR("per-channel types and supercompression metadata")
                    default: break
                    }
                }
                cursor = start + count
            }
        } else {
            guard marker == UInt32(Self.headerSize), word(44) == Self.v2 else {
                throw Failure.invalidPVR("the version marker or legacy header size is invalid")
            }
            let flags = word(16)
            let encoding = flags & 0xFF
            guard encoding == 24 || encoding == 25 else { throw Failure.unsupportedPVR("v2 pixel format \(encoding)") }
            // Allow legacy mipmap, twiddle, cube and alpha flags.
            guard flags & ~UInt32(0x93FF) == 0 else { throw Failure.unsupportedPVR("v2 flags \(flags)") }
            let bitCount: UInt32 = encoding == 24 ? 2 : 4
            guard word(24) == bitCount else { throw Failure.invalidPVR("the v2 bit count does not match the encoding") }
            let hasAlpha = word(40) != 0
            format = encoding == 24 ? (hasAlpha ? .rgba2 : .rgb2) : (hasAlpha ? .rgba4 : .rgb4)
            isSRGB = false
            width = Int(word(8))
            height = Int(word(4))
            mipCount = Int(word(12)) + 1
            faceCount = Int(word(48))
            payloadStart = Self.headerSize
            legacyPayloadSize = Int(word(20))
            // Explicit legacy cubemaps store each face's mip chain together.
            // Without the cube flag, six surfaces use the r187 level-major layout.
            faceMajor = flags & 0x1000 != 0
            guard !faceMajor || faceCount == 6 else { throw Failure.invalidPVR("the v2 cube flag requires six surfaces") }
        }

        guard faceCount == 1 || faceCount == 6 else { throw Failure.unsupportedPVR("face or surface counts other than one or six") }
        guard allowingCube || faceCount == 1 else { throw Failure.unsupportedPVR("cube maps through the 2D loader") }
        guard faceCount == 1 || width == height else { throw Failure.invalidPVR("cube faces must be square") }
        guard KTX2Container.dimensionsFit(width: width, height: height, depth: 0, faces: faceCount) else {
            throw Failure.imageDimensions(width: width, height: height)
        }
        guard width.nonzeroBitCount == 1, height.nonzeroBitCount == 1 else {
            throw Failure.invalidPVR("PVRTC1 requires power-of-two dimensions")
        }
        guard mipCount > 0, mipCount <= Int.bitWidth - max(width, height).leadingZeroBitCount else {
            throw Failure.invalidPVR("the mip count must be within a full chain")
        }

        let payloadCount = data.count - payloadStart
        var mipOffset = 0
        var shapes: [(width: Int, height: Int, count: Int, offset: Int)] = []
        for mip in 0..<mipCount {
            let w = max(1, width >> mip), h = max(1, height >> mip)
            let count = format.byteCount(width: w, height: h)
            shapes.append((w, h, count, mipOffset))
            mipOffset += count
        }
        let bytesPerFace = mipOffset
        guard payloadCount == bytesPerFace * faceCount else {
            throw Failure.invalidPVR("the file length does not match the complete mip and face payload")
        }
        if let legacyPayloadSize, legacyPayloadSize != payloadCount,
           !(faceMajor && legacyPayloadSize == bytesPerFace) {
            throw Failure.invalidPVR("the v2 data length does not match the payload")
        }

        self.data = data
        self.format = format
        self.isSRGB = isSRGB
        self.faceCount = faceCount
        self.levels = shapes.map { shape in
            let faces = faceOrder.prefix(faceCount).map { face in
                let offset = faceMajor ? face * bytesPerFace + shape.offset : shape.offset * faceCount + face * shape.count
                let start = payloadStart + offset
                return start..<(start + shape.count)
            }
            return StoredTextureLevel(width: shape.width, height: shape.height, faces: faces)
        }
    }
}
