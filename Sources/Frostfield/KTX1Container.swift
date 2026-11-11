import Foundation
internal import FrostfieldBasis
import Metal

/// A KTX 1.0 compressed 2D image or complete cube, with level-major surfaces.
struct KTX1Container {
    typealias Failure = TextureLoader.Failure

    static let identifier: [UInt8] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]

    static func hasIdentifier(_ data: Data) -> Bool { data.starts(with: identifier) }

    let format: KTX1Format
    let width: Int
    let height: Int
    let faceCount: Int
    let levels: [StoredTextureLevel]
    let data: Data

    init(_ source: Data, allowingCube: Bool) throws {
        let data = Data(source)
        guard Self.hasIdentifier(data), data.count >= 64 else { throw Failure.invalidKTX1("the header is truncated") }
        let marker = data[12..<16]
        let littleEndian: Bool
        if marker.elementsEqual([1, 2, 3, 4]) {
            littleEndian = true
        } else if marker.elementsEqual([4, 3, 2, 1]) {
            littleEndian = false
        } else {
            throw Failure.invalidKTX1("the byte-order marker is invalid")
        }
        func word(_ offset: Int) -> UInt32 {
            let value = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
            return littleEndian ? UInt32(littleEndian: value) : UInt32(bigEndian: value)
        }

        guard word(16) == 0, word(24) == 0 else { throw Failure.unsupportedKTX1("uncompressed textures") }
        guard word(20) == 1 else { throw Failure.invalidKTX1("compressed textures require glTypeSize 1") }
        let gl = word(28)
        guard let format = KTX1Format(gl: gl) else { throw Failure.unsupportedKTX1("GL internal format \(gl)") }
        guard word(32) == format.base else { throw Failure.invalidKTX1("the GL base format does not match the encoding") }
        let width = Int(word(36)), height = Int(word(40)), faceCount = Int(word(52))
        guard height > 0, word(44) == 0 else { throw Failure.unsupportedKTX1("1D and 3D textures") }
        guard word(48) == 0 else { throw Failure.unsupportedKTX1("texture arrays") }
        guard faceCount == 1 || faceCount == 6 else { throw Failure.invalidKTX1("face count must be one or six") }
        guard allowingCube || faceCount == 1 else { throw Failure.unsupportedKTX1("cube maps through the 2D loader") }
        guard faceCount == 1 || width == height else { throw Failure.invalidKTX1("cube faces must be square") }
        guard KTX2Container.dimensionsFit(width: width, height: height, depth: 0, faces: faceCount) else {
            throw Failure.imageDimensions(width: width, height: height)
        }
        if format.pvrtc != nil, width.nonzeroBitCount != 1 || height.nonzeroBitCount != 1 {
            throw Failure.invalidKTX1("PVRTC1 requires power-of-two dimensions")
        }
        let mipCount = max(1, Int(word(56)))
        guard mipCount <= Int.bitWidth - max(width, height).leadingZeroBitCount else {
            throw Failure.invalidKTX1("the mip count exceeds a full chain")
        }

        let metadataBytes = Int(word(60))
        guard metadataBytes.isMultiple(of: 4), metadataBytes <= data.count - 64 else {
            throw Failure.invalidKTX1("the metadata length is invalid")
        }
        var cursor = 64
        let metadataEnd = cursor + metadataBytes
        while cursor < metadataEnd {
            let size = Int(word(cursor))
            cursor += 4
            let padded = (size + 3) & ~3
            guard size > 0, padded <= metadataEnd - cursor else {
                throw Failure.invalidKTX1("a metadata entry is truncated")
            }
            let end = cursor + size
            guard data[end..<(cursor + padded)].allSatisfy({ $0 == 0 }),
                  let separator = data[cursor..<end].firstIndex(of: 0), separator > cursor,
                  let key = String(bytes: data[cursor..<separator], encoding: .utf8) else {
                throw Failure.invalidKTX1("a metadata entry has an invalid key or padding")
            }
            let value = data[(separator + 1)..<end].prefix { $0 != 0 }
            if key.lowercased() == "ktxorientation" {
                let orientation = String(decoding: value, as: UTF8.self)
                guard ["S=r,T=d", "S=r,T=d,R=i", "S=r,T=d,R=o"].contains(orientation) else {
                    throw Failure.unsupportedKTX1("orientation \(orientation)")
                }
            }
            if key == "KTXswizzle", !value.elementsEqual("rgba".utf8) {
                throw Failure.unsupportedKTX1("nonidentity channel swizzles")
            }
            cursor += padded
        }

        var levels: [StoredTextureLevel] = []
        for mip in 0..<mipCount {
            guard data.count - cursor >= 4 else { throw Failure.invalidKTX1("a mip header is truncated") }
            let imageSize = Int(word(cursor))
            cursor += 4
            let mipWidth = max(1, width >> mip), mipHeight = max(1, height >> mip)
            guard imageSize == format.byteCount(width: mipWidth, height: mipHeight) else {
                throw Failure.invalidKTX1("mip \(mip) imageSize does not match a face")
            }
            // Every supported encoding stores eight or sixteen bytes per block.
            guard imageSize <= (data.count - cursor) / faceCount else {
                throw Failure.invalidKTX1("mip \(mip) face data is truncated")
            }
            var faces: [Range<Int>] = []
            for _ in 0..<faceCount {
                faces.append(cursor..<(cursor + imageSize))
                cursor += imageSize
            }
            levels.append(StoredTextureLevel(width: mipWidth, height: mipHeight, faces: faces))
        }
        guard cursor == data.count else { throw Failure.invalidKTX1("unexpected bytes after the last mip") }

        self.format = format
        self.width = width
        self.height = height
        self.faceCount = faceCount
        self.levels = levels
        self.data = data
    }
}

struct KTX1Format {
    let base: UInt32
    let linear: MTLPixelFormat
    let sRGB: MTLPixelFormat?
    let isSRGB: Bool
    let blockWidth: Int
    let blockHeight: Int
    let bytesPerBlock: Int
    let pvrtc: PVRTC1Format?
    let bc: FrostfieldBCFormat?

    func bytesPerRow(width: Int) -> Int {
        ((width + blockWidth - 1) / blockWidth) * bytesPerBlock
    }

    func byteCount(width: Int, height: Int) -> Int {
        if let pvrtc { return pvrtc.byteCount(width: width, height: height) }
        return bytesPerRow(width: width) * ((height + blockHeight - 1) / blockHeight)
    }

    init?(gl: UInt32) {
        let rgb: UInt32 = 0x1907, rgba: UInt32 = 0x1908, red: UInt32 = 0x1903, rg: UInt32 = 0x8227
        let sRGB = (0x8C4C...0x8C4F).contains(gl)
        let s3tc = sRGB ? gl - 0x8C4C + 0x83F0 : gl
        switch s3tc {
        case 0x83F0, 0x83F1:
            self.init(s3tc == 0x83F0 ? rgb : rgba, .bc1_rgba, .bc1_rgba_srgb, sRGB, 8,
                      bc: s3tc == 0x83F0 ? FrostfieldBC1Opaque : FrostfieldBC1RGBA)
            return
        case 0x83F2:
            self.init(rgba, .bc2_rgba, .bc2_rgba_srgb, sRGB, 16, bc: FrostfieldBC2)
            return
        case 0x83F3:
            self.init(rgba, .bc3_rgba, .bc3_rgba_srgb, sRGB, 16, bc: FrostfieldBC3)
            return
        default: break
        }

        switch gl {
        case 0x8C00: self.init(rgb, pvrtc: .rgb4)
        case 0x8C01: self.init(rgb, pvrtc: .rgb2)
        case 0x8C02: self.init(rgba, pvrtc: .rgba4)
        case 0x8C03: self.init(rgba, pvrtc: .rgba2)
        case 0x8D64: self.init(rgb, vk: 147)
        case 0x9270: self.init(red, .eac_r11Unorm, nil, false, 8)
        case 0x9271: self.init(red, .eac_r11Snorm, nil, false, 8)
        case 0x9272: self.init(rg, .eac_rg11Unorm, nil, false, 16)
        case 0x9273: self.init(rg, .eac_rg11Snorm, nil, false, 16)
        case 0x9274, 0x9275: self.init(rgb, vk: 147 + gl - 0x9274)
        case 0x9276, 0x9277: self.init(rgba, vk: 149 + gl - 0x9276)
        case 0x9278, 0x9279: self.init(rgba, vk: 151 + gl - 0x9278)
        case 0x8DBB: self.init(red, .bc4_rUnorm, nil, false, 8, bc: FrostfieldBC4)
        case 0x8DBC: self.init(red, .bc4_rSnorm, nil, false, 8, bc: FrostfieldBC4Signed)
        case 0x8DBD: self.init(rg, .bc5_rgUnorm, nil, false, 16, bc: FrostfieldBC5)
        case 0x8DBE: self.init(rg, .bc5_rgSnorm, nil, false, 16, bc: FrostfieldBC5Signed)
        case 0x8E8C, 0x8E8D:
            self.init(rgba, .bc7_rgbaUnorm, .bc7_rgbaUnorm_srgb, gl == 0x8E8D, 16, bc: FrostfieldBC7)
        case 0x8E8E: self.init(rgb, .bc6H_rgbFloat, nil, false, 16, bc: FrostfieldBC6HSigned)
        case 0x8E8F: self.init(rgb, .bc6H_rgbuFloat, nil, false, 16, bc: FrostfieldBC6HUnsigned)
        case 0x93B0...0x93BD: self.init(rgba, vk: 157 + 2 * (gl - 0x93B0))
        case 0x93D0...0x93DD: self.init(rgba, vk: 158 + 2 * (gl - 0x93D0))
        default: return nil
        }
    }

    private init(_ base: UInt32, pvrtc: PVRTC1Format) {
        self.init(base, pvrtc.pixelFormat(sRGB: false), pvrtc.pixelFormat(sRGB: true), false, 8,
                  width: pvrtc.minimumWidth / 2, pvrtc: pvrtc)
    }

    private init?(_ base: UInt32, vk: UInt32) {
        guard let ktx = KTX2Format(vkFormat: vk) else { return nil }
        self.init(base, ktx.linear, ktx.sRGB, ktx.isSRGB, ktx.bytesPerBlock,
                  width: ktx.blockWidth, height: ktx.blockHeight)
    }

    private init(
        _ base: UInt32, _ linear: MTLPixelFormat, _ sRGB: MTLPixelFormat?, _ isSRGB: Bool, _ bytes: Int,
        width: Int = 4, height: Int = 4, pvrtc: PVRTC1Format? = nil, bc: FrostfieldBCFormat? = nil
    ) {
        self.base = base
        self.linear = linear
        self.sRGB = sRGB
        self.isSRGB = isSRGB
        blockWidth = width
        blockHeight = height
        bytesPerBlock = bytes
        self.pvrtc = pvrtc
        self.bc = bc
    }
}
