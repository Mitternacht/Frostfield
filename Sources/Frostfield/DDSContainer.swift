import Foundation
import Metal

/// A validated face-major DDS image, with tightly stored blocks or byte rows.
struct DDSContainer {
    typealias Failure = TextureLoader.Failure

    static func hasIdentifier(_ data: Data) -> Bool {
        data.starts(with: [0x44, 0x44, 0x53, 0x20])
    }

    enum Format: String {
        case bc1 = "DXT1"
        case bc2 = "DXT3"
        case bc3 = "DXT5"
        case etc1 = "ETC1"
        case bc6HSigned = "BC6H signed"
        case bc6HUnsigned = "BC6H unsigned"
        case bgr = "BGR24"
        case bgra = "BGRA32"

        var isCompressed: Bool { self != .bgr && self != .bgra }

        var bytesPerBlock: Int {
            switch self {
            case .bgr: 3
            case .bgra: 4
            case .bc1, .etc1: 8
            default: 16
            }
        }

        func pixelFormat(sRGB: Bool) -> MTLPixelFormat {
            switch self {
            case .bc1: sRGB ? .bc1_rgba_srgb : .bc1_rgba
            case .bc2: sRGB ? .bc2_rgba_srgb : .bc2_rgba
            case .bc3: sRGB ? .bc3_rgba_srgb : .bc3_rgba
            case .etc1: sRGB ? .etc2_rgb8_srgb : .etc2_rgb8
            case .bc6HSigned: .bc6H_rgbFloat
            case .bc6HUnsigned: .bc6H_rgbuFloat
            case .bgr, .bgra: sRGB ? .rgba8Unorm_srgb : .rgba8Unorm
            }
        }

        func rowBytes(width: Int) -> Int {
            (isCompressed ? (width + 3) / 4 : width) * bytesPerBlock
        }

        func rowCount(height: Int) -> Int { isCompressed ? (height + 3) / 4 : height }
    }

    struct Surface {
        let width: Int
        let height: Int
        let rowBytes: Int
        let range: Range<Int>
    }

    let format: Format
    let width: Int
    let height: Int
    let mipCount: Int
    let faceCount: Int
    let surfaces: [Surface]
    let data: Data

    init(_ source: Data, allowingCube: Bool) throws {
        let data = Data(source)
        guard Self.hasIdentifier(data), data.count >= 128 else { throw Failure.invalidDDS("the header is truncated") }
        func word(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }

        guard word(4) == 124, word(76) == 32 else { throw Failure.invalidDDS("the header sizes are invalid") }
        let width = Int(word(16)), height = Int(word(12))
        let flags = word(8), caps = word(112)
        guard caps & 0x20_0000 == 0, word(24) <= 1 else { throw Failure.unsupportedDDS("volume textures") }
        let legacyCube = caps & 0x200 != 0
        let cubeFaces: UInt32 = 0xFC00
        guard legacyCube ? caps & cubeFaces == cubeFaces : caps & cubeFaces == 0 else {
            throw Failure.invalidDDS("the cube face flags are incomplete")
        }

        let format: Format
        var isCube = legacyCube
        var payloadOffset = 128
        if word(80) & 4 != 0 {
            let fourCC = String(decoding: data[84..<88], as: UTF8.self)
            switch fourCC {
            case "DXT1": format = .bc1
            case "DXT3": format = .bc2
            case "DXT5": format = .bc3
            case "ETC1": format = .etc1
            case "DX10":
                guard data.count >= 148 else { throw Failure.invalidDDS("the DX10 header is truncated") }
                guard word(132) == 3 else { throw Failure.unsupportedDDS("DX10 resources other than 2D and cube textures") }
                guard word(140) == 1 else { throw Failure.unsupportedDDS("DX10 texture arrays") }
                isCube = word(136) & 4 != 0
                guard !legacyCube || isCube else { throw Failure.invalidDDS("the cube flags disagree") }
                let dxgi = word(128)
                switch dxgi {
                case 95: format = .bc6HUnsigned
                case 96: format = .bc6HSigned
                default: throw Failure.unsupportedDDS("DXGI format \(dxgi)")
                }
                payloadOffset = 148
            default:
                throw Failure.unsupportedDDS("FourCC \(fourCC.debugDescription)")
            }
        } else {
            guard word(80) & 0x40 != 0,
                  word(92) == 0xFF0000, word(96) == 0xFF00, word(100) == 0xFF else {
                throw Failure.unsupportedDDS("RGB channel masks")
            }
            switch (word(88), word(104)) {
            case (24, 0): format = .bgr
            case (32, 0xFF000000): format = .bgra
            default: throw Failure.unsupportedDDS("the uncompressed pixel layout")
            }
        }

        guard allowingCube || !isCube else { throw Failure.unsupportedDDS("cube maps through the 2D loader") }
        guard !isCube || width == height else { throw Failure.invalidDDS("cube faces must be square") }
        let faceCount = isCube ? 6 : 1
        guard KTX2Container.dimensionsFit(width: width, height: height, depth: 0, faces: faceCount) else {
            throw Failure.imageDimensions(width: width, height: height)
        }
        let mipCount = flags & 0x20_000 != 0 ? max(1, Int(word(28))) : 1
        let maximumMips = Int.bitWidth - max(width, height).leadingZeroBitCount
        guard mipCount <= maximumMips else { throw Failure.invalidDDS("the mip count exceeds a full chain") }

        let sizes = (0..<mipCount).map { (max(1, width >> $0), max(1, height >> $0)) }
        let tight = sizes.map { format.rowBytes(width: $0.0) }
        var rowCandidates = [tight]
        if !format.isCompressed {
            let aligned = tight.map { ($0 + 3) & ~3 }
            let pitch = flags & 8 != 0 ? Int(word(20)) : 0
            if aligned != tight, pitch == 0 || pitch == aligned[0] { rowCandidates.append(aligned) }
            if pitch > tight[0], pitch <= data.count { rowCandidates.append([pitch] + tight.dropFirst()) }
        }

        let payloadBytes = data.count - payloadOffset
        let rowCounts = sizes.map { format.rowCount(height: $0.1) }
        guard let rows = rowCandidates.first(where: { bytesPerRow in
            zip(bytesPerRow, rowCounts).reduce(0) { $0 + $1.0 * $1.1 } * faceCount == payloadBytes
        }) else { throw Failure.invalidDDS("the payload length does not match the faces and levels") }

        var cursor = payloadOffset
        var surfaces: [Surface] = []
        for _ in 0..<faceCount {
            for level in 0..<mipCount {
                let end = cursor + rows[level] * rowCounts[level]
                let (w, h) = sizes[level]
                surfaces.append(Surface(width: w, height: h, rowBytes: rows[level], range: cursor..<end))
                cursor = end
            }
        }
        self.format = format
        self.width = width
        self.height = height
        self.mipCount = mipCount
        self.faceCount = faceCount
        self.surfaces = surfaces
        self.data = data
    }
}
