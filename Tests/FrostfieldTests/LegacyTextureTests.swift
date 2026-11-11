import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LegacyTextureTests {
    private let loader: TextureLoader

    init() throws {
        loader = TextureLoader(device: try #require(MTLCreateSystemDefaultDevice()))
    }

    private static let rgb: [[UInt8]] = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
    ]

    @Test("TGA storage origins all produce the same top-left texels", arguments: [UInt8(0), 16, 32, 48])
    func targaOrigins(flags: UInt8) throws {
        let rows = flags & 32 == 0 ? [1, 0] : [0, 1]
        let columns = flags & 16 == 0 ? [0, 1, 2] : [2, 1, 0]
        let payload = rows.flatMap { row in columns.flatMap { column in Self.rgb[3 * row + column].reversed() } }
        let texture = try loader.load(data: targa(kind: 2, depth: 24, flags: flags, pixels: payload), interpretation: .linearData)
        #expect(texture.width == 3 && texture.height == 2)
        #expect(rgba(texture) == Self.rgb.flatMap { $0 + [255] })
    }

    @Test("TGA RLE raw packets can cross rows and BGRX remains opaque")
    func targaPacketBoundaries() throws {
        let payload = Self.rgb.flatMap { $0.reversed() }
        let rle = targa(kind: 10, depth: 24, pixels: [5] + payload)
        func bgrx(_ order: [Int]) -> [UInt8] {
            order.flatMap { index -> [UInt8] in
                let rgb = Self.rgb[index]
                return [rgb[2], rgb[1], rgb[0], 0]
            }
        }
        let files = [
            rle, targa(kind: 2, depth: 32, pixels: bgrx([0, 1, 2, 3, 4, 5])),
            targa(kind: 2, depth: 32, flags: 0x30, pixels: bgrx([2, 1, 0, 5, 4, 3])),
        ]
        for data in files {
            for interpretation in [TextureLoader.Interpretation.linearData, .color] {
                let texture = try loader.load(data: data, interpretation: interpretation)
                #expect(rgba(texture) == Self.rgb.flatMap { $0 + [255] })
            }
        }
    }

    @Test("TGA indexed and grayscale packets preserve palette offsets and alpha")
    func targaIndexedAndGray() throws {
        let palette = Self.rgb.prefix(3).flatMap { $0.reversed() }
        let raw = targa(kind: 1, depth: 8, first: 17, palette: palette, pixels: [17, 18, 19, 19, 18, 17])
        let run = targa(kind: 9, depth: 8, first: 17, palette: palette, pixels: [0x81, 17, 3, 19, 19, 18, 17])
        for (data, expected) in [(raw, [0, 1, 2, 2, 1, 0]), (run, [0, 0, 2, 2, 1, 0])] {
            let texture = try loader.load(data: data, interpretation: .encodedSRGB)
            #expect(texture.pixelFormat == .rgba8Unorm_srgb)
            #expect(rgba(texture) == expected.flatMap { Self.rgb[$0] + [255] })
        }

        let grayscale: [UInt8] = [30, 30, 100, 180, 220, 255]
        let alpha: [UInt8] = [100, 100, 1, 0, 200, 255]
        let grayRun = targa(kind: 11, depth: 8, pixels: [0x81, 30, 3, 100, 180, 220, 255])
        let grayAlphaRun = targa(kind: 11, depth: 16, flags: 0x28,
                                 pixels: [0x81, 30, 100, 3, 100, 1, 180, 0, 220, 200, 255, 255])
        let grayAlphaRaw = targa(kind: 3, depth: 16, flags: 0x28,
                                 pixels: zip(grayscale, alpha).flatMap { [$0, $1] })
        let gray = try loader.load(data: grayRun, interpretation: .linearData)
        #expect(rgba(gray) == grayscale.flatMap { [$0, $0, $0, 255] })
        for data in [grayAlphaRun, grayAlphaRaw] {
            let grayWithAlpha = try loader.load(data: data, interpretation: .linearData)
            #expect(rgba(grayWithAlpha) == zip(grayscale, alpha).flatMap { [$0, $0, $0, $1] })
        }
    }

    @Test("TGA truecolor retains r187 RGB555 and straight alpha semantics",
          arguments: [TextureLoader.Interpretation.color, .linearData, .encodedSRGB])
    func targaTruecolor(interpretation: TextureLoader.Interpretation) throws {
        let rgb555 = targa(kind: 10, depth: 16, flags: 0x21,
                           pixels: [0x81, 0x00, 0x7C, 3, 0x1F, 0, 0, 0xFC, 0xE0, 0x83, 0x1F, 0x80])
        let texture = try loader.load(data: rgb555, interpretation: interpretation)
        let opaque: [UInt8] = [248, 0, 0, 255, 248, 0, 0, 255, 0, 0, 248, 255]
        let hidden: [UInt8] = [248, 0, 0, 0, 0, 248, 0, 0, 0, 0, 248, 0]
        #expect(rgba(texture) == opaque + (interpretation == .color ? Array(repeating: 0, count: 12) : hidden))

        let bgrRun = targa(kind: 10, depth: 32, flags: 0x28, id: [78, 65, 77, 69],
                           pixels: [0x81, 50, 100, 200, 1, 3, 255, 0, 0, 200, 0, 255, 0, 255, 0, 0, 255, 1, 17, 20, 29, 0],
                           footer: true)
        let rgbaRun = try loader.load(data: bgrRun, interpretation: interpretation, mipmapped: true)
        let first: [UInt8] = [200, 100, 50, 1, 200, 100, 50, 1, 0, 0, 255, 200, 0, 255, 0, 255, 255, 0, 0, 1]
        #expect(rgba(rgbaRun) == first + (interpretation == .color ? [0, 0, 0, 0] : [29, 20, 17, 0]))
        #expect(rgbaRun.mipmapLevelCount == 2)
        #expect(rgbaRun.pixelFormat == (interpretation == .linearData ? .rgba8Unorm : .rgba8Unorm_srgb))
    }

    @Test("Malformed native TGA payloads fail before texture upload")
    func malformedTarga() throws {
        let badIndex = targa(kind: 1, depth: 8, first: 17, palette: [0, 0, 255],
                             pixels: [17, 17, 17, 17, 16, 17])
        let excessRun = targa(kind: 11, depth: 8, pixels: [0x86, 99])
        let shortRawPacket = targa(kind: 11, depth: 8, pixels: [5, 1, 2, 3])
        let shortID = targa(kind: 10, depth: 32, flags: 0x28, id: [1, 2, 3, 4], pixels: [])
        var badFooter = targa(kind: 11, depth: 8, pixels: [0x85, 90], footer: true)
        badFooter[badFooter.count - 26] = 2
        for data in [badIndex, excessRun, shortRawPacket, shortID, badFooter] {
            #expect(throws: TextureLoader.Failure.invalidImage) { try loader.load(data: data) }
        }
        let oversized = targa(kind: 11, depth: 8, width: 8_193, height: 1, pixels: [0x85, 90], footer: true)
        #expect(throws: TextureLoader.Failure.imageDimensions(width: 8_193, height: 1)) {
            try loader.load(data: oversized)
        }
    }

    @Test("TIFF color applies orientation while encoded and linear data retain storage order")
    func tiffOrientation() throws {
        // Three by two RGB TIFF with orientation 6, clockwise ninety degrees.
        let data = try #require(Data(base64Encoded:
            "SUkqAAgAAAALAAABBAABAAAAAwAAAAEBBAABAAAAAgAAAAIBAwADAAAAkgAAAAMBAwABAAAAAQAAAAYBAwABAAAAAgAAABEBBAABAAAAmAAAABIBAwABAAAABgAAABUBAwABAAAAAwAAABYBBAABAAAAAgAAABcBBAABAAAAEgAAABwBAwABAAAAAQAAAAAAAAAIAAgACAD/AAAA/wAAAP///wD/AP8A//8="
        ))
        let color = try loader.load(data: data, interpretation: .color, mipmapped: true)
        #expect(color.width == 2 && color.height == 3 && color.mipmapLevelCount == 2)
        #expect(color.pixelFormat == .rgba8Unorm_srgb)
        #expect(rgba(color) == [3, 0, 4, 1, 5, 2].flatMap { Self.rgb[$0] + [255] })
        let flipped = try loader.load(data: data, interpretation: .color, uploadOptions: .init(flipY: true))
        #expect(flipped.width == 2 && flipped.height == 3)
        #expect(rgba(flipped) == [5, 2, 4, 1, 3, 0].flatMap { Self.rgb[$0] + [255] })
        for mode in [TextureLoader.Interpretation.encodedSRGB, .linearData] {
            let texture = try loader.load(data: data, interpretation: mode)
            #expect(texture.width == 3 && texture.height == 2 && texture.mipmapLevelCount == 1)
            #expect(texture.pixelFormat == (mode == .linearData ? .rgba8Unorm : .rgba8Unorm_srgb))
            #expect(rgba(texture) == Self.rgb.flatMap { $0 + [255] })
        }
    }

    @Test("ImageIO loads TIFF PackBits, palette and white-is-zero grayscale texels")
    func tiffLayouts() throws {
        let packBits = try #require(Data(base64Encoded:
            "TU0AKgAAAAgACwEAAAQAAAABAAAAAwEBAAQAAAABAAAAAgECAAMAAAADAAAAkgEDAAMAAAABgAUAAAEGAAMAAAABAAIAAAERAAQAAAABAAAAmAESAAMAAAABAAEAAAEVAAMAAAABAAMAAAEWAAQAAAABAAAAAgEXAAQAAAABAAAAEwEcAAMAAAABAAEAAAAAAAAACAAIAAgR/wAAAP8AAAD///8A/wD/AP//"
        ))
        let palette = try #require(Data(base64Encoded:
            "TU0AKgAAAAgADAEAAAQAAAABAAAAAgEBAAQAAAABAAAAAgECAAMAAAABAAIAAAEDAAMAAAABAAEAAAEGAAMAAAABAAMAAAERAAQAAAABAAAAtgESAAMAAAABAAEAAAEVAAMAAAABAAEAAAEWAAQAAAABAAAAAgEXAAQAAAABAAAAAgEcAAMAAAABAAEAAAFAAAMAAAAMAAAAngAAAAD//wAAAAD//wAA//8AAP//AAAAAP//AAAQsA=="
        ))
        let gray = try #require(Data(base64Encoded:
            "SUkqAAgAAAALAAABBAABAAAAAwAAAAEBBAABAAAAAgAAAAIBAwABAAAACAAAAAMBAwABAAAAAQAAAAYBAwABAAAAAAAAABEBBAABAAAAkgAAABIBAwABAAAAAQAAABUBAwABAAAAAQAAABYBBAABAAAAAgAAABcBBAABAAAABgAAABwBAwABAAAAAQAAAAAAAAAAHmS03P8="
        ))
        #expect(rgba(try loader.load(data: packBits, interpretation: .linearData)) == Self.rgb.flatMap { $0 + [255] })
        #expect(rgba(try loader.load(data: palette, interpretation: .linearData)) == Self.rgb.prefix(4).flatMap { $0 + [255] })
        let expected: [UInt8] = [255, 225, 155, 75, 35, 0].flatMap { [$0, $0, $0, 255] }
        #expect(rgba(try loader.load(data: gray, interpretation: .linearData)) == expected)
    }

    private func rgba(_ texture: any MTLTexture) -> [UInt8] {
        let width = texture.width, height = texture.height
        var result = [UInt8](repeating: 0, count: width * height * 4)
        result.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!, bytesPerRow: width * 4,
                             from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        return result
    }

    private func targa(
        kind: UInt8, depth: UInt8, flags: UInt8 = 0x20, width: Int = 3, height: Int = 2,
        first: UInt8 = 0, palette: [UInt8] = [], id: [UInt8] = [], pixels: [UInt8], footer: Bool = false
    ) -> Data {
        let header: [UInt8] = [
            UInt8(id.count), palette.isEmpty ? 0 : 1, kind, first, 0, UInt8(palette.count / 3), 0, palette.isEmpty ? 0 : 24,
            0, 0, 0, 0, UInt8(truncatingIfNeeded: width), UInt8(width >> 8),
            UInt8(truncatingIfNeeded: height), UInt8(height >> 8), depth, flags,
        ]
        var result = Data(header + id + palette + pixels)
        if footer {
            result.append(Data(count: 8))
            result.append(contentsOf: "TRUEVISION-XFILE.\0".utf8)
        }
        return result
    }
}
