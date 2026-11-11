import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import zlib

struct ColorLookupImageTests {
    @Test("Both LUT strip orientations preserve red-fastest RGB, including zero-alpha PNG", arguments: [false, true])
    func axes(horizontal: Bool) throws {
        let png = try LookupStripFixture.rgba(horizontal: horizontal)
        let table = try ColorLookupTable(image: png)
        #expect(table.size == 2)
        #expect(table.domainMinimum == .zero && table.domainMaximum == .one)
        #expect(table.values == LookupStripFixture.expected)

        let flipped = try ColorLookupTable(image: png, flipVertically: true)
        let indices = horizontal ? [2, 3, 0, 1, 6, 7, 4, 5] : [6, 7, 4, 5, 2, 3, 0, 1]
        #expect(flipped.values == indices.map { LookupStripFixture.expected[$0] })
    }

    @Test("LUT PNG grayscale and indexed values expand without applying metadata")
    func grayscaleAndPalette() throws {
        let gray: [UInt8] = [0, 22, 80, 100, 120, 140, 160, 255]
        let grayAlpha: [UInt8] = gray.flatMap { [$0, 0] }
        let fixtures: [(UInt8, [UInt8])] = [(0, gray), (4, grayAlpha)]
        for (layout, pixels) in fixtures {
            let data = try LookupStripFixture.png(width: 2, height: 4, layout: layout, pixels: pixels)
            let table = try ColorLookupTable(image: data)
            #expect(table.values == gray.map { SIMD3(repeating: Float($0) / 255) })
        }

        let palette: [UInt8] = [5, 25, 95, 230, 20, 60]
        let indices: [UInt8] = [1, 0, 1, 0, 0, 1, 0, 1]
        let indexed = try LookupStripFixture.png(width: 2, height: 4, layout: 3, pixels: indices, chunks: [
            ("PLTE", palette), ("tRNS", [0, 1]),
        ])
        let table = try ColorLookupTable(image: indexed)
        let colors = [SIMD3<Float>(5, 25, 95) / 255, SIMD3<Float>(230, 20, 60) / 255]
        #expect(table.values == indices.map { colors[Int($0)] })
    }

    @Test("An opaque Display P3 TIFF strip keeps decoded numeric RGB")
    func opaqueImage() throws {
        let values = LookupStripFixture.codes.flatMap { [$0.x, $0.y, $0.z] }
        let provider = try #require(CGDataProvider(data: Data(values) as CFData))
        let space = try #require(CGColorSpace(name: CGColorSpace.displayP3))
        let image = try #require(CGImage(width: 2, height: 4, bitsPerComponent: 8, bitsPerPixel: 24,
                                         bytesPerRow: 6, space: space,
                                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let buffer = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(buffer, "public.tiff" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        try #require(CGImageDestinationFinalize(destination))
        #expect(try ColorLookupTable(image: buffer as Data).values == LookupStripFixture.expected)
    }

    @Test("LUT image file loading supports automatic layout and explicit flipping")
    func file() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("png")
        defer { try? FileManager.default.removeItem(at: url) }
        try LookupStripFixture.rgba(horizontal: true).write(to: url)
        #expect(try ColorLookupTable(contentsOf: url).values == LookupStripFixture.expected)
        #expect(try ColorLookupTable(imageContentsOf: url, flipVertically: true).values.first ==
                LookupStripFixture.expected[2])
    }

    @Test("LUT images reject a malformed strip shape, oversized lattice, and corrupt compressed pixels")
    func invalidImages() throws {
        let wrongShape = try LookupStripFixture.png(width: 3, height: 3, layout: 0, pixels: [UInt8](repeating: 0, count: 9))
        #expect(throws: ColorLookupTable.Failure.imageLayout(width: 3, height: 3)) {
            try ColorLookupTable(image: wrongShape)
        }
        let tooLarge = try LookupStripFixture.png(width: 66, height: 66, layout: 0,
                                                   pixels: [UInt8](repeating: 0, count: 66 * 66))
        #expect(throws: ColorLookupTable.Failure.size(66)) { try ColorLookupTable(image: tooLarge) }
        let corrupt = try LookupStripFixture.png(width: 2, height: 4, layout: 0,
                                                  pixels: [UInt8](repeating: 0, count: 8), filter: 5)
        #expect(throws: ColorLookupTable.Failure.imageDecode) { try ColorLookupTable(image: corrupt) }
        #expect(throws: ColorLookupTable.Failure.imageDecode) { try ColorLookupTable(image: Data()) }
    }

    @Test("An imported LUT grades all lattice inputs in Metal", .enabled(if: MTLCreateSystemDefaultDevice() != nil),
          arguments: [ColorLookupTable.Interpolation.trilinear, .tetrahedral])
    func metalGrade(interpolation: ColorLookupTable.Interpolation) throws {
        let gpu = try PostProcessFixture()
        let table = try ColorLookupTable(image: LookupStripFixture.rgba(horizontal: true))
        let pass = try ColorLookupPass(device: gpu.device, table: table,
                                       settings: .init(interpolation: interpolation, inputEncoding: .linear))
        let source = try gpu.color(width: 8, height: 1) { x, _ in
            SIMD4(Float(x & 1) * 0.5, Float((x >> 1) & 1) * 0.5, Float(x >> 2) * 0.5, 0.5)
        }
        let destination = try gpu.color(width: 8, height: 1)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        let samples = try gpu.read(destination)
        for (index, sample) in samples.enumerated() {
            let expected = SIMD4(LookupStripFixture.expected[index] * 0.5, 0.5)
            #expect(PostProcessFixture.close(sample, expected, tolerance: 4e-4))
        }
    }
}

private enum LookupStripFixture {
    // Independent coordinate reference, ordered R then G then B.
    static let codes: [SIMD3<UInt8>] = [
        SIMD3(20, 120, 190), SIMD3(101, 37, 251), SIMD3(210, 70, 12), SIMD3(75, 200, 89),
        SIMD3(14, 166, 97), SIMD3(244, 60, 145), SIMD3(43, 231, 105), SIMD3(98, 24, 167),
    ]
    static let expected = codes.map { SIMD3(Float($0.x), Float($0.y), Float($0.z)) / 255 }

    static func rgba(horizontal: Bool) throws -> Data {
        let order = horizontal ? [0, 1, 4, 5, 2, 3, 6, 7] : Array(0..<8)
        let pixels = order.flatMap { i -> [UInt8] in
            let color = codes[i]
            return [color.x, color.y, color.z, UInt8(i % 2)]
        }
        return try png(width: horizontal ? 4 : 2, height: horizontal ? 2 : 4, layout: 6, pixels: pixels,
                       chunks: [("gAMA", [0, 1, 134, 160])])
    }

    static func png(
        width: Int, height: Int, layout: UInt8, pixels: [UInt8], chunks: [(String, [UInt8])] = [],
        filter: UInt8 = 0
    ) throws -> Data {
        let rowBytes = pixels.count / height
        var scanlines: [UInt8] = []
        for row in 0..<height {
            scanlines.append(filter)
            scanlines.append(contentsOf: pixels[(row * rowBytes)..<((row + 1) * rowBytes)])
        }
        var length = compressBound(uLong(scanlines.count))
        var compressed = [UInt8](repeating: 0, count: Int(length))
        let status = compressed.withUnsafeMutableBufferPointer { target in
            scanlines.withUnsafeBufferPointer { source in
                compress2(target.baseAddress, &length, source.baseAddress, uLong(source.count), Z_DEFAULT_COMPRESSION)
            }
        }
        try #require(status == Z_OK)

        var data = Data([0x89, 0x50, 0x4E, 0x47, 13, 10, 26, 10])
        let header = word(UInt32(width)) + word(UInt32(height)) + [8, layout, 0, 0, 0]
        data.append(chunk("IHDR", header))
        for (name, value) in chunks { data.append(chunk(name, value)) }
        data.append(chunk("IDAT", Array(compressed.prefix(Int(length)))))
        data.append(chunk("IEND", []))
        return data
    }

    private static func word(_ value: UInt32) -> [UInt8] {
        withUnsafeBytes(of: value.bigEndian, Array.init)
    }

    private static func chunk(_ name: String, _ value: [UInt8]) -> Data {
        let contents = Array(name.utf8) + value
        let crc = contents.withUnsafeBufferPointer { crc32(0, $0.baseAddress, UInt32($0.count)) }
        return Data(word(UInt32(value.count)) + contents + word(UInt32(crc)))
    }
}
