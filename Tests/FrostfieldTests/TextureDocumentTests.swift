import Metal
import Testing
@testable import Frostfield

private struct FixedDocument: TextureDocumentParser {
    let document: TextureDocument
    func parse(_ data: Data) throws -> TextureDocument { document }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureDocumentTests {
    @Test func uploadsParsedDataAndBlocks() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let loader = TextureLoader(device: device)
        let pixels: [UInt8] = [
            10, 20, 30, 255, 40, 50, 60, 255,
            70, 80, 90, 255, 1, 2, 3, 4,
        ]
        let color = try loader.load(
            data: Data(pixels), parser: FixedDocument(document: TextureDocument(width: 2, height: 2, contents: .rgba8(Data(pixels))))
        )
        #expect(color.pixelFormat == .rgba8Unorm && color.width == 2 && color.height == 2)
        var read = [UInt8](repeating: 0, count: pixels.count)
        color.getBytes(&read, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0)
        #expect(read == pixels)

        let short = TextureDocument(width: 2, height: 2, contents: .rgba8(Data([1, 2, 3, 4])))
        #expect(throws: TextureLoader.Failure.invalidDocument("RGBA8 needs 16 bytes, not 4.")) {
            try loader.upload(short)
        }

        let format: MTLPixelFormat = device.supportsFamily(.apple2) ? .astc_4x4_ldr : .bc1_rgba
        let block = format == .astc_4x4_ldr
            ? Data([0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 17, 17, 128, 128, 0x40, 0x40, 0xFF, 0xFF])
            : Data(repeating: 0xAB, count: 8)
        let compressed = try loader.load(
            data: block,
            parser: FixedDocument(document: TextureDocument(width: 4, height: 4, contents: .blocks(format: format, levels: [block])))
        )
        #expect(compressed.pixelFormat == format && compressed.width == 4 && compressed.mipmapLevelCount == 1)
        var stored = [UInt8](repeating: 0, count: block.count)
        compressed.getBytes(&stored, bytesPerRow: block.count, from: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0)
        #expect(Data(stored) == block)

        #expect(throws: TextureLoader.Failure.invalidImage) { try loader.load(data: Data([1, 2, 3])) }
    }
}
