import Foundation
@testable import Frostfield
import ImageIO
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureLoaderTests {
    // A 1 × 2 sRGB PNG. Top: RGBA(128, 64, 255, 128). Bottom: RGBA(20, 230, 100, 255).
    private static let image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAACCAYAAACZgbYnAAAAAXNSR0IArs4c6QAAABJJREFUeJxjaHD438Ag8izlPwAVTASdmr2NEwAAAABJRU5ErkJggg=="
    // Two pixels on each row. The rows use filters 0 through 4 in order and
    // the zlib stream spans seven IDAT chunks.
    private static let filteredRGBA = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAAFCAYAAABvsz2cAAAACUlEQVR4nAEtANL/AMgoW2MBAAAACUlEQVRkMkAKFB7AAQFY0uWdAAAACUlEQVQC/QEkPV7/AvcM9uGsAAAACUlEQVR94AEQjc+AA5YygU3eAAAACUlEQVSK7AJYQbjABOuOOO8nAAAACUlEQVQxn/0GGA3/adSUyY3LAAAAAklEQVQRphCXmHcAAAAASUVORK5CYII="
    private static let rgba: [UInt8] = [
        200, 100, 50, 64, 10, 20, 30, 192,
        1, 2, 253, 1, 37, 63, 91, 0,
        248, 127, 221, 2, 53, 204, 42, 128,
        18, 201, 90, 3, 123, 11, 250, 1,
        253, 250, 249, 0, 3, 35, 7, 255,
    ]

    @Test("Data PNG loads keep exact RGB at low and zero alpha",
          arguments: [TextureLoader.Interpretation.linearData, .encodedSRGB])
    func exactPNGChannels(interpretation: TextureLoader.Interpretation) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let data = try #require(Data(base64Encoded: Self.filteredRGBA))
        let texture = try TextureLoader(device: device).load(data: data, interpretation: interpretation)
        #expect(texture.pixelFormat == (interpretation == .linearData ? .rgba8Unorm : .rgba8Unorm_srgb))
        #expect(texture.width == 2 && texture.height == 5)
        var pixels = [UInt8](repeating: 0, count: Self.rgba.count)
        pixels.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 2, 5), mipmapLevel: 0)
        }
        #expect(pixels == Self.rgba)
    }

    @Test("The straight PNG fallback rejects truncated, corrupt, and invalid-filter streams")
    func invalidPNGChannels() throws {
        let data = try #require(Data(base64Encoded: Self.filteredRGBA))
        var pixels = [UInt8](repeating: 0, count: Self.rgba.count)
        #expect(pixels.withUnsafeMutableBytes { PNGChannels.copyRGBA8(data, width: 2, height: 5, into: $0) })
        #expect(pixels == Self.rgba)

        var badCRC = data
        badCRC[50] ^= 1  // A byte in the first IDAT chunk's CRC.
        // Same image with a filter byte of 5 in the last row, and valid CRCs and zlib checksum.
        let badFilter = try #require(Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAAFCAYAAABvsz2cAAAACUlEQVR4nAEtANL/AMgoW2MBAAAACUlEQVRkMkAKFB7AAQFY0uWdAAAACUlEQVQC/QEkPV7/AvcM9uGsAAAACUlEQVR94AEQjc+AA5YygU3eAAAACUlEQVSK7AJYQbjABeuXI95mAAAACUlEQVQxn/0GGA3/ad3tFTVvAAAAAklEQVQRp2eQqOEAAAAASUVORK5CYII="
        ))
        let sentinel = [UInt8](repeating: 0xA5, count: Self.rgba.count)
        for malformed in [Data(data.dropLast(3)), badCRC, badFilter] {
            pixels = sentinel
            #expect(!pixels.withUnsafeMutableBytes { PNGChannels.copyRGBA8(malformed, width: 2, height: 5, into: $0) })
            #expect(pixels == sentinel)
        }
    }

    @Test("The GPU samples top-left sRGB color as linear RGB with straight alpha")
    func sampling() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let data = try #require(Data(base64Encoded: Self.image))
        let texture = try TextureLoader(device: device).load(data: data)
        #expect(texture.pixelFormat == .rgba8Unorm_srgb)
        #expect(texture.width == 1)
        #expect(texture.height == 2)
        #expect(texture.mipmapLevelCount == 1)

        let library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;

            kernel void inspect(
                texture2d<float> image [[texture(0)]],
                device float4* samples [[buffer(0)]],
                uint i [[thread_position_in_grid]]
            ) {
                constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest);
                samples[i] = image.sample(nearest, float2(0.5, (float(i) + 0.5) * 0.5));
            }
            """, options: nil)
        let function = try #require(library.makeFunction(name: "inspect"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        let output = try #require(device.makeBuffer(
            length: 2 * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared
        ))
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 2, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)

        let samples = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)
        let top = SIMD4<Float>(0.216, 0.051, 1, 0.502)
        let bottom = SIMD4<Float>(0.007, 0.791, 0.127, 1)
        for channel in 0..<4 {
            #expect(abs(samples[0][channel] - top[channel]) < 0.012)
            #expect(abs(samples[1][channel] - bottom[channel]) < 0.012)
        }
    }

    @Test("A regular local file loads, while a remote URL or malformed image fails")
    func fileAndMalformedData() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let loader = TextureLoader(device: device)
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(path: "texture.png")
        let data = try #require(Data(base64Encoded: Self.image))
        try data.write(to: file)

        let texture = try loader.load(contentsOf: file)
        #expect(texture.width == 1)
        #expect(texture.height == 2)
        let remote = try #require(URL(string: "https://example.invalid/texture.png"))
        #expect(throws: TextureLoader.Failure.localFileRequired) { try loader.load(contentsOf: remote) }
        #expect(throws: TextureLoader.Failure.invalidImage) { try loader.load(data: Data([1, 2, 3])) }
    }

    private static func imageIO(decodes type: String) -> Bool {
        (CGImageSourceCopyTypeIdentifiers() as? [String])?.contains(type) == true
    }

    private static func imageIO(encodes type: String) -> Bool {
        (CGImageDestinationCopyTypeIdentifiers() as? [String])?.contains(type) == true
    }

    @Test("A lossless WebP decodes through ImageIO with exact pixels",
          .enabled(if: imageIO(decodes: "org.webmproject.webp"), "ImageIO cannot decode WebP on this system"))
    func webP() throws {
        // A 2 × 2 image from `cwebp -lossless -exact` (libwebp 1.6.0). Top: red, green.
        // Bottom: blue, RGB(32, 64, 128). Every pixel is opaque.
        let data = try #require(Data(base64Encoded:
            "UklGRjIAAABXRUJQVlA4TCUAAAAvAUAAAB8gECA48z9lQyCQpLI/5AIBgnPn5j/wyVHBDRgi+h8CAA=="
        ))
        let device = try #require(MTLCreateSystemDefaultDevice())
        let texture = try TextureLoader(device: device).load(data: data, interpretation: .linearData)
        #expect(texture.pixelFormat == .rgba8Unorm)
        #expect(texture.width == 2 && texture.height == 2)
        var pixels = [UInt8](repeating: 0, count: 16)
        try pixels.withUnsafeMutableBytes { bytes in
            texture.getBytes(try #require(bytes.baseAddress), bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0)
        }
        #expect(pixels == [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 32, 64, 128, 255])
    }

    @Test("An AVIF written by ImageIO decodes to its solid color",
          .enabled(if: imageIO(decodes: "public.avif") && imageIO(encodes: "public.avif"),
                   "ImageIO cannot both encode and decode AVIF on this system"))
    func avif() throws {
        let color: [UInt8] = [200, 100, 50, 255]
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try #require(CGContext(
            data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ))
        context.setFillColor(CGColor(
            colorSpace: space, components: color.map { CGFloat($0) / 255 }
        ) ?? CGColor(gray: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        let image = try #require(context.makeImage())
        let encoded = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(encoded, "public.avif" as CFString, 1, nil))
        // ImageIO's AVIF encoder rejects quality 1.0, so use a high lossy quality.
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        try #require(CGImageDestinationFinalize(destination))

        let device = try #require(MTLCreateSystemDefaultDevice())
        let texture = try TextureLoader(device: device).load(data: encoded as Data, interpretation: .encodedSRGB)
        #expect(texture.pixelFormat == .rgba8Unorm_srgb)
        #expect(texture.width == 8 && texture.height == 8)
        var pixel = [UInt8](repeating: 0, count: 4)
        try pixel.withUnsafeMutableBytes { bytes in
            texture.getBytes(try #require(bytes.baseAddress), bytesPerRow: 4, from: MTLRegionMake2D(4, 4, 1, 1), mipmapLevel: 0)
        }
        // Lossy YCbCr storage allows a small error per channel.
        #expect(zip(pixel, color).allSatisfy { abs(Int($0) - Int($1)) <= 6 }, "\(pixel)")
    }

    @Test("An oversized image is rejected from its header")
    func oversizedImage() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let data = try #require(Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAIAEAAAABCAYAAACZiUteAAAAAXNSR0IArs4c6QAAADZJREFUeJztwQENAAAAwqD3T+1mDqAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgBuABQABHQsoSAAAAABJRU5ErkJggg=="
        ))
        #expect(throws: TextureLoader.Failure.imageDimensions(width: 8_193, height: 1)) {
            try TextureLoader(device: device).load(data: data)
        }
    }
}
