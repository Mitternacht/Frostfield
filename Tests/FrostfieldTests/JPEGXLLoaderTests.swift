import Foundation
import Frostfield
import Metal
import Testing

struct JPEGXLLoaderTests {
    // Lossless 2 × 2, Float32 RGBA in linear sRGB. Encoded by libjxl 0.11.0.
    static let hdrFixture = "AAAADEpYTCANCocKAAAAFGZ0eXBqeGwgAAAAAGp4bCAAAAAJanhsbAoAAABianhsY/8KCBByQg6AokUIEBAAKAFLGIsVUnxot7HzbmCj4esErdMQcgwAAAT7///7/7/nH4BpXgQAAHQCAJwB8EMEmq0ZAAAA+v//x/9fE8CnCwAA8AUAoPG/4gkPMA=="
    // Lossless sRGB UInt8, bare codestream, two authored RGB texels.
    static let sRGB = "/woAcBAJCAQBAFQASxiLFcIJbNxm1wVCABCA/1f9iY9S"
    static let gray = "/woAcBAUNwIIBAEAJABLGIsVwklBjgA="

    @Test("JPEG XL preserves HDR RGB and straight alpha through Float32 and half-float Metal sampling")
    func hdr() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let data = try #require(Data(base64Encoded: Self.hdrFixture))
        let loader = TextureLoader(device: device)
        let expected: [SIMD4<Float>] = [
            SIMD4(4, 0.25, 2, 0.5), SIMD4(0.125, 0.5, 0.25, 1),
            SIMD4(0, 3, 0.5, 0.25), SIMD4(8, 4, 0.5, 1),
        ]

        let full = try loader.loadHighDynamicRange(data: data, precision: .float)
        #expect(full.source == .highDynamicRangeEncoding)
        #expect(full.maximumValue == 8)
        #expect(full.texture.pixelFormat == .rgba32Float)
        #expect(try Self.readFloat(full.texture) == expected)

        let half = try loader.loadHighDynamicRange(data: data, precision: .half, mipmapped: true)
        #expect(half.texture.pixelFormat == .rgba16Float)
        #expect(half.texture.mipmapLevelCount == 2)
        #expect(try Self.sample(half.texture, device: device, level: 0, count: 4) == expected)
        let mip = try Self.sample(half.texture, device: device, level: 1, count: 1)
        #expect(mip[0] == SIMD4(3.03125, 1.9375, 0.8125, 0.6875))
    }

    @Test("JPEG XL converts sRGB and grayscale transfer values in bare codestreams")
    func colorProfiles() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let loader = TextureLoader(device: device)
        let data = try #require(Data(base64Encoded: Self.sRGB))
        let image = try loader.loadHighDynamicRange(data: data, precision: .float)
        #expect(image.source == .standardDynamicRange)
        let pixels = try Self.readFloat(image.texture)
        #expect(pixels.count == 2)
        #expect(isNearly(pixels[0], SIMD4(0.21586, 0.0512695, 0.0144438, 1)))
        #expect(isNearly(pixels[1], SIMD4(0, 1, 0.57758, 1)))

        let gray = try #require(Data(base64Encoded: Self.gray))
        let grayImage = try loader.loadHighDynamicRange(data: gray, precision: .float)
        let grayPixels = try Self.readFloat(grayImage.texture)
        #expect(grayPixels[0] == SIMD4(0, 0, 0, 1))
        #expect(isNearly(grayPixels[1], SIMD4(0.21586, 0.21586, 0.21586, 1)))
    }

    @Test("JPEG XL embedded Display P3 ICC converts gamut and applies orientation")
    func icc() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let encoded = "/woAcAuE5E5hQMIsADZzfR64zutojAque3pC5Z8VIQSVFBKLj2VVocC6gJkyesrr0ho7S1qt1tAZBAibMW1mFvCJu2IqQ3RoMw2QhjeZE+bOnDF05LD8hHz9x7jDvat15hIfy6WhLGKj53I1kBD+JWUqeHNoxs3EleHMgXb+Npshw0IetGvl8OB0G6cZAW56/ft3vuUx5wTsC/Hv370h//7tK4IqFwTGBQd+Ybs///79u4Mo/sDYXmiZg0WLYaIFIs09xqHETArKPUE3GcwIhVY2CAQBADQASxiLFcJB4AUCAEAAAQ=="
        let data = try #require(Data(base64Encoded: encoded))
        let image = try TextureLoader(device: device).loadHighDynamicRange(data: data, precision: .float)
        #expect(image.texture.width == 1 && image.texture.height == 2)
        let pixels = try Self.readFloat(image.texture)
        #expect(isNearly(pixels[0], SIMD4(1.22494, -0.04206, -0.01964, 1)))
        #expect(isNearly(pixels[1], SIMD4(0, 0, 1.09827, 1)))
    }

    @Test("JPEG XL PQ and HLG preserve display radiance relative to 203-nit white")
    func displayTransfers() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let loader = TextureLoader(device: device)
        let pq = "AAAADEpYTCANCocKAAAAFGZ0eXBqeGwgAAAAAGp4bCAAAAAJanhsbAoAAAA7anhsY/8KAHCBHEAePdBjAAAAAAgIBAEAdABLGIsVUnzmQjWfA1SKt+zcB6gUb9m5D1Ap3rJzAw=="
        let hlg = "AAAADEpYTCANCocKAAAAFGZ0eXBqeGwgAAAAAGp4bCAAAAAJanhsbAoAAAA7anhsY/8KAHCBHECegUCPAQAAACAIBAEAdABLGIsVUnzkYiWfAACgBwA4AQBADwBwAgCAHgDgAA=="
        for encoded in [pq, hlg] {
            let data = try #require(Data(base64Encoded: encoded))
            let image = try loader.loadHighDynamicRange(data: data, precision: .float)
            #expect(image.source == .highDynamicRangeEncoding)
            let peak: Float = 1_000 / 203
            #expect(image.contentHeadroom == peak)
            let pixels = try Self.readFloat(image.texture)
            for channel in 0..<3 {
                #expect(abs(pixels[0][channel] - 1) < 0.002)
                #expect(abs(pixels[1][channel] - peak) < 0.002)
            }
            #expect(pixels.allSatisfy { $0.w == 1 })
        }
    }

    @Test("A recognized but truncated JPEG XL cannot fall back to a clipped image")
    func truncated() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let data = try #require(Data(base64Encoded: Self.hdrFixture))
        let loader = TextureLoader(device: device)
        for damaged in [data.prefix(12), data.dropLast(24), Data([0xff, 0x0a])] {
            #expect(throws: TextureLoader.HighDynamicRangeFailure.invalidImage("The JPEG XL data is invalid or truncated.")) {
                try loader.loadHighDynamicRange(data: damaged)
            }
        }
    }

    static func readFloat(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        try #require(texture.pixelFormat == .rgba32Float)
        var result = [SIMD4<Float>](repeating: .zero, count: texture.width * texture.height)
        result.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                bytes.baseAddress!, bytesPerRow: texture.width * 16,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return result
    }

    private static func sample(
        _ texture: any MTLTexture, device: any MTLDevice, level: UInt32, count: Int
    ) throws -> [SIMD4<Float>] {
        let library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void sample_jxl(texture2d<float> image [[texture(0)]], constant uint &mip [[buffer(0)]],
                                   device float4 *pixels [[buffer(1)]], uint i [[thread_position_in_grid]]) {
                constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::nearest);
                uint width = image.get_width(mip);
                float2 uv = (float2(i % width, i / width) + 0.5) / float2(width, image.get_height(mip));
                pixels[i] = image.sample(nearest, uv, level(mip));
            }
            """, options: nil)
        let pipeline = try device.makeComputePipelineState(function: #require(library.makeFunction(name: "sample_jxl")))
        let buffer = try #require(device.makeBuffer(length: count * 16, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        var mip = level
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBytes(&mip, length: MemoryLayout<UInt32>.size, index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(command.error?.localizedDescription ?? "Metal command failed")")
        return Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: count))
    }
}
