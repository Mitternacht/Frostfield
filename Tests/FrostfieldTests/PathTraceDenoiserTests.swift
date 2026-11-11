import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathTraceDenoiserTests {
    @Test("Guided denoising suppresses surface noise and preserves a silhouette and material edge")
    func guidedFilter() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let width = 17
        let height = 5
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        let source = try #require(device.makeTexture(descriptor: descriptor))
        let normalDepth = try #require(device.makeTexture(descriptor: descriptor))
        let albedo = try #require(device.makeTexture(descriptor: descriptor))
        let output = try #require(device.makeTexture(descriptor: descriptor))
        var radiance: [SIMD4<Float16>] = []
        var geometry: [SIMD4<Float16>] = []
        var surface: [SIMD4<Float16>] = []
        for y in 0..<height {
            for x in 0..<width {
                let left = x < 8
                let miss = x == width - 1
                let signal: Float16 = (x + y).isMultiple(of: 2) ? 0.7 : 0.3
                radiance.append(miss ? SIMD4(0, 0, 2, 1) : (left ? SIMD4(signal, 0, 0, 1) : SIMD4(0, signal, 0, 1)))
                geometry.append(left ? SIMD4(0, 0, 1, 2) : SIMD4(0, 1, 0, miss ? 0 : 4))
                surface.append(left ? SIMD4(1, 0, 0, 0.8) : SIMD4(0, 1, 0, 0.8))
            }
        }
        let region = MTLRegionMake2D(0, 0, width, height)
        for (texture, values) in [(source, radiance), (normalDepth, geometry), (albedo, surface)] {
            try values.withUnsafeBytes { bytes in
                texture.replace(region: region, mipmapLevel: 0, withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8)
            }
        }
        let denoiser = try PathTraceDenoiser(device: device)
        let buffer = try #require(queue.makeCommandBuffer())
        try denoiser.encode(
            source: source, normalDepth: normalDepth, albedoRoughness: albedo,
            destination: output, passes: 3, commandBuffer: buffer
        )
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        func pixel(_ x: Int) -> SIMD4<Float16> {
            var value = SIMD4<Float16>.zero
            output.getBytes(&value, bytesPerRow: 8, from: MTLRegionMake2D(x, 2, 1, 1), mipmapLevel: 0)
            return value
        }
        #expect(abs(Float(pixel(2).x) - Float(pixel(3).x)) < 0.2)
        #expect(pixel(7).y == 0)
        #expect(pixel(8).x == 0)
        #expect(pixel(width - 1) == SIMD4(0, 0, 2, 1))
    }
}
