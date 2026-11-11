import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderTextureViewTests {
    @Test("sRGB outputs declare the linear view they are written through")
    func srgbOutputsNeedPixelFormatViewUsage() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let generator = try ShaderTextureGenerator(
            graph: TextureShaderGraph { t in t.constant(SIMD3<Float>(0.5, 0.25, 0.125)) }, device: device
        )
        let made = try generator.makeTexture(width: 4, height: 4, format: .rgba8Unorm_srgb, queue: queue)
        #expect(made.usage.contains(.pixelFormatView))

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb, width: 4, height: 4, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        let undeclared = try #require(device.makeTexture(descriptor: descriptor))
        let buffer = try #require(queue.makeCommandBuffer())
        #expect(throws: ShaderComputeFailure.outputTexture) {
            try generator.encode(into: undeclared, commandBuffer: buffer)
        }
        descriptor.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
        let declared = try #require(device.makeTexture(descriptor: descriptor))
        try generator.encode(into: declared, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        #expect(buffer.status == .completed)
    }
}
