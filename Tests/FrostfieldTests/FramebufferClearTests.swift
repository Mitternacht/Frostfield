import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct FramebufferClearTests {
    @Test("Independent clears preserve the other stored buffers and the caller's pass actions")
    func selectiveClear() async throws {
        let fixture = try GeometryRenderFixture(side: 8)
        let target = try fixture.makeTarget()
        let stencilDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .stencil8, width: 8, height: 8, mipmapped: false
        )
        stencilDescriptor.storageMode = .private
        stencilDescriptor.usage = .renderTarget
        let stencil = try #require(fixture.device.makeTexture(descriptor: stencilDescriptor))
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.depthAttachment.storeAction = .store
        pass.stencilAttachment.texture = stencil
        pass.stencilAttachment.storeAction = .store
        try FramebufferClear(color: SIMD4(0, 0.5, 0, 1), depth: 0.75, stencil: 7).apply(to: pass)

        let command = try #require(fixture.queue.makeCommandBuffer())
        try #require(command.makeRenderCommandEncoder(descriptor: pass)).endEncoding()
        try FramebufferClear(depth: 0.25).encode(in: pass, commandBuffer: command)
        try FramebufferClear(color: SIMD4(0.5, 0, 0.25, 0.5)).encode(in: pass, commandBuffer: command)
        try FramebufferClear(stencil: 13).encode(in: pass, commandBuffer: command)
        #expect(pass.depthAttachment.clearDepth == 0.75)
        #expect(pass.colorAttachments[0].clearColor.green == 0.5)
        #expect(pass.stencilAttachment.clearStencil == 7)

        let probe = try fixture.makeTarget().color
        try stencilProbe(stencil, color: probe, fixture: fixture, command: command)
        try fixture.finish(command)
        #expect(try fixture.pixels(of: target.color).allSatisfy { $0 == SIMD4(0.5, 0, 0.25, 0.5) })
        let sample = try await TextureTransfer.read(
            from: target.depth, region: MTLRegionMake2D(4, 4, 1, 1), commandQueue: fixture.queue
        )
        #expect(sample.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) } == 0.25)
        #expect(try fixture.pixels(of: probe).allSatisfy { $0 == SIMD4(1, 0, 0, 1) })
    }

    @Test("Automatic clears touch selected attachments only and validate before mutation")
    func loadActions() throws {
        let fixture = try GeometryRenderFixture(side: 8)
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.colorAttachments[0].loadAction = .load
        pass.depthAttachment.loadAction = .load
        #expect(throws: FramebufferClear.Failure.missingAttachment("stencil")) {
            try FramebufferClear(color: SIMD4(0, 0, 0, 1), stencil: 1).apply(to: pass)
        }
        #expect(pass.colorAttachments[0].loadAction == .load)
        try FramebufferClear(color: SIMD4(0, 0, 0, 0)).apply(to: pass)
        #expect(pass.colorAttachments[0].loadAction == .clear)
        #expect(pass.colorAttachments[0].clearColor.alpha == 0)
        #expect(pass.depthAttachment.loadAction == .load)
        #expect(throws: FramebufferClear.Failure.color) { try FramebufferClear(color: SIMD4(.nan, 0, 0, 1)) }
        #expect(throws: FramebufferClear.Failure.depth) { try FramebufferClear(depth: 1.1) }
        #expect(throws: FramebufferClear.Failure.stencil) { try FramebufferClear(stencil: 256) }
    }

    private func stencilProbe(
        _ stencil: any MTLTexture, color: any MTLTexture, fixture: GeometryRenderFixture, command: any MTLCommandBuffer
    ) throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 stencilProbeVertex(uint index [[vertex_id]]) {
            constexpr float2 corners[] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
            return float4(corners[index], 0, 1);
        }
        fragment half4 stencilProbeFragment() { return half4(1, 0, 0, 1); }
        """
        let library = try fixture.device.makeLibrary(source: source, options: nil)
        let pipeline = MTLRenderPipelineDescriptor()
        pipeline.vertexFunction = library.makeFunction(name: "stencilProbeVertex")
        pipeline.fragmentFunction = library.makeFunction(name: "stencilProbeFragment")
        pipeline.colorAttachments[0].pixelFormat = color.pixelFormat
        pipeline.stencilAttachmentPixelFormat = .stencil8
        let state = try fixture.device.makeRenderPipelineState(descriptor: pipeline)
        let stencilState = MTLStencilDescriptor()
        stencilState.stencilCompareFunction = .equal
        let depthStencil = MTLDepthStencilDescriptor()
        depthStencil.frontFaceStencil = stencilState
        depthStencil.backFaceStencil = stencilState
        let compare = try #require(fixture.device.makeDepthStencilState(descriptor: depthStencil))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].storeAction = .store
        pass.stencilAttachment.texture = stencil
        pass.stencilAttachment.loadAction = .load
        try FramebufferClear(color: .zero).apply(to: pass)
        let encoder = try #require(command.makeRenderCommandEncoder(descriptor: pass))
        encoder.setRenderPipelineState(state)
        encoder.setDepthStencilState(compare)
        encoder.setStencilReferenceValue(13)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }
}
