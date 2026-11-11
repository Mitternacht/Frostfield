import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RenderGPUTimingTests {
    @Test("Graph and custom draw timing preserve the ordinary output and caller descriptor")
    func graphAndCustomPass() throws {
        let gpu = try PostProcessFixture()
        let graph = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            return p.vector(color.a.xxx - color.rgb, color.a)
        }
        let effect = try PostProcessPass(graph: graph, device: gpu.device)
        let source = try gpu.color(width: 128, height: 64) { x, y in
            SIMD4(Float(x % 8) / 16, Float(y % 8) / 16, 0.25, 0.5)
        }
        let reference = try gpu.color(width: 128, height: 64)
        let timedOutput = try gpu.color(width: 128, height: 64)
        let customOutput = try gpu.color(width: 128, height: 64)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = customOutput
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        let command = try #require(gpu.queue.makeCommandBuffer())
        try effect.encode(source: source, destination: reference, commandBuffer: command)

        guard RenderGPUTiming.isSupported(on: gpu.device) else {
            #expect(throws: RenderGPUTiming.Failure.unsupported) {
                try effect.encodeTimed(source: source, destination: timedOutput, commandBuffer: command)
            }
            try effect.encode(source: source, destination: timedOutput, commandBuffer: command)
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed)
            #expect(try gpu.read(timedOutput) == gpu.read(reference))
            return
        }

        let graphTiming = try effect.encodeTimed(source: source, destination: timedOutput, commandBuffer: command)
        let customTiming = try RenderGPUTiming.encode(pass: pass, commandBuffer: command) { encoder in
            encoder.setRenderPipelineState(effect.pipelineState)
            encoder.setFragmentTexture(source, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        #expect(pass.sampleBufferAttachments[0].sampleBuffer == nil)
        #expect(throws: RenderGPUTiming.Failure.notReady) { try graphTiming.results() }
        #expect(throws: RenderGPUTiming.Failure.notReady) { try customTiming.results() }
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        for timing in [graphTiming, customTiming] {
            let stages = try timing.results()
            let vertex = try #require(stages.vertexSeconds)
            let fragment = try #require(stages.fragmentSeconds)
            let elapsed = try #require(stages.elapsedSeconds)
            #expect(elapsed.isFinite && elapsed > 0 && elapsed >= max(vertex, fragment))
            #expect(elapsed <= (command.gpuEndTime - command.gpuStartTime) * 1.05 + 0.0001)
        }
        #expect(try gpu.read(timedOutput) == gpu.read(reference))
        #expect(try gpu.read(customOutput) == gpu.read(reference))

        let counters = try #require(gpu.device.counterSets?.first { $0.name == MTLCommonCounterSet.timestamp.rawValue })
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counters
        descriptor.sampleCount = 4
        descriptor.storageMode = .shared
        pass.sampleBufferAttachments[0].sampleBuffer = try gpu.device.makeCounterSampleBuffer(descriptor: descriptor)
        let pending = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: RenderGPUTiming.Failure.counterAttachmentInUse) {
            try RenderGPUTiming.encode(pass: pass, commandBuffer: pending) { _ in Issue.record("Unexpected render encode") }
        }
        #expect(throws: RenderGPUTiming.Failure.commandBufferState) {
            try RenderGPUTiming.encode(pass: pass, commandBuffer: command) { _ in Issue.record("Unexpected render encode") }
        }
    }

    @Test("Async chain timing includes scaled effects and the fully bypassed copy")
    func wholeChain() async throws {
        let gpu = try PostProcessFixture()
        let bloom = try MipmapBloom(device: gpu.device, settings: .init(strength: 0.25, levels: 3))
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(bloom), .colorGrade(strength: 0.5)])
        try chain.setResolutionScale(0.5, at: 0)
        let source = try gpu.color(width: 64, height: 32) { x, y in
            SIMD4(Float(x % 8), Float(y % 8), 0.25, 1)
        }
        let reference = try gpu.color(width: 64, height: 32)
        let timedOutput = try gpu.color(width: 64, height: 32)
        try gpu.run { try chain.encode(source: source, destination: reference, commandBuffer: $0) }
        let seconds = try await chain.renderTimed(source: source, destination: timedOutput)
        #expect(seconds.isFinite && seconds > 0)
        #expect(try gpu.read(timedOutput) == gpu.read(reference))
        #expect(try gpu.read(timedOutput) != gpu.read(source))

        try chain.setEnabled(false, at: 0)
        try chain.setEnabled(false, at: 1)
        let copySeconds = try await chain.renderTimed(source: source, destination: timedOutput)
        #expect(copySeconds.isFinite && copySeconds > 0)
        #expect(try gpu.read(timedOutput) == gpu.read(source))
    }
}
