import Foundation
import Frostfield
import Metal
import Testing

struct GPUTimingBatchTests {
    @Test("An empty frame has no measured GPU passes")
    func empty() async throws {
        let summary = try await GPUTimingBatch().resolve()
        #expect(summary.render.seconds == 0 && summary.compute.seconds == 0)
        #expect(summary.render.measuredPassCount == 0 && summary.compute.measuredPassCount == 0)
        #expect(summary.render.unavailablePassCount == 0 && summary.compute.unavailablePassCount == 0)
    }

    @Test("A frame sums scene, post-process and compute passes across submissions exactly once",
          .enabled(if: MTLCreateSystemDefaultDevice().map { RenderGPUTiming.isSupported(on: $0) } ?? false,
                   "Requires stage-boundary counters"))
    func frame() async throws {
        let scene = try GeometryRenderFixture()
        let post = try PostProcessFixture()
        let mesh = try scene.unlit(GeometryRenderFixture.square(1))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let graph = SceneGraph(roots: [node], camera: scene.camera)
        let targets = try scene.makeTarget()
        let sceneCommand = try #require(scene.queue.makeCommandBuffer())
        let sceneTiming = try graph.encodeTimed(
            pass: scene.pass(color: targets.color, depth: targets.depth), commandBuffer: sceneCommand
        )

        let effect = try PostProcessPass(
            graph: PostProcessGraph { p in p.sceneColor(at: p.screenUV) }, device: post.device
        )
        let source = try post.color(width: 128, height: 64) { _, _ in SIMD4(0.25, 0.5, 0.75, 1) }
        let output = try post.color(width: 128, height: 64)
        let effectsCommand = try #require(post.queue.makeCommandBuffer())
        let renderTiming = try effect.encodeTimed(source: source, destination: output, commandBuffer: effectsCommand)

        let textureGraph = try TextureShaderGraph { t in t.constant(SIMD4<Float>(1, 0.5, 0.25, 1)) }
        let generator = try ShaderTextureGenerator(graph: textureGraph, device: post.device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 128, height: 64, mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        let generated = try #require(post.device.makeTexture(descriptor: descriptor))
        let computeTiming = try generator.encodeTimed(into: generated, commandBuffer: effectsCommand)

        var building = GPUTimingBatch()
        try building.track(sceneTiming)
        try building.track(renderTiming)
        try building.track(computeTiming)
        try building.track(renderTiming)
        try building.track(sceneTiming)
        try building.track(computeTiming)
        let batch = building
        await #expect(throws: GPUTimingBatch.Failure.commandBufferNotSubmitted) { try await batch.resolve() }
        sceneCommand.commit()
        effectsCommand.commit()
        async let first = batch.resolve()
        async let second = batch.resolve()
        let (summary, repeated) = try await (first, second)
        #expect(summary == repeated)

        let sceneSpans = try sceneTiming.results().compactMap(\.elapsedSeconds)
        let renderSpan = try #require(try renderTiming.results().elapsedSeconds)
        #expect(summary.render.measuredPassCount == sceneSpans.count + 1)
        #expect(summary.render.measuredPassCount >= 2)
        #expect(summary.render.seconds == sceneSpans.reduce(0, +) + renderSpan)
        #expect(summary.compute.measuredPassCount == 1)
        #expect(try summary.compute.seconds == computeTiming.duration())
        #expect(summary.render.seconds > 0 && summary.compute.seconds > 0)
        #expect(try post.read(output) == post.read(source))
    }

    @Test("A discarded uncommitted capture reports its state instead of waiting",
          .enabled(if: MTLCreateSystemDefaultDevice().map { ComputeGPUTiming.isSupported(on: $0) } ?? false,
                   "Requires stage-boundary counters"))
    func discarded() async throws {
        let gpu = try PostProcessFixture()
        let graph = try TextureShaderGraph { t in t.constant(SIMD4<Float>(0.25, 0.5, 0.75, 1)) }
        let generator = try ShaderTextureGenerator(graph: graph, device: gpu.device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 32, height: 32, mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let batch = try autoreleasepool {
            let command = try #require(gpu.queue.makeCommandBuffer())
            let timing = try generator.encodeTimed(into: texture, commandBuffer: command)
            var batch = GPUTimingBatch()
            try batch.track(timing)
            return batch
        }
        await #expect(throws: GPUTimingBatch.Failure.commandBufferDiscarded) { try await batch.resolve() }
    }
}
