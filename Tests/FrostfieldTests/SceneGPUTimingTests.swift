import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneGPUTimingTests {
    @Test("Owned scene timing reports native counters or a device capability error in either depth mode",
          arguments: [MeshDepthMode.forward, .reversed])
    func colorTiming(_ depthMode: MeshDepthMode) throws {
        let fixture = try GeometryRenderFixture()
        let mesh = try fixture.unlit(GeometryRenderFixture.square(1), depthMode: depthMode)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let targets = try fixture.makeTarget()
        let pass = fixture.pass(color: targets.color, depth: targets.depth, depthMode: depthMode)
        let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        guard SceneGPUTimingFrame.isSupported(on: fixture.device) else {
            #expect(throws: SceneGPUTimingError.unsupported) {
                try graph.encodeTimed(pass: pass, commandBuffer: command, depthMode: depthMode)
            }
            try graph.encode(pass: pass, commandBuffer: command, depthMode: depthMode)
            try fixture.finish(command)
            #expect(try fixture.pixel(fixture.pixels(of: targets.color), x: 32, y: 32).w == 1)
            return
        }
        let frame = try graph.encodeTimed(pass: pass, commandBuffer: command, depthMode: depthMode)
        #expect(pass.sampleBufferAttachments[0].sampleBuffer == nil)
        #expect(throws: SceneGPUTimingError.notReady) { try frame.results() }
        try fixture.finish(command)
        let passes = try frame.results()
        #expect(passes.map(\.pass) == [.color])
        let measured = try #require(passes.first)
        let vertex = try #require(measured.vertexSeconds)
        let fragment = try #require(measured.fragmentSeconds)
        let elapsed = try #require(measured.elapsedSeconds)
        #expect(vertex >= 0 && fragment >= 0 && elapsed > 0)
        #expect(elapsed >= max(vertex, fragment))
        #expect(elapsed <= (command.gpuEndTime - command.gpuStartTime) * 1.05 + 0.0001)
        #expect(try fixture.pixels(of: targets.color) == fixture.render(graph, depthMode: depthMode).pixels)
        #expect(try fixture.pixel(fixture.pixels(of: targets.color), x: 32, y: 32).w == 1)
        #expect(frame.statistics.drawnMeshCount == 1)
        #expect(throws: SceneGPUTimingError.commandBufferState) {
            try graph.encodeTimed(pass: pass, commandBuffer: command, depthMode: depthMode)
        }
        let counters = try #require(fixture.device.counterSets?.first { $0.name == MTLCommonCounterSet.timestamp.rawValue })
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counters
        descriptor.sampleCount = 4
        descriptor.storageMode = .shared
        pass.sampleBufferAttachments[0].sampleBuffer = try fixture.device.makeCounterSampleBuffer(descriptor: descriptor)
        let pending = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: SceneGPUTimingError.counterAttachmentInUse) {
            try graph.encodeTimed(pass: pass, commandBuffer: pending, depthMode: depthMode)
        }
    }

    @Test("Explicit views separate shadow and transmission timestamps",
          .enabled(if: MTLCreateSystemDefaultDevice().map { SceneGPUTimingFrame.isSupported(on: $0) } ?? false,
                   "Requires stage-boundary counters"))
    func shadowAndTransmissionTiming() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try GeometryRenderFixture.square(1)
        let opaque = try fixture.unlit(geometry, material: .physicallyBased(PhysicallyBasedMaterial(receivesShadows: true)))
        let glass = try fixture.unlit(
            geometry, material: .physicallyBased(PhysicallyBasedMaterial(transmission: .init(factor: 1, indexOfRefraction: 1)))
        )
        let back = SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: opaque)
        let front = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: glass)
        let light = try SceneNode(light: PunctualLight(kind: .spot(), shadow: .init()))
        let graph = SceneGraph(roots: [back, front, light], camera: fixture.camera)
        let directional = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 64, directionToLight: SIMD3(0, 0, 1),
            center: SIMD3(0, 0, -3), halfExtent: 4
        )
        let punctual = try PunctualShadowMaps(queue: fixture.queue, resolution: 64, spotCapacity: 1, pointCapacity: 0)
        let targets = try fixture.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: fixture.side, height: fixture.side, mipmapped: true
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let backdrop = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let pass = fixture.pass(color: targets.color, depth: targets.depth)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let frame = try graph.encodeTimed(
            pass: pass, commandBuffer: command,
            clipFromWorld: fixture.camera.projection(aspectRatio: 1), cameraPosition: .zero,
            shadowMap: directional, transmissionBackdrop: backdrop, punctualShadowMaps: punctual
        )
        try fixture.finish(command)
        let result = try frame.results()
        try #require(result.map(\.pass) == [.directionalShadow(cascade: 0), .punctualShadow(slice: 0), .opaqueColor, .transmissionColor])
        #expect(result.allSatisfy { ($0.elapsedSeconds ?? 0) > 0 })
        #expect(result[2].fragmentSeconds != nil)
        #expect(result[3].fragmentSeconds != nil)
        #expect(pass.sampleBufferAttachments[0].sampleBuffer == nil)
        #expect(frame.statistics.shadowDrawCount >= 2)
    }
}
