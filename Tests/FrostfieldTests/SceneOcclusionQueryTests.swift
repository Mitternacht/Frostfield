import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneOcclusionQueryTests {
    @Test("Owned visibility matches the color pass in both depth modes",
          arguments: [MeshDepthMode.forward, .reversed])
    func colorVisibility(_ depthMode: MeshDepthMode) throws {
        let fixture = try GeometryRenderFixture()
        let large = try fixture.unlit(GeometryRenderFixture.square(1), depthMode: depthMode)
        let smallGeometry = try GeometryRenderFixture.square(0.4)
        let small = try fixture.unlit(smallGeometry, depthMode: depthMode)
        let front = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: large)
        let occluded = SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: small)
        let unrelated = SceneNode(transform: .init(translation: SIMD3(-2.8, 0, -4)), mesh: small)
        let hidden = SceneNode(mesh: small, isVisible: false)
        let outside = SceneNode(
            transform: .init(translation: SIMD3(25, 0, -4)), mesh: small, frustumCulled: true
        )
        let batch = try BatchedMesh(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [smallGeometry], capacity: 1,
            depthMode: depthMode, submission: BatchedMesh.supportsIndirectSubmission(on: fixture.device) ? .indirectCommandBuffer : .direct
        )
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(2.8, 0, -4))
        let visibleBatch = SceneNode(batch: batch)
        let graph = SceneGraph(roots: [front, occluded, unrelated, hidden, outside, visibleBatch], camera: fixture.camera)
        let targets = try fixture.makeTarget()
        let pass = fixture.pass(color: targets.color, depth: targets.depth, depthMode: depthMode)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let missing = UUID()
        let frame = try graph.encodeWithOcclusionQueries(
            for: [front.id, occluded.id, hidden.id, outside.id, visibleBatch.id, missing],
            pass: pass, commandBuffer: command, depthMode: depthMode
        )
        #expect(pass.visibilityResultBuffer == nil)
        #expect(throws: SceneOcclusionError.notReady) { try frame.results() }
        try fixture.finish(command)
        let results = try frame.results()
        #expect(results == [
            front.id: .visible, occluded.id: .occluded, hidden.id: .notDrawn,
            outside.id: .notDrawn, visibleBatch.id: .visible, missing: .notDrawn,
        ])
        let pixels = try fixture.pixels(of: targets.color)
        #expect(fixture.pixel(pixels, x: fixture.column(x: 2.8, depth: 4), y: 32).w > 0.99)
        #expect(frame.statistics.drawnMeshCount == 4)
        #expect(try fixture.render(graph, depthMode: depthMode).pixels == pixels)
    }

    @Test("Visibility merges a node's primitive and mesh across transmission passes")
    func splitPasses() throws {
        let fixture = try GeometryRenderFixture()
        let large = try fixture.unlit(GeometryRenderFixture.square(1))
        let glass = try fixture.unlit(
            GeometryRenderFixture.square(0.4),
            material: .physicallyBased(PhysicallyBasedMaterial(transmission: .init(factor: 1, indexOfRefraction: 1)))
        )
        let point = try PointsRenderer(
            device: fixture.device, cloud: PointCloud(positions: [SIMD3(2.8, 0, 0)]),
            material: PointsMaterial(size: .pixels(4), blending: .opaque)
        )
        let front = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: large)
        let hiddenGlass = SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: glass)
        var mixed = SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: glass)
        mixed.primitive = .points(point)
        let visibleGlass = SceneNode(transform: .init(translation: SIMD3(-2.8, 0, -4)), mesh: glass)
        let graph = SceneGraph(roots: [front, mixed, hiddenGlass, visibleGlass], camera: fixture.camera)
        let targets = try fixture.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: fixture.side, height: fixture.side, mipmapped: true
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let backdrop = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let command = try #require(fixture.queue.makeCommandBuffer())
        let pass = fixture.pass(color: targets.color, depth: targets.depth)
        let frame = try graph.encodeWithOcclusionQueries(
            for: [front.id, mixed.id, hiddenGlass.id, visibleGlass.id], pass: pass,
            commandBuffer: command, transmissionBackdrop: backdrop
        )
        try fixture.finish(command)
        #expect(try frame.results() == [
            front.id: .visible, mixed.id: .visible, hiddenGlass.id: .occluded, visibleGlass.id: .visible,
        ])
        #expect(frame.statistics.drawnPrimitiveCount == 1)
    }

    @Test("Two encodes on an unretained command buffer keep independent visibility results")
    func independentSnapshots() throws {
        let fixture = try GeometryRenderFixture()
        let mesh = try fixture.unlit(GeometryRenderFixture.square(0.5))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let first = try graph.encodeWithOcclusionQueries(
            for: [node.id], pass: pass, commandBuffer: command,
            clipFromWorld: fixture.camera.projection(aspectRatio: 1), cameraPosition: .zero
        )
        let second = try graph.encodeWithOcclusionQueries(
            for: [node.id], pass: pass, commandBuffer: command, scissor: MTLScissorRect(x: 0, y: 0, width: 0, height: 0)
        )
        try fixture.finish(command)
        #expect(try first.results()[node.id] == .visible)
        #expect(try second.results()[node.id] == .notDrawn)
        #expect(throws: SceneOcclusionError.commandBufferState) {
            try graph.encodeWithOcclusionQueries(for: [node.id], pass: pass, commandBuffer: command)
        }
        let existing = try #require(fixture.device.makeBuffer(length: 8, options: .storageModeShared))
        pass.visibilityResultBuffer = existing
        let pending = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: SceneOcclusionError.existingVisibilityBuffer) {
            try graph.encodeWithOcclusionQueries(for: [node.id], pass: pass, commandBuffer: pending)
        }
    }
}
