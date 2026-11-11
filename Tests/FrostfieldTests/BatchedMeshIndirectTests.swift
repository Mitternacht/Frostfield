import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct BatchedMeshIndirectCapabilityTests {
    @Test("Indirect submission has an explicit device check and leaves default batches direct")
    func capability() throws {
        let fixture = try GeometryRenderFixture()
        let square = try GeometryRenderFixture.square(0.25)
        func render(submission: BatchedMesh.Submission) throws -> SceneDrawStatistics {
            let batch = try BatchedMesh(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [square],
                capacity: 1, submission: submission
            )
            try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -2))
            return try fixture.render(SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)).statistics
        }
        #expect(try render(submission: .direct).indirectBatchExecutionCount == 0)
        if BatchedMesh.supportsIndirectSubmission(on: fixture.device) {
            #expect(try render(submission: .indirectCommandBuffer).indirectBatchExecutionCount == 1)
        } else {
            #expect(throws: BatchedMesh.Failure.indirectUnsupported) { try render(submission: .indirectCommandBuffer) }
        }
    }
}

private var supportsBatchBundles: Bool {
    MTLCreateSystemDefaultDevice().map { BatchedMesh.supportsIndirectSubmission(on: $0) } ?? false
}

@Suite(.enabled(if: supportsBatchBundles, "Requires Metal indirect render commands"))
struct BatchedMeshIndirectTests {
    let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    @Test("Indexed bundles preserve draw order, mixed winding and per-run instance offsets")
    func renders() throws {
        let square = try GeometryRenderFixture.square(0.25)
        let triangle = try triangleGeometry()
        let direct = try makeBatch([square, triangle], submission: .direct)
        let bundled = try makeBatch([square, triangle])
        let transforms = [
            SceneTransform(translation: SIMD3(-0.75, 0, -2)),
            SceneTransform(translation: SIMD3(0, 0, -2)),
            SceneTransform(translation: SIMD3(0.75, 0, -2), scale: SIMD3(-1, 1, 1)),
            SceneTransform(translation: SIMD3(0, 0.65, -2)),
        ]
        let tints: [SIMD4<Float>] = [SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1), SIMD4(1, 0, 1, 1)]
        func fill(_ batch: BatchedMesh) throws -> BatchedMesh.InstanceID {
            batch.sortsObjects = false
            var handles: [BatchedMesh.InstanceID] = []
            for (index, placement) in transforms.enumerated() {
                handles.append(try batch.addInstance(geometryIndex: index % 2, transform: placement.matrix, tint: tints[index]))
            }
            return handles[0]
        }
        let directFirst = try fill(direct)
        let indirectFirst = try fill(bundled)
        let reference = try render(direct)
        let first = try render(bundled)
        #expect(first.statistics.drawnMeshCount == 4 && first.statistics.drawnInstanceCount == 4)
        #expect(first.statistics.indirectBatchExecutionCount == 3)
        #expect(reference.statistics.submittedTriangleCount == 6 && first.statistics.submittedTriangleCount == 6)
        #expect(first.pixels == reference.pixels)
        #expect(fixture.pixel(first.pixels, x: 44, y: 32) == SIMD4(0, 0, 1, 1))
        #expect(bundled.indirectBundleBuildCount == 1)

        try direct.update(directFirst, transform: translationMatrix(-0.85, 0.25, -2), tint: SIMD4(0, 1, 1, 1))
        try bundled.update(indirectFirst, transform: translationMatrix(-0.85, 0.25, -2), tint: SIMD4(0, 1, 1, 1))
        let edited = try render(bundled)
        #expect(edited.pixels == (try render(direct)).pixels)
        #expect(edited.pixels != first.pixels)
        #expect(bundled.indirectBundleBuildCount == 1)
    }

    @Test("Blended commands keep depth order across indirect winding spans")
    func blendedOrder() throws {
        let square = try GeometryRenderFixture.square(0.45)
        let batch = try makeBatch([square, try triangleGeometry()], state: .transparent)
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -1), tint: SIMD4(1, 0, 0, 0.5))
        try batch.addInstance(geometryIndex: 1, transform: translationMatrix(0, 0, -3), tint: SIMD4(0, 0, 1, 0.5))
        let mirrored = SceneTransform(translation: SIMD3(0, 0, -2), scale: SIMD3(-1, 1, 1)).matrix
        try batch.addInstance(geometryIndex: 0, transform: mirrored, tint: SIMD4(0, 1, 0, 0.5))
        let output = try render(batch)
        #expect(output.statistics.drawnMeshCount == 3 && output.statistics.indirectBatchExecutionCount == 3)
        #expect(isNearly(fixture.pixel(output.pixels, x: 32, y: 32), SIMD4(0.11, 0.055, 0.125, 0.875)))
    }

    @Test("Indirect draws inherit the prepared material's fragment texture and tint")
    func textured() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        var rgba = SIMD4<UInt8>(64, 128, 192, 255)
        texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &rgba, bytesPerRow: 4)
        let square = try GeometryRenderFixture.square(0.25)
        func renderTextured(_ submission: BatchedMesh.Submission) throws -> [SIMD4<Float>] {
            let batch = try BatchedMesh(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [square], capacity: 1,
                material: .textured(texture), submission: submission
            )
            try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -2), tint: SIMD4(0.5, 0.75, 0.25, 1))
            return try render(batch).pixels
        }
        let indirect = try renderTextured(.indirectCommandBuffer)
        #expect(indirect == (try renderTextured(.direct)))
        #expect(isNearly(fixture.pixel(indirect, x: 32, y: 32), SIMD4(32.0 / 255, 96.0 / 255, 48.0 / 255, 1)))
    }

    @Test("An earlier unretained command survives bundle eviction, palette replacement and teardown")
    func leases() throws {
        let target = try fixture.makeTarget()
        let previous = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        try autoreleasepool {
            let square = try GeometryRenderFixture.square(0.25)
            let triangle = try triangleGeometry()
            let batch = try makeBatch([square, triangle])
            batch.sortsObjects = false
            try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -2), tint: SIMD4(0, 0, 1, 1))
            let graph = SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)
            try graph.encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: previous)
            // A unique count and ordered range each time, enough to evict the original.
            for index in 1...5 {
                try batch.addInstance(
                    geometryIndex: index % 2, transform: translationMatrix(1, 0.15 * Float(index), -2)
                )
                let output = try render(batch)
                #expect(output.statistics.drawnInstanceCount == index + 1)
            }
            #expect(batch.indirectBundleBuildCount == 6)
            let id = try batch.geometryID(at: 0)
            try batch.replaceGeometry(id, with: triangle)
            #expect(try render(batch).statistics.indirectBatchExecutionCount == 1)
        }
        try fixture.finish(previous)
        let old = try fixture.pixels(of: target.color)
        #expect(fixture.pixel(old, x: 32, y: 32) == SIMD4(0, 0, 1, 1))
        #expect(fixture.pixel(old, x: 48, y: 32) == .zero)
    }

    @Test("The same bundle submits through color, shadows and surface attributes")
    func depthPasses() throws {
        let square = try GeometryRenderFixture.square(0.25)
        let blocker = try MeshGeometry(vertices: square.vertices.map {
            MeshVertex(position: $0.position, normal: simd_normalize(SIMD3(0, 1, 1)))
        }, indices: square.indices)
        let batch = try makeBatch([blocker])
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0.75, 0, -2))
        let receiver = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let graph = SceneGraph(roots: [receiver, SceneNode(batch: batch)], camera: fixture.camera)
        let shadow = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        let output = try fixture.render(graph, shadowMap: shadow)
        #expect(output.statistics.shadowDrawCount == 1 && output.statistics.indirectBatchExecutionCount == 2)
        #expect(abs(fixture.pixel(output.pixels, x: 32, y: 32).x - 0.16) < 0.01)
        #expect(batch.indirectBundleBuildCount == 1)

        let target = try fixture.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: fixture.side, height: fixture.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let attributes = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let command = try #require(fixture.queue.makeCommandBuffer())
        let stats = try graph.encodeSurfaceAttributes(
            pass: fixture.pass(color: attributes, depth: target.depth), commandBuffer: command
        )
        #expect(stats.indirectBatchExecutionCount == 1)
        try fixture.finish(command)
        var pixel = SIMD2<UInt32>.zero
        attributes.getBytes(&pixel, bytesPerRow: MemoryLayout<SIMD2<UInt32>>.stride,
                            from: MTLRegionMake2D(44, 32, 1, 1), mipmapLevel: 0)
        #expect(pixel.x != 0)
        #expect(batch.indirectBundleBuildCount == 1)
    }

    private func makeBatch(
        _ geometries: [MeshGeometry], submission: BatchedMesh.Submission = .indirectCommandBuffer,
        state: MeshRenderState? = nil
    ) throws -> BatchedMesh {
        try BatchedMesh(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometries: geometries,
            capacity: 8, renderState: state, submission: submission
        )
    }

    private func render(_ batch: BatchedMesh) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
        try fixture.render(SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera))
    }

    private func triangleGeometry() throws -> MeshGeometry {
        try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(-0.27, -0.23, 0), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0.27, -0.23, 0), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0, 0.3, 0), normal: SIMD3(0, 0, 1)),
        ], indices: [0, 1, 2])
    }
}
