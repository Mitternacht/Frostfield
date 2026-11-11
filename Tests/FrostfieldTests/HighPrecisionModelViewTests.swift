import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionModelViewTests {
    // Even adding a small local offset to an absolute Double at this origin
    // would round it away. Rebase before walking the hierarchy.
    private let farOrigin: Float = 0x1p60

    @Test("Camera-relative mesh transforms preserve both visible child offsets and cull the offscreen sibling")
    func childPlacementAndCulling() throws {
        let gpu = try GeometryRenderFixture()
        let mesh = try unlit(gpu, halfExtent: 0.18)
        let nodes = [
            SceneNode(transform: .init(translation: SIMD3(-0.75, 0, -2)), mesh: mesh, materialParameters: SIMD4(1, 0, 0, 1)),
            SceneNode(transform: .init(translation: SIMD3(0.75, 0, -2)), mesh: mesh, materialParameters: SIMD4(0, 1, 0, 1)),
            SceneNode(transform: .init(translation: SIMD3(4, 0, -2)), mesh: mesh, materialParameters: SIMD4(0, 0, 1, 1)),
        ]
        let reference = try gpu.render(graph(gpu, origin: 0, nodes: nodes))
        let relocated = graph(gpu, origin: farOrigin, nodes: nodes, highPrecision: true)
        let precise = try gpu.render(relocated)
        #expect(precise.pixels == reference.pixels)
        #expect(precise.statistics.drawnMeshCount == 2)
        #expect(precise.statistics.culledMeshCount == 1)
        #expect(gpu.pixel(precise.pixels, x: gpu.column(x: -0.75, depth: 2), y: 32) == SIMD4(1, 0, 0, 1))
        #expect(gpu.pixel(precise.pixels, x: gpu.column(x: 0.75, depth: 2), y: 32) == SIMD4(0, 1, 0, 1))

        var floatFrame = relocated
        floatFrame.highPrecision = false
        let rounded = try gpu.render(floatFrame)
        #expect(rounded.pixels != reference.pixels)
        #expect(gpu.pixel(rounded.pixels, x: 32, y: 32).w == 1)
    }

    @Test("Rotated cameras retain pivots, world-oriented children, normal shading and projection jitter")
    func orientedHierarchy() throws {
        let gpu = try GeometryRenderFixture(side: 45)
        let mesh = try gpu.unlit(GeometryRenderFixture.square(0.38), material: .classic(try ClassicMaterial(model: .normal)))
        let child = SceneNode(
            transform: .init(
                translation: SIMD3(0.55, 0.22, -2.4),
                rotation: simd_quatf(angle: 0.35, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(0.9, 1.15, 1)
            ),
            mesh: mesh, pivot: SIMD3(0.11, -0.16, 0.08), orientationMode: .world
        )
        func oriented(_ origin: Float) -> SceneGraph {
            var scene = graph(gpu, origin: origin, nodes: [child], highPrecision: true)
            scene.roots[0].transform.rotation = simd_quatf(angle: 0.4, axis: SIMD3(0, 0, 1))
            scene.roots[0].transform.scale = SIMD3(1.2, 0.75, 1)
            scene.cameraTransform.rotation = simd_quatf(angle: 0.19, axis: SIMD3(0, 0, 1))
            scene.projectionJitter = SIMD2(0.25, -0.125)
            return scene
        }
        var nearOrigin = oriented(0)
        nearOrigin.highPrecision = false
        let reference = try gpu.render(nearOrigin)
        let moved = try gpu.render(oriented(farOrigin))
        #expect(reference.pixels.contains { $0.w == 1 && $0.x > 0.6 })
        #expect(zip(reference.pixels, moved.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.001) })
        #expect(moved.statistics.drawnMeshCount == 1)
        var steady = oriented(farOrigin)
        steady.projectionJitter = .zero
        #expect(try gpu.render(steady).pixels != moved.pixels)
    }

    @Test("High-precision LOD selection uses the relative origin while shaders keep world coordinates")
    func detailAndShading() throws {
        let gpu = try GeometryRenderFixture()
        let surface = try SurfaceShaderGraph { b in
            let worldX = b.worldPosition.x * Float(0x1p-60)
            return .init(baseColor: b.vector(b.constant(Float(0)), worldX, b.constant(Float(0))))
        }
        let material = try PhysicallyBasedMaterial(surfaceGraph: surface, shadingModel: .unlit)
        let detailed = try unlit(gpu, halfExtent: 0.12)
        let coarse = try gpu.unlit(GeometryRenderFixture.square(0.12), material: .physicallyBased(material))
        let lod = try LevelOfDetail(levels: [.init(mesh: detailed, distance: 0), .init(mesh: coarse, distance: 1.125)])
        let node = SceneNode(
            transform: .init(translation: SIMD3(0.75, 0, -1)),
            materialParameters: SIMD4(1, 0, 0, 1), levelsOfDetail: lod
        )
        var scene = graph(gpu, origin: farOrigin, nodes: [node], highPrecision: true)
        let precise = try gpu.render(scene)
        #expect(gpu.pixel(precise.pixels, x: gpu.column(x: 0.75, depth: 1), y: 32) == SIMD4(0, 1, 0, 1))
        scene.highPrecision = false
        let rounded = try gpu.render(scene)
        #expect(gpu.pixel(rounded.pixels, x: 32, y: 32) == SIMD4(1, 0, 0, 1))
    }

    @Test("An external Float projection cannot silently use the graph-owned high-precision path")
    func externalView() throws {
        let gpu = try GeometryRenderFixture()
        let scene = graph(gpu, origin: 0, nodes: [], highPrecision: true)
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let clip = try gpu.camera.projection(aspectRatio: 1)
        #expect(throws: MeshRenderer.Failure.highPrecisionView) {
            try scene.encode(
                pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
                clipFromWorld: clip, cameraPosition: .zero
            )
        }
        #expect(command.status == .notEnqueued)
    }

    private func unlit(_ gpu: GeometryRenderFixture, halfExtent: Float) throws -> MeshRenderer {
        let material = try PhysicallyBasedMaterial(shadingModel: .unlit)
        return try gpu.unlit(GeometryRenderFixture.square(halfExtent), material: .physicallyBased(material))
    }

    private func graph(
        _ gpu: GeometryRenderFixture, origin: Float, nodes: [SceneNode], highPrecision: Bool = false
    ) -> SceneGraph {
        let translation = SIMD3(origin, 0, 0)
        let root = SceneNode(transform: .init(translation: translation), children: nodes)
        var scene = SceneGraph(roots: [root], camera: gpu.camera, cameraTransform: .init(translation: translation))
        scene.highPrecision = highPrecision
        return scene
    }
}
