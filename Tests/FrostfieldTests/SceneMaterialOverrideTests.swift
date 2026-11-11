import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneMaterialOverrideTests {
    private let gpu: RasterLightingFixture
    private let camera: PerspectiveCamera

    init() throws {
        gpu = try RasterLightingFixture(width: 64, height: 64)
        camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
    }

    @Test("A shared mesh keeps its authored material on an excluded node and restores after clearing the override")
    func sharedMeshAndExclusions() throws {
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.8, height: 0.8)
        )
        let left = node(x: -0.7, mesh: mesh), right = node(x: 0.7, mesh: mesh)
        var graph = SceneGraph(roots: [left, right], camera: camera)
        let original = try gpu.pixels(of: gpu.render(graph).color)
        let override = try SceneMaterialOverride(
            material: .classic(ClassicMaterial(model: .basic, color: SIMD3(0, 1, 0))), excluding: [right.id]
        )
        try override.prepare(in: graph, colorPixelFormat: .rgba16Float)
        graph.overrideMaterial = override
        let changed = try gpu.pixels(of: gpu.render(graph).color)
        #expect(original[32 * gpu.width + 18].x > 0)
        #expect(changed[32 * gpu.width + 18] == SIMD4(0, 1, 0, 1))
        #expect(changed[32 * gpu.width + 46] == original[32 * gpu.width + 46])
        #expect(graph.roots[0].mesh === mesh && graph.roots[1].mesh === mesh)
        graph.overrideMaterial = nil
        #expect(try gpu.pixels(of: gpu.render(graph).color) == original)
    }

    @Test("The override prepares the pass attachment and depth convention from the live target")
    func attachmentAndReversedDepth() throws {
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1, height: 1)
        )
        var graph = SceneGraph(roots: [node(x: 0, mesh: mesh)], camera: camera)
        graph.overrideMaterial = try SceneMaterialOverride(material: .unlit, parameters: SIMD4(0, 0, 1, 1))
        let color = try gpu.texture(.rgba8Unorm, usage: .renderTarget, storage: .shared)
        let depth = try gpu.depthTarget()
        try gpu.run { command in
            let statistics = try graph.encode(
                pass: gpu.pass(color: color, depth: depth, depthMode: .reversed),
                commandBuffer: command, depthMode: .reversed
            )
            #expect(statistics.submittedTriangleCount == 2)
        }
        var center = SIMD4<UInt8>.zero
        withUnsafeMutableBytes(of: &center) { bytes in
            color.getBytes(bytes.baseAddress!, bytesPerRow: 4, from: MTLRegionMake2D(32, 32, 1, 1), mipmapLevel: 0)
        }
        #expect(center == SIMD4(0, 0, 255, 255))
    }

    @Test("Replacing a blended mesh's color material preserves the authored shadow caster set")
    func authoredShadowCasters() throws {
        let plane = try MeshGeometry.plane(width: 1, height: 1)
        let caster = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane)
        let blended = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane,
            material: .unlit, renderState: .transparent
        )
        var graph = SceneGraph(roots: [node(x: -0.7, mesh: caster), node(x: 0.7, mesh: blended)], camera: camera)
        graph.overrideMaterial = try SceneMaterialOverride(material: .shadowedLambert)
        let shadow = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 64, directionToLight: SIMD3(0, 0, 1),
            center: SIMD3(0, 0, -3), halfExtent: 2
        )
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        try gpu.run { command in
            let statistics = try graph.encode(
                pass: gpu.pass(color: color, depth: depth), commandBuffer: command, shadowMap: shadow
            )
            #expect(statistics.submittedTriangleCount == 4)
            #expect(statistics.shadowTriangleCount == 2)
        }
    }

    @Test("An override follows a live batch palette in direct and indirect submission",
          arguments: [BatchedMesh.Submission.direct, .indirectCommandBuffer])
    func liveBatch(submission: BatchedMesh.Submission) throws {
        guard submission == .direct || BatchedMesh.supportsIndirectSubmission(on: gpu.device) else { return }
        let batch = try BatchedMesh(
            device: gpu.device, colorPixelFormat: .rgba16Float,
            geometries: [.plane(width: 0.8, height: 0.8)], capacity: 2, submission: submission
        )
        _ = try batch.addInstance(geometryIndex: 0, transform: SceneTransform(translation: SIMD3(-0.7, 0, 0)).matrix)
        _ = try batch.addInstance(geometryIndex: 0, transform: SceneTransform(translation: SIMD3(0.7, 0, 0)).matrix)
        var graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), batch: batch)], camera: camera
        )
        graph.overrideMaterial = try SceneMaterialOverride(material: .classic(ClassicMaterial(model: .basic, color: SIMD3(1, 0, 0))))
        let first = try gpu.pixels(of: gpu.render(graph).color)
        let center = 32 * gpu.width
        #expect(first[center + 12] == SIMD4(1, 0, 0, 1))
        #expect(first[center + 46] == SIMD4(1, 0, 0, 1))
        try batch.replaceGeometry(batch.geometryID(at: 0), with: .plane(width: 0.25, height: 0.8))
        let updated = try gpu.pixels(of: gpu.render(graph).color)
        #expect(updated[center + 12] == .zero)
        #expect(updated[center + 46] == SIMD4(1, 0, 0, 1))
    }

    @Test("The override's fragment draws the authored surface graph's deformed vertices")
    func deformedSource() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(positionOffset: s.constant(SIMD3<Float>(0.65, 0, 0)))
        }
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.55, height: 0.55),
            material: .physicallyBased(PhysicallyBasedMaterial(surfaceGraph: graph, shadingModel: .unlit))
        )
        var scene = SceneGraph(roots: [node(x: 0, mesh: mesh)], camera: camera)
        scene.overrideMaterial = try SceneMaterialOverride(
            material: .classic(ClassicMaterial(model: .basic, color: SIMD3(0, 1, 0)))
        )
        let changed = try gpu.pixels(of: gpu.render(scene).color)
        scene.overrideMaterial = nil
        let authored = try gpu.pixels(of: gpu.render(scene).color)
        #expect(changed[32 * gpu.width + 32] == .zero)
        #expect(changed[32 * gpu.width + 45] == SIMD4(0, 1, 0, 1))
        #expect(changed.map { $0.w } == authored.map { $0.w })
    }

    private func node(x: Float, mesh: MeshRenderer) -> SceneNode {
        SceneNode(
            transform: .init(translation: SIMD3(x, 0, -3)), mesh: mesh,
            materialParameters: SIMD4(1, 0, 0, 1)
        )
    }
}
