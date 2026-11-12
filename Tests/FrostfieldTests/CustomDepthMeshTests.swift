import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CustomDepthMeshTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    private func mesh(_ halfExtent: Float) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: try GeometryRenderFixture.square(halfExtent),
            material: .classic(try ClassicMaterial(model: .basic)),
            renderState: MeshRenderState(colorWriteMask: [])
        )
    }

    private func image(depthMesh: MeshRenderer?) throws -> SIMD4<Float> {
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float,
                geometry: try GeometryRenderFixture.square(1.6),
                material: .classic(try ClassicMaterial(
                    model: .lambert, ambient: SIMD3(repeating: 0.1 * .pi), receivesShadows: true
                ))
            ),
            castsShadow: false
        )
        var blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3(2.625, 0, 0.5)),
            mesh: try mesh(0.25)
        )
        blocker.depthMesh = depthMesh
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 512, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 5
        )
        let pixels = try fixture.render(
            SceneGraph(roots: [receiver, blocker], camera: fixture.camera), shadowMap: shadowMap
        ).pixels
        return fixture.pixel(pixels, x: 32, y: 32)
    }

    @Test("A depth mesh casts the shadow while the color mesh stays in the picture")
    func depthMeshCasts() throws {
        let wide = try image(depthMesh: nil)
        let narrow = try image(depthMesh: mesh(0.02))
        let lit = 0.1 + 0.8 / Float.pi
        #expect(wide.x < 0.2, "\(wide)")
        #expect(abs(narrow.x - lit) < 0.05, "\(narrow)")
        var stored = SceneNode(
            transform: SceneTransform(translation: SIMD3(2.625, 0, 0.5)),
            mesh: try mesh(0.25)
        )
        stored.depthMesh = try mesh(0.02)
        let archive = try SceneArchive(
            capturing: SceneGraph(roots: [stored], camera: fixture.camera), commandQueue: fixture.queue
        )
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        #expect(loaded.roots[0].depthMesh != nil)
        let again = try image(depthMesh: loaded.roots[0].depthMesh)
        #expect(abs(again.x - lit) < 0.05, "\(again)")
    }
}
