import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShadowSideTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    private func mesh(_ material: ClassicMaterial, state: MeshRenderState, halfExtent: Float) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: try GeometryRenderFixture.square(halfExtent), material: .classic(material), renderState: state
        )
    }

    private func image(shadowSide: MeshRenderState.Side?) throws -> [SIMD4<Float>] {
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try mesh(
                try ClassicMaterial(model: .lambert, ambient: SIMD3(repeating: 0.1 * .pi), receivesShadows: true),
                state: MeshRenderState(), halfExtent: 1.6
            ),
            castsShadow: false
        )
        let blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3(2.625, 0, 0.5)),
            mesh: try mesh(
                try ClassicMaterial(model: .basic),
                state: MeshRenderState(colorWriteMask: [], shadowSide: shadowSide), halfExtent: 0.25
            )
        )
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 512, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 5
        )
        return try fixture.render(
            SceneGraph(roots: [receiver, blocker], camera: fixture.camera), shadowMap: shadowMap
        ).pixels
    }

    @Test("Shadow maps can cast the front or the back without changing the color pass")
    func shadowSideChoosesTheCaster() throws {
        let open = fixture.pixel(try image(shadowSide: nil), x: 32, y: 32)
        let front = fixture.pixel(try image(shadowSide: .front), x: 32, y: 32)
        let back = fixture.pixel(try image(shadowSide: .back), x: 32, y: 32)
        let lit = 0.1 + 0.8 / Float.pi
        #expect(open.x < 0.2)
        #expect(front.x < 0.2)
        #expect(abs(back.x - lit) < 0.05, "\(back)")
        let colored = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -2)),
            mesh: try mesh(
                try ClassicMaterial(model: .basic, color: SIMD3(1, 0, 0)),
                state: MeshRenderState(shadowSide: .back), halfExtent: 1
            )
        )
        let color = try fixture.render(SceneGraph(roots: [colored], camera: fixture.camera)).pixels
        #expect(fixture.pixel(color, x: 32, y: 32).x > 0.5)
        let stored = SceneGraph(
            roots: [
                SceneNode(
                    transform: SceneTransform(translation: SIMD3(2.625, 0, 0.5)),
                    mesh: try mesh(
                        try ClassicMaterial(model: .basic),
                        state: MeshRenderState(colorWriteMask: [], shadowSide: .back), halfExtent: 0.25
                    )
                ),
            ],
            camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: stored, commandQueue: fixture.queue)
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        let again = try fixture.render(SceneGraph(
            roots: [
                SceneNode(
                    transform: SceneTransform(translation: SIMD3(0, 0, -3)),
                    mesh: try mesh(
                        try ClassicMaterial(model: .lambert, ambient: SIMD3(repeating: 0.1 * .pi), receivesShadows: true),
                        state: MeshRenderState(), halfExtent: 1.6
                    ),
                    castsShadow: false
                ),
                loaded.roots[0],
            ],
            camera: fixture.camera
        ), shadowMap: try DirectionalShadowMap(
            queue: fixture.queue, resolution: 512, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 5
        )).pixels
        #expect(abs(fixture.pixel(again, x: 32, y: 32).x - lit) < 0.05)
    }
}
