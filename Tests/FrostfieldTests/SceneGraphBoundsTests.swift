import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneGraphBoundsTests {
    @Test("Hierarchy bounds use ancestor transforms and can tighten to vertices")
    func hierarchy() throws {
        let fixture = try GeometryRenderFixture()
        let vertices = [SIMD3<Float>(0, 0, 0), SIMD3(2, 0, 0), SIMD3(0, 2, 0)].map {
            MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
        }
        let mesh = try fixture.unlit(MeshGeometry(vertices: vertices, indices: [0, 1, 2]))
        let child = SceneNode(
            transform: SceneTransform(
                translation: SIMD3(0, 1, 0), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1))
            ),
            mesh: mesh
        )
        let parent = SceneNode(
            transform: SceneTransform(translation: SIMD3(5, 3, -2), scale: SIMD3(1.5, 2, 0.5)), children: [child]
        )
        let hidden = SceneNode(
            transform: SceneTransform(translation: SIMD3(30, 0, -2)), children: [SceneNode(mesh: mesh)], isVisible: false
        )
        let offLayer = SceneNode(
            transform: SceneTransform(translation: SIMD3(-30, 0, -2)), mesh: mesh, layers: SceneLayers(layer: 4)
        )
        let graph = SceneGraph(roots: [parent, hidden, offLayer], camera: fixture.camera)
        let fast = try #require(try graph.worldBounds(of: parent.id))
        let precise = try #require(try graph.worldBounds(of: child.id, precise: true))
        #expect(precise.minimum.x > 2)
        #expect(precise.maximum.y < fast.maximum.y - 1)
        #expect(precise.minimum.z == -2 && precise.maximum.z == -2)
        #expect(try graph.worldBounds(of: parent.id, precise: true) == precise)
        #expect(try graph.worldBounds(precise: true, visibleOnly: true) == precise)
        let all = try #require(try graph.worldBounds(precise: true))
        #expect(all.minimum.x == -30 && all.maximum.x == 32)
        #expect(try graph.worldBounds(of: hidden.id, visibleOnly: true) == nil)
        let missing = UUID()
        #expect(throws: SceneGraph.EditingError.missingNode(missing)) { try graph.worldBounds(of: missing) }
    }

    @Test("Bounds track current poses and exclude empty instance sets")
    func posedAndInstancedGeometry() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try GeometryRenderFixture.square(1)
        let jointCount = 1
        let skin = try MeshSkin(
            jointCount: jointCount,
            influences: geometry.vertices.map { _ in
                VertexSkinInfluence(joints: SIMD4(repeating: 0), weights: SIMD4(1, 0, 0, 0))
            }
        )
        let morph = try MeshMorphTargets(
            vertexCount: geometry.vertices.count,
            targets: [.init(positionOffsets: geometry.vertices.map { _ in SIMD3(0, 2, 0) })]
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, skin: skin,
            morphTargets: morph, supportsInstancing: true
        )
        func translation(_ x: Float) -> simd_float4x4 {
            var result = matrix_identity_float4x4
            result.columns.3.x = x
            return result
        }
        let palette = try JointPalette(matrices: [translation(3)])
        let node = SceneNode(mesh: mesh, jointPalette: palette, morphWeights: [0.5])
        var graph = SceneGraph(roots: [node], camera: fixture.camera)
        let precise = try #require(try graph.worldBounds(precise: true))
        #expect(precise.minimum == SIMD3(2, 0, 0))
        #expect(precise.maximum == SIMD3(4, 2, 0))
        #expect(try graph.worldBounds() == precise)

        let instances = try InstanceSet(device: fixture.device, transforms: [translation(-6), translation(4)])
        try graph.updateNode(withID: node.id) { $0.instances = instances }
        let instanced = try #require(try graph.worldBounds(precise: true))
        #expect(instanced.minimum == SIMD3(-4, 0, 0))
        #expect(instanced.maximum == SIMD3(8, 2, 0))
        try instances.replaceAll(transforms: [])
        #expect(try graph.worldBounds() == nil)
    }

    @Test("Bounds include world-sized point padding and pixel sprite anchors")
    func primitives() throws {
        let fixture = try GeometryRenderFixture()
        let cloud = try PointCloud(positions: [SIMD3(2, 1, -3)], sizes: [2])
        let points = try PointsRenderer(device: fixture.device, cloud: cloud, material: PointsMaterial(size: .world(1), shape: .circle))
        let sprites = try SpriteRenderer(
            device: fixture.device, sprites: [Sprite(position: SIMD3(-5, 0, -4), size: SIMD2(repeating: 99))],
            material: SpriteMaterial(sizeAttenuation: false)
        )
        let pointNode = SceneNode(
            transform: SceneTransform(translation: SIMD3(4, 0, 0)), primitive: .points(points)
        )
        let spriteNode = SceneNode(primitive: .sprites(sprites))
        let graph = SceneGraph(roots: [pointNode, spriteNode], camera: fixture.camera)
        let all = try #require(try graph.worldBounds())
        #expect(all.minimum == SIMD3(-5, 0, -4))
        #expect(all.maximum == SIMD3(7, 2, -2))
        #expect(try graph.worldBounds(precise: true) == all)

        let geometry = try LineGeometry(positions: [SIMD3(0, 0, -1), SIMD3(1, 0, -1), SIMD3(1, 1, -1)])
        let line = try LineRenderer(
            device: fixture.device, geometry: geometry, material: LineMaterial(width: .world(2), join: .miter(limit: 4))
        )
        let lineGraph = SceneGraph(roots: [SceneNode(primitive: .lines(line))], camera: fixture.camera)
        let stroke = try #require(try lineGraph.worldBounds())
        #expect(stroke.minimum == SIMD3(-4, -4, -5))
        #expect(stroke.maximum == SIMD3(5, 5, 3))
    }
}
