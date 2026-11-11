import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneSelectionTests {
    @Test("A reversed drag selects posed mesh and primitive centers in hierarchy order")
    func perspectiveSelection() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try GeometryRenderFixture.square(2)
        let morph = try MeshMorphTargets(vertexCount: geometry.vertices.count, targets: [
            .init(positionOffsets: geometry.vertices.map { _ in SIMD3(2, 0, 0) })
        ])
        let mesh = try fixture.unlit(geometry, morphTargets: morph)
        let points = try PointsRenderer(
            device: fixture.device, cloud: PointCloud(positions: [SIMD3(0.8, 0.9, 0), SIMD3(1.2, 1.1, 0)]),
            material: PointsMaterial()
        )
        let line = try LineRenderer(
            device: fixture.device, geometry: LineGeometry(positions: [SIMD3(0.5, 1, 0), SIMD3(1.5, 1, 0)]),
            material: LineMaterial()
        )
        let posed = SceneNode(
            transform: .init(translation: SIMD3(-1, 1, 0)), mesh: mesh, morphWeights: [1]
        )
        let rest = SceneNode(transform: .init(translation: SIMD3(-1, 1, 0)), mesh: mesh)
        let pointNode = SceneNode(primitive: .points(points))
        let lineNode = SceneNode(primitive: .lines(line))
        let hidden = SceneNode(children: [SceneNode(primitive: .points(points))], isVisible: false)
        var offLayer = SceneNode(primitive: .lines(line))
        offLayer.layers = SceneLayers(layer: 3)
        let root = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4)),
            children: [posed, rest, pointNode, hidden, lineNode, offLayer]
        )
        let behind = SceneNode(transform: .init(translation: SIMD3(-1, -1, 4)), mesh: mesh)
        let tooFar = SceneNode(transform: .init(translation: SIMD3(4, 4, -16)), mesh: mesh)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 1, far: 8)
        var graph = SceneGraph(roots: [root, behind, tooFar], camera: camera)
        let rectangle = try SceneSelectionRectangle(from: SIMD2(0.65, 0.4), to: SIMD2(0.6, 0.35))
        #expect(rectangle.minimum == SIMD2(0.6, 0.35))
        #expect(try graph.select(in: rectangle, aspectRatio: 1).map(\.nodeID) == [posed.id, pointNode.id, lineNode.id])
        graph.cameraLayers = SceneLayers(layer: 3)
        #expect(try graph.select(in: rectangle, aspectRatio: 1).map(\.nodeID) == [offLayer.id])
        #expect(throws: ScenePickingError.invalidNormalizedPoint) {
            try SceneSelectionRectangle(from: .zero, to: SIMD2(.nan, 1))
        }
    }

    @Test("Orthographic selection respects instance origins, empty sets and reversed depth")
    func orthographicInstances() throws {
        let fixture = try GeometryRenderFixture()
        let mesh = try fixture.unlit(GeometryRenderFixture.square(2), supportsInstancing: true)
        let instances = try InstanceSet(device: fixture.device, transforms: [
            translationMatrix(-1, 0, -2), translationMatrix(0, 0, -4),
            translationMatrix(0, 0, -7), translationMatrix(-1, 0, 2),
        ])
        let node = SceneNode(transform: .init(translation: SIMD3(1, 0, 0)), mesh: mesh, instances: instances)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let rectangle = try SceneSelectionRectangle(from: SIMD2(0.5, 0.45), to: SIMD2(0.75, 0.55))
        for depthMode in [MeshDepthMode.forward, .reversed] {
            let camera = try orthographicCamera(depthMode)
            let selected = try graph.select(in: rectangle, camera: camera)
            #expect(selected.map(\.nodeID) == [node.id, node.id])
            #expect(selected.map(\.instanceIndex) == [0, 1])
        }
        try instances.replaceAll(transforms: [])
        #expect(try graph.select(in: rectangle, camera: orthographicCamera(.forward)).isEmpty)
    }

    @Test("Perspective selection includes centers on the camera's clip planes")
    func inclusiveClipPlanes() throws {
        let fixture = try GeometryRenderFixture()
        let mesh = try fixture.unlit(GeometryRenderFixture.square(2))
        let nodes = [-0.999, -1, -8, -8.001].map { (z: Float) in
            SceneNode(transform: .init(translation: SIMD3(0, 0, z)), mesh: mesh)
        }
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 1, far: 8)
        let graph = SceneGraph(roots: nodes, camera: camera)
        let rectangle = try SceneSelectionRectangle(from: SIMD2(0.4, 0.4), to: SIMD2(0.6, 0.6))
        #expect(try graph.select(in: rectangle, aspectRatio: 1).map(\.nodeID) == [nodes[1].id, nodes[2].id])
    }

    @Test("Selection depth may extend past the render camera without changing its framing")
    func customSelectionDepth() throws {
        let fixture = try GeometryRenderFixture()
        let mesh = try fixture.unlit(GeometryRenderFixture.square(2))
        let near = SceneNode(transform: .init(translation: SIMD3(0.6, 0, -3)), mesh: mesh)
        let far = SceneNode(transform: .init(translation: SIMD3(1.4, 0, -7)), mesh: mesh)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 1, far: 5)
        let graph = SceneGraph(roots: [near, far], camera: camera)
        let rectangle = try SceneSelectionRectangle(from: SIMD2(0.58, 0.4), to: SIMD2(0.62, 0.6))
        #expect(try graph.select(in: rectangle, aspectRatio: 1).map(\.nodeID) == [near.id])
        #expect(try graph.select(in: rectangle, aspectRatio: 1, maximumDepth: 7).map(\.nodeID) == [near.id, far.id])
        #expect(try graph.select(in: rectangle, aspectRatio: 1, maximumDepth: 2).isEmpty)
        #expect(throws: PerspectiveCamera.ValidationError.invalidClippingPlanes) {
            try graph.select(in: rectangle, aspectRatio: 1, maximumDepth: .nan)
        }
    }

    @Test("All mesh hits sort by distance, node, instance, and triangle")
    func sortedIntersections() throws {
        let fixture = try GeometryRenderFixture()
        let vertices = [SIMD3<Float>(-1, -1, 0), SIMD3(1, -1, 0), SIMD3(0, 1, 0)].map {
            MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
        }
        let mesh = try fixture.unlit(
            MeshGeometry(vertices: vertices, indices: [0, 1, 2, 0, 1, 2]), supportsInstancing: true
        )
        let instances = try InstanceSet(device: fixture.device, transforms: [
            translationMatrix(0, 0, -4), translationMatrix(0, 0, -2), translationMatrix(0, 0, -2)
        ])
        let first = SceneNode(mesh: mesh, instances: instances)
        let second = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let excluded = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: mesh, layers: SceneLayers(layer: 2)
        )
        let graph = SceneGraph(roots: [first, second, excluded], camera: fixture.camera)
        let hits = try graph.intersections(at: SIMD2(0.5, 0.5), camera: orthographicCamera(.forward))
        #expect(hits.map(\.nodeID) == [first.id, first.id, first.id, first.id, second.id, second.id, first.id, first.id])
        #expect(hits.map(\.instanceIndex) == [1, 1, 2, 2, nil, nil, 0, 0])
        #expect(hits.map(\.intersection.distance) == [1, 1, 1, 1, 1, 1, 3, 3])
        #expect(hits.map(\.intersection.triangleIndex) == [0, 1, 0, 1, 0, 1, 0, 1])
        let short = try SceneRay(origin: .zero, direction: SIMD3(0, 0, -1), maximumDistance: 2)
        #expect(try graph.intersections(ray: short).count == 6)
        #expect(try graph.pick(ray: short)?.instanceIndex == 1)
        #expect(try graph.intersections(at: SIMD2(0.5, 0.5), aspectRatio: 1).count == hits.count)
    }

    @Test("Native input reaches only the nearest direct child and preserves the hit UV")
    func groupedEvents() throws {
        let fixture = try GeometryRenderFixture()
        let normal = SIMD3<Float>(0, 0, 1)
        let vertices = [
            MeshVertex(position: SIMD3(0, 0, 0), normal: normal, textureCoordinate: SIMD2(0, 1)),
            MeshVertex(position: SIMD3(1, 0, 0), normal: normal, textureCoordinate: SIMD2(1, 1)),
            MeshVertex(position: SIMD3(0, 1, 0), normal: normal, textureCoordinate: SIMD2(0, 0)),
        ]
        let mesh = try fixture.unlit(MeshGeometry(vertices: vertices, indices: [0, 1, 2]))
        func node(_ z: Float) -> SceneNode {
            SceneNode(transform: .init(translation: SIMD3(0, 0, z)), mesh: mesh)
        }
        let outside = node(-1.5)
        let nested = SceneNode(children: [node(-2)])
        let front = node(-3)
        let rear = node(-4)
        let parent = SceneNode(children: [nested, rear, front])
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 1, far: 8)
        var graph = SceneGraph(roots: [outside, parent], camera: camera)
        let ray = try SceneRay(origin: SIMD3(0.25, 0.5, 0), direction: SIMD3(0, 0, -1))
        #expect(try graph.pick(ray: ray)?.nodeID == outside.id)
        var events: [SceneInteractionGroup.Event] = []
        var group = SceneInteractionGroup(nodeID: parent.id)
        group.setHandler(for: rear.id) { events.append($0) }
        #expect(try group.dispatch(.down, ray: ray, in: graph)?.nodeID == front.id)
        #expect(events.isEmpty)
        group.setHandler(for: front.id) { events.append($0) }
        let orthographic = try orthographicCamera(.reversed)
        let point = SIMD2<Float>(0.5625, 0.375)
        #expect(try group.dispatch(.click, at: point, camera: orthographic, depthMode: .reversed, in: graph)?.nodeID == front.id)
        let click = try #require(events.first)
        #expect(click.phase == .click)
        #expect(click.hit.intersection.barycentric == SIMD3(0.25, 0.25, 0.5))
        #expect(click.textureCoordinate == SIMD2(0.25, 0.5))
        #expect(click.hit.intersection.distance == 2)
        #expect(try group.dispatch(.move, at: SIMD2(0.54, 0.42), aspectRatio: 1, in: graph)?.nodeID == front.id)
        group.setHandler(for: front.id, nil)
        #expect(try group.dispatch(.up, ray: ray, in: graph)?.nodeID == front.id)
        #expect(events.count == 2)
        try graph.updateNode(withID: front.id) { $0.isVisible = false }
        #expect(try group.dispatch(.click, ray: ray, in: graph)?.nodeID == rear.id)
        #expect(events.last?.hit.nodeID == rear.id)
        let missing = SceneInteractionGroup(nodeID: UUID())
        #expect(throws: SceneGraph.EditingError.missingNode(missing.nodeID)) {
            try missing.dispatch(.click, ray: ray, in: graph)
        }
    }

    @Test("An asymmetric infinite perspective view emits a bounded-origin unbounded ray")
    func asymmetricCameraRay() throws {
        let near: Float = 0.5
        let camera = try ShaderGraphCamera(
            viewMatrix: translationMatrix(-3),
            projectionMatrix: simd_float4x4(columns: (
                SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0),
                SIMD4(0.25, -0.5, -1, -1), SIMD4(0, 0, -near, 0)
            )),
            near: near, far: .infinity
        )
        let ray = try camera.ray(at: SIMD2(0.5, 0.5))
        #expect(ray.origin == SIMD3(3.125, -0.25, -0.5))
        #expect(simd_length(ray.direction - simd_normalize(SIMD3<Float>(0.25, -0.5, -1))) < 1e-6)
        #expect(ray.maximumDistance == .infinity)
    }

    private func orthographicCamera(_ depthMode: MeshDepthMode) throws -> ShaderGraphCamera {
        let depthScale: Float = depthMode == .forward ? -0.25 : 0.25
        let depthOffset: Float = depthMode == .forward ? -0.25 : 1.25
        return try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: simd_float4x4(columns: (
                SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0),
                SIMD4(0, 0, depthScale, 0), SIMD4(0, 0, depthOffset, 1)
            )), near: 1, far: 5
        )
    }
}
