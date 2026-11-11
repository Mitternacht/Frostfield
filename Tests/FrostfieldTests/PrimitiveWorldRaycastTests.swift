import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PrimitiveWorldRaycastTests {
    @Test("Point hits use exact world thresholds and retain index order under reflected nonuniform scale")
    func transformedPoints() throws {
        let cloud = try PointCloud(positions: [
            SIMD3(0, 2, 4), SIMD3(0, 0, 2), SIMD3(0, -1, 4), SIMD3(0.5, 0, 3),
            SIMD3(0, 0, 5.5), SIMD3(0, 0, -0.25), SIMD3(0, 0, 5),
        ], sizes: [1, 0, 1, 1, 1, 1, 1])
        let transform = SceneTransform(translation: SIMD3(1, 0, 0), scale: SIMD3(-2, 0.25, 1)).matrix
        let ray = try SceneRay(origin: SIMD3(1, 0, 0), direction: SIMD3(0, 0, 4), maximumDistance: 5)
        let hits = try cloud.intersections(ray: ray, localToWorld: transform, threshold: 0.5)
        #expect(hits.map(\.element) == [.point(5), .point(0), .point(2), .point(6)])
        #expect(hits.map(\.distance) == [0, 4, 4, 5])
        #expect(hits.map(\.distanceToRay) == [0.25, 0.5, 0.25, 0])
        #expect(hits[1].position == SIMD3(1, 0.5, 4))
        #expect(hits[1].rayPosition == SIMD3(1, 0, 4))
        #expect(hits[0].rayPosition == ray.origin)
        #expect(hits.allSatisfy { $0.segmentParameter == nil })
        #expect(try cloud.intersections(ray: ray, localToWorld: transform, threshold: 0.49).map(\.element)
            == [.point(5), .point(2), .point(6)])
    }

    @Test("A zero threshold can hit an exact point on a non-axis-aligned ray")
    func exactPoint() throws {
        let cloud = try PointCloud(positions: [SIMD3(1, 1, 0), SIMD3(1, 1, 0.001)])
        let ray = try SceneRay(origin: .zero, direction: SIMD3(4, 4, 0))
        let hits = try cloud.intersections(ray: ray, threshold: 0)
        #expect(hits.map(\.element) == [.point(0)])
        #expect(hits.first?.distanceToRay == 0)
        #expect(hits.first?.rayPosition == SIMD3(1, 1, 0))
    }

    @Test("Line hits handle interiors, parallel pairs, ray endpoints and degenerate segments")
    func segmentCases() throws {
        let geometry = try LineGeometry(positions: [
            SIMD3(-2, 0.25, 3), SIMD3(2, 0.25, 3),
            SIMD3(0.25, 0, 4), SIMD3(0.25, 0, 1),
            SIMD3(0, 0.25, -2), SIMD3(0, 0.25, 2),
            SIMD3(0, 0, 2), SIMD3(0, 0, 2),
            SIMD3(-1, 0, -0.25), SIMD3(1, 0, -0.25),
            SIMD3(0, 0, 5.25), SIMD3(1, 0, 5.25),
            SIMD3(0.125, 0, 1), SIMD3(-0.125, 0, 5),
            SIMD3(-1, 0, 5), SIMD3(1, 0, 5),
            SIMD3(0.25, 0, -2), SIMD3(0.25, 0, -1),
        ], topology: .segments)
        let ray = try SceneRay(origin: .zero, direction: SIMD3(0, 0, 1), maximumDistance: 5)
        let hits = try geometry.intersections(ray: ray, threshold: 0.25)
        #expect(hits.map(\.element) == [.segment(2), .segment(4), .segment(1), .segment(3), .segment(0), .segment(6), .segment(7)])
        #expect(hits.map(\.distance) == [0, 0, 1, 2, 3, 3, 5])
        #expect(hits.map(\.segmentParameter) == [0.5, 0.5, 1, 0, 0.5, 0.5, 0.5])
        #expect(hits.map(\.distanceToRay) == [0.25, 0.25, 0.25, 0, 0.25, 0, 0])
        #expect(hits[2].position == SIMD3(0.25, 0, 1) && hits[2].rayPosition == SIMD3(0, 0, 1))
        #expect(try geometry.intersections(ray: ray, threshold: 0).map(\.element) == [.segment(3), .segment(6), .segment(7)])
    }

    @Test("Loops include their closing segment and strips use drawn segment indices")
    func topologyAndTransform() throws {
        let positions = [SIMD3<Float>(-1, -1, 3), SIMD3(-1, 1, 3), SIMD3(1, 1, 3), SIMD3(1, -1, 3)]
        let ray = try SceneRay(origin: .zero, direction: SIMD3(0, 0, 1))
        let transform = SceneTransform(scale: SIMD3(-2, 0.25, 1)).matrix
        let loop = try LineGeometry(positions: positions, topology: .loop)
        let strip = try LineGeometry(positions: [positions[0]] + positions)
        let loopHits = try loop.intersections(ray: ray, localToWorld: transform, threshold: 0.25)
        #expect(loopHits.map(\.element) == [.segment(1), .segment(3)])
        #expect(loopHits.map(\.position) == [SIMD3(0, 0.25, 3), SIMD3(0, -0.25, 3)])
        #expect(loopHits.map(\.segmentParameter) == [0.5, 0.5])
        #expect(try strip.intersections(ray: ray, localToWorld: transform, threshold: 0.25).map(\.element) == [.segment(1)])
    }

    @Test("World ray thresholds, affine transforms and representable hit values are validated")
    func invalidInputs() throws {
        let cloud = try PointCloud(positions: [.zero])
        let line = try LineGeometry(positions: [.zero, SIMD3(0, 1, 0)])
        let ray = try SceneRay(origin: .zero, direction: SIMD3(1, 0, 0), maximumDistance: 0)
        for invalid: Float in [-1, .nan, .infinity] {
            #expect(throws: PrimitivePickingError.tolerance) { try cloud.intersections(ray: ray, threshold: invalid) }
            #expect(throws: PrimitivePickingError.tolerance) { try line.intersections(ray: ray, threshold: invalid) }
        }
        let collapsed = SceneTransform(scale: SIMD3(1, 0, 1)).matrix
        #expect(throws: ScenePickingError.invalidTransform) { try cloud.intersections(ray: ray, localToWorld: collapsed) }
        var projective = matrix_identity_float4x4
        projective.columns.2.w = 1
        #expect(throws: ScenePickingError.invalidTransform) { try line.intersections(ray: ray, localToWorld: projective) }
        #expect(try cloud.intersections(ray: ray, threshold: 0).first?.distance == 0)

        let distant = try PointCloud(positions: [SIMD3(.greatestFiniteMagnitude, 0, 0)])
        let unbounded = try SceneRay(origin: .zero, direction: SIMD3(1, 0, 0))
        let scale = SceneTransform(scale: SIMD3(2, 1, 1)).matrix
        #expect(throws: ScenePickingError.unrepresentableResult) {
            try distant.intersections(ray: unbounded, localToWorld: scale, threshold: 0)
        }
        #expect(try distant.intersections(ray: ray, localToWorld: scale, threshold: 0).isEmpty)
    }

    @Test("A scene world ray merges points, lines, meshes, instances and batch slots")
    func mixedGraph() throws {
        let fixture = try GeometryRenderFixture()
        let device = fixture.device
        let ray = try SceneRay(origin: SIMD3(0, 0, 2), direction: SIMD3(0, 0, -1), maximumDistance: 5)
        let pointRenderer = try PointsRenderer(
            device: device, cloud: PointCloud(positions: [SIMD3(0.25, 0, 0)]), material: .init(size: .pixels(100))
        )
        let point = SceneNode(transform: .init(translation: SIMD3(0, 0, 0.5)), primitive: .points(pointRenderer))
        let lineRenderer = try LineRenderer(
            device: device, geometry: LineGeometry(positions: [SIMD3(-1, 0.125, -1), SIMD3(1, 0.125, -1)]),
            material: .init(width: .pixels(100))
        )
        let line = SceneNode(primitive: .lines(lineRenderer))
        let vertices = [SIMD3<Float>(-1, -1, 0), SIMD3(1, -1, 0), SIMD3(0, 1, 0)].map {
            MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
        }
        let geometry = try MeshGeometry(vertices: vertices, indices: [0, 1, 2])
        let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: geometry, supportsInstancing: true)
        let meshNode = SceneNode(transform: .init(translation: SIMD3(0, 0, -1)), mesh: mesh)
        let instances = try InstanceSet(device: device, transforms: [matrix_identity_float4x4, translationMatrix(3)])
        let instanceNode = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh, instances: instances)
        let batch = try BatchedMesh(device: device, colorPixelFormat: .rgba16Float, geometries: [geometry], capacity: 2)
        let batchID = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -3))
        try batch.addInstance(geometryIndex: 0, isVisible: false)
        let batchNode = SceneNode(batch: batch)
        let sprite = try SpriteRenderer(device: device, sprites: [.init(position: .zero, size: SIMD2(2, 2))])
        let spriteNode = SceneNode(primitive: .sprites(sprite))
        var offLayer = SceneNode(primitive: .points(pointRenderer))
        offLayer.layers = .init(layer: 4)
        let hidden = SceneNode(children: [SceneNode(primitive: .points(pointRenderer))], isVisible: false)
        let group = SceneNode(children: [line, meshNode, point, SceneNode(children: [instanceNode])])
        let graph = SceneGraph(roots: [group, batchNode, spriteNode, offLayer, hidden], camera: fixture.camera)

        let hits = try graph.raycast(ray: ray, pointThreshold: 0.25, lineThreshold: 0.125)
        #expect(hits.map(\.nodeID) == [point.id, line.id, meshNode.id, instanceNode.id, batchNode.id])
        #expect(hits.map(\.distance) == [1.5, 3, 3, 4, 5])
        let meshHits = hits.compactMap { if case .mesh(let hit) = $0 { hit } else { nil } }
        #expect(meshHits.map(\.instanceIndex) == [nil, 0, nil])
        #expect(meshHits.map(\.batchInstanceID) == [nil, nil, batchID])
        #expect(try graph.raycast(ray: ray, pointThreshold: 0.249, lineThreshold: 0.124).map(\.nodeID)
            == [meshNode.id, instanceNode.id, batchNode.id])
        #expect(try graph.raycast(ray: ray, in: group.id).map(\.nodeID) == [point.id, line.id, meshNode.id])
        #expect(try graph.raycast(ray: ray, layers: .init(layer: 4)).map(\.nodeID) == [offLayer.id])
        let missing = UUID()
        #expect(throws: SceneGraph.EditingError.missingNode(missing)) { try graph.raycast(ray: ray, in: missing) }
        #expect(throws: PrimitivePickingError.tolerance) { try graph.raycast(ray: ray, lineThreshold: .infinity) }
    }

    @Test(arguments: [false, true])
    func `A controller ray hits the rendered sprite plane and atlas from another origin`(orthographic: Bool) throws {
        let fixture = try GeometryRenderFixture()
        let projection: simd_float4x4
        if orthographic {
            projection = simd_float4x4(columns: (
                SIMD4(0.5, 0, 0, 0), SIMD4(0, 1, 0, 0),
                SIMD4(0, 0, -0.1, 0), SIMD4(0, 0, 0, 1)
            ))
        } else {
            projection = try fixture.camera.projection(aspectRatio: 1)
        }
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection,
            near: orthographic ? 0 : fixture.camera.near, far: orthographic ? 10 : fixture.camera.far
        )
        let x: Float = orthographic ? 0.25 : 0.5
        let sprite = try SpriteRenderer(
            device: fixture.device,
            sprites: [Sprite(position: .zero, size: SIMD2(16, 16), textureRegion: SIMD4(0.25, 0.5, 0.5, 0.25))],
            material: SpriteMaterial(sizeAttenuation: false)
        )
        let pose = SceneTransform(translation: SIMD3(0, 0, -4))
        let viewport = SIMD2<Float>(64, 64)
        let ray = try SceneRay(origin: SIMD3(x + 1, 0, 0), direction: SIMD3(-0.25, 0, -1))
        let hit = try #require(try sprite.intersections(
            ray: ray, localToWorld: pose.matrix, camera: camera, drawableSize: viewport
        ).first)
        #expect(hit.element == .sprite(0))
        #expect(simd_distance(hit.position, SIMD3(x, 0, -4)) < 1e-6)
        #expect(hit.position == hit.rayPosition)
        #expect(abs(hit.distance - Float(17).squareRoot()) < 1e-6)
        #expect(hit.textureCoordinate == SIMD2(0.625, 0.625))

        var node = SceneNode(primitive: .sprites(sprite))
        node.transform = pose
        let parent = SceneNode(children: [node])
        let graph = SceneGraph(roots: [parent], camera: fixture.camera)
        var group = SceneInteractionGroup(nodeID: parent.id)
        var received: [SceneRaycastHit] = []
        group.setDrawableHandler(for: node.id) { received.append($0.hit) }
        let routed = try #require(try group.dispatch(.click, ray: ray, camera: camera, drawableSize: viewport, in: graph))
        #expect(routed.nodeID == node.id)
        #expect(routed.textureCoordinate == hit.textureCoordinate)
        #expect(received == [routed])
        let short = try SceneRay(origin: ray.origin, direction: ray.direction, maximumDistance: 4)
        #expect(try graph.raycast(ray: short, camera: camera, drawableSize: viewport).isEmpty)
        let parallel = try SceneRay(origin: SIMD3(0, 0, -4), direction: SIMD3(1, 0, 0))
        #expect(try graph.raycast(ray: parallel, camera: camera, drawableSize: viewport).isEmpty)
    }
}
