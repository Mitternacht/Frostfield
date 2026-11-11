import Frostfield
import Metal
import Testing
import simd

struct CollisionGeometryTests {
    @Test("Oriented boxes use all separating axes and clamp in box space")
    func orientedBoxes() throws {
        let rotation = simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1))
        let center = SIMD3<Float>(3, 5, 0)
        let halfSize = SIMD3<Float>(2, 0.25, 0.5)
        let box = try OrientedBoundingBox(center: center, halfSize: halfSize, rotation: rotation)
        let shifted = try OrientedBoundingBox(
            center: center + rotation.act(SIMD3(0, 0.75, 0)), halfSize: halfSize, rotation: rotation
        )
        let overlapping = try OrientedBoundingBox(
            center: center + rotation.act(SIMD3(0, 0.4, 0)), halfSize: halfSize, rotation: rotation
        )
        #expect(try box.axisAlignedBounds().intersects(shifted.axisAlignedBounds()))
        #expect(!box.intersects(shifted))
        #expect(box.intersects(overlapping) && overlapping.intersects(box))
        #expect(box.contains(center + rotation.act(SIMD3(1, 0.2, 0.4))))
        #expect(!box.contains(center + rotation.act(SIMD3(1, 0.3, 0.4))))
        let closest = try box.closestPoint(to: center + rotation.act(SIMD3(1, 2, -2)))
        #expect(simd_distance(closest, center + rotation.act(SIMD3(1, 0.25, -0.5))) < 1e-6)
        #expect(box.intersectsPlane(normal: SIMD3(0, 0, 2), offset: -1))
        #expect(!box.intersectsPlane(normal: SIMD3(0, 0, 2), offset: -1.01))
        #expect(box.intersects(try BoundingSphere(center: center + SIMD3(0, 0, 1), radius: 0.5)))
        #expect(!box.intersects(try BoundingSphere(center: center + SIMD3(0, 0, 1), radius: 0.49)))
    }

    @Test("Oriented box rays preserve world distance, clipping and inside exit")
    func orientedBoxRays() throws {
        let rotation = simd_quatf(angle: 0.6, axis: SIMD3(0, 1, 0))
        let center = SIMD3<Float>(3, 5, 1)
        let box = try OrientedBoundingBox(center: center, halfSize: SIMD3(1, 2, 0.5), rotation: rotation)
        let forward = rotation.act(SIMD3(0, 0, 1))
        let inside = try SceneRay(origin: center, direction: forward)
        let exit = try #require(try box.intersection(ray: inside))
        #expect(abs(exit.distance - 0.5) < 1e-6)
        #expect(simd_distance(exit.position, center + forward * 0.5) < 1e-6)
        let outside = center + forward * 3
        let entry = try #require(try box.intersection(ray: SceneRay(origin: outside, direction: -forward)))
        #expect(abs(entry.distance - 2.5) < 1e-6)
        #expect(try box.intersection(ray: SceneRay(origin: outside, direction: -forward, maximumDistance: 2)) == nil)
        #expect(try box.intersection(ray: SceneRay(origin: outside, direction: forward)) == nil)

        let axisAligned = try OrientedBoundingBox(center: .zero, halfSize: SIMD3(repeating: 1))
        let faceRay = try SceneRay(origin: SIMD3(1, 0, 0), direction: SIMD3(0, 1, 0), maximumDistance: 0)
        #expect(try axisAligned.intersection(ray: faceRay)?.distance == 0)
    }

    @Test("Oriented box transforms retain reflected corners and reject shear")
    func orientedBoxTransforms() throws {
        let local = try AxisAlignedBox(minimum: SIMD3(-1, -2, -3), maximum: SIMD3(2, 4, 6))
        let box = OrientedBoundingBox(local)
        let matrix = SceneTransform(
            translation: SIMD3(5, -4, 2),
            rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 0, 1)),
            scale: SIMD3(-2, 0.5, 1)
        ).matrix
        let moved = try box.transformed(by: matrix)
        let expectedCenter = matrix * SIMD4(local.center, 1)
        #expect(simd_distance(moved.center, SIMD3(expectedCenter.x, expectedCenter.y, expectedCenter.z)) < 1e-6)
        #expect(simd_distance(moved.halfSize, SIMD3(3, 1.5, 4.5)) < 1e-5)
        let expected = try local.transformed(by: matrix)
        let enclosure = try moved.axisAlignedBounds()
        #expect(enclosure.contains(expected.minimum) && enclosure.contains(expected.maximum))
        var shear = matrix_identity_float4x4
        shear.columns.1.x = 0.25
        #expect(throws: OrientedBoundingBox.Failure.invalidTransform) { try box.transformed(by: shear) }
        #expect(throws: OrientedBoundingBox.Failure.invalidHalfSize) {
            try OrientedBoundingBox(center: .zero, halfSize: SIMD3(1, -1, 1))
        }
        #expect(throws: OrientedBoundingBox.Failure.invalidRotation) {
            try OrientedBoundingBox(center: .zero, halfSize: .zero, rotation: simd_quatf(vector: .zero))
        }
    }

    @Test("Capsules detect segment, cap, box corner and capsule contact")
    func capsules() throws {
        let box = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 1))
        let corner = SIMD3<Float>(1.5, 1.5, 1.5)
        let missesCorner = try Capsule(start: corner, end: corner, radius: 0.75)
        #expect(try missesCorner.axisAlignedBounds().intersects(box))
        #expect(!missesCorner.intersects(box))
        #expect(!missesCorner.intersects(OrientedBoundingBox(box)))

        let tangent = try Capsule(start: SIMD3(-2, 2, 0.5), end: SIMD3(3, 2, 0.5), radius: 1)
        #expect(tangent.intersects(box))
        #expect(tangent.intersects(OrientedBoundingBox(box)))
        #expect(tangent.contains(SIMD3(-3, 2, 0.5)))
        #expect(!tangent.contains(SIMD3(-3.1, 2, 0.5)))
        #expect(try tangent.translated(by: SIMD3(0, 1, 0)).center == SIMD3(0.5, 3, 0.5))
        let segment = try Capsule(start: SIMD3(-2, 0.5, 0.5), end: SIMD3(3, 0.5, 0.5), radius: 0)
        #expect(segment.intersects(box))
        #expect(!segment.intersects(tangent))
        let crossing = try Capsule(start: SIMD3(0, -2, 0.5), end: SIMD3(0, 3, 0.5), radius: 0)
        #expect(segment.intersects(crossing) && crossing.intersects(segment))
        #expect(segment.intersects(try BoundingSphere(center: SIMD3(0, 0.75, 0.5), radius: 0.25)))
        #expect(throws: Capsule.Failure.invalidRadius) { try Capsule(start: .zero, end: .zero, radius: .infinity) }
        #expect(throws: Capsule.Failure.invalidEndpoint) {
            try Capsule(start: SIMD3(.nan, 0, 0), end: .zero, radius: 1)
        }
    }
}

struct MeshCollisionTests {
    @Test("Accelerated capsule and sphere contacts cover faces, edges and spine crossings")
    func roundedContacts() throws {
        let bvh = try MeshBVH(geometry: .plane(width: 4, height: 4))
        let parallel = try Capsule(start: SIMD3(0.75, 0, 0.25), end: SIMD3(1.2, 0, 0.25), radius: 0.5)
        let contact = try #require(try bvh.contacts(intersecting: parallel).first)
        #expect(contact.triangleIndex == 1)
        #expect(contact.normal == SIMD3(0, 0, 1))
        #expect(abs(contact.depth - 0.25) < 1e-6)
        #expect(contact.position == SIMD3(0.75, 0, 0))

        let piercing = try Capsule(start: SIMD3(1, 0, -0.2), end: SIMD3(1, 0, 0.65), radius: 0.2)
        let crossing = try #require(try bvh.contacts(intersecting: piercing).first)
        #expect(crossing.normal == SIMD3(0, 0, 1))
        #expect(abs(crossing.depth - 0.4) < 1e-6)
        #expect(crossing.position == SIMD3(1, 0, 0))

        let edgeSphere = try BoundingSphere(center: SIMD3(0, 2.1, 0.1), radius: 0.2)
        let edge = try #require(try bvh.contacts(intersecting: edgeSphere).first)
        #expect(edge.position == SIMD3(0, 2, 0))
        #expect(simd_distance(edge.normal, simd_normalize(SIMD3(0, 1, 1))) < 1e-6)
        #expect(abs(edge.depth - (0.2 - Float(0.02).squareRoot())) < 1e-6)

        let tangent = try BoundingSphere(center: SIMD3(1, 0, 0.25), radius: 0.25)
        #expect(try bvh.contacts(intersecting: tangent).first?.depth == 0)
        let misses = try BoundingSphere(center: SIMD3(0, 2.1, 0.2), radius: 0.2)
        #expect(try bvh.contacts(intersecting: misses).isEmpty)
    }

    @Test("Box contacts give the separating direction and agree with BVH box queries")
    func boxContacts() throws {
        let bvh = try MeshBVH(geometry: .plane(width: 4, height: 4))
        let box = try AxisAlignedBox(minimum: SIMD3(0.8, -0.2, -0.125), maximum: SIMD3(1.2, 0.2, 0.375))
        let contact = try #require(try bvh.contacts(intersecting: box).first)
        #expect(contact.normal == SIMD3(0, 0, 1))
        #expect(contact.depth == 0.125)
        let rotated = try OrientedBoundingBox(
            center: SIMD3(1, 0, 0.2), halfSize: SIMD3(0.2, 0.25, 0.35),
            rotation: simd_quatf(angle: .pi / 6, axis: SIMD3(1, 0, 0))
        )
        let tilted = try #require(try bvh.contacts(intersecting: rotated).first)
        #expect(simd_distance(tilted.normal, SIMD3(0, 0, 1)) < 1e-6)
        #expect(abs(tilted.depth - (0.25 * sin(Float.pi / 6) + 0.35 * cos(Float.pi / 6) - 0.2)) < 1e-6)

        let grid = try MeshBVH(geometry: .plane(width: 10, height: 10, columns: 10, rows: 10))
        var random = BVHTestRandom(seed: 187)
        for _ in 0..<100 {
            let center = random.vector(in: -5...5)
            let reach = SIMD3<Float>(0.6, 0.6, 0.9)
            let query = try AxisAlignedBox(minimum: center - reach, maximum: center + reach)
            let indices = grid.triangles(intersecting: query)
            #expect(try grid.contacts(intersecting: query).map(\.triangleIndex) == indices)
        }
    }

    @Test("Mesh contacts exclude degenerate triangles while box queries retain them")
    func degenerateSurface() throws {
        let geometry = try MeshGeometry(
            vertices: [MeshVertex(position: .zero, normal: SIMD3(0, 0, 1))], indices: [0, 0, 0]
        )
        let bvh = try MeshBVH(geometry: geometry)
        let bounds = try AxisAlignedBox(minimum: .zero, maximum: .zero)
        #expect(bvh.triangles(intersecting: bounds) == [0])
        #expect(try bvh.contacts(intersecting: bounds).isEmpty)
        #expect(try bvh.contacts(intersecting: BoundingSphere(center: .zero, radius: 1)).isEmpty)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneCollisionTests {
    @Test("Collision snapshots bake world transforms and instances with independent visibility")
    func sceneCollision() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(), supportsInstancing: true
        )
        let instances = try InstanceSet(device: device, transforms: [
            SceneTransform(translation: SIMD3(0, 0, -1), scale: SIMD3(-1, 1, 1)).matrix,
            SceneTransform(translation: SIMD3(2, 0, -1), scale: SIMD3(-1, 1, 1)).matrix,
        ])
        let collisionLayer = SceneLayers(layer: 4)
        let panels = SceneNode(mesh: mesh, instances: instances, layers: collisionLayer)
        let hidden = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -1)),
            mesh: mesh, isVisible: false, layers: collisionLayer
        )
        let root = SceneNode(
            transform: SceneTransform(translation: SIMD3(3, 0, -4)), children: [panels]
        )
        let graph = SceneGraph(
            roots: [root, hidden], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20)
        )
        #expect(try graph.collisionBVH() == nil)
        let visible = try #require(try graph.collisionBVH(layers: collisionLayer, visibleOnly: true))
        #expect(visible.triangleCount == 4)
        #expect(try visible.intersection(ray: SceneRay(origin: SIMD3(3, 0, 0), direction: SIMD3(0, 0, -1)))?.distance == 5)
        #expect(try visible.intersection(ray: SceneRay(origin: SIMD3(5, 0, 0), direction: SIMD3(0, 0, -1)))?.distance == 5)
        let capsule = try Capsule(start: SIMD3(5, 0, -4.75), end: SIMD3(5, 0.1, -4.75), radius: 0.5)
        #expect(try visible.contacts(intersecting: capsule).allSatisfy { $0.depth > 0 && $0.normal.z > 0 })
        #expect(try visible.contacts(intersecting: capsule).count == 2)
        let correction = try #require(try visible.depenetration(for: capsule))
        #expect(correction.isResolved && correction.iterations == 1)
        #expect(correction.translation == SIMD3(0, 0, 0.25))

        let complete = try #require(try graph.collisionBVH(layers: collisionLayer))
        #expect(complete.triangleCount == 6)
        #expect(try complete.intersection(ray: SceneRay(origin: .zero, direction: SIMD3(0, 0, -1)))?.distance == 1)
        try instances.replaceAll(transforms: [])
        #expect(try graph.collisionBVH(layers: collisionLayer, visibleOnly: true) == nil)
        #expect(visible.triangleCount == 4)
    }
}
