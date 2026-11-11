import Frostfield
import Testing
import simd

struct ScenePickingTests {
    @Test
    func `Camera ray maps the top-left pixel through both clip planes`() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 1, far: 5)
        let ray = try camera.ray(at: .zero, aspectRatio: 2)
        #expect(simd_length(ray.origin - SIMD3<Float>(-2, 1, -1)) < 1e-5)
        #expect(abs(simd_length(ray.direction) - 1) < 1e-6)
        let end = ray.origin + ray.direction * ray.maximumDistance
        #expect(simd_length(end - SIMD3<Float>(-10, 5, -5)) < 1e-5)
    }

    @Test
    func `Camera transform moves and scales the near-to-far ray`() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 1, far: 5)
        let transform = SceneTransform(
            translation: SIMD3(1, 2, 3),
            rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)),
            scale: SIMD3(1, 1, 2)
        )
        let ray = try camera.ray(at: SIMD2(0.5, 0.5), aspectRatio: 1, cameraTransform: transform)
        #expect(simd_length(ray.origin - SIMD3<Float>(-1, 2, 3)) < 1e-5)
        #expect(simd_length(ray.direction - SIMD3<Float>(-1, 0, 0)) < 1e-5)
        #expect(abs(ray.maximumDistance - 8) < 1e-5)
    }

    @Test
    func `Ray validates inputs and normalizes even large directions`() throws {
        let ray = try SceneRay(origin: .zero, direction: SIMD3(.greatestFiniteMagnitude, .greatestFiniteMagnitude, 0))
        #expect(abs(simd_length(ray.direction) - 1) < 1e-6)
        #expect(ray.maximumDistance == .infinity)
        #expect(throws: ScenePickingError.invalidOrigin) {
            try SceneRay(origin: SIMD3(.nan, 0, 0), direction: SIMD3(0, 0, -1))
        }
        #expect(throws: ScenePickingError.invalidDirection) {
            try SceneRay(origin: .zero, direction: .zero)
        }
        #expect(throws: ScenePickingError.invalidMaximumDistance) {
            try SceneRay(origin: .zero, direction: SIMD3(0, 0, -1), maximumDistance: -1)
        }

        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 1, far: 5)
        #expect(throws: ScenePickingError.invalidNormalizedPoint) {
            try camera.ray(at: SIMD2(1.1, 0.5), aspectRatio: 1)
        }
        #expect(throws: PerspectiveCamera.ValidationError.invalidAspectRatio) {
            try camera.ray(at: SIMD2(0.5, 0.5), aspectRatio: 0)
        }
    }

    @Test
    func `Nearest front triangle wins and equal distances keep the first index`() throws {
        let vertices = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(0, 0, 1), SIMD3<Float>(1, 0, 1), SIMD3<Float>(0, 1, 1),
        ].map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) }
        let mesh = try MeshGeometry(vertices: vertices, indices: [0, 0, 0, 0, 1, 2, 3, 4, 5, 3, 4, 5])
        let ray = try SceneRay(origin: SIMD3(0.25, 0.25, 3), direction: SIMD3(0, 0, -1))
        let hit = try #require(try mesh.intersection(ray: ray, localToWorld: matrix_identity_float4x4))
        #expect(hit.triangleIndex == 2)
        #expect(hit.distance == 2)
        #expect(hit.position == SIMD3<Float>(0.25, 0.25, 1))
    }

    @Test
    func `Mirrored nonuniform scale preserves the front face and world distance`() throws {
        let mesh = try MeshGeometry(
            vertices: [SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0)].map {
                MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
            },
            indices: [0, 1, 2]
        )
        let model = SceneTransform(translation: SIMD3(2, 1, -2), scale: SIMD3(-2, 3, 0.5)).matrix
        let origin = SIMD3<Float>(1.5, 1.75, 1)
        let front = try SceneRay(origin: origin, direction: SIMD3(0, 0, -2), maximumDistance: 3)
        let hit = try #require(try mesh.intersection(ray: front, localToWorld: model))
        #expect(hit.distance == 3)
        #expect(hit.position == SIMD3<Float>(1.5, 1.75, -2))

        let short = try SceneRay(origin: origin, direction: front.direction, maximumDistance: 2.99)
        #expect(try mesh.intersection(ray: short, localToWorld: model) == nil)
        let back = try SceneRay(origin: SIMD3(1.5, 1.75, -3), direction: SIMD3(0, 0, 1))
        #expect(try mesh.intersection(ray: back, localToWorld: model) == nil)
        let backHit = try #require(try mesh.intersection(ray: back, localToWorld: model, cullsBackFaces: false))
        #expect(backHit.distance == 1)
        #expect(backHit.position == hit.position)
        #expect(throws: ScenePickingError.invalidTransform) {
            try mesh.intersection(ray: front, localToWorld: SceneTransform(scale: SIMD3(1, 0, 1)).matrix)
        }
    }

    @Test
    func `Hit normals and both UVs follow the accepted side through a mirrored transform`() throws {
        let mesh = try MeshGeometry(vertices: [
            MeshVertex(position: .zero, normal: SIMD3(1, 0, 1), secondaryTextureCoordinate: SIMD2(0.25, 0.5)),
            MeshVertex(position: SIMD3(1, 0, 0), normal: SIMD3(0, 1, 1), textureCoordinate: SIMD2(1, 0),
                       secondaryTextureCoordinate: SIMD2(0.75, 0.5)),
            MeshVertex(position: SIMD3(0, 1, 0), normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(0, 1),
                       secondaryTextureCoordinate: SIMD2(0.25, 1)),
        ], indices: [0, 1, 2])
        let model = SceneTransform(translation: SIMD3(2, 1, -2), scale: SIMD3(-2, 3, 0.5)).matrix
        let front = try SceneRay(origin: SIMD3(1.5, 1.75, 0), direction: SIMD3(0, 0, -1))
        let back = try SceneRay(origin: SIMD3(1.5, 1.75, -3), direction: SIMD3(0, 0, 1))
        let hit = try #require(try mesh.intersection(ray: front, localToWorld: model, side: .front))
        #expect(hit.isFrontFace)
        #expect(hit.barycentric == SIMD3(0.5, 0.25, 0.25))
        #expect(hit.textureCoordinate == SIMD2(0.25, 0.25))
        #expect(hit.secondaryTextureCoordinate == SIMD2(0.375, 0.625))
        #expect(hit.faceNormal == SIMD3(0, 0, 1))
        #expect(hit.localFaceNormal == SIMD3(0, 0, 1))
        let localNormal = simd_normalize(SIMD3<Float>(0.5, 0.25, 1))
        let worldNormal = simd_normalize(SIMD3<Float>(-0.25, 0.25 / 3, 2))
        #expect(simd_distance(hit.localNormal, localNormal) < 1e-6)
        #expect(simd_distance(hit.normal, worldNormal) < 1e-6)
        #expect(try mesh.intersections(ray: front, localToWorld: model, side: .back).isEmpty)
        #expect(try mesh.intersections(ray: back, localToWorld: model, side: .front).isEmpty)
        let backHit = try #require(try mesh.intersection(ray: back, localToWorld: model, side: .back))
        #expect(!backHit.isFrontFace)
        #expect(backHit.faceNormal == hit.faceNormal)
        #expect(backHit.normal == -hit.normal)
        #expect(backHit.localNormal == -hit.localNormal)
        let bvh = try MeshBVH(geometry: mesh)
        #expect(try bvh.intersection(ray: back, localToWorld: model, side: .back) == backHit)
        #expect(try bvh.intersections(ray: front, localToWorld: model, side: .back).isEmpty)
    }

    @Test
    func `A reused BVH returns attributes of the caller geometry`() throws {
        let positions = [SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0)]
        let original = try MeshGeometry(vertices: positions.map {
            MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
        }, indices: [0, 1, 2])
        let edited = try MeshGeometry(vertices: positions.map {
            MeshVertex(position: $0, normal: SIMD3(0, 1, 1), textureCoordinate: SIMD2(0.5, 0.25))
        }, indices: original.indices)
        let ray = try SceneRay(origin: SIMD3(0.25, 0.25, 2), direction: SIMD3(0, 0, -1))
        let bvh = try MeshBVH(geometry: original)
        let linear = try #require(try edited.intersection(ray: ray, localToWorld: matrix_identity_float4x4))
        #expect(linear.textureCoordinate == SIMD2(0.5, 0.25))
        #expect(linear.secondaryTextureCoordinate == nil)
        #expect(simd_distance(linear.normal, simd_normalize(SIMD3<Float>(0, 1, 1))) < 1e-6)
        #expect(try edited.intersection(ray: ray, localToWorld: matrix_identity_float4x4, cullsBackFaces: true, bvh: bvh) == linear)
        #expect(try edited.intersections(ray: ray, localToWorld: matrix_identity_float4x4, side: .front, bvh: bvh) == [linear])
    }

    @Test
    func `Back-only scene raycasts agree with raster for plain instanced and batched meshes`() throws {
        let fixture = try GeometryRenderFixture()
        let mesh = try MeshGeometry(vertices: [SIMD3<Float>(-1, -1, 0), SIMD3<Float>(1, -1, 0), SIMD3<Float>(0, 1, 0)].map {
            MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
        }, indices: [0, 1, 2])
        let state = MeshRenderState(side: .back)
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: mesh, supportsInstancing: true, renderState: state
        )
        let front = SceneTransform(translation: SIMD3(0, 0, -3), scale: SIMD3(-1, 1, 1))
        let back = SceneTransform(
            translation: SIMD3(0, 0, -4), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        )
        let instances = try InstanceSet(device: fixture.device, transforms: [back.matrix])
        let batch = try BatchedMesh(device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [mesh],
                                    capacity: 2, renderState: state)
        try batch.addInstance(geometryIndex: 0, transform: front.matrix)
        let backID = try batch.addInstance(geometryIndex: 0, transform: back.matrix)
        let frontNode = SceneNode(transform: front, mesh: renderer)
        let backNode = SceneNode(transform: back, mesh: renderer)
        let instanceNode = SceneNode(mesh: renderer, instances: instances)
        let batchNode = SceneNode(batch: batch)
        let center = SIMD2<Float>(repeating: 0.5)
        let frontOnly = SceneGraph(roots: [frontNode], camera: fixture.camera)
        #expect(try frontOnly.pick(at: center, aspectRatio: 1) == nil)
        #expect(try fixture.pixel(fixture.render(frontOnly).pixels, x: 32, y: 32).w == 0)
        for node in [backNode, instanceNode, batchNode] {
            let graph = SceneGraph(roots: [frontNode, node], camera: fixture.camera)
            let hit = try #require(try graph.pick(at: center, aspectRatio: 1))
            #expect(hit.nodeID == node.id)
            #expect(!hit.intersection.isFrontFace)
            #expect(hit.intersection.normal.z > 0.999)
            #expect(try graph.intersections(at: center, aspectRatio: 1) == [hit])
            #expect(try fixture.pixel(fixture.render(graph).pixels, x: 32, y: 32).w > 0.99)
            if node.id == batchNode.id { #expect(hit.batchInstanceID == backID) }
            if node.id == instanceNode.id { #expect(hit.instanceIndex == 0) }
        }
    }
}
