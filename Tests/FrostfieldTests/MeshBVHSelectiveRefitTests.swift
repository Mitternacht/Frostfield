import Frostfield
import Testing
import simd

struct MeshBVHSelectiveRefitTests {
    @Test func `A shared indexed vertex refits distant leaves and its repeated-index point`() throws {
        let geometry = try sharedVertexMesh()
        let original = try MeshBVH(geometry: geometry)
        let leaves = try leafBounds(original)
        let leftLeaf = try #require(leaves.first { $0.value.triangles.contains(0) }?.key)
        let rightLeaf = try #require(leaves.first { $0.value.triangles.contains(1) }?.key)
        #expect(leftLeaf != rightLeaf)

        let apex = SIMD3<Float>(0, 0, 4)
        let revised = try original.refitted(updatingPositions: [0: apex])
        let expectedGeometry = try replacing(geometry, positions: [0: apex])
        let full = try original.refitted(to: expectedGeometry)
        let rebuilt = try MeshBVH(geometry: expectedGeometry)
        let movedLeaves = try leafBounds(revised)
        #expect(revised.nodeCount == original.nodeCount)
        #expect(revised.depth == original.depth)
        #expect(revised.bounds == full.bounds)
        #expect(revised.bounds == rebuilt.bounds)
        #expect(revised.geometry.indices == original.geometry.indices)
        #expect(revised.geometry.groups == original.geometry.groups)
        #expect(revised.geometry.vertices[0].normal == geometry.vertices[0].normal)
        #expect(revised.geometry.vertices[0].textureCoordinate == geometry.vertices[0].textureCoordinate)
        #expect(revised.geometry.vertices[0].secondaryTextureCoordinate == geometry.vertices[0].secondaryTextureCoordinate)
        #expect(revised.geometry.vertices[0].tangent == geometry.vertices[0].tangent)
        #expect(revised.geometry.vertices[0].color == geometry.vertices[0].color)
        for (index, leaf) in leaves {
            let moved = try #require(movedLeaves[index])
            #expect(moved.triangles == leaf.triangles)
            if leaf.triangles.contains(where: { $0 <= 2 }) {
                #expect(moved.bounds.maximum.z == 4)
            } else {
                #expect(moved.bounds == leaf.bounds)
            }
        }

        let probes: [SIMD3<Float>] = [SIMD3(-2, -1, 3.5), SIMD3(2, 1, 3.5)]
        for (triangle, origin) in probes.enumerated() {
            let ray = try SceneRay(origin: origin, direction: SIMD3(0, 0, -1), maximumDistance: 1.5)
            let hit = try #require(try revised.intersection(ray: ray))
            #expect(hit.triangleIndex == triangle)
            #expect(hit.distance == 1.5)
            #expect(hit.position.z == 2)
            #expect(try original.intersection(ray: ray) == nil)
            #expect(try full.intersection(ray: ray) == hit)
            #expect(try rebuilt.intersection(ray: ray) == hit)
            let closest = try #require(try revised.closestPoint(to: hit.position, maximumDistance: 0))
            #expect(closest.triangleIndex == triangle)
            #expect(closest.distance == 0)
        }
        let apexBox = try AxisAlignedBox(minimum: apex, maximum: apex)
        #expect(revised.triangles(intersecting: apexBox) == [0, 1, 2])
        #expect(original.triangles(intersecting: apexBox).isEmpty)

        let probe = try MeshBVH(geometry: .plane(width: 0.25, height: 0.25))
        let placement = SceneTransform(translation: SIMD3(-2, -1, 2)).matrix
        let pair = try #require(try revised.closestPoints(to: probe, otherLocalToWorld: placement, maximumDistance: 0))
        #expect(pair.first.triangleIndex == 0)
        #expect(pair.distance == 0)
        #expect(try rebuilt.closestPoints(to: probe, otherLocalToWorld: placement, maximumDistance: 0) == pair)

        let restored = try revised.refitted(updatingPositions: [0: .zero])
        #expect(restored.bounds == original.bounds)
        #expect(restored.triangles(intersecting: apexBox).isEmpty)
        #expect(revised.triangles(intersecting: apexBox) == [0, 1, 2])
        #expect(original.geometry.vertices[0].position == .zero)
        #expect(revised.geometry.vertices[0].position == apex)
    }

    @Test func `No-op and unused vertex updates preserve bounds and validate every edit`() throws {
        let source = try sharedVertexMesh()
        let original = try MeshBVH(geometry: source)
        for edits: [Int: SIMD3<Float>] in [[:], [0: source.vertices[0].position]] {
            let unchanged = try original.refitted(updatingPositions: edits)
            #expect(unchanged.bounds == original.bounds)
            #expect(unchanged.geometry.vertices.withUnsafeBufferPointer { updated in
                source.vertices.withUnsafeBufferPointer { $0.baseAddress == updated.baseAddress }
            })
        }

        let unused = source.vertices.count - 1
        #expect(!source.indices.contains(UInt32(unused)))
        let moved = try original.refitted(updatingPositions: [unused: SIMD3(90, 100, 110)])
        #expect(moved.bounds == original.bounds)
        #expect(moved.geometry.vertices[unused].position == SIMD3(90, 100, 110))
        #expect(moved.geometry.groups == source.groups)
        #expect(moved.geometry.vertices[0].position == source.vertices[0].position)
        #expect(try leafBounds(moved).keys.sorted() == leafBounds(original).keys.sorted())

        #expect(throws: MeshBVH.Failure.refitVertexIndex(-1)) {
            try original.refitted(updatingPositions: [-1: .zero, 0: SIMD3(0, 0, 4)])
        }
        #expect(throws: MeshBVH.Failure.refitVertexIndex(source.vertices.count)) {
            try original.refitted(updatingPositions: [source.vertices.count: .zero])
        }
        #expect(throws: MeshBVH.Failure.nonfiniteRefitPosition(0)) {
            try original.refitted(updatingPositions: [0: SIMD3(.nan, 0, 0), 1: SIMD3(1, 1, 1)])
        }
        #expect(throws: MeshBVH.Failure.nonfiniteRefitPosition(1)) {
            try original.refitted(updatingPositions: [1: SIMD3(0, .infinity, 0)])
        }
        #expect(original.geometry.vertices[0].position == source.vertices[0].position)
    }

    @Test func `Repeated sparse edits match rebuilds through bounded ray and box queries`() throws {
        let source = try MeshGeometry.plane(width: 12, height: 10, columns: 14, rows: 12)
        var snapshot = try MeshBVH(geometry: source)
        var reference = source
        var random = BVHTestRandom(seed: 0x4F27)
        for revision in 0..<3 {
            let start = revision * 20
            let edits = Dictionary(uniqueKeysWithValues: (start..<(start + 15)).map { index in
                (index, reference.vertices[index].position + random.vector(in: -1.5...1.5))
            })
            snapshot = try snapshot.refitted(updatingPositions: edits)
            reference = try replacing(reference, positions: edits)
            let rebuilt = try MeshBVH(geometry: reference)
            #expect(snapshot.bounds == rebuilt.bounds)
            for _ in 0..<40 {
                let point = SIMD3(random.float(in: -8...8), random.float(in: -7...7), Float(2))
                let ray = try SceneRay(origin: point, direction: SIMD3(0, 0, -1), maximumDistance: 3)
                #expect(try snapshot.intersection(ray: ray, cullsBackFaces: false)
                    == rebuilt.intersection(ray: ray, cullsBackFaces: false))
                #expect(try snapshot.intersections(ray: ray, cullsBackFaces: false)
                    == rebuilt.intersections(ray: ray, cullsBackFaces: false))
                #expect(try snapshot.closestPoint(to: point, maximumDistance: 3)
                    == rebuilt.closestPoint(to: point, maximumDistance: 3))
                let extent = SIMD3<Float>(0.25, 0.25, 2)
                let box = try AxisAlignedBox(minimum: point - extent, maximum: point + extent)
                #expect(snapshot.triangles(intersecting: box) == rebuilt.triangles(intersecting: box))
            }
        }
    }

    private func sharedVertexMesh() throws -> MeshGeometry {
        let positions: [SIMD3<Float>] = [
            .zero, SIMD3(-4, 0, 0), SIMD3(-4, -4, 0), SIMD3(4, 0, 0), SIMD3(4, 4, 0),
        ]
        var vertices = positions.map { point in
            MeshVertex(position: point, normal: SIMD3(0.25, 0, 1), textureCoordinate: SIMD2(0.125, 0.5),
                       secondaryTextureCoordinate: SIMD2(0.75, 0.25), tangent: SIMD4(1, 0, 0, 1),
                       color: SIMD4(0.25, 0.5, 0.75, 1))
        }
        var indices: [UInt32] = [0, 1, 2, 0, 3, 4, 0, 0, 0]
        for x: Float in [-32, -26, -20, -14, -8, 8, 14, 20, 26, 32] {
            let start = UInt32(vertices.count)
            vertices += [SIMD3(x, -1, 0), SIMD3(x + 2, -1, 0), SIMD3(x, 1, 0)].map {
                MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
            }
            indices += [start, start + 1, start + 2]
        }
        vertices.append(MeshVertex(position: SIMD3(1, 2, 3), normal: SIMD3(0, 0, 1)))
        return try MeshGeometry(vertices: vertices, indices: indices).withGroups([
            MeshGeometryGroup(start: 0, count: 9, materialIndex: 2),
            MeshGeometryGroup(start: 9, count: 3, materialIndex: 3),
        ])
    }

    private func replacing(_ geometry: MeshGeometry, positions: [Int: SIMD3<Float>]) throws -> MeshGeometry {
        let vertices = geometry.vertices.enumerated().map { index, vertex in
            MeshVertex(
                position: positions[index] ?? vertex.position, normal: vertex.normal,
                textureCoordinate: vertex.textureCoordinate, secondaryTextureCoordinate: vertex.secondaryTextureCoordinate,
                tangent: vertex.tangent, color: vertex.color
            )
        }
        return try MeshGeometry(vertices: vertices, indices: geometry.indices).withGroups(geometry.groups)
    }

    private func leafBounds(_ bvh: MeshBVH) throws -> [Int: (bounds: AxisAlignedBox, triangles: [Int])] {
        var result: [Int: (bounds: AxisAlignedBox, triangles: [Int])] = [:]
        _ = try bvh.shapecast(classifyBounds: { _, _ in .intersecting }, visitRange: { node, range, _ in
            result[node.index] = (node.bounds, Array(range))
            return false
        })
        return result
    }
}
