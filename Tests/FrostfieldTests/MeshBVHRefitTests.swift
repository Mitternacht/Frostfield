import Frostfield
import Testing
import simd

struct MeshBVHRefitTests {
    @Test func `Refit moves bounded ray and proximity witnesses and changes overlap pairs`() throws {
        let source = try MeshGeometry.plane(width: 4, height: 4, columns: 4, rows: 4)
        let original = try MeshBVH(geometry: source)
        let changed = try remap(source) { _, vertex in
            MeshVertex(position: vertex.position + SIMD3(10, 0, 2), normal: vertex.normal,
                       textureCoordinate: SIMD2(0.25, 0.75))
        }
        let refitted = try original.refitted(to: changed)
        let rebuilt = try MeshBVH(geometry: changed)
        #expect(refitted.nodeCount == original.nodeCount)
        #expect(refitted.depth == original.depth)
        #expect(refitted.triangleCount == original.triangleCount)
        #expect(refitted.bounds == rebuilt.bounds)
        #expect(refitted.bounds.minimum == SIMD3(8, -2, 2))
        #expect(original.bounds.minimum == SIMD3(-2, -2, 0))

        let query = SIMD3<Float>(9.75, 0.125, 3)
        let ray = try SceneRay(origin: query, direction: SIMD3(0, 0, -1), maximumDistance: 1)
        let hit = try #require(try refitted.intersection(ray: ray))
        #expect(hit.distance == 1)
        #expect(hit.position == SIMD3(9.75, 0.125, 2))
        #expect(hit.textureCoordinate == SIMD2(0.25, 0.75))
        #expect(try rebuilt.intersection(ray: ray) == hit)
        #expect(try changed.intersection(ray: ray, localToWorld: matrix_identity_float4x4, cullsBackFaces: true, bvh: refitted) == hit)
        #expect(try original.intersection(ray: ray) == nil)
        #expect(throws: MeshBVH.Failure.geometryMismatch) {
            try changed.intersection(ray: ray, localToWorld: matrix_identity_float4x4, cullsBackFaces: true, bvh: original)
        }
        let nearest = try #require(try refitted.closestPoint(to: query, maximumDistance: 1))
        #expect(nearest.position == hit.position)
        #expect(nearest.distance == 1)
        #expect(try rebuilt.closestPoint(to: query, maximumDistance: 1) == nearest)
        #expect(try original.closestPoint(to: query, maximumDistance: 1) == nil)
        #expect(try refitted.closestPoint(to: query, maximumDistance: Double(1).nextDown) == nil)

        let probe = try MeshBVH(geometry: .plane(width: 0.25, height: 0.25))
        let placement = SceneTransform(translation: SIMD3(9.75, 0.125, 2)).matrix
        let pair = try #require(try refitted.closestPoints(to: probe, otherLocalToWorld: placement, maximumDistance: 0))
        #expect(pair.distance == 0)
        #expect(try rebuilt.closestPoints(to: probe, otherLocalToWorld: placement, maximumDistance: 0) == pair)
        #expect(try original.closestPoints(to: probe, otherLocalToWorld: placement, maximumDistance: 0) == nil)
        #expect(try overlaps(refitted, probe, placement: placement) == overlaps(rebuilt, probe, placement: placement))
        #expect(try !overlaps(refitted, probe, placement: placement).isEmpty)
        #expect(try overlaps(original, probe, placement: placement).isEmpty)
    }

    @Test func `Refit matches fresh bounds and queries after nonuniform vertex movement`() throws {
        let source = try MeshGeometry.plane(width: 6, height: 8, columns: 6, rows: 8)
        let original = try MeshBVH(geometry: source)
        let posed = try remap(source) { index, vertex in
            let p = vertex.position
            let ripple = Float(index % 4) * 0.2
            return MeshVertex(position: SIMD3(-p.y * 1.4, p.x * 0.75 + p.y * 0.4, ripple),
                              normal: vertex.normal, textureCoordinate: vertex.textureCoordinate)
        }
        let refitted = try original.refitted(to: posed)
        let rebuilt = try MeshBVH(geometry: posed)
        #expect(refitted.bounds == rebuilt.bounds)
        #expect(refitted.nodeCount == original.nodeCount)
        #expect(refitted.depth == original.depth)

        var random = BVHTestRandom(seed: 0x5EED)
        var hits = 0, misses = 0
        for _ in 0..<100 {
            let query = SIMD3(random.float(in: -7...7), random.float(in: -4...4), Float(2))
            let ray = try SceneRay(origin: query, direction: SIMD3(0, 0, -1), maximumDistance: 2)
            let hit = try refitted.intersection(ray: ray, cullsBackFaces: false)
            #expect(try rebuilt.intersection(ray: ray, cullsBackFaces: false) == hit)
            #expect(try refitted.intersections(ray: ray, cullsBackFaces: false)
                == rebuilt.intersections(ray: ray, cullsBackFaces: false))
            if hit == nil { misses += 1 } else { hits += 1 }
            #expect(try refitted.closestPoint(to: query, maximumDistance: 2.5)
                == rebuilt.closestPoint(to: query, maximumDistance: 2.5))
            let extent = SIMD3<Float>(0.4, 0.3, 2)
            let box = try AxisAlignedBox(minimum: query - extent, maximum: query + extent)
            #expect(refitted.triangles(intersecting: box) == rebuilt.triangles(intersecting: box))
        }
        #expect(hits > 25)
        #expect(misses > 25)

        let other = try MeshBVH(geometry: .plane(width: 2, height: 1, columns: 3, rows: 3))
        for z: Float in [0.25, 0.75, 4] {
            let placement = SceneTransform(translation: SIMD3(0.3, -0.25, z), scale: SIMD3(1.5, 0.4, 2)).matrix
            #expect(try refitted.closestPoints(to: other, otherLocalToWorld: placement)
                == rebuilt.closestPoints(to: other, otherLocalToWorld: placement))
        }
    }

    @Test func `An oversized degenerate leaf retains topology as its triangles spread`() throws {
        let surface = try MeshGeometry.plane(width: 4, height: 4, columns: 4, rows: 4)
        let collapsed = try remap(surface) { _, vertex in
            MeshVertex(position: .zero, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate)
        }
        let original = try MeshBVH(geometry: collapsed)
        let refitted = try original.refitted(to: surface)
        let rebuilt = try MeshBVH(geometry: surface)
        #expect(original.nodeCount == 1)
        #expect(refitted.nodeCount == 1)
        #expect(rebuilt.nodeCount > 1)
        #expect(refitted.bounds == rebuilt.bounds)
        #expect(try refitted.closestPoint(to: SIMD3(1.75, 1.75, 1))
            == rebuilt.closestPoint(to: SIMD3(1.75, 1.75, 1)))
        #expect(try original.closestPoint(to: SIMD3(1.75, 1.75, 1))?.position == .zero)
    }

    @Test func `Refit rejects vertex count and exact source index mismatches`() throws {
        let source = try MeshGeometry.plane(width: 4, height: 4)
        let original = try MeshBVH(geometry: source)
        let extra = try MeshGeometry(vertices: source.vertices + [source.vertices[0]], indices: source.indices)
        let short = try MeshGeometry(vertices: source.vertices, indices: Array(source.indices.prefix(3)))
        var swapped = source.indices
        swapped.swapAt(0, 1)
        let different = try MeshGeometry(vertices: source.vertices, indices: swapped)
        let reordered = try MeshGeometry(vertices: source.vertices, indices: Array(source.indices.suffix(3) + source.indices.prefix(3)))
        for mesh in [extra, short, different, reordered] {
            #expect(throws: MeshBVH.Failure.refitTopologyMismatch) {
                try original.refitted(to: mesh)
            }
        }
        #expect(try original.refitted(to: source).bounds == original.bounds)
    }

    @Test func `Readers keep an old snapshot while another task refits the same tree`() async throws {
        let source = try MeshGeometry.plane(width: 4, height: 4, columns: 8, rows: 8)
        let original = try MeshBVH(geometry: source)
        let moved = try remap(source) { _, vertex in
            MeshVertex(position: vertex.position + SIMD3(0, 0, 3), normal: vertex.normal)
        }
        let ray = try SceneRay(origin: SIMD3(0, 0, 4), direction: SIMD3(0, 0, -1))
        async let refitted = original.refitted(to: moved)
        let oldDistances = try await withThrowingTaskGroup(of: Float?.self) { group in
            for _ in 0..<16 { group.addTask { try original.intersection(ray: ray)?.distance } }
            var distances: [Float?] = []
            for try await value in group { distances.append(value) }
            return distances
        }
        #expect(oldDistances == Array(repeating: 4, count: 16))
        let newSnapshot = try await refitted
        #expect(try newSnapshot.intersection(ray: ray)?.distance == 1)
    }

    private func remap(_ source: MeshGeometry, _ vertex: (Int, MeshVertex) -> MeshVertex) throws -> MeshGeometry {
        try MeshGeometry(vertices: source.vertices.enumerated().map { vertex($0.offset, $0.element) }, indices: source.indices)
    }

    private func overlaps(_ a: MeshBVH, _ b: MeshBVH, placement: simd_float4x4) throws -> Set<SIMD2<Int>> {
        var pairs = Set<SIMD2<Int>>()
        _ = try a.forEachOverlappingTrianglePair(with: b, otherLocalToWorld: placement) { i, j in
            pairs.insert(SIMD2(i, j))
            return false
        }
        return pairs
    }
}
