import Frostfield
import Testing
import simd

struct MeshBVHClosestPointTests {
    @Test func `An interior point returns source corner weights and an inclusive world bound`() throws {
        let bvh = try hierarchy([[.zero, SIMD3(4, 0, 0), SIMD3(0, 4, 0)]])
        let query = SIMD3<Float>(1, 1, 3)
        let hit = try #require(try bvh.closestPoint(to: query, maximumDistance: 3))
        #expect(hit.position == SIMD3(1, 1, 0))
        #expect(hit.barycentric == SIMD3(0.5, 0.25, 0.25))
        #expect(hit.distance == 3)
        #expect(hit.triangleIndex == 0)
        #expect(try bvh.closestPoint(to: query, maximumDistance: Double(3).nextDown) == nil)

        let onSurface = try #require(try bvh.closestPoint(to: hit.position, maximumDistance: 0))
        #expect(onSurface.distance == 0)
        #expect(onSurface.position == hit.position)

        let edge = try #require(try bvh.closestPoint(to: SIMD3(3, 3, 1)))
        #expect(edge.position == SIMD3(2, 2, 0))
        #expect(edge.barycentric == SIMD3(0, 0.5, 0.5))
        #expect(edge.distance == 3.0.squareRoot())
        #expect(try bvh.closestPoint(to: SIMD3(3, 3, 1), maximumDistance: edge.distance) == edge)
    }

    @Test func `A mirrored shear uses the world metric to select an interior edge witness`() throws {
        let bvh = try hierarchy([[.zero, SIMD3(2, 0, 0), SIMD3(0, 2, 0)]])
        let shear = simd_float4x4(
            SIMD4(-2, 0, 0, 0), SIMD4(1, 1, 0, 0), SIMD4(0, 0, 0.5, 0), SIMD4(5, 3, 7, 1)
        )
        // The CA edge runs from (7,5,7) to (5,3,7).
        let hit = try #require(try bvh.closestPoint(to: SIMD3(7, 3, 9), localToWorld: shear))
        #expect(hit.position == SIMD3(6, 4, 7))
        #expect(hit.distance == 6.0.squareRoot())
        #expect(hit.barycentric == SIMD3(0.5, 0, 0.5))
    }

    @Test func `Degenerate triangles and axis collapse retain a convex barycentric witness`() throws {
        let bvh = try hierarchy([
            [SIMD3(-3, 0, 0), SIMD3(-1, 0, 0), SIMD3(-2, 0, 0)],
            [SIMD3(2, 0, 0), SIMD3(2, 0, 0), SIMD3(2, 0, 0)],
            [SIMD3(6, 0, 0), SIMD3(8, 0, 0), SIMD3(6, 2, 0)],
        ])
        let line = try #require(try bvh.closestPoint(to: SIMD3(-2.5, 1, 0)))
        #expect(line.triangleIndex == 0)
        #expect(line.position == SIMD3(-2.5, 0, 0))
        #expect(line.barycentric == SIMD3(0.75, 0.25, 0))
        let point = try #require(try bvh.closestPoint(to: SIMD3(2, 0, 1)))
        #expect(point.triangleIndex == 1)
        #expect(point.position == SIMD3(2, 0, 0))
        #expect(point.barycentric == SIMD3(1, 0, 0))

        let collapsed = SceneTransform(scale: SIMD3(1, 0, 1)).matrix
        let flattened = try #require(try bvh.closestPoint(to: SIMD3(7, 3, 0), localToWorld: collapsed))
        #expect(flattened.triangleIndex == 2)
        #expect(flattened.position == SIMD3(7, 0, 0))
        #expect(flattened.distance == 3)
        #expect(flattened.barycentric == SIMD3(0.5, 0.5, 0))
    }

    @Test func `Equal-distance leaves choose the smallest source index after visiting the other side`() throws {
        let right = Array(repeating: SIMD3<Float>(1, 0, 0), count: 3)
        let left = Array(repeating: SIMD3<Float>(-1, 0, 0), count: 3)
        let bvh = try hierarchy([right, left, right, left, right, left])
        #expect(bvh.depth > 1)
        let hit = try #require(try bvh.closestPoint(to: .zero, maximumDistance: 1))
        #expect(hit.triangleIndex == 0)
        #expect(hit.position == SIMD3(1, 0, 0))
    }

    @Test func `Far points and large affine scales preserve representable witnesses`() throws {
        let bvh = try hierarchy([[SIMD3(0, 0, 1), SIMD3(1, 0, 1), SIMD3(0, 1, 1)]])
        let far = try #require(try bvh.closestPoint(to: SIMD3(0.25, 0.25, 1e30)))
        #expect(far.position == SIMD3(0.25, 0.25, 1))
        #expect(far.barycentric == SIMD3(0.5, 0.25, 0.25))

        let large = Float(1e37)
        let scaled = SceneTransform(scale: SIMD3(large, large, 1)).matrix
        let hit = try #require(try bvh.closestPoint(to: SIMD3(large * 0.25, large * 0.25, 2), localToWorld: scaled))
        #expect(hit.position == SIMD3(large * 0.25, large * 0.25, 1))
        #expect(abs(hit.barycentric.x - 0.5) < 1e-14)
        #expect(hit.distance == 1)
    }

    @Test func `Nonfinite inputs and unrepresentable results fail explicitly`() throws {
        let bvh = try hierarchy([[SIMD3(2, 0, 0), SIMD3(2, 1, 0), SIMD3(2, 0, 1)]])
        #expect(throws: MeshBVH.Failure.nonfiniteProximityPoint) {
            try bvh.closestPoint(to: SIMD3(.nan, 0, 0))
        }
        for limit: Double in [-1, -.infinity, .nan] {
            #expect(throws: MeshBVH.Failure.invalidProximityDistance) {
                try bvh.closestPoint(to: .zero, maximumDistance: limit)
            }
        }
        var projective = matrix_identity_float4x4
        projective[1].w = 1
        #expect(throws: MeshBVH.Failure.invalidProximityTransform) {
            try bvh.closestPoint(to: .zero, localToWorld: projective)
        }
        let overflow = SceneTransform(scale: SIMD3(.greatestFiniteMagnitude, 1, 1)).matrix
        #expect(throws: MeshBVH.Failure.unrepresentableProximityPoint) {
            try bvh.closestPoint(to: .zero, localToWorld: overflow)
        }
        #expect(try bvh.closestPoint(to: .zero, localToWorld: overflow, maximumDistance: 100) == nil)
    }

    @Test func `Pruned affine traversal matches a brute-force independent reference`() throws {
        var random = BVHTestRandom(seed: 0x3E1A)
        var triangles: [[SIMD3<Float>]] = []
        for index in 0..<180 {
            let base = random.vector(in: -8...8)
            let a = random.vector(in: -2...2), b = random.vector(in: -2...2)
            if index % 7 == 0 {
                triangles.append([base, base, base])
            } else if index % 5 == 0 {
                triangles.append([base, base + a, base])
            } else {
                triangles.append([base, base + a, base + b])
            }
        }
        let bvh = try hierarchy(triangles)
        #expect(bvh.depth > 3)
        let transforms = [
            matrix_identity_float4x4,
            simd_float4x4(SIMD4(-2.4, 0.25, 0, 0), SIMD4(1, 0.35, 0.25, 0),
                          SIMD4(0, 0.3, 3, 0), SIMD4(1, -4, 2, 1)),
            SceneTransform(scale: SIMD3(0.01, 5, 0.25)).matrix,
        ]
        var hits = 0
        var misses = 0
        for transform in transforms {
            let matrix = simd_double4x4(
                SIMD4<Double>(transform[0]), SIMD4<Double>(transform[1]),
                SIMD4<Double>(transform[2]), SIMD4<Double>(transform[3])
            )
            let moved = triangles.map { triangle in
                triangle.map { point in
                    let p = matrix * SIMD4(SIMD3<Double>(point), 1)
                    return SIMD3(p.x, p.y, p.z)
                }
            }
            for queryIndex in 0..<80 {
                let query = random.vector(in: -24...24)
                let world = SIMD3<Double>(query)
                let limit = queryIndex.isMultiple(of: 2) ? Double(random.float(in: 0...20)) : .infinity
                let reference = moved.enumerated().map { index, triangle in
                    (index, referencePoint(world, triangle: triangle))
                }.min { simd_length_squared($0.1 - world) < simd_length_squared($1.1 - world) }!
                let distance = simd_length(reference.1 - world)
                let hit = try bvh.closestPoint(to: query, localToWorld: transform, maximumDistance: limit)
                if distance <= limit {
                    let hit = try #require(hit)
                    #expect(hit.triangleIndex == reference.0)
                    #expect(abs(hit.distance - distance) < 1e-11)
                    #expect(simd_length(SIMD3<Double>(hit.position) - reference.1) < 3e-6)
                    #expect(all(hit.barycentric .>= .zero))
                    #expect(abs(hit.barycentric.sum() - 1) < 1e-14)
                    let corners = moved[hit.triangleIndex]
                    let reconstructed = corners[0] * hit.barycentric.x + corners[1] * hit.barycentric.y
                        + corners[2] * hit.barycentric.z
                    #expect(simd_length(reconstructed - reference.1) < 1e-10)
                    hits += 1
                } else {
                    #expect(hit == nil)
                    misses += 1
                }
            }
        }
        #expect(hits > 140)
        #expect(misses > 10)
    }

    private func hierarchy(_ triangles: [[SIMD3<Float>]]) throws -> MeshBVH {
        let vertices = triangles.flatMap { $0 }.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) }
        return try MeshBVH(geometry: MeshGeometry(vertices: vertices, indices: vertices.indices.map(UInt32.init)))
    }

    // Independent barycentric reference. Project by the unnormalized plane,
    // then test the Gram-coordinate triangle and the three closed boundaries.
    private func referencePoint(_ query: SIMD3<Double>, triangle: [SIMD3<Double>]) -> SIMD3<Double> {
        let a = triangle[0], b = triangle[1], c = triangle[2]
        let e = b - a, f = c - a
        let normal = simd_cross(e, f)
        let areaSquared = simd_length_squared(normal)
        if areaSquared > 0 {
            let projected = query - normal * (simd_dot(query - a, normal) / areaSquared)
            let v = projected - a
            let ef = simd_dot(e, f), ee = simd_dot(e, e), ff = simd_dot(f, f)
            let x = (ff * simd_dot(v, e) - ef * simd_dot(v, f)) / areaSquared
            let y = (ee * simd_dot(v, f) - ef * simd_dot(v, e)) / areaSquared
            if x >= 0, y >= 0, x + y <= 1 { return projected }
        }
        return [(a, b), (b, c), (c, a)].map { start, end in
            let edge = end - start
            let squared = simd_length_squared(edge)
            guard squared > 0 else { return start }
            let parameter = simd_dot(query - start, edge) / squared
            if parameter < 0 { return start }
            if parameter > 1 { return end }
            return start + edge * parameter
        }.min { simd_length_squared($0 - query) < simd_length_squared($1 - query) }!
    }
}
