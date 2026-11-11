import Frostfield
import Testing
import simd

struct MeshBVHPairTests {
    @Test func `A triangle edge can cross the interior of the other face at zero distance`() throws {
        let first = try hierarchy([[.zero, SIMD3(6, 0, 0), SIMD3(0, 6, 0)]])
        let second = try hierarchy([[SIMD3(1, 1, -2), SIMD3(1, 1, 2), SIMD3(1, 3, 2)]])
        let pair = try #require(try first.closestPoints(to: second, maximumDistance: 0))
        #expect(pair.first.position == SIMD3(1, 1, 0))
        #expect(pair.second.position == pair.first.position)
        #expect(pair.distance == 0)
        validate(pair.first, bvh: first)
        validate(pair.second, bvh: second)
    }

    @Test func `Coplanar overlapping triangles find a shared witness without vertex containment`() throws {
        let first = try hierarchy([[SIMD3(-2, 0, 0), SIMD3(2, 0, 0), SIMD3(0, 3, 0)]])
        let second = try hierarchy([[SIMD3(-2, 2, 0), SIMD3(2, 2, 0), SIMD3(0, -1, 0)]])
        let pair = try #require(try first.closestPoints(to: second, maximumDistance: 0))
        #expect(pair.first.position == pair.second.position)
        #expect(pair.distance == 0)
        validate(pair.first, bvh: first)
        validate(pair.second, bvh: second)
    }

    @Test func `Separated segment triangles find the two interior edge witnesses`() throws {
        let first = try hierarchy([[SIMD3(-2, 0, 0), SIMD3(2, 0, 0), SIMD3(-2, 0, 0)]])
        let second = try hierarchy([[SIMD3(0, -3, 0), SIMD3(0, 3, 0), SIMD3(0, -3, 0)]])
        let firstTransform = SceneTransform(translation: SIMD3(2, 0, 4), scale: SIMD3(-2, 1, 1)).matrix
        let secondTransform = simd_float4x4(
            SIMD4(1, 0, 0, 0), SIMD4(1, 2, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(2, 0, 5, 1)
        )
        let pair = try #require(try first.closestPoints(
            to: second, localToWorld: firstTransform, otherLocalToWorld: secondTransform, maximumDistance: 1
        ))
        #expect(pair.first.position == SIMD3(2, 0, 4))
        #expect(pair.second.position == SIMD3(2, 0, 5))
        #expect(pair.distance == 1)
        validate(pair.first, bvh: first, transform: firstTransform)
        validate(pair.second, bvh: second, transform: secondTransform)
        #expect(try first.closestPoints(
            to: second, localToWorld: firstTransform, otherLocalToWorld: secondTransform,
            maximumDistance: Double(1).nextDown
        ) == nil)
    }

    @Test func `A collinear contact survives rounded interior segment parameters`() throws {
        let first = try hierarchy([[.zero, SIMD3(7, 0, 0), .zero]])
        let endpoint = SIMD3<Float>(37.0 / 41, 0, 0)
        let second = try hierarchy([[endpoint, SIMD3(1.2, 0, 0), endpoint]])
        let nearest = try #require(try first.closestPoint(to: endpoint, maximumDistance: 0))
        #expect(nearest.position == endpoint)
        #expect(nearest.distance == 0)
        let pair = try #require(try first.closestPoints(to: second, maximumDistance: 0))
        #expect(pair.first.position == pair.second.position)
        #expect(pair.distance == 0)
        validate(pair.first, bvh: first)
        validate(pair.second, bvh: second)
    }

    @Test func `Dual traversal agrees with exhaustive transformed triangle pairs`() throws {
        let first = try splitPlane()
        let second = try splitPlane()
        let transforms = [
            SceneTransform(translation: SIMD3(0.25, -0.5, 2)).matrix,
            SceneTransform(translation: SIMD3(4, 3, 1), rotation: simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0)),
                           scale: SIMD3(-2, 0.75, 1)).matrix,
            SceneTransform(translation: SIMD3(6, 2, 2), rotation: simd_quatf(angle: 0.6, axis: SIMD3(1, 0, 0)),
                           scale: SIMD3(1, 2, 0.4)).matrix,
        ]
        #expect(first.bvh.depth > 1)
        #expect(second.bvh.depth > 1)
        for placement in transforms {
            var reference: (first: Int, second: Int, distance: Double)?
            for (i, a) in first.faces.enumerated() {
                for (j, b) in second.faces.enumerated() {
                    let pair = try #require(try a.closestPoints(to: b, otherLocalToWorld: placement))
                    if pair.distance < (reference?.distance ?? .infinity) { reference = (i, j, pair.distance) }
                }
            }
            let expected = try #require(reference)
            let actual = try #require(try first.bvh.closestPoints(to: second.bvh, otherLocalToWorld: placement))
            #expect(actual.first.triangleIndex == expected.first)
            #expect(actual.second.triangleIndex == expected.second)
            #expect(actual.distance == expected.distance)
            validate(actual.first, bvh: first.bvh)
            validate(actual.second, bvh: second.bvh, transform: placement)
        }
    }

    @Test func `Pair cast prunes disjoint bounds and accepts or stops at candidate callbacks`() throws {
        let first = try splitPlane()
        let second = try hierarchy([
            [SIMD3(-0.75, -0.6, 0), SIMD3(-0.55, -0.6, 0), SIMD3(-0.75, -0.5, 0)],
            [SIMD3(20, 20, 0), SIMD3(21, 20, 0), SIMD3(20, 21, 0)],
        ])
        var pairs = Set<SIMD2<Int>>()
        let stopped = try first.bvh.forEachOverlappingTrianglePair(with: second) { a, b in
            #expect(pairs.insert(SIMD2(a, b)).inserted)
            return false
        }
        #expect(!stopped)
        #expect(!pairs.isEmpty)
        #expect(pairs.allSatisfy { $0.y == 0 })
        var count = 0
        #expect(try first.bvh.forEachOverlappingTrianglePair(with: second) { _, _ in
            count += 1
            return true
        })
        #expect(count == 1)
        let far = SceneTransform(translation: SIMD3(0, 0, 0.01)).matrix
        #expect(try !first.bvh.forEachOverlappingTrianglePair(with: second, otherLocalToWorld: far) { _, _ in
            Issue.record("Disjoint world bounds must skip the callback")
            return true
        })

        enum CallbackError: Error { case stop }
        #expect(throws: CallbackError.stop) {
            try first.bvh.forEachOverlappingTrianglePair(with: second) { _, _ in throw CallbackError.stop }
        }
    }

    @Test func `Pair casts expose bounds candidates while closest pairs check actual surfaces`() throws {
        let a = try hierarchy([[.zero, SIMD3(2, 0, 0), SIMD3(0, 2, 0)]])
        let b = try hierarchy([[SIMD3(1.5, 1.5, 0), SIMD3(2, 1.5, 0), SIMD3(1.5, 2, 0)]])
        #expect(try a.forEachOverlappingTrianglePair(with: b) { $0 == 0 && $1 == 0 })
        #expect(try a.closestPoints(to: b, maximumDistance: 0) == nil)
        let hit = try #require(try a.closestPoints(to: b))
        #expect(hit.first.position == SIMD3(1, 1, 0))
        #expect(hit.second.position == SIMD3(1.5, 1.5, 0))
        #expect(hit.distance == 0.5.squareRoot())
    }

    private func hierarchy(_ triangles: [[SIMD3<Float>]]) throws -> MeshBVH {
        let vertices = triangles.flatMap { $0 }.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) }
        return try MeshBVH(geometry: MeshGeometry(vertices: vertices, indices: vertices.indices.map(UInt32.init)))
    }

    private func splitPlane() throws -> (bvh: MeshBVH, faces: [MeshBVH]) {
        let bvh = try MeshBVH(geometry: .plane(width: 4, height: 3, columns: 3, rows: 3))
        let geometry = bvh.geometry
        let faces = try (0..<bvh.triangleCount).map { triangle in
            let corners = geometry.indices[(triangle * 3)..<(triangle * 3 + 3)].map { geometry.vertices[Int($0)].position }
            return try hierarchy([corners])
        }
        return (bvh, faces)
    }

    private func validate(
        _ witness: MeshClosestPoint, bvh: MeshBVH, transform: simd_float4x4 = matrix_identity_float4x4
    ) {
        let geometry = bvh.geometry
        let offset = witness.triangleIndex * 3
        let matrix = simd_double4x4(
            SIMD4<Double>(transform[0]), SIMD4<Double>(transform[1]), SIMD4<Double>(transform[2]), SIMD4<Double>(transform[3])
        )
        var reconstructed = SIMD4<Double>.zero
        for corner in 0..<3 {
            let local = geometry.vertices[Int(geometry.indices[offset + corner])].position
            reconstructed += (matrix * SIMD4(SIMD3<Double>(local), 1)) * witness.barycentric[corner]
        }
        #expect(abs(witness.barycentric.sum() - 1) < 1e-13)
        #expect(all(witness.barycentric .>= .zero))
        let world = SIMD3(reconstructed.x, reconstructed.y, reconstructed.z)
        #expect(simd_length(world - SIMD3<Double>(witness.position)) < 5e-7)
    }
}
