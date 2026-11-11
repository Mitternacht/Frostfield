import Foundation
import Frostfield
import Testing
import simd

struct ConvexHullTests {
    /// Checks the hull is a closed convex polyhedron that contains every input point.
    private func verify(_ hull: ConvexHull) {
        let positions = hull.points.map(SIMD3<Double>.init)
        var directed: [SIMD2<Int>: Int] = [:]
        for triangle in hull.triangles {
            for (a, b) in [(triangle.x, triangle.y), (triangle.y, triangle.z), (triangle.z, triangle.x)] {
                directed[SIMD2(a, b), default: 0] += 1
            }
        }
        #expect(directed.allSatisfy { edge, count in count == 1 && directed[SIMD2(edge.y, edge.x)] == 1 })
        // Euler's formula for a sphere-like polyhedron: V − E + F = 2.
        #expect(hull.vertexIndices.count - directed.count / 2 + hull.triangles.count == 2)
        let vertices = hull.vertexIndices.map { positions[$0] }
        var maximumOutside = -Double.infinity
        for triangle in hull.triangles {
            let a = positions[triangle.x], b = positions[triangle.y], c = positions[triangle.z]
            let normal = simd_normalize(simd_cross(b - a, c - a))
            // Convex: every hull vertex lies on or behind every face.
            for vertex in vertices { maximumOutside = max(maximumOutside, simd_dot(normal, vertex - a)) }
            // Contains: every input point lies on or behind every face.
            for point in positions { maximumOutside = max(maximumOutside, simd_dot(normal, point - a)) }
        }
        #expect(maximumOutside <= 8 * hull.tolerance)
        #expect(hull.points.allSatisfy { hull.contains($0) })
    }

    @Test("Hulls of random points contain every point, are convex and closed", arguments: [11, 12, 13] as [UInt64])
    func randomPoints(seed: UInt64) throws {
        var random = ShapeTestRandom(seed: seed)
        let points = (0..<2_000).map { _ in
            SIMD3<Float>(Float.random(in: -1...1, using: &random), Float.random(in: -2...2, using: &random), Float.random(in: -0.5...0.5, using: &random))
        }
        let hull = try ConvexHull(points: points)
        verify(hull)
        // The hull of points in a box has less volume than the box but most of it.
        #expect(hull.volume < 8 && hull.volume > 7)
        let mesh = try hull.geometry()
        #expect(ShapeCheck.isClosed(mesh))
        #expect(abs(mesh.signedVolume - hull.volume) < 1e-5)
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
        // Flat shading: each triangle's three normals equal its winding normal.
        for triangle in 0..<(mesh.indices.count / 3) {
            let (a, b, c) = ShapeCheck.corners(mesh, triangle)
            let winding = SIMD3<Float>(simd_normalize(simd_cross(b - a, c - a)))
            for slot in 0..<3 {
                #expect(simd_distance(mesh.vertices[Int(mesh.indices[triangle * 3 + slot])].normal, winding) < 1e-4)
            }
        }
    }

    @Test("Every point on a sphere is a hull vertex and the volume approaches the ball's")
    func spherePoints() throws {
        var random = ShapeTestRandom(seed: 7)
        let points = (0..<400).map { _ -> SIMD3<Float> in
            var direction = SIMD3<Double>.zero
            repeat {
                direction = SIMD3(Double.random(in: -1...1, using: &random), Double.random(in: -1...1, using: &random), Double.random(in: -1...1, using: &random))
            } while simd_length(direction) < 0.1 || simd_length(direction) > 1
            return SIMD3<Float>(simd_normalize(direction))
        }
        let hull = try ConvexHull(points: points)
        verify(hull)
        #expect(hull.vertexIndices.count == 400)
        // Euler's formula for a triangulated sphere: F = 2V − 4.
        #expect(hull.triangles.count == 2 * 400 - 4)
        #expect(hull.volume < 4 / 3 * .pi && hull.volume > 0.9 * 4 / 3 * .pi)
    }

    @Test("Coplanar, duplicate and interior points add no vertices: a gridded cube has eight")
    func degeneratePoints() throws {
        var points: [SIMD3<Float>] = []
        // A 5 × 5 × 5 lattice: every face holds 25 coplanar points, every edge 5 collinear ones.
        for x in 0..<5 { for y in 0..<5 { for z in 0..<5 {
            points.append(SIMD3(Float(x), Float(y), Float(z)) * 0.25 - 0.5)
        } } }
        points += points.prefix(30)
        let hull = try ConvexHull(points: points)
        verify(hull)
        #expect(hull.vertexIndices.count == 8)
        #expect(hull.triangles.count == 12)
        #expect(hull.volume == 1)
        #expect(hull.contains(SIMD3(0.5, 0.5, 0.5)))
        #expect(!hull.contains(SIMD3(0.5, 0.5, 0.5001)))
        #expect(abs(hull.signedDistance(to: SIMD3(0, 0, 0)) + 0.5) < 1e-12)
    }

    @Test("A shallow point above a face is retained")
    func shallowCorner() throws {
        let points: [SIMD3<Float>] = [
            SIMD3(-0.5, -0.5, 0), SIMD3(0.5, -0.5, 0),
            SIMD3(0.5, 0.5, 0), SIMD3(-0.5, 0.5, 0),
            SIMD3(0, 0, -1), SIMD3(0, 0, 1e-5),
        ]
        let hull = try ConvexHull(points: points)
        verify(hull)
        #expect(hull.vertexIndices == Array(points.indices))
        #expect(abs(hull.volume - (1 + Double(points[5].z)) / 3) < 1e-12)
    }

    @Test("A thin square frustum builds a closed hull with nonzero volume")
    func thinFrustum() throws {
        let height: Float = 1e-5
        let points: [SIMD3<Float>] = [
            SIMD3(-1, -1, 0), SIMD3(1, -1, 0), SIMD3(1, 1, 0), SIMD3(-1, 1, 0),
            SIMD3(-0.5, -0.5, height), SIMD3(0.5, -0.5, height),
            SIMD3(0.5, 0.5, height), SIMD3(-0.5, 0.5, height),
        ]
        let hull = try ConvexHull(points: points)
        verify(hull)
        #expect(hull.vertexIndices == Array(points.indices))
        #expect(abs(hull.volume - 7 * Double(height) / 3) < 1e-12)
        #expect(try ShapeCheck.isClosed(hull.geometry()))
    }

    @Test("Points without volume, too few points and non-finite points are rejected")
    func validation() {
        let square: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(0, 1, 0), SIMD3(0.5, 0.5, 0)]
        #expect(throws: ConvexHull.Failure.degenerate) { try ConvexHull(points: square) }
        let line = (0..<6).map { SIMD3<Float>(Float($0), Float($0) * 2, 0) }
        #expect(throws: ConvexHull.Failure.degenerate) { try ConvexHull(points: line) }
        #expect(throws: ConvexHull.Failure.tooFewPoints) { try ConvexHull(points: Array(square.prefix(3))) }
        #expect(throws: ConvexHull.Failure.nonFinitePoint(1)) {
            try ConvexHull(points: [.zero, SIMD3(.nan, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)])
        }
    }
}
