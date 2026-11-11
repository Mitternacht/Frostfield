import Frostfield
import Foundation
import Testing
import simd

struct PolygonTriangulatorTests {
    private func area(_ triangles: [SIMD3<Int>], _ points: [SIMD2<Double>]) -> (total: Double, smallest: Double) {
        var total = 0.0, smallest = Double.infinity
        for t in triangles {
            let a = points[t.x], b = points[t.y], c = points[t.z]
            let doubled = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
            total += doubled / 2
            smallest = min(smallest, doubled / 2)
        }
        return (total, smallest)
    }

    @Test("A square with a square hole covers exactly its area with V + 2H − 2 counterclockwise triangles")
    func squareWithHole() throws {
        let outer: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(4, 0), SIMD2(4, 4), SIMD2(0, 4)]
        // The hole winds the same way as the outer ring; the triangulator orients each ring itself.
        let hole: [SIMD2<Double>] = [SIMD2(1, 1), SIMD2(3, 1), SIMD2(3, 3), SIMD2(1, 3)]
        let triangles = try PolygonTriangulator.triangulate(outer: outer, holes: [hole])
        let (total, smallest) = area(triangles, outer + hole)
        #expect(triangles.count == 8)
        #expect(total == 12)
        #expect(smallest > 0)
        // No triangle covers the hole's center.
        for t in triangles {
            let points = [t.x, t.y, t.z].map { (outer + hole)[$0] }
            let center = SIMD2<Double>(2, 2)
            let inside = (0..<3).allSatisfy { i in
                let a = points[i], b = points[(i + 1) % 3]
                return (b.x - a.x) * (center.y - a.y) - (b.y - a.y) * (center.x - a.x) > 0
            }
            #expect(!inside)
        }
    }

    @Test("Random star polygons with many holes triangulate to their exact shoelace area", arguments: [1, 2, 3, 4, 5] as [UInt64])
    func randomStars(seed: UInt64) throws {
        var random = ShapeTestRandom(seed: seed)
        let count = 40 + Int(seed) * 17
        // A star-shaped outer ring: angles increase, radii vary, so the ring is simple but deeply concave.
        let outer: [SIMD2<Double>] = (0..<count).map { index in
            let angle = 2 * Double.pi * Double(index) / Double(count)
            let radius = Double.random(in: 4...10, using: &random)
            return SIMD2(cos(angle), sin(angle)) * radius
        }
        // Small square holes on a grid well inside the inner radius of 4.
        var holes: [[SIMD2<Double>]] = []
        for x in stride(from: -2.25, through: 1.75, by: 1.0) {
            for y in stride(from: -2.25, through: 1.75, by: 1.0) {
                let size = Double.random(in: 0.1...0.4, using: &random)
                holes.append([SIMD2(x, y), SIMD2(x, y + size), SIMD2(x + size, y + size), SIMD2(x + size, y)])
            }
        }
        let points = outer + holes.flatMap { $0 }
        let triangles = try PolygonTriangulator.triangulate(outer: outer, holes: holes)
        let expected = PolygonTriangulator.doubledArea(outer) / 2 - holes.reduce(0) { $0 + abs(PolygonTriangulator.doubledArea($1)) / 2 }
        let (total, smallest) = area(triangles, points)
        #expect(abs(total - expected) <= 1e-9 * expected)
        #expect(smallest > 0)
        #expect(triangles.count == points.count + 2 * holes.count - 2)
        // Every boundary vertex is used.
        #expect(Set(triangles.flatMap { [$0.x, $0.y, $0.z] }).count == points.count)
    }

    @Test("Collinear boundary points stay in the triangulation so neighbours sharing them remain conforming")
    func collinearPointsKept() throws {
        // A square with three extra points on its bottom edge and two on its right edge.
        let outer: [SIMD2<Double>] = [
            SIMD2(0, 0), SIMD2(0.25, 0), SIMD2(0.5, 0), SIMD2(0.75, 0), SIMD2(1, 0),
            SIMD2(1, 1.0 / 3), SIMD2(1, 2.0 / 3), SIMD2(1, 1), SIMD2(0, 1),
        ]
        let triangles = try PolygonTriangulator.triangulate(outer: outer)
        #expect(Set(triangles.flatMap { [$0.x, $0.y, $0.z] }) == Set(outer.indices))
        let (total, smallest) = area(triangles, outer)
        #expect(abs(total - 1) < 1e-15)
        #expect(smallest > 1e-3)
        #expect(triangles.count == outer.count - 2)
        // Each boundary segment between consecutive points is an edge of exactly one triangle.
        var edges: [SIMD2<Int>: Int] = [:]
        for t in triangles {
            for (a, b) in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] { edges[SIMD2(a, b), default: 0] += 1 }
        }
        for index in outer.indices {
            #expect(edges[SIMD2(index, (index + 1) % outer.count)] == 1)
        }
    }

    @Test("Crossing rings fail instead of returning overlapping triangles")
    func rejectsCrossingRings() {
        let bowTie: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(2, 2), SIMD2(2, 0), SIMD2(0, 2)]
        #expect(throws: PolygonTriangulator.Failure.self) {
            try PolygonTriangulator.triangulate(outer: bowTie)
        }
        let outer: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(2, 0), SIMD2(2, 2), SIMD2(0, 2)]
        let escapingHole: [SIMD2<Double>] = [SIMD2(1, 1), SIMD2(3, 1), SIMD2(3, 1.5), SIMD2(1, 1.5)]
        #expect(throws: PolygonTriangulator.Failure.self) {
            try PolygonTriangulator.triangulate(outer: outer, holes: [escapingHole])
        }
        #expect(throws: PolygonTriangulator.Failure.ringTooSmall(0)) {
            try PolygonTriangulator.triangulate(outer: [SIMD2(0, 0), SIMD2(1, 0)])
        }
    }
}
