import Frostfield
import Testing

struct PlanarTriangulationTests {
    private static let square: [SIMD2<Float>] = [.zero, SIMD2(4, 0), SIMD2(4, 4), SIMD2(0, 4)]
    private static let squareHole: [SIMD2<Float>] = [SIMD2(1, 1), SIMD2(3, 1), SIMD2(3, 3), SIMD2(1, 3)]
    private static let concave: [SIMD2<Float>] = [
        .zero, SIMD2(5, 0), SIMD2(5, 2), SIMD2(2, 2), SIMD2(2, 5), SIMD2(0, 5),
    ]

    @Test("Clockwise convex and concave contours match the r187 triangle counts and filled areas")
    func referenceContours() throws {
        let clockwise: [SIMD2<Float>] = [.zero, SIMD2(0, 3), SIMD2(4, 3), SIMD2(4, 0), .zero]
        let convex = try PlanarShape.triangulate(outer: clockwise)
        #expect(convex.count == 2)
        #expect(try checkTriangles(convex, points: clockwise, boundary: [[0, 1, 2, 3]]) == 12)

        let closedConcave = Self.concave + [.zero]
        let concave = try PlanarShape.triangulate(outer: closedConcave)
        #expect(concave.count == 4)
        #expect(try checkTriangles(concave, points: closedConcave, boundary: [[0, 1, 2, 3, 4, 5]]) == 16)
        #expect(!covers(SIMD2(3, 3), triangles: concave, points: closedConcave))
    }

    @Test("Closed and consecutive duplicates preserve the raw offsets of both holes")
    func originalHoleOffsets() throws {
        var outer = Self.concave + [.zero, .zero]
        outer.insert(outer[1], at: 2)
        let a = SIMD2<Float>(0.5, 0.5), b = SIMD2<Float>(0.5, 3)
        let first = [a, SIMD2(1.5, 0.5), SIMD2(1.5, 1.5), SIMD2(0.5, 1.5), a]
        let second = [b, b, SIMD2(1.5, 3), SIMD2(1.5, 4), SIMD2(0.5, 4), b]
        let points = outer + first + second
        let triangles = try PlanarShape.triangulate(outer: outer, holes: [first, second])
        let boundary = [[0, 1, 3, 4, 5, 6], [9, 10, 11, 12], [14, 16, 17, 18]]
        #expect(triangles.count == 16)
        #expect(try checkTriangles(triangles, points: points, boundary: boundary) == 14)
        #expect(!covers(SIMD2(1, 1), triangles: triangles, points: points))
        #expect(!covers(SIMD2(1, 3.5), triangles: triangles, points: points))
        #expect(covers(SIMD2(0.25, 0.25), triangles: triangles, points: points))
    }

    @Test("A normalized PlanarShape uses the same indices as its flat mesh")
    func normalizedShapeAndMesh() throws {
        let hole: [SIMD2<Float>] = [SIMD2(0.5, 0.5), SIMD2(1.5, 0.5), SIMD2(1.5, 1.5), SIMD2(0.5, 1.5)]
        let shape = try PlanarShape(outer: Array(Self.concave.reversed()), holes: [hole + [hole[0]]])
        let triangles = try shape.triangulatedIndices()
        let points = shape.outer + shape.holes.flatMap { $0 }
        #expect(try checkTriangles(triangles, points: points, boundary: [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]) == shape.area)
        let mesh = try MeshGeometry.shape([shape])
        #expect(mesh.vertices.map { SIMD2($0.position.x, $0.position.y) } == points)
        #expect(mesh.indices == triangles.flatMap { [UInt32($0.x), UInt32($0.y), UInt32($0.z)] })
    }

    @Test("The native triangulation keeps collinear boundary vertices")
    func collinearBoundary() throws {
        let points: [SIMD2<Float>] = [.zero, SIMD2(1, 0), SIMD2(2, 0), SIMD2(2, 1), SIMD2(0, 1)]
        let triangles = try PlanarShape.triangulate(outer: points)
        #expect(triangles.count == 3)
        #expect(try checkTriangles(triangles, points: points, boundary: [[0, 1, 2, 3, 4]]) == 2)
    }

    @Test("A hole touching the outer ring at one vertex retains its unfilled area")
    func sharedVertex() throws {
        let outer: [SIMD2<Float>] = [.zero, SIMD2(4, 0), SIMD2(4, 4), SIMD2(0, 4)]
        let hole: [SIMD2<Float>] = [.zero, SIMD2(1, 0.5), SIMD2(0.5, 1)]
        let triangles = try PlanarShape.triangulate(outer: outer, holes: [hole])
        let points = outer + hole
        #expect(triangles.count == 5)
        #expect(try checkTriangles(triangles, points: points) == 15.625)
        #expect(!covers(SIMD2(0.5, 0.5), triangles: triangles, points: points))
    }

    @Test("A filled Steiner point refines one triangle and duplicate coordinates keep the first source")
    func interiorSteiner() throws {
        let point = SIMD2<Float>(1, 2)
        let holes = [[.zero], [point], [point]]
        let triangles = try PlanarShape.triangulate(outer: Self.square, holes: holes)
        #expect(triangles.count == 4)
        #expect(try checkTriangles(
            triangles, points: Self.square + holes.flatMap { $0 }, boundary: [[0, 1, 2, 3]], interior: [5]
        ) == 16)
    }

    @Test("Steiner points exactly on a shared diagonal split both incident faces")
    func sharedDiagonal() throws {
        let baseline = try PlanarShape.triangulate(outer: Self.square)
        try #require(baseline.count == 2)
        let shared = Set(corners(baseline[0])).intersection(corners(baseline[1])).sorted()
        try #require(shared.count == 2)
        let a = Self.square[shared[0]], b = Self.square[shared[1]]
        let inserted = [(a + b) / 2, (3 * a + b) / 4, (a + 3 * b) / 4]
        let triangles = try PlanarShape.triangulate(outer: Self.square, holes: inserted.map { [$0] })
        #expect(triangles.count == 8)
        #expect(try checkTriangles(
            triangles, points: Self.square + inserted, boundary: [[0, 1, 2, 3]], interior: [4, 5, 6]
        ) == 16)
        for (start, end) in zip([shared[0], 5, 4, 6], [5, 4, 6, shared[1]]) {
            #expect(triangles.filter { corners($0).contains(start) && corners($0).contains(end) }.count == 2)
        }
    }

    @Test("A Steiner point on a contour edge splits that boundary without changing area")
    func outerBoundarySteiner() throws {
        let point = SIMD2<Float>(0, 2)
        let triangles = try PlanarShape.triangulate(outer: Self.square, holes: [[point]])
        #expect(triangles.count == 3)
        #expect(try checkTriangles(triangles, points: Self.square + [point], boundary: [[0, 1, 2, 3, 4]]) == 16)
    }

    @Test("Mixed hole rings preserve original offsets, area and a split hole edge")
    func mixedSteinerAndHole() throws {
        let outer = Self.square + [.zero]
        let hole = Self.squareHole + [Self.squareHole[0]]
        let interior = SIMD2<Float>(0.5, 2)
        let holes = [[interior], hole, [SIMD2(3, 2)], [SIMD2(3.5, 2)], [interior], [SIMD2(2, 2)], [SIMD2(5, 2)], [SIMD2(1, 1)]]
        let points = outer + holes.flatMap { $0 }
        let triangles = try PlanarShape.triangulate(outer: outer, holes: holes)
        #expect(triangles.count == 13)
        #expect(try checkTriangles(
            triangles, points: points, boundary: [[0, 1, 2, 3], [6, 7, 11, 8, 9]], interior: [5, 12]
        ) == 12)
        #expect(!covers(SIMD2(2, 2), triangles: triangles, points: points))
        #expect(!covers(SIMD2(5, 2), triangles: triangles, points: points))
    }

    @Test("Steiner points beyond the contour or inside a real hole leave the triangulation intact")
    func outsideSteiner() throws {
        let baseline = try PlanarShape.triangulate(outer: Self.square, holes: [Self.squareHole])
        let holes = [[SIMD2<Float>(-1, 2)], Self.squareHole, [SIMD2(5, 2)], [SIMD2(2, -1)], [SIMD2(2, 5)], [SIMD2(2, 2)]]
        let result = try PlanarShape.triangulate(outer: Self.square, holes: holes)
        // The exterior singleton precedes the real hole and shifts its source
        // offsets. The triangles themselves and their order stay the same.
        let remapped = baseline.map { triangle in
            SIMD3(triangle.x >= 4 ? triangle.x + 1 : triangle.x,
                  triangle.y >= 4 ? triangle.y + 1 : triangle.y,
                  triangle.z >= 4 ? triangle.z + 1 : triangle.z)
        }
        #expect(result == remapped)
    }

    @Test("Exact predicates distinguish a boundary point from nearby points at large coordinate ranges")
    func nearCancellation() throws {
        let outer: [SIMD2<Float>] = [.zero, SIMD2(1e20, 0), SIMD2(1e20, 1e20)]
        let baseline = try PlanarShape.triangulate(outer: outer)
        let outside = try PlanarShape.triangulate(outer: outer, holes: [[SIMD2(1, 2)]])
        let onEdge = try PlanarShape.triangulate(outer: outer, holes: [[SIMD2(2, 2)]])
        let inside = try PlanarShape.triangulate(outer: outer, holes: [[SIMD2(2, 1)]])
        #expect(outside == baseline)
        #expect(onEdge.count == 2 && onEdge.allSatisfy { corners($0).contains(3) })
        #expect(inside.count == 3 && inside.allSatisfy { corners($0).contains(3) })
    }

    @Test("Steiner input keeps ring diagnostics and rejects work beyond the search budget")
    func steinerValidation() {
        #expect(throws: PlanarShape.Failure.nonFinitePoint(1)) {
            try PlanarShape.triangulate(outer: Self.square, holes: [[SIMD2(.nan, 1)]])
        }
        #expect(throws: PlanarShape.Failure.degenerateRing(1)) {
            try PlanarShape(outer: Self.square, holes: [[SIMD2(1, 1)]])
        }
        #expect(throws: PlanarShape.Failure.degenerateRing(1)) {
            try PlanarShape.triangulate(outer: Self.square, holes: [[SIMD2(1, 1), SIMD2(1, 1), SIMD2(1, 1)]])
        }
        #expect(throws: PlanarShape.Failure.degenerateRing(2)) {
            try PlanarShape.triangulate(outer: Self.square, holes: [[SIMD2(1, 1)], [SIMD2(1, 1), SIMD2(2, 2), SIMD2(3, 3)]])
        }
        #expect(throws: PlanarShape.Failure.holeOutsideOuterRing(1)) {
            try PlanarShape.triangulate(outer: Self.square, holes: [[SIMD2(1, 1)], [SIMD2(5, 0), SIMD2(6, 0), SIMD2(6, 1)]])
        }
        let tooMany = (0..<3_000).map { [SIMD2(Float($0) / 1_000 + 0.5, 0.5)] }
        #expect(throws: PlanarShape.TriangulationFailure.steinerWorkLimit) {
            try PlanarShape.triangulate(outer: Self.square, holes: tooMany)
        }
    }

    @Test("Invalid rings return the existing PlanarShape errors")
    func invalidRings() {
        let outer: [SIMD2<Float>] = [.zero, SIMD2(4, 0), SIMD2(4, 4), SIMD2(0, 4)]
        #expect(throws: PlanarShape.Failure.degenerateRing(0)) {
            try PlanarShape.triangulate(outer: [.zero, SIMD2(1, 0), .zero])
        }
        #expect(throws: PlanarShape.Failure.nonFinitePoint(1)) {
            try PlanarShape.triangulate(outer: outer, holes: [[.zero, SIMD2(1, 0), SIMD2(0, .nan)]])
        }
        #expect(throws: PlanarShape.Failure.crossingRings) {
            try PlanarShape.triangulate(outer: [.zero, SIMD2(3, 3), SIMD2(0, 4), SIMD2(4, 0)])
        }
        #expect(throws: PlanarShape.Failure.holeOutsideOuterRing(0)) {
            try PlanarShape.triangulate(outer: outer, holes: [[SIMD2(5, 0), SIMD2(6, 0), SIMD2(6, 1)]])
        }
        #expect(throws: PlanarShape.Failure.nestedHole(1)) {
            try PlanarShape.triangulate(outer: outer, holes: [
                [SIMD2(1, 1), SIMD2(3, 1), SIMD2(3, 3), SIMD2(1, 3)],
                [SIMD2(1.5, 1.5), SIMD2(2, 1.5), SIMD2(2, 2)],
            ])
        }
    }

    private func checkTriangles(
        _ triangles: [SIMD3<Int>], points: [SIMD2<Float>], boundary: [[Int]] = [], interior: Set<Int> = []
    ) throws -> Double {
        var total = 0.0
        var edgeIncidence: [SIMD2<Int>: Int] = [:]
        var used = Set<Int>()
        for triangle in triangles {
            let corners = self.corners(triangle)
            try #require(corners.allSatisfy { points.indices.contains($0) })
            used.formUnion(corners)
            let area = cross(points[triangle.x], points[triangle.y], points[triangle.z]) / 2
            #expect(area > 0)
            total += area
            for (a, b) in zip(corners, corners.dropFirst() + [corners[0]]) {
                edgeIncidence[SIMD2(min(a, b), max(a, b)), default: 0] += 1
            }
        }
        if !boundary.isEmpty {
            let expectedEdges = Set(boundary.flatMap { ring in
                zip(ring, ring.dropFirst() + [ring[0]]).map { a, b in SIMD2(min(a, b), max(a, b)) }
            })
            #expect(used == Set(boundary.flatMap { $0 }).union(interior))
            #expect(Set(edgeIncidence.filter { $0.value == 1 }.keys) == expectedEdges)
            #expect(edgeIncidence.values.allSatisfy { $0 == 1 || $0 == 2 })
        }
        return total
    }

    private func corners(_ triangle: SIMD3<Int>) -> [Int] { [triangle.x, triangle.y, triangle.z] }

    private func covers(_ point: SIMD2<Float>, triangles: [SIMD3<Int>], points: [SIMD2<Float>]) -> Bool {
        triangles.contains { triangle in
            let a = points[triangle.x], b = points[triangle.y], c = points[triangle.z]
            return cross(a, b, point) >= 0 && cross(b, c, point) >= 0 && cross(c, a, point) >= 0
        }
    }

    private func cross(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ c: SIMD2<Float>) -> Double {
        let ab = SIMD2<Double>(b) - SIMD2<Double>(a), ac = SIMD2<Double>(c) - SIMD2<Double>(a)
        return ab.x * ac.y - ab.y * ac.x
    }
}
