import CoreGraphics
import Foundation
import simd

/// A filled region of the XY plane: one counterclockwise outer ring and any
/// number of clockwise holes, as seen looking down negative Z.
///
/// Rings list their points without a closing duplicate. Every ring is simple,
/// no two rings cross, each hole lies inside the outer ring, and no hole lies
/// inside another. Rings may touch only at shared vertices.
public struct PlanarShape: Equatable, Sendable {
    /// The counterclockwise boundary.
    public let outer: [SIMD2<Float>]
    /// The clockwise boundaries of the regions removed from the outer ring.
    public let holes: [[SIMD2<Float>]]

    /// Validates explicit rings and orients the outer ring counterclockwise and
    /// each hole clockwise. Throws when a ring has fewer than three distinct
    /// points or no area, when rings cross or overlap, or when a hole is outside
    /// the outer ring or inside another hole.
    public init(outer: [SIMD2<Float>], holes: [[SIMD2<Float>]] = []) throws {
        let rings = try ([outer] + holes).enumerated().map { index, ring in
            try Self.retainedIndices(in: ring, ring: index).map { SIMD2<Double>(ring[$0]) }
        }
        let oriented = try Self.validatedRings(rings)
        self.init(validatedOuter: oriented[0], holes: Array(oriented.dropFirst()))
    }

    /// Triangulates one simple outer ring and its holes in the XY plane.
    /// Input rings may wind either way. Each triangle winds counterclockwise.
    /// Its indices address the original `outer` followed by each original
    /// hole in order. Consecutive duplicate points use the first occurrence,
    /// and a closing copy of the first point is omitted. These omissions do
    /// not shift the original offsets of subsequent rings.
    ///
    /// A one-point hole inserts a Steiner vertex in the filled region or
    /// along one of its edges. Points outside the fill are skipped. An
    /// existing ring vertex takes precedence at a duplicate position,
    /// followed by the first singleton there. An insertion workload over
    /// the search budget throws `TriangulationFailure.steinerWorkLimit`.
    ///
    /// Other holes use the same validation as `PlanarShape(outer:holes:)`.
    /// For intersecting paths that need a fill rule, first use
    /// `shapes(rings:fillRule:)`, then triangulate each resolved shape.
    public static func triangulate(
        outer: [SIMD2<Float>], holes: [[SIMD2<Float>]] = []
    ) throws -> [SIMD3<Int>] {
        var rings: [[SIMD2<Double>]] = []
        var ringIndices: [Int] = []
        var sources: [Int] = []
        var steinerPoints: [SIMD2<Double>] = []
        var steinerSources: [Int] = []
        var offset = 0
        for (index, ring) in ([outer] + holes).enumerated() {
            if index > 0, ring.count == 1 {
                let point = ring[0]
                guard point.x.isFinite && point.y.isFinite else { throw Failure.nonFinitePoint(index) }
                steinerPoints.append(SIMD2<Double>(point))
                steinerSources.append(offset)
            } else {
                let retained = try retainedIndices(in: ring, ring: index)
                rings.append(retained.map { SIMD2<Double>(ring[$0]) })
                ringIndices.append(index)
                sources.append(contentsOf: retained.lazy.map { offset + $0 })
            }
            offset += ring.count
        }
        // Keep the original ring order for index mapping. The clipping step
        // can orient the rings itself after validation.
        _ = try validatedRings(rings, sourceIndices: ringIndices)
        sources.append(contentsOf: steinerSources)
        return try triangulateRings(rings, steinerPoints: steinerPoints).map {
            SIMD3(sources[$0.x], sources[$0.y], sources[$0.z])
        }
    }

    /// Returns counterclockwise triangle indices into this shape's stored
    /// `outer` followed by its stored `holes` in order.
    public func triangulatedIndices() throws -> [SIMD3<Int>] {
        try Self.triangulateRings(([outer] + holes).map { $0.map(SIMD2<Double>.init) })
    }

    private static func triangulateRings(
        _ rings: [[SIMD2<Double>]], steinerPoints: [SIMD2<Double>] = []
    ) throws -> [SIMD3<Int>] {
        var triangles: [SIMD3<Int>]
        do {
            triangles = try PolygonTriangulator.triangulate(outer: rings[0], holes: Array(rings.dropFirst()))
        } catch {
            throw Failure.unresolvedBoundary
        }
        let points = rings.flatMap { $0 }
        guard !triangles.isEmpty, triangles.allSatisfy({ triangle in
            let a = points[triangle.x], b = points[triangle.y], c = points[triangle.z]
            return PolygonTriangulator.cross(b - a, c - a) > 0
        }) else { throw Failure.unresolvedBoundary }
        if !steinerPoints.isEmpty {
            try PolygonSteinerInsertion.apply(points: points + steinerPoints, boundaryCount: points.count, to: &triangles)
        }
        return triangles
    }

    private static func validatedRings(
        _ rings: [[SIMD2<Double>]], sourceIndices: [Int]? = nil
    ) throws -> [[SIMD2<Double>]] {
        var oriented: [[SIMD2<Double>]] = []
        for (index, ring) in rings.enumerated() {
            let area = PolygonTriangulator.doubledArea(ring)
            guard area != 0 else { throw Failure.degenerateRing(sourceIndices?[index] ?? index) }
            oriented.append((area > 0) == (index == 0) ? ring : ring.reversed())
        }
        guard !PlanarArrangement.hasCrossing(oriented) else { throw Failure.crossingRings }
        for (index, hole) in oriented.enumerated().dropFirst() {
            let sourceHole = (sourceIndices?[index] ?? index) - 1
            let probe = PlanarArrangement.interiorProbe(of: hole)
            guard PlanarArrangement.winding(of: probe, in: oriented[0]) != 0 else {
                throw Failure.holeOutsideOuterRing(sourceHole)
            }
            for (otherIndex, other) in oriented.enumerated().dropFirst() where otherIndex != index {
                if PlanarArrangement.winding(of: probe, in: other) != 0 {
                    throw Failure.nestedHole(sourceHole)
                }
            }
        }
        return oriented
    }

    init(validatedOuter outer: [SIMD2<Double>], holes: [[SIMD2<Double>]]) {
        self.outer = outer.map(SIMD2<Float>.init)
        self.holes = holes.map { $0.map(SIMD2<Float>.init) }
    }

    /// The enclosed area: the outer ring's area less its holes'.
    public var area: Double {
        let outerArea = PolygonTriangulator.doubledArea(outer.map(SIMD2<Double>.init)) / 2
        return holes.reduce(outerArea) { $0 + PolygonTriangulator.doubledArea($1.map(SIMD2<Double>.init)) / 2 }
    }

    /// Resolves closed rings that may wind either way, overlap, touch or
    /// cross themselves into disjoint shapes that cover exactly the points the
    /// fill rule fills.
    ///
    /// Crossing edges are split at their intersections, which round to Float.
    /// An edge is kept as a boundary when the fill rule fills one side and not
    /// the other; edges with the same fill on both sides, such as the shared
    /// edge of two abutting squares, disappear. Each hole is assigned to the
    /// smallest outer ring that contains it.
    public static func shapes(rings: [[SIMD2<Float>]], fillRule: CGPathFillRule = .winding) throws -> [PlanarShape] {
        try shapes(rings: rings, fillRule: fillRule, maximumEdges: .max)
    }

    static func shapes(
        rings: [[SIMD2<Float>]], fillRule: CGPathFillRule, maximumEdges: Int
    ) throws -> [PlanarShape] {
        var cleaned: [[SIMD2<Double>]] = []
        for (index, ring) in rings.enumerated() {
            guard ring.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { throw Failure.nonFinitePoint(index) }
            let points = PlanarShape.removingRepeats(ring.map(SIMD2<Double>.init))
            if points.count >= 3 { cleaned.append(points) }
        }
        guard !cleaned.isEmpty else { return [] }
        return try PlanarArrangement.resolve(cleaned, evenOdd: fillRule == .evenOdd, maximumEdges: maximumEdges)
    }

    /// Flattens every subpath of a Core Graphics path and resolves the result
    /// with ``shapes(rings:fillRule:)``.
    ///
    /// Quadratic and cubic Béziers are divided uniformly into the fewest
    /// segments that Wang's bound guarantees stay within `tolerance` of the
    /// curve, up to 1,024 per curve. Open subpaths close implicitly, as Core
    /// Graphics closes them for filling.
    /// [Wang's formula](https://doi.org/10.1145/1015706.1015757)
    public static func shapes(
        from path: CGPath, fillRule: CGPathFillRule = .winding, tolerance: Float
    ) throws -> [PlanarShape] {
        guard tolerance.isFinite, tolerance > 0 else { throw Failure.invalidTolerance }
        let rings = try flatten(path, tolerance: Double(tolerance))
        return try shapes(rings: rings, fillRule: fillRule)
    }

    /// The subpaths of `path` as closed polylines, with curves flattened to `tolerance`.
    static func flatten(_ path: CGPath, tolerance: Double, maximumPoints: Int = .max) throws -> [[SIMD2<Float>]] {
        try flattenedSubpaths(path, tolerance: tolerance, maximumPoints: maximumPoints)
            .filter { $0.points.count >= 3 }
            .map { removingRepeats($0.points).map(SIMD2<Float>.init) }
    }

    struct FlattenedSubpath {
        let points: [SIMD2<Double>]
        let closed: Bool
    }

    /// Retains open two-point subpaths and explicit closure for stroke tessellation.
    static func flattenedSubpaths(
        _ path: CGPath, tolerance: Double, maximumPoints: Int
    ) throws -> [FlattenedSubpath] {
        var subpaths: [FlattenedSubpath] = []
        var current: [SIMD2<Double>] = []
        var nonFinite = false
        var remaining = maximumPoints
        var exceeded = false
        func append(_ point: SIMD2<Double>) {
            guard remaining > 0 else { exceeded = true; return }
            remaining -= 1
            current.append(point)
        }
        func finishSubpath(closed: Bool = false) {
            if current.count >= 2 { subpaths.append(FlattenedSubpath(points: current, closed: closed)) }
            current = []
        }
        func point(_ value: CGPoint) -> SIMD2<Double> {
            let result = SIMD2<Double>(Double(value.x), Double(value.y))
            if !result.x.isFinite || !result.y.isFinite { nonFinite = true }
            return result
        }
        path.applyWithBlock { pointer in
            guard !exceeded else { return }
            let element = pointer.pointee
            switch element.type {
            case .moveToPoint:
                finishSubpath()
                append(point(element.points[0]))
            case .addLineToPoint:
                append(point(element.points[0]))
            case .addQuadCurveToPoint:
                let start = current.last ?? .zero
                let control = point(element.points[0]), end = point(element.points[1])
                let bend = simd_length(start - 2 * control + end)
                let count = curveSegmentCount(bend: bend, degreeFactor: 0.25, tolerance: tolerance)
                for step in 1...count {
                    let t = Double(step) / Double(count), s = 1 - t
                    append(step == count ? end : s * s * start + 2 * s * t * control + t * t * end)
                }
            case .addCurveToPoint:
                let start = current.last ?? .zero
                let first = point(element.points[0]), second = point(element.points[1]), end = point(element.points[2])
                let bend = max(simd_length(start - 2 * first + second), simd_length(first - 2 * second + end))
                let count = curveSegmentCount(bend: bend, degreeFactor: 0.75, tolerance: tolerance)
                for step in 1...count {
                    let t = Double(step) / Double(count), s = 1 - t
                    let value = s * s * s * start + 3 * s * s * t * first + 3 * s * t * t * second + t * t * t * end
                    append(step == count ? end : value)
                }
            case .closeSubpath:
                let start = current.first
                finishSubpath(closed: true)
                if let start { append(start) }
            @unknown default:
                break
            }
        }
        finishSubpath()
        guard !exceeded else { throw PlanarComplexityLimit.exceeded }
        guard !nonFinite else {
            throw Failure.nonFinitePoint(subpaths.lazy.filter { $0.points.count >= 3 }.count)
        }
        return subpaths
    }

    /// Wang's bound: `n = ⌈√(d(d−1)/8 · M / tolerance)⌉` for degree `d` and largest second difference `M`.
    private static func curveSegmentCount(bend: Double, degreeFactor: Double, tolerance: Double) -> Int {
        guard bend.isFinite, bend > 0 else { return 1 }
        let count = (degreeFactor * bend / tolerance).squareRoot().rounded(.up)
        return Int(min(max(count, 1), 1_024))
    }

    /// Drops consecutive repeats after rounding to Float, and a closing duplicate.
    static func removingRepeats(_ ring: [SIMD2<Double>]) -> [SIMD2<Double>] {
        var result: [SIMD2<Double>] = []
        result.reserveCapacity(ring.count)
        for point in ring {
            let rounded = SIMD2<Double>(SIMD2<Float>(point))
            if result.last != rounded { result.append(rounded) }
        }
        while result.count > 1, result.first == result.last { result.removeLast() }
        return result
    }

    private static func retainedIndices(in points: [SIMD2<Float>], ring: Int) throws -> [Int] {
        var retained: [Int] = []
        retained.reserveCapacity(points.count)
        for index in points.indices {
            let point = points[index]
            guard point.x.isFinite && point.y.isFinite else { throw Failure.nonFinitePoint(ring) }
            if let previous = retained.last, points[previous] == point { continue }
            retained.append(index)
        }
        if retained.count > 1, points[retained[0]] == points[retained[retained.count - 1]] {
            retained.removeLast()
        }
        guard retained.count >= 3 else { throw Failure.degenerateRing(ring) }
        return retained
    }

    public enum Failure: LocalizedError, Equatable {
        case nonFinitePoint(Int)
        case degenerateRing(Int)
        case crossingRings
        case holeOutsideOuterRing(Int)
        case nestedHole(Int)
        case invalidTolerance
        case unresolvedBoundary

        public var errorDescription: String? {
            switch self {
            case .nonFinitePoint(let ring): "Shape ring \(ring) contains a non-finite point."
            case .degenerateRing(let ring): "Shape ring \(ring) needs at least three distinct points enclosing an area."
            case .crossingRings: "Shape rings must not cross or overlap; use PlanarShape.shapes(rings:fillRule:) to resolve them."
            case .holeOutsideOuterRing(let hole): "Hole \(hole) lies outside the shape's outer ring."
            case .nestedHole(let hole): "Hole \(hole) lies inside another hole."
            case .invalidTolerance: "The curve flattening tolerance must be finite and positive."
            case .unresolvedBoundary:
                "The rings' intersections could not be resolved at Float precision; separate nearly coincident edges."
            }
        }
    }

    public enum TriangulationFailure: LocalizedError, Equatable {
        case steinerWorkLimit

        public var errorDescription: String? {
            "The Steiner points and polygon triangles exceed the triangulation search budget."
        }
    }
}

enum PlanarComplexityLimit: Error { case exceeded }

/// Splits rings at their intersections and keeps the edges that separate filled from empty space.
enum PlanarArrangement {
    static func resolve(
        _ rings: [[SIMD2<Double>]], evenOdd: Bool, maximumEdges: Int
    ) throws -> [PlanarShape] {
        var pool = PointPool()
        var segments: [(Int, Int)] = []
        for ring in rings {
            let ids = ring.map { pool.id(for: $0) }
            for index in ids.indices {
                let a = ids[index], b = ids[(index + 1) % ids.count]
                if a != b { segments.append((a, b)) }
            }
        }
        let pieces = try split(segments, pool: &pool, maximumEdges: maximumEdges)

        // Net multiplicity of each undirected edge, positive along (low, high).
        var multiplicity: [EdgeKey: Int] = [:]
        for (a, b) in pieces {
            let key = EdgeKey(a, b)
            multiplicity[key, default: 0] += a < b ? 1 : -1
        }
        let edges = multiplicity.keys.sorted { ($0.low, $0.high) < ($1.low, $1.high) }
        let points = pool.points
        func filled(_ winding: Int) -> Bool { evenOdd ? winding & 1 != 0 : winding != 0 }

        var boundary: [(Int, Int)] = []
        for key in edges {
            let m = multiplicity[key]!
            let a = points[key.low], b = points[key.high]
            let middle = (a + b) * 0.5
            let direction = b - a
            // Cast along +X unless the edge is horizontal, then along +Y.
            let horizontalRay = abs(direction.y) >= abs(direction.x)
            var winding = 0
            for other in edges where other != key {
                let otherM = multiplicity[other]!
                guard otherM != 0 else { continue }
                winding += otherM * crossing(from: points[other.low], to: points[other.high], at: middle, alongX: horizontalRay)
            }
            // The ray's side of the edge: right of low→high when the ray points to the right of the edge.
            let rayIsRight = horizontalRay ? direction.y > 0 : direction.x < 0
            let right = rayIsRight ? winding : winding - m
            let left = right + m
            guard filled(left) != filled(right) else { continue }
            boundary.append(filled(left) ? (key.low, key.high) : (key.high, key.low))
        }
        let loops = try link(boundary, points: points)
        return try nest(loops, points: points)
    }

    /// Sunday's signed crossing of a ray from `point` along +X (or +Y) by the edge `a → b`.
    private static func crossing(from a: SIMD2<Double>, to b: SIMD2<Double>, at point: SIMD2<Double>, alongX: Bool) -> Int {
        // Map the +Y ray onto +X by rotating coordinates a quarter turn clockwise.
        let p = alongX ? a - point : SIMD2(a.y - point.y, point.x - a.x)
        let q = alongX ? b - point : SIMD2(b.y - point.y, point.x - b.x)
        if p.y <= 0 {
            if q.y > 0, PolygonTriangulator.cross(q - p, -p) > 0 { return 1 }
        } else if q.y <= 0, PolygonTriangulator.cross(q - p, -p) < 0 {
            return -1
        }
        return 0
    }

    /// Splits segments at every crossing, touching endpoint and collinear overlap.
    private static func split(
        _ segments: [(Int, Int)], pool: inout PointPool, maximumEdges: Int
    ) throws -> [(Int, Int)] {
        guard segments.count <= maximumEdges else { throw PlanarComplexityLimit.exceeded }
        var cuts = segments.map { Set([$0.0, $0.1]) }
        var edgesLeft = maximumEdges - segments.count
        func addCut(_ point: Int, to index: Int) throws {
            guard cuts[index].insert(point).inserted else { return }
            guard edgesLeft > 0 else { throw PlanarComplexityLimit.exceeded }
            edgesLeft -= 1
        }
        let order = segments.indices.sorted {
            min(pool.points[segments[$0].0].x, pool.points[segments[$0].1].x)
                < min(pool.points[segments[$1].0].x, pool.points[segments[$1].1].x)
        }
        for (position, i) in order.enumerated() {
            let a = pool.points[segments[i].0], b = pool.points[segments[i].1]
            let upperX = max(a.x, b.x)
            for j in order[(position + 1)...] {
                let c = pool.points[segments[j].0], d = pool.points[segments[j].1]
                if min(c.x, d.x) > upperX { break }
                guard min(c.y, d.y) <= max(a.y, b.y), min(a.y, b.y) <= max(c.y, d.y) else { continue }
                let o1 = orientation(a, b, c), o2 = orientation(a, b, d)
                let o3 = orientation(c, d, a), o4 = orientation(c, d, b)
                if o1 * o2 < 0, o3 * o4 < 0 {
                    let t = o3 / (o3 - o4)
                    let id = pool.id(for: SIMD2<Double>(SIMD2<Float>(a + (b - a) * t)))
                    try addCut(id, to: i)
                    try addCut(id, to: j)
                    continue
                }
                if o1 == 0, within(c, a, b) { try addCut(segments[j].0, to: i) }
                if o2 == 0, within(d, a, b) { try addCut(segments[j].1, to: i) }
                if o3 == 0, within(a, c, d) { try addCut(segments[i].0, to: j) }
                if o4 == 0, within(b, c, d) { try addCut(segments[i].1, to: j) }
            }
        }
        var pieces: [(Int, Int)] = []
        for (index, segment) in segments.enumerated() {
            let a = pool.points[segment.0], direction = pool.points[segment.1] - a
            let sorted = cuts[index].sorted { lhs, rhs in
                let first = simd_dot(pool.points[lhs] - a, direction)
                let second = simd_dot(pool.points[rhs] - a, direction)
                return first == second ? lhs < rhs : first < second
            }
            for (from, to) in zip(sorted, sorted.dropFirst()) where from != to {
                pieces.append((from, to))
            }
        }
        return pieces
    }

    /// Joins directed boundary edges into loops, turning at shared vertices so
    /// the filled side stays on the left of each loop.
    private static func link(_ edges: [(Int, Int)], points: [SIMD2<Double>]) throws -> [[Int]] {
        var outgoing: [Int: [Int]] = [:]
        for (index, edge) in edges.enumerated() { outgoing[edge.0, default: []].append(index) }
        var used = [Bool](repeating: false, count: edges.count)
        var loops: [[Int]] = []
        for start in edges.indices where !used[start] {
            var loop: [Int] = []
            var current = start
            while !used[current] {
                used[current] = true
                let (from, to) = edges[current]
                loop.append(from)
                guard let candidates = outgoing[to] else { throw PlanarShape.Failure.unresolvedBoundary }
                let back = points[from] - points[to]
                let backAngle = atan2(back.y, back.x)
                // The first outgoing edge clockwise from the reversed incoming edge.
                current = candidates.min { lhs, rhs in
                    clockwiseTurn(from: backAngle, to: points[edges[lhs].1] - points[to])
                        < clockwiseTurn(from: backAngle, to: points[edges[rhs].1] - points[to])
                }!
            }
            guard current == start else { throw PlanarShape.Failure.unresolvedBoundary }
            if loop.count >= 3 { loops.append(loop) }
        }
        return loops
    }

    private static func clockwiseTurn(from angle: Double, to direction: SIMD2<Double>) -> Double {
        var turn = angle - atan2(direction.y, direction.x)
        while turn <= 0 { turn += 2 * .pi }
        while turn > 2 * .pi { turn -= 2 * .pi }
        return turn
    }

    /// Pairs clockwise loops with the smallest counterclockwise loop containing them.
    private static func nest(_ loops: [[Int]], points: [SIMD2<Double>]) throws -> [PlanarShape] {
        var outers: [(ring: [SIMD2<Double>], area: Double)] = []
        var holes: [[SIMD2<Double>]] = []
        for loop in loops {
            let ring = loop.map { points[$0] }
            let area = PolygonTriangulator.doubledArea(ring)
            if area > 0 { outers.append((ring, area)) } else if area < 0 { holes.append(ring) }
        }
        var assigned = [[[SIMD2<Double>]]](repeating: [], count: outers.count)
        for hole in holes {
            let probe = interiorProbe(of: hole)
            var best: Int?
            for (index, outer) in outers.enumerated() where winding(of: probe, in: outer.ring) != 0 {
                if best.map({ outer.area < outers[$0].area }) ?? true { best = index }
            }
            guard let best else { throw PlanarShape.Failure.unresolvedBoundary }
            assigned[best].append(hole)
        }
        let shapes = outers.indices.map { PlanarShape(validatedOuter: outers[$0].ring, holes: assigned[$0]) }
        // Rounding intersections to Float can, in rare cases, introduce a new crossing.
        for shape in shapes {
            let rings = ([shape.outer] + shape.holes).map { $0.map(SIMD2<Double>.init) }
            guard !hasCrossing(rings) else { throw PlanarShape.Failure.unresolvedBoundary }
        }
        return shapes
    }

    /// The midpoint of a ring's first edge, which lies on no other ring of a resolved shape.
    static func interiorProbe(of ring: [SIMD2<Double>]) -> SIMD2<Double> {
        (ring[0] + ring[1]) * 0.5
    }

    /// The winding number of a closed ring around a point.
    static func winding(of point: SIMD2<Double>, in ring: [SIMD2<Double>]) -> Int {
        var total = 0
        var previous = ring[ring.count - 1]
        for current in ring {
            total += crossing(from: previous, to: current, at: point, alongX: true)
            previous = current
        }
        return total
    }

    /// Whether any two edges of the rings cross or overlap other than at a shared endpoint.
    static func hasCrossing(_ rings: [[SIMD2<Double>]]) -> Bool {
        var edges: [(SIMD2<Double>, SIMD2<Double>)] = []
        for ring in rings {
            for index in ring.indices { edges.append((ring[index], ring[(index + 1) % ring.count])) }
        }
        let order = edges.indices.sorted { min(edges[$0].0.x, edges[$0].1.x) < min(edges[$1].0.x, edges[$1].1.x) }
        for (position, i) in order.enumerated() {
            let (a, b) = edges[i]
            let upperX = max(a.x, b.x)
            for j in order[(position + 1)...] {
                let (c, d) = edges[j]
                if min(c.x, d.x) > upperX { break }
                guard min(c.y, d.y) <= max(a.y, b.y), min(a.y, b.y) <= max(c.y, d.y) else { continue }
                let o1 = orientation(a, b, c), o2 = orientation(a, b, d)
                let o3 = orientation(c, d, a), o4 = orientation(c, d, b)
                if o1 * o2 < 0, o3 * o4 < 0 { return true }
                // Touching at a shared endpoint is allowed; any other contact is a crossing.
                if o1 == 0, c != a, c != b, within(c, a, b) { return true }
                if o2 == 0, d != a, d != b, within(d, a, b) { return true }
                if o3 == 0, a != c, a != d, within(a, c, d) { return true }
                if o4 == 0, b != c, b != d, within(b, c, d) { return true }
                if o1 == 0, o2 == 0, (a == c && b == d) || (a == d && b == c) { return true }
            }
        }
        return false
    }

    /// Twice the signed area of `abc`, or zero when rounding cannot decide its sign.
    static func orientation(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>) -> Double {
        let left = (b.x - a.x) * (c.y - a.y), right = (b.y - a.y) * (c.x - a.x)
        let value = left - right
        return abs(value) <= 8 * Double.ulpOfOne * (abs(left) + abs(right)) ? 0 : value
    }

    /// Whether `p`, collinear with `a` and `b`, lies on the closed segment between them.
    private static func within(_ p: SIMD2<Double>, _ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Bool {
        p.x >= min(a.x, b.x) && p.x <= max(a.x, b.x) && p.y >= min(a.y, b.y) && p.y <= max(a.y, b.y)
    }

    private struct EdgeKey: Hashable {
        let low: Int
        let high: Int
        init(_ a: Int, _ b: Int) {
            low = min(a, b)
            high = max(a, b)
        }
    }

    /// Distinct Float-representable points.
    private struct PointPool {
        var points: [SIMD2<Double>] = []
        var lookup: [SIMD2<Double>: Int] = [:]

        mutating func id(for point: SIMD2<Double>) -> Int {
            if let id = lookup[point] { return id }
            points.append(point)
            lookup[point] = points.count - 1
            return points.count - 1
        }
    }
}
