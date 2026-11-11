import Foundation
import simd

/// Ear-clipping triangulation of a simple polygon with holes.
///
/// Holes join the outer ring through bridges found with Eberly's visibility
/// method, and the merged ring is clipped ear by ear with the fallbacks of the
/// earcut algorithm: collinear points are filtered, local self-intersections
/// cured, and remaining rings split along valid diagonals.
/// [Eberly, Triangulation by Ear Clipping](https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf),
/// [earcut](https://github.com/mapbox/earcut) (ISC License, Copyright (c) 2024 Mapbox).
///
/// Unlike earcut, no boundary vertex is lost. Points dropped as collinear are
/// put back by splitting the output triangle edge that passes through them,
/// so neighbouring meshes that share those points stay conforming. The result
/// is checked before it is returned: every triangle must be counterclockwise
/// and the triangle areas must sum to the polygon's area.
package enum PolygonTriangulator {
    /// The relative sine below which three points count as collinear.
    static let collinearTolerance = 1e-12

    /// Triangulates one outer ring and its holes, each listed without a closing
    /// duplicate. Rings may wind either way. The result indexes the outer ring's
    /// points first, then each hole's in order, and every triangle winds
    /// counterclockwise in the XY plane.
    package static func triangulate(outer: [SIMD2<Double>], holes: [[SIMD2<Double>]] = []) throws -> [SIMD3<Int>] {
        let rings = [outer] + holes
        for (ringIndex, ring) in rings.enumerated() {
            guard ring.count >= 3 else { throw Failure.ringTooSmall(ringIndex) }
            guard ring.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { throw Failure.nonFinitePoint(ringIndex) }
        }
        var state = State()
        var offset = 0
        var starts: [Int] = []
        for (ringIndex, ring) in rings.enumerated() {
            if let start = state.linkedList(ring, sourceOffset: offset, counterclockwise: ringIndex == 0) {
                starts.append(start)
            } else if ringIndex == 0 {
                throw Failure.degenerateOuterRing
            }
            offset += ring.count
        }
        var outerNode = starts[0]
        if starts.count > 1 {
            outerNode = state.eliminateHoles(Array(starts.dropFirst()), outerNode: outerNode)
        }
        state.earcutLinked(outerNode, pass: 0)

        let points = rings.flatMap { $0 }
        var triangles = state.triangles
        reinsertCollinearPoints(state.removed, into: &triangles, points: points)
        try verify(triangles, points: points, rings: rings)
        return triangles
    }

    package enum Failure: LocalizedError, Equatable {
        case ringTooSmall(Int)
        case nonFinitePoint(Int)
        case degenerateOuterRing
        case areaMismatch
        case invertedTriangle

        package var errorDescription: String? {
            switch self {
            case .ringTooSmall(let ring): "Polygon ring \(ring) needs at least three points."
            case .nonFinitePoint(let ring): "Polygon ring \(ring) contains a non-finite point."
            case .degenerateOuterRing: "The outer polygon ring encloses no area."
            case .areaMismatch:
                "The polygon could not be triangulated: its rings cross, touch or overlap at the supplied precision."
            case .invertedTriangle: "The polygon could not be triangulated without an inverted triangle."
            }
        }
    }

    /// The doubled signed area of a ring: positive when counterclockwise.
    /// Coordinates are taken relative to the first point, so rounding scales
    /// with the ring's size rather than its distance from the origin.
    package static func doubledArea(_ ring: [SIMD2<Double>]) -> Double {
        let origin = ring[0]
        var sum = 0.0
        var previous = ring[ring.count - 1] - origin
        for point in ring {
            let current = point - origin
            sum += previous.x * current.y - current.x * previous.y
            previous = current
        }
        return sum
    }

    private static func verify(_ triangles: [SIMD3<Int>], points: [SIMD2<Double>], rings: [[SIMD2<Double>]]) throws {
        let expected = abs(doubledArea(rings[0])) - rings.dropFirst().reduce(0) { $0 + abs(doubledArea($1)) }
        var lower = points[0], upper = points[0]
        for point in points {
            lower = simd_min(lower, point)
            upper = simd_max(upper, point)
        }
        let extent = simd_reduce_max(upper - lower)
        // Rounding in each doubled triangle area is a few ulps of the extent squared.
        let slack = Double(points.count) * 64 * Double.ulpOfOne * extent * extent
        var total = 0.0
        for triangle in triangles {
            let a = points[triangle.x], b = points[triangle.y], c = points[triangle.z]
            let area = cross(b - a, c - a)
            guard area >= -slack else { throw Failure.invertedTriangle }
            total += area
        }
        guard abs(total - expected) <= max(1e-9 * abs(expected), slack) else { throw Failure.areaMismatch }
    }

    /// Splits each triangle edge that passes through a point removed as collinear.
    private static func reinsertCollinearPoints(
        _ removed: [Int], into triangles: inout [SIMD3<Int>], points: [SIMD2<Double>]
    ) {
        var seen = Set<Int>()
        for source in removed where seen.insert(source).inserted {
            let p = points[source]
            var index = 0
            while index < triangles.count {
                let triangle = triangles[index]
                let corners = [triangle.x, triangle.y, triangle.z]
                var split = false
                for edge in 0..<3 {
                    let u = corners[edge], v = corners[(edge + 1) % 3], w = corners[(edge + 2) % 3]
                    guard u != source, v != source, w != source, points[u] != p, points[v] != p else { continue }
                    if liesStrictlyOn(p, points[u], points[v]) {
                        triangles[index] = SIMD3(u, source, w)
                        triangles.append(SIMD3(source, v, w))
                        split = true
                        break
                    }
                }
                if !split { index += 1 }
            }
        }
    }

    private static func liesStrictlyOn(_ p: SIMD2<Double>, _ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Bool {
        let ab = b - a
        let lengthSquared = simd_length_squared(ab)
        guard lengthSquared > 0 else { return false }
        let t = simd_dot(p - a, ab) / lengthSquared
        guard t > 0, t < 1 else { return false }
        let offset = cross(ab, p - a)
        return abs(offset) <= collinearTolerance * 8 * lengthSquared.squareRoot() * simd_length(p - a)
            || abs(offset) <= 16 * Double.ulpOfOne * lengthSquared
    }

    @inline(__always)
    static func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        a.x * b.y - a.y * b.x
    }

    /// A doubly linked ring of polygon vertices stored in parallel arrays.
    private struct State {
        var x: [Double] = []
        var y: [Double] = []
        var source: [Int] = []
        var previous: [Int] = []
        var following: [Int] = []
        var triangles: [SIMD3<Int>] = []
        /// Source indices removed as duplicates or collinear points.
        var removed: [Int] = []

        mutating func makeNode(_ index: Int, _ point: SIMD2<Double>) -> Int {
            let node = x.count
            x.append(point.x)
            y.append(point.y)
            source.append(index)
            previous.append(node)
            following.append(node)
            return node
        }

        mutating func insert(_ index: Int, _ point: SIMD2<Double>, after last: Int?) -> Int {
            let node = makeNode(index, point)
            if let last {
                following[node] = following[last]
                previous[node] = last
                previous[following[last]] = node
                following[last] = node
            }
            return node
        }

        mutating func remove(_ node: Int) {
            following[previous[node]] = following[node]
            previous[following[node]] = previous[node]
        }

        /// Links a ring in the requested orientation and returns its last node.
        mutating func linkedList(_ ring: [SIMD2<Double>], sourceOffset: Int, counterclockwise: Bool) -> Int? {
            var last: Int?
            let area = PolygonTriangulator.doubledArea(ring)
            if counterclockwise == (area > 0) {
                for (index, point) in ring.enumerated() { last = insert(sourceOffset + index, point, after: last) }
            } else {
                for (index, point) in ring.enumerated().reversed() { last = insert(sourceOffset + index, point, after: last) }
            }
            if let node = last, equals(node, following[node]) {
                removed.append(source[node])
                remove(node)
                last = following[node]
            }
            guard let last, following[last] != previous[last] else { return nil }
            return last
        }

        func equals(_ a: Int, _ b: Int) -> Bool { x[a] == x[b] && y[a] == y[b] }

        /// Earcut's orientation: negative for a counterclockwise (convex) turn, zero when collinear.
        func area(_ p: Int, _ q: Int, _ r: Int) -> Double {
            let first = (y[q] - y[p]) * (x[r] - x[q])
            let second = (x[q] - x[p]) * (y[r] - y[q])
            let value = first - second
            let pq = SIMD2(x[q] - x[p], y[q] - y[p]), qr = SIMD2(x[r] - x[q], y[r] - y[q])
            let scale = simd_length(pq) * simd_length(qr)
            if abs(value) <= PolygonTriangulator.collinearTolerance * scale
                || abs(value) <= 4 * Double.ulpOfOne * (abs(first) + abs(second)) {
                return 0
            }
            return value
        }

        /// Whether `p` lies inside or on the counterclockwise triangle `abc`.
        func pointInTriangle(_ a: Int, _ b: Int, _ c: Int, _ p: Int) -> Bool {
            area(a, b, p) <= 0 && area(b, c, p) <= 0 && area(c, a, p) <= 0
        }

        /// Removes duplicate and collinear points between `start` and `end`.
        mutating func filterPoints(_ start: Int, _ end: Int? = nil) -> Int {
            var end = end ?? start
            var p = start
            var again: Bool
            repeat {
                again = false
                if equals(p, following[p]) || area(previous[p], p, following[p]) == 0 {
                    removed.append(source[p])
                    remove(p)
                    p = previous[p]
                    end = p
                    if p == following[p] { break }
                    again = true
                } else {
                    p = following[p]
                }
            } while again || p != end
            return end
        }

        mutating func earcutLinked(_ start: Int, pass: Int) {
            var ear = start
            var stop = ear
            while previous[ear] != following[ear] {
                let before = previous[ear], after = following[ear]
                if isEar(ear) {
                    triangles.append(SIMD3(source[before], source[ear], source[after]))
                    remove(ear)
                    // Skipping the next vertex leads to fewer sliver triangles.
                    ear = following[after]
                    stop = following[after]
                    continue
                }
                ear = after
                if ear == stop {
                    switch pass {
                    case 0:
                        earcutLinked(filterPoints(ear), pass: 1)
                    case 1:
                        let cured = cureLocalIntersections(filterPoints(ear))
                        earcutLinked(cured, pass: 2)
                    default:
                        splitEarcut(ear)
                    }
                    break
                }
            }
        }

        func isEar(_ ear: Int) -> Bool {
            let a = previous[ear], b = ear, c = following[ear]
            guard area(a, b, c) < 0 else { return false }
            let lowerX = min(x[a], x[b], x[c]), upperX = max(x[a], x[b], x[c])
            let lowerY = min(y[a], y[b], y[c]), upperY = max(y[a], y[b], y[c])
            var p = following[c]
            while p != a {
                if x[p] >= lowerX, x[p] <= upperX, y[p] >= lowerY, y[p] <= upperY,
                   !equals(p, a), pointInTriangle(a, b, c, p), area(previous[p], p, following[p]) >= 0 {
                    return false
                }
                p = following[p]
            }
            return true
        }

        mutating func cureLocalIntersections(_ start: Int) -> Int {
            var start = start
            var p = start
            repeat {
                let a = previous[p], b = following[following[p]]
                if !equals(a, b), intersects(a, p, following[p], b), locallyInside(a, b), locallyInside(b, a) {
                    triangles.append(SIMD3(source[a], source[p], source[b]))
                    let next = following[p]
                    remove(p)
                    remove(next)
                    p = b
                    start = b
                }
                p = following[p]
            } while p != start
            return filterPoints(p)
        }

        mutating func splitEarcut(_ start: Int) {
            var a = start
            repeat {
                var b = following[following[a]]
                while b != previous[a] {
                    if source[a] != source[b], isValidDiagonal(a, b) {
                        var c = splitPolygon(a, b)
                        let first = filterPoints(a, following[a])
                        c = filterPoints(c, following[c])
                        earcutLinked(first, pass: 0)
                        earcutLinked(c, pass: 0)
                        return
                    }
                    b = following[b]
                }
                a = following[a]
            } while a != start
        }

        mutating func eliminateHoles(_ holes: [Int], outerNode: Int) -> Int {
            var queue = holes.map(leftmost)
            queue.sort { lhs, rhs in
                if x[lhs] != x[rhs] { return x[lhs] < x[rhs] }
                if y[lhs] != y[rhs] { return y[lhs] < y[rhs] }
                // Holes meeting at their leftmost point sort counterclockwise so each bridges at that point.
                let lhsSlope = (y[following[lhs]] - y[lhs]) / (x[following[lhs]] - x[lhs])
                let rhsSlope = (y[following[rhs]] - y[rhs]) / (x[following[rhs]] - x[rhs])
                return lhsSlope < rhsSlope
            }
            var outer = outerNode
            for hole in queue {
                outer = eliminateHole(hole, outerNode: outer)
            }
            return outer
        }

        mutating func eliminateHole(_ hole: Int, outerNode: Int) -> Int {
            guard let bridge = findHoleBridge(hole, outerNode: outerNode) else { return outerNode }
            let bridgeReverse = splitPolygon(bridge, hole)
            _ = filterPoints(bridgeReverse, following[bridgeReverse])
            return filterPoints(bridge, following[bridge])
        }

        /// David Eberly's search for an outer vertex visible from the hole's leftmost point.
        func findHoleBridge(_ hole: Int, outerNode: Int) -> Int? {
            var p = outerNode
            let hx = x[hole], hy = y[hole]
            var qx = -Double.infinity
            var candidate: Int?
            if equals(hole, p) { return p }
            repeat {
                let next = following[p]
                if equals(hole, next) { return next }
                if hy <= y[p], hy >= y[next], y[next] != y[p] {
                    let crossing = x[p] + (hy - y[p]) * (x[next] - x[p]) / (y[next] - y[p])
                    if crossing <= hx, crossing > qx {
                        qx = crossing
                        candidate = x[p] < x[next] ? p : next
                        if crossing == hx { return candidate }
                    }
                }
                p = next
            } while p != outerNode
            guard var m = candidate else { return nil }

            // Choose the reflex vertex inside the hole–crossing–candidate triangle
            // with the smallest angle to the ray, when there is one.
            let stop = m
            let mx = x[m], my = y[m]
            var tanMin = Double.infinity
            p = m
            repeat {
                if hx >= x[p], x[p] >= mx, hx != x[p],
                   earcutPointInTriangle(
                       SIMD2(hy < my ? hx : qx, hy), SIMD2(mx, my), SIMD2(hy < my ? qx : hx, hy), SIMD2(x[p], y[p])
                   ) {
                    let tangent = abs(hy - y[p]) / (hx - x[p])
                    if locallyInside(p, hole),
                       tangent < tanMin
                        || (tangent == tanMin && (x[p] > x[m] || (x[p] == x[m] && sectorContainsSector(m, p)))) {
                        m = p
                        tanMin = tangent
                    }
                }
                p = following[p]
            } while p != stop
            return m
        }

        func earcutPointInTriangle(
            _ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>, _ p: SIMD2<Double>
        ) -> Bool {
            (c.x - p.x) * (a.y - p.y) >= (a.x - p.x) * (c.y - p.y)
                && (a.x - p.x) * (b.y - p.y) >= (b.x - p.x) * (a.y - p.y)
                && (b.x - p.x) * (c.y - p.y) >= (c.x - p.x) * (b.y - p.y)
        }

        func sectorContainsSector(_ m: Int, _ p: Int) -> Bool {
            area(previous[m], m, previous[p]) < 0 && area(following[p], m, following[m]) < 0
        }

        func leftmost(_ start: Int) -> Int {
            var p = start, best = start
            repeat {
                if x[p] < x[best] || (x[p] == x[best] && y[p] < y[best]) { best = p }
                p = following[p]
            } while p != start
            return best
        }

        func isValidDiagonal(_ a: Int, _ b: Int) -> Bool {
            guard source[following[a]] != source[b], source[previous[a]] != source[b], !intersectsPolygon(a, b) else {
                return false
            }
            let visible = locallyInside(a, b) && locallyInside(b, a) && middleInside(a, b)
                && (area(previous[a], a, previous[b]) != 0 || area(a, previous[b], b) != 0)
            let zeroLength = equals(a, b) && area(previous[a], a, following[a]) > 0
                && area(previous[b], b, following[b]) > 0
            return visible || zeroLength
        }

        func intersects(_ p1: Int, _ q1: Int, _ p2: Int, _ q2: Int) -> Bool {
            let o1 = sign(area(p1, q1, p2)), o2 = sign(area(p1, q1, q2))
            let o3 = sign(area(p2, q2, p1)), o4 = sign(area(p2, q2, q1))
            if o1 != o2 && o3 != o4 { return true }
            if o1 == 0 && onSegment(p1, p2, q1) { return true }
            if o2 == 0 && onSegment(p1, q2, q1) { return true }
            if o3 == 0 && onSegment(p2, p1, q2) { return true }
            if o4 == 0 && onSegment(p2, q1, q2) { return true }
            return false
        }

        func onSegment(_ p: Int, _ q: Int, _ r: Int) -> Bool {
            x[q] <= max(x[p], x[r]) && x[q] >= min(x[p], x[r]) && y[q] <= max(y[p], y[r]) && y[q] >= min(y[p], y[r])
        }

        func sign(_ value: Double) -> Int { value > 0 ? 1 : (value < 0 ? -1 : 0) }

        func intersectsPolygon(_ a: Int, _ b: Int) -> Bool {
            var p = a
            repeat {
                let next = following[p]
                if source[p] != source[a], source[next] != source[a], source[p] != source[b], source[next] != source[b],
                   intersects(p, next, a, b) {
                    return true
                }
                p = next
            } while p != a
            return false
        }

        func locallyInside(_ a: Int, _ b: Int) -> Bool {
            area(previous[a], a, following[a]) < 0
                ? area(a, b, following[a]) >= 0 && area(a, previous[a], b) >= 0
                : area(a, b, previous[a]) < 0 || area(a, following[a], b) < 0
        }

        func middleInside(_ a: Int, _ b: Int) -> Bool {
            var p = a
            var inside = false
            let px = (x[a] + x[b]) / 2, py = (y[a] + y[b]) / 2
            repeat {
                let next = following[p]
                if (y[p] > py) != (y[next] > py), y[next] != y[p],
                   px < (x[next] - x[p]) * (py - y[p]) / (y[next] - y[p]) + x[p] {
                    inside.toggle()
                }
                p = next
            } while p != a
            return inside
        }

        /// Links `a` to `b` with a two-way bridge and returns the duplicate of `b`.
        mutating func splitPolygon(_ a: Int, _ b: Int) -> Int {
            let a2 = makeNode(source[a], SIMD2(x[a], y[a]))
            let b2 = makeNode(source[b], SIMD2(x[b], y[b]))
            let an = following[a], bp = previous[b]
            following[a] = b
            previous[b] = a
            following[a2] = an
            previous[an] = a2
            following[b2] = a2
            previous[a2] = b2
            following[bp] = b2
            previous[b2] = bp
            return b2
        }
    }
}
