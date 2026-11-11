import simd

struct ProximityTriangle {
    struct Point {
        let position: SIMD3<Double>
        let barycentric: SIMD3<Double>
    }

    let a: SIMD3<Double>
    let b: SIMD3<Double>
    let c: SIMD3<Double>

    var normal: SIMD3<Double>? {
        let cross = simd_cross(b - a, c - a)
        let scale = simd_reduce_max(simd_abs(cross))
        return scale > 0 ? simd_normalize(cross / scale) : nil
    }

    func closestPoint(to point: SIMD3<Double>, normal: SIMD3<Double>?) -> Point {
        if let normal, let weights = projectionWeights(of: point, normal: normal) {
            let displacement = normal * simd_dot(point - a, normal)
            let projected = point - displacement
            // Keep tangent coordinates from the projection. Where a large
            // normal offset cancels a coordinate, reconstruct it from corners.
            let cancellation = simd_abs(projected) .< (simd_abs(point) * 0.5)
            let interpolated = weights.x * a + weights.y * b + weights.z * c
            return Point(position: projected.replacing(with: interpolated, where: cancellation), barycentric: weights)
        }

        // The closest feature is an edge or vertex. The same search treats
        // collinear and coincident corners as their closed segments or point.
        let ab = CollisionSegment(start: a, end: b)
        let bc = CollisionSegment(start: b, end: c)
        let ca = CollisionSegment(start: c, end: a)
        let (onAB, s) = ab.closestWitness(to: point)
        let (onBC, t) = bc.closestWitness(to: point)
        let (onCA, u) = ca.closestWitness(to: point)
        let abDistance = simd_length_squared(point - onAB)
        let bcDistance = simd_length_squared(point - onBC)
        let caDistance = simd_length_squared(point - onCA)
        if abDistance <= bcDistance, abDistance <= caDistance {
            return Point(position: onAB, barycentric: SIMD3(1 - s, s, 0))
        }
        if bcDistance <= caDistance {
            return Point(position: onBC, barycentric: SIMD3(0, 1 - t, t))
        }
        return Point(position: onCA, barycentric: SIMD3(u, 0, 1 - u))
    }

    /// The projected subtriangle areas divided by their sum. Orienting with a
    /// unit normal avoids squaring a cross product at extreme scene scales.
    func projectionWeights(of point: SIMD3<Double>, normal: SIMD3<Double>) -> SIMD3<Double>? {
        let weights = SIMD3(
            simd_dot(simd_cross(c - b, point - b), normal),
            simd_dot(simd_cross(a - c, point - c), normal),
            simd_dot(simd_cross(b - a, point - a), normal)
        )
        let total = weights.sum()
        guard all(weights .>= .zero), total > 0 else { return nil }
        return weights / total
    }
}

extension ProximityTriangle {
    struct Pair {
        let first: Point
        let second: Point
        var squaredDistance: Double { simd_length_squared(first.position - second.position) }
    }

    var bounds: CollisionBounds {
        CollisionBounds(minimum: simd_min(a, simd_min(b, c)), maximum: simd_max(a, simd_max(b, c)))
    }

    func closestPoints(to other: Self) -> Pair {
        let firstNormal = normal, secondNormal = other.normal
        let firstEdges = edges, secondEdges = other.edges

        // A noncoplanar intersection includes an edge that crosses the other
        // face, even when no vertices or edges touch one another.
        for index in 0..<3 {
            if let secondNormal,
               let crossing = other.crossing(firstEdges[index], normal: secondNormal) {
                return Pair(first: edgePoint(index, crossing.parameter, at: crossing.point.position), second: crossing.point)
            }
            if let firstNormal,
               let crossing = self.crossing(secondEdges[index], normal: firstNormal) {
                return Pair(first: crossing.point, second: other.edgePoint(index, crossing.parameter, at: crossing.point.position))
            }
        }

        // Otherwise a minimum is a vertex-to-face or edge-to-edge pair.
        // These include containment in a coplanar face and degenerate corners.
        var best = Pair(first: edgePoint(0, 0, at: a), second: other.closestPoint(to: a, normal: secondNormal))
        func consider(_ first: Point, _ second: Point) {
            let pair = Pair(first: first, second: second)
            if pair.squaredDistance < best.squaredDistance { best = pair }
        }
        for index in 0..<3 {
            let start = firstEdges[index].start, otherStart = secondEdges[index].start
            consider(edgePoint(index, 0, at: start), other.closestPoint(to: start, normal: secondNormal))
            consider(closestPoint(to: otherStart, normal: firstNormal), other.edgePoint(index, 0, at: otherStart))
            for secondIndex in 0..<3 {
                let edge = firstEdges[index], otherEdge = secondEdges[secondIndex]
                let parameters = edge.closestParameters(to: otherEdge)
                let first = parameters.intersection ?? edge.point(at: parameters.first)
                let second = parameters.intersection ?? otherEdge.point(at: parameters.second)
                consider(edgePoint(index, parameters.first, at: first), other.edgePoint(secondIndex, parameters.second, at: second))
            }
        }
        return best
    }

    private var edges: InlineArray<3, CollisionSegment> {
        [CollisionSegment(start: a, end: b), CollisionSegment(start: b, end: c), CollisionSegment(start: c, end: a)]
    }

    private func edgePoint(_ edge: Int, _ parameter: Double, at position: SIMD3<Double>) -> Point {
        var weights = SIMD3<Double>.zero
        weights[edge] = 1 - parameter
        weights[(edge + 1) % 3] = parameter
        return Point(position: position, barycentric: weights)
    }

    private func crossing(_ segment: CollisionSegment, normal: SIMD3<Double>) -> (point: Point, parameter: Double)? {
        let slope = simd_dot(segment.direction, normal)
        guard slope != 0 else { return nil }
        let parameter = simd_dot(a - segment.start, normal) / slope
        guard (0...1).contains(parameter) else { return nil }
        let point = segment.point(at: parameter)
        guard let weights = projectionWeights(of: point, normal: normal) else { return nil }
        return (Point(position: point, barycentric: weights), parameter)
    }
}
