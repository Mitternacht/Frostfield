import Foundation
import simd

/// A closed sphere swept along a line segment, like Three.js `Capsule`.
public struct Capsule: Equatable, Sendable {
    public let start: SIMD3<Float>
    public let end: SIMD3<Float>
    public let radius: Float

    /// The endpoints must be finite and the radius finite and nonnegative.
    /// Equal endpoints produce a sphere; a zero radius produces a segment.
    public init(start: SIMD3<Float>, end: SIMD3<Float>, radius: Float) throws {
        guard boundsFinite(start), boundsFinite(end) else { throw Failure.invalidEndpoint }
        guard radius.isFinite, radius >= 0 else { throw Failure.invalidRadius }
        self.start = start
        self.end = end
        self.radius = radius
    }

    public var center: SIMD3<Float> { SIMD3<Float>((boundsDouble(start) + boundsDouble(end)) * 0.5) }

    public func translated(by offset: SIMD3<Float>) throws -> Capsule {
        try Capsule(start: start + offset, end: end + offset, radius: radius)
    }

    /// The axis-aligned enclosure, rounded outward to Float.
    public func axisAlignedBounds() throws -> AxisAlignedBox {
        let bounds = exactBounds
        return try AxisAlignedBox(minimum: boundsFloatBelow(bounds.minimum), maximum: boundsFloatAbove(bounds.maximum))
    }

    public func contains(_ point: SIMD3<Float>) -> Bool {
        let point = boundsDouble(point)
        return simd_length_squared(point - spine.closestPoint(to: point)) <= radiusSquared
    }

    public func intersects(_ sphere: BoundingSphere) -> Bool {
        let point = boundsDouble(sphere.center)
        let sum = Double(radius) + Double(sphere.radius)
        return simd_length_squared(point - spine.closestPoint(to: point)) <= sum * sum
    }

    public func intersects(_ other: Capsule) -> Bool {
        let closest = spine.closestPoints(to: other.spine)
        let sum = Double(radius) + Double(other.radius)
        return simd_length_squared(closest.0 - closest.1) <= sum * sum
    }

    /// Tests the full segment-to-box distance, including rounded capsule caps.
    /// Touching counts as overlap.
    public func intersects(_ box: AxisAlignedBox) -> Bool {
        spine.squaredDistance(minimum: boundsDouble(box.minimum), maximum: boundsDouble(box.maximum)) <= radiusSquared
    }

    public func intersects(_ box: OrientedBoundingBox) -> Bool {
        let inverse = box.basis.transpose
        let origin = boundsDouble(box.center)
        let extent = boundsDouble(box.halfSize)
        let local = CollisionSegment(start: inverse * (spine.start - origin), end: inverse * (spine.end - origin))
        return local.squaredDistance(minimum: -extent, maximum: extent) <= radiusSquared
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidEndpoint
        case invalidRadius

        public var errorDescription: String? {
            switch self {
            case .invalidEndpoint: "A capsule needs finite endpoints."
            case .invalidRadius: "A capsule's radius must be finite and nonnegative."
            }
        }
    }

    var spine: CollisionSegment { CollisionSegment(start: boundsDouble(start), end: boundsDouble(end)) }
    var radiusSquared: Double { Double(radius) * Double(radius) }

    var exactBounds: CollisionBounds {
        let spine = spine
        let expansion = SIMD3<Double>(repeating: Double(radius))
        return CollisionBounds(
            minimum: simd_min(spine.start, spine.end) - expansion,
            maximum: simd_max(spine.start, spine.end) + expansion
        )
    }
}

struct CollisionSegment {
    let start: SIMD3<Double>
    let end: SIMD3<Double>
    var direction: SIMD3<Double> { end - start }

    func closestParameter(to point: SIMD3<Double>) -> Double {
        let step = direction
        let lengthSquared = simd_length_squared(step)
        guard lengthSquared > 0 else { return 0 }
        return min(max(simd_dot(point - start, step) / lengthSquared, 0), 1)
    }

    func point(at parameter: Double) -> SIMD3<Double> {
        if parameter == 0 { return start }
        if parameter == 1 { return end }
        return start + parameter * direction
    }

    func closestWitness(to point: SIMD3<Double>) -> (position: SIMD3<Double>, parameter: Double) {
        let parameter = closestParameter(to: point)
        let position = contains(point, at: parameter) ? point : self.point(at: parameter)
        return (position, parameter)
    }

    func closestPoint(to point: SIMD3<Double>) -> SIMD3<Double> {
        closestWitness(to: point).position
    }

    private func contains(_ point: SIMD3<Double>, at closestParameter: Double) -> Bool {
        if closestParameter == 0 { return point == start }
        if closestParameter == 1 { return point == end }
        return simd_cross(point - start, direction) == .zero
    }

    /// Minimizes squared distance over the square of the two segment parameters.
    /// The minimum is at a stationary point or on one of the four boundaries.
    func closestParameters(
        to other: CollisionSegment
    ) -> (first: Double, second: Double, intersection: SIMD3<Double>?) {
        let u = direction, v = other.direction
        let cross = simd_cross(u, v)
        let scale = simd_reduce_max(simd_abs(cross))
        if scale > 0 {
            let normal = cross / scale
            let denominator = simd_dot(cross, normal)
            let offset = other.start - start
            let s = simd_dot(simd_cross(offset, v), normal) / denominator
            let t = simd_dot(simd_cross(offset, u), normal) / denominator
            if (0...1).contains(s), (0...1).contains(t) {
                let intersection = simd_dot(offset, normal) == 0 ? point(at: s) : nil
                return (s, t, intersection)
            }
        }
        let candidates: InlineArray<4, (Double, Double)> = [
            (0, other.closestParameter(to: start)), (1, other.closestParameter(to: end)),
            (closestParameter(to: other.start), 0), (closestParameter(to: other.end), 1),
        ]
        func squaredDistance(_ candidate: (Double, Double)) -> Double {
            simd_length_squared(point(at: candidate.0) - other.point(at: candidate.1))
        }
        // A collinear endpoint is a shared point even when computing its
        // position again from a rounded interior parameter would move it.
        for index in 0..<4 {
            let (s, t) = candidates[index]
            if index < 2 {
                let endpoint = point(at: s)
                if other.contains(endpoint, at: t) { return (s, t, endpoint) }
            } else {
                let endpoint = other.point(at: t)
                if contains(endpoint, at: s) { return (s, t, endpoint) }
            }
        }
        var best = candidates[0]
        for index in 1..<4 where squaredDistance(candidates[index]) < squaredDistance(best) { best = candidates[index] }
        return (best.0, best.1, nil)
    }

    func closestPoints(to other: CollisionSegment) -> (SIMD3<Double>, SIMD3<Double>) {
        let parameters = closestParameters(to: other)
        if let intersection = parameters.intersection { return (intersection, intersection) }
        return (point(at: parameters.first), other.point(at: parameters.second))
    }

    /// Each interval between box-plane crossings has a fixed set of clamped
    /// coordinates, so squared distance there is a quadratic in the parameter.
    func squaredDistance(minimum: SIMD3<Double>, maximum: SIMD3<Double>) -> Double {
        let step = direction
        var breaks: [Double] = [0, 1]
        for axis in 0..<3 where step[axis] != 0 {
            for boundary in [minimum[axis], maximum[axis]] {
                let t = (boundary - start[axis]) / step[axis]
                if t > 0, t < 1 { breaks.append(t) }
            }
        }
        breaks.sort()
        var best = Double.infinity
        for (first, last) in zip(breaks, breaks.dropFirst()) {
            let middle = start + (first + last) * 0.5 * step
            if all(middle .>= minimum), all(middle .<= maximum) { return 0 }
            var slope = 0.0
            var curvature = 0.0
            for axis in 0..<3 {
                if middle[axis] < minimum[axis] {
                    slope += step[axis] * (start[axis] - minimum[axis])
                } else if middle[axis] > maximum[axis] {
                    slope += step[axis] * (start[axis] - maximum[axis])
                } else {
                    continue
                }
                curvature += step[axis] * step[axis]
            }
            let t = curvature > 0 ? min(max(-slope / curvature, first), last) : first
            let point = start + t * step
            let squared = simd_length_squared(point - simd_clamp(point, minimum, maximum))
            best = min(best, squared)
        }
        return best
    }
}
