import Foundation
import simd

/// A Euclidean control point and its positive rational weight.
public struct NURBSControlPoint: Equatable, Sendable {
    public let position: SIMD3<Float>
    public let weight: Float

    public init(_ position: SIMD3<Float>, weight: Float = 1) {
        self.position = position
        self.weight = weight
    }

    fileprivate func homogeneous() throws -> SIMD4<Double> {
        guard position.x.isFinite, position.y.isFinite, position.z.isFinite, weight.isFinite, weight > 0 else {
            throw NURBSFailure.controlPoint
        }
        let w = Double(weight)
        return SIMD4(SIMD3<Double>(position) * w, w)
    }
}

public extension Curve3D {
    /// Evaluates a rational B-spline on the active knot domain
    /// knots[degree]...knots[controlPoints.count]. An optional knot index range
    /// selects a subdomain. Tangents use the analytic rational derivative.
    static func nurbs(
        degree: Int, knots: [Double], controlPoints: [NURBSControlPoint],
        knotRange: ClosedRange<Int>? = nil
    ) throws -> Curve3D {
        let basis = try NURBSBasis(degree: degree, knots: knots, controlCount: controlPoints.count)
        let range = try basis.domain(for: knotRange)
        let points = try controlPoints.map { try $0.homogeneous() }
        return Curve3D(position: { parameter in
            let (value, span) = basis.coordinates(parameter, in: range)
            let weights = basis.weights(at: value, span: span, degree: degree)
            return try euclidean(nurbsBlend(points, from: span - degree, weights: weights))
        }, derivative: { parameter in
            let (value, span) = basis.coordinates(parameter, in: range)
            let weights = basis.weights(at: value, span: span, degree: degree)
            let first = span - degree
            let point = nurbsBlend(points, from: first, weights: weights)
            let direction = nurbsBlend(points, from: first, weights: basis.derivatives(at: value, span: span))
            return SIMD3(direction.x, direction.y, direction.z) - (try euclidean(point)) * direction.w
        })
    }
}

/// A tensor-product rational B-spline surface. The control net is indexed
/// as [v][u], with U varying across a row and V varying between rows.
public struct NURBSSurface: Sendable {
    private let uBasis: NURBSBasis
    private let vBasis: NURBSBasis
    private let points: [SIMD4<Double>]

    public init(
        uDegree: Int, vDegree: Int, uKnots: [Double], vKnots: [Double],
        controlPoints: [[NURBSControlPoint]]
    ) throws {
        guard let first = controlPoints.first, !first.isEmpty,
              controlPoints.count <= 262_144 / first.count,
              controlPoints.allSatisfy({ $0.count == first.count }) else { throw NURBSFailure.controlNet }
        uBasis = try NURBSBasis(degree: uDegree, knots: uKnots, controlCount: first.count)
        vBasis = try NURBSBasis(degree: vDegree, knots: vKnots, controlCount: controlPoints.count)
        points = try controlPoints.flatMap { row in try row.map { try $0.homogeneous() } }
    }

    public func point(u: Double, v: Double) throws -> SIMD3<Float> {
        guard (0...1).contains(u), (0...1).contains(v) else { throw NURBSFailure.parameter }
        let (uValue, uSpan) = uBasis.coordinates(u, in: uBasis.activeDomain)
        let (vValue, vSpan) = vBasis.coordinates(v, in: vBasis.activeDomain)
        let uWeights = uBasis.weights(at: uValue, span: uSpan, degree: uBasis.degree)
        let vWeights = vBasis.weights(at: vValue, span: vSpan, degree: vBasis.degree)
        let firstU = uSpan - uBasis.degree
        let firstV = vSpan - vBasis.degree
        var point = SIMD4<Double>.zero
        for (row, weight) in vWeights.enumerated() {
            let rowStart = (firstV + row) * uBasis.controlCount + firstU
            point += weight * nurbsBlend(points, from: rowStart, weights: uWeights)
        }
        return SIMD3<Float>(try euclidean(point))
    }

    /// Tessellates the patch with the regular surface builder. Singular
    /// samples such as a collapsed pole report its usual mesh error.
    public func mesh(uSegments: Int, vSegments: Int) throws -> MeshGeometry {
        try MeshGeometry.parametricSurface(uSegments: uSegments, vSegments: vSegments) { uv in
            try point(u: Double(uv.x), v: Double(uv.y))
        }
    }
}

public enum NURBSFailure: LocalizedError, Equatable {
    case degree
    case knots
    case controlPoint
    case controlNet
    case knotRange
    case parameter

    public var errorDescription: String? {
        switch self {
        case .degree: "Use degree 1 through 32 and between degree + 1 and 262,144 control points per axis."
        case .knots: "Provide a finite nondecreasing knot vector with valid length, domain and multiplicities."
        case .controlPoint: "NURBS control points need finite coordinates and a positive finite weight."
        case .controlNet: "Use a rectangular control net with at most 262,144 points."
        case .knotRange: "Select an increasing knot range inside the active domain."
        case .parameter: "Surface parameters must be in 0...1."
        }
    }
}

private struct NURBSBasis: Sendable {
    let degree: Int
    let knots: [Double]
    let controlCount: Int
    let activeDomain: ClosedRange<Double>

    init(degree: Int, knots: [Double], controlCount: Int) throws {
        guard (1...32).contains(degree), ((degree + 1)...262_144).contains(controlCount) else {
            throw NURBSFailure.degree
        }
        guard knots.count == controlCount + degree + 1, knots.allSatisfy(\.isFinite),
              knots[degree] < knots[controlCount], (knots[knots.count - 1] - knots[0]).isFinite else {
            throw NURBSFailure.knots
        }
        var repetitions = 1
        for index in 1..<knots.count {
            guard knots[index] >= knots[index - 1] else { throw NURBSFailure.knots }
            repetitions = knots[index] == knots[index - 1] ? repetitions + 1 : 1
            guard repetitions <= degree + 1 else { throw NURBSFailure.knots }
        }
        self.degree = degree
        self.knots = knots
        self.controlCount = controlCount
        activeDomain = knots[degree]...knots[controlCount]
    }

    func domain(for range: ClosedRange<Int>?) throws -> ClosedRange<Double> {
        guard let range else { return activeDomain }
        guard knots.indices.contains(range.lowerBound), knots.indices.contains(range.upperBound) else {
            throw NURBSFailure.knotRange
        }
        let first = knots[range.lowerBound], last = knots[range.upperBound]
        guard first < last, activeDomain.contains(first), activeDomain.contains(last) else { throw NURBSFailure.knotRange }
        return first...last
    }

    func coordinates(_ parameter: Double, in domain: ClosedRange<Double>) -> (Double, Int) {
        let value = parameter == 1 ? domain.upperBound
            : min(domain.upperBound, domain.lowerBound + parameter * (domain.upperBound - domain.lowerBound))
        // Use the right-hand span at internal knots and the left-hand span
        // at the active upper endpoint, including repeated nonclamped knots.
        var low = degree
        var high = controlCount
        while low < high {
            let middle = (low + high) / 2
            if knots[middle] < value || (knots[middle] == value && value < activeDomain.upperBound) {
                low = middle + 1
            } else {
                high = middle
            }
        }
        return (value, max(degree, low - 1))
    }

    func weights(at value: Double, span: Int, degree: Int) -> [Double] {
        var weights = [1.0] + Array(repeating: 0.0, count: degree)
        guard degree > 0 else { return weights }
        for level in 1...degree {
            var carried = 0.0
            for i in 0..<level {
                let left = value - knots[span + i + 1 - level]
                let right = knots[span + i + 1] - value
                let span = left + right
                let toLeft = span > 0 ? weights[i] * (right / span) : 0
                let toRight = span > 0 ? weights[i] * (left / span) : 0
                weights[i] = carried + toLeft
                carried = toRight
            }
            weights[level] = carried
        }
        return weights
    }

    func derivatives(at value: Double, span: Int) -> [Double] {
        let lower = weights(at: value, span: span, degree: degree - 1)
        // Scale the derivative by the active interval. The positive factor
        // cancels in the unit tangent and avoids division by a tiny knot step.
        let interval = knots[span + 1] - knots[span]
        return (0...degree).map { offset in
            let i = span - degree + offset
            let a = knots[i + degree] - knots[i]
            let b = knots[i + degree + 1] - knots[i + 1]
            let left = offset > 0 && a > 0 ? lower[offset - 1] * (interval / a) : 0
            let right = offset < degree && b > 0 ? lower[offset] * (interval / b) : 0
            return Double(degree) * (left - right)
        }
    }
}

private func nurbsBlend(_ points: [SIMD4<Double>], from start: Int, weights: [Double]) -> SIMD4<Double> {
    weights.enumerated().reduce(into: .zero) { sum, entry in
        sum += entry.element * points[start + entry.offset]
    }
}

private func euclidean(_ homogeneous: SIMD4<Double>) throws -> SIMD3<Double> {
    guard homogeneous.w.isFinite, homogeneous.w > 0 else { throw Curve3D.Failure.unrepresentablePoint }
    let position = SIMD3(homogeneous.x, homogeneous.y, homogeneous.z) / homogeneous.w
    let rounded = SIMD3<Float>(position)
    guard rounded.x.isFinite, rounded.y.isFinite, rounded.z.isFinite else { throw Curve3D.Failure.unrepresentablePoint }
    return position
}
