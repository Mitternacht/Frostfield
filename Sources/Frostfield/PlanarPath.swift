import CoreGraphics
import Foundation
import simd

/// A value-semantic list of drawing commands in the Y-up XY plane.
/// Coordinates and control points are absolute. Each move starts a new
/// subpath. An elliptical arc can also start an empty path.
public struct PlanarPath: Equatable, Sendable {
    public enum Command: Equatable, Sendable {
        case move(to: SIMD2<Float>)
        case line(to: SIMD2<Float>)
        case quadratic(to: SIMD2<Float>, control: SIMD2<Float>)
        case cubic(to: SIMD2<Float>, control1: SIMD2<Float>, control2: SIMD2<Float>)
        /// A uniform Catmull–Rom spline through the current point and the
        /// supplied points. The first and last controls are clamped.
        case spline(through: [SIMD2<Float>])
        /// Radians in the rotated ellipse's local frame. A positive sweep
        /// moves counterclockwise and a negative one clockwise, up to one
        /// full turn. A line connects a current point to the arc's start
        /// unless `connecting` is false, which begins a new subpath.
        case ellipticalArc(
            center: SIMD2<Float>, radii: SIMD2<Float>, startAngle: Double, sweepAngle: Double,
            rotation: Double = 0, connecting: Bool = true
        )
        /// Joins the current point to the subpath's start. A later drawing
        /// command begins a new subpath there unless a move changes it.
        case close
    }

    public var commands: [Command]

    public init(_ commands: [Command] = []) {
        self.commands = commands
    }

    /// Builds a Core Graphics path. Béziers remain curves, and elliptical
    /// arcs use chords with the given world-space deviation tolerance.
    public func makeCGPath(tolerance: Float = 0.001) throws -> CGPath {
        guard tolerance.isFinite, tolerance > 0 else { throw Failure.invalidTolerance }
        guard commands.count <= 16_384 else { throw Failure.complexity }
        var builder = Builder()
        for (index, command) in commands.enumerated() {
            try builder.append(command, index: index, tolerance: Double(tolerance))
        }
        return builder.path
    }

    /// Samples drawable subpaths in order. Adjacent repeated points are
    /// removed. Explicitly closed subpaths omit the closing copy of the
    /// first point, and their length and spaced samples include the closing
    /// segment. Bézier flattening uses up to 1,024 samples per segment.
    public func sampledSubpaths(tolerance: Float = 0.001) throws -> [Subpath] {
        let path = try makeCGPath(tolerance: tolerance)
        let flattened: [PlanarShape.FlattenedSubpath]
        do {
            flattened = try PlanarShape.flattenedSubpaths(
                path, tolerance: Double(tolerance), maximumPoints: Self.pointBudget
            )
        } catch is PlanarComplexityLimit {
            throw Failure.complexity
        }
        return flattened.compactMap { subpath in
            var points: [SIMD2<Float>] = []
            for sample in subpath.points {
                let point = SIMD2<Float>(sample)
                if points.last != point { points.append(point) }
            }
            if subpath.closed, points.count > 1, points.first == points.last { points.removeLast() }
            return points.count >= 2 ? Subpath(points: points, isClosed: subpath.closed) : nil
        }
    }

    /// Resolves the filled region using nonzero winding or even-odd parity.
    /// Fill implicitly closes each drawable subpath.
    public func shapes(fillRule: CGPathFillRule = .winding, tolerance: Float = 0.001) throws -> [PlanarShape] {
        let rings = try sampledSubpaths(tolerance: tolerance).map(\.points)
        do {
            return try PlanarShape.shapes(rings: rings, fillRule: fillRule, maximumEdges: Self.fillBudget)
        } catch is PlanarComplexityLimit {
            throw Failure.complexity
        }
    }

    public struct Subpath: Equatable, Sendable {
        public let points: [SIMD2<Float>]
        public let isClosed: Bool

        /// Approximate arc length of the sampled polyline.
        public var length: Double { arcLengths().last! }

        /// Evaluates the sampled polyline by normalized arc length.
        public func point(atLengthFraction fraction: Double) throws -> SIMD2<Float> {
            guard (0...1).contains(fraction) else { throw Failure.invalidFraction }
            return point(at: fraction, lengths: arcLengths())
        }

        /// Samples both endpoints at equal arc-length intervals. The last
        /// point repeats the first for an explicitly closed subpath.
        public func spacedPoints(segments: Int) throws -> [SIMD2<Float>] {
            guard (1...65_536).contains(segments) else { throw Failure.invalidResolution }
            let lengths = arcLengths()
            return (0...segments).map { point(at: Double($0) / Double(segments), lengths: lengths) }
        }

        private func arcLengths() -> [Double] {
            var lengths = [0.0]
            var previous = SIMD2<Double>(points[0])
            var total = 0.0
            let count = points.count + (isClosed ? 1 : 0)
            lengths.reserveCapacity(count)
            for index in 1..<count {
                let next = SIMD2<Double>(points[index % points.count])
                total += simd_distance(previous, next)
                lengths.append(total)
                previous = next
            }
            return lengths
        }

        private func point(at fraction: Double, lengths: [Double]) -> SIMD2<Float> {
            if fraction == 0 { return points[0] }
            if fraction == 1 { return isClosed ? points[0] : points[points.count - 1] }
            let distance = fraction * lengths[lengths.count - 1]
            var lower = 1, upper = lengths.count - 1
            while lower < upper {
                let middle = (lower + upper) / 2
                if lengths[middle] < distance { lower = middle + 1 } else { upper = middle }
            }
            let a = SIMD2<Double>(points[lower - 1]), b = SIMD2<Double>(points[lower % points.count])
            let span = lengths[lower] - lengths[lower - 1]
            let t = span > 0 ? (distance - lengths[lower - 1]) / span : 0
            return SIMD2<Float>(a + t * (b - a))
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidTolerance
        case invalidPoint(Int)
        case missingMove(Int)
        case invalidEllipse(Int)
        case invalidSpline(Int)
        case invalidFraction
        case invalidResolution
        case expectedSingleContour(Int)
        case complexity

        public var errorDescription: String? {
            switch self {
            case .invalidTolerance: "The path tolerance must be finite and positive."
            case .invalidPoint(let command): "Path command \(command) needs finite Float coordinates."
            case .missingMove(let command): "Path command \(command) needs an initial move or ellipse."
            case .invalidEllipse(let command): "Ellipse command \(command) needs positive radii, finite angles and a nonzero sweep of at most one turn."
            case .invalidSpline(let command): "Spline command \(command) needs at least one destination."
            case .invalidFraction: "The path distance fraction must lie in 0...1."
            case .invalidResolution: "Use 1 to 65,536 path sampling intervals."
            case .expectedSingleContour(let ring): "Path ring \(ring) needs exactly one drawable subpath."
            case .complexity: "The path exceeds the command, sampling or fill budget."
            }
        }
    }

    private static let pointBudget = 65_536
    fileprivate static let fillBudget = 16_384

    private struct Builder {
        let path = CGMutablePath()
        private var current: SIMD2<Double>?
        private var start: SIMD2<Double>?
        private var closed = false
        private var remaining = PlanarPath.pointBudget

        mutating func append(_ command: Command, index: Int, tolerance: Double) throws {
            switch command {
            case .move(let point):
                try move(checked(SIMD2<Double>(point), index))
            case .line(let point):
                try beginSegment(index)
                let end = try checked(SIMD2<Double>(point), index)
                try reserve()
                path.addLine(to: cg(end))
                current = end
            case .quadratic(let point, let control):
                try beginSegment(index)
                let end = try checked(SIMD2<Double>(point), index)
                let control = try checked(SIMD2<Double>(control), index)
                try reserve()
                path.addQuadCurve(to: cg(end), control: cg(control))
                current = end
            case .cubic(let point, let first, let second):
                try beginSegment(index)
                let end = try checked(SIMD2<Double>(point), index)
                let first = try checked(SIMD2<Double>(first), index)
                let second = try checked(SIMD2<Double>(second), index)
                try reserve()
                path.addCurve(to: cg(end), control1: cg(first), control2: cg(second))
                current = end
            case .spline(let points):
                let origin = try beginSegment(index)
                try spline(from: origin, through: points, index: index)
            case .ellipticalArc(let center, let radii, let start, let sweep, let rotation, let connecting):
                try ellipse(center: center, radii: radii, startAngle: start, sweep: sweep, rotation: rotation,
                            connecting: connecting, index: index, tolerance: tolerance)
            case .close:
                guard let start else { throw Failure.missingMove(index) }
                if closed { return }
                try reserve()
                path.closeSubpath()
                current = start
                closed = true
            }
        }

        private mutating func move(_ point: SIMD2<Double>) throws {
            try reserve()
            path.move(to: cg(point))
            current = point
            start = point
            closed = false
        }

        @discardableResult
        private mutating func beginSegment(_ index: Int) throws -> SIMD2<Double> {
            guard let current else { throw Failure.missingMove(index) }
            if closed { try move(current) }
            return current
        }

        private mutating func spline(from origin: SIMD2<Double>, through values: [SIMD2<Float>], index: Int) throws {
            guard !values.isEmpty else { throw Failure.invalidSpline(index) }
            try reserve(values.count)
            let points = try [origin] + values.map { try checked(SIMD2<Double>($0), index) }
            for edge in 0..<(points.count - 1) {
                let a = points[max(0, edge - 1)], b = points[edge]
                let c = points[edge + 1], d = points[min(edge + 2, points.count - 1)]
                let first = try checked(b + (c - a) / 6, index), second = try checked(c - (d - b) / 6, index)
                path.addCurve(to: cg(c), control1: cg(first), control2: cg(second))
            }
            current = points[points.count - 1]
        }

        private mutating func ellipse(
            center: SIMD2<Float>, radii: SIMD2<Float>, startAngle: Double, sweep: Double, rotation: Double,
            connecting: Bool, index: Int, tolerance: Double
        ) throws {
            let turn = 2 * Double.pi
            guard radii.x.isFinite, radii.y.isFinite, radii.x > 0, radii.y > 0,
                  startAngle.isFinite, rotation.isFinite, abs(sweep) > 0, abs(sweep) <= turn else {
                throw Failure.invalidEllipse(index)
            }
            let center = try checked(SIMD2<Double>(center), index)
            let radius = SIMD2<Double>(radii)
            let angle = rotation.remainder(dividingBy: turn)
            let start = startAngle.remainder(dividingBy: turn)
            let cosine = cos(angle), sine = sin(angle)
            func sample(_ theta: Double) -> SIMD2<Double> {
                let x = radius.x * cos(theta), y = radius.y * sin(theta)
                return center + SIMD2(cosine * x - sine * y, sine * x + cosine * y)
            }
            let first = try checked(sample(start), index)
            if current == nil || !connecting {
                try move(first)
            } else {
                let previous = try beginSegment(index)
                if previous != first {
                    try reserve()
                    path.addLine(to: cg(first))
                }
            }
            // The unit-circle sagitta is 2 sin²(step/4). The largest
            // ellipse radius bounds how much the affine map scales that error.
            let ratio = min(1, tolerance / (2 * max(radius.x, radius.y)))
            let step = min(.pi / 2, 4 * asin(sqrt(ratio)))
            let count = max(1, (abs(sweep) / step).rounded(.up))
            guard count <= Double(remaining) else { throw Failure.complexity }
            let segments = Int(count)
            try reserve(segments)
            for sampleIndex in 1...segments {
                let fraction = Double(sampleIndex) / Double(segments)
                let point = sampleIndex == segments && abs(sweep) == turn ? first : sample(start + fraction * sweep)
                let end = try checked(point, index)
                path.addLine(to: cg(end))
                current = end
            }
        }

        private mutating func reserve(_ count: Int = 1) throws {
            guard count <= remaining else { throw Failure.complexity }
            remaining -= count
        }

        private func checked(_ point: SIMD2<Double>, _ index: Int) throws -> SIMD2<Double> {
            guard Float(point.x).isFinite, Float(point.y).isFinite else { throw Failure.invalidPoint(index) }
            return point
        }

        private func cg(_ point: SIMD2<Double>) -> CGPoint { CGPoint(x: point.x, y: point.y) }
    }
}

extension PlanarShape {
    /// Creates one shape from an explicit outer path and hole paths. Each
    /// path must have one drawable subpath and closes implicitly for filling.
    public init(outerPath: PlanarPath, holePaths: [PlanarPath] = [], tolerance: Float = 0.001) throws {
        var remaining = PlanarPath.fillBudget
        let rings = try ([outerPath] + holePaths).enumerated().map { index, path in
            let subpaths = try path.sampledSubpaths(tolerance: tolerance)
            guard subpaths.count == 1 else { throw PlanarPath.Failure.expectedSingleContour(index) }
            let points = subpaths[0].points
            guard points.count <= remaining else { throw PlanarPath.Failure.complexity }
            remaining -= points.count
            return points
        }
        try self.init(outer: rings[0], holes: Array(rings.dropFirst()))
    }
}
