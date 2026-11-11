import Foundation
import simd

/// A three-dimensional curve over the normalized parameter range 0...1.
/// Evaluation and distance accumulation use Double. Samples must fit Float
/// for use with procedural meshes, animation and the scene graph.
public struct Curve3D: Sendable {
    public enum Spacing: Equatable, Sendable {
        case parameter
        /// Approximates distance using this many uniform parameter intervals.
        case arcLength(subdivisions: Int = 256)
    }

    private let position: @Sendable (Double) throws -> SIMD3<Double>
    private let derivative: (@Sendable (Double) throws -> SIMD3<Double>)?

    /// The function should evaluate deterministically throughout 0...1.
    public init(position: @escaping @Sendable (Double) throws -> SIMD3<Double>) {
        self.init(position: position, derivative: nil)
    }

    init(
        position: @escaping @Sendable (Double) throws -> SIMD3<Double>,
        derivative: (@Sendable (Double) throws -> SIMD3<Double>)?
    ) {
        self.position = position
        self.derivative = derivative
    }

    public func point(at parameter: Double) throws -> SIMD3<Float> {
        SIMD3<Float>(try checkedPosition(parameter))
    }

    /// Returns a unit tangent. Curves without an analytic derivative use
    /// a finite difference, with one-sided samples at the endpoints.
    public func tangent(at parameter: Double) throws -> SIMD3<Float> {
        SIMD3<Float>(try unitTangent(at: parameter))
    }

    private func unitTangent(at parameter: Double) throws -> SIMD3<Double> {
        guard (0...1).contains(parameter) else { throw Failure.parameter }
        let direction: SIMD3<Double>
        if let derivative {
            direction = try derivative(parameter)
        } else {
            let delta = 1e-5
            direction = try checkedPosition(min(1, parameter + delta)) - checkedPosition(max(0, parameter - delta))
        }
        let scale = max(abs(direction.x), abs(direction.y), abs(direction.z))
        guard scale.isFinite, scale > 0 else { throw Failure.undefinedTangent }
        return simd_normalize(direction / scale)
    }

    /// Samples both endpoints and the requested number of intervals.
    public func points(segments: Int, spacing: Spacing = .parameter) throws -> [SIMD3<Float>] {
        guard (1...65_536).contains(segments) else { throw Failure.resolution }
        let table: ArcTable?
        switch spacing {
        case .parameter: table = nil
        case .arcLength(let subdivisions): table = try arcTable(subdivisions)
        }
        return try (0...segments).map { index in
            let fraction = Double(index) / Double(segments)
            return try point(at: table?.parameter(at: fraction) ?? fraction)
        }
    }

    public func length(subdivisions: Int = 256) throws -> Double {
        try arcTable(subdivisions).total
    }

    /// Evaluates a fraction of the approximate arc length, including endpoints.
    public func point(atLengthFraction fraction: Double, subdivisions: Int = 256) throws -> SIMD3<Float> {
        guard (0...1).contains(fraction) else { throw Failure.parameter }
        return try point(at: arcTable(subdivisions).parameter(at: fraction))
    }

    func arcSamples(
        segments: Int, subdivisions: Int
    ) throws -> (samples: [(point: SIMD3<Double>, tangent: SIMD3<Double>)], length: Double) {
        let table = try arcTable(subdivisions)
        let samples = try (0...segments).map { index in
            let parameter = table.parameter(at: Double(index) / Double(segments))
            return (try checkedPosition(parameter), try unitTangent(at: parameter))
        }
        return (samples, table.total)
    }

    private func checkedPosition(_ parameter: Double) throws -> SIMD3<Double> {
        guard (0...1).contains(parameter) else { throw Failure.parameter }
        let value = try position(parameter)
        let sample = SIMD3<Float>(value)
        guard sample.x.isFinite, sample.y.isFinite, sample.z.isFinite else { throw Failure.unrepresentablePoint }
        return value
    }

    private func arcTable(_ subdivisions: Int) throws -> ArcTable {
        guard (1...65_536).contains(subdivisions) else { throw Failure.resolution }
        var lengths = [0.0]
        lengths.reserveCapacity(subdivisions + 1)
        var previous = try checkedPosition(0)
        var total = 0.0
        for index in 1...subdivisions {
            let current = try checkedPosition(Double(index) / Double(subdivisions))
            total += simd_distance(current, previous)
            lengths.append(total)
            previous = current
        }
        return ArcTable(lengths: lengths, total: total)
    }

    private struct ArcTable {
        let lengths: [Double]
        let total: Double

        func parameter(at fraction: Double) -> Double {
            guard fraction > 0, fraction < 1, total > 0 else { return fraction }
            let distance = total * fraction
            var low = 1
            var high = lengths.count - 1
            while low < high {
                let middle = (low + high) / 2
                if lengths[middle] < distance { low = middle + 1 } else { high = middle }
            }
            let first = lengths[low - 1]
            let span = lengths[low] - first
            let offset = span > 0 ? (distance - first) / span : 0
            return (Double(low - 1) + offset) / Double(lengths.count - 1)
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case parameter
        case resolution
        case unrepresentablePoint
        case undefinedTangent
        case controlPoints

        public var errorDescription: String? {
            switch self {
            case .parameter: "Curve parameters and distance fractions must be in 0...1."
            case .resolution: "Use 1 to 65,536 sampling intervals."
            case .unrepresentablePoint: "The evaluated point exceeds the finite Float range."
            case .undefinedTangent: "The curve has no unit tangent at this parameter."
            case .controlPoints: "Use finite control points and valid curve settings."
            }
        }
    }
}
