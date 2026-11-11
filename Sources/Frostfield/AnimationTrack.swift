import Foundation
import simd

/// Keyframes for one typed property, like a Three.js `KeyframeTrack` bound to a
/// `PropertyBinding` path. Combine tracks into an `AnimationClip` and play it with an
/// `AnimationMixer`, or sample a track directly with `value(at:)`.
///
/// Times are seconds, finite, nonnegative and strictly increasing. Outside its key
/// range a track holds its first or last value.
public struct AnimationTrack {
    public enum Interpolation: String, Equatable, Sendable, Codable {
        /// Holds each key until the next one, like Three.js `InterpolateDiscrete` and glTF `STEP`.
        case step
        /// Componentwise linear interpolation. Quaternions use spherical linear
        /// interpolation along the shortest arc, like `QuaternionLinearInterpolant`.
        case linear
        /// Three.js `InterpolateSmooth` (`CubicInterpolant`) with its default
        /// zero-curvature endings: a Catmull-Rom cubic through every key whose
        /// tangents weigh neighboring keys by their spacing. Not for quaternions.
        case smooth
        /// glTF `CUBICSPLINE`: each key stores an in-tangent, a value and an
        /// out-tangent, with tangents in value units per second. Quaternions are
        /// normalized after interpolation.
        case cubicSpline
    }

    public let property: AnimationProperty
    public let interpolation: Interpolation
    public let times: [Float]
    /// One value per key, or for `cubicSpline` an in-tangent, value and out-tangent per key.
    public let values: [AnimationValue]
    public let valueKind: AnimationValue.Kind

    /// Continuous values flattened, `stride` components per value.
    private let components: [Double]
    private let stride: Int

    /// Validates the keys against each other and the property: every value has one
    /// kind that the property accepts, values are finite, quaternion keys are unit
    /// length within 0.001 and are normalized, booleans and strings use `step`, and
    /// quaternions do not use `smooth`.
    public init(
        property: AnimationProperty, times: [Float], values: [AnimationValue], interpolation: Interpolation = .linear
    ) throws {
        try self.init(property: property, times: times, values: values, interpolation: interpolation, unitKeys: true)
    }

    /// `unitKeys: false` keeps cubic quaternion keys at their length, which an exact
    /// split of a cubic segment needs; sampling normalizes them.
    init(
        property: AnimationProperty, times: [Float], values: [AnimationValue], interpolation: Interpolation,
        unitKeys: Bool, normalizeQuaternionKeys: Bool = true
    ) throws {
        if case .uniform(_, let name) = property, property.valueKind == nil { throw Failure.unknownUniform(name) }
        guard !times.isEmpty else { throw Failure.emptyTrack(property.description) }
        for (index, time) in times.enumerated() {
            guard time.isFinite, time >= 0, index == 0 || time > times[index - 1] else {
                throw Failure.invalidKeyTime(property.description, key: index)
            }
        }
        let perKey = interpolation == .cubicSpline ? 3 : 1
        guard values.count == times.count * perKey else {
            throw Failure.valueCount(property.description, expected: times.count * perKey, actual: values.count)
        }
        let kind = values[0].kind
        guard values.allSatisfy({ $0.kind == kind }) else { throw Failure.mixedKinds(property.description) }
        if let expected = property.valueKind {
            guard expected.accepts(kind) else { throw Failure.incompatibleKind(property.description, kind) }
        } else if case .node(_, .morphWeights) = property {
            guard case .numbers(let count) = kind, count > 0 else { throw Failure.incompatibleKind(property.description, kind) }
        }
        switch interpolation {
        case .step: break
        case .linear, .cubicSpline:
            guard kind.isContinuous else { throw Failure.unsupportedInterpolation(property.description, interpolation) }
        case .smooth:
            guard kind.isContinuous, kind != .quaternion else {
                throw Failure.unsupportedInterpolation(property.description, interpolation)
            }
        }

        var normalized = values
        if kind == .quaternion, unitKeys || interpolation != .cubicSpline {
            for key in Swift.stride(from: perKey == 3 ? 1 : 0, to: values.count, by: perKey) {
                guard let rotation = values[key].quaternionValue else { continue }
                let length = simd_length(SIMD4<Double>(rotation.vector))
                guard length.isFinite, abs(length - 1) <= 0.001 else {
                    throw Failure.nonunitQuaternion(property.description, key: key / perKey)
                }
                if normalizeQuaternionKeys {
                    normalized[key] = .quaternion(simd_quatf(vector: SIMD4<Float>(SIMD4<Double>(rotation.vector) / length)))
                }
            }
        }
        guard normalized.allSatisfy(\.isFinite) else { throw Failure.nonfiniteValue(property.description) }

        self.init(property: property, times: times, validatedValues: normalized, interpolation: interpolation)
    }

    private init(
        property: AnimationProperty, times: [Float], validatedValues: [AnimationValue], interpolation: Interpolation
    ) {
        let kind = validatedValues[0].kind
        self.property = property
        self.interpolation = interpolation
        self.times = times
        values = validatedValues
        valueKind = kind
        stride = kind.componentCount
        components = kind.isContinuous ? validatedValues.flatMap { $0.components ?? [] } : []
    }

    /// A track of numbers, like Three.js `NumberKeyframeTrack`.
    public static func number(
        _ property: AnimationProperty, times: [Float], values: [Float], interpolation: Interpolation = .linear
    ) throws -> AnimationTrack {
        try AnimationTrack(property: property, times: times, values: values.map { .number($0) }, interpolation: interpolation)
    }

    /// A track of three-component vectors, like Three.js `VectorKeyframeTrack`.
    public static func vector(
        _ property: AnimationProperty, times: [Float], values: [SIMD3<Float>], interpolation: Interpolation = .linear
    ) throws -> AnimationTrack {
        try AnimationTrack(property: property, times: times, values: values.map { .vector3($0) }, interpolation: interpolation)
    }

    /// A track of unit quaternions, like Three.js `QuaternionKeyframeTrack`.
    public static func quaternion(
        _ property: AnimationProperty, times: [Float], values: [simd_quatf], interpolation: Interpolation = .linear
    ) throws -> AnimationTrack {
        try AnimationTrack(property: property, times: times, values: values.map { .quaternion($0) }, interpolation: interpolation)
    }

    /// A track of linear RGB colors, like Three.js `ColorKeyframeTrack`.
    public static func color(
        _ property: AnimationProperty, times: [Float], values: [SIMD3<Float>], interpolation: Interpolation = .linear
    ) throws -> AnimationTrack {
        try AnimationTrack(property: property, times: times, values: values.map { .color($0) }, interpolation: interpolation)
    }

    /// A stepped track of booleans, like Three.js `BooleanKeyframeTrack`.
    public static func boolean(_ property: AnimationProperty, times: [Float], values: [Bool]) throws -> AnimationTrack {
        try AnimationTrack(property: property, times: times, values: values.map { .boolean($0) }, interpolation: .step)
    }

    /// A stepped track of strings, like Three.js `StringKeyframeTrack`.
    public static func string(_ property: AnimationProperty, times: [Float], values: [String]) throws -> AnimationTrack {
        try AnimationTrack(property: property, times: times, values: values.map { .string($0) }, interpolation: .step)
    }

    /// The last key time in seconds.
    public var endTime: Float { times[times.count - 1] }

    /// Removes interior keys of flat runs in step and linear tracks. The run's
    /// endpoints and the track's time range stay intact. Spline keys are preserved.
    public func optimized() -> Self {
        guard interpolation == .step || interpolation == .linear,
              let keys = retainedFlatRunKeys(times.count, matches: { values[$0] == values[$1] }) else { return self }
        return Self(property: property, times: keys.map { times[$0] }, validatedValues: keys.map { values[$0] },
                    interpolation: interpolation)
    }

    /// The value at a time in seconds, holding the first or last key outside the
    /// key range. Throws `Failure.invalidSampleTime` for a nonfinite time.
    public func value(at time: Float) throws -> AnimationValue {
        guard time.isFinite else { throw Failure.invalidSampleTime }
        return try sample(at: Double(time))
    }

    // MARK: - Sampling

    private var perKey: Int { interpolation == .cubicSpline ? 3 : 1 }
    private var valueOffset: Int { interpolation == .cubicSpline ? 1 : 0 }

    /// The key index and fraction for a time: `(index, nil)` on a key, or the
    /// segment from `index` to `index + 1` with the fraction across it.
    private func segment(at time: Double) -> (key: Int, fraction: Double?) {
        let last = times.count - 1
        if time <= Double(times[0]) { return (0, nil) }
        if time >= Double(times[last]) { return (last, nil) }
        var lower = 1
        var upper = last
        while lower < upper {
            let middle = lower + (upper - lower) / 2
            if Double(times[middle]) < time { lower = middle + 1 } else { upper = middle }
        }
        if Double(times[lower]) == time { return (lower, nil) }
        let start = Double(times[lower - 1])
        return (lower - 1, (time - start) / (Double(times[lower]) - start))
    }

    private func key(_ index: Int, element: Int = -1) -> ArraySlice<Double> {
        let value = index * perKey + (element < 0 ? valueOffset : element)
        return components[(value * stride)..<((value + 1) * stride)]
    }

    func sample(at time: Double) throws -> AnimationValue {
        let (key, fraction) = segment(at: time)
        guard let fraction else {
            let value = values[key * perKey + valueOffset]
            if case .quaternion(let rotation) = value { return .quaternion(simd_normalize(rotation)) }
            return value
        }
        if !valueKind.isContinuous || interpolation == .step { return values[key * perKey + valueOffset] }

        var result: [Double]
        switch interpolation {
        case .step:
            return values[key * perKey + valueOffset]
        case .linear:
            if valueKind == .quaternion {
                let a = simd_quatd(vector: SIMD4(Array(self.key(key))))
                let b = simd_quatd(vector: SIMD4(Array(self.key(key + 1))))
                let rotation = simd_normalize(simd_slerp(a, b, fraction).vector)
                return .quaternion(simd_quatf(vector: SIMD4<Float>(rotation)))
            }
            result = zip(self.key(key), self.key(key + 1)).map { $0 * (1 - fraction) + $1 * fraction }
        case .smooth:
            result = smoothSample(segmentStart: key, fraction: fraction)
        case .cubicSpline:
            let span = Double(times[key + 1]) - Double(times[key])
            result = Self.hermite(
                start: self.key(key), outgoing: self.key(key, element: 2), end: self.key(key + 1),
                incoming: self.key(key + 1, element: 0), span: span, fraction: fraction
            )
        }
        if valueKind == .quaternion {
            let length = simd_length(SIMD4(result))
            guard length.isFinite, length > 0 else { throw Failure.nonfiniteValue(property.description) }
            result = result.map { $0 / length }
        }
        let value = AnimationValue.continuous(valueKind, result)
        guard value.isFinite else { throw Failure.nonfiniteValue(property.description) }
        return value
    }

    /// The cubic Hermite value at a fraction of a segment with tangents per second.
    static func hermite(
        start: ArraySlice<Double>, outgoing: ArraySlice<Double>, end: ArraySlice<Double>, incoming: ArraySlice<Double>,
        span: Double, fraction t: Double
    ) -> [Double] {
        let t2 = t * t
        let t3 = t2 * t
        let h00 = 2 * t3 - 3 * t2 + 1
        let h10 = (t3 - 2 * t2 + t) * span
        let h01 = -2 * t3 + 3 * t2
        let h11 = (t3 - t2) * span
        return (0..<start.count).map { index in
            h00 * start[start.startIndex + index] + h10 * outgoing[outgoing.startIndex + index]
                + h01 * end[end.startIndex + index] + h11 * incoming[incoming.startIndex + index]
        }
    }

    /// The time derivative of the cubic Hermite segment at a fraction.
    static func hermiteDerivative(
        start: ArraySlice<Double>, outgoing: ArraySlice<Double>, end: ArraySlice<Double>, incoming: ArraySlice<Double>,
        span: Double, fraction t: Double
    ) -> [Double] {
        let t2 = t * t
        let d00 = (6 * t2 - 6 * t) / span
        let d10 = 3 * t2 - 4 * t + 1
        let d01 = (-6 * t2 + 6 * t) / span
        let d11 = 3 * t2 - 2 * t
        return (0..<start.count).map { index in
            d00 * start[start.startIndex + index] + d10 * outgoing[outgoing.startIndex + index]
                + d01 * end[end.startIndex + index] + d11 * incoming[incoming.startIndex + index]
        }
    }

    /// Three.js `CubicInterpolant` neighbors and weights for the segment ending at key `i1`.
    private func smoothWeights(endKey i1: Int) -> (previous: Int, next: Int, weightPrevious: Double, weightNext: Double) {
        let t0 = Double(times[i1 - 1])
        let t1 = Double(times[i1])
        var previous = i1 - 2
        var next = i1 + 1
        var tPrevious: Double
        var tNext: Double
        if previous >= 0 {
            tPrevious = Double(times[previous])
        } else {
            // Zero-curvature ending, as Three.js writes it.
            previous = i1
            tPrevious = t1
        }
        if next < times.count {
            tNext = Double(times[next])
        } else {
            next = i1 - 1
            tNext = t0
        }
        let half = (t1 - t0) * 0.5
        return (previous, next, half / (t0 - tPrevious), half / (tNext - t1))
    }

    private func smoothSample(segmentStart: Int, fraction p: Double) -> [Double] {
        let i1 = segmentStart + 1
        let (previous, next, wP, wN) = smoothWeights(endKey: i1)
        let pp = p * p
        let ppp = pp * p
        let sP = -wP * ppp + 2 * wP * pp - wP * p
        let s0 = (1 + wP) * ppp + (-1.5 - 2 * wP) * pp + (-0.5 + wP) * p + 1
        let s1 = (-1 - wN) * ppp + (1.5 + wN) * pp + 0.5 * p
        let sN = wN * ppp - wN * pp
        let vP = key(previous)
        let v0 = key(i1 - 1)
        let v1 = key(i1)
        let vN = key(next)
        return (0..<stride).map { index in
            sP * vP[vP.startIndex + index] + s0 * v0[v0.startIndex + index]
                + s1 * v1[v1.startIndex + index] + sN * vN[vN.startIndex + index]
        }
    }

    // MARK: - Transformations

    /// The same curve as a glTF cubic spline. A `smooth` track converts exactly:
    /// each Three.js `CubicInterpolant` segment is a cubic, so its end values and
    /// one-sided time derivatives define it. Linear quaternion and step tracks are
    /// returned unchanged.
    func cubicSplineEquivalent() throws -> AnimationTrack {
        switch interpolation {
        case .cubicSpline, .step: return self
        case .linear where valueKind == .quaternion: return self
        case .linear, .smooth: break
        }
        let zero = [Double](repeating: 0, count: stride)
        var incoming = Array(repeating: zero, count: times.count)
        var outgoing = Array(repeating: zero, count: times.count)
        for segment in 0..<(times.count - 1) {
            let span = Double(times[segment + 1]) - Double(times[segment])
            let v0 = key(segment)
            let v1 = key(segment + 1)
            if interpolation == .linear {
                let slope = (0..<stride).map { (v1[v1.startIndex + $0] - v0[v0.startIndex + $0]) / span }
                outgoing[segment] = slope
                incoming[segment + 1] = slope
                continue
            }
            let (previous, next, wP, wN) = smoothWeights(endKey: segment + 1)
            let vP = key(previous)
            let vN = key(next)
            outgoing[segment] = (0..<stride).map { index in
                (-wP * vP[vP.startIndex + index] + (wP - 0.5) * v0[v0.startIndex + index] + 0.5 * v1[v1.startIndex + index])
                    / span
            }
            incoming[segment + 1] = (0..<stride).map { index in
                (-0.5 * v0[v0.startIndex + index] + (0.5 - wN) * v1[v1.startIndex + index] + wN * vN[vN.startIndex + index])
                    / span
            }
        }
        var triples: [AnimationValue] = []
        for index in times.indices {
            triples.append(.continuous(valueKind, incoming[index]))
            triples.append(values[index])
            triples.append(.continuous(valueKind, outgoing[index]))
        }
        return try AnimationTrack(property: property, times: times, values: triples, interpolation: .cubicSpline)
    }

    /// The track between two times, shifted to start at zero, reproducing this
    /// track exactly on that interval. Boundary keys are sampled; a cubic segment
    /// cut by a boundary gets the curve's exact value and derivative there, and a
    /// `smooth` track becomes its exact cubic spline first. Before its first key
    /// and after its last the track holds a constant value, so a cubic boundary
    /// outside the keys gets zero tangents, as does the tangent of the neighboring
    /// key that faces it (a tangent the source never uses).
    func trimmed(from start: Double, to end: Double) throws -> AnimationTrack {
        if interpolation == .smooth { return try cubicSplineEquivalent().trimmed(from: start, to: end) }
        let first = Double(times[0])
        let last = Double(times[times.count - 1])
        let zero = valueKind.isContinuous ? AnimationValue.continuous(valueKind, [Double](repeating: 0, count: stride)) : nil
        var newTimes: [Float] = [0]
        var newValues: [AnimationValue] = []
        var endBoundary = false
        func append(boundary time: Double) throws {
            if interpolation == .cubicSpline, let zero {
                let (key, fraction) = segment(at: time)
                guard let fraction else {
                    if time < first || time > last {
                        newValues += [zero, values[key * 3 + 1], zero]
                    } else {
                        newValues += values[(key * 3)..<(key * 3 + 3)]
                    }
                    return
                }
                let span = Double(times[key + 1]) - Double(times[key])
                let arguments = (
                    self.key(key), self.key(key, element: 2), self.key(key + 1), self.key(key + 1, element: 0)
                )
                let value = Self.hermite(
                    start: arguments.0, outgoing: arguments.1, end: arguments.2, incoming: arguments.3,
                    span: span, fraction: fraction
                )
                let slope = Self.hermiteDerivative(
                    start: arguments.0, outgoing: arguments.1, end: arguments.2, incoming: arguments.3,
                    span: span, fraction: fraction
                )
                let tangent = AnimationValue.continuous(valueKind, slope)
                newValues += [tangent, .continuous(valueKind, value), tangent]
            } else {
                newValues.append(try sample(at: time))
            }
        }
        try append(boundary: start)
        if end > start {
            for (index, time) in times.enumerated() where Double(time) > start && Double(time) < end {
                newTimes.append(Float(Double(time) - start))
                newValues += values[(index * perKey)..<((index + 1) * perKey)]
            }
            let length = Float(end - start)
            if length > newTimes[newTimes.count - 1] {
                newTimes.append(length)
                try append(boundary: end)
                endBoundary = true
            }
        }
        if interpolation == .cubicSpline, let zero, newTimes.count > 1 {
            // The key after a held start is the source's first key (or a held end), and the key
            // before a held end is its last key (or a held start); their facing tangents are unused.
            if start < first { newValues[3] = zero }
            if end > last, endBoundary { newValues[newValues.count - 4] = zero }
        }
        // A cubic quaternion boundary keeps the Hermite sum's length; sampling normalizes it.
        return try AnimationTrack(
            property: property, times: newTimes, values: newValues, interpolation: interpolation, unitKeys: false
        )
    }

    /// A copy targeting another property with the same values.
    func retargeted(to property: AnimationProperty) throws -> AnimationTrack {
        try AnimationTrack(property: property, times: times, values: values, interpolation: interpolation, unitKeys: false)
    }

    public enum Failure: LocalizedError, Equatable {
        case emptyTrack(String)
        case invalidKeyTime(String, key: Int)
        case valueCount(String, expected: Int, actual: Int)
        case mixedKinds(String)
        case incompatibleKind(String, AnimationValue.Kind)
        case unknownUniform(String)
        case unsupportedInterpolation(String, Interpolation)
        case nonunitQuaternion(String, key: Int)
        case nonfiniteValue(String)
        case invalidSampleTime

        public var errorDescription: String? {
            switch self {
            case .emptyTrack(let property): "The animation track for \(property) needs at least one key."
            case .invalidKeyTime(let property, let key):
                "Key \(key) of the \(property) track needs a finite, nonnegative time greater than its predecessor."
            case .valueCount(let property, let expected, let actual):
                "The \(property) track needs \(expected) values for its keys but has \(actual)."
            case .mixedKinds(let property): "Every value of the \(property) track must have the same kind."
            case .incompatibleKind(let property, let kind): "The \(property) track cannot drive the property with \(kind) values."
            case .unknownUniform(let name): "The shader uniforms declare no value named \"\(name)\"."
            case .unsupportedInterpolation(let property, let mode):
                "The \(property) track cannot use \(mode) interpolation for its value kind."
            case .nonunitQuaternion(let property, let key): "Quaternion key \(key) of the \(property) track is not unit length."
            case .nonfiniteValue(let property): "The \(property) track has or produces a nonfinite value."
            case .invalidSampleTime: "The animation track sample time must be finite."
            }
        }
    }
}
