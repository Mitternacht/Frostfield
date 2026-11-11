import Foundation
import simd

/// An easing curve that maps linear progress to eased progress, with the tween.js
/// `Easing` family (Robert Penner's equations), tween.js `generatePow`, CSS
/// `cubic-bezier()` and CSS `steps()`.
///
/// The Penner curves use tween.js's exact formulas, including its elastic and back
/// constants. Every curve maps zero to zero and one to one; elastic and back
/// curves overshoot in between.
public struct Easing: Equatable, Sendable {
    public enum Direction: Equatable, Sendable { case `in`, out, inOut }
    /// Where CSS `steps()` jumps.
    public enum StepPosition: Equatable, Sendable { case jumpStart, jumpEnd, jumpNone, jumpBoth }

    private enum Curve: Equatable, Sendable {
        case linear
        case quadratic, cubic, quartic, quintic, sinusoidal, exponential, circular, elastic, back, bounce
        case power(Double)
        case bezier(Double, Double, Double, Double)
        case steps(Int, StepPosition)
    }

    private let curve: Curve
    private let direction: Direction

    private init(_ curve: Curve, _ direction: Direction = .in) {
        self.curve = curve
        self.direction = direction
    }

    public static let linear = Easing(.linear)
    public static func quadratic(_ direction: Direction) -> Easing { Easing(.quadratic, direction) }
    public static func cubic(_ direction: Direction) -> Easing { Easing(.cubic, direction) }
    public static func quartic(_ direction: Direction) -> Easing { Easing(.quartic, direction) }
    public static func quintic(_ direction: Direction) -> Easing { Easing(.quintic, direction) }
    public static func sinusoidal(_ direction: Direction) -> Easing { Easing(.sinusoidal, direction) }
    public static func exponential(_ direction: Direction) -> Easing { Easing(.exponential, direction) }
    public static func circular(_ direction: Direction) -> Easing { Easing(.circular, direction) }
    public static func elastic(_ direction: Direction) -> Easing { Easing(.elastic, direction) }
    public static func back(_ direction: Direction) -> Easing { Easing(.back, direction) }
    public static func bounce(_ direction: Direction) -> Easing { Easing(.bounce, direction) }

    /// Every Penner curve in every direction, plus linear.
    public static var pennerCurves: [Easing] {
        [Easing.linear] + [Curve.quadratic, .cubic, .quartic, .quintic, .sinusoidal, .exponential, .circular, .elastic, .back, .bounce]
            .flatMap { curve in [Direction.in, .out, .inOut].map { Easing(curve, $0) } }
    }

    /// tween.js `Easing.generatePow(power)`. The power is clamped to tween.js's range.
    public static func power(_ exponent: Float, _ direction: Direction) throws -> Easing {
        guard exponent.isFinite else { throw Failure.invalidParameters }
        return Easing(.power(min(max(Double(exponent), Double(Float.ulpOfOne)), 10_000)), direction)
    }

    /// CSS `cubic-bezier(x1, y1, x2, y2)`: a curve from (0, 0) to (1, 1) whose
    /// control points need x in zero to one. Progress outside zero to one extends
    /// along the end tangents, as CSS specifies.
    public static func cubicBezier(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float) throws -> Easing {
        guard (0...1).contains(x1), (0...1).contains(x2), y1.isFinite, y2.isFinite else { throw Failure.invalidParameters }
        return Easing(.bezier(Double(x1), Double(y1), Double(x2), Double(y2)))
    }

    /// CSS `steps(count, position)`. `jumpNone` needs at least two steps.
    public static func steps(_ count: Int, position: StepPosition = .jumpEnd) throws -> Easing {
        guard count > 0, position != .jumpNone || count > 1 else { throw Failure.invalidParameters }
        return Easing(.steps(count, position))
    }

    /// The CSS keywords `ease`, `ease-in`, `ease-out` and `ease-in-out`.
    public static let ease = Easing(.bezier(0.25, 0.1, 0.25, 1))
    public static let easeIn = Easing(.bezier(0.42, 0, 1, 1))
    public static let easeOut = Easing(.bezier(0, 0, 0.58, 1))
    public static let easeInOut = Easing(.bezier(0.42, 0, 0.58, 1))

    /// The eased progress.
    public func callAsFunction(_ progress: Float) -> Float { Float(value(Double(progress))) }

    /// The eased progress in double precision.
    public func value(_ amount: Double) -> Double {
        switch curve {
        case .linear: return amount
        case .bezier(let x1, let y1, let x2, let y2): return Self.bezier(amount, x1, y1, x2, y2)
        case .steps(let count, let position): return Self.steps(amount, count, position)
        default: break
        }
        switch direction {
        case .in: return eased(amount)
        case .out:
            return switch curve {
            case .quadratic: amount * (2 - amount)
            case .cubic: pow(amount - 1, 3) + 1
            case .quartic: 1 - pow(amount - 1, 4)
            case .quintic: pow(amount - 1, 5) + 1
            case .sinusoidal: sin(amount * .pi / 2)
            case .exponential: amount == 1 ? 1 : 1 - pow(2, -10 * amount)
            case .circular: (1 - (amount - 1) * (amount - 1)).squareRoot()
            case .elastic: amount == 0 ? 0 : amount == 1 ? 1 : pow(2, -10 * amount) * sin((amount - 0.1) * 5 * .pi) + 1
            case .back: amount == 0 ? 0 : (amount - 1) * (amount - 1) * ((Self.backS + 1) * (amount - 1) + Self.backS) + 1
            case .bounce: Self.bounceOut(amount)
            case .power(let exponent): 1 - pow(1 - amount, exponent)
            default: amount
            }
        case .inOut:
            switch curve {
            case .sinusoidal: return 0.5 * (1 - sin(.pi * (0.5 - amount)))
            case .exponential:
                if amount == 0 { return 0 }
                if amount == 1 { return 1 }
                let a = amount * 2
                return a < 1 ? 0.5 * pow(1024, a - 1) : 0.5 * (-pow(2, -10 * (a - 1)) + 2)
            case .circular:
                let a = amount * 2
                return a < 1 ? -0.5 * ((1 - a * a).squareRoot() - 1) : 0.5 * ((1 - (a - 2) * (a - 2)).squareRoot() + 1)
            case .elastic:
                if amount == 0 { return 0 }
                if amount == 1 { return 1 }
                let a = amount * 2
                return a < 1 ? -0.5 * pow(2, 10 * (a - 1)) * sin((a - 1.1) * 5 * .pi)
                    : 0.5 * pow(2, -10 * (a - 1)) * sin((a - 1.1) * 5 * .pi) + 1
            case .back:
                let s = Self.backS * 1.525
                let a = amount * 2
                return a < 1 ? 0.5 * (a * a * ((s + 1) * a - s)) : 0.5 * ((a - 2) * (a - 2) * ((s + 1) * (a - 2) + s) + 2)
            case .bounce:
                return amount < 0.5 ? (1 - Self.bounceOut(1 - amount * 2)) * 0.5 : Self.bounceOut(amount * 2 - 1) * 0.5 + 0.5
            case .power(let exponent):
                return amount < 0.5 ? pow(amount * 2, exponent) / 2 : (1 - pow(2 - amount * 2, exponent)) / 2 + 0.5
            default:
                // Polynomials: tween.js halves the in curve and mirrors it.
                let order: Double = switch curve {
                case .quadratic: 2
                case .cubic: 3
                case .quartic: 4
                default: 5
                }
                let a = amount * 2
                if a < 1 { return 0.5 * pow(a, order) }
                let b = a - 2
                return order.truncatingRemainder(dividingBy: 2) == 0 ? -0.5 * (pow(b, order) - 2) : 0.5 * (pow(b, order) + 2)
            }
        }
    }

    private static let backS = 1.70158

    private func eased(_ amount: Double) -> Double {
        switch curve {
        case .quadratic: amount * amount
        case .cubic: amount * amount * amount
        case .quartic: pow(amount, 4)
        case .quintic: pow(amount, 5)
        case .sinusoidal: 1 - sin((1 - amount) * .pi / 2)
        case .exponential: amount == 0 ? 0 : pow(1024, amount - 1)
        case .circular: 1 - (1 - amount * amount).squareRoot()
        case .elastic: amount == 0 ? 0 : amount == 1 ? 1 : -pow(2, 10 * (amount - 1)) * sin((amount - 1.1) * 5 * .pi)
        case .back: amount == 1 ? 1 : amount * amount * ((Self.backS + 1) * amount - Self.backS)
        case .bounce: 1 - Self.bounceOut(1 - amount)
        case .power(let exponent): pow(amount, exponent)
        default: amount
        }
    }

    private static func bounceOut(_ amount: Double) -> Double {
        if amount < 1 / 2.75 { return 7.5625 * amount * amount }
        if amount < 2 / 2.75 {
            let a = amount - 1.5 / 2.75
            return 7.5625 * a * a + 0.75
        }
        if amount < 2.5 / 2.75 {
            let a = amount - 2.25 / 2.75
            return 7.5625 * a * a + 0.9375
        }
        let a = amount - 2.625 / 2.75
        return 7.5625 * a * a + 0.984375
    }

    /// Solves x(t) = progress for the bezier parameter and returns y(t), following
    /// the CSS Easing specification, including its linear extension outside zero to one.
    private static func bezier(_ progress: Double, _ x1: Double, _ y1: Double, _ x2: Double, _ y2: Double) -> Double {
        if progress < 0 {
            // CSS Easing: the tangent at P0 is the line to P1, or to P2 when P1 lies on P0's x.
            let slope = x1 > 0 ? y1 / x1 : (x2 > 0 ? y2 / x2 : 0)
            return slope * progress
        }
        if progress > 1 {
            let slope = x2 < 1 ? (y2 - 1) / (x2 - 1) : (x1 < 1 ? (y1 - 1) / (x1 - 1) : 0)
            return 1 + slope * (progress - 1)
        }
        func coordinate(_ t: Double, _ p1: Double, _ p2: Double) -> Double {
            let u = 1 - t
            return 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t
        }
        func slope(_ t: Double, _ p1: Double, _ p2: Double) -> Double {
            let u = 1 - t
            return 3 * u * u * p1 + 6 * u * t * (p2 - p1) + 3 * t * t * (1 - p2)
        }
        // Newton's method from the progress, with bisection when the slope vanishes.
        var t = progress
        for _ in 0..<8 {
            let error = coordinate(t, x1, x2) - progress
            if abs(error) < 1e-12 { return coordinate(t, y1, y2) }
            let derivative = slope(t, x1, x2)
            guard abs(derivative) > 1e-9 else { break }
            t -= error / derivative
        }
        var lower = 0.0
        var upper = 1.0
        t = progress
        for _ in 0..<64 {
            let x = coordinate(t, x1, x2)
            if abs(x - progress) < 1e-14 { break }
            if x < progress { lower = t } else { upper = t }
            t = (lower + upper) / 2
        }
        return coordinate(t, y1, y2)
    }

    /// CSS Easing Functions Level 1, step easing function steps 1–7, without a before flag.
    private static func steps(_ progress: Double, _ count: Int, _ position: StepPosition) -> Double {
        var step = (progress * Double(count)).rounded(.down)
        if position == .jumpStart || position == .jumpBoth { step += 1 }
        if progress >= 0, step < 0 { step = 0 }
        let jumps = switch position {
        case .jumpStart, .jumpEnd: Double(count)
        case .jumpNone: Double(count - 1)
        case .jumpBoth: Double(count + 1)
        }
        if progress <= 1, step > jumps { step = jumps }
        return step / jumps
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidParameters

        public var errorDescription: String? {
            "Use a finite power, cubic-bezier control x values in zero to one, and a positive step count (two or more for jumpNone)."
        }
    }
}

/// A value animated from one `AnimationValue` to another over time with an easing
/// curve, delay, repeats and yoyo, like a tween.js `Tween`. Numbers, vectors and
/// colors interpolate componentwise; quaternions turn along the shortest arc,
/// extrapolating along it when an easing overshoots.
public struct Tween: Sendable {
    public let from: AnimationValue
    public let to: AnimationValue
    /// Seconds per cycle.
    public let duration: Double
    public let easing: Easing
    /// Seconds before the first cycle starts.
    public let delay: Double
    /// Extra cycles after the first; nil repeats forever.
    public let repeatCount: Int?
    /// Alternate cycles run from `to` back to `from` with the same easing, as tween.js does.
    public let yoyo: Bool

    public init(
        from: AnimationValue, to: AnimationValue, duration: Double, easing: Easing = .linear, delay: Double = 0,
        repeatCount: Int? = 0, yoyo: Bool = false
    ) throws {
        guard from.kind == to.kind, from.kind.isContinuous, from.isFinite, to.isFinite else { throw Failure.values }
        guard duration.isFinite, duration > 0, delay.isFinite, delay >= 0, repeatCount.map({ $0 >= 0 }) ?? true else {
            throw Failure.timing
        }
        self.from = from
        self.to = to
        self.duration = duration
        self.easing = easing
        self.delay = delay
        self.repeatCount = repeatCount
        self.yoyo = yoyo
    }

    /// The value at a time in seconds since the tween started. Before the delay it
    /// is `from`; after the last cycle it holds that cycle's end.
    public func value(at time: Double) throws -> AnimationValue {
        guard time.isFinite else { throw Failure.timing }
        let elapsed = time - delay
        guard elapsed > 0 else { return from }
        var cycle = (elapsed / duration).rounded(.down)
        var local = elapsed - cycle * duration
        if let repeatCount, cycle > Double(repeatCount) {
            cycle = Double(repeatCount)
            local = duration
        } else if local == 0, cycle > 0 {
            // A cycle boundary belongs to the cycle that just ended.
            cycle -= 1
            local = duration
        }
        let reversed = yoyo && cycle.truncatingRemainder(dividingBy: 2) == 1
        let progress = easing.value(local / duration)
        return reversed ? Self.mix(to, from, progress) : Self.mix(from, to, progress)
    }

    /// Whether the last cycle has ended. Infinite tweens never finish.
    public func isFinished(at time: Double) -> Bool {
        guard let repeatCount else { return false }
        return time - delay >= duration * Double(repeatCount + 1)
    }

    static func mix(_ a: AnimationValue, _ b: AnimationValue, _ t: Double) -> AnimationValue {
        if case .quaternion(let start) = a, case .quaternion(let end) = b {
            var first = simd_quatd(vector: SIMD4<Double>(start.vector))
            var target = simd_quatd(vector: SIMD4<Double>(end.vector))
            first = simd_normalize(first)
            target = simd_normalize(target)
            if simd_dot(first.vector, target.vector) < 0 { target = simd_quatd(vector: -target.vector) }
            let delta = first.inverse * target
            let angle = delta.angle
            let result = angle < 1e-12 ? first : simd_normalize(first * simd_quatd(angle: angle * t, axis: delta.axis))
            return .quaternion(simd_quatf(vector: SIMD4<Float>(result.vector)))
        }
        let start = a.components ?? []
        let end = b.components ?? []
        return .continuous(a.kind, zip(start, end).map { $0 + ($1 - $0) * t })
    }

    public enum Failure: LocalizedError, Equatable {
        case values
        case timing

        public var errorDescription: String? {
            switch self {
            case .values: "A tween needs finite start and end values of the same interpolating kind."
            case .timing: "A tween needs a positive duration, a nonnegative delay and repeat count, and finite times."
            }
        }
    }
}

/// A damped harmonic spring advanced by its exact analytic solution, so the state
/// after a span of time does not depend on how the span is divided into frames.
/// It covers maath's `damp` springs and Unity-style `SmoothDamp`, whose usual
/// implementation approximates the exponential and drifts with the frame rate.
///
/// The displacement `x - target` obeys `x'' = -ω²(x - target) - 2ζω x'`.
/// A critically damped spring (ζ = 1) released from rest approaches its target
/// without crossing it.
public struct Spring: Equatable, Sendable {
    /// The undamped angular frequency ω in radians per second.
    public let angularFrequency: Double
    /// ζ: below one oscillates, one is critically damped, above one creeps.
    public let dampingRatio: Double

    public init(angularFrequency: Float, dampingRatio: Float = 1) throws {
        guard angularFrequency.isFinite, angularFrequency > 0, dampingRatio.isFinite, dampingRatio >= 0 else {
            throw Failure.parameters
        }
        self.angularFrequency = Double(angularFrequency)
        self.dampingRatio = Double(dampingRatio)
    }

    /// A mass on a spring with a viscous damper: ω = √(k/m) and ζ = c / (2√(km)).
    public init(stiffness: Float, damping: Float, mass: Float = 1) throws {
        guard stiffness.isFinite, stiffness > 0, damping.isFinite, damping >= 0, mass.isFinite, mass > 0 else {
            throw Failure.parameters
        }
        let k = Double(stiffness)
        let m = Double(mass)
        angularFrequency = (k / m).squareRoot()
        dampingRatio = Double(damping) / (2 * (k * m).squareRoot())
    }

    /// The critically damped spring with ω = 2 / smoothTime, the convention of Unity's `SmoothDamp`.
    public static func criticallyDamped(smoothTime: Float) throws -> Spring {
        guard smoothTime.isFinite, smoothTime > 0 else { throw Failure.parameters }
        return try Spring(angularFrequency: 2 / smoothTime, dampingRatio: 1)
    }

    /// The linear map from (displacement, velocity) now to their values after `time`.
    func propagator(_ time: Double) -> (Double, Double, Double, Double) {
        let omega = angularFrequency
        let zeta = dampingRatio
        if abs(zeta - 1) < 1e-6 {
            let decay = exp(-omega * time)
            return ((1 + omega * time) * decay, time * decay, -omega * omega * time * decay, (1 - omega * time) * decay)
        }
        if zeta < 1 {
            let a = zeta * omega
            let b = omega * (1 - zeta * zeta).squareRoot()
            let decay = exp(-a * time)
            let c = cos(b * time)
            let s = sin(b * time)
            return (decay * (c + a / b * s), decay * s / b, -decay * omega * omega / b * s, decay * (c - a / b * s))
        }
        let root = (zeta * zeta - 1).squareRoot()
        let r1 = -omega * (zeta - root)
        let r2 = -omega * (zeta + root)
        let e1 = exp(r1 * time)
        let e2 = exp(r2 * time)
        let span = r2 - r1
        return ((r2 * e1 - r1 * e2) / span, (e2 - e1) / span, r1 * r2 * (e1 - e2) / span, (r2 * e2 - r1 * e1) / span)
    }

    /// Advances a scalar toward a target by a nonnegative time step.
    public func step(position: inout Float, velocity: inout Float, target: Float, deltaTime: Float) throws {
        var x = SIMD2<Float>(position, 0)
        var v = SIMD2<Float>(velocity, 0)
        try step(position: &x, velocity: &v, target: SIMD2(target, 0), deltaTime: deltaTime)
        position = x.x
        velocity = v.x
    }

    /// Advances every component of a vector toward a target by a nonnegative time step.
    public func step<Vector: SIMD>(
        position: inout Vector, velocity: inout Vector, target: Vector, deltaTime: Float
    ) throws where Vector.Scalar == Float {
        guard deltaTime.isFinite, deltaTime >= 0 else { throw Failure.deltaTime }
        let (a, b, c, d) = propagator(Double(deltaTime))
        for index in position.indices {
            let displacement = Double(position[index]) - Double(target[index])
            let speed = Double(velocity[index])
            position[index] = Float(Double(target[index]) + a * displacement + b * speed)
            velocity[index] = Float(c * displacement + d * speed)
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case parameters
        case deltaTime

        public var errorDescription: String? {
            switch self {
            case .parameters: "A spring needs a positive finite frequency, stiffness, mass or smooth time, and nonnegative damping."
            case .deltaTime: "A spring time step must be finite and nonnegative."
            }
        }
    }
}

/// Frame-rate-independent exponential smoothing, like Three.js `MathUtils.damp`:
/// `x + (target - x)(1 - e^(-λ dt))`. Two steps of `dt₁` and `dt₂` equal one step of `dt₁ + dt₂`.
public enum Damping {
    /// Moves a value toward a target with a rate λ in 1/s over a nonnegative time step.
    public static func damp(_ current: Float, toward target: Float, rate: Float, deltaTime: Float) -> Float {
        let keep = exp(-Double(rate) * Double(deltaTime))
        return Float(Double(target) + (Double(current) - Double(target)) * keep)
    }

    public static func damp<Vector: SIMD>(
        _ current: Vector, toward target: Vector, rate: Float, deltaTime: Float
    ) -> Vector where Vector.Scalar == Float {
        let keep = Float(exp(-Double(rate) * Double(deltaTime)))
        return target + (current - target) * keep
    }

    /// Turns a rotation toward a target along the shortest arc, keeping the fraction
    /// `e^(-λ dt)` of the remaining angle, like maath `dampQ` with an exact exponential.
    public static func damp(_ current: simd_quatf, toward target: simd_quatf, rate: Float, deltaTime: Float) -> simd_quatf {
        let fraction = 1 - exp(-Double(rate) * Double(deltaTime))
        guard case .quaternion(let result) = Tween.mix(.quaternion(current), .quaternion(target), fraction) else {
            return current
        }
        return result
    }
}
