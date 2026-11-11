import Foundation
import Frostfield
import Testing
import simd

struct TweeningTests {
    @Test
    func `Penner curves match tween.js values and end points`() throws {
        for easing in Easing.pennerCurves {
            #expect(abs(easing.value(0)) < 1e-12)
            #expect(abs(easing.value(1) - 1) < 1e-12)
        }
        // Reference values from the tween.js formulas.
        let cases: [(Easing, Double, Double)] = [
            (.quadratic(.in), 0.5, 0.25), (.quadratic(.out), 0.25, 0.4375), (.quadratic(.inOut), 0.75, 0.875),
            (.cubic(.out), 0.5, 0.875), (.cubic(.inOut), 0.25, 0.0625), (.quartic(.inOut), 0.75, 0.96875),
            (.quintic(.in), 0.5, 0.03125), (.quintic(.inOut), 0.75, 0.984375),
            (.sinusoidal(.inOut), 0.5, 0.5), (.sinusoidal(.in), 0.5, 1 - sin(Double.pi / 4)),
            (.exponential(.in), 0.5, pow(1024, -0.5)), (.exponential(.out), 0.3, 1 - pow(2, -3)),
            (.circular(.out), 0.5, sqrt(0.75)), (.circular(.inOut), 0.25, 0.5 * (1 - sqrt(0.75))),
            (.elastic(.out), 0.25, pow(2, -2.5) * sin(0.15 * 5 * .pi) + 1), (.elastic(.in), 0.5, -pow(2, -5) * sin(-0.6 * 5 * .pi)),
            (.back(.in), 0.5, 0.25 * (2.70158 * 0.5 - 1.70158)), (.back(.out), 0.5, 0.25 * (2.70158 * -0.5 + 1.70158) + 1),
            (.bounce(.out), 1 / 2.75, 1), (.bounce(.out), 0.5, 7.5625 * pow(0.5 - 1.5 / 2.75, 2) + 0.75),
            (.bounce(.inOut), 0.5, 0.5),
        ]
        for (easing, input, expected) in cases {
            #expect(abs(easing.value(input) - expected) < 1e-12, "\(easing) at \(input)")
        }
        // In and out curves of the symmetric families mirror each other.
        for family in [Easing.quadratic, Easing.cubic, Easing.quartic, Easing.quintic, Easing.sinusoidal, Easing.circular, Easing.bounce] {
            for t in stride(from: 0.0, through: 1.0, by: 0.1) {
                #expect(abs(family(.out).value(t) - (1 - family(.in).value(1 - t))) < 1e-12)
                #expect(abs(family(.inOut).value(t) - (1 - family(.inOut).value(1 - t))) < 1e-12)
            }
        }
        let power = try Easing.power(3, .inOut)
        #expect(abs(power.value(0.25) - 0.0625) < 1e-12)
        #expect(abs(Easing.back(.in)(0.5) - Float(0.25 * (2.70158 * 0.5 - 1.70158))) < 1e-6)
    }

    @Test
    func `CSS cubic-bezier and steps follow the CSS Easing specification`() throws {
        let linear = try Easing.cubicBezier(0.3, 0.3, 0.7, 0.7)
        for t in stride(from: 0.0, through: 1.0, by: 0.05) { #expect(abs(linear.value(t) - t) < 1e-9) }
        // Reference: solve x(s) = t by bisection in double precision and evaluate y(s).
        func reference(_ t: Double, _ p: (Double, Double, Double, Double)) -> Double {
            func bezier(_ s: Double, _ a: Double, _ b: Double) -> Double { 3 * (1 - s) * (1 - s) * s * a + 3 * (1 - s) * s * s * b + s * s * s }
            var low = 0.0, high = 1.0
            for _ in 0..<200 {
                let mid = (low + high) / 2
                if bezier(mid, p.0, p.2) < t { low = mid } else { high = mid }
            }
            return bezier((low + high) / 2, p.1, p.3)
        }
        for t in stride(from: 0.0, through: 1.0, by: 0.01) {
            #expect(abs(Easing.ease.value(t) - reference(t, (0.25, 0.1, 0.25, 1))) < 1e-9)
            #expect(abs(Easing.easeInOut.value(t) - reference(t, (0.42, 0, 0.58, 1))) < 1e-9)
        }
        #expect(abs(Easing.ease.value(0.5) - 0.8024033877399112) < 1e-9)
        // Outside zero to one the curve extends along its end tangents.
        let overshoot = try Easing.cubicBezier(0.5, 1.5, 0.5, -0.5)
        #expect(abs(overshoot.value(-1) - (-3)) < 1e-12)
        #expect(abs(overshoot.value(2) - 4) < 1e-12)
        // x1 = 0: the line through P0 and P2 below zero; x2 = 1: the line through P1 and P3 above one.
        let vertical = try Easing.cubicBezier(0, 0.5, 0.5, 1)
        #expect(abs(vertical.value(-0.1) - (-0.2)) < 1e-12)
        let late = try Easing.cubicBezier(0.5, 0, 1, 0.5)
        #expect(abs(late.value(1.1) - 1.2) < 1e-12)
        // Both control points on the end's x: the output holds.
        let flat = try Easing.cubicBezier(0, 0.3, 0, 0.6)
        #expect(flat.value(-0.5) == 0)
        #expect(throws: Easing.Failure.invalidParameters) { try Easing.cubicBezier(1.2, 0, 0.5, 1) }

        let end = try Easing.steps(4)
        #expect([0.0, 0.24, 0.25, 0.99, 1].map(end.value) == [0, 0, 0.25, 0.75, 1])
        let start = try Easing.steps(4, position: .jumpStart)
        #expect([0.0, 0.24, 0.25, 1].map(start.value) == [0.25, 0.25, 0.5, 1])
        let none = try Easing.steps(5, position: .jumpNone)
        #expect([0.0, 0.2, 0.99, 1].map(none.value) == [0, 0.25, 1, 1])
        let both = try Easing.steps(3, position: .jumpBoth)
        #expect([0.0, 0.5, 1].map(both.value) == [0.25, 0.5, 1])
        #expect(throws: Easing.Failure.invalidParameters) { try Easing.steps(1, position: .jumpNone) }
    }

    @Test
    func `Tweens apply delay, easing, repeats and yoyo`() throws {
        let tween = try Tween(
            from: .vector3(.zero), to: .vector3(SIMD3(10, 20, 0)), duration: 2, easing: .quadratic(.in),
            delay: 1, repeatCount: 1, yoyo: true
        )
        #expect(try tween.value(at: 0.5) == .vector3(.zero))
        #expect(try tween.value(at: 2) == .vector3(SIMD3(2.5, 5, 0)))
        #expect(try tween.value(at: 3) == .vector3(SIMD3(10, 20, 0)))
        // The yoyo cycle runs from the end back to the start with the same easing.
        #expect(try tween.value(at: 4) == .vector3(SIMD3(7.5, 15, 0)))
        #expect(try tween.value(at: 9) == .vector3(.zero))
        #expect(tween.isFinished(at: 5) && !tween.isFinished(at: 4.9))

        let z = SIMD3<Float>(0, 0, 1)
        let turn = try Tween(
            from: .quaternion(simd_quatf(angle: 0, axis: z)), to: .quaternion(simd_quatf(angle: 1, axis: z)),
            duration: 1, easing: .back(.in)
        )
        guard case .quaternion(let rotation) = try turn.value(at: 0.5) else { Issue.record("kind"); return }
        // back-in dips below zero, so the rotation turns slightly the other way.
        let expected = Float(0.25 * (2.70158 * 0.5 - 1.70158))
        #expect(abs(rotation.angle - abs(expected)) < 1e-5 && rotation.axis.z < 0)
        #expect(throws: Tween.Failure.values) { try Tween(from: .boolean(true), to: .boolean(false), duration: 1) }
        #expect(throws: Tween.Failure.timing) { try Tween(from: .number(0), to: .number(1), duration: 0) }
    }

    @Test
    func `Critically damped springs settle without overshoot and ignore the frame rate`() throws {
        let spring = try Spring.criticallyDamped(smoothTime: 0.3)
        var position: Float = 0
        var velocity: Float = 0
        var previous = position
        for _ in 0..<600 {
            try spring.step(position: &position, velocity: &velocity, target: 1, deltaTime: 1.0 / 120)
            #expect(position <= 1 && position >= previous)
            previous = position
        }
        #expect(abs(position - 1) < 1e-6)

        // One 1/30 s step equals four 1/120 s steps, for every damping regime.
        var once: Float = 0.25
        var onceVelocity: Float = 3
        var four = once
        var fourVelocity = onceVelocity
        try spring.step(position: &once, velocity: &onceVelocity, target: 1, deltaTime: 1.0 / 30)
        for _ in 0..<4 { try spring.step(position: &four, velocity: &fourVelocity, target: 1, deltaTime: 1.0 / 120) }
        #expect(abs(once - four) < 1e-6 && abs(onceVelocity - fourVelocity) < 1e-5)
        for spring in try [spring, Spring(angularFrequency: 9, dampingRatio: 0.2), Spring(stiffness: 40, damping: 30, mass: 0.5)] {
            var coarse = SIMD3<Float>(0.3, -2, 5)
            var coarseVelocity = SIMD3<Float>(1, 0, -4)
            var fine = coarse
            var fineVelocity = coarseVelocity
            let target = SIMD3<Float>(1, 1, 1)
            for _ in 0..<30 { try spring.step(position: &coarse, velocity: &coarseVelocity, target: target, deltaTime: 1.0 / 30) }
            for _ in 0..<120 { try spring.step(position: &fine, velocity: &fineVelocity, target: target, deltaTime: 1.0 / 120) }
            #expect(simd_reduce_max(abs(coarse - fine)) < 1e-5)
            #expect(simd_reduce_max(abs(coarseVelocity - fineVelocity)) < 1e-4)
        }

        // The state matches the closed form x(t) = (x₀ + (v₀ + ωx₀)t)e^(−ωt).
        let omega: Float = 4
        let critical = try Spring(angularFrequency: omega)
        var x: Float = 2
        var v: Float = -1
        try critical.step(position: &x, velocity: &v, target: 0, deltaTime: 0.7)
        #expect(abs(x - (2 + (-1 + omega * 2) * 0.7) * exp(-omega * 0.7)) < 1e-6)

        // An underdamped spring overshoots.
        let bouncy = try Spring(angularFrequency: 10, dampingRatio: 0.1)
        var y: Float = 0
        var w: Float = 0
        var peak: Float = 0
        for _ in 0..<60 {
            try bouncy.step(position: &y, velocity: &w, target: 1, deltaTime: 1.0 / 60)
            peak = max(peak, y)
        }
        #expect(peak > 1.5)
    }

    @Test
    func `Exponential damping is frame-rate independent for scalars, vectors and rotations`() throws {
        var a: Float = 0
        var b: Float = 0
        a = Damping.damp(a, toward: 10, rate: 3, deltaTime: 0.1)
        for _ in 0..<4 { b = Damping.damp(b, toward: 10, rate: 3, deltaTime: 0.025) }
        #expect(abs(a - b) < 1e-5)
        #expect(abs(a - 10 * (1 - exp(-0.3))) < 1e-5)
        let vector = Damping.damp(SIMD3<Float>(0, 0, 0), toward: SIMD3(1, 2, 3), rate: 2, deltaTime: 0.5)
        #expect(simd_distance(vector, SIMD3(1, 2, 3) * (1 - exp(-1))) < 1e-6)
        let z = SIMD3<Float>(0, 0, 1)
        var coarse = simd_quatf(angle: 0, axis: z)
        var fine = coarse
        let target = simd_quatf(angle: 2, axis: z)
        coarse = Damping.damp(coarse, toward: target, rate: 4, deltaTime: 0.2)
        for _ in 0..<8 { fine = Damping.damp(fine, toward: target, rate: 4, deltaTime: 0.025) }
        #expect(abs(coarse.angle - fine.angle) < 1e-5)
        #expect(abs(coarse.angle - 2 * (1 - exp(-0.8))) < 1e-5)
    }
}
