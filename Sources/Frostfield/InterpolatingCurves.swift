import Foundation
import simd

public extension Curve3D {
    enum CatmullRomParameterization: Equatable, Sendable {
        case centripetal
        case chordal
        case uniform(tension: Double = 0.5)
    }

    static func cubicBezier(
        from start: SIMD3<Float>, control1: SIMD3<Float>, control2: SIMD3<Float>, to end: SIMD3<Float>
    ) throws -> Curve3D {
        let points = [start, control1, control2, end]
        guard points.allSatisfy(curveFinite) else { throw Failure.controlPoints }
        let a = SIMD3<Double>(start), b = SIMD3<Double>(control1)
        let c = SIMD3<Double>(control2), d = SIMD3<Double>(end)
        return Curve3D(position: { t in
            if t == 0 { return a }
            if t == 1 { return d }
            let s = 1 - t
            let ab = s * a + t * b
            let bc = s * b + t * c
            let cd = s * c + t * d
            return s * (s * ab + t * bc) + t * (s * bc + t * cd)
        }, derivative: { t in
            let s = 1 - t
            let first = (s * s) * (b - a)
            let middle = (2 * s * t) * (c - b)
            let last = (t * t) * (d - c)
            return 3 * (first + middle + last)
        })
    }

    /// Interpolates the supplied points. Closed curves connect the final
    /// control point back to the first. Open endpoints use extrapolated
    /// neighbors. The nonuniform variants use the r187 short-interval rule.
    static func catmullRom(
        _ points: [SIMD3<Float>], closed: Bool = false,
        parameterization: CatmullRomParameterization = .centripetal
    ) throws -> Curve3D {
        guard ((closed ? 3 : 2)...65_537).contains(points.count), points.allSatisfy(curveFinite) else {
            throw Failure.controlPoints
        }
        if case .uniform(let tension) = parameterization, !tension.isFinite { throw Failure.controlPoints }
        let control = points.map(SIMD3<Double>.init)
        let segments = (0..<(closed ? control.count : control.count - 1)).map { index in
            let b = control[index]
            let c = control[(index + 1) % control.count]
            let a = index > 0 ? control[index - 1] : (closed ? control[control.count - 1] : 2 * b - c)
            let d = index + 2 < control.count ? control[index + 2]
                : (closed ? control[(index + 2) % control.count] : 2 * c - b)
            return CatmullRomSegment(a, b, c, d, parameterization: parameterization)
        }
        let first = control[0]
        let last = closed ? first : control[control.count - 1]
        let evaluateSegment: @Sendable (Double) -> (CatmullRomSegment, Double) = { parameter in
            let coordinate = parameter * Double(segments.count)
            let index = min(Int(coordinate), segments.count - 1)
            return (segments[index], coordinate - Double(index))
        }
        return Curve3D(position: { parameter in
            if parameter == 0 { return first }
            if parameter == 1 { return last }
            let (segment, t) = evaluateSegment(parameter)
            return segment.point(at: t)
        }, derivative: { parameter in
            let (segment, t) = evaluateSegment(parameter)
            return segment.derivative(at: t)
        })
    }
}

private struct CatmullRomSegment: Sendable {
    let constant: SIMD3<Double>
    let linear: SIMD3<Double>
    let quadratic: SIMD3<Double>
    let cubic: SIMD3<Double>

    init(
        _ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>, _ d: SIMD3<Double>,
        parameterization: Curve3D.CatmullRomParameterization
    ) {
        let startTangent: SIMD3<Double>
        let endTangent: SIMD3<Double>
        switch parameterization {
        case .uniform(let tension):
            startTangent = tension * (c - a)
            endTangent = tension * (d - b)
        case .centripetal, .chordal:
            let exponent = parameterization == .centripetal ? 0.25 : 0.5
            let first = pow(simd_distance_squared(a, b), exponent)
            let middle = pow(simd_distance_squared(b, c), exponent)
            let last = pow(simd_distance_squared(c, d), exponent)
            let step = middle < 1e-4 ? 1 : middle
            let before = first < 1e-4 ? step : first
            let after = last < 1e-4 ? step : last
            let slope = (c - b) / step
            let startSlope = (b - a) / before - (c - a) / (before + step)
            let endSlope = (d - c) / after - (d - b) / (step + after)
            startTangent = (startSlope + slope) * step
            endTangent = (endSlope + slope) * step
        }
        constant = b
        linear = startTangent
        quadratic = 3 * (c - b) - 2 * startTangent - endTangent
        cubic = 2 * (b - c) + startTangent + endTangent
    }

    func point(at t: Double) -> SIMD3<Double> {
        constant + t * (linear + t * (quadratic + t * cubic))
    }

    func derivative(at t: Double) -> SIMD3<Double> {
        linear + t * (2 * quadratic + 3 * t * cubic)
    }
}

private func curveFinite(_ point: SIMD3<Float>) -> Bool {
    point.x.isFinite && point.y.isFinite && point.z.isFinite
}
