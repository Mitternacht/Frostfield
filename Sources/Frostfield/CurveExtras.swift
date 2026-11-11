import Foundation

// Parameterizations adapted from Three.js r187 CurveExtras.js.
// See LICENSES/Three-CurveExtras.txt.

public extension Curve3D {
    /// The fixed-size Granny Knot from the Three.js curve catalog.
    static let grannyKnot = Curve3D(position: { t in
        let angle = 2 * Double.pi * t
        let x = -0.22 * cos(angle) - 1.28 * sin(angle) - 0.44 * cos(3 * angle) - 0.78 * sin(3 * angle)
        let y = -0.10 * cos(2 * angle) - 0.27 * sin(2 * angle) + 0.38 * cos(4 * angle) + 0.46 * sin(4 * angle)
        let z = 0.70 * cos(3 * angle) - 0.40 * sin(3 * angle)
        return SIMD3(x, y, z) * 20
    })

    /// A planar heart in the XY plane. Scale may be positive, zero or negative.
    static func heart(scale: Double = 5) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let angle = 2 * Double.pi * t
            let sine = sin(angle)
            let x = 16 * sine * sine * sine
            let y = 13 * cos(angle) - 5 * cos(2 * angle) - 2 * cos(3 * angle) - cos(4 * angle)
            return SIMD3(x, y, 0)
        }
    }

    /// Viviani's curve, the intersection of a sphere and an offset cylinder.
    static func viviani(scale: Double = 70) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let angle = 4 * Double.pi * t
            return SIMD3(0.5 * (1 + cos(angle)), 0.5 * sin(angle), sin(0.5 * angle))
        }
    }

    /// The fixed-size `KnotCurve` from the Three.js catalog.
    static let knot = Curve3D(position: { t in
        let angle = 2 * Double.pi * t
        let sine = sin(angle)
        let cosine = cos(angle)
        let radius = 10 + 50 * cosine
        return SIMD3(50 * sine, radius * cosine, radius * sine)
    })

    /// A five-turn helix of radius 30 and height 150 along +Z.
    static let helix = Curve3D(position: { t in
        polarPoint(angle: 10 * Double.pi * t, radius: 30, height: 150 * t)
    })

    /// A toroidal trefoil with two azimuthal turns and three radial turns.
    static func trefoilKnot(scale: Double = 10) throws -> Curve3D {
        try toroidalKnot(azimuthalTurns: 2, radialTurns: 3, scale: scale)
    }

    /// The catalog torus knot with three azimuthal turns and four radial turns.
    static func torusKnot(scale: Double = 10) throws -> Curve3D {
        try toroidalKnot(azimuthalTurns: 3, radialTurns: 4, scale: scale)
    }

    /// A toroidal cinquefoil with two azimuthal turns and five radial turns.
    static func cinquefoilKnot(scale: Double = 10) throws -> Curve3D {
        try toroidalKnot(azimuthalTurns: 2, radialTurns: 5, scale: scale)
    }

    /// A trefoil polynomial over its catalog domain −2...2.
    static func trefoilPolynomialKnot(scale: Double = 10) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let u = 4 * t - 2
            let squared = u * u
            return SIMD3(u * (squared - 3), squared * (squared - 4), u * (0.2 * squared * squared - 2))
        }
    }

    /// A figure-eight polynomial knot over its catalog domain −4...4.
    static func figureEightPolynomialKnot(scale: Double = 1) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let u = 8 * t - 4
            let squared = u * u
            let x = 0.4 * u * (squared - 7) * (squared - 10)
            let y = squared * (squared - 13)
            let z = 0.1 * u * (squared - 4) * (squared - 9) * (squared - 12)
            return SIMD3(x, y, z)
        }
    }

    /// The catalog's decorated torus knot 4a.
    static func decoratedTorusKnot4a(scale: Double = 40) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let angle = 2 * Double.pi * t
            let radius = 1 + 0.6 * (cos(5 * angle) + 0.75 * cos(10 * angle))
            return polarPoint(angle: 2 * angle, radius: radius, height: 0.35 * sin(5 * angle))
        }
    }

    /// The catalog's decorated torus knot 4b.
    static func decoratedTorusKnot4b(scale: Double = 40) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let angle = 2 * Double.pi * t
            let radius = 1 + 0.45 * cos(3 * angle) + 0.4 * cos(9 * angle)
            return polarPoint(angle: 2 * angle, radius: radius, height: 0.2 * sin(9 * angle))
        }
    }

    /// The catalog's decorated torus knot 5a.
    static func decoratedTorusKnot5a(scale: Double = 40) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let angle = 2 * Double.pi * t
            let radius = 1 + 0.3 * cos(5 * angle) + 0.5 * cos(10 * angle)
            return polarPoint(angle: 3 * angle, radius: radius, height: 0.2 * sin(20 * angle))
        }
    }

    /// The catalog's decorated torus knot 5c.
    static func decoratedTorusKnot5c(scale: Double = 40) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let angle = 2 * Double.pi * t
            let radius = 1 + 0.5 * (cos(5 * angle) + 0.4 * cos(20 * angle))
            return polarPoint(angle: 4 * angle, radius: radius, height: 0.35 * sin(15 * angle))
        }
    }

    private static func toroidalKnot(azimuthalTurns: Double, radialTurns: Double, scale: Double) throws -> Curve3D {
        try scaledExtra(scale) { t in
            let angle = 2 * Double.pi * t
            let radialAngle = radialTurns * angle
            return polarPoint(angle: azimuthalTurns * angle, radius: 2 + cos(radialAngle), height: sin(radialAngle))
        }
    }

    private static func polarPoint(angle: Double, radius: Double, height: Double) -> SIMD3<Double> {
        SIMD3(radius * cos(angle), radius * sin(angle), height)
    }

    private static func scaledExtra(
        _ scale: Double, _ position: @escaping @Sendable (Double) -> SIMD3<Double>
    ) throws -> Curve3D {
        guard scale.isFinite else { throw Failure.controlPoints }
        return Curve3D(position: { t in scale * position(t) })
    }
}
