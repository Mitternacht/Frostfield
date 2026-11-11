import Foundation

/// Independent CPU references of the Three.js r187 dithering chunks, written from the GLSL
/// in `dithering_pars_fragment`, `dithering_fragment` and the `rand` of the `common` chunk,
/// not from the Metal port.
enum R187DitheringReference {
    // The GLSL literals are float32 values. The Double reference widens those same values.
    static let a = Double(Float(12.9898))
    static let b = Double(Float(78.233))
    static let c = Double(Float(43758.5453))
    static let pi = Double(Float(3.141592653589793))

    static func fract(_ x: Double) -> Double { x - x.rounded(.down) }

    /// The distance between two values on the circle of circumference 1, where fract wraps.
    static func circularDistance(_ x: Double, _ y: Double) -> Double {
        let difference = abs(x - y)
        return min(difference, 1 - difference)
    }

    /// r187 `rand`: `fract( sin( mod( dot( uv, vec2( a, b ) ), PI ) ) * c )` in Double
    /// arithmetic with the shader's float32 constants.
    static func rand(x: Double, y: Double) -> Double {
        let dt = a * x + b * y
        let sn = dt - pi * (dt / pi).rounded(.down)
        return fract(sin(sn) * c)
    }

    /// The same formula with the literal decimal constants, to show what the float32
    /// rounding of the constants alone does.
    static func randWithDecimalConstants(x: Double, y: Double) -> Double {
        let dt = 12.9898 * x + 78.233 * y
        let sn = dt - Double.pi * (dt / Double.pi).rounded(.down)
        return fract(sin(sn) * 43758.5453)
    }

    /// r187 `mix( 2.0 * shift, -2.0 * shift, grid_position )` in Double arithmetic.
    static func shift(gridPosition g: Double) -> SIMD3<Double> {
        let base = SIMD3<Double>(0.25 / 255, -0.25 / 255, 0.25 / 255)
        return (2 * base) * (1 - g) + (-2 * base) * g
    }

    /// The shader's float32 operations on the CPU: a rounded product and a fused
    /// multiply-add, a correctly rounded divide and a fused multiply-subtract, then the
    /// C library `sinf`.
    static func mirrorRand(x: Float, y: Float) -> Float {
        let dt = (x * 12.9898).addingProduct(y, 78.233)
        let pi: Float = 3.141592653589793
        let sn = dt.addingProduct(-pi, (dt / pi).rounded(.down))
        let scaled = sin(sn) * 43758.5453
        return min(scaled - scaled.rounded(.down), 0x1.fffffep-1)
    }

    /// The shift at the pixel with top-left coordinates (`column`, `row`) of an image `height`
    /// rows tall: `gl_FragCoord` is the pixel center counted from the bottom-left corner.
    static func mirrorShift(column: Int, row: Int, height: Int, flipsRows: Bool = true) -> SIMD3<Double> {
        let y = flipsRows ? Float(height) - Float(row) - 0.5 : Float(row) + 0.5
        return shift(gridPosition: Double(mirrorRand(x: Float(column) + 0.5, y: y)))
    }

    static func srgbEncode(_ x: Double) -> Double {
        let m = abs(x)
        return (m <= 0.0031308 ? m * 12.92 : 1.055 * pow(m, 1 / 2.4) - 0.055) * (x < 0 ? -1 : 1)
    }

    static func srgbDecode(_ x: Double) -> Double {
        let m = abs(x)
        return (m <= 0.04045 ? m / 12.92 : pow((m + 0.055) / 1.055, 2.4)) * (x < 0 ? -1 : 1)
    }
}
