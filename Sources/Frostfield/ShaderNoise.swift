import Foundation
import simd

/// CPU versions of the shader graph's noise, hash and dithering nodes.
///
/// Each function repeats the generated Metal operation for operation: the same
/// integer hash, the same `floor` and `fract`, and every multiply-add written as a
/// fused multiply-add on both sides. Use them to place objects, scatter instances or
/// bake data on the CPU where the GPU shades with the same pattern. On an M4 Max the
/// results match the GPU within 1e-6 at 1,000 random points per function
/// (`ShaderNoiseTests`); the lattice hashes and cell choices match exactly.
public enum ShaderNoise {
    /// The integer lattice hash behind every noise node, in 0...1 with 24-bit resolution.
    public static func hash(_ cell: SIMD3<Int32>, seed: UInt32 = 0) -> Float {
        let x = UInt32(bitPattern: cell.x)
        let y = UInt32(bitPattern: cell.y)
        let z = UInt32(bitPattern: cell.z)
        var h = (x &* 0x8da6_b343) ^ (y &* 0xd816_3841) ^ (z &* 0xcb1a_b31f) ^ (seed &* 0x9e37_79b9)
        h ^= h >> 16
        h = h &* 0x7feb_352d
        h ^= h >> 15
        h = h &* 0x846c_a68b
        h ^= h >> 16
        return Float(h >> 8) * 0x1p-24
    }

    /// `MaterialGraph.Node.hash` of a scalar.
    public static func hash(_ value: Float) -> Float { hashBits(SIMD4((value + 0).bitPattern, 0, 0, 0)) }
    /// `MaterialGraph.Node.hash` of a vector2.
    public static func hash(_ value: SIMD2<Float>) -> Float {
        hashBits(SIMD4((value.x + 0).bitPattern, (value.y + 0).bitPattern, 0, 0))
    }
    /// `MaterialGraph.Node.hash` of a vector3.
    public static func hash(_ value: SIMD3<Float>) -> Float {
        hashBits(SIMD4((value.x + 0).bitPattern, (value.y + 0).bitPattern, (value.z + 0).bitPattern, 0))
    }
    /// `MaterialGraph.Node.hash` of a vector4.
    public static func hash(_ value: SIMD4<Float>) -> Float {
        hashBits(SIMD4(
            (value.x + 0).bitPattern, (value.y + 0).bitPattern, (value.z + 0).bitPattern, (value.w + 0).bitPattern
        ))
    }

    /// Lattice noise in 0...1, as `MaterialGraph.Node.latticeNoise`.
    public static func noise(_ position: SIMD3<Float>, kind: MaterialGraph.NoiseKind) -> Float {
        kind == .value ? value(position) : gradient(position)
    }

    /// Trilinear-smoothed value noise in 0...1.
    public static func value(_ position: SIMD3<Float>) -> Float {
        let point = noisePoint(position)
        let cell = SIMD3<Int32>(Int32(point.x.rounded(.down)), Int32(point.y.rounded(.down)), Int32(point.z.rounded(.down)))
        let f = SIMD3(fract(point.x), fract(point.y), fract(point.z))
        let w = SIMD3(smooth(f.x), smooth(f.y), smooth(f.z))
        func h(_ dx: Int32, _ dy: Int32, _ dz: Int32) -> Float { hash(cell &+ SIMD3(dx, dy, dz)) }
        let lowerA = lerp(h(0, 0, 0), h(1, 0, 0), w.x)
        let lowerB = lerp(h(0, 1, 0), h(1, 1, 0), w.x)
        let upperA = lerp(h(0, 0, 1), h(1, 0, 1), w.x)
        let upperB = lerp(h(0, 1, 1), h(1, 1, 1), w.x)
        return lerp(lerp(lowerA, lowerB, w.y), lerp(upperA, upperB, w.y), w.z)
    }

    /// Quintic-faded gradient noise in 0...1.
    public static func gradient(_ position: SIMD3<Float>) -> Float {
        let point = noisePoint(position)
        let cell = SIMD3<Int32>(Int32(point.x.rounded(.down)), Int32(point.y.rounded(.down)), Int32(point.z.rounded(.down)))
        let local = SIMD3(fract(point.x), fract(point.y), fract(point.z))
        let w = SIMD3(quintic(local.x), quintic(local.y), quintic(local.z))
        func c(_ dx: Int32, _ dy: Int32, _ dz: Int32) -> Float { corner(cell, SIMD3(dx, dy, dz), local) }
        let x00 = lerp(c(0, 0, 0), c(1, 0, 0), w.x)
        let x10 = lerp(c(0, 1, 0), c(1, 1, 0), w.x)
        let x01 = lerp(c(0, 0, 1), c(1, 0, 1), w.x)
        let x11 = lerp(c(0, 1, 1), c(1, 1, 1), w.x)
        let value = lerp(lerp(x00, x10, w.y), lerp(x01, x11, w.y), w.z)
        return min(max(Float(0.5).addingProduct(0.57735, value), 0), 1)
    }

    /// Fractal Brownian motion, as `MaterialGraph.Node.fractalNoise`. Octaves must be
    /// 1...8 and lacunarity and gain finite and positive, as the graph requires.
    public static func fractal(
        _ position: SIMD3<Float>, kind: MaterialGraph.NoiseKind = .gradient,
        octaves: Int = 5, lacunarity: Float = 2, gain: Float = 0.5
    ) -> Float {
        var sum: Float = 0
        var total: Float = 0
        var amplitude: Float = 1
        var point = position
        let offset = SIMD3<Float>(19.19, 7.31, 13.73)
        for _ in 0..<max(0, min(octaves, 8)) {
            let value = noise(point, kind: kind)
            sum = sum.addingProduct(value, amplitude)
            total += amplitude
            amplitude *= gain
            point = SIMD3(
                offset.x.addingProduct(point.x, lacunarity), offset.y.addingProduct(point.y, lacunarity),
                offset.z.addingProduct(point.z, lacunarity)
            )
        }
        let result = sum / total
        return total > 0 && result.isFinite ? result : 0
    }

    /// Cellular (Worley) noise: the nearest feature distance in x and a 0...1 cell identifier in y.
    public static func cellular(_ position: SIMD3<Float>) -> SIMD2<Float> {
        let point = noisePoint(position)
        let cell = SIMD3<Int32>(Int32(point.x.rounded(.down)), Int32(point.y.rounded(.down)), Int32(point.z.rounded(.down)))
        let local = SIMD3(fract(point.x), fract(point.y), fract(point.z))
        var nearest: Float = 8
        var identifier: Float = 0
        for z in Int32(-1)...1 {
            for y in Int32(-1)...1 {
                for x in Int32(-1)...1 {
                    let offset = SIMD3(x, y, z)
                    let neighbor = cell &+ offset
                    let feature = SIMD3<Float>(offset) + SIMD3(
                        hash(neighbor, seed: 4), hash(neighbor, seed: 5), hash(neighbor, seed: 6)
                    )
                    let delta = feature - local
                    let distanceSquared = dot3(delta, delta)
                    if distanceSquared < nearest {
                        nearest = distanceSquared
                        identifier = hash(neighbor, seed: 7)
                    }
                }
            }
        }
        return SIMD2(nearest.squareRoot(), identifier)
    }

    /// Jimenez's interleaved gradient noise of pixel coordinates, in 0...1.
    public static func interleavedGradient(_ pixel: SIMD2<Float>) -> Float {
        let inner = fract((0.06711056 * pixel.x).addingProduct(0.00583715, pixel.y))
        return fract(52.9829189 * inner)
    }

    // MARK: - Shared steps

    private static func hashBits(_ bits: SIMD4<UInt32>) -> Float {
        hash(SIMD3(Int32(bitPattern: bits.x), Int32(bitPattern: bits.y), Int32(bitPattern: bits.z)), seed: bits.w ^ 0x68e3_1da4)
    }

    /// Nonfinite components become zero, then each clamps to ±2²⁰.
    private static func noisePoint(_ position: SIMD3<Float>) -> SIMD3<Float> {
        func clamp(_ x: Float) -> Float { x.isFinite ? min(max(x, -1_048_576), 1_048_576) : 0 }
        return SIMD3(clamp(position.x), clamp(position.y), clamp(position.z))
    }

    /// Metal's `fract`: `x − floor(x)`, kept below 1.
    static func fract(_ x: Float) -> Float { min(x - x.rounded(.down), 0x1.fffffep-1) }

    private static func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float { a.addingProduct(t, b - a) }

    private static func smooth(_ f: Float) -> Float { f * f * Float(3).addingProduct(-2, f) }

    private static func quintic(_ l: Float) -> Float {
        l * l * l * Float(10).addingProduct(l, Float(-15).addingProduct(l, 6))
    }

    private static func dot3(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        (a.x * b.x).addingProduct(a.y, b.y).addingProduct(a.z, b.z)
    }

    private static func corner(_ cell: SIMD3<Int32>, _ offset: SIMD3<Int32>, _ local: SIMD3<Float>) -> Float {
        let corner = cell &+ offset
        let gradient = SIMD3(
            Float(-1).addingProduct(hash(corner, seed: 1), 2), Float(-1).addingProduct(hash(corner, seed: 2), 2),
            Float(-1).addingProduct(hash(corner, seed: 3), 2)
        )
        let lengthSquared = dot3(gradient, gradient)
        let unit = lengthSquared > 0 ? gradient / lengthSquared.squareRoot() : .zero
        return dot3(unit, local - SIMD3<Float>(offset))
    }
}
