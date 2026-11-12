import Foundation
import simd

/// Park-Miller random numbers, the generator behind Three.js `MathUtils.seededRandom`.
///
/// Each value keeps its own seed. The first sample after `init(seed: 1)` is
/// `16806 / 2147483646`.
public struct SeededRandom: Equatable, Sendable {
    private var state: Int

    /// `seed` uses the same wrap as Three.js: the remainder modulo 2147483647, then
    /// 2147483646 added while the remainder is not positive.
    public init(seed: Int = 1_234_567) {
        var seed = seed % 2_147_483_647
        if seed <= 0 { seed += 2_147_483_646 }
        state = seed
    }

    /// The next sample in `[0, 1)`.
    public mutating func next() -> Double {
        state = Int((Int64(state) * 16_807) % 2_147_483_647)
        return Double(state - 1) / 2_147_483_646
    }

    /// `lower + (upper - lower) * next()`. The bounds must be finite and ordered.
    public mutating func next(from lower: Double, through upper: Double) throws -> Double {
        guard lower.isFinite, upper.isFinite, lower <= upper else { throw Failure.nonfinite }
        return lower + (upper - lower) * next()
    }

    /// An integer in the closed range. `next()` selects the slot, and the last slot covers the top.
    public mutating func integer(in range: ClosedRange<Int>) -> Int {
        let width = range.count
        let index = min(Int(next() * Double(width)), width - 1)
        return range.lowerBound + index
    }

    /// A unit vector from Marsaglia's method.
    public mutating func direction() -> SIMD3<Double> {
        while true {
            let x = next() * 2 - 1
            let y = next() * 2 - 1
            let squared = x * x + y * y
            if squared > 0, squared < 1 {
                let scale = 2 * (1 - squared).squareRoot()
                return SIMD3(x * scale, y * scale, 1 - 2 * squared)
            }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case nonfinite

        public var errorDescription: String? {
            "A random range needs finite bounds in order."
        }
    }
}
