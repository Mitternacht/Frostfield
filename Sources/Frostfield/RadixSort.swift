import Foundation

/// A stable CPU sort of indices by finite float keys, in the role of Three.js
/// `SortUtils.radixSort`.
///
/// Equal keys keep their original order. Negative zero sorts before positive zero.
/// The key order is the IEEE bit order used by a four-pass radix sort, so every
/// finite value has one place.
public enum RadixSort {
    public static let maximumCount = 1 << 20

    public enum Failure: LocalizedError, Equatable {
        case count
        case nonfinite

        public var errorDescription: String? {
            switch self {
            case .count: "A radix sort accepts at most \(RadixSort.maximumCount) keys."
            case .nonfinite: "A radix sort key must be finite."
            }
        }
    }

    /// Indices `0..<keys.count` ordered so `keys[result[i]]` is nondecreasing
    /// when `ascending` is true.
    public static func indices(of keys: [Float], ascending: Bool = true) throws -> [UInt32] {
        guard keys.count <= maximumCount else { throw Failure.count }
        guard keys.allSatisfy(\.isFinite) else { throw Failure.nonfinite }
        let count = keys.count
        if count <= 1 { return keys.indices.map { UInt32($0) } }
        var codes = keys.map { ordered($0.bitPattern, ascending: ascending) }
        var indices = (0..<count).map { UInt32($0) }
        var scratchCodes = codes
        var scratchIndices = indices
        var histogram = [Int](repeating: 0, count: 256)
        for shift in stride(from: 0, to: 32, by: 8) {
            histogram.replaceSubrange(0..<256, with: repeatElement(0, count: 256))
            for code in codes {
                histogram[Int((code >> shift) & 0xff)] += 1
            }
            var sum = 0
            for bucket in 0..<256 {
                let frequency = histogram[bucket]
                histogram[bucket] = sum
                sum += frequency
            }
            for index in 0..<count {
                let bucket = Int((codes[index] >> shift) & 0xff)
                let destination = histogram[bucket]
                histogram[bucket] = destination + 1
                scratchIndices[destination] = indices[index]
                scratchCodes[destination] = codes[index]
            }
            swap(&codes, &scratchCodes)
            swap(&indices, &scratchIndices)
        }
        return indices
    }

    /// Maps an IEEE bit pattern onto an unsigned order. Positive numbers follow
    /// negative ones, and a more negative value sorts first.
    private static func ordered(_ bits: UInt32, ascending: Bool) -> UInt32 {
        let sortable = (bits & 0x8000_0000) == 0 ? bits ^ 0x8000_0000 : ~bits
        return ascending ? sortable : ~sortable
    }
}
