import Foundation
import Frostfield
import Metal
import Testing

struct SortTests {
    @Test("Radix sort is stable for finite keys and rejects the rest")
    func radix() throws {
        #expect(try RadixSort.indices(of: []) == [])
        #expect(try RadixSort.indices(of: [4]) == [0])
        #expect(try RadixSort.indices(of: [1, 3, -2, 3, 0]) == [2, 4, 0, 1, 3])
        #expect(try RadixSort.indices(of: [1, 3, -2, 3, 0], ascending: false) == [1, 3, 0, 4, 2])
        let ties = try RadixSort.indices(of: [2, 1, 2, 1])
        #expect(ties == [1, 3, 0, 2])
        #expect(throws: RadixSort.Failure.nonfinite) { try RadixSort.indices(of: [.nan]) }
        #expect(throws: RadixSort.Failure.nonfinite) { try RadixSort.indices(of: [.infinity]) }
        #expect(throws: RadixSort.Failure.count) {
            try RadixSort.indices(of: Array(repeating: 0, count: RadixSort.maximumCount + 1))
        }
    }

    @Test("Bitonic sort matches the radix order for distinct finite keys")
    func bitonic() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let sorter = try BitonicSort(device: device)
        var generator = SplitMix(seed: 0xB170_51C5)
        let keys = (0..<300).map { Float($0) * 0.25 - 20 + generator.nextFloat(in: 0...0.2) }
        let expected = try RadixSort.indices(of: keys)
        let indices = try sort(keys, sorter: sorter, queue: queue)
        #expect(indices == expected)

        let descending = try sort([0.25, -4, 2, -4.5, 7], sorter: sorter, queue: queue, ascending: false)
        #expect(descending == (try RadixSort.indices(of: [0.25, -4, 2, -4.5, 7], ascending: false)))
        #expect(throws: BitonicSort.Failure.nonfinite) {
            try sort([.nan], sorter: sorter, queue: queue)
        }
    }

    private func sort(
        _ keys: [Float], sorter: BitonicSort, queue: any MTLCommandQueue, ascending: Bool = true
    ) throws -> [UInt32] {
        let indices = try #require(sorter.device.makeBuffer(
            length: max(keys.count, 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared
        ))
        let buffer = try #require(queue.makeCommandBuffer())
        try sorter.encode(keys: keys, into: indices, ascending: ascending, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        return Array(UnsafeBufferPointer(
            start: indices.contents().assumingMemoryBound(to: UInt32.self), count: keys.count
        ))
    }
}

private struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let unit = Float(next() >> 40) / Float(1 << 24)
        return range.lowerBound + (range.upperBound - range.lowerBound) * unit
    }
}
