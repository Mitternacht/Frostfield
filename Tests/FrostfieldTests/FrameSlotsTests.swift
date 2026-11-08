import Testing
@testable import Frostfield

/// The slot pool is the renderer's in-flight bound: it must never hand out a
/// slot the GPU may still be reading, and it must decline instead of waiting.
struct FrameSlotsTests {
    @Test func `Hands out each slot once until it completes`() throws {
        let slots = FrameSlots(capacity: 2)
        let first = try #require(slots.acquire())
        let second = try #require(slots.acquire())
        #expect(first != second)
        #expect(slots.acquire() == nil)

        _ = slots.markSubmitted(first)
        _ = slots.markSubmitted(second)
        slots.complete(first, failed: false)
        #expect(slots.acquire() == first)
        #expect(slots.acquire() == nil)
    }

    @Test func `Declining a frame for lack of a slot is counted, not waited out`() {
        let slots = FrameSlots(capacity: 1)
        let slot = slots.acquire()
        #expect(slot != nil)
        #expect(slots.acquire() == nil)
        #expect(slots.acquire() == nil)
        #expect(slots.statistics.skippedNoSlot == 2)
        #expect(slots.statistics.submitted == 0)
    }

    @Test func `A slot released for a missing target is free again and not submitted`() throws {
        let slots = FrameSlots(capacity: 1)
        let slot = try #require(slots.acquire())
        slots.releaseUnusedForMissingTarget(slot)
        #expect(slots.acquire() == slot)
        #expect(slots.statistics == {
            var expected = FrameStatistics()
            expected.skippedNoTarget = 1
            return expected
        }())
    }

    @Test func `Counts submissions, completions, failures, and frames in flight`() throws {
        let slots = FrameSlots(capacity: 3)
        let a = try #require(slots.acquire())
        let b = try #require(slots.acquire())
        #expect(slots.markSubmitted(a) == 1)
        #expect(slots.markSubmitted(b) == 2)
        #expect(slots.statistics.inFlight == 2)

        slots.complete(a, failed: false)
        slots.complete(b, failed: true)
        let statistics = slots.statistics
        #expect(statistics.submitted == 2)
        #expect(statistics.completed == 2)
        #expect(statistics.failed == 1)
        #expect(statistics.inFlight == 0)
    }

    @Test func `Prefers the least recently used free slot`() throws {
        let slots = FrameSlots(capacity: 3)
        var order: [Int] = []
        for _ in 0..<6 {
            let slot = try #require(slots.acquire())
            order.append(slot)
            _ = slots.markSubmitted(slot)
            slots.complete(slot, failed: false)
        }
        #expect(order == [0, 1, 2, 0, 1, 2])
    }

    /// Completions arrive on Metal's thread while frames are acquired on the
    /// host's. The pool must stay consistent: never more slots in use than its
    /// capacity, and every acquisition accounted for.
    @Test func `Stays consistent when completions race acquisitions`() async {
        let slots = FrameSlots(capacity: 2)
        let attempts = 2_000
        let (stream, continuation) = AsyncStream.makeStream(of: Int.self)

        let completer = Task.detached {
            for await slot in stream {
                slots.complete(slot, failed: false)
            }
        }

        var acquired = 0
        for _ in 0..<attempts {
            if let slot = slots.acquire() {
                acquired += 1
                _ = slots.markSubmitted(slot)
                #expect(slots.statistics.inFlight <= 2)
                continuation.yield(slot)
            } else {
                await Task.yield()
            }
        }
        continuation.finish()
        await completer.value

        let statistics = slots.statistics
        #expect(statistics.submitted == acquired)
        #expect(statistics.completed == acquired)
        #expect(statistics.submitted + statistics.skippedNoSlot == attempts)
        #expect(statistics.inFlight == 0)
    }
}
