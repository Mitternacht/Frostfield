import Synchronization

/// Counters for the frames a renderer attempted.
///
/// `submitted` counts command buffers committed to the GPU, so a host can check
/// that hidden or still content stops submitting work. The skip counters record
/// frames the renderer declined without blocking.
public struct FrameStatistics: Sendable, Equatable {
    /// Command buffers committed to the GPU.
    public var submitted = 0
    /// Committed command buffers whose completion the GPU has reported,
    /// successful or not.
    public var completed = 0
    /// Completed command buffers that finished with an error status.
    public var failed = 0
    /// Frames declined because every frame slot was still in use by the GPU.
    public var skippedNoSlot = 0
    /// Frames declined because no render target (drawable) was available.
    public var skippedNoTarget = 0
    /// Frames abandoned because Metal could not create a command buffer or
    /// encoder.
    public var encodingFailures = 0

    /// Submitted frames whose GPU completion has not been reported yet.
    public var inFlight: Int { submitted - completed }

    public init() {}
}

/// A fixed set of per-frame resource slots shared between the thread that
/// encodes frames and the thread Metal uses for completion handlers.
///
/// A slot is handed out only when the GPU has completed the last frame that
/// used it, so the per-slot resources it guards are never written while the GPU
/// may still read them. Acquisition never waits: when every slot is busy it
/// reports `nil` and the caller skips the frame.
final class FrameSlots: Sendable {
    private struct State {
        var busy: [Bool]
        var next = 0
        var statistics = FrameStatistics()
    }

    let capacity: Int
    private let state: Mutex<State>

    init(capacity: Int) {
        precondition(capacity >= 1, "A renderer needs at least one frame slot")
        self.capacity = capacity
        self.state = Mutex(State(busy: Array(repeating: false, count: capacity)))
    }

    var statistics: FrameStatistics {
        state.withLock { $0.statistics }
    }

    /// Returns a free slot, searching in round-robin order, or `nil` without
    /// waiting when every slot is still in flight.
    func acquire() -> Int? {
        state.withLock { state in
            for offset in 0..<capacity {
                let slot = (state.next + offset) % capacity
                if !state.busy[slot] {
                    state.busy[slot] = true
                    state.next = (slot + 1) % capacity
                    return slot
                }
            }
            state.statistics.skippedNoSlot += 1
            return nil
        }
    }

    /// Returns an acquired slot that was never submitted because no render
    /// target was available.
    func releaseUnusedForMissingTarget(_ slot: Int) {
        state.withLock { state in
            precondition(state.busy[slot], "Releasing slot \(slot), which is not acquired")
            state.busy[slot] = false
            state.statistics.skippedNoTarget += 1
        }
    }

    /// Returns an acquired slot that was never submitted because encoding
    /// could not start.
    func releaseUnusedForEncodingFailure(_ slot: Int) {
        state.withLock { state in
            precondition(state.busy[slot], "Releasing slot \(slot), which is not acquired")
            state.busy[slot] = false
            state.statistics.encodingFailures += 1
        }
    }

    /// Records that the frame using `slot` is being committed and returns its
    /// one-based frame number. The slot stays busy until `complete(_:failed:)`.
    func markSubmitted(_ slot: Int) -> UInt64 {
        state.withLock { state in
            precondition(state.busy[slot], "Submitting slot \(slot), which is not acquired")
            state.statistics.submitted += 1
            return UInt64(state.statistics.submitted)
        }
    }

    /// Frees `slot` after the GPU reported completion of its frame.
    func complete(_ slot: Int, failed: Bool) {
        state.withLock { state in
            precondition(state.busy[slot], "Completing slot \(slot), which is not in flight")
            state.busy[slot] = false
            state.statistics.completed += 1
            if failed { state.statistics.failed += 1 }
        }
    }
}
