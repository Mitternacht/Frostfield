import Foundation
import Metal
import Synchronization

/// Collects timed Metal captures for one frame, including captures on different command buffers.
public struct GPUTimingBatch: Sendable {
    public static let maximumCaptureCount = 1_024

    public struct Totals: Equatable, Sendable {
        /// Sum of available pass spans in seconds. Concurrent passes may overlap.
        public private(set) var seconds: Double = 0
        public private(set) var measuredPassCount = 0
        public private(set) var unavailablePassCount = 0

        fileprivate mutating func add(_ intervals: [Double?]) {
            for interval in intervals {
                if let interval {
                    seconds += interval
                    measuredPassCount += 1
                } else {
                    unavailablePassCount += 1
                }
            }
        }
    }

    public struct Summary: Equatable, Sendable {
        public let render: Totals
        public let compute: Totals
    }

    public enum Failure: LocalizedError, Equatable {
        case captureLimit
        case commandBufferNotSubmitted
        case commandBufferDiscarded

        public var errorDescription: String? {
            switch self {
            case .captureLimit: "A GPU timing batch can track at most \(GPUTimingBatch.maximumCaptureCount) captures."
            case .commandBufferNotSubmitted: "Submit all timed command buffers before resolving the batch."
            case .commandBufferDiscarded: "A timed command buffer was released without submitting it."
            }
        }
    }

    private enum Kind: Sendable { case render, compute }

    private struct Capture: Sendable {
        let kind: Kind
        let validateSubmission: @Sendable () throws -> Void
        let intervals: @Sendable () async throws -> [Double?]
    }

    private var identities: Set<ObjectIdentifier> = []
    private var captures: [Capture] = []

    public init() {}

    /// Tracks a scene's render passes. Tracking the same capture again has no effect.
    public mutating func track(_ frame: SceneGPUTimingFrame) throws {
        try track(frame.completion, kind: .render) { $0.map(\.elapsedSeconds) }
    }

    /// Tracks one render encoder, including a timed post-process graph.
    public mutating func track(_ timing: RenderGPUTiming) throws {
        try track(timing.completion, kind: .render) { [$0.elapsedSeconds] }
    }

    /// Tracks one compute encoder, including a timed shader graph.
    public mutating func track(_ timing: ComputeGPUTiming) throws {
        try track(timing.completion, kind: .compute) { [$0] }
    }

    /// Resolves the same frame whenever called. Submit every tracked command
    /// buffer first. Cancellation waits for any capture currently being read.
    public func resolve() async throws -> Summary {
        try Task.checkCancellation()
        for capture in captures { try capture.validateSubmission() }
        var render = Totals()
        var compute = Totals()
        for capture in captures {
            let intervals = try await capture.intervals()
            try Task.checkCancellation()
            switch capture.kind {
            case .render: render.add(intervals)
            case .compute: compute.add(intervals)
            }
        }
        return Summary(render: render, compute: compute)
    }

    private mutating func track<Value, ErrorType>(
        _ completion: GPUTimingCompletion<Value, ErrorType>, kind: Kind,
        intervals: @escaping @Sendable (Value) -> [Double?]
    ) throws {
        let identity = ObjectIdentifier(completion)
        guard !identities.contains(identity) else { return }
        guard captures.count < Self.maximumCaptureCount else { throw Failure.captureLimit }
        captures.append(Capture(
            kind: kind, validateSubmission: { try completion.validateSubmission() },
            intervals: { intervals(try await completion.wait()) }
        ))
        identities.insert(identity)
    }
}

/// Publishes a resolved sample after the Metal callback has read it, independently
/// of when the command buffer first reports `.completed`.
final class GPUTimingCompletion<Value: Sendable, ErrorType: Error>: Sendable {
    private typealias Waiter = CheckedContinuation<Result<Value, ErrorType>, Never>

    // Access to the weak Metal reference and all resolution state is locked.
    private struct State: @unchecked Sendable {
        weak var command: (any MTLCommandBuffer)?
        var result: Result<Value, ErrorType>?
        var discarded = false
        var waiters: [Waiter] = []
    }

    private let state: Mutex<State>

    init(commandBuffer: any MTLCommandBuffer) {
        state = Mutex(State(command: commandBuffer))
    }

    var result: Result<Value, ErrorType>? { state.withLock { $0.result } }

    func validateSubmission() throws {
        try state.withLock { state in
            guard !state.discarded else { throw GPUTimingBatch.Failure.commandBufferDiscarded }
            if state.result != nil { return }
            guard let command = state.command else { throw GPUTimingBatch.Failure.commandBufferDiscarded }
            guard command.status != .notEnqueued, command.status != .enqueued else {
                throw GPUTimingBatch.Failure.commandBufferNotSubmitted
            }
        }
    }

    func finish(_ result: Result<Value, ErrorType>, status: MTLCommandBufferStatus) {
        let waiters = state.withLock { state in
            // Metal may invoke completion handlers as it releases an unsubmitted buffer.
            state.discarded = status == .notEnqueued || status == .enqueued
            state.result = result
            let pending = state.waiters
            state.waiters.removeAll()
            return pending
        }
        for waiter in waiters { waiter.resume(returning: result) }
    }

    func wait() async throws -> Value {
        let result: Result<Value, ErrorType> = await withCheckedContinuation { waiter in
            let ready = state.withLock { state in
                if state.result == nil { state.waiters.append(waiter) }
                return state.result
            }
            if let ready { waiter.resume(returning: ready) }
        }
        return try result.get()
    }
}
