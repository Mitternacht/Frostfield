import Foundation
import Metal
import Synchronization

extension PathTracer {
    /// Recommends the next path-tracing sample batch from completed GPU timings.
    ///
    /// Feed it `Timing.duration` and the number of samples per pixel in that
    /// encode. The controller smooths seconds per sample and selects the largest
    /// count in its range predicted to fit the target. The minimum count still
    /// applies when even that count is predicted to exceed the target.
    public struct FrameBudget: Equatable, Sendable {
        public let targetSeconds: Double
        public let range: ClosedRange<Int>
        /// The weight of the newest measurement, greater than zero and at most one.
        public let responsiveness: Double
        /// Samples per pixel to pass to the next encode.
        public private(set) var samplesPerFrame: Int
        /// Smoothed GPU seconds per sample once a valid measurement arrives.
        public private(set) var secondsPerSample: Double?

        /// If omitted, the initial sample count is the lower bound of `range`.
        public init(
            targetSeconds: Double, range: ClosedRange<Int> = 1...16,
            initialSamples: Int? = nil, responsiveness: Double = 0.5
        ) throws {
            let initial = initialSamples ?? range.lowerBound
            guard targetSeconds.isFinite, targetSeconds > 0,
                  range.lowerBound >= 1, range.upperBound <= 16, range.contains(initial),
                  responsiveness > 0, responsiveness <= 1 else { throw Failure.frameBudget }
            self.targetSeconds = targetSeconds
            self.range = range
            self.responsiveness = responsiveness
            samplesPerFrame = initial
        }

        /// Ignores unavailable, nonpositive, or non-finite timings and sample
        /// counts outside the tracer's 1...16 per-encode range.
        @discardableResult
        public mutating func record(seconds: Double?, samples: Int) -> Int {
            guard let seconds, seconds.isFinite, seconds > 0, (1...16).contains(samples) else {
                return samplesPerFrame
            }
            let measured = max(seconds / Double(samples), .leastNonzeroMagnitude)
            let smoothed: Double
            if let previous = secondsPerSample {
                let weighted = previous * (1 - responsiveness) + measured * responsiveness
                smoothed = min(max(weighted, min(previous, measured)), max(previous, measured))
            } else {
                smoothed = measured
            }
            secondsPerSample = smoothed
            // Clamp before converting to Int, since a valid tiny duration can
            // put the predicted count beyond the representable integer range.
            let predicted = targetSeconds / smoothed
            samplesPerFrame = Int(min(max(predicted, Double(range.lowerBound)), Double(range.upperBound)))
            return samplesPerFrame
        }
    }

    /// GPU timestamps around the compute pass of one `encode` call.
    ///
    /// The device's timestamp counters are sampled at the start and end of the
    /// path-tracing pass, so the duration covers that dispatch's GPU execution and
    /// excludes queueing and the command buffer's other passes. Timestamps convert to
    /// CPU seconds with `MTLDevice.sampleTimestamps()` pairs taken when the pass is
    /// encoded and when its command buffer completes. One timing measures one pending
    /// command buffer; pass it to the next encode after that buffer finishes, or
    /// after releasing an uncommitted one.
    // The counter buffer is written only by the GPU and read after completion; the
    // remaining state is locked.
    public final class Timing: @unchecked Sendable {
        public let device: any MTLDevice
        private let sampleBuffer: any MTLCounterSampleBuffer
        private let state = Mutex(State())

        /// Throws `PathTracer.Failure.timingUnsupported` when the device has no
        /// timestamp counter set or cannot sample counters at pass boundaries.
        public init(device: any MTLDevice) throws {
            guard let counters = GPUTimestamps.counterSet(on: device) else { throw Failure.timingUnsupported }
            let descriptor = MTLCounterSampleBufferDescriptor()
            descriptor.counterSet = counters
            descriptor.sampleCount = 2
            descriptor.storageMode = .shared
            descriptor.label = "Frostfield path-tracing timestamps"
            do {
                sampleBuffer = try device.makeCounterSampleBuffer(descriptor: descriptor)
            } catch {
                throw Failure.timingUnsupported
            }
            self.device = device
        }

        /// Seconds of GPU execution for the latest timed pass. Nil until its command
        /// buffer completes successfully, or when the counters reported no valid range.
        public var duration: Double? { state.withLock { $0.duration } }

        /// The command can report completion before its timing handler has run.
        var isMeasuring: Bool { state.withLock { $0.commandBuffer != nil } }

        /// Claims the timing for a command buffer and returns the pass descriptor that samples it.
        func begin(_ commandBuffer: any MTLCommandBuffer) throws -> MTLComputePassDescriptor {
            let sequence = try state.withLock { state in
                // A completed buffer can still be running its completion handler.
                guard state.commandBuffer == nil else { throw Failure.timingBusy }
                state.sequence &+= 1
                state.commandBuffer = commandBuffer
                state.duration = nil
                return state.sequence
            }
            let start = device.sampleTimestamps()
            commandBuffer.addCompletedHandler { [self] buffer in
                state.withLock { state in
                    guard state.sequence == sequence else { return }
                    state.duration = buffer.status == .completed
                        ? Self.duration(sampleBuffer, start: start, end: device.sampleTimestamps()) : nil
                    state.commandBuffer = nil
                }
            }
            let descriptor = MTLComputePassDescriptor()
            descriptor.dispatchType = .serial
            let attachment: MTLComputePassSampleBufferAttachmentDescriptor = descriptor.sampleBufferAttachments[0]
            attachment.sampleBuffer = sampleBuffer
            attachment.startOfEncoderSampleIndex = 0
            attachment.endOfEncoderSampleIndex = 1
            return descriptor
        }

        /// Releases a claim whose pass was never encoded.
        func cancel() {
            state.withLock { state in
                state.sequence &+= 1
                state.commandBuffer = nil
            }
        }

        private static func duration(
            _ buffer: any MTLCounterSampleBuffer,
            start: (cpu: MTLTimestamp, gpu: MTLTimestamp), end: (cpu: MTLTimestamp, gpu: MTLTimestamp)
        ) -> Double? {
            guard let samples = GPUTimestamps.samples(buffer, count: 2),
                  let interval = GPUTimestamps.interval(samples[0], samples[1]),
                  let calibration = GPUTimestampCalibration(start: start, end: end) else { return nil }
            return calibration.seconds(interval)
        }

        // The weak buffer identity is only read and written under the lock.
        private struct State: @unchecked Sendable {
            weak var commandBuffer: (any MTLCommandBuffer)?
            var sequence: UInt64 = 0
            var duration: Double?
        }
    }
}
