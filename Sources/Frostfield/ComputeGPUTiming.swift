import Foundation
import Metal

/// The GPU execution time of one Metal compute encoder.
public struct ComputeGPUTiming: Sendable {
    public enum Failure: LocalizedError, Equatable {
        case unsupported
        case commandBufferState
        case resource
        case encoder
        case notReady
        case commandFailed(String?)
        case samplesUnavailable

        public var errorDescription: String? {
            switch self {
            case .unsupported: "The Metal device does not expose stage-boundary timestamp counters."
            case .commandBufferState: "Encode compute timing before committing the command buffer."
            case .resource: "Metal could not allocate a compute timestamp sample buffer."
            case .encoder: "Metal could not create the timed compute encoder."
            case .notReady: "The compute timing command buffer has not completed."
            case .commandFailed(let reason): reason ?? "Metal did not complete the timed compute pass."
            case .samplesUnavailable: "Metal did not provide usable compute timestamp samples."
            }
        }
    }

    let completion: GPUTimingCompletion<Double, Failure>

    init(commandBuffer: any MTLCommandBuffer) {
        completion = GPUTimingCompletion(commandBuffer: commandBuffer)
    }

    public static func isSupported(on device: any MTLDevice) -> Bool {
        GPUTimestamps.counterSet(on: device) != nil
    }

    /// Creates and ends one serial compute encoder. Encode dispatches in `body`,
    /// then commit the command buffer and read `duration()` after completion.
    /// Callers using an unretained command buffer still own the resources they bind.
    public static func encode(
        commandBuffer: any MTLCommandBuffer,
        _ body: (any MTLComputeCommandEncoder) -> Void
    ) throws -> ComputeGPUTiming {
        let timing = ComputeGPUTiming(commandBuffer: commandBuffer)
        let encoder = try timing.makeEncoder(in: commandBuffer)
        body(encoder)
        encoder.endEncoding()
        return timing
    }

    /// Reads GPU seconds after the command buffer completes. CPU encoding,
    /// queue wait, and other encoders in the command buffer are outside the interval.
    public func duration() throws -> Double {
        guard let result = completion.result else { throw Failure.notReady }
        return try result.get()
    }

    func makeEncoder(in commandBuffer: any MTLCommandBuffer) throws -> any MTLComputeCommandEncoder {
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBufferState
        }
        let device = commandBuffer.device
        guard let counters = GPUTimestamps.counterSet(on: device) else { throw Failure.unsupported }
        let sampleDescriptor = MTLCounterSampleBufferDescriptor()
        sampleDescriptor.counterSet = counters
        sampleDescriptor.sampleCount = 2
        sampleDescriptor.storageMode = .shared
        sampleDescriptor.label = "Frostfield compute GPU timestamps"
        guard let buffer = try? device.makeCounterSampleBuffer(descriptor: sampleDescriptor) else { throw Failure.resource }

        let pass = MTLComputePassDescriptor()
        pass.dispatchType = .serial
        let attachment: MTLComputePassSampleBufferAttachmentDescriptor = pass.sampleBufferAttachments[0]
        attachment.sampleBuffer = buffer
        attachment.startOfEncoderSampleIndex = 0
        attachment.endOfEncoderSampleIndex = 1
        let start = device.sampleTimestamps()
        guard let encoder = commandBuffer.makeComputeCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        let samples = GPUTimingSampleBuffer(buffer: buffer)
        commandBuffer.addCompletedHandler { [completion] command in
            let result: Result<Double, Failure>
            if command.status != .completed {
                result = .failure(.commandFailed(command.error?.localizedDescription))
            } else if let calibration = GPUTimestampCalibration(start: start, end: command.device.sampleTimestamps()),
                      let timestamps = GPUTimestamps.samples(samples.buffer, count: 2),
                      let interval = GPUTimestamps.interval(timestamps[0], timestamps[1]) {
                result = .success(calibration.seconds(interval))
            } else {
                result = .failure(.samplesUnavailable)
            }
            completion.finish(result, status: command.status)
        }
        return encoder
    }
}
