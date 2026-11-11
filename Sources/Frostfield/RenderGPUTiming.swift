import Foundation
import Metal

/// Stage-boundary GPU timing for one Metal render encoder.
public struct RenderGPUTiming: Sendable {
    public struct Measurement: Equatable, Sendable {
        public let vertexSeconds: Double?
        public let fragmentSeconds: Double?
        /// The span of the available stages. Stage intervals can overlap.
        public let elapsedSeconds: Double?
    }

    public enum Failure: LocalizedError, Equatable {
        case unsupported
        case commandBufferState
        case counterAttachmentInUse
        case resource
        case encoder
        case notReady
        case commandFailed(String?)
        case samplesUnavailable

        public var errorDescription: String? {
            switch self {
            case .unsupported: "The Metal device does not expose stage-boundary timestamp counters."
            case .commandBufferState: "Encode render timing before committing the command buffer."
            case .counterAttachmentInUse: "The render pass already uses counter attachment zero."
            case .resource: "Metal could not prepare the render timestamp resources."
            case .encoder: "Metal could not create the timed render encoder."
            case .notReady: "The render timing command buffer has not completed."
            case .commandFailed(let reason): reason ?? "Metal did not complete the timed render pass."
            case .samplesUnavailable: "Metal did not provide usable render timestamp samples."
            }
        }
    }

    let completion: GPUTimingCompletion<Measurement, Failure>

    public static func isSupported(on device: any MTLDevice) -> Bool {
        GPUTimestamps.counterSet(on: device) != nil
    }

    /// Creates and ends a render encoder around the supplied draw commands.
    /// The descriptor is copied and its counter attachment zero must be free.
    /// Callers using an unretained command buffer own the resources they bind.
    public static func encode(
        pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer,
        _ body: (any MTLRenderCommandEncoder) -> Void
    ) throws -> RenderGPUTiming {
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBufferState
        }
        guard pass.sampleBufferAttachments[0].sampleBuffer == nil else { throw Failure.counterAttachmentInUse }
        let device = commandBuffer.device
        guard let counters = GPUTimestamps.counterSet(on: device) else { throw Failure.unsupported }
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counters
        descriptor.sampleCount = 4
        descriptor.storageMode = .shared
        descriptor.label = "Frostfield render GPU timestamps"
        guard let buffer = try? device.makeCounterSampleBuffer(descriptor: descriptor),
              let timedPass = pass.copy() as? MTLRenderPassDescriptor else { throw Failure.resource }
        GPUTimestamps.attachRenderSamples(buffer, to: timedPass.sampleBufferAttachments[0], at: 0)
        let start = device.sampleTimestamps()
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: timedPass) else { throw Failure.encoder }
        body(encoder)
        encoder.endEncoding()

        let completion = GPUTimingCompletion<Measurement, Failure>(commandBuffer: commandBuffer)
        let samples = GPUTimingSampleBuffer(buffer: buffer)
        commandBuffer.addCompletedHandler { command in
            let result: Result<Measurement, Failure>
            if command.status != .completed {
                result = .failure(.commandFailed(command.error?.localizedDescription))
            } else if let calibration = GPUTimestampCalibration(start: start, end: command.device.sampleTimestamps()),
                      let timestamps = GPUTimestamps.samples(samples.buffer, count: 4) {
                result = .success(GPUTimestamps.renderStages(timestamps, at: 0, calibration: calibration))
            } else {
                result = .failure(.samplesUnavailable)
            }
            completion.finish(result, status: command.status)
        }
        return RenderGPUTiming(completion: completion)
    }

    /// Reads the available stage intervals after command-buffer completion.
    /// CPU encoding, queue wait and other encoders are outside the intervals.
    public func results() throws -> Measurement {
        guard let result = completion.result else { throw Failure.notReady }
        return try result.get()
    }
}
