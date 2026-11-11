import Foundation
import Metal
import simd
import Synchronization

/// GPU stage timing for one encoded scene view.
public struct SceneGPUTimingFrame: Sendable {
    public enum Pass: Equatable, Sendable {
        case directionalShadow(cascade: Int)
        case punctualShadow(slice: Int)
        case color
        case opaqueColor
        case transmissionColor
    }

    public struct Measurement: Equatable, Sendable {
        public let pass: Pass
        public let vertexSeconds: Double?
        public let fragmentSeconds: Double?
        /// The span of the available stages. Stage intervals can overlap.
        public let elapsedSeconds: Double?
    }

    public let statistics: SceneDrawStatistics
    let completion: GPUTimingCompletion<[Measurement], SceneGPUTimingError>

    /// Whether the device exposes stage-boundary GPU timestamp counters.
    public static func isSupported(on device: any MTLDevice) -> Bool {
        GPUTimestamps.counterSet(on: device) != nil
    }

    /// Reads the timed passes in encode order after command-buffer completion.
    /// An unavailable stage has no interval. Timings exclude CPU encoding,
    /// queue wait, blits and uninstrumented passes.
    public func results() throws -> [Measurement] {
        guard let result = completion.result else { throw SceneGPUTimingError.notReady }
        return try result.get()
    }

    fileprivate static func measurements(
        for passes: [Pass], samples: [MTLTimestamp], calibration: GPUTimestampCalibration
    ) -> [Measurement] {
        return passes.enumerated().map { index, pass in
            let stages = GPUTimestamps.renderStages(samples, at: index * 4, calibration: calibration)
            return Measurement(
                pass: pass, vertexSeconds: stages.vertexSeconds, fragmentSeconds: stages.fragmentSeconds,
                elapsedSeconds: stages.elapsedSeconds
            )
        }
    }
}

public enum SceneGPUTimingError: LocalizedError, Equatable {
    case unsupported
    case commandBufferState
    case counterAttachmentInUse
    case passLimit
    case resource
    case notReady
    case commandFailed(String?)
    case samplesUnavailable

    public var errorDescription: String? {
        switch self {
        case .unsupported: "The Metal device does not expose stage-boundary timestamp counters."
        case .commandBufferState: "Encode scene timing before committing the command buffer."
        case .counterAttachmentInUse: "The render pass already uses counter attachment zero."
        case .passLimit: "Scene timing can measure at most 128 passes in one view."
        case .resource: "Metal could not prepare the scene timing buffer or render pass."
        case .notReady: "The scene timing command buffer has not completed."
        case .commandFailed(let reason): reason ?? "Metal did not complete the timed scene."
        case .samplesUnavailable: "Metal did not provide usable GPU timestamp samples."
        }
    }
}

public extension SceneGraph {
    /// Encodes the usual color and shadow passes with GPU timestamps. Submit
    /// the command buffer and read the frame after completion. The supplied
    /// descriptor's counter attachment zero must be free. Prepare meshes and
    /// clear depth for the selected mode as in the normal color pass.
    func encodeTimed(
        pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer,
        shadowMap: DirectionalShadowMap? = nil, transmissionBackdrop: (any MTLTexture)? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, depthMode: MeshDepthMode = .forward,
        screenSpaceOcclusion: (any MTLTexture)? = nil,
        scissor: MTLScissorRect? = nil, indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> SceneGPUTimingFrame {
        let timing = try SceneGPUTimingRecording(pass: pass, commandBuffer: commandBuffer)
        let statistics = try MeshRenderer.encode(
            self, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps,
            depthMode: depthMode, transmissionBackdrop: transmissionBackdrop,
            screenOcclusion: screenSpaceOcclusion, scissor: scissor,
            indirectDraws: indirectDraws, gpuTiming: timing
        )
        return timing.frame(statistics: statistics)
    }

    /// Encodes GPU timings for an explicit raster view. View, depth mode,
    /// viewport and attachments follow the matching `encode` overload.
    func encodeTimed(
        pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer,
        clipFromWorld: simd_float4x4, cameraPosition: SIMD3<Float>, directionToCamera: SIMD3<Float>? = nil,
        viewport: MTLViewport? = nil, depthMode: MeshDepthMode = .forward, shadowMap: DirectionalShadowMap? = nil,
        transmissionBackdrop: (any MTLTexture)? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        graphCamera: ShaderGraphCamera? = nil, screenSpaceOcclusion: (any MTLTexture)? = nil,
        scissor: MTLScissorRect? = nil, indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> SceneGPUTimingFrame {
        let timing = try SceneGPUTimingRecording(pass: pass, commandBuffer: commandBuffer)
        let statistics = try MeshRenderer.encode(
            self, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps,
            view: (clipFromWorld, cameraPosition, directionToCamera), viewport: viewport, depthMode: depthMode,
            transmissionBackdrop: transmissionBackdrop, graphCamera: graphCamera, screenOcclusion: screenSpaceOcclusion,
            scissor: scissor, indirectDraws: indirectDraws, gpuTiming: timing
        )
        return timing.frame(statistics: statistics)
    }
}

final class SceneGPUTimingRecording {
    private static let maximumPasses = 128
    private let buffer: any MTLCounterSampleBuffer
    private let completion: GPUTimingCompletion<[SceneGPUTimingFrame.Measurement], SceneGPUTimingError>
    private let submitted = SubmittedPasses()
    private var passes: [SceneGPUTimingFrame.Pass] = []

    private final class SubmittedPasses: Sendable {
        let values = Mutex<[SceneGPUTimingFrame.Pass]>([])
    }

    init(pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw SceneGPUTimingError.commandBufferState
        }
        guard pass.sampleBufferAttachments[0].sampleBuffer == nil else {
            throw SceneGPUTimingError.counterAttachmentInUse
        }
        let device = commandBuffer.device
        guard let counters = GPUTimestamps.counterSet(on: device) else { throw SceneGPUTimingError.unsupported }
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counters
        descriptor.sampleCount = Self.maximumPasses * 4
        descriptor.storageMode = .shared
        descriptor.label = "Frostfield scene GPU timestamps"
        do {
            buffer = try device.makeCounterSampleBuffer(descriptor: descriptor)
        } catch {
            throw SceneGPUTimingError.resource
        }
        completion = GPUTimingCompletion(commandBuffer: commandBuffer)
        let start = device.sampleTimestamps()
        let retained = GPUTimingSampleBuffer(buffer: buffer)
        commandBuffer.addCompletedHandler { [completion, submitted] command in
            let status = command.status
            let result: Result<[SceneGPUTimingFrame.Measurement], SceneGPUTimingError>
            if status != .completed {
                result = .failure(.commandFailed(command.error?.localizedDescription))
            } else {
                let passes = submitted.values.withLock { $0 }
                if passes.isEmpty {
                    result = .success([])
                } else if let calibration = GPUTimestampCalibration(start: start, end: command.device.sampleTimestamps()),
                          let samples = GPUTimestamps.samples(retained.buffer, count: passes.count * 4) {
                    result = .success(SceneGPUTimingFrame.measurements(for: passes, samples: samples, calibration: calibration))
                } else {
                    result = .failure(.samplesUnavailable)
                }
            }
            completion.finish(result, status: status)
        }
    }

    /// Shadow passes already own their descriptor. Color descriptors belong
    /// to the caller and are copied by the renderer before this method runs.
    func record(_ kind: SceneGPUTimingFrame.Pass, into pass: MTLRenderPassDescriptor) throws {
        guard passes.count < Self.maximumPasses else { throw SceneGPUTimingError.passLimit }
        let attachment: MTLRenderPassSampleBufferAttachmentDescriptor = pass.sampleBufferAttachments[0]
        guard attachment.sampleBuffer == nil else { throw SceneGPUTimingError.counterAttachmentInUse }
        GPUTimestamps.attachRenderSamples(buffer, to: attachment, at: passes.count * 4)
        passes.append(kind)
    }

    func frame(statistics: SceneDrawStatistics) -> SceneGPUTimingFrame {
        submitted.values.withLock { $0 = passes }
        return SceneGPUTimingFrame(statistics: statistics, completion: completion)
    }
}
