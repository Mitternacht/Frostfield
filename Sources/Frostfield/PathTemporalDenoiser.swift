import Foundation
import Metal
import simd

#if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
import MetalFX
#endif

/// Traces one jittered sample per pixel and uses MetalFX to denoise and optionally
/// upscale it for a planar pinhole camera and a visible background. Camera and
/// instance transforms can change between frames. Rebuilding scene geometry,
/// repacking scene materials, or replacing instance slots restarts history.
///
/// Keep one instance per view. Call it serially on retained command buffers from
/// its path tracer's queue. Encode consumers of `outputTexture` after this pass
/// on that queue. The texture is reused and is valid after the command succeeds.
public final class PathTemporalDenoiser {
    public struct Frame: Equatable, Sendable {
        /// Camera sample offset in input pixels, X right and Y down.
        public let jitter: SIMD2<Float>
        /// Whether the pass received a preceding successful frame.
        public let reusedHistory: Bool
    }

    public static func isSupported(device: any MTLDevice) -> Bool {
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        return MTLFXTemporalDenoisedScalerDescriptor.supportsDevice(device)
        #else
        return false
        #endif
    }

    public let inputWidth: Int
    public let inputHeight: Int
    /// The denoised RGBA16Float image in a private, tracked Metal texture.
    public let outputTexture: any MTLTexture

    #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
    private let tracer: PathTracer
    private let scaler: any MTLFXTemporalDenoisedScaler
    private var accumulation: PathTracer.Accumulation
    private let color: any MTLTexture
    private let depth: any MTLTexture
    private let motion: any MTLTexture
    private let diffuseAlbedo: any MTLTexture
    private let specularAlbedo: any MTLTexture
    private let normal: any MTLTexture
    private let roughness: any MTLTexture

    private struct PendingFrame {
        let commandBuffer: any MTLCommandBuffer
        let history: PathTracer.History
        let generation: UInt64
    }

    private var pending: PendingFrame?
    private var previous: PathTracer.History?
    private var generation: UInt64 = 0
    private var frameIndex: UInt32 = 0
    #endif

    /// Use equal input and output sizes to denoise without upscaling.
    /// The output must be at least as large as the input and share its aspect ratio.
    public init(
        tracer: PathTracer, inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int
    ) throws {
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        guard Self.isSupported(device: tracer.device) else { throw Failure.unsupported }
        guard (1...4_096).contains(inputWidth), (1...4_096).contains(inputHeight),
              (inputWidth...16_384).contains(outputWidth), (inputHeight...16_384).contains(outputHeight),
              inputWidth * outputHeight == inputHeight * outputWidth else { throw Failure.dimensions }

        self.tracer = tracer
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        accumulation = try tracer.makeAccumulation(width: inputWidth, height: inputHeight)

        let descriptor = MTLFXTemporalDenoisedScalerDescriptor()
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .rgba16Float
        descriptor.depthTextureFormat = .r32Float
        descriptor.motionTextureFormat = .rg16Float
        descriptor.diffuseAlbedoTextureFormat = .rgba16Float
        descriptor.specularAlbedoTextureFormat = .rgba16Float
        descriptor.normalTextureFormat = .rgba16Float
        descriptor.roughnessTextureFormat = .r16Float
        descriptor.outputTextureFormat = .rgba16Float
        descriptor.isAutoExposureEnabled = true
        descriptor.requiresSynchronousInitialization = true
        guard let scaler = descriptor.makeTemporalDenoisedScaler(device: tracer.device) else { throw Failure.scaler }
        self.scaler = scaler

        func texture(
            _ format: MTLPixelFormat, _ usage: MTLTextureUsage, _ label: String, output: Bool = false
        ) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: output ? outputWidth : inputWidth,
                height: output ? outputHeight : inputHeight, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = usage.union(output ? .shaderRead : .shaderWrite)
            guard let result = tracer.device.makeTexture(descriptor: descriptor) else { throw Failure.texture(label) }
            result.label = "Frostfield temporal \(label)"
            return result
        }

        color = try texture(.rgba16Float, scaler.colorTextureUsage, "color")
        depth = try texture(.r32Float, scaler.depthTextureUsage, "depth")
        motion = try texture(.rg16Float, scaler.motionTextureUsage, "motion")
        diffuseAlbedo = try texture(.rgba16Float, scaler.diffuseAlbedoTextureUsage, "diffuse albedo")
        specularAlbedo = try texture(.rgba16Float, scaler.specularAlbedoTextureUsage, "specular albedo")
        normal = try texture(.rgba16Float, scaler.normalTextureUsage, "normal")
        roughness = try texture(.r16Float, scaler.roughnessTextureUsage, "roughness")
        outputTexture = try texture(.rgba16Float, scaler.outputTextureUsage, "output", output: true)

        scaler.colorTexture = color
        scaler.depthTexture = depth
        scaler.motionTexture = motion
        scaler.diffuseAlbedoTexture = diffuseAlbedo
        scaler.specularAlbedoTexture = specularAlbedo
        scaler.normalTexture = normal
        scaler.roughnessTexture = roughness
        scaler.outputTexture = outputTexture
        scaler.isDepthReversed = false
        scaler.motionVectorScaleX = 1
        scaler.motionVectorScaleY = 1
        scaler.preExposure = 1
        #else
        throw Failure.unsupported
        #endif
    }

    /// Encodes the trace and native denoiser without submitting the command buffer.
    /// Finish the prior frame before encoding the next one. A failed command
    /// starts a new history. Call `reset()` for camera cuts and changes to the
    /// lighting, live surface uniforms, or in-place texture contents.
    @discardableResult
    public func encode(
        cameraModel: PathTracer.CameraModel, cameraTransform: SceneTransform,
        settings: PathTracer.Settings = .init(), lighting: PathTracer.Lighting? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws -> Frame {
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        guard commandBuffer.commandQueue === tracer.queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        try finishPendingFrame()
        let lighting = lighting ?? tracer.defaultLighting
        guard lighting.backgroundVisible, lighting.fogVolumes.isEmpty else { throw Failure.lighting }

        let matrices = try PathTracingScene.planarPathCamera(
            model: cameraModel, cameraTransform: cameraTransform, width: inputWidth, height: inputHeight
        )
        if let previous, !tracer.canReuseMotionScene(previous) { reset() }
        let jitter = TemporalAntiAliasing.jitter(at: Int(frameIndex % 32))
        var settings = settings
        settings.seed &+= frameIndex
        let history = try tracer.encode(
            cameraModel: cameraModel, cameraTransform: cameraTransform, destination: color, accumulation: accumulation,
            commandBuffer: commandBuffer, settings: settings, lighting: lighting,
            aovs: .init(
                albedo: diffuseAlbedo, normal: normal, roughness: roughness,
                specularAlbedo: specularAlbedo, projectedDepth: depth,
                cameraMotion: .init(vectors: motion, previousFrame: previous, jitter: jitter)
            )
        )

        let reusedHistory = previous != nil
        // Moving the ray sample right/down moves the image content left/up.
        // MetalFX's jitter is the offset of the content from the pixel center.
        let contentJitter = -jitter
        scaler.jitterOffsetX = contentJitter.x
        scaler.jitterOffsetY = contentJitter.y
        scaler.worldToViewMatrix = matrices.viewMatrix
        scaler.viewToClipMatrix = TemporalAntiAliasing.jitteredClipFromWorld(
            matrices.projectionMatrix, jitter: contentJitter, width: Double(inputWidth), height: Double(inputHeight)
        )
        scaler.shouldResetHistory = !reusedHistory
        scaler.encode(commandBuffer: commandBuffer)
        pending = PendingFrame(commandBuffer: commandBuffer, history: history, generation: generation)
        frameIndex &+= 1
        return Frame(jitter: jitter, reusedHistory: reusedHistory)
        #else
        throw Failure.unsupported
        #endif
    }

    /// Discards temporal history for the next frame and restarts the jitter
    /// sequence. A frame already in flight must finish before the next encode.
    public func reset() {
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        generation &+= 1
        previous = nil
        frameIndex = 0
        #endif
    }

    /// Abandons the outstanding frame if its command buffer was never enqueued.
    /// Discard that entire buffer. The next frame starts a new history.
    public func discardUnsubmittedFrame(_ commandBuffer: any MTLCommandBuffer) throws {
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        guard let pending, pending.commandBuffer === commandBuffer, commandBuffer.status == .notEnqueued else {
            throw Failure.discard
        }
        accumulation = try tracer.makeAccumulation(width: inputWidth, height: inputHeight)
        self.pending = nil
        reset()
        #else
        throw Failure.unsupported
        #endif
    }

    #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
    private func finishPendingFrame() throws {
        guard let pending else { return }
        switch pending.commandBuffer.status {
        case .completed:
            if pending.generation == generation { previous = pending.history }
        case .error:
            reset()
        default:
            throw Failure.frameInFlight
        }
        self.pending = nil
    }
    #endif

    public enum Failure: LocalizedError, Equatable {
        case unsupported
        case dimensions
        case scaler
        case texture(String)
        case lighting
        case commandBuffer
        case frameInFlight
        case discard

        public var errorDescription: String? {
            switch self {
            case .unsupported: "MetalFX temporal denoising is unavailable on this platform or GPU."
            case .dimensions: "Use positive input and output sizes with the same aspect ratio and an output at least as large as the input."
            case .scaler: "MetalFX could not create a temporal denoiser for these sizes and formats."
            case .texture(let name): "Metal could not allocate the temporal denoiser's \(name) texture."
            case .lighting: "Temporal path denoising requires a visible background and a scene without world fog."
            case .commandBuffer: "Use a retained, uncommitted command buffer from the path tracer's queue."
            case .frameInFlight: "Finish or discard the denoiser's outstanding frame before encoding another."
            case .discard: "Only the outstanding denoiser command buffer can be discarded before enqueue."
            }
        }
    }
}
