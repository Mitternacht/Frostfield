import Foundation
import Metal
import simd

#if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
import MetalFX
#endif

/// Motion-guided temporal antialiasing and upscaling for resolved raster frames.
/// Keep one instance per view and encode its consumers afterward on the same queue.
public final class RasterTemporalUpscaler {
    public enum Backend: Sendable, Hashable {
        case metalFX
        /// Resolves motion and depth at input resolution, then upsamples linear color.
        case native
    }

    public enum ColorMode: Sendable, Hashable {
        /// Linear RGBA with alpha equal to one across the frame.
        case opaque
        /// Linear premultiplied RGBA with alpha retained by the native backend.
        case premultiplied
    }

    public struct Frame: Sendable, Equatable {
        /// A successful preceding frame was available. Individual pixels may reject it.
        public let reusedHistory: Bool
    }

    public static func isMetalFXSupported(device: any MTLDevice) -> Bool {
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        return MTLFXTemporalScalerDescriptor.supportsDevice(device)
        #else
        return false
        #endif
    }

    public let queue: any MTLCommandQueue
    public let inputWidth: Int
    public let inputHeight: Int
    public let backend: Backend
    /// The reusable, tracked RGBA16Float result, available after the encode succeeds.
    public let outputTexture: any MTLTexture

    private struct NativeState {
        let taa: TemporalAntiAliasing
        let resolved: any MTLTexture
        let resampler: (any MTLRenderPipelineState)?
    }

    private enum Processor {
        case native(NativeState)
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        case metalFX(MetalFXState)
        #endif
    }

    private struct History {
        let depthMode: MeshDepthMode
        let slot: Int
    }

    private struct Pending {
        let commandBuffer: any MTLCommandBuffer
        let history: History
        let generation: UInt64
    }

    private let processor: Processor
    private var previous: History?
    private var pending: Pending?
    private var generation: UInt64 = 0
    private var frameIndex: UInt32 = 0

    /// A Halton offset for the next raster projection, in input pixels, x right and y down.
    public var nextJitter: SIMD2<Float> {
        let index = pending?.commandBuffer.status == .error ? 0 : frameIndex
        return TemporalAntiAliasing.jitter(at: Int(index % 16))
    }

    /// Use equal aspect ratios and an output at least as large as the input.
    /// With an automatic backend, opaque frames use MetalFX when the device
    /// and size support it. Premultiplied frames use the native backend.
    public init(
        queue: any MTLCommandQueue, inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int,
        colorMode: ColorMode = .premultiplied, preferredBackend: Backend? = nil
    ) throws {
        guard (1...16_384).contains(inputWidth), (1...16_384).contains(inputHeight),
              (inputWidth...16_384).contains(outputWidth), (inputHeight...16_384).contains(outputHeight),
              inputWidth * outputHeight == inputHeight * outputWidth else { throw Failure.dimensions }
        guard preferredBackend != .metalFX || colorMode == .opaque else { throw Failure.colorMode }
        self.queue = queue
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        let input = SIMD2(inputWidth, inputHeight), output = SIMD2(outputWidth, outputHeight)
        let device = queue.device
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        if preferredBackend != .native, colorMode == .opaque, Self.isMetalFXSupported(device: device) {
            if let state = try MetalFXState.make(device: device, input: input, output: output) {
                processor = .metalFX(state)
                outputTexture = state.output
                backend = .metalFX
                return
            }
            if preferredBackend == .metalFX { throw Failure.scaler }
        }
        #endif
        guard preferredBackend != .metalFX else { throw Failure.unsupported }

        let result = try Self.texture(device: device, size: output, format: .rgba16Float, usage: [.renderTarget, .shaderRead], label: "output")
        let resolved = input == output ? result : try Self.texture(
            device: device, size: input, format: .rgba16Float, usage: [.renderTarget, .shaderRead], label: "resolved input"
        )
        let resampler = input == output ? nil : try PostProcessShaders(device: device)
            .render("frostfield_resample_fragment", formats: [.rgba16Float])
        processor = .native(NativeState(taa: try TemporalAntiAliasing(device: device), resolved: resolved, resampler: resampler))
        outputTexture = result
        backend = .native
    }

    /// Encodes one frame without submitting it. Finish the preceding buffer
    /// before encoding another. Supply single-sample scene color, stored Depth32Float,
    /// and the matching RGBA16Float raster motion attachment. Motion RG is
    /// previous-minus-current in top-left input pixels including projection jitter,
    /// B is previous stored depth, and A is one for valid history.
    @discardableResult
    public func encode(
        source: any MTLTexture, depth: any MTLTexture, motionTexture: any MTLTexture,
        jitter: SIMD2<Float>, depthMode: MeshDepthMode = .forward, commandBuffer: any MTLCommandBuffer
    ) throws -> Frame {
        try depthMode.requireProjectionDepth()
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else { throw Failure.commandBuffer }
        guard TemporalAntiAliasing.isValidJitter(jitter) else { throw Failure.jitter }
        try validate(source: source, depth: depth, motion: motionTexture)
        try finishPendingFrame()
        let reuse = previous?.depthMode == depthMode
        let slot = previous?.slot == 0 ? 1 : 0

        switch processor {
        case .native(let state):
            if !reuse { state.taa.reset() }
            do {
                // The motion-only resolve follows the supplied vectors and depth.
                try state.taa.encode(
                    source: source, depth: depth, clipFromWorld: matrix_identity_float4x4,
                    jitter: jitter, depthMode: depthMode, motionTexture: motionTexture,
                    destination: state.resolved, commandBuffer: commandBuffer
                )
                if let pipeline = state.resampler {
                    var size = SIMD4(Float(outputTexture.width), Float(outputTexture.height), 0, 0)
                    try PostProcessShaders.drawFullScreen(
                        into: outputTexture, label: "Frostfield temporal upscale", commandBuffer: commandBuffer,
                        pipeline: pipeline
                    ) { encoder in
                        encoder.setFragmentTexture(state.resolved, index: 0)
                        encoder.setFragmentBytes(&size, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
                    }
                }
            } catch {
                reset()
                throw error
            }
        #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
        case .metalFX(let state):
            try state.encode(
                source: source, depth: depth, motion: motionTexture, jitter: jitter,
                depthMode: depthMode, slot: slot, reuse: reuse, commandBuffer: commandBuffer
            )
        #endif
        }
        pending = Pending(commandBuffer: commandBuffer, history: History(depthMode: depthMode, slot: slot), generation: generation)
        frameIndex &+= 1
        return Frame(reusedHistory: reuse)
    }

    /// Restarts temporal history and the jitter sequence after a cut.
    public func reset() {
        generation &+= 1
        previous = nil
        frameIndex = 0
        if case .native(let state) = processor { state.taa.reset() }
    }

    /// Abandons a frame whose buffer was never enqueued. Discard the entire buffer.
    public func discardUnsubmittedFrame(_ commandBuffer: any MTLCommandBuffer) throws {
        guard let pending, pending.commandBuffer === commandBuffer, commandBuffer.status == .notEnqueued else {
            throw Failure.discard
        }
        self.pending = nil
        reset()
    }

    private func finishPendingFrame() throws {
        guard let pending else { return }
        switch pending.commandBuffer.status {
        case .completed:
            if pending.generation == generation { previous = pending.history }
        case .error: reset()
        default: throw Failure.frameInFlight
        }
        self.pending = nil
    }

    private func validate(source: any MTLTexture, depth: any MTLTexture, motion: any MTLTexture) throws {
        guard [source, depth, motion].allSatisfy({ $0.device === queue.device }) else { throw Failure.wrongDevice }
        guard PostProcessPass.isReadable(source, format: .rgba16Float) else { throw Failure.source }
        guard PostProcessPass.isReadable(depth, format: .depth32Float) else { throw Failure.depth }
        guard PostProcessPass.isReadable(motion, format: .rgba16Float) else { throw Failure.motion }
        guard [source, depth, motion].allSatisfy({ $0.width == inputWidth && $0.height == inputHeight }) else {
            throw Failure.dimensions
        }
        guard [source, depth, motion].allSatisfy({ !TextureTransfer.sharesStorage($0, outputTexture) }) else {
            throw Failure.alias
        }
    }

    private static func texture(
        device: any MTLDevice, size: SIMD2<Int>, format: MTLPixelFormat, usage: MTLTextureUsage, label: String
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: size.x, height: size.y, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = usage
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.texture(label) }
        texture.label = "Frostfield raster temporal \(label)"
        return texture
    }

    public enum Failure: LocalizedError, Equatable {
        case dimensions, colorMode, unsupported, scaler, texture(String), commandBuffer, jitter
        case wrongDevice, source, depth, motion, alias, frameInFlight, discard

        public var errorDescription: String? {
            switch self {
            case .dimensions: "Use matching input textures and positive sizes of equal aspect ratio, with output at least as large as input."
            case .colorMode: "MetalFX temporal scaling requires an opaque color frame."
            case .unsupported: "MetalFX temporal scaling is unavailable on this platform or GPU."
            case .scaler: "MetalFX could not create a temporal scaler for these sizes and formats."
            case .texture(let label): "Metal could not allocate the raster temporal \(label) texture."
            case .commandBuffer: "Use a retained, uncommitted command buffer from the upscaler's queue."
            case .jitter: "Projection jitter must be finite and within one input pixel on each axis."
            case .wrongDevice: "Raster temporal inputs must use the upscaler's device."
            case .source: "Raster temporal color must be a tracked, shader-readable 2D single-sample RGBA16Float texture."
            case .depth: "Raster temporal depth must be a tracked, shader-readable 2D single-sample Depth32Float texture."
            case .motion: "Raster temporal motion must be a tracked, shader-readable 2D single-sample RGBA16Float texture."
            case .alias: "Raster temporal inputs must use storage separate from the output."
            case .frameInFlight: "Finish or discard the outstanding upscale before encoding another."
            case .discard: "Only the outstanding command buffer can be discarded before enqueue."
            }
        }
    }

    #if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
    private struct MetalFXState {
        let scaler: any MTLFXTemporalScaler
        let prepare: any MTLComputePipelineState
        let depth: [any MTLTexture]
        let motion: any MTLTexture
        let reactive: any MTLTexture
        let output: any MTLTexture

        static func make(device: any MTLDevice, input: SIMD2<Int>, output: SIMD2<Int>) throws -> MetalFXState? {
            // MetalFX constructs internal reduced guides. Tiny textures and out-of-range
            // scale factors can trigger framework assertions before it returns a scaler.
            let factor = Float(output.x) / Float(input.x)
            guard input.x >= 16, input.y >= 16,
                  factor >= MTLFXTemporalScalerDescriptor.supportedInputContentMinScale(device: device),
                  factor <= MTLFXTemporalScalerDescriptor.supportedInputContentMaxScale(device: device) else { return nil }
            let descriptor = MTLFXTemporalScalerDescriptor()
            descriptor.inputWidth = input.x
            descriptor.inputHeight = input.y
            descriptor.outputWidth = output.x
            descriptor.outputHeight = output.y
            descriptor.colorTextureFormat = .rgba16Float
            descriptor.depthTextureFormat = .r32Float
            descriptor.motionTextureFormat = .rg16Float
            descriptor.reactiveMaskTextureFormat = .r8Unorm
            descriptor.outputTextureFormat = .rgba16Float
            descriptor.isReactiveMaskTextureEnabled = true
            descriptor.isJitteredMotionVectorsEnabled = true
            descriptor.isAutoExposureEnabled = true
            descriptor.requiresSynchronousInitialization = true
            guard let scaler = descriptor.makeTemporalScaler(device: device) else { return nil }

            func inputTexture(_ format: MTLPixelFormat, _ usage: MTLTextureUsage, _ label: String) throws -> any MTLTexture {
                try RasterTemporalUpscaler.texture(device: device, size: input, format: format, usage: usage.union([.shaderRead, .shaderWrite]), label: label)
            }
            let depths = try (0..<2).map { index in
                try inputTexture(.r32Float, scaler.depthTextureUsage, "depth \(index)")
            }
            let motion = try inputTexture(.rg16Float, scaler.motionTextureUsage, "motion")
            let reactive = try inputTexture(.r8Unorm, scaler.reactiveMaskTextureUsage, "reactive mask")
            let result = try RasterTemporalUpscaler.texture(
                device: device, size: output, format: .rgba16Float, usage: scaler.outputTextureUsage.union(.shaderRead), label: "MetalFX output"
            )
            scaler.motionTexture = motion
            scaler.reactiveMaskTexture = reactive
            scaler.outputTexture = result
            scaler.inputContentWidth = input.x
            scaler.inputContentHeight = input.y
            scaler.motionVectorScaleX = 1
            scaler.motionVectorScaleY = 1
            scaler.preExposure = 1
            let pipeline = try PostProcessShaders(device: device).compute("frostfield_raster_temporal_prepare")
            return MetalFXState(scaler: scaler, prepare: pipeline, depth: depths, motion: motion, reactive: reactive, output: result)
        }

        func encode(
            source: any MTLTexture, depth currentDepth: any MTLTexture, motion rasterMotion: any MTLTexture,
            jitter: SIMD2<Float>, depthMode: MeshDepthMode, slot: Int, reuse: Bool, commandBuffer: any MTLCommandBuffer
        ) throws {
            guard source.usage.contains(scaler.colorTextureUsage) else { throw Failure.source }
            let reversed = depthMode == .reversed
            var options = SIMD4<Float>(reuse ? 1 : 0, reversed ? 1 : 0, TemporalAntiAliasing.Settings().motionDepthThreshold, 0)
            try PostProcessShaders.compute(
                label: "Frostfield raster temporal guides", commandBuffer: commandBuffer, pipeline: prepare,
                width: currentDepth.width, height: currentDepth.height
            ) { encoder in
                encoder.setTexture(currentDepth, index: 0)
                encoder.setTexture(rasterMotion, index: 1)
                encoder.setTexture(depth[1 - slot], index: 2)
                encoder.setTexture(depth[slot], index: 3)
                encoder.setTexture(motion, index: 4)
                encoder.setTexture(reactive, index: 5)
                encoder.setBytes(&options, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            }
            scaler.colorTexture = source
            scaler.depthTexture = depth[slot]
            scaler.isDepthReversed = reversed
            scaler.jitterOffsetX = jitter.x
            scaler.jitterOffsetY = jitter.y
            scaler.reset = !reuse
            scaler.encode(commandBuffer: commandBuffer)
        }
    }
    #endif
}
