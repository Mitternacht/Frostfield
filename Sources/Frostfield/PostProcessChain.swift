import Foundation
import Metal

/// An ordered list of full-screen effects applied to a premultiplied linear
/// RGBA16Float frame, like Three.js `EffectComposer` passes.
///
/// Each stage can be bypassed. Enabled stages run in order through ping-pong
/// RGBA16Float intermediates that the chain owns and reuses until the output size
/// changes. The last enabled stage writes the caller's destination in
/// `destinationPixelFormat`. Every graph stage receives the scene depth passed to
/// `encode`. Graph stages are prepared for both the intermediate format and the
/// destination format, so enabling or bypassing stages never compiles shaders.
///
/// The chain is not thread-safe. Serialize calls, and encode on retained command
/// buffers from its queue, so older buffers keep replaced intermediates alive.
public final class PostProcessChain {
    public static let maximumStageCount = 32

    public enum Effect {
        /// A graph pass. Nil uniforms creates live values at the graph's defaults;
        /// read them with `uniforms(at:)`. Textures bind the graph's declared textures by name.
        case graph(PostProcessGraph, uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:])
        /// The warm `ColorGrade`, with a strength from 0 to 1.
        case colorGrade(strength: Float)
        /// `ScreenEffects` bloom, fringing and grain, with settings in their documented ranges.
        case screenEffects(ScreenEffects.Settings)
        /// `HDRDisplay` tone mapping to display-referred color, like Three.js `OutputPass`.
        /// A headroom above 1 needs the stage to write RGBA16Float. `Settings.dithering` needs
        /// the opposite: the stage must write the chain's 8-bit destination directly, so it
        /// must be the final stage, without a resolution scale or a mask, in a chain whose
        /// destination is BGRA8Unorm_sRGB or RGBA8Unorm_sRGB.
        case toneMapping(HDRDisplay.Settings)
        /// `FXAA` edge smoothing, like Three.js `FXAAPass`. Place it after `.toneMapping`.
        case fxaa(FXAA.Settings)
        /// `SMAA` 1x, like Three.js `SMAAPass`. Place it after `.toneMapping`.
        case smaa(SMAA.Quality)
        /// A prepared effect such as `MipmapBloom`, `BokehDepthOfField` or `ColorLookupPass`,
        /// on the chain's device. Edit the effect's own settings between encodes. Each stage
        /// should hold its own effect object.
        case pass(any PostProcessEffect)
        /// Copies the current linear RGBA16Float frame to a caller-owned texture.
        /// The saved frame remains the input to the next stage.
        case save(to: any MTLTexture)
    }

    /// A binary mask read from a linear texture at the chain's working resolution.
    public struct Mask {
        public enum Channel: Sendable { case red, alpha }

        public let texture: any MTLTexture
        public let channel: Channel
        /// Values greater than or equal to this threshold select the processed pixels.
        public let threshold: Float
        public let inverse: Bool

        public init(texture: any MTLTexture, channel: Channel = .red, threshold: Float = 0.5, inverse: Bool = false) {
            self.texture = texture
            self.channel = channel
            self.threshold = threshold
            self.inverse = inverse
        }
    }

    public let queue: any MTLCommandQueue
    /// The format of the destination, which the last enabled stage writes.
    public let destinationPixelFormat: MTLPixelFormat
    /// The number of intermediate textures allocated so far. It stays constant while
    /// the output size stays the same and no more stages are enabled than before.
    public private(set) var intermediateAllocationCount = 0

    private var stages: [Stage] = []
    private var intermediates: [any MTLTexture] = []
    /// Reduced-resolution input and output pairs keyed by size.
    private var scaledIntermediates: [SIMD2<Int>: [any MTLTexture]] = [:]
    private var scaledSceneOutput: (any MTLTexture)?
    private var resamplers: PostProcessShaders?
    private var compositionShaders: PostProcessShaders?
    private var maskedIntermediate: (any MTLTexture)?
    private var device: any MTLDevice { queue.device }

    private struct Stage {
        var kind: Kind
        var isEnabled = true
        var resolutionScale: Float = 1
        var mask: Mask?

        var readsDepth: Bool {
            switch kind {
            case .graph(let pass, _): pass.graph.readsDepth
            case .pass(let effect): effect.readsDepth
            default: false
            }
        }
    }

    // Each effect is prepared for intermediate RGBA16Float output and for the
    // destination format. They are the same object when the formats agree.
    private enum Kind {
        case graph(intermediate: PostProcessPass, final: PostProcessPass)
        case colorGrade(intermediate: ColorGrade, final: ColorGrade, strength: Float)
        case screenEffects(intermediate: ScreenEffects, final: ScreenEffects, settings: ScreenEffects.Settings)
        case toneMapping(intermediate: HDRDisplay, final: HDRDisplay, settings: HDRDisplay.Settings)
        case fxaa(intermediate: FXAA, final: FXAA, settings: FXAA.Settings)
        case smaa(intermediate: SMAA, final: SMAA, quality: SMAA.Quality)
        case pass(any PostProcessEffect)
        case save(any MTLTexture)

        var isSave: Bool {
            if case .save = self { return true }
            return false
        }
    }

    /// Prepares every stage. The destination format is RGBA16Float for further
    /// processing or BGRA8Unorm_sRGB or RGBA8Unorm_sRGB for display. Throws the
    /// failing stage's own error, such as `PostProcessPass.Failure.missingTexture`.
    public init(
        queue: any MTLCommandQueue,
        destinationPixelFormat: MTLPixelFormat = .rgba16Float,
        effects: [Effect]
    ) throws {
        guard PostProcessPass.supportedPixelFormats.contains(destinationPixelFormat) else {
            throw Failure.destinationFormat
        }
        guard effects.count <= Self.maximumStageCount else { throw Failure.tooManyStages(limit: Self.maximumStageCount) }
        self.queue = queue
        self.destinationPixelFormat = destinationPixelFormat
        stages = try effects.map { Stage(kind: try prepare($0, reusing: nil)) }
    }

    public var stageCount: Int { stages.count }

    /// Prepares and inserts an enabled stage. Use `stageCount` to append it.
    /// Existing stages keep their settings, and a failed insert leaves the chain unchanged.
    public func insertEffect(_ effect: Effect, at index: Int) throws {
        guard (0...stages.count).contains(index) else { throw Failure.stageIndex(index) }
        guard stages.count < Self.maximumStageCount else { throw Failure.tooManyStages(limit: Self.maximumStageCount) }
        let kind = try prepare(effect, reusing: nil)
        stages.insert(Stage(kind: kind), at: index)
    }

    /// Removes a stage. Later stages shift toward the start without changing their settings.
    public func removeEffect(at index: Int) throws {
        try checkIndex(index)
        stages.remove(at: index)
    }

    public func isEnabled(at index: Int) throws -> Bool {
        try checkIndex(index)
        return stages[index].isEnabled
    }

    /// Bypassing a stage passes its input to the next enabled stage unchanged.
    public func setEnabled(_ isEnabled: Bool, at index: Int) throws {
        try checkIndex(index)
        stages[index].isEnabled = isEnabled
    }

    /// Applies or clears a mask over a range of stages. Each effect reads the
    /// preceding masked frame. Save stages capture that full frame.
    /// Use the alpha channel of a transparent-clear scene capture for a geometry mask.
    public func setMask(_ mask: Mask?, stages range: Range<Int>) throws {
        guard range.lowerBound >= 0 else { throw Failure.stageIndex(range.lowerBound) }
        guard range.upperBound <= stages.count else { throw Failure.stageIndex(range.upperBound - 1) }
        if let mask { try validateMask(mask) }
        for index in range { stages[index].mask = mask }
    }

    /// The live uniforms of a graph stage, or nil for a built-in effect.
    public func uniforms(at index: Int) throws -> ShaderUniforms? {
        try checkIndex(index)
        guard case .graph(let pass, _) = stages[index].kind else { return nil }
        return pass.uniforms
    }

    /// The fraction of the output size at which a stage runs, from 0.125 to 1.
    public func resolutionScale(at index: Int) throws -> Float {
        try checkIndex(index)
        return stages[index].resolutionScale
    }

    /// Runs a stage at a fraction of the output size, like pmndrs `resolutionScale`.
    ///
    /// A scaled stage reads a reduced copy of its input at `round(size · scale)`, in
    /// which each texel averages its exact source footprint weighted by overlap, so the
    /// mean is kept at any scale and small highlights never fall between taps. The
    /// stage runs there, and its result is bilinearly upsampled for the next stage, so
    /// it suits low-frequency effects such as blurs and color filters. Stages that read depth must run at full resolution. The chain keeps
    /// one reduced pair of RGBA16Float intermediates per scaled size.
    public func setResolutionScale(_ scale: Float, at index: Int) throws {
        try checkIndex(index)
        guard scale.isFinite, (0.125...1).contains(scale) else { throw Failure.resolutionScale }
        guard scale == 1 || !stages[index].readsDepth else { throw Failure.scaledDepth(stage: index) }
        guard scale == 1 || !stages[index].kind.isSave else { throw Failure.scaledSave(stage: index) }
        if scale != 1, resamplers == nil {
            let shaders = try PostProcessShaders(device: device)
            _ = try shaders.render("frostfield_resample_fragment", formats: [.rgba16Float])
            if destinationPixelFormat != .rgba16Float {
                _ = try shaders.render("frostfield_resample_fragment", formats: [destinationPixelFormat])
            }
            resamplers = shaders
        }
        stages[index].resolutionScale = scale
    }

    /// Replaces a stage's effect and keeps whether it is enabled. Built-in effects
    /// of the same kind keep their prepared pipelines, so use this to change their
    /// settings. A graph stage prepares new pipelines from cached libraries; edit its
    /// `ShaderUniforms` instead to change values. The stage is unchanged on failure.
    public func replaceEffect(at index: Int, with effect: Effect) throws {
        try checkIndex(index)
        let kind = try prepare(effect, reusing: stages[index].kind)
        if stages[index].resolutionScale != 1 {
            if Stage(kind: kind).readsDepth { throw Failure.scaledDepth(stage: index) }
            if kind.isSave { throw Failure.scaledSave(stage: index) }
        }
        stages[index].kind = kind
    }

    /// Encodes every enabled stage on the caller's command buffer without submitting it.
    ///
    /// The source is a shader-readable, tracked, single-sample RGBA16Float texture
    /// of premultiplied linear color. `depth`, when given, is shader-readable,
    /// tracked Depth32Float, and is required when an enabled graph reads depth. The
    /// destination is a separate render target in `destinationPixelFormat`, and all
    /// textures share one size. With no stage enabled the chain copies the source,
    /// which needs an RGBA16Float destination that is not framebuffer-only.
    ///
    /// Every stage is validated and every intermediate allocated before anything
    /// is encoded. If Metal later refuses to create an encoder, discard the buffer.
    public func encode(
        source: any MTLTexture,
        depth: (any MTLTexture)? = nil,
        destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let encodeStages = try prepareEncoding(
            source: source, depth: depth, destination: destination, commandBuffer: commandBuffer, sceneSource: false
        )
        try encodeStages()
    }

    /// Captures a scene as the chain's source, then encodes the enabled effects.
    /// Effects start with the capture's resolution and matching depth. A reduced
    /// capture's final color is bilinearly resampled into the destination. With
    /// no enabled effects, the chain displays the capture directly.
    @discardableResult
    public func encode(
        scene: SceneGraph, using scenePass: PostProcessScenePass, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer, camera: ShaderGraphCamera? = nil, depthMode: MeshDepthMode = .forward,
        clearColor: MTLClearColor = MTLClearColor(), shadowMap: DirectionalShadowMap? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, transmissionBackdrop: (any MTLTexture)? = nil
    ) throws -> PostProcessScenePass.Frame {
        let targets = try scenePass.prepare(width: destination.width, height: destination.height, commandBuffer: commandBuffer)
        let encodeStages = try prepareEncoding(
            source: targets.color, depth: targets.depth, destination: destination,
            commandBuffer: commandBuffer, sceneSource: true
        )
        let frame = try scenePass.encode(
            scene, width: destination.width, height: destination.height, targets: targets, commandBuffer: commandBuffer,
            camera: camera, depthMode: depthMode, clearColor: clearColor, shadowMap: shadowMap,
            punctualShadowMaps: punctualShadowMaps, transmissionBackdrop: transmissionBackdrop
        )
        try encodeStages()
        return frame
    }

    private func prepareEncoding(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer, sceneSource: Bool
    ) throws -> () throws -> Void {
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard source.device === device, destination.device === device,
              depth.map({ $0.device === device }) ?? true else { throw Failure.wrongDevice }
        guard PostProcessPass.isReadable(source, format: .rgba16Float), source.width > 0, source.height > 0 else {
            throw Failure.source
        }
        guard !TextureTransfer.sharesStorage(source, destination), destination.pixelFormat == destinationPixelFormat,
              destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless else {
            throw Failure.destination
        }
        if let depth, !PostProcessPass.isReadable(depth, format: .depth32Float) { throw Failure.depth }
        let resizing = destination.width != source.width || destination.height != source.height
        guard (sceneSource || !resizing),
              depth.map({ $0.width == source.width && $0.height == source.height }) ?? true else {
            throw Failure.sizeMismatch
        }

        let enabled = stages.indices.filter { stages[$0].isEnabled }
        if depth == nil, let index = enabled.first(where: { stages[$0].readsDepth }) {
            throw Failure.missingDepth(stage: index)
        }
        guard let lastIndex = enabled.last else {
            if sceneSource {
                let shaders = try prepareResampler(for: destination.pixelFormat)
                return { try Step.resample(shaders, from: source, to: destination, commandBuffer: commandBuffer) }
            }
            guard destination.pixelFormat == source.pixelFormat, !destination.isFramebufferOnly else {
                throw Failure.bypassCopy
            }
            return { try Self.copy(source: source, destination: destination, commandBuffer: commandBuffer) }
        }

        let rendersLast = !stages[lastIndex].kind.isSave
        // A reduced scene capture ends in a resample, which would blur the dither the last stage wrote.
        if resizing, rendersLast, case .toneMapping(_, _, let settings) = stages[lastIndex].kind, settings.dithering {
            throw HDRDisplay.Failure.ditheringNeedsEightBitDestination
        }
        let output = try resizing && rendersLast ? prepareSceneOutput(width: source.width, height: source.height) : destination
        let outputResampler = try resizing ? prepareResampler(for: destination.pixelFormat) : nil
        let effectCount = enabled.count { !stages[$0].kind.isSave }
        let bufferCount = min(effectCount - (rendersLast ? 1 : 0), 2)
        let buffers = try intermediates(count: bufferCount, width: source.width, height: source.height)
        var steps: [Step] = []
        var input = source
        var effectIndex = 0
        for index in enabled {
            let stage = stages[index]
            if case .save(let target) = stage.kind {
                guard target.width == source.width, target.height == source.height else { throw Failure.sizeMismatch }
                guard !TextureTransfer.sharesStorage(target, source),
                      !TextureTransfer.sharesStorage(target, input),
                      !TextureTransfer.sharesStorage(target, destination),
                      depth.map({ !TextureTransfer.sharesStorage(target, $0) }) ?? true else { throw Failure.saveDestination }
                steps.append(Step(kind: stage.kind, isFinal: false, input: input, output: target))
                continue
            }

            let isFinal = index == lastIndex
            let stageOutput = isFinal ? output : buffers[effectIndex % 2]
            var step = Step(kind: stage.kind, isFinal: isFinal, input: input, output: stageOutput)
            effectIndex += 1
            let scale = stage.resolutionScale
            if scale != 1 {
                if depth != nil, stage.readsDepth { throw Failure.scaledDepth(stage: index) }
                let width = max(1, Int((Float(source.width) * scale).rounded()))
                let height = max(1, Int((Float(source.height) * scale).rounded()))
                let pair = try scaledPair(width: width, height: height)
                step.scaled = Scaled(input: pair[0], output: pair[1], resamplers: try resamplers ?? PostProcessShaders(device: device))
            }
            if let mask = stage.mask {
                guard mask.texture.width == source.width, mask.texture.height == source.height else { throw Failure.sizeMismatch }
                guard !TextureTransfer.sharesStorage(mask.texture, destination),
                      !TextureTransfer.sharesStorage(mask.texture, stageOutput) else { throw Failure.mask }
                let working = try prepareMaskedIntermediate(width: source.width, height: source.height)
                let pipeline = try prepareComposition("frostfield_post_process_mask", format: stageOutput.pixelFormat)
                step.masked = Masked(mask: mask, working: working, pipeline: pipeline)
            }
            steps.append(step)
            input = stageOutput
        }
        for step in steps {
            try step.validate(depth: depth, commandBuffer: commandBuffer)
        }
        let finalInput = rendersLast ? output : input
        let finalCopy = try !rendersLast && !resizing
            ? prepareComposition("frostfield_copy_fragment", format: destination.pixelFormat) : nil
        return {
            for step in steps {
                try step.encode(depth: depth, commandBuffer: commandBuffer)
            }
            if let outputResampler {
                try Step.resample(outputResampler, from: finalInput, to: destination, commandBuffer: commandBuffer)
            } else if let finalCopy {
                try PostProcessShaders.drawFullScreen(
                    into: destination, label: "Frostfield post-process presentation", commandBuffer: commandBuffer, pipeline: finalCopy
                ) { $0.setFragmentTexture(finalInput, index: 0) }
            }
        }
    }

    private func validateMask(_ mask: Mask) throws {
        guard mask.texture.device === device else { throw Failure.wrongDevice }
        let rgba: Set<MTLPixelFormat> = [.rgba8Unorm, .rgba16Float]
        let red: Set<MTLPixelFormat> = [.r8Unorm, .r16Float, .r32Float]
        let format = mask.texture.pixelFormat
        guard (0...1).contains(mask.threshold), (rgba.contains(format) || (mask.channel == .red && red.contains(format))),
              PostProcessPass.isReadable(mask.texture, format: format) else { throw Failure.mask }
    }

    private func prepareComposition(_ function: String, format: MTLPixelFormat) throws -> any MTLRenderPipelineState {
        let shaders = try compositionShaders ?? PostProcessShaders(device: device)
        let pipeline = try shaders.render(function, formats: [format])
        compositionShaders = shaders
        return pipeline
    }

    private func prepareMaskedIntermediate(width: Int, height: Int) throws -> any MTLTexture {
        if let maskedIntermediate, maskedIntermediate.width == width, maskedIntermediate.height == height {
            return maskedIntermediate
        }
        let texture = try PostProcessShaders.scratch(
            device: device, width: width, height: height, usage: [.renderTarget, .shaderRead],
            label: "Frostfield post-process masked stage"
        )
        maskedIntermediate = texture
        intermediateAllocationCount += 1
        return texture
    }

    private func prepareResampler(for format: MTLPixelFormat) throws -> PostProcessShaders {
        let shaders = try resamplers ?? PostProcessShaders(device: device)
        _ = try shaders.render("frostfield_resample_fragment", formats: [format])
        resamplers = shaders
        return shaders
    }

    private func prepareSceneOutput(width: Int, height: Int) throws -> any MTLTexture {
        if let scaledSceneOutput, scaledSceneOutput.width == width, scaledSceneOutput.height == height {
            return scaledSceneOutput
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: destinationPixelFormat, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.intermediateAllocation }
        texture.label = "Frostfield post-process scaled scene output"
        scaledSceneOutput = texture
        intermediateAllocationCount += 1
        return texture
    }

    /// Submits the chain in its own command buffer and returns its total GPU
    /// execution time in seconds, including internal resampling and copies.
    /// Commit any source producer on this queue first. Cancellation after
    /// submission waits for Metal completion before returning.
    public func renderTimed(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Double {
        try Task.checkCancellation()
        guard let command = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
        command.label = "Frostfield timed post-process chain"
        try encode(source: source, depth: depth, destination: destination, commandBuffer: command)
        let duration: Double = try await withCheckedThrowingContinuation { continuation in
            command.addCompletedHandler { completed in
                guard completed.status == .completed else {
                    continuation.resume(throwing: Failure.commandFailed(completed.error?.localizedDescription))
                    return
                }
                let start = completed.gpuStartTime
                let end = completed.gpuEndTime
                guard start.isFinite, end.isFinite, start > 0, end >= start else {
                    continuation.resume(throwing: Failure.timingUnavailable)
                    return
                }
                continuation.resume(returning: end - start)
            }
            command.commit()
        }
        try Task.checkCancellation()
        return duration
    }

    // MARK: - Private

    private struct Scaled {
        let input: any MTLTexture
        let output: any MTLTexture
        let resamplers: PostProcessShaders
    }

    private struct Masked {
        let mask: Mask
        let working: any MTLTexture
        let pipeline: any MTLRenderPipelineState

        func encode(source: any MTLTexture, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws {
            var parameters = SIMD4<Float>(mask.threshold, mask.channel == .alpha ? 1 : 0, mask.inverse ? 1 : 0, 0)
            try PostProcessShaders.drawFullScreen(
                into: destination, label: "Frostfield post-process mask", commandBuffer: commandBuffer, pipeline: pipeline
            ) { encoder in
                encoder.setFragmentTexture(source, index: 0)
                encoder.setFragmentTexture(working, index: 1)
                encoder.setFragmentTexture(mask.texture, index: 2)
                encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            }
        }
    }

    private struct Step {
        let kind: Kind
        let isFinal: Bool
        let input: any MTLTexture
        let output: any MTLTexture
        var scaled: Scaled?
        var masked: Masked?

        /// The textures the stage itself reads and writes: reduced copies when scaled.
        private var stageInput: any MTLTexture { scaled?.input ?? input }
        private var stageOutput: any MTLTexture { scaled?.output ?? masked?.working ?? output }
        /// Scaled and masked effects write linear intermediates before presentation.
        private var stageIsFinal: Bool { isFinal && scaled == nil && masked == nil }

        func validate(depth: (any MTLTexture)?, commandBuffer: any MTLCommandBuffer) throws {
            if scaled != nil {
                try validateStage(depth: nil, commandBuffer: commandBuffer)
            } else {
                try validateStage(depth: depth, commandBuffer: commandBuffer)
            }
        }

        func encode(depth: (any MTLTexture)?, commandBuffer: any MTLCommandBuffer) throws {
            if let scaled {
                try Self.resample(scaled.resamplers, from: input, to: scaled.input, commandBuffer: commandBuffer)
                try encodeStage(depth: nil, commandBuffer: commandBuffer)
                try Self.resample(scaled.resamplers, from: scaled.output, to: masked?.working ?? output, commandBuffer: commandBuffer)
            } else {
                try encodeStage(depth: depth, commandBuffer: commandBuffer)
            }
            try masked?.encode(source: input, destination: output, commandBuffer: commandBuffer)
        }

        static func resample(
            _ shaders: PostProcessShaders, from source: any MTLTexture, to destination: any MTLTexture,
            commandBuffer: any MTLCommandBuffer
        ) throws {
            let pipeline = try shaders.render("frostfield_resample_fragment", formats: [destination.pixelFormat])
            let downsampling = destination.width < source.width || destination.height < source.height
            var parameters = SIMD4(Float(destination.width), Float(destination.height), downsampling ? 1 : 0, 0)
            try PostProcessShaders.drawFullScreen(
                into: destination, label: "Frostfield post-process resample", commandBuffer: commandBuffer, pipeline: pipeline
            ) { encoder in
                encoder.setFragmentTexture(source, index: 0)
                encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            }
        }

        private func validateStage(depth: (any MTLTexture)?, commandBuffer: any MTLCommandBuffer) throws {
            let input = stageInput
            let output = stageOutput
            let isFinal = stageIsFinal
            switch kind {
            case .graph(let intermediate, let final):
                try (isFinal ? final : intermediate).validate(
                    source: input, depth: depth, destination: output, commandBuffer: commandBuffer
                )
            case .colorGrade(let intermediate, let final, let strength):
                try (isFinal ? final : intermediate).validate(
                    source: input, destination: output, strength: strength, commandBuffer: commandBuffer
                )
            case .screenEffects(let intermediate, let final, let settings):
                try (isFinal ? final : intermediate).prepare(
                    source: input, destination: output, settings: settings, commandBuffer: commandBuffer
                )
            case .toneMapping(let intermediate, let final, let settings):
                try (isFinal ? final : intermediate).check(
                    source: input, destination: output, settings: settings, commandBuffer: commandBuffer
                )
            case .fxaa(let intermediate, let final, let settings):
                try (isFinal ? final : intermediate).validate(
                    source: input, destination: output, settings: settings, commandBuffer: commandBuffer
                )
            case .smaa(let intermediate, let final, _):
                try (isFinal ? final : intermediate).validate(
                    source: input, destination: output, commandBuffer: commandBuffer
                )
            case .pass(let effect):
                try effect.validate(source: input, depth: depth, destination: output, commandBuffer: commandBuffer)
            case .save:
                break  // The chain validates save targets against the complete frame plan.
            }
        }

        private func encodeStage(depth: (any MTLTexture)?, commandBuffer: any MTLCommandBuffer) throws {
            let input = stageInput
            let output = stageOutput
            let isFinal = stageIsFinal
            switch kind {
            case .graph(let intermediate, let final):
                try (isFinal ? final : intermediate).encode(
                    source: input, depth: depth, destination: output, commandBuffer: commandBuffer
                )
            case .colorGrade(let intermediate, let final, let strength):
                try (isFinal ? final : intermediate).encode(
                    source: input, destination: output, strength: strength, commandBuffer: commandBuffer
                )
            case .screenEffects(let intermediate, let final, let settings):
                try (isFinal ? final : intermediate).encode(
                    source: input, destination: output, settings: settings, commandBuffer: commandBuffer
                )
            case .toneMapping(let intermediate, let final, let settings):
                try (isFinal ? final : intermediate).encode(
                    source: input, destination: output, settings: settings, commandBuffer: commandBuffer
                )
            case .fxaa(let intermediate, let final, let settings):
                try (isFinal ? final : intermediate).encode(
                    source: input, destination: output, settings: settings, commandBuffer: commandBuffer
                )
            case .smaa(let intermediate, let final, let quality):
                try (isFinal ? final : intermediate).encode(
                    source: input, destination: output, quality: quality, commandBuffer: commandBuffer
                )
            case .pass(let effect):
                try effect.encode(source: input, depth: depth, destination: output, commandBuffer: commandBuffer)
            case .save:
                try PostProcessChain.copy(source: input, destination: output, commandBuffer: commandBuffer, label: "Frostfield post-process save")
            }
        }
    }

    private func checkIndex(_ index: Int) throws {
        guard stages.indices.contains(index) else { throw Failure.stageIndex(index) }
    }

    private func prepare(_ effect: Effect, reusing existing: Kind?) throws -> Kind {
        let format = destinationPixelFormat
        switch effect {
        case .graph(let graph, let uniforms, let textures):
            let intermediate = try PostProcessPass(
                graph: graph, device: device, pixelFormat: .rgba16Float, uniforms: uniforms, textures: textures
            )
            let final = format == .rgba16Float ? intermediate : try PostProcessPass(
                graph: graph, device: device, pixelFormat: format, uniforms: intermediate.uniforms, textures: textures
            )
            return .graph(intermediate: intermediate, final: final)
        case .colorGrade(let strength):
            guard strength.isFinite, (0...1).contains(strength) else { throw ColorGrade.Failure.invalidStrength }
            if case .colorGrade(let intermediate, let final, _) = existing {
                return .colorGrade(intermediate: intermediate, final: final, strength: strength)
            }
            let intermediate = try ColorGrade(device: device, destinationPixelFormat: .rgba16Float)
            let final = format == .rgba16Float
                ? intermediate : try ColorGrade(device: device, destinationPixelFormat: format)
            return .colorGrade(intermediate: intermediate, final: final, strength: strength)
        case .screenEffects(let settings):
            guard settings.isValid else { throw ScreenEffects.Failure.invalidSettings }
            if case .screenEffects(let intermediate, let final, _) = existing {
                return .screenEffects(intermediate: intermediate, final: final, settings: settings)
            }
            let intermediate = try ScreenEffects(queue: queue, destinationPixelFormat: .rgba16Float)
            let final = format == .rgba16Float
                ? intermediate : try ScreenEffects(queue: queue, destinationPixelFormat: format)
            return .screenEffects(intermediate: intermediate, final: final, settings: settings)
        case .toneMapping(let settings):
            _ = try HDRDisplay.check(settings)
            // Dithering applies where the color is quantized, so only a chain that writes 8 bits can use it.
            if settings.dithering, format == .rgba16Float { throw HDRDisplay.Failure.ditheringNeedsEightBitDestination }
            if case .toneMapping(let intermediate, let final, _) = existing {
                return .toneMapping(intermediate: intermediate, final: final, settings: settings)
            }
            let intermediate = try HDRDisplay(device: device, destinationPixelFormat: .rgba16Float)
            let final = format == .rgba16Float
                ? intermediate : try HDRDisplay(device: device, destinationPixelFormat: format)
            return .toneMapping(intermediate: intermediate, final: final, settings: settings)
        case .fxaa(let settings):
            guard settings.isValid else { throw FXAA.Failure.invalidSettings }
            if case .fxaa(let intermediate, let final, _) = existing {
                return .fxaa(intermediate: intermediate, final: final, settings: settings)
            }
            let intermediate = try FXAA(device: device, destinationPixelFormat: .rgba16Float)
            let final = format == .rgba16Float
                ? intermediate : try FXAA(device: device, destinationPixelFormat: format)
            return .fxaa(intermediate: intermediate, final: final, settings: settings)
        case .smaa(let quality):
            if case .smaa(let intermediate, let final, _) = existing {
                return .smaa(intermediate: intermediate, final: final, quality: quality)
            }
            let intermediate = try SMAA(device: device, destinationPixelFormat: .rgba16Float)
            let final = format == .rgba16Float
                ? intermediate : try SMAA(device: device, destinationPixelFormat: format)
            return .smaa(intermediate: intermediate, final: final, quality: quality)
        case .pass(let effect):
            guard effect.device === device else { throw Failure.wrongDevice }
            try effect.prepare(pixelFormat: .rgba16Float)
            if format != .rgba16Float { try effect.prepare(pixelFormat: format) }
            return .pass(effect)
        case .save(let target):
            guard target.device === device else { throw Failure.wrongDevice }
            guard PostProcessPass.isReadable(target, format: .rgba16Float) else { throw Failure.saveDestination }
            return .save(target)
        }
    }

    private func scaledPair(width: Int, height: Int) throws -> [any MTLTexture] {
        let key = SIMD2(width, height)
        if let pair = scaledIntermediates[key] { return pair }
        let pair = try (0..<2).map { slot -> any MTLTexture in
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.usage = [.renderTarget, .shaderRead]
            descriptor.hazardTrackingMode = .tracked
            guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.intermediateAllocation }
            texture.label = "Frostfield post-process scaled \(slot)"
            intermediateAllocationCount += 1
            return texture
        }
        // Keep the cache bounded to the sizes in use.
        if scaledIntermediates.count >= 8 { scaledIntermediates.removeAll() }
        scaledIntermediates[key] = pair
        return pair
    }

    private func intermediates(count: Int, width: Int, height: Int) throws -> [any MTLTexture] {
        if let first = intermediates.first, first.width != width || first.height != height {
            intermediates.removeAll()
        }
        while intermediates.count < count {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.usage = [.renderTarget, .shaderRead]
            descriptor.hazardTrackingMode = .tracked
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw Failure.intermediateAllocation
            }
            texture.label = "Frostfield post-process intermediate \(intermediates.count)"
            intermediates.append(texture)
            intermediateAllocationCount += 1
        }
        return intermediates
    }

    private static func copy(
        source: any MTLTexture, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        label: String = "Frostfield post-process bypass"
    ) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
        blit.label = label
        blit.copy(
            from: source, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: source.width, height: source.height, depth: 1),
            to: destination, destinationSlice: 0, destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
    }

    public enum Failure: LocalizedError, Equatable {
        case destinationFormat
        case tooManyStages(limit: Int)
        case stageIndex(Int)
        case commandBuffer
        case wrongDevice
        case source
        case destination
        case depth
        case missingDepth(stage: Int)
        case sizeMismatch
        case bypassCopy
        case intermediateAllocation
        case encoder
        case resolutionScale
        case scaledDepth(stage: Int)
        case scaledSave(stage: Int)
        case mask
        case saveDestination
        case commandFailed(String?)
        case timingUnavailable

        public var errorDescription: String? {
            switch self {
            case .destinationFormat: "A post-process chain writes RGBA16Float, BGRA8Unorm_sRGB or RGBA8Unorm_sRGB."
            case .tooManyStages(let limit): "A post-process chain may contain at most \(limit) stages."
            case .stageIndex(let index): "The post-process chain has no stage at index \(index)."
            case .commandBuffer:
                "Use an uncommitted command buffer from the chain's queue that retains its references."
            case .wrongDevice: "Post-process textures must use the chain's Metal device."
            case .source: "The chain source must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
            case .destination:
                "The chain destination must be a separate single-sample 2D render target in the chain's pixel format."
            case .depth: "Chain depth must be a shader-readable, tracked, single-sample 2D Depth32Float texture."
            case .missingDepth(let stage): "Stage \(stage) samples scene depth. Pass the scene's Depth32Float texture."
            case .sizeMismatch: "Chain color, depth, mask and save textures must match the working frame dimensions."
            case .bypassCopy:
                "With every stage bypassed, the chain copies its source and needs an RGBA16Float destination that is not framebuffer-only."
            case .intermediateAllocation: "Metal could not allocate the post-process intermediates."
            case .encoder: "Metal could not create a post-process encoder."
            case .resolutionScale: "A stage's resolution scale must be from 0.125 to 1."
            case .scaledDepth(let stage): "Stage \(stage) reads scene depth and must run at full resolution."
            case .scaledSave(let stage): "Save stage \(stage) must use the full working resolution."
            case .mask: "Use a separate, tracked linear mask texture and a threshold from 0 to 1."
            case .saveDestination: "The save target must use separate tracked, shader-readable RGBA16Float storage."
            case .commandFailed(let reason): reason ?? "Metal did not complete the timed post-process chain."
            case .timingUnavailable: "Metal did not provide GPU execution times for the post-process chain."
            }
        }
    }
}
