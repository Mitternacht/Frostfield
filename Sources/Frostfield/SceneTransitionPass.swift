import Metal

/// Composites two resolved scene frames in premultiplied linear color.
/// Render each scene with its own camera into a caller-owned RGBA16Float
/// texture, then encode this pass on the same queue. In a post-process chain,
/// the previous stage's output is the first frame.
public final class SceneTransitionPass: PostProcessEffect {
    public struct Settings: Equatable, Sendable {
        /// From 0 for the first frame to 1 for the second frame.
        public var progress: Float
        /// Mask blend width in 0...1. Zero makes a hard cut at the mask value.
        public var threshold: Float
        /// Plays the same transition backward, including its endpoints.
        public var reversed: Bool
        /// Reveals high red mask values before low values. The default reveals low values first.
        public var invertMask: Bool

        public init(progress: Float = 0, threshold: Float = 0.1, reversed: Bool = false, invertMask: Bool = false) {
            self.progress = progress
            self.threshold = threshold
            self.reversed = reversed
            self.invertMask = invertMask
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    /// The second resolved premultiplied linear RGBA16Float frame.
    public var toTexture: any MTLTexture
    /// Optional linear red-channel mask. Its dimensions may differ from the frames.
    /// The mask is sampled bilinearly in top-left normalized coordinates, clamped to its edge.
    public var mask: (any MTLTexture)?
    public var settings: Settings
    private let shaders: PostProcessShaders

    public init(
        device: any MTLDevice, toTexture: any MTLTexture, mask: (any MTLTexture)? = nil, settings: Settings = .init()
    ) throws {
        self.device = device
        self.toTexture = toTexture
        self.mask = mask
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render(Self.fade, formats: [pixelFormat])
        _ = try shaders.render(Self.masked, formats: [pixelFormat])
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard (0...1).contains(settings.progress), (0...1).contains(settings.threshold) else {
            throw PostProcessEffectError.invalidSettings("Transition progress and threshold must be from 0 to 1.")
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination,
            commandBuffer: commandBuffer, requiresDepth: false
        )
        guard toTexture.device === device, mask.map({ $0.device === device }) ?? true else {
            throw PostProcessEffectError.wrongDevice
        }
        guard PostProcessPass.isReadable(toTexture, format: .rgba16Float) else {
            throw PostProcessEffectError.unsupportedInput(
                "The second transition frame must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
            )
        }
        guard toTexture.width == source.width, toTexture.height == source.height else {
            throw PostProcessEffectError.sizeMismatch
        }
        if let mask {
            let valid = PhysicallyBasedMaterial.isSurfaceTexture(mask, content: .data)
                && mask.usage.contains(.shaderRead) && mask.storageMode != .memoryless
                && mask.hazardTrackingMode != .untracked
                && PostProcessPass.isFloatSampleable(mask.pixelFormat, linear: true, device: device)
            guard valid else {
                throw PostProcessEffectError.unsupportedInput("The transition mask must be a shader-readable, tracked, linear 2D texture.")
            }
        }
        guard !TextureDebugPass.aliases(source, destination), !TextureDebugPass.aliases(toTexture, destination),
              mask.map({ !TextureDebugPass.aliases($0, destination) }) ?? true else {
            throw PostProcessEffectError.destination
        }
    }

    /// Encodes one draw without submitting it. Source textures may share
    /// storage with one another. The destination must use separate storage.
    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render(mask == nil ? Self.fade : Self.masked, formats: [destination.pixelFormat])
        let progress = settings.reversed ? 1 - settings.progress : settings.progress
        var parameters = SIMD4<Float>(progress, settings.threshold, settings.invertMask ? 1 : 0, 0)
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield scene transition", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(toTexture, index: 1)
            if let mask { encoder.setFragmentTexture(mask, index: 2) }
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        }
    }

    private static let fade = "frostfield_transition_fade"
    private static let masked = "frostfield_transition_masked"
}

extension SceneTransitionPass: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}

public extension ShaderGraphBuilder {
    /// Blends two premultiplied frames, optionally revealing the second through a
    /// red-channel mask. Supply sampled RGBA and scalar values so the graph can
    /// choose each texture's UVs. Progress and threshold are clamped to 0...1.
    /// A nil mask gives a crossfade. Zero threshold cuts where mask ≤ progress.
    func transition(
        from: ShaderValue<SIMD4<Float>>, to: ShaderValue<SIMD4<Float>>, progress: ShaderValue<Float>,
        maskRed: ShaderValue<Float>? = nil, threshold: ShaderValue<Float>? = nil,
        reversed: Bool = false, invertMask: Bool = false
    ) -> ShaderValue<SIMD4<Float>> {
        let zero = constant(Float(0))
        let one = constant(Float(1))
        let fraction = reversed ? (1 - progress).saturated() : progress.saturated()
        let weight: ShaderValue<Float>
        if let maskRed {
            let mask = invertMask ? 1 - maskRed.saturated() : maskRed.saturated()
            let softness = (threshold ?? constant(Float(0.1))).saturated()
            let feathered = softness.greaterThan(zero)
            let divisor = select(feathered, softness, one)
            let ramp = ((fraction * (1 + 2 * softness) - mask) / divisor).saturated()
            weight = select(feathered, ramp, step(mask, fraction))
        } else {
            weight = fraction
        }
        let blended = from.mix(to, by: weight)
        return select(fraction.greaterThan(zero), select(fraction.lessThan(one), blended, to), from)
    }
}
