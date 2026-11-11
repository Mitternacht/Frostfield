import Metal

/// A screen-space radial streak effect in linear premultiplied RGBA. It adds
/// weighted samples toward a screen position. Use it directly or with
/// `PostProcessChain.Effect.pass(_:)`.
public final class RadialBlur: PostProcessEffect {
    public struct Settings: Sendable, Hashable {
        /// The ray target in top-left normalized screen coordinates, within 0...1 on both axes.
        public var center: SIMD2<Float>
        /// A fraction of the pixel-to-center distance, in 0...4. One gives the
        /// Three.js r187 travel distance, including its sub-step jitter. Zero bypasses.
        public var radius: Float
        /// The number of taps along the ray, in 1...64.
        public var sampleCount: Int
        /// The first tap's weight, in 0...1. Zero bypasses.
        public var weight: Float
        /// The ratio between consecutive tap weights, in 0...1.
        public var decay: Float
        /// Scales the added light, in 0...64. Zero bypasses.
        public var exposure: Float

        public init(
            center: SIMD2<Float> = SIMD2(repeating: 0.5), radius: Float = 1, sampleCount: Int = 32,
            weight: Float = 0.9, decay: Float = 0.95, exposure: Float = 5
        ) {
            self.center = center
            self.radius = radius
            self.sampleCount = sampleCount
            self.weight = weight
            self.decay = decay
            self.exposure = exposure
        }

        fileprivate func check() throws {
            guard (0...1).contains(center.x), (0...1).contains(center.y),
                  (0...4).contains(radius), (1...64).contains(sampleCount),
                  (0...1).contains(weight), (0...1).contains(decay), (0...64).contains(exposure) else {
                throw PostProcessEffectError.invalidSettings(
                    "Radial blur needs center, weight and decay from 0 to 1, radius from 0 to 4, 1 to 64 samples, and exposure from 0 to 64."
                )
            }
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    /// Edits take effect on the next encode and reuse the prepared pipeline.
    public var settings: Settings
    private let shaders: PostProcessShaders

    private struct Parameters {
        var center: SIMD2<Float>
        var radius: Float
        var weight: Float
        var decay: Float
        var exposure: Float
        var sampleCount: UInt32
    }

    public init(device: any MTLDevice, settings: Settings = .init()) throws {
        try settings.check()
        self.device = device
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_radial_blur_fragment", formats: [pixelFormat])
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try settings.check()
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination,
            commandBuffer: commandBuffer, requiresDepth: false
        )
        guard !TextureTransfer.sharesStorage(source, destination) else { throw PostProcessEffectError.destination }
    }

    /// Encodes one full-screen draw on the caller's command buffer. Taps use
    /// bilinear filtering with edge clamp. The added alpha saturates to one;
    /// HDR RGB remains linear within the half-float range.
    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render("frostfield_radial_blur_fragment", formats: [destination.pixelFormat])
        var parameters = Parameters(
            center: settings.center, radius: settings.radius, weight: settings.weight,
            decay: settings.decay, exposure: settings.exposure, sampleCount: UInt32(settings.sampleCount)
        )
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield radial blur", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
    }
}

extension RadialBlur: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
