import Metal

/// Screens the RGB channels of a linear premultiplied frame on independently
/// rotated grids. Use directly or as a `PostProcessChain.Effect.pass(_:)`.
public final class HalftonePass: PostProcessEffect {
    public enum Shape: UInt32, Sendable, Hashable, CaseIterable {
        case dot = 1, ellipse, line, square, diamond
    }

    public enum BlendMode: UInt32, Sendable, Hashable, CaseIterable {
        /// Interpolates from the source to the screen.
        case linear = 1
        /// Other modes interpolate from the screen to the indicated combination.
        case multiply, add, lighter, darker
    }

    public struct Settings: Sendable, Hashable {
        public var shape: Shape
        /// Screen grid spacing and maximum dot radius, in pixels, from 0.25 to 256.
        public var radius: Float
        /// RGB grid angles in radians, measured counterclockwise from the positive x axis.
        public var rotations: SIMD3<Float>
        /// Deterministic positional scatter, from zero to one.
        public var scatter: Float
        /// Blend fraction in the chosen mode, from zero to one.
        public var blending: Float
        public var blendMode: BlendMode
        /// Uses the arithmetic mean of the resulting RGB channels.
        public var grayscale: Bool
        /// False copies the source exactly.
        public var isEnabled: Bool

        public init(
            shape: Shape = .dot, radius: Float = 4,
            rotations: SIMD3<Float> = SIMD3(.pi / 12, .pi / 6, .pi / 4),
            scatter: Float = 0, blending: Float = 1, blendMode: BlendMode = .linear,
            grayscale: Bool = false, isEnabled: Bool = true
        ) {
            self.shape = shape
            self.radius = radius
            self.rotations = rotations
            self.scatter = scatter
            self.blending = blending
            self.blendMode = blendMode
            self.grayscale = grayscale
            self.isEnabled = isEnabled
        }

        fileprivate func check() throws {
            guard (0.25...256).contains(radius), (0...1).contains(scatter), (0...1).contains(blending),
                  rotations.x.isFinite, rotations.y.isFinite, rotations.z.isFinite else {
                throw PostProcessEffectError.invalidSettings(
                    "Halftone needs a radius from 0.25 to 256, finite rotations, and normalized scatter and blending."
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
        var rotationsAndRadius: SIMD4<Float>
        var scatterAndBlend: SIMD2<Float>
        var shape: UInt32
        var blendMode: UInt32
        var grayscale: UInt32
        var enabled: UInt32
    }

    public init(device: any MTLDevice, settings: Settings = .init()) throws {
        try settings.check()
        self.device = device
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_halftone_fragment", formats: [pixelFormat])
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

    /// Encodes one full-screen draw. The screen and blend use the SDR portion
    /// of straight RGB. Signed HDR excess and the original pixel alpha are retained.
    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render("frostfield_halftone_fragment", formats: [destination.pixelFormat])
        let rotations = settings.rotations
        let turn = Float.pi * 2
        var parameters = Parameters(
            rotationsAndRadius: SIMD4(
                rotations.x.truncatingRemainder(dividingBy: turn), rotations.y.truncatingRemainder(dividingBy: turn),
                rotations.z.truncatingRemainder(dividingBy: turn), settings.radius
            ),
            scatterAndBlend: SIMD2(settings.scatter, settings.blending),
            shape: settings.shape.rawValue, blendMode: settings.blendMode.rawValue,
            grayscale: settings.grayscale ? 1 : 0, enabled: settings.isEnabled ? 1 : 0
        )
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield halftone", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
    }
}

extension HalftonePass: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
