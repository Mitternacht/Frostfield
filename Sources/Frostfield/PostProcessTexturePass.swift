import Metal

/// Draws a full-screen color texture. At opacity one the texture replaces the
/// frame, including its alpha. Lower opacity composites it over the frame in
/// premultiplied linear color. No depth is read or written.
public final class PostProcessTexturePass: PostProcessEffect {
    public enum AlphaMode: Sendable {
        /// The RGB values already include alpha in linear space.
        case premultiplied
        /// The pass multiplies the RGB of each texel by its alpha before filtering.
        case straight
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    /// A tracked, shader-readable RGBA16Float, RGBA8Unorm, or BGRA8Unorm 2D texture.
    /// The 8-bit sRGB variants are decoded to linear RGB by Metal.
    /// The pass samples mip zero in top-left normalized coordinates, with
    /// bilinear filtering and edge clamp. The texture may differ in size.
    public var texture: any MTLTexture
    public var alphaMode: AlphaMode
    /// A factor from zero to one. Zero preserves the incoming frame.
    public var opacity: Float
    private let shaders: PostProcessShaders

    public init(
        device: any MTLDevice, texture: any MTLTexture, opacity: Float = 1, alphaMode: AlphaMode = .premultiplied
    ) throws {
        self.device = device
        self.texture = texture
        self.opacity = opacity
        self.alphaMode = alphaMode
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render(Self.fragment, formats: [pixelFormat])
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination,
            commandBuffer: commandBuffer, requiresDepth: false
        )
        guard (0...1).contains(opacity) else {
            throw PostProcessEffectError.invalidSettings("Texture pass opacity must be from 0 to 1.")
        }
        guard texture.device === device else { throw PostProcessEffectError.wrongDevice }
        let format = texture.pixelFormat
        guard Self.sourceFormats.contains(format), PostProcessPass.isReadable(texture, format: format) else {
            throw PostProcessEffectError.unsupportedInput(
                "The overlay must be a shader-readable, tracked, single-sample 2D RGBA16Float or 8-bit RGBA/BGRA texture."
            )
        }
        guard !TextureTransfer.sharesStorage(source, destination), !TextureTransfer.sharesStorage(texture, destination) else {
            throw PostProcessEffectError.destination
        }
    }

    /// Encodes one draw without submitting it. The overlay and source may
    /// share storage. The destination must use separate storage.
    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render(Self.fragment, formats: [destination.pixelFormat])
        var parameters = SIMD4(Float(destination.width), Float(destination.height), opacity, alphaMode == .straight ? 1 : 0)
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield post-process texture", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(texture, index: 1)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        }
    }

    private static let fragment = "frostfield_post_process_texture"
    private static let sourceFormats: Set<MTLPixelFormat> = [.rgba16Float, .rgba8Unorm, .bgra8Unorm, .rgba8Unorm_srgb, .bgra8Unorm_srgb]
}

extension PostProcessTexturePass: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
