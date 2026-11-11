import Foundation
import Metal

/// Fast approximate anti-aliasing: the quality path of Timothy Lottes' FXAA 3.11,
/// like Three.js `FXAAPass` and the `fxaa` TSL node.
///
/// FXAA finds luma edges in one image and blends each edge pixel with its
/// neighbor across the edge, by an amount estimated from the distance to the
/// edge's ends, plus a sub-pixel term for isolated detail. Run it on
/// display-referred color, after tone mapping: render `HDRDisplay` into an
/// RGBA16Float texture and run FXAA from that into the drawable. On scene-linear
/// HDR color, bright highlights dominate the luma range and edges are missed.
///
/// The pass uses two draws: one writes perceptual luma (the square root of Rec.709
/// luminance) into an R16Float texture that this object owns, and one resolves the
/// edges. Premultiplied RGBA is filtered as one value, so alpha edges are smoothed
/// with color. The object is not thread-safe; serialize its encodes.
public final class FXAA {
    /// FXAA 3.11 quality presets, which set the edge-end search steps.
    public enum Quality: Sendable, Hashable {
        /// Preset 12, the FXAA 3.11 default: 5 search steps reaching 20.5 pixels.
        case medium
        /// Preset 29: 12 search steps reaching 30.5 pixels.
        case high
        /// Preset 39: 12 search steps reaching 26.5 pixels, with finer near steps.
        case extreme

        var steps: [Float] {
            switch self {
            case .medium: [1, 1.5, 2, 4, 12]
            case .high: [1, 1.5, 2, 2, 2, 2, 2, 2, 2, 2, 4, 8]
            case .extreme: [1, 1, 1, 1, 1, 1.5, 2, 2, 2, 2, 4, 8]
            }
        }
    }

    /// FXAA 3.11's quality parameters. The thresholds default to FXAA's defaults;
    /// the preset defaults to `.high` (29) rather than FXAA's 12.
    public struct Settings: Sendable, Hashable {
        public var quality: Quality
        /// `fxaaQualitySubpix`, 0...1: how much isolated, sub-pixel detail is
        /// softened. 0 turns it off; 0.75 is FXAA's default; 1 is softest.
        public var subpixelBlending: Float
        /// `fxaaQualityEdgeThreshold`, 0.031...0.5: the local contrast, relative to
        /// the brightest neighbor, needed to process a pixel. 0.166 is the default;
        /// 0.125 is high quality; 0.063 is FXAA's "overkill".
        public var edgeThreshold: Float
        /// `fxaaQualityEdgeThresholdMin`, 0...0.25: skips darks below this luma
        /// contrast. 0.0833 is FXAA's upper limit and default; 0.0312 is its
        /// visible limit.
        public var edgeThresholdMinimum: Float

        public init(
            quality: Quality = .high,
            subpixelBlending: Float = 0.75,
            edgeThreshold: Float = 0.166,
            edgeThresholdMinimum: Float = 0.0833
        ) {
            self.quality = quality
            self.subpixelBlending = subpixelBlending
            self.edgeThreshold = edgeThreshold
            self.edgeThresholdMinimum = edgeThresholdMinimum
        }

        var isValid: Bool {
            (0...1).contains(subpixelBlending) && (0.031...0.5).contains(edgeThreshold)
                && (0...0.25).contains(edgeThresholdMinimum)
        }
    }

    /// Linear RGBA16Float for further processing or presentation with extended
    /// range, or an 8-bit sRGB format, where Metal encodes the linear result.
    public static let supportedPixelFormats: [MTLPixelFormat] = [.rgba16Float, .bgra8Unorm_srgb, .rgba8Unorm_srgb]

    public let device: any MTLDevice
    public let destinationPixelFormat: MTLPixelFormat
    private let lumaPipeline: any MTLRenderPipelineState
    private let resolvePipeline: any MTLRenderPipelineState
    private var luma: (any MTLTexture)?

    /// Matches `FXAAOptions` in AntiAliasing.metal.
    private struct ShaderOptions {
        var texelSize: SIMD2<Float>
        var subpixel: Float
        var edgeThreshold: Float
        var edgeThresholdMinimum: Float
        var stepCount: UInt32
        var steps: (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float)
    }

    /// Prepares both pipelines from Frostfield's precompiled shaders.
    public init(device: any MTLDevice, destinationPixelFormat: MTLPixelFormat = .rgba16Float) throws {
        guard Self.supportedPixelFormats.contains(destinationPixelFormat) else { throw Failure.destinationFormat }
        self.device = device
        self.destinationPixelFormat = destinationPixelFormat
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_aa_vertex"),
                  let lumaFragment = library.makeFunction(name: "frostfield_fxaa_luma_fragment"),
                  let resolveFragment = library.makeFunction(name: "frostfield_fxaa_fragment") else {
                throw Failure.missingShader
            }
            let lumaDescriptor = MTLRenderPipelineDescriptor()
            lumaDescriptor.label = "Frostfield FXAA luma"
            lumaDescriptor.vertexFunction = vertex
            lumaDescriptor.fragmentFunction = lumaFragment
            lumaDescriptor.colorAttachments[0].pixelFormat = .r16Float
            lumaPipeline = try device.makeRenderPipelineState(descriptor: lumaDescriptor)
            let resolveDescriptor = MTLRenderPipelineDescriptor()
            resolveDescriptor.label = "Frostfield FXAA"
            resolveDescriptor.vertexFunction = vertex
            resolveDescriptor.fragmentFunction = resolveFragment
            resolveDescriptor.colorAttachments[0].pixelFormat = destinationPixelFormat
            resolvePipeline = try device.makeRenderPipelineState(descriptor: resolveDescriptor)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Encodes the luma and edge passes on the caller's command buffer without
    /// submitting it.
    ///
    /// The source is a shader-readable, tracked, single-sample 2D RGBA16Float
    /// texture of premultiplied, display-referred linear color. The destination is a
    /// separate, equal-sized single-sample render target in `destinationPixelFormat`.
    /// Use an uncommitted command buffer that retains its references. Validation
    /// happens before anything is encoded. The luma texture is reallocated when the
    /// size changes; earlier command buffers keep the old one alive.
    public func encode(
        source: any MTLTexture,
        destination: any MTLTexture,
        settings: Settings = Settings(),
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, destination: destination, settings: settings, commandBuffer: commandBuffer)
        let luma = try lumaTexture(width: source.width, height: source.height)

        let lumaPass = MTLRenderPassDescriptor()
        lumaPass.colorAttachments[0].texture = luma
        lumaPass.colorAttachments[0].loadAction = .dontCare
        lumaPass.colorAttachments[0].storeAction = .store
        guard let lumaEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: lumaPass) else {
            throw Failure.encoder
        }
        lumaEncoder.label = "Frostfield FXAA luma"
        lumaEncoder.setRenderPipelineState(lumaPipeline)
        lumaEncoder.setFragmentTexture(source, index: 0)
        lumaEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        lumaEncoder.endEncoding()

        var steps = settings.quality.steps
        let stepCount = steps.count
        steps += Array(repeating: 0, count: 12 - steps.count)
        var options = ShaderOptions(
            texelSize: SIMD2(1 / Float(source.width), 1 / Float(source.height)),
            subpixel: settings.subpixelBlending,
            edgeThreshold: settings.edgeThreshold,
            edgeThresholdMinimum: settings.edgeThresholdMinimum,
            stepCount: UInt32(stepCount),
            steps: (steps[0], steps[1], steps[2], steps[3], steps[4], steps[5],
                    steps[6], steps[7], steps[8], steps[9], steps[10], steps[11])
        )
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield FXAA"
        encoder.setRenderPipelineState(resolvePipeline)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(luma, index: 1)
        encoder.setFragmentBytes(&options, length: MemoryLayout<ShaderOptions>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    /// Checks everything `encode` requires without encoding.
    func validate(
        source: any MTLTexture, destination: any MTLTexture, settings: Settings, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard settings.isValid else { throw Failure.invalidSettings }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device, source.device === device, destination.device === device else {
            throw Failure.wrongDevice
        }
        guard PostProcessPass.isReadable(source, format: .rgba16Float), source.width > 0, source.height > 0 else {
            throw Failure.source
        }
        guard destination !== source, destination.pixelFormat == destinationPixelFormat,
              destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless else {
            throw Failure.destination
        }
        guard destination.width == source.width, destination.height == source.height else {
            throw Failure.sizeMismatch
        }
    }

    private func lumaTexture(width: Int, height: Int) throws -> any MTLTexture {
        if let luma, luma.width == width, luma.height == height { return luma }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
        texture.label = "Frostfield FXAA luma"
        luma = texture
        return texture
    }

    public enum Failure: LocalizedError, Equatable {
        case destinationFormat
        case missingShader
        case pipeline(String)
        case invalidSettings
        case commandBuffer
        case wrongDevice
        case source
        case destination
        case sizeMismatch
        case allocation
        case encoder

        public var errorDescription: String? {
            switch self {
            case .destinationFormat: "FXAA renders to RGBA16Float, BGRA8Unorm_sRGB or RGBA8Unorm_sRGB."
            case .missingShader: "Frostfield's FXAA shaders are missing."
            case .pipeline(let detail): "Could not prepare FXAA. \(detail)"
            case .invalidSettings:
                "FXAA needs sub-pixel blending in 0...1, an edge threshold in 0.031...0.5 and a minimum threshold in 0...0.25."
            case .commandBuffer: "FXAA needs an uncommitted command buffer that retains its references."
            case .wrongDevice: "FXAA textures and the command buffer must use its Metal device."
            case .source: "The FXAA source must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
            case .destination:
                "The FXAA destination must be a separate single-sample 2D render target in the prepared pixel format."
            case .sizeMismatch: "The FXAA source and destination must have the same dimensions."
            case .allocation: "Metal could not allocate the FXAA luma texture."
            case .encoder: "Metal could not create an FXAA encoder."
            }
        }
    }
}
