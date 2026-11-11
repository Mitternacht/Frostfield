import Foundation
import Metal

/// Subpixel morphological anti-aliasing, SMAA 1x (Jimenez et al. 2012), like
/// Three.js `SMAAPass` and the `smaa` TSL node.
///
/// Three passes run on display-referred color: luma edge detection on the
/// sRGB-encoded image, blending weights from the edge patterns found by searching
/// along each edge (with diagonal and corner detection at high quality), and
/// neighborhood blending. The area and search lookup tables are generated in code
/// by a port of the reference generator scripts, and equal the published tables
/// for SMAA 1x byte for byte; nothing is loaded from disk.
///
/// Run it after tone mapping on an RGBA16Float image, like `FXAA`. Values above
/// SDR white, as `HDRDisplay` writes with a display headroom, are encoded on the
/// same sRGB curve continued above 1, so edges between EDR highlights are found
/// with the preset's threshold. SMAA keeps straight, long edges and flat regions
/// sharper than FXAA, and at high and ultra quality recovers diagonal lines and
/// keeps corners sharp. It needs more passes and two intermediate textures, which
/// this object owns and reallocates when the size changes. The object is not
/// thread-safe; serialize its encodes.
public final class SMAA {
    /// SMAA's presets from SMAA.hlsl.
    public enum Quality: Sendable, Hashable {
        /// Threshold 0.15, 4 search steps, no diagonal or corner detection.
        case low
        /// Threshold 0.1, 8 search steps, no diagonal or corner detection.
        case medium
        /// Threshold 0.1, 16 search steps, 8 diagonal steps, 25% corner rounding.
        case high
        /// Threshold 0.05, 32 search steps, 16 diagonal steps, 25% corner rounding.
        case ultra
    }

    /// Linear RGBA16Float for further processing or presentation with extended
    /// range, or an 8-bit sRGB format, where Metal encodes the linear result.
    public static let supportedPixelFormats: [MTLPixelFormat] = [.rgba16Float, .bgra8Unorm_srgb, .rgba8Unorm_srgb]

    public let device: any MTLDevice
    public let destinationPixelFormat: MTLPixelFormat
    private let edgePipeline: any MTLRenderPipelineState
    private let weightPipeline: any MTLRenderPipelineState
    private let blendPipeline: any MTLRenderPipelineState
    private let areaTexture: any MTLTexture
    private let searchTexture: any MTLTexture
    private var scratch: (edges: any MTLTexture, weights: any MTLTexture)?

    // Generated once per process: under 10 ms in optimized builds.
    private static let areaBytes = SMAALookupTextures.areaBytes()
    private static let searchBytes = SMAALookupTextures.searchBytes()

    /// Matches `SMAAOptions` in SMAA.metal.
    private struct ShaderOptions {
        var metrics: SIMD4<Float>
        var threshold: Float
        var maxSearchSteps: Float
        var maxSearchStepsDiagonal: Float
        var cornerRounding: Float
        var flags: UInt32
        var padding0: Float = 0
        var padding1: Float = 0
        var padding2: Float = 0
    }

    /// Prepares the three pipelines and uploads the generated lookup tables.
    public init(device: any MTLDevice, destinationPixelFormat: MTLPixelFormat = .rgba16Float) throws {
        guard Self.supportedPixelFormats.contains(destinationPixelFormat) else { throw Failure.destinationFormat }
        self.device = device
        self.destinationPixelFormat = destinationPixelFormat
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            func pipeline(_ fragment: String, format: MTLPixelFormat, label: String) throws -> any MTLRenderPipelineState {
                guard let vertexFunction = library.makeFunction(name: "frostfield_smaa_vertex"),
                      let fragmentFunction = library.makeFunction(name: fragment) else { throw Failure.missingShader }
                let descriptor = MTLRenderPipelineDescriptor()
                descriptor.label = label
                descriptor.vertexFunction = vertexFunction
                descriptor.fragmentFunction = fragmentFunction
                descriptor.colorAttachments[0].pixelFormat = format
                return try device.makeRenderPipelineState(descriptor: descriptor)
            }
            edgePipeline = try pipeline("frostfield_smaa_edges_fragment", format: .rg8Unorm, label: "Frostfield SMAA edges")
            weightPipeline = try pipeline("frostfield_smaa_weights_fragment", format: .rgba8Unorm, label: "Frostfield SMAA weights")
            blendPipeline = try pipeline("frostfield_smaa_blend_fragment", format: destinationPixelFormat, label: "Frostfield SMAA")
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        areaTexture = try Self.lookupTexture(
            device: device, format: .rg8Unorm, width: SMAALookupTextures.areaWidth,
            height: SMAALookupTextures.areaHeight, bytes: Self.areaBytes, label: "Frostfield SMAA area"
        )
        searchTexture = try Self.lookupTexture(
            device: device, format: .r8Unorm, width: SMAALookupTextures.searchWidth,
            height: SMAALookupTextures.searchHeight, bytes: Self.searchBytes, label: "Frostfield SMAA search"
        )
    }

    /// Encodes the three passes on the caller's command buffer without submitting it.
    ///
    /// The source is a shader-readable, tracked, single-sample 2D RGBA16Float
    /// texture of premultiplied, display-referred linear color. The destination is a
    /// separate, equal-sized single-sample render target in `destinationPixelFormat`.
    /// Use an uncommitted command buffer that retains its references. Validation
    /// happens before anything is encoded.
    public func encode(
        source: any MTLTexture,
        destination: any MTLTexture,
        quality: Quality = .high,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, destination: destination, commandBuffer: commandBuffer)
        let (edges, weights) = try scratchTextures(width: source.width, height: source.height)
        var options = Self.options(quality: quality, width: source.width, height: source.height)

        func draw(
            _ pipeline: any MTLRenderPipelineState, into target: any MTLTexture, clear: Bool,
            textures: [any MTLTexture], label: String
        ) throws {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = clear ? .clear : .dontCare
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
            encoder.label = label
            encoder.setRenderPipelineState(pipeline)
            for (index, texture) in textures.enumerated() { encoder.setFragmentTexture(texture, index: index) }
            encoder.setFragmentBytes(&options, length: MemoryLayout<ShaderOptions>.stride, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
        }
        // Edge detection discards pixels without edges, so its target starts cleared.
        try draw(edgePipeline, into: edges, clear: true, textures: [source], label: "Frostfield SMAA edges")
        try draw(weightPipeline, into: weights, clear: false, textures: [edges, areaTexture, searchTexture], label: "Frostfield SMAA weights")
        try draw(blendPipeline, into: destination, clear: false, textures: [source, weights], label: "Frostfield SMAA")
    }

    /// Checks everything `encode` requires without encoding.
    func validate(source: any MTLTexture, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws {
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

    private static func options(quality: Quality, width: Int, height: Int) -> ShaderOptions {
        let (threshold, steps, diagonalSteps, corners): (Float, Float, Float, Bool) = switch quality {
        case .low: (0.15, 4, 0, false)
        case .medium: (0.1, 8, 0, false)
        case .high: (0.1, 16, 8, true)
        case .ultra: (0.05, 32, 16, true)
        }
        return ShaderOptions(
            metrics: SIMD4(1 / Float(width), 1 / Float(height), Float(width), Float(height)),
            threshold: threshold, maxSearchSteps: steps, maxSearchStepsDiagonal: diagonalSteps,
            cornerRounding: 0.25, flags: (diagonalSteps > 0 ? 1 : 0) | (corners ? 2 : 0)
        )
    }

    private func scratchTextures(width: Int, height: Int) throws -> (any MTLTexture, any MTLTexture) {
        if let scratch, scratch.edges.width == width, scratch.edges.height == height { return scratch }
        func make(_ format: MTLPixelFormat, _ label: String) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.usage = [.renderTarget, .shaderRead]
            guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
            texture.label = label
            return texture
        }
        let made = (edges: try make(.rg8Unorm, "Frostfield SMAA edges"), weights: try make(.rgba8Unorm, "Frostfield SMAA weights"))
        scratch = made
        return made
    }

    private static func lookupTexture(
        device: any MTLDevice, format: MTLPixelFormat, width: Int, height: Int, bytes: [UInt8], label: String
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
        texture.label = label
        let bytesPerRow = bytes.count / height
        bytes.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: base, bytesPerRow: bytesPerRow)
            }
        }
        return texture
    }

    public enum Failure: LocalizedError, Equatable {
        case destinationFormat
        case missingShader
        case pipeline(String)
        case commandBuffer
        case wrongDevice
        case source
        case destination
        case sizeMismatch
        case allocation
        case encoder

        public var errorDescription: String? {
            switch self {
            case .destinationFormat: "SMAA renders to RGBA16Float, BGRA8Unorm_sRGB or RGBA8Unorm_sRGB."
            case .missingShader: "Frostfield's SMAA shaders are missing."
            case .pipeline(let detail): "Could not prepare SMAA. \(detail)"
            case .commandBuffer: "SMAA needs an uncommitted command buffer that retains its references."
            case .wrongDevice: "SMAA textures and the command buffer must use its Metal device."
            case .source: "The SMAA source must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
            case .destination:
                "The SMAA destination must be a separate single-sample 2D render target in the prepared pixel format."
            case .sizeMismatch: "The SMAA source and destination must have the same dimensions."
            case .allocation: "Metal could not allocate SMAA's textures."
            case .encoder: "Metal could not create an SMAA encoder."
            }
        }
    }
}
