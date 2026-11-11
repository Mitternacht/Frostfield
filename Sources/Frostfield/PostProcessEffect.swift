import Foundation
import Metal

/// A prepared full-screen effect that runs on its own or as a `PostProcessChain`
/// stage through `PostProcessChain.Effect.pass(_:)`.
///
/// Every effect reads a shader-readable, tracked, single-sample RGBA16Float
/// source of premultiplied linear color and writes a separate render target of the
/// same size in one of `PostProcessPass.supportedPixelFormats`. Effects that read
/// depth take the scene's Depth32Float texture at the same size. Effects keep
/// their settings as properties, so editing them never rebuilds a pipeline.
///
/// Effects are not thread-safe. Serialize encodes and settings changes, and encode
/// on command buffers that retain their references so scratch textures replaced
/// after a resize stay alive in earlier buffers.
public protocol PostProcessEffect: AnyObject {
    /// The device whose textures and command buffers the effect accepts.
    var device: any MTLDevice { get }
    /// Whether `encode` needs the scene depth.
    var readsDepth: Bool { get }
    /// Prepares the pipelines that write `pixelFormat`. Encoding prepares them on
    /// first use otherwise; `PostProcessChain` calls this for its intermediate and
    /// destination formats when it adds the effect, so enabling or bypassing a
    /// stage never builds pipelines.
    func prepare(pixelFormat: MTLPixelFormat) throws
    /// Checks everything `encode` requires and allocates the scratch it will use,
    /// without encoding.
    func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws
    /// Encodes the effect on the caller's command buffer without submitting it.
    /// Nothing is encoded when validation throws.
    func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws
}

/// Reports how many pipelines a built-in effect has created, so tests can check that
/// `prepare(pixelFormat:)` builds everything `encode` uses.
package protocol PostProcessPipelineCounting {
    var builtPipelineCount: Int { get }
}

/// Errors shared by Frostfield's built-in `PostProcessEffect` implementations.
public enum PostProcessEffectError: LocalizedError, Equatable {
    case missingShader(String)
    case pipeline(String)
    case invalidSettings(String)
    case pixelFormat
    case commandBuffer
    case wrongDevice
    case source
    case destination
    case depth
    case missingDepth
    case sizeMismatch
    case allocation
    case encoder
    case missingSelection
    case unsupportedInput(String)

    public var errorDescription: String? {
        switch self {
        case .missingShader(let name): "Frostfield's post-process shader \(name) is missing."
        case .pipeline(let detail): "Could not prepare a post-process pipeline. \(detail)"
        case .invalidSettings(let detail): "Invalid post-process settings. \(detail)"
        case .pixelFormat: "Post-process effects write RGBA16Float, BGRA8Unorm_sRGB or RGBA8Unorm_sRGB."
        case .commandBuffer: "Post-processing needs an uncommitted command buffer that retains its references."
        case .wrongDevice: "Post-process textures and the command buffer must use the effect's Metal device."
        case .source: "The post-process source must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
        case .destination:
            "The post-process destination must be a separate single-sample 2D render target in a supported format."
        case .depth: "Post-process depth must be a shader-readable, tracked, single-sample 2D Depth32Float texture."
        case .missingDepth: "This effect reads scene depth. Pass the scene's Depth32Float texture."
        case .sizeMismatch: "The post-process source, depth and destination must have the same dimensions."
        case .allocation: "Metal could not allocate post-process scratch textures."
        case .encoder: "Metal could not create a post-process encoder."
        case .missingSelection:
            "Encode the selection mask for this frame's size before encoding the outline."
        case .unsupportedInput(let detail): detail
        }
    }
}

/// Precompiled post-process pipelines for one device, prepared on first use.
final class PostProcessShaders {
    let device: any MTLDevice
    private let library: any MTLLibrary
    private var renderPipelines: [String: any MTLRenderPipelineState] = [:]
    private var computePipelines: [String: any MTLComputePipelineState] = [:]
    private var placeholder: (any MTLTexture)?

    /// A 1 × 1 Depth32Float bound where a shader declares depth that a configuration does not read.
    func placeholderDepth() throws -> any MTLTexture {
        if let placeholder { return placeholder }
        let texture = try Self.scratch(
            device: device, width: 1, height: 1, pixelFormat: .depth32Float, usage: .shaderRead,
            label: "Frostfield unused depth"
        )
        placeholder = texture
        return texture
    }

    /// The number of render and compute pipelines built so far.
    var builtPipelineCount: Int { renderPipelines.count + computePipelines.count }

    init(device: any MTLDevice) throws {
        self.device = device
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw PostProcessEffectError.pipeline(error.localizedDescription)
        }
    }

    func compute(_ name: String) throws -> any MTLComputePipelineState {
        if let pipeline = computePipelines[name] { return pipeline }
        guard let function = library.makeFunction(name: name) else { throw PostProcessEffectError.missingShader(name) }
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            computePipelines[name] = pipeline
            return pipeline
        } catch {
            throw PostProcessEffectError.pipeline(error.localizedDescription)
        }
    }

    /// A full-screen pipeline whose color attachments use `formats` in order.
    /// `additive` blends every attachment as source plus destination.
    func render(_ fragment: String, formats: [MTLPixelFormat], additive: Bool = false) throws -> any MTLRenderPipelineState {
        let key = "\(fragment)|\(formats.map { String($0.rawValue) }.joined(separator: ","))|\(additive)"
        if let pipeline = renderPipelines[key] { return pipeline }
        guard let vertex = library.makeFunction(name: "frostfield_fx_vertex") else {
            throw PostProcessEffectError.missingShader("frostfield_fx_vertex")
        }
        guard let function = library.makeFunction(name: fragment) else { throw PostProcessEffectError.missingShader(fragment) }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield \(fragment)"
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = function
        for (index, format) in formats.enumerated() {
            descriptor.colorAttachments[index].pixelFormat = format
            if additive {
                let attachment = descriptor.colorAttachments[index]!
                attachment.isBlendingEnabled = true
                attachment.rgbBlendOperation = .add
                attachment.alphaBlendOperation = .add
                attachment.sourceRGBBlendFactor = .one
                attachment.sourceAlphaBlendFactor = .one
                attachment.destinationRGBBlendFactor = .one
                attachment.destinationAlphaBlendFactor = .one
            }
        }
        do {
            let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            renderPipelines[key] = pipeline
            return pipeline
        } catch {
            throw PostProcessEffectError.pipeline(error.localizedDescription)
        }
    }

    /// Dispatches one thread per pixel of a width × height grid in whole threadgroups.
    /// Kernels must bounds-check their pixel.
    static func dispatch(
        _ encoder: any MTLComputeCommandEncoder, pipeline: any MTLComputePipelineState, width: Int, height: Int
    ) {
        let threadWidth = pipeline.threadExecutionWidth
        let threadHeight = max(1, min(8, pipeline.maxTotalThreadsPerThreadgroup / threadWidth))
        let groups = MTLSize(
            width: (width + threadWidth - 1) / threadWidth, height: (height + threadHeight - 1) / threadHeight, depth: 1
        )
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1))
    }

    /// Encodes one full-screen triangle into `destination` (and optional extra attachments).
    static func drawFullScreen(
        into destination: any MTLTexture, extra: [any MTLTexture] = [], load: Bool = false, label: String,
        commandBuffer: any MTLCommandBuffer, pipeline: any MTLRenderPipelineState,
        bind: (any MTLRenderCommandEncoder) -> Void
    ) throws {
        let pass = MTLRenderPassDescriptor()
        for (index, texture) in ([destination] + extra).enumerated() {
            pass.colorAttachments[index].texture = texture
            pass.colorAttachments[index].loadAction = load ? .load : .dontCare
            pass.colorAttachments[index].storeAction = .store
        }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw PostProcessEffectError.encoder
        }
        encoder.label = label
        encoder.setRenderPipelineState(pipeline)
        bind(encoder)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    /// Encodes one compute dispatch over a grid.
    static func compute(
        label: String, commandBuffer: any MTLCommandBuffer, pipeline: any MTLComputePipelineState,
        width: Int, height: Int, bind: (any MTLComputeCommandEncoder) -> Void
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw PostProcessEffectError.encoder }
        encoder.label = label
        encoder.setComputePipelineState(pipeline)
        bind(encoder)
        dispatch(encoder, pipeline: pipeline, width: width, height: height)
        encoder.endEncoding()
    }

    /// A private, tracked scratch texture.
    static func scratch(
        device: any MTLDevice, width: Int, height: Int, pixelFormat: MTLPixelFormat = .rgba16Float,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite], label: String
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: max(1, width), height: max(1, height), mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = usage
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw PostProcessEffectError.allocation }
        texture.label = label
        return texture
    }
}

/// Shared input validation for the built-in effects.
enum PostProcessValidation {
    static func check(
        device: any MTLDevice, source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer, requiresDepth: Bool
    ) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw PostProcessEffectError.commandBuffer
        }
        guard commandBuffer.device === device, source.device === device, destination.device === device,
              depth.map({ $0.device === device }) ?? true else { throw PostProcessEffectError.wrongDevice }
        guard PostProcessPass.isReadable(source, format: .rgba16Float), source.width > 0, source.height > 0 else {
            throw PostProcessEffectError.source
        }
        guard destination !== source, PostProcessPass.supportedPixelFormats.contains(destination.pixelFormat),
              destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless else {
            throw PostProcessEffectError.destination
        }
        if let depth {
            guard PostProcessPass.isReadable(depth, format: .depth32Float) else { throw PostProcessEffectError.depth }
        } else if requiresDepth {
            throw PostProcessEffectError.missingDepth
        }
        guard destination.width == source.width, destination.height == source.height,
              depth.map({ $0.width == source.width && $0.height == source.height }) ?? true else {
            throw PostProcessEffectError.sizeMismatch
        }
    }

    static func checkFormat(_ pixelFormat: MTLPixelFormat) throws {
        guard PostProcessPass.supportedPixelFormats.contains(pixelFormat) else { throw PostProcessEffectError.pixelFormat }
    }

    /// Converts stored Metal depth to view-axis distance: `x` near, `y` far, `z` 1 for reversed depth.
    static func depthParameters(camera: PerspectiveCamera, depthMode: MeshDepthMode) -> SIMD4<Float> {
        SIMD4(camera.near, camera.far, depthMode == .reversed ? 1 : 0, 0)
    }
}
