import Foundation
import Metal

/// A prepared linear-color post process with threshold bloom, radial color
/// fringing, and deterministic film grain.
public final class ScreenEffects {
    public struct Settings: Sendable {
        /// Minimum peak RGB channel in linear space, from 0 to 65,504.
        public var bloomThreshold: Float
        /// Additive glow gain from 0 to 4. Zero skips the bloom passes.
        public var bloomStrength: Float
        /// Full-resolution pixels to the outer blur taps, from 0 to 32.
        public var bloomRadius: Float
        /// Maximum radial channel displacement per axis, from 0 to 16 pixels.
        public var chromaticAberration: Float
        /// Monochromatic grain amplitude from 0 to 0.25 in linear space.
        public var grainStrength: Float
        /// Repeat a seed for a fixed pattern or change it between frames.
        public var grainSeed: UInt32

        public init(
            bloomThreshold: Float = 1,
            bloomStrength: Float = 0,
            bloomRadius: Float = 8,
            chromaticAberration: Float = 0,
            grainStrength: Float = 0,
            grainSeed: UInt32 = 0
        ) {
            self.bloomThreshold = bloomThreshold
            self.bloomStrength = bloomStrength
            self.bloomRadius = bloomRadius
            self.chromaticAberration = chromaticAberration
            self.grainStrength = grainStrength
            self.grainSeed = grainSeed
        }

        var isValid: Bool {
            (0...65_504).contains(bloomThreshold) && (0...4).contains(bloomStrength)
                && (0...32).contains(bloomRadius) && (0...16).contains(chromaticAberration)
                && (0...0.25).contains(grainStrength)
        }
    }

    private struct Scratch {
        let first: any MTLTexture
        let second: any MTLTexture
    }

    private static let maxScratchPixels = 4_194_304
    private let queue: any MTLCommandQueue
    private let destinationFormat: MTLPixelFormat
    private let threshold: any MTLComputePipelineState
    private let blur: any MTLComputePipelineState
    private let composite: any MTLRenderPipelineState
    private var scratch: Scratch?

    public init(queue: any MTLCommandQueue, destinationPixelFormat: MTLPixelFormat) throws {
        switch destinationPixelFormat {
        case .rgba16Float, .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb: break
        default: throw Failure.destinationFormat
        }

        self.queue = queue
        self.destinationFormat = destinationPixelFormat
        let device = queue.device
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let threshold = library.makeFunction(name: "frostfield_screen_threshold"),
                  let blur = library.makeFunction(name: "frostfield_screen_blur"),
                  let vertex = library.makeFunction(name: "frostfield_screen_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_screen_fragment") else {
                throw Failure.missingShader
            }
            self.threshold = try device.makeComputePipelineState(function: threshold)
            self.blur = try device.makeComputePipelineState(function: blur)

            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Frostfield screen effects"
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = destinationPixelFormat
            composite = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Encodes into the caller's buffer without submitting it. Source must be
    /// a standalone, tracked RGBA16Float texture containing premultiplied linear
    /// RGB. Use a separate equal-sized render target as the destination. Effects
    /// preserve source alpha, and glow is visible only where that alpha is nonzero.
    /// The target performs its normal format conversion, with no tone mapping.
    ///
    /// Serialize calls on each ScreenEffects instance and use retained command
    /// buffers from its queue. Scratch textures remain alive in earlier buffers
    /// when resized. Discard the command buffer if encoding throws.
    public func encode(
        source: any MTLTexture,
        destination: any MTLTexture,
        settings: Settings = Settings(),
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try prepare(source: source, destination: destination, settings: settings, commandBuffer: commandBuffer)

        var glow = source
        if settings.bloomStrength > 0 {
            let width = source.width / 2 + source.width % 2
            let height = source.height / 2 + source.height % 2
            let targets = try bloomTargets(width: width, height: height)
            try compute(
                threshold, label: "Frostfield bloom threshold", source: source, destination: targets.first,
                values: SIMD2(settings.bloomThreshold, 0), commandBuffer: commandBuffer
            )
            if settings.bloomRadius > 0 {
                let spacing = settings.bloomRadius * 0.5
                try compute(
                    blur, label: "Frostfield bloom horizontal", source: targets.first, destination: targets.second,
                    values: SIMD2(spacing / Float(source.width), 0), commandBuffer: commandBuffer
                )
                try compute(
                    blur, label: "Frostfield bloom vertical", source: targets.second, destination: targets.first,
                    values: SIMD2(0, spacing / Float(source.height)), commandBuffer: commandBuffer
                )
            }
            glow = targets.first
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }
        var amounts = SIMD4(settings.bloomStrength, settings.chromaticAberration, settings.grainStrength, 0)
        var seed = settings.grainSeed
        encoder.label = "Frostfield screen composite"
        encoder.setRenderPipelineState(composite)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(glow, index: 1)
        encoder.setFragmentBytes(&amounts, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        encoder.setFragmentBytes(&seed, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    /// Checks everything `encode` requires and allocates the bloom scratch it will
    /// use, without encoding.
    func prepare(
        source: any MTLTexture, destination: any MTLTexture, settings: Settings, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard settings.isValid else { throw Failure.invalidSettings }
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences else {
            throw Failure.commandBuffer
        }
        guard source.device === queue.device, destination.device === queue.device else {
            throw Failure.wrongDevice
        }
        guard source !== destination, source.textureType == .type2D, destination.textureType == .type2D,
              source.sampleCount == 1, destination.sampleCount == 1,
              source.width > 0, source.height > 0,
              source.width == destination.width, source.height == destination.height,
              source.pixelFormat == .rgba16Float, destination.pixelFormat == destinationFormat,
              source.usage.contains(.shaderRead), !source.isFramebufferOnly,
              destination.usage.contains(.renderTarget),
              source.hazardTrackingMode != .untracked,
              source.parent == nil, destination.parent == nil,
              source.buffer == nil, destination.buffer == nil, source.iosurface == nil,
              source.heap == nil else {
            throw Failure.incompatibleTextures
        }
        if settings.bloomStrength > 0 {
            _ = try bloomTargets(width: source.width / 2 + source.width % 2, height: source.height / 2 + source.height % 2)
        }
    }

    private func bloomTargets(width: Int, height: Int) throws -> Scratch {
        guard width <= Self.maxScratchPixels / height else { throw Failure.bloomSize }
        if let scratch, scratch.first.width == width, scratch.first.height == height { return scratch }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let first = queue.device.makeTexture(descriptor: descriptor),
              let second = queue.device.makeTexture(descriptor: descriptor) else {
            throw Failure.bloomAllocation
        }
        first.label = "Frostfield bloom threshold and vertical"
        second.label = "Frostfield bloom horizontal"
        let result = Scratch(first: first, second: second)
        scratch = result
        return result
    }

    private func compute(
        _ pipeline: any MTLComputePipelineState,
        label: String,
        source: any MTLTexture,
        destination: any MTLTexture,
        values: SIMD2<Float>,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        var values = values
        let width = pipeline.threadExecutionWidth
        let height = max(1, min(8, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.label = label
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setBytes(&values, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        let groups = MTLSize(
            width: (destination.width + width - 1) / width,
            height: (destination.height + height - 1) / height,
            depth: 1
        )
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case destinationFormat
        case invalidSettings
        case commandBuffer
        case wrongDevice
        case incompatibleTextures
        case bloomSize
        case bloomAllocation
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield’s screen effects shaders are missing."
            case .pipeline(let detail): "Could not prepare Frostfield’s screen effects. \(detail)"
            case .destinationFormat: "Screen effects need RGBA16Float or an 8-bit RGBA or BGRA target."
            case .invalidSettings: "Screen effect settings must be finite and within their documented ranges."
            case .commandBuffer: "Use a retained command buffer from the screen effects queue."
            case .wrongDevice: "The screen effects textures must use the command queue’s Metal device."
            case .incompatibleTextures:
                "Use a separate, equal-sized render target and a tracked, shader-readable RGBA16Float source."
            case .bloomSize: "The half-resolution bloom target exceeds 4,194,304 pixels."
            case .bloomAllocation: "Metal could not allocate the bloom scratch textures."
            case .encoder: "Metal could not create a screen effects encoder."
            }
        }
    }
}
