import Foundation
import Metal

/// Edge-adaptive spatial upsampling with optional contrast-adaptive sharpening.
/// Encode on retained buffers from the pass's queue. Intermediate storage is reused.
public final class FSR1Upscaler {
    public let queue: any MTLCommandQueue
    /// Set to nil for EASU alone. Edits apply on the next encode.
    public var sharpening: RCASSharpen.Settings?

    private let shaders: PostProcessShaders
    private let sharpenPass: RCASSharpen
    private var intermediate: (any MTLTexture)?

    public init(queue: any MTLCommandQueue, sharpening: RCASSharpen.Settings? = .init()) throws {
        self.queue = queue
        self.sharpening = sharpening
        shaders = try PostProcessShaders(device: queue.device)
        sharpenPass = try RCASSharpen(device: queue.device, settings: sharpening ?? .init())
    }

    /// Upsamples tracked, premultiplied linear RGBA16Float into a separate render target.
    /// Both destination dimensions must be at least as large as the source.
    /// Equal-size input uses an exact copy or the standalone sharpening stage.
    public func encode(
        source: any MTLTexture, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw PostProcessEffectError.commandBuffer
        }
        guard source.device === queue.device, destination.device === queue.device else {
            throw PostProcessEffectError.wrongDevice
        }
        guard PostProcessPass.isReadable(source, format: .rgba16Float) else { throw PostProcessEffectError.source }
        try PostProcessValidation.checkFormat(destination.pixelFormat)
        guard destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              !TextureTransfer.sharesStorage(source, destination) else { throw PostProcessEffectError.destination }
        guard destination.width >= source.width, destination.height >= source.height else {
            throw PostProcessEffectError.sizeMismatch
        }

        if let sharpening {
            sharpenPass.settings = sharpening
            try sharpenPass.prepare(pixelFormat: destination.pixelFormat)
            if source.width == destination.width, source.height == destination.height {
                try sharpenPass.encode(source: source, destination: destination, commandBuffer: commandBuffer)
                return
            }
            let staged = try sharpeningInput(width: destination.width, height: destination.height)
            try sharpenPass.validate(source: staged, depth: nil, destination: destination, commandBuffer: commandBuffer)
            try encodeEASU(source: source, destination: staged, commandBuffer: commandBuffer)
            try sharpenPass.encode(source: staged, destination: destination, commandBuffer: commandBuffer)
        } else {
            try encodeEASU(source: source, destination: destination, commandBuffer: commandBuffer)
        }
    }

    private func sharpeningInput(width: Int, height: Int) throws -> any MTLTexture {
        if let intermediate, intermediate.width == width, intermediate.height == height { return intermediate }
        let texture = try PostProcessShaders.scratch(
            device: queue.device, width: width, height: height, usage: [.renderTarget, .shaderRead],
            label: "Frostfield FSR1 intermediate"
        )
        intermediate = texture
        return texture
    }

    private func encodeEASU(
        source: any MTLTexture, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let pipeline = try shaders.render("frostfield_fsr1_easu", formats: [destination.pixelFormat])
        var size = SIMD2(Float(destination.width), Float(destination.height))
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield EASU", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentBytes(&size, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        }
    }
}

/// A five-tap robust contrast-adaptive sharpening pass at the input size.
/// The pass sharpens straight linear RGB and retains the center pixel's alpha.
public final class RCASSharpen: PostProcessEffect {
    public struct Settings: Sendable, Hashable {
        /// Attenuation in stops, from 0 (strongest) to 16.
        public var sharpness: Float
        /// Attenuates sharpening in noisy neighborhoods.
        public var denoise: Bool

        public init(sharpness: Float = 0.2, denoise: Bool = false) {
            self.sharpness = sharpness
            self.denoise = denoise
        }

        fileprivate func check() throws {
            guard (0...16).contains(sharpness) else {
                throw PostProcessEffectError.invalidSettings("RCAS needs sharpness attenuation from 0 to 16 stops.")
            }
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    public var settings: Settings
    private let shaders: PostProcessShaders

    public init(device: any MTLDevice, settings: Settings = .init()) throws {
        try settings.check()
        self.device = device
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_fsr1_rcas", formats: [pixelFormat])
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

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render("frostfield_fsr1_rcas", formats: [destination.pixelFormat])
        var parameters = SIMD2(exp2(-settings.sharpness), settings.denoise ? 1 : 0)
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield RCAS", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        }
    }
}

extension RCASSharpen: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
