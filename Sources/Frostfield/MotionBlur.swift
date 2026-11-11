import Foundation
import Metal

/// Applies screen-space motion blur to linear premultiplied color.
/// Use directly or with `PostProcessChain.Effect.pass(_:)`.
public final class MotionBlur: PostProcessEffect {
    public struct Settings: Sendable, Hashable {
        /// The number of taps along the motion vector, from 2 to 64, plus the original texel.
        public var sampleCount: Int
        /// Scales the supplied motion vector, from zero to four. Zero copies the source.
        public var shutterScale: Float
        /// Shifts the sampling span using per-axis blue-noise ranks, from zero to one.
        /// Zero preserves the regular tap positions.
        public var jitter: Float
        /// Selects the deterministic spatial phase. Change it for each rendered frame
        /// when the blur should vary over time.
        public var frameSeed: UInt32

        public init(sampleCount: Int = 16, shutterScale: Float = 1, jitter: Float = 0, frameSeed: UInt32 = 0) {
            self.sampleCount = sampleCount
            self.shutterScale = shutterScale
            self.jitter = jitter
            self.frameSeed = frameSeed
        }

        fileprivate func check() throws {
            guard (2...64).contains(sampleCount), (0...4).contains(shutterScale), (0...1).contains(jitter) else {
                throw PostProcessEffectError.invalidSettings(
                    "Motion blur needs 2 to 64 samples, a shutter scale from 0 to 4, and jitter from 0 to 1."
                )
            }
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    /// A tracked, shader-readable RGBA16Float texture of the same size as the source.
    /// RG is previous-minus-current motion in top-left pixels. A is one for valid
    /// motion and zero without history. B stores the previous depth.
    /// Set this to the current view's motion attachment before encoding.
    public var motionTexture: any MTLTexture
    /// Edits take effect on the next encode and reuse the prepared variant.
    public var settings: Settings

    private let shaders: PostProcessShaders
    private var blueNoiseRanks: (any MTLBuffer)?

    private struct Parameters {
        var sampleCount: UInt32
        var shutterScale: Float
    }

    private struct Jitter {
        var amount: Float
        var phase: UInt32
    }

    private var usesJitter: Bool { settings.jitter > 0 && settings.shutterScale > 0 }

    private var fragmentName: String {
        usesJitter ? "frostfield_motion_blur_jittered_fragment" : "frostfield_motion_blur_fragment"
    }

    private func rankBuffer() throws -> any MTLBuffer {
        if let blueNoiseRanks { return blueNoiseRanks }
        guard let url = Bundle.module.url(forResource: "PathBlueNoiseRanks", withExtension: "bin"),
              let data = try? Data(contentsOf: url), data.count == 64 * 64 * MemoryLayout<UInt16>.stride,
              let buffer = data.withUnsafeBytes({
                  device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: [.storageModeShared, .hazardTrackingModeTracked])
              }) else { throw PostProcessEffectError.allocation }
        buffer.label = "Frostfield motion blur blue-noise ranks"
        blueNoiseRanks = buffer
        return buffer
    }

    public init(device: any MTLDevice, motionTexture: any MTLTexture, settings: Settings = .init()) throws {
        try settings.check()
        self.device = device
        self.motionTexture = motionTexture
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render(fragmentName, formats: [pixelFormat])
        if usesJitter { _ = try rankBuffer() }
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try settings.check()
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination,
            commandBuffer: commandBuffer, requiresDepth: false
        )
        guard motionTexture.device === device else { throw PostProcessEffectError.wrongDevice }
        guard PostProcessPass.isReadable(motionTexture, format: .rgba16Float) else {
            throw PostProcessEffectError.unsupportedInput("Motion blur needs a tracked, shader-readable, single-sample 2D RGBA16Float motion texture.")
        }
        guard motionTexture.width == source.width, motionTexture.height == source.height else {
            throw PostProcessEffectError.sizeMismatch
        }
        guard !TextureTransfer.sharesStorage(source, destination),
              !TextureTransfer.sharesStorage(motionTexture, destination) else {
            throw PostProcessEffectError.destination
        }
    }

    /// Encodes one draw without submitting the caller's command buffer. Each pixel
    /// reads its motion texel directly and samples color bilinearly with edge clamp.
    /// Invalid or zero motion leaves the original texel exactly as stored.
    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render(fragmentName, formats: [destination.pixelFormat])
        let ranks = try usesJitter ? rankBuffer() : nil
        var parameters = Parameters(sampleCount: UInt32(settings.sampleCount), shutterScale: settings.shutterScale)
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield motion blur", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(motionTexture, index: 1)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
            if let ranks {
                let phase = UInt32(ShaderNoise.hash(.zero, seed: settings.frameSeed) * 0x1p24)
                var jitter = Jitter(amount: settings.jitter, phase: phase)
                encoder.setFragmentBytes(&jitter, length: MemoryLayout<Jitter>.stride, index: 1)
                encoder.setFragmentBuffer(ranks, offset: 0, index: 2)
            }
        }
    }
}

extension MotionBlur: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
