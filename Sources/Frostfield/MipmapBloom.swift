import Foundation
import Metal

/// Physically based bloom over a mip chain, after Jimenez (Call of Duty: Advanced
/// Warfare, SIGGRAPH 2014) and pmndrs/postprocessing `MipmapBlurPass`.
///
/// The frame is downsampled to `levels` successively halved RGBA16Float textures
/// with the 13-tap filter (five overlapping 2 × 2 boxes weighted ½ and 4 × ⅛), then
/// upsampled with a 3 × 3 tent, where each level becomes
/// `mix(downsampled, tent(upsampled coarser level), radius)`. The finest level is
/// tent-filtered to full size and combined with the scene.
///
/// A threshold is applied to each source texel before any filtering, so the chain is
/// a linear filter of the thresholded frame. `.energyConserving` blending writes
/// `scene + strength · (bloom − thresholded(scene))`: each pixel gives up the
/// fraction of its above-threshold light that scatters and receives the scattered
/// light of its neighbors. Without a threshold this is `mix(scene, bloom, strength)`;
/// with one, pixels below the knee that receive no bloom are unchanged. Every filter
/// is normalized and area-preserving, so with `fireflySuppression` off and
/// `.energyConserving` blending the output's total RGB equals the input's away from
/// the frame edges, with or without a threshold: bloom only redistributes light, like
/// scattering in a lens. The Karis average used for firefly suppression is nonlinear
/// and removes that guarantee. Alpha passes through unchanged.
public final class MipmapBloom: PostProcessEffect {
    public static let maximumLevels = 10

    public enum Blending: Sendable, Hashable {
        /// `scene + strength · (bloom − thresholded(scene))`, which is
        /// `mix(scene, bloom, strength)` without a threshold: total light is preserved.
        case energyConserving
        /// `scene + strength · bloom`, like Three.js `UnrealBloomPass`.
        case additive
    }

    public struct Settings: Sendable, Hashable {
        /// Fraction of light scattered (energy conserving, 0...1) or additive gain (0...16).
        public var strength: Float
        /// How much each level takes from the coarser levels, from 0 to 1. Larger values widen the glow.
        public var radius: Float
        /// Mip levels below full resolution, 1 to 10. Levels stop early once a side reaches one pixel.
        public var levels: Int
        /// Minimum peak RGB that blooms, or nil for threshold-free bloom of all light.
        public var threshold: Float?
        /// Soft-knee width below the threshold, in the same units.
        public var knee: Float
        /// Weights the first downsample's boxes by `1 / (1 + luma)` to suppress single-pixel highlights.
        public var fireflySuppression: Bool
        public var blending: Blending

        public init(
            strength: Float = 0.04, radius: Float = 0.85, levels: Int = 6, threshold: Float? = nil, knee: Float = 0.5,
            fireflySuppression: Bool = true, blending: Blending = .energyConserving
        ) {
            self.strength = strength
            self.radius = radius
            self.levels = levels
            self.threshold = threshold
            self.knee = knee
            self.fireflySuppression = fireflySuppression
            self.blending = blending
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    public var settings: Settings
    private let shaders: PostProcessShaders
    private var chain: (down: [any MTLTexture], up: [any MTLTexture])?

    public init(device: any MTLDevice, settings: Settings = Settings()) throws {
        self.device = device
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_bloom_composite", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_bloom_downsample")
        _ = try shaders.compute("frostfield_bloom_upsample")
    }

    /// The mip sizes used for a frame, finest first.
    public func levelSizes(width: Int, height: Int) -> [(width: Int, height: Int)] {
        var sizes: [(Int, Int)] = []
        var (w, h) = (width, height)
        while sizes.count < settings.levels, w > 1, h > 1 {
            w = (w + 1) / 2
            h = (h + 1) / 2
            sizes.append((w, h))
        }
        return sizes
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let maximumStrength: Float = settings.blending == .energyConserving ? 1 : 16
        guard settings.strength.isFinite, (0...maximumStrength).contains(settings.strength),
              settings.radius.isFinite, (0...1).contains(settings.radius),
              (1...Self.maximumLevels).contains(settings.levels),
              settings.knee.isFinite, settings.knee >= 0,
              settings.threshold.map({ $0.isFinite && $0 >= 0 }) ?? true else {
            throw PostProcessEffectError.invalidSettings(
                "Bloom strength must be 0...1 (energy conserving) or 0...16 (additive), radius 0...1, levels 1...10, and threshold and knee nonnegative."
            )
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: false
        )
        let sizes = levelSizes(width: source.width, height: source.height)
        guard !sizes.isEmpty else { throw PostProcessEffectError.unsupportedInput("Bloom needs a frame at least 2 pixels on each side.") }
        if let chain, chain.down.count == sizes.count,
           zip(chain.down, sizes).allSatisfy({ $0.width == $1.width && $0.height == $1.height }) {
            return
        }
        let down = try sizes.enumerated().map { index, size in
            try PostProcessShaders.scratch(
                device: device, width: size.width, height: size.height, label: "Frostfield bloom down \(index)"
            )
        }
        let up = try sizes.dropLast().enumerated().map { index, size in
            try PostProcessShaders.scratch(device: device, width: size.width, height: size.height, label: "Frostfield bloom up \(index)")
        }
        chain = (down, up)
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let composite = try shaders.render("frostfield_bloom_composite", formats: [destination.pixelFormat])
        let glow = try encodePyramid(source: source, commandBuffer: commandBuffer)
        var parameters = Composite(
            strength: settings.strength, additive: settings.blending == .energyConserving ? 0 : 1,
            inverseSize: SIMD2(1 / Float(destination.width), 1 / Float(destination.height)),
            prefilter: Prefilter(
                threshold: settings.threshold ?? 0, knee: settings.knee, useThreshold: settings.threshold != nil ? 1 : 0,
                karis: 0
            )
        )
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield bloom composite", commandBuffer: commandBuffer, pipeline: composite
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(glow, index: 1)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Composite>.stride, index: 0)
        }
    }

    /// Writes the strength-weighted HDR bloom alone, with alpha one. Use this as a
    /// named input to a post-process graph that composites the glow separately.
    /// The tracked RGBA16Float destination must be shader-readable and match the source size.
    /// `settings.blending` still determines the permitted strength range.
    public func encodeGlow(
        source: any MTLTexture, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard destination.pixelFormat == .rgba16Float, destination.usage.contains(.shaderRead),
              destination.hazardTrackingMode != .untracked else {
            throw PostProcessEffectError.destination
        }
        try validate(source: source, depth: nil, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render("frostfield_bloom_glow", formats: [.rgba16Float])
        let glow = try encodePyramid(source: source, commandBuffer: commandBuffer)
        var parameters = SIMD4(settings.strength, 0, 1 / Float(destination.width), 1 / Float(destination.height))
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield bloom glow", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(glow, index: 0)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        }
    }

    private func encodePyramid(source: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws -> any MTLTexture {
        guard let chain else { throw PostProcessEffectError.allocation }
        let downsample = try shaders.compute("frostfield_bloom_downsample")
        let upsample = try shaders.compute("frostfield_bloom_upsample")

        var input = source
        for (index, output) in chain.down.enumerated() {
            var prefilter = Prefilter(
                threshold: index == 0 ? settings.threshold ?? 0 : 0, knee: settings.knee,
                useThreshold: index == 0 && settings.threshold != nil ? 1 : 0,
                karis: index == 0 && settings.fireflySuppression ? 1 : 0
            )
            try PostProcessShaders.compute(
                label: "Frostfield bloom downsample \(index)", commandBuffer: commandBuffer, pipeline: downsample,
                width: output.width, height: output.height
            ) { encoder in
                encoder.setTexture(input, index: 0)
                encoder.setTexture(output, index: 1)
                encoder.setBytes(&prefilter, length: MemoryLayout<Prefilter>.stride, index: 0)
            }
            input = output
        }
        var lower = chain.down[chain.down.count - 1]
        var radius = settings.radius
        for index in stride(from: chain.up.count - 1, through: 0, by: -1) {
            let output = chain.up[index]
            let current = chain.down[index]
            let coarser = lower
            try PostProcessShaders.compute(
                label: "Frostfield bloom upsample \(index)", commandBuffer: commandBuffer, pipeline: upsample,
                width: output.width, height: output.height
            ) { encoder in
                encoder.setTexture(coarser, index: 0)
                encoder.setTexture(current, index: 1)
                encoder.setTexture(output, index: 2)
                encoder.setBytes(&radius, length: MemoryLayout<Float>.size, index: 0)
            }
            lower = output
        }
        return lower
    }

    private struct Prefilter {
        var threshold: Float
        var knee: Float
        var useThreshold: UInt32
        var karis: UInt32
    }

    /// Matches `FFBloomComposite`: 32 bytes.
    private struct Composite {
        var strength: Float
        var additive: UInt32
        var inverseSize: SIMD2<Float>
        var prefilter: Prefilter
    }
}

extension MipmapBloom: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
