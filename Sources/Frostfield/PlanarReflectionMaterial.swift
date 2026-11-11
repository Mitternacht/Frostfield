import Foundation
import Metal
import simd

/// Filters a planar capture in linear HDR color with five four-tap passes.
/// Serialize settings changes and encodes. Each output is caller-owned and is
/// available to later draws on the same retained-reference command buffer.
public final class PlanarReflectionBlur {
    public struct Settings: Sendable, Hashable {
        /// Maximum horizontal and vertical tap offsets in capture pixels, 0...64.
        public var radius: SIMD2<Float>
        /// Smoothstep edges for proximity in normalized device depth, 0...1.
        /// Proximity is `1 - storedDepth` in forward depth and `storedDepth` in reversed depth.
        public var depthThresholds: SIMD2<Float>
        /// Strength of the proximity response, 0...64. Zero uses a constant footprint.
        public var depthScale: Float

        public init(radius: SIMD2<Float> = .zero, depthThresholds: SIMD2<Float> = SIMD2(0.9, 1), depthScale: Float = 0) {
            self.radius = radius
            self.depthThresholds = depthThresholds
            self.depthScale = depthScale
        }

        var needsPass: Bool { radius != .zero || depthScale > 0 }

        var isValid: Bool {
            (0...64).contains(radius.x) && (0...64).contains(radius.y) && (0...64).contains(depthScale)
                && (0...1).contains(depthThresholds.x) && (0...1).contains(depthThresholds.y)
                && depthThresholds.x < depthThresholds.y
        }
    }

    public let device: any MTLDevice
    public var settings: Settings
    private let shaders: PostProcessShaders
    private var scratch: [any MTLTexture] = []
    private weak var scratchQueue: (any MTLCommandQueue)?

    public init(device: any MTLDevice, settings: Settings = .init()) throws {
        self.device = device
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    /// Filter the RGBA16Float color from `PlanarReflection.encode`. When
    /// depth scaling is enabled, give the capture a shader-readable Depth32Float
    /// attachment. The optional R16Float target stores the proximity response
    /// for `PlanarReflectionMix`. Both outputs must be tracked, shader-readable
    /// render targets of the capture size with separate storage.
    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        depthFactor: (any MTLTexture)? = nil, depthMode: MeshDepthMode = .forward, commandBuffer: any MTLCommandBuffer
    ) throws {
        let settings = self.settings
        guard settings.isValid else { throw Failure.settings }
        try depthMode.requireProjectionDepth()
        let usesDepth = settings.depthScale > 0
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: usesDepth
        )
        guard PostProcessPass.isReadable(destination, format: .rgba16Float) else { throw PostProcessEffectError.destination }
        if let depthFactor {
            guard depthFactor.device === device else { throw PostProcessEffectError.wrongDevice }
            guard PostProcessPass.isReadable(depthFactor, format: .r16Float), depthFactor.usage.contains(.renderTarget),
                  depthFactor.width == source.width, depthFactor.height == source.height else { throw Failure.depthFactor }
        }
        let textures = [source, destination] + [depth, depthFactor].compactMap { $0 }
        for (index, texture) in textures.enumerated() {
            guard !texture.isAliasable(), !textures[..<index].contains(where: { TextureTransfer.sharesStorage(texture, $0) }) else {
                throw Failure.feedback
            }
        }
        let offsets: [Float] = settings.radius == .zero ? [0] : [0.5, 1.5, 2.5, 2.5, 3.5]
        let pipeline = try shaders.render("frostfield_planar_blur_fragment", formats: [.rgba16Float])
        let final = try depthFactor.map { _ in
            try shaders.render("frostfield_planar_blur_depth_fragment", formats: [.rgba16Float, .r16Float])
        } ?? pipeline
        if offsets.count > 1,
           scratchQueue !== commandBuffer.commandQueue || scratch.first?.width != source.width || scratch.first?.height != source.height {
            scratch = try (0..<2).map { index in
                try PostProcessShaders.scratch(
                    device: device, width: source.width, height: source.height, usage: [.renderTarget, .shaderRead],
                    label: "Frostfield planar blur scratch \(index)"
                )
            }
            scratchQueue = commandBuffer.commandQueue
        }
        let depth = try depth ?? shaders.placeholderDepth()
        let texel = SIMD2(1 / Float(source.width), 1 / Float(source.height))
        let response = SIMD4(settings.depthThresholds.x, settings.depthThresholds.y, settings.depthScale, depthMode == .reversed ? 1 : 0)
        var input = source
        for (index, offset) in offsets.enumerated() {
            let last = index == offsets.count - 1
            let output = last ? destination : scratch[index % 2]
            let extra = last ? [depthFactor].compactMap { $0 } : []
            let radius = settings.radius * (offset / 3.5) * texel
            var parameters = Parameters(filter: SIMD4(radius.x, radius.y, texel.x, texel.y), response: response)
            try PostProcessShaders.drawFullScreen(
                into: output, extra: extra, label: "Frostfield depth-aware planar blur", commandBuffer: commandBuffer,
                pipeline: last ? final : pipeline
            ) { encoder in
                encoder.setFragmentTexture(input, index: 0)
                encoder.setFragmentTexture(depth, index: 1)
                encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
            }
            input = output
        }
    }

    private struct Parameters {
        var filter: SIMD4<Float>
        var response: SIMD4<Float>
    }

    public enum Failure: LocalizedError, Equatable {
        case settings, depthFactor, feedback

        public var errorDescription: String? {
            switch self {
            case .settings: "Use planar blur radii and depth scale in 0...64 and strictly ordered depth thresholds in 0...1."
            case .depthFactor: "The depth-factor output must be a tracked shader-readable R16Float render target of the capture size."
            case .feedback: "Planar blur inputs and outputs require distinct non-aliasable texture storage."
            }
        }
    }
}

/// Artist controls for combining sharp and blurred planar reflection color in
/// a typed raster surface graph. Settings are authored in linear color space.
public struct PlanarReflectionMix: Sendable, Hashable {
    public let blurMix: Float
    public let strength: Float
    public let mirror: Float
    public let contrast: Float
    public let distortion: Float
    public let depthBlurBias: Float

    /// `mirror` suppresses the underlying diffuse contribution from 0 to 1.
    /// `distortion` adds a red-channel offset in homogeneous clip units.
    public init(
        blurMix: Float = 0, strength: Float = 1, mirror: Float = 0, contrast: Float = 1,
        distortion: Float = 1, depthBlurBias: Float = 0.25
    ) throws {
        guard [blurMix, strength, contrast].allSatisfy({ (0...64).contains($0) }),
              (0...1).contains(mirror), (0...1).contains(depthBlurBias), (-16...16).contains(distortion) else {
            throw Failure.settings
        }
        self.blurMix = blurMix
        self.strength = strength
        self.mirror = mirror
        self.contrast = contrast
        self.distortion = distortion
        self.depthBlurBias = depthBlurBias
    }

    /// Multiply the authored linear base color by this result. The capture inputs
    /// use clamp-to-edge sampling. Bind the optional depth factor and distortion
    /// map as linear `.data` textures. Distortion uses mesh UV0 by default and
    /// reads the camera matrices when enabled. This is a raster surface operation.
    public func multiplier(
        sharp: ShaderTexture, blurred: ShaderTexture? = nil, depthFactor: ShaderTexture? = nil,
        distortionMap: ShaderTexture? = nil, roughness: ShaderValue<Float>? = nil,
        coordinates: ShaderValue<SIMD2<Float>>? = nil, distortionCoordinates: ShaderValue<SIMD2<Float>>? = nil
    ) -> ShaderValue<SIMD3<Float>> {
        multiplier(
            sharp: sharp, blurred: blurred, depthFactor: depthFactor, distortionMap: distortionMap,
            roughness: roughness, coordinates: coordinates, distortionCoordinates: distortionCoordinates, normalMap: nil
        )
    }

    /// Adds the RGB normal-map projection offset used by a drei reflector. Pass
    /// normal coordinates after the binding's transform and any normal UV scale.
    /// The normal offsets color only. The depth weight follows the distorted ray.
    /// Supply `normalProjectedDepth` in Metal device depth when the normal
    /// offset should use a different projection from the current draw.
    public func multiplier(
        sharp: ShaderTexture, blurred: ShaderTexture? = nil, depthFactor: ShaderTexture? = nil,
        distortionMap: ShaderTexture? = nil, roughness: ShaderValue<Float>? = nil,
        coordinates: ShaderValue<SIMD2<Float>>? = nil, distortionCoordinates: ShaderValue<SIMD2<Float>>? = nil,
        normalMap: ShaderTexture?, normalCoordinates: ShaderValue<SIMD2<Float>>? = nil,
        normalProjectedDepth: ShaderValue<Float>? = nil, depthMode: MeshDepthMode = .forward
    ) -> ShaderValue<SIMD3<Float>> {
        let graph = sharp.builder
        let viewport = coordinates ?? graph.screenUV
        var uv = graph.vector(1 - viewport.x, viewport.y)
        var colorUV = uv
        if normalMap != nil || (distortionMap != nil && distortion != 0) {
            let view = graph.viewMatrix * graph.vector(graph.worldPosition, graph.constant(Float(1)))
            let clip = graph.projectionMatrix * view
            let w = clip.w.max(0.000001)
            if let distortionMap, distortion != 0 {
                let offset = distortionMap.sample(distortionCoordinates ?? graph.uv0).r * distortion / w
                uv = uv + graph.vector(offset, 0 - offset)
            }
            colorUV = uv
            if let normalMap {
                let encoded = normalMap.sample(normalCoordinates ?? graph.uv0)
                let axes = graph.vector(encoded.r * 2 - 1, encoded.b, encoded.g * 2 - 1)
                let unit = axes / axes.length().max(0.000001)
                let depth = normalProjectedDepth ?? clip.z / w
                let forwardDepth = depthMode == .reversed ? depth.oneMinus() : depth
                colorUV = uv + graph.vector(unit.x, 0 - unit.z) * (forwardDepth * 0.05)
            }
        }
        let factor = depthFactor?.sample(uv).r.clamped(0.0001, 1)
        var result = sharp.sample(colorUV).rgb
        if let blurred {
            var soft = blurred.sample(colorUV).rgb
            if let factor {
                result = result * (factor + 0.5).min(1)
                soft = soft * (factor + depthBlurBias).min(1)
            }
            result = result.mix(soft, by: ((roughness ?? graph.roughness) * blurMix).saturated())
        } else if let factor {
            result = result * factor
        }
        return ((result - 0.5) * contrast + 0.5) * strength + (1 - mirror)
    }

    public enum Failure: LocalizedError, Equatable {
        case settings

        public var errorDescription: String? {
            "Use blur mix, strength and contrast in 0...64, mirror and depth blur bias in 0...1, and distortion in -16...16."
        }
    }
}
