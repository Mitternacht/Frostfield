import Foundation
import Metal
import simd

/// Screen-space light shafts radiating from a light position, after Mitchell,
/// "Volumetric Light Scattering as a Post-Process" (GPU Gems 3, chapter 13), the
/// method of the `GodRaysShader` that older Three.js god-ray examples used. It is not
/// volumetric: current Three.js examples ray-march shadow maps instead.
///
/// A mask pass keeps light only where depth is cleared (open sky): either a soft sun
/// disk around `lightPosition` or scene color above a threshold. Up to three radial
/// passes then average `samples` taps toward the light, weighting tap `i` by
/// `decay^i` and normalizing, each pass with steps `samples` times finer than the
/// last. The composite adds `intensity · color · shafts` to the scene RGB, so a
/// uniform mask of value `m` adds exactly `intensity · color · m`. Occluders between
/// a pixel and the light carry their zero mask along the ray as a shadow shaft.
public final class GodRays: PostProcessEffect {
    public enum Source: Sendable, Hashable {
        /// A soft disk of the given radius, as a fraction of the frame height, around the light.
        case sun(radius: Float)
        /// Scene RGB minus a threshold, where the sky is visible.
        case sky(threshold: Float)
    }

    public struct Settings: Sendable, Hashable {
        /// Light position in top-left normalized frame coordinates; it may lie outside 0...1.
        public var lightPosition: SIMD2<Float>
        /// Linear RGB tint of the shafts.
        public var color: SIMD3<Float>
        /// Gain applied to the shafts, from 0 to 64.
        public var intensity: Float
        /// Fraction of the distance toward the light that one pass spans, from 0 to 1.
        public var density: Float
        /// Per-tap falloff from 0 to 1; 1 weights every tap equally.
        public var decay: Float
        /// Taps per pass, from 2 to 128.
        public var samples: Int
        /// Radial passes, from 1 to 3.
        public var passes: Int
        /// Size of the mask and blur textures relative to the frame, from 0.25 to 1.
        public var resolutionScale: Float
        public var source: Source

        public init(
            lightPosition: SIMD2<Float>, color: SIMD3<Float> = SIMD3(1, 0.95, 0.85), intensity: Float = 1,
            density: Float = 0.9, decay: Float = 0.97, samples: Int = 32, passes: Int = 2, resolutionScale: Float = 0.5,
            source: Source = .sun(radius: 0.08)
        ) {
            self.lightPosition = lightPosition
            self.color = color
            self.intensity = intensity
            self.density = density
            self.decay = decay
            self.samples = samples
            self.passes = passes
            self.resolutionScale = resolutionScale
            self.source = source
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { true }
    public var settings: Settings
    public var depthMode: MeshDepthMode
    private let shaders: PostProcessShaders
    private var scratch: (any MTLTexture, any MTLTexture)?

    public init(device: any MTLDevice, depthMode: MeshDepthMode = .forward, settings: Settings) throws {
        self.device = device
        self.depthMode = depthMode
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    /// The top-left normalized frame position of a light infinitely far along
    /// `direction` (from the camera toward the light, in world space), or nil when
    /// it lies behind the camera.
    public static func screenPosition(
        ofDirection direction: SIMD3<Float>, camera: PerspectiveCamera, cameraTransform: SceneTransform, aspectRatio: Float
    ) throws -> SIMD2<Float>? {
        let clipFromWorld = try camera.projection(aspectRatio: aspectRatio) * simd_inverse(cameraTransform.matrix)
        return screenPosition(clip: clipFromWorld * SIMD4(direction, 0))
    }

    static func screenPosition(clip: SIMD4<Float>) -> SIMD2<Float>? {
        guard clip.w > 1e-6, clip.x.isFinite, clip.y.isFinite else { return nil }
        let ndc = SIMD2(clip.x, clip.y) / clip.w
        return SIMD2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_godrays_composite", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_godrays_mask")
        _ = try shaders.compute("frostfield_godrays_blur")
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let s = settings
        let sourceValid: Bool = switch s.source {
        case .sun(let radius): radius.isFinite && radius > 0 && radius <= 4
        case .sky(let threshold): threshold.isFinite && threshold >= 0
        }
        guard sourceValid, s.lightPosition.x.isFinite, s.lightPosition.y.isFinite,
              abs(s.lightPosition.x) < 1e4, abs(s.lightPosition.y) < 1e4,
              s.color.x.isFinite, s.color.y.isFinite, s.color.z.isFinite, s.color.min() >= 0,
              s.intensity.isFinite, (0...64).contains(s.intensity), s.density.isFinite, (0...1).contains(s.density),
              s.decay.isFinite, (0...1).contains(s.decay), (2...128).contains(s.samples), (1...3).contains(s.passes),
              s.resolutionScale.isFinite, (0.25...1).contains(s.resolutionScale) else {
            throw PostProcessEffectError.invalidSettings(
                "God rays need a finite light position, nonnegative color, intensity 0...64, density and decay 0...1, 2 to 128 samples, 1 to 3 passes, a resolution scale 0.25...1 and a positive sun radius or nonnegative threshold."
            )
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: true
        )
        let width = max(1, Int((Float(source.width) * s.resolutionScale).rounded()))
        let height = max(1, Int((Float(source.height) * s.resolutionScale).rounded()))
        if scratch?.0.width != width || scratch?.0.height != height {
            scratch = (
                try PostProcessShaders.scratch(device: device, width: width, height: height, label: "Frostfield god rays A"),
                try PostProcessShaders.scratch(device: device, width: width, height: height, label: "Frostfield god rays B")
            )
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        guard let depth, let scratch else { throw PostProcessEffectError.missingDepth }
        try depthMode.requireProjectionDepth()
        let mask = try shaders.compute("frostfield_godrays_mask")
        let blur = try shaders.compute("frostfield_godrays_blur")
        let composite = try shaders.render("frostfield_godrays_composite", formats: [destination.pixelFormat])
        let s = settings
        let sourceValues: SIMD2<Float> = switch s.source {
        case .sun(let radius): SIMD2(0, radius)
        case .sky(let threshold): SIMD2(1, threshold)
        }
        var parameters = Parameters(
            light: SIMD4(s.lightPosition.x, s.lightPosition.y, s.density, s.decay),
            source: SIMD4(sourceValues.x, sourceValues.y, Float(source.width) / Float(source.height), depthMode == .reversed ? 1 : 0),
            color: SIMD4(s.color, s.intensity),
            pass: SIMD4(Float(s.samples), 1 / Float(s.samples), 0, 0)
        )
        try PostProcessShaders.compute(
            label: "Frostfield god rays mask", commandBuffer: commandBuffer, pipeline: mask,
            width: scratch.0.width, height: scratch.0.height
        ) { encoder in
            encoder.setTexture(source, index: 0)
            encoder.setTexture(depth, index: 1)
            encoder.setTexture(scratch.0, index: 2)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
        var input = scratch.0
        var output = scratch.1
        for pass in 0..<s.passes {
            parameters.pass.y = 1 / pow(Float(s.samples), Float(pass + 1))
            let (from, to) = (input, output)
            try PostProcessShaders.compute(
                label: "Frostfield god rays pass \(pass)", commandBuffer: commandBuffer, pipeline: blur,
                width: to.width, height: to.height
            ) { encoder in
                encoder.setTexture(from, index: 0)
                encoder.setTexture(to, index: 1)
                encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
            }
            swap(&input, &output)
        }
        let shafts = input
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield god rays composite", commandBuffer: commandBuffer, pipeline: composite
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(shafts, index: 1)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
    }

    private struct Parameters {
        var light: SIMD4<Float>
        var source: SIMD4<Float>
        var color: SIMD4<Float>
        var pass: SIMD4<Float>
    }
}

/// A procedural lens flare for bright lights, like the Three.js `Lensflare` addon,
/// with a glare core, starburst streaks, a dispersive halo ring and ghosts.
///
/// Each light's visibility is the fraction of `occlusionSamples` depth taps within
/// `occlusionRadius` pixels of its position that see past the scene: cleared depth
/// for a light at infinity, or depth no nearer than the light's own stored depth.
/// Taps outside the frame count as hidden, so flares fade as their light leaves it.
/// A compute pass writes the visibilities on the GPU, so nothing is read back.
///
/// Elements are evaluated per pixel in frame-height units centered on the frame.
/// A ghost at `distance` sits at `light · (1 − 2 · distance)` from the center, as in
/// Three.js: 0 on the light, ½ at the center, 1 mirrored. The composite adds
/// `visibility · intensity · color ·` (glare, starburst, halo and ghost profiles) to
/// the scene RGB and keeps alpha.
public final class LensFlare: PostProcessEffect {
    public static let maximumLights = 8
    public static let maximumGhosts = 16

    public struct Light: Sendable, Hashable {
        /// Top-left normalized frame position.
        public var position: SIMD2<Float>
        /// Linear RGB color of the flare.
        public var color: SIMD3<Float>
        public var intensity: Float
        /// The light's stored Metal depth, or nil for a light at infinity seen only through cleared depth.
        public var depth: Float?

        public init(position: SIMD2<Float>, color: SIMD3<Float> = .one, intensity: Float = 1, depth: Float? = nil) {
            self.position = position
            self.color = color
            self.intensity = intensity
            self.depth = depth
        }

        var isValid: Bool {
            position.x.isFinite && position.y.isFinite && abs(position.x) < 1e4 && abs(position.y) < 1e4
                && intensity.isFinite && intensity >= 0 && color.isFiniteVector && color.min() >= 0
                && (depth.map { $0.isFinite && (0...1).contains($0) } ?? true)
        }

        /// Projects a world-space point light, returning nil when it is behind the camera.
        ///
        /// A light beyond the far plane gets a nil depth, so only cleared depth reveals
        /// it, as for a light at infinity: every rendered surface is nearer. A light
        /// between the camera and the near plane gets the near-plane depth, so no
        /// rendered surface can hide it.
        public init?(
            worldPosition: SIMD3<Float>, camera: PerspectiveCamera, cameraTransform: SceneTransform, aspectRatio: Float,
            depthMode: MeshDepthMode = .forward, color: SIMD3<Float> = .one, intensity: Float = 1
        ) throws {
            try depthMode.requireProjectionDepth()
            let clipFromWorld = try camera.projection(aspectRatio: aspectRatio) * simd_inverse(cameraTransform.matrix)
            let clip = clipFromWorld * SIMD4(worldPosition, 1)
            guard let position = GodRays.screenPosition(clip: clip), clip.z.isFinite else { return nil }
            let stored = clip.z / clip.w
            let forward: Float? = stored >= 1 ? nil : max(stored, 0)
            self.init(
                position: position, color: color, intensity: intensity,
                depth: forward.map { depthMode == .forward ? $0 : 1 - $0 }
            )
        }
    }

    public struct Ghost: Sendable, Hashable {
        /// Position along the light–center axis: 0 on the light, ½ at the center, 1 mirrored.
        public var distance: Float
        /// Radius as a fraction of the frame height.
        public var radius: Float
        /// Linear RGB tint, multiplied by the light color.
        public var color: SIMD3<Float>
        /// Fraction of the radius over which the edge fades, from 0 to 1.
        public var softness: Float

        public init(distance: Float, radius: Float, color: SIMD3<Float>, softness: Float = 0.3) {
            self.distance = distance
            self.radius = radius
            self.color = color
            self.softness = softness
        }
    }

    public struct Settings: Sendable, Hashable {
        /// Gaussian glare core standard deviation as a fraction of the frame height; 0 disables it.
        public var glareRadius: Float
        public var glareIntensity: Float
        /// Number of starburst streaks; 0 disables them.
        public var starburstRays: Int
        public var starburstIntensity: Float
        /// Exponential falloff length of the streaks as a fraction of the frame height.
        public var starburstLength: Float
        /// Exponent that narrows the streaks.
        public var starburstSharpness: Float
        public var starburstRotation: Float
        /// Halo ring radius and Gaussian width as fractions of the frame height.
        public var haloRadius: Float
        public var haloWidth: Float
        /// Zero disables the halo.
        public var haloIntensity: Float
        /// Relative red and blue radius offsets that split the ring into a spectrum.
        public var haloDispersion: Float
        public var ghosts: [Ghost]
        /// Ghost aperture blades: 0 for discs, 3 to 16 for polygons.
        public var ghostBlades: Int
        public var ghostRotation: Float
        /// Radius in pixels of the visibility test around each light, from 0 to 64.
        public var occlusionRadius: Float
        /// Depth taps per light, from 1 to 256.
        public var occlusionSamples: Int
        /// Tolerance in stored depth when testing a light with a depth.
        public var depthBias: Float

        public init(
            glareRadius: Float = 0.03, glareIntensity: Float = 1, starburstRays: Int = 6, starburstIntensity: Float = 0.4,
            starburstLength: Float = 0.12, starburstSharpness: Float = 24, starburstRotation: Float = 0.3,
            haloRadius: Float = 0.25, haloWidth: Float = 0.012, haloIntensity: Float = 0.08, haloDispersion: Float = 0.04,
            ghosts: [Ghost] = LensFlare.defaultGhosts, ghostBlades: Int = 6, ghostRotation: Float = 0.2,
            occlusionRadius: Float = 6, occlusionSamples: Int = 32, depthBias: Float = 1e-4
        ) {
            self.glareRadius = glareRadius
            self.glareIntensity = glareIntensity
            self.starburstRays = starburstRays
            self.starburstIntensity = starburstIntensity
            self.starburstLength = starburstLength
            self.starburstSharpness = starburstSharpness
            self.starburstRotation = starburstRotation
            self.haloRadius = haloRadius
            self.haloWidth = haloWidth
            self.haloIntensity = haloIntensity
            self.haloDispersion = haloDispersion
            self.ghosts = ghosts
            self.ghostBlades = ghostBlades
            self.ghostRotation = ghostRotation
            self.occlusionRadius = occlusionRadius
            self.occlusionSamples = occlusionSamples
            self.depthBias = depthBias
        }
    }

    /// Six hexagonal ghosts along the axis with warm-to-cool tints.
    public static let defaultGhosts: [Ghost] = [
        Ghost(distance: 0.6, radius: 0.035, color: SIMD3(0.10, 0.14, 0.22)),
        Ghost(distance: 0.7, radius: 0.06, color: SIMD3(0.12, 0.09, 0.05)),
        Ghost(distance: 0.9, radius: 0.025, color: SIMD3(0.06, 0.16, 0.10)),
        Ghost(distance: 1.0, radius: 0.09, color: SIMD3(0.05, 0.06, 0.12), softness: 0.6),
        Ghost(distance: 1.25, radius: 0.04, color: SIMD3(0.15, 0.08, 0.12)),
        Ghost(distance: 0.35, radius: 0.02, color: SIMD3(0.16, 0.12, 0.06)),
    ]

    public let device: any MTLDevice
    public var readsDepth: Bool { true }
    public var settings: Settings
    /// Up to eight lights; an empty list leaves the frame unchanged.
    public var lights: [Light]
    public var depthMode: MeshDepthMode
    /// One visibility per light slot, written on the GPU before the composite reads it.
    private let visibility: any MTLBuffer
    private let shaders: PostProcessShaders

    public init(
        device: any MTLDevice, lights: [Light] = [], depthMode: MeshDepthMode = .forward, settings: Settings = Settings()
    ) throws {
        self.device = device
        self.lights = lights
        self.depthMode = depthMode
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
        guard let buffer = device.makeBuffer(length: Self.maximumLights * MemoryLayout<Float>.stride, options: .storageModePrivate) else {
            throw PostProcessEffectError.allocation
        }
        buffer.label = "Frostfield lens flare visibility"
        visibility = buffer
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_flare_composite", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_flare_visibility")
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let s = settings
        let scalars = [
            s.glareRadius, s.glareIntensity, s.starburstIntensity, s.starburstLength, s.starburstSharpness,
            s.haloRadius, s.haloWidth, s.haloIntensity, s.haloDispersion, s.occlusionRadius, s.depthBias,
        ]
        guard scalars.allSatisfy({ $0.isFinite && $0 >= 0 }), s.starburstRotation.isFinite, s.ghostRotation.isFinite,
              (0...64).contains(s.starburstRays), s.ghostBlades == 0 || (3...16).contains(s.ghostBlades),
              s.ghosts.count <= Self.maximumGhosts, (0...64).contains(s.occlusionRadius),
              (1...256).contains(s.occlusionSamples),
              s.ghosts.allSatisfy({ ghost in
                  ghost.distance.isFinite && ghost.radius.isFinite && ghost.radius > 0 && ghost.softness.isFinite
                      && (0...1).contains(ghost.softness) && ghost.color.min() >= 0 && ghost.color.max().isFinite
              }) else {
            throw PostProcessEffectError.invalidSettings(
                "Lens flare values must be finite and nonnegative, with at most 16 ghosts of positive radius, 0 or 3 to 16 blades, 0 to 64 streaks, an occlusion radius up to 64 pixels and 1 to 256 samples."
            )
        }
        guard lights.count <= Self.maximumLights, lights.allSatisfy(\.isValid) else {
            throw PostProcessEffectError.invalidSettings(
                "Lens flare takes up to eight lights with finite positions, nonnegative color and intensity, and depth from 0 to 1."
            )
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: true
        )
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        guard let depth else { throw PostProcessEffectError.missingDepth }
        try depthMode.requireProjectionDepth()
        let visibilityKernel = try shaders.compute("frostfield_flare_visibility")
        let composite = try shaders.render("frostfield_flare_composite", formats: [destination.pixelFormat])
        let s = settings
        var packedLights = lights.map(FlareDepthVisibility.Light.init)
        // Metal requires a nonempty binding; an unused slot keeps zero intensity.
        if packedLights.isEmpty { packedLights.append(.init(.init(position: .zero, intensity: 0))) }
        var packedGhosts = s.ghosts.map { ghost in
            PackedGhost(geometry: SIMD4(ghost.distance, ghost.radius, ghost.softness, 0), color: SIMD4(ghost.color, 0))
        }
        if packedGhosts.isEmpty { packedGhosts.append(PackedGhost(geometry: .zero, color: .zero)) }
        var parameters = Parameters(
            glare: SIMD4(s.glareRadius, s.glareIntensity, Float(source.width) / Float(source.height), Float(lights.count)),
            starburst: SIMD4(Float(s.starburstRays), s.starburstIntensity, s.starburstLength, s.starburstSharpness),
            halo: SIMD4(s.haloRadius, s.haloWidth, s.haloIntensity, s.haloDispersion),
            shape: SIMD4(s.starburstRotation, Float(s.ghostBlades), s.ghostRotation, Float(s.ghosts.count))
        )
        if !lights.isEmpty {
            try FlareDepthVisibility.encode(
                lights: packedLights, radius: s.occlusionRadius, samples: s.occlusionSamples, bias: s.depthBias,
                depthMode: depthMode, depth: depth, output: visibility, pipeline: visibilityKernel, commandBuffer: commandBuffer
            )
        }
        try packedLights.withUnsafeBytes { lightBytes in
            try packedGhosts.withUnsafeBytes { ghostBytes in
                try PostProcessShaders.drawFullScreen(
                    into: destination, label: "Frostfield lens flare", commandBuffer: commandBuffer, pipeline: composite
                ) { encoder in
                    encoder.setFragmentTexture(source, index: 0)
                    encoder.setFragmentBytes(lightBytes.baseAddress!, length: lightBytes.count, index: 0)
                    encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 1)
                    encoder.setFragmentBuffer(visibility, offset: 0, index: 2)
                    encoder.setFragmentBytes(ghostBytes.baseAddress!, length: ghostBytes.count, index: 3)
                }
            }
        }
    }

    private struct PackedGhost {
        var geometry: SIMD4<Float>
        var color: SIMD4<Float>
    }

    private struct Parameters {
        var glare: SIMD4<Float>
        var starburst: SIMD4<Float>
        var halo: SIMD4<Float>
        var shape: SIMD4<Float>
    }
}

enum FlareDepthVisibility {
    struct Light {
        let position: SIMD4<Float>
        let color: SIMD4<Float>

        init(_ light: LensFlare.Light) {
            position = SIMD4(light.position.x, light.position.y, light.depth ?? -1, light.intensity)
            color = SIMD4(light.color, 0)
        }
    }

    static func encode(
        lights: [Light], radius: Float, samples: Int, bias: Float, depthMode: MeshDepthMode,
        depth: any MTLTexture, output: any MTLBuffer, pipeline: any MTLComputePipelineState,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw PostProcessEffectError.encoder }
        encoder.label = "Frostfield lens flare visibility"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(depth, index: 0)
        lights.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        var parameters = SIMD4(radius, bias, depthMode == .reversed ? 1 : 0, Float(samples))
        var count = UInt32(lights.count)
        encoder.setBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: lights.count, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }
}

extension GodRays: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}

extension LensFlare: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
