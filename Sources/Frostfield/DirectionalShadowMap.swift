import Foundation
import Metal
import simd

/// Orthographic depth maps for a directional light on a single Metal queue.
///
/// A single map covers a cube centered at `center` in the light's coordinate system.
/// A cascaded map, like the Three.js CSM addon, splits the camera's view range into
/// up to four slices and gives each its own map. `SunLight` fits its cascades
/// during scene encoding. For a manually managed map, call `fitCascades(camera:cameraTransform:aspectRatio:)`
/// or `fitCascades(view:splitFractions:)` whenever the camera moves. Each map is fitted
/// to a bounding sphere of its slice and snapped to whole texels.
public final class DirectionalShadowMap {
    /// The largest light angular radius, in radians, that the contact-hardening filter accepts.
    public static let maximumLightAngularRadius: Float = 0.25
    /// The most cascades one map holds.
    public static let maximumCascadeCount = 4
    /// The most transmissive surface crossings stored per shadow texel.
    public static let maximumTransmissionLayers = 4

    /// How a cascaded map divides the camera's view range.
    public struct Cascades: Sendable, Equatable {
        /// The number of maps, from 1 to `maximumCascadeCount`.
        public var count: Int
        /// The practical split scheme's blend, from 0 (uniform splits) to 1 (logarithmic splits).
        /// Split `i` of `n` lies at `λ·near·(far/near)^(i/n) + (1−λ)·(near + (far−near)·i/n)`.
        public var splitLambda: Float
        /// The farthest view distance that receives shadows, at most the camera's far
        /// plane. Nil uses the far plane. Beyond it surfaces are unshadowed.
        public var maximumDistance: Float?
        /// The fraction of each cascade's depth range, from 0 to 0.5, that cross-fades into
        /// the next cascade. The last cascade fades to unshadowed over the same fraction.
        public var blendFraction: Float
        /// World units that each map's depth range extends toward the light beyond the
        /// slice's bounding sphere, so casters outside the view still cast. Nonnegative.
        public var casterMargin: Float

        public init(
            count: Int = 4, splitLambda: Float = 0.75, maximumDistance: Float? = nil,
            blendFraction: Float = 0.1, casterMargin: Float = 20
        ) {
            self.count = count
            self.splitLambda = splitLambda
            self.maximumDistance = maximumDistance
            self.blendFraction = blendFraction
            self.casterMargin = casterMargin
        }

        var isValid: Bool {
            (1...DirectionalShadowMap.maximumCascadeCount).contains(count)
                && splitLambda.isFinite && (0...1).contains(splitLambda)
                && (maximumDistance.map { $0.isFinite && $0 > 0 } ?? true)
                && blendFraction.isFinite && (0...0.5).contains(blendFraction)
                && casterMargin.isFinite && casterMargin >= 0
        }
    }

    /// The light-space projection of each map, one per array slice. Empty for a
    /// cascaded map until it has been fitted to a camera.
    package private(set) var cascadeViews: [simd_float4x4]
    /// The single map's projection, or the first cascade's.
    package var clipFromWorld: simd_float4x4 { cascadeViews.first ?? matrix_identity_float4x4 }
    /// A Depth32Float 2D array with one slice per map.
    package let texture: any MTLTexture
    /// Filtered first and second moments, or an unused one-texel binding.
    package let shadowMoments: any MTLTexture
    private let variancePass: VarianceShadowPass?
    let transmittedShadows: DirectionalTransmissionPass?
    /// The number of transmissive crossings per texel. Zero disables colored shadows.
    /// Supports stock PBR glass and PBR materials whose surface graph authors `SurfaceShaderGraph.Outputs.castShadowColor`.
    /// Opaque depth is separate and other receiver materials still use it.
    /// Additional glass crossings beyond this count do not contribute.
    public var transmissionLayers: Int { transmittedShadows?.layerCount ?? 0 }
    package private(set) var lightParameters: SIMD4<Float>
    /// Basic sampling uses one depth comparison and the configured receiver bias.
    /// The light's angular radius applies to filtered sampling.
    public let sampling: ShadowMapSampling
    /// Box filter radius in texels for a variance map.
    public let varianceBlurRadius: Int?
    /// On-demand maps hold one command buffer at a time on their creation queue.
    /// The first use and changes to the fitted light-space projections rebuild depth.
    public let updateMode: ShadowMapUpdateMode
    private let updateCache: ShadowMapUpdateCache?
    package private(set) var shadowSettings: SIMD2<Float>
    /// The record matching `FrostfieldDirectionalShadow` in Metal.
    package private(set) var shaderUniforms: [SIMD4<Float>] = []
    private static let shaderUniformCount = 28
    /// An unsampled record for material variants that can draw without a directional map.
    static let unusedShaderUniforms = Array(repeating: SIMD4<Float>.zero, count: shaderUniformCount)

    /// Linear RGB tint of the directional incident light, with components in 0...1.
    public private(set) var color: SIMD3<Float>

    /// Signed world-space distance to shift receivers along their geometric normal for shadow lookup.
    public private(set) var normalBias: Float
    /// How strongly the map attenuates direct light, from zero to full strength at one.
    public private(set) var shadowStrength: Float

    /// The cascade layout, or nil for a single map.
    public let cascades: Cascades?
    /// Sun-owned maps fit the active scene view before the shadow draw.
    package var automaticallyFitsCamera = false
    /// The far view distance of each fitted cascade, in world units. Empty for a single
    /// map or before the first fit.
    public private(set) var cascadeSplits: [Float] = []
    /// The world-space width of one shadow texel in each map.
    public private(set) var texelSizes: [Float]
    /// The texel resolution of each square map.
    public let resolution: Int

    // Normalized depth per unit of map UV in each map; one for a single map.
    private var depthPerUV: [Float]
    private var viewForward = SIMD3<Float>(0, 0, -1)
    private var viewOrigin = SIMD3<Float>.zero
    private var viewNear: Float = 0
    private var directionToLight: SIMD3<Float>
    private var fit: CascadeFit?
    private var relativeViews: [RelativeView] = []

    struct RelativeDraw {
        let clips: [simd_float4x4]
        let splitPlane: SIMD4<Float>
    }

    /// Each light-space projection is anchored in world space. Small
    /// offsets within its map stay separate from the potentially large origin.
    private struct RelativeView {
        let center: SIMD3<Double>
        let clipFromCenter: simd_double4x4

        func clipFromLocal(_ relativeWorld: simd_double4x4, cameraOrigin: SIMD3<Double>) -> simd_float4x4 {
            var centeredWorld = relativeWorld
            centeredWorld[3] += SIMD4(cameraOrigin - center, 0)
            return DirectionalShadowMap.pack(clipFromCenter * centeredWorld)
        }

        /// Keep the previous representation for the same snapped grid cell.
        /// A changed cell moves at least one texel along one of the light axes.
        func keepingGridAnchor(_ previous: Self?, clipTexel: SIMD3<Double>) -> Self {
            guard let previous,
                  (0..<3).allSatisfy({ previous.clipFromCenter[$0] == clipFromCenter[$0] }) else { return self }
            let rebased = previous.clipFromCenter * SIMD4(center - previous.center, 1)
            let translation = clipFromCenter[3]
            let sameCell = (0..<3).allSatisfy { abs(rebased[$0] - translation[$0]) < clipTexel[$0] / 4 }
            return sameCell ? previous : self
        }
    }

    private struct CascadeFit {
        enum Projection {
            case symmetricPerspective(spread: Double)
            case explicit(clipFromView: simd_float4x4, worldFromView: simd_float4x4)
        }

        let origin: SIMD3<Float>
        let forward: SIMD3<Float>
        let near: Float
        let splits: [Float]
        let projection: Projection

        func sphere(near: Double, far: Double) -> (center: SIMD3<Double>, offset: SIMD3<Double>, radius: Double) {
            let origin = SIMD3<Double>(origin)
            switch projection {
            case .symmetricPerspective(let spread):
                let distance = min(far, (near + far) * (1 + spread) / 2)
                let radius = sqrt((far - distance) * (far - distance) + spread * far * far)
                let offset = SIMD3<Double>(forward) * distance
                return (origin + offset, offset, radius)

            case .explicit(let p, let world):
                let scales = SIMD2(Double(p.columns.0.x), Double(p.columns.1.y))
                let slope = SIMD2(Double(p.columns.2.x), Double(p.columns.2.y))
                let offset = SIMD2(Double(p.columns.3.x), Double(p.columns.3.y))
                // At each depth, solve clip.xy = ndc.xy * clip.w for view-space x and y.
                let nearCenter = (slope * near - offset) / scales
                let farCenter = (slope * far - offset) / scales
                let nearWidth = (Double(p.columns.3.w) - Double(p.columns.2.w) * near) / scales
                let farWidth = (Double(p.columns.3.w) - Double(p.columns.2.w) * far) / scales
                let halfShift = abs(farCenter - nearCenter) / 2
                let centerXY = (nearCenter + farCenter) / 2
                let halfDepth = (far - near) / 2
                let radius = simd_length(SIMD3(simd_max(nearWidth, farWidth) + halfShift, halfDepth))
                let right = SIMD3<Double>(SIMD3(world.columns.0.x, world.columns.0.y, world.columns.0.z))
                let up = SIMD3<Double>(SIMD3(world.columns.1.x, world.columns.1.y, world.columns.1.z))
                let x = right * centerXY.x, y = up * centerXY.y
                let z = SIMD3<Double>(forward) * ((near + far) / 2)
                return (origin + x + y + z, x + y + z, radius)
            }
        }
    }

    private let queue: any MTLCommandQueue
    private let pipeline: any MTLRenderPipelineState
    private let skinnedPipeline: any MTLRenderPipelineState
    private let maskedPipeline: any MTLRenderPipelineState
    private let skinnedMaskedPipeline: any MTLRenderPipelineState
    private let depthState: any MTLDepthStencilState

    /// Creates a single map. Resolution is clamped to 1...4096. Bias is measured in
    /// normalized depth. An angular radius from 0 to 0.25 radians enables contact-hardening shadows.
    /// Variance sampling needs a resolution in 1...2048, a blur radius in 0...16,
    /// and a zero angular radius. A positive `transmissionLayers` enables up to four
    /// nearest glass crossings per texel for PBR receivers, with a 128 MiB resource
    /// limit. See `transmissionLayers` for the supported caster inputs.
    public init(
        queue: any MTLCommandQueue,
        resolution: Int,
        directionToLight: SIMD3<Float>,
        center: SIMD3<Float> = .zero,
        halfExtent: Float,
        color: SIMD3<Float> = SIMD3(repeating: 1),
        intensity: Float = 1,
        bias: Float = 0.0015,
        normalBias: Float = 0,
        shadowStrength: Float = 1,
        lightAngularRadius: Float = 0,
        sampling: ShadowMapSampling = .filtered,
        varianceBlurRadius: Int = 3,
        updateMode: ShadowMapUpdateMode = .automatic,
        transmissionLayers: Int = 0
    ) throws {
        let projection = try Self.singleProjection(directionToLight: directionToLight, center: center, halfExtent: halfExtent)
        try Self.validate(intensity: intensity, bias: bias, lightAngularRadius: lightAngularRadius)
        try Self.validateColor(color)
        try Self.validateReceiver(normalBias: normalBias, shadowStrength: shadowStrength)
        let size = min(max(resolution, 1), 4096)
        let variance = try Self.prepareVariance(
            device: queue.device, sampling: sampling, resolution: resolution, slices: 1,
            radius: varianceBlurRadius, angularRadius: lightAngularRadius
        )
        let transmitted = try Self.prepareTransmission(
            device: queue.device, resolution: size, slices: 1, layerCount: transmissionLayers
        )
        let prepared = try Self.prepare(queue: queue, resolution: size, slices: 1)
        transmittedShadows = transmitted
        self.queue = queue
        self.sampling = sampling
        self.updateMode = updateMode
        updateCache = updateMode == .onDemand ? ShadowMapUpdateCache() : nil
        pipeline = prepared.pipelines[0]
        skinnedPipeline = prepared.pipelines[1]
        maskedPipeline = prepared.pipelines[2]
        skinnedMaskedPipeline = prepared.pipelines[3]
        depthState = prepared.depthState
        texture = prepared.texture
        variancePass = variance
        self.varianceBlurRadius = variance == nil ? nil : varianceBlurRadius
        shadowMoments = try variance?.moments ?? VarianceShadowPass.fallback(device: queue.device)
        self.resolution = size
        self.color = color
        self.normalBias = normalBias
        self.shadowStrength = shadowStrength
        cascades = nil
        relativeViews = [projection.relative]
        cascadeViews = [projection.clipFromWorld]
        texelSizes = [2 * halfExtent / Float(size)]
        depthPerUV = [1]
        self.directionToLight = projection.directionToLight
        lightParameters = SIMD4<Float>(projection.directionToLight, intensity)
        shadowSettings = SIMD2(bias, tan(lightAngularRadius))
        shaderUniforms = Self.uniforms(for: self)
    }

    /// Creates a cascaded map with `cascades.count` maps of `resolution` texels square,
    /// clamped to 1...4096. Fit it to the viewing camera before encoding a scene. Bias is
    /// a fraction of each map's width, as the single map's normalized depth is a fraction
    /// of its cube. The contact-hardening filter widens with blocker distance in world
    /// units in every cascade, up to 16 texels of that cascade. Colored transmission
    /// uses separate layers for each cascade.
    public init(
        queue: any MTLCommandQueue,
        resolution: Int,
        directionToLight: SIMD3<Float>,
        cascades: Cascades,
        color: SIMD3<Float> = SIMD3(repeating: 1),
        intensity: Float = 1,
        bias: Float = 0.0015,
        normalBias: Float = 0,
        shadowStrength: Float = 1,
        lightAngularRadius: Float = 0,
        sampling: ShadowMapSampling = .filtered,
        varianceBlurRadius: Int = 3,
        updateMode: ShadowMapUpdateMode = .automatic,
        transmissionLayers: Int = 0
    ) throws {
        guard cascades.isValid else { throw Failure.invalidCascades }
        let light = try Self.projection(directionToLight: directionToLight, center: .zero, halfExtent: 1).directionToLight
        try Self.validate(intensity: intensity, bias: bias, lightAngularRadius: lightAngularRadius)
        try Self.validateColor(color)
        try Self.validateReceiver(normalBias: normalBias, shadowStrength: shadowStrength)
        let size = min(max(resolution, 1), 4096)
        let variance = try Self.prepareVariance(
            device: queue.device, sampling: sampling, resolution: resolution, slices: cascades.count,
            radius: varianceBlurRadius, angularRadius: lightAngularRadius
        )
        let transmitted = try Self.prepareTransmission(
            device: queue.device, resolution: size, slices: cascades.count, layerCount: transmissionLayers
        )
        let prepared = try Self.prepare(queue: queue, resolution: size, slices: cascades.count)
        transmittedShadows = transmitted
        self.queue = queue
        self.sampling = sampling
        self.updateMode = updateMode
        updateCache = updateMode == .onDemand ? ShadowMapUpdateCache() : nil
        pipeline = prepared.pipelines[0]
        skinnedPipeline = prepared.pipelines[1]
        maskedPipeline = prepared.pipelines[2]
        skinnedMaskedPipeline = prepared.pipelines[3]
        depthState = prepared.depthState
        texture = prepared.texture
        variancePass = variance
        self.varianceBlurRadius = variance == nil ? nil : varianceBlurRadius
        shadowMoments = try variance?.moments ?? VarianceShadowPass.fallback(device: queue.device)
        self.resolution = size
        self.color = color
        self.normalBias = normalBias
        self.shadowStrength = shadowStrength
        self.cascades = cascades
        cascadeViews = []
        texelSizes = []
        depthPerUV = []
        self.directionToLight = light
        lightParameters = SIMD4<Float>(light, intensity)
        shadowSettings = SIMD2(bias, tan(lightAngularRadius))
        shaderUniforms = Self.uniforms(for: self)
    }

    /// Re-aims and re-scales a single map for the next encoded frame, reusing the depth
    /// texture and pipelines. Uses the same validation as the initializer; on failure
    /// the previous settings stay in place. Earlier command buffers keep the values they
    /// encoded. Throws `Failure.cascaded` for a cascaded map.
    public func update(
        directionToLight: SIMD3<Float>, center: SIMD3<Float> = .zero, halfExtent: Float,
        color: SIMD3<Float>? = nil, intensity: Float = 1, bias: Float = 0.0015, lightAngularRadius: Float = 0
    ) throws {
        guard cascades == nil else { throw Failure.cascaded }
        let projection = try Self.singleProjection(directionToLight: directionToLight, center: center, halfExtent: halfExtent)
        try Self.validate(intensity: intensity, bias: bias, lightAngularRadius: lightAngularRadius)
        let color = color ?? self.color
        try Self.validateColor(color)
        guard sampling != .variance || lightAngularRadius == 0 else { throw Failure.varianceSettings }
        relativeViews = [projection.relative]
        cascadeViews = [projection.clipFromWorld]
        texelSizes = [2 * halfExtent / Float(resolution)]
        self.directionToLight = projection.directionToLight
        lightParameters = SIMD4<Float>(projection.directionToLight, intensity)
        self.color = color
        shadowSettings = SIMD2(bias, tan(lightAngularRadius))
        shaderUniforms = Self.uniforms(for: self)
    }

    /// Re-aims a cascaded map's light. A fitted map refits its cascades to the same
    /// camera. On failure the previous settings stay in place. Throws `Failure.notCascaded`
    /// for a single map.
    public func update(
        directionToLight: SIMD3<Float>, color: SIMD3<Float>? = nil,
        intensity: Float = 1, bias: Float = 0.0015, lightAngularRadius: Float = 0
    ) throws {
        guard cascades != nil else { throw Failure.notCascaded }
        let light = try Self.projection(directionToLight: directionToLight, center: .zero, halfExtent: 1).directionToLight
        try Self.validate(intensity: intensity, bias: bias, lightAngularRadius: lightAngularRadius)
        let color = color ?? self.color
        try Self.validateColor(color)
        guard sampling != .variance || lightAngularRadius == 0 else { throw Failure.varianceSettings }
        if let fit {
            try refit(fit, light: light)
        }
        self.directionToLight = light
        lightParameters = SIMD4<Float>(light, intensity)
        self.color = color
        shadowSettings = SIMD2(bias, tan(lightAngularRadius))
        shaderUniforms = Self.uniforms(for: self)
    }

    /// Changes the directional light's tint for subsequent draws, preserving
    /// the fitted projection, depth contents and on-demand update state.
    public func update(color: SIMD3<Float>) throws {
        try Self.validateColor(color)
        self.color = color
        shaderUniforms = Self.uniforms(for: self)
    }

    /// Changes the lookup for subsequent draws. The light's intensity, map contents
    /// and on-demand update state stay intact. Invalid input leaves both values intact.
    public func updateReceiver(normalBias: Float, shadowStrength: Float) throws {
        try Self.validateReceiver(normalBias: normalBias, shadowStrength: shadowStrength)
        self.normalBias = normalBias
        self.shadowStrength = shadowStrength
        shaderUniforms = Self.uniforms(for: self)
    }

    /// Fits each cascade to its slice of the camera's view, for a drawable with this
    /// width-to-height aspect ratio. Call it after every camera change and before encoding.
    /// On failure the previous fit stays in place. Throws `Failure.notCascaded` for a single map.
    public func fitCascades(camera: PerspectiveCamera, cameraTransform: SceneTransform, aspectRatio: Float) throws {
        try fitPerspective(camera: camera, transform: cameraTransform, aspectRatio: aspectRatio, splitFractions: nil)
    }

    /// Fits a perspective camera with caller-authored split positions. Supply exactly
    /// `cascades.count - 1` strictly increasing fractions between 0 and 1. Zero is the
    /// camera's near distance and one is the fitted far distance after applying
    /// `Cascades.maximumDistance`. The last split is always at one.
    public func fitCascades(
        camera: PerspectiveCamera, cameraTransform: SceneTransform, aspectRatio: Float, splitFractions: [Float]
    ) throws {
        try fitPerspective(
            camera: camera, transform: cameraTransform, aspectRatio: aspectRatio, splitFractions: splitFractions
        )
    }

    /// Fits the perspective or orthographic camera for an explicit scene pass. A glTF
    /// camera view supplies this input as `GLTFScene.Camera.View.graphCamera`. The view
    /// must be rigid and the projection may be off-center. Forward and reversed depth
    /// give the same fit. An infinite perspective far plane needs `Cascades.maximumDistance`.
    /// By default the configured practical splits apply, with uniform spacing for an
    /// orthographic near plane at zero. Pass `splitFractions` to place intermediate
    /// splits as in the perspective overload. On failure the previous fit stays in place.
    public func fitCascades(view: ShaderGraphCamera, splitFractions: [Float]? = nil) throws {
        guard cascades != nil else { throw Failure.notCascaded }
        let projection = view.projectionMatrix
        let world = simd_inverse(view.viewMatrix)
        let isOrthographic = projection.columns.2.w == 0 && projection.columns.3.w > 0
        let isPerspective = projection.columns.2.w < 0 && projection.columns.3.w == 0 && view.near > 0
        guard isOrthographic || isPerspective,
              projection.columns.0.x > 0, projection.columns.1.y > 0,
              projection.columns.0.y == 0, projection.columns.1.x == 0,
              projection.columns.0.w == 0, projection.columns.1.w == 0,
              world.columns.0.w == 0, world.columns.1.w == 0, world.columns.2.w == 0,
              abs(world.columns.3.w - 1) <= 1e-4,
              ScreenSpaceProjection.viewFromWorld(world) != nil else { throw Failure.invalidCamera }
        let origin = SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
        guard Self.isFinite(origin) else { throw Failure.invalidCamera }
        let forward = simd_normalize(-SIMD3(world.columns.2.x, world.columns.2.y, world.columns.2.z))
        let splits = try splitDistances(near: view.near, far: view.far, fractions: splitFractions)
        let fit = CascadeFit(
            origin: origin, forward: forward, near: view.near, splits: splits,
            projection: .explicit(clipFromView: projection, worldFromView: world)
        )
        try refit(fit, light: directionToLight)
        self.fit = fit
    }

    private func fitPerspective(
        camera: PerspectiveCamera, transform: SceneTransform, aspectRatio: Float, splitFractions: [Float]?
    ) throws {
        guard cascades != nil else { throw Failure.notCascaded }
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw Failure.invalidCamera }
        let determinant = simd_determinant(transform.matrix)
        guard determinant.isFinite, determinant != 0 else { throw Failure.invalidCamera }
        let rotated = transform.rotation.act(SIMD3<Float>(0, 0, -1))
        let forwardLength = simd_length(rotated)
        guard forwardLength.isNormal, Self.isFinite(transform.translation) else { throw Failure.invalidCamera }
        let scale = abs(SIMD3<Double>(transform.scale))
        let near = Float(Double(camera.near) * scale.z)
        let far = Float(Double(camera.far) * scale.z)
        guard near.isNormal, far.isFinite, far > near else { throw Failure.invalidCamera }
        let tangent = tan(Double(camera.verticalFieldOfView) / 2) / scale.z
        let halfWidth = Double(aspectRatio) * scale.x
        let spread = tangent * tangent * (halfWidth * halfWidth + scale.y * scale.y)
        let forward = rotated / forwardLength * (transform.scale.z > 0 ? 1 : -1)
        let splits = try splitDistances(near: near, far: far, fractions: splitFractions)
        let fit = CascadeFit(
            origin: transform.translation, forward: forward, near: near, splits: splits,
            projection: .symmetricPerspective(spread: spread)
        )
        try refit(fit, light: directionToLight)
        self.fit = fit
    }

    /// The practical split distances for a view range, as in Zhang et al. 2006: the far
    /// distance of each of `count` cascades between `near` and `far`. Throws
    /// `Failure.invalidSplits` unless `0 < near < far` are finite, `count` is 1 to
    /// `maximumCascadeCount` and `lambda` is 0 to 1.
    public static func practicalSplits(near: Float, far: Float, count: Int, lambda: Float) throws -> [Float] {
        guard near.isFinite, far.isFinite, near > 0, far > near,
              (1...maximumCascadeCount).contains(count), lambda.isFinite, (0...1).contains(lambda) else {
            throw Failure.invalidSplits
        }
        return (1...count).map { index in
            let fraction = Double(index) / Double(count)
            let logarithmic = Double(near) * pow(Double(far) / Double(near), fraction)
            let uniform = Double(near) + (Double(far) - Double(near)) * fraction
            return index == count ? far : Float(Double(lambda) * logarithmic + (1 - Double(lambda)) * uniform)
        }
    }

    private func splitDistances(near: Float, far: Float, fractions: [Float]?) throws -> [Float] {
        guard let cascades else { throw Failure.notCascaded }
        let end = min(cascades.maximumDistance ?? far, far)
        guard end.isFinite else { throw Failure.invalidCamera }
        guard end > near else { throw Failure.invalidCascades }

        let splits: [Float]
        if let fractions {
            guard fractions.count == cascades.count - 1 else { throw Failure.invalidSplits }
            var previous: Float = 0
            for fraction in fractions {
                guard fraction.isFinite, fraction > previous, fraction < 1 else { throw Failure.invalidSplits }
                previous = fraction
            }
            let span = Double(end) - Double(near)
            splits = fractions.map { Float(Double(near) + Double($0) * span) } + [end]
        } else if near == 0 {
            splits = (1...cascades.count).map { index in
                Float(Double(end) * Double(index) / Double(cascades.count))
            }
        } else {
            splits = try Self.practicalSplits(near: near, far: end, count: cascades.count, lambda: cascades.splitLambda)
        }
        // Even strictly increasing inputs can round to equal Float distances.
        var previous = near
        for split in splits {
            guard split > previous else { throw Failure.invalidSplits }
            previous = split
        }
        return splits
    }

    private func refit(_ fit: CascadeFit, light: SIMD3<Float>) throws {
        guard let cascades else { throw Failure.notCascaded }
        let basis = Self.basis(light)
        let resolution = Double(self.resolution)
        var views: [simd_float4x4] = []
        var relative: [RelativeView] = []
        var texels: [Float] = []
        var depths: [Float] = []
        var previousFar = Double(fit.near)
        var previousNear = Double(fit.near)
        let origin = SIMD3<Double>(fit.origin)
        for (slice, split) in fit.splits.enumerated() {
            // Each sphere also covers the previous cascade's blend band, where both are sampled.
            let sliceFar = Double(split)
            let sliceNear = max(Double(fit.near), previousFar - Double(cascades.blendFraction) * (previousFar - previousNear))
            previousNear = previousFar
            previousFar = sliceFar
            let (center, offset, radius) = fit.sphere(near: sliceNear, far: sliceFar)
            let texel = 2 * radius / resolution
            guard Float(texel).isNormal else { throw Failure.unrepresentableRegion }
            // Snapping the center to whole texels in light space keeps the grid fixed in the world.
            let x = (simd_dot(basis.right, center) / texel).rounded() * texel
            let y = (simd_dot(basis.up, center) / texel).rounded() * texel
            let z = (simd_dot(basis.light, center) / texel).rounded() * texel
            let depthRange = 2 * radius + Double(cascades.casterMargin)
            let view = try Self.matrix(
                basis: basis, lightSpaceCenter: SIMD3(x, y, z), halfExtent: radius,
                nearOffset: radius + Double(cascades.casterMargin), depthRange: depthRange
            )
            let snappedOffset = SIMD3(
                Self.snappedOffset(origin: origin, offset: offset, axis: basis.right, texel: texel),
                Self.snappedOffset(origin: origin, offset: offset, axis: basis.up, texel: texel),
                Self.snappedOffset(origin: origin, offset: offset, axis: basis.light, texel: texel)
            )
            let local = RelativeView(center: origin, clipFromCenter: Self.projectionMatrix(
                basis: basis, lightSpaceCenter: snappedOffset, halfExtent: radius,
                nearOffset: radius + Double(cascades.casterMargin), depthRange: depthRange
            ))
            let previous = relativeViews.indices.contains(slice) ? relativeViews[slice] : nil
            let clipTexel = SIMD3(texel / radius, texel / radius, texel / depthRange)
            relative.append(local.keepingGridAnchor(previous, clipTexel: clipTexel))
            views.append(view)
            texels.append(Float(texel))
            depths.append(Float(2 * radius / depthRange))
        }
        guard depths.allSatisfy(\.isNormal) else { throw Failure.unrepresentableRegion }
        cascadeViews = views
        relativeViews = relative
        texelSizes = texels
        depthPerUV = depths
        cascadeSplits = fit.splits
        viewForward = fit.forward
        viewOrigin = fit.origin
        viewNear = fit.near
        shaderUniforms = Self.uniforms(for: self)
    }

    /// Snap to the existing world texel grid without adding the small local
    /// offset to a potentially large absolute light-space position.
    private static func snappedOffset(
        origin: SIMD3<Double>, offset: SIMD3<Double>, axis: SIMD3<Double>, texel: Double
    ) -> Double {
        var phase = 0.0
        for index in 0..<3 {
            let product = origin[index] * axis[index]
            // FMA recovers the part of the product that rounded out in Double.
            let remainder = (-product).addingProduct(origin[index], axis[index])
            phase += product.remainder(dividingBy: texel) + remainder.remainder(dividingBy: texel)
        }
        phase = phase.remainder(dividingBy: texel)
        let local = simd_dot(axis, offset)
        let cell = (phase + local) / texel
        let lower = cell.rounded(.down)
        let snapped: Double
        if cell - lower == 0.5 {
            // Preserve away-from-zero rounding on an exact world-grid tie.
            snapped = simd_dot(axis, origin) + local < 0 ? lower : lower + 1
        } else {
            snapped = cell.rounded()
        }
        return snapped * texel - phase
    }

    private static func uniforms(for map: DirectionalShadowMap) -> [SIMD4<Float>] {
        var values: [SIMD4<Float>] = []
        values.reserveCapacity(shaderUniformCount)
        for index in 0..<maximumCascadeCount {
            let view = index < map.cascadeViews.count ? map.cascadeViews[index] : matrix_identity_float4x4
            values += [view.columns.0, view.columns.1, view.columns.2, view.columns.3]
        }
        for index in 0..<maximumCascadeCount {
            let scale = index < map.depthPerUV.count ? map.depthPerUV[index] : 1
            let inverseHalfExtent = map.relativeViews.indices.contains(index)
                ? Float(simd_length(linearRow(0, map.relativeViews[index].clipFromCenter))) : 0
            // The single map's bias is already normalized depth; cascades scale their width fraction.
            values.append(SIMD4(scale, map.shadowSettings.x * scale, 0, inverseHalfExtent))
        }
        var splits = SIMD4<Float>(repeating: .greatestFiniteMagnitude)
        for (index, split) in map.cascadeSplits.enumerated() { splits[index] = split }
        values.append(splits)
        values.append(SIMD4(map.viewForward, map.cascades?.blendFraction ?? 0))
        values.append(SIMD4(map.viewOrigin, Float(map.cascadeViews.count)))
        let sampling: Float = switch map.sampling {
        case .filtered: 0
        case .basic: 1
        case .variance: 2
        }
        values.append(SIMD4(map.cascades == nil ? 0 : 1, map.shadowSettings.y, map.viewNear, sampling))
        // Keep the light axes in world space even when the position projection is rebased to a model.
        let view = map.relativeViews.first?.clipFromCenter ?? matrix_identity_double4x4
        values.append(SIMD4(SIMD3<Float>(simd_normalize(linearRow(0, view))), map.normalBias))
        values.append(SIMD4(SIMD3<Float>(simd_normalize(linearRow(1, view))), map.shadowStrength))
        values.append(SIMD4(SIMD3<Float>(simd_normalize(linearRow(2, view))), Float(map.transmissionLayers)))
        values.append(SIMD4(map.color, 0))
        return values
    }

    private static func linearRow(_ row: Int, _ matrix: simd_double4x4) -> SIMD3<Double> {
        SIMD3(matrix[0][row], matrix[1][row], matrix[2][row])
    }

    /// Throws `Failure.cascadesNotFitted` for a cascaded map that has never been fitted.
    package func validateFitted() throws {
        guard !cascadeViews.isEmpty else { throw Failure.cascadesNotFitted }
    }

    /// Requests new depth on the next usable encode of an on-demand map.
    /// Geometry, visibility and material edits are reflected after that update.
    public func requestUpdate() {
        updateCache?.requestUpdate()
    }

    func beginUse(
        commandBuffer: any MTLCommandBuffer, usingRelativeView: Bool = false
    ) throws -> ShadowMapUpdateCache.Use {
        guard let updateCache else { return .automatic }
        try validate(commandBuffer: commandBuffer)
        let relative = usingRelativeView ? relativeViews : []
        let layout = ShadowMapUpdateCache.Layout(groups: [
            .init(
                slices: cascadeViews.indices, views: cascadeViews[...],
                relativeProjections: relative.map { .init(origin: $0.center, clipFromOrigin: $0.clipFromCenter) }
            )
        ])
        return try updateCache.begin(layout, commandBuffer: commandBuffer)
    }

    func relativeDraw(_ relativeWorld: simd_double4x4, cameraOrigin: SIMD3<Double>) -> RelativeDraw {
        let clips = relativeViews.map { $0.clipFromLocal(relativeWorld, cameraOrigin: cameraOrigin) }
        var splitPlane = SIMD4<Float>.zero
        if cascades != nil {
            // Signed depth from the fitted camera as a plane in model coordinates.
            let forward = SIMD3<Double>(viewForward)
            for column in 0..<3 {
                let axis = relativeWorld[column]
                splitPlane[column] = Float(simd_dot(forward, SIMD3(axis.x, axis.y, axis.z)))
            }
            let origin = relativeWorld[3] + SIMD4(cameraOrigin - SIMD3<Double>(viewOrigin), 0)
            splitPlane.w = Float(simd_dot(forward, SIMD3(origin.x, origin.y, origin.z)))
        }
        return RelativeDraw(clips: clips, splitPlane: splitPlane)
    }

    func relativeShaderUniforms(for draw: RelativeDraw) -> [SIMD4<Float>] {
        var values = shaderUniforms
        for (slice, clip) in draw.clips.enumerated() {
            for column in 0..<4 { values[4 * slice + column] = clip[column] }
        }
        // The first map's z switches to local coordinates.
        // Preserve the world-space light axes, blend fraction and map count.
        let cascadeRecords = Self.maximumCascadeCount * 4
        values[cascadeRecords].z = 1
        if cascades != nil {
            let forward = cascadeRecords + Self.maximumCascadeCount + 1
            values[forward] = SIMD4(draw.splitPlane.x, draw.splitPlane.y, draw.splitPlane.z, values[forward].w)
            values[forward + 1].x = draw.splitPlane.w
        }
        return values
    }

    func beginRead(slice: Int, commandBuffer: any MTLCommandBuffer) throws -> ShadowMapUpdateCache.Use? {
        try updateCache?.beginRead(slice: slice, commandBuffer: commandBuffer)
    }

    func validate(commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.commandQueue === queue else { throw Failure.wrongQueue }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
    }

    /// Starts the depth pass for one map. End this encoder before sampling the map on the same queue.
    package func makeEncoder(
        commandBuffer: any MTLCommandBuffer, slice: Int = 0,
        configure: ((MTLRenderPassDescriptor) throws -> Void)? = nil
    ) throws -> any MTLRenderCommandEncoder {
        try validate(commandBuffer: commandBuffer)
        let pass = MTLRenderPassDescriptor()
        pass.depthAttachment.texture = texture
        pass.depthAttachment.slice = slice
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        try configure?(pass)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }
        encoder.label = cascades == nil ? "Frostfield directional shadow pass" : "Frostfield shadow cascade \(slice)"
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        return encoder
    }

    func encodeMoments(commandBuffer: any MTLCommandBuffer) throws {
        guard let variancePass, let varianceBlurRadius else { return }
        let slices = cascadeViews.indices.map { SIMD2(UInt32($0), UInt32(varianceBlurRadius)) }
        try variancePass.encode(depth: texture, slices: slices, commandBuffer: commandBuffer)
    }

    package func setPipeline(on encoder: any MTLRenderCommandEncoder, skinned: Bool, masked: Bool) {
        if masked {
            encoder.setRenderPipelineState(skinned ? skinnedMaskedPipeline : maskedPipeline)
        } else {
            encoder.setRenderPipelineState(skinned ? skinnedPipeline : pipeline)
        }
    }

    package static func projection(
        directionToLight: SIMD3<Float>, center: SIMD3<Float>, halfExtent: Float
    ) throws -> (clipFromWorld: simd_float4x4, directionToLight: SIMD3<Float>) {
        let projection = try singleProjection(directionToLight: directionToLight, center: center, halfExtent: halfExtent)
        return (projection.clipFromWorld, projection.directionToLight)
    }

    private static func singleProjection(
        directionToLight: SIMD3<Float>, center: SIMD3<Float>, halfExtent: Float
    ) throws -> (clipFromWorld: simd_float4x4, directionToLight: SIMD3<Float>, relative: RelativeView) {
        guard isFinite(directionToLight), directionToLight != .zero else {
            throw Failure.invalidDirection
        }
        guard isFinite(center), halfExtent.isFinite, halfExtent > 0 else {
            throw Failure.invalidRegion
        }

        // Compute the basis and translation in Double before packing the Float matrix.
        let basis = basis(directionToLight)
        let origin = SIMD3<Double>(center)
        let lightSpace = SIMD3(simd_dot(basis.right, origin), simd_dot(basis.up, origin), simd_dot(basis.light, origin))
        // Looking away from the light, the near face maps to depth 0 and the far face to 1.
        let matrix = try matrix(
            basis: basis, lightSpaceCenter: lightSpace, halfExtent: Double(halfExtent),
            nearOffset: Double(halfExtent), depthRange: 2 * Double(halfExtent)
        )
        let relative = projectionMatrix(
            basis: basis, lightSpaceCenter: .zero, halfExtent: Double(halfExtent),
            nearOffset: Double(halfExtent), depthRange: 2 * Double(halfExtent)
        )
        return (matrix, SIMD3<Float>(basis.light), RelativeView(center: origin, clipFromCenter: relative))
    }

    private static func basis(_ directionToLight: SIMD3<Float>) -> (right: SIMD3<Double>, up: SIMD3<Double>, light: SIMD3<Double>) {
        let light = simd_normalize(SIMD3<Double>(directionToLight))
        let referenceUp = abs(light.y) < 0.99 ? SIMD3<Double>(0, 1, 0) : SIMD3<Double>(1, 0, 0)
        let right = simd_normalize(simd_cross(referenceUp, light))
        return (right, simd_cross(light, right), light)
    }

    /// An orthographic light view: x and y span ±`halfExtent` around the light-space
    /// center, and depth runs from `nearOffset` toward the light to `depthRange` beyond it.
    private static func matrix(
        basis: (right: SIMD3<Double>, up: SIMD3<Double>, light: SIMD3<Double>), lightSpaceCenter: SIMD3<Double>,
        halfExtent: Double, nearOffset: Double, depthRange: Double
    ) throws -> simd_float4x4 {
        let matrix = pack(projectionMatrix(
            basis: basis, lightSpaceCenter: lightSpaceCenter, halfExtent: halfExtent,
            nearOffset: nearOffset, depthRange: depthRange
        ))
        guard Float(1 / halfExtent).isNormal, Float(1 / depthRange).isNormal,
              (0..<4).allSatisfy({ isFinite(matrix[$0]) }) else {
            throw Failure.unrepresentableRegion
        }
        return matrix
    }

    private static func projectionMatrix(
        basis: (right: SIMD3<Double>, up: SIMD3<Double>, light: SIMD3<Double>), lightSpaceCenter: SIMD3<Double>,
        halfExtent: Double, nearOffset: Double, depthRange: Double
    ) -> simd_double4x4 {
        let (right, up, light) = basis
        let xyScale = 1 / halfExtent
        let depthScale = 1 / depthRange
        return simd_double4x4(
            SIMD4(right.x * xyScale, up.x * xyScale, -light.x * depthScale, 0),
            SIMD4(right.y * xyScale, up.y * xyScale, -light.y * depthScale, 0),
            SIMD4(right.z * xyScale, up.z * xyScale, -light.z * depthScale, 0),
            SIMD4(-lightSpaceCenter.x * xyScale, -lightSpaceCenter.y * xyScale,
                  (lightSpaceCenter.z + nearOffset) * depthScale, 1)
        )
    }

    private static func pack(_ matrix: simd_double4x4) -> simd_float4x4 {
        simd_float4x4(SIMD4<Float>(matrix[0]), SIMD4<Float>(matrix[1]),
                      SIMD4<Float>(matrix[2]), SIMD4<Float>(matrix[3]))
    }

    private static func validate(intensity: Float, bias: Float, lightAngularRadius: Float) throws {
        guard intensity.isFinite, intensity >= 0 else { throw Failure.invalidIntensity }
        guard bias.isFinite, (0...1).contains(bias) else { throw Failure.invalidBias }
        guard (0...Self.maximumLightAngularRadius).contains(lightAngularRadius) else { throw Failure.invalidAngularRadius }
    }

    private static func validateColor(_ color: SIMD3<Float>) throws {
        guard (0...1).contains(color.x), (0...1).contains(color.y), (0...1).contains(color.z) else {
            throw Failure.invalidColor
        }
    }

    private static func validateReceiver(normalBias: Float, shadowStrength: Float) throws {
        guard normalBias.isFinite else { throw Failure.invalidNormalBias }
        guard (0...1).contains(shadowStrength) else { throw Failure.invalidShadowStrength }
    }

    private static func prepareTransmission(
        device: any MTLDevice, resolution: Int, slices: Int, layerCount: Int
    ) throws -> DirectionalTransmissionPass? {
        guard (0...maximumTransmissionLayers).contains(layerCount) else { throw Failure.transmissionLayers }
        return try layerCount == 0 ? nil
            : DirectionalTransmissionPass(device: device, resolution: resolution, slices: slices, layerCount: layerCount)
    }

    private static func prepare(
        queue: any MTLCommandQueue, resolution: Int, slices: Int
    ) throws -> (pipelines: [any MTLRenderPipelineState], depthState: any MTLDepthStencilState, texture: any MTLTexture) {
        let device = queue.device
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }
        guard let vertex = library.makeFunction(name: "frostfield_shadow_vertex"),
              let skinnedVertex = library.makeFunction(name: "frostfield_skinned_shadow_vertex"),
              let mask = library.makeFunction(name: "frostfield_shadow_mask") else {
            throw Failure.missingShader
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield directional shadow depth"
        descriptor.vertexFunction = vertex
        descriptor.depthAttachmentPixelFormat = .depth32Float
        descriptor.inputPrimitiveTopology = .triangle
        var pipelines: [any MTLRenderPipelineState] = []
        do {
            pipelines.append(try device.makeRenderPipelineState(descriptor: descriptor))
            descriptor.label = "Frostfield directional skinned shadow depth"
            descriptor.vertexFunction = skinnedVertex
            let skinned = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.label = "Frostfield directional masked skinned shadow depth"
            descriptor.fragmentFunction = mask
            let skinnedMasked = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.label = "Frostfield directional masked shadow depth"
            descriptor.vertexFunction = vertex
            let masked = try device.makeRenderPipelineState(descriptor: descriptor)
            pipelines += [skinned, masked, skinnedMasked]
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            throw Failure.resource("shadow depth state")
        }

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: resolution, height: resolution, mipmapped: false
        )
        textureDescriptor.textureType = .type2DArray
        textureDescriptor.arrayLength = slices
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        textureDescriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw Failure.resource("shadow depth texture")
        }
        texture.label = slices == 1 ? "Frostfield directional shadow map" : "Frostfield cascaded shadow maps"
        return (pipelines, depthState, texture)
    }

    private static func isFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    private static func isFinite(_ value: SIMD4<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
    }

    private static func prepareVariance(
        device: any MTLDevice, sampling: ShadowMapSampling, resolution: Int, slices: Int,
        radius: Int, angularRadius: Float
    ) throws -> VarianceShadowPass? {
        guard sampling == .variance else { return nil }
        guard (1...2048).contains(resolution), (0...16).contains(radius), angularRadius == 0 else {
            throw Failure.varianceSettings
        }
        return try VarianceShadowPass(device: device, resolution: resolution, slices: slices)
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidDirection
        case invalidRegion
        case unrepresentableRegion
        case invalidIntensity
        case invalidColor
        case invalidBias
        case invalidNormalBias
        case invalidShadowStrength
        case invalidAngularRadius
        case varianceSettings
        case transmissionLayers
        case invalidCascades
        case invalidSplits
        case invalidCamera
        case cascaded
        case notCascaded
        case cascadesNotFitted
        case library(String)
        case missingShader
        case pipeline(String)
        case resource(String)
        case wrongQueue
        case commandBuffer
        case encoder
        case momentEncoder

        public var errorDescription: String? {
            switch self {
            case .invalidDirection: "The shadow light direction must be finite and nonzero."
            case .invalidRegion: "The shadow region needs a finite center and a positive finite half extent."
            case .unrepresentableRegion: "The shadow region exceeds the Float projection range."
            case .invalidIntensity: "The shadow light intensity must be nonnegative and finite."
            case .invalidColor: "The directional light color needs linear RGB components between zero and one."
            case .invalidBias: "The shadow bias must be finite and between zero and one."
            case .invalidNormalBias: "The shadow normal bias must be finite."
            case .invalidShadowStrength: "The shadow strength must be between zero and one."
            case .invalidAngularRadius: "The shadow light's angular radius must be between zero and 0.25 radians."
            case .varianceSettings: "Variance shadows need a 1–2048 texel resolution, a 0–16 texel blur radius, and zero angular radius."
            case .transmissionLayers: "Use zero to four directional transmission layers per cascade and at most 128 MiB of transmission textures."
            case .invalidCascades:
                "Use 1 to \(DirectionalShadowMap.maximumCascadeCount) cascades, a split lambda from 0 to 1, a blend fraction from 0 to 0.5, a nonnegative caster margin, and a maximum distance beyond the camera's near plane."
            case .invalidSplits:
                "Use a representable split layout with increasing distances. Custom fractions need one fewer entry than the number of cascades and values strictly increasing between zero and one."
            case .invalidCamera:
                "Use a finite camera fit. An explicit view needs a rigid pose and a perspective or orthographic projection, with a finite shadow distance."
            case .cascaded: "Re-aim a cascaded shadow map with update(directionToLight:color:intensity:bias:lightAngularRadius:)."
            case .notCascaded: "Only a cascaded shadow map can be fitted to a camera."
            case .cascadesNotFitted: "Fit the cascaded shadow map to the camera before encoding a scene with it."
            case .library(let reason): "Could not load Frostfield's shader library: \(reason)"
            case .missingShader: "A required directional shadow shader is missing."
            case .pipeline(let reason): "Could not prepare the directional shadow pipeline: \(reason)"
            case .resource(let name): "Metal could not allocate \(name)."
            case .wrongQueue: "Use a command buffer from the shadow map's queue."
            case .commandBuffer: "Use an uncommitted command buffer with retained references."
            case .encoder: "Metal could not create the shadow render encoder."
            case .momentEncoder: "Metal could not create the shadow moment compute encoder."
            }
        }
    }
}
