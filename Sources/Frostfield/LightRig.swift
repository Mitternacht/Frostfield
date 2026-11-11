import Foundation
import simd

/// An editable set of studio lights that renders consistently in the raster
/// renderer and the path tracer.
///
/// Each light is placed by a position and an aim target, has a linear color, an
/// emission given as total power or as the kind's photometric intensity, and a
/// size where the kind has one. Frostfield shades photometric numbers directly as
/// radiometric scene units, the same convention `PunctualLight` and
/// `PathTracer.AreaLight` use, so one rig drives both renderers.
///
/// Three kinds of change stay distinct:
/// - Editing lights: rebuild the raster light nodes or the tracer's `Lighting`
///   value. No geometry or shader is prepared again; path tracing restarts
///   accumulation because history rejects a different rig.
/// - Updating geometry: move meshes with `PathTracingScene.updateTransforms(_:)`,
///   re-pose skinned or deformed meshes with `updateGeometry(for:in:replacements:)`,
///   and prepare a new `PathTracingScene` to add or remove meshes. Lights are unaffected.
/// - Display exposure: change `HDRDisplay.Settings.exposureStops`. Neither the rig
///   nor accumulated radiance changes.
public struct LightRig: Equatable, Sendable {
    public struct Light: Equatable, Sendable, Identifiable {
        public enum Kind: Equatable, Sendable {
            /// Parallel light along `target - position`. A positive angular diameter in
            /// radians softens shadows where a renderer supports it; the Sun is about 0.0093.
            case directional(angularDiameter: Float = 0)
            /// An omnidirectional light with an optional cutoff range in meters.
            case point(range: Float? = nil)
            /// A cone with full intensity inside `innerAngle` and a smooth falloff to `outerAngle`, in radians.
            case spot(innerAngle: Float, outerAngle: Float, range: Float? = nil)
            /// A one-sided Lambertian panel facing the target, in meters.
            case rectangle(width: Float, height: Float)
        }

        public enum Emission: Equatable, Sendable {
            /// Total emitted power. Point and spot intensity follow from the solid angle
            /// they illuminate, and a rectangle's luminance from its area, so resizing a
            /// cone or panel keeps its total output. Not available for directional lights.
            case power(Float)
            /// Illuminance in lux for directional lights, intensity in candela for point
            /// and spot lights, or luminance in nits for rectangles.
            case intensity(Float)
        }

        public var id: String
        public var kind: Kind
        public var position: SIMD3<Float>
        public var target: SIMD3<Float>
        /// Linear RGB with components in 0...1. See `Light.color(temperature:)`.
        public var color: SIMD3<Float>
        public var emission: Emission
        public var castsShadow: Bool
        public var isEnabled: Bool
        /// Shows a rectangle's emitting face to path-traced camera rays or a
        /// finite directional disk to camera and scattered rays.
        public var isVisibleToCamera: Bool
        /// An IES distribution for a point or spot light, aimed at the target with
        /// horizontal angle 0° along the light's local +X (see `transform()`). Power
        /// then divides by the profile's solid angle, times the cone falloff for a
        /// spot, so total flux stays as authored; `.intensity` sets the peak candela.
        /// `.intensity(profile.maximumIntensity)` reproduces the file's photometry.
        public var iesProfile: IESProfile?
        /// A projected image for a spot light with an outer cone up to 89°. Like a
        /// physical gobo it filters the beam, so `.power` is the flux before the cookie.
        public var cookie: LightCookie?

        public init(
            id: String, kind: Kind, position: SIMD3<Float>, target: SIMD3<Float>,
            color: SIMD3<Float> = SIMD3(repeating: 1), emission: Emission,
            castsShadow: Bool = true, isEnabled: Bool = true, isVisibleToCamera: Bool = false,
            iesProfile: IESProfile? = nil, cookie: LightCookie? = nil
        ) {
            self.id = id
            self.kind = kind
            self.position = position
            self.target = target
            self.color = color
            self.emission = emission
            self.castsShadow = castsShadow
            self.isEnabled = isEnabled
            self.isVisibleToCamera = isVisibleToCamera
            self.iesProfile = iesProfile
            self.cookie = cookie
        }

        /// The unit direction the light faces.
        public var direction: SIMD3<Float> { simd_normalize(target - position) }

        /// The kind's photometric value: lux, candela, or nits.
        public func intensity() throws -> Float {
            switch emission {
            case .intensity(let value):
                guard value.isFinite, value >= 0 else { throw Failure.emission(id) }
                return value
            case .power(let power):
                guard power.isFinite, power >= 0 else { throw Failure.emission(id) }
                let solidAngle: Double
                switch kind {
                case .directional:
                    throw Failure.directionalPower(id)
                case .point:
                    solidAngle = iesProfile.map { $0.weightedSolidAngle { _ in 1 } } ?? 4 * .pi
                case .spot(let inner, let outer, _):
                    solidAngle = profiledSpotSolidAngle(inner: inner, outer: outer)
                case .rectangle(let width, let height):
                    // A Lambertian panel emits π·A·L over its front hemisphere.
                    solidAngle = .pi * Double(width) * Double(height)
                }
                let value = Double(power) / solidAngle
                guard value.isFinite, value <= Double(Float.greatestFiniteMagnitude) else { throw Failure.emission(id) }
                return Float(value)
            }
        }

        /// The total power implied by the current emission, or nil for directional lights.
        public func power() throws -> Float? {
            if case .power(let value) = emission { return value }
            let intensity = try intensity()
            switch kind {
            case .directional: return nil
            case .point: return Float(Double(intensity) * (iesProfile.map { $0.weightedSolidAngle { _ in 1 } } ?? 4 * .pi))
            case .spot(let inner, let outer, _): return Float(Double(intensity) * profiledSpotSolidAngle(inner: inner, outer: outer))
            case .rectangle(let width, let height): return Float(Double(intensity) * .pi * Double(width) * Double(height))
            }
        }

        /// The effective solid angle of the raster and path-traced spot falloff:
        /// full intensity inside the inner cone, then a squared linear ramp in the
        /// cosine of the angle to zero at the outer cone. It is 2π[(1 − cos i) + (cos i − cos o)/3].
        public static func spotSolidAngle(inner: Float, outer: Float) -> Double {
            let cosInner = cos(Double(inner))
            let cosOuter = cos(Double(outer))
            return 2 * .pi * ((1 - cosInner) + (cosInner - cosOuter) / 3)
        }

        /// The spot's effective solid angle, weighted by its IES profile when it has one.
        private func profiledSpotSolidAngle(inner: Float, outer: Float) -> Double {
            guard let iesProfile else { return Self.spotSolidAngle(inner: inner, outer: outer) }
            // The shading cone: a squared, clamped ramp in cosine from max(cos outer, 0) to cos inner.
            let cosInner = Double(cos(inner))
            let cosOuter = Double(max(cos(outer), 0))
            return iesProfile.weightedSolidAngle { cosine in
                let cone = min(max((cosine - cosOuter) / (cosInner - cosOuter), 0), 1)
                return cone * cone
            }
        }

        /// A normalized linear-sRGB color for a blackbody at 1,000–40,000 kelvin,
        /// scaled so its largest component is one.
        public static func color(temperature kelvin: Float) throws -> SIMD3<Float> {
            guard kelvin.isFinite, (1_000...40_000).contains(kelvin) else { throw Failure.temperature }
            // Integrate Planck's law against the CIE 1931 2° observer (Wyman et al. 2013 fit).
            func gaussian(_ x: Double, _ mu: Double, _ low: Double, _ high: Double) -> Double {
                let t = (x - mu) / (x < mu ? low : high)
                return exp(-0.5 * t * t)
            }
            var xyz = SIMD3<Double>.zero
            let t = Double(kelvin)
            for nanometers in stride(from: 380.0, through: 780.0, by: 5.0) {
                let meters = nanometers * 1e-9
                let radiance = 1 / (pow(meters, 5) * (exp(1.4387769e-2 / (meters * t)) - 1))
                let x = 1.056 * gaussian(nanometers, 599.8, 37.9, 31.0) + 0.362 * gaussian(nanometers, 442.0, 16.0, 26.7)
                    - 0.065 * gaussian(nanometers, 501.1, 20.4, 26.2)
                let y = 0.821 * gaussian(nanometers, 568.8, 46.9, 40.5) + 0.286 * gaussian(nanometers, 530.9, 16.3, 31.1)
                let z = 1.217 * gaussian(nanometers, 437.0, 11.8, 36.0) + 0.681 * gaussian(nanometers, 459.0, 26.0, 13.8)
                xyz += radiance * SIMD3(x, y, z)
            }
            let rgb = SIMD3<Double>(
                3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
                -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
                0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z
            )
            let positive = simd_max(rgb, .zero)
            return SIMD3<Float>(positive / simd_reduce_max(positive))
        }

        /// A node transform at `position` whose local -Z faces the target, with local +Y kept near world up.
        public func transform() throws -> SceneTransform {
            let delta = target - position
            let length = simd_length(delta)
            guard position.x.isFinite, position.y.isFinite, position.z.isFinite, length.isFinite, length > 1e-6 else {
                throw Failure.aim(id)
            }
            let forward = delta / length
            let right = ResolvedPunctualLight.defaultRight(for: forward)
            let trueUp = simd_cross(right, forward)
            let rotation = simd_quatf(simd_float3x3(right, trueUp, -forward))
            return SceneTransform(translation: position, rotation: rotation)
        }

        func validate() throws {
            guard (0...1).contains(color.x), (0...1).contains(color.y), (0...1).contains(color.z) else {
                throw Failure.color(id)
            }
            switch kind {
            case .directional(let diameter):
                guard diameter.isFinite, (0...(.pi / 4)).contains(diameter) else { throw Failure.size(id) }
            case .point(let range):
                if let range { guard range.isFinite, range > 0 else { throw Failure.size(id) } }
            case .spot(let inner, let outer, let range):
                guard inner.isFinite, outer.isFinite, inner >= 0, inner < outer, outer <= .pi / 2 else { throw Failure.size(id) }
                if let range { guard range.isFinite, range > 0 else { throw Failure.size(id) } }
            case .rectangle(let width, let height):
                guard width.isFinite, height.isFinite, width > 0, height > 0 else { throw Failure.size(id) }
            }
            if iesProfile != nil {
                switch kind {
                case .point, .spot: break
                case .directional, .rectangle: throw Failure.lightTables(id)
                }
            }
            if cookie != nil {
                guard case .spot(_, let outer, _) = kind, outer <= PunctualLight.maximumCookieConeAngle else {
                    throw Failure.lightTables(id)
                }
            }
            _ = try intensity()
            _ = try transform()
            _ = try pathEmitter()
        }

        /// The light in the path tracer's form: an area emitter for a rectangle,
        /// otherwise an analytic light. Throws `unsupported` for a light the tracer
        /// rejects, such as a rectangle whose luminance times its brightest color
        /// component exceeds the tracer's 65,504 radiance limit.
        fileprivate func pathEmitter() throws -> PathEmitter {
            let intensity = try intensity()
            if case .rectangle = kind, simd_reduce_max(color) * intensity > LightRig.maximumRadiance {
                let peak = Int((simd_reduce_max(color) * intensity).rounded())
                throw Failure.unsupported(
                    id, "its radiance peaks at \(peak), above the 65,504 limit; lower its power or enlarge the panel"
                )
            }
            do {
                switch kind {
                case .rectangle(let width, let height):
                    // Width follows local X and the front faces local -Z, as in the raster panel.
                    let rotation = try transform().rotation
                    let right = rotation.act(SIMD3<Float>(1, 0, 0))
                    let up = rotation.act(SIMD3<Float>(0, 1, 0))
                    return .area(try PathTracer.AreaLight(
                        center: position, halfU: right * (width / 2), halfV: -up * (height / 2),
                        radiance: color * intensity, cameraVisible: isVisibleToCamera
                    ))
                case .directional(let diameter):
                    return .analytic(try PathTracer.AnalyticLight(
                        PunctualLight(kind: .directional, color: color, intensity: intensity),
                        direction: direction, angularDiameter: diameter, diskVisible: isVisibleToCamera && diameter > 0
                    ))
                case .point(let range):
                    // The aim orients an IES profile; the light's local +X follows transform().
                    return .analytic(try PathTracer.AnalyticLight(
                        PunctualLight(kind: .point(range: range), color: color, intensity: intensity, iesProfile: iesProfile),
                        position: position, direction: direction
                    ))
                case .spot(let inner, let outer, let range):
                    return .analytic(try PathTracer.AnalyticLight(
                        PunctualLight(
                            kind: .spot(range: range, innerConeAngle: inner, outerConeAngle: outer),
                            color: color, intensity: intensity, iesProfile: iesProfile, cookie: cookie
                        ),
                        position: position, direction: direction
                    ))
                }
            } catch is PathTracer.Failure {
                throw Failure.unsupported(id, "its size or position is outside the path tracer's range")
            }
        }
    }

    fileprivate enum PathEmitter {
        case area(PathTracer.AreaLight)
        case analytic(PathTracer.AnalyticLight)
    }

    /// The path tracer's per-channel limit for area light radiance, the largest half float.
    static let maximumRadiance: Float = 65_504

    public var lights: [Light]

    /// Validates every light and requires unique identifiers. Each light must also be
    /// one the path tracer accepts: a rectangle's luminance times its brightest color
    /// component may not exceed 65,504, the tracer's half-float radiance limit.
    public init(lights: [Light]) throws {
        var seen = Set<String>()
        for light in lights {
            guard seen.insert(light.id).inserted else { throw Failure.duplicateID(light.id) }
            try light.validate()
        }
        self.lights = lights
    }

    public subscript(id: String) -> Light? {
        get { lights.first { $0.id == id } }
        set {
            guard let index = lights.firstIndex(where: { $0.id == id }) else { return }
            if let newValue { lights[index] = newValue } else { lights.remove(at: index) }
        }
    }

    // MARK: - Raster

    /// Scene nodes that carry the enabled point, spot, rectangle, and directional lights
    /// for `SceneGraph`. Shadow-casting point, spot, and rectangle lights use `shadow`
    /// and need `PunctualShadowMaps` when encoding; directional shadows use
    /// `DirectionalShadowMap`, configured from `directionalShadow`.
    ///
    /// Set `omittingDirectionalShadow` when `directionalShadow` drives a
    /// `DirectionalShadowMap`, whose light already shades receiving materials.
    public func sceneNodes(shadow: PunctualLight.Shadow? = nil, omittingDirectionalShadow: Bool = false) throws -> [SceneNode] {
        let defaultShadow = try shadow ?? PunctualLight.Shadow()
        let omitted = omittingDirectionalShadow ? directionalShadowID : nil
        return try lights.filter { $0.isEnabled && $0.id != omitted }.compactMap { light -> SceneNode? in
            try light.validate()
            let kind: PunctualLight.Kind
            switch light.kind {
            case .directional:
                kind = .directional
            case .point(let range):
                kind = .point(range: range)
            case .spot(let inner, let outer, let range):
                kind = .spot(range: range, innerConeAngle: inner, outerConeAngle: outer)
            case .rectangle(let width, let height):
                kind = .rectangle(width: width, height: height)
            }
            var castsShadow = light.castsShadow
            if case .directional = light.kind { castsShadow = false }
            if case .spot(_, let outer, _) = light.kind, outer > PunctualLight.Shadow.maximumSpotConeAngle {
                castsShadow = false
            }
            let punctual = try PunctualLight(
                kind: kind, color: light.color, intensity: try light.intensity(), shadow: castsShadow ? defaultShadow : nil,
                iesProfile: light.iesProfile, cookie: light.cookie
            )
            return SceneNode(transform: try light.transform(), light: punctual)
        }
    }

    /// The first enabled, shadow-casting directional light as a direction toward the
    /// light, its illuminance times its color's luminance-preserving average, and its
    /// angular radius, for `DirectionalShadowMap`, which shades in white light. The
    /// radius is limited to `DirectionalShadowMap.maximumLightAngularRadius`, so a sun
    /// wider than 0.5 radians casts the softest raster shadow the map supports while
    /// the path tracer uses its full diameter.
    public var directionalShadow: (directionToLight: SIMD3<Float>, intensity: Float, angularRadius: Float)? {
        guard let id = directionalShadowID, let light = self[id], let intensity = try? light.intensity(),
              case .directional(let diameter) = light.kind else { return nil }
        let tint = simd_dot(light.color, SIMD3<Float>(0.2126, 0.7152, 0.0722))
        return (-light.direction, intensity * tint, min(diameter / 2, DirectionalShadowMap.maximumLightAngularRadius))
    }

    private var directionalShadowID: String? {
        lights.first { light in
            guard light.isEnabled, light.castsShadow, case .directional = light.kind else { return false }
            return (try? light.intensity()) != nil
        }?.id
    }

    // MARK: - Path tracing

    /// The same enabled lights for the path tracer. Rectangles become area emitters
    /// sampled by area; point, spot and directional lights become analytic lights.
    /// Path-traced lights always cast shadows, so `castsShadow` affects raster only.
    public func pathLighting(
        sky: PathTracer.Environment = .constant(.zero), backgroundVisible: Bool = true
    ) throws -> PathTracer.Lighting {
        var areas: [PathTracer.AreaLight] = []
        var analytic: [PathTracer.AnalyticLight] = []
        for light in lights where light.isEnabled {
            try light.validate()
            switch try light.pathEmitter() {
            case .area(let area): areas.append(area)
            case .analytic(let emitter): analytic.append(emitter)
            }
        }
        return try PathTracer.Lighting(lights: areas, analyticLights: analytic, sky: sky, backgroundVisible: backgroundVisible)
    }

    public enum Failure: LocalizedError, Equatable {
        case duplicateID(String)
        case color(String)
        case size(String)
        case emission(String)
        case directionalPower(String)
        case aim(String)
        case temperature
        case unsupported(String, String)
        case lightTables(String)

        public var errorDescription: String? {
            switch self {
            case .duplicateID(let id): "Light rig identifier \(id) is used more than once."
            case .color(let id): "Light \(id) needs linear color components in zero to one."
            case .size(let id):
                "Light \(id) needs a positive finite size or range, spot cones with 0 ≤ inner < outer ≤ π/2, and a sun diameter up to π/4."
            case .emission(let id): "Light \(id) needs finite, nonnegative power or intensity."
            case .directionalPower(let id): "Directional light \(id) takes illuminance in lux, not power."
            case .aim(let id): "Light \(id) needs a finite position and a target at a different point."
            case .temperature: "Color temperature must be between 1,000 and 40,000 kelvin."
            case .unsupported(let id, let detail): "Light \(id) cannot be path traced: \(detail)."
            case .lightTables(let id):
                "Light \(id): only point and spot lights take an IES profile, and only spots with an outer cone up to 89° take a cookie."
            }
        }
    }
}
