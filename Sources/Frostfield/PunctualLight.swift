import Foundation
import simd

/// The depth comparison used by a raster shadow map.
public enum ShadowMapSampling: String, Codable, Sendable {
    /// Use the map's configured percentage-closer or contact-hardening filter.
    case filtered
    /// Compare the receiver with a single nearest depth texel.
    case basic
    /// Estimate visibility from filtered depth moments on directional and spot maps.
    case variance
}

/// A light attached to a scene node. Directional, spot, and rectangle lights point along local -Z.
public struct PunctualLight: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case directional
        case point(range: Float? = nil)
        case spot(range: Float? = nil, innerConeAngle: Float = 0, outerConeAngle: Float = .pi / 4)
        /// A one-sided Lambertian rectangle centered on the node in its local XY plane.
        /// `width` spans local X and `height` spans local Y, in meters. It emits only
        /// toward local -Z, so surfaces behind its plane receive no light. Node scale
        /// does not resize the rectangle, and a skewed node frame is orthogonalized.
        case rectangle(width: Float, height: Float)
    }

    /// Depth-map shadow settings for a point, spot, or rectangle light.
    ///
    /// Shadows render only when the scene graph is encoded with `PunctualShadowMaps`
    /// that have a free slot for the light. A rectangle light casts from its center
    /// through a point-light slot.
    public struct Shadow: Equatable, Sendable {
        /// Spot shadows support outer cone angles up to 75 degrees, in radians.
        public static let maximumSpotConeAngle: Float = 5 * .pi / 12
        /// The largest normal bias, in shadow-map texels.
        public static let maximumNormalBias: Float = 16
        /// The largest PCF filter radius, in shadow-map texels.
        public static let maximumFilterRadius: Float = 8

        /// A constant receiver offset toward the light along the shadow view's axis, in meters.
        public let depthBias: Float
        /// The receiver offset along its geometric normal, in shadow-map texels at the
        /// receiver's distance, scaled by the sine of the light's incidence angle.
        /// Filtered maps add the filter radius plus one texel to account for the filter footprint.
        public let normalBias: Float
        /// The percentage-closer filter radius in shadow-map texels. Zero uses one
        /// bilinear depth comparison. Larger radii use 16 rotated comparisons.
        /// Basic and variance sampling ignore this radius.
        public let filterRadius: Float
        /// The distance from the light to the shadow projection's near plane, in meters.
        /// Geometry closer to the light than this distance casts no shadow.
        public let nearPlane: Float
        public let sampling: ShadowMapSampling
        /// Box filter radius in texels for spot variance sampling.
        public let varianceBlurRadius: Int
        /// How strongly the raster shadow map attenuates this light, from zero to one.
        /// The light's emitted intensity and shadow projection are independent.
        public let shadowStrength: Float

        /// Depth bias must be finite and nonnegative, normal bias in 0...16, filter
        /// radius in 0...8, variance blur radius in 0...16, shadow strength in 0...1,
        /// and the near plane positive and finite.
        public init(
            depthBias: Float = 0, normalBias: Float = 1, filterRadius: Float = 1.5, nearPlane: Float = 0.05,
            sampling: ShadowMapSampling = .filtered, varianceBlurRadius: Int = 3, shadowStrength: Float = 1
        ) throws {
            guard depthBias.isFinite, depthBias >= 0,
                  normalBias.isFinite, (0...Self.maximumNormalBias).contains(normalBias),
                  filterRadius.isFinite, (0...Self.maximumFilterRadius).contains(filterRadius),
                  (0...16).contains(varianceBlurRadius), (0...1).contains(shadowStrength),
                  nearPlane.isFinite, nearPlane > 0 else {
                throw Failure.shadow
            }
            self.depthBias = depthBias
            self.normalBias = normalBias
            self.filterRadius = filterRadius
            self.nearPlane = nearPlane
            self.sampling = sampling
            self.varianceBlurRadius = varianceBlurRadius
            self.shadowStrength = shadowStrength
        }
    }

    public let kind: Kind
    /// A linear RGB multiplier, with components between zero and one.
    public let color: SIMD3<Float>
    /// Illuminance in lux for directional lights, luminous intensity in candela for
    /// point and spot lights, or luminance in nits (cd/m²) for rectangle lights.
    /// Use meters for scene positions, range, and rectangle size. Shading uses these
    /// values directly as radiometric units, so a rectangle's intensity is the same
    /// radiance that `PathTracer.AreaLight` uses.
    public let intensity: Float
    /// Optional depth-map shadows. Directional lights use `DirectionalShadowMap` instead.
    public let shadow: Shadow?
    /// An IES candela distribution for a point or spot light. With a profile,
    /// `intensity` is the candela in the profile's brightest direction, the
    /// distribution follows the node's local frame (−Z is the photometric axis and
    /// +X horizontal angle 0°), and a spot's cone still applies. Both renderers
    /// sample the same table; see `IESProfile`.
    public let iesProfile: IESProfile?
    /// A color image a spot light projects across the square around its outer cone;
    /// see `LightCookie`. Requires an outer cone angle up to 89°.
    public let cookie: LightCookie?

    public init(
        kind: Kind, color: SIMD3<Float> = SIMD3(repeating: 1), intensity: Float = 1, shadow: Shadow? = nil,
        iesProfile: IESProfile? = nil, cookie: LightCookie? = nil
    ) throws {
        guard (0...1).contains(color.x), (0...1).contains(color.y), (0...1).contains(color.z) else {
            throw Failure.color
        }
        guard intensity.isFinite, intensity >= 0 else { throw Failure.intensity }
        switch kind {
        case .directional:
            guard shadow == nil else { throw Failure.shadowKind }
        case .point(let range):
            try Self.validateRange(range, shadow: shadow)
        case .spot(let range, let inner, let outer):
            try Self.validateRange(range, shadow: shadow)
            guard inner >= 0, inner < outer, outer <= .pi / 2 else { throw Failure.cone }
            if shadow != nil, outer > Shadow.maximumSpotConeAngle { throw Failure.shadowCone }
        case .rectangle(let width, let height):
            guard width.isFinite, height.isFinite, width > 0, height > 0 else { throw Failure.rectangle }
        }
        if iesProfile != nil {
            switch kind {
            case .point, .spot: break
            case .directional, .rectangle: throw Failure.profileKind
            }
        }
        if shadow?.sampling == .variance {
            guard case .spot = kind else { throw Failure.shadowSampling }
        }
        if cookie != nil {
            guard case .spot(_, _, let outer) = kind, outer <= Self.maximumCookieConeAngle else { throw Failure.cookieKind }
        }
        self.kind = kind
        self.color = color
        self.intensity = intensity
        self.shadow = shadow
        self.iesProfile = iesProfile
        self.cookie = cookie
    }

    /// The widest outer cone a cookie may cover, 89° in radians.
    public static let maximumCookieConeAngle: Float = 89 * .pi / 180

    /// A rectangle light from its total emitted power over the front hemisphere.
    /// A Lambertian emitter has luminance `power / (π × width × height)`, the same
    /// conversion as `PathTracer.AreaLight(center:aim:size:radiantPower:)`.
    public init(
        rectangleWidth width: Float, height: Float, power: Float,
        color: SIMD3<Float> = SIMD3(repeating: 1), shadow: Shadow? = nil
    ) throws {
        guard width.isFinite, height.isFinite, width > 0, height > 0 else { throw Failure.rectangle }
        guard power.isFinite, power >= 0 else { throw Failure.power }
        let luminance = Double(power) / (.pi * Double(width) * Double(height))
        guard luminance <= Double(Float.greatestFiniteMagnitude) else { throw Failure.power }
        try self.init(kind: .rectangle(width: width, height: height), color: color, intensity: Float(luminance), shadow: shadow)
    }

    private static func validateRange(_ range: Float?, shadow: Shadow?) throws {
        if let range, !range.isFinite || range <= 0 { throw Failure.range }
        if let range, let shadow, shadow.nearPlane >= range { throw Failure.shadowRange }
    }

    public enum Failure: LocalizedError, Equatable {
        case color
        case intensity
        case range
        case cone
        case rectangle
        case power
        case shadow
        case shadowKind
        case shadowSampling
        case shadowCone
        case shadowRange
        case profileKind
        case cookieKind

        public var errorDescription: String? {
            switch self {
            case .color: "Punctual light color components must be in zero to one."
            case .intensity: "Punctual light intensity must be finite and nonnegative."
            case .range: "Punctual light range must be positive and finite when supplied."
            case .cone: "A spot cone needs 0 ≤ inner < outer ≤ π/2 radians."
            case .rectangle: "A rectangle light needs a finite, positive width and height."
            case .power: "Rectangle light power must be finite and nonnegative, with a representable luminance."
            case .shadow:
                "Use a finite, nonnegative depth bias, a normal bias of 0–16 texels, a filter radius of 0–8 texels, a variance radius of 0–16 texels, a shadow strength from zero to one, and a positive, finite near plane."
            case .shadowKind: "Directional lights cast shadows through DirectionalShadowMap, not a punctual shadow."
            case .shadowSampling: "Only spot lights support variance sampling in punctual shadow maps."
            case .shadowCone: "Spot shadows support outer cone angles up to 75°. Use a point light for wider shadowed cones."
            case .shadowRange: "A shadow's near plane must be closer to the light than its range."
            case .profileKind: "Only point and spot lights take an IES profile."
            case .cookieKind: "Only spot lights with an outer cone of at most 89° take a cookie."
            }
        }
    }
}

/// A punctual light resolved through its scene node's world transform.
public struct ResolvedPunctualLight: Equatable, Sendable {
    public let light: PunctualLight
    public let position: SIMD3<Float>
    /// The unit vector pointing away from the light along its transformed local -Z.
    public let direction: SIMD3<Float>
    /// Unit world axes for the light's local X and Y, orthogonal to `direction`.
    let right: SIMD3<Float>
    let up: SIMD3<Float>

    init(light: PunctualLight, world: simd_float4x4) {
        self.light = light
        let z = world.columns.2
        let origin = world.columns.3
        position = SIMD3(origin.x, origin.y, origin.z)
        let scale = max(abs(z.x), max(abs(z.y), abs(z.z)))
        let direction = simd_normalize(-SIMD3(z.x, z.y, z.z) / scale)
        self.direction = direction
        // Remove any component along the emission direction so a sheared frame still
        // yields a rectangle perpendicular to the direction it emits.
        let x = world.columns.0
        let xScale = max(abs(x.x), max(abs(x.y), abs(x.z)))
        let localX = SIMD3(x.x, x.y, x.z) / xScale
        let tangent = localX - direction * simd_dot(localX, direction)
        let right = simd_length_squared(tangent) > 1e-12 ? simd_normalize(tangent) : Self.perpendicular(to: direction)
        self.right = right
        up = simd_cross(right, direction)
    }

    /// Frame for a light placed with an aim but without an authored roll.
    static func defaultRight(for direction: SIMD3<Float>) -> SIMD3<Float> {
        let reference: SIMD3<Float> = abs(direction.y) < 0.999 ? SIMD3(0, 1, 0) : SIMD3(0, 0, 1)
        return simd_normalize(simd_cross(direction, reference))
    }

    private static func perpendicular(to direction: SIMD3<Float>) -> SIMD3<Float> {
        let reference = abs(direction.y) < 0.99 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
        return simd_normalize(simd_cross(reference, direction))
    }
}
