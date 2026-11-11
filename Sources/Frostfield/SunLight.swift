import Foundation
import Metal
import simd

/// A directional sun with camera-fitted cascaded shadows. Pass its shadow map
/// to a scene encode that uses directional-shadow receiver materials. The
/// renderer fits the cascades to that view and reuses their textures.
public final class SunLight {
    /// Pass this map to `SceneGraph.encode(...shadowMap:)`. For an explicit
    /// projection, supply the matching `ShaderGraphCamera` as well.
    public let shadowMap: DirectionalShadowMap

    /// Unit world-space direction from the scene toward the sun.
    public var directionToLight: SIMD3<Float> {
        let light = shadowMap.lightParameters
        return SIMD3(light.x, light.y, light.z)
    }

    public var intensity: Float { shadowMap.lightParameters.w }

    /// Linear RGB tint of the sun's incident light, with components in 0...1.
    public var color: SIMD3<Float> { shadowMap.color }

    /// Defaults to an overhead sun and two 1024² practical cascades, with
    /// a 500-unit camera distance cap and 10% blend bands.
    public init(
        queue: any MTLCommandQueue, directionToLight: SIMD3<Float> = SIMD3(0, 1, 0),
        color: SIMD3<Float> = SIMD3(repeating: 1), intensity: Float = 1,
        resolution: Int = 1024,
        cascades: DirectionalShadowMap.Cascades = .init(count: 2, splitLambda: 0.5, maximumDistance: 500),
        bias: Float = 0.0015, normalBias: Float = 0, shadowStrength: Float = 1,
        lightAngularRadius: Float = 0, sampling: ShadowMapSampling = .filtered,
        varianceBlurRadius: Int = 3, updateMode: ShadowMapUpdateMode = .automatic, transmissionLayers: Int = 0
    ) throws {
        shadowMap = try DirectionalShadowMap(
            queue: queue, resolution: resolution, directionToLight: directionToLight, cascades: cascades,
            color: color, intensity: intensity, bias: bias, normalBias: normalBias, shadowStrength: shadowStrength,
            lightAngularRadius: lightAngularRadius, sampling: sampling,
            varianceBlurRadius: varianceBlurRadius, updateMode: updateMode,
            transmissionLayers: transmissionLayers
        )
        shadowMap.automaticallyFitsCamera = true
    }

    /// A sun aimed using the native solar position model and its world axes.
    public convenience init(
        queue: any MTLCommandQueue, solarPosition: SolarPosition, refracted: Bool = true,
        color: SIMD3<Float> = SIMD3(repeating: 1), intensity: Float = 1,
        resolution: Int = 1024,
        cascades: DirectionalShadowMap.Cascades = .init(count: 2, splitLambda: 0.5, maximumDistance: 500),
        bias: Float = 0.0015, normalBias: Float = 0, shadowStrength: Float = 1,
        lightAngularRadius: Float = 0, sampling: ShadowMapSampling = .filtered,
        varianceBlurRadius: Int = 3, updateMode: ShadowMapUpdateMode = .automatic, transmissionLayers: Int = 0
    ) throws {
        try self.init(
            queue: queue, directionToLight: solarPosition.direction(refracted: refracted),
            color: color, intensity: intensity, resolution: resolution, cascades: cascades, bias: bias, normalBias: normalBias,
            shadowStrength: shadowStrength, lightAngularRadius: lightAngularRadius,
            sampling: sampling, varianceBlurRadius: varianceBlurRadius, updateMode: updateMode,
            transmissionLayers: transmissionLayers
        )
    }

    /// Re-aims the same light and map. Omitted values keep the current
    /// color, intensity and filtering settings. A failed update preserves the light.
    public func update(
        directionToLight: SIMD3<Float>, color: SIMD3<Float>? = nil, intensity: Float? = nil,
        bias: Float? = nil, lightAngularRadius: Float? = nil
    ) throws {
        let currentRadius = min(DirectionalShadowMap.maximumLightAngularRadius, atan(shadowMap.shadowSettings.y))
        try shadowMap.update(
            directionToLight: directionToLight, color: color, intensity: intensity ?? self.intensity,
            bias: bias ?? shadowMap.shadowSettings.x, lightAngularRadius: lightAngularRadius ?? currentRadius
        )
    }

    public func update(
        solarPosition: SolarPosition, refracted: Bool = true, color: SIMD3<Float>? = nil, intensity: Float? = nil
    ) throws {
        try update(directionToLight: solarPosition.direction(refracted: refracted), color: color, intensity: intensity)
    }

    /// Changes only the sun's tint. The shadow map keeps its fit and cached depth.
    public func update(color: SIMD3<Float>) throws {
        try shadowMap.update(color: color)
    }
}
