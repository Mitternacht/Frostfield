import Foundation
import Metal
import simd

/// Raster water materials that sample caller-owned planar captures. Use
/// `PlanarReflection` and, for flowing water, `PlanarRefraction` to encode
/// the captures before drawing the water in the same viewport space.
public enum PlanarWater {
    /// Camera fog applied after water lighting. Distance is view-axis depth in world units.
    public struct Fog: Sendable {
        public enum Falloff: Sendable {
            /// Smoothstep between the near and far view distances.
            case linear(near: Float, far: Float)
            /// Visibility falls as exp(-(density * viewDepth)²).
            case exponentialSquared(density: Float)
        }

        public let color: SIMD3<Float>
        public let falloff: Falloff

        public init(color: SIMD3<Float>, falloff: Falloff) throws {
            let validFalloff = switch falloff {
            case .linear(let near, let far): near.isFinite && far.isFinite && near >= 0 && near < far
            case .exponentialSquared(let density): density.isFinite && density >= 0
            }
            guard validFalloff, PlanarWater.validColor(color) else { throw Failure.fog }
            self.color = color
            self.falloff = falloff
        }
    }

    /// Inputs for a custom flowing-water graph, using the same typed builder.
    public struct FlowInputs {
        public let reflection: ShaderTexture
        public let refraction: ShaderTexture
        public let firstNormal: ShaderTexture
        public let secondNormal: ShaderTexture
        /// The optional decoded flow map or the constant flow direction.
        public let flowDirection: ShaderValue<SIMD2<Float>>
        /// Phase in 0...0.15 before the half-cycle stagger.
        public let phase: ShaderValue<Float>
        public let normal: ShaderValue<SIMD3<Float>>
        public let fresnel: ShaderValue<Float>
        /// Distorted viewport UV before the reflection sampler's X correction.
        public let coordinates: ShaderValue<SIMD2<Float>>
        /// Straight, tinted water RGB and alpha before fog.
        public let color: ShaderValue<SIMD3<Float>>
        public let opacity: ShaderValue<Float>
    }

    public struct FlowOptions: Sendable {
        /// Linear RGB multiplier applied to both captures.
        public var tint: SIMD3<Float> = SIMD3(repeating: 1)
        /// Apparent direction of the normal pattern in top-left UV0.
        /// An optional flow texture supplies its own direction per texel.
        public var direction: SIMD2<Float> = SIMD2(1, 0)
        /// UV phase units per second. The blend cycle spans 0.15 phase units.
        public var speed: Float = 0.03
        /// Normal-map repeats per unit of mesh UV0.
        public var scale: Float = 1
        /// Reflectance at normal incidence. Grazing reflectance tends to one.
        public var reflectivity: Float = 0.02
        /// Viewport UV gain for normal tilt. Planar refraction also multiplies
        /// it by forward projective depth.
        public var distortion: Float = 0.05
        public var fog: Fog?

        public init() {}
    }

    public struct ReflectionOptions: Sendable {
        /// Linear RGB scatter color. The default is sRGB 0x7f7f7f converted to linear.
        public var tint: SIMD3<Float> = SIMD3(repeating: 0.21223076)
        public var opacity: Float = 1
        /// Scale of the world XZ coordinates used for the four wave samples.
        public var scale: Float = 1
        /// Gain of distance-dependent distortion in top-left viewport UV.
        public var distortion: Float = 20
        /// Unit or non-unit world-space vector from the surface to the sun.
        public var sunDirection: SIMD3<Float> = SIMD3(1, 1, 0)
        /// Linear RGB color and intensity of the authored sun term.
        public var sunColor: SIMD3<Float> = SIMD3(repeating: 1)
        /// Shadows attenuate scatter and diffuse sunlight. Supply a matching
        /// directional shadow map to the scene encode when enabled.
        public var receivesSunShadows = false
        public var fog: Fog?

        public init() {}
    }

    /// Water2-style dual captures and smoothly crossfaded normal maps. The
    /// surface uses the mesh UV0 tangent frame. The normals store signed XY in
    /// encoded RG and positive Z in B. An optional RG flow map encodes each
    /// direction component from -1...1 into 0...1. Bind linear data textures.
    /// Advance `material.surfaceUniforms` to animate without rebuilding.
    /// A custom `shade` graph can replace color, opacity, mask, alpha test and
    /// object-space vertex offset, using these same textures and live flow inputs.
    /// Declare any extra textures in that graph and bind them by name here.
    public static func flowing(
        reflection: any MTLTexture, refraction: any MTLTexture,
        firstNormal: any MTLTexture, secondNormal: any MTLTexture, flowMap: (any MTLTexture)? = nil,
        options: FlowOptions = .init(), additionalTextures: [String: any MTLTexture] = [:],
        shade: ((ShaderGraphBuilder, FlowInputs) throws -> SurfaceShaderGraph.Result)? = nil
    ) throws -> PhysicallyBasedMaterial {
        try makeFlowing(
            reflection: reflection, refraction: refraction, firstNormal: firstNormal, secondNormal: secondNormal,
            flowMap: flowMap, options: options, additionalTextures: additionalTextures, shade: shade
        )
    }

    /// Water2Mesh-style reflection and depth-safe refraction of the opaque scene.
    /// Bind color and depth on `SceneGraph.viewportSnapshot` from the main view.
    /// The scene can capture its opaque draws automatically with `.opaquePass`.
    /// Encode the separate planar reflection before drawing the water.
    public static func flowingViewport(
        reflection: any MTLTexture, firstNormal: any MTLTexture, secondNormal: any MTLTexture,
        flowMap: (any MTLTexture)? = nil, options: FlowOptions = .init()
    ) throws -> PhysicallyBasedMaterial {
        try makeFlowing(
            reflection: reflection, refraction: nil, firstNormal: firstNormal, secondNormal: secondNormal,
            flowMap: flowMap, options: options, additionalTextures: [:], shade: nil
        )
    }

    private static func makeFlowing(
        reflection: any MTLTexture, refraction: (any MTLTexture)?,
        firstNormal: any MTLTexture, secondNormal: any MTLTexture, flowMap: (any MTLTexture)?,
        options: FlowOptions, additionalTextures: [String: any MTLTexture],
        shade: ((ShaderGraphBuilder, FlowInputs) throws -> SurfaceShaderGraph.Result)?
    ) throws -> PhysicallyBasedMaterial {
        let cycle: Float = 0.15
        guard validColor(options.tint), options.direction.x.isFinite, options.direction.y.isFinite,
              options.speed.isFinite, options.speed >= 0, (options.speed / cycle).isFinite,
              options.scale.isFinite, options.scale > 0, (0...1).contains(options.reflectivity),
              options.distortion.isFinite, options.distortion >= 0
        else { throw Failure.flowOptions }
        let captures = [reflection, refraction].compactMap { $0 }
        let reservedNames = ["WaterReflection", "WaterRefraction", "WaterNormal0", "WaterNormal1", "WaterFlow"]
        if let collision = reservedNames.first(where: { additionalTextures[$0] != nil }) {
            throw Failure.reservedTexture(collision)
        }
        guard refraction.map({ !TextureTransfer.sharesStorage(reflection, $0) }) ?? true,
              captures.allSatisfy({ $0.pixelFormat == .rgba16Float }),
              (captures + [firstNormal, secondNormal] + [flowMap].compactMap { $0 } + additionalTextures.values)
                .allSatisfy({ $0.device === reflection.device })
        else { throw Failure.textures }

        let graph = try SurfaceShaderGraph { surface in
            let first = surface.texture("WaterNormal0", content: .data)
            let second = surface.texture("WaterNormal1", content: .data)
            let reflected = surface.texture("WaterReflection", addressing: .clampToEdge)
            let refracted = refraction.map { _ in surface.texture("WaterRefraction", addressing: .clampToEdge) }
            let flow: ShaderValue<SIMD2<Float>>
            if flowMap != nil {
                flow = surface.texture("WaterFlow", content: .data).sample(surface.uv0).xy * 2 - 1
            } else {
                flow = surface.constant(options.direction)
            }

            let half = cycle * 0.5
            let phase = (surface.time * (options.speed / cycle)).fract() * cycle
            let staggered = (phase + half).mod(cycle)
            let uv = surface.uv0 * options.scale
            let color0 = first.sample(uv - flow * phase)
            let color1 = second.sample(uv - flow * staggered)
            let encoded = color0.mix(color1, by: (phase - half).abs() / half)
            let tilt = surface.vector(encoded.r * 2 - 1, encoded.g * 2 - 1, encoded.b).normalized()
            let normal = (surface.tangent * tilt.x + surface.bitangent * tilt.y + surface.normal * tilt.z).normalized()
            let fresnel = schlick(surface.viewDirection.dot(normal), reflectivity: options.reflectivity)
            let offset = if refracted != nil {
                tilt.xy * forwardProjectiveDepth(in: surface) * options.distortion
            } else {
                tilt.xy * options.distortion
            }
            let sampleUV = surface.screenUV + offset
            let refractionSample: ShaderValue<SIMD4<Float>>
            if let refracted {
                refractionSample = refracted.samplePlanarRefraction(at: sampleUV)
            } else {
                let original = surface.attachmentUV(at: surface.screenCoordinate)
                let viewportScale = surface.attachmentUV(at: surface.viewportSize)
                let distorted = original + offset * viewportScale
                refractionSample = surface.sceneColor(at: surface.viewportSafeUV(at: distorted, fallback: original))
            }
            let reflection = reflected.samplePlanarReflection(at: sampleUV)
            let mixed = refractionSample.mix(reflection, by: fresnel)
            let color = mixed.rgb / mixed.a * options.tint
            var result = SurfaceShaderGraph.Result(baseColor: color, opacity: mixed.a)
            if let shade, let refracted {
                result = try shade(surface, FlowInputs(
                    reflection: reflected, refraction: refracted, firstNormal: first, secondNormal: second,
                    flowDirection: flow, phase: phase, normal: normal, fresnel: fresnel, coordinates: sampleUV,
                    color: color, opacity: mixed.a
                ))
            }
            let radiance = result.baseColor ?? surface.baseColor
            result.baseColor = nil
            result.emission = options.fog.map { applyFog(radiance, $0, in: surface) } ?? radiance
            return result
        }
        var textures = additionalTextures
        textures["WaterReflection"] = reflection
        textures["WaterRefraction"] = refraction
        textures["WaterNormal0"] = firstNormal
        textures["WaterNormal1"] = secondNormal
        textures["WaterFlow"] = flowMap
        return try PhysicallyBasedMaterial(
            alphaMode: .blend, surfaceGraph: graph, surfaceTextures: textures, shadingModel: .unlit,
            unlitUsesGraphEmission: true
        )
    }

    /// Water/WaterMesh-style reflective water with four animated XZ normal
    /// samples, Fresnel reflection, scatter, and an authored sun highlight.
    /// Use a horizontal Y-up surface and a linear RGB normal texture.
    /// Advance `material.surfaceUniforms` to animate without rebuilding.
    public static func reflective(
        reflection: any MTLTexture, normalMap: any MTLTexture, options: ReflectionOptions = .init()
    ) throws -> PhysicallyBasedMaterial {
        let largest = simd_reduce_max(abs(options.sunDirection))
        guard validColor(options.tint), validColor(options.sunColor), (0...1).contains(options.opacity),
              options.scale.isFinite, options.scale > 0, options.distortion.isFinite, options.distortion >= 0,
              [options.sunDirection.x, options.sunDirection.y, options.sunDirection.z].allSatisfy(\.isFinite),
              largest > 0 else { throw Failure.reflectionOptions }
        guard reflection.pixelFormat == .rgba16Float, normalMap.device === reflection.device else { throw Failure.textures }

        let sun = simd_normalize(options.sunDirection / largest)
        let graph = try SurfaceShaderGraph { surface in
            let map = surface.texture("WaterNormal", content: .data)
            let reflected = surface.texture("WaterReflection", addressing: .clampToEdge)
            let uv = surface.worldPosition.xz * options.scale
            let time = surface.time
            let first = uv / 103 + surface.vector(time / 17, time / 29)
            let second = uv / 107 + surface.vector(time / 19, -time / 31)
            let third = uv * SIMD2<Float>(1 / 8907, 1 / 9803) + surface.vector(time / 101, time / 97)
            let fourth = uv * SIMD2<Float>(1 / 1091, 1 / 1027) + surface.vector(-time / 109, time / 113)
            let wave = (map.sample(first).rgb + map.sample(second).rgb + map.sample(third).rgb + map.sample(fourth).rgb)
                * 0.5 - 1
            let normal = surface.vector(wave.x * 1.5, wave.z, wave.y * 1.5).normalized()
            let view = surface.viewDirection
            let sunDirection = surface.constant(sun)
            let highlight = view.dot((-sunDirection).reflect(normal)).saturated().pow(100) * 2
            let diffuse = sunDirection.dot(normal).saturated() * 0.15
            let facing = view.dot(normal).saturated()
            let distance = surface.cameraPosition.distance(to: surface.worldPosition).max(1e-4)
            let distortion = normal.xz * (0.001 + surface.constant(Float(1)) / distance) * options.distortion
            let reflectedColor = reflected.samplePlanarReflection(at: surface.screenUV + distortion).rgb
            let scattered = surface.constant(options.tint) * facing + surface.constant(options.sunColor * options.sunColor) * diffuse
            let visibility = options.receivesSunShadows ? surface.directionalShadowVisibility : surface.constant(Float(1))
            let litReflection = reflectedColor + surface.constant(options.sunColor) * highlight
            let color = (scattered * visibility).mix(litReflection, by: schlick(facing, reflectivity: 0.02))
            return .init(emission: options.fog.map { applyFog(color, $0, in: surface) } ?? color)
        }
        return try PhysicallyBasedMaterial(
            directionToLight: sun, receivesShadows: options.receivesSunShadows,
            baseColorFactor: SIMD4(1, 1, 1, options.opacity), alphaMode: options.opacity < 1 ? .blend : .opaque,
            surfaceGraph: graph, surfaceTextures: ["WaterReflection": reflection, "WaterNormal": normalMap],
            shadingModel: .unlit, unlitUsesGraphEmission: true
        )
    }

    private static func forwardProjectiveDepth(in surface: ShaderGraphBuilder) -> ShaderValue<Float> {
        let projection = surface.projectionMatrix
        let world = surface.vector(surface.worldPosition, surface.constant(Float(1)))
        let clip = projection * (surface.viewMatrix * world)
        let depth = clip.z / clip.w
        let z = projection.column(2)
        let w = projection.column(3)
        let slope = z.z * w.w - w.z * z.w
        return surface.select(slope.greaterThan(surface.constant(Float(0))), 1 - depth, depth).saturated()
    }

    private static func applyFog(
        _ color: ShaderValue<SIMD3<Float>>, _ fog: Fog, in surface: ShaderGraphBuilder
    ) -> ShaderValue<SIMD3<Float>> {
        let depth = (-surface.viewMatrix.transformPoint(surface.worldPosition).z).max(0)
        let factor: ShaderValue<Float>
        switch fog.falloff {
        case .linear(let near, let far):
            factor = depth.smoothstep(near, far)
        case .exponentialSquared(let density):
            let opticalDepth = depth * density
            factor = 1 - (-(opticalDepth * opticalDepth)).exp()
        }
        return color.mix(surface.constant(fog.color), by: factor)
    }

    private static func schlick(_ facing: ShaderValue<Float>, reflectivity: Float) -> ShaderValue<Float> {
        reflectivity + (1 - reflectivity) * (1 - facing.saturated()).pow(5)
    }

    private static func validColor(_ color: SIMD3<Float>) -> Bool {
        [color.x, color.y, color.z].allSatisfy { (0...65_504).contains($0) }
    }

    public enum Failure: LocalizedError, Equatable {
        case flowOptions, reflectionOptions, textures, fog
        case reservedTexture(String)

        public var errorDescription: String? {
            switch self {
            case .flowOptions: "Use finite flowing-water controls with positive scale and reflectivity in 0...1."
            case .reflectionOptions: "Use finite reflective-water controls with positive scale, a nonzero sun direction and opacity in 0...1."
            case .textures: "Use distinct RGBA16Float water captures and graph textures from the same Metal device."
            case .fog: "Use finite linear fog color and increasing nonnegative distances or a nonnegative density."
            case .reservedTexture(let name): "The water graph already reserves the texture name \(name)."
            }
        }
    }
}
