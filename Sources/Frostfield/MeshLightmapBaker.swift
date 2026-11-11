import Foundation
import Metal
import simd

/// A static irradiance lightmap in a receiver's authored UV atlas.
///
/// The initializer bakes direct directional irradiance. Use
/// ``ProgressiveLightmapBaker`` for a snapshot that also includes indirect light.
/// Both use the coverage, overlap and padding rules of ``MeshTextureBaker``.
public struct MeshLightmapBaker {
    public struct DirectionalLight: Sendable {
        /// Unit world direction from the receiver toward the light.
        public let directionToLight: SIMD3<Float>
        /// Linear RGB irradiance at normal incidence.
        public let irradiance: SIMD3<Float>

        public init(directionToLight: SIMD3<Float>, irradiance: SIMD3<Float>) throws {
            guard directionToLight != .zero,
                  [directionToLight.x, directionToLight.y, directionToLight.z].allSatisfy(\.isFinite),
                  [irradiance.x, irradiance.y, irradiance.z].allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw Failure.light
            }
            self.directionToLight = SIMD3<Float>(simd_normalize(SIMD3<Double>(directionToLight)))
            self.irradiance = irradiance
        }
    }

    public static let maximumLightCount = 16
    /// The UV set to use when sampling the result on the receiver.
    public let textureCoordinateSet: MeshTextureCoordinateSet
    /// Linear RGB irradiance in `rgba16Float`, with alpha one on covered and padded
    /// texels. For Lambertian reflected radiance multiply RGB by linear albedo / π.
    public let irradianceTexture: any MTLTexture
    /// Stored bytes 255 for interior samples, 128 for padding, and 0 for empty texels.
    public let coverageTexture: any MTLTexture

    /// Linear irradiance for a classic material's `lightMap` input. The material
    /// multiplies this by its diffuse color divided by π. Clamp atlas edges so
    /// filtering near the atlas boundary cannot wrap into an unrelated chart.
    public func lightMapBinding() throws -> ClassicMaterial.TextureBinding {
        try .init(
            texture: irradianceTexture,
            sampling: .init(addressU: .clampToEdge, addressV: .clampToEdge, mipFilter: .notMipmapped),
            device: irradianceTexture.device, textureCoordinateSet: textureCoordinateSet
        )
    }

    /// The same linear irradiance for a physically based raster material.
    public func pbrLightMapBinding() throws -> PhysicallyBasedMaterial.LightMap {
        try .init(texture: irradianceTexture, textureCoordinateSet: textureCoordinateSet)
    }

    init(
        textureCoordinateSet: MeshTextureCoordinateSet, irradianceTexture: any MTLTexture, coverageTexture: any MTLTexture
    ) {
        self.textureCoordinateSet = textureCoordinateSet
        self.irradianceTexture = irradianceTexture
        self.coverageTexture = coverageTexture
    }

    /// Bakes synchronously on the CPU. Uploads ready-to-sample, single-level Metal textures.
    ///
    /// `worldFromObject` places the receiver, and each caster supplies its own affine
    /// world transform. When `receiverCastsShadows` is true, other receiver triangles
    /// can occlude the light. The sampled triangle is excluded from its own shadow rays.
    /// Rays start `shadowBias` world units above the surface along its geometric normal.
    /// Surfaces receive light only on the front of their geometric normal. Use the light
    /// and caster lists that should contribute to this static direct-light bake.
    public init(
        device: any MTLDevice, geometry: MeshGeometry, width: Int, height: Int,
        textureCoordinateSet: MeshTextureCoordinateSet = .secondary,
        worldFromObject: simd_float4x4 = matrix_identity_float4x4,
        lights: [DirectionalLight], casters: [MeshGeometry.MergePart] = [], receiverCastsShadows: Bool = true,
        shadowBias: Float = 0.0001, overlap: MeshTextureBaker.Overlap = .reject, padding: Int = 0
    ) throws {
        guard lights.count <= Self.maximumLightCount else { throw Failure.lightCount }
        guard shadowBias.isFinite, shadowBias >= 0 else { throw Failure.shadowBias }

        var atlas = try UVBakeAtlas(width: width, height: height, padding: padding)
        let receiver = try MeshGeometry.merged([.init(geometry, transform: worldFromObject)]).geometry
        let coordinates = try atlas.coordinates(of: receiver, set: textureCoordinateSet)
        let visibility: MeshBVH?
        if casters.isEmpty {
            visibility = receiverCastsShadows ? try MeshBVH(geometry: receiver) : nil
        } else {
            let parts = receiverCastsShadows ? [.init(receiver)] + casters : casters
            visibility = try MeshBVH(geometry: MeshGeometry.merged(parts).geometry)
        }
        let lighting = LightmapDirectionalLighting(lights)
        let bias = Double(shadowBias)
        let limit = Double(Float16.greatestFiniteMagnitude)
        var irradiance = [SIMD4<Float16>](repeating: .zero, count: width * height)
        for start in stride(from: 0, to: receiver.indices.count, by: 3) {
            let ia = Int(receiver.indices[start]), ib = Int(receiver.indices[start + 1]), ic = Int(receiver.indices[start + 2])
            let a = receiver.vertices[ia], b = receiver.vertices[ib], c = receiver.vertices[ic]
            let pa = SIMD3<Double>(a.position), pb = SIMD3<Double>(b.position), pc = SIMD3<Double>(c.position)
            let face = simd_cross(pb - pa, pc - pa)
            let geometricNormal = face == .zero ? SIMD3<Double>.zero : simd_normalize(face)
            let na = simd_normalize(SIMD3<Double>(a.normal))
            let nb = simd_normalize(SIMD3<Double>(b.normal))
            let nc = simd_normalize(SIMD3<Double>(c.normal))
            let frontLighting = lighting.facing(geometricNormal)
            let ignoredTriangle = receiverCastsShadows ? start / 3 : nil
            try atlas.rasterize(coordinates[ia], coordinates[ib], coordinates[ic], triangle: start / 3, overlap: overlap) { index, weights in
                let normal = UVBakeAtlas.interpolatedNormal(na, nb, nc, weights: weights)
                let origin = pa * weights.x + pb * weights.y + pc * weights.z + geometricNormal * bias
                let total = frontLighting.irradiance(
                    at: origin, normal: normal, visibility: visibility, ignoringTriangle: ignoredTriangle
                )
                guard simd_reduce_max(total) <= limit else { throw Failure.irradianceRange }
                irradiance[index] = SIMD4<Float16>(SIMD4(SIMD3<Float>(total), 1))
            }
        }
        try atlas.pad(by: padding) { source, destination in irradiance[destination] = irradiance[source] }
        self.textureCoordinateSet = textureCoordinateSet
        irradianceTexture = try atlas.upload(
            irradiance, device: device, format: .rgba16Float, label: "Frostfield baked directional irradiance"
        )
        coverageTexture = try atlas.coverageTexture(device: device)
    }

    public enum Failure: LocalizedError, Equatable {
        case light
        case lightCount
        case shadowBias
        case irradianceRange

        public var errorDescription: String? {
            switch self {
            case .light: "A baked directional light needs a finite, nonzero direction and finite nonnegative RGB irradiance."
            case .lightCount: "A direct lightmap supports up to \(MeshLightmapBaker.maximumLightCount) directional lights."
            case .shadowBias: "Lightmap shadow bias must be finite and nonnegative."
            case .irradianceRange: "Baked irradiance exceeds the finite RGBA16Float range."
            }
        }
    }
}

struct LightmapDirectionalLighting {
    private let lights: [(direction: SIMD3<Double>, irradiance: SIMD3<Double>)]

    init(_ lights: [MeshLightmapBaker.DirectionalLight]) {
        self.lights = lights.filter { $0.irradiance != .zero }.map {
            (simd_normalize(SIMD3<Double>($0.directionToLight)), SIMD3<Double>($0.irradiance))
        }
    }

    private init(prepared lights: [(direction: SIMD3<Double>, irradiance: SIMD3<Double>)]) {
        self.lights = lights
    }

    /// Hoists the geometric hemisphere test out of the static bake's texel loop.
    func facing(_ normal: SIMD3<Double>) -> LightmapDirectionalLighting {
        .init(prepared: lights.filter { simd_dot(normal, $0.direction) > 0 })
    }

    func irradiance(
        at origin: SIMD3<Double>, normal: SIMD3<Double>, geometricNormal: SIMD3<Double>? = nil,
        visibility: MeshBVH?, ignoringTriangle: Int?, acceptsHit: MeshBVH.HitFilter? = nil
    ) -> SIMD3<Double> {
        var total = SIMD3<Double>.zero
        for light in lights {
            let cosine = simd_dot(normal, light.direction)
            guard cosine > 0 else { continue }
            if let geometricNormal, simd_dot(geometricNormal, light.direction) <= 0 { continue }
            if visibility?.occludes(
                origin: origin, unitDirection: light.direction, ignoringTriangle: ignoringTriangle, acceptsHit: acceptsHit
            ) == true {
                continue
            }
            total += light.irradiance * cosine
        }
        return total
    }
}
