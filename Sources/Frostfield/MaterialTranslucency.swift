import Foundation
import simd

extension PhysicallyBasedMaterial {
    /// KHR_materials_diffuse_transmission: a Lambertian BTDF that takes `factor` of the light
    /// entering the surface (after specular reflection) from the diffuse reflection and sends
    /// it to the other side, tinted by `colorFactor`. Specular transmission, where present,
    /// overrides it, as the extension specifies.
    ///
    /// Without a volume (`Transmission.thickness` zero) the surface is an infinitely thin
    /// translucent sheet such as a leaf or paper. With a volume, light crosses the boundary
    /// diffusely into the medium that `Transmission` (attenuation) and `VolumeScatter` describe;
    /// the path tracer random-walks it and lets the walk leave diffusely where it reaches the
    /// boundary again. Raster rendering lights thin sheets from behind with the exact Lambertian
    /// BTDF and approximates volumes with wrapped diffuse light and thickness attenuation.
    public struct DiffuseTransmission {
        /// The fraction of non-specularly reflected light that is transmitted, zero to one.
        public let factor: Float
        /// Linear RGB, zero to one.
        public let colorFactor: SIMD3<Float>
        /// Linear alpha multiplies the factor. Use an RGBA format.
        public let factorTexture: TextureBinding?
        /// RGB multiplies the color. sRGB formats decode when sampled.
        public let colorTexture: TextureBinding?

        /// Throws `Failure.invalidDiffuseTransmission` outside the documented ranges.
        public init(
            factor: Float = 0, colorFactor: SIMD3<Float> = SIMD3(repeating: 1),
            factorTexture: TextureBinding? = nil, colorTexture: TextureBinding? = nil
        ) throws {
            guard (0...1).contains(factor), (0..<3).allSatisfy({ (0...1).contains(colorFactor[$0]) }) else {
                throw Failure.invalidDiffuseTransmission
            }
            self.factor = factor
            self.colorFactor = colorFactor
            self.factorTexture = factorTexture
            self.colorTexture = colorTexture
        }
    }

    /// KHR_materials_diffuse_roughness: the diffuse lobe becomes the energy-preserving
    /// Oren-Nayar BRDF (EON, Portsmouth, Kutz and Hill 2025) that OpenPBR uses for
    /// `base_diffuse_roughness`, with roughness `factor` (zero is Lambertian). Rough surfaces
    /// back-scatter: they brighten where light and view align and darken toward the
    /// terminator, while a white surface keeps unit albedo. Both renderers evaluate the BRDF
    /// for direct light; raster image lighting scales the irradiance by the BRDF's exact
    /// directional albedo, which matches it under uniform light.
    public struct DiffuseRoughness {
        /// Zero to one.
        public let factor: Float
        /// Linear red multiplies the factor.
        public let texture: TextureBinding?

        /// Throws `Failure.invalidDiffuseRoughness` outside zero to one.
        public init(factor: Float = 0, texture: TextureBinding? = nil) throws {
            guard (0...1).contains(factor) else { throw Failure.invalidDiffuseRoughness }
            self.factor = factor
            self.texture = texture
        }
    }

    /// KHR_materials_volume_scatter (Khronos draft): scattering inside the material's volume.
    /// The volume's attenuation (`Transmission.attenuationColor` and `attenuationDistance`)
    /// becomes the extinction coefficient `sigma_t = -ln(color) / distance` of a homogeneous
    /// medium that absorbs and scatters. The single-scattering albedo comes from
    /// `multiscatterColor` through Kulla and Conty's inversion, so a deep, dense medium reflects
    /// about that color under diffuse light, and scattering follows the Henyey-Greenstein phase
    /// function with `anisotropy`. Requires a volume: a `transmission` with nonzero thickness.
    ///
    /// The path tracer samples free paths in the medium per color channel with multiple
    /// importance sampling. Raster rendering blurs and hazes the transmission backdrop by the
    /// scattering depth.
    public struct VolumeScatter {
        /// The multiple-scattering albedo, linear RGB from zero to one. Zero scatters nothing.
        public let multiscatterColor: SIMD3<Float>
        /// The Henyey-Greenstein g, greater than -1 and less than 1. Positive scatters forward.
        public let anisotropy: Float
        /// RGB multiplies the color at the point where light enters the volume. sRGB formats decode.
        public let multiscatterColorTexture: TextureBinding?

        /// Throws `Failure.invalidVolumeScatter` outside the documented ranges.
        public init(
            multiscatterColor: SIMD3<Float> = .zero, anisotropy: Float = 0, multiscatterColorTexture: TextureBinding? = nil
        ) throws {
            guard (0..<3).allSatisfy({ (0...1).contains(multiscatterColor[$0]) }), anisotropy > -1, anisotropy < 1 else {
                throw Failure.invalidVolumeScatter
            }
            self.multiscatterColor = multiscatterColor
            self.anisotropy = anisotropy
            self.multiscatterColorTexture = multiscatterColorTexture
        }
    }

    /// Random-walk subsurface scattering for opaque-looking translucent solids such as skin,
    /// wax, marble, jade and milk, with parameters based on OpenPBR and Disney subsurface materials.
    ///
    /// `weight` of the diffuse lobe enters the surface diffusely and travels through a closed,
    /// homogeneous medium with mean free path `radius * scale` per channel (extinction is its
    /// reciprocal) and the single-scattering albedo that makes a deep slab reflect `color`
    /// (Kulla and Conty's inversion). The walk leaves the surface diffusely where it reaches
    /// the boundary, and the dielectric specular layer stays on top. A zero mean free path
    /// shades as the diffuse lobe with `color`. The mesh must be closed and outward-facing.
    ///
    /// The path tracer runs up to 256 scattering events per walk. Raster rendering
    /// approximates the look with per-channel wrapped diffuse light, whose width follows the
    /// diffusion length and the surface curvature, and back-lighting attenuated through the
    /// volume thickness of `transmission` when one is set.
    public struct Subsurface {
        /// The fraction of the diffuse lobe that scatters below the surface, zero to one.
        public let weight: Float
        /// The multiple-scattering albedo, linear RGB from zero to one.
        public let color: SIMD3<Float>
        /// Per-channel multipliers of `scale`, finite and nonnegative.
        public let radius: SIMD3<Float>
        /// The mean free path in world units where a radius component is one.
        public let scale: Float
        /// The Henyey-Greenstein g, greater than -1 and less than 1.
        public let anisotropy: Float
        /// RGB multiplies `color` at the entry point. sRGB formats decode when sampled.
        public let colorTexture: TextureBinding?

        /// Throws `Failure.invalidSubsurface` outside the documented ranges.
        public init(
            weight: Float = 1, color: SIMD3<Float> = SIMD3(repeating: 0.8),
            radius: SIMD3<Float> = SIMD3(repeating: 1), scale: Float = 0.01, anisotropy: Float = 0,
            colorTexture: TextureBinding? = nil
        ) throws {
            guard (0...1).contains(weight), (0..<3).allSatisfy({ (0...1).contains(color[$0]) }),
                  (0..<3).allSatisfy({ radius[$0].isFinite && radius[$0] >= 0 }),
                  scale.isFinite, scale >= 0, anisotropy > -1, anisotropy < 1 else {
                throw Failure.invalidSubsurface
            }
            self.weight = weight
            self.color = color
            self.radius = radius
            self.scale = scale
            self.anisotropy = anisotropy
            self.colorTexture = colorTexture
        }

        /// The mean free path per channel in world units.
        public var meanFreePath: SIMD3<Float> { radius * scale }
    }
}

/// The homogeneous medium inside a material's volume, as both renderers derive it.
struct MaterialMedium: Equatable {
    /// How light crosses the volume's boundary.
    enum Boundary: Equatable {
        /// Refraction and reflection by the dielectric BSDF (KHR_materials_transmission).
        case dielectric
        /// Diffuse entry and exit (KHR_materials_diffuse_transmission with a volume, or subsurface).
        case diffuse
    }

    /// The largest extinction coefficient, per world unit, kept finite for sampling.
    static let maximumExtinction: Float = 1e8

    let boundary: Boundary
    /// Extinction per world unit.
    let extinction: SIMD3<Float>
    /// Multiple-scattering albedo before the entry-point texture; zero scatters nothing.
    let multiscatterColor: SIMD3<Float>
    let anisotropy: Float

    /// Extinction from an attenuation color and distance, as KHR_materials_volume defines it.
    static func extinction(color: SIMD3<Float>, distance: Float?) -> SIMD3<Float> {
        guard let distance, distance.isFinite, distance > 0 else { return .zero }
        var result = SIMD3<Float>.zero
        for channel in 0..<3 {
            let value = color[channel]
            result[channel] = value <= 0 ? maximumExtinction : min(-log(min(value, 1)) / distance, maximumExtinction)
        }
        return result
    }

    /// The Kulla and Conty single-scattering albedo, matching `frostfieldSingleScatteringAlbedo`.
    static func singleScatteringAlbedo(_ color: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3((0..<3).map { channel in
            let albedo = min(max(color[channel], 0), 1)
            let root = 4.09712 + 4.20863 * albedo - (9.59217 + 41.6808 * albedo + 17.7126 * albedo * albedo).squareRoot()
            return min(max(1 - root * root, 0), 1)
        })
    }

    /// The diffusion length per channel, 1 / sqrt(3 sigma_a (sigma_a + sigma_s (1 - g))), for
    /// the raster approximation; infinity where nothing is absorbed.
    var diffusionLength: SIMD3<Float> {
        let albedo = Self.singleScatteringAlbedo(multiscatterColor)
        var result = SIMD3<Float>(repeating: .infinity)
        for channel in 0..<3 {
            let absorption = extinction[channel] * (1 - albedo[channel])
            let reduced = absorption + extinction[channel] * albedo[channel] * (1 - anisotropy)
            let transport = (3 * absorption * reduced).squareRoot()
            if transport > 0 { result[channel] = 1 / transport }
        }
        return result
    }

    /// The attenuation per world unit that the raster back-lighting uses: the smaller of the
    /// extinction and the diffusion transport coefficient.
    var rasterAttenuation: SIMD3<Float> {
        let length = diffusionLength
        return SIMD3<Float>((0..<3).map { min(extinction[$0], length[$0].isFinite ? 1 / length[$0] : 0) })
    }
}

/// Splices the shared material-extension shader source into run-time compiled Metal source,
/// for diagnostics and tests of the shared shader functions.
package enum MaterialShaderSource {
    /// The `#include` line that `splicingExtensions(into:)` replaces.
    package static let includeLine = MaterialExtensionsSource.includeLine

    package static func splicingExtensions(into source: String) throws -> String {
        try MaterialExtensionsSource.splice(into: source)
    }
}
