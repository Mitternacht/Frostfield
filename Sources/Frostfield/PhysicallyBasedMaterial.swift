import Foundation
import Metal
import simd

/// A metallic-roughness surface with directional and optional environment lighting.
public struct PhysicallyBasedMaterial {
    public struct Environment {
        public let lighting: EnvironmentLighting
        public let intensity: Float
        /// Rotation of the environment around the scene's positive Y axis, in radians.
        public let yaw: Float
        /// Optional world-space correction for raster specular and clearcoat IBL.
        public let boxProjection: EnvironmentBoxProjection?

        public init(lighting: EnvironmentLighting, intensity: Float = 1, yaw: Float = 0) throws {
            try self.init(lighting: lighting, intensity: intensity, yaw: yaw, boxProjection: nil)
        }

        /// Binds prepared, read-only lighting to a fixed world-space projection.
        /// Keep the material resources alive through GPU completion when using
        /// an unretained command buffer. A new capture needs a new binding.
        public init(
            lighting: EnvironmentLighting, intensity: Float = 1, yaw: Float = 0, boxProjection: EnvironmentBoxProjection?
        ) throws {
            guard intensity.isFinite, intensity >= 0, yaw.isFinite else { throw Failure.invalidEnvironment }
            self.lighting = lighting
            self.intensity = intensity
            self.yaw = yaw
            self.boxProjection = boxProjection
        }
    }

    /// An affine texture-coordinate transform, as in KHR_texture_transform and
    /// Three.js `Texture.offset`, `repeat` and `rotation`. Sampling reads
    /// `u' = cos(r) sx u + sin(r) sy v + ox` and `v' = -sin(r) sx u + cos(r) sy v + oy`:
    /// scale, then rotate by `rotation` radians, then offset, in the top-left UV space
    /// of the bound coordinate set. Normal-map tangent frames follow the transformed
    /// coordinates in both renderers.
    public struct TextureTransform: Hashable, Sendable {
        public static let identity = TextureTransform(validOffset: .zero, rotation: 0, scale: SIMD2(repeating: 1))

        /// Added after scale and rotation, in texture coordinates.
        public let offset: SIMD2<Float>
        /// Radians.
        public let rotation: Float
        /// Texture repeats per unit of the source coordinates. Components must be nonzero.
        public let scale: SIMD2<Float>

        /// Throws `Failure.invalidTextureTransform` for nonfinite values or a zero scale component.
        public init(offset: SIMD2<Float> = .zero, rotation: Float = 0, scale: SIMD2<Float> = SIMD2(repeating: 1)) throws {
            guard offset.x.isFinite, offset.y.isFinite, rotation.isFinite,
                  scale.x.isFinite, scale.y.isFinite, scale.x != 0, scale.y != 0 else {
                throw Failure.invalidTextureTransform
            }
            self.init(validOffset: offset, rotation: rotation, scale: scale)
        }

        private init(validOffset: SIMD2<Float>, rotation: Float, scale: SIMD2<Float>) {
            offset = validOffset
            self.rotation = rotation
            self.scale = scale
        }

        /// The transformed coordinate that a bound map samples.
        public func apply(to coordinate: SIMD2<Float>) -> SIMD2<Float> {
            let (first, second) = rows
            return SIMD2(
                first.x * coordinate.x + first.y * coordinate.y + first.z,
                second.x * coordinate.x + second.y * coordinate.y + second.z
            )
        }

        /// The two shader rows: `u' = dot(r0.xy, uv) + r0.z` and `v' = dot(r1.xy, uv) + r1.z`.
        var rows: (SIMD4<Float>, SIMD4<Float>) {
            let cosine = cos(rotation)
            let sine = sin(rotation)
            return (
                SIMD4(cosine * scale.x, sine * scale.y, offset.x, 0),
                SIMD4(-sine * scale.x, cosine * scale.y, offset.y, 0)
            )
        }
    }

    /// Sampler settings that both renderers can read, as in glTF samplers and Three.js
    /// `wrapS`, `wrapT`, `magFilter` and `minFilter`. Metal sampler states cannot be
    /// inspected, so the path tracer reads this description instead of a `sampler`.
    ///
    /// The path tracer honors the addressing of each axis and the magnification filter,
    /// and always blends its mip levels linearly. Raster draws honor every setting.
    /// The default repeats both axes with linear filtering and linear mip blending.
    public struct TextureSampling: Hashable, Sendable, Codable {
        /// The addressing modes surface graph textures use: repeat, clamp to edge or mirrored repeat.
        public typealias Addressing = ShaderTextureInput.Addressing

        /// The texel filter of one stage. Unlike `ShaderTextureInput.Filtering`, it does not
        /// choose the mip filter, which `MipFiltering` sets separately.
        public enum Filtering: String, Hashable, Sendable, Codable {
            case nearest
            case linear
        }

        public enum MipFiltering: String, Hashable, Sendable, Codable {
            /// Samples only the base level.
            case notMipmapped
            case nearest
            case linear
        }

        /// Repeat addressing, linear filtering and linear mip blending.
        public static let `default` = TextureSampling()

        /// Addressing of the first texture coordinate (U, glTF `wrapS`).
        public let addressU: Addressing
        /// Addressing of the second texture coordinate (V, glTF `wrapT`).
        public let addressV: Addressing
        public let magnification: Filtering
        public let minification: Filtering
        public let mipFilter: MipFiltering

        public init(
            addressU: Addressing = .repeat, addressV: Addressing = .repeat,
            magnification: Filtering = .linear, minification: Filtering = .linear, mipFilter: MipFiltering = .linear
        ) {
            self.addressU = addressU
            self.addressV = addressV
            self.magnification = magnification
            self.minification = minification
            self.mipFilter = mipFilter
        }

        /// A new Metal descriptor with these settings.
        public var descriptor: MTLSamplerDescriptor {
            func metal(_ mode: Addressing) -> MTLSamplerAddressMode {
                switch mode {
                case .repeat: .repeat
                case .clampToEdge: .clampToEdge
                case .mirroredRepeat: .mirrorRepeat
                }
            }
            let descriptor = MTLSamplerDescriptor()
            descriptor.sAddressMode = metal(addressU)
            descriptor.tAddressMode = metal(addressV)
            descriptor.magFilter = magnification == .nearest ? .nearest : .linear
            descriptor.minFilter = minification == .nearest ? .nearest : .linear
            switch mipFilter {
            case .notMipmapped: descriptor.mipFilter = .notMipmapped
            case .nearest: descriptor.mipFilter = .nearest
            case .linear: descriptor.mipFilter = .linear
            }
            return descriptor
        }

        /// Throws `MeshRenderer.Failure.resource` when Metal cannot create the sampler.
        public func makeSamplerState(device: any MTLDevice) throws -> any MTLSamplerState {
            guard let sampler = device.makeSamplerState(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("texture sampler")
            }
            return sampler
        }
    }

    /// How a stock normal map stores its vector.
    public enum NormalEncoding: UInt32, Codable, Sendable {
        /// RGB channels in zero to one, remapped to minus one to one.
        case rgb = 0
        /// Red and green in zero to one, with the positive Z component reconstructed.
        case xyUNorm = 1
        /// Red and green in minus one to one, with the positive Z component reconstructed.
        case xySNorm = 2

        func acceptsPackedTexture(_ format: MTLPixelFormat) -> Bool {
            switch format {
            case .rg16Float, .rgba16Float:
                return self != .rgb
            case .rg8Unorm, .rg16Unorm, .rgba8Unorm, .bgra8Unorm, .rgba16Unorm, .bc5_rgUnorm, .eac_rg11Unorm:
                return self == .xyUNorm
            case .rg8Snorm, .rg16Snorm, .rgba8Snorm, .rgba16Snorm, .bc5_rgSnorm, .eac_rg11Snorm:
                return self == .xySNorm
            default:
                return false
            }
        }
    }

    public enum NormalSpace: UInt32, Codable, Sendable {
        /// Interpret the normal in the selected UV tangent frame and apply normal scale.
        case tangentSpace = 0
        /// Interpret the normal in mesh coordinates, before the node and instance transforms.
        /// Normal scale is ignored. The texture still uses its selected UV set and transform.
        case objectSpace = 1
    }

    /// The sampler is optional. The default repeats with linear filtering.
    ///
    /// `sampling` describes the binding's sampler for renderers that cannot read a Metal
    /// sampler state. The path tracer samples with it and rejects a `sampler` without
    /// one. Raster draws use `sampler` when it is set and otherwise create one from
    /// `sampling`, so a description passed with a sampler must match it.
    public struct TextureBinding {
        public let texture: any MTLTexture
        public let sampler: (any MTLSamplerState)?
        public let textureCoordinateSet: MeshTextureCoordinateSet
        /// Applied to the coordinate set before sampling.
        public let transform: TextureTransform
        /// The settings of `sampler`, or of the default sampler when nil.
        public let sampling: TextureSampling?
        /// Used when this binding supplies a stock base or clearcoat normal map.
        public let normalEncoding: NormalEncoding
        public let normalSpace: NormalSpace

        public init(
            texture: any MTLTexture, sampler: (any MTLSamplerState)? = nil,
            textureCoordinateSet: MeshTextureCoordinateSet = .primary,
            transform: TextureTransform = .identity,
            sampling: TextureSampling? = nil, normalEncoding: NormalEncoding = .rgb,
            normalSpace: NormalSpace = .tangentSpace
        ) {
            self.texture = texture
            self.sampler = sampler
            self.textureCoordinateSet = textureCoordinateSet
            self.transform = transform
            self.sampling = sampling
            self.normalEncoding = normalEncoding
            self.normalSpace = normalSpace
        }

        /// Creates the sampler state for `sampling` on `device` and keeps both, so the
        /// raster renderer and the path tracer sample alike.
        public init(
            texture: any MTLTexture, sampling: TextureSampling, device: any MTLDevice,
            textureCoordinateSet: MeshTextureCoordinateSet = .primary,
            transform: TextureTransform = .identity, normalEncoding: NormalEncoding = .rgb,
            normalSpace: NormalSpace = .tangentSpace
        ) throws {
            self.init(
                texture: texture, sampler: try sampling.makeSamplerState(device: device),
                textureCoordinateSet: textureCoordinateSet, transform: transform, sampling: sampling,
                normalEncoding: normalEncoding, normalSpace: normalSpace
            )
        }
    }

    /// A linear red-channel height texture that perturbs the base shading normal.
    public struct BumpMap {
        public let texture: TextureBinding
        /// Signed multiplier for a height change across the local sampling footprint.
        public let scale: Float

        /// A normal texture takes precedence when both are present. The binding's
        /// coordinate set, transform and sampling apply to the height texture.
        public init(texture: TextureBinding, scale: Float = 1) throws {
            guard scale.isFinite else { throw Failure.invalidBumpScale }
            let height = texture.texture
            guard PhysicallyBasedMaterial.isSurfaceTexture(height, content: .data),
                  height.storageMode != .memoryless, !MaterialTextureFormat.isFloat32(height.pixelFormat) else {
                throw Failure.invalidBumpTexture
            }
            self.texture = texture
            self.scale = scale
        }
    }

    /// A dielectric GGX layer with IOR 1.5 over the base material.
    public struct Clearcoat {
        public let factor: Float
        public let roughness: Float
        /// Multiplies the layer factor by the linear red channel.
        public let factorTexture: TextureBinding?
        /// Multiplies perceptual roughness by the linear green channel.
        public let roughnessTexture: TextureBinding?
        /// An independent, linear tangent-space normal map for the layer.
        public let normalTexture: TextureBinding?
        public let normalScale: Float

        public init(
            factor: Float = 0, roughness: Float = 0,
            factorTexture: TextureBinding? = nil, roughnessTexture: TextureBinding? = nil,
            normalTexture: TextureBinding? = nil, normalScale: Float = 1
        ) throws {
            guard (0...1).contains(factor), (0...1).contains(roughness), normalScale.isFinite else {
                throw Failure.invalidClearcoat
            }
            self.factor = factor
            self.roughness = roughness
            self.factorTexture = factorTexture
            self.roughnessTexture = roughnessTexture
            self.normalTexture = normalTexture
            self.normalScale = normalScale
        }
    }

    /// Optical transmission through a dielectric surface or volume. A factor of
    /// zero can still specify the dielectric index of refraction.
    ///
    /// Active transmission samples an opaque-scene HDR snapshot. The estimate
    /// cannot see geometry outside that image or behind another glass layer.
    public struct Transmission {
        public let factor: Float
        public let indexOfRefraction: Float
        /// The thickness of the medium in mesh units. Zero is a thin surface.
        public let thickness: Float
        /// The absorption distance in world units. Nil represents infinity.
        public let attenuationDistance: Float?
        public let attenuationColor: SIMD3<Float>
        /// Linear red multiplies the transmission factor.
        public let factorTexture: TextureBinding?
        /// Linear green multiplies mesh-space thickness.
        public let thicknessTexture: TextureBinding?
        /// KHR_materials_dispersion: 20 divided by the Abbe number, or zero for none.
        /// The path tracer refracts volumes (nonzero thickness) with a per-wavelength
        /// index of refraction, sampling one of red, green or blue per path at the
        /// Fraunhofer C, d and F lines, where green uses `indexOfRefraction` exactly.
        /// Raster samples the backdrop once per wavelength when this is positive and the
        /// thickness is positive. Zero keeps the single base-index sample.
        public let dispersion: Float

        public init(
            factor: Float,
            indexOfRefraction: Float = 1.5,
            thickness: Float = 0,
            attenuationDistance: Float? = nil,
            attenuationColor: SIMD3<Float> = SIMD3(repeating: 1),
            factorTexture: TextureBinding? = nil,
            thicknessTexture: TextureBinding? = nil,
            dispersion: Float = 0
        ) throws {
            guard dispersion.isFinite, dispersion >= 0 else { throw Failure.invalidDispersion }
            guard (0...1).contains(factor), thickness.isFinite, thickness >= 0,
                  (0..<3).allSatisfy({ (0...1).contains(attenuationColor[$0]) }),
                  attenuationDistance.map({ $0 > 0 }) ?? true else {
                throw Failure.invalidTransmission
            }
            guard indexOfRefraction.isFinite, indexOfRefraction == 0 || indexOfRefraction >= 1 else {
                throw Failure.invalidIndexOfRefraction
            }
            self.factor = factor
            self.indexOfRefraction = indexOfRefraction
            self.thickness = thickness
            self.attenuationDistance = attenuationDistance == .infinity ? nil : attenuationDistance
            self.attenuationColor = attenuationColor
            self.factorTexture = factorTexture
            self.thicknessTexture = thicknessTexture
            self.dispersion = dispersion
        }
    }

    /// KHR_materials_specular, Three.js `specularIntensity` and `specularColor`.
    /// The dielectric reflectance at normal incidence becomes
    /// `min(F0(ior) * color, 1) * factor` and at grazing incidence `factor`, so a
    /// zero factor removes dielectric reflection. Metals are unaffected.
    public struct Specular {
        /// Zero to one.
        public let factor: Float
        /// Finite, nonnegative linear RGB.
        public let colorFactor: SIMD3<Float>
        /// Linear alpha multiplies the factor. Use an RGBA format.
        public let factorTexture: TextureBinding?
        /// RGB multiplies the color factor. sRGB formats decode when sampled.
        public let colorTexture: TextureBinding?

        /// Throws `Failure.invalidSpecular` outside the documented ranges.
        public init(
            factor: Float = 1, colorFactor: SIMD3<Float> = SIMD3(repeating: 1),
            factorTexture: TextureBinding? = nil, colorTexture: TextureBinding? = nil
        ) throws {
            guard (0...1).contains(factor), (0..<3).allSatisfy({ colorFactor[$0].isFinite && colorFactor[$0] >= 0 }) else {
                throw Failure.invalidSpecular
            }
            self.factor = factor
            self.colorFactor = colorFactor
            self.factorTexture = factorTexture
            self.colorTexture = colorTexture
        }
    }

    /// KHR_materials_sheen, Three.js `sheen`: a Charlie microfacet lobe with Neubelt
    /// visibility above the base layer and below clearcoat. The base layer is scaled
    /// by `min(1 - max(color) E(NoV), 1 - max(color) E(NoL))` using a tabulated
    /// directional albedo E, clamped at zero. Raster image lighting approximates
    /// the lobe as the diffuse irradiance times `color * E(NoV)`. Perceptual
    /// roughness below 0.07 shades as 0.07.
    public struct Sheen {
        /// Linear RGB from zero to one. Zero disables the layer.
        public let colorFactor: SIMD3<Float>
        /// Perceptual roughness from zero to one.
        public let roughness: Float
        /// RGB multiplies the color. sRGB formats decode when sampled.
        public let colorTexture: TextureBinding?
        /// Linear alpha multiplies the roughness. Use an RGBA format.
        public let roughnessTexture: TextureBinding?

        /// Throws `Failure.invalidSheen` outside the documented ranges.
        public init(
            colorFactor: SIMD3<Float> = .zero, roughness: Float = 0,
            colorTexture: TextureBinding? = nil, roughnessTexture: TextureBinding? = nil
        ) throws {
            guard (0..<3).allSatisfy({ (0...1).contains(colorFactor[$0]) }), (0...1).contains(roughness) else {
                throw Failure.invalidSheen
            }
            self.colorFactor = colorFactor
            self.roughness = roughness
            self.colorTexture = colorTexture
            self.roughnessTexture = roughnessTexture
        }
    }

    /// KHR_materials_anisotropy, Three.js `anisotropy`: anisotropic GGX with
    /// `alpha_t = mix(alpha, 1, strength^2)` along the anisotropy direction and
    /// `alpha_b = alpha` across it. The direction starts at the mesh tangent (the
    /// vertex tangent in raster when present, otherwise derived from primary UVs)
    /// and turns toward the bitangent by `rotation`. Raster image lighting uses a
    /// bent reflection normal. Where the UVs define no frame the surface stays isotropic.
    public struct Anisotropy {
        /// Zero to one.
        public let strength: Float
        /// Radians, counterclockwise from the tangent toward the bitangent.
        public let rotation: Float
        /// Linear red and green encode a tangent-space direction in [-1, 1]; blue
        /// multiplies the strength. Use an RGBA format.
        public let texture: TextureBinding?

        /// Throws `Failure.invalidAnisotropy` outside the documented ranges.
        public init(strength: Float = 0, rotation: Float = 0, texture: TextureBinding? = nil) throws {
            guard (0...1).contains(strength), rotation.isFinite else { throw Failure.invalidAnisotropy }
            self.strength = strength
            self.rotation = rotation
            self.texture = texture
        }
    }

    /// KHR_materials_iridescence, Three.js `iridescence`: a thin dielectric film whose
    /// Belcour and Barla Fresnel replaces the specular Fresnel by `factor`. Raster
    /// shading evaluates the film once per pixel at the view angle; the path tracer
    /// evaluates it per microfacet sample.
    public struct Iridescence {
        /// Zero to one.
        public let factor: Float
        /// The film's index of refraction, at least one.
        public let indexOfRefraction: Float
        /// Film thickness in nanometers where the thickness texture's green is zero.
        public let minimumThickness: Float
        /// Film thickness in nanometers where green is one, or without a texture.
        public let maximumThickness: Float
        /// Linear red multiplies the factor.
        public let factorTexture: TextureBinding?
        /// Linear green interpolates from the minimum to the maximum thickness.
        public let thicknessTexture: TextureBinding?

        /// Throws `Failure.invalidIridescence` outside the documented ranges.
        public init(
            factor: Float = 0, indexOfRefraction: Float = 1.3,
            minimumThickness: Float = 100, maximumThickness: Float = 400,
            factorTexture: TextureBinding? = nil, thicknessTexture: TextureBinding? = nil
        ) throws {
            guard (0...1).contains(factor), indexOfRefraction.isFinite, indexOfRefraction >= 1,
                  minimumThickness.isFinite, minimumThickness >= 0,
                  maximumThickness.isFinite, maximumThickness >= 0 else {
                throw Failure.invalidIridescence
            }
            self.factor = factor
            self.indexOfRefraction = indexOfRefraction
            self.minimumThickness = minimumThickness
            self.maximumThickness = maximumThickness
            self.factorTexture = factorTexture
            self.thicknessTexture = thicknessTexture
        }
    }

    /// How the surface responds to light.
    public enum ShadingModel: Equatable, Sendable {
        case physicallyBased
        /// KHR_materials_unlit, Three.js `MeshBasicMaterial`: outputs the base color
        /// (factor, texture, tint and surface graph) with the material's alpha
        /// mode, ignoring lights, emission, normals and optical layers. Set
        /// `unlitUsesGraphEmission` to use a graph's HDR emission output as color.
        /// The path tracer emits the selected color as radiance without scattering.
        case unlit
    }

    /// The path tracer tests a mask's cutoff on every ray, and traces blended surfaces
    /// as stochastic opacity: each ray accepts a hit with probability alpha, which
    /// converges to alpha coverage without sorting. Material-alpha surfaces stay opaque
    /// to every ray, as their raster depth writes and shadows are, and a camera ray that
    /// first hits one scales its radiance and coverage by the alpha, as raster writes it.
    public enum AlphaMode: Equatable, Sendable {
        /// Write the material alpha and depth using the original Frostfield behavior.
        case materialAlpha
        case opaque
        case mask(cutoff: Float)
        /// Premultiplied alpha blending with depth testing and no depth writes.
        case blend
    }

    /// Linear RGB irradiance baked into a mesh's UV atlas. Alpha is ignored.
    /// Raster and path tracing add the diffuse response from the atlas to live
    /// lighting. Use the atlas for light sources excluded from live lighting.
    public struct LightMap {
        public let texture: any MTLTexture
        public let textureCoordinateSet: MeshTextureCoordinateSet
        public let transform: TextureTransform

        public init(
            texture: any MTLTexture, textureCoordinateSet: MeshTextureCoordinateSet = .secondary,
            transform: TextureTransform = .identity
        ) throws {
            guard texture.textureType == .type2D, texture.sampleCount == 1, !texture.isFramebufferOnly,
                  texture.storageMode != .memoryless, texture.hazardTrackingMode != .untracked,
                  texture.usage.isEmpty || texture.usage.contains(.shaderRead) else {
                throw Failure.invalidLightMap
            }
            switch texture.pixelFormat {
            case .rgba16Float, .rgba32Float, .rgba8Unorm, .bgra8Unorm: break
            default: throw Failure.invalidLightMap
            }
            self.texture = texture
            self.textureCoordinateSet = textureCoordinateSet
            self.transform = transform
        }
    }

    /// A straight-alpha color texture. Metal decodes sRGB texture formats when sampling.
    /// Without a texture, the material uses the scene tint.
    public let baseColorTexture: (any MTLTexture)?
    /// The base color map with its sampler, UV set, and current UV transform.
    public var baseColorTextureBinding: TextureBinding? { current.bindings[0] }
    public let baseColorFactor: SIMD4<Float>
    public let alphaMode: AlphaMode
    public let isDoubleSided: Bool
    /// Uses the scene's directional shadow map for the direct light when enabled.
    public let receivesShadows: Bool
    /// When false, a display pass that receives this material's exclusion mask copies the
    /// scene color without exposure or a tone curve. Defaults to true.
    public let toneMapped: Bool
    /// Draws each triangle's edges instead of its face. Defaults to false.
    public let wireframe: Bool
    /// Shades with the triangle's face normal instead of the interpolated vertex normal.
    public let flatShading: Bool
    /// When false, the material is not drawn and does not cast a shadow. Picking still hits it.
    /// Defaults to true, like Three.js `Material.visible`.
    public let isVisible: Bool
    /// When false, a fog pass that samples this material's exclusion mask leaves the pixel unchanged.
    /// Defaults to true, like Three.js `Material.fog`.
    public let fog: Bool
    /// Adds the r187 material dither when this mesh is drawn into an 8-bit color target.
    /// A float target leaves the color unchanged. Defaults to false.
    public let dithering: Bool
    /// Replaces the constant ambient fill with prefiltered diffuse and specular light.
    public let environment: Environment?
    public let clearcoat: Clearcoat?
    public let transmission: Transmission?
    /// Overrides selected PBR channels after texture sampling. This leaves
    /// alpha coverage, normal mapping, lighting, and optical layers in place.
    /// With a graph, draw material parameters are its four inputs. Base color
    /// and alpha use the material's own factor and texture as their defaults.
    public let surfaceGraph: SurfaceShaderGraph?
    /// Live values for the surface graph's declared uniforms, shared by every
    /// renderer prepared with this material. Edits apply to the next encoded draw
    /// or path-tracing dispatch without rebuilding a pipeline.
    public let surfaceUniforms: ShaderUniforms?
    /// The textures bound to the surface graph's declared inputs, in declaration order.
    public let surfaceTextures: [any MTLTexture]
    /// Applied to the base color texture's coordinates.
    public let baseColorTextureTransform: TextureTransform
    /// Describes the base color sampler, or the default sampler when nil.
    public let baseColorSampling: TextureSampling?
    /// KHR_materials_emissive_strength: a finite, nonnegative multiplier of the
    /// emissive factor, in the same linear radiance units, without an upper bound.
    public let emissiveStrength: Float
    /// Dielectric specular strength and tint. Nil keeps F0 from the index of refraction.
    public let specular: Specular?
    /// A sheen layer; inactive while its color is zero.
    public let sheen: Sheen?
    /// Anisotropic roughness; inactive at zero strength.
    public let anisotropy: Anisotropy?
    /// A thin-film layer; inactive at a zero factor.
    public let iridescence: Iridescence?
    /// Unlit materials ignore lighting, emission and every optical layer.
    public let shadingModel: ShadingModel
    /// Opts an unlit material into graph emission as its linear color. Base
    /// texture, tint and graph coverage still determine alpha.
    public let unlitUsesGraphEmission: Bool
    /// The emissive factor before `emissiveStrength` scales it.
    public let emissiveFactor: SIMD3<Float>
    /// The emissive map with its sampler, UV set, and current UV transform.
    public var emissiveTextureBinding: TextureBinding? { current.bindings[4] }
    /// Factors that can change after preparation, shared by every renderer prepared
    /// with this material. Nil unless the material came from `withLiveFactors()`.
    public internal(set) var liveFactors: LiveFactors?
    /// The material with its live factors applied, which encoding reads.
    var current: PhysicallyBasedMaterial { liveFactors?.material ?? self }
    /// Three.js `alphaMap`: the linear green channel multiplies alpha before the alpha
    /// mode applies. It shares sampler slot 0 with the base color map: with a base color
    /// texture, its binding's sampler must be nil or `baseColorSampler`; without one, its
    /// own sampler (or the default) applies. Raster only; the path tracer rejects it.
    public let alphaTexture: TextureBinding?
    /// Base-layer height map, used when no base normal texture is present.
    public let bumpMap: BumpMap?
    /// Diffuse transmission through a thin sheet or into the volume; inactive at a zero factor.
    public let diffuseTransmission: DiffuseTransmission?
    /// Rough (Oren-Nayar) diffuse reflection; inactive at a zero factor.
    public let diffuseRoughness: DiffuseRoughness?
    /// Scattering inside the volume that `transmission` defines.
    public let volumeScatter: VolumeScatter?
    /// Random-walk subsurface scattering; inactive at a zero weight.
    public let subsurface: Subsurface?
    /// Optional baked irradiance for the diffuse reflection lobe.
    public let lightMap: LightMap?
    /// Finite nonnegative multiplier for the baked irradiance.
    public let lightMapIntensity: Float
    /// Blends ordinary microfacet reflection toward reflection back to the light,
    /// from zero (ordinary) to one (fully retroreflective).
    public let retroreflectivity: Float
    var hasRetroreflection: Bool { shadingModel == .physicallyBased && retroreflectivity > 0 }
    var hasLightMap: Bool { shadingModel == .physicallyBased && lightMap != nil && lightMapIntensity > 0 }
    // Unlit shading ignores the optical layers.
    var hasTransmission: Bool {
        shadingModel == .physicallyBased && transmission.map { $0.factor > 0 && $0.indexOfRefraction != 0 } ?? false
    }
    var hasClearcoat: Bool { shadingModel == .physicallyBased && (clearcoat?.factor ?? 0) > 0 }
    var hasSheen: Bool { sheen.map { $0.colorFactor != .zero } ?? false }
    var hasAnisotropy: Bool { (anisotropy?.strength ?? 0) > 0 }
    var hasIridescence: Bool { (iridescence?.factor ?? 0) > 0 }
    var hasDiffuseTransmission: Bool { shadingModel == .physicallyBased && (diffuseTransmission?.factor ?? 0) > 0 }
    var hasDiffuseRoughness: Bool { shadingModel == .physicallyBased && (diffuseRoughness?.factor ?? 0) > 0 }
    var hasSubsurface: Bool { shadingModel == .physicallyBased && (subsurface?.weight ?? 0) > 0 }
    /// Whether the raster PBR shader compiles its specular, sheen, anisotropy, iridescence,
    /// diffuse transmission, diffuse roughness, subsurface and volume-scatter layers.
    var hasLayers: Bool {
        shadingModel == .physicallyBased && (specular != nil || hasSheen || hasAnisotropy || hasIridescence
            || hasDiffuseTransmission || hasDiffuseRoughness || hasSubsurface || (volumeScatter != nil && medium != nil))
    }
    /// The volume's homogeneous medium: subsurface, a diffuse-transmission volume, or a
    /// transmission volume. Nil without a volume.
    let medium: MaterialMedium?
    package let lightParameters: SIMD4<Float>
    package let settings: SIMD4<Float>
    private let bindings: [TextureBinding?]
    /// Specular, specular color, sheen color, sheen roughness, anisotropy, iridescence,
    /// iridescence thickness, diffuse transmission, diffuse transmission color, diffuse
    /// roughness and multiple-scattering color maps, in shader order. Inactive layers hold nil.
    let layerBindings: [TextureBinding?]
    private let controls: SIMD4<Float>
    private let textureCoordinateMask: UInt32
    var usesSecondaryTextureCoordinates: Bool {
        textureCoordinateMask != 0 || alphaTexture?.textureCoordinateSet == .secondary
            || surfaceGraph?.bumpUsesSecondaryTextureCoordinates == true
    }
    private let emission: SIMD4<Float>
    package let packedExtensions: PackedExtensions
    /// Initializer arguments the material does not otherwise expose, for `SceneArchive`.
    package let sourceInputs: SourceInputs

    var pathTracingMaps: [TextureBinding?] { [bindings[0], bindings[2], bindings[1], bindings[4]] }
    var pathTracingNormalScale: Float { controls.x }
    var pathTracingEmission: SIMD3<Float> { SIMD3(emission.x, emission.y, emission.z) }
    var hasActiveBumpMap: Bool { sourceInputs.normalTexture == nil && (bumpMap?.scale ?? 0) != 0 }
    // Code 3 shares the existing base-normal slot. Clearcoat only decodes normal vectors.
    var baseNormalKind: UInt32 { hasActiveBumpMap ? 3 : baseNormalEncoding.rawValue }
    var baseNormalEncoding: NormalEncoding { sourceInputs.normalTexture?.normalEncoding ?? .rgb }
    var coatNormalEncoding: NormalEncoding { hasClearcoat ? clearcoat?.normalTexture?.normalEncoding ?? .rgb : .rgb }
    /// Object-space flags for the base normal (bit 0) and clearcoat normal (bit 1).
    var objectNormalMaps: UInt32 {
        let base = sourceInputs.normalTexture?.normalSpace ?? .tangentSpace
        let coat = hasClearcoat ? clearcoat?.normalTexture?.normalSpace ?? .tangentSpace : .tangentSpace
        return base.rawValue | (coat.rawValue << 1)
    }

    /// Base color, metallic-roughness, normal, occlusion, and emissive bindings, for glTF export.
    var coreMaps: (bindings: [TextureBinding?], normalScale: Float, occlusionStrength: Float) {
        var maps = Array(bindings.prefix(5))
        maps[2] = sourceInputs.normalTexture
        return (maps, sourceInputs.normalTexture == nil ? 0 : controls.x, controls.y)
    }

    /// Clearcoat factor, clearcoat roughness and transmission factor maps, then the layer maps.
    var pathTracingLayerMaps: [TextureBinding?] {
        let coat = hasClearcoat ? clearcoat : nil
        let transmission = hasTransmission ? transmission : nil
        return [coat?.factorTexture, coat?.roughnessTexture, transmission?.factorTexture] + layerBindings.prefix(7)
    }

    /// Clearcoat normal, diffuse transmission, diffuse transmission color, diffuse roughness,
    /// multiple-scattering color, and baked irradiance, the path tracer's lobe slots.
    var pathTracingLobeMaps: [TextureBinding?] {
        let irradiance: TextureBinding? = hasLightMap ? lightMap.map {
            TextureBinding(
                texture: $0.texture, textureCoordinateSet: $0.textureCoordinateSet, transform: $0.transform,
                sampling: TextureSampling(addressU: .clampToEdge, addressV: .clampToEdge)
            )
        } : nil
        return [hasClearcoat ? clearcoat?.normalTexture : nil] + layerBindings.suffix(4) + [irradiance]
    }

    /// Fragment buffer 21 of the raster PBR shader, matching `FrostfieldPBRExtensions` in
    /// PhysicallyBasedShading.h: layer factors, flags, and two transform rows for each
    /// of the 22 maps (0 base color, 1 metallic-roughness, 2 normal, 3 occlusion,
    /// 4 emissive, 5–7 clearcoat, 8 transmission, 9 thickness, 10–20 layer maps, 21 light map), then
    /// the translucency factors.
    package struct PackedExtensions {
        /// Specular color factor (rgb) and factor (a).
        package var specular: SIMD4<Float>
        /// Sheen color (rgb) and perceptual roughness (a).
        package var sheen: SIMD4<Float>
        /// Strength, cosine and sine of the rotation, then 1 when flat shading is on.
        package var anisotropy: SIMD4<Float>
        /// Factor, film index of refraction, minimum and maximum thickness in nanometers.
        package var iridescence: SIMD4<Float>
        /// One for unlit shading, light-map intensity, retroreflectivity, then unused.
        package var options: SIMD4<Float>
        /// Bound layer maps (bit i is map 10 + i), then three-bit sampler slots for layer maps
        /// 0–6 in y and 7–10 in z.
        package var maps: SIMD4<UInt32>
        package var transforms: InlineArray<44, SIMD4<Float>>
        /// Diffuse transmission color (rgb) and the translucent share of the diffuse lobe (a):
        /// the diffuse transmission factor or subsurface weight.
        package var translucency: SIMD4<Float>
        /// Multiple-scattering color (rgb) and mode (a): 0 none, 1 thin sheet, 2 diffuse volume.
        package var scattering: SIMD4<Float>
        /// Raster attenuation per world unit (rgb) and volume thickness in mesh units (a).
        package var attenuation: SIMD4<Float>
        /// Diffusion length in world units (rgb, zero for no wrap) and diffuse roughness (a).
        package var diffusion: SIMD4<Float>
    }

    /// Metallic and perceptual roughness range from zero to one. Intensities are
    /// nonnegative linear values. The light direction points from the surface to the light.
    /// With shadow reception enabled, encode the material in a graph with a shadow map.
    ///
    /// Layer maps (specular, sheen, anisotropy, iridescence) may use at most six
    /// distinct samplers; preparing a renderer otherwise throws `Failure.layerSamplerLimit`.
    /// `baseColorSampling` describes `baseColorSampler`, as `TextureBinding.sampling` does.
    public init(
        baseColorTexture: (any MTLTexture)? = nil,
        metallic: Float = 0,
        roughness: Float = 0.5,
        directionToLight: SIMD3<Float> = SIMD3(0, 1, 1),
        lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12,
        receivesShadows: Bool = false,
        baseColorFactor: SIMD4<Float> = SIMD4(repeating: 1),
        baseColorSampler: (any MTLSamplerState)? = nil,
        metallicRoughnessTexture: TextureBinding? = nil,
        normalTexture: TextureBinding? = nil,
        normalScale: Float = 1,
        occlusionTexture: TextureBinding? = nil,
        occlusionStrength: Float = 1,
        emissiveTexture: TextureBinding? = nil,
        emissiveFactor: SIMD3<Float> = .zero,
        alphaMode: AlphaMode = .materialAlpha,
        isDoubleSided: Bool = false,
        environment: Environment? = nil,
        baseColorTextureCoordinateSet: MeshTextureCoordinateSet = .primary,
        clearcoat: Clearcoat? = nil,
        transmission: Transmission? = nil,
        surfaceGraph: SurfaceShaderGraph? = nil,
        surfaceUniforms: ShaderUniforms? = nil,
        surfaceTextures: [String: any MTLTexture] = [:],
        baseColorTextureTransform: TextureTransform = .identity,
        emissiveStrength: Float = 1,
        specular: Specular? = nil,
        sheen: Sheen? = nil,
        anisotropy: Anisotropy? = nil,
        iridescence: Iridescence? = nil,
        shadingModel: ShadingModel = .physicallyBased,
        baseColorSampling: TextureSampling? = nil,
        alphaTexture: TextureBinding? = nil,
        diffuseTransmission: DiffuseTransmission? = nil,
        diffuseRoughness: DiffuseRoughness? = nil,
        volumeScatter: VolumeScatter? = nil,
        subsurface: Subsurface? = nil,
        bumpMap: BumpMap? = nil,
        lightMap: LightMap? = nil,
        lightMapIntensity: Float = 1,
        retroreflectivity: Float = 0,
        unlitUsesGraphEmission: Bool = false,
        toneMapped: Bool = true,
        wireframe: Bool = false,
        flatShading: Bool = false,
        isVisible: Bool = true,
        fog: Bool = true,
        dithering: Bool = false
    ) throws {
        guard !unlitUsesGraphEmission || (shadingModel == .unlit && surfaceGraph?.outputs.emission != nil) else {
            throw Failure.unlitGraphEmission
        }
        if let alphaTexture, let sampler = alphaTexture.sampler, baseColorTexture != nil, sampler !== baseColorSampler {
            throw Failure.alphaTextureSampler
        }
        guard (0...1).contains(metallic) else { throw Failure.invalidMetallic }
        guard (0...1).contains(roughness) else { throw Failure.invalidRoughness }
        guard directionToLight.x.isFinite, directionToLight.y.isFinite, directionToLight.z.isFinite else {
            throw Failure.invalidDirection
        }
        let scale = max(abs(directionToLight.x), abs(directionToLight.y), abs(directionToLight.z))
        guard scale > 0 else { throw Failure.invalidDirection }
        guard lightIntensity.isFinite, lightIntensity >= 0 else { throw Failure.invalidLightIntensity }
        guard ambientIntensity.isFinite, ambientIntensity >= 0 else { throw Failure.invalidAmbientIntensity }
        guard lightMapIntensity.isFinite, lightMapIntensity >= 0 else { throw Failure.invalidLightMapIntensity }
        self.lightMap = lightMap
        self.lightMapIntensity = lightMapIntensity
        guard (0...1).contains(retroreflectivity) else { throw Failure.invalidRetroreflectivity }
        self.retroreflectivity = retroreflectivity
        guard (0..<4).allSatisfy({ (0...1).contains(baseColorFactor[$0]) }) else {
            throw Failure.invalidBaseColorFactor
        }
        guard normalScale.isFinite, (0...1).contains(occlusionStrength),
              (0..<3).allSatisfy({ (0...1).contains(emissiveFactor[$0]) }) else {
            throw Failure.invalidSurfaceParameters
        }
        guard emissiveStrength.isFinite, emissiveStrength >= 0 else { throw Failure.invalidEmissiveStrength }
        let physical = shadingModel == .physicallyBased
        let volumeThickness = transmission.map { $0.indexOfRefraction != 0 ? $0.thickness : 0 } ?? 0
        if physical, (subsurface?.weight ?? 0) > 0,
           (diffuseTransmission?.factor ?? 0) > 0 || (transmission?.factor ?? 0) > 0 || volumeScatter != nil {
            throw Failure.subsurfaceConflict
        }
        if physical, volumeScatter != nil, volumeThickness <= 0 {
            throw Failure.volumeScatterWithoutVolume
        }

        let alphaCode: Float
        var cutoff: Float = 0
        switch alphaMode {
        case .materialAlpha: alphaCode = 0
        case .opaque: alphaCode = 1
        case .mask(let value):
            guard value.isFinite, value >= 0 else { throw Failure.invalidAlphaCutoff }
            alphaCode = 2
            cutoff = value
        case .blend: alphaCode = 3
        }

        sourceInputs = SourceInputs(
            metallic: metallic, roughness: roughness, directionToLight: directionToLight,
            lightIntensity: lightIntensity, ambientIntensity: ambientIntensity, baseColorSampler: baseColorSampler,
            metallicRoughnessTexture: metallicRoughnessTexture, normalTexture: normalTexture, normalScale: normalScale,
            occlusionTexture: occlusionTexture, occlusionStrength: occlusionStrength, emissiveTexture: emissiveTexture,
            emissiveFactor: emissiveFactor, baseColorTextureCoordinateSet: baseColorTextureCoordinateSet
        )
        self.baseColorTexture = baseColorTexture
        self.baseColorFactor = baseColorFactor
        self.alphaMode = alphaMode
        let hasVolume = shadingModel == .physicallyBased
            && transmission.map { $0.factor > 0 && $0.indexOfRefraction != 0 && $0.thickness > 0 } ?? false
        self.isDoubleSided = isDoubleSided && !hasVolume
        self.receivesShadows = receivesShadows
        self.toneMapped = toneMapped
        self.wireframe = wireframe
        self.flatShading = flatShading
        self.isVisible = isVisible
        self.fog = fog
        self.dithering = dithering
        self.environment = environment
        self.clearcoat = clearcoat
        self.transmission = transmission
        self.surfaceGraph = surfaceGraph
        let declaredUniforms = surfaceGraph?.uniforms ?? []
        if let surfaceUniforms {
            guard surfaceGraph != nil, surfaceUniforms.matches(declaredUniforms) else { throw Failure.surfaceUniforms }
            self.surfaceUniforms = surfaceUniforms
        } else {
            // A graph that reads the clock needs live values even without declared uniforms.
            let readsClock = surfaceGraph?.program.readsClock ?? false
            self.surfaceUniforms = declaredUniforms.isEmpty && !readsClock ? nil : try ShaderUniforms(declaredUniforms)
        }
        let declaredTextures = surfaceGraph?.textures ?? []
        if surfaceGraph?.outputs.pointSize != nil || surfaceGraph?.outputs.lineWidth != nil {
            throw Failure.pointSizeOutput
        }
        if let extra = surfaceTextures.keys.first(where: { name in !declaredTextures.contains { $0.name == name } }) {
            throw Failure.surfaceTexture(extra)
        }
        self.surfaceTextures = try declaredTextures.map { declaration in
            guard declaration.shape == .twoDimensional else {
                throw Failure.surfaceTexture(declaration.name)
            }
            guard let texture = surfaceTextures[declaration.name] else { throw Failure.surfaceTexture(declaration.name) }
            guard Self.isSurfaceTexture(texture, content: declaration.content) else {
                throw Failure.surfaceTexture(declaration.name)
            }
            // Raster and deformation sample with the declared filter; only some GPUs filter 32-bit floats.
            if declaration.filtering == .linear, MaterialTextureFormat.isFloat32(texture.pixelFormat),
               !texture.device.supports32BitFloatFiltering {
                throw Failure.surfaceTexture(declaration.name)
            }
            return texture
        }
        self.baseColorTextureTransform = baseColorTextureTransform
        self.baseColorSampling = baseColorSampling
        self.emissiveStrength = emissiveStrength
        self.emissiveFactor = emissiveFactor
        self.specular = specular
        self.sheen = sheen
        self.anisotropy = anisotropy
        self.iridescence = iridescence
        self.shadingModel = shadingModel
        self.unlitUsesGraphEmission = unlitUsesGraphEmission
        self.alphaTexture = alphaTexture
        self.bumpMap = bumpMap
        self.diffuseTransmission = diffuseTransmission
        self.diffuseRoughness = diffuseRoughness
        self.volumeScatter = volumeScatter
        self.subsurface = subsurface
        let activeSubsurface = physical && (subsurface?.weight ?? 0) > 0 ? subsurface : nil
        let activeDiffuseTransmission = physical && (diffuseTransmission?.factor ?? 0) > 0 ? diffuseTransmission : nil
        let activeDiffuseRoughness = physical && (diffuseRoughness?.factor ?? 0) > 0 ? diffuseRoughness : nil
        let specularTransmission = physical && (transmission?.factor ?? 0) > 0 && transmission?.indexOfRefraction != 0
        let medium: MaterialMedium?
        if let activeSubsurface {
            let path = activeSubsurface.meanFreePath
            medium = MaterialMedium(
                boundary: .diffuse,
                extinction: SIMD3((0..<3).map { path[$0] > 0 ? min(1 / path[$0], MaterialMedium.maximumExtinction) : MaterialMedium.maximumExtinction }),
                multiscatterColor: activeSubsurface.color, anisotropy: activeSubsurface.anisotropy
            )
        } else if physical, volumeThickness > 0, let transmission, specularTransmission || activeDiffuseTransmission != nil {
            medium = MaterialMedium(
                boundary: specularTransmission ? .dielectric : .diffuse,
                extinction: MaterialMedium.extinction(color: transmission.attenuationColor, distance: transmission.attenuationDistance),
                multiscatterColor: volumeScatter?.multiscatterColor ?? .zero, anisotropy: volumeScatter?.anisotropy ?? 0
            )
        } else {
            medium = nil
        }
        self.medium = medium
        let activeBump = normalTexture == nil && (bumpMap?.scale ?? 0) != 0 ? bumpMap : nil
        var bindings: [TextureBinding?] = [
            baseColorTexture.map {
                TextureBinding(
                    texture: $0, sampler: baseColorSampler, textureCoordinateSet: baseColorTextureCoordinateSet,
                    transform: baseColorTextureTransform, sampling: baseColorSampling
                )
            },
            metallicRoughnessTexture, normalTexture ?? activeBump?.texture, occlusionTexture, emissiveTexture,
        ]
        if let clearcoat, clearcoat.factor > 0 {
            bindings += [clearcoat.factorTexture, clearcoat.roughnessTexture, clearcoat.normalTexture]
        }
        if let transmission, transmission.factor > 0, transmission.indexOfRefraction != 0 {
            if bindings.count == 5 { bindings += [nil, nil, nil] }
            bindings += [transmission.factorTexture, transmission.thicknessTexture]
        }
        self.bindings = bindings
        let layersActive = shadingModel == .physicallyBased
        let activeSheen = layersActive && sheen.map { $0.colorFactor != .zero } == true ? sheen : nil
        let activeAnisotropy = layersActive && (anisotropy?.strength ?? 0) > 0 ? anisotropy : nil
        let activeIridescence = layersActive && (iridescence?.factor ?? 0) > 0 ? iridescence : nil
        let activeSpecular = layersActive ? specular : nil
        let scatterTexture = activeSubsurface?.colorTexture ?? (medium == nil ? nil : volumeScatter?.multiscatterColorTexture)
        let layerBindings: [TextureBinding?] = [
            activeSpecular?.factorTexture, activeSpecular?.colorTexture,
            activeSheen?.colorTexture, activeSheen?.roughnessTexture, activeAnisotropy?.texture,
            activeIridescence?.factorTexture, activeIridescence?.thicknessTexture,
            activeSubsurface == nil ? activeDiffuseTransmission?.factorTexture : nil,
            activeSubsurface == nil ? activeDiffuseTransmission?.colorTexture : nil,
            activeDiffuseRoughness?.texture, scatterTexture,
        ]
        self.layerBindings = layerBindings
        let allBindings = bindings + Array(repeating: nil, count: 10 - bindings.count) + layerBindings
        let activeLightMap = shadingModel == .physicallyBased && lightMapIntensity > 0 ? lightMap : nil
        textureCoordinateMask = allBindings.enumerated().reduce((activeLightMap?.textureCoordinateSet.rawValue ?? 0) << 21) {
            mask, binding in mask | ((binding.element?.textureCoordinateSet.rawValue ?? 0) << binding.offset)
        }
        controls = SIMD4(
            normalTexture == nil ? activeBump?.scale ?? 0 : normalScale,
            occlusionTexture == nil ? 0 : occlusionStrength,
            cutoff,
            metallicRoughnessTexture == nil ? 0 : 1
        )
        emission = SIMD4(emissiveFactor * emissiveStrength, emissiveTexture == nil ? 0 : 1)
        var transforms = InlineArray<44, SIMD4<Float>>(repeating: .zero)
        for (index, binding) in allBindings.enumerated() {
            let rows = (binding?.transform ?? .identity).rows
            transforms[2 * index] = rows.0
            transforms[2 * index + 1] = rows.1
        }
        (transforms[42], transforms[43]) = (activeLightMap?.transform ?? .identity).rows
        let rotation = activeAnisotropy?.rotation ?? 0
        let unlitCode: Float = shadingModel == .physicallyBased ? 0 : (unlitUsesGraphEmission ? 2 : 1)
        var extensions = PackedExtensions(
            specular: SIMD4(activeSpecular?.colorFactor ?? SIMD3(repeating: 1), activeSpecular?.factor ?? 1),
            sheen: SIMD4(activeSheen?.colorFactor ?? .zero, activeSheen?.roughness ?? 0),
            anisotropy: SIMD4(activeAnisotropy?.strength ?? 0, cos(rotation), sin(rotation), flatShading ? 1 : 0),
            iridescence: SIMD4(
                activeIridescence?.factor ?? 0, activeIridescence?.indexOfRefraction ?? 1.3,
                activeIridescence?.minimumThickness ?? 100, activeIridescence?.maximumThickness ?? 400
            ),
            options: SIMD4(
                unlitCode, activeLightMap == nil ? 0 : lightMapIntensity,
                shadingModel == .physicallyBased ? retroreflectivity : 0, 0
            ),
            maps: SIMD4(
                layerBindings.enumerated().reduce(0) { $0 | ($1.element == nil ? 0 : 1 << UInt32($1.offset)) }, 0, 0, 0
            ),
            transforms: transforms,
            translucency: .zero, scattering: .zero, attenuation: .zero,
            diffusion: SIMD4(0, 0, 0, activeDiffuseRoughness?.factor ?? 0)
        )
        if let activeSubsurface {
            extensions.translucency = SIMD4(1, 1, 1, activeSubsurface.weight)
        } else if let activeDiffuseTransmission {
            extensions.translucency = SIMD4(activeDiffuseTransmission.colorFactor, activeDiffuseTransmission.factor)
        }
        if extensions.translucency.w > 0 {
            if let medium {
                // Raster treats the diffusion length as zero (no wrap) where nothing is absorbed or scattered.
                let length = medium.diffusionLength
                let scattering = simd_reduce_max(medium.multiscatterColor) > 0
                extensions.scattering = SIMD4(medium.multiscatterColor, 2)
                extensions.attenuation = SIMD4(medium.rasterAttenuation, volumeThickness)
                extensions.diffusion = SIMD4(
                    scattering ? SIMD3((0..<3).map { length[$0].isFinite ? length[$0] : 1e6 }) : .zero,
                    extensions.diffusion.w
                )
            } else {
                extensions.scattering = SIMD4(0, 0, 0, 1)
            }
        } else if let medium, volumeScatter != nil {
            // Volume scattering behind specular transmission hazes the raster backdrop.
            extensions.scattering = SIMD4(medium.multiscatterColor, 0)
        }
        packedExtensions = extensions
        // Scale before normalizing so finite vectors near Float's limits do not overflow or underflow.
        lightParameters = SIMD4(simd_normalize(directionToLight / scale), lightIntensity)
        settings = SIMD4(roughness, metallic, ambientIntensity, alphaCode)
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidMetallic
        case invalidRoughness
        case invalidDirection
        case invalidLightIntensity
        case invalidAmbientIntensity
        case invalidLightMap
        case invalidLightMapIntensity
        case invalidRetroreflectivity
        case invalidBaseColorFactor
        case invalidSurfaceParameters
        case invalidAlphaCutoff
        case invalidDataTexture
        case invalidPackedNormalTexture
        case invalidBumpScale
        case invalidBumpTexture
        case invalidEnvironment
        case invalidClearcoat
        case invalidTransmission
        case invalidIndexOfRefraction
        case surfaceUniforms
        case unlitGraphEmission
        case surfaceTexture(String)
        case pointSizeOutput
        case invalidTextureTransform
        case invalidEmissiveStrength
        case invalidSpecular
        case invalidSheen
        case invalidAnisotropy
        case invalidIridescence
        case invalidDispersion
        case layerSamplerLimit
        case alphaTextureSampler
        case invalidDiffuseTransmission
        case invalidDiffuseRoughness
        case invalidVolumeScatter
        case volumeScatterWithoutVolume
        case invalidSubsurface
        case subsurfaceConflict

        public var errorDescription: String? {
            switch self {
            case .invalidMetallic: "Metallic must be between zero and one."
            case .invalidRoughness: "Roughness must be between zero and one."
            case .invalidDirection: "The direction to the light must be finite and nonzero."
            case .invalidLightIntensity: "The light intensity must be finite and nonnegative."
            case .invalidAmbientIntensity: "The ambient intensity must be finite and nonnegative."
            case .invalidLightMap:
                "A light map needs a shader-readable linear 2D RGB texture."
            case .invalidLightMapIntensity: "Light-map intensity must be finite and nonnegative."
            case .invalidRetroreflectivity: "Retroreflectivity must be between zero and one."
            case .invalidBaseColorFactor: "Base color factors must be between zero and one."
            case .invalidSurfaceParameters: "Normal scale must be finite. Occlusion strength and emissive factors must be between zero and one."
            case .invalidAlphaCutoff: "The alpha mask cutoff must be finite and nonnegative."
            case .invalidPackedNormalTexture:
                "A packed normal map needs a linear texture with two channels and matching signedness."
            case .invalidBumpScale: "Bump scale must be finite."
            case .invalidBumpTexture: "Use a shader-readable, linear 2D height texture. Use normalized or 16-bit floating-point data."
            case .invalidDataTexture: "Use a shader-readable linear texture with the channels required by the PBR data map."
            case .invalidEnvironment: "Environment intensity must be finite and nonnegative, and its rotation must be finite."
            case .invalidClearcoat: "Clearcoat factor and roughness must be in zero to one, and normal scale must be finite."
            case .invalidTransmission: "Use a transmission factor and attenuation color in zero to one, finite nonnegative thickness, and positive attenuation distance."
            case .invalidIndexOfRefraction: "Use an index of refraction of zero or a finite value of at least one."
            case .surfaceUniforms: "Surface uniforms must match the surface graph's declared names and types."
            case .unlitGraphEmission: "Graph emission as color requires unlit shading and a connected emission output."
            case .surfaceTexture(let name):
                "Bind each declared surface graph texture exactly once with a shader-readable 2D texture. Color inputs accept RGBA8, BGRA8, sRGB, float, or RGBA block-compressed formats; data inputs use linear formats. Linear filtering of 32-bit float textures needs a GPU that supports it. Problem: \(name)."
            case .pointSizeOutput: "Point size and line width outputs belong on point and line materials."
            case .invalidTextureTransform: "Texture transforms need a finite offset and rotation and finite, nonzero scale components."
            case .invalidEmissiveStrength: "Emissive strength must be finite and nonnegative."
            case .invalidSpecular: "Specular factor must be between zero and one, and specular color components finite and nonnegative."
            case .invalidSheen: "Sheen color components and roughness must be between zero and one."
            case .invalidAnisotropy: "Anisotropy strength must be between zero and one, and its rotation finite."
            case .invalidIridescence: "Iridescence factor must be between zero and one, its index of refraction finite and at least one, and its thicknesses finite and nonnegative."
            case .invalidDispersion: "Dispersion must be finite and nonnegative."
            case .layerSamplerLimit:
                "The specular, sheen, anisotropy, iridescence, diffuse transmission, diffuse roughness and scattering maps of one material can use at most six distinct samplers."
            case .invalidDiffuseTransmission: "Diffuse transmission factor and color components must be between zero and one."
            case .invalidDiffuseRoughness: "Diffuse roughness must be between zero and one."
            case .invalidVolumeScatter:
                "Multiple-scattering color components must be between zero and one, and scatter anisotropy greater than -1 and less than 1."
            case .volumeScatterWithoutVolume:
                "Volume scattering needs a transmission layer with nonzero thickness and a finite index of refraction of at least one."
            case .invalidSubsurface:
                "Subsurface weight and color components must be between zero and one, radius components and scale finite and nonnegative, and anisotropy greater than -1 and less than 1."
            case .subsurfaceConflict:
                "Subsurface scattering defines the material's volume; use it without specular transmission, diffuse transmission or volume scattering."
            case .alphaTextureSampler: "With a base color texture, the alpha map samples with the base color sampler; leave its sampler nil or pass that sampler."
            }
        }
    }

    static func isSurfaceTexture(_ texture: any MTLTexture, content: ShaderTextureInput.Content) -> Bool {
        guard texture.textureType == .type2D, !texture.isFramebufferOnly, texture.sampleCount == 1,
              texture.usage.isEmpty || texture.usage.contains(.shaderRead) else { return false }
        switch texture.pixelFormat {
        case .rgba8Unorm, .bgra8Unorm, .rgba16Float, .rgba32Float: return true
        case .rgba8Unorm_srgb, .bgra8Unorm_srgb: return content == .color
        case .r8Unorm, .r16Float, .r32Float, .rg8Unorm, .rg16Float, .rg32Float: return content == .data
        case let format:
            // Block-compressed RGBA samples like RGBA8; sRGB variants carry color only.
            let encoding = MaterialTextureFormat.compressedEncoding(format)
            return encoding.isCompressed && (!encoding.isSRGB || content == .color)
        }
    }

    struct Prepared {
        private let textures: [any MTLTexture]
        private let samplers: [any MTLSamplerState]
        private let bindingSlots: [Int]
        private let material: PhysicallyBasedMaterial

        init(material: PhysicallyBasedMaterial, device: any MTLDevice) throws {
            self.material = material
            if let environment = material.environment, environment.lighting.device !== device {
                throw MeshRenderer.Failure.wrongDevice
            }
            let sampler = MTLSamplerDescriptor()
            sampler.minFilter = .linear
            sampler.magFilter = .linear
            sampler.mipFilter = .linear
            sampler.sAddressMode = .repeat
            sampler.tAddressMode = .repeat
            guard let defaultSampler = device.makeSamplerState(descriptor: sampler) else {
                throw MeshRenderer.Failure.resource("PBR sampler")
            }
            // A binding's own sampler wins; otherwise equal descriptions share one sampler state.
            var describedSamplers: [TextureSampling: any MTLSamplerState] = [:]
            func samplerState(for binding: TextureBinding) throws -> any MTLSamplerState {
                if let sampler = binding.sampler { return sampler }
                guard let sampling = binding.sampling, sampling != .default else { return defaultSampler }
                if let existing = describedSamplers[sampling] { return existing }
                let created = try sampling.makeSamplerState(device: device)
                describedSamplers[sampling] = created
                return created
            }

            var fallback: (any MTLTexture)?
            var textures: [any MTLTexture] = []
            var samplers: [any MTLSamplerState] = []
            var bindingSlots: [Int] = []
            for (index, binding) in material.bindings.enumerated() {
                if (5...7).contains(index), !material.hasClearcoat { continue }
                bindingSlots.append(index)
                if let binding {
                    let texture = binding.texture
                    guard texture.device === device, binding.sampler?.device === device || binding.sampler == nil else {
                        throw MeshRenderer.Failure.wrongDevice
                    }
                    let isColor = index == 0 || index == 4
                    let invalidTexture: any Error = isColor
                        ? MeshRenderer.Failure.colorTexture : Failure.invalidDataTexture
                    // BumpMap already validates the effective height binding.
                    let isBumpMap = index == 2 && material.hasActiveBumpMap
                    if !isBumpMap, (index == 2 || index == 7), binding.normalEncoding != .rgb {
                        guard binding.normalEncoding.acceptsPackedTexture(texture.pixelFormat) else {
                            throw Failure.invalidPackedNormalTexture
                        }
                    } else if !isBumpMap {
                        switch texture.pixelFormat {
                        case .rgba8Unorm, .bgra8Unorm, .rgba16Float: break
                        case .r8Unorm, .r16Float:
                            guard index == 5 || index == 8 else { throw invalidTexture }
                        case .rg8Unorm, .rg16Float:
                            guard index == 5 || index == 6 || index == 8 || index == 9 else { throw invalidTexture }
                        case .rgba8Unorm_srgb, .bgra8Unorm_srgb:
                            guard isColor else { throw invalidTexture }
                        case let format:
                            // Block-compressed RGBA maps follow the RGBA8 rules for color and data.
                            let encoding = MaterialTextureFormat.compressedEncoding(format)
                            guard encoding.isCompressed, isColor || !encoding.isSRGB else { throw invalidTexture }
                        }
                    }
                    guard texture.textureType == .type2D, !texture.isFramebufferOnly,
                          texture.usage.isEmpty || texture.usage.contains(.shaderRead) else {
                        throw invalidTexture
                    }
                    textures.append(texture)
                    samplers.append(try samplerState(for: binding))
                } else {
                    let white = try fallback ?? Self.makeWhiteTexture(device: device)
                    fallback = white
                    textures.append(white)
                    // Without a base color map, slot 0's sampler serves the alpha map.
                    let alphaSampler = index == 0 ? material.alphaTexture?.sampler : nil
                    if let alphaSampler, alphaSampler.device !== device { throw MeshRenderer.Failure.wrongDevice }
                    samplers.append(alphaSampler ?? defaultSampler)
                }
            }
            self.textures = textures
            self.samplers = samplers
            self.bindingSlots = bindingSlots
            if material.surfaceGraph != nil {
                guard material.surfaceTextures.allSatisfy({ $0.device === device }) else { throw MeshRenderer.Failure.wrongDevice }
                let filler = try fallback ?? Self.makeWhiteTexture(device: device)
                graphTextures = material.surfaceTextures
                    + Array(repeating: filler, count: MaterialGraph.maximumTextureCount - material.surfaceTextures.count)
            } else {
                graphTextures = []
            }

            // Layer and light maps share the texture-handle buffer at fragment slot 22.
            // The optional optical layers use sampler slots 10 to 15.
            var extensions = material.packedExtensions
            if material.hasLayers || material.hasLightMap {
                var layerTextures: [any MTLTexture] = []
                var layerSamplers: [any MTLSamplerState] = []
                var samplerSlots: UInt32 = 0
                var laterSamplerSlots: UInt32 = 0
                for (index, binding) in material.layerBindings.enumerated() {
                    guard let binding else {
                        let white = try fallback ?? Self.makeWhiteTexture(device: device)
                        fallback = white
                        layerTextures.append(white)
                        continue
                    }
                    let texture = binding.texture
                    guard texture.device === device, binding.sampler?.device === device || binding.sampler == nil else {
                        throw MeshRenderer.Failure.wrongDevice
                    }
                    try Self.validateLayerTexture(texture, map: index)
                    layerTextures.append(texture)
                    let sampler = try samplerState(for: binding)
                    let slot: Int
                    if let existing = layerSamplers.firstIndex(where: { $0 === sampler }) {
                        slot = existing
                    } else {
                        guard layerSamplers.count < 6 else { throw Failure.layerSamplerLimit }
                        slot = layerSamplers.count
                        layerSamplers.append(sampler)
                    }
                    if index < 7 {
                        samplerSlots |= UInt32(slot) << UInt32(3 * index)
                    } else {
                        laterSamplerSlots |= UInt32(slot) << UInt32(3 * (index - 7))
                    }
                }
                if material.hasLightMap, let lightMap = material.lightMap {
                    guard lightMap.texture.device === device else { throw MeshRenderer.Failure.wrongDevice }
                    layerTextures.append(lightMap.texture)
                } else {
                    layerTextures.append(try fallback ?? Self.makeWhiteTexture(device: device))
                }
                extensions.maps.y = samplerSlots
                extensions.maps.z = laterSamplerSlots
                // Ask Metal to use the device's layout, including Tier 1's private layout.
                let textureArray = MTLArgumentDescriptor()
                textureArray.index = 0
                textureArray.dataType = .texture
                textureArray.textureType = .type2D
                textureArray.arrayLength = layerTextures.count
                guard let argumentEncoder = device.makeArgumentEncoder(arguments: [textureArray]),
                      let arguments = device.makeBuffer(length: argumentEncoder.encodedLength, options: .storageModeShared) else {
                    throw MeshRenderer.Failure.resource("PBR layer map handles")
                }
                arguments.label = "Frostfield PBR layer maps"
                argumentEncoder.setArgumentBuffer(arguments, offset: 0)
                argumentEncoder.setTextures(layerTextures, range: 0..<layerTextures.count)
                layerArguments = arguments
                self.layerTextures = layerTextures
                self.layerSamplers = layerSamplers + Array(repeating: defaultSampler, count: 6 - layerSamplers.count)
            } else {
                layerArguments = nil
                layerTextures = []
                layerSamplers = []
            }
            self.extensions = extensions
            if let alpha = material.alphaTexture {
                let texture = alpha.texture
                guard texture.device === device else { throw MeshRenderer.Failure.wrongDevice }
                guard texture.textureType == .type2D, !texture.isFramebufferOnly,
                      texture.usage.isEmpty || texture.usage.contains(.shaderRead) else {
                    throw Failure.invalidDataTexture
                }
                switch texture.pixelFormat {
                case .rgba8Unorm, .bgra8Unorm, .rg8Unorm, .rgba16Float, .rg16Float: break
                case let format:
                    let encoding = MaterialTextureFormat.compressedEncoding(format)
                    guard encoding.isCompressed, !encoding.isSRGB else { throw Failure.invalidDataTexture }
                }
                let rows = alpha.transform.rows
                alphaInfo = [rows.0, rows.1, SIMD4(Float(alpha.textureCoordinateSet.rawValue), 0, 0, 0)]
            } else {
                alphaInfo = []
            }
        }

        /// The alpha map's two transform rows and its UV set, for fragment buffer 23.
        private let alphaInfo: [SIMD4<Float>]
        private let graphTextures: [any MTLTexture]
        private let extensions: PackedExtensions
        private let layerArguments: (any MTLBuffer)?
        private let layerTextures: [any MTLTexture]
        private let layerSamplers: [any MTLSamplerState]

        func recordResources(in inventory: inout SceneResourceInventory) {
            for texture in textures { inventory.recordTexture(texture) }
            for texture in layerTextures { inventory.recordTexture(texture) }
            for texture in graphTextures { inventory.recordTexture(texture) }
            inventory.recordBuffer(layerArguments)
            inventory.recordTexture(material.alphaTexture?.texture)
            if let lighting = material.environment?.lighting {
                inventory.recordTexture(lighting.skyCube)
                inventory.recordTexture(lighting.diffuseCube)
                inventory.recordTexture(lighting.specularCube)
                inventory.recordTexture(lighting.brdfLUT)
            }
        }

        /// Validates a specular, sheen, anisotropy, iridescence, diffuse transmission, diffuse
        /// roughness or scattering map by its sampled channels.
        private static func validateLayerTexture(_ texture: any MTLTexture, map: Int) throws {
            let isColor = map == 1 || map == 2 || map == 8 || map == 10
            let invalid: any Error = isColor ? MeshRenderer.Failure.colorTexture : Failure.invalidDataTexture
            guard texture.textureType == .type2D, !texture.isFramebufferOnly,
                  texture.usage.isEmpty || texture.usage.contains(.shaderRead) else { throw invalid }
            switch texture.pixelFormat {
            case .rgba8Unorm, .bgra8Unorm, .rgba16Float: break
            case .rgba8Unorm_srgb, .bgra8Unorm_srgb: guard isColor else { throw invalid }
            case .r8Unorm, .r16Float: guard map == 5 || map == 9 else { throw invalid }
            case .rg8Unorm, .rg16Float: guard map == 5 || map == 6 || map == 9 else { throw invalid }
            default: throw invalid
            }
        }

        private func encodeTexture(at index: Int, slot: Int, into encoder: any MTLRenderCommandEncoder) {
            let textureIndex = switch slot {
            case 0: 0
            case 1...4: slot + 1
            case 5...8: slot + 4
            default: 14
            }
            encoder.setFragmentTexture(textures[index], index: textureIndex)
            encoder.setFragmentSamplerState(samplers[index], index: slot)
        }

        func encode(
            into encoder: any MTLRenderCommandEncoder, sceneEnvironment: RasterSceneEnvironment,
            object: ShaderObjectContext? = nil
        ) {
            for (index, slot) in bindingSlots.enumerated() {
                encodeTexture(at: index, slot: slot, into: encoder)
            }
            // Live factors replace the values copied into this draw; bindings stay as prepared.
            let material = self.material.current
            var controls = material.controls
            var emission = material.emission
            var factor = material.baseColorFactor
            // Low bits choose UV sets. Normal sources occupy bits 24–27, spaces bits 28–29.
            var coordinateMask = material.textureCoordinateMask | (material.baseNormalKind << 24)
                | (material.coatNormalEncoding.rawValue << 26) | (material.objectNormalMaps << 28)
            var extensions = self.material.liveFactors == nil ? extensions : material.packedExtensions
            extensions.maps = self.extensions.maps
            if let alpha = material.alphaTexture {
                encoder.setFragmentTexture(alpha.texture, index: 15)
                alphaInfo.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 23) }
            }
            encoder.setFragmentBytes(&controls, length: MemoryLayout<SIMD4<Float>>.size, index: 6)
            encoder.setFragmentBytes(&emission, length: MemoryLayout<SIMD4<Float>>.size, index: 7)
            encoder.setFragmentBytes(&factor, length: MemoryLayout<SIMD4<Float>>.size, index: 8)
            encoder.setFragmentBytes(&coordinateMask, length: MemoryLayout<UInt32>.size, index: 9)
            encoder.setFragmentBytes(&extensions, length: MemoryLayout<PackedExtensions>.stride, index: 21)
            if let layerArguments {
                encoder.setFragmentBuffer(layerArguments, offset: 0, index: 22)
                encoder.useResources(layerTextures, usage: .read, stages: .fragment)
                if self.material.hasLayers { encoder.setFragmentSamplerStates(layerSamplers, range: 10..<16) }
            }
            if let (lighting, lookup) = sceneEnvironment.resolve(material) {
                var uniforms = lookup
                encoder.setFragmentTexture(lighting.diffuseCube, index: 6)
                encoder.setFragmentTexture(lighting.specularCube, index: 7)
                encoder.setFragmentTexture(lighting.brdfLUT, index: 8)
                encoder.setFragmentBytes(&uniforms, length: MemoryLayout<RasterEnvironmentLookup>.size, index: 10)
            }
            if let clearcoat = material.clearcoat, self.material.hasClearcoat {
                var layer = SIMD4(
                    clearcoat.factor, clearcoat.roughness,
                    clearcoat.normalTexture == nil ? 0 : clearcoat.normalScale,
                    material.sourceInputs.normalTexture?.normalSpace == .tangentSpace ? 1 : 0
                )
                encoder.setFragmentBytes(&layer, length: MemoryLayout<SIMD4<Float>>.size, index: 13)
            }
            let transmission = material.transmission
            var optical = SIMD4(
                transmission?.factor ?? 0, transmission?.indexOfRefraction ?? 1.5,
                transmission?.thickness ?? 0, transmission?.attenuationDistance ?? 0
            )
            encoder.setFragmentBytes(&optical, length: MemoryLayout<SIMD4<Float>>.size, index: 14)
            if let transmission, self.material.hasTransmission {
                var attenuation = SIMD4(transmission.attenuationColor, transmission.dispersion)
                encoder.setFragmentBytes(&attenuation, length: MemoryLayout<SIMD4<Float>>.size, index: 15)
            }
            if material.surfaceGraph != nil {
                // Copy the live uniform values into this draw. A graph without uniforms binds one unused slot.
                let values = material.surfaceUniforms?.bufferSnapshot(object: object).values ?? []
                if values.isEmpty {
                    var unused = SIMD4<Float>.zero
                    encoder.setFragmentBytes(&unused, length: MemoryLayout<SIMD4<Float>>.stride, index: 19)
                } else {
                    values.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 19) }
                }
                encoder.setFragmentTextures(graphTextures, range: 16..<(16 + graphTextures.count))
            }
        }

        /// Matches FrostfieldTexturedTransmissionShadow in Shadows.metal.
        struct TransmissionShadow {
            var tint: SIMD4<Float>
            var optical: SIMD4<Float>
            var attenuation: SIMD4<Float>
            var coverage: SIMD4<Float>
            var transforms: InlineArray<10, SIMD4<Float>> = .init(repeating: .zero)
        }

        private static let transmissionShadowSlots = [0, 1, 8, 9]

        func transmissionShadow(
            material: PhysicallyBasedMaterial, optical: Transmission, tint: SIMD4<Float>, opticalDepth: Float
        ) -> TransmissionShadow {
            var packed = TransmissionShadow(
                tint: tint, optical: SIMD4(optical.factor, material.settings.y, opticalDepth, 0),
                attenuation: SIMD4(optical.attenuationColor, 0),
                coverage: SIMD4(material.settings.w, material.controls.z, alphaInfo.isEmpty ? 0 : 1, 0)
            )
            for (index, slot) in Self.transmissionShadowSlots.enumerated() {
                let row = slot * 2
                var first = material.packedExtensions.transforms[row]
                first.w = Float((material.textureCoordinateMask >> slot) & 1)
                packed.transforms[2 * index] = first
                packed.transforms[2 * index + 1] = material.packedExtensions.transforms[row + 1]
            }
            if !alphaInfo.isEmpty {
                packed.transforms[8] = alphaInfo[0]
                packed.transforms[8].w = alphaInfo[2].x
                packed.transforms[9] = alphaInfo[1]
            }
            return packed
        }

        func encodeTransmissionShadow(_ materialInputs: TransmissionShadow, into encoder: any MTLRenderCommandEncoder) {
            var inputs = materialInputs
            encoder.setFragmentBytes(&inputs, length: MemoryLayout<TransmissionShadow>.stride, index: 0)
            for (index, slot) in bindingSlots.enumerated() where Self.transmissionShadowSlots.contains(slot) {
                encodeTexture(at: index, slot: slot, into: encoder)
            }
            encoder.setFragmentTexture(material.alphaTexture?.texture ?? textures[0], index: 15)
        }

        func encodeShadowMask(tintAlpha: Float, cutoff: Float, into encoder: any MTLRenderCommandEncoder) {
            let material = self.material.current
            let drawAlpha = material.surfaceGraph == nil ? tintAlpha : 1
            var coverage = SIMD4(
                min(max(drawAlpha * material.baseColorFactor.w, 0), 1), cutoff,
                Float(material.textureCoordinateMask & 1), 0
            )
            var transform = [material.packedExtensions.transforms[0], material.packedExtensions.transforms[1]]
            encoder.setFragmentTexture(textures[0], index: 0)
            encoder.setFragmentSamplerState(samplers[0], index: 0)
            encoder.setFragmentBytes(&coverage, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            encoder.setFragmentBytes(&transform, length: 2 * MemoryLayout<SIMD4<Float>>.stride, index: 1)
        }

        static func makeWhiteTexture(device: any MTLDevice) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("white PBR texture")
            }
            var white: UInt32 = .max
            withUnsafePointer(to: &white) {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0,
                    withBytes: $0, bytesPerRow: MemoryLayout<UInt32>.size
                )
            }
            return texture
        }
    }
}

/// Splices FrostfieldMaterialExtensions.h into shader sources compiled at run time,
/// where the bundled headers' relative #include lines do not resolve.
enum MaterialExtensionsSource {
    static let includeLine = "#include \"FrostfieldMaterialExtensions.h\""

    static func splice(into source: String) throws -> String {
        guard let url = Bundle.module.url(forResource: "FrostfieldMaterialExtensions", withExtension: "h"),
              let header = try? String(contentsOf: url, encoding: .utf8),
              source.contains(includeLine) else {
            throw MeshRenderer.Failure.library("The package is missing its material extension shader source.")
        }
        return source.replacingOccurrences(of: includeLine, with: header)
    }
}
