import Foundation
import Metal
import simd

/// The non-PBR material models of Three.js: `MeshBasicMaterial`, `MeshLambertMaterial`,
/// `MeshPhongMaterial`, `MeshToonMaterial`, `MeshMatcapMaterial`, `MeshNormalMaterial`,
/// `MeshDepthMaterial`, `MeshDistanceMaterial` and `ShadowMaterial`. Draw one with
/// `MeshMaterial.classic`.
///
/// Every model multiplies `color` and `opacity` by the draw's material parameters,
/// an instance tint and vertex colors. Lit models (Lambert, Phong, toon) receive
/// the scene graph's directional, point and spot lights with their shadows, the
/// directional shadow map's light when `receivesShadows` is set, and `ambient`
/// irradiance. As in Three.js, Lambertian terms divide irradiance by π, so a white
/// surface under a light of π lux at normal incidence reflects one. Rectangle lights
/// are ignored, as Three.js does for these models. The path tracer does not trace
/// classic materials.
///
/// With a render state that does not blend (and no alpha to coverage) the fragment
/// writes opaque color and uses alpha only for `alphaTest` and hashing, like
/// Three.js with `transparent` false. A blending state receives premultiplied color.
/// Explicit depth packing and red distance output write straight data channels.
public struct ClassicMaterial {
    public typealias TextureBinding = PhysicallyBasedMaterial.TextureBinding
    public typealias BumpMap = PhysicallyBasedMaterial.BumpMap

    public enum Model {
        /// Unlit color: `color × map × light map / π × occlusion`, then the
        /// environment map. Like `MeshBasicMaterial`, the light map stores irradiance.
        case basic
        /// Lambertian diffuse, like `MeshLambertMaterial`.
        case lambert
        /// Lambert diffuse plus the Blinn-Phong lobe of `MeshPhongMaterial`:
        /// `F_Schlick(specular, VoH) × ¼ × (shininess / 2 + 1) / π × NoHˢʰⁱⁿⁱⁿᵉˢˢ`
        /// per unit irradiance. `specular` is linear RGB; the default is Three.js's
        /// `0x111111` decoded from sRGB, as its color management does. Shininess is
        /// finite and nonnegative.
        case phong(specular: SIMD3<Float> = ClassicMaterial.defaultPhongSpecular, shininess: Float = 30)
        /// Lambert diffuse whose irradiance factor N·L is replaced by the gradient
        /// map's red channel at U = N·L × 0.5 + 0.5, like `MeshToonMaterial`. Use
        /// nearest filtering and clamped addressing for hard bands. Without a map,
        /// N·L below 0.4 gives 0.7 and above gives 1, with a pixel-wide blend.
        case toon(gradient: TextureBinding? = nil)
        /// `color × map × matcap(view normal)`, like `MeshMatcapMaterial`. The view
        /// normal's X and Y, measured in a frame that faces the view direction, map
        /// to U = 0.5 + 0.495 x and V = 0.5 − 0.495 y in the top-left texture.
        case matcap(TextureBinding)
        /// The view-space normal packed as `n × 0.5 + 0.5`, like `MeshNormalMaterial`.
        /// Front faces toward the camera have positive Z.
        case normal
        /// `1 − z` of the forward-depth window position (`z` with reversed depth), so
        /// near is bright. Set `depthPacking` for the exact Three.js r187 output.
        case depth
        /// `(|p − reference| − near) / (far − near)`, clamped, in every color channel,
        /// unless `distanceOutput` selects red. Requires `0 ≤ near < far`.
        case distance(reference: SIMD3<Float>, near: Float, far: Float)
        /// Writes `color` with alpha `opacity × (1 − visibility)`, so only shadows show,
        /// like `ShadowMaterial`. Visibility multiplies the directional shadow map (with
        /// `receivesShadows`) and every shadowed punctual light. Blends by default.
        case shadow
    }

    /// Three.js r187 depth material encodings of the raster window depth.
    /// Use a single-sample linear target with blending disabled for these modes.
    public enum DepthPacking: UInt32, CaseIterable, Codable, Sendable {
        /// Inverted window depth in RGB and the authored material opacity in alpha.
        case basic = 3200
        /// Base-256 packing in RGBA, most significant channel first.
        case rgba = 3201
        /// Base-256 packing in RGB, with alpha one.
        case rgb = 3202
        /// Base-256 packing in RG, with blue zero and alpha one.
        case rg = 3203
    }

    /// The normalized world-distance material's color output.
    public enum DistanceOutput: String, Codable, Sendable {
        /// The distance in each RGB channel, the default Frostfield output.
        case grayscale
        /// The r187 layout with distance in red, green and blue zero, and alpha one.
        case red
    }

    /// How the environment sample combines with the shaded color, like Three.js `combine`.
    public enum EnvironmentBlend: Sendable {
        /// `mix(c, c × env, reflectivity)`.
        case multiply
        /// `mix(c, env, reflectivity)`.
        case mix
        /// `c + env × reflectivity`.
        case add
    }

    /// The world-space direction used to look up the environment cube.
    public enum EnvironmentMapping: Equatable, Sendable {
        /// Mirrors the incoming camera direction about the shading normal.
        case reflection
        /// Bends the incoming camera direction by the ratio of air to material IOR.
        /// The ratio is between zero and one, inclusive. This samples only the
        /// environment cube, like Three.js `CubeRefractionMapping`.
        case refraction(ratio: Float = 0.98)

        var refractionRatio: Float? {
            if case .refraction(let ratio) = self { ratio } else { nil }
        }
    }

    /// A classic `envMap` for Basic, Lambert or Phong materials.
    /// Directions use the same cube orientation and yaw as `EnvironmentLighting`.
    public struct EnvironmentMap {
        public let texture: any MTLTexture
        public let blend: EnvironmentBlend
        public let reflectivity: Float
        public let intensity: Float
        public let yaw: Float
        public let mapping: EnvironmentMapping

        /// Use a shader-readable cube texture. Reflectivity is zero to one; intensity is
        /// finite and nonnegative.
        public init(
            texture: any MTLTexture, blend: EnvironmentBlend = .multiply, reflectivity: Float = 1,
            intensity: Float = 1, yaw: Float = 0, mapping: EnvironmentMapping = .reflection
        ) throws {
            guard texture.textureType == .typeCube, !texture.isFramebufferOnly,
                  texture.usage.isEmpty || texture.usage.contains(.shaderRead) else {
                throw Failure.environmentTexture
            }
            guard (0...1).contains(reflectivity), intensity.isFinite, intensity >= 0, yaw.isFinite else {
                throw Failure.invalidValue("environment reflectivity, intensity or yaw")
            }
            if let ratio = mapping.refractionRatio, !(0...1).contains(ratio) {
                throw Failure.invalidValue("refractionRatio")
            }
            self.texture = texture
            self.blend = blend
            self.reflectivity = reflectivity
            self.intensity = intensity
            self.yaw = yaw
            self.mapping = mapping
        }
    }

    /// The linear value of Three.js `MeshPhongMaterial.specular` (`0x111111`, an sRGB hex color).
    public static let defaultPhongSpecular = SIMD3<Float>(repeating: 0.005_605_392)

    public let model: Model
    /// Linear RGB, finite and nonnegative.
    public let color: SIMD3<Float>
    /// Zero to one.
    public let opacity: Float
    /// Color texture; RGB multiplies color and alpha multiplies opacity. sRGB formats decode when sampled.
    public let map: TextureBinding?
    /// The linear green channel multiplies opacity, like Three.js `alphaMap`. Needs a
    /// format with a green channel, so not R8, BC4 or EAC R11.
    public let alphaMap: TextureBinding?
    /// Discards fragments whose alpha is below this value; zero disables the test.
    public let alphaTest: Float
    /// The red channel attenuates indirect light: `(r − 1) × intensity + 1`.
    public let aoMap: TextureBinding?
    public let aoMapIntensity: Float
    /// Baked irradiance in linear RGB, scaled by its intensity.
    public let lightMap: TextureBinding?
    public let lightMapIntensity: Float
    /// Emitted linear radiance, scaled by the emissive map's RGB when present.
    public let emissive: SIMD3<Float>
    public let emissiveMap: TextureBinding?
    public let environmentMap: EnvironmentMap?
    /// Constant irradiance for lit models, like a Three.js `AmbientLight` color times intensity.
    public let ambient: SIMD3<Float>
    /// Uses the scene graph's directional shadow map and its light. Requires a shadow map when drawn.
    public let receivesShadows: Bool
    /// When false, a display pass that receives this material's exclusion mask copies the
    /// scene color without exposure or a tone curve. Defaults to true.
    public let toneMapped: Bool
    /// Draws each triangle's edges instead of its face. Defaults to false.
    public let wireframe: Bool
    /// When false, the material is not drawn and does not cast a shadow. Picking still hits it.
    /// Defaults to true, like Three.js `Material.visible`.
    public let isVisible: Bool
    /// When false, a fog pass that samples this material's exclusion mask leaves the pixel unchanged.
    /// Defaults to true, like Three.js `Material.fog`.
    public let fog: Bool
    /// Adds the r187 material dither when this mesh is drawn into an 8-bit color target.
    /// A float target leaves the color unchanged. Defaults to false.
    public let dithering: Bool
    /// Shades with the triangle's face normal, like Three.js `flatShading`.
    public let flatShading: Bool
    /// Optional depth packing for the depth model. Nil uses near-white grayscale.
    public let depthPacking: DepthPacking?
    /// The distance model's output layout.
    public let distanceOutput: DistanceOutput
    /// Red-channel height for Lambert, Phong, toon, matcap and normal visualization.
    /// A normal map takes precedence when both maps are authored.
    public let bumpMap: BumpMap?
    /// A linear tangent-space or object-space shading normal texture.
    public let normalMap: TextureBinding?
    /// Signed X and Y multipliers for tangent-space normals.
    public let normalScale: SIMD2<Float>
    /// Replaces the diffuse color after maps and before Lambert, Phong, toon, or basic lighting.
    /// Other graph outputs are rejected.
    public let surfaceGraph: SurfaceShaderGraph?
    /// Live values for `surfaceGraph`, shared by every renderer prepared with this material.
    public let surfaceUniforms: ShaderUniforms?
    /// Textures for `surfaceGraph`, in declaration order.
    public let surfaceTextures: [any MTLTexture]

    /// Creates a transparent, black-by-default shadow catcher. It receives punctual
    /// shadows assigned to the graph. Enable directional reception when the graph
    /// supplies a directional map. The default mesh state tests depth without
    /// writing it. Supply a blending `MeshRenderState` with depth writes to occlude
    /// later draws where the catcher passes its depth test.
    public static func shadowCatcher(
        color: SIMD3<Float> = .zero, opacity: Float = 1, receivesDirectionalShadows: Bool = false
    ) throws -> ClassicMaterial {
        try ClassicMaterial(model: .shadow, color: color, opacity: opacity, receivesShadows: receivesDirectionalShadows)
    }

    /// Throws `Failure` for values outside the documented ranges or maps the model ignores:
    /// light, ambient-occlusion and emissive maps belong to basic and lit models (basic
    /// has no emissive), environment maps to basic, Lambert and Phong, and the depth,
    /// distance, normal and shadow models use `map` and `alphaMap` for alpha.
    /// Lambert, Phong, toon, matcap and normal models accept surface detail maps.
    public init(
        model: Model,
        color: SIMD3<Float> = SIMD3(repeating: 1),
        opacity: Float = 1,
        map: TextureBinding? = nil,
        alphaMap: TextureBinding? = nil,
        alphaTest: Float = 0,
        aoMap: TextureBinding? = nil,
        aoMapIntensity: Float = 1,
        lightMap: TextureBinding? = nil,
        lightMapIntensity: Float = 1,
        emissive: SIMD3<Float> = .zero,
        emissiveMap: TextureBinding? = nil,
        environmentMap: EnvironmentMap? = nil,
        ambient: SIMD3<Float> = .zero,
        receivesShadows: Bool = false,
        flatShading: Bool = false,
        depthPacking: DepthPacking? = nil,
        distanceOutput: DistanceOutput = .grayscale,
        bumpMap: BumpMap? = nil,
        normalMap: TextureBinding? = nil,
        normalScale: SIMD2<Float> = SIMD2(repeating: 1),
        surfaceGraph: SurfaceShaderGraph? = nil,
        surfaceUniforms: ShaderUniforms? = nil,
        surfaceTextures: [String: any MTLTexture] = [:],
        toneMapped: Bool = true,
        wireframe: Bool = false,
        isVisible: Bool = true,
        fog: Bool = true,
        dithering: Bool = false
    ) throws {
        func nonnegative(_ value: SIMD3<Float>) -> Bool { (0..<3).allSatisfy { value[$0].isFinite && value[$0] >= 0 } }
        guard nonnegative(color) else { throw Failure.invalidValue("color") }
        guard (0...1).contains(opacity) else { throw Failure.invalidValue("opacity") }
        guard alphaTest.isFinite, (0...1).contains(alphaTest) else { throw Failure.invalidValue("alphaTest") }
        guard aoMapIntensity.isFinite, aoMapIntensity >= 0, lightMapIntensity.isFinite, lightMapIntensity >= 0 else {
            throw Failure.invalidValue("map intensity")
        }
        guard nonnegative(emissive), nonnegative(ambient) else { throw Failure.invalidValue("emissive or ambient") }
        let lit: Bool
        switch model {
        case .lambert, .phong, .toon: lit = true
        default: lit = false
        }
        switch model {
        case .phong(let specular, let shininess):
            guard nonnegative(specular), shininess.isFinite, shininess >= 0 else {
                throw Failure.invalidValue("specular or shininess")
            }
        case .distance(let reference, let near, let far):
            guard (0..<3).allSatisfy({ reference[$0].isFinite }), near.isFinite, far.isFinite, near >= 0, far > near else {
                throw Failure.invalidValue("distance reference or range")
            }
        default: break
        }
        if depthPacking != nil {
            guard case .depth = model else { throw Failure.unusedProperty("depthPacking") }
        }
        if distanceOutput == .red {
            guard case .distance = model else { throw Failure.unusedProperty("distanceOutput") }
        }
        var isBasic = false
        if case .basic = model { isBasic = true }
        var takesEnvironment = isBasic
        switch model {
        case .lambert, .phong: takesEnvironment = true
        default: break
        }
        if (aoMap != nil || lightMap != nil) && !(lit || isBasic) { throw Failure.unusedProperty("aoMap or lightMap") }
        if (emissiveMap != nil || emissive != .zero) && !lit { throw Failure.unusedProperty("emissive") }
        if environmentMap != nil && !takesEnvironment { throw Failure.unusedProperty("environmentMap") }
        if ambient != .zero && !lit { throw Failure.unusedProperty("ambient") }
        var shadowModel = false
        if case .shadow = model { shadowModel = true }
        if receivesShadows && !(lit || shadowModel) { throw Failure.unusedProperty("receivesShadows") }
        switch model {
        case .lambert, .phong, .toon, .matcap, .normal:
            break
        default:
            if bumpMap != nil || normalMap != nil { throw Failure.unusedProperty("bumpMap or normalMap") }
        }
        guard normalScale.x.isFinite, normalScale.y.isFinite else { throw Failure.invalidValue("normalScale") }
        if let surfaceGraph {
            let channels = Set(surfaceGraph.outputs.connected.map(\.0))
            guard !channels.isEmpty, channels.isSubset(of: [.baseColor, .opacity, .normal]) else {
                throw Failure.nodeGraph("A classic node material reads base color, opacity, and normal.")
            }
            for node in surfaceGraph.nodes {
                switch node {
                case .sampleScene, .sampleSceneLevel, .sampleDepth, .viewportSafeUV, .softParticleOpacity,
                     .viewPositionFromDepth, .normalFromDepth, .denoise, .linearDepth, .derivative, .bumpMap:
                    throw Failure.nodeGraph("A classic node material cannot sample the frame, take a derivative, or read a bump map.")
                case .input(let input) where input.isScreenSpace:
                    throw Failure.nodeGraph("A classic node material cannot read screen coordinates.")
                default:
                    break
                }
            }
        } else if surfaceUniforms != nil || !surfaceTextures.isEmpty {
            throw Failure.nodeGraph("Surface uniforms and textures need a surface graph.")
        }
        self.model = model
        self.color = color
        self.opacity = opacity
        self.map = map
        self.alphaMap = alphaMap
        self.alphaTest = alphaTest
        self.aoMap = aoMap
        self.aoMapIntensity = aoMapIntensity
        self.lightMap = lightMap
        self.lightMapIntensity = lightMapIntensity
        self.emissive = emissive
        self.emissiveMap = emissiveMap
        self.environmentMap = environmentMap
        self.ambient = ambient
        self.receivesShadows = receivesShadows
        self.toneMapped = toneMapped
        self.wireframe = wireframe
        self.isVisible = isVisible
        self.fog = fog
        self.dithering = dithering
        self.flatShading = flatShading
        self.depthPacking = depthPacking
        self.distanceOutput = distanceOutput
        self.bumpMap = bumpMap
        self.normalMap = normalMap
        self.normalScale = normalScale
        self.surfaceGraph = surfaceGraph
        let declaredUniforms = surfaceGraph?.uniforms ?? []
        if let surfaceUniforms {
            guard surfaceGraph != nil, surfaceUniforms.matches(declaredUniforms) else { throw Failure.nodeGraph("Surface uniforms must match the graph.") }
            self.surfaceUniforms = surfaceUniforms
        } else {
            let readsClock = surfaceGraph?.program.readsClock ?? false
            self.surfaceUniforms = declaredUniforms.isEmpty && !readsClock ? nil : try ShaderUniforms(declaredUniforms)
        }
        let declaredTextures = surfaceGraph?.textures ?? []
        if let extra = surfaceTextures.keys.first(where: { name in !declaredTextures.contains { $0.name == name } }) {
            throw Failure.nodeGraph("The graph does not declare a texture named \(extra).")
        }
        self.surfaceTextures = try declaredTextures.map { declaration in
            guard declaration.shape == .twoDimensional else {
                throw Failure.nodeGraph("Texture \(declaration.name) is not a 2D image. Compute graphs sample cube and 3D textures.")
            }
            guard let texture = surfaceTextures[declaration.name] else {
                throw Failure.nodeGraph("The graph is missing texture \(declaration.name).")
            }
            guard PhysicallyBasedMaterial.isSurfaceTexture(texture, content: declaration.content) else {
                throw Failure.nodeGraph("Texture \(declaration.name) cannot be sampled by the graph.")
            }
            return texture
        }
    }

    /// Whether the model reads scene lights.
    var isLit: Bool {
        switch model {
        case .lambert, .phong, .toon: true
        default: false
        }
    }

    /// Whether the model reads punctual light records and their shadows.
    var usesPunctualLights: Bool {
        if case .shadow = model { return true }
        return isLit
    }

    var defaultRenderState: MeshRenderState {
        if case .shadow = model { return .transparent }
        return .opaque
    }

    /// The model code shared with `FrostfieldClassicModel` in ClassicMaterials.metal.
    var modelCode: UInt32 {
        switch model {
        case .basic: 0
        case .lambert: 1
        case .phong: 2
        case .toon: 3
        case .matcap: 4
        case .normal: 5
        case .depth: 6
        case .distance: 7
        case .shadow: 8
        }
    }

    // Function constant 43. Zero selects the established grayscale behavior.
    var depthOutputCode: UInt32 { depthPacking?.rawValue ?? (distanceOutput == .red ? 1 : 0) }

    func supportsDepthOutput(_ format: MTLPixelFormat) -> Bool {
        switch format {
        case .rgba8Unorm, .bgra8Unorm, .rgba16Unorm, .rgba16Float, .rgba32Float: true
        case .rg8Unorm, .rg16Unorm, .rg16Float, .rg32Float: depthPacking == .rg || distanceOutput == .red
        case .r8Unorm, .r16Unorm, .r16Float, .r32Float: distanceOutput == .red
        default: false
        }
    }

    /// Whether any map samples the secondary texture coordinates.
    var usesSecondaryTextureCoordinates: Bool {
        textureBindings.contains { $0?.textureCoordinateSet == .secondary }
    }

    var activeBumpMap: BumpMap? { normalMap == nil && (bumpMap?.scale ?? 0) != 0 ? bumpMap : nil }
    var hasSurfaceDetail: Bool { normalMap != nil || activeBumpMap != nil }

    /// Color, alpha, occlusion, light, emissive, lookup, and a normal or height map.
    var textureBindings: [TextureBinding?] {
        let lookup: TextureBinding?
        switch model {
        case .toon(let gradient): lookup = gradient
        case .matcap(let binding): lookup = binding
        default: lookup = nil
        }
        return [map, alphaMap, aoMap, lightMap, emissiveMap, lookup, normalMap ?? activeBumpMap?.texture]
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidValue(String)
        case unusedProperty(String)
        case environmentTexture
        case texture(Int)
        case nodeGraph(String)

        public var errorDescription: String? {
            switch self {
            case .invalidValue(let name): "The classic material's \(name) is outside its documented range."
            case .unusedProperty(let name): "The classic material model does not use \(name)."
            case .environmentTexture: "Use a shader-readable cube texture for the environment map."
            case .texture(6): "Classic surface detail needs a linear, shader-readable 2D normal or bump map and sampler on the renderer's device."
            case .texture(let index):
                "Classic material map \(index) (0 map, 1 alpha, 2 occlusion, 3 light, 4 emissive, 5 gradient or matcap) needs a shader-readable 2D texture on the renderer's device in RGBA8, BGRA8 (linear or sRGB), RGBA16Float, RG8, R8 or a block-compressed format; the alpha map needs a green channel, so not R8, BC4 or EAC R11."
            case .nodeGraph(let detail): detail
            }
        }
    }

    /// Fragment buffer 1 of `frostfield_mesh_classic`, matching `FrostfieldClassicUniforms`.
    struct Packed {
        var color: SIMD4<Float>
        /// Emissive radiance (rgb), then the environment refraction ratio.
        var emissive: SIMD4<Float>
        var specular: SIMD4<Float>
        var ambient: SIMD4<Float>
        /// Occlusion intensity, light map intensity, reflectivity, environment intensity.
        var intensities: SIMD4<Float>
        /// Distance reference (xyz); w is the cosine of the environment yaw.
        var reference: SIMD4<Float>
        /// Near, far, 1 for reversed depth, sine of the environment yaw.
        var range: SIMD4<Float>
        /// Signed XY scale, normal encoding (3 for height), object-space flag.
        var normal: SIMD4<Float>
        /// Bound maps (bit i is 2D map i, bit 7 the environment), secondary-UV bits,
        /// environment blend, and options (bit 0 refraction, bit 1 flat shading).
        var flags: SIMD4<UInt32>
        var transforms: InlineArray<14, SIMD4<Float>>
    }

    struct Prepared {
        let textures: [any MTLTexture]
        let samplers: [any MTLSamplerState]
        let environment: any MTLTexture
        let environmentSampler: any MTLSamplerState
        let shadowFallback: any MTLTexture
        let graphTextures: [any MTLTexture]
        let packed: Packed

        init(material: ClassicMaterial, device: any MTLDevice, depthMode: MeshDepthMode) throws {
            let descriptor = MTLSamplerDescriptor()
            descriptor.minFilter = .linear
            descriptor.magFilter = .linear
            descriptor.mipFilter = .linear
            descriptor.sAddressMode = .repeat
            descriptor.tAddressMode = .repeat
            guard let defaultSampler = device.makeSamplerState(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("classic material sampler")
            }
            descriptor.sAddressMode = .clampToEdge
            descriptor.tAddressMode = .clampToEdge
            descriptor.rAddressMode = .clampToEdge
            guard let clampSampler = device.makeSamplerState(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("classic material sampler")
            }
            let white = try PhysicallyBasedMaterial.Prepared.makeWhiteTexture(device: device)
            var textures: [any MTLTexture] = []
            var samplers: [any MTLSamplerState] = []
            var bound: UInt32 = 0
            var secondary: UInt32 = 0
            var transforms = InlineArray<14, SIMD4<Float>>(repeating: .zero)
            for (index, binding) in material.textureBindings.enumerated() {
                let rows = (binding?.transform ?? .identity).rows
                transforms[2 * index] = rows.0
                transforms[2 * index + 1] = rows.1
                guard let binding else {
                    textures.append(white)
                    samplers.append(index == 5 ? clampSampler : defaultSampler)
                    continue
                }
                guard binding.texture.device === device, binding.sampler.map({ $0.device === device }) ?? true,
                      index == 6
                        ? material.activeBumpMap != nil || Self.isNormalTexture(binding)
                        : Self.isMapTexture(binding.texture, needsGreen: index == 1) else {
                    throw Failure.texture(index)
                }
                textures.append(binding.texture)
                // Gradient and matcap lookups clamp by default, like their usual Three.js setup.
                if index == 6, binding.sampler == nil, let sampling = binding.sampling {
                    samplers.append(try sampling.makeSamplerState(device: device))
                } else {
                    samplers.append(binding.sampler ?? (index == 5 ? clampSampler : defaultSampler))
                }
                bound |= 1 << UInt32(index)
                secondary |= binding.textureCoordinateSet.rawValue << UInt32(index)
            }
            self.textures = textures
            self.samplers = samplers
            environmentSampler = defaultSampler
            if let map = material.environmentMap {
                guard map.texture.device === device else { throw MeshRenderer.Failure.wrongDevice }
                environment = map.texture
                bound |= 1 << 7
            } else {
                environment = try Self.makeWhiteCube(device: device)
            }
            shadowFallback = try Self.makeDepthFallback(device: device)

            var specular = SIMD4<Float>(0, 0, 0, 0)
            var reference = SIMD4<Float>(0, 0, 0, 1)
            var range = SIMD4<Float>(0, 1, depthMode == .reversed ? 1 : 0, 0)
            switch material.model {
            case .phong(let color, let shininess): specular = SIMD4(color, shininess)
            case .distance(let point, let near, let far):
                reference = SIMD4(point, 1)
                range.x = near
                range.y = far
            default: break
            }
            let environment = material.environmentMap
            let yaw = environment?.yaw ?? 0
            reference.w = cos(yaw)
            range.w = sin(yaw)
            let blend: UInt32 = switch environment?.blend ?? .multiply {
            case .multiply: 0
            case .mix: 1
            case .add: 2
            }
            let refractionRatio = environment?.mapping.refractionRatio
            let options: UInt32 = (material.flatShading ? 2 : 0) | (refractionRatio == nil ? 0 : 1)
            let normal: SIMD4<Float> = if let bump = material.activeBumpMap {
                SIMD4(bump.scale, 0, 3, 0)
            } else {
                SIMD4(
                    material.normalScale.x, material.normalScale.y, Float(material.normalMap?.normalEncoding.rawValue ?? 0),
                    material.normalMap?.normalSpace == .objectSpace ? 1 : 0
                )
            }
            packed = Packed(
                color: SIMD4(material.color, material.opacity),
                emissive: SIMD4(material.emissive, refractionRatio ?? 0),
                specular: specular,
                ambient: SIMD4(material.ambient, material.alphaTest),
                intensities: SIMD4(
                    material.aoMapIntensity, material.lightMapIntensity,
                    environment?.reflectivity ?? 0, environment?.intensity ?? 0
                ),
                reference: reference,
                range: range,
                normal: normal,
                flags: SIMD4(bound, secondary, blend, options),
                transforms: transforms
            )
            if material.surfaceGraph != nil {
                guard material.surfaceTextures.allSatisfy({ $0.device === device }) else { throw MeshRenderer.Failure.wrongDevice }
                graphTextures = material.surfaceTextures
                    + Array(repeating: white, count: MaterialGraph.maximumTextureCount - material.surfaceTextures.count)
                materialUniforms = material.surfaceUniforms
            } else {
                graphTextures = []
                materialUniforms = nil
            }
        }

        func encode(into encoder: any MTLRenderCommandEncoder) {
            for (index, texture) in textures.enumerated() {
                let slot = index == 0 ? 0 : index + (index == 6 ? 2 : 1)
                encoder.setFragmentTexture(texture, index: slot)
                encoder.setFragmentSamplerState(samplers[index], index: slot)
            }
            encoder.setFragmentTexture(environment, index: 7)
            encoder.setFragmentSamplerState(environmentSampler, index: 7)
            var packed = packed
            encoder.setFragmentBytes(&packed, length: MemoryLayout<Packed>.stride, index: 1)
        }

        /// Binds the node graph's uniforms and textures. The color fragment declares them;
        /// the attribute fragment does not, so call this only for the color pass.
        func encodeNodeGraph(into encoder: any MTLRenderCommandEncoder, object: ShaderObjectContext? = nil) {
            guard !graphTextures.isEmpty else { return }
            let values = materialUniforms?.bufferSnapshot(object: object).values ?? []
            if values.isEmpty {
                var unused = SIMD4<Float>.zero
                encoder.setFragmentBytes(&unused, length: MemoryLayout<SIMD4<Float>>.stride, index: 19)
            } else {
                values.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 19) }
            }
            encoder.setFragmentTextures(graphTextures, range: 16..<(16 + graphTextures.count))
        }

        /// The uniforms bound with the graph textures. Nil when the material has no node graph.
        var materialUniforms: ShaderUniforms?

        private static func isNormalTexture(_ binding: TextureBinding) -> Bool {
            let texture = binding.texture
            guard PhysicallyBasedMaterial.isSurfaceTexture(texture, content: .data),
                  texture.storageMode != .memoryless else { return false }
            let format = texture.pixelFormat
            if binding.normalEncoding != .rgb { return binding.normalEncoding.acceptsPackedTexture(format) }
            switch format {
            case .rgba8Unorm, .bgra8Unorm, .rgba16Float: return true
            default: return MaterialTextureFormat.compressedEncoding(format).isCompressed
            }
        }

        /// Whether a texture can be a 2D map. The alpha map reads green, which
        /// single-channel formats sample as zero.
        private static func isMapTexture(_ texture: any MTLTexture, needsGreen: Bool) -> Bool {
            guard texture.textureType == .type2D, !texture.isFramebufferOnly,
                  texture.usage.isEmpty || texture.usage.contains(.shaderRead) else { return false }
            switch texture.pixelFormat {
            case .r8Unorm, .bc4_rUnorm, .bc4_rSnorm, .eac_r11Unorm, .eac_r11Snorm:
                return !needsGreen
            case .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb, .rgba16Float, .rg8Unorm:
                return true
            case let format:
                return MaterialTextureFormat.compressedEncoding(format).isCompressed
            }
        }

        private static func makeWhiteCube(device: any MTLDevice) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba8Unorm, size: 1, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("classic environment fallback")
            }
            var white: UInt32 = .max
            for face in 0..<6 {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, slice: face,
                    withBytes: &white, bytesPerRow: 4, bytesPerImage: 4
                )
            }
            return texture
        }

        private static func makeDepthFallback(device: any MTLDevice) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false
            )
            descriptor.textureType = .type2DArray
            descriptor.arrayLength = 1
            descriptor.storageMode = .private
            descriptor.usage = .shaderRead
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("classic shadow fallback")
            }
            return texture
        }
    }
}
