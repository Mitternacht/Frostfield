import Foundation
import simd

extension GLTFScene {
    /// The metallic-roughness factors and texture bindings of a used glTF material.
    public struct Material {
        public enum AlphaMode: Equatable, Sendable {
            case opaque
            case mask(cutoff: Float)
            case blend
        }

        /// Native height-map metadata carried from an imported MTL material.
        /// This is outside the portable glTF material schema.
        public struct BumpMap {
            /// A linear height field in the red channel.
            public let texture: Texture
            /// The signed `-bm` strength from the material library.
            public let scale: Float
        }

        /// A dielectric layer on top of the base material.
        public struct Clearcoat {
            public let factor: Float
            public let roughness: Float
            public let factorTexture: Texture?
            public let roughnessTexture: Texture?
            public let normalTexture: Texture?
            public let normalScale: Float
        }

        public struct Transmission {
            public let factor: Float
            /// The transmission multiplier is stored in the linear red channel.
            public let factorTexture: Texture?
        }

        /// KHR_materials_diffuse_transmission. Factors default to zero and white.
        public struct DiffuseTransmission {
            /// Zero to one: the fraction of non-specularly reflected light that is transmitted.
            public let factor: Float
            /// Linear alpha multiplies the factor.
            public let factorTexture: Texture?
            /// Linear RGB from zero to one.
            public let colorFactor: SIMD3<Float>
            /// sRGB color multiplying the color factor.
            public let colorTexture: Texture?
        }

        /// KHR_materials_diffuse_roughness (Khronos draft). The factor defaults to zero.
        public struct DiffuseRoughness {
            /// Zero to one.
            public let factor: Float
            /// Linear red multiplies the factor.
            public let texture: Texture?
        }

        /// KHR_materials_volume_scatter (Khronos draft). The color defaults to black and the anisotropy to zero.
        public struct VolumeScatter {
            /// The multiple-scattering albedo, linear RGB from zero to one.
            public let multiscatterColorFactor: SIMD3<Float>
            /// sRGB color multiplying the factor where light enters the volume.
            public let multiscatterColorTexture: Texture?
            /// The Henyey-Greenstein g, greater than -1 and less than 1.
            public let scatterAnisotropy: Float
        }

        public struct Volume {
            /// Thickness in mesh-space units before applying the green-channel multiplier.
            public let thickness: Float
            public let thicknessTexture: Texture?
            /// Distance in world-space units. Infinity means no attenuation.
            public let attenuationDistance: Float
            public let attenuationColor: SIMD3<Float>
        }

        /// KHR_materials_specular. Factors default to one.
        public struct Specular {
            /// Zero to one; scales dielectric F0 and F90.
            public let factor: Float
            /// Linear alpha multiplies the factor.
            public let factorTexture: Texture?
            /// Nonnegative linear RGB multiplying dielectric F0.
            public let colorFactor: SIMD3<Float>
            /// sRGB color multiplying the color factor.
            public let colorTexture: Texture?
        }

        /// KHR_materials_sheen. Factors default to zero.
        public struct Sheen {
            /// Linear RGB from zero to one.
            public let colorFactor: SIMD3<Float>
            /// sRGB color multiplying the color factor.
            public let colorTexture: Texture?
            /// Perceptual roughness from zero to one.
            public let roughness: Float
            /// Linear alpha multiplies the roughness.
            public let roughnessTexture: Texture?
        }

        /// KHR_materials_anisotropy. Factors default to zero.
        public struct Anisotropy {
            /// Zero to one.
            public let strength: Float
            /// Radians, counterclockwise from the tangent toward the bitangent.
            public let rotation: Float
            /// Linear red and green encode a direction; blue multiplies the strength.
            public let texture: Texture?
        }

        /// KHR_materials_iridescence. Defaults: factor zero, IOR 1.3, thickness 100 to 400 nm.
        public struct Iridescence {
            /// Zero to one.
            public let factor: Float
            /// Linear red multiplies the factor.
            public let factorTexture: Texture?
            /// The film's index of refraction, at least one.
            public let indexOfRefraction: Float
            /// Nanometers at green zero of the thickness texture.
            public let minimumThickness: Float
            /// Nanometers at green one, or without a thickness texture.
            public let maximumThickness: Float
            public let thicknessTexture: Texture?
        }

        public let sourceIndex: Int
        public let name: String?
        public let baseColorFactor: SIMD4<Float>
        public let metallicFactor: Float
        public let roughnessFactor: Float
        public var baseColorImage: Image? { baseColorTexture?.image }
        public let baseColorTexture: Texture?
        public let metallicRoughnessTexture: Texture?
        public let normalTexture: Texture?
        public let normalScale: Float
        public let occlusionTexture: Texture?
        public let occlusionStrength: Float
        public let emissiveTexture: Texture?
        public let emissiveFactor: SIMD3<Float>
        public let alphaMode: AlphaMode
        public let isDoubleSided: Bool
        public let clearcoat: Clearcoat?
        public let transmission: Transmission?
        public let volume: Volume?
        /// Defaults to 1.5. An authored zero selects the glTF infinite-IOR compatibility mode.
        public let indexOfRefraction: Float
        /// KHR_materials_emissive_strength: a nonnegative emissive multiplier; one when absent.
        public let emissiveStrength: Float
        public let specular: Specular?
        public let sheen: Sheen?
        public let anisotropy: Anisotropy?
        public let iridescence: Iridescence?
        /// KHR_materials_dispersion: 20 divided by the Abbe number; zero when absent.
        /// Preparation passes it to the material's transmission, which only the path tracer refracts by wavelength.
        public let dispersion: Float
        /// KHR_materials_unlit. Preparation selects `PhysicallyBasedMaterial.ShadingModel.unlit`.
        public let isUnlit: Bool
        /// KHR_materials_diffuse_transmission; nil when absent.
        public let diffuseTransmission: DiffuseTransmission?
        /// KHR_materials_diffuse_roughness; nil when absent.
        public let diffuseRoughness: DiffuseRoughness?
        /// KHR_materials_volume_scatter, which needs KHR_materials_volume; nil when absent.
        public let volumeScatter: VolumeScatter?
        /// An MTL height map. A normal texture takes precedence when both are present.
        public let bumpMap: BumpMap?

        var hasActiveBumpMap: Bool {
            !isUnlit && normalTexture == nil && (bumpMap?.scale ?? 0) != 0
        }
    }

    public struct Texture {
        public let sourceIndex: Int
        /// The image to sample: a texture extension's image when this system can use
        /// it, otherwise the texture's core PNG or JPEG `source`.
        public let image: Image
        public let sampling: Sampling
        /// The effective set, including a KHR_texture_transform `texCoord` override.
        public let textureCoordinateSet: MeshTextureCoordinateSet
        /// KHR_texture_transform; identity when absent.
        public let transform: PhysicallyBasedMaterial.TextureTransform

        /// The core PNG or JPEG `source` kept when `image` is a KTX2 payload, for
        /// devices that cannot sample its block format. Material preparation uses it
        /// in that case. Nil otherwise.
        public let fallbackImage: Image?
    }

    public struct Sampling: Hashable, Sendable {
        public enum Magnification: Int, Sendable {
            case nearest = 9728
            case linear = 9729
        }

        public enum Minification: Int, Sendable {
            case nearest = 9728
            case linear = 9729
            case nearestMipmapNearest = 9984
            case linearMipmapNearest = 9985
            case nearestMipmapLinear = 9986
            case linearMipmapLinear = 9987
        }

        public enum Wrap: Int, Sendable {
            case clampToEdge = 33071
            case mirroredRepeat = 33648
            case `repeat` = 10497
        }

        public let magnification: Magnification
        public let minification: Minification
        public let wrapS: Wrap
        public let wrapT: Wrap
    }

    /// Original encoded bytes. `mimeType` comes from the byte signature: `image/png`
    /// or `image/jpeg`, or `image/webp`, `image/avif`, or `image/ktx2` from texture
    /// extensions. `TextureLoader` decodes all of them; decoding remains the consumer's step.
    public struct Image {
        public let sourceIndex: Int
        public let name: String?
        public let mimeType: String
        public let data: Data
    }
}

struct GLTFMaterialDocument: Decodable {
    static let clearcoatExtension = "KHR_materials_clearcoat"
    static let transmissionExtension = "KHR_materials_transmission"
    static let volumeExtension = "KHR_materials_volume"
    static let iorExtension = "KHR_materials_ior"
    static let emissiveStrengthExtension = "KHR_materials_emissive_strength"
    static let specularExtension = "KHR_materials_specular"
    static let sheenExtension = "KHR_materials_sheen"
    static let anisotropyExtension = "KHR_materials_anisotropy"
    static let iridescenceExtension = "KHR_materials_iridescence"
    static let dispersionExtension = "KHR_materials_dispersion"
    static let unlitExtension = "KHR_materials_unlit"
    static let diffuseTransmissionExtension = "KHR_materials_diffuse_transmission"
    static let diffuseRoughnessExtension = "KHR_materials_diffuse_roughness"
    static let volumeScatterExtension = "KHR_materials_volume_scatter"
    static let textureTransformExtension = "KHR_texture_transform"
    /// Extensions accepted inside a material record.
    static let materialExtensions: Set<String> = [
        clearcoatExtension, transmissionExtension, volumeExtension, iorExtension, emissiveStrengthExtension,
        specularExtension, sheenExtension, anisotropyExtension, iridescenceExtension, dispersionExtension, unlitExtension,
        diffuseTransmissionExtension, diffuseRoughnessExtension, volumeScatterExtension,
    ]
    /// Extensions a scene loaded with materials can require.
    static let supportedExtensions = materialExtensions.union([textureTransformExtension, GLTFMaterialVariantsDocument.extensionName])
    static let basisuExtension = "KHR_texture_basisu"
    static let webPExtension = "EXT_texture_webp"
    static let avifExtension = "EXT_texture_avif"
    /// Texture extensions in preference order: GPU-native KTX2, then AVIF, then WebP.
    static let texturePreference = [basisuExtension, avifExtension, webPExtension]
    static let textureExtensions = Set(texturePreference)

    /// Attributes present on every primitive that uses one material.
    struct Usage {
        var coordinates: Set<MeshTextureCoordinateSet>
        var hasAuthoredTangentSpace: Bool
    }

    private typealias Failure = GLTFMeshLoader.Failure

    let materials: [MaterialRecord]?
    let textures: [TextureRecord]?
    let images: [ImageRecord]?
    let samplers: [SamplerRecord]?

    struct MaterialRecord: Decodable {
        let name: String?
        let pbrMetallicRoughness: Surface?
        let alphaMode: String?
        let alphaCutoff: Float?
        let doubleSided: Bool?
        let normalTexture: TextureBinding?
        let occlusionTexture: TextureBinding?
        let emissiveTexture: TextureBinding?
        let emissiveFactor: [Float]?
        let extensions: MaterialExtensions?
    }

    struct MaterialExtensions: Decodable {
        let clearcoat: Result<ClearcoatRecord, any Error>?
        let transmission: Result<TransmissionRecord, any Error>?
        let volume: Result<VolumeRecord, any Error>?
        let ior: Result<IORRecord, any Error>?
        let emissiveStrength: Result<EmissiveStrengthRecord, any Error>?
        let specular: Result<SpecularRecord, any Error>?
        let sheen: Result<SheenRecord, any Error>?
        let anisotropy: Result<AnisotropyRecord, any Error>?
        let iridescence: Result<IridescenceRecord, any Error>?
        let dispersion: Result<DispersionRecord, any Error>?
        let unlit: Result<UnlitRecord, any Error>?
        let diffuseTransmission: Result<DiffuseTransmissionRecord, any Error>?
        let diffuseRoughness: Result<DiffuseRoughnessRecord, any Error>?
        let volumeScatter: Result<VolumeScatterRecord, any Error>?
        let unsupported: String?

        private struct Name: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }

        init(from decoder: any Decoder) throws {
            let values = try decoder.container(keyedBy: Name.self)
            // Keep decode failures with their source material so unused materials
            // remain independent of the selected scene.
            func record<T: Decodable>(_ name: String, as type: T.Type) -> Result<T, any Error>? {
                let key = Name(stringValue: name)
                guard values.contains(key) else { return nil }
                return Result { try values.decode(type, forKey: key) }
            }
            clearcoat = record(GLTFMaterialDocument.clearcoatExtension, as: ClearcoatRecord.self)
            transmission = record(GLTFMaterialDocument.transmissionExtension, as: TransmissionRecord.self)
            volume = record(GLTFMaterialDocument.volumeExtension, as: VolumeRecord.self)
            ior = record(GLTFMaterialDocument.iorExtension, as: IORRecord.self)
            emissiveStrength = record(GLTFMaterialDocument.emissiveStrengthExtension, as: EmissiveStrengthRecord.self)
            specular = record(GLTFMaterialDocument.specularExtension, as: SpecularRecord.self)
            sheen = record(GLTFMaterialDocument.sheenExtension, as: SheenRecord.self)
            anisotropy = record(GLTFMaterialDocument.anisotropyExtension, as: AnisotropyRecord.self)
            iridescence = record(GLTFMaterialDocument.iridescenceExtension, as: IridescenceRecord.self)
            dispersion = record(GLTFMaterialDocument.dispersionExtension, as: DispersionRecord.self)
            unlit = record(GLTFMaterialDocument.unlitExtension, as: UnlitRecord.self)
            diffuseTransmission = record(GLTFMaterialDocument.diffuseTransmissionExtension, as: DiffuseTransmissionRecord.self)
            diffuseRoughness = record(GLTFMaterialDocument.diffuseRoughnessExtension, as: DiffuseRoughnessRecord.self)
            volumeScatter = record(GLTFMaterialDocument.volumeScatterExtension, as: VolumeScatterRecord.self)
            unsupported = values.allKeys.map(\.stringValue).filter { !GLTFMaterialDocument.materialExtensions.contains($0) }.min()
        }
    }

    struct EmissiveStrengthRecord: Decodable {
        let emissiveStrength: Float?
        let extensions: [String: Present]?
    }

    struct SpecularRecord: Decodable {
        let specularFactor: Float?
        let specularTexture: TextureBinding?
        let specularColorFactor: [Float]?
        let specularColorTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct SheenRecord: Decodable {
        let sheenColorFactor: [Float]?
        let sheenColorTexture: TextureBinding?
        let sheenRoughnessFactor: Float?
        let sheenRoughnessTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct AnisotropyRecord: Decodable {
        let anisotropyStrength: Float?
        let anisotropyRotation: Float?
        let anisotropyTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct IridescenceRecord: Decodable {
        let iridescenceFactor: Float?
        let iridescenceTexture: TextureBinding?
        let iridescenceIor: Float?
        let iridescenceThicknessMinimum: Float?
        let iridescenceThicknessMaximum: Float?
        let iridescenceThicknessTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct DispersionRecord: Decodable {
        let dispersion: Float?
        let extensions: [String: Present]?
    }

    struct UnlitRecord: Decodable {
        let extensions: [String: Present]?
    }

    struct DiffuseTransmissionRecord: Decodable {
        let diffuseTransmissionFactor: Float?
        let diffuseTransmissionTexture: TextureBinding?
        let diffuseTransmissionColorFactor: [Float]?
        let diffuseTransmissionColorTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct DiffuseRoughnessRecord: Decodable {
        let diffuseRoughnessFactor: Float?
        let diffuseRoughnessTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct VolumeScatterRecord: Decodable {
        let multiscatterColorFactor: [Float]?
        let multiscatterColorTexture: TextureBinding?
        let scatterAnisotropy: Float?
        let extensions: [String: Present]?
    }

    struct ClearcoatRecord: Decodable {
        let clearcoatFactor: Float?
        let clearcoatRoughnessFactor: Float?
        let clearcoatTexture: TextureBinding?
        let clearcoatRoughnessTexture: TextureBinding?
        let clearcoatNormalTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct TransmissionRecord: Decodable {
        let transmissionFactor: Float?
        let transmissionTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct VolumeRecord: Decodable {
        let thicknessFactor: Float?
        let thicknessTexture: TextureBinding?
        let attenuationDistance: Float?
        let attenuationColor: [Float]?
        let extensions: [String: Present]?
    }

    struct IORRecord: Decodable {
        let ior: Float?
        let extensions: [String: Present]?
    }

    struct Surface: Decodable {
        let baseColorFactor: [Float]?
        let metallicFactor: Float?
        let roughnessFactor: Float?
        let baseColorTexture: TextureBinding?
        let metallicRoughnessTexture: TextureBinding?
        let extensions: [String: Present]?
    }

    struct TextureBinding: Decodable {
        let index: Int
        let texCoord: Int?
        let scale: Float?
        let strength: Float?
        let extensions: TextureInfoExtensions?
    }

    struct TextureInfoExtensions: Decodable {
        let transform: Result<TransformRecord, any Error>?
        let unsupported: String?

        private struct Name: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }

        init(from decoder: any Decoder) throws {
            let values = try decoder.container(keyedBy: Name.self)
            let key = Name(stringValue: GLTFMaterialDocument.textureTransformExtension)
            transform = values.contains(key) ? Result { try values.decode(TransformRecord.self, forKey: key) } : nil
            unsupported = values.allKeys.map(\.stringValue).filter { $0 != GLTFMaterialDocument.textureTransformExtension }.min()
        }
    }

    struct TransformRecord: Decodable {
        let offset: [Float]?
        let rotation: Float?
        let scale: [Float]?
        let texCoord: Int?
        let extensions: [String: Present]?
    }

    struct TextureRecord: Decodable {
        let source: Int?
        let sampler: Int?
        let extensions: TextureExtensions?
    }

    /// Image sources from supported texture extensions, keyed by extension name.
    struct TextureExtensions: Decodable {
        let sources: [String: Int?]
        let unsupported: String?

        private struct Name: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }

        private struct Source: Decodable { let source: Int? }

        init(from decoder: any Decoder) throws {
            let values = try decoder.container(keyedBy: Name.self)
            var sources: [String: Int?] = [:]
            for key in values.allKeys where GLTFMaterialDocument.textureExtensions.contains(key.stringValue) {
                sources[key.stringValue] = try values.decode(Source.self, forKey: key).source
            }
            self.sources = sources
            unsupported = values.allKeys.map(\.stringValue).filter { !GLTFMaterialDocument.textureExtensions.contains($0) }.min()
        }
    }

    struct SamplerRecord: Decodable {
        let magFilter: Int?
        let minFilter: Int?
        let wrapS: Int?
        let wrapT: Int?
        let extensions: [String: Present]?
    }

    struct ImageRecord: Decodable {
        let name: String?
        let uri: String?
        let bufferView: Int?
        let mimeType: String?
        let extensions: [String: Present]?
    }

    struct Present: Decodable {
        init(from decoder: any Decoder) throws {}
    }

    private struct TextureKey: Hashable {
        let index: Int
        let coordinates: MeshTextureCoordinateSet
        let transform: PhysicallyBasedMaterial.TextureTransform
    }

    func resolve(
        usage: [Int: Usage],
        maximumBytes: Int,
        readView: (Int) throws -> Data,
        readAdjacent: (String) throws -> Data
    ) throws -> [GLTFScene.Material] {
        let records = materials ?? []
        var resolved: [GLTFScene.Material] = []
        var imageCache: [Int: GLTFScene.Image] = [:]
        var textureCache: [TextureKey: GLTFScene.Texture] = [:]
        var imageBytes = 0

        func texture(_ binding: TextureBinding?, channel: String, material: Int) throws -> GLTFScene.Texture? {
            guard let binding else { return nil }
            var setIndex = binding.texCoord ?? 0
            var transform = PhysicallyBasedMaterial.TextureTransform.identity
            if let unsupported = binding.extensions?.unsupported {
                throw Failure.unsupportedFeature("\(unsupported) on \(channel) binding on material \(material)")
            }
            if let result = binding.extensions?.transform {
                guard case .success(let record) = result else {
                    throw Failure.invalidMaterial(material, "the \(channel) texture transform requires an object with valid properties")
                }
                try rejectExtensions(record.extensions, context: "\(channel) texture transform on material \(material)")
                let offset = record.offset ?? [0, 0]
                let scale = record.scale ?? [1, 1]
                guard offset.count == 2, scale.count == 2,
                      let value = try? PhysicallyBasedMaterial.TextureTransform(
                        offset: SIMD2(offset[0], offset[1]), rotation: record.rotation ?? 0, scale: SIMD2(scale[0], scale[1])
                      ) else {
                    throw Failure.invalidMaterial(
                        material, "the \(channel) texture transform needs two finite offsets, a finite rotation, and two finite nonzero scales"
                    )
                }
                transform = value
                // The extension's texCoord replaces the binding's coordinate set.
                if let override = record.texCoord { setIndex = override }
            }
            guard let rawValue = UInt32(exactly: setIndex),
                  let coordinates = MeshTextureCoordinateSet(rawValue: rawValue) else {
                throw Failure.unsupportedFeature("\(channel) texture coordinate set \(setIndex) on material \(material)")
            }
            guard usage[material]?.coordinates.contains(coordinates) == true else {
                throw Failure.invalidMaterial(material, "each primitive using the \(channel) texture needs TEXCOORD_\(setIndex)")
            }
            let key = TextureKey(index: binding.index, coordinates: coordinates, transform: transform)
            if let cached = textureCache[key] { return cached }
            guard let textures, textures.indices.contains(binding.index) else { throw Failure.missingTexture(binding.index) }
            let record = textures[binding.index]
            if let unsupported = record.extensions?.unsupported {
                throw Failure.unsupportedFeature("\(unsupported) on texture \(binding.index)")
            }
            func image(_ source: Int, kinds: Set<ImageSignature>, context: String) throws -> GLTFScene.Image {
                let image: GLTFScene.Image
                if let existing = imageCache[source] {
                    image = existing
                } else {
                    image = try readImage(source, readView: readView, readAdjacent: readAdjacent)
                    guard image.data.count <= maximumBytes - imageBytes else { throw Failure.sizeLimit }
                    imageBytes += image.data.count
                    imageCache[source] = image
                }
                guard let kind = ImageSignature(mimeType: image.mimeType), kinds.contains(kind) else {
                    throw Failure.invalidImage(source, "\(context) needs \(kinds.map(\.name).sorted().joined(separator: " or ")) data")
                }
                return image
            }
            let (selected, fallback) = try textureImages(
                record, texture: binding.index,
                core: { try image($0, kinds: [.png, .jpeg], context: "a texture source") },
                extensionImage: { name, source in
                    try image(source, kinds: [Self.imageKind(forExtension: name)], context: name)
                }
            )
            let result = GLTFScene.Texture(
                sourceIndex: binding.index, image: selected, sampling: try sampling(record.sampler, texture: binding.index),
                textureCoordinateSet: coordinates, transform: transform, fallbackImage: fallback
            )
            textureCache[key] = result
            return result
        }

        for index in usage.keys.sorted() {
            guard records.indices.contains(index) else { throw Failure.missingMaterial(index) }
            let record = records[index]
            let surface = record.pbrMetallicRoughness
            if let unsupported = record.extensions?.unsupported {
                throw Failure.unsupportedFeature("\(unsupported) on material \(index)")
            }
            try rejectExtensions(surface?.extensions, context: "surface on material \(index)")
            let factor = surface?.baseColorFactor ?? [1, 1, 1, 1]
            let metallic = surface?.metallicFactor ?? 1
            let roughness = surface?.roughnessFactor ?? 1
            let emission = record.emissiveFactor ?? [0, 0, 0]
            let layers = try resolveLayers(
                record, hasTangentFrame: usage[index].map { $0.hasAuthoredTangentSpace || $0.coordinates.contains(.primary) } == true,
                material: index, texture: texture
            )
            let normalScale = record.normalTexture?.scale ?? 1
            let occlusionStrength = record.occlusionTexture?.strength ?? 1
            guard factor.count == 4, factor.allSatisfy({ (0...1).contains($0) }),
                  (0...1).contains(metallic), (0...1).contains(roughness),
                  emission.count == 3, emission.allSatisfy({ (0...1).contains($0) }),
                  normalScale.isFinite, (0...1).contains(occlusionStrength) else {
                throw Failure.invalidMaterial(index, "material factors are outside the supported ranges")
            }

            let alpha: GLTFScene.Material.AlphaMode
            switch record.alphaMode ?? "OPAQUE" {
            case "OPAQUE": alpha = .opaque
            case "BLEND": alpha = .blend
            case "MASK":
                let cutoff = record.alphaCutoff ?? 0.5
                guard cutoff.isFinite, cutoff >= 0 else {
                    throw Failure.invalidMaterial(index, "alpha cutoff must be finite and nonnegative")
                }
                alpha = .mask(cutoff: cutoff)
            case let mode:
                throw Failure.unsupportedFeature("alpha mode \(mode) on material \(index)")
            }

            let transmission = try resolveTransmission(record, material: index, texture: texture)
            let volume = try resolveVolume(
                record, hasTransmission: transmission != nil || layers.diffuseTransmission != nil, material: index, texture: texture
            )
            let volumeScatter = try resolveVolumeScatter(record, hasVolume: volume != nil, material: index, texture: texture)

            resolved.append(GLTFScene.Material(
                sourceIndex: index, name: record.name,
                baseColorFactor: SIMD4(factor[0], factor[1], factor[2], factor[3]),
                metallicFactor: metallic, roughnessFactor: roughness,
                baseColorTexture: try texture(surface?.baseColorTexture, channel: "base color", material: index),
                metallicRoughnessTexture: try texture(surface?.metallicRoughnessTexture, channel: "metallic-roughness", material: index),
                normalTexture: try texture(record.normalTexture, channel: "normal", material: index),
                normalScale: normalScale,
                occlusionTexture: try texture(record.occlusionTexture, channel: "occlusion", material: index),
                occlusionStrength: occlusionStrength,
                emissiveTexture: try texture(record.emissiveTexture, channel: "emissive", material: index),
                emissiveFactor: SIMD3(emission[0], emission[1], emission[2]),
                alphaMode: alpha, isDoubleSided: record.doubleSided ?? false,
                clearcoat: try resolveClearcoat(
                    record, hasAuthoredTangentSpace: usage[index]?.hasAuthoredTangentSpace == true,
                    material: index, texture: texture
                ),
                transmission: transmission, volume: volume, indexOfRefraction: try resolveIOR(record, material: index),
                emissiveStrength: layers.emissiveStrength, specular: layers.specular, sheen: layers.sheen,
                anisotropy: layers.anisotropy, iridescence: layers.iridescence, dispersion: layers.dispersion,
                isUnlit: layers.isUnlit, diffuseTransmission: layers.diffuseTransmission,
                diffuseRoughness: layers.diffuseRoughness, volumeScatter: volumeScatter, bumpMap: nil
            ))
        }
        return resolved
    }

    private func resolveClearcoat(
        _ record: MaterialRecord, hasAuthoredTangentSpace: Bool, material index: Int,
        texture: (TextureBinding?, String, Int) throws -> GLTFScene.Texture?
    ) throws -> GLTFScene.Material.Clearcoat? {
        guard let coat = try extensionRecord(record.extensions?.clearcoat, name: "clearcoat", material: index) else {
            return nil
        }
        try rejectExtensions(coat.extensions, context: "clearcoat on material \(index)")
        let factor = coat.clearcoatFactor ?? 0
        let roughness = coat.clearcoatRoughnessFactor ?? 0
        let normalScale = coat.clearcoatNormalTexture?.scale ?? 1
        guard (0...1).contains(factor), (0...1).contains(roughness), normalScale.isFinite else {
            throw Failure.invalidMaterial(index, "clearcoat factors must be between zero and one and normal scale must be finite")
        }
        if coat.clearcoatNormalTexture != nil, record.normalTexture == nil, !hasAuthoredTangentSpace {
            throw Failure.invalidMaterial(index, "a clearcoat normal texture needs NORMAL and TANGENT on every primitive or a base normal texture")
        }
        return GLTFScene.Material.Clearcoat(
            factor: factor, roughness: roughness,
            factorTexture: try texture(coat.clearcoatTexture, "clearcoat", index),
            roughnessTexture: try texture(coat.clearcoatRoughnessTexture, "clearcoat roughness", index),
            normalTexture: try texture(coat.clearcoatNormalTexture, "clearcoat normal", index),
            normalScale: normalScale
        )
    }

    private func resolveTransmission(
        _ record: MaterialRecord, material index: Int,
        texture: (TextureBinding?, String, Int) throws -> GLTFScene.Texture?
    ) throws -> GLTFScene.Material.Transmission? {
        guard let transmission = try extensionRecord(record.extensions?.transmission, name: "transmission", material: index) else {
            return nil
        }
        try rejectExtensions(transmission.extensions, context: "transmission on material \(index)")
        let factor = transmission.transmissionFactor ?? 0
        guard (0...1).contains(factor) else {
            throw Failure.invalidMaterial(index, "transmission factor must be between zero and one")
        }
        return GLTFScene.Material.Transmission(
            factor: factor, factorTexture: try texture(transmission.transmissionTexture, "transmission", index)
        )
    }

    private func resolveVolume(
        _ record: MaterialRecord, hasTransmission: Bool, material index: Int,
        texture: (TextureBinding?, String, Int) throws -> GLTFScene.Texture?
    ) throws -> GLTFScene.Material.Volume? {
        guard let volume = try extensionRecord(record.extensions?.volume, name: "volume", material: index) else {
            return nil
        }
        guard hasTransmission else {
            throw Failure.unsupportedFeature(
                "KHR_materials_volume without KHR_materials_transmission or KHR_materials_diffuse_transmission on material \(index)"
            )
        }
        try rejectExtensions(volume.extensions, context: "volume on material \(index)")
        let thickness = volume.thicknessFactor ?? 0
        let distance = volume.attenuationDistance ?? .infinity
        let color = volume.attenuationColor ?? [1, 1, 1]
        guard thickness.isFinite, thickness >= 0, distance > 0,
              volume.attenuationDistance?.isFinite != false,
              color.count == 3, color.allSatisfy({ (0...1).contains($0) }) else {
            throw Failure.invalidMaterial(index, "volume requires nonnegative thickness, a positive attenuation distance, and three linear color components between zero and one")
        }
        return GLTFScene.Material.Volume(
            thickness: thickness,
            thicknessTexture: try texture(volume.thicknessTexture, "thickness", index),
            attenuationDistance: distance, attenuationColor: SIMD3(color[0], color[1], color[2])
        )
    }

    private func resolveVolumeScatter(
        _ record: MaterialRecord, hasVolume: Bool, material index: Int,
        texture: (TextureBinding?, String, Int) throws -> GLTFScene.Texture?
    ) throws -> GLTFScene.Material.VolumeScatter? {
        guard let scatter = try extensionRecord(record.extensions?.volumeScatter, name: "volume scatter", material: index) else {
            return nil
        }
        guard hasVolume else {
            throw Failure.unsupportedFeature("KHR_materials_volume_scatter without KHR_materials_volume on material \(index)")
        }
        try rejectExtensions(scatter.extensions, context: "volume scatter on material \(index)")
        let color = scatter.multiscatterColorFactor ?? [0, 0, 0]
        let anisotropy = scatter.scatterAnisotropy ?? 0
        guard color.count == 3, color.allSatisfy({ (0...1).contains($0) }), anisotropy > -1, anisotropy < 1 else {
            throw Failure.invalidMaterial(
                index, "volume scatter needs three multiscatter color components between zero and one and an anisotropy between -1 and 1"
            )
        }
        return GLTFScene.Material.VolumeScatter(
            multiscatterColorFactor: SIMD3(color[0], color[1], color[2]),
            multiscatterColorTexture: try texture(scatter.multiscatterColorTexture, "multiscatter color", index),
            scatterAnisotropy: anisotropy
        )
    }

    private struct Layers {
        var emissiveStrength: Float = 1
        var diffuseTransmission: GLTFScene.Material.DiffuseTransmission?
        var diffuseRoughness: GLTFScene.Material.DiffuseRoughness?
        var specular: GLTFScene.Material.Specular?
        var sheen: GLTFScene.Material.Sheen?
        var anisotropy: GLTFScene.Material.Anisotropy?
        var iridescence: GLTFScene.Material.Iridescence?
        var dispersion: Float = 0
        var isUnlit = false
    }

    private func resolveLayers(
        _ record: MaterialRecord, hasTangentFrame: Bool, material index: Int,
        texture: (TextureBinding?, String, Int) throws -> GLTFScene.Texture?
    ) throws -> Layers {
        var layers = Layers()
        let extensions = record.extensions
        func invalid(_ reason: String) -> GLTFMeshLoader.Failure { .invalidMaterial(index, reason) }
        func color(_ values: [Float]?, default fallback: SIMD3<Float>, in range: ClosedRange<Float>) -> SIMD3<Float>? {
            guard let values else { return fallback }
            guard values.count == 3, values.allSatisfy({ range.contains($0) }) else { return nil }
            return SIMD3(values[0], values[1], values[2])
        }

        if let strength = try extensionRecord(extensions?.emissiveStrength, name: "emissive strength", material: index) {
            try rejectExtensions(strength.extensions, context: "emissive strength on material \(index)")
            let value = strength.emissiveStrength ?? 1
            guard value.isFinite, value >= 0 else { throw invalid("emissive strength must be finite and nonnegative") }
            layers.emissiveStrength = value
        }
        if let specular = try extensionRecord(extensions?.specular, name: "specular", material: index) {
            try rejectExtensions(specular.extensions, context: "specular on material \(index)")
            let factor = specular.specularFactor ?? 1
            guard (0...1).contains(factor),
                  let tint = color(specular.specularColorFactor, default: SIMD3(repeating: 1), in: 0...Float.greatestFiniteMagnitude) else {
                throw invalid("specular needs a factor between zero and one and three finite nonnegative color components")
            }
            layers.specular = .init(
                factor: factor, factorTexture: try texture(specular.specularTexture, "specular", index),
                colorFactor: tint, colorTexture: try texture(specular.specularColorTexture, "specular color", index)
            )
        }
        if let sheen = try extensionRecord(extensions?.sheen, name: "sheen", material: index) {
            try rejectExtensions(sheen.extensions, context: "sheen on material \(index)")
            let roughness = sheen.sheenRoughnessFactor ?? 0
            guard (0...1).contains(roughness), let tint = color(sheen.sheenColorFactor, default: .zero, in: 0...1) else {
                throw invalid("sheen color components and roughness must be between zero and one")
            }
            layers.sheen = .init(
                colorFactor: tint, colorTexture: try texture(sheen.sheenColorTexture, "sheen color", index),
                roughness: roughness, roughnessTexture: try texture(sheen.sheenRoughnessTexture, "sheen roughness", index)
            )
        }
        if let anisotropy = try extensionRecord(extensions?.anisotropy, name: "anisotropy", material: index) {
            try rejectExtensions(anisotropy.extensions, context: "anisotropy on material \(index)")
            let strength = anisotropy.anisotropyStrength ?? 0
            let rotation = anisotropy.anisotropyRotation ?? 0
            guard (0...1).contains(strength), rotation.isFinite else {
                throw invalid("anisotropy strength must be between zero and one and its rotation finite")
            }
            if strength > 0, !hasTangentFrame {
                throw invalid("anisotropy needs NORMAL and TANGENT or TEXCOORD_0 on every primitive")
            }
            layers.anisotropy = .init(
                strength: strength, rotation: rotation, texture: try texture(anisotropy.anisotropyTexture, "anisotropy", index)
            )
        }
        if let iridescence = try extensionRecord(extensions?.iridescence, name: "iridescence", material: index) {
            try rejectExtensions(iridescence.extensions, context: "iridescence on material \(index)")
            let factor = iridescence.iridescenceFactor ?? 0
            let ior = iridescence.iridescenceIor ?? 1.3
            let minimum = iridescence.iridescenceThicknessMinimum ?? 100
            let maximum = iridescence.iridescenceThicknessMaximum ?? 400
            guard (0...1).contains(factor), ior.isFinite, ior >= 1,
                  minimum.isFinite, minimum >= 0, maximum.isFinite, maximum >= 0 else {
                throw invalid("iridescence needs a factor between zero and one, an IOR of at least one, and finite nonnegative thicknesses")
            }
            layers.iridescence = .init(
                factor: factor, factorTexture: try texture(iridescence.iridescenceTexture, "iridescence", index),
                indexOfRefraction: ior, minimumThickness: minimum, maximumThickness: maximum,
                thicknessTexture: try texture(iridescence.iridescenceThicknessTexture, "iridescence thickness", index)
            )
        }
        if let dispersion = try extensionRecord(extensions?.dispersion, name: "dispersion", material: index) {
            try rejectExtensions(dispersion.extensions, context: "dispersion on material \(index)")
            let value = dispersion.dispersion ?? 0
            guard value.isFinite, value >= 0 else { throw invalid("dispersion must be finite and nonnegative") }
            layers.dispersion = value
        }
        if let unlit = try extensionRecord(extensions?.unlit, name: "unlit", material: index) {
            try rejectExtensions(unlit.extensions, context: "unlit on material \(index)")
            layers.isUnlit = true
        }
        if let diffuse = try extensionRecord(extensions?.diffuseTransmission, name: "diffuse transmission", material: index) {
            try rejectExtensions(diffuse.extensions, context: "diffuse transmission on material \(index)")
            let factor = diffuse.diffuseTransmissionFactor ?? 0
            guard (0...1).contains(factor),
                  let tint = color(diffuse.diffuseTransmissionColorFactor, default: SIMD3(repeating: 1), in: 0...1) else {
                throw invalid("diffuse transmission needs a factor and three color components between zero and one")
            }
            layers.diffuseTransmission = .init(
                factor: factor, factorTexture: try texture(diffuse.diffuseTransmissionTexture, "diffuse transmission", index),
                colorFactor: tint,
                colorTexture: try texture(diffuse.diffuseTransmissionColorTexture, "diffuse transmission color", index)
            )
        }
        if let rough = try extensionRecord(extensions?.diffuseRoughness, name: "diffuse roughness", material: index) {
            try rejectExtensions(rough.extensions, context: "diffuse roughness on material \(index)")
            let factor = rough.diffuseRoughnessFactor ?? 0
            guard (0...1).contains(factor) else { throw invalid("diffuse roughness must be between zero and one") }
            layers.diffuseRoughness = .init(
                factor: factor, texture: try texture(rough.diffuseRoughnessTexture, "diffuse roughness", index)
            )
        }
        return layers
    }

    private func resolveIOR(_ record: MaterialRecord, material index: Int) throws -> Float {
        guard let ior = try extensionRecord(record.extensions?.ior, name: "IOR", material: index) else { return 1.5 }
        try rejectExtensions(ior.extensions, context: "IOR on material \(index)")
        let value = ior.ior ?? 1.5
        guard value.isFinite, value == 0 || value >= 1 else {
            throw Failure.invalidMaterial(index, "IOR must be zero or at least one")
        }
        return value
    }

    private func extensionRecord<T>(
        _ result: Result<T, any Error>?, name: String, material: Int
    ) throws -> T? {
        guard let result else { return nil }
        guard case .success(let record) = result else {
            throw Failure.invalidMaterial(material, "\(name) requires an object with valid properties")
        }
        return record
    }

    private func sampling(_ index: Int?, texture: Int) throws -> GLTFScene.Sampling {
        let record: SamplerRecord?
        if let index {
            guard let samplers, samplers.indices.contains(index) else {
                throw Failure.invalidTexture(texture, "missing sampler \(index)")
            }
            record = samplers[index]
            try rejectExtensions(record?.extensions, context: "sampler \(index)")
        } else {
            record = nil
        }
        typealias Sampling = GLTFScene.Sampling
        guard let mag = Sampling.Magnification(rawValue: record?.magFilter ?? 9729),
              let min = Sampling.Minification(rawValue: record?.minFilter ?? 9987),
              let wrapS = Sampling.Wrap(rawValue: record?.wrapS ?? 10497),
              let wrapT = Sampling.Wrap(rawValue: record?.wrapT ?? 10497) else {
            throw Failure.invalidTexture(texture, "invalid sampler filter or wrapping mode")
        }
        return Sampling(magnification: mag, minification: min, wrapS: wrapS, wrapT: wrapT)
    }

    private static func imageKind(forExtension name: String) -> ImageSignature {
        switch name {
        case basisuExtension: .ktx2
        case avifExtension: .avif
        default: .webP
        }
    }

    /// Picks the first usable extension image in preference order, else the core source.
    /// Returns the core image as a fallback only when the selection is a KTX2 payload,
    /// whose block format may not be sampleable on the device used later.
    private func textureImages(
        _ record: TextureRecord, texture: Int,
        core: (Int) throws -> GLTFScene.Image,
        extensionImage: (String, Int) throws -> GLTFScene.Image
    ) throws -> (GLTFScene.Image, GLTFScene.Image?) {
        var reason: Failure?
        for name in Self.texturePreference {
            guard let entry = record.extensions?.sources[name] else { continue }
            guard let source = entry else { throw Failure.invalidTexture(texture, "\(name) needs an image source") }
            let kind = Self.imageKind(forExtension: name)
            guard kind.isDecodable else {
                reason = reason ?? .unsupportedFeature("\(kind.name) decoding for \(name) on texture \(texture)")
                continue
            }
            let image = try extensionImage(name, source)
            guard kind == .ktx2 else { return (image, nil) }
            do {
                _ = try KTX2Container(image.data)
            } catch TextureLoader.Failure.ktx2TranscodingRequired {
                do {
                    try BasisKTX2.validate(image.data)
                } catch TextureLoader.Failure.unsupportedKTX2(let feature) {
                    reason = reason ?? .unsupportedFeature("KTX2 \(feature) on image \(source)")
                    continue
                } catch let failure as TextureLoader.Failure {
                    if record.source != nil { continue }
                    throw Failure.invalidImage(source, failure.localizedDescription)
                }
            } catch TextureLoader.Failure.unsupportedKTX2(let feature) {
                reason = reason ?? .unsupportedFeature("KTX2 \(feature) on image \(source)")
                continue
            } catch let failure as TextureLoader.Failure {
                throw Failure.invalidImage(source, failure.localizedDescription)
            }
            return (image, try record.source.map(core))
        }
        if let source = record.source { return (try core(source), nil) }
        throw reason ?? Failure.invalidTexture(texture, "missing image source")
    }

    private func readImage(
        _ index: Int,
        readView: (Int) throws -> Data,
        readAdjacent: (String) throws -> Data
    ) throws -> GLTFScene.Image {
        guard let images, images.indices.contains(index) else { throw Failure.missingImage(index) }
        let record = images[index]
        try rejectExtensions(record.extensions, context: "image \(index)")

        let declared = record.mimeType?.lowercased()
        if let declared, ImageSignature(mimeType: declared) == nil {
            throw Failure.unsupportedFeature("MIME type \(declared) on image \(index)")
        }
        let data: Data
        var uriType: String?
        switch (record.uri, record.bufferView) {
        case (.some(let uri), nil):
            if uri.prefix(5).lowercased() == "data:" {
                let embedded = try dataURI(uri, at: index)
                data = embedded.data
                uriType = embedded.type
            } else {
                data = try readAdjacent(uri)
            }
        case (nil, .some(let view)):
            guard declared != nil else {
                throw Failure.invalidImage(index, "a buffer-view image needs a MIME type")
            }
            data = try readView(view)
        default:
            throw Failure.invalidImage(index, "an image needs exactly one URI or buffer view")
        }

        guard let actual = ImageSignature(data)?.mimeType else {
            throw Failure.invalidImage(index, "the bytes do not have a PNG, JPEG, WebP, AVIF, or KTX2 signature")
        }
        guard declared == nil || declared == actual, uriType == nil || uriType == actual else {
            throw Failure.invalidImage(index, "the MIME type does not match the image signature")
        }
        return GLTFScene.Image(sourceIndex: index, name: record.name, mimeType: actual, data: data)
    }

    private func dataURI(_ uri: String, at index: Int) throws -> (type: String, data: Data) {
        guard let comma = uri.firstIndex(of: ",") else {
            throw Failure.invalidImage(index, "missing data URI payload")
        }
        let header = uri[..<comma].lowercased()
        guard let type = ImageSignature.allCases.map(\.mimeType).first(where: { header == "data:\($0);base64" }) else {
            throw Failure.unsupportedFeature("non-base64 or unsupported image data URI on image \(index)")
        }
        guard let payload = String(uri[uri.index(after: comma)...]).removingPercentEncoding,
              let bytes = Data(base64Encoded: payload) else {
            throw Failure.invalidImage(index, "malformed base64 payload")
        }
        return (type, bytes)
    }

    private func rejectExtensions(_ extensions: [String: Present]?, context: String) throws {
        if let extensionName = extensions?.keys.sorted().first {
            throw Failure.unsupportedFeature("\(extensionName) on \(context)")
        }
    }

}
