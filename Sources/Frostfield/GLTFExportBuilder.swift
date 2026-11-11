import Foundation
import simd

/// A material whose maps already refer to exported texture indices.
struct GLTFExportMaterial: Hashable {
    struct TextureReference: Hashable {
        let index: Int
        /// The effective set. With a transform it is written to both `texCoord` fields.
        let coordinates: MeshTextureCoordinateSet
        /// Written as KHR_texture_transform unless it is the identity.
        var transform: PhysicallyBasedMaterial.TextureTransform = .identity
    }

    enum Alpha: Hashable {
        case opaque
        case mask(Float)
        case blend
    }

    struct Clearcoat: Hashable {
        let factor: Float
        let roughness: Float
        let factorTexture: TextureReference?
        let roughnessTexture: TextureReference?
        let normalTexture: TextureReference?
        let normalScale: Float
    }

    struct Transmission: Hashable {
        let factor: Float
        let texture: TextureReference?
    }

    struct Volume: Hashable {
        let thickness: Float
        let thicknessTexture: TextureReference?
        /// Nil represents an infinite distance.
        let attenuationDistance: Float?
        let attenuationColor: SIMD3<Float>
    }

    struct Specular: Hashable {
        let factor: Float
        let factorTexture: TextureReference?
        let colorFactor: SIMD3<Float>
        let colorTexture: TextureReference?
    }

    struct Sheen: Hashable {
        let colorFactor: SIMD3<Float>
        let colorTexture: TextureReference?
        let roughness: Float
        let roughnessTexture: TextureReference?
    }

    struct Anisotropy: Hashable {
        let strength: Float
        let rotation: Float
        let texture: TextureReference?
    }

    struct Iridescence: Hashable {
        let factor: Float
        let factorTexture: TextureReference?
        let indexOfRefraction: Float
        let minimumThickness: Float
        let maximumThickness: Float
        let thicknessTexture: TextureReference?
    }

    struct DiffuseTransmission: Hashable {
        let factor: Float
        let factorTexture: TextureReference?
        let colorFactor: SIMD3<Float>
        let colorTexture: TextureReference?
    }

    struct DiffuseRoughness: Hashable {
        let factor: Float
        let texture: TextureReference?
    }

    struct VolumeScatter: Hashable {
        let colorFactor: SIMD3<Float>
        let colorTexture: TextureReference?
        let anisotropy: Float
    }

    var name: String?
    var baseColorFactor: SIMD4<Float>
    var metallicFactor: Float
    var roughnessFactor: Float
    var baseColorTexture: TextureReference?
    var metallicRoughnessTexture: TextureReference?
    var normalTexture: TextureReference?
    var normalScale: Float
    var occlusionTexture: TextureReference?
    var occlusionStrength: Float
    var emissiveTexture: TextureReference?
    var emissiveFactor: SIMD3<Float>
    var alpha: Alpha
    var isDoubleSided: Bool
    var clearcoat: Clearcoat?
    var transmission: Transmission?
    var volume: Volume?
    var indexOfRefraction: Float
    /// KHR_materials_emissive_strength, written when it is not one.
    var emissiveStrength: Float = 1
    var specular: Specular?
    var sheen: Sheen?
    var anisotropy: Anisotropy?
    var iridescence: Iridescence?
    /// KHR_materials_dispersion, written when it is not zero.
    var dispersion: Float = 0
    /// KHR_materials_unlit.
    var isUnlit = false
    var diffuseTransmission: DiffuseTransmission?
    var diffuseRoughness: DiffuseRoughness?
    var volumeScatter: VolumeScatter?
}

/// Accumulates glTF JSON records and one little-endian binary buffer.
struct GLTFExportBuilder {
    private typealias Failure = GLTFExporter.Failure

    static let generator = "Frostfield"
    static let importerLimit = 64 * 1_024 * 1_024
    private static let arrayBuffer = 34_962
    private static let elementArrayBuffer = 34_963
    private static let floatComponent = 5_126

    struct Channel {
        let node: Int
        let path: String
        let times: [Float]
        let values: [Float]
        let components: Int
        let interpolation: String
    }

    private struct SamplerKey: Hashable {
        let magnification: Int
        let minification: Int
        let wrapS: Int
        let wrapT: Int
    }

    private struct TextureKey: Hashable {
        let image: Int?
        let extensionName: String?
        let extensionImage: Int?
        let sampler: Int
    }

    let format: GLTFExporter.Format
    private(set) var warnings: [GLTFExporter.Warning] = []
    private var binary = Data()
    private var bufferViews: [[String: Any]] = []
    private var accessors: [[String: Any]] = []
    private var nodes: [[String: Any]] = []
    private var meshes: [[String: Any]] = []
    private var materials: [[String: Any]] = []
    private var textures: [[String: Any]] = []
    private var images: [[String: Any]] = []
    private var samplers: [[String: Any]] = []
    private var skins: [[String: Any]] = []
    private var cameras: [[String: Any]] = []
    private var lights: [[String: Any]] = []
    private var animations: [[String: Any]] = []
    private var materialVariants: [String] = []
    private var extensionsUsed = Set<String>()
    private var extensionsRequired = Set<String>()
    private var imageIndices: [Data: Int] = [:]
    private var samplerIndices: [SamplerKey: Int] = [:]
    private var textureIndices: [TextureKey: Int] = [:]
    private var materialIndices: [GLTFExportMaterial: Int] = [:]
    private var timeAccessors: [[Float]: Int] = [:]

    init(format: GLTFExporter.Format) {
        self.format = format
    }

    var meshCount: Int { meshes.count }

    mutating func warn(_ warning: GLTFExporter.Warning) {
        if !warnings.contains(warning) { warnings.append(warning) }
    }

    // MARK: Binary data

    private mutating func view(_ bytes: Data, target: Int?, context: String) throws -> Int {
        guard !bytes.isEmpty else { throw Failure.invalidData("\(context) has no data.") }
        // Every view starts on a four-byte boundary, which covers all component widths.
        let padding = (4 - binary.count % 4) % 4
        guard bytes.count <= Int(UInt32.max) - 64 - binary.count - padding else { throw Failure.sizeLimit }
        binary.append(contentsOf: repeatElement(0, count: padding))
        var record: [String: Any] = ["buffer": 0, "byteLength": bytes.count]
        if binary.count > 0 { record["byteOffset"] = binary.count }
        if let target { record["target"] = target }
        binary.append(bytes)
        bufferViews.append(record)
        return bufferViews.count - 1
    }

    /// Writes tightly packed float components. Apple platforms store Float little-endian, as glTF requires.
    mutating func floats(
        _ values: [Float], type: String, components: Int, target: Int? = nil, bounds: Bool = false, context: String
    ) throws -> Int {
        guard !values.isEmpty, values.count.isMultiple(of: components) else {
            throw Failure.invalidData("\(context) needs a positive multiple of \(components) components.")
        }
        guard values.allSatisfy(\.isFinite) else { throw Failure.invalidData("\(context) contains a non-finite value.") }
        let count = values.count / components
        let data = values.withUnsafeBytes { Data($0) }
        var record: [String: Any] = [
            "bufferView": try view(data, target: target, context: context),
            "componentType": Self.floatComponent, "count": count, "type": type,
        ]
        if bounds {
            var minimum = Array(values[0..<components])
            var maximum = minimum
            for offset in stride(from: components, to: values.count, by: components) {
                for lane in 0..<components {
                    minimum[lane] = min(minimum[lane], values[offset + lane])
                    maximum[lane] = max(maximum[lane], values[offset + lane])
                }
            }
            record["min"] = minimum.map(Self.number)
            record["max"] = maximum.map(Self.number)
        }
        accessors.append(record)
        return accessors.count - 1
    }

    private mutating func integers(
        _ data: Data, component: Int, type: String, count: Int, target: Int?, context: String
    ) throws -> Int {
        accessors.append([
            "bufferView": try view(data, target: target, context: context),
            "componentType": component, "count": count, "type": type,
        ])
        return accessors.count - 1
    }

    /// Animation inputs need exact bounds, and identical key times share one accessor.
    private mutating func times(_ values: [Float], context: String) throws -> Int {
        if let existing = timeAccessors[values] { return existing }
        guard let first = values.first, let last = values.last, first.isFinite, first >= 0,
              zip(values, values.dropFirst()).allSatisfy({ $0.0 < $0.1 }), last.isFinite else {
            throw Failure.invalidData("\(context) needs finite, nonnegative, strictly increasing key times.")
        }
        let index = try floats(values, type: "SCALAR", components: 1, bounds: true, context: context)
        timeAccessors[values] = index
        return index
    }

    // MARK: Geometry

    /// Writes one triangle primitive. TEXCOORD_0 is always written; TEXCOORD_1 and TANGENT
    /// only when every vertex has them, and float COLOR_0 when any vertex has a color. Indices use UInt16 when every index fits.
    mutating func primitive(
        _ geometry: MeshGeometry, influences: [VertexSkinInfluence]?, mesh: Int, primitive: Int
    ) throws -> [String: Any] {
        let vertices = geometry.vertices
        let context = "Mesh \(mesh) primitive \(primitive)"
        guard !vertices.isEmpty, !geometry.indices.isEmpty, geometry.indices.count.isMultiple(of: 3) else {
            throw Failure.invalidData("\(context) needs vertices and whole triangles.")
        }
        if let index = geometry.indices.first(where: { Int($0) >= vertices.count }) {
            throw Failure.indexOutOfBounds(mesh: mesh, primitive: primitive, index: index)
        }

        var positions: [Float] = [], normals: [Float] = [], primaryUV: [Float] = []
        positions.reserveCapacity(vertices.count * 3)
        normals.reserveCapacity(vertices.count * 3)
        primaryUV.reserveCapacity(vertices.count * 2)
        for vertex in vertices {
            positions += [vertex.position.x, vertex.position.y, vertex.position.z]
            var normal = vertex.normal
            let lengthSquared = simd_length_squared(normal)
            guard lengthSquared.isFinite, lengthSquared > 0 else {
                throw Failure.invalidData("\(context) has a zero or non-finite normal.")
            }
            // glTF requires unit normals. Leave already-unit values bit-exact.
            if abs(lengthSquared - 1) > 0.0001 { normal = simd_normalize(normal) }
            normals += [normal.x, normal.y, normal.z]
            primaryUV += [vertex.textureCoordinate.x, vertex.textureCoordinate.y]
        }
        var attributes: [String: Int] = [
            "POSITION": try floats(positions, type: "VEC3", components: 3, target: Self.arrayBuffer, bounds: true, context: "\(context) POSITION"),
            "NORMAL": try floats(normals, type: "VEC3", components: 3, target: Self.arrayBuffer, context: "\(context) NORMAL"),
            "TEXCOORD_0": try floats(primaryUV, type: "VEC2", components: 2, target: Self.arrayBuffer, context: "\(context) TEXCOORD_0"),
        ]

        let secondary = vertices.compactMap(\.secondaryTextureCoordinate)
        if secondary.count == vertices.count {
            attributes["TEXCOORD_1"] = try floats(
                secondary.flatMap { [$0.x, $0.y] }, type: "VEC2", components: 2, target: Self.arrayBuffer,
                context: "\(context) TEXCOORD_1"
            )
        } else if !secondary.isEmpty {
            warn(.attributeDropped(mesh: mesh, primitive: primitive, attribute: "TEXCOORD_1"))
        }
        let tangents = vertices.compactMap(\.tangent)
        if tangents.count == vertices.count {
            guard tangents.allSatisfy(MeshGeometry.isValidTangent) else {
                throw Failure.invalidData("\(context) needs unit tangents with a handedness of -1 or +1.")
            }
            attributes["TANGENT"] = try floats(
                tangents.flatMap { [$0.x, $0.y, $0.z, $0.w] }, type: "VEC4", components: 4, target: Self.arrayBuffer,
                context: "\(context) TANGENT"
            )
        } else if !tangents.isEmpty {
            warn(.attributeDropped(mesh: mesh, primitive: primitive, attribute: "TANGENT"))
        }
        if geometry.hasVertexColors {
            // Vertices without a color draw as opaque white, so they export that way.
            attributes["COLOR_0"] = try floats(
                vertices.flatMap { vertex -> [Float] in
                    let color = vertex.color ?? SIMD4(repeating: 1)
                    return [color.x, color.y, color.z, color.w]
                }, type: "VEC4", components: 4, target: Self.arrayBuffer, context: "\(context) COLOR_0"
            )
        }

        if let influences {
            guard influences.count == vertices.count else {
                throw Failure.invalidData("\(context) needs one skin influence per vertex.")
            }
            let wide = influences.contains { $0.joints.max() > UInt16(UInt8.max) }
            var joints = Data(capacity: influences.count * (wide ? 8 : 4))
            for influence in influences {
                for lane in 0..<4 {
                    let joint = influence.joints[lane]
                    if wide {
                        withUnsafeBytes(of: joint.littleEndian) { joints.append(contentsOf: $0) }
                    } else {
                        joints.append(UInt8(joint))
                    }
                }
            }
            attributes["JOINTS_0"] = try integers(
                joints, component: wide ? 5_123 : 5_121, type: "VEC4", count: influences.count,
                target: Self.arrayBuffer, context: "\(context) JOINTS_0"
            )
            attributes["WEIGHTS_0"] = try floats(
                influences.flatMap { [$0.weights.x, $0.weights.y, $0.weights.z, $0.weights.w] }, type: "VEC4",
                components: 4, target: Self.arrayBuffer, context: "\(context) WEIGHTS_0"
            )
        }

        let indexData: Data
        let component: Int
        if vertices.count <= Int(UInt16.max) {
            // The largest index is then at most 65,534, below the reserved restart value.
            indexData = geometry.indices.map { UInt16($0).littleEndian }.withUnsafeBytes { Data($0) }
            component = 5_123
        } else {
            indexData = geometry.indices.map(\.littleEndian).withUnsafeBytes { Data($0) }
            component = 5_125
        }
        let indices = try integers(
            indexData, component: component, type: "SCALAR", count: geometry.indices.count,
            target: Self.elementArrayBuffer, context: "\(context) indices"
        )
        return ["attributes": attributes, "indices": indices]
    }

    mutating func addMesh(name: String?, primitives: [[String: Any]]) {
        var record: [String: Any] = ["primitives": primitives]
        if let name { record["name"] = name }
        meshes.append(record)
    }

    mutating func addSkin(name: String?, joints: [Int], inverseBindMatrices: [simd_float4x4], skeleton: Int?) throws {
        guard !joints.isEmpty, joints.count == inverseBindMatrices.count else {
            throw Failure.invalidData("A skin needs one inverse bind matrix per joint.")
        }
        // simd matrices store columns, matching glTF's column-major MAT4 layout.
        let values = inverseBindMatrices.flatMap { matrix in
            [matrix.columns.0, matrix.columns.1, matrix.columns.2, matrix.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
        }
        var record: [String: Any] = [
            "joints": joints,
            "inverseBindMatrices": try floats(values, type: "MAT4", components: 16, context: "Skin inverse bind matrices"),
        ]
        if let name { record["name"] = name }
        if let skeleton { record["skeleton"] = skeleton }
        skins.append(record)
    }

    // MARK: Materials and textures

    mutating func image(_ data: Data, mimeType: String, name: String?) throws -> Int {
        if let existing = imageIndices[data] { return existing }
        var record: [String: Any] = ["mimeType": mimeType]
        if let name, !name.isEmpty { record["name"] = name }
        switch format {
        case .glb: record["bufferView"] = try view(data, target: nil, context: "Image")
        case .gltf: record["uri"] = "data:\(mimeType);base64,\(data.base64EncodedString())"
        }
        images.append(record)
        imageIndices[data] = images.count - 1
        return images.count - 1
    }

    mutating func sampler(_ sampling: GLTFScene.Sampling) -> Int {
        let key = SamplerKey(
            magnification: sampling.magnification.rawValue, minification: sampling.minification.rawValue,
            wrapS: sampling.wrapS.rawValue, wrapT: sampling.wrapT.rawValue
        )
        if let existing = samplerIndices[key] { return existing }
        samplers.append([
            "magFilter": key.magnification, "minFilter": key.minification, "wrapS": key.wrapS, "wrapT": key.wrapT,
        ])
        samplerIndices[key] = samplers.count - 1
        return samplers.count - 1
    }

    mutating func texture(image: Int, sampler: Int) -> Int {
        texture(source: image, extensionImage: nil, sampler: sampler)
    }

    /// A texture whose core `source` is a PNG or JPEG image, and optionally a texture
    /// extension (KHR_texture_basisu, EXT_texture_webp, or EXT_texture_avif) naming
    /// another image. Without a core source the extension is also required, as each
    /// of those extension specifications prescribes for textures without a fallback.
    mutating func texture(source: Int?, extensionImage: (name: String, image: Int)?, sampler: Int) -> Int {
        let key = TextureKey(
            image: source, extensionName: extensionImage?.name, extensionImage: extensionImage?.image, sampler: sampler
        )
        if let existing = textureIndices[key] { return existing }
        var record: [String: Any] = ["sampler": sampler]
        record["source"] = source
        if let extensionImage {
            record["extensions"] = [extensionImage.name: ["source": extensionImage.image]]
            extensionsUsed.insert(extensionImage.name)
            if source == nil { extensionsRequired.insert(extensionImage.name) }
        }
        textures.append(record)
        textureIndices[key] = textures.count - 1
        return textures.count - 1
    }

    /// The texture extension that carries an image of this media type, or nil for the
    /// core PNG and JPEG types.
    static func textureExtension(forMIMEType mimeType: String) throws -> String? {
        switch mimeType {
        case "image/png", "image/jpeg": nil
        case "image/ktx2": GLTFMaterialDocument.basisuExtension
        case "image/webp": GLTFMaterialDocument.webPExtension
        case "image/avif": GLTFMaterialDocument.avifExtension
        default: throw Failure.invalidData("Images must be PNG, JPEG, KTX2, WebP, or AVIF, not \(mimeType).")
        }
    }

    /// Whether a KTX2 payload is Basis Universal (BasisLZ/ETC1S or UASTC), the only
    /// payloads KHR_texture_basisu specifies. Those leave `vkFormat` undefined (zero).
    static func isBasisUniversal(_ ktx2: Data) -> Bool {
        guard ktx2.count >= 16 else { return false }
        return ktx2.dropFirst(12).prefix(4).allSatisfy { $0 == 0 }
    }

    /// Writes the root KHR_materials_variants names.
    mutating func setMaterialVariants(_ names: [String]) {
        materialVariants = names
        if !names.isEmpty { extensionsUsed.insert(GLTFMaterialVariantsDocument.extensionName) }
    }

    /// A primitive's KHR_materials_variants extension from exported material indices
    /// keyed by variant index, with one mapping per material. Nil when nothing is mapped.
    func variantMappings(_ materials: [Int: Int]) throws -> [String: Any]? {
        guard !materials.isEmpty else { return nil }
        if let variant = materials.keys.sorted().first(where: { !materialVariants.indices.contains($0) }) {
            throw Failure.missingReference("Material variant \(variant) is not in the scene's variant list.")
        }
        var variantsByMaterial: [Int: [Int]] = [:]
        for (variant, material) in materials.sorted(by: { $0.key < $1.key }) {
            variantsByMaterial[material, default: []].append(variant)
        }
        let mappings = variantsByMaterial
            .sorted { $0.value[0] < $1.value[0] }
            .map { ["variants": $0.value, "material": $0.key] as [String: Any] }
        return [GLTFMaterialVariantsDocument.extensionName: ["mappings": mappings]]
    }

    /// Validates the glTF factor ranges and returns the material index. Deduplication shares
    /// one index between identical materials.
    mutating func material(_ material: GLTFExportMaterial, deduplicate: Bool) throws -> Int {
        if deduplicate, let existing = materialIndices[material] { return existing }
        let unit: ClosedRange<Float> = 0...1
        let context = "Material \(material.name ?? String(materials.count))"
        guard (0..<4).allSatisfy({ unit.contains(material.baseColorFactor[$0]) }),
              unit.contains(material.metallicFactor), unit.contains(material.roughnessFactor),
              (0..<3).allSatisfy({ unit.contains(material.emissiveFactor[$0]) }),
              material.normalScale.isFinite, unit.contains(material.occlusionStrength) else {
            throw Failure.invalidData("\(context) has factors outside zero to one or a non-finite normal scale.")
        }

        var surface: [String: Any] = [
            "metallicFactor": Self.number(material.metallicFactor),
            "roughnessFactor": Self.number(material.roughnessFactor),
        ]
        if material.baseColorFactor != SIMD4(repeating: 1) { surface["baseColorFactor"] = Self.numbers(material.baseColorFactor) }
        surface["baseColorTexture"] = info(material.baseColorTexture)
        surface["metallicRoughnessTexture"] = info(material.metallicRoughnessTexture)
        var record: [String: Any] = ["pbrMetallicRoughness": surface]
        if let name = material.name { record["name"] = name }
        record["normalTexture"] = info(material.normalTexture, scale: material.normalScale)
        if var occlusion = info(material.occlusionTexture) {
            if material.occlusionStrength != 1 { occlusion["strength"] = Self.number(material.occlusionStrength) }
            record["occlusionTexture"] = occlusion
        }
        record["emissiveTexture"] = info(material.emissiveTexture)
        if material.emissiveFactor != .zero { record["emissiveFactor"] = Self.numbers(material.emissiveFactor) }
        switch material.alpha {
        case .opaque: break
        case .mask(let cutoff):
            guard cutoff.isFinite, cutoff >= 0 else { throw Failure.invalidData("\(context) needs a nonnegative alpha cutoff.") }
            record["alphaMode"] = "MASK"
            record["alphaCutoff"] = Self.number(cutoff)
        case .blend: record["alphaMode"] = "BLEND"
        }
        if material.isDoubleSided { record["doubleSided"] = true }

        var extensions: [String: Any] = [:]
        if let coat = material.clearcoat {
            guard unit.contains(coat.factor), unit.contains(coat.roughness), coat.normalScale.isFinite else {
                throw Failure.invalidData("\(context) clearcoat factors must be in zero to one.")
            }
            var layer: [String: Any] = [
                "clearcoatFactor": Self.number(coat.factor), "clearcoatRoughnessFactor": Self.number(coat.roughness),
            ]
            layer["clearcoatTexture"] = info(coat.factorTexture)
            layer["clearcoatRoughnessTexture"] = info(coat.roughnessTexture)
            layer["clearcoatNormalTexture"] = info(coat.normalTexture, scale: coat.normalScale)
            extensions[GLTFMaterialDocument.clearcoatExtension] = layer
        }
        if let transmission = material.transmission {
            guard unit.contains(transmission.factor) else {
                throw Failure.invalidData("\(context) transmission factor must be in zero to one.")
            }
            var layer: [String: Any] = ["transmissionFactor": Self.number(transmission.factor)]
            layer["transmissionTexture"] = info(transmission.texture)
            extensions[GLTFMaterialDocument.transmissionExtension] = layer
        }
        if let volume = material.volume {
            // KHR_materials_volume extends a transmissive surface.
            guard material.transmission != nil || material.diffuseTransmission != nil,
                  volume.thickness.isFinite, volume.thickness >= 0,
                  volume.attenuationDistance.map({ $0.isFinite && $0 > 0 }) ?? true,
                  (0..<3).allSatisfy({ unit.contains(volume.attenuationColor[$0]) }) else {
                throw Failure.invalidData("\(context) volume needs transmission and valid thickness and attenuation.")
            }
            var layer: [String: Any] = ["thicknessFactor": Self.number(volume.thickness)]
            layer["thicknessTexture"] = info(volume.thicknessTexture)
            layer["attenuationDistance"] = volume.attenuationDistance.map(Self.number)
            if volume.attenuationColor != SIMD3(repeating: 1) {
                layer["attenuationColor"] = Self.numbers(volume.attenuationColor)
            }
            extensions[GLTFMaterialDocument.volumeExtension] = layer
        }
        if material.indexOfRefraction != 1.5 {
            guard material.indexOfRefraction.isFinite, material.indexOfRefraction == 0 || material.indexOfRefraction >= 1 else {
                throw Failure.invalidData("\(context) IOR must be zero or at least one.")
            }
            extensions[GLTFMaterialDocument.iorExtension] = ["ior": Self.number(material.indexOfRefraction)]
        }
        try layerExtensions(material, context: context).forEach { extensions[$0.key] = $0.value }
        if !extensions.isEmpty {
            record["extensions"] = extensions
            extensionsUsed.formUnion(extensions.keys)
        }
        materials.append(record)
        if deduplicate { materialIndices[material] = materials.count - 1 }
        return materials.count - 1
    }

    /// Material layer records with the extension property names and the importer's ranges.
    private mutating func layerExtensions(_ material: GLTFExportMaterial, context: String) throws -> [String: Any] {
        let unit: ClosedRange<Float> = 0...1
        func nonnegative(_ value: Float) -> Bool { value.isFinite && value >= 0 }
        var extensions: [String: Any] = [:]
        if material.emissiveStrength != 1 {
            guard nonnegative(material.emissiveStrength) else {
                throw Failure.invalidData("\(context) emissive strength must be finite and nonnegative.")
            }
            extensions[GLTFMaterialDocument.emissiveStrengthExtension] = [
                "emissiveStrength": Self.number(material.emissiveStrength),
            ]
        }
        if let specular = material.specular {
            guard unit.contains(specular.factor), (0..<3).allSatisfy({ nonnegative(specular.colorFactor[$0]) }) else {
                throw Failure.invalidData("\(context) specular needs a factor in zero to one and a finite, nonnegative color.")
            }
            var layer: [String: Any] = [
                "specularFactor": Self.number(specular.factor), "specularColorFactor": Self.numbers(specular.colorFactor),
            ]
            layer["specularTexture"] = info(specular.factorTexture)
            layer["specularColorTexture"] = info(specular.colorTexture)
            extensions[GLTFMaterialDocument.specularExtension] = layer
        }
        if let sheen = material.sheen {
            guard (0..<3).allSatisfy({ unit.contains(sheen.colorFactor[$0]) }), unit.contains(sheen.roughness) else {
                throw Failure.invalidData("\(context) sheen color and roughness must be in zero to one.")
            }
            var layer: [String: Any] = [
                "sheenColorFactor": Self.numbers(sheen.colorFactor), "sheenRoughnessFactor": Self.number(sheen.roughness),
            ]
            layer["sheenColorTexture"] = info(sheen.colorTexture)
            layer["sheenRoughnessTexture"] = info(sheen.roughnessTexture)
            extensions[GLTFMaterialDocument.sheenExtension] = layer
        }
        if let anisotropy = material.anisotropy {
            guard unit.contains(anisotropy.strength), anisotropy.rotation.isFinite else {
                throw Failure.invalidData("\(context) anisotropy needs a strength in zero to one and a finite rotation.")
            }
            var layer: [String: Any] = [
                "anisotropyStrength": Self.number(anisotropy.strength), "anisotropyRotation": Self.number(anisotropy.rotation),
            ]
            layer["anisotropyTexture"] = info(anisotropy.texture)
            extensions[GLTFMaterialDocument.anisotropyExtension] = layer
        }
        if let film = material.iridescence {
            guard unit.contains(film.factor), film.indexOfRefraction.isFinite, film.indexOfRefraction >= 1,
                  nonnegative(film.minimumThickness), nonnegative(film.maximumThickness) else {
                throw Failure.invalidData(
                    "\(context) iridescence needs a factor in zero to one, an IOR of at least one, and nonnegative thicknesses."
                )
            }
            var layer: [String: Any] = [
                "iridescenceFactor": Self.number(film.factor), "iridescenceIor": Self.number(film.indexOfRefraction),
                "iridescenceThicknessMinimum": Self.number(film.minimumThickness),
                "iridescenceThicknessMaximum": Self.number(film.maximumThickness),
            ]
            layer["iridescenceTexture"] = info(film.factorTexture)
            layer["iridescenceThicknessTexture"] = info(film.thicknessTexture)
            extensions[GLTFMaterialDocument.iridescenceExtension] = layer
        }
        if material.dispersion != 0 {
            guard nonnegative(material.dispersion) else {
                throw Failure.invalidData("\(context) dispersion must be finite and nonnegative.")
            }
            extensions[GLTFMaterialDocument.dispersionExtension] = ["dispersion": Self.number(material.dispersion)]
        }
        if let layer = material.diffuseTransmission {
            guard unit.contains(layer.factor), (0..<3).allSatisfy({ unit.contains(layer.colorFactor[$0]) }) else {
                throw Failure.invalidData("\(context) diffuse transmission needs a factor and color in zero to one.")
            }
            var values: [String: Any] = [
                "diffuseTransmissionFactor": Self.number(layer.factor),
                "diffuseTransmissionColorFactor": Self.numbers(layer.colorFactor),
            ]
            values["diffuseTransmissionTexture"] = info(layer.factorTexture)
            values["diffuseTransmissionColorTexture"] = info(layer.colorTexture)
            extensions[GLTFMaterialDocument.diffuseTransmissionExtension] = values
        }
        if let layer = material.diffuseRoughness {
            guard unit.contains(layer.factor) else {
                throw Failure.invalidData("\(context) diffuse roughness must be between zero and one.")
            }
            var values: [String: Any] = ["diffuseRoughnessFactor": Self.number(layer.factor)]
            values["diffuseRoughnessTexture"] = info(layer.texture)
            extensions[GLTFMaterialDocument.diffuseRoughnessExtension] = values
        }
        if let layer = material.volumeScatter {
            guard material.volume != nil, layer.anisotropy > -1, layer.anisotropy < 1,
                  (0..<3).allSatisfy({ unit.contains(layer.colorFactor[$0]) }) else {
                throw Failure.invalidData("\(context) scattering needs a volume, color in zero to one, and anisotropy between -1 and 1.")
            }
            var values: [String: Any] = [
                "multiscatterColorFactor": Self.numbers(layer.colorFactor),
                "scatterAnisotropy": Self.number(layer.anisotropy),
            ]
            values["multiscatterColorTexture"] = info(layer.colorTexture)
            extensions[GLTFMaterialDocument.volumeScatterExtension] = values
        }
        if material.isUnlit { extensions[GLTFMaterialDocument.unlitExtension] = [String: Any]() }
        return extensions
    }

    /// A textureInfo record. A non-identity transform adds KHR_texture_transform; its
    /// `texCoord` repeats the effective set, as the extension allows for clients without it.
    private mutating func info(_ texture: GLTFExportMaterial.TextureReference?, scale: Float? = nil) -> [String: Any]? {
        guard let texture else { return nil }
        var info: [String: Any] = ["index": texture.index]
        let coordinates = texture.coordinates == .primary ? nil : Int(texture.coordinates.rawValue)
        info["texCoord"] = coordinates
        if let scale, scale != 1 { info["scale"] = Self.number(scale) }
        let transform = texture.transform
        if transform != .identity {
            var values: [String: Any] = [:]
            if transform.offset != .zero { values["offset"] = [Self.number(transform.offset.x), Self.number(transform.offset.y)] }
            if transform.rotation != 0 { values["rotation"] = Self.number(transform.rotation) }
            if transform.scale != SIMD2(repeating: 1) {
                values["scale"] = [Self.number(transform.scale.x), Self.number(transform.scale.y)]
            }
            values["texCoord"] = coordinates
            info["extensions"] = [GLTFMaterialDocument.textureTransformExtension: values]
            extensionsUsed.insert(GLTFMaterialDocument.textureTransformExtension)
        }
        return info
    }

    // MARK: Lights, cameras, nodes, and animation

    @discardableResult
    mutating func addLight(_ light: PunctualLight, name: String?) throws -> Int {
        var record: [String: Any] = [
            "color": Self.numbers(light.color), "intensity": Self.number(light.intensity),
        ]
        func range(_ value: Float?) { record["range"] = value.map(Self.number) }
        switch light.kind {
        case .directional:
            record["type"] = "directional"
        case .point(let value):
            record["type"] = "point"
            range(value)
        case .spot(let value, let inner, let outer):
            record["type"] = "spot"
            range(value)
            record["spot"] = ["innerConeAngle": Self.number(inner), "outerConeAngle": Self.number(outer)]
        case .rectangle:
            throw Failure.invalidData("KHR_lights_punctual cannot represent rectangle lights.")
        }
        if let name { record["name"] = name }
        lights.append(record)
        extensionsUsed.insert(GLTFLightDocument.extensionName)
        return lights.count - 1
    }

    @discardableResult
    mutating func addCamera(_ projection: GLTFScene.Camera.Projection, name: String?) throws -> Int {
        var record: [String: Any]
        switch projection {
        case .perspective(let fieldOfView, let aspectRatio, let near, let far):
            guard fieldOfView.isFinite, fieldOfView > 0, fieldOfView < .pi, near.isFinite, near > 0,
                  far.map({ $0.isFinite && $0 > near }) ?? true, aspectRatio.map({ $0.isFinite && $0 > 0 }) ?? true else {
                throw Failure.invalidData("A perspective camera needs a field of view in zero to pi and valid clip planes.")
            }
            var values: [String: Any] = ["yfov": Self.number(fieldOfView), "znear": Self.number(near)]
            values["zfar"] = far.map(Self.number)
            values["aspectRatio"] = aspectRatio.map(Self.number)
            record = ["type": "perspective", "perspective": values]
        case .orthographic(let halfWidth, let halfHeight, let near, let far):
            guard [halfWidth, halfHeight, near, far].allSatisfy(\.isFinite), halfWidth > 0, halfHeight > 0,
                  near >= 0, far > near else {
                throw Failure.invalidData("An orthographic camera needs positive extents and valid clip planes.")
            }
            record = ["type": "orthographic", "orthographic": [
                "xmag": Self.number(halfWidth), "ymag": Self.number(halfHeight),
                "znear": Self.number(near), "zfar": Self.number(far),
            ]]
        }
        if let name { record["name"] = name }
        cameras.append(record)
        return cameras.count - 1
    }

    /// Appends a node with TRS components, omitting identity values. Instances are written
    /// as EXT_mesh_gpu_instancing TRANSLATION, ROTATION, and SCALE float accessors.
    @discardableResult
    mutating func addNode(
        name: String?, transform: SceneTransform, context: String, children: [Int] = [],
        mesh: Int? = nil, skin: Int? = nil, light: Int? = nil, camera: Int? = nil, instances: [SceneTransform]? = nil
    ) throws -> Int {
        guard Self.isValid(transform) else { throw Failure.invalidTransform(context) }
        let t = transform.translation, r = transform.rotation.vector, s = transform.scale
        var record: [String: Any] = [:]
        if let name { record["name"] = name }
        if !children.isEmpty { record["children"] = children }
        if t != .zero { record["translation"] = Self.numbers(t) }
        if r != SIMD4(0, 0, 0, 1) { record["rotation"] = Self.numbers(r) }
        if s != SIMD3(repeating: 1) { record["scale"] = Self.numbers(s) }
        record["mesh"] = mesh
        record["skin"] = skin
        record["camera"] = camera
        var extensions: [String: Any] = [:]
        if let light { extensions[GLTFLightDocument.extensionName] = ["light": light] }
        if let instances {
            guard mesh != nil else { throw Failure.invalidData("The \(context) has instances but no mesh.") }
            extensions[GLTFHierarchy.gpuInstancingExtension] = try instancing(instances, context: context)
        }
        if !extensions.isEmpty { record["extensions"] = extensions }
        nodes.append(record)
        return nodes.count - 1
    }

    private mutating func instancing(_ instances: [SceneTransform], context: String) throws -> [String: Any] {
        guard !instances.isEmpty else { throw Failure.invalidData("The \(context) has an empty instance list.") }
        for (index, instance) in instances.enumerated() where !Self.isValid(instance) {
            throw Failure.invalidTransform("\(context) instance \(index)")
        }
        let translations = try floats(
            instances.flatMap { [$0.translation.x, $0.translation.y, $0.translation.z] }, type: "VEC3", components: 3,
            context: "\(context) instance translations"
        )
        let rotations = try floats(
            instances.flatMap { [$0.rotation.vector.x, $0.rotation.vector.y, $0.rotation.vector.z, $0.rotation.vector.w] },
            type: "VEC4", components: 4, context: "\(context) instance rotations"
        )
        let scales = try floats(
            instances.flatMap { [$0.scale.x, $0.scale.y, $0.scale.z] }, type: "VEC3", components: 3,
            context: "\(context) instance scales"
        )
        extensionsUsed.insert(GLTFHierarchy.gpuInstancingExtension)
        return ["attributes": ["TRANSLATION": translations, "ROTATION": rotations, "SCALE": scales]]
    }

    /// Finite TRS values, a unit rotation, and an invertible matrix.
    private static func isValid(_ transform: SceneTransform) -> Bool {
        let t = transform.translation, r = transform.rotation.vector, s = transform.scale
        let matrix = transform.matrix
        let determinant = simd_determinant(matrix)
        return [t.x, t.y, t.z, r.x, r.y, r.z, r.w, s.x, s.y, s.z].allSatisfy(\.isFinite)
            && abs(simd_length_squared(r) - 1) <= 0.0001 && determinant.isFinite && determinant != 0
            && isFinite(simd_inverse(matrix))
    }

    mutating func addAnimation(name: String?, channels: [Channel]) throws {
        guard !channels.isEmpty else { return }
        var samplerRecords: [[String: Any]] = []
        var channelRecords: [[String: Any]] = []
        for channel in channels {
            let context = "Animation \(animations.count) \(channel.path) channel"
            let keysPerTime = channel.interpolation == "CUBICSPLINE" ? 3 : 1
            guard channel.values.count == channel.times.count * keysPerTime * channel.components else {
                throw Failure.invalidData("\(context) needs \(keysPerTime) output values per key time.")
            }
            let input = try times(channel.times, context: "\(context) times")
            let output = try floats(
                channel.values, type: "VEC\(channel.components)", components: channel.components, context: "\(context) values"
            )
            samplerRecords.append(["input": input, "output": output, "interpolation": channel.interpolation])
            channelRecords.append([
                "sampler": samplerRecords.count - 1, "target": ["node": channel.node, "path": channel.path],
            ])
        }
        var record: [String: Any] = ["channels": channelRecords, "samplers": samplerRecords]
        if let name { record["name"] = name }
        animations.append(record)
    }

    // MARK: Encoding

    /// Encodes the document. A GLB pads JSON with spaces and BIN with zeros to four-byte boundaries.
    mutating func encode(sceneName: String?, roots: [Int]) throws -> Data {
        var scene: [String: Any] = [:]
        if !roots.isEmpty { scene["nodes"] = roots }
        if let sceneName { scene["name"] = sceneName }
        var document: [String: Any] = [
            "asset": ["version": "2.0", "generator": Self.generator], "scene": 0, "scenes": [scene],
        ]
        let sections: [(String, [[String: Any]])] = [
            ("nodes", nodes), ("meshes", meshes), ("materials", materials), ("textures", textures),
            ("images", images), ("samplers", samplers), ("skins", skins), ("cameras", cameras),
            ("animations", animations), ("accessors", accessors), ("bufferViews", bufferViews),
        ]
        for (key, records) in sections where !records.isEmpty { document[key] = records }
        var extensions: [String: Any] = [:]
        if !lights.isEmpty { extensions[GLTFLightDocument.extensionName] = ["lights": lights] }
        if !materialVariants.isEmpty {
            extensions[GLTFMaterialVariantsDocument.extensionName] = ["variants": materialVariants.map { ["name": $0] }]
        }
        if !extensions.isEmpty { document["extensions"] = extensions }
        if !extensionsUsed.isEmpty { document["extensionsUsed"] = extensionsUsed.sorted() }
        if !extensionsRequired.isEmpty { document["extensionsRequired"] = extensionsRequired.sorted() }

        // The importer accepts up to three zero bytes after the declared buffer length.
        let padding = (4 - binary.count % 4) % 4
        let chunk = binary + Data(repeating: 0, count: padding)
        if !binary.isEmpty {
            var buffer: [String: Any] = ["byteLength": binary.count]
            if format == .gltf { buffer["uri"] = "data:application/octet-stream;base64,\(binary.base64EncodedString())" }
            document["buffers"] = [buffer]
        }

        guard JSONSerialization.isValidJSONObject(document) else {
            throw Failure.invalidData("The document contains a value JSON cannot represent.")
        }
        var json = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys, .withoutEscapingSlashes])
        let output: Data
        switch format {
        case .gltf:
            output = json
        case .glb:
            json.append(contentsOf: repeatElement(0x20, count: (4 - json.count % 4) % 4))
            let total = 12 + 8 + json.count + (binary.isEmpty ? 0 : 8 + chunk.count)
            guard total <= Int(UInt32.max) else { throw Failure.sizeLimit }
            var data = Data(capacity: total)
            func word(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
            word(0x4654_6C67)
            word(2)
            word(UInt32(total))
            word(UInt32(json.count))
            word(0x4E4F_534A)
            data.append(json)
            if !binary.isEmpty {
                word(UInt32(chunk.count))
                word(0x004E_4942)
                data.append(chunk)
            }
            output = data
        }
        if output.count > Self.importerLimit { warn(.exceedsImporterLimit(bytes: output.count)) }
        return output
    }

    /// Returns a Double that decodes to the same Float, preferring Float's shortest decimal form.
    static func number(_ value: Float) -> Double {
        if let shortest = Double(value.description), Float(shortest) == value { return shortest }
        return Double(value)
    }

    private static func numbers(_ value: SIMD3<Float>) -> [Double] { [number(value.x), number(value.y), number(value.z)] }
    private static func numbers(_ value: SIMD4<Float>) -> [Double] {
        [number(value.x), number(value.y), number(value.z), number(value.w)]
    }

    private static func isFinite(_ matrix: simd_float4x4) -> Bool {
        let (x, y, z, w) = matrix.columns
        return [x, y, z, w].allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }
    }
}
