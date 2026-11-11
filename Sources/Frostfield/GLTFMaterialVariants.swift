import Foundation

extension GLTFScene {
    /// Returns this scene with each primitive's material replaced by its
    /// KHR_materials_variants mapping for the variant at `index`, or by its default
    /// material where the variant has no mapping. Nil restores the default
    /// materials. Prepare the returned scene, for example with `PreparedGLTFScene`,
    /// to render the variant; prepared renderers keep the materials they were made with.
    ///
    /// Throws `GLTFMeshLoader.Failure.missingMaterialVariant` for an index outside
    /// `materialVariants`.
    public func selectingMaterialVariant(_ index: Int?) throws -> GLTFScene {
        if let index, !materialVariants.indices.contains(index) {
            throw GLTFMeshLoader.Failure.missingMaterialVariant("\(index)")
        }
        var scene = self
        scene.meshes = meshes.map { mesh in
            var mesh = mesh
            let defaults = mesh.defaultPrimitiveMaterialIndices ?? mesh.primitiveMaterialIndices
            mesh.defaultPrimitiveMaterialIndices = defaults
            mesh.primitiveMaterialIndices = defaults.enumerated().map { primitive, material in
                guard let index, mesh.primitiveMaterialVariants.indices.contains(primitive) else { return material }
                return mesh.primitiveMaterialVariants[primitive][index] ?? material
            }
            return mesh
        }
        scene.selectedMaterialVariant = index
        return scene
    }

    /// Selects the first variant with this name, as `selectingMaterialVariant(_:)` does.
    public func selectingMaterialVariant(named name: String) throws -> GLTFScene {
        guard let index = materialVariants.firstIndex(of: name) else {
            throw GLTFMeshLoader.Failure.missingMaterialVariant(name)
        }
        return try selectingMaterialVariant(index)
    }
}

/// The root variant names and per-primitive mappings of KHR_materials_variants.
struct GLTFMaterialVariantsDocument: Decodable {
    static let extensionName = "KHR_materials_variants"
    private typealias Failure = GLTFMeshLoader.Failure

    let extensions: Extensions<VariantsRecord>?
    let meshes: [MeshRecord]?

    /// Decodes only this extension's entry and keeps a malformed one as a failure.
    struct Extensions<Record: Decodable>: Decodable {
        let record: Result<Record, any Error>?

        private struct Name: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }

        init(from decoder: any Decoder) throws {
            let values = try decoder.container(keyedBy: Name.self)
            let key = Name(stringValue: GLTFMaterialVariantsDocument.extensionName)
            record = values.contains(key) ? Result { try values.decode(Record.self, forKey: key) } : nil
        }
    }

    struct VariantsRecord: Decodable {
        let variants: [VariantRecord]
    }

    struct VariantRecord: Decodable {
        let name: String
    }

    struct MeshRecord: Decodable {
        let primitives: [PrimitiveRecord]?
    }

    struct PrimitiveRecord: Decodable {
        let extensions: Extensions<MappingsRecord>?
    }

    struct MappingsRecord: Decodable {
        let mappings: [MappingRecord]
    }

    struct MappingRecord: Decodable {
        let variants: [Int]
        let material: Int
    }

    /// Variant names, and for each listed mesh one variant-to-material map per primitive.
    func resolve(meshIndices: Set<Int>) throws -> (names: [String], mappings: [Int: [[Int: Int]]]) {
        let names: [String]
        switch extensions?.record {
        case nil: names = []
        case .success(let record): names = record.variants.map(\.name)
        case .failure: throw Failure.invalidMaterialVariants("the root extension needs a variants array of named objects")
        }
        var mappings: [Int: [[Int: Int]]] = [:]
        for meshIndex in meshIndices.sorted() {
            guard let mesh = meshes?[safe: meshIndex] else { continue }
            mappings[meshIndex] = try (mesh.primitives ?? []).enumerated().map { primitive, record in
                switch record.extensions?.record {
                case nil:
                    return [:]
                case .failure:
                    throw Failure.invalidMaterialVariants(
                        "primitive \(primitive) of mesh \(meshIndex) needs a mappings array of variant lists and materials"
                    )
                case .success(let record):
                    var materials: [Int: Int] = [:]
                    for mapping in record.mappings {
                        for variant in mapping.variants {
                            guard names.indices.contains(variant) else {
                                throw Failure.invalidMaterialVariants(
                                    "primitive \(primitive) of mesh \(meshIndex) maps missing variant \(variant)"
                                )
                            }
                            guard materials.updateValue(mapping.material, forKey: variant) == nil else {
                                throw Failure.invalidMaterialVariants(
                                    "primitive \(primitive) of mesh \(meshIndex) maps variant \(variant) more than once"
                                )
                            }
                        }
                    }
                    return materials
                }
            }
        }
        return (names, mappings)
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? { indices.contains(index) ? self[index] : nil }
}
