import Foundation
import Frostfield
import ImageIO
import Metal
import simd
import Testing

/// Import, export, and re-import of material extensions, texture transforms, texture
/// extension images, material variants, and GPU instances.
struct GLTFExtensionExportTests {
    private static let webP = "UklGRjIAAABXRUJQVlA4TCUAAAAvAUAAAB8gECA48z9lQyCQpLI/5AIBgnPn5j/wyVHBDRgi+h8CAA=="
    private static let materialExtensions = [
        "KHR_materials_anisotropy", "KHR_materials_clearcoat", "KHR_materials_dispersion",
        "KHR_materials_emissive_strength", "KHR_materials_ior", "KHR_materials_iridescence", "KHR_materials_sheen",
        "KHR_materials_specular", "KHR_materials_transmission", "KHR_materials_unlit", "KHR_materials_volume",
    ]

    // MARK: Materials and texture transforms

    @Test
    func `Material extensions and texture transforms survive GLB and glTF round trips`() throws {
        let first = try GLTFMeshLoader.loadScene(data: layeredAsset(), includeMaterials: true)
        try #require(first.materials.count == 2)
        let source = first.materials[0]
        // Guard against comparing absent values: the fixture sets every layer and map.
        try #require(source.specular != nil && source.sheen != nil && source.anisotropy != nil && source.iridescence != nil)
        try #require(source.clearcoat != nil && source.transmission != nil && source.volume != nil)
        #expect(textures(of: source).count == 17)
        #expect(first.materials[1].isUnlit)

        for format in [GLTFExporter.Format.glb, .gltf] {
            let output = try GLTFExporter(format: format).export(first)
            #expect(output.warnings.isEmpty, "\(format)")
            let document = try json(output)
            #expect(document["extensionsUsed"] as? [String] == (Self.materialExtensions + ["KHR_texture_transform"]).sorted())
            #expect(document["extensionsRequired"] == nil)
            try expectCoreSources(document)

            // Spec property names, and the effective texCoord in both places for a transformed map.
            let materials = try #require(document["materials"] as? [[String: Any]])
            let surface = try #require(materials[0]["pbrMetallicRoughness"] as? [String: Any])
            let base = try #require(surface["baseColorTexture"] as? [String: Any])
            #expect(base["texCoord"] as? Int == 1)
            let transform = try #require((base["extensions"] as? [String: Any])?["KHR_texture_transform"] as? [String: Any])
            #expect(transform["offset"] as? [Double] == [0.25, -0.5])
            #expect(transform["rotation"] as? Double == 0.75)
            #expect(transform["scale"] as? [Double] == [2, 3])
            #expect(transform["texCoord"] as? Int == 1)
            #expect((surface["metallicRoughnessTexture"] as? [String: Any])?["extensions"] == nil)
            let normalTransform = try #require(
                ((materials[0]["normalTexture"] as? [String: Any])?["extensions"] as? [String: Any])?["KHR_texture_transform"]
                    as? [String: Any]
            )
            #expect(normalTransform.keys.sorted() == ["rotation"])
            let layers = try #require(materials[0]["extensions"] as? [String: Any])
            #expect(Set(layers.keys) == Set(Self.materialExtensions).subtracting(["KHR_materials_unlit"]))
            #expect(Set((layers["KHR_materials_specular"] as? [String: Any])?.keys.map { $0 } ?? []) == [
                "specularFactor", "specularTexture", "specularColorFactor", "specularColorTexture",
            ])
            #expect(Set((layers["KHR_materials_sheen"] as? [String: Any])?.keys.map { $0 } ?? []) == [
                "sheenColorFactor", "sheenColorTexture", "sheenRoughnessFactor", "sheenRoughnessTexture",
            ])
            #expect(Set((layers["KHR_materials_anisotropy"] as? [String: Any])?.keys.map { $0 } ?? []) == [
                "anisotropyStrength", "anisotropyRotation", "anisotropyTexture",
            ])
            #expect(Set((layers["KHR_materials_iridescence"] as? [String: Any])?.keys.map { $0 } ?? []) == [
                "iridescenceFactor", "iridescenceTexture", "iridescenceIor", "iridescenceThicknessMinimum",
                "iridescenceThicknessMaximum", "iridescenceThicknessTexture",
            ])
            #expect((layers["KHR_materials_emissive_strength"] as? [String: Double])?["emissiveStrength"] == 12.5)
            #expect((layers["KHR_materials_dispersion"] as? [String: Double])?["dispersion"] == 0.6)
            #expect((materials[1]["extensions"] as? [String: [String: Any]])?["KHR_materials_unlit"]?.isEmpty == true)

            let second = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true)
            try #require(second.materials.count == first.materials.count)
            for (x, y) in zip(first.materials, second.materials) { expectSameMaterial(x, y) }
            #expect(second.meshes.map(\.primitiveMaterialIndices) == first.meshes.map(\.primitiveMaterialIndices))
        }
    }

    // MARK: Texture extension images

    @Test
    func `KTX2, WebP and AVIF images export through their extensions with only PNG or JPEG core sources`() throws {
        let decodesWebP = Self.imageIO(decodes: "org.webmproject.webp")
        let avif = Self.imageIO(decodes: "public.avif") ? try Self.encode([200, 100, 50, 255], type: "public.avif") : nil
        let first = try GLTFMeshLoader.loadScene(data: textureExtensionAsset(avif: avif), includeMaterials: true)
        let source = try #require(first.materials.first)
        // The importer keeps a KTX2 texture's PNG source as its fallback.
        #expect(source.baseColorTexture?.image.mimeType == "image/ktx2")
        #expect(source.baseColorTexture?.fallbackImage?.mimeType == "image/png")
        #expect(source.occlusionTexture?.image.mimeType == "image/ktx2" && source.occlusionTexture?.fallbackImage == nil)
        #expect(source.emissiveTexture?.image.mimeType == (decodesWebP ? "image/webp" : "image/png"))
        if avif != nil { #expect(source.metallicRoughnessTexture?.image.mimeType == "image/avif") }

        for format in [GLTFExporter.Format.glb, .gltf] {
            let output = try GLTFExporter(format: format).export(first)
            let document = try json(output)
            try expectCoreSources(document)
            let images = try #require(document["images"] as? [[String: Any]])
            // Both GPU-native KTX2 payloads are flagged, since KHR_texture_basisu specifies Basis Universal.
            #expect(output.warnings.count == 2, "\(output.warnings)")
            for warning in output.warnings {
                guard case .nonBasisKTX2(let image) = warning else {
                    Issue.record("Unexpected warning \(warning)")
                    continue
                }
                #expect(images[image]["mimeType"] as? String == "image/ktx2")
                #expect(!warning.description.isEmpty)
            }

            // Textures without a core source require their extension; a fallback keeps it optional.
            var used = ["KHR_texture_basisu"], required = ["KHR_texture_basisu"]
            if decodesWebP {
                used.append("EXT_texture_webp")
                required.append("EXT_texture_webp")
            }
            if avif != nil {
                used.append("EXT_texture_avif")
                required.append("EXT_texture_avif")
            }
            #expect(document["extensionsUsed"] as? [String] == used.sorted())
            #expect(document["extensionsRequired"] as? [String] == required.sorted())
            let textures = try #require(document["textures"] as? [[String: Any]])
            let fallbackTexture = try #require(textures.first { ($0["extensions"] as? [String: Any])?["KHR_texture_basisu"] != nil && $0["source"] != nil })
            #expect(images[try #require(fallbackTexture["source"] as? Int)]["mimeType"] as? String == "image/png")
            #expect(textures.contains { ($0["extensions"] as? [String: Any])?["KHR_texture_basisu"] != nil && $0["source"] == nil })

            let second = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true)
            try #require(second.materials.count == 1)
            expectSameMaterial(source, second.materials[0])
        }

        // A KTX2 texture with a fallback alone keeps KHR_texture_basisu optional.
        let optional = try GLTFMeshLoader.loadScene(data: textureExtensionAsset(avif: nil, onlyFallback: true), includeMaterials: true)
        let document = try json(GLTFExporter().export(optional))
        #expect(document["extensionsUsed"] as? [String] == ["KHR_texture_basisu"])
        #expect(document["extensionsRequired"] == nil)
    }

    // MARK: Material variants

    @Test
    func `Material variants keep their names, mappings and default materials`() throws {
        let first = try GLTFMeshLoader.loadScene(data: variantAsset(), includeMaterials: true)
        #expect(first.materialVariants == ["Green", "Blue", "Gold"])
        let mesh = try #require(first.meshes.first)
        #expect(mesh.primitiveMaterialIndices == [0, 0])
        #expect(mesh.primitiveMaterialVariants == [[0: 1, 1: 2, 2: 1], [1: 3]])

        for format in [GLTFExporter.Format.glb, .gltf] {
            let output = try GLTFExporter(format: format).export(first)
            #expect(output.warnings.isEmpty)
            let document = try json(output)
            #expect(document["extensionsUsed"] as? [String] == ["KHR_materials_unlit", "KHR_materials_variants"])
            let root = try #require((document["extensions"] as? [String: Any])?["KHR_materials_variants"] as? [String: Any])
            #expect((root["variants"] as? [[String: String]])?.map { $0["name"] } == ["Green", "Blue", "Gold"])
            let primitives = try #require((document["meshes"] as? [[String: Any]])?.first?["primitives"] as? [[String: Any]])
            let mappings = try #require(
                ((primitives[0]["extensions"] as? [String: Any])?["KHR_materials_variants"] as? [String: Any])?["mappings"]
                    as? [[String: Any]]
            )
            #expect(mappings.map { $0["variants"] as? [Int] } == [[0, 2], [1]])
            #expect(mappings.map { $0["material"] as? Int } == [1, 2])

            let second = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true)
            #expect(second.materialVariants == first.materialVariants)
            #expect(second.meshes.map(\.primitiveMaterialIndices) == first.meshes.map(\.primitiveMaterialIndices))
            #expect(second.meshes.map(\.primitiveMaterialVariants) == first.meshes.map(\.primitiveMaterialVariants))
            for (x, y) in zip(first.materials, second.materials) { expectSameMaterial(x, y) }
            #expect(try second.selectingMaterialVariant(named: "Gold").meshes.first?.primitiveMaterialIndices == [1, 0])
        }

        // A selected variant is reported; the file keeps the authored defaults and every variant.
        let blue = try first.selectingMaterialVariant(named: "Blue")
        #expect(blue.meshes.first?.primitiveMaterialIndices == [2, 3])
        let output = try GLTFExporter().export(blue)
        #expect(output.warnings == [.materialVariantSelectionNotStored(variant: 1)])
        let reloaded = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true)
        #expect(reloaded.selectedMaterialVariant == nil)
        #expect(reloaded.meshes.first?.primitiveMaterialIndices == [0, 0])
        #expect(reloaded.meshes.first?.primitiveMaterialVariants == mesh.primitiveMaterialVariants)
        #expect(try reloaded.selectingMaterialVariant(named: "Blue").meshes.first?.primitiveMaterialIndices == [2, 3])
    }

    // MARK: GPU instancing

    @Test
    func `GPU instances export as EXT_mesh_gpu_instancing accessors`() throws {
        let first = try GLTFMeshLoader.loadScene(data: instancedAsset(), includeGPUInstancing: true)
        let instances = try #require(first.nodes.first?.instances)
        try #require(instances.count == 3)

        for format in [GLTFExporter.Format.glb, .gltf] {
            let output = try GLTFExporter(format: format).export(first)
            #expect(output.warnings.isEmpty)
            let document = try json(output)
            #expect(document["extensionsUsed"] as? [String] == ["EXT_mesh_gpu_instancing"])
            #expect(document["extensionsRequired"] == nil)
            let node = try #require((document["nodes"] as? [[String: Any]])?.first)
            let attributes = try #require(
                ((node["extensions"] as? [String: Any])?["EXT_mesh_gpu_instancing"] as? [String: Any])?["attributes"]
                    as? [String: Int]
            )
            #expect(attributes.keys.sorted() == ["ROTATION", "SCALE", "TRANSLATION"])
            let accessors = try #require(document["accessors"] as? [[String: Any]])
            for (name, type) in [("TRANSLATION", "VEC3"), ("ROTATION", "VEC4"), ("SCALE", "VEC3")] {
                let accessor = accessors[try #require(attributes[name])]
                #expect(accessor["type"] as? String == type && accessor["componentType"] as? Int == 5_126 && accessor["count"] as? Int == 3)
            }

            #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("EXT_mesh_gpu_instancing on node 0")) {
                try GLTFMeshLoader.loadScene(data: output.data)
            }
            let second = try GLTFMeshLoader.loadScene(data: output.data, includeGPUInstancing: true)
            let reloaded = try #require(second.nodes.first?.instances)
            try #require(reloaded.count == instances.count)
            #expect(close(second.nodes[0].localTransform, first.nodes[0].localTransform, tolerance: 0))
            for (a, b) in zip(instances, reloaded) {
                #expect(a.translation == b.translation && a.scale == b.scale)
                // The importer renormalizes rotations, which can move the last bit.
                #expect(close(a, b, tolerance: 0.000_001))
            }
        }
    }

    // MARK: Scene graphs

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Scene graph materials, map transforms and instances export with their extensions`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 50)
        let color = try texture(device, [255, 128, 0, 255, 0, 64, 255, 255, 32, 32, 32, 255, 200, 200, 10, 128], format: .rgba8Unorm_srgb)
        let data = try texture(device, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160])
        let baseTransform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(0.125, 0.25), rotation: 0.3, scale: SIMD2(2, 4))
        let normalTransform = try PhysicallyBasedMaterial.TextureTransform(rotation: -0.5)
        let sheenTransform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(0.5, 0), scale: SIMD2(-1, 1))
        let layered = try PhysicallyBasedMaterial(
            baseColorTexture: color, metallic: 0.5, roughness: 0.4, baseColorFactor: SIMD4(0.9, 0.8, 0.7, 1),
            normalTexture: .init(texture: data, transform: normalTransform), normalScale: 0.5,
            emissiveTexture: .init(texture: color), emissiveFactor: SIMD3(0.5, 0.25, 0.125), alphaMode: .opaque,
            transmission: .init(factor: 0.5, indexOfRefraction: 1.4, thickness: 0.3, factorTexture: .init(texture: data), dispersion: 0.4),
            baseColorTextureTransform: baseTransform, emissiveStrength: 4,
            specular: .init(factor: 0.6, colorFactor: SIMD3(1.5, 1, 0.5), factorTexture: .init(texture: data), colorTexture: .init(texture: color)),
            sheen: .init(
                colorFactor: SIMD3(0.4, 0.3, 0.2), roughness: 0.7, colorTexture: .init(texture: color, transform: sheenTransform),
                roughnessTexture: .init(texture: data)
            ),
            anisotropy: .init(strength: 0.5, rotation: 1, texture: .init(texture: data)),
            iridescence: .init(
                factor: 0.8, indexOfRefraction: 1.5, minimumThickness: 120, maximumThickness: 480,
                factorTexture: .init(texture: data), thicknessTexture: .init(texture: data)
            )
        )
        let unlit = try PhysicallyBasedMaterial(baseColorFactor: SIMD4(0.2, 0.4, 0.6, 1), alphaMode: .opaque, shadingModel: .unlit)
        let plain = try PhysicallyBasedMaterial(roughness: 0.25, alphaMode: .opaque)
        func renderer(
            _ material: PhysicallyBasedMaterial, instancing: Bool = false, morph: MeshMorphTargets? = nil
        ) throws -> MeshRenderer {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: .cube(), material: .physicallyBased(material),
                morphTargets: morph, supportsInstancing: instancing
            )
        }
        let placements = [
            SceneTransform(translation: SIMD3(1, 0, 0)),
            SceneTransform(translation: SIMD3(0, 2, 0), rotation: simd_quatf(angle: 0.8, axis: simd_normalize(SIMD3(1, 1, 0))), scale: SIMD3(1, 2, 0.5)),
            SceneTransform(translation: SIMD3(0, 0, -3), rotation: simd_quatf(angle: 2.5, axis: SIMD3(0, 0, 1)), scale: SIMD3(repeating: 0.25)),
        ]
        let instanceSet = try InstanceSet(
            device: device, transforms: placements.map(\.matrix),
            tints: [SIMD4(repeating: 1), SIMD4(1, 0, 0, 1), SIMD4(repeating: 1)]
        )
        let levels = try LevelOfDetail(levels: [
            .init(mesh: renderer(plain), distance: 0), .init(mesh: renderer(unlit), distance: 10),
        ])
        let morph = try MeshMorphTargets(vertexCount: MeshGeometry.cube().vertices.count, targets: [
            .init(positionOffsets: Array(repeating: SIMD3(0, 0.1, 0), count: MeshGeometry.cube().vertices.count)),
        ])
        let roots = [
            SceneNode(mesh: try renderer(layered)),
            SceneNode(transform: .init(translation: SIMD3(2, 0, 0)), mesh: try renderer(unlit)),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: try renderer(plain, instancing: true), instances: instanceSet),
            SceneNode(levelsOfDetail: levels),
            SceneNode(mesh: try renderer(plain, instancing: true), instances: try InstanceSet(device: device, transforms: [], capacity: 1)),
            SceneNode(mesh: try renderer(plain, morph: morph), morphWeights: [0.5]),
        ]
        let graph = SceneGraph(roots: roots, camera: camera)

        for format in [GLTFExporter.Format.glb, .gltf] {
            let output = try GLTFExporter(format: format, commandQueue: queue).export(graph, includesCamera: false)
            #expect(output.warnings == [
                .instanceTintsDropped(node: 2), .levelsOfDetailReduced(node: 3, levelCount: 2),
                .emptyInstancesDropped(node: 4), .morphTargetsDropped(mesh: 4),
            ])
            #expect(output.warnings.allSatisfy { !$0.description.isEmpty })
            let document = try json(output)
            try expectCoreSources(document)
            #expect(document["extensionsUsed"] as? [String] == (Self.materialExtensions + [
                "EXT_mesh_gpu_instancing", "KHR_texture_transform",
            ]).filter { $0 != "KHR_materials_clearcoat" }.sorted())
            #expect(document["extensionsRequired"] == nil)

            let scene = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true, includeGPUInstancing: true)
            func material(ofNode index: Int) throws -> GLTFScene.Material {
                let mesh = try #require(scene.meshes.first { $0.sourceIndex == scene.nodes[index].meshIndex })
                let source = try #require(mesh.primitiveMaterialIndices.first ?? nil)
                return try #require(scene.materials.first { $0.sourceIndex == source })
            }
            let exported = try material(ofNode: 0)
            #expect(exported.baseColorFactor == SIMD4(0.9, 0.8, 0.7, 1))
            #expect(exported.metallicFactor == 0.5 && exported.roughnessFactor == 0.4)
            // The material stores factor × strength; export separates them again.
            #expect(exported.emissiveFactor == SIMD3(0.5, 0.25, 0.125) && exported.emissiveStrength == 4)
            #expect(exported.baseColorTexture?.transform == baseTransform)
            #expect(exported.normalTexture?.transform == normalTransform && exported.normalScale == 0.5)
            #expect(exported.emissiveTexture?.transform == .identity)
            #expect(exported.transmission?.factor == 0.5 && exported.transmission?.factorTexture != nil)
            #expect(exported.volume?.thickness == 0.3 && exported.indexOfRefraction == 1.4 && exported.dispersion == 0.4)
            let specular = try #require(exported.specular)
            #expect(specular.factor == 0.6 && specular.colorFactor == SIMD3(1.5, 1, 0.5))
            #expect(specular.factorTexture != nil && specular.colorTexture != nil)
            let sheen = try #require(exported.sheen)
            #expect(sheen.colorFactor == SIMD3(0.4, 0.3, 0.2) && sheen.roughness == 0.7)
            #expect(sheen.colorTexture?.transform == sheenTransform && sheen.roughnessTexture != nil)
            #expect(exported.anisotropy?.strength == 0.5 && exported.anisotropy?.rotation == 1 && exported.anisotropy?.texture != nil)
            let film = try #require(exported.iridescence)
            #expect(film.factor == 0.8 && film.indexOfRefraction == 1.5 && film.minimumThickness == 120 && film.maximumThickness == 480)
            #expect(film.factorTexture != nil && film.thicknessTexture != nil)
            #expect(!exported.isUnlit)
            // Color layer maps are sRGB-encoded PNGs, data maps linear, as the importer prepares them.
            #expect(try pixels(device, specular.factorTexture, .linearData) == pixels(data))
            #expect(maximumDifference(try pixels(device, sheen.colorTexture, .encodedSRGB), pixels(color)) <= 1)

            let prepared = try exported.prepare(device: device)
            #expect(prepared.emissiveStrength == 4 && prepared.baseColorTextureTransform == baseTransform)
            #expect(prepared.transmission?.dispersion == 0.4)

            let flat = try material(ofNode: 1)
            #expect(flat.isUnlit && flat.baseColorFactor == SIMD4(0.2, 0.4, 0.6, 1))
            #expect(try material(ofNode: 1).prepare(device: device).shadingModel == .unlit)

            let instances = try #require(scene.nodes[2].instances)
            try #require(instances.count == placements.count)
            for (actual, expected) in zip(instances, placements) {
                #expect(maximumDifference(actual.matrix, expected.matrix) < 0.000_01)
            }
            #expect(scene.nodes[3].meshIndex != nil && scene.nodes[3].instances == nil)
            #expect(scene.nodes[4].meshIndex == nil)
            #expect(scene.meshes.allSatisfy { $0.morphTargets == nil })
        }
    }

    @Test
    func `Morph targets are reported and weight animation is skipped`() throws {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3)
        let normals = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3)
        let offsets = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3)
        let times = fixture.floats([0, 1], type: "SCALAR", count: 2, bounds: [0, 1])
        let weights = fixture.floats([0, 1], type: "SCALAR", count: 2)
        let translations = fixture.floats([0, 0, 0, 1, 0, 0], type: "VEC3", count: 2)
        let data = try fixture.document([
            "nodes": [["mesh": 0]],
            "meshes": [["weights": [0.5], "primitives": [[
                "attributes": ["POSITION": positions, "NORMAL": normals], "targets": [["POSITION": offsets]],
            ]]]],
            "animations": [[
                "channels": [
                    ["sampler": 0, "target": ["node": 0, "path": "weights"]],
                    ["sampler": 1, "target": ["node": 0, "path": "translation"]],
                ],
                "samplers": [["input": times, "output": weights], ["input": times, "output": translations]],
            ]],
        ])
        let first = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true, includeMorphTargets: true)
        try #require(first.meshes.first?.morphTargets != nil)
        let output = try GLTFExporter().export(first)
        #expect(output.warnings == [.morphTargetsDropped(mesh: 0)])
        let second = try GLTFMeshLoader.loadScene(data: output.data, includeAnimations: true, includeMorphTargets: true)
        #expect(second.meshes.first?.morphTargets == nil)
        #expect(second.animations.first?.channels.map(\.target) == [.translation])
    }

    // MARK: Fixtures

    /// A two-primitive triangle mesh: material 0 sets every layer extension and transforms
    /// several maps; material 1 is unlit with a transformed base color map.
    private func layeredAsset() throws -> Data {
        var fixture = GLTFMotionFixture()
        let attributes = [
            "POSITION": fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3),
            "NORMAL": fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3),
            "TANGENT": fixture.floats([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], type: "VEC4", count: 3),
            "TEXCOORD_0": fixture.floats([0, 0, 1, 0, 0, 1], type: "VEC2", count: 3),
            "TEXCOORD_1": fixture.floats([0, 1, 0, 0, 1, 0], type: "VEC2", count: 3),
        ]
        func map(_ index: Int, texCoord: Int? = nil, transform: [String: Any]? = nil, extra: [String: Any] = [:]) -> [String: Any] {
            var info: [String: Any] = ["index": index]
            info["texCoord"] = texCoord
            if let transform { info["extensions"] = ["KHR_texture_transform": transform] }
            for (key, value) in extra { info[key] = value }
            return info
        }
        let layered: [String: Any] = [
            "name": "Layered",
            "pbrMetallicRoughness": [
                "baseColorFactor": [0.9, 0.8, 0.7, 1], "metallicFactor": 0.25, "roughnessFactor": 0.6,
                "baseColorTexture": map(0, transform: ["offset": [0.25, -0.5], "rotation": 0.75, "scale": [2, 3], "texCoord": 1]),
                "metallicRoughnessTexture": map(1, texCoord: 1),
            ],
            "normalTexture": map(1, transform: ["rotation": 1.5], extra: ["scale": 0.5]),
            "occlusionTexture": map(1, extra: ["strength": 0.75]),
            "emissiveTexture": map(0, transform: ["scale": [4, 4]]),
            "emissiveFactor": [1, 0.5, 0.25],
            "extensions": [
                "KHR_materials_emissive_strength": ["emissiveStrength": 12.5],
                "KHR_materials_specular": [
                    "specularFactor": 0.4, "specularColorFactor": [2, 1, 0.5],
                    "specularTexture": map(1, transform: ["offset": [0.5, 0.5]]), "specularColorTexture": map(0, texCoord: 1),
                ],
                "KHR_materials_sheen": [
                    "sheenColorFactor": [0.9, 0.5, 0.1], "sheenRoughnessFactor": 0.3,
                    "sheenColorTexture": map(0, transform: ["rotation": -0.25]), "sheenRoughnessTexture": map(1),
                ],
                "KHR_materials_anisotropy": ["anisotropyStrength": 0.7, "anisotropyRotation": 1.25, "anisotropyTexture": map(1)],
                "KHR_materials_iridescence": [
                    "iridescenceFactor": 0.8, "iridescenceIor": 1.45, "iridescenceThicknessMinimum": 150,
                    "iridescenceThicknessMaximum": 520, "iridescenceTexture": map(1, texCoord: 1),
                    "iridescenceThicknessTexture": map(1, transform: ["scale": [0.5, 2], "texCoord": 0]),
                ],
                "KHR_materials_clearcoat": [
                    "clearcoatFactor": 0.5, "clearcoatRoughnessFactor": 0.2, "clearcoatTexture": map(1),
                    "clearcoatRoughnessTexture": map(1), "clearcoatNormalTexture": map(1, transform: ["offset": [0, 1]], extra: ["scale": 0.8]),
                ],
                "KHR_materials_transmission": ["transmissionFactor": 0.9, "transmissionTexture": map(1)],
                "KHR_materials_volume": [
                    "thicknessFactor": 0.5, "thicknessTexture": map(1), "attenuationDistance": 3, "attenuationColor": [0.9, 0.8, 0.7],
                ],
                "KHR_materials_ior": ["ior": 1.33],
                "KHR_materials_dispersion": ["dispersion": 0.6],
            ],
        ]
        let unlit: [String: Any] = [
            "pbrMetallicRoughness": [
                "baseColorFactor": [0.2, 0.4, 0.6, 1], "baseColorTexture": map(0, transform: ["offset": [0.5, 0]]),
            ],
            "extensions": ["KHR_materials_unlit": [:] as [String: Any]],
        ]
        let used = Self.materialExtensions + ["KHR_texture_transform"]
        return try fixture.document([
            "extensionsUsed": used, "extensionsRequired": ["KHR_texture_transform"],
            "nodes": [["mesh": 0]],
            "meshes": [["primitives": [["attributes": attributes, "material": 0], ["attributes": attributes, "material": 1]]]],
            "materials": [layered, unlit],
            "textures": [["source": 0], ["source": 1, "sampler": 0]],
            "samplers": [["magFilter": 9728, "minFilter": 9729, "wrapS": 33071, "wrapT": 33648]],
            "images": [
                ["name": "Color", "uri": "data:image/png;base64,\(try Self.encode([255, 64, 0, 255], type: "public.png").base64EncodedString())"],
                ["uri": "data:image/png;base64,\(try Self.encode([10, 200, 90, 128], type: "public.png").base64EncodedString())"],
            ],
        ])
    }

    /// Base color: KTX2 with a PNG source. Occlusion: KTX2 alone. Emissive: WebP with a PNG
    /// source. Metallic-roughness: AVIF alone, when available.
    private func textureExtensionAsset(avif: Data?, onlyFallback: Bool = false) throws -> Data {
        var fixture = GLTFMotionFixture()
        let attributes = [
            "POSITION": fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3),
            "TEXCOORD_0": fixture.floats([0, 0, 1, 0, 0, 1], type: "VEC2", count: 3),
        ]
        let color = KTX2Fixture.container(vkFormat: 43, width: 1, height: 1, levels: [[255, 128, 0, 255]])
        let linear = KTX2Fixture.container(vkFormat: 37, width: 1, height: 1, levels: [[90, 255, 255, 255]])
        let png = try Self.encode([0, 128, 255, 255], type: "public.png")
        var images: [[String: Any]] = [
            ["uri": "data:image/png;base64,\(png.base64EncodedString())"],
            ["name": "Color KTX2", "uri": "data:image/ktx2;base64,\(color.base64EncodedString())"],
            ["uri": "data:image/ktx2;base64,\(linear.base64EncodedString())"],
            ["uri": "data:image/webp;base64,\(Self.webP)"],
        ]
        var textures: [[String: Any]] = [
            ["source": 0, "extensions": ["KHR_texture_basisu": ["source": 1]]],
            ["extensions": ["KHR_texture_basisu": ["source": 2]]],
            ["source": 0, "extensions": ["EXT_texture_webp": ["source": 3]]],
        ]
        var surface: [String: Any] = ["baseColorTexture": ["index": 0]]
        var material: [String: Any] = [:]
        var required = ["KHR_texture_basisu"]
        if !onlyFallback {
            material["occlusionTexture"] = ["index": 1]
            material["emissiveTexture"] = ["index": 2]
        }
        if let avif, !onlyFallback {
            images.append(["uri": "data:image/avif;base64,\(avif.base64EncodedString())"])
            textures.append(["extensions": ["EXT_texture_avif": ["source": 4]]])
            surface["metallicRoughnessTexture"] = ["index": 3]
            required.append("EXT_texture_avif")
        }
        material["pbrMetallicRoughness"] = surface
        return try fixture.document([
            "extensionsUsed": required + ["EXT_texture_webp"], "extensionsRequired": required,
            "nodes": [["mesh": 0]], "meshes": [["primitives": [["attributes": attributes, "material": 0]]]],
            "materials": [material], "textures": textures, "images": images,
        ])
    }

    /// Two primitives with unlit materials and three variants.
    private func variantAsset() throws -> Data {
        var fixture = GLTFMotionFixture()
        let attributes = [
            "POSITION": fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3),
            "NORMAL": fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3),
        ]
        let materials: [[String: Any]] = [
            SIMD3<Float>(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 0.8, 0),
        ].enumerated().map { index, color in
            [
                "name": "Look \(index)", "pbrMetallicRoughness": ["baseColorFactor": [color.x, color.y, color.z, 1]],
                "extensions": ["KHR_materials_unlit": [:] as [String: Any]],
            ]
        }
        func primitive(_ mappings: [[String: Any]]) -> [String: Any] {
            ["attributes": attributes, "material": 0, "extensions": ["KHR_materials_variants": ["mappings": mappings]]]
        }
        return try fixture.document([
            "nodes": [["mesh": 0]],
            "meshes": [["primitives": [
                primitive([["variants": [2, 0], "material": 1], ["variants": [1], "material": 2]]),
                primitive([["variants": [1], "material": 3]]),
            ]]],
            "materials": materials,
            "extensions": ["KHR_materials_variants": ["variants": [["name": "Green"], ["name": "Blue"], ["name": "Gold"]]]],
            "extensionsUsed": ["KHR_materials_variants", "KHR_materials_unlit"],
        ])
    }

    /// A node at z = -5 with three instances, rotations stored as normalized shorts.
    private func instancedAsset() throws -> Data {
        var fixture = GLTFExtensionFixture()
        let position = fixture.accessor(
            GLTFExtensionFixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0]), component: 5126, type: "VEC3", count: 3
        )
        let translation = fixture.accessor(
            GLTFExtensionFixture.floats([1, 0, 0, 0, 2, 0, 0, 0, 3]), component: 5126, type: "VEC3", count: 3
        )
        let rotation = fixture.accessor(
            GLTFExtensionFixture.bytes([Int16(0), 0, 0, 32_767, 0, 23_170, 0, 23_170, 0, 0, 32_767, 0]),
            component: 5122, type: "VEC4", count: 3, normalized: true
        )
        let scale = fixture.accessor(
            GLTFExtensionFixture.floats([1, 1, 1, 2, 2, 2, 0.5, 1, -1]), component: 5126, type: "VEC3", count: 3
        )
        let node: [String: Any] = [
            "mesh": 0, "translation": [0, 0, -5],
            "extensions": ["EXT_mesh_gpu_instancing": ["attributes": ["TRANSLATION": translation, "ROTATION": rotation, "SCALE": scale]]],
        ]
        return try fixture.document([
            "extensionsUsed": ["EXT_mesh_gpu_instancing"],
            "meshes": [["primitives": [["attributes": ["POSITION": position]]]]],
            "nodes": [node], "scenes": [["nodes": [0]]],
        ])
    }

    // MARK: Helpers

    private static func imageIO(decodes type: String) -> Bool {
        (CGImageSourceCopyTypeIdentifiers() as? [String])?.contains(type) == true
            && (type != "public.avif" || (CGImageDestinationCopyTypeIdentifiers() as? [String])?.contains(type) == true)
    }

    /// A 2 × 2 image of one straight-alpha color, encoded by ImageIO.
    private static func encode(_ rgba: [UInt8], type: String) throws -> Data {
        let pixels = Array([[UInt8]](repeating: rgba, count: 4).joined())
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let alpha = type == "public.avif" ? CGImageAlphaInfo.noneSkipLast : .last
        let image = try #require(CGImage(
            width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue), provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        ))
        let output = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(output as CFMutableData, type as CFString, 1, nil))
        // ImageIO's AVIF encoder rejects quality 1.0.
        let options = type == "public.avif" ? [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary : nil
        CGImageDestinationAddImage(destination, image, options)
        try #require(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// The JSON document of a GLB or embedded glTF.
    private func json(_ output: GLTFExporter.Output) throws -> [String: Any] {
        var data = output.data
        if output.format == .glb {
            let length = data.subdata(in: 12..<16).withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))) }
            data = data.subdata(in: 20..<(20 + length))
        }
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// Every core texture `source` is a PNG or JPEG image, and every extension source has
    /// its extension's media type. A texture without `source` requires its extension.
    private func expectCoreSources(_ document: [String: Any]) throws {
        let images = document["images"] as? [[String: Any]] ?? []
        let required = Set(document["extensionsRequired"] as? [String] ?? [])
        func mimeType(_ index: Int) -> String? {
            guard images.indices.contains(index) else { return nil }
            let declared = images[index]["mimeType"] as? String
            if let uri = images[index]["uri"] as? String {
                #expect(declared.map { uri.hasPrefix("data:\($0);base64,") } ?? false)
            }
            return declared
        }
        let types = ["KHR_texture_basisu": "image/ktx2", "EXT_texture_webp": "image/webp", "EXT_texture_avif": "image/avif"]
        for texture in document["textures"] as? [[String: Any]] ?? [] {
            let extensions = texture["extensions"] as? [String: [String: Int]] ?? [:]
            if let source = texture["source"] as? Int {
                #expect(["image/png", "image/jpeg"].contains(mimeType(source)), "core source \(source)")
            } else {
                #expect(extensions.count == 1 && extensions.keys.allSatisfy(required.contains))
            }
            for (name, record) in extensions {
                #expect(mimeType(try #require(record["source"])) == types[name], "\(name)")
            }
        }
    }

    private func textures(of material: GLTFScene.Material) -> [(String, GLTFScene.Texture)] {
        let all: [(String, GLTFScene.Texture?)] = [
            ("base color", material.baseColorTexture), ("metallic-roughness", material.metallicRoughnessTexture),
            ("normal", material.normalTexture), ("occlusion", material.occlusionTexture), ("emissive", material.emissiveTexture),
            ("clearcoat", material.clearcoat?.factorTexture), ("clearcoat roughness", material.clearcoat?.roughnessTexture),
            ("clearcoat normal", material.clearcoat?.normalTexture), ("transmission", material.transmission?.factorTexture),
            ("thickness", material.volume?.thicknessTexture), ("specular", material.specular?.factorTexture),
            ("specular color", material.specular?.colorTexture), ("sheen color", material.sheen?.colorTexture),
            ("sheen roughness", material.sheen?.roughnessTexture), ("anisotropy", material.anisotropy?.texture),
            ("iridescence", material.iridescence?.factorTexture), ("iridescence thickness", material.iridescence?.thicknessTexture),
        ]
        return all.compactMap { name, texture in texture.map { (name, $0) } }
    }

    /// Compares every factor and every map, including encoded image and fallback bytes.
    private func expectSameMaterial(_ x: GLTFScene.Material, _ y: GLTFScene.Material) {
        #expect(x.name == y.name && x.baseColorFactor == y.baseColorFactor)
        #expect(x.metallicFactor == y.metallicFactor && x.roughnessFactor == y.roughnessFactor)
        #expect(x.normalScale == y.normalScale && x.occlusionStrength == y.occlusionStrength)
        #expect(x.emissiveFactor == y.emissiveFactor && x.emissiveStrength == y.emissiveStrength)
        #expect(x.alphaMode == y.alphaMode && x.isDoubleSided == y.isDoubleSided && x.isUnlit == y.isUnlit)
        #expect(x.indexOfRefraction == y.indexOfRefraction && x.dispersion == y.dispersion)
        #expect(x.clearcoat?.factor == y.clearcoat?.factor && x.clearcoat?.roughness == y.clearcoat?.roughness)
        #expect(x.clearcoat?.normalScale == y.clearcoat?.normalScale && x.transmission?.factor == y.transmission?.factor)
        #expect(x.volume?.thickness == y.volume?.thickness && x.volume?.attenuationDistance == y.volume?.attenuationDistance)
        #expect(x.volume?.attenuationColor == y.volume?.attenuationColor)
        #expect(x.specular?.factor == y.specular?.factor && x.specular?.colorFactor == y.specular?.colorFactor)
        #expect(x.sheen?.colorFactor == y.sheen?.colorFactor && x.sheen?.roughness == y.sheen?.roughness)
        #expect(x.anisotropy?.strength == y.anisotropy?.strength && x.anisotropy?.rotation == y.anisotropy?.rotation)
        #expect(x.iridescence?.factor == y.iridescence?.factor && x.iridescence?.indexOfRefraction == y.iridescence?.indexOfRefraction)
        #expect(x.iridescence?.minimumThickness == y.iridescence?.minimumThickness)
        #expect(x.iridescence?.maximumThickness == y.iridescence?.maximumThickness)
        let left = textures(of: x), right = textures(of: y)
        #expect(left.map(\.0) == right.map(\.0))
        for ((name, p), (_, q)) in zip(left, right) {
            #expect(p.image.data == q.image.data && p.image.mimeType == q.image.mimeType && p.image.name == q.image.name, "\(name)")
            #expect(p.fallbackImage?.data == q.fallbackImage?.data && p.fallbackImage?.mimeType == q.fallbackImage?.mimeType, "\(name)")
            #expect(p.sampling == q.sampling && p.textureCoordinateSet == q.textureCoordinateSet, "\(name)")
            #expect(p.transform == q.transform, "\(name)")
        }
    }

    private func close(_ a: SceneTransform, _ b: SceneTransform, tolerance: Float) -> Bool {
        simd_distance(a.translation, b.translation) <= tolerance && simd_distance(a.scale, b.scale) <= tolerance
            && 1 - abs(simd_dot(a.rotation.vector, b.rotation.vector)) <= max(tolerance, 0.000_000_1)
    }

    private func maximumDifference(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let (a0, a1, a2, a3) = a.columns, (b0, b1, b2, b3) = b.columns
        return simd_reduce_max(abs(a0 - b0) + abs(a1 - b1) + abs(a2 - b2) + abs(a3 - b3))
    }

    private func maximumDifference(_ a: [UInt8], _ b: [UInt8]) -> Int {
        guard a.count == b.count else { return .max }
        return zip(a, b).map { abs(Int($0) - Int($1)) }.max() ?? 0
    }

    /// A shared 2 × 2 four-byte texture.
    private func texture(_ device: any MTLDevice, _ bytes: [UInt8], format: MTLPixelFormat = .rgba8Unorm) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 2, height: 2, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        bytes.withUnsafeBytes { texture.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 8) }
        return texture
    }

    private func pixels(_ texture: any MTLTexture) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: texture.width * texture.height * 4)
        bytes.withUnsafeMutableBytes { raw in
            texture.getBytes(
                raw.baseAddress!, bytesPerRow: texture.width * 4, from: MTLRegionMake2D(0, 0, texture.width, texture.height),
                mipmapLevel: 0
            )
        }
        return bytes
    }

    /// Decodes an imported map the way material preparation does.
    private func pixels(
        _ device: any MTLDevice, _ texture: GLTFScene.Texture?, _ interpretation: TextureLoader.Interpretation
    ) throws -> [UInt8] {
        let image = try #require(texture?.image)
        return pixels(try TextureLoader(device: device).load(data: image.data, interpretation: interpretation))
    }
}
