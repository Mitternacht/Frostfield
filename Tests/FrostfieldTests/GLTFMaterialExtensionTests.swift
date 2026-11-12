import Foundation
import Frostfield
import Metal
import Testing
import simd

struct GLTFMaterialExtensionTests {
    private static let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="

    /// One triangle with POSITION, NORMAL, TEXCOORD_0 and optionally TEXCOORD_1 and
    /// TANGENT, using `materials[0]` unless the sections override the meshes.
    private func asset(
        materials: [[String: Any]], sections: [String: Any] = [:], secondaryUV: Bool = true,
        primary: Bool = true, tangents: Bool = false, primitiveExtensions: [String: Any]? = nil
    ) throws -> Data {
        var fixture = GLTFMotionFixture()
        var attributes = [
            "POSITION": fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3),
            "NORMAL": fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3),
        ]
        if primary { attributes["TEXCOORD_0"] = fixture.floats([0, 0, 1, 0, 0, 1], type: "VEC2", count: 3) }
        if secondaryUV, primary { attributes["TEXCOORD_1"] = fixture.floats([0, 1, 0, 0, 1, 0], type: "VEC2", count: 3) }
        if tangents { attributes["TANGENT"] = fixture.floats([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], type: "VEC4", count: 3) }
        var primitive: [String: Any] = ["attributes": attributes, "material": 0]
        if let primitiveExtensions { primitive["extensions"] = primitiveExtensions }
        var document: [String: Any] = [
            "nodes": [["mesh": 0]], "meshes": [["primitives": [primitive]]], "materials": materials,
            "textures": [["source": 0], ["source": 0, "sampler": 0]], "samplers": [["magFilter": 9728]],
            "images": [["uri": "data:image/png;base64,\(Self.png)"]],
        ]
        for (key, value) in sections { document[key] = value }
        return try fixture.document(document)
    }

    private func load(_ data: Data) throws -> GLTFScene.Material {
        try #require(GLTFMeshLoader.loadScene(data: data, includeMaterials: true).materials.first)
    }

    @Test
    func `Imports every material extension with its factors and textures`() throws {
        let extensions: [String: Any] = [
            "KHR_materials_emissive_strength": ["emissiveStrength": 12.5],
            "KHR_materials_specular": [
                "specularFactor": 0.4, "specularColorFactor": [2, 1, 0.5],
                "specularTexture": ["index": 0], "specularColorTexture": ["index": 1, "texCoord": 1],
            ],
            "KHR_materials_sheen": [
                "sheenColorFactor": [0.9, 0.5, 0.1], "sheenRoughnessFactor": 0.3,
                "sheenColorTexture": ["index": 0], "sheenRoughnessTexture": ["index": 0],
            ],
            "KHR_materials_anisotropy": [
                "anisotropyStrength": 0.7, "anisotropyRotation": 1.25, "anisotropyTexture": ["index": 0],
            ],
            "KHR_materials_iridescence": [
                "iridescenceFactor": 0.8, "iridescenceIor": 1.45, "iridescenceThicknessMinimum": 150,
                "iridescenceThicknessMaximum": 520, "iridescenceTexture": ["index": 0],
                "iridescenceThicknessTexture": ["index": 0],
            ],
            "KHR_materials_transmission": ["transmissionFactor": 1],
            "KHR_materials_volume": ["thicknessFactor": 0.5],
            "KHR_materials_dispersion": ["dispersion": 0.6],
        ]
        let required = [
            "KHR_materials_emissive_strength", "KHR_materials_specular", "KHR_materials_sheen",
            "KHR_materials_anisotropy", "KHR_materials_iridescence", "KHR_materials_dispersion",
        ]
        let data = try asset(
            materials: [["emissiveFactor": [1, 0.5, 0.25], "extensions": extensions]],
            sections: ["extensionsUsed": required, "extensionsRequired": required]
        )
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_materials_emissive_strength")) {
            try GLTFMeshLoader.loadScene(data: data)
        }
        let material = try load(data)
        #expect(material.emissiveStrength == 12.5)
        let specular = try #require(material.specular)
        #expect(specular.factor == 0.4 && specular.colorFactor == SIMD3(2, 1, 0.5))
        #expect(specular.factorTexture?.textureCoordinateSet == .primary)
        #expect(specular.colorTexture?.textureCoordinateSet == .secondary)
        #expect(specular.colorTexture?.sampling.magnification == .nearest)
        let sheen = try #require(material.sheen)
        #expect(sheen.colorFactor == SIMD3(0.9, 0.5, 0.1) && sheen.roughness == 0.3)
        #expect(sheen.colorTexture != nil && sheen.roughnessTexture != nil)
        let anisotropy = try #require(material.anisotropy)
        #expect(anisotropy.strength == 0.7 && anisotropy.rotation == 1.25 && anisotropy.texture != nil)
        let iridescence = try #require(material.iridescence)
        #expect(iridescence.factor == 0.8 && iridescence.indexOfRefraction == 1.45)
        #expect(iridescence.minimumThickness == 150 && iridescence.maximumThickness == 520)
        #expect(iridescence.factorTexture != nil && iridescence.thicknessTexture != nil)
        #expect(material.dispersion == 0.6)
        #expect(!material.isUnlit)

        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let prepared = try material.prepare(device: device)
        #expect(prepared.emissiveStrength == 12.5)
        #expect(prepared.specular?.factor == 0.4 && prepared.specular?.colorTexture?.textureCoordinateSet == .secondary)
        #expect(prepared.specular?.colorTexture?.texture.pixelFormat == .rgba8Unorm_srgb)
        #expect(prepared.specular?.factorTexture?.texture.pixelFormat == .rgba8Unorm)
        #expect(prepared.sheen?.colorTexture?.texture.pixelFormat == .rgba8Unorm_srgb)
        #expect(prepared.anisotropy?.rotation == 1.25)
        #expect(prepared.iridescence?.maximumThickness == 520)
        #expect(prepared.transmission?.dispersion == 0.6)
        // Equal glTF sampling shares one sampler state across maps.
        #expect(prepared.specular?.factorTexture?.sampler === prepared.sheen?.colorTexture?.sampler)
        let geometry = try #require(GLTFMeshLoader.loadScene(data: data, includeMaterials: true).meshes.first?.primitives.first)
        _ = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(prepared)
        )
    }

    @Test
    func `Uses specification defaults for empty extension objects`() throws {
        let names = [
            "KHR_materials_emissive_strength", "KHR_materials_specular", "KHR_materials_sheen",
            "KHR_materials_anisotropy", "KHR_materials_iridescence", "KHR_materials_dispersion",
        ]
        let extensions = Dictionary(uniqueKeysWithValues: names.map { ($0, [:] as [String: Any]) })
        let material = try load(asset(materials: [["extensions": extensions]]))
        #expect(material.emissiveStrength == 1)
        #expect(material.specular?.factor == 1 && material.specular?.colorFactor == SIMD3(repeating: 1))
        #expect(material.sheen?.colorFactor == .zero && material.sheen?.roughness == 0)
        #expect(material.anisotropy?.strength == 0 && material.anisotropy?.rotation == 0)
        #expect(material.iridescence?.factor == 0 && material.iridescence?.indexOfRefraction == 1.3)
        #expect(material.iridescence?.minimumThickness == 100 && material.iridescence?.maximumThickness == 400)
        #expect(material.dispersion == 0)
        let plain = try load(asset(materials: [[:]]))
        #expect(plain.emissiveStrength == 1 && plain.specular == nil && plain.sheen == nil && plain.dispersion == 0)
    }

    @Test
    func `Imports texture transforms with offsets, rotation, scale and a texCoord override`() throws {
        let transform: [String: Any] = ["offset": [0.25, -0.5], "rotation": 0.75, "scale": [2, 3], "texCoord": 1]
        let data = try asset(materials: [[
            "pbrMetallicRoughness": [
                "baseColorTexture": ["index": 0, "extensions": ["KHR_texture_transform": transform]],
                "metallicRoughnessTexture": ["index": 0, "texCoord": 1],
            ],
            "normalTexture": ["index": 0, "extensions": ["KHR_texture_transform": ["rotation": 1]]],
        ]], sections: ["extensionsUsed": ["KHR_texture_transform"], "extensionsRequired": ["KHR_texture_transform"]])
        let material = try load(data)
        let base = try #require(material.baseColorTexture)
        #expect(base.textureCoordinateSet == .secondary)
        #expect(base.transform == (try .init(offset: SIMD2(0.25, -0.5), rotation: 0.75, scale: SIMD2(2, 3))))
        #expect(material.metallicRoughnessTexture?.transform == .identity)
        #expect(material.normalTexture?.transform.rotation == 1)
        #expect(material.normalTexture?.textureCoordinateSet == .primary)

        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let prepared = try material.prepare(device: device)
        #expect(prepared.baseColorTextureTransform == base.transform)
    }

    @Test
    func `Rejects invalid extension values with material errors`() throws {
        func rejects(_ extensions: [String: Any], _ reason: String, tangents: Bool = false, primary: Bool = true) {
            #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(0, reason)) {
                try GLTFMeshLoader.loadScene(
                    data: asset(materials: [["extensions": extensions]], primary: primary, tangents: tangents),
                    includeMaterials: true
                )
            }
        }
        rejects(["KHR_materials_emissive_strength": ["emissiveStrength": -1]], "emissive strength must be finite and nonnegative")
        rejects(
            ["KHR_materials_specular": ["specularColorFactor": [1, 1]]],
            "specular needs a factor between zero and one and three finite nonnegative color components"
        )
        rejects(["KHR_materials_sheen": ["sheenRoughnessFactor": 2]], "sheen color components and roughness must be between zero and one")
        rejects(
            ["KHR_materials_anisotropy": ["anisotropyStrength": 1.5]],
            "anisotropy strength must be between zero and one and its rotation finite"
        )
        rejects(
            ["KHR_materials_iridescence": ["iridescenceIor": 0.5]],
            "iridescence needs a factor between zero and one, an IOR of at least one, and finite nonnegative thicknesses"
        )
        rejects(["KHR_materials_dispersion": ["dispersion": -0.1]], "dispersion must be finite and nonnegative")
        rejects(["KHR_materials_sheen": "not an object"], "sheen requires an object with valid properties")
        rejects(
            ["KHR_materials_anisotropy": ["anisotropyStrength": 0.5]],
            "anisotropy needs NORMAL and TANGENT or TEXCOORD_0 on every primitive", primary: false
        )
        // Authored tangents provide the frame without TEXCOORD_0.
        let tangentFrame = try load(asset(
            materials: [["extensions": ["KHR_materials_anisotropy": ["anisotropyStrength": 0.5]]]],
            primary: false, tangents: true
        ))
        #expect(tangentFrame.anisotropy?.strength == 0.5)

        func transformRejects(_ transform: Any, _ failure: GLTFMeshLoader.Failure) {
            #expect(throws: failure) {
                let surface = ["baseColorTexture": ["index": 0, "extensions": ["KHR_texture_transform": transform]]]
                try GLTFMeshLoader.loadScene(
                    data: asset(materials: [["pbrMetallicRoughness": surface]]), includeMaterials: true
                )
            }
        }
        transformRejects(["scale": [0, 1]], .invalidMaterial(
            0, "the base color texture transform needs two finite offsets, a finite rotation, and two finite nonzero scales"
        ))
        transformRejects(["offset": [1]], .invalidMaterial(
            0, "the base color texture transform needs two finite offsets, a finite rotation, and two finite nonzero scales"
        ))
        transformRejects(["texCoord": 2], .unsupportedFeature("base color texture coordinate set 2 on material 0"))
        transformRejects(5, .invalidMaterial(0, "the base color texture transform requires an object with valid properties"))
        transformRejects(
            ["extensions": ["EXT_future": [:] as [String: Any]]],
            .unsupportedFeature("EXT_future on base color texture transform on material 0")
        )
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(0, "each primitive using the base color texture needs TEXCOORD_1")) {
            let surface = ["baseColorTexture": ["index": 0, "extensions": ["KHR_texture_transform": ["texCoord": 1]]]]
            try GLTFMeshLoader.loadScene(
                data: asset(materials: [["pbrMetallicRoughness": surface]], secondaryUV: false), includeMaterials: true
            )
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("EXT_other on base color binding on material 0")) {
            let surface = ["baseColorTexture": ["index": 0, "extensions": ["EXT_other": [:] as [String: Any]]]]
            try GLTFMeshLoader.loadScene(data: asset(materials: [["pbrMetallicRoughness": surface]]), includeMaterials: true)
        }
    }

    @Test
    func `Unlit materials prepare to the unlit shading model with base color only`() throws {
        let data = try asset(materials: [[
            "pbrMetallicRoughness": ["baseColorFactor": [0.2, 0.4, 0.6, 1], "baseColorTexture": ["index": 0]],
            "emissiveFactor": [1, 1, 1],
            "extensions": ["KHR_materials_unlit": [:] as [String: Any], "KHR_materials_clearcoat": ["clearcoatFactor": 1]],
        ]])
        let material = try load(data)
        #expect(material.isUnlit)
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let prepared = try material.prepare(device: device)
        #expect(prepared.shadingModel == .unlit)
        #expect(prepared.baseColorFactor == SIMD4(0.2, 0.4, 0.6, 1))
        #expect(prepared.baseColorTexture != nil)
        #expect(prepared.clearcoat == nil && prepared.emissiveStrength == 1)
    }

    // MARK: - KHR_materials_variants

    private func variantAsset(mappings: Any, variants: Any = [["name": "Green"], ["name": "Blue"]]) throws -> Data {
        // A quad covering a 90° view at z = -1 with unlit red, green and blue materials.
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([-2, -2, -1, 2, -2, -1, 2, 2, -1, -2, 2, -1], type: "VEC3", count: 4)
        let normals = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 4)
        let indices = fixture.bytes(
            [0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 3, 0], type: "SCALAR", count: 6, component: 5123
        )
        let unlit = ["KHR_materials_unlit": [:] as [String: Any]]
        let materials: [[String: Any]] = [
            SIMD3<Float>(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1),
        ].map { color in
            ["pbrMetallicRoughness": ["baseColorFactor": [color.x, color.y, color.z, 1]], "extensions": unlit]
        }
        let primitive: [String: Any] = [
            "attributes": ["POSITION": positions, "NORMAL": normals], "indices": indices, "material": 0,
            "extensions": ["KHR_materials_variants": ["mappings": mappings]],
        ]
        return try fixture.document([
            "nodes": [["mesh": 0]], "meshes": [["primitives": [primitive]]], "materials": materials,
            "extensions": ["KHR_materials_variants": ["variants": variants]],
            "extensionsUsed": ["KHR_materials_variants", "KHR_materials_unlit"],
        ])
    }

    @Test
    func `Lists material variants and selects them by index or name`() throws {
        let data = try variantAsset(mappings: [
            ["variants": [0], "material": 1], ["variants": [1], "material": 2, "name": "blue look"],
        ])
        #expect(try GLTFMeshLoader.loadScene(data: data).materialVariants.isEmpty)
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        #expect(scene.materialVariants == ["Green", "Blue"])
        #expect(scene.selectedMaterialVariant == nil)
        // Materials that only a variant uses are still resolved.
        #expect(scene.materials.map(\.sourceIndex) == [0, 1, 2])
        let mesh = try #require(scene.meshes.first)
        #expect(mesh.primitiveMaterialIndices == [0])
        #expect(mesh.primitiveMaterialVariants == [[0: 1, 1: 2]])

        let blue = try scene.selectingMaterialVariant(named: "Blue")
        #expect(blue.selectedMaterialVariant == 1)
        #expect(blue.meshes.first?.primitiveMaterialIndices == [2])
        let green = try blue.selectingMaterialVariant(0)
        #expect(green.meshes.first?.primitiveMaterialIndices == [1])
        #expect(try green.selectingMaterialVariant(nil).meshes.first?.primitiveMaterialIndices == [0])
        #expect(throws: GLTFMeshLoader.Failure.missingMaterialVariant("Purple")) {
            try scene.selectingMaterialVariant(named: "Purple")
        }
        #expect(throws: GLTFMeshLoader.Failure.missingMaterialVariant("2")) { try scene.selectingMaterialVariant(2) }

        // An unmapped variant keeps the default material.
        let partial = try GLTFMeshLoader.loadScene(
            data: variantAsset(mappings: [["variants": [1], "material": 2]]), includeMaterials: true
        )
        #expect(try partial.selectingMaterialVariant(0).meshes.first?.primitiveMaterialIndices == [0])
    }

    @Test
    func `Validates variant mappings`() throws {
        func rejects(_ mappings: Any, variants: Any = [["name": "Green"], ["name": "Blue"]], _ reason: String) {
            #expect(throws: GLTFMeshLoader.Failure.invalidMaterialVariants(reason)) {
                try GLTFMeshLoader.loadScene(data: variantAsset(mappings: mappings, variants: variants), includeMaterials: true)
            }
        }
        rejects([["variants": [2], "material": 1]], "primitive 0 of mesh 0 maps missing variant 2")
        rejects(
            [["variants": [0], "material": 1], ["variants": [0], "material": 2]],
            "primitive 0 of mesh 0 maps variant 0 more than once"
        )
        rejects("none", "primitive 0 of mesh 0 needs a mappings array of variant lists and materials")
        rejects([["variants": [0], "material": 1]], variants: [["label": "x"]], "the root extension needs a variants array of named objects")
        #expect(throws: GLTFMeshLoader.Failure.missingMaterial(7)) {
            try GLTFMeshLoader.loadScene(data: variantAsset(mappings: [["variants": [0], "material": 7]]), includeMaterials: true)
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `A selected variant renders its material after re-preparation`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let scene = try GLTFMeshLoader.loadScene(
            data: variantAsset(mappings: [["variants": [0], "material": 1], ["variants": [1], "material": 2]]),
            includeMaterials: true
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 8, height: 8, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: descriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: 8, height: 8, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))

        func center(_ variant: String?) throws -> SIMD4<Float> {
            let selected = try variant.map { try scene.selectingMaterialVariant(named: $0) } ?? scene
            let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .rgba16Float, scene: selected)
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = 1
            let buffer = try #require(queue.makeCommandBuffer())
            try prepared.graph(camera: camera).encode(pass: pass, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
            var pixel = SIMD4<Float16>.zero
            color.getBytes(&pixel, bytesPerRow: 64, from: MTLRegionMake2D(4, 4, 1, 1), mipmapLevel: 0)
            return SIMD4<Float>(pixel)
        }
        #expect(try center(nil) == SIMD4(1, 0, 0, 1))
        #expect(try center("Green") == SIMD4(0, 1, 0, 1))
        #expect(try center("Blue") == SIMD4(0, 0, 1, 1))
    }

    @Test("EXT_materials_bump imports a height map and a normal texture still wins")
    func bumpExtension() throws {
        let required = ["EXT_materials_bump"]
        let data = try asset(
            materials: [[
                "extensions": ["EXT_materials_bump": ["bumpFactor": 2.0, "bumpTexture": ["index": 0, "texCoord": 0]]],
            ]],
            sections: ["extensionsUsed": required, "extensionsRequired": required]
        )
        let material = try load(data)
        let bump = try #require(material.bumpMap)
        #expect(bump.scale == 2)
        #expect(bump.texture.sourceIndex == 0)
        #expect(material.normalTexture == nil)
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let prepared = try material.prepare(device: device)
        #expect(prepared.bumpMap?.scale == 2)
        #expect(prepared.bumpMap != nil)

        let withNormal = try asset(
            materials: [[
                "normalTexture": ["index": 0],
                "extensions": ["EXT_materials_bump": ["bumpFactor": 2.0, "bumpTexture": ["index": 0]]],
            ]],
            sections: ["extensionsUsed": required, "extensionsRequired": required]
        )
        let suppressed = try load(withNormal)
        #expect(suppressed.bumpMap?.scale == 2)
        #expect(suppressed.normalTexture != nil)
        #expect(try suppressed.prepare(device: device).bumpMap == nil)

        let missing = try asset(
            materials: [["extensions": ["EXT_materials_bump": ["bumpFactor": 1]]]],
            sections: ["extensionsUsed": required, "extensionsRequired": required]
        )
        #expect(throws: GLTFMeshLoader.Failure.self) { try load(missing) }
    }
}
