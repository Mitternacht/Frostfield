import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import simd

struct GLTFExtensionPolicyTests {
    private static let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="
    private static let webP = "UklGRjIAAABXRUJQVlA4TCUAAAAvAUAAAB8gECA48z9lQyCQpLI/5AIBgnPn5j/wyVHBDRgi+h8CAA=="

    @Test
    func `Lists supported extensions and applies the option each one needs`() throws {
        #expect(GLTFMeshLoader.supportedExtensions == [
            "KHR_mesh_quantization", "EXT_meshopt_compression", "KHR_meshopt_compression", "KHR_draco_mesh_compression",
            "EXT_mesh_gpu_instancing",
            "KHR_lights_punctual", "KHR_animation_pointer", "KHR_materials_clearcoat", "KHR_materials_transmission", "KHR_materials_volume",
            "KHR_materials_ior", "KHR_materials_emissive_strength", "KHR_materials_specular", "KHR_materials_sheen",
            "KHR_materials_anisotropy", "KHR_materials_iridescence", "KHR_materials_dispersion", "KHR_materials_unlit",
            "EXT_materials_bump",
            "KHR_materials_variants", "KHR_texture_transform", "KHR_texture_basisu", "EXT_texture_webp", "EXT_texture_avif",
            "KHR_materials_diffuse_transmission", "KHR_materials_diffuse_roughness", "KHR_materials_volume_scatter",
        ])
        func scene(requiring name: String) throws -> Data {
            var fixture = GLTFExtensionFixture()
            let position = fixture.accessor(
                GLTFExtensionFixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0]), component: 5126, type: "VEC3", count: 3
            )
            return try fixture.document([
                "extensionsUsed": [name], "extensionsRequired": [name],
                "meshes": [["primitives": [["attributes": ["POSITION": position]]]]],
                "nodes": [["mesh": 0]], "scenes": [["nodes": [0]]],
            ])
        }
        for name in ["KHR_mesh_quantization", "EXT_meshopt_compression", "KHR_meshopt_compression"] {
            #expect(try GLTFMeshLoader.load(data: scene(requiring: name)).indices == [0, 1, 2])
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("EXT_mesh_gpu_instancing")) {
            try GLTFMeshLoader.loadScene(data: scene(requiring: "EXT_mesh_gpu_instancing"))
        }
        #expect(try GLTFMeshLoader.loadScene(data: scene(requiring: "EXT_mesh_gpu_instancing"), includeGPUInstancing: true)
            .nodes.first?.instances == nil)
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_animation_pointer")) {
            try GLTFMeshLoader.loadScene(data: scene(requiring: "KHR_animation_pointer"))
        }
        #expect(try GLTFMeshLoader.loadScene(data: scene(requiring: "KHR_animation_pointer"), includeAnimations: true).animations.isEmpty)
        for name in ["KHR_texture_basisu", "EXT_texture_webp", "EXT_texture_avif", "KHR_texture_transform"] {
            #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature(name)) {
                try GLTFMeshLoader.loadScene(data: scene(requiring: name))
            }
            #expect(try GLTFMeshLoader.loadScene(data: scene(requiring: name), includeMaterials: true).meshes.count == 1)
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_materials_pbrSpecularGlossiness")) {
            try GLTFMeshLoader.loadScene(
                data: scene(requiring: "KHR_materials_pbrSpecularGlossiness"), includeMaterials: true, includeAnimations: true,
                includeSkins: true, includeLights: true, includeCameras: true, includeMorphTargets: true,
                includeGPUInstancing: true
            )
        }
    }

    @Test
    func `Parses GPU instance transforms and validates their accessors`() throws {
        let instanced = try instancedScene()
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("EXT_mesh_gpu_instancing on node 0")) {
            try GLTFMeshLoader.loadScene(data: instanced)
        }
        let scene = try GLTFMeshLoader.loadScene(data: instanced, includeGPUInstancing: true)
        let instances = try #require(scene.nodes.first?.instances)
        try #require(instances.count == 3)
        #expect(instances.map(\.translation) == [SIMD3(1, 0, 0), SIMD3(0, 2, 0), SIMD3(0, 0, 3)])
        #expect(instances.map(\.scale) == [SIMD3(repeating: 1), SIMD3(repeating: 2), SIMD3(0.5, 1, 1)])
        let rotations = [
            simd_quatf(real: 1, imag: .zero), simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)),
            simd_quatf(angle: .pi, axis: SIMD3(0, 0, 1)),
        ]
        for (actual, expected) in zip(instances.map(\.rotation), rotations) {
            #expect(abs(abs(simd_dot(actual.vector, expected.vector)) - 1) < 1e-4)
        }

        let invalid: [(GLTFMeshLoader.Failure, Data)] = try [
            (.invalidInstancing(0, "attribute SCALE has a different instance count"), instancedScene(scaleCount: 2)),
            (.invalidInstancing(0, "unknown attribute COLOR; custom attributes need a leading underscore"),
             instancedScene(extra: "COLOR")),
            (.invalidInstancing(0, "instance 0 rotation must be a unit quaternion"), instancedScene(rotation: [0, 0, 0, 2])),
            (.invalidInstancing(0, "instance 0 transform must be finite and invertible"), instancedScene(scale: [0, 1, 1])),
            (.invalidInstancing(0, "instancing needs a node with a mesh"), instancedScene(mesh: false)),
        ]
        for (failure, data) in invalid {
            #expect(throws: failure) { try GLTFMeshLoader.loadScene(data: data, includeGPUInstancing: true) }
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Prepared scenes draw each instance with the shared renderer`() throws {
        let scene = try GLTFMeshLoader.loadScene(data: instancedScene(), includeGPUInstancing: true)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .bgra8Unorm, scene: scene)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20)
        let draws = try prepared.graph(camera: camera).resolve().filter { $0.mesh != nil }
        try #require(draws.count == 3)
        #expect(draws.allSatisfy { prepared.sourceNodeIndicesByID[$0.id] == 0 && $0.mesh === draws[0].mesh })
        let node = try #require(scene.nodes.first)
        let instances = try #require(node.instances)
        for (draw, instance) in zip(draws, instances) {
            let expected = node.localTransform.matrix * instance.matrix
            let (a, b, c, d) = draw.worldTransform.columns
            let (e, f, g, h) = expected.columns
            #expect(simd_reduce_max(abs(a - e) + abs(b - f) + abs(c - g) + abs(d - h)) < 1e-5)
        }
    }

    @Test
    func `Chooses texture extension images and keeps a fallback for KTX2`() throws {
        let rgba = KTX2Fixture.container(vkFormat: 43, width: 1, height: 1, levels: [[255, 128, 0, 255]])
        let basis = KTX2Fixture.container(vkFormat: 0, width: 4, height: 4, levels: [Array(repeating: 0, count: 16)], supercompression: 1, uncompressedLengths: [0], colorModel: 163)
        func material(_ texture: [String: Any], second: Data, mimeType: String, required: [String] = []) throws -> Data {
            var fixture = GLTFExtensionFixture()
            let position = fixture.accessor(
                GLTFExtensionFixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0]), component: 5126, type: "VEC3", count: 3
            )
            let uv = fixture.accessor(GLTFExtensionFixture.floats([0, 0, 1, 0, 0, 1]), component: 5126, type: "VEC2", count: 3)
            return try fixture.document([
                "extensionsRequired": required,
                "meshes": [["primitives": [["attributes": ["POSITION": position, "TEXCOORD_0": uv], "material": 0]]]],
                "nodes": [["mesh": 0]], "scenes": [["nodes": [0]]],
                "materials": [["pbrMetallicRoughness": ["baseColorTexture": ["index": 0]]]],
                "textures": [texture],
                "images": [
                    ["uri": "data:image/png;base64,\(Self.png)"],
                    ["uri": "data:\(mimeType);base64,\(second.base64EncodedString())"],
                ],
            ])
        }
        func texture(_ data: Data) throws -> GLTFScene.Texture {
            try #require(GLTFMeshLoader.loadScene(data: data, includeMaterials: true).materials.first?.baseColorTexture)
        }
        let basisu = ["KHR_texture_basisu": ["source": 1]]

        let native = try texture(material(["source": 0, "extensions": basisu], second: rgba, mimeType: "image/ktx2"))
        #expect(native.image.mimeType == "image/ktx2" && native.image.sourceIndex == 1)
        #expect(native.fallbackImage?.mimeType == "image/png")
        let transcoded = try texture(material(["source": 0, "extensions": basisu], second: basis, mimeType: "image/ktx2"))
        #expect(transcoded.image.mimeType == "image/png" && transcoded.fallbackImage == nil)
        #expect(throws: GLTFMeshLoader.Failure.invalidImage(
            1, "Invalid KTX2 texture. invalid ETC1S or UASTC Basis Universal data"
        )) {
            try texture(material(["extensions": basisu], second: basis, mimeType: "image/ktx2", required: ["KHR_texture_basisu"]))
        }

        if let device = MTLCreateSystemDefaultDevice() {
            for encoding in BasisTextureFixture.Encoding.allCases {
                let compressed = try BasisTextureFixture.data(encoding)
                let required = try material(
                    ["extensions": basisu], second: compressed, mimeType: "image/ktx2", required: ["KHR_texture_basisu"]
                )
                let scene = try GLTFMeshLoader.loadScene(data: required, includeMaterials: true)
                let description = try #require(scene.materials.first)
                #expect(description.baseColorTexture?.image.data == compressed)
                #expect(description.baseColorTexture?.fallbackImage == nil)
                let prepared = try description.prepare(device: device)
                #expect(prepared.baseColorTexture?.width == 8)
                #expect(prepared.baseColorTexture?.mipmapLevelCount == 4)
                #expect(prepared.baseColorTexture?.pixelFormat == .rgba8Unorm_srgb)
            }

            let broken = try BasisTextureFixture.corruptZstd(BasisTextureFixture.data(.uastcZstd))
            let optional = try material(["source": 0, "extensions": basisu], second: broken, mimeType: "image/ktx2")
            let scene = try GLTFMeshLoader.loadScene(data: optional, includeMaterials: true)
            let description = try #require(scene.materials.first)
            #expect(description.baseColorTexture?.image.mimeType == "image/ktx2")
            #expect(description.baseColorTexture?.fallbackImage?.mimeType == "image/png")
            #expect(try description.prepare(device: device).baseColorTexture?.width == 1)
        }

        let webP = try #require(Data(base64Encoded: Self.webP))
        let chosen = try texture(material(["source": 0, "extensions": ["EXT_texture_webp": ["source": 1]]], second: webP, mimeType: "image/webp"))
        let decodesWebP = (CGImageSourceCopyTypeIdentifiers() as? [String])?.contains("org.webmproject.webp") == true
        #expect(chosen.image.mimeType == (decodesWebP ? "image/webp" : "image/png"))
        #expect(chosen.fallbackImage == nil)

        #expect(throws: GLTFMeshLoader.Failure.invalidImage(1, "a texture source needs JPEG or PNG data")) {
            try texture(material(["source": 1], second: rgba, mimeType: "image/ktx2"))
        }
        if decodesWebP {
            #expect(throws: GLTFMeshLoader.Failure.invalidImage(1, "EXT_texture_webp needs WebP data")) {
                try texture(material(["source": 0, "extensions": ["EXT_texture_webp": ["source": 1]]], second: rgba, mimeType: "image/ktx2"))
            }
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("MSFT_texture_dds on texture 0")) {
            try texture(material(["source": 0, "extensions": ["MSFT_texture_dds": ["source": 1]]], second: rgba, mimeType: "image/ktx2"))
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidTexture(0, "KHR_texture_basisu needs an image source")) {
            try texture(material(["source": 0, "extensions": ["KHR_texture_basisu": [:] as [String: Int]]], second: rgba, mimeType: "image/ktx2"))
        }

        if let device = MTLCreateSystemDefaultDevice() {
            let scene = try GLTFMeshLoader.loadScene(
                data: material(["source": 0, "extensions": basisu], second: rgba, mimeType: "image/ktx2"), includeMaterials: true
            )
            let prepared = try #require(scene.materials.first).prepare(device: device)
            #expect(prepared.baseColorTexture?.pixelFormat == .rgba8Unorm_srgb)
        }
    }

    /// A node at z = -5 with three instances and an ignored application attribute.
    private func instancedScene(
        scaleCount: Int = 3, extra: String? = nil, rotation: [Float]? = nil, scale: [Float]? = nil, mesh: Bool = true
    ) throws -> Data {
        var fixture = GLTFExtensionFixture()
        let position = fixture.accessor(
            GLTFExtensionFixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0]), component: 5126, type: "VEC3", count: 3
        )
        let translation = fixture.accessor(
            GLTFExtensionFixture.floats([1, 0, 0, 0, 2, 0, 0, 0, 3]), component: 5126, type: "VEC3", count: 3
        )
        // Identity, 90° about Y, and 180° about Z as normalized SHORT components.
        let rotationAccessor = rotation.map {
            fixture.accessor(GLTFExtensionFixture.floats($0 + [0, 0, 0, 1, 0, 0, 0, 1]), component: 5126, type: "VEC4", count: 3)
        } ?? fixture.accessor(
            GLTFExtensionFixture.bytes([Int16(0), 0, 0, 32_767, 0, 23_170, 0, 23_170, 0, 0, 32_767, 0]),
            component: 5122, type: "VEC4", count: 3, normalized: true
        )
        let scales = (scale ?? [1, 1, 1]) + [2, 2, 2, 0.5, 1, 1]
        let scaleAccessor = fixture.accessor(
            GLTFExtensionFixture.floats(Array(scales.prefix(scaleCount * 3))), component: 5126, type: "VEC3", count: scaleCount
        )
        let feature = fixture.accessor(GLTFExtensionFixture.floats([7, 8, 9]), component: 5126, type: "SCALAR", count: 3)
        var attributes = [
            "TRANSLATION": translation, "ROTATION": rotationAccessor, "SCALE": scaleAccessor, "_FEATURE_ID_0": feature,
        ]
        if let extra { attributes[extra] = feature }
        var node: [String: Any] = [
            "translation": [0, 0, -5], "extensions": ["EXT_mesh_gpu_instancing": ["attributes": attributes]],
        ]
        if mesh { node["mesh"] = 0 }
        return try fixture.document([
            "extensionsUsed": ["EXT_mesh_gpu_instancing"],
            "meshes": [["primitives": [["attributes": ["POSITION": position]]]]],
            "nodes": [node], "scenes": [["nodes": [0]]],
        ])
    }
}
