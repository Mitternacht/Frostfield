import Foundation
import Frostfield
import Testing

struct GLTFTransmissionTests {
    private static let transmission = "KHR_materials_transmission"
    private static let volume = "KHR_materials_volume"
    private static let ior = "KHR_materials_ior"

    @Test
    func importsTransmissionVolumeAndIORTogether() throws {
        let data = try asset(
            extensions: [
                Self.transmission: ["transmissionFactor": 0.85, "transmissionTexture": ["index": 0, "texCoord": 1]],
                Self.volume: [
                    "thicknessFactor": 2, "thicknessTexture": ["index": 0],
                    "attenuationDistance": 0.5, "attenuationColor": [0, 0.4, 1],
                ],
                Self.ior: ["ior": 1.33],
            ],
            required: true, alphaMode: "MASK"
        )
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature(Self.ior)) {
            try GLTFMeshLoader.loadScene(data: data)
        }

        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        let material = try #require(scene.materials.first)
        let transmission = try #require(material.transmission)
        let volume = try #require(material.volume)
        #expect(material.alphaMode == .mask(cutoff: 0.5))
        #expect(material.indexOfRefraction == 1.33)
        #expect(transmission.factor == 0.85)
        #expect(transmission.factorTexture?.textureCoordinateSet == .secondary)
        #expect(transmission.factorTexture?.sampling.wrapT == .clampToEdge)
        #expect(volume.thickness == 2)
        #expect(volume.thicknessTexture?.textureCoordinateSet == .primary)
        #expect(volume.attenuationDistance == 0.5)
        #expect(volume.attenuationColor == SIMD3(0, 0.4, 1))
    }

    @Test
    func defaultsToThinClearVolumeAndPreservesStandaloneOpaqueIOR() throws {
        let data = try asset(extensions: [Self.transmission: [:], Self.volume: [:], Self.ior: [:]])
        #expect(try GLTFMeshLoader.loadScene(data: data).materials.isEmpty)
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        let material = try #require(scene.materials.first)
        let volume = try #require(material.volume)
        #expect(material.transmission?.factor == 0)
        #expect(material.indexOfRefraction == 1.5)
        #expect(volume.thickness == 0)
        #expect(volume.attenuationDistance == .infinity)
        #expect(volume.attenuationColor == SIMD3(repeating: 1))

        for index: Float in [0, 1, 2.42] {
            let opaque = try GLTFMeshLoader.loadScene(
                data: asset(extensions: [Self.ior: ["ior": index]]), includeMaterials: true
            )
            #expect(opaque.materials.first?.indexOfRefraction == index)
            #expect(opaque.materials.first?.transmission == nil)
            #expect(opaque.materials.first?.alphaMode == .opaque)
        }
    }

    @Test
    func rejectsInvalidOpticalPropertiesAndMissingDependencies() throws {
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(0, "IOR must be zero or at least one")) {
            try GLTFMeshLoader.loadScene(data: asset(extensions: [Self.ior: ["ior": 0.8]]), includeMaterials: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(0, "transmission factor must be between zero and one")) {
            try GLTFMeshLoader.loadScene(
                data: asset(extensions: [Self.transmission: ["transmissionFactor": -0.01]]), includeMaterials: true
            )
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature(
            "KHR_materials_volume without KHR_materials_transmission or KHR_materials_diffuse_transmission on material 0"
        )) {
            try GLTFMeshLoader.loadScene(
                data: asset(extensions: [Self.volume: ["thicknessFactor": 2]]), includeMaterials: true
            )
        }

        let malformed: [[String: Any]] = [
            ["thicknessFactor": -0.01], ["attenuationDistance": 0], ["attenuationColor": [0.5, 1]],
            ["attenuationColor": [0.5, 1.1, 1]],
        ]
        for volume in malformed {
            #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(
                0, "volume requires nonnegative thickness, a positive attenuation distance, and three linear color components between zero and one"
            )) {
                try GLTFMeshLoader.loadScene(
                    data: asset(extensions: [Self.transmission: [:], Self.volume: volume]), includeMaterials: true
                )
            }
        }
    }

    @Test
    func validatesTextureUVsAndDoesNotIgnoreOtherExtensions() throws {
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(
            0, "each primitive using the thickness texture needs TEXCOORD_1"
        )) {
            try GLTFMeshLoader.loadScene(
                data: asset(extensions: [
                    Self.transmission: ["transmissionFactor": 1],
                    Self.volume: ["thicknessTexture": ["index": 0, "texCoord": 1]],
                ], includesSecondaryUV: false), includeMaterials: true
            )
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_materials_pbrSpecularGlossiness on material 0")) {
            try GLTFMeshLoader.loadScene(
                data: asset(extensions: [Self.transmission: [:], "KHR_materials_pbrSpecularGlossiness": [:]]),
                includeMaterials: true
            )
        }
    }

    private func asset(
        extensions: [String: [String: Any]], required: Bool = false,
        alphaMode: String = "OPAQUE", includesSecondaryUV: Bool = true
    ) throws -> Data {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3)
        let uv0 = fixture.floats([0, 0, 1, 0, 0, 1], type: "VEC2", count: 3)
        let uv1 = fixture.floats([0, 1, 0, 0, 1, 0], type: "VEC2", count: 3)
        var attributes = ["POSITION": positions, "TEXCOORD_0": uv0]
        if includesSecondaryUV { attributes["TEXCOORD_1"] = uv1 }
        let primitive: [String: Any] = ["attributes": attributes, "material": 0]
        let material: [String: Any] = ["extensions": extensions, "alphaMode": alphaMode]
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="
        var sections: [String: Any] = [
            "nodes": [["mesh": 0]], "meshes": [["primitives": [primitive]]],
            "materials": [material], "textures": [["source": 0, "sampler": 0]], "samplers": [["wrapT": 33071]],
            "images": [["uri": "data:image/png;base64,\(png)"]],
            "extensionsUsed": extensions.keys.sorted(),
        ]
        if required { sections["extensionsRequired"] = extensions.keys.sorted() }
        return try fixture.document(sections)
    }
}
