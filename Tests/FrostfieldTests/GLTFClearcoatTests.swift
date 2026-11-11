import Foundation
import Frostfield
import Testing

struct GLTFClearcoatTests {
    @Test
    func importsRequiredClearcoatFactorsAndThreeIndependentBindings() throws {
        let data = try asset(
            clearcoat: [
                "clearcoatFactor": 0.8, "clearcoatRoughnessFactor": 0.35,
                "clearcoatTexture": ["index": 0],
                "clearcoatRoughnessTexture": ["index": 0, "texCoord": 1],
                "clearcoatNormalTexture": ["index": 0, "texCoord": 1, "scale": 0.6],
            ], required: true
        )
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_materials_clearcoat")) {
            try GLTFMeshLoader.loadScene(data: data)
        }
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        let clearcoat = try #require(scene.materials.first?.clearcoat)
        #expect(clearcoat.factor == 0.8)
        #expect(clearcoat.roughness == 0.35)
        #expect(clearcoat.normalScale == 0.6)
        #expect(clearcoat.factorTexture?.textureCoordinateSet == .primary)
        #expect(clearcoat.roughnessTexture?.textureCoordinateSet == .secondary)
        #expect(clearcoat.normalTexture?.textureCoordinateSet == .secondary)
        #expect(clearcoat.roughnessTexture?.sampling.wrapS == .clampToEdge)
        #expect(clearcoat.normalTexture?.image.data == clearcoat.factorTexture?.image.data)
    }

    @Test
    func usesExtensionDefaultsAndPreservesTheGeometryOnlyPath() throws {
        let data = try asset(clearcoat: [:] as [String: Int])
        #expect(try GLTFMeshLoader.loadScene(data: data).materials.isEmpty)
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        let clearcoat = try #require(scene.materials.first?.clearcoat)
        #expect(clearcoat.factor == 0)
        #expect(clearcoat.roughness == 0)
        #expect(clearcoat.normalScale == 1)
        #expect(clearcoat.factorTexture == nil)
        #expect(clearcoat.roughnessTexture == nil)
        #expect(clearcoat.normalTexture == nil)
    }

    @Test
    func validatesClearcoatRangesAndPerPrimitiveTangentSpace() throws {
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(
            0, "clearcoat factors must be between zero and one and normal scale must be finite"
        )) {
            try GLTFMeshLoader.loadScene(data: asset(clearcoat: ["clearcoatFactor": 1.1]), includeMaterials: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(
            0, "each primitive using the clearcoat roughness texture needs TEXCOORD_1"
        )) {
            let data = try asset(
                clearcoat: ["clearcoatRoughnessTexture": ["index": 0, "texCoord": 1]],
                incompleteSecondPrimitive: true
            )
            try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        }
        let normal = ["clearcoatFactor": 1, "clearcoatNormalTexture": ["index": 0]] as [String: Any]
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(
            0, "a clearcoat normal texture needs NORMAL and TANGENT on every primitive or a base normal texture"
        )) {
            try GLTFMeshLoader.loadScene(data: asset(clearcoat: normal, authoredTangents: false), includeMaterials: true)
        }
        let derived = try GLTFMeshLoader.loadScene(
            data: asset(clearcoat: normal, authoredTangents: false, baseNormal: true), includeMaterials: true
        )
        #expect(derived.materials.first?.clearcoat?.normalTexture?.textureCoordinateSet == .primary)
    }

    private func asset(
        clearcoat: Any, required: Bool = false, authoredTangents: Bool = true,
        baseNormal: Bool = false, incompleteSecondPrimitive: Bool = false
    ) throws -> Data {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3)
        let normals = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3)
        let uv0 = fixture.floats([0, 0, 1, 0, 0, 1], type: "VEC2", count: 3)
        let uv1 = fixture.floats([0, 1, 0, 0, 1, 0], type: "VEC2", count: 3)
        let tangents = fixture.floats([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], type: "VEC4", count: 3)
        var attributes = ["POSITION": positions, "NORMAL": normals, "TEXCOORD_0": uv0, "TEXCOORD_1": uv1]
        if authoredTangents { attributes["TANGENT"] = tangents }
        var primitives: [[String: Any]] = [["attributes": attributes, "material": 0]]
        if incompleteSecondPrimitive {
            attributes.removeValue(forKey: "TEXCOORD_1")
            primitives.append(["attributes": attributes, "material": 0])
        }

        let key = "KHR_materials_clearcoat"
        var material: [String: Any] = ["extensions": [key: clearcoat]]
        if baseNormal { material["normalTexture"] = ["index": 0] }
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="
        var sections: [String: Any] = [
            "nodes": [["mesh": 0]], "meshes": [["primitives": primitives]],
            "materials": [material, ["extensions": [key: "unused malformed coat"]]],
            "textures": [["source": 0, "sampler": 0]], "samplers": [["wrapS": 33071]],
            "images": [["uri": "data:image/png;base64,\(png)"]],
            "extensionsUsed": [key],
        ]
        if required { sections["extensionsRequired"] = [key] }
        return try fixture.document(sections)
    }
}
