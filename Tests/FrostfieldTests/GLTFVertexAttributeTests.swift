import Foundation
import Frostfield
import Testing
import simd

struct GLTFVertexAttributeTests {
    @Test
    func importsAuthoredTangentsAndRoutesEachMaterialBinding() throws {
        let scene = try GLTFMeshLoader.loadScene(data: asset(), includeMaterials: true)
        let geometry = try #require(scene.meshes.first?.primitives.first)
        let material = try #require(scene.materials.first)
        #expect(geometry.vertices.map(\.tangent) == Array(repeating: SIMD4<Float>(1, 0, 0, -1), count: 3))
        #expect(geometry.vertices.map(\.secondaryTextureCoordinate) == [
            SIMD2<Float>(0, 1), SIMD2<Float>(1, 0), SIMD2<Float>(Float(32_768) / 65_535, 0),
        ])
        #expect(material.baseColorTexture?.textureCoordinateSet == .secondary)
        #expect(material.normalTexture?.textureCoordinateSet == .primary)
        #expect(material.emissiveTexture?.textureCoordinateSet == .secondary)
        #expect(material.baseColorTexture?.sourceIndex == material.normalTexture?.sourceIndex)
    }

    @Test
    func validatesEveryPrimitiveUVSetAndIgnoresTangentsWithoutNormals() throws {
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(
            0, "each primitive using the base color texture needs TEXCOORD_1"
        )) {
            try GLTFMeshLoader.loadScene(data: asset(incompleteSecondPrimitive: true), includeMaterials: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(
            4, "tangents require a unit direction and a handedness sign of -1 or +1"
        )) {
            try GLTFMeshLoader.load(data: asset(tangentSign: 0))
        }
        let flat = try GLTFMeshLoader.load(data: asset(includesNormals: false, tangentSign: 0))
        #expect(flat.vertices.allSatisfy { $0.tangent == nil && $0.secondaryTextureCoordinate != nil })
    }

    @Test
    func preservesUVsAndAdjustsTangentHandednessDuringSkinning() throws {
        let geometry = try GLTFMeshLoader.load(data: asset())
        let influence = VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0))
        let skin = try MeshSkin(jointCount: 1, influences: Array(repeating: influence, count: geometry.vertices.count))
        let mirror = SceneTransform(scale: SIMD3(-2, 3, 1)).matrix
        let deformed = try skin.deform(geometry, using: JointPalette(matrices: [mirror]))
        #expect(deformed.vertices.map(\.secondaryTextureCoordinate) == geometry.vertices.map(\.secondaryTextureCoordinate))
        #expect(deformed.vertices.map(\.tangent) == Array(repeating: SIMD4<Float>(-1, 0, 0, 1), count: 3))
    }

    private func asset(
        includesNormals: Bool = true, tangentSign: Float = -1, incompleteSecondPrimitive: Bool = false
    ) throws -> Data {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3)
        let normals = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3)
        let uv0 = fixture.floats([0, 0, 1, 0, 0, 1], type: "VEC2", count: 3)
        let uv1 = fixture.bytes(
            [0, 0, 255, 255, 255, 255, 0, 0, 0, 128, 0, 0],
            type: "VEC2", count: 3, component: 5123, normalized: true
        )
        let tangents = fixture.floats(
            [1, 0, 0, tangentSign, 1, 0, 0, tangentSign, 1, 0, 0, tangentSign], type: "VEC4", count: 3
        )
        var attributes = ["POSITION": positions, "TEXCOORD_0": uv0, "TEXCOORD_1": uv1, "TANGENT": tangents]
        if includesNormals { attributes["NORMAL"] = normals }
        var primitives: [[String: Any]] = [["attributes": attributes, "material": 0]]
        if incompleteSecondPrimitive {
            attributes.removeValue(forKey: "TEXCOORD_1")
            primitives.append(["attributes": attributes, "material": 0])
        }
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="
        let material: [String: Any] = [
            "pbrMetallicRoughness": ["baseColorTexture": ["index": 0, "texCoord": 1]],
            "normalTexture": ["index": 0],
            "emissiveTexture": ["index": 0, "texCoord": 1],
        ]
        return try fixture.document([
            "nodes": [["mesh": 0]], "meshes": [["primitives": primitives]],
            "materials": [material], "textures": [["source": 0]],
            "images": [["uri": "data:image/png;base64,\(png)"]],
        ])
    }
}
