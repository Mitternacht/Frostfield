import Foundation
import Frostfield
import Testing
import simd

struct GLTFSkinTests {
    @Test
    func importsOrderedJointsAndExpandsInfluencesForFlatNormals() throws {
        let data = try asset()
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("skin on node 0")) {
            try GLTFMeshLoader.loadScene(data: data)
        }
        let scene = try GLTFMeshLoader.loadScene(data: data, includeSkins: true)
        let skin = try #require(scene.skins.first)
        #expect(scene.nodes.first?.skinIndex == 0)
        #expect(skin.sourceIndex == 0)
        #expect(skin.name == "Rig")
        #expect(skin.jointNodeIndices == [2, 1])
        #expect(skin.skeletonRootNodeIndex == 1)
        #expect(skin.inverseBindMatrices.map { $0.columns.3 } == [SIMD4(3, 0, 0, 1), SIMD4(0, 0, 0, 1)])

        let mesh = try #require(scene.meshes.first)
        let geometry = try #require(mesh.primitives.first)
        let influences = try #require(mesh.primitiveSkinInfluences.first ?? nil)
        #expect(geometry.vertices.count == 6)
        #expect(influences.count == 6)
        #expect(influences.map { $0.joints.x } == [0, 1, 0, 0, 0, 1])
        #expect(influences[2] == influences[4])
        #expect(abs(influences[2].weights.x - Float(128) / 255) < 0.00001)
        #expect(try MeshSkin(jointCount: 2, influences: influences).influences.count == 6)
    }

    @Test
    func defaultsBindMatricesAndRejectsInvalidInfluences() throws {
        let data = try asset(explicitBinds: false)
        let scene = try GLTFMeshLoader.loadScene(data: data, includeSkins: true)
        let skin = try #require(scene.skins.first)
        #expect(skin.inverseBindMatrices.map { $0.columns.3 } == [SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0, 1)])
        #expect(throws: GLTFMeshLoader.Failure.invalidSkin(0, "a skin requires one to 256 distinct joints")) {
            try GLTFMeshLoader.loadScene(data: asset(joints: [1, 1]), includeSkins: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidSkinPrimitive("joint indices or weights are out of range")) {
            try GLTFMeshLoader.loadScene(data: asset(invalidJoint: true), includeSkins: true)
        }
    }

    @Test
    func normalizesAuthoredFloatWeights() throws {
        let data = try asset(floatWeights: true)
        let scene = try GLTFMeshLoader.loadScene(data: data, includeSkins: true)
        let mesh = try #require(scene.meshes.first)
        let influences = try #require(mesh.primitiveSkinInfluences.first ?? nil)
        #expect(influences[0].weights == SIMD4(1, 0, 0, 0))
        #expect(abs(influences[2].weights.sum() - 1) < 0.000001)
    }

    private func asset(
        explicitBinds: Bool = true, joints: [Int] = [2, 1], invalidJoint: Bool = false, floatWeights: Bool = false
    ) throws -> Data {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], type: "VEC3", count: 4)
        let faces = fixture.bytes([0, 1, 2, 0, 2, 3], type: "SCALAR", count: 6, component: 5121)
        let indexLanes: [UInt8] = [0, 0, 0, 0, invalidJoint ? 2 : 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        let jointIndices = fixture.bytes(
            indexLanes.flatMap { [$0, 0] }, type: "VEC4", count: 4, component: 5123
        )
        let weights: Int
        if floatWeights {
            weights = fixture.floats(
                [0.9998279, 0, 0, 0, 1, 0, 0, 0, 0.6, 0.3999, 0, 0, 1, 0, 0, 0], type: "VEC4", count: 4
            )
        } else {
            weights = fixture.bytes(
                [255, 0, 0, 0, 255, 0, 0, 0, 128, 127, 0, 0, 255, 0, 0, 0],
                type: "VEC4", count: 4, component: 5121, normalized: true
            )
        }

        let primitive: [String: Any] = [
            "attributes": ["POSITION": positions, "JOINTS_0": jointIndices, "WEIGHTS_0": weights],
            "indices": faces,
        ]
        let nodes: [[String: Any]] = [["mesh": 0, "skin": 0], ["children": [2]], [:]]
        var skin: [String: Any] = ["name": "Rig", "joints": joints, "skeleton": 1]
        if explicitBinds {
            skin["inverseBindMatrices"] = fixture.floats([
                1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  3, 0, 0, 1,
                1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
            ], type: "MAT4", count: 2)
        }
        return try fixture.document([
            "scenes": [["nodes": [0, 1]]], "nodes": nodes, "meshes": [["primitives": [primitive]]], "skins": [skin],
        ])
    }
}
