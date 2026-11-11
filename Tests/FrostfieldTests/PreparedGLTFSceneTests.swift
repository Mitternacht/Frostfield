import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PreparedGLTFSceneTests {
    @Test
    func sharesPreparedPrimitivesAndUpdatesSkinnedInstances() throws {
        var fixture = GLTFMotionFixture()
        let position = fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3)
        let joints = fixture.bytes(Array(repeating: 0, count: 12), type: "VEC4", count: 3, component: 5121)
        let weights = fixture.bytes(
            [255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0], type: "VEC4", count: 3, component: 5121, normalized: true
        )
        let times = fixture.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let movement = fixture.floats([0, 0, 0, 4, 0, 0], type: "VEC3", count: 2)
        let primitive = ["attributes": ["POSITION": position, "JOINTS_0": joints, "WEIGHTS_0": weights]]
        let nodes: [[String: Any]] = [
            ["mesh": 0, "skin": 0, "scale": [-2, 1, 1], "children": [3]],
            [:],
            ["mesh": 0, "skin": 0, "translation": [6, 0, 0]],
            ["translation": [1, 0, 0]],
            ["mesh": 0],
        ]
        let animation: [String: Any] = [
            "channels": [["sampler": 0, "target": ["node": 1, "path": "translation"]]],
            "samplers": [["input": times, "output": movement]],
        ]
        let data = try fixture.document([
            "scenes": [["nodes": [0, 1, 2, 4]]], "nodes": nodes,
            "meshes": [["primitives": [primitive, primitive]]], "skins": [["joints": [1]]],
            "animations": [animation],
        ])
        let source = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true, includeSkins: true)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .bgra8Unorm, scene: source)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20)
        let cameraTransform = SceneTransform(translation: SIMD3(0, 0, 5))
        let rest = try prepared.graph(camera: camera, cameraTransform: cameraTransform)
        let animated = try prepared.graph(camera: camera, cameraTransform: cameraTransform, animationIndex: 0, time: 1)
        let restDraws = try rest.resolve().filter { $0.mesh != nil }
        let resolved = try animated.resolve()
        let draws = resolved.filter { $0.mesh != nil }
        let bySource = Dictionary(grouping: draws) { prepared.sourceNodeIndicesByID[$0.id] }

        #expect(restDraws.map(\.id) == draws.map(\.id))
        #expect(animated.cameraTransform.translation == cameraTransform.translation)
        let firstSkin = try #require(bySource[0])
        let secondSkin = try #require(bySource[2])
        let staticNodes = try #require(bySource[4])
        try #require(firstSkin.count == 2 && secondSkin.count == 2 && staticNodes.count == 2)
        #expect(firstSkin[0].mesh === secondSkin[0].mesh)
        #expect(firstSkin[1].mesh === secondSkin[1].mesh)
        #expect(firstSkin[0].mesh !== staticNodes[0].mesh)
        #expect(staticNodes[0].jointPalette == nil)

        let sourceID = try #require(prepared.nodeIDs[0])
        let sourceNode = try #require(resolved.first { $0.id == sourceID })
        let childID = try #require(prepared.nodeIDs[3])
        let childNode = try #require(resolved.first { $0.id == childID })
        #expect(sourceNode.worldTransform.columns.0.x == -2)
        #expect(childNode.worldTransform.columns.3.x == -2)
        #expect(firstSkin[0].worldTransform.columns.0.x == 1)
        #expect(firstSkin[0].jointPalette?.matrices.first?.columns.3 == SIMD4(2, 0, 0, 1))
        #expect(firstSkin[0].jointPalette == secondSkin[0].jointPalette)
        #expect(restDraws.first { $0.id == firstSkin[0].id }?.jointPalette?.matrices.first?.columns.3 == SIMD4(0, 0, 0, 1))

        let clone = prepared.cloned()
        let cloneRest = try clone.graph(camera: camera, cameraTransform: cameraTransform)
        let cloneDraws = try cloneRest.resolve().filter { $0.mesh != nil }
        #expect(Set(try rest.resolve().map(\.id)).isDisjoint(with: Set(try cloneRest.resolve().map(\.id))))
        #expect(clone.nodeIDs.keys == prepared.nodeIDs.keys)
        #expect(cloneDraws.count == restDraws.count)
        for (copy, original) in zip(cloneDraws, restDraws) {
            #expect(copy.mesh === original.mesh)
            #expect(clone.sourceNodeIndicesByID[copy.id] == prepared.sourceNodeIndicesByID[original.id])
            #expect(copy.jointPalette == original.jointPalette)
        }

        var joined = rest
        joined.roots += cloneRest.roots
        var mixer = AnimationMixer()
        let originalAction = try mixer.addAction(for: prepared.animationClip(0))
        let cloneAction = try mixer.addAction(for: clone.animationClip(0))
        try mixer.play(originalAction)
        try mixer.play(cloneAction)
        try mixer.setTime(0.5, for: originalAction)
        try mixer.setTime(1.5, for: cloneAction)
        let posed = try clone.posing(prepared.posing(mixer.apply(to: joined)))
        let posedDraws = try posed.resolve()
        let originalID = try #require(restDraws.first { prepared.sourceNodeIndicesByID[$0.id] == 0 }?.id)
        let cloneID = try #require(cloneDraws.first { clone.sourceNodeIndicesByID[$0.id] == 0 }?.id)
        let originalDraw = try #require(posedDraws.first { $0.id == originalID })
        let cloneDraw = try #require(posedDraws.first { $0.id == cloneID })
        let mesh = try #require(originalDraw.mesh)
        let skin = try #require(mesh.skin)
        let originalGeometry = try skin.deform(mesh.geometry, using: #require(originalDraw.jointPalette))
        let cloneGeometry = try skin.deform(mesh.geometry, using: #require(cloneDraw.jointPalette))
        #expect(originalGeometry.vertices[0].position == SIMD3(1, 0, 0))
        #expect(cloneGeometry.vertices[0].position == SIMD3(3, 0, 0))
    }
}
