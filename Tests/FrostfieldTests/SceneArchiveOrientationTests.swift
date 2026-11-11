import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneArchiveOrientationTests {
    @Test("Version ten preserves world-oriented poses and earlier archives keep inherited orientation")
    func roundTrip() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20)
        let child = SceneNode(
            transform: .init(
                translation: SIMD3(0.3, -0.4, 0.2), rotation: simd_quatf(angle: 0.5, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(1.5, -1, 0.75)
            ),
            pivot: SIMD3(0.15, 0.2, -0.1), orientationMode: .world
        )
        let parent = SceneNode(
            transform: .init(
                translation: SIMD3(0.1, 0.3, -4), rotation: simd_quatf(angle: 0.8, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(2, 1, 0.5)
            ), children: [child]
        )
        var source = SceneGraph(roots: [parent], camera: camera)
        let archive = try SceneArchive(capturing: source)
        let manifest = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 10)
        let restored = try archive.load(device: device).graph
        #expect(restored.roots[0].orientationMode == .inherited)
        #expect(restored.roots[0].children[0].orientationMode == .world)
        #expect(restored.roots[0].children[0].pivot == child.pivot)
        #expect(try restored.worldTransform(of: child.id) == source.worldTransform(of: child.id))
        #expect(try SceneArchive(capturing: restored) == archive)

        var downgraded = manifest
        downgraded["version"] = 9
        #expect(throws: SceneArchive.Failure.decoding("world orientation requires archive version 10")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: downgraded), binary: archive.binary)
        }

        source.roots[0].children[0].orientationMode = .inherited
        let legacy = try SceneArchive(capturing: source)
        let legacyManifest = try #require(try JSONSerialization.jsonObject(with: legacy.manifest) as? [String: Any])
        #expect(legacyManifest["version"] as? Int == 8)
        let legacyScene = try legacy.load(device: device).graph
        #expect(legacyScene.roots[0].children[0].orientationMode == .inherited)
        #expect(try legacyScene.worldTransform(of: child.id) == source.worldTransform(of: child.id))
    }
}
