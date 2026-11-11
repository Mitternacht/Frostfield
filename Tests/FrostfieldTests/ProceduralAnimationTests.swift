import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

struct ProceduralAnimationTests {
    @Test("A turn follows the chosen local axis through the full revolution",
          arguments: [AnimationClipCreator.Axis.x, .y, .z])
    func rotate(axis: AnimationClipCreator.Axis) throws {
        let node = SceneNode()
        let initial = simd_quatf(angle: 0.43, axis: SIMD3<Float>(0, 1, 0))
        let clip = try AnimationClipCreator.rotation(node: node.id, period: 2, axis: axis, initialRotation: initial)
        let track = try #require(clip.tracks.first)
        var direction = SIMD3<Float>.zero
        direction[axis.rawValue] = 1
        let vertex = SIMD3<Float>(0.2, 0.4, 0.8)
        for step in 0...16 {
            let progress = Float(step) / 16
            let sampled = try #require(try track.value(at: progress * clip.duration).quaternionValue)
            let expected = initial * simd_quatf(angle: progress * (2 * .pi), axis: direction)
            #expect(simd_length(sampled.act(vertex) - expected.act(vertex)) < 1e-5)
        }
    }

    @Test("Axis scale plays on a graph and keeps both other scale components")
    func scale() throws {
        let node = SceneNode()
        let graph = SceneGraph(roots: [node], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let clip = try AnimationClipCreator.scaleAxis(node: node.id, period: 4, axis: .z, restScale: SIMD3(2, 3, 4))
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: clip)
        try mixer.play(action)
        try mixer.setTime(1, for: action)
        let posed = try mixer.apply(to: graph)
        #expect(posed.roots[0].transform.scale == SIMD3(2, 3, 0.25))
    }

    @Test("Seeded shake and pulse follow 10 Hz sampling and hold after their last key")
    func seededClips() throws {
        let id = UUID()
        let amplitude = SIMD3<Float>(2, 3, 4)
        let center = SIMD3<Float>(1, 2, 3)
        func shake(_ seed: UInt64) throws -> AnimationTrack {
            try #require(AnimationClipCreator.shake(
                node: id, duration: 0.3, amplitude: amplitude, center: center, seed: seed
            ).tracks.first)
        }
        let first = try shake(42)
        #expect(first.times == [0, 0.1, 0.2])
        #expect(try shake(42).values == first.values)
        #expect(try shake(43).values != first.values)
        let unit = SIMD3<Float>(0.74156487, 0.15991038, 0.27860111)
        let expected = center + (2 * unit - 1) * amplitude
        #expect(first.values[0].vector3Value == expected)
        #expect(try first.value(at: 0.29) == first.values[2])
        let pulse = try AnimationClipCreator.pulsation(node: id, duration: 0.25, maximumScale: 2, seed: 42)
        let track = try #require(pulse.tracks.first)
        #expect(track.times == [0, 0.1, 0.2])
        #expect(track.values[0].vector3Value == SIMD3(repeating: 2 * unit.x))
        #expect(track.values.allSatisfy { value in
            guard let scale = value.vector3Value else { return false }
            return (0...2).contains(scale.x) && scale.x == scale.y && scale.y == scale.z
        })
        #expect(try AnimationClipCreator.pulsation(node: id, duration: 0.02, maximumScale: 0).tracks[0].values == [.vector3(.zero)])
        #expect(throws: AnimationClipCreator.Failure.sampleCount) {
            try AnimationClipCreator.shake(node: id, duration: 1e8, amplitude: amplitude)
        }
        #expect(throws: AnimationClipCreator.Failure.scale) {
            try AnimationClipCreator.pulsation(node: id, duration: 1, maximumScale: -.infinity)
        }
        #expect(throws: AnimationClip.Failure.invalidDuration) {
            try AnimationClipCreator.visibility(node: id, duration: 0)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProceduralAnimationSceneTests {
    @Test("Visibility and RGB clips survive an archive and drive real scene draws")
    func archivedClips() throws {
        let fixture = try GeometryRenderFixture()
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(0.25, 0.5, 0.75, 0.5), alphaMode: .blend, shadingModel: .unlit
        ).withLiveFactors()
        let live = try #require(material.liveFactors)
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.5),
            material: .physicallyBased(material)
        )
        let group = SceneNode(children: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)])
        let graph = SceneGraph(roots: [group], camera: fixture.camera)
        let visibility = try AnimationClipCreator.visibility(node: group.id, duration: 2)
        #expect(try AnimationProperty(path: ".visible", nodes: [:], defaultNode: group.id) == .node(group.id, .isVisible))
        let color = try AnimationClipCreator.materialColor(live, duration: 2, colors: [SIMD3(1, 0, 0), SIMD3(0, 0, 1)])
        #expect(throws: AnimationClipCreator.Failure.colors) {
            try AnimationClipCreator.materialColor(live, duration: 1, colors: [])
        }
        let constant = try AnimationClipCreator.materialColor(live, duration: 2, colors: [SIMD3(0, 1, 0)])
        #expect(try constant.tracks[0].value(at: 1.9) == .color(SIMD3(0, 1, 0)))

        let archive = try SceneArchive(capturing: graph, animations: [visibility, color])
        var manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 25)
        manifest["version"] = 24
        #expect(throws: SceneArchive.Failure.self) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: manifest), binary: archive.binary)
        }
        let decoded = try archive.load(device: fixture.device)
        #expect(try SceneArchive(capturing: decoded.graph, animations: decoded.animations) == archive)
        let factors = try #require(decoded.graph.roots[0].children[0].mesh?.pathTracingMaterial?.liveFactors)
        try factors.update { $0.baseColorFactor.w = 0.625 }

        var mixer = AnimationMixer()
        let visibleAction = try mixer.addAction(for: decoded.animations[0], loop: .once, clampWhenFinished: true)
        let colorAction = try mixer.addAction(for: decoded.animations[1], loop: .once, clampWhenFinished: true)
        try mixer.play(visibleAction)
        try mixer.play(colorAction)
        try mixer.setTime(0.5, for: colorAction)
        try mixer.applyLiveProperties()
        #expect(factors.current.baseColorFactor == SIMD4(0.75, 0, 0.25, 0.625))

        func centerPixel(_ scene: SceneGraph) throws -> SIMD4<Float> {
            fixture.pixel(try fixture.render(scene).pixels, x: fixture.side / 2, y: fixture.side / 2)
        }
        let initial = try centerPixel(mixer.apply(to: decoded.graph))
        #expect(initial.w == 0.625 && initial.x > 0.4 && initial.z > 0.1)
        try mixer.setTime(1, for: visibleAction)
        #expect(try centerPixel(mixer.apply(to: decoded.graph)) == .zero)
        try mixer.setTime(2, for: visibleAction)
        #expect(try centerPixel(mixer.apply(to: decoded.graph)) == initial)

        let full = try AnimationClip(tracks: [.init(
            property: .material(factors, .baseColorFactor), times: [0], values: [.vector4(SIMD4(0, 1, 0, 0.25))]
        )])
        let fullAction = try mixer.addAction(for: full)
        try mixer.play(fullAction)
        try mixer.applyLiveProperties()
        #expect(factors.current.baseColorFactor == SIMD4(0.75, 0, 0.25, 0.25))
        try mixer.stop(colorAction)
        try mixer.applyLiveProperties()
        #expect(factors.current.baseColorFactor == SIMD4(0, 1, 0, 0.25))
    }
}
