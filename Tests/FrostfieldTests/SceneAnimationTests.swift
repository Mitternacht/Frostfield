import Foundation
import Frostfield
import Testing
import simd

struct SceneAnimationTests {
    @Test
    func `Interpolates a child along the shortest rotation arc and keeps the hierarchy`() throws {
        let child = SceneNode(materialParameters: SIMD4(0.2, 0.4, 0.6, 1))
        let sibling = SceneNode(transform: .init(translation: SIMD3(0, 4, 0)))
        let parent = SceneNode(transform: .init(translation: SIMD3(5, 0, 0)), children: [child, sibling])
        let otherRoot = SceneNode()
        let graph = try makeGraph([parent, otherRoot])
        let zAxis = SIMD3<Float>(0, 0, 1)
        let start = SceneTransform(
            translation: SIMD3(0, 0, 0),
            rotation: simd_quatf(angle: 170 * .pi / 180, axis: zAxis),
            scale: SIMD3(-1, 1, 1)
        )
        let end = SceneTransform(
            translation: SIMD3(4, 2, 0),
            rotation: simd_quatf(angle: -170 * .pi / 180, axis: zAxis),
            scale: SIMD3(-3, 3, 1)
        )
        let track = try SceneAnimationTrack(nodeID: child.id, keyframes: [
            .init(time: 0, transform: start), .init(time: 2, transform: end),
        ])
        let animation = try SceneAnimation(tracks: [track])
        let sampled = try animation.applying(to: graph, at: 1)

        #expect(animation.duration == 2)
        #expect(sampled.roots.map(\.id) == [parent.id, otherRoot.id])
        #expect(sampled.roots[0].children.map(\.id) == [child.id, sibling.id])
        let animated = sampled.roots[0].children[0]
        #expect(animated.transform.translation == SIMD3<Float>(2, 1, 0))
        #expect(animated.transform.scale == SIMD3<Float>(-2, 2, 1))
        let rotated = simd_act(animated.transform.rotation, SIMD3<Float>(1, 0, 0))
        #expect(simd_length(rotated - SIMD3<Float>(-1, 0, 0)) < 1e-5)
        #expect(animated.materialParameters == child.materialParameters)
        #expect(sampled.roots[0].children[1].transform.translation == sibling.transform.translation)
        #expect(sampled.cameraTransform.translation == graph.cameraTransform.translation)
        #expect(graph.roots[0].children[0].transform.translation == .zero)
    }

    @Test
    func `Clamps and wraps time while shorter tracks hold their endpoints`() throws {
        let first = SceneNode()
        let second = SceneNode()
        let graph = try makeGraph([first, second])
        let animation = try SceneAnimation(tracks: [
            SceneAnimationTrack(nodeID: first.id, keyframes: [key(0, x: 0), key(4, x: 4)]),
            SceneAnimationTrack(nodeID: second.id, keyframes: [key(1, x: 10), key(2, x: 20)]),
        ])
        #expect(animation.duration == 4)
        #expect(try positions(animation.applying(to: graph, at: -1)) == [0, 10])
        #expect(try positions(animation.applying(to: graph, at: 9)) == [4, 20])
        #expect(try positions(animation.applying(to: graph, at: -1, looping: true)) == [3, 20])
        #expect(try positions(animation.applying(to: graph, at: 5, looping: true)) == [1, 10])
        #expect(try positions(animation.applying(to: graph, at: 4, looping: true)) == [0, 10])
    }

    @Test
    func `A zero-duration track holds its only key when looping`() throws {
        let node = SceneNode()
        let graph = try makeGraph([node])
        let track = try SceneAnimationTrack(nodeID: node.id, keyframes: [key(0, x: 7)])
        let animation = try SceneAnimation(tracks: [track])
        #expect(animation.duration == 0)
        #expect(try positions(animation.applying(to: graph, at: -3, looping: true)) == [7])
    }

    @Test
    func `Rejects invalid key times transforms and zero crossings`() {
        let id = UUID()
        #expect(throws: SceneAnimation.ValidationError.emptyTrack(id)) {
            try SceneAnimationTrack(nodeID: id, keyframes: [])
        }
        #expect(throws: SceneAnimation.ValidationError.invalidKeyTime(id, keyframe: 1)) {
            try SceneAnimationTrack(nodeID: id, keyframes: [key(0, x: 0), key(0, x: 1)])
        }
        #expect(throws: SceneAnimation.ValidationError.invalidTransform(id, keyframe: 0)) {
            try SceneAnimationTrack(nodeID: id, keyframes: [
                .init(time: 0, transform: .init(scale: SIMD3(1, 0, 1))),
            ])
        }
        #expect(throws: SceneAnimation.ValidationError.scaleCrossesZero(id, keyframe: 1)) {
            try SceneAnimationTrack(nodeID: id, keyframes: [
                key(0, x: 0), .init(time: 1, transform: .init(scale: SIMD3(-1, 1, 1))),
            ])
        }
    }

    @Test
    func `Rejects duplicate or absent animation targets and invalid sample times`() throws {
        let track = try SceneAnimationTrack(nodeID: UUID(), keyframes: [key(0, x: 0)])
        let graph = try makeGraph([SceneNode()])
        #expect(throws: SceneAnimation.ValidationError.duplicateTrack(track.nodeID)) {
            try SceneAnimation(tracks: [track, track])
        }
        let animation = try SceneAnimation(tracks: [track])
        #expect(throws: SceneAnimation.ValidationError.missingNode(track.nodeID)) {
            try animation.applying(to: graph, at: 0)
        }
        #expect(throws: SceneAnimation.ValidationError.invalidSampleTime) {
            try animation.applying(to: graph, at: .infinity)
        }
    }

    @Test
    func `Rejects sampled transforms whose world composition overflows`() throws {
        let child = SceneNode()
        let parent = SceneNode(transform: .init(translation: SIMD3(2e38, 0, 0)), children: [child])
        let graph = try makeGraph([parent])
        let track = try SceneAnimationTrack(nodeID: child.id, keyframes: [key(0, x: 2e38)])
        let animation = try SceneAnimation(tracks: [track])
        #expect(throws: SceneGraph.ValidationError.invalidWorldTransform(child.id)) {
            try animation.applying(to: graph, at: 0)
        }
    }

    private func key(_ time: Float, x: Float) -> TransformKeyframe {
        TransformKeyframe(time: time, transform: .init(translation: SIMD3(x, 0, 0)))
    }

    private func positions(_ graph: SceneGraph) -> [Float] {
        graph.roots.map { $0.transform.translation.x }
    }

    private func makeGraph(_ roots: [SceneNode]) throws -> SceneGraph {
        SceneGraph(
            roots: roots,
            camera: try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20),
            cameraTransform: .init(translation: SIMD3(0, 0, 3))
        )
    }
}
