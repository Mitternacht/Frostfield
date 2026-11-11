import Foundation
import Frostfield
import Testing
import simd

struct AnimationMixerTests {
    private let zAxis = SIMD3<Float>(0, 0, 1)

    @Test
    func `Blends two clips by normalized weights with hemisphere correction`() throws {
        let node = SceneNode(transform: .init(translation: SIMD3(9, 9, 9)))
        let bystander = SceneNode(transform: .init(translation: SIMD3(-1, 0, 0), scale: SIMD3(repeating: 2)))
        let graph = try makeGraph([node, bystander])
        let quarterTurn = simd_quatf(angle: .pi / 2, axis: zAxis)
        let first = try constantClip(node.id, .init(translation: SIMD3(4, 0, 0)))
        // The negated quaternion is the same rotation and must not cancel the blend.
        let second = try constantClip(node.id, .init(
            translation: SIMD3(0, 8, 0), rotation: simd_quatf(vector: -quarterTurn.vector), scale: SIMD3(repeating: 3)
        ))

        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: first, weight: 0.25))
        try mixer.play(mixer.addAction(for: second, weight: 0.75))
        let blended = try mixer.apply(to: graph)

        let transform = blended.roots[0].transform
        #expect(close(transform.translation, SIMD3(1, 6, 0)))
        #expect(close(transform.scale, SIMD3(repeating: 2.5)))
        let half = Float(0.5).squareRoot()
        let expected = simd_normalize(SIMD4<Float>(0, 0, 0.75 * half, 0.25 + 0.75 * half))
        #expect(close(alignedVector(transform.rotation, to: expected), expected))
        #expect(blended.roots[1].transform.translation == bystander.transform.translation)
        #expect(blended.roots[1].transform.scale == bystander.transform.scale)
        #expect(graph.roots[0].transform.translation == SIMD3(9, 9, 9))
    }

    @Test
    func `Partial weights blend toward the graph's rest pose`() throws {
        let rest = SceneTransform(translation: SIMD3(2, 0, 0))
        let node = SceneNode(transform: rest)
        let graph = try makeGraph([node])
        let yTurn = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
        let clip = try constantClip(node.id, .init(translation: SIMD3(6, 0, 0), rotation: yTurn, scale: SIMD3(repeating: 3)))

        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: clip, weight: 0.25)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation == rest.translation)
        try mixer.play(action)
        let transform = try mixer.apply(to: graph).roots[0].transform
        #expect(close(transform.translation, SIMD3(3, 0, 0)))
        #expect(close(transform.scale, SIMD3(repeating: 1.5)))
        let expected = simd_normalize(yTurn.vector * 0.25 + SIMD4(0, 0, 0, 1) * 0.75)
        #expect(close(alignedVector(transform.rotation, to: expected), expected))

        try mixer.setWeight(0, for: action)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation == rest.translation)
    }

    @Test
    func `Cross-fades weights over mixer time and stops the outgoing action`() throws {
        let node = SceneNode()
        let graph = try makeGraph([node])
        var mixer = AnimationMixer()
        let walk = try mixer.addAction(for: constantClip(node.id, .init(translation: SIMD3(10, 0, 0))))
        let run = try mixer.addAction(for: constantClip(node.id, .init(translation: SIMD3(20, 0, 0))))
        try mixer.play(walk)
        try mixer.crossFade(from: walk, to: run, duration: 1)

        #expect(try mixer.action(walk).effectiveWeight == 1)
        #expect(try mixer.action(run).effectiveWeight == 0)
        #expect(try mixer.action(run).isPlaying)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 10)

        #expect(try mixer.update(deltaTime: 0.5).isEmpty)
        #expect(try mixer.action(walk).effectiveWeight == 0.5)
        #expect(try mixer.action(run).effectiveWeight == 0.5)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 15)

        #expect(try mixer.update(deltaTime: 0.5).isEmpty)
        #expect(mixer.time == 1)
        #expect(try !mixer.action(walk).isPlaying)
        #expect(try mixer.action(walk).effectiveWeight == 0)
        #expect(try mixer.action(run).effectiveWeight == 1)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 20)
    }

    @Test
    func `Warped cross-fades synchronize loop periods by integrating the ramp`() throws {
        let node = SceneNode()
        var mixer = AnimationMixer()
        let long = try mixer.addAction(for: rampClip(node.id, duration: 2))
        let short = try mixer.addAction(for: rampClip(node.id, duration: 1))
        try mixer.play(long)
        try mixer.crossFade(from: long, to: short, duration: 1, warp: true)
        #expect(try mixer.action(long).effectiveTimeScale == 1)
        #expect(try mixer.action(short).effectiveTimeScale == 0.5)

        _ = try mixer.update(deltaTime: 0.5)
        #expect(try mixer.action(long).time == 0.625)
        #expect(try mixer.action(long).effectiveTimeScale == 1.5)
        #expect(try mixer.action(short).time == 0.3125)
        #expect(try mixer.action(short).effectiveTimeScale == 0.75)

        _ = try mixer.update(deltaTime: 0.5)
        #expect(try !mixer.action(long).isPlaying)
        #expect(try mixer.action(short).time == 0.75)
        #expect(try mixer.action(short).timeScale == 1)
        #expect(try mixer.action(short).effectiveTimeScale == 1)
    }

    @Test
    func `A warp to zero halts at the integrated time and keeps the configured rate`() throws {
        let node = SceneNode()
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: rampClip(node.id, duration: 4), timeScale: 2)
        try mixer.play(action)
        try mixer.warp(action, from: 2, to: 0, duration: 1)
        _ = try mixer.update(deltaTime: 2)
        let halted = try mixer.action(action)
        #expect(halted.time == 1)
        #expect(halted.isPaused)
        #expect(!halted.isRunning)
        #expect(halted.timeScale == 2)
        _ = try mixer.update(deltaTime: 1)
        #expect(try mixer.action(action).time == 1)
    }

    @Test
    func `Once actions clamp on their final pose or stop and release the rest pose`() throws {
        let node = SceneNode(transform: .init(translation: SIMD3(-1, 0, 0)))
        let graph = try makeGraph([node])
        let clip = try rampClip(node.id, duration: 2)
        var mixer = AnimationMixer()
        let clamped = try mixer.addAction(for: clip, loop: .once, clampWhenFinished: true)
        try mixer.play(clamped)
        #expect(try mixer.update(deltaTime: 1.5).isEmpty)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 3)
        #expect(try mixer.update(deltaTime: 1) == [.finished(clamped, direction: .forward)])
        let finished = try mixer.action(clamped)
        #expect(finished.isPlaying)
        #expect(finished.isPaused)
        #expect(finished.time == 2)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 4)
        #expect(try mixer.update(deltaTime: 1).isEmpty)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 4)

        try mixer.stop(clamped)
        let released = try mixer.addAction(for: clip, loop: .once)
        try mixer.play(released)
        #expect(try mixer.update(deltaTime: 2.5) == [.finished(released, direction: .forward)])
        #expect(try !mixer.action(released).isPlaying)
        #expect(try mixer.action(released).time == 0)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == -1)
    }

    @Test
    func `Ping-pong reverses odd passes and clamps where the last pass ends`() throws {
        let node = SceneNode()
        let graph = try makeGraph([node])
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: rampClip(node.id, duration: 2), loop: .pingPong(count: 2), clampWhenFinished: true)
        try mixer.play(action)

        #expect(try mixer.update(deltaTime: 1.5).isEmpty)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 3)
        #expect(try mixer.update(deltaTime: 1) == [.looped(action, loopDelta: 1)])
        #expect(try mixer.action(action).clipTime == 1.5)
        #expect(try mixer.update(deltaTime: 1).isEmpty)
        #expect(try mixer.action(action).clipTime == 0.5)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 1)
        #expect(try mixer.update(deltaTime: 1) == [.finished(action, direction: .forward)])
        let finished = try mixer.action(action)
        #expect(finished.isPaused)
        #expect(finished.clipTime == 0)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 0)
    }

    @Test
    func `Repeating actions report every wrapped pass and finish after their count`() throws {
        let node = SceneNode()
        var mixer = AnimationMixer()
        let endless = try mixer.addAction(for: rampClip(node.id, duration: 2))
        let limited = try mixer.addAction(for: rampClip(node.id, duration: 2), loop: .repeat(count: 2))
        try mixer.play(endless)
        try mixer.play(limited)
        let events = try mixer.update(deltaTime: 5)
        #expect(events == [.looped(endless, loopDelta: 2), .finished(limited, direction: .forward)])
        #expect(try mixer.action(endless).time == 1)
        #expect(try !mixer.action(limited).isPlaying)
    }

    @Test
    func `Negative and zero time scales play backward or hold the playhead`() throws {
        let node = SceneNode()
        let graph = try makeGraph([node])
        let clip = try rampClip(node.id, duration: 2)
        var mixer = AnimationMixer()
        let reversed = try mixer.addAction(for: clip, timeScale: -1)
        try mixer.play(reversed)
        // Entering a reversed loop from zero starts the first pass at the end.
        #expect(try mixer.update(deltaTime: 0.5).isEmpty)
        #expect(try mixer.action(reversed).time == 1.5)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 3)
        #expect(try mixer.update(deltaTime: 2) == [.looped(reversed, loopDelta: -1)])
        #expect(try mixer.action(reversed).time == 1.5)
        try mixer.stop(reversed)

        let once = try mixer.addAction(for: clip, timeScale: -1, loop: .once)
        try mixer.setTime(2, for: once)
        try mixer.play(once)
        #expect(try mixer.update(deltaTime: 0.5).isEmpty)
        #expect(try mixer.action(once).time == 1.5)
        #expect(try mixer.update(deltaTime: 2) == [.finished(once, direction: .backward)])
        #expect(try !mixer.action(once).isPlaying)

        let held = try mixer.addAction(for: clip, timeScale: 0)
        try mixer.setTime(0.5, for: held)
        try mixer.play(held)
        #expect(try mixer.update(deltaTime: 3).isEmpty)
        #expect(try mixer.action(held).time == 0.5)
        #expect(try !mixer.action(held).isRunning)
        #expect(try mixer.apply(to: graph).roots[0].transform.translation.x == 1)
    }

    @Test
    func `Additive clips add weighted offsets from their reference pose`() throws {
        let node = SceneNode(transform: .init(translation: SIMD3(5, 5, 5)))
        let graph = try makeGraph([node])
        let base = try constantClip(node.id, .init(
            translation: SIMD3(1, 0, 0), rotation: simd_quatf(angle: .pi / 6, axis: zAxis)
        ))
        let nod = try AnimationClip(SceneAnimation(tracks: [SceneAnimationTrack(nodeID: node.id, keyframes: [
            .init(time: 0, transform: .init()),
            .init(time: 1, transform: .init(
                translation: SIMD3(0, 2, 0), rotation: simd_quatf(angle: .pi / 2, axis: zAxis), scale: SIMD3(2, 1, 1)
            )),
        ])]))
        let additive = try nod.makingAdditive()
        #expect(additive.blendMode == .additive)
        let offset = try #require(additive.sample(at: 1)[node.id])
        #expect(offset.translation == SIMD3(0, 2, 0))
        #expect(offset.scale == SIMD3(1, 0, 0))

        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: base))
        let layer = try mixer.addAction(for: additive, weight: 0.5, timeScale: 0)
        try mixer.setTime(1, for: layer)
        try mixer.play(layer)
        var transform = try mixer.apply(to: graph).roots[0].transform
        #expect(close(transform.translation, SIMD3(1, 1, 0)))
        #expect(close(transform.scale, SIMD3(1.5, 1, 1)))
        #expect(close(transform.rotation.act(SIMD3(1, 0, 0)), SIMD3(cos(5 * .pi / 12), sin(5 * .pi / 12), 0)))

        // Without a normal action, the offset applies on top of the rest pose.
        try mixer.stop(mixer.actions[0].id)
        try mixer.setWeight(1, for: layer)
        transform = try mixer.apply(to: graph).roots[0].transform
        #expect(close(transform.translation, SIMD3(5, 7, 5)))
        #expect(close(transform.rotation.act(SIMD3(1, 0, 0)), SIMD3(0, 1, 0)))
    }

    @Test
    func `Additive rotation offsets apply in the base's local frame`() throws {
        let node = SceneNode()
        let graph = try makeGraph([node])
        let tilt = simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))
        let spin = simd_quatf(angle: .pi / 2, axis: zAxis)
        let reference = try constantClip(node.id, .init(rotation: tilt))
        let pose = try constantClip(node.id, .init(rotation: tilt * spin))
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: constantClip(node.id, .init(rotation: tilt))))
        try mixer.play(mixer.addAction(for: pose.makingAdditive(reference: reference)))
        let rotation = try mixer.apply(to: graph).roots[0].transform.rotation
        #expect(close(rotation.act(SIMD3(1, 0, 0)), (tilt * spin).act(SIMD3(1, 0, 0))))
        #expect(close(rotation.act(SIMD3(0, 1, 0)), (tilt * spin).act(SIMD3(0, 1, 0))))
    }

    @Test
    func `Fade-in starts a stopped action and fade-out stops it`() throws {
        let node = SceneNode()
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: constantClip(node.id, .init(translation: SIMD3(1, 0, 0))), weight: 0.5)
        try mixer.fadeIn(action, duration: 2)
        #expect(try mixer.action(action).isPlaying)
        #expect(try mixer.action(action).effectiveWeight == 0)
        _ = try mixer.update(deltaTime: 1)
        #expect(try mixer.action(action).effectiveWeight == 0.25)
        try mixer.fadeOut(action, duration: 1)
        _ = try mixer.update(deltaTime: 0.5)
        #expect(try mixer.action(action).fadeFactor == 0.25)
        _ = try mixer.update(deltaTime: 0.5)
        #expect(try !mixer.action(action).isPlaying)
        #expect(try mixer.action(action).fadeFactor == 1)
    }

    @Test
    func `Imported glTF clips mix only the properties they animate`() throws {
        var fixture = GLTFMotionFixture()
        let times = fixture.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let moves = fixture.floats([0, 0, 0, 4, 0, 0], type: "VEC3", count: 2)
        let animation: [String: Any] = [
            "channels": [["target": ["node": 0, "path": "translation"], "sampler": 0]],
            "samplers": [["input": times, "output": moves]],
        ]
        let scene = try GLTFMeshLoader.loadScene(data: fixture.document(["animations": [animation]]), includeAnimations: true)
        let imported = try #require(scene.animations.first)
        let node = SceneNode(transform: .init(rotation: simd_quatf(angle: 1, axis: zAxis), scale: SIMD3(repeating: 2)))
        let graph = try makeGraph([node])
        #expect(throws: GLTFAnimationClip.SamplingError.missingNodeMapping(0)) { try AnimationClip(imported, nodeIDs: [:]) }
        let clip = try AnimationClip(imported, nodeIDs: [0: node.id])
        #expect(clip.duration == 2)
        #expect(clip.nodeIDs == [node.id])
        #expect(try clip.sample(at: 1)[node.id] == AnimationNodePose(translation: SIMD3(2, 0, 0)))

        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: clip, weight: 0.5))
        _ = try mixer.update(deltaTime: 1)
        let transform = try mixer.apply(to: graph).roots[0].transform
        #expect(transform.translation == SIMD3(1, 0, 0))
        #expect(transform.rotation == node.transform.rotation)
        #expect(transform.scale == node.transform.scale)
    }

    @Test
    func `Scene animations sample transforms without a graph`() throws {
        let id = UUID()
        let animation = try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: id, keyframes: [
            .init(time: 0, transform: .init()), .init(time: 2, transform: .init(translation: SIMD3(4, 0, 0))),
        ])])
        #expect(try animation.sample(at: 1)[id]?.translation == SIMD3(2, 0, 0))
        #expect(try animation.sample(at: 3, looping: true)[id]?.translation == SIMD3(2, 0, 0))
        #expect(throws: SceneAnimation.ValidationError.invalidSampleTime) { try animation.sample(at: .nan) }
    }

    @Test
    func `Rejects invalid mixer inputs without changing state`() throws {
        let node = SceneNode()
        let clip = try rampClip(node.id, duration: 2)
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: clip)
        let other = try mixer.addAction(for: try constantClip(node.id, .init()))
        var otherMixer = AnimationMixer()
        let stranger = try otherMixer.addAction(for: clip)

        #expect(throws: AnimationMixer.Failure.invalidDeltaTime) { try mixer.update(deltaTime: -0.1) }
        #expect(throws: AnimationMixer.Failure.invalidDeltaTime) { try mixer.update(deltaTime: .nan) }
        #expect(throws: AnimationMixer.Failure.invalidWeight) { try mixer.setWeight(-1, for: action) }
        #expect(throws: AnimationMixer.Failure.invalidWeight) { try mixer.addAction(for: clip, weight: .infinity) }
        #expect(throws: AnimationMixer.Failure.invalidTimeScale) { try mixer.setTimeScale(.nan, for: action) }
        #expect(throws: AnimationMixer.Failure.invalidLoopCount) { try mixer.setLoop(.pingPong(count: 0), for: action) }
        #expect(throws: AnimationMixer.Failure.invalidTime) { try mixer.setTime(2.5, for: action) }
        #expect(throws: AnimationMixer.Failure.invalidDuration) { try mixer.fadeIn(action, duration: -1) }
        #expect(throws: AnimationMixer.Failure.unknownAction(stranger)) { try mixer.play(stranger) }
        #expect(throws: AnimationMixer.Failure.notPlaying(action)) { try mixer.fadeOut(action, duration: 1) }
        #expect(throws: AnimationMixer.Failure.notPlaying(action)) { try mixer.warp(action, from: 1, to: 2, duration: 1) }
        #expect(throws: AnimationMixer.Failure.notPlaying(action)) { try mixer.crossFade(from: action, to: other, duration: 1) }
        try mixer.play(action)
        #expect(throws: AnimationMixer.Failure.sameAction) { try mixer.crossFade(from: action, to: action, duration: 1) }
        #expect(throws: AnimationMixer.Failure.warpRequiresDuration) {
            try mixer.crossFade(from: action, to: other, duration: 1, warp: true)
        }
        #expect(try !mixer.action(other).isPlaying)
        #expect(mixer.time == 0)

        let emptyGraph = try makeGraph([SceneNode()])
        #expect(throws: AnimationMixer.Failure.missingNode(node.id)) { try mixer.apply(to: emptyGraph) }
        #expect(throws: AnimationMixer.Failure.invalidRestWeights(node.id)) { try mixer.morphWeights(for: node.id, rest: []) }
        #expect(try mixer.morphWeights(for: node.id, rest: [0.5]) == [0.5])
        #expect(throws: AnimationClip.Failure.missingReference(node.id)) {
            try clip.makingAdditive(reference: constantClip(UUID(), .init()))
        }
        #expect(throws: AnimationClip.Failure.alreadyAdditive) { try clip.makingAdditive().makingAdditive() }
        #expect(throws: AnimationClip.Failure.invalidReferenceTime) { try clip.makingAdditive(referenceTime: -1) }
        try mixer.removeAction(action)
        #expect(throws: AnimationMixer.Failure.unknownAction(action)) { try mixer.action(action) }
    }

    // MARK: - Helpers

    /// A clip that moves along X from 0 to twice the duration.
    private func rampClip(_ id: UUID, duration: Float) throws -> AnimationClip {
        try AnimationClip(SceneAnimation(tracks: [SceneAnimationTrack(nodeID: id, keyframes: [
            .init(time: 0, transform: .init()),
            .init(time: duration, transform: .init(translation: SIMD3(2 * duration, 0, 0))),
        ])]))
    }

    private func constantClip(_ id: UUID, _ transform: SceneTransform) throws -> AnimationClip {
        try AnimationClip(SceneAnimation(tracks: [SceneAnimationTrack(nodeID: id, keyframes: [.init(time: 0, transform: transform)])]))
    }

    private func makeGraph(_ roots: [SceneNode]) throws -> SceneGraph {
        SceneGraph(roots: roots, camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
    }

    private func alignedVector(_ rotation: simd_quatf, to reference: SIMD4<Float>) -> SIMD4<Float> {
        simd_dot(rotation.vector, reference) < 0 ? -rotation.vector : rotation.vector
    }

    private func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>, tolerance: Float = 1e-5) -> Bool {
        simd_length(a - b) <= tolerance
    }

    private func close(_ a: SIMD4<Float>, _ b: SIMD4<Float>, tolerance: Float = 1e-5) -> Bool {
        simd_length(a - b) <= tolerance
    }
}
