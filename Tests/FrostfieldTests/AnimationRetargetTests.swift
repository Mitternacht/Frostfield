import Foundation
import Frostfield
import Testing
import simd

struct AnimationRetargetTests {
    /// A chain hips → spine → arm → hand along +Y with unit bones and identity frames.
    private func sourceSkeleton() throws -> AnimationSkeleton {
        try AnimationSkeleton(joints: [
            .init(id: UUID(), name: "hips", parent: nil, rest: .init(translation: SIMD3(0, 1, 0))),
            .init(id: UUID(), name: "spine", parent: 0, rest: .init(translation: SIMD3(0, 1, 0))),
            .init(id: UUID(), name: "arm", parent: 1, rest: .init(translation: SIMD3(0, 1, 0))),
            .init(id: UUID(), name: "hand", parent: 2, rest: .init(translation: SIMD3(0, 1, 0))),
        ])
    }

    /// The same rest shape scaled by two, with every joint frame rolled differently,
    /// as rigs exported from different tools are. Child offsets are written in the
    /// rolled parent frames so the world-space rest pose matches.
    private func targetSkeleton(names: [String]) throws -> AnimationSkeleton {
        let rolls = [
            simd_quatf(angle: 0.7, axis: SIMD3(0, 1, 0)), simd_quatf(angle: -1.1, axis: simd_normalize(SIMD3(1, 0, 1))),
            simd_quatf(angle: 2.0, axis: SIMD3(0, 0, 1)), simd_quatf(angle: 0.4, axis: SIMD3(1, 0, 0)),
        ]
        var joints: [AnimationSkeleton.Joint] = []
        var parentWorld = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
        for index in 0..<4 {
            let world = rolls[index]
            let local = parentWorld.inverse * world
            let offset = parentWorld.inverse.act(SIMD3(0, 2, 0))
            joints.append(.init(id: UUID(), name: names[index], parent: index == 0 ? nil : index - 1,
                                rest: .init(translation: offset, rotation: local)))
            parentWorld = world
        }
        return try AnimationSkeleton(joints: joints)
    }

    private func worldPositions(_ skeleton: AnimationSkeleton, poses: [UUID: AnimationNodePose]) -> [SIMD3<Float>] {
        var worlds: [simd_float4x4] = []
        for joint in skeleton.joints {
            let local = (poses[joint.id]?.applied(to: joint.rest) ?? joint.rest).matrix
            worlds.append(joint.parent.map { worlds[$0] * local } ?? local)
        }
        return worlds.map { SIMD3($0.columns.3.x, $0.columns.3.y, $0.columns.3.z) }
    }

    @Test("Additive retargeting layers source offsets over rest and preserves the target blend mode")
    func additiveRetarget() throws {
        let source = try sourceSkeleton()
        let target = try targetSkeleton(names: source.joints.map(\.name))
        let id = source.joints[0].id
        let referenceRotation = simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0))
        let delta = simd_quatf(angle: 0.8, axis: SIMD3(1, 0, 0))
        let move = SIMD3<Float>(1, 2, 1)
        let authored = try AnimationClip(tracks: [
            .quaternion(.node(id, .rotation), times: [0, 1], values: [referenceRotation, referenceRotation * delta]),
            .vector(.node(id, .translation), times: [0, 1], values: [SIMD3(3, 5, 7), SIMD3(3, 5, 7) + move]),
        ])
        let additive = try authored.makingAdditive().retargeted(from: source, to: target, framesPerSecond: 8)
        let sourceRest = source.joints[0].rest
        let absolute = try AnimationClip(tracks: [
            .quaternion(.node(id, .rotation), times: [0, 1], values: [sourceRest.rotation, sourceRest.rotation * delta]),
            .vector(.node(id, .translation), times: [0, 1], values: [sourceRest.translation, sourceRest.translation + move]),
        ]).retargeted(from: source, to: target, framesPerSecond: 8)
        #expect(additive.blendMode == .additive)
        for time: Float in [0, 0.5, 1] {
            let offsets = try additive.sample(at: time)
            let poses = try absolute.sample(at: time)
            let root = target.joints[0]
            let translation = try #require(offsets[root.id]?.translation)
            let expectedPosition = try #require(poses[root.id]?.translation)
            #expect(simd_distance(root.rest.translation + translation, expectedPosition) < 1e-5)
            for joint in target.joints {
                let offset = try #require(offsets[joint.id]?.rotation)
                let expected = try #require(poses[joint.id]?.rotation)
                let actual = joint.rest.rotation * offset
                #expect(abs(simd_dot(actual.vector, expected.vector)) > 0.99999)
            }
        }
        #expect(throws: AnimationClip.Failure.invalidFrameRange) {
            try authored.retargeted(from: source, to: target, framesPerSecond: .greatestFiniteMagnitude)
        }
    }

    @Test
    func `Retargeting reproduces the source end effector across joint frames and proportions`() throws {
        let source = try sourceSkeleton()
        let target = try targetSkeleton(names: ["Hips", "Chest", "UpperArm", "Hand"])
        let ids = source.joints.map(\.id)
        let clip = try AnimationClip(name: "wave", tracks: [
            .vector(.node(ids[0], .translation), times: [0, 1], values: [SIMD3(0, 1, 0), SIMD3(0.5, 1.2, -0.3)]),
            .quaternion(.node(ids[0], .rotation), times: [0, 1], values: [
                simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)), simd_quatf(angle: 0.6, axis: SIMD3(0, 1, 0)),
            ]),
            .quaternion(.node(ids[1], .rotation), times: [0, 1], values: [
                simd_quatf(angle: 0, axis: SIMD3(1, 0, 0)), simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3(1, 0, 1))),
            ]),
            .quaternion(.node(ids[2], .rotation), times: [0, 1], values: [
                simd_quatf(angle: 0.2, axis: SIMD3(0, 0, 1)), simd_quatf(angle: 1.4, axis: SIMD3(0, 0, 1)),
            ]),
        ])
        let retargeted = try clip.retargeted(
            from: source, to: target,
            boneNames: ["Hips": "hips", "Chest": "spine", "UpperArm": "arm", "Hand": "hand"], framesPerSecond: 10
        )
        #expect(retargeted.duration == 1)
        for frame in 0...10 {
            let time = Float(frame) / 10
            let sourceWorld = worldPositions(source, poses: try clip.sample(at: time))
            let targetWorld = worldPositions(target, poses: try retargeted.sample(at: time))
            // The target is the source scaled by two about the skeleton origin's rest root.
            let sourceRoot = sourceWorld[0]
            let targetRoot = targetWorld[0]
            let expectedRoot = SIMD3<Float>(0, 2, 0) + (sourceRoot - SIMD3(0, 1, 0)) * 2
            #expect(simd_distance(targetRoot, expectedRoot) < 1e-4, "root at \(time)")
            for joint in 1..<4 {
                let expected = targetRoot + (sourceWorld[joint] - sourceRoot) * 2
                #expect(simd_distance(targetWorld[joint], expected) < 1e-4, "joint \(joint) at \(time)")
            }
        }
    }

    @Test
    func `Retargeting handles rolled source frames, scaled joints and a scaled unmapped parent`() throws {
        // The source of the first test, with rolled joint frames and the clip rewritten in them.
        let plain = try sourceSkeleton()
        let rolls = [
            simd_quatf(angle: -0.9, axis: SIMD3(0, 0, 1)), simd_quatf(angle: 1.3, axis: simd_normalize(SIMD3(0, 1, 1))),
            simd_quatf(angle: 0.6, axis: SIMD3(1, 0, 0)), simd_quatf(angle: 2.2, axis: SIMD3(0, 1, 0)),
        ]
        let source = try AnimationSkeleton(joints: (0..<4).map { index in
            let parentRoll = index == 0 ? simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)) : rolls[index - 1]
            return .init(
                id: UUID(), name: plain.joints[index].name, parent: plain.joints[index].parent,
                rest: .init(translation: parentRoll.inverse.act(SIMD3(0, 1, 0)), rotation: parentRoll.inverse * rolls[index])
            )
        })
        let ids = plain.joints.map(\.id)
        let plainClip = try AnimationClip(tracks: [
            .vector(.node(ids[0], .translation), times: [0, 1], values: [SIMD3(0, 1, 0), SIMD3(0.5, 1.2, -0.3)]),
            .quaternion(.node(ids[0], .rotation), times: [0, 1], values: [
                simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)), simd_quatf(angle: 0.6, axis: SIMD3(0, 1, 0)),
            ]),
            .quaternion(.node(ids[1], .rotation), times: [0, 1], values: [
                simd_quatf(angle: 0, axis: SIMD3(1, 0, 0)), simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3(1, 0, 1))),
            ]),
            .quaternion(.node(ids[2], .rotation), times: [0, 1], values: [
                simd_quatf(angle: 0.2, axis: SIMD3(0, 0, 1)), simd_quatf(angle: 1.4, axis: SIMD3(0, 0, 1)),
            ]),
        ])
        let times = (0...10).map { Float($0) / 10 }
        var rotationKeys = Array(repeating: [simd_quatf](), count: 4)
        var rootKeys: [SIMD3<Float>] = []
        for time in times {
            let poses = try plainClip.sample(at: time)
            var world: [simd_quatf] = []
            for (index, joint) in plain.joints.enumerated() {
                let local = poses[joint.id]?.rotation ?? simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
                world.append(index == 0 ? local : world[index - 1] * local)
                let parentFrame = index == 0 ? simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)) : world[index - 1] * rolls[index - 1]
                rotationKeys[index].append(simd_normalize(parentFrame.inverse * world[index] * rolls[index]))
            }
            rootKeys.append(try #require(poses[ids[0]]?.translation))
        }
        var tracks = try (0..<4).map { try AnimationTrack.quaternion(.node(source.joints[$0].id, .rotation), times: times, values: rotationKeys[$0]) }
        tracks.append(try .vector(.node(source.joints[0].id, .translation), times: times, values: rootKeys))
        let clip = try AnimationClip(tracks: tracks)

        // Target bones are twice as long. An unmapped armature, rotated and scaled by one half, holds
        // a hips joint scaled by three whose child undoes it, so local offsets differ from bone lengths.
        func local(position: SIMD3<Float>, rotation: simd_quatf, scale: Float, parent: (SIMD3<Float>, simd_quatf, Float)?)
            -> SceneTransform {
            guard let parent else {
                return SceneTransform(translation: position, rotation: rotation, scale: SIMD3(repeating: scale))
            }
            return SceneTransform(
                translation: parent.1.inverse.act(position - parent.0) / parent.2,
                rotation: parent.1.inverse * rotation, scale: SIMD3(repeating: scale / parent.2)
            )
        }
        let armature = (SIMD3<Float>(3, 0, 1), simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)), Float(0.5))
        let frames: [(SIMD3<Float>, simd_quatf, Float)] = [
            (SIMD3(0, 2, 0), simd_quatf(angle: 0.7, axis: SIMD3(0, 1, 0)), 1.5),
            (SIMD3(0, 4, 0), simd_quatf(angle: -1.1, axis: simd_normalize(SIMD3(1, 0, 1))), 1),
            (SIMD3(0, 6, 0), simd_quatf(angle: 2.0, axis: SIMD3(0, 0, 1)), 1),
            (SIMD3(0, 8, 0), simd_quatf(angle: 0.4, axis: SIMD3(1, 0, 0)), 1),
        ]
        var targetJoints: [AnimationSkeleton.Joint] = [
            .init(id: UUID(), name: "Armature", parent: nil, rest: local(position: armature.0, rotation: armature.1, scale: armature.2, parent: nil)),
        ]
        for (index, frame) in frames.enumerated() {
            let parent = index == 0 ? armature : frames[index - 1]
            targetJoints.append(.init(
                id: UUID(), name: ["Hips", "Chest", "UpperArm", "Hand"][index], parent: index,
                rest: local(position: frame.0, rotation: frame.1, scale: frame.2, parent: parent)
            ))
        }
        let target = try AnimationSkeleton(joints: targetJoints)
        #expect(simd_distance(worldPositions(target, poses: [:])[4], SIMD3(0, 8, 0)) < 1e-5)

        let retargeted = try clip.retargeted(
            from: source, to: target,
            boneNames: ["Hips": "hips", "Chest": "spine", "UpperArm": "arm", "Hand": "hand"], framesPerSecond: 10
        )
        for time in times {
            let sourceWorld = worldPositions(source, poses: try clip.sample(at: time))
            let plainWorld = worldPositions(plain, poses: try plainClip.sample(at: time))
            #expect(zip(sourceWorld, plainWorld).allSatisfy { simd_distance($0, $1) < 1e-5 }, "rolled source at \(time)")
            let targetWorld = Array(worldPositions(target, poses: try retargeted.sample(at: time)).dropFirst())
            let expectedRoot = SIMD3<Float>(0, 2, 0) + (sourceWorld[0] - SIMD3(0, 1, 0)) * 2
            #expect(simd_distance(targetWorld[0], expectedRoot) < 1e-4, "root at \(time)")
            for joint in 1..<4 {
                let expected = targetWorld[0] + (sourceWorld[joint] - sourceWorld[0]) * 2
                #expect(simd_distance(targetWorld[joint], expected) < 1e-4, "joint \(joint) at \(time)")
            }
        }

        // A root scaled by two doubles its child bone, so root motion doubles too.
        let small = try AnimationSkeleton(joints: [
            .init(id: UUID(), name: "hips", parent: nil, rest: .init()),
            .init(id: UUID(), name: "spine", parent: 0, rest: .init(translation: SIMD3(0, 1, 0))),
        ])
        let large = try AnimationSkeleton(joints: [
            .init(id: UUID(), name: "hips", parent: nil, rest: .init(scale: SIMD3(repeating: 2))),
            .init(id: UUID(), name: "spine", parent: 0, rest: .init(translation: SIMD3(0, 1, 0))),
        ])
        let walk = try AnimationClip(tracks: [
            .vector(.node(small.joints[0].id, .translation), times: [0, 1], values: [.zero, SIMD3(1, 0, 0)]),
        ])
        let moved = try walk.retargeted(from: small, to: large).sample(at: 1)
        #expect(simd_distance(try #require(moved[large.joints[0].id]?.translation), SIMD3(2, 0, 0)) < 1e-6)
    }

    @Test
    func `Unmapped joints keep their rest rotation and names must exist`() throws {
        let source = try sourceSkeleton()
        let target = try targetSkeleton(names: ["hips", "spine", "Elbow", "hand"])
        let clip = try AnimationClip(tracks: [
            .quaternion(.node(source.joints[1].id, .rotation), times: [0], values: [simd_quatf(angle: 0.3, axis: SIMD3(1, 0, 0))]),
        ])
        let retargeted = try clip.retargeted(from: source, to: target)
        #expect(Set(retargeted.nodeIDs) == Set([target.joints[0].id, target.joints[1].id, target.joints[3].id]))
        #expect(throws: AnimationSkeleton.Failure.unmappedJoint("Wrist")) {
            try clip.retargeted(from: source, to: target, boneNames: ["Wrist": "hand"])
        }
        #expect(throws: AnimationSkeleton.Failure.duplicateJoint("a")) {
            try AnimationSkeleton(joints: [
                .init(id: UUID(), name: "a", parent: nil, rest: .init()), .init(id: UUID(), name: "a", parent: 0, rest: .init()),
            ])
        }
        #expect(throws: AnimationSkeleton.Failure.parentOrder("b")) {
            try AnimationSkeleton(joints: [.init(id: UUID(), name: "b", parent: 1, rest: .init())])
        }
    }
}
