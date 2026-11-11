import Foundation
@testable import Frostfield
import Testing

struct AnimationObjectGroupTests {
    @Test("Members join the running fade, leave without a reset, and can reactivate")
    func dynamicMembership() throws {
        let template = SceneNode(children: [SceneNode()])
        let templateChild = template.children[0].id
        let first = SceneNode(children: [SceneNode()])
        let second = SceneNode(transform: .init(translation: SIMD3(6, 0, 0)), children: [SceneNode()])
        let firstMap = [template.id: first.id, templateChild: first.children[0].id]
        let secondMap = [template.id: second.id, templateChild: second.children[0].id]
        let graph = SceneGraph(
            roots: [template, first, second], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
        let group = try AnimationObjectGroup(sourceNodes: [template.id, templateChild])
        let clip = try AnimationClip(tracks: [
            .vector(.node(template.id, .translation), times: [0, 2], values: [.zero, SIMD3(8, 0, 0)]),
            .boolean(.node(templateChild, .isVisible), times: [0, 0.25], values: [true, false]),
            .number(.environment(.intensity), times: [0, 2], values: [1, 5]),
        ])
        try group.add(member: first.id, mapping: firstMap)
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: clip, on: group, loop: .once)
        try mixer.fadeIn(action, duration: 0.8)
        try mixer.update(deltaTime: 0.5)
        var posed = try mixer.apply(to: graph)
        #expect(posed.roots.map(\.transform.translation.x) == [0, 1.25, 6])
        #expect(!posed.roots[1].children[0].isVisible)

        try group.add(member: second.id, mapping: secondMap)
        try group.add(member: second.id, mapping: secondMap)
        posed = try mixer.apply(to: graph)
        #expect(posed.roots.map(\.transform.translation.x) == [0, 1.25, 3.5])
        #expect(!posed.roots[2].children[0].isVisible)
        #expect(posed.environmentIntensity == 1.625)
        #expect(try mixer.value(of: .node(second.children[0].id, .isVisible), rest: .boolean(true)) == .boolean(false))
        #expect(try mixer.value(of: .node(templateChild, .isVisible), rest: .boolean(true)) == .boolean(true))
        #expect(try mixer.action(action).time == 0.5)
        #expect(try mixer.action(action).effectiveWeight == 0.625)

        group.remove(member: first.id)
        group.remove(member: second.id)
        #expect(group.cachedMemberCount == 2)
        posed = try mixer.apply(to: graph)
        #expect(posed.roots.map(\.transform.translation.x) == [0, 0, 6])
        #expect(posed.environmentIntensity == 1.625)
        try mixer.update(deltaTime: 0.5)
        try group.add(member: first.id, mapping: firstMap)
        posed = try mixer.apply(to: graph)
        #expect(posed.roots.map(\.transform.translation.x) == [0, 4, 6])
        #expect(group.activeMembers == [first.id])
        #expect(try mixer.action(action).time == 1)
        #expect(try mixer.action(action).effectiveWeight == 1)
    }

    @Test("Group validation is atomic and uncaching releases destination IDs")
    func validationAndUncache() throws {
        let source = UUID(), child = UUID(), member = UUID(), target = UUID(), other = UUID()
        let group = try AnimationObjectGroup(sourceNodes: [source, child])
        #expect(throws: AnimationObjectGroup.Failure.invalidMapping(member)) {
            try group.add(member: member, mapping: [source: target, child: target])
        }
        #expect(group.memberCount == 0)
        let mapping = [source: target, child: other]
        try group.add(member: member, mapping: mapping)
        group.remove(member: member)
        let replacement = UUID()
        #expect(throws: AnimationObjectGroup.Failure.duplicateDestination(target)) {
            try group.add(member: replacement, mapping: [source: target, child: replacement])
        }
        #expect(throws: AnimationObjectGroup.Failure.changedMapping(member)) {
            try group.add(member: member, mapping: [source: replacement, child: other])
        }
        #expect(group.cachedMemberCount == 1 && group.activeMembers.isEmpty)
        group.uncache(member: member)
        try group.add(member: replacement, mapping: mapping)
        #expect(group.cachedMemberCount == 0 && group.activeMembers == [replacement])

        var mixer = AnimationMixer()
        let clip = try AnimationClipCreator.visibility(node: replacement, duration: 1)
        #expect(throws: AnimationObjectGroup.Failure.unknownClipNode(replacement)) {
            try mixer.addAction(for: clip, on: group)
        }
        #expect(mixer.actions.isEmpty)
    }

    @Test("Grouped imported and typed morph tracks combine with each destination's rest weights")
    func morphBindings() throws {
        let source = UUID(), first = UUID(), second = UUID()
        let imported = GLTFAnimationClip(
            sourceIndex: 0, name: "Morphs", channels: [
                .init(sourceNodeIndex: 0, interpolation: .linear, times: [0, 1], weights: [0, 0.5, 1, 1], weightCount: 2),
            ]
        )
        let importedClip = try AnimationClip(imported, nodeIDs: [0: source])
        let single = try AnimationClip(tracks: [
            .number(.node(source, .morphWeight(1)), times: [0, 1], values: [0.25, 0.75]),
        ]).makingAdditive()
        let group = try AnimationObjectGroup(sourceNodes: [source])
        try group.add(member: first, mapping: [source: first])
        try group.add(member: second, mapping: [source: second])
        var mixer = AnimationMixer()
        let weights = try mixer.addAction(for: importedClip, on: group, weight: 0.5)
        let additive = try mixer.addAction(for: single, on: group)
        try mixer.play(weights)
        try mixer.play(additive)
        try mixer.update(deltaTime: 0.5)
        #expect(try mixer.morphWeights(for: first, rest: [0, 0]) == [0.25, 0.625])
        #expect(try mixer.morphWeights(for: second, rest: [0.5, 0.5]) == [0.5, 0.875])
        group.uncache(member: second)
        #expect(try mixer.morphWeights(for: second, rest: [0.5, 0.5]) == [0.5, 0.5])
        #expect(try mixer.morphWeights(for: source, rest: [0, 0]) == [0, 0])
    }
}
