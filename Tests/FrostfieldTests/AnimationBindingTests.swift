import Foundation
import Frostfield
import Metal
import Testing
import simd

struct AnimationBindingTests {
    private func graph(_ roots: [SceneNode]) throws -> SceneGraph {
        SceneGraph(roots: roots, camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100))
    }

    @Test
    func `Typed tracks drive light, morph, tint and shadow properties of graph nodes`() throws {
        let lamp = SceneNode(light: try PunctualLight(
            kind: .spot(range: 10, innerConeAngle: 0.2, outerConeAngle: 0.6), color: SIMD3(1, 1, 1), intensity: 100
        ))
        let face = SceneNode(morphWeights: [0, 0, 0])
        let clip = try AnimationClip(name: "props", tracks: [
            .number(.node(lamp.id, .lightIntensity), times: [0, 2], values: [0, 200]),
            .color(.node(lamp.id, .lightColor), times: [0, 2], values: [SIMD3(1, 0, 0), SIMD3(0, 0, 1)]),
            .number(.node(lamp.id, .spotOuterConeAngle), times: [0, 2], values: [0.6, 1.0]),
            .number(.node(lamp.id, .lightRange), times: [0, 2], values: [10, 20]),
            AnimationTrack(property: .node(face.id, .morphWeights), times: [0, 2], values: [.numbers([0, 0, 0]), .numbers([1, 2, 4])]),
            .number(.node(face.id, .morphWeight(2)), times: [0, 2], values: [1, 0]),
            AnimationTrack(property: .node(face.id, .materialParameters), times: [0, 2], values: [.vector4(.zero), .vector4(SIMD4(8, 8, 8, 8))]),
            .boolean(.node(face.id, .castsShadow), times: [0, 0.25], values: [true, false]),
        ])
        let source = try graph([lamp, face])
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: clip)
        try mixer.play(action)
        try mixer.update(deltaTime: 0.5)
        let posed = try mixer.apply(to: source)

        let light = try #require(posed.roots[0].light)
        #expect(abs(light.intensity - 50) < 1e-4)
        #expect(simd_distance(light.color, SIMD3(0.75, 0, 0.25)) < 1e-6)
        guard case .spot(let range, let inner, let outer) = light.kind else { Issue.record("kind"); return }
        #expect(abs(try #require(range) - 12.5) < 1e-5)
        #expect(inner == 0.2)
        #expect(abs(outer - 0.7) < 1e-6)
        // The single-weight track applies on top of the whole-array track.
        let weights = try #require(posed.roots[1].morphWeights)
        #expect(zip(weights, [0.25, 0.5, 0.75]).allSatisfy { abs($0 - $1) < 1e-6 })
        #expect(posed.roots[1].materialParameters == SIMD4(repeating: 2))
        #expect(posed.roots[1].castsShadow == false)
        #expect(source.roots[0].light?.intensity == 100)

        // Half weight blends toward the graph's rest values; discrete values need more than half.
        try mixer.setWeight(0.5, for: action)
        let half = try mixer.apply(to: source)
        #expect(abs(try #require(half.roots[0].light).intensity - 75) < 1e-4)
        #expect(half.roots[1].castsShadow == true)
        try mixer.setWeight(0.51, for: action)
        #expect(try mixer.apply(to: source).roots[1].castsShadow == false)

        // A node without a light cannot take light tracks.
        let bare = SceneNode()
        var lonely = AnimationMixer()
        try lonely.play(lonely.addAction(for: AnimationClip(tracks: [.number(.node(bare.id, .lightIntensity), times: [0], values: [1])])))
        #expect(throws: AnimationMixer.Failure.unanimatableProperty(AnimationProperty.node(bare.id, .lightIntensity).description, "the node has no light")) {
            try lonely.apply(to: graph([bare]))
        }
    }

    @Test
    func `Discrete values select like Three.js PropertyMixer`() throws {
        let node = SceneNode(castsShadow: true)
        let source = try graph([node])
        let on = try AnimationClip(tracks: [.boolean(.node(node.id, .castsShadow), times: [0], values: [true])])
        let off = try AnimationClip(tracks: [.boolean(.node(node.id, .castsShadow), times: [0], values: [false])])
        var mixer = AnimationMixer()
        let first = try mixer.addAction(for: on, weight: 0.3)
        let second = try mixer.addAction(for: off, weight: 0.6)
        try mixer.play(second)
        // 0.6 alone exceeds one half, so it beats the rest value.
        #expect(try mixer.apply(to: source).roots[0].castsShadow == false)
        try mixer.setWeight(0.4, for: second)
        #expect(try mixer.apply(to: source).roots[0].castsShadow == true)
        try mixer.play(first)
        // 0.3 then 0.4: 0.4 / 0.7 ≥ ½ selects false, and the total 0.7 beats the rest.
        #expect(try mixer.apply(to: source).roots[0].castsShadow == false)
    }

    @Test
    func `Typed transform tracks blend with node tracks`() throws {
        let node = SceneNode()
        let nodeClip = AnimationClip(try SceneAnimation(tracks: [
            SceneAnimationTrack(nodeID: node.id, keyframes: [TransformKeyframe(time: 0, transform: .init(translation: SIMD3(4, 0, 0)))]),
        ]))
        let typed = try AnimationClip(tracks: [.vector(.node(node.id, .translation), times: [0], values: [SIMD3(0, 8, 0)])])
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: nodeClip, weight: 0.5))
        try mixer.play(mixer.addAction(for: typed, weight: 0.5))
        let posed = try mixer.apply(to: graph([node]))
        #expect(simd_distance(posed.roots[0].transform.translation, SIMD3(2, 4, 0)) < 1e-6)
        // The typed track keeps the node's scale and rotation.
        #expect(posed.roots[0].transform.scale == SIMD3(repeating: 1))

        let missing = UUID()
        var orphan = AnimationMixer()
        try orphan.play(orphan.addAction(for: AnimationClip(tracks: [.vector(.node(missing, .scale), times: [0], values: [.one])])))
        #expect(throws: AnimationMixer.Failure.missingNode(missing)) { try orphan.apply(to: graph([node])) }
    }

    @Test
    func `One replicated clip drives a group of objects like AnimationObjectGroup`() throws {
        let source = UUID()
        let a = SceneNode()
        let b = SceneNode(transform: .init(scale: SIMD3(repeating: 2)))
        let uniforms = try ShaderUniforms([ShaderUniform("glow", 0)])
        let clip = try AnimationClip(tracks: [
            .vector(.node(source, .translation), times: [0, 1], values: [.zero, SIMD3(0, 4, 0)]),
            .number(.uniform(uniforms, name: "glow"), times: [0, 1], values: [0, 1]),
        ])
        let group = try clip.replicated(onto: [[source: a.id], [source: b.id]])
        #expect(group.properties.count == 3)
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: group, weight: 0.5)
        try mixer.play(action)
        try mixer.update(deltaTime: 0.5)
        let posed = try mixer.apply(to: graph([a, b]))
        #expect(posed.roots.map(\.transform.translation) == [SIMD3(0, 1, 0), SIMD3(0, 1, 0)])
        #expect(posed.roots[1].transform.scale == SIMD3(repeating: 2))
        #expect(throws: AnimationClip.Failure.unmappedTrack(AnimationProperty.node(source, .translation).description)) {
            try clip.replicated(onto: [[UUID(): a.id]])
        }
    }

    @Test
    func `Camera and environment tracks change the graph`() throws {
        let clip = try AnimationClip(tracks: [
            .number(.camera(.verticalFieldOfView), times: [0, 1], values: [0.5, 1.5]),
            .number(.camera(.far), times: [0], values: [50]),
            .vector(.camera(.translation), times: [0, 1], values: [.zero, SIMD3(0, 0, 10)]),
            .number(.environment(.intensity), times: [0, 1], values: [0, 2]),
            .number(.environment(.rotation), times: [0, 1], values: [0, .pi]),
        ])
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: clip))
        try mixer.update(deltaTime: 0.25)
        let posed = try mixer.apply(to: graph([SceneNode()]))
        #expect(abs(posed.camera.verticalFieldOfView - 0.75) < 1e-6)
        #expect(posed.camera.near == 0.1)
        #expect(posed.camera.far == 50)
        #expect(abs(posed.cameraTransform.translation.z - 2.5) < 1e-6)
        #expect(abs(posed.environmentIntensity - 0.5) < 1e-6)
        #expect(abs(posed.environmentRotation - .pi / 4) < 1e-6)
    }

    @Test
    func `Live properties write uniforms and material factors, and stopping restores rest values`() throws {
        let uniforms = try ShaderUniforms([ShaderUniform("glow", 0.1), ShaderUniform("tint", SIMD3<Float>(1, 1, 1))])
        let material = try PhysicallyBasedMaterial(roughness: 0.5, emissiveFactor: SIMD3(0.2, 0.2, 0.2)).withLiveFactors()
        let live = try #require(material.liveFactors)
        let clip = try AnimationClip(tracks: [
            .number(.uniform(uniforms, name: "glow"), times: [0, 1], values: [0, 2]),
            .color(.uniform(uniforms, name: "tint"), times: [0, 1], values: [SIMD3(1, 0, 0), SIMD3(0, 1, 0)]),
            .number(.material(live, .roughness), times: [0, 1], values: [0.2, 0.8]),
            .color(.material(live, .emissiveFactor), times: [0, 1], values: [.zero, SIMD3(1, 0.5, 0)]),
        ])
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: clip)
        try mixer.play(action)
        try mixer.update(deltaTime: 0.25)
        let revision = live.revision
        try mixer.applyLiveProperties()
        #expect(uniforms.value(named: "glow") == SIMD4(0.5, 0, 0, 0))
        #expect(uniforms.value(named: "tint") == SIMD4(0.75, 0.25, 0, 0))
        #expect(abs(live.current.roughness - 0.35) < 1e-6)
        #expect(simd_distance(live.current.emissiveFactor, SIMD3(0.25, 0.125, 0)) < 1e-6)
        #expect(live.revision == revision + 1)
        #expect(live.current.metallic == live.rest.metallic)

        try mixer.stop(action)
        try mixer.applyLiveProperties()
        #expect(uniforms.value(named: "glow") == SIMD4(0.1, 0, 0, 0))
        #expect(live.current == live.rest)

        // Invalid blended factors are rejected by the material's own validation.
        var rough = AnimationMixer()
        try rough.play(rough.addAction(for: AnimationClip(tracks: [.number(.material(live, .roughness), times: [0], values: [1.5])])))
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidRoughness) { try rough.applyLiveProperties() }
        #expect(live.current == live.rest)
        // A factor the material lacks cannot be animated.
        var coat = AnimationMixer()
        try coat.play(coat.addAction(for: AnimationClip(tracks: [.number(.material(live, .clearcoatFactor), times: [0], values: [1])])))
        #expect(throws: PhysicallyBasedMaterial.LiveFactors.Failure.missingFactor(.clearcoatFactor)) { try coat.applyLiveProperties() }
        #expect(throws: AnimationTrack.Failure.unknownUniform("missing")) {
            try AnimationTrack.number(.uniform(uniforms, name: "missing"), times: [0], values: [0])
        }
    }

    @Test
    func `Light rigs, custom values and additive typed clips blend`() throws {
        let rig = try LightRig(lights: [
            LightRig.Light(id: "key", kind: .spot(innerAngle: 0.2, outerAngle: 0.5), position: SIMD3(0, 2, 0), target: .zero, emission: .intensity(100)),
        ])
        let clip = try AnimationClip(tracks: [
            .number(.rigLight(id: "key", .intensity), times: [0, 1], values: [100, 300]),
            .vector(.rigLight(id: "key", .position), times: [0, 1], values: [SIMD3(0, 2, 0), SIMD3(0, 4, 0)]),
            .number(.custom("door"), times: [0, 1], values: [0, 90]),
        ])
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: clip))
        try mixer.update(deltaTime: 0.5)
        let animated = try mixer.apply(to: rig)
        let key = try #require(animated["key"])
        #expect(key.emission == .intensity(200))
        #expect(key.position == SIMD3(0, 3, 0))
        #expect(try mixer.value(of: .custom("door"), rest: .number(0)) == .number(45))
        #expect(try mixer.value(of: .custom("unused"), rest: .number(7)) == .number(7))

        var power = AnimationMixer()
        try power.play(power.addAction(for: AnimationClip(tracks: [.number(.rigLight(id: "key", .power), times: [0], values: [1])])))
        #expect(throws: AnimationMixer.Failure.unanimatableProperty(AnimationProperty.rigLight(id: "key", .power).description, "its emission is an intensity")) {
            try power.apply(to: rig)
        }

        // An additive clip adds its offset from the reference time to the normal blend.
        let base = try AnimationClip(tracks: [.number(.custom("x"), times: [0], values: [10])])
        let wave = try AnimationClip(tracks: [
            .number(.custom("x"), times: [0, 2], values: [1, 5]),
            .boolean(.custom("flag"), times: [0], values: [true]),
        ]).makingAdditive()
        // Boolean and string tracks stay, unchanged, as Three.js leaves them.
        #expect(wave.properties == [.custom("x"), .custom("flag")])
        var layered = AnimationMixer()
        try layered.play(layered.addAction(for: base))
        let additive = try layered.addAction(for: wave, weight: 0.5)
        try layered.play(additive)
        try layered.setTime(1, for: additive)
        #expect(try layered.value(of: .custom("x"), rest: .number(0)) == .number(11))
        // An additive discrete value at weight one half or more replaces the blend; a lighter one does not.
        #expect(try layered.value(of: .custom("flag"), rest: .boolean(false)) == .boolean(true))
        try layered.setWeight(0.3, for: additive)
        #expect(try layered.value(of: .custom("flag"), rest: .boolean(false)) == .boolean(false))
    }

    @Test
    func `Subclips and merged clips of additive clips stay additive with the same offsets`() throws {
        let clip = try AnimationClip(tracks: [
            .number(.custom("x"), times: [0, 2], values: [10, 14]),
            .vector(.node(UUID(), .translation), times: [0, 2], values: [SIMD3(1, 0, 0), SIMD3(3, 2, 0)]),
        ]).makingAdditive()
        for (start, end) in [(Float(0), Float(60)), (30, 60)] {
            let sub = try clip.subclip(startFrame: start, endFrame: end)
            #expect(sub.blendMode == .additive)
            for time: Float in [0, 0.25, 0.5, 1] where time <= sub.duration {
                let trimmed = try sub.sampleProperties(at: time)
                for (property, value) in try clip.sampleProperties(at: time + start / 30) {
                    switch (value, trimmed[property]) {
                    case (.number(let a), .number(let b)?): #expect(abs(a - b) < 1e-5, "\(property) at \(time)")
                    case (.vector3(let a), .vector3(let b)?): #expect(simd_distance(a, b) < 1e-5, "\(property) at \(time)")
                    default: Issue.record("\(property) at \(time): \(value) vs \(String(describing: trimmed[property]))")
                    }
                }
            }
            var mixer = AnimationMixer()
            let action = try mixer.addAction(for: sub)
            try mixer.play(action)
            try mixer.setTime(Double(1 - start / 30), for: action)
            // The offset from the reference value 10 at the source's 1 s is 2.
            #expect(try mixer.value(of: .custom("x"), rest: .number(0)) == .number(2))
        }

        let node = UUID()
        let rest: [Float] = [0.25, 0.5, 0.75]
        let singles = try AnimationClip(tracks: [
            .number(.node(node, .morphWeight(0)), times: [0, 2], values: [0.2, 1]),
            .number(.node(node, .morphWeight(2)), times: [1, 3], values: [1, 0]),
        ]).makingAdditive(referenceTime: 0.5)
        let merged = try singles.mergingMorphWeightTracks(restWeights: [node: rest])
        #expect(merged.blendMode == .additive)
        #expect(merged.properties == [.node(node, .morphWeights)])
        for time in [0.0, 0.5, 1.2, 2.5, 3] {
            func weights(_ clip: AnimationClip) throws -> [Float] {
                var mixer = AnimationMixer()
                let action = try mixer.addAction(for: clip, weight: 0.7)
                try mixer.play(action)
                try mixer.setTime(time, for: action)
                return try mixer.morphWeights(for: node, rest: rest)
            }
            let expected = try weights(singles)
            let actual = try weights(merged)
            #expect(zip(expected, actual).allSatisfy { abs($0 - $1) < 1e-6 }, "\(expected) vs \(actual) at \(time)")
        }

        let joint = UUID()
        let skeleton = try AnimationSkeleton(joints: [.init(id: joint, name: "root", parent: nil, rest: .init())])
        let retargeted = try clip.retargeted(from: skeleton, to: skeleton)
        let neutralPose = try #require(try retargeted.sample(at: 1)[joint])
        #expect(retargeted.blendMode == .additive)
        #expect(neutralPose.translation == .zero)
        #expect(neutralPose.rotation?.vector == SIMD4(0, 0, 0, 1))
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Edited glTF clips keep weights channels on the morph path of skinned meshes`() throws {
        // A skinned quad with one morph target: the weights channel targets node 0, whose draws are separate roots.
        var fixture = GLTFMotionFixture()
        let h: Float = 0.25
        let corners: [SIMD3<Float>] = [SIMD3(-h, -h, 0), SIMD3(h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, h, 0)]
        let positions = fixture.floats(corners.flatMap { [$0.x, $0.y, $0.z] }, type: "VEC3", count: 6)
        let normals = fixture.floats(Array(repeating: [Float(0), 0, 1], count: 6).flatMap { $0 }, type: "VEC3", count: 6)
        let slide = fixture.floats(Array(repeating: [Float(1), 0, 0], count: 6).flatMap { $0 }, type: "VEC3", count: 6)
        let joints = fixture.bytes(Array(repeating: UInt8(0), count: 24), type: "VEC4", count: 6, component: 5121)
        let weights = fixture.floats(Array(repeating: [Float(1), 0, 0, 0], count: 6).flatMap { $0 }, type: "VEC4", count: 6)
        let times = fixture.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let values = fixture.floats([0, 1], type: "SCALAR", count: 2)
        let moves = fixture.floats([0, 0, -2, 1, 0, -2], type: "VEC3", count: 2)
        let data = try fixture.document([
            "scenes": [["nodes": [0, 1]]],
            "nodes": [["mesh": 0, "skin": 0], ["translation": [0, 0, -2]]],
            "skins": [["joints": [1]]],
            "meshes": [[
                "primitives": [[
                    "attributes": ["POSITION": positions, "NORMAL": normals, "JOINTS_0": joints, "WEIGHTS_0": weights],
                    "targets": [["POSITION": slide]],
                ]],
                "weights": [0],
            ]],
            "animations": [[
                "channels": [
                    ["target": ["node": 0, "path": "weights"], "sampler": 0],
                    ["target": ["node": 1, "path": "translation"], "sampler": 1],
                ],
                "samplers": [["input": times, "output": values], ["input": times, "output": moves]],
            ]],
        ])
        let device = try #require(MTLCreateSystemDefaultDevice())
        let scene = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true, includeSkins: true, includeMorphTargets: true)
        let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .rgba16Float, scene: scene)
        let graph = try prepared.graph(camera: PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let source = try AnimationClip(scene.animations[0], nodeIDs: prepared.nodeIDs)
        let mesh = try #require(prepared.nodeIDs[0])
        let joint = try #require(prepared.nodeIDs[1])
        let rest = try #require(prepared.restMorphWeights[mesh])

        func play(_ clip: AnimationClip, at time: Double) throws -> AnimationMixer {
            var mixer = AnimationMixer()
            let action = try mixer.addAction(for: clip)
            try mixer.play(action)
            try mixer.setTime(time, for: action)
            return mixer
        }
        let edits = [
            ("subclip", try source.subclip(startFrame: 15, endFrame: 60), 0.5),
            ("merged", try source.mergingMorphWeightTracks(restWeights: [:]), 0.0),
            ("replicated", try source.replicated(onto: [[mesh: mesh, joint: joint]]), 0.0),
            ("additive subclip", try source.makingAdditive().subclip(startFrame: 15, endFrame: 60), 0.5),
        ]
        for (name, clip, offset) in edits {
            let additive = clip.blendMode == .additive
            for time in [0.0, 0.4, 1.5] where time + offset <= 2 {
                let original = try play(additive ? source.makingAdditive() : source, at: time + offset)
                let edited = try play(clip, at: time)
                // The graph pass leaves morph weights alone, as for the source clip, and moves the joint the same way.
                let posed = try edited.apply(to: graph)
                let expectedGraph = try original.apply(to: graph)
                let a = try posed.resolve().first { $0.id == joint }?.worldTransform
                let b = try expectedGraph.resolve().first { $0.id == joint }?.worldTransform
                #expect(a != nil && b != nil && simd_distance(a!.columns.3, b!.columns.3) < 1e-5, "\(name) at \(time)")
                #expect(try posed.resolve().map(\.morphWeights) == graph.resolve().map(\.morphWeights), "\(name) at \(time)")
                // The morph path gives the same weights, which the prepared scene writes to the skinned draws.
                let expected = try original.morphWeights(for: mesh, rest: rest)
                let actual = try edited.morphWeights(for: mesh, rest: rest)
                #expect(abs(expected[0] - actual[0]) < 1e-6, "\(name) at \(time): \(expected) vs \(actual)")
                _ = try prepared.applyingMorphWeights([mesh: actual], to: posed)
            }
        }
        // A clip rebuilt from the typed tracks writes graph weights, which the skinned source node cannot hold.
        let typed = try AnimationClip(tracks: source.typedTracks())
        #expect(throws: AnimationMixer.Failure.self) { try play(typed, at: 0.5).apply(to: graph) }
    }

    @Test
    func `Three.js binding paths parse to typed properties`() throws {
        let face = UUID()
        let nodes = ["Face": face]
        #expect(try AnimationProperty(path: "Face.quaternion", nodes: nodes) == .node(face, .rotation))
        #expect(try AnimationProperty(path: ".position", nodes: [:], defaultNode: face) == .node(face, .translation))
        #expect(try AnimationProperty(path: "Face.morphTargetInfluences[2]", nodes: nodes) == .node(face, .morphWeight(2)))
        #expect(try AnimationProperty(
            path: "Face.morphTargetInfluences[smile]", nodes: nodes, morphTargetNames: ["Face": ["blink", "smile"]]
        ) == .node(face, .morphWeight(1)))
        #expect(try AnimationProperty(path: "Face.intensity", nodes: nodes) == .node(face, .lightIntensity))
        #expect(throws: AnimationProperty.PathError.unknownMorphTarget("frown")) {
            try AnimationProperty(path: "Face.morphTargetInfluences[frown]", nodes: nodes)
        }
        #expect(throws: AnimationProperty.PathError.unknownNode("Arm")) { try AnimationProperty(path: "Arm.scale", nodes: nodes) }
        #expect(try AnimationProperty(path: "Face.visible", nodes: nodes) == .node(face, .isVisible))
        #expect(throws: AnimationProperty.PathError.unknownProperty("unknown")) {
            try AnimationProperty(path: "Face.unknown", nodes: nodes)
        }

        // Keys and node names with dots, as Blender exports them.
        let names = ["Face": ["smile", "smile.L", "smile.001"], "Armature.001": ["blink"]]
        let rig = UUID()
        let dotted = ["Face": face, "Armature.001": rig]
        #expect(try AnimationProperty(path: "Face.morphTargetInfluences[smile.L]", nodes: dotted, morphTargetNames: names)
            == .node(face, .morphWeight(1)))
        #expect(try AnimationProperty(path: "Face.morphTargetInfluences[smile.001]", nodes: dotted, morphTargetNames: names)
            == .node(face, .morphWeight(2)))
        #expect(try AnimationProperty(path: "Armature.001.position", nodes: dotted) == .node(rig, .translation))
        #expect(try AnimationProperty(path: "Armature.001.morphTargetInfluences[blink]", nodes: dotted, morphTargetNames: names)
            == .node(rig, .morphWeight(0)))
        // The default node finds names under "" or under its own name.
        #expect(try AnimationProperty(path: ".morphTargetInfluences[smile.L]", nodes: dotted, defaultNode: face, morphTargetNames: names)
            == .node(face, .morphWeight(1)))
        #expect(try AnimationProperty(path: ".morphTargetInfluences[jaw]", nodes: [:], defaultNode: face, morphTargetNames: ["": ["jaw"]])
            == .node(face, .morphWeight(0)))
        #expect(throws: AnimationProperty.PathError.malformed("Face.morphTargetInfluences[]")) {
            try AnimationProperty(path: "Face.morphTargetInfluences[]", nodes: nodes)
        }
        #expect(throws: AnimationProperty.PathError.malformed("Face.morphTargetInfluences[smile")) {
            try AnimationProperty(path: "Face.morphTargetInfluences[smile", nodes: nodes)
        }
    }

    @Test
    func `Mixer morph weights include typed weight tracks`() throws {
        let node = UUID()
        let clip = try AnimationClip(tracks: [
            AnimationTrack(property: .node(node, .morphWeights), times: [0, 1], values: [.numbers([0, 0]), .numbers([1, 1])]),
            .number(.node(node, .morphWeight(1)), times: [0], values: [0.9]),
        ])
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: clip))
        try mixer.update(deltaTime: 0.5)
        let weights = try mixer.morphWeights(for: node, rest: [0, 0])
        #expect(abs(weights[0] - 0.5) < 1e-6)
        #expect(abs(weights[1] - 0.9) < 1e-6)
    }
}
