import Foundation
import Frostfield
import Testing
import simd

struct AnimationTrackTests {
    private let node = UUID()

    @Test("Track optimization preserves flat-run boundaries, splines and additive clip references")
    func redundantKeys() throws {
        let times: [Float] = [0, 1, 2, 3, 4, 5]
        let ramp = try AnimationTrack.number(.custom("ramp"), times: times, values: [1, 1, 1, 2, 2, 2])
        let toggle = try AnimationTrack.boolean(.custom("toggle"), times: times, values: [false, false, false, true, true, true])
        let smooth = try AnimationTrack.number(.custom("smooth"), times: times, values: [1, 1, 1, 2, 2, 2], interpolation: .smooth)
        let cubic = try AnimationTrack.number(.custom("cubic"), times: times,
                                              values: Array(repeating: [Float(1), 2, 3], count: times.count).flatMap { $0 },
                                              interpolation: .cubicSpline)
        let source = try AnimationClip(name: "Held poses", tracks: [ramp, toggle, smooth, cubic], duration: 7)
            .makingAdditive(referenceTime: 3.5)
        let reduced = source.optimized()
        #expect(reduced.tracks.map(\.times) == [[0, 2, 3, 5], [0, 2, 3, 5], times, times])
        #expect(reduced.tracks[2].values == smooth.values && reduced.tracks[3].values == cubic.values)
        #expect(reduced.name == source.name && reduced.duration == 7 && reduced.blendMode == .additive)
        #expect(reduced.properties == source.properties && source.tracks[0].times == times)
        for frame in 0...70 {
            let time = Float(frame) / 10
            #expect(try reduced.sampleProperties(at: time) == source.sampleProperties(at: time))
        }
    }

    @Test("Wrapped transform clips keep the original pose through flat-run compression")
    func wrappedTransformOptimization() throws {
        let times: [Float] = [0, 1, 2, 3, 4, 5]
        let keys = zip(times, [Float(0), 0, 0, 1, 1, 1]).map { time, x in
            TransformKeyframe(time: time, transform: .init(translation: SIMD3(x, 0, 0)))
        }
        let source = try AnimationClip(SceneAnimation(tracks: [SceneAnimationTrack(nodeID: node, keyframes: keys)]))
        let reduced = source.optimized()
        #expect(reduced.nodeIDs == [node])
        #expect(try reduced.typedTracks().map(\.times) == Array(repeating: [0, 2, 3, 5], count: 3))
        for frame in 0...50 {
            let time = Float(frame) / 10
            #expect(try reduced.sample(at: time) == source.sample(at: time))
        }
    }

    @Test
    func `Number tracks step and interpolate linearly`() throws {
        let linear = try AnimationTrack.number(.custom("x"), times: [1, 3, 4], values: [2, 6, -2])
        #expect(try linear.value(at: 0) == .number(2))
        #expect(try linear.value(at: 2) == .number(4))
        #expect(try linear.value(at: 3.5) == .number(2))
        #expect(try linear.value(at: 9) == .number(-2))
        let step = try AnimationTrack.number(.custom("x"), times: [1, 3, 4], values: [2, 6, -2], interpolation: .step)
        #expect(try step.value(at: 2.999) == .number(2))
        #expect(try step.value(at: 3) == .number(6))
        #expect(throws: AnimationTrack.Failure.invalidSampleTime) { try linear.value(at: .nan) }
    }

    @Test
    func `Cubic spline tracks reproduce a cubic polynomial from its derivatives`() throws {
        // f(t) = t³ − 2t + 1 per component, with exact tangents f'(t) = 3t² − 2.
        func f(_ t: Float) -> Float { t * t * t - 2 * t + 1 }
        func slope(_ t: Float) -> Float { 3 * t * t - 2 }
        let times: [Float] = [0, 1, 3]
        let values = times.flatMap { t -> [AnimationValue] in
            let tangent = AnimationValue.vector3(SIMD3(slope(t), 2 * slope(t), -slope(t)))
            return [tangent, .vector3(SIMD3(f(t), 2 * f(t), -f(t))), tangent]
        }
        let track = try AnimationTrack(property: .custom("v"), times: times, values: values, interpolation: .cubicSpline)
        for t: Float in [0.25, 0.5, 1.7, 2.2, 2.9] {
            guard case .vector3(let value) = try track.value(at: t) else { Issue.record("kind"); return }
            #expect(abs(value.x - f(t)) < 1e-5)
            #expect(abs(value.y - 2 * f(t)) < 1e-5)
            #expect(abs(value.z + f(t)) < 1e-5)
        }
    }

    @Test
    func `Smooth tracks match Three.js CubicInterpolant`() throws {
        // Catmull-Rom with uniform spacing reproduces a quadratic exactly between interior keys.
        let times: [Float] = [0, 1, 2, 3, 4]
        let track = try AnimationTrack.number(.custom("q"), times: times, values: times.map { $0 * $0 }, interpolation: .smooth)
        for t: Float in [1.25, 1.5, 2.1, 2.75] {
            #expect(abs(try #require(track.value(at: t).floatValueForTest) - t * t) < 1e-5)
        }
        // The first segment uses Three.js's zero-curvature ending: wP = −½ and the previous key is key 1.
        let p: Float = 0.4
        let wP: Float = -0.5, wN: Float = 0.5
        let sP = -wP * p * p * p + 2 * wP * p * p - wP * p
        let s0 = (1 + wP) * p * p * p + (-1.5 - 2 * wP) * p * p + (-0.5 + wP) * p + 1
        let s1 = (-1 - wN) * p * p * p + (1.5 + wN) * p * p + 0.5 * p
        let sN = wN * p * p * p - wN * p * p
        let expected = sP * 1 + s0 * 0 + s1 * 1 + sN * 4
        #expect(abs(try #require(track.value(at: p).floatValueForTest) - expected) < 1e-6)
        // Two keys reduce to a straight line, as in Three.js.
        let pair = try AnimationTrack.number(.custom("q"), times: [0, 2], values: [1, 5], interpolation: .smooth)
        #expect(abs(try #require(pair.value(at: 0.5).floatValueForTest) - 2) < 1e-6)
    }

    @Test
    func `Quaternion tracks slerp along the shortest arc`() throws {
        let z = SIMD3<Float>(0, 0, 1)
        let end = simd_quatf(angle: 2 * .pi / 3, axis: z)
        // The negated end key is the same rotation; interpolation must not take the long way.
        let track = try AnimationTrack.quaternion(
            .node(node, .rotation), times: [0, 2], values: [simd_quatf(angle: 0, axis: z), simd_quatf(vector: -end.vector)]
        )
        let rotation = try #require(track.value(at: 0.5).quaternionValueForTest)
        let turned = rotation.act(SIMD3(1, 0, 0))
        #expect(abs(turned.x - cos(.pi / 6)) < 1e-5)
        #expect(abs(turned.y - sin(.pi / 6)) < 1e-5)
        #expect(throws: AnimationTrack.Failure.unsupportedInterpolation(AnimationProperty.node(node, .rotation).description, .smooth)) {
            try AnimationTrack.quaternion(.node(node, .rotation), times: [0], values: [end], interpolation: .smooth)
        }
        #expect(throws: AnimationTrack.Failure.nonunitQuaternion(AnimationProperty.node(node, .rotation).description, key: 0)) {
            try AnimationTrack.quaternion(.node(node, .rotation), times: [0], values: [simd_quatf(vector: SIMD4(0, 0, 0, 2))])
        }
    }

    @Test
    func `Color, boolean and string tracks interpolate or step by kind`() throws {
        let color = try AnimationTrack.color(.node(node, .lightColor), times: [0, 4], values: [SIMD3(1, 0, 0), SIMD3(0, 0.5, 1)])
        #expect(try color.value(at: 1) == .color(SIMD3(0.75, 0.125, 0.25)))
        let flag = try AnimationTrack.boolean(.node(node, .castsShadow), times: [0, 1, 2], values: [true, false, true])
        #expect(try flag.value(at: 0.99) == .boolean(true))
        #expect(try flag.value(at: 1.5) == .boolean(false))
        let label = try AnimationTrack.string(.custom("label"), times: [0, 1], values: ["idle", "run"])
        #expect(try label.value(at: 0.5) == .string("idle"))
        #expect(try label.value(at: 1) == .string("run"))
        #expect(throws: AnimationTrack.Failure.unsupportedInterpolation("custom label", .linear)) {
            try AnimationTrack(property: .custom("label"), times: [0], values: [.string("a")], interpolation: .linear)
        }
        #expect(throws: AnimationTrack.Failure.incompatibleKind(AnimationProperty.node(node, .castsShadow).description, .number)) {
            try AnimationTrack.number(.node(node, .castsShadow), times: [0], values: [1])
        }
        #expect(throws: AnimationTrack.Failure.invalidKeyTime("custom x", key: 1)) {
            try AnimationTrack.number(.custom("x"), times: [1, 1], values: [0, 0])
        }
    }

    @Test
    func `Subclips reproduce every interpolation exactly between their boundaries`() throws {
        let z = SIMD3<Float>(0, 0, 1)
        let tracks = try [
            AnimationTrack.number(.custom("linear"), times: [0, 1, 2.5, 4], values: [0, 3, -1, 2]),
            // Keys between samples: a step is discontinuous, so sampling at its key compares rounding.
            AnimationTrack.number(.custom("step"), times: [0, 1.23, 3], values: [5, 6, 7], interpolation: .step),
            AnimationTrack.number(.custom("smooth"), times: [0, 0.7, 1.9, 3.1, 4], values: [0, 2, -1, 3, 1], interpolation: .smooth),
            AnimationTrack(
                property: .custom("cubic"), times: [0, 2, 4],
                values: [.number(0), .number(1), .number(3), .number(-2), .number(0), .number(4), .number(1), .number(2), .number(0)],
                interpolation: .cubicSpline
            ),
            AnimationTrack.quaternion(
                .node(node, .rotation), times: [0, 2, 4],
                values: [simd_quatf(angle: 0, axis: z), simd_quatf(angle: 1.2, axis: z), simd_quatf(angle: -0.4, axis: SIMD3(1, 0, 0))]
            ),
        ]
        let clip = try AnimationClip(name: "source", tracks: tracks)
        // Frames 15 to 105 at 30 fps: 0.5 s to 3.5 s.
        let sub = try clip.subclip(name: "middle", startFrame: 15, endFrame: 105)
        #expect(sub.name == "middle")
        #expect(abs(sub.duration - 3) < 1e-6)
        for step in 0...60 {
            let local = Float(step) * 0.05
            let original = try clip.sampleProperties(at: local + 0.5)
            let trimmed = try sub.sampleProperties(at: local)
            for property in clip.properties {
                let a = try #require(original[property])
                let b = try #require(trimmed[property])
                switch (a, b) {
                case (.number(let x), .number(let y)): #expect(abs(x - y) < 2e-5, "\(property) at \(local): \(x) vs \(y)")
                case (.quaternion(let x), .quaternion(let y)):
                    #expect(abs(abs(simd_dot(x.vector, y.vector)) - 1) < 1e-5, "\(property) at \(local)")
                default: Issue.record("Unexpected kinds for \(property)")
                }
            }
        }
        #expect(throws: AnimationClip.Failure.invalidFrameRange) { try clip.subclip(startFrame: 10, endFrame: 10) }
        #expect(throws: AnimationClip.Failure.invalidFrameRange) { try clip.subclip(startFrame: 200, endFrame: 300) }
    }

    @Test
    func `Subclips hold curves exactly where the range extends past a track's keys`() throws {
        func cubic(_ name: String, _ times: [Float], _ triples: [Float]) throws -> AnimationTrack {
            try AnimationTrack(property: .custom(name), times: times, values: triples.map { .number($0) }, interpolation: .cubicSpline)
        }
        let clip = try AnimationClip(tracks: [
            // Starts at 1 s with out-tangent 5; holds 0 before.
            cubic("late start", [1, 2], [0, 0, 5, 0, 1, 0]),
            // Ends at 1 s with out-tangent 5; holds 1 after.
            cubic("early end", [0, 1], [0, 0, 0, 0, 1, 5]),
            // Keys only between 3 s and 3.5 s, with tangents facing outward on both ends.
            cubic("short", [3, 3.5], [4, 2, 1, 1, 3, -6]),
            AnimationTrack.number(.custom("smooth"), times: [1, 2, 3], values: [0, 1, 0], interpolation: .smooth),
            AnimationTrack.number(.custom("clock"), times: [0, 4], values: [0, 4]),
        ])
        #expect(clip.duration == 4)
        for (startFrame, endFrame) in [(Float(0), Float(120)), (7.5, 112.5), (0, 60), (75, 120), (105, 120)] {
            let sub = try clip.subclip(startFrame: startFrame, endFrame: endFrame)
            let offset = startFrame / 30
            for step in 0...80 {
                let local = sub.duration * Float(step) / 80
                let original = try clip.sampleProperties(at: local + offset)
                let trimmed = try sub.sampleProperties(at: local)
                for property in clip.properties {
                    let a = try #require(original[property]?.floatValueForTest)
                    let b = try #require(trimmed[property]?.floatValueForTest)
                    #expect(abs(a - b) < 1e-5, "\(property) in frames \(startFrame)–\(endFrame) at \(local): \(a) vs \(b)")
                }
            }
        }
    }

    @Test
    func `Morph target sequences peak one target per frame like Three.js`() throws {
        let clip = try AnimationClip.morphTargetSequence(node: node, targetCount: 4, framesPerSecond: 10)
        #expect(abs(clip.duration - 0.4) < 1e-6)
        func weights(at time: Float) throws -> [Float] {
            let values = try clip.sampleProperties(at: time)
            return try (0..<4).map { try #require(values[.node(node, .morphWeight($0))]?.floatValueForTest) }
        }
        #expect(try weights(at: 0.1) == [0, 1, 0, 0])
        let between = try weights(at: 0.15)
        #expect(zip(between, [0, 0.5, 0.5, 0]).allSatisfy { abs($0 - $1) < 1e-5 })
        // Looping returns to the first target at the end.
        let wrapped = try weights(at: 0.35)
        #expect(zip(wrapped, [0.5, 0, 0, 0.5]).allSatisfy { abs($0 - $1) < 1e-5 })
        let once = try AnimationClip.morphTargetSequence(node: node, targetCount: 4, framesPerSecond: 10, looping: false)
        #expect(abs(once.duration - 0.3) < 1e-6)
    }

    @Test
    func `Single morph weight tracks merge into one exact weights track`() throws {
        let clip = try AnimationClip(tracks: [
            AnimationTrack.number(.node(node, .morphWeight(0)), times: [0, 2], values: [0, 1]),
            AnimationTrack.number(.node(node, .morphWeight(2)), times: [1, 3], values: [1, 0]),
        ])
        let merged = try clip.mergingMorphWeightTracks(restWeights: [node: [0.25, 0.5, 0.75]])
        #expect(merged.properties == [.node(node, .morphWeights)])
        for time: Float in [0, 0.5, 1, 1.7, 2.5, 3] {
            let single = try clip.sampleProperties(at: time)
            guard case .numbers(let weights) = try #require(merged.sampleProperties(at: time)[.node(node, .morphWeights)]) else {
                Issue.record("kind"); return
            }
            #expect(abs(weights[0] - (single[.node(node, .morphWeight(0))]?.floatValueForTest ?? .nan)) < 1e-6)
            #expect(weights[1] == 0.5)
            #expect(abs(weights[2] - (single[.node(node, .morphWeight(2))]?.floatValueForTest ?? .nan)) < 1e-6)
        }
    }
}

extension AnimationValue {
    var floatValueForTest: Float? { if case .number(let value) = self { value } else { nil } }
    var quaternionValueForTest: simd_quatf? { if case .quaternion(let value) = self { value } else { nil } }
    var vector3ValueForTest: SIMD3<Float>? {
        switch self {
        case .vector3(let value), .color(let value): value
        default: nil
        }
    }
}
