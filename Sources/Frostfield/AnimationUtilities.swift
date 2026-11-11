import Foundation
import simd

// Clip editing and procedural clips, following Three.js `AnimationUtils`,
// `AnimationClip.CreateFromMorphTargetSequence` and `SkeletonUtils.retargetClip`.

extension AnimationClip {
    /// Removes redundant interior keys from step and linear tracks. The clip
    /// keeps its name, duration, blend mode, additive references and bindings.
    /// Imported channels remain imported channels, including morph weights.
    public func optimized() -> AnimationClip {
        let targets = targets.map { target in
            AnimationNodeTracks(nodeID: target.nodeID, transform: target.transform?.optimized(),
                                channels: target.channels.map { $0.optimized() })
        }
        return AnimationClip(name: name, duration: duration, blendMode: blendMode, targets: targets,
                             reference: reference, tracks: tracks.map { $0.optimized() }, trackReference: trackReference)
    }

    /// Every track of the clip as a typed track: node transform and glTF channels
    /// become translation, rotation, scale and morph weight tracks with the same
    /// keys and interpolation. Typed tracks are returned unchanged. An additive
    /// clip's tracks hold absolute values, not offsets.
    ///
    /// A glTF weights channel becomes a `.morphWeights` track. In a clip built with
    /// `init(name:tracks:duration:)` such a track writes the node's own weights in
    /// `AnimationMixer.apply(to:)`, which for a skinned or multi-primitive glTF mesh
    /// is not the node that draws. `subclip`, `mergingMorphWeightTracks` and
    /// `replicated(onto:)` keep weights channels as channels, which only
    /// `AnimationMixer.morphWeights(for:rest:)` reads, for `PreparedGLTFScene.applyingMorphWeights`.
    public func typedTracks() throws -> [AnimationTrack] {
        try editable().tracks
    }

    /// The clip's tracks for editing, with what rebuilding it needs.
    func editable() throws -> EditableClip {
        var result = EditableClip()
        for target in targets {
            let id = target.nodeID
            let base = blendMode == .additive ? reference[id] : nil
            func add(_ track: AnimationTrack, reference: AnimationValue?) {
                result.tracks.append(track)
                if let reference { result.references[track.property] = reference }
            }
            if let track = target.transform {
                let times = track.keyframes.map(\.time)
                add(
                    try .vector(.node(id, .translation), times: times, values: track.keyframes.map(\.transform.translation)),
                    reference: base?.translation.map { .vector3($0) }
                )
                add(
                    try .quaternion(.node(id, .rotation), times: times, values: track.keyframes.map(\.transform.rotation)),
                    reference: base?.rotation.map { .quaternion($0) }
                )
                add(
                    try .vector(.node(id, .scale), times: times, values: track.keyframes.map(\.transform.scale)),
                    reference: base?.scale.map { .vector3($0) }
                )
            }
            for channel in target.channels {
                let interpolation: AnimationTrack.Interpolation = switch channel.interpolation {
                case .linear: .linear
                case .step: .step
                case .cubicSpline: .cubicSpline
                }
                let values: [AnimationValue]
                let property: AnimationProperty
                let origin: AnimationValue?
                switch channel.target {
                case .translation:
                    property = .node(id, .translation)
                    values = channel.values.map { .vector3(SIMD3($0.x, $0.y, $0.z)) }
                    origin = base?.translation.map { .vector3($0) }
                case .scale:
                    property = .node(id, .scale)
                    values = channel.values.map { .vector3(SIMD3($0.x, $0.y, $0.z)) }
                    origin = base?.scale.map { .vector3($0) }
                case .rotation:
                    property = .node(id, .rotation)
                    values = channel.values.map { .quaternion(simd_quatf(vector: $0)) }
                    origin = base?.rotation.map { .quaternion($0) }
                case .weights:
                    property = .node(id, .morphWeights)
                    let count = channel.weightCount
                    values = stride(from: 0, to: channel.weightValues.count, by: count).map {
                        .numbers(Array(channel.weightValues[$0..<($0 + count)]))
                    }
                    origin = base?.morphWeights.map { .numbers($0) }
                    result.weightChannels[property] = channel.sourceNodeIndex
                }
                add(
                    try AnimationTrack(property: property, times: channel.times, values: values, interpolation: interpolation, unitKeys: false),
                    reference: origin
                )
            }
        }
        for track in tracks {
            result.tracks.append(track)
            if let reference = trackReference[track.property] { result.references[track.property] = reference }
        }
        return result
    }

    /// Rebuilds an edited clip. Tracks marked as glTF weights channels become weights
    /// channels again; an additive clip keeps the reference value of every track that has one.
    init(edited: EditableClip, name: String?, duration: Float, blendMode: AnimationBlendMode) throws {
        guard !edited.tracks.isEmpty else { throw Failure.emptyClip }
        var targets: [AnimationNodeTracks] = []
        var reference: [UUID: AnimationNodePose] = [:]
        var tracks: [AnimationTrack] = []
        var trackReference: [AnimationProperty: AnimationValue] = [:]
        var seen = Set<AnimationProperty>()
        for var track in edited.tracks {
            guard seen.insert(track.property).inserted else { throw Failure.duplicateTrack(track.property.description) }
            let origin = blendMode == .additive ? edited.references[track.property] : nil
            guard let source = edited.weightChannels[track.property], case .node(let id, .morphWeights) = track.property,
                  case .numbers(let count) = track.valueKind else {
                tracks.append(track)
                if let origin { trackReference[track.property] = origin }
                continue
            }
            if track.interpolation == .smooth { track = try track.cubicSplineEquivalent() }
            let interpolation: GLTFAnimationClip.Channel.Interpolation = switch track.interpolation {
            case .step: .step
            case .linear: .linear
            case .smooth, .cubicSpline: .cubicSpline
            }
            let weights = track.values.flatMap { value -> [Float] in
                if case .numbers(let values) = value { values } else { [] }
            }
            targets.append(AnimationNodeTracks(nodeID: id, channels: [GLTFAnimationClip.Channel(
                sourceNodeIndex: source, interpolation: interpolation, times: track.times, weights: weights, weightCount: count
            )]))
            if case .numbers(let values)? = origin { reference[id] = AnimationNodePose(morphWeights: values) }
        }
        self.init(
            name: name, duration: duration, blendMode: blendMode, targets: targets, reference: reference,
            tracks: tracks, trackReference: trackReference
        )
    }

    /// The part of the clip between two frames, starting at time zero, like Three.js
    /// `AnimationUtils.subclip(clip, name, startFrame, endFrame, fps)`.
    ///
    /// Unlike Three.js, which keeps only the keys inside the range, every track gets
    /// keys at both boundaries, so the subclip reproduces the source exactly over
    /// the range: step and linear values are sampled, a cubic spline segment cut by
    /// a boundary is split with the curve's exact value and derivative, a boundary
    /// past a track's keys holds its end value, and a `smooth` track becomes its
    /// exact cubic spline. Frames may be fractional; the end clamps to the clip's
    /// duration. Node transform channels become typed tracks; glTF weights channels
    /// stay channels. An additive clip stays additive with the same reference values.
    /// Throws `Failure.invalidFrameRange` unless `0 ≤ start < end` and the frame rate is positive.
    public func subclip(
        name: String? = nil, startFrame: Float, endFrame: Float, framesPerSecond: Float = 30
    ) throws -> AnimationClip {
        guard framesPerSecond.isFinite, framesPerSecond > 0, startFrame.isFinite, endFrame.isFinite,
              startFrame >= 0, startFrame < endFrame else {
            throw Failure.invalidFrameRange
        }
        let start = Double(startFrame) / Double(framesPerSecond)
        let end = min(Double(endFrame) / Double(framesPerSecond), Double(duration))
        guard start < end else { throw Failure.invalidFrameRange }
        var edited = try editable()
        edited.tracks = try edited.tracks.map { try $0.trimmed(from: start, to: end) }
        return try AnimationClip(edited: edited, name: name ?? self.name, duration: Float(end - start), blendMode: blendMode)
    }

    /// Merges each node's single morph weight tracks (`.morphWeight(i)`) into one
    /// `.morphWeights` track over the union of their key times. Weights without a
    /// track hold the node's rest weight. Exact for step and linear tracks; the
    /// tracks merged for one node must share their interpolation, and other
    /// interpolations throw `AnimationTrack.Failure.unsupportedInterpolation`.
    /// An additive clip stays additive: the merged reference is the rest weights
    /// with each merged track's reference value in its slot, so every offset is unchanged.
    public func mergingMorphWeightTracks(restWeights: [UUID: [Float]]) throws -> AnimationClip {
        var edited = try editable()
        var singles: [UUID: [(Int, AnimationTrack)]] = [:]
        var kept: [AnimationTrack] = []
        for track in edited.tracks {
            if case .node(let id, .morphWeight(let index)) = track.property {
                singles[id, default: []].append((index, track))
            } else {
                kept.append(track)
            }
        }
        for id in singles.keys.sorted(by: { $0.uuidString < $1.uuidString }) {
            let entries = singles[id]!
            let property = AnimationProperty.node(id, .morphWeights)
            guard let rest = restWeights[id], !rest.isEmpty else { throw Failure.unmappedTrack(property.description) }
            guard !kept.contains(where: { $0.property == property }) else { throw Failure.duplicateTrack(property.description) }
            let interpolation = entries[0].1.interpolation
            guard interpolation == .linear || interpolation == .step, entries.allSatisfy({ $0.1.interpolation == interpolation }) else {
                throw AnimationTrack.Failure.unsupportedInterpolation(property.description, entries[0].1.interpolation)
            }
            guard entries.allSatisfy({ rest.indices.contains($0.0) }) else {
                throw AnimationMixer.Failure.unanimatableProperty(property.description, "the rest weights have \(rest.count) targets")
            }
            let times = Array(Set(entries.flatMap(\.1.times))).sorted()
            let values = try times.map { time -> AnimationValue in
                var weights = rest
                for (index, track) in entries {
                    guard let value = try track.value(at: time).floatValue else { throw AnimationTrack.Failure.mixedKinds(property.description) }
                    weights[index] = value
                }
                return .numbers(weights)
            }
            kept.append(try AnimationTrack(property: property, times: times, values: values, interpolation: interpolation))
            if blendMode == .additive {
                var origin = rest
                for (index, track) in entries {
                    if let value = edited.references.removeValue(forKey: track.property)?.floatValue { origin[index] = value }
                }
                edited.references[property] = .numbers(origin)
            }
        }
        edited.tracks = kept
        return try AnimationClip(edited: edited, name: name, duration: duration, blendMode: blendMode)
    }

    /// One clip that drives several objects with one timing and weight, like playing
    /// a clip on a Three.js `AnimationObjectGroup`. Each map sends the clip's node IDs
    /// to one object's nodes; every animated node must be in every map. Tracks for
    /// other properties, such as shader uniforms, are kept once. glTF weights
    /// channels stay channels. Make the result additive afterwards if needed; an
    /// additive clip throws `Failure.alreadyAdditive`.
    public func replicated(onto nodeMaps: [[UUID: UUID]]) throws -> AnimationClip {
        guard blendMode == .normal else { throw Failure.alreadyAdditive }
        guard !nodeMaps.isEmpty else { throw Failure.emptyClip }
        let source = try editable()
        var edited = EditableClip()
        for track in source.tracks {
            guard case .node(let id, let property) = track.property else {
                edited.tracks.append(track)
                continue
            }
            for map in nodeMaps {
                guard let target = map[id] else { throw Failure.unmappedTrack(track.property.description) }
                let copy = try track.retargeted(to: .node(target, property))
                edited.tracks.append(copy)
                if let channel = source.weightChannels[track.property] { edited.weightChannels[copy.property] = channel }
            }
        }
        return try AnimationClip(edited: edited, name: name, duration: duration, blendMode: .normal)
    }

    /// A clip that cycles through a node's morph targets, like Three.js
    /// `AnimationClip.CreateFromMorphTargetSequence(name, sequence, fps, noLoop)`.
    /// Target `i` has weight one at frame `i` and falls linearly to zero one frame
    /// before and after. When looping, the clip lasts `targetCount` frames and the
    /// first target returns at the end; otherwise it lasts `targetCount - 1` frames.
    public static func morphTargetSequence(
        name: String? = nil, node: UUID, targetCount: Int, framesPerSecond: Float, looping: Bool = true
    ) throws -> AnimationClip {
        guard targetCount > 0, framesPerSecond.isFinite, framesPerSecond > 0 else { throw Failure.invalidFrameRange }
        let frames = looping ? targetCount : max(targetCount - 1, 1)
        let times = (0...frames).map { Float(Double($0) / Double(framesPerSecond)) }
        let tracks = try (0..<targetCount).map { target in
            let values = (0...frames).map { frame -> Float in
                var distance = abs(frame - target)
                if looping { distance = min(distance, abs(frame - target - targetCount), abs(frame - target + targetCount)) }
                return distance == 0 ? 1 : 0
            }
            return try AnimationTrack.number(.node(node, .morphWeight(target)), times: times, values: values)
        }
        return try AnimationClip(name: name, tracks: tracks)
    }
}

/// Keep both ends of each flat run so the ramps into and out of it retain their times.
func retainedFlatRunKeys(_ count: Int, matches: (Int, Int) -> Bool) -> [Int]? {
    guard count > 2 else { return nil }
    let last = count - 1
    var keys = [0]
    for key in 1..<last where !matches(key - 1, key) || !matches(key, key + 1) {
        keys.append(key)
    }
    keys.append(last)
    return keys.count < count ? keys : nil
}

/// A clip's tracks as typed tracks, for the editing functions of `AnimationClip`.
struct EditableClip {
    var tracks: [AnimationTrack] = []
    /// Additive reference values by property; empty for a normal clip.
    var references: [AnimationProperty: AnimationValue] = [:]
    /// The source node index of each track converted from a glTF weights channel.
    var weightChannels: [AnimationProperty: Int] = [:]
}

/// A joint hierarchy with a rest pose, for retargeting clips between skeletons.
public struct AnimationSkeleton {
    public struct Joint {
        /// The scene graph node the joint's animation tracks target.
        public let id: UUID
        public let name: String
        /// The parent joint's index, which must precede this joint.
        public let parent: Int?
        /// The local transform of the reference pose that both skeletons share, such as a T-pose.
        public let rest: SceneTransform

        public init(id: UUID, name: String, parent: Int?, rest: SceneTransform) {
            self.id = id
            self.name = name
            self.parent = parent
            self.rest = rest
        }
    }

    public let joints: [Joint]

    /// Requires unique names and IDs, parents before children, and unit rest rotations.
    public init(joints: [Joint]) throws {
        guard !joints.isEmpty else { throw Failure.empty }
        var names = Set<String>()
        var ids = Set<UUID>()
        for (index, joint) in joints.enumerated() {
            guard names.insert(joint.name).inserted, ids.insert(joint.id).inserted else { throw Failure.duplicateJoint(joint.name) }
            if let parent = joint.parent, !(0..<index).contains(parent) { throw Failure.parentOrder(joint.name) }
            guard abs(simd_length(joint.rest.rotation.vector) - 1) <= 0.0001,
                  simd_reduce_min(abs(joint.rest.scale)) > 0 else { throw Failure.invalidRest(joint.name) }
        }
        self.joints = joints
    }

    /// The joints of an imported skin, named by their glTF node names and posed by
    /// their authored local transforms. Each joint's parent must be another joint of
    /// the skin or outside it.
    public init(scene: GLTFScene, skin: GLTFScene.Skin, nodeIDs: [Int: UUID]) throws {
        let nodes = Dictionary(uniqueKeysWithValues: scene.nodes.map { ($0.sourceIndex, $0) })
        let members = Set(skin.jointNodeIndices)
        var order: [Int] = []
        var placed = Set<Int>()
        // Parents before children, keeping the skin's order otherwise.
        func place(_ index: Int, depth: Int) throws {
            guard !placed.contains(index) else { return }
            guard depth <= 512, let node = nodes[index] else { throw Failure.parentOrder("\(index)") }
            if let parent = node.parentIndex, members.contains(parent) { try place(parent, depth: depth + 1) }
            placed.insert(index)
            order.append(index)
        }
        for index in skin.jointNodeIndices { try place(index, depth: 0) }
        let positions = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($0.element, $0.offset) })
        try self.init(joints: order.map { index in
            let node = nodes[index]!
            guard let name = node.name, let id = nodeIDs[index] else { throw Failure.unnamedJoint(index) }
            return Joint(id: id, name: name, parent: node.parentIndex.flatMap { positions[$0] }, rest: node.localTransform)
        })
    }

    /// Joint world matrices in skeleton space for local transforms in joint order.
    func worldMatrices(_ locals: [SceneTransform]) -> [simd_float4x4] {
        var worlds: [simd_float4x4] = []
        worlds.reserveCapacity(joints.count)
        for (index, joint) in joints.enumerated() {
            let local = locals[index].matrix
            worlds.append(joint.parent.map { worlds[$0] * local } ?? local)
        }
        return worlds
    }

    /// Joint world rotations in skeleton space, ignoring scale.
    func worldRotations(_ locals: [SceneTransform]) -> [simd_quatd] {
        var worlds: [simd_quatd] = []
        worlds.reserveCapacity(joints.count)
        for (index, joint) in joints.enumerated() {
            let local = simd_quatd(vector: SIMD4<Double>(locals[index].rotation.vector))
            worlds.append(simd_normalize(joint.parent.map { worlds[$0] * local } ?? local))
        }
        return worlds
    }

    public enum Failure: LocalizedError, Equatable {
        case empty
        case duplicateJoint(String)
        case parentOrder(String)
        case invalidRest(String)
        case unnamedJoint(Int)
        case unmappedJoint(String)
        case noMappedJoints

        public var errorDescription: String? {
            switch self {
            case .empty: "A skeleton needs at least one joint."
            case .duplicateJoint(let name): "The skeleton has more than one joint named or identified as \(name)."
            case .parentOrder(let name): "Joint \(name) must come after its parent, which must be in the skeleton."
            case .invalidRest(let name): "Joint \(name) needs a unit rest rotation and a nonzero rest scale."
            case .unnamedJoint(let index): "Joint node \(index) needs a name and a graph ID."
            case .unmappedJoint(let name): "The bone map names joint \(name), which is not in its skeleton."
            case .noMappedJoints: "Retargeting needs at least one joint mapped between the skeletons."
            }
        }
    }
}

extension AnimationClip {
    /// Retargets a clip that animates `source` onto `target`, like Three.js
    /// `SkeletonUtils.retargetClip`. Both skeletons' rest transforms must describe
    /// the same reference pose, such as a T-pose; their joint frames, bone lengths
    /// and proportions may differ.
    ///
    /// The clip is sampled at `framesPerSecond` and at its end. At each sample every
    /// mapped target joint takes the world rotation that turns its rest orientation
    /// the way its source joint turned from rest, `R_source · R_source,rest⁻¹ ·
    /// R_target,rest`, so bones point the same way in skeleton space. Unmapped joints
    /// keep their rest rotations. The mapped joint nearest the root also moves by the
    /// source's skeleton-space displacement from rest times `translationScale`, which
    /// defaults to the ratio of the mapped target bones' rest lengths to the source's,
    /// measured between the skeleton-space rest positions of each mapped joint and
    /// its mapped parent, so joint scales count. The displacement is converted into
    /// the root joint's parent space with the parent's full rest transform.
    ///
    /// `boneNames` maps target joint names to source joint names; by default joints
    /// with equal names map. The result has a linear rotation track per mapped joint
    /// and a translation track for the root joint. Additive offsets are applied
    /// to the source rest pose and returned relative to the target rest pose.
    /// The duration and sampling rate may produce up to one million frame steps.
    public func retargeted(
        from source: AnimationSkeleton, to target: AnimationSkeleton, boneNames: [String: String]? = nil,
        framesPerSecond: Float = 30, translationScale: Float? = nil
    ) throws -> AnimationClip {
        guard framesPerSecond.isFinite, framesPerSecond > 0 else { throw Failure.invalidFrameRange }
        let sourceIndex = Dictionary(uniqueKeysWithValues: source.joints.enumerated().map { ($0.element.name, $0.offset) })
        var mapping: [Int: Int] = [:]
        if let boneNames {
            for (targetName, sourceName) in boneNames {
                guard let t = target.joints.firstIndex(where: { $0.name == targetName }) else {
                    throw AnimationSkeleton.Failure.unmappedJoint(targetName)
                }
                guard let s = sourceIndex[sourceName] else { throw AnimationSkeleton.Failure.unmappedJoint(sourceName) }
                mapping[t] = s
            }
        } else {
            for (t, joint) in target.joints.enumerated() { if let s = sourceIndex[joint.name] { mapping[t] = s } }
        }
        guard !mapping.isEmpty else { throw AnimationSkeleton.Failure.noMappedJoints }

        let sourceRest = source.joints.map(\.rest)
        let targetRest = target.joints.map(\.rest)
        let sourceRestWorld = source.worldRotations(sourceRest)
        let targetRestWorld = target.worldRotations(targetRest)
        let sourceRestMatrices = source.worldMatrices(sourceRest)
        let targetRestMatrices = target.worldMatrices(targetRest)
        func position(_ matrix: simd_float4x4) -> SIMD3<Double> {
            SIMD3(Double(matrix.columns.3.x), Double(matrix.columns.3.y), Double(matrix.columns.3.z))
        }

        // The topmost mapped target joint carries the root motion.
        func depth(_ joints: [AnimationSkeleton.Joint], _ index: Int) -> Int {
            var count = 0
            var current = joints[index].parent
            while let parent = current { count += 1; current = joints[parent].parent }
            return count
        }
        let root = mapping.keys.min { (depth(target.joints, $0), $0) < (depth(target.joints, $1), $1) }!
        let scale: Double
        if let translationScale {
            guard translationScale.isFinite else { throw Failure.invalidFrameRange }
            scale = Double(translationScale)
        } else {
            var targetLength = 0.0
            var sourceLength = 0.0
            // Each bone runs from a mapped joint to its mapped parent, on both skeletons.
            for (t, s) in mapping {
                guard let parent = target.joints[t].parent, let sourceParent = mapping[parent] else { continue }
                targetLength += simd_distance(position(targetRestMatrices[t]), position(targetRestMatrices[parent]))
                sourceLength += simd_distance(position(sourceRestMatrices[s]), position(sourceRestMatrices[sourceParent]))
            }
            scale = sourceLength > 0 && targetLength > 0 ? targetLength / sourceLength : 1
        }

        // The root's ancestors are unmapped, so they stay at rest; their full linear part
        // (rotation and scale) takes skeleton-space displacements into the root's parent space.
        let parentFromSkeleton: simd_double3x3 = target.joints[root].parent.map { parent in
            let m = targetRestMatrices[parent]
            func column(_ c: SIMD4<Float>) -> SIMD3<Double> { SIMD3(Double(c.x), Double(c.y), Double(c.z)) }
            return simd_double3x3(columns: (column(m.columns.0), column(m.columns.1), column(m.columns.2))).inverse
        } ?? matrix_identity_double3x3

        let frameSteps = (Double(duration) * Double(framesPerSecond)).rounded(.down)
        guard frameSteps <= 1_000_000 else { throw Failure.invalidFrameRange }
        let steps = Int(frameSteps)
        var times = (0...steps).map { Float(Double($0) / Double(framesPerSecond)) }.filter { $0 < duration }
        if times.isEmpty || times[times.count - 1] < duration { times.append(duration) }

        var rotations = Array(repeating: [simd_quatf](), count: target.joints.count)
        var rootTranslations: [SIMD3<Float>] = []
        let additive = blendMode == .additive
        for time in times {
            let poses = try sample(at: time)
            let locals = source.joints.map { joint -> SceneTransform in
                guard let pose = poses[joint.id] else { return joint.rest }
                if !additive { return pose.applied(to: joint.rest) }
                var local = joint.rest
                if let translation = pose.translation { local.translation += translation }
                if let rotation = pose.rotation { local.rotation = simd_normalize(local.rotation * rotation) }
                if let scale = pose.scale { local.scale += scale }
                return local
            }
            let sourceWorld = source.worldRotations(locals)
            var targetWorld: [simd_quatd] = []
            targetWorld.reserveCapacity(target.joints.count)
            for (index, joint) in target.joints.enumerated() {
                let parentWorld = joint.parent.map { targetWorld[$0] } ?? simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
                let world: simd_quatd
                if let s = mapping[index] {
                    world = simd_normalize(sourceWorld[s] * sourceRestWorld[s].inverse * targetRestWorld[index])
                    var local = simd_normalize(parentWorld.inverse * world)
                    // Keep neighboring samples in one hemisphere so linear tracks take the short way.
                    if let previous = rotations[index].last, simd_dot(SIMD4<Double>(previous.vector), local.vector) < 0 {
                        local = simd_quatd(vector: -local.vector)
                    }
                    rotations[index].append(simd_quatf(vector: SIMD4<Float>(local.vector)))
                } else {
                    world = simd_normalize(parentWorld * simd_quatd(vector: SIMD4<Double>(joint.rest.rotation.vector)))
                }
                targetWorld.append(world)
            }
            let s = mapping[root]!
            let displacement = position(source.worldMatrices(locals)[s]) - position(sourceRestMatrices[s])
            let local = parentFromSkeleton * (displacement * scale)
            rootTranslations.append(target.joints[root].rest.translation + SIMD3<Float>(local))
        }

        var tracks: [AnimationTrack] = []
        for index in target.joints.indices where mapping[index] != nil {
            tracks.append(try .quaternion(.node(target.joints[index].id, .rotation), times: times, values: rotations[index]))
        }
        tracks.append(try .vector(.node(target.joints[root].id, .translation), times: times, values: rootTranslations))
        guard additive else { return try AnimationClip(name: name, tracks: tracks, duration: duration) }
        var references: [AnimationProperty: AnimationValue] = [
            .node(target.joints[root].id, .translation): .vector3(targetRest[root].translation),
        ]
        for index in target.joints.indices where mapping[index] != nil {
            references[.node(target.joints[index].id, .rotation)] = .quaternion(targetRest[index].rotation)
        }
        return AnimationClip(
            name: name, duration: duration, blendMode: .additive, targets: [], reference: [:],
            tracks: tracks, trackReference: references
        )
    }
}
