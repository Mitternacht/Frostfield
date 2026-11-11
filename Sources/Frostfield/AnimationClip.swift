import Foundation
import simd

/// The local properties one animation source sets on a node at one time.
/// A nil property is not animated by that source.
public struct AnimationNodePose: Equatable, Sendable {
    public var translation: SIMD3<Float>?
    /// A unit quaternion.
    public var rotation: simd_quatf?
    public var scale: SIMD3<Float>?
    /// Morph target weights in the mesh's target order.
    public var morphWeights: [Float]?

    public init(
        translation: SIMD3<Float>? = nil, rotation: simd_quatf? = nil, scale: SIMD3<Float>? = nil,
        morphWeights: [Float]? = nil
    ) {
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.morphWeights = morphWeights
    }

    /// Replaces the animated translation, rotation, and scale of a local transform
    /// and keeps its other properties.
    public func applied(to transform: SceneTransform) -> SceneTransform {
        var result = transform
        if let translation { result.translation = translation }
        if let rotation { result.rotation = rotation }
        if let scale { result.scale = scale }
        return result
    }
}

/// How an action's sampled values combine with the other actions in a mixer.
public enum AnimationBlendMode: Equatable, Sendable {
    /// Values are averaged with other normal actions by weight, and blend
    /// toward the rest pose when the total weight is below one.
    case normal
    /// Values are offsets from a reference pose, added after the normal blend.
    case additive
}

/// A playable animation whose tracks target scene graph node IDs and other typed
/// properties. Build one from a `SceneAnimation`, from an imported glTF clip and its
/// node ID map, or from `AnimationTrack`s, then play it with an `AnimationMixer`.
public struct AnimationClip {
    public let name: String?
    /// The latest key time across the clip's tracks, in seconds.
    public let duration: Float
    public let blendMode: AnimationBlendMode

    let targets: [AnimationNodeTracks]
    private let targetIndices: [UUID: Int]
    /// Additive reference values by node; empty for a normal clip.
    let reference: [UUID: AnimationNodePose]
    /// Typed property tracks, each property at most once.
    public let tracks: [AnimationTrack]
    private let trackIndices: [AnimationProperty: Int]
    /// Additive reference values by property; empty for a normal clip.
    let trackReference: [AnimationProperty: AnimationValue]

    /// The animated node IDs: node tracks in order, then nodes that only property tracks animate.
    public var nodeIDs: [UUID] {
        var seen = Set(targets.map(\.nodeID))
        var ids = targets.map(\.nodeID)
        for track in tracks {
            if case .node(let id, _) = track.property, seen.insert(id).inserted { ids.append(id) }
        }
        return ids
    }

    /// The animated properties of the typed tracks, in track order.
    public var properties: [AnimationProperty] { tracks.map(\.property) }

    /// Wraps a transform animation. Every track sets translation, rotation, and scale.
    public init(_ animation: SceneAnimation, name: String? = nil) {
        self.init(
            name: name, duration: animation.duration, blendMode: .normal,
            targets: animation.tracks.map { AnimationNodeTracks(nodeID: $0.nodeID, transform: $0) }, reference: [:],
            tracks: [], trackReference: [:]
        )
    }

    /// Binds an imported glTF clip to graph nodes, keeping its channels' own
    /// interpolation, including morph weight channels. The map must contain a
    /// distinct graph ID for every animated source node, such as `PreparedGLTFScene.nodeIDs`;
    /// otherwise throws `GLTFAnimationClip.SamplingError.missingNodeMapping` or `duplicateNodeMapping`.
    public init(_ clip: GLTFAnimationClip, nodeIDs: [Int: UUID]) throws {
        var targets: [AnimationNodeTracks] = []
        var indices: [UUID: Int] = [:]
        var sources: [UUID: Int] = [:]
        for channel in clip.channels {
            let source = channel.sourceNodeIndex
            guard let id = nodeIDs[source] else { throw GLTFAnimationClip.SamplingError.missingNodeMapping(source) }
            if let previous = sources[id], previous != source {
                throw GLTFAnimationClip.SamplingError.duplicateNodeMapping(id)
            }
            sources[id] = source
            if let index = indices[id] {
                targets[index].channels.append(channel)
            } else {
                indices[id] = targets.count
                targets.append(AnimationNodeTracks(nodeID: id, channels: [channel]))
            }
        }
        self.init(
            name: clip.name, duration: clip.duration, blendMode: .normal, targets: targets, reference: [:],
            tracks: [], trackReference: [:]
        )
    }

    /// Builds a clip from typed tracks, like Three.js `new AnimationClip(name, duration, tracks)`.
    /// Each property may have one track. The duration defaults to the latest key
    /// time; a shorter one stops playback before the last keys. Throws
    /// `Failure.emptyClip`, `duplicateTrack` or `invalidDuration`.
    public init(name: String? = nil, tracks: [AnimationTrack], duration: Float? = nil) throws {
        guard !tracks.isEmpty else { throw Failure.emptyClip }
        var seen = Set<AnimationProperty>()
        for track in tracks where !seen.insert(track.property).inserted {
            throw Failure.duplicateTrack(track.property.description)
        }
        let end = duration ?? tracks.reduce(0) { max($0, $1.endTime) }
        guard end.isFinite, end >= 0 else { throw Failure.invalidDuration }
        self.init(
            name: name, duration: end, blendMode: .normal, targets: [], reference: [:], tracks: tracks, trackReference: [:]
        )
    }

    init(
        name: String?, duration: Float, blendMode: AnimationBlendMode, targets: [AnimationNodeTracks],
        reference: [UUID: AnimationNodePose], tracks: [AnimationTrack], trackReference: [AnimationProperty: AnimationValue]
    ) {
        self.name = name
        self.duration = duration
        self.blendMode = blendMode
        self.targets = targets
        targetIndices = Dictionary(uniqueKeysWithValues: targets.enumerated().map { ($0.element.nodeID, $0.offset) })
        self.reference = reference
        self.tracks = tracks
        trackIndices = Dictionary(uniqueKeysWithValues: tracks.enumerated().map { ($0.element.property, $0.offset) })
        self.trackReference = trackReference
    }

    /// Returns the values this clip contributes at a time in seconds, keyed by node ID.
    /// Time clamps to the clip's bounds. An additive clip returns offsets from its
    /// reference pose: translation, scale, and weight differences, and the rotation
    /// `reference⁻¹ · sampled`.
    /// Typed tracks that animate a node's translation, rotation, scale or morph
    /// weights contribute to its pose.
    public func sample(at time: Float) throws -> [UUID: AnimationNodePose] {
        guard time.isFinite else { throw Failure.invalidSampleTime }
        var poses: [UUID: AnimationNodePose] = [:]
        for index in targets.indices {
            poses[targets[index].nodeID] = try pose(of: index, at: Double(time), transform: true, morphWeights: true)
        }
        for index in tracks.indices {
            guard case .node(let id, let property) = tracks[index].property else { continue }
            let value = try trackValue(index, at: Double(time))
            switch property {
            case .translation: poses[id, default: AnimationNodePose()].translation = value.vector3Value
            case .rotation: poses[id, default: AnimationNodePose()].rotation = value.quaternionValue
            case .scale: poses[id, default: AnimationNodePose()].scale = value.vector3Value
            case .morphWeights:
                if case .numbers(let weights) = value { poses[id, default: AnimationNodePose()].morphWeights = weights }
            default: continue
            }
        }
        return poses
    }

    /// Returns every typed track's value at a time in seconds, clamped to the clip.
    /// An additive clip returns offsets from its reference values.
    public func sampleProperties(at time: Float) throws -> [AnimationProperty: AnimationValue] {
        guard time.isFinite else { throw Failure.invalidSampleTime }
        var values: [AnimationProperty: AnimationValue] = [:]
        for index in tracks.indices { values[tracks[index].property] = try trackValue(index, at: Double(time)) }
        return values
    }

    /// Returns an additive copy whose values are offsets from a reference pose,
    /// following Three.js `AnimationUtils.makeClipAdditive`. The reference pose is
    /// sampled from `reference`, or from this clip, at `referenceTime` seconds,
    /// clamped to that clip's duration. The reference must animate every property
    /// this clip animates, with the same morph weight counts.
    ///
    /// Typed tracks become offsets from the reference clip's track for the same
    /// property. Boolean and string tracks stay unchanged and need no reference, as
    /// Three.js leaves them; see `AnimationMixer` for how additive actions apply them.
    public func makingAdditive(referenceTime: Float = 0, reference: AnimationClip? = nil) throws -> AnimationClip {
        guard blendMode == .normal else { throw Failure.alreadyAdditive }
        guard referenceTime.isFinite, referenceTime >= 0 else { throw Failure.invalidReferenceTime }
        let source = reference ?? self
        let time = min(Double(referenceTime), Double(source.duration))
        var kept: [AnimationTrack] = []
        var references: [AnimationProperty: AnimationValue] = [:]
        for track in tracks {
            guard track.valueKind.isContinuous else {
                kept.append(track)
                continue
            }
            guard let index = source.trackIndices[track.property] else {
                throw Failure.missingPropertyReference(track.property.description)
            }
            let base = try source.tracks[index].sample(at: time)
            guard base.kind == track.valueKind else { throw Failure.missingPropertyReference(track.property.description) }
            kept.append(track)
            references[track.property] = base
        }
        var poses: [UUID: AnimationNodePose] = [:]
        for target in targets {
            guard let index = source.targetIndices[target.nodeID] else { throw Failure.missingReference(target.nodeID) }
            let base = try source.targets[index].sample(at: time, transform: true, morphWeights: true)
            let animated = target.animatedProperties
            guard !animated.translation || base.translation != nil,
                  !animated.rotation || base.rotation != nil,
                  !animated.scale || base.scale != nil,
                  animated.morphWeightCount.map({ $0 == base.morphWeights?.count }) != false else {
                throw Failure.missingReference(target.nodeID)
            }
            poses[target.nodeID] = AnimationNodePose(
                translation: animated.translation ? base.translation : nil,
                rotation: animated.rotation ? base.rotation.map(simd_normalize) : nil,
                scale: animated.scale ? base.scale : nil,
                morphWeights: animated.morphWeightCount == nil ? nil : base.morphWeights
            )
        }
        return AnimationClip(
            name: name, duration: duration, blendMode: .additive, targets: targets, reference: poses,
            tracks: kept, trackReference: references
        )
    }

    func targetIndex(of nodeID: UUID) -> Int? { targetIndices[nodeID] }

    func trackIndex(of property: AnimationProperty) -> Int? { trackIndices[property] }

    /// Samples one typed track at a time clamped to the clip, converting to an
    /// offset for an additive clip: a difference, or `reference⁻¹ · value` for a quaternion.
    func trackValue(_ index: Int, at time: Double) throws -> AnimationValue {
        let track = tracks[index]
        let value = try track.sample(at: min(max(time, 0), Double(duration)))
        guard blendMode == .additive, let base = trackReference[track.property] else { return value }
        if case .quaternion(let rotation) = value, case .quaternion(let origin) = base {
            return .quaternion(simd_normalize(origin.conjugate * rotation))
        }
        guard let components = value.components, let origins = base.components else { return value }
        return .continuous(value.kind, zip(components, origins).map { $0 - $1 })
    }

    /// Samples one node's tracks at a time clamped to the clip, converting to
    /// offsets for an additive clip.
    func pose(of index: Int, at time: Double, transform: Bool, morphWeights: Bool) throws -> AnimationNodePose {
        let target = targets[index]
        let sampled = try target.sample(
            at: min(max(time, 0), Double(duration)), transform: transform, morphWeights: morphWeights
        )
        guard blendMode == .additive, let base = reference[target.nodeID] else { return sampled }
        var offset = AnimationNodePose()
        if let value = sampled.translation, let origin = base.translation { offset.translation = value - origin }
        if let value = sampled.rotation, let origin = base.rotation {
            offset.rotation = simd_normalize(origin.conjugate * value)
        }
        if let value = sampled.scale, let origin = base.scale { offset.scale = value - origin }
        if let values = sampled.morphWeights, let origins = base.morphWeights {
            offset.morphWeights = zip(values, origins).map { $0 - $1 }
        }
        return offset
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidSampleTime
        case invalidReferenceTime
        case alreadyAdditive
        case missingReference(UUID)
        case missingPropertyReference(String)
        case emptyClip
        case duplicateTrack(String)
        case invalidDuration
        case invalidFrameRange
        case unmappedTrack(String)

        public var errorDescription: String? {
            switch self {
            case .invalidSampleTime: "The animation clip sample time must be finite."
            case .invalidReferenceTime: "The additive reference time must be finite and nonnegative."
            case .alreadyAdditive: "The animation clip is already additive."
            case .missingReference(let id):
                "The additive reference does not animate every property of node \(id) with matching morph weight counts."
            case .missingPropertyReference(let property):
                "The additive reference has no \(property) track with the same value kind."
            case .emptyClip: "An animation clip needs at least one track."
            case .duplicateTrack(let property): "The clip has more than one track for \(property)."
            case .invalidDuration: "A clip duration must be finite and nonnegative."
            case .invalidFrameRange: "A subclip needs a positive frame rate and finite frames with start < end."
            case .unmappedTrack(let property): "The clip's \(property) track has no mapping in the target."
            }
        }
    }
}

/// One node's tracks within a clip: a full transform track, or individual glTF channels.
struct AnimationNodeTracks {
    let nodeID: UUID
    var transform: SceneAnimationTrack?
    var channels: [GLTFAnimationClip.Channel] = []

    init(nodeID: UUID, transform: SceneAnimationTrack? = nil, channels: [GLTFAnimationClip.Channel] = []) {
        self.nodeID = nodeID
        self.transform = transform
        self.channels = channels
    }

    struct Properties {
        var translation = false
        var rotation = false
        var scale = false
        var morphWeightCount: Int?
    }

    var animatedProperties: Properties {
        var result = Properties()
        if transform != nil {
            result.translation = true
            result.rotation = true
            result.scale = true
        }
        for channel in channels {
            switch channel.target {
            case .translation: result.translation = true
            case .rotation: result.rotation = true
            case .scale: result.scale = true
            case .weights: result.morphWeightCount = channel.weightCount
            }
        }
        return result
    }

    var animatesTransform: Bool { transform != nil || channels.contains { $0.target != .weights } }
    var animatesMorphWeights: Bool { channels.contains { $0.target == .weights } }

    func sample(at time: Double, transform includeTransform: Bool, morphWeights: Bool) throws -> AnimationNodePose {
        var pose = AnimationNodePose()
        if includeTransform, let track = transform {
            let sampled = track.sample(at: time)
            pose.translation = sampled.translation
            pose.rotation = sampled.rotation
            pose.scale = sampled.scale
        }
        for channel in channels where channel.target == .weights ? morphWeights : includeTransform {
            try channel.sample(at: time, into: &pose)
        }
        return pose
    }
}
