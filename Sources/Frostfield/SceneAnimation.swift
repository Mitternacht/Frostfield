import Foundation
import simd

/// A local node transform at a nonnegative time in seconds.
public struct TransformKeyframe {
    public let time: Float
    public let transform: SceneTransform

    public init(time: Float, transform: SceneTransform) {
        self.time = time
        self.transform = transform
    }
}

/// An ordered sequence of local transforms for one node.
public struct SceneAnimationTrack {
    public let nodeID: UUID
    public let keyframes: [TransformKeyframe]

    /// Adjacent scale components must keep their signs so the interpolation
    /// stays clear of a zero scale. A track with one key holds its transform.
    public init(nodeID: UUID, keyframes: [TransformKeyframe]) throws {
        guard !keyframes.isEmpty else { throw SceneAnimation.ValidationError.emptyTrack(nodeID) }

        for (index, frame) in keyframes.enumerated() {
            guard frame.time.isFinite, frame.time >= 0,
                  index == 0 || frame.time > keyframes[index - 1].time else {
                throw SceneAnimation.ValidationError.invalidKeyTime(nodeID, keyframe: index)
            }
            guard Self.isValid(frame.transform) else {
                throw SceneAnimation.ValidationError.invalidTransform(nodeID, keyframe: index)
            }
            if index > 0 {
                let previous = keyframes[index - 1].transform.scale
                let next = frame.transform.scale
                guard previous.x.sign == next.x.sign,
                      previous.y.sign == next.y.sign,
                      previous.z.sign == next.z.sign else {
                    throw SceneAnimation.ValidationError.scaleCrossesZero(nodeID, keyframe: index)
                }
            }
        }

        self.init(nodeID: nodeID, validatedKeyframes: keyframes)
    }

    private init(nodeID: UUID, validatedKeyframes: [TransformKeyframe]) {
        self.nodeID = nodeID
        keyframes = validatedKeyframes
    }

    func optimized() -> Self {
        guard let keys = retainedFlatRunKeys(keyframes.count, matches: { keyframes[$0].transform == keyframes[$1].transform })
        else { return self }
        return Self(nodeID: nodeID, validatedKeyframes: keys.map { keyframes[$0] })
    }

    var endTime: Float { keyframes[keyframes.count - 1].time }

    func sample(at time: Double) -> SceneTransform {
        if time <= Double(keyframes[0].time) { return keyframes[0].transform }
        if time >= Double(endTime) { return keyframes[keyframes.count - 1].transform }

        var lower = 1
        var upper = keyframes.count - 1
        while lower < upper {
            let middle = lower + (upper - lower) / 2
            if Double(keyframes[middle].time) < time {
                lower = middle + 1
            } else {
                upper = middle
            }
        }

        let next = keyframes[lower]
        if Double(next.time) == time { return next.transform }
        let previous = keyframes[lower - 1]
        let fraction = (time - Double(previous.time)) / (Double(next.time) - Double(previous.time))
        return SceneTransform(
            translation: Self.interpolate(previous.transform.translation, next.transform.translation, fraction),
            rotation: simd_slerp(previous.transform.rotation, next.transform.rotation, Float(fraction)),
            scale: Self.interpolate(previous.transform.scale, next.transform.scale, fraction)
        )
    }

    private static func interpolate(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ fraction: Double) -> SIMD3<Float> {
        let remaining = 1 - fraction
        return SIMD3(
            Float(Double(a.x) * remaining + Double(b.x) * fraction),
            Float(Double(a.y) * remaining + Double(b.y) * fraction),
            Float(Double(a.z) * remaining + Double(b.z) * fraction)
        )
    }

    private static func isValid(_ transform: SceneTransform) -> Bool {
        guard abs(simd_length_squared(transform.rotation.vector) - 1) <= 0.0001 else { return false }
        let matrix = transform.matrix
        guard isFinite(matrix) else { return false }
        let determinant = simd_determinant(matrix)
        return determinant.isFinite && determinant != 0 && isFinite(simd_inverse(matrix))
    }

    private static func isFinite(_ matrix: simd_float4x4) -> Bool {
        let (x, y, z, w) = matrix.columns
        return [x, y, z, w].allSatisfy {
            $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite
        }
    }
}

/// Samples one transform track per animated node, leaving the original graph intact.
public struct SceneAnimation {
    public let tracks: [SceneAnimationTrack]
    /// The latest key time across all tracks.
    public let duration: Float

    public init(tracks: [SceneAnimationTrack]) throws {
        guard !tracks.isEmpty else { throw ValidationError.emptyAnimation }
        var seen = Set<UUID>()
        for track in tracks {
            guard seen.insert(track.nodeID).inserted else { throw ValidationError.duplicateTrack(track.nodeID) }
        }
        self.tracks = tracks
        duration = tracks.reduce(0) { max($0, $1.endTime) }
    }

    /// Returns each track's local transform at a time in seconds, keyed by node ID,
    /// without reading or changing a graph. Time clamps to the animation's bounds,
    /// or wraps when looping. A shorter track holds its first or last key outside
    /// its own time range. Throws `invalidSampleTime` for a nonfinite time.
    public func sample(at time: Float, looping: Bool = false) throws -> [UUID: SceneTransform] {
        guard time.isFinite else { throw ValidationError.invalidSampleTime }
        let sampleTime = animationSampleTime(time, duration: duration, looping: looping)
        return Dictionary(uniqueKeysWithValues: tracks.map { ($0.nodeID, $0.sample(at: sampleTime)) })
    }

    /// Replaces the targeted local transforms in a graph value with `sample(at:looping:)`.
    /// Throws when a target node is missing or the sampled graph no longer resolves.
    public func applying(to graph: SceneGraph, at time: Float, looping: Bool = false) throws -> SceneGraph {
        var remaining = try sample(at: time, looping: looping)
        let result = try graph.updatingTransforms { id, transform in
            if let sampled = remaining.removeValue(forKey: id) { transform = sampled }
        }
        if let missing = tracks.first(where: { remaining[$0.nodeID] != nil }) {
            throw ValidationError.missingNode(missing.nodeID)
        }
        return result
    }

    public enum ValidationError: LocalizedError, Equatable {
        case emptyAnimation
        case emptyTrack(UUID)
        case invalidKeyTime(UUID, keyframe: Int)
        case invalidTransform(UUID, keyframe: Int)
        case scaleCrossesZero(UUID, keyframe: Int)
        case duplicateTrack(UUID)
        case missingNode(UUID)
        case invalidSampleTime

        public var errorDescription: String? {
            switch self {
            case .emptyAnimation: "A scene animation needs at least one transform track."
            case .emptyTrack(let id): "Animation track for node \(id) needs at least one keyframe."
            case .invalidKeyTime(let id, let index):
                "Keyframe \(index) for node \(id) must have a finite, nonnegative time greater than its predecessor."
            case .invalidTransform(let id, let index):
                "Keyframe \(index) for node \(id) needs a finite, invertible transform with a unit rotation."
            case .scaleCrossesZero(let id, let index):
                "Scale cannot cross zero between keyframe \(index) for node \(id) and its predecessor."
            case .duplicateTrack(let id): "Scene animation contains more than one track for node \(id)."
            case .missingNode(let id): "Animated node \(id) is missing from the scene."
            case .invalidSampleTime: "Scene animation sample time must be finite."
            }
        }
    }
}

func animationSampleTime(_ time: Float, duration: Float, looping: Bool) -> Double {
    let end = Double(duration)
    if looping, duration > 0 {
        let remainder = Double(time).truncatingRemainder(dividingBy: end)
        return remainder < 0 ? remainder + end : remainder
    }
    return min(max(Double(time), 0), end)
}

extension SceneGraph {
    func updatingTransforms(_ update: (UUID, inout SceneTransform) -> Void) throws -> SceneGraph {
        let result = updatingNodes { node in update(node.id, &node.transform) }
        _ = try result.resolve()
        return result
    }

    func updatingNodes(_ update: (inout SceneNode) -> Void) -> SceneGraph {
        // Copy the graph so environment settings carry over with the camera.
        var result = self
        result.roots = SceneNode.updatingForest(roots, update)
        return result
    }
}

extension SceneNode {
    static func updatingForest(_ roots: [SceneNode], _ update: (inout SceneNode) -> Void) -> [SceneNode] {
        var pending = roots.reversed().map { (node: $0, visited: false) }
        var rebuilt: [SceneNode] = []

        // Rebuild children before their parent without recursing into the hierarchy.
        while let (original, visited) = pending.popLast() {
            var node = original
            if !visited, !node.children.isEmpty {
                pending.append((node, true))
                for child in node.children.reversed() { pending.append((child, false)) }
                continue
            }
            let count = node.children.count
            if count > 0 {
                node.children = Array(rebuilt.suffix(count))
                rebuilt.removeLast(count)
            }
            update(&node)
            rebuilt.append(node)
        }

        return rebuilt
    }
}
