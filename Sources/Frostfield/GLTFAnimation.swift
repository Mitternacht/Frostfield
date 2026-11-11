import Foundation
import simd

/// A selected scene's glTF animation channels in their source order.
public struct GLTFAnimationClip {
    public let sourceIndex: Int
    public let name: String?
    public let channels: [Channel]
    /// `KHR_animation_pointer` channels that target materials, lights or cameras.
    /// Pointers to node transforms and weights import into `channels`. Bind them
    /// with `AnimationClip(_:in:)`; the other APIs of this type ignore them.
    public let pointerChannels: [PointerChannel]
    /// `KHR_animation_pointer` pointers that Frostfield cannot bind, such as
    /// `KHR_node_visibility` or `KHR_materials_volume` properties, whose channels
    /// were skipped. An asset that lists the extension in `extensionsRequired`
    /// fails to load with such a pointer instead.
    public let unsupportedPointers: [String]
    /// The latest key time among the selected scene's channels, in seconds.
    public let duration: Float

    public struct Channel {
        public enum Target: String, Hashable, Sendable {
            case translation, rotation, scale
            /// The morph target weights of the node's mesh. Imported only when
            /// the scene is loaded with morph targets.
            case weights
        }

        public enum Interpolation: String, Equatable {
            case linear = "LINEAR"
            case step = "STEP"
            case cubicSpline = "CUBICSPLINE"
        }

        public let sourceNodeIndex: Int
        public let target: Target
        public let interpolation: Interpolation
        public let times: [Float]
        /// Translation, rotation, or scale keys. Cubic splines store in-tangent, value, out-tangent triples.
        let values: [SIMD4<Float>]
        /// Morph weight keys, `weightCount` values per key or per spline element.
        let weightValues: [Float]
        let weightCount: Int

        init(sourceNodeIndex: Int, target: Target, interpolation: Interpolation, times: [Float], values: [SIMD4<Float>]) {
            self.sourceNodeIndex = sourceNodeIndex
            self.target = target
            self.interpolation = interpolation
            self.times = times
            self.values = values
            weightValues = []
            weightCount = 0
        }

        init(sourceNodeIndex: Int, interpolation: Interpolation, times: [Float], weights: [Float], weightCount: Int) {
            self.sourceNodeIndex = sourceNodeIndex
            target = .weights
            self.interpolation = interpolation
            self.times = times
            values = []
            weightValues = weights
            self.weightCount = weightCount
        }

        func optimized() -> Self {
            guard interpolation != .cubicSpline else { return self }
            if target == .weights {
                func weights(at key: Int) -> ArraySlice<Float> {
                    let start = key * weightCount
                    return weightValues[start..<(start + weightCount)]
                }
                guard let keys = retainedFlatRunKeys(times.count, matches: { weights(at: $0) == weights(at: $1) })
                else { return self }
                return Self(sourceNodeIndex: sourceNodeIndex, interpolation: interpolation, times: keys.map { times[$0] },
                            weights: keys.flatMap(weights(at:)), weightCount: weightCount)
            }
            guard let keys = retainedFlatRunKeys(times.count, matches: { values[$0] == values[$1] }) else { return self }
            return Self(sourceNodeIndex: sourceNodeIndex, target: target, interpolation: interpolation,
                        times: keys.map { times[$0] }, values: keys.map { values[$0] })
        }

        private var valueStride: Int { interpolation == .cubicSpline ? 3 : 1 }
        private var valueOffset: Int { interpolation == .cubicSpline ? 1 : 0 }

        private enum Segment {
            case key(Int)
            /// Between `key` and `key + 1`.
            case between(key: Int, fraction: Double, span: Double)
        }

        private func segment(at time: Double) -> Segment {
            let last = times.count - 1
            if time <= Double(times[0]) { return .key(0) }
            if time >= Double(times[last]) { return .key(last) }

            var lower = 1
            var upper = last
            while lower < upper {
                let middle = lower + (upper - lower) / 2
                if Double(times[middle]) < time {
                    lower = middle + 1
                } else {
                    upper = middle
                }
            }
            if Double(times[lower]) == time { return .key(lower) }
            let span = Double(times[lower]) - Double(times[lower - 1])
            return .between(key: lower - 1, fraction: (time - Double(times[lower - 1])) / span, span: span)
        }

        /// Writes this channel's property into a node pose.
        func sample(at time: Double, into pose: inout AnimationNodePose) throws {
            switch target {
            case .translation:
                let value = try sampleVector(at: time)
                pose.translation = SIMD3(value.x, value.y, value.z)
            case .rotation:
                pose.rotation = simd_quatf(vector: try sampleVector(at: time))
            case .scale:
                let value = try sampleVector(at: time)
                pose.scale = SIMD3(value.x, value.y, value.z)
            case .weights:
                pose.morphWeights = try sampleWeights(at: time)
            }
        }

        private func sampleVector(at time: Double) throws -> SIMD4<Float> {
            let key: Int
            let fraction: Double
            let span: Double
            switch segment(at: time) {
            case .key(let index): return values[index * valueStride + valueOffset]
            case .between(let index, let between, let length): (key, fraction, span) = (index, between, length)
            }

            let previous = key * valueStride + valueOffset
            let next = previous + valueStride
            let first = values[previous]
            let lastValue = values[next]
            let sampled: SIMD4<Double>
            switch interpolation {
            case .step:
                return first
            case .linear:
                if target == .rotation {
                    let a = simd_quatf(vector: first)
                    let b = simd_quatf(vector: lastValue)
                    return simd_normalize(simd_slerp(a, b, Float(fraction))).vector
                }
                sampled = SIMD4<Double>(first) * (1 - fraction) + SIMD4<Double>(lastValue) * fraction
            case .cubicSpline:
                let square = fraction * fraction
                let cube = square * fraction
                let outgoing = SIMD4<Double>(values[previous + 1])
                let incoming = SIMD4<Double>(values[next - 1])
                let firstValue = SIMD4<Double>(first) * (2 * cube - 3 * square + 1)
                let lastValue = SIMD4<Double>(lastValue) * (-2 * cube + 3 * square)
                let firstTangent = outgoing * (span * (cube - 2 * square + fraction))
                let lastTangent = incoming * (span * (cube - square))
                sampled = firstValue + firstTangent + lastValue + lastTangent
            }

            if target == .rotation {
                let length = simd_length(sampled)
                guard length.isFinite, length > 0 else {
                    throw SamplingError.invalidValue(sourceNodeIndex, target)
                }
                return SIMD4<Float>(sampled / length)
            }
            let value = SIMD4<Float>(sampled)
            guard value.x.isFinite, value.y.isFinite, value.z.isFinite else {
                throw SamplingError.invalidValue(sourceNodeIndex, target)
            }
            return value
        }

        private func sampleWeights(at time: Double) throws -> [Float] {
            let count = weightCount
            func start(_ key: Int, _ element: Int) -> Int { (key * valueStride + element) * count }

            let key: Int
            let fraction: Double
            let span: Double
            switch segment(at: time) {
            case .key(let index):
                let first = start(index, valueOffset)
                return Array(weightValues[first..<(first + count)])
            case .between(let index, let between, let length): (key, fraction, span) = (index, between, length)
            }

            let previous = start(key, valueOffset)
            if interpolation == .step { return Array(weightValues[previous..<(previous + count)]) }
            let next = start(key + 1, valueOffset)
            var result = [Float](repeating: 0, count: count)
            for index in 0..<count {
                let a = Double(weightValues[previous + index])
                let b = Double(weightValues[next + index])
                let value: Double
                if interpolation == .linear {
                    value = a * (1 - fraction) + b * fraction
                } else {
                    let square = fraction * fraction
                    let cube = square * fraction
                    let outgoing = Double(weightValues[start(key, 2) + index])
                    let incoming = Double(weightValues[start(key + 1, 0) + index])
                    value = a * (2 * cube - 3 * square + 1) + outgoing * (span * (cube - 2 * square + fraction))
                        + b * (-2 * cube + 3 * square) + incoming * (span * (cube - square))
                }
                result[index] = Float(value)
                guard result[index].isFinite else { throw SamplingError.invalidValue(sourceNodeIndex, target) }
            }
            return result
        }
    }

    init(
        sourceIndex: Int, name: String?, channels: [Channel], pointerChannels: [PointerChannel] = [],
        unsupportedPointers: [String] = []
    ) {
        self.sourceIndex = sourceIndex
        self.name = name
        self.channels = channels
        self.pointerChannels = pointerChannels
        self.unsupportedPointers = unsupportedPointers
        duration = max(
            channels.reduce(0) { max($0, $1.times.last ?? 0) }, pointerChannels.reduce(0) { max($0, $1.times.last ?? 0) }
        )
    }

    /// Returns the animated local properties at a time in seconds, keyed by graph
    /// node ID, without reading or changing a graph. Properties without a channel
    /// stay nil. Each channel holds its nearest key outside its own input range.
    /// Time clamps to the clip's bounds, or wraps when looping. The map must contain
    /// a distinct graph ID for every animated source node.
    public func sample(at time: Float, nodeIDs: [Int: UUID], looping: Bool = false) throws -> [UUID: AnimationNodePose] {
        try sampledPoses(at: time, nodeIDs: nodeIDs, looping: looping).poses
    }

    /// Samples a copy of the graph. Each channel changes only its target
    /// property and holds its nearest key outside its own input range.
    /// The map must contain a distinct graph ID for every animated source node.
    /// Weights channels leave the graph unchanged, since one glTF node's primitives
    /// can be several draw nodes; `PreparedGLTFScene` applies them to its draws, or
    /// read them with `sample(at:nodeIDs:looping:)` or `AnimationMixer` and set
    /// `SceneNode.morphWeights`.
    public func applying(
        to graph: SceneGraph, nodeIDs: [Int: UUID], at time: Float, looping: Bool = false
    ) throws -> SceneGraph {
        let sampled = try sampledPoses(at: time, nodeIDs: nodeIDs, looping: looping)
        var pending = sampled.poses
        let result = try graph.updatingTransforms { id, transform in
            if let pose = pending.removeValue(forKey: id) { transform = pose.applied(to: transform) }
        }
        if let source = pending.keys.compactMap({ sampled.sources[$0] }).min() {
            throw SamplingError.missingGraphNode(source)
        }
        return result
    }

    private func sampledPoses(
        at time: Float, nodeIDs: [Int: UUID], looping: Bool
    ) throws -> (poses: [UUID: AnimationNodePose], sources: [UUID: Int]) {
        guard time.isFinite else { throw SamplingError.invalidSampleTime }
        let sampleTime = animationSampleTime(time, duration: duration, looping: looping)
        var sources: [UUID: Int] = [:]
        var poses: [UUID: AnimationNodePose] = [:]
        for channel in channels {
            let source = channel.sourceNodeIndex
            guard let id = nodeIDs[source] else { throw SamplingError.missingNodeMapping(source) }
            if let previous = sources[id], previous != source { throw SamplingError.duplicateNodeMapping(id) }
            sources[id] = source
            try channel.sample(at: sampleTime, into: &poses[id, default: AnimationNodePose()])
        }
        return (poses, sources)
    }

    public enum SamplingError: LocalizedError, Equatable {
        case invalidSampleTime
        case invalidValue(Int, Channel.Target)
        case missingNodeMapping(Int)
        case duplicateNodeMapping(UUID)
        case missingGraphNode(Int)

        public var errorDescription: String? {
            switch self {
            case .invalidSampleTime: "The glTF animation sample time must be finite."
            case .invalidValue(let node, let target): "Invalid sampled \(target.rawValue) for glTF node \(node)."
            case .missingNodeMapping(let node): "The glTF animation has no graph ID for node \(node)."
            case .duplicateNodeMapping(let id): "More than one animated glTF node maps to graph ID \(id)."
            case .missingGraphNode(let node): "The graph is missing the mapped glTF node \(node)."
            }
        }
    }
}

struct GLTFAnimationDocument: Decodable {
    private typealias Failure = GLTFMeshLoader.Failure

    private let animations: [Record]?

    private struct Record: Decodable {
        let name: String?
        let channels: [Channel]
        let samplers: [Sampler]
    }

    private struct Channel: Decodable {
        let sampler: Int
        let target: Target
    }

    private struct Target: Decodable {
        let node: Int?
        let path: String
        let extensions: TargetExtensions?
    }

    private struct TargetExtensions: Decodable {
        let pointer: Pointer?

        private enum CodingKeys: String, CodingKey { case pointer = "KHR_animation_pointer" }
    }

    private struct Pointer: Decodable {
        let pointer: String
    }

    /// Which pointer targets were imported, so channels for others are skipped.
    struct PointerTargets {
        /// Imported material indices, or nil when materials were not loaded.
        var materials: Set<Int>?
        /// Imported light indices, or nil when lights were not loaded.
        var lights: Set<Int>?
        /// Imported camera indices, or nil when cameras were not loaded.
        var cameras: Set<Int>?
        /// Whether the asset lists the extension in `extensionsRequired`, which makes
        /// a pointer Frostfield cannot bind an error instead of a skipped channel.
        var required = false
    }

    private struct Sampler: Decodable {
        let input: Int
        let output: Int
        let interpolation: String?
    }

    private struct ChannelKey: Hashable {
        let node: Int
        let target: GLTFAnimationClip.Channel.Target
    }

    /// `morphTargetCounts` maps selected nodes whose meshes have morph targets to
    /// their target counts; nil means morph targets were not requested.
    /// `readPointer` reads a pointer sampler's times and float outputs: input,
    /// output, components per value, values per key and the remaining element budget.
    func resolve(
        nodes: [GLTFHierarchy.NodeRecord], selectedNodes: Set<Int>, maximumElements: Int,
        morphTargetCounts: [Int: Int]?, pointerTargets: PointerTargets = PointerTargets(),
        readSampler: (Int, Int, Bool, Int, Int) throws -> (times: [Float], values: [SIMD4<Float>], unitTolerance: Float),
        readWeights: (Int, Int, Int, Int) throws -> (times: [Float], values: [Float]),
        readPointer: (Int, Int, Int, Int, Int) throws -> (times: [Float], values: [Float]) = { _, _, _, _, _ in
            throw GLTFMeshLoader.Failure.unsupportedFeature(GLTFAnimationPointer.extensionName)
        }
    ) throws -> [GLTFAnimationClip] {
        let records = animations ?? []
        guard records.count <= maximumElements else { throw Failure.animationSizeLimit }
        var remaining = maximumElements
        var clips: [GLTFAnimationClip] = []

        for (index, record) in records.enumerated() {
            func invalid(_ reason: String) -> Failure { .invalidAnimation(index, reason) }
            guard !record.channels.isEmpty, !record.samplers.isEmpty else {
                throw invalid("at least one channel and sampler are required")
            }
            guard record.channels.count <= remaining, record.samplers.count <= remaining - record.channels.count else {
                throw Failure.animationSizeLimit
            }
            remaining -= record.channels.count + record.samplers.count
            var seen = Set<ChannelKey>()
            var seenPointers = Set<GLTFAnimationPointer>()
            var channels: [GLTFAnimationClip.Channel] = []
            var pointerChannels: [GLTFAnimationClip.PointerChannel] = []
            var unsupportedPointers: [String] = []

            for channel in record.channels {
                var targetNode = channel.target.node
                var targetPath = channel.target.path
                if targetPath == "pointer" {
                    guard targetNode == nil, let pointer = channel.target.extensions?.pointer?.pointer else {
                        throw invalid("a pointer channel needs a KHR_animation_pointer pointer and no node")
                    }
                    switch GLTFAnimationPointer.parse(pointer) {
                    case .node(let node, let path)?:
                        targetNode = node
                        targetPath = path.rawValue
                    case .pointer(let target)?:
                        let imported: Set<Int>?
                        let targetIndex: Int
                        switch target {
                        case .material(let index, _): (imported, targetIndex) = (pointerTargets.materials, index)
                        case .light(let index, _): (imported, targetIndex) = (pointerTargets.lights, index)
                        case .camera(let index, _): (imported, targetIndex) = (pointerTargets.cameras, index)
                        }
                        // Pointers to data that was not loaded, or is unused by the selected scene, are skipped.
                        guard let imported, imported.contains(targetIndex) else { continue }
                        guard seenPointers.insert(target).inserted else { throw invalid("pointer \(pointer) has duplicate channels") }
                        guard record.samplers.indices.contains(channel.sampler) else {
                            throw invalid("sampler \(channel.sampler) does not exist")
                        }
                        let sampler = record.samplers[channel.sampler]
                        let mode = sampler.interpolation ?? "LINEAR"
                        guard let interpolation = GLTFAnimationClip.Channel.Interpolation(rawValue: mode) else {
                            throw Failure.unsupportedFeature("animation interpolation \(mode)")
                        }
                        let cubic = interpolation == .cubicSpline
                        let output = try readPointer(
                            sampler.input, sampler.output, target.componentCount, cubic ? 3 : 1, remaining
                        )
                        if cubic, output.times.count < 2 { throw invalid("a cubic spline requires at least two keys") }
                        remaining -= output.times.count + output.values.count
                        pointerChannels.append(.init(
                            pointer: pointer, target: target, interpolation: interpolation, times: output.times,
                            values: output.values
                        ))
                        continue
                    case nil:
                        // Clients without the extension ignore pointer channels, so an asset that does not
                        // require it keeps its other animation; one that requires it needs every pointer.
                        guard !pointerTargets.required else {
                            throw Failure.unsupportedFeature("\(GLTFAnimationPointer.extensionName) target \(pointer)")
                        }
                        unsupportedPointers.append(pointer)
                        continue
                    }
                }
                guard let node = targetNode else { continue }
                guard nodes.indices.contains(node) else { throw Failure.missingNode(node) }
                guard selectedNodes.contains(node) else { continue }
                guard let target = GLTFAnimationClip.Channel.Target(rawValue: targetPath),
                      target != .weights || morphTargetCounts != nil else {
                    throw Failure.unsupportedFeature("animation target \(targetPath)")
                }
                guard nodes[node].matrix == nil else { throw invalid("node \(node) uses a matrix instead of TRS") }
                guard seen.insert(ChannelKey(node: node, target: target)).inserted else {
                    throw invalid("node \(node) has duplicate \(target.rawValue) channels")
                }
                guard record.samplers.indices.contains(channel.sampler) else {
                    throw invalid("sampler \(channel.sampler) does not exist")
                }
                let sampler = record.samplers[channel.sampler]
                let mode = sampler.interpolation ?? "LINEAR"
                guard let interpolation = GLTFAnimationClip.Channel.Interpolation(rawValue: mode) else {
                    throw Failure.unsupportedFeature("animation interpolation \(mode)")
                }

                let cubic = interpolation == .cubicSpline
                if target == .weights {
                    guard let count = morphTargetCounts?[node] else {
                        throw invalid("node \(node) has no morph targets for its weights channel")
                    }
                    let output = try readWeights(sampler.input, sampler.output, count * (cubic ? 3 : 1), remaining)
                    if cubic, output.times.count < 2 { throw invalid("a cubic spline requires at least two keys") }
                    remaining -= output.times.count + output.values.count
                    channels.append(.init(
                        sourceNodeIndex: node, interpolation: interpolation, times: output.times,
                        weights: output.values, weightCount: count
                    ))
                    continue
                }

                var output = try readSampler(sampler.input, sampler.output, target == .rotation, cubic ? 3 : 1, remaining)
                if cubic, output.times.count < 2 { throw invalid("a cubic spline requires at least two keys") }
                let outputCount = output.values.count
                remaining -= output.times.count + outputCount
                if target == .rotation {
                    // Tangents are derivatives, not unit quaternions.
                    for key in stride(from: cubic ? 1 : 0, to: outputCount, by: cubic ? 3 : 1) {
                        let length = simd_length(output.values[key])
                        guard abs(length - 1) <= output.unitTolerance else {
                            throw invalid("rotation key is not a unit quaternion")
                        }
                        output.values[key] /= length
                    }
                }
                channels.append(.init(
                    sourceNodeIndex: node, target: target, interpolation: interpolation, times: output.times, values: output.values
                ))
            }
            if !channels.isEmpty || !pointerChannels.isEmpty || !unsupportedPointers.isEmpty {
                clips.append(.init(
                    sourceIndex: index, name: record.name, channels: channels, pointerChannels: pointerChannels,
                    unsupportedPointers: unsupportedPointers
                ))
            }
        }
        return clips
    }
}
