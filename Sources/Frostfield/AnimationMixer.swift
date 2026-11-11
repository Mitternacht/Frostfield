import Foundation
import simd

/// How an action repeats its clip.
public enum AnimationLoopMode: Equatable, Sendable {
    /// One pass through the clip, then the action finishes.
    case once
    /// Wraps to the opposite end after each pass. A nil count repeats without
    /// limit; otherwise the action finishes after `count` passes.
    case `repeat`(count: Int? = nil)
    /// Alternates forward and reversed passes. A nil count repeats without
    /// limit; otherwise the action finishes after `count` passes.
    case pingPong(count: Int? = nil)

    var passLimit: Int? {
        switch self {
        case .once: 1
        case .repeat(let count), .pingPong(let count): count
        }
    }
}

/// The direction a playhead moved when an action finished.
public enum AnimationPlaybackDirection: Equatable, Sendable {
    case forward, backward
}

/// One clip's playback state in an `AnimationMixer`. Read it with
/// `AnimationMixer.action(_:)` and change it through the mixer.
public struct AnimationAction: Identifiable {
    public struct ID: Hashable, Sendable {
        fileprivate let value = UUID()
    }

    public let id: ID
    public let clip: AnimationClip
    /// The live destination group, when this action is attached to one.
    public internal(set) var objectGroup: AnimationObjectGroup?
    /// The configured blend weight, finite and nonnegative. Fades scale it.
    public internal(set) var weight: Float
    /// The configured playback rate. Negative values play backward and zero holds the playhead.
    public internal(set) var timeScale: Float
    public internal(set) var loop: AnimationLoopMode
    /// When true, a finished action pauses on its final pose. Otherwise it stops
    /// and its nodes return to the rest pose.
    public internal(set) var clampWhenFinished: Bool
    /// Seconds into the current pass, from zero to the clip duration.
    public internal(set) var time: Double = 0
    /// Whether the mixer advances and blends this action.
    public internal(set) var isPlaying = false
    /// A paused action keeps its weight but does not advance.
    public internal(set) var isPaused = false
    /// The fade multiplier at the mixer's current time, from zero to one.
    public internal(set) var fadeFactor: Float = 1

    /// The zero-based pass number, or -1 before the first pass starts.
    var pass = -1
    var fade: AnimationRamp?
    var warp: AnimationRamp?

    init(clip: AnimationClip, weight: Float, timeScale: Float, loop: AnimationLoopMode, clampWhenFinished: Bool) {
        id = ID()
        self.clip = clip
        self.weight = weight
        self.timeScale = timeScale
        self.loop = loop
        self.clampWhenFinished = clampWhenFinished
    }

    /// The weight used for blending: zero unless playing, otherwise the weight times the fade factor.
    public var effectiveWeight: Float { isPlaying ? weight * fadeFactor : 0 }

    /// The playback rate at the mixer's current time, including warps; zero while
    /// stopped or paused.
    public var effectiveTimeScale: Float {
        guard isPlaying, !isPaused else { return 0 }
        return currentTimeScale
    }

    /// Whether the playhead moves on the next update.
    public var isRunning: Bool { effectiveTimeScale != 0 }

    /// The clip time sampled for blending. Ping-pong mode reverses odd passes.
    public var clipTime: Double {
        if case .pingPong = loop, pass > 0, pass % 2 == 1 { return Double(clip.duration) - time }
        return time
    }

    var currentTimeScale: Float { warp?.current ?? timeScale }

    func destinationNodes(for source: UUID) -> [UUID] {
        objectGroup?.destinationNodes(for: source) ?? [source]
    }

    func sourceNode(for destination: UUID) -> UUID? {
        guard let objectGroup else { return destination }
        return objectGroup.sourceNode(for: destination)
    }

    mutating func stop() {
        isPlaying = false
        reset()
    }

    mutating func reset() {
        time = 0
        pass = -1
        isPaused = false
        fadeFactor = 1
        fade = nil
        warp = nil
    }

    /// Moves the playhead by a signed clip-time delta and reports wraps and finishes.
    mutating func advance(by delta: Double, events: inout [AnimationMixer.Event]) {
        guard delta != 0 else { return }
        let duration = Double(clip.duration)
        let direction: AnimationPlaybackDirection = delta > 0 ? .forward : .backward
        if pass < 0, delta > 0 || loop == .once { pass = 0 }
        let next = time + delta

        if loop == .once {
            if next >= duration {
                finish(at: duration, direction: direction, events: &events)
            } else if next < 0 {
                finish(at: 0, direction: direction, events: &events)
            } else {
                time = next
            }
            return
        }

        // A zero-length clip has no passes to count, so it holds its pose.
        guard duration > 0 else { return }
        guard next >= duration || next < 0 else {
            time = next
            return
        }
        let wraps = (next / duration).rounded(.down)
        let crossed = Int(min(abs(wraps), 1e15))
        // A reversed start wraps from zero into its first pass without completing one.
        let entering = pass < 0
        let (sum, overflow) = pass.addingReportingOverflow(crossed)
        let passes = overflow ? Int.max : sum
        if let limit = loop.passLimit, passes >= limit {
            pass = limit - 1
            finish(at: direction == .forward ? duration : 0, direction: direction, events: &events)
            return
        }
        pass = passes
        time = min(max(next - duration * wraps, 0), duration)
        let completed = entering ? crossed - 1 : crossed
        if completed > 0 { events.append(.looped(id, loopDelta: direction == .forward ? completed : -completed)) }
    }

    private mutating func finish(
        at boundary: Double, direction: AnimationPlaybackDirection, events: inout [AnimationMixer.Event]
    ) {
        time = boundary
        events.append(.finished(id, direction: direction))
        if clampWhenFinished { isPaused = true } else { stop() }
    }
}

/// A linear ramp over mixer time.
struct AnimationRamp {
    let start: Double
    let end: Double
    let from: Float
    let to: Float
    /// The value at the mixer's current time.
    var current: Float

    init(start: Double, duration: Double, from: Float, to: Float) {
        self.start = start
        end = start + duration
        self.from = from
        self.to = to
        current = from
    }

    func value(at time: Double) -> Float {
        if time >= end { return to }
        if time <= start { return from }
        let fraction = (time - start) / (end - start)
        return Float(Double(from) + (Double(to) - Double(from)) * fraction)
    }

    /// The exact area under the ramp between two mixer times.
    func integral(from first: Double, to last: Double) -> Double {
        var area = 0.0
        let rampEnd = min(last, end)
        if rampEnd > first {
            area += (Double(value(at: first)) + Double(value(at: rampEnd))) / 2 * (rampEnd - first)
        }
        let holdStart = max(first, end)
        if last > holdStart { area += Double(to) * (last - holdStart) }
        return area
    }
}

/// Blends any number of animation clips into scene graph poses, following the
/// Three.js `AnimationMixer` and `AnimationAction` model.
///
/// The mixer is a value type: it owns no graph, keeps no callbacks, and changes
/// only through its mutating methods, so the same calls always produce the same
/// state and a copy is an independent snapshot. Call `update(deltaTime:)` once a
/// frame, then `apply(to:)` and `morphWeights(for:rest:)` to read the blend.
///
/// Blending runs per node and property, in action order:
/// - Normal actions average by effective weight. Translation and scale blend
///   linearly. Rotation is the normalized weighted sum of quaternions, each
///   flipped into the hemisphere of the first contribution. When the total
///   weight is below one, the graph's current local value (the rest pose)
///   fills the remaining weight.
/// - Additive actions then add their weighted offsets: translation, scale, and
///   morph weights add linearly, and rotation multiplies `rest · offsetʷ` in
///   action order using the shortest-arc offset.
///
/// Properties that no playing action animates keep the graph's value.
public struct AnimationMixer {
    /// The mixer clock in seconds. Fades and warps are scheduled on this clock.
    public private(set) var time: Double = 0
    /// Actions in the order they were added, which is also their blend order.
    public private(set) var actions: [AnimationAction] = []

    public init() {}

    /// A lifecycle change found while advancing the mixer.
    public enum Event: Equatable, Sendable {
        /// A repeating action completed one or more passes and continued.
        /// The delta is negative when the playhead moved backward.
        case looped(AnimationAction.ID, loopDelta: Int)
        /// A once action, or a repeating action that used its last pass, reached an end.
        case finished(AnimationAction.ID, direction: AnimationPlaybackDirection)
    }

    /// Adds a stopped action for a clip. The weight must be finite and
    /// nonnegative, the time scale finite, and loop counts positive.
    public mutating func addAction(
        for clip: AnimationClip, weight: Float = 1, timeScale: Float = 1,
        loop: AnimationLoopMode = .repeat(), clampWhenFinished: Bool = false
    ) throws -> AnimationAction.ID {
        try Self.validate(weight: weight)
        try Self.validate(timeScale: timeScale)
        try Self.validate(loop: loop)
        let action = AnimationAction(
            clip: clip, weight: weight, timeScale: timeScale, loop: loop, clampWhenFinished: clampWhenFinished
        )
        actions.append(action)
        return action.id
    }

    /// Adds one action whose node tracks fan out to the current group members.
    /// Other typed properties are applied once. The group template must cover
    /// every animated node in the clip.
    public mutating func addAction(
        for clip: AnimationClip, on group: AnimationObjectGroup, weight: Float = 1, timeScale: Float = 1,
        loop: AnimationLoopMode = .repeat(), clampWhenFinished: Bool = false
    ) throws -> AnimationAction.ID {
        if let missing = clip.nodeIDs.first(where: { !group.sourceNodes.contains($0) }) {
            throw AnimationObjectGroup.Failure.unknownClipNode(missing)
        }
        let id = try addAction(
            for: clip, weight: weight, timeScale: timeScale, loop: loop, clampWhenFinished: clampWhenFinished
        )
        actions[actions.count - 1].objectGroup = group
        return id
    }

    public func action(_ id: AnimationAction.ID) throws -> AnimationAction {
        actions[try index(of: id)]
    }

    public mutating func removeAction(_ id: AnimationAction.ID) throws {
        actions.remove(at: try index(of: id))
    }

    /// Starts blending and advancing an action from its current time.
    /// Playing an action that is already playing changes nothing.
    public mutating func play(_ id: AnimationAction.ID) throws {
        actions[try index(of: id)].isPlaying = true
    }

    /// Stops an action and resets its time, loop count, pause, fade, and warp.
    /// Its weight, time scale, and loop settings remain.
    public mutating func stop(_ id: AnimationAction.ID) throws {
        actions[try index(of: id)].stop()
    }

    public mutating func stopAll() {
        for index in actions.indices { actions[index].stop() }
    }

    /// Rewinds an action and clears its pause, fade, and warp without changing whether it plays.
    public mutating func reset(_ id: AnimationAction.ID) throws {
        actions[try index(of: id)].reset()
    }

    /// Sets the configured weight, which must be finite and nonnegative. An active fade keeps scaling it.
    public mutating func setWeight(_ weight: Float, for id: AnimationAction.ID) throws {
        let index = try index(of: id)
        try Self.validate(weight: weight)
        actions[index].weight = weight
    }

    /// Sets a finite playback rate and cancels any warp.
    public mutating func setTimeScale(_ timeScale: Float, for id: AnimationAction.ID) throws {
        let index = try index(of: id)
        try Self.validate(timeScale: timeScale)
        actions[index].timeScale = timeScale
        actions[index].warp = nil
    }

    /// Changes the loop mode without moving the playhead. Counts must be positive.
    public mutating func setLoop(_ loop: AnimationLoopMode, for id: AnimationAction.ID) throws {
        let index = try index(of: id)
        try Self.validate(loop: loop)
        actions[index].loop = loop
    }

    public mutating func setClampWhenFinished(_ clamp: Bool, for id: AnimationAction.ID) throws {
        actions[try index(of: id)].clampWhenFinished = clamp
    }

    public mutating func setPaused(_ paused: Bool, for id: AnimationAction.ID) throws {
        actions[try index(of: id)].isPaused = paused
    }

    /// Moves the playhead within the current pass, from zero to the clip duration in seconds.
    /// To play a once action backward from its end, set its time to the duration first.
    public mutating func setTime(_ time: Double, for id: AnimationAction.ID) throws {
        let index = try index(of: id)
        guard time.isFinite, time >= 0, time <= Double(actions[index].clip.duration) else {
            throw Failure.invalidTime
        }
        actions[index].time = time
        actions[index].pass = max(actions[index].pass, 0)
    }

    /// Ramps the fade factor to one over a duration in mixer seconds, starting
    /// from its current value. A stopped action starts playing at zero weight.
    public mutating func fadeIn(_ id: AnimationAction.ID, duration: Double) throws {
        let index = try index(of: id)
        try Self.validate(duration: duration)
        startFadeIn(index, duration: duration)
    }

    /// Ramps the fade factor of a playing action to zero over a duration in
    /// mixer seconds, then stops the action.
    public mutating func fadeOut(_ id: AnimationAction.ID, duration: Double) throws {
        let index = try index(of: id)
        try Self.validate(duration: duration)
        guard actions[index].isPlaying else { throw Failure.notPlaying(id) }
        startFadeOut(index, duration: duration)
    }

    /// Fades a playing action out and another in over the same duration, starting
    /// the incoming action if needed. With `warp`, both playback rates ramp so the
    /// outgoing action ends with the incoming action's loop period and the incoming
    /// action starts with the outgoing period, like Three.js `crossFadeTo(_:_:true)`.
    /// Warping requires clips with nonzero durations.
    public mutating func crossFade(
        from outgoing: AnimationAction.ID, to incoming: AnimationAction.ID, duration: Double, warp: Bool = false
    ) throws {
        let source = try index(of: outgoing)
        let target = try index(of: incoming)
        guard source != target else { throw Failure.sameAction }
        try Self.validate(duration: duration)
        guard actions[source].isPlaying else { throw Failure.notPlaying(outgoing) }

        if warp {
            let outgoingDuration = Double(actions[source].clip.duration)
            let incomingDuration = Double(actions[target].clip.duration)
            guard outgoingDuration > 0, incomingDuration > 0 else { throw Failure.warpRequiresDuration }
            let outgoingScale = Double(actions[source].currentTimeScale)
            let incomingScale = Double(actions[target].timeScale)
            let outgoingEnd = Float(incomingScale * outgoingDuration / incomingDuration)
            let incomingStart = Float(outgoingScale * incomingDuration / outgoingDuration)
            guard outgoingEnd.isFinite, incomingStart.isFinite else { throw Failure.invalidTimeScale }
            startFadeIn(target, duration: duration)
            startWarp(source, from: Float(outgoingScale), to: outgoingEnd, duration: duration)
            startWarp(target, from: incomingStart, to: Float(incomingScale), duration: duration)
        } else {
            startFadeIn(target, duration: duration)
        }
        startFadeOut(source, duration: duration)
    }

    /// Ramps a playing action's rate between two finite time scales over a
    /// duration in mixer seconds. The playhead integrates the ramp exactly, so the
    /// result does not depend on the update step. A warp that ends at zero pauses
    /// the action and keeps its configured time scale; otherwise the end value
    /// becomes the configured time scale.
    public mutating func warp(_ id: AnimationAction.ID, from start: Float, to end: Float, duration: Double) throws {
        let index = try index(of: id)
        try Self.validate(timeScale: start)
        try Self.validate(timeScale: end)
        try Self.validate(duration: duration)
        guard actions[index].isPlaying else { throw Failure.notPlaying(id) }
        startWarp(index, from: start, to: end, duration: duration)
    }

    /// Advances the mixer clock by a finite, nonnegative number of seconds, then
    /// advances each playing action by its effective time scale, completes fades
    /// and warps, and returns loop and finish events in action order. A completed
    /// fade-out stops its action. A finished action without clamping stops.
    @discardableResult
    public mutating func update(deltaTime: Double) throws -> [Event] {
        guard deltaTime.isFinite, deltaTime >= 0, (time + deltaTime).isFinite else { throw Failure.invalidDeltaTime }
        let start = time
        let end = time + deltaTime
        time = end
        var events: [Event] = []

        for index in actions.indices where actions[index].isPlaying {
            var action = actions[index]
            // A warp that halts during this step still moves the playhead until it ends.
            let wasPaused = action.isPaused
            var delta: Double
            if let warp = action.warp {
                delta = warp.integral(from: start, to: end)
                if end >= warp.end {
                    action.warp = nil
                    if warp.to == 0 { action.isPaused = true } else { action.timeScale = warp.to }
                } else {
                    action.warp?.current = warp.value(at: end)
                }
            } else {
                delta = Double(action.timeScale) * deltaTime
            }
            if wasPaused { delta = 0 }
            action.advance(by: delta, events: &events)

            if action.isPlaying, let fade = action.fade {
                action.fadeFactor = fade.value(at: end)
                if end >= fade.end {
                    action.fade = nil
                    if fade.to == 0 { action.stop() }
                }
            }
            actions[index] = action
        }
        return events
    }

    /// Returns a copy of the graph with blended local transforms written to every
    /// node that a playing action with positive weight animates, and the blended
    /// values of typed graph properties: node morph weights, material parameters,
    /// visibility, shadow casting and light parameters, and the camera and environment. Other
    /// nodes and properties keep the graph's values. Throws `missingNode` when a
    /// blended node is absent, `unanimatableProperty` when a node lacks an animated
    /// property, such as a light, and rethrows sampling and validation errors.
    public func apply(to graph: SceneGraph) throws -> SceneGraph {
        var blends: [UUID: TransformBlend] = [:]
        var order: [UUID] = []
        var ordered = Set<UUID>()
        var properties = PropertyBlends()
        for action in actions {
            let weight = Double(action.effectiveWeight)
            guard weight > 0 else { continue }
            let clip = action.clip
            for index in clip.targets.indices where clip.targets[index].animatesTransform {
                let destinations = action.destinationNodes(for: clip.targets[index].nodeID)
                guard !destinations.isEmpty else { continue }
                let pose = try clip.pose(of: index, at: action.clipTime, transform: true, morphWeights: false)
                for id in destinations {
                    if ordered.insert(id).inserted { order.append(id) }
                    blends[id, default: TransformBlend()].add(pose, weight: weight, mode: clip.blendMode)
                }
            }
            for index in clip.tracks.indices where clip.tracks[index].property.isGraphProperty {
                let track = clip.tracks[index]
                guard case .node(let source, let field) = track.property else {
                    try properties.add(
                        try clip.trackValue(index, at: action.clipTime), for: track.property, kind: track.valueKind,
                        weight: weight, additive: clip.blendMode == .additive
                    )
                    continue
                }
                let destinations = action.destinationNodes(for: source)
                guard !destinations.isEmpty else { continue }
                let value = try clip.trackValue(index, at: action.clipTime)
                let pose: AnimationNodePose? = switch field {
                case .translation: .init(translation: value.vector3Value)
                case .rotation: .init(rotation: value.quaternionValue)
                case .scale: .init(scale: value.vector3Value)
                default: nil
                }
                for id in destinations {
                    if ordered.insert(id).inserted { order.append(id) }
                    if let pose {
                        blends[id, default: TransformBlend()].add(pose, weight: weight, mode: clip.blendMode)
                    } else {
                        try properties.add(
                            value, for: .node(id, field), kind: track.valueKind,
                            weight: weight, additive: clip.blendMode == .additive
                        )
                    }
                }
            }
        }
        guard !blends.isEmpty || !properties.isEmpty else { return graph }

        var pending = blends
        var result = graph.updatingNodes { node in
            if let blend = pending.removeValue(forKey: node.id) { node.transform = blend.resolve(rest: node.transform) }
        }
        let applied = try applyGraphProperties(properties, to: result)
        result = applied.graph
        _ = try result.resolve()
        let missing = Set(applied.missing)
        if let absent = order.first(where: { pending[$0] != nil || missing.contains($0) }) { throw Failure.missingNode(absent) }
        return result
    }

    /// Returns the blended morph target weights for a node. The rest weights,
    /// usually the mesh or node defaults, fill weight below one and receive
    /// additive offsets. Returns the rest weights when no playing action animates
    /// the node's morph weights. Rest weights must be finite, nonempty, and match
    /// every contributing track's count.
    public func morphWeights(for nodeID: UUID, rest: [Float]) throws -> [Float] {
        guard !rest.isEmpty, rest.allSatisfy(\.isFinite) else { throw Failure.invalidRestWeights(nodeID) }
        var blend = WeightBlend(count: rest.count)
        for action in actions {
            let weight = Double(action.effectiveWeight)
            let clip = action.clip
            guard weight > 0, let source = action.sourceNode(for: nodeID),
                  let index = clip.targetIndex(of: source), clip.targets[index].animatesMorphWeights else {
                continue
            }
            let pose = try clip.pose(of: index, at: action.clipTime, transform: false, morphWeights: true)
            guard let values = pose.morphWeights, values.count == rest.count else {
                throw Failure.morphWeightCountMismatch(nodeID, expected: rest.count, actual: pose.morphWeights?.count ?? 0)
            }
            blend.add(values, weight: weight, additive: clip.blendMode == .additive)
        }
        // Typed tracks for every weight blend like weights channels; single-weight tracks then apply on top.
        for action in actions {
            let weight = Double(action.effectiveWeight)
            let clip = action.clip
            guard weight > 0, let source = action.sourceNode(for: nodeID),
                  let index = clip.trackIndex(of: .node(source, .morphWeights)) else { continue }
            guard case .numbers(let values) = try clip.trackValue(index, at: action.clipTime), values.count == rest.count else {
                throw Failure.morphWeightCountMismatch(nodeID, expected: rest.count, actual: clip.tracks[index].valueKind.componentCount)
            }
            blend.add(values, weight: weight, additive: clip.blendMode == .additive)
        }
        var result = try blend.resolve(rest: rest, nodeID: nodeID)
        let single = try blendTracks(targetNode: nodeID) { property in
            if case .node(nodeID, .morphWeight) = property { true } else { false }
        }
        for property in single.order {
            guard case .node(_, .morphWeight(let target)) = property, let weights = single[property] else { continue }
            guard result.indices.contains(target) else {
                throw Failure.unanimatableProperty(property.description, "the mesh has \(result.count) morph targets")
            }
            guard let value = try weights.resolve(rest: .number(result[target])).floatValue else {
                throw Failure.valueKindMismatch(property.description)
            }
            result[target] = value
        }
        return result
    }

    private func index(of id: AnimationAction.ID) throws -> Int {
        guard let index = actions.firstIndex(where: { $0.id == id }) else { throw Failure.unknownAction(id) }
        return index
    }

    private mutating func startFadeIn(_ index: Int, duration: Double) {
        if !actions[index].isPlaying {
            actions[index].isPlaying = true
            actions[index].fadeFactor = 0
        }
        startFade(index, to: 1, duration: duration)
    }

    private mutating func startFadeOut(_ index: Int, duration: Double) {
        startFade(index, to: 0, duration: duration)
        if duration == 0 { actions[index].stop() }
    }

    private mutating func startFade(_ index: Int, to target: Float, duration: Double) {
        if duration == 0 {
            actions[index].fade = nil
            actions[index].fadeFactor = target
        } else {
            actions[index].fade = AnimationRamp(start: time, duration: duration, from: actions[index].fadeFactor, to: target)
        }
    }

    private mutating func startWarp(_ index: Int, from start: Float, to end: Float, duration: Double) {
        if duration == 0 {
            actions[index].warp = nil
            if end == 0 { actions[index].isPaused = true } else { actions[index].timeScale = end }
        } else {
            actions[index].warp = AnimationRamp(start: time, duration: duration, from: start, to: end)
        }
    }

    private static func validate(weight: Float) throws {
        guard weight.isFinite, weight >= 0 else { throw Failure.invalidWeight }
    }

    private static func validate(timeScale: Float) throws {
        guard timeScale.isFinite else { throw Failure.invalidTimeScale }
    }

    private static func validate(duration: Double) throws {
        guard duration.isFinite, duration >= 0 else { throw Failure.invalidDuration }
    }

    private static func validate(loop: AnimationLoopMode) throws {
        if let count = loop.passLimit, count < 1 { throw Failure.invalidLoopCount }
    }

    public enum Failure: LocalizedError, Equatable {
        case unknownAction(AnimationAction.ID)
        case notPlaying(AnimationAction.ID)
        case sameAction
        case invalidDeltaTime
        case invalidWeight
        case invalidTimeScale
        case invalidDuration
        case invalidTime
        case invalidLoopCount
        case warpRequiresDuration
        case missingNode(UUID)
        case invalidRestWeights(UUID)
        case morphWeightCountMismatch(UUID, expected: Int, actual: Int)
        case unanimatableProperty(String, String)
        case valueKindMismatch(String)
        case invalidPropertyValue(String)
        case missingRigLight(String)

        public var errorDescription: String? {
            switch self {
            case .unknownAction: "The animation mixer has no such action."
            case .notPlaying: "The animation action must be playing."
            case .sameAction: "A cross-fade needs two different actions."
            case .invalidDeltaTime: "The mixer time step must be finite and nonnegative."
            case .invalidWeight: "An action weight must be finite and nonnegative."
            case .invalidTimeScale: "An action time scale must be finite."
            case .invalidDuration: "A fade or warp duration must be finite and nonnegative."
            case .invalidTime: "An action time must be finite and within the clip duration."
            case .invalidLoopCount: "A loop count must be at least one."
            case .warpRequiresDuration: "Synchronizing a cross-fade requires clips with nonzero durations."
            case .missingNode(let id): "Animated node \(id) is missing from the scene graph."
            case .invalidRestWeights(let id): "Rest morph weights for node \(id) must be finite and nonempty."
            case .morphWeightCountMismatch(let id, let expected, let actual):
                "Node \(id) has \(expected) rest morph weights, but an action animates \(actual)."
            case .unanimatableProperty(let property, let reason): "Cannot animate \(property): \(reason)."
            case .valueKindMismatch(let property): "Animated values for \(property) do not have the property's kind."
            case .invalidPropertyValue(let property): "The blended value for \(property) is invalid."
            case .missingRigLight(let id): "The light rig has no light \"\(id)\"."
            }
        }
    }
}

/// Weighted per-property accumulation for one node's transform.
private struct TransformBlend {
    private var translation = VectorBlend()
    private var scale = VectorBlend()
    private var rotation = RotationBlend()

    mutating func add(_ pose: AnimationNodePose, weight: Double, mode: AnimationBlendMode) {
        let additive = mode == .additive
        if let value = pose.translation { translation.add(value, weight: weight, additive: additive) }
        if let value = pose.scale { scale.add(value, weight: weight, additive: additive) }
        if let value = pose.rotation { rotation.add(value, weight: weight, additive: additive) }
    }

    func resolve(rest: SceneTransform) -> SceneTransform {
        SceneTransform(
            translation: translation.resolve(rest: rest.translation),
            rotation: rotation.resolve(rest: rest.rotation),
            scale: scale.resolve(rest: rest.scale)
        )
    }
}

private struct VectorBlend {
    private var sum = SIMD3<Double>.zero
    private var weight = 0.0
    private var offset = SIMD3<Double>.zero
    private var hasOffset = false

    mutating func add(_ value: SIMD3<Float>, weight: Double, additive: Bool) {
        if additive {
            offset += SIMD3<Double>(value) * weight
            hasOffset = true
        } else {
            sum += SIMD3<Double>(value) * weight
            self.weight += weight
        }
    }

    func resolve(rest: SIMD3<Float>) -> SIMD3<Float> {
        guard weight > 0 || hasOffset else { return rest }
        let base = weight >= 1 ? sum / weight : sum + SIMD3<Double>(rest) * (1 - weight)
        return SIMD3<Float>(base + offset)
    }
}

struct RotationBlend {
    private var sum = SIMD4<Double>.zero
    private var reference: SIMD4<Double>?
    private var weight = 0.0
    private var offset = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
    private var hasOffset = false

    mutating func add(_ value: simd_quatf, weight: Double, additive: Bool) {
        let vector = SIMD4<Double>(value.vector)
        if additive {
            offset = offset * Self.power(simd_quatd(vector: vector), weight)
            hasOffset = true
            return
        }
        if let reference {
            sum += (simd_dot(vector, reference) < 0 ? -vector : vector) * weight
        } else {
            reference = vector
            sum += vector * weight
        }
        self.weight += weight
    }

    func resolve(rest: simd_quatf) -> simd_quatf {
        guard weight > 0 || hasOffset else { return rest }
        let restVector = SIMD4<Double>(rest.vector)
        var base = restVector
        if let reference, weight > 0 {
            // The first contribution has positive weight and every other term lies in
            // its hemisphere, so the sum's projection onto it is positive.
            var blended = sum
            if weight < 1 { blended += (simd_dot(restVector, reference) < 0 ? -restVector : restVector) * (1 - weight) }
            base = simd_normalize(blended)
        }
        let result = simd_normalize((simd_quatd(vector: base) * offset).vector)
        return simd_quatf(vector: SIMD4<Float>(result))
    }

    /// Raises a rotation to a real power along its shortest arc.
    private static func power(_ rotation: simd_quatd, _ exponent: Double) -> simd_quatd {
        var vector = simd_normalize(rotation.vector)
        if vector.w < 0 { vector = -vector }
        let imaginary = SIMD3(vector.x, vector.y, vector.z)
        let length = simd_length(imaginary)
        guard length > 0 else { return simd_quatd(ix: 0, iy: 0, iz: 0, r: 1) }
        let halfAngle = atan2(length, vector.w) * exponent
        return simd_quatd(real: cos(halfAngle), imag: imaginary / length * sin(halfAngle))
    }
}

private struct WeightBlend {
    private var sum: [Double]
    private var weight = 0.0
    private var offset: [Double]
    private var hasOffset = false

    init(count: Int) {
        sum = Array(repeating: 0, count: count)
        offset = Array(repeating: 0, count: count)
    }

    mutating func add(_ values: [Float], weight: Double, additive: Bool) {
        for index in values.indices {
            if additive {
                offset[index] += Double(values[index]) * weight
            } else {
                sum[index] += Double(values[index]) * weight
            }
        }
        if additive { hasOffset = true } else { self.weight += weight }
    }

    func resolve(rest: [Float], nodeID: UUID) throws -> [Float] {
        guard weight > 0 || hasOffset else { return rest }
        let result = rest.indices.map { index in
            let base = weight >= 1 ? sum[index] / weight : sum[index] + Double(rest[index]) * (1 - weight)
            return Float(base + offset[index])
        }
        guard result.allSatisfy(\.isFinite) else { throw AnimationMixer.Failure.invalidRestWeights(nodeID) }
        return result
    }
}
