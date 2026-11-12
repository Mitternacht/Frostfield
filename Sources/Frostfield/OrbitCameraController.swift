import Foundation
import simd

/// Orbits a camera around a target point, in the style of Three.js `OrbitControls`
/// and, with ``PanMode/horizontal``, `MapControls`.
///
/// Coordinates are right-handed with +Y up, and the camera looks down its local −Z
/// axis at the target, matching ``SceneTransform`` and ``PerspectiveCamera``. Yaw is a
/// counterclockwise rotation about +Y seen from above; at zero yaw the camera sits on
/// the target's +Z side. Positive pitch raises the camera so it looks down. Angles are
/// radians and lengths world units.
///
/// Input takes abstract deltas, not gestures. Each input method changes ``goal``
/// immediately, within ``limits``. ``update(deltaTime:)`` then moves the presented
/// ``pose`` toward the goal with frame-rate independent exponential damping, or
/// jumps to it when ``dampingRate`` is nil. Read ``cameraTransform`` after updating.
public struct OrbitCameraController: Equatable, Sendable {
    /// The projection the controller's pan, zoom and framing must agree with.
    public enum Projection: Equatable, Sendable {
        /// Dolly changes the distance. The field of view is in radians, between zero and pi.
        case perspective(verticalFieldOfView: Float)
        /// Dolly changes ``Pose/orthographicHalfHeight``, the world-space half height
        /// of the view. Build the projection from it and the drawable aspect ratio.
        case orthographic
    }

    public enum PanMode: Equatable, Sendable {
        /// Moves the target in the camera's view plane, like screen-space `OrbitControls` panning.
        case viewPlane
        /// Moves the target horizontally, like `MapControls`. Vertical drags move it
        /// along the camera's forward direction projected onto the XZ plane.
        case horizontal
    }

    public enum AxisTransition: Equatable, Sendable {
        case immediate
        case damped
        /// Rotates around the orbit's yaw and pitch axes at a fixed camera
        /// angular speed. The default makes a quarter turn in 0.25 seconds.
        case constantSpeed(radiansPerSecond: Float = Float(2 * Double.pi))
    }

    /// Ranges that every goal pose satisfies. The controller validates them on creation.
    public struct Limits: Equatable, Sendable {
        /// Target-to-camera distance. Its bounds must be positive and finite.
        public var distance: ClosedRange<Float>
        /// Orthographic half height. Its bounds must be positive and finite.
        public var orthographicHalfHeight: ClosedRange<Float>
        /// Pitch, within −pi/2...pi/2.
        public var pitch: ClosedRange<Float>
        /// An optional finite yaw range. Without one, yaw wraps freely.
        public var yaw: ClosedRange<Float>?

        /// The defaults allow distances and half heights from 0.0001 to 1,000,000,
        /// the full pitch range, and free yaw.
        public init(
            distance: ClosedRange<Float> = 0.0001...1_000_000,
            orthographicHalfHeight: ClosedRange<Float> = 0.0001...1_000_000,
            pitch: ClosedRange<Float> = -Float.pi / 2...Float.pi / 2,
            yaw: ClosedRange<Float>? = nil
        ) {
            self.distance = distance
            self.orthographicHalfHeight = orthographicHalfHeight
            self.pitch = pitch
            self.yaw = yaw
        }

        var isValid: Bool {
            func positiveFinite(_ range: ClosedRange<Float>) -> Bool {
                range.lowerBound > 0 && range.upperBound.isFinite
            }
            let pitchFinite = pitch.lowerBound.isFinite && pitch.upperBound.isFinite
            let yawFinite = yaw.map { $0.lowerBound.isFinite && $0.upperBound.isFinite } ?? true
            return positiveFinite(distance) && positiveFinite(orthographicHalfHeight) && pitchFinite
                && pitch.lowerBound >= -Float.pi / 2 && pitch.upperBound <= Float.pi / 2 && yawFinite
        }
    }

    /// A camera placement around a target.
    public struct Pose: Equatable, Sendable {
        public internal(set) var target: SIMD3<Float>
        public internal(set) var distance: Float
        public internal(set) var yaw: Float
        public internal(set) var pitch: Float
        public internal(set) var orthographicHalfHeight: Float

        /// The camera rotation: yaw about +Y, then pitch about the camera's X axis.
        public var rotation: simd_quatf {
            let yawRotation = simd_quatd(angle: Double(yaw), axis: SIMD3(0, 1, 0))
            let pitchRotation = simd_quatd(angle: -Double(pitch), axis: SIMD3(1, 0, 0))
            let vector = (yawRotation * pitchRotation).vector
            return simd_quatf(vector: SIMD4(Float(vector.x), Float(vector.y), Float(vector.z), Float(vector.w)))
        }

        public var cameraPosition: SIMD3<Float> {
            let position = orbitDouble(target) + basis.back * Double(distance)
            return SIMD3(Float(position.x), Float(position.y), Float(position.z))
        }

        /// The camera's local-to-world transform, with unit scale.
        public var cameraTransform: SceneTransform {
            SceneTransform(translation: cameraPosition, rotation: rotation)
        }

        /// The camera's right, up and backward axes in world space.
        var basis: (right: SIMD3<Double>, up: SIMD3<Double>, back: SIMD3<Double>) {
            let (sinYaw, cosYaw) = (sin(Double(yaw)), cos(Double(yaw)))
            let (sinPitch, cosPitch) = (sin(Double(pitch)), cos(Double(pitch)))
            return (
                SIMD3(cosYaw, 0, -sinYaw),
                SIMD3(-sinPitch * sinYaw, cosPitch, -sinPitch * cosYaw),
                SIMD3(sinYaw * cosPitch, sinPitch, cosYaw * cosPitch)
            )
        }
    }

    /// Where input has sent the camera.
    public private(set) var goal: Pose
    /// Where the camera is presented, following ``goal`` as ``update(deltaTime:)`` runs.
    public private(set) var pose: Pose
    public let projection: Projection
    public let limits: Limits
    public let panMode: PanMode
    /// Nil presents input immediately. Otherwise the remaining gap between ``pose``
    /// and ``goal`` shrinks by `exp(−dampingRate × seconds)`, so results do not depend
    /// on the frame rate. Three.js's default damping at 60 Hz is about 3.1 per second.
    public let dampingRate: Float?
    /// When false, ``handle(_:)`` and ``update(deltaTime:)`` leave the pose where it is.
    /// Direct orbit, pan and dolly calls still edit the goal. Defaults to true.
    public var isEnabled = true
    /// When set, ``update(deltaTime:)`` moves from the pose at the latest goal change to the
    /// goal in this many seconds, and reaches it exactly. Nil keeps exponential damping,
    /// or a snap when ``dampingRate`` is also nil. Zero snaps on the next update.
    /// A new goal restarts the interval from the presented pose. This wins over damping.
    public var transitionDuration: Float? = nil
    /// The orbit target stays inside this box. Nil leaves the target free.
    public var targetBoundary: AxisAlignedBox? = nil
    /// Surfaces the camera cannot enter. ``collisionRadius`` is the clearance.
    public var colliders: [MeshBVH] = []
    /// How far the camera stays from ``colliders``. Zero disables the push.
    public var collisionRadius: Float = 0

    /// Pose, limits and collider count. Two meshes with the same count compare equal.
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.goal == rhs.goal && lhs.pose == rhs.pose && lhs.projection == rhs.projection
            && lhs.limits == rhs.limits && lhs.panMode == rhs.panMode && lhs.dampingRate == rhs.dampingRate
            && lhs.isEnabled == rhs.isEnabled && lhs.transitionDuration == rhs.transitionDuration
            && lhs.axisTurn == rhs.axisTurn && lhs.timedMove == rhs.timedMove && lhs.timedGoal == rhs.timedGoal
            && lhs.targetBoundary == rhs.targetBoundary && lhs.collisionRadius == rhs.collisionRadius
            && lhs.colliders.count == rhs.colliders.count
    }

    private struct AxisTurn: Equatable, Sendable {
        let start: SIMD2<Double>
        let delta: SIMD2<Double>
        let duration: Double
        var elapsed: Double = 0
    }

    private var axisTurn: AxisTurn?
    private var timedMove: TimedMove?
    private var timedGoal: Pose?

    private struct TimedMove: Equatable {
        var start: Pose
        var elapsed: Double
        let duration: Double
    }

    /// Creates a settled controller. The pose must lie within the limits.
    public init(
        target: SIMD3<Float> = .zero,
        distance: Float,
        yaw: Float = 0,
        pitch: Float = 0,
        projection: Projection,
        orthographicHalfHeight: Float = 1,
        limits: Limits = Limits(),
        panMode: PanMode = .viewPlane,
        dampingRate: Float? = nil
    ) throws {
        guard orbitFinite(target) else { throw Failure.invalidTarget }
        guard yaw.isFinite, pitch.isFinite else { throw Failure.invalidAngle }
        if case .perspective(let fieldOfView) = projection {
            guard fieldOfView.isFinite, fieldOfView > 0, fieldOfView < .pi else { throw Failure.invalidFieldOfView }
        }
        guard limits.isValid else { throw Failure.invalidLimits }
        guard limits.distance.contains(distance), limits.pitch.contains(pitch),
              limits.orthographicHalfHeight.contains(orthographicHalfHeight),
              limits.yaw?.contains(yaw) ?? true else { throw Failure.poseOutsideLimits }
        if let dampingRate {
            guard dampingRate.isFinite, dampingRate > 0 else { throw Failure.invalidDampingRate }
        }
        goal = Pose(
            target: target, distance: distance, yaw: yaw, pitch: pitch, orthographicHalfHeight: orthographicHalfHeight
        )
        pose = goal
        self.projection = projection
        self.limits = limits
        self.panMode = panMode
        self.dampingRate = dampingRate
    }

    /// The presented camera's local-to-world transform.
    public var cameraTransform: SceneTransform { pose.cameraTransform }

    /// Whether the presented pose has reached the goal.
    public var isSettled: Bool { axisTurn == nil && pose == goal }

    /// Adds yaw (x) and pitch (y) in radians. For content that follows a drag with
    /// a top-left origin, pass `(−dx, dy) × radiansPerPoint`.
    public mutating func rotate(by radians: SIMD2<Float>) throws {
        guard radians.x.isFinite, radians.y.isFinite else { throw Failure.invalidDelta }
        axisTurn = nil
        goal.pitch = clamp(goal.pitch + radians.y, to: limits.pitch)
        let yaw = Double(goal.yaw) + Double(radians.x)
        if let range = limits.yaw {
            goal.yaw = clamp(Float(yaw), to: range)
        } else {
            // Wrap the goal near zero and shift the presented yaw by the same turns,
            // so damping still follows the user's direction of rotation.
            let turns = (yaw / (2 * .pi)).rounded()
            goal.yaw = Float(yaw - turns * 2 * .pi)
            pose.yaw = Float(Double(pose.yaw) - turns * 2 * .pi)
        }
        try constrain()
    }

    /// Aligns the camera with a signed world axis, preserving the presented
    /// target, distance and orthographic half height. This supersedes pending
    /// orbit, pan and zoom input. Call `update(deltaTime:)` to animate with the
    /// controller's damping, or pass `animated: false` to snap immediately.
    /// Throws if the controller limits exclude the requested axis.
    public mutating func orient(to axis: AxisViewHelper.Axis, animated: Bool = true) throws {
        try orient(to: axis, transition: animated ? .damped : .immediate)
    }

    /// Aligns the camera with a signed world axis using the selected timing.
    /// Constant speed keeps the controller's no-roll yaw and pitch path. Orbit
    /// input switches subsequent updates back to the usual controller timing.
    public mutating func orient(to axis: AxisViewHelper.Axis, transition: AxisTransition) throws {
        let (axisYaw, axisPitch) = axis.orbitAngles
        guard limits.pitch.contains(axisPitch) else { throw Failure.poseOutsideLimits }
        let cycle = 2 * Double.pi
        var startingPose = pose
        let yaw: Float
        if let limit = limits.yaw {
            // Accept a cardinal angle expressed with Swift's Float pi constant.
            let tolerance = 1e-6
            let minimum = Double(limit.lowerBound) - tolerance
            let maximum = Double(limit.upperBound) + tolerance
            let firstTurn = ((minimum - axisYaw) / cycle).rounded(.up)
            let lastTurn = ((maximum - axisYaw) / cycle).rounded(.down)
            guard firstTurn <= lastTurn else { throw Failure.poseOutsideLimits }
            let nearest = ((Double(pose.yaw) - axisYaw) / cycle).rounded()
            let turn = min(max(nearest, firstTurn), lastTurn)
            yaw = clamp(Float(axisYaw + turn * cycle), to: limit)
            let difference = Double(yaw) - axisYaw
            guard abs(atan2(sin(difference), cos(difference))) <= tolerance else { throw Failure.poseOutsideLimits }
        } else {
            yaw = Float(axisYaw)
            let current = Double(pose.yaw)
            let wrapped = atan2(sin(current), cos(current))
            startingPose.yaw = Float(axisYaw + (wrapped - axisYaw).remainder(dividingBy: cycle))
        }
        var destination = startingPose
        destination.yaw = yaw
        destination.pitch = axisPitch
        guard orbitFinite(destination.cameraPosition) else { throw Failure.unrepresentablePose }

        let turn: AxisTurn?
        let startsInMotion: Bool
        switch transition {
        case .immediate:
            turn = nil
            startsInMotion = false
        case .damped:
            turn = nil
            startsInMotion = dampingRate != nil
        case .constantSpeed(let radiansPerSecond):
            guard radiansPerSecond.isFinite, radiansPerSecond > 0 else { throw Failure.invalidAngularSpeed }
            let start = SIMD2(Double(startingPose.yaw), Double(startingPose.pitch))
            let delta = SIMD2(Double(yaw), Double(axisPitch)) - start
            // World yaw and the rotated local pitch axis are perpendicular.
            // Their angular rates combine by length throughout this path.
            let duration = simd_length(delta) / Double(radiansPerSecond)
            startsInMotion = duration > 0
            turn = startsInMotion ? AxisTurn(start: start, delta: delta, duration: duration) : nil
        }
        goal = destination
        pose = startsInMotion ? startingPose : destination
        axisTurn = turn
        try constrain()
    }

    /// Moves the target so the scene follows a drag, with +x right and +y down in
    /// any unit, such as points. `viewportHeight` is the view's height in the same
    /// unit, so a drag of the full height moves the target by the visible height at
    /// the target's depth. Throws for a non-finite delta, a nonpositive height, or a
    /// target outside the `Float` range.
    public mutating func pan(by delta: SIMD2<Float>, viewportHeight: Float) throws {
        guard delta.x.isFinite, delta.y.isFinite else { throw Failure.invalidDelta }
        guard viewportHeight.isFinite, viewportHeight > 0 else { throw Failure.invalidViewportHeight }
        let worldPerUnit = visibleHalfHeight(goal) * 2 / Double(viewportHeight)
        let basis = goal.basis
        let vertical: SIMD3<Double>
        switch panMode {
        case .viewPlane:
            vertical = basis.up
        case .horizontal:
            // The camera's forward direction projected onto the XZ plane.
            vertical = SIMD3(basis.right.z, 0, -basis.right.x)
        }
        let offset = (vertical * Double(delta.y) - basis.right * Double(delta.x)) * worldPerUnit
        goal.target = try orbitFloat(orbitDouble(goal.target) + offset)
        axisTurn = nil
        try constrain()
    }

    /// Divides the distance, or the orthographic half height, by a positive finite
    /// factor within the limits. A factor above one moves closer, like a pinch-out
    /// magnification.
    public mutating func dolly(by factor: Float) throws {
        guard factor.isFinite, factor > 0 else { throw Failure.invalidFactor }
        switch projection {
        case .perspective:
            goal.distance = clamp(Float(Double(goal.distance) / Double(factor)), to: limits.distance)
        case .orthographic:
            goal.orthographicHalfHeight = clamp(
                Float(Double(goal.orthographicHalfHeight) / Double(factor)), to: limits.orthographicHalfHeight
            )
        }
        axisTurn = nil
        try constrain()
    }

    /// Dollies like ``dolly(by:)`` while keeping whatever lies under a normalized
    /// view point fixed on screen once the pose settles. (0, 0) is the top-left and
    /// (1, 1) the bottom-right, as in ``PerspectiveCamera/ray(at:aspectRatio:cameraTransform:)``.
    /// A perspective camera moves along the ray through the point, so every depth
    /// along it stays put. The aspect ratio is the drawable width over its height.
    public mutating func zoom(by factor: Float, toward point: SIMD2<Float>, aspectRatio: Float) throws {
        guard factor.isFinite, factor > 0 else { throw Failure.invalidFactor }
        guard (0...1).contains(point.x), (0...1).contains(point.y) else { throw Failure.invalidNormalizedPoint }
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw Failure.invalidAspectRatio }
        let before = goal
        let pendingTurn = axisTurn
        try dolly(by: factor)
        // The view-plane offset of the point at unit depth (perspective) or unit half height.
        let screen = SIMD2(Double(2 * point.x - 1) * Double(aspectRatio), Double(1 - 2 * point.y))
        let shrink: Double
        let reach: Double
        switch projection {
        case .perspective(let fieldOfView):
            reach = tan(Double(fieldOfView) / 2)
            shrink = Double(before.distance) - Double(goal.distance)
        case .orthographic:
            reach = 1
            shrink = Double(before.orthographicHalfHeight) - Double(goal.orthographicHalfHeight)
        }
        let basis = before.basis
        let offset = (basis.right * screen.x + basis.up * screen.y) * reach * shrink
        do {
            goal.target = try orbitFloat(orbitDouble(before.target) + offset)
            try constrain()
        } catch {
            goal = before
            axisTurn = pendingTurn
            throw error
        }
    }

    /// Places the goal so the camera sits at `eye` and looks at `target`.
    /// Yaw and pitch follow the shortest orbit that puts the target on the view axis.
    /// The orthographic half height is unchanged. Call ``update(deltaTime:)`` to present it.
    public mutating func look(at target: SIMD3<Float>, from eye: SIMD3<Float>) throws {
        guard orbitFinite(target), orbitFinite(eye) else { throw Failure.invalidTarget }
        let offset = orbitDouble(eye) - orbitDouble(target)
        let distance = simd_length(offset)
        guard distance.isFinite, distance > 0 else { throw Failure.invalidTarget }
        let back = offset / distance
        let pitch = Float(asin(min(1, max(-1, back.y))))
        let horizontal = cos(Double(pitch))
        let yaw = abs(horizontal) < 1e-6 ? goal.yaw : Float(atan2(back.x, back.z))
        guard limits.distance.contains(Float(distance)), limits.pitch.contains(pitch),
              limits.yaw?.contains(yaw) ?? true else { throw Failure.poseOutsideLimits }
        axisTurn = nil
        goal.target = target
        goal.distance = Float(distance)
        goal.yaw = yaw
        goal.pitch = pitch
        try constrain()
    }

    /// Centers the target on a box and fits its bounding sphere in the view, keeping
    /// yaw and pitch. A perspective camera fits the sphere within the narrower of the
    /// vertical and horizontal fields of view. An orthographic camera sets the half
    /// height and backs away to at least twice the radius, so near planes up to one
    /// radius from the camera keep the box visible. Distances and half heights are
    /// clamped to the limits, which can leave a large box only partly in view.
    /// `padding` is the fraction of the view kept clear around the sphere, from 0 up to
    /// but not including 1. Zero matches a tight fit.
    public mutating func frame(_ bounds: AxisAlignedBox, aspectRatio: Float, padding: Float = 0) throws {
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw Failure.invalidAspectRatio }
        guard padding.isFinite, (0..<1).contains(padding) else { throw Failure.invalidPadding }
        let sphere = bounds.boundingSphere
        axisTurn = nil
        let radius = Double(sphere.radius)
        let margin = 1 - Double(padding)
        goal.target = sphere.center
        switch projection {
        case .perspective(let fieldOfView):
            let halfVertical = Double(fieldOfView) / 2
            let halfHorizontal = atan(tan(halfVertical) * Double(aspectRatio))
            let distance = radius / sin(min(halfVertical, halfHorizontal)) / margin
            goal.distance = clamp(Float(distance), to: limits.distance)
        case .orthographic:
            let halfHeight = radius / min(1, Double(aspectRatio)) / margin
            goal.orthographicHalfHeight = clamp(Float(halfHeight), to: limits.orthographicHalfHeight)
            goal.distance = clamp(max(goal.distance, Float(2 * radius)), to: limits.distance)
        }
        try constrain()
    }

    /// Advances the presented pose by a nonnegative, finite number of seconds.
    /// A timed axis turn ends at its duration. Damping snaps to the goal once
    /// every remaining difference is below 1e-5 radians or 1e-5 of the visible half height.
    public mutating func update(deltaTime: Float) throws {
        guard deltaTime.isFinite, deltaTime >= 0 else { throw Failure.invalidDeltaTime }
        guard isEnabled else { return }
        if let duration = transitionDuration {
            guard duration.isFinite, duration >= 0 else { throw Failure.invalidTransitionDuration }
            if duration == 0 || pose == goal {
                pose = goal
                timedMove = nil
                timedGoal = nil
                try constrain()
                return
            }
            if timedMove == nil || timedGoal != goal {
                timedMove = TimedMove(start: pose, elapsed: 0, duration: Double(duration))
                timedGoal = goal
            }
            timedMove?.elapsed += Double(deltaTime)
            let elapsed = timedMove?.elapsed ?? 0
            let span = timedMove?.duration ?? 1
            let t = min(1, elapsed / span)
            if let start = timedMove?.start {
                pose = blend(start, toward: goal, by: t * t * (3 - 2 * t))
            }
            if t >= 1 {
                pose = goal
                timedMove = nil
            }
            try constrain()
            return
        }
        timedMove = nil
        timedGoal = nil
        if var turn = axisTurn {
            turn.elapsed += Double(deltaTime)
            if turn.elapsed >= turn.duration {
                settle()
            } else {
                let angles = turn.start + turn.delta * (turn.elapsed / turn.duration)
                pose.yaw = Float(angles.x)
                pose.pitch = Float(angles.y)
                axisTurn = turn
            }
            try constrain()
            return
        }
        guard let dampingRate else {
            pose = goal
            try constrain()
            return
        }
        let remaining = exp(-Double(dampingRate) * Double(deltaTime))
        func approach(_ current: Float, _ goal: Float) -> Float {
            Float(Double(goal) + (Double(current) - Double(goal)) * remaining)
        }
        let target = orbitDouble(goal.target) + (orbitDouble(pose.target) - orbitDouble(goal.target)) * remaining
        pose.target = SIMD3(Float(target.x), Float(target.y), Float(target.z))
        pose.distance = approach(pose.distance, goal.distance)
        pose.yaw = approach(pose.yaw, goal.yaw)
        pose.pitch = approach(pose.pitch, goal.pitch)
        pose.orthographicHalfHeight = approach(pose.orthographicHalfHeight, goal.orthographicHalfHeight)

        let tolerance = 1e-5
        let length = visibleHalfHeight(goal) * tolerance
        let near = simd_length(orbitDouble(pose.target) - orbitDouble(goal.target)) <= length
            && abs(Double(pose.distance) - Double(goal.distance)) <= length
            && abs(Double(pose.orthographicHalfHeight) - Double(goal.orthographicHalfHeight))
                <= Double(goal.orthographicHalfHeight) * tolerance
            && abs(Double(pose.yaw) - Double(goal.yaw)) <= tolerance
            && abs(Double(pose.pitch) - Double(goal.pitch)) <= tolerance
        if near { pose = goal }
        try constrain()
    }

    /// Presents the goal immediately, skipping any remaining transition.
    public mutating func settle() {
        pose = goal
        axisTurn = nil
        timedMove = nil
        timedGoal = nil
    }

    private func blend(_ start: Pose, toward goal: Pose, by amount: Double) -> Pose {
        func mix(_ current: Float, _ goal: Float) -> Float {
            Float(Double(current) + (Double(goal) - Double(current)) * amount)
        }
        var yawDelta = Double(goal.yaw) - Double(start.yaw)
        if yawDelta > .pi { yawDelta -= 2 * .pi }
        if yawDelta < -.pi { yawDelta += 2 * .pi }
        let target = orbitDouble(start.target) + (orbitDouble(goal.target) - orbitDouble(start.target)) * amount
        var presented = start
        presented.target = SIMD3(Float(target.x), Float(target.y), Float(target.z))
        presented.distance = mix(start.distance, goal.distance)
        presented.yaw = Float(Double(start.yaw) + yawDelta * amount)
        presented.pitch = mix(start.pitch, goal.pitch)
        presented.orthographicHalfHeight = mix(start.orthographicHalfHeight, goal.orthographicHalfHeight)
        return presented
    }

    private func visibleHalfHeight(_ pose: Pose) -> Double {
        switch projection {
        case .perspective(let fieldOfView): Double(pose.distance) * tan(Double(fieldOfView) / 2)
        case .orthographic: Double(pose.orthographicHalfHeight)
        }
    }

    private mutating func constrain() throws {
        if let boundary = targetBoundary {
            goal.target = simd_min(boundary.maximum, simd_max(boundary.minimum, goal.target))
            pose.target = simd_min(boundary.maximum, simd_max(boundary.minimum, pose.target))
        }
        guard collisionRadius.isFinite, collisionRadius >= 0 else { throw Failure.invalidCollisionRadius }
        guard collisionRadius > 0, !colliders.isEmpty else { return }
        try push(outOf: colliders, pose: &goal)
        try push(outOf: colliders, pose: &pose)
    }

    private func push(outOf colliders: [MeshBVH], pose: inout Pose) throws {
        var position = pose.cameraPosition
        for collider in colliders {
            guard let hit = try collider.closestPoint(to: position), hit.distance < Double(collisionRadius) else { continue }
            let away = SIMD3<Double>(position) - SIMD3<Double>(hit.position)
            let length = simd_length(away)
            let direction = length > 1e-8 ? away / length : SIMD3<Double>(0, 0, 1)
            let pushed = SIMD3<Double>(hit.position) + direction * Double(collisionRadius)
            position = SIMD3(Float(pushed.x), Float(pushed.y), Float(pushed.z))
        }
        let offset = orbitDouble(position) - orbitDouble(pose.target)
        let separation = simd_length(offset)
        guard separation > 1e-8 else { return }
        pose.distance = clamp(Float(separation), to: limits.distance)
        pose.pitch = clamp(Float(asin(min(max(offset.y / separation, -1), 1))), to: limits.pitch)
        let yaw = Float(atan2(offset.x, offset.z))
        pose.yaw = limits.yaw.map { clamp(yaw, to: $0) } ?? yaw
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidTarget
        case invalidAngle
        case invalidFieldOfView
        case invalidLimits
        case poseOutsideLimits
        case invalidDampingRate
        case invalidAngularSpeed
        case invalidDelta
        case invalidViewportHeight
        case invalidFactor
        case invalidNormalizedPoint
        case invalidAspectRatio
        case invalidDeltaTime
        case invalidTransitionDuration
        case invalidPadding
        case invalidCollisionRadius
        case unrepresentablePose

        public var errorDescription: String? {
            switch self {
            case .invalidTarget: "An orbit target must be finite."
            case .invalidAngle: "Orbit yaw and pitch must be finite."
            case .invalidFieldOfView: "The vertical field of view must be between zero and pi radians."
            case .invalidLimits:
                "Orbit limits need positive finite distance and half-height bounds, pitch within ±pi/2, and a finite yaw range."
            case .poseOutsideLimits: "The orbit distance, half height, pitch and yaw must lie within the limits."
            case .invalidDampingRate: "A damping rate must be positive and finite."
            case .invalidAngularSpeed: "An axis turn's angular speed must be positive and finite."
            case .invalidDelta: "Camera input deltas must be finite."
            case .invalidViewportHeight: "The viewport height must be positive and finite."
            case .invalidFactor: "A dolly or zoom factor must be positive and finite."
            case .invalidNormalizedPoint: "A zoom point must lie inside the normalized drawable."
            case .invalidAspectRatio: "The drawable aspect ratio must be positive and finite."
            case .invalidDeltaTime: "An update's elapsed time must be finite and nonnegative."
            case .invalidTransitionDuration: "A transition duration must be nonnegative and finite."
            case .invalidPadding: "Frame padding must be finite and in 0 up to, but not including, 1."
            case .invalidCollisionRadius: "The camera collision radius must be finite and nonnegative."
            case .unrepresentablePose: "The camera input moves the target outside the Float range."
            }
        }
    }
}

private func clamp(_ value: Float, to range: ClosedRange<Float>) -> Float {
    min(max(value, range.lowerBound), range.upperBound)
}

private func orbitFinite(_ value: SIMD3<Float>) -> Bool {
    value.x.isFinite && value.y.isFinite && value.z.isFinite
}

private func orbitDouble(_ value: SIMD3<Float>) -> SIMD3<Double> {
    SIMD3(Double(value.x), Double(value.y), Double(value.z))
}

private func orbitFloat(_ value: SIMD3<Double>) throws -> SIMD3<Float> {
    let result = SIMD3(Float(value.x), Float(value.y), Float(value.z))
    guard orbitFinite(result) else { throw OrbitCameraController.Failure.unrepresentablePose }
    return result
}
