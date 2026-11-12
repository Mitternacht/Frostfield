import simd

/// A pointer or keyboard sample for ``OrbitCameraController`` and ``FlyCameraController``.
///
/// Deltas are in the controller's own units: radians for a pointer, a positive
/// magnification factor's logarithm for a scroll, and camera-axis direction for
/// movement. The caller converts pixels and key state before constructing the event.
public struct CameraControlEvent: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        case began
        case changed
        case ended
    }

    public enum Kind: Equatable, Sendable {
        /// A pointer drag. Only ``Phase/changed`` moves the camera.
        case pointer(phase: Phase, delta: SIMD2<Float>)
        /// A scroll wheel or trackpad delta. Orbit uses `y` as the natural log of a dolly factor.
        case scroll(delta: SIMD2<Float>)
        /// Camera-space movement for one step: x right, y up, z forward. Orbit ignores it.
        case movement(direction: SIMD3<Float>, deltaTime: Float)
    }

    public var kind: Kind

    public init(_ kind: Kind) {
        self.kind = kind
    }

    func validate() throws {
        switch kind {
        case .pointer(_, let delta), .scroll(let delta):
            guard delta.x.isFinite, delta.y.isFinite else { throw OrbitCameraController.Failure.invalidDelta }
        case .movement(let direction, let deltaTime):
            guard direction.x.isFinite, direction.y.isFinite, direction.z.isFinite else {
                throw FlyCameraController.Failure.invalidMovement
            }
            guard deltaTime.isFinite, deltaTime >= 0 else { throw FlyCameraController.Failure.invalidDeltaTime }
        }
    }
}

/// The shared controls surface: an enabled flag and one event entry point.
///
/// A disabled controller accepts a well-formed event and then does nothing.
/// A nonfinite delta still throws, so a bad event is not mistaken for a disabled one.
public protocol CameraControls {
    var isEnabled: Bool { get set }
    mutating func handle(_ event: CameraControlEvent) throws
}

extension OrbitCameraController: CameraControls {
    public mutating func handle(_ event: CameraControlEvent) throws {
        try event.validate()
        guard isEnabled else { return }
        switch event.kind {
        case .pointer(.changed, let delta):
            try rotate(by: delta)
        case .pointer:
            break
        case .scroll(let delta):
            try dolly(by: exp(delta.y))
        case .movement:
            break
        }
    }
}

extension FlyCameraController: CameraControls {
    public mutating func handle(_ event: CameraControlEvent) throws {
        try event.validate()
        guard isEnabled else { return }
        switch event.kind {
        case .pointer(.changed, let delta):
            try look(by: delta)
        case .pointer:
            break
        case .scroll(let delta):
            try look(by: delta)
        case .movement(let direction, let deltaTime):
            try update(deltaTime: deltaTime, movement: direction)
        }
    }
}
