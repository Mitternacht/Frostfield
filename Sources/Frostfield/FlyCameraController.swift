import Foundation
import simd

/// A free-flying camera with smoothed velocity, in the style of Three.js
/// `FlyControls` and first-person WASD navigation.
///
/// Coordinates are right-handed with +Y up, and the camera looks down its local −Z
/// axis. Yaw is a counterclockwise rotation about +Y seen from above; zero yaw looks
/// down −Z. Positive pitch looks up. Angles are radians, positions world units, and
/// speeds world units per second.
///
/// Movement input is a commanded direction in camera axes. The velocity approaches
/// the commanded velocity exponentially at ``accelerationRate`` and decays toward
/// rest at ``dampingRate`` without input. Position integrates that velocity exactly
/// over each step, so constant input gives the same result at any frame rate.
public struct FlyCameraController: Equatable, Sendable {
    public private(set) var position: SIMD3<Float>
    public private(set) var yaw: Float
    public private(set) var pitch: Float
    /// World-space velocity in units per second.
    public private(set) var velocity: SIMD3<Float>
    /// The speed full movement input approaches, in units per second.
    public let maximumSpeed: Float
    /// How quickly velocity approaches the commanded velocity, per second.
    public let accelerationRate: Float
    /// How quickly velocity decays without movement input, per second.
    public let dampingRate: Float
    /// Pitch range, within −pi/2...pi/2.
    public let pitchLimits: ClosedRange<Float>
    /// When false, ``handle(_:)`` and ``update(deltaTime:movement:)`` leave the camera where it is.
    /// ``look(by:)`` still turns it. Defaults to true.
    public var isEnabled = true

    public init(
        position: SIMD3<Float> = .zero,
        yaw: Float = 0,
        pitch: Float = 0,
        maximumSpeed: Float,
        accelerationRate: Float = 10,
        dampingRate: Float = 6,
        pitchLimits: ClosedRange<Float> = -Float.pi / 2...Float.pi / 2
    ) throws {
        guard flyFinite(position) else { throw Failure.invalidPosition }
        guard pitchLimits.lowerBound >= -Float.pi / 2, pitchLimits.upperBound <= Float.pi / 2 else {
            throw Failure.invalidPitchLimits
        }
        guard yaw.isFinite, pitchLimits.contains(pitch) else { throw Failure.invalidAngle }
        guard maximumSpeed.isFinite, maximumSpeed > 0 else { throw Failure.invalidSpeed }
        guard accelerationRate.isFinite, accelerationRate > 0, dampingRate.isFinite, dampingRate > 0 else {
            throw Failure.invalidRate
        }
        self.position = position
        self.yaw = yaw
        self.pitch = pitch
        velocity = .zero
        self.maximumSpeed = maximumSpeed
        self.accelerationRate = accelerationRate
        self.dampingRate = dampingRate
        self.pitchLimits = pitchLimits
    }

    /// The camera rotation: yaw about +Y, then pitch about the camera's X axis.
    public var rotation: simd_quatf {
        let yawRotation = simd_quatd(angle: Double(yaw), axis: SIMD3(0, 1, 0))
        let pitchRotation = simd_quatd(angle: Double(pitch), axis: SIMD3(1, 0, 0))
        let vector = (yawRotation * pitchRotation).vector
        return simd_quatf(vector: SIMD4(Float(vector.x), Float(vector.y), Float(vector.z), Float(vector.w)))
    }

    /// The camera's local-to-world transform, with unit scale.
    public var cameraTransform: SceneTransform {
        SceneTransform(translation: position, rotation: rotation)
    }

    /// Turns by yaw (x) and pitch (y) radians; positive x turns left and positive
    /// y looks up. For a drag with a top-left origin, pass `(−dx, −dy) × radiansPerPoint`.
    /// Yaw wraps to `−pi...pi` and pitch clamps to ``pitchLimits``.
    public mutating func look(by radians: SIMD2<Float>) throws {
        guard radians.x.isFinite, radians.y.isFinite else { throw Failure.invalidDelta }
        let turned = Double(yaw) + Double(radians.x)
        yaw = Float(turned - (turned / (2 * .pi)).rounded() * 2 * .pi)
        pitch = min(max(pitch + radians.y, pitchLimits.lowerBound), pitchLimits.upperBound)
    }

    /// Advances by a nonnegative, finite number of seconds. `movement` is in camera
    /// axes: x right, y up and z forward along the view direction, as from WASD with
    /// Q/E. Components are clamped to −1...1 and a longer vector is shortened to unit
    /// length, so diagonal input is no faster. The movement direction uses the
    /// current orientation.
    public mutating func update(deltaTime: Float, movement: SIMD3<Float> = .zero) throws {
        guard deltaTime.isFinite, deltaTime >= 0 else { throw Failure.invalidDeltaTime }
        guard flyFinite(movement) else { throw Failure.invalidMovement }
        guard isEnabled else { return }
        var input = flyDouble(simd_clamp(movement, SIMD3(repeating: -1), SIMD3(repeating: 1)))
        let length = simd_length(input)
        if length > 1 { input /= length }

        let (sinYaw, cosYaw) = (sin(Double(yaw)), cos(Double(yaw)))
        let (sinPitch, cosPitch) = (sin(Double(pitch)), cos(Double(pitch)))
        let right = SIMD3(cosYaw, 0, -sinYaw)
        let up = SIMD3(sinPitch * sinYaw, cosPitch, sinPitch * cosYaw)
        let forward = SIMD3(-cosPitch * sinYaw, sinPitch, -cosPitch * cosYaw)
        let commanded = (right * input.x + up * input.y + forward * input.z) * Double(maximumSpeed)

        let rate = Double(input == .zero ? dampingRate : accelerationRate)
        let seconds = Double(deltaTime)
        let initial = flyDouble(velocity)
        // v(t) = c + (v0 − c)e^(−rt), so the displacement is ct + (v0 − c)(1 − e^(−rt)) / r.
        let settled = -expm1(-rate * seconds)
        let displacement = commanded * seconds + (initial - commanded) * (settled / rate)
        let newVelocity = commanded + (initial - commanded) * (1 - settled)
        let newPosition = flyDouble(position) + displacement

        let positionResult = SIMD3(Float(newPosition.x), Float(newPosition.y), Float(newPosition.z))
        guard flyFinite(positionResult) else { throw Failure.unrepresentablePosition }
        position = positionResult
        // Stop exactly once coasting falls below a millionth of the maximum speed.
        velocity = input == .zero && simd_length(newVelocity) < Double(maximumSpeed) * 1e-6
            ? .zero
            : SIMD3(Float(newVelocity.x), Float(newVelocity.y), Float(newVelocity.z))
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidPosition
        case invalidAngle
        case invalidPitchLimits
        case invalidSpeed
        case invalidRate
        case invalidDelta
        case invalidMovement
        case invalidDeltaTime
        case unrepresentablePosition

        public var errorDescription: String? {
            switch self {
            case .invalidPosition: "A fly camera position must be finite."
            case .invalidAngle: "Fly camera yaw must be finite and pitch within the pitch limits."
            case .invalidPitchLimits: "Fly camera pitch limits must lie within ±pi/2."
            case .invalidSpeed: "The maximum speed must be positive and finite."
            case .invalidRate: "Acceleration and damping rates must be positive and finite."
            case .invalidDelta: "Look deltas must be finite."
            case .invalidMovement: "Movement input must be finite."
            case .invalidDeltaTime: "An update's elapsed time must be finite and nonnegative."
            case .unrepresentablePosition: "The camera moved outside the Float range."
            }
        }
    }
}

private func flyFinite(_ value: SIMD3<Float>) -> Bool {
    value.x.isFinite && value.y.isFinite && value.z.isFinite
}

private func flyDouble(_ value: SIMD3<Float>) -> SIMD3<Double> {
    SIMD3(Double(value.x), Double(value.y), Double(value.z))
}
