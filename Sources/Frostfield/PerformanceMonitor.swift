import Foundation

/// Steps a quality factor from measured frame times, in the role of drei
/// `PerformanceMonitor` and `AdaptiveDpr`.
///
/// Every `iterations` frames, the average frame rate is compared with the bounds.
/// A rate below the lower bound decreases `factor` by `step`. A rate above the upper
/// bound increases it. The first move does not count as a reversal. After
/// `maximumReversals` direction changes, the factor stays where it is.
/// `pixelScale(minimum:maximum:)` maps the factor onto a drawable scale.
/// The app records each frame; this type does not subscribe to a display link.
public struct PerformanceMonitor: Equatable, Sendable {
    public let lowerFramesPerSecond: Float
    public let upperFramesPerSecond: Float
    public let iterations: Int
    public let step: Float
    public let maximumReversals: Int?
    public private(set) var factor: Float
    public private(set) var fellBack = false

    public enum Event: Equatable, Sendable {
        case incline(factor: Float)
        case decline(factor: Float)
    }

    public enum Failure: LocalizedError, Equatable {
        case bounds
        case iterations
        case step
        case reversals
        case factor
        case frameTime
        case pixelScale

        public var errorDescription: String? {
            switch self {
            case .bounds: "The lower frame rate must be below the upper frame rate, and both must be finite and positive."
            case .iterations: "A performance sample needs from 1 to 240 frames."
            case .step: "The quality step must be finite and from 0 to 1, excluding 0."
            case .reversals: "The reversal limit must be at least 1."
            case .factor: "The starting quality factor must be finite and from 0 to 1."
            case .frameTime: "A frame time must be finite and greater than zero."
            case .pixelScale: "The pixel-scale bounds must be finite, and the minimum must be from 0 to the maximum."
            }
        }
    }

    public init(
        lowerFramesPerSecond: Float = 40,
        upperFramesPerSecond: Float = 60,
        iterations: Int = 10,
        step: Float = 0.1,
        maximumReversals: Int? = nil,
        factor: Float = 1
    ) throws {
        guard lowerFramesPerSecond.isFinite, upperFramesPerSecond.isFinite, lowerFramesPerSecond > 0,
              upperFramesPerSecond > lowerFramesPerSecond else { throw Failure.bounds }
        guard (1...240).contains(iterations) else { throw Failure.iterations }
        guard step.isFinite, step > 0, step <= 1 else { throw Failure.step }
        if let maximumReversals, maximumReversals < 1 { throw Failure.reversals }
        guard factor.isFinite, (0...1).contains(factor) else { throw Failure.factor }
        self.lowerFramesPerSecond = lowerFramesPerSecond
        self.upperFramesPerSecond = upperFramesPerSecond
        self.iterations = iterations
        self.step = step
        self.maximumReversals = maximumReversals
        self.factor = factor
    }

    /// Records one frame. Returns an event when a completed sample changes the factor.
    public mutating func record(frameTime: Float) throws -> Event? {
        guard frameTime.isFinite, frameTime > 0 else { throw Failure.frameTime }
        guard !fellBack else { return nil }
        accumulatedTime += frameTime
        recordedFrames += 1
        guard recordedFrames == iterations else { return nil }
        let framesPerSecond = Float(iterations) / accumulatedTime
        accumulatedTime = 0
        recordedFrames = 0
        let direction: Int
        if framesPerSecond < lowerFramesPerSecond {
            direction = -1
        } else if framesPerSecond > upperFramesPerSecond {
            direction = 1
        } else {
            previousDirection = 0
            return nil
        }
        if previousDirection != 0, direction != previousDirection {
            reversals += 1
            if let maximumReversals, reversals >= maximumReversals {
                fellBack = true
                previousDirection = direction
                return nil
            }
        }
        previousDirection = direction
        let next = min(1, max(0, factor + Float(direction) * step))
        guard next != factor else { return nil }
        factor = next
        return direction > 0 ? .incline(factor: next) : .decline(factor: next)
    }

    /// The drawable scale for the current factor. `minimum` is factor 0 and `maximum` is factor 1.
    public func pixelScale(minimum: Float, maximum: Float) throws -> Float {
        guard minimum.isFinite, maximum.isFinite, minimum >= 0, maximum >= minimum else { throw Failure.pixelScale }
        return minimum + (maximum - minimum) * factor
    }

    private var accumulatedTime: Float = 0
    private var recordedFrames = 0
    private var previousDirection = 0
    private var reversals = 0
}
