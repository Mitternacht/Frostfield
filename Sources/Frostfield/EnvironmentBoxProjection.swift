import Foundation
import simd

/// A world-aligned proxy box and the world position that captured its environment.
/// Specular rays from inside the box sample toward their exit point from the probe.
public struct EnvironmentBoxProjection: Equatable, Sendable {
    public let bounds: AxisAlignedBox
    public let capturePosition: SIMD3<Float>

    /// The box needs finite, positive normal extents, with the capture position
    /// strictly inside. Use the same world coordinates as the receiving meshes.
    public init(bounds: AxisAlignedBox, capturePosition: SIMD3<Float>) throws {
        let size = bounds.size
        guard size.x.isNormal, size.y.isNormal, size.z.isNormal else { throw Failure.bounds }
        guard all(capturePosition .> bounds.minimum), all(capturePosition .< bounds.maximum) else {
            throw Failure.capturePosition
        }
        self.bounds = bounds
        self.capturePosition = capturePosition
    }

    /// Returns a unit world direction before the environment's rotation.
    /// Points outside the box use the original direction. Parallel axes do not
    /// restrict the exit distance, including for a ray on the side of a box.
    public func direction(from position: SIMD3<Float>, along direction: SIMD3<Float>) throws -> SIMD3<Float> {
        guard position.isFiniteVector, direction.isFiniteVector, direction != .zero else { throw Failure.ray }
        let ray = simd_normalize(SIMD3<Double>(direction))
        guard bounds.contains(position) else { return SIMD3<Float>(ray) }
        let start = SIMD3<Double>(position)
        var exit = Double.infinity
        var hitAxis = 0
        for axis in 0..<3 where ray[axis] != 0 {
            let boundary = ray[axis] > 0 ? bounds.maximum[axis] : bounds.minimum[axis]
            let distance = (Double(boundary) - start[axis]) / ray[axis]
            if distance < exit {
                exit = distance
                hitAxis = axis
            }
        }
        var corrected = (start - SIMD3<Double>(capturePosition)) + ray * exit
        // Pin the exit coordinate to its wall to retain small capture offsets.
        let boundary = ray[hitAxis] > 0 ? bounds.maximum[hitAxis] : bounds.minimum[hitAxis]
        corrected[hitAxis] = Double(boundary) - Double(capturePosition[hitAxis])
        return SIMD3<Float>(simd_normalize(corrected))
    }

    public enum Failure: LocalizedError, Equatable {
        case bounds, capturePosition, ray

        public var errorDescription: String? {
            switch self {
            case .bounds: "Use finite, positive normal extents for the environment projection box."
            case .capturePosition: "Place the environment capture strictly inside the projection box."
            case .ray: "Use a finite reflection origin and a finite, nonzero direction."
            }
        }
    }
}
