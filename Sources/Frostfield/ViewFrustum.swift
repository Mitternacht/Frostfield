import Foundation
import simd

/// The world-space region that a world-to-clip matrix maps into Metal's clip
/// volume: −w ≤ x ≤ w, −w ≤ y ≤ w and 0 ≤ z ≤ w.
///
/// Forward and reversed depth both map into `0...w` and differ only in which
/// end is near, so one extraction serves either convention. Perspective,
/// infinite-far and orthographic projections are supported. Box and sphere tests
/// are conservative: an object outside the frustum but near one of its corners
/// or edges can report an intersection, never the reverse.
public struct ViewFrustum: Equatable, Sendable {
    /// A half-space bounded by a plane.
    public struct Plane: Equatable, Sendable {
        /// The unit normal, pointing into the frustum.
        public let normal: SIMD3<Float>
        /// A point is inside when `dot(normal, point) + offset` is nonnegative.
        public let offset: Float

        /// The signed distance from the plane in world units, positive inside.
        public func signedDistance(to point: SIMD3<Float>) -> Float {
            simd_dot(normal, point) + offset
        }
    }

    /// How a volume relates to the frustum.
    public enum Containment: Equatable, Sendable {
        case outside
        case intersecting
        case inside
    }

    /// The left, right, bottom and top planes, then `z ≥ 0` and `z ≤ w`. A depth
    /// plane that contains all of space, such as an infinite far plane, is omitted.
    public let planes: [Plane]
    // Double copies of the normalized planes keep tests independent of Float rounding.
    private let equations: [SIMD4<Double>]

    /// Extracts the planes of a Metal clip-from-world matrix, such as a projection
    /// times the inverse camera transform. Throws for a non-finite matrix or
    /// one whose clip volume has a plane with no orientation.
    public init(clipFromWorld: simd_float4x4) throws {
        let (c0, c1, c2, c3) = clipFromWorld.columns
        func row(_ index: Int) -> SIMD4<Double> {
            SIMD4(Double(c0[index]), Double(c1[index]), Double(c2[index]), Double(c3[index]))
        }
        let rows = (0..<4).map(row)
        guard rows.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) else {
            throw Failure.nonFiniteMatrix
        }
        let (x, y, z, w) = (rows[0], rows[1], rows[2], rows[3])

        var equations: [SIMD4<Double>] = []
        for (index, raw) in [w + x, w - x, w + y, w - y, z, w - z].enumerated() {
            let length = simd_length(SIMD3(raw.x, raw.y, raw.z))
            guard length > 0 else {
                // A depth plane with no normal and a positive offset holds everywhere.
                guard index >= 4, raw.w > 0 else { throw Failure.degenerateMatrix }
                continue
            }
            let plane = raw / length
            guard plane.x.isFinite, plane.y.isFinite, plane.z.isFinite, plane.w.isFinite else {
                throw Failure.degenerateMatrix
            }
            equations.append(plane)
        }
        self.equations = equations
        planes = equations.map {
            Plane(normal: SIMD3(Float($0.x), Float($0.y), Float($0.z)), offset: Float($0.w))
        }
    }

    /// Whether the closed frustum contains the point.
    public func contains(point: SIMD3<Float>) -> Bool {
        let point = SIMD4(boundsDouble(point), 1)
        return equations.allSatisfy { simd_dot($0, point) >= 0 }
    }

    /// Classifies a world-space box. `.inside` means the whole box is inside;
    /// `.outside` means it lies entirely behind one plane.
    public func classify(box: AxisAlignedBox) -> Containment {
        let lower = boundsDouble(box.minimum)
        let upper = boundsDouble(box.maximum)
        var straddles = false
        for plane in equations {
            let normal = SIMD3(plane.x, plane.y, plane.z)
            let positive = SIMD3<Double>(
                normal.x >= 0 ? upper.x : lower.x, normal.y >= 0 ? upper.y : lower.y, normal.z >= 0 ? upper.z : lower.z
            )
            if simd_dot(normal, positive) + plane.w < 0 { return .outside }
            let negative = SIMD3<Double>(
                normal.x >= 0 ? lower.x : upper.x, normal.y >= 0 ? lower.y : upper.y, normal.z >= 0 ? lower.z : upper.z
            )
            if simd_dot(normal, negative) + plane.w < 0 { straddles = true }
        }
        return straddles ? .intersecting : .inside
    }

    /// Whether any part of a world-space box may be visible.
    public func intersects(box: AxisAlignedBox) -> Bool {
        classify(box: box) != .outside
    }

    /// Whether any part of a world-space sphere may be visible.
    public func intersects(sphere: BoundingSphere) -> Bool {
        let center = boundsDouble(sphere.center)
        let radius = Double(sphere.radius)
        return equations.allSatisfy { simd_dot(SIMD3($0.x, $0.y, $0.z), center) + $0.w >= -radius }
    }

    public enum Failure: LocalizedError, Equatable {
        case nonFiniteMatrix
        case degenerateMatrix

        public var errorDescription: String? {
            switch self {
            case .nonFiniteMatrix: "A view frustum needs a finite clip-from-world matrix."
            case .degenerateMatrix: "The clip-from-world matrix does not bound a view volume."
            }
        }
    }
}
