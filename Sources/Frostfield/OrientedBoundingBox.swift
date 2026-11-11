import Foundation
import simd

/// A closed box with an arbitrary orientation, like Three.js `OBB`.
public struct OrientedBoundingBox: Equatable, Sendable {
    public let center: SIMD3<Float>
    /// Nonnegative half extents in box-local coordinates. Zero flattens that axis.
    public let halfSize: SIMD3<Float>
    /// A unit quaternion that rotates box-local axes into world axes.
    public let rotation: simd_quatf

    public init(
        center: SIMD3<Float>, halfSize: SIMD3<Float>,
        rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    ) throws {
        guard boundsFinite(center) else { throw Failure.invalidPoint }
        guard boundsFinite(halfSize), all(halfSize .>= .zero) else { throw Failure.invalidHalfSize }
        let length = simd_length_squared(rotation.vector)
        guard abs(length - 1) <= 1e-4 else { throw Failure.invalidRotation }
        self.init(validatedCenter: center, halfSize: halfSize, rotation: simd_normalize(rotation))
    }

    private init(validatedCenter center: SIMD3<Float>, halfSize: SIMD3<Float>, rotation: simd_quatf) {
        self.center = center
        self.halfSize = halfSize
        self.rotation = rotation
    }

    /// Uses the same axes and encloses the original box even when its midpoint rounds.
    public init(_ box: AxisAlignedBox) {
        center = SIMD3<Float>((boundsDouble(box.minimum) + boundsDouble(box.maximum)) * 0.5)
        let midpoint = boundsDouble(center)
        halfSize = boundsFloatAbove(simd_max(boundsDouble(box.maximum) - midpoint, midpoint - boundsDouble(box.minimum)))
        rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }

    /// The extent along each box-local axis.
    public var size: SIMD3<Float> { halfSize * 2 }

    /// Moves the center by a world-space offset. The result must stay finite.
    public func translated(by offset: SIMD3<Float>) throws -> OrientedBoundingBox {
        let movedCenter = center + offset
        guard boundsFinite(movedCenter) else { throw Failure.invalidPoint }
        return OrientedBoundingBox(validatedCenter: movedCenter, halfSize: halfSize, rotation: rotation)
    }

    /// The axis-aligned enclosure, rounded outward to Float.
    public func axisAlignedBounds() throws -> AxisAlignedBox {
        let bounds = exactBounds
        return try AxisAlignedBox(minimum: boundsFloatBelow(bounds.minimum), maximum: boundsFloatAbove(bounds.maximum))
    }

    public func contains(_ point: SIMD3<Float>) -> Bool {
        let local = basis.transpose * (boundsDouble(point) - boundsDouble(center))
        return all(simd_abs(local) .<= boundsDouble(halfSize))
    }

    /// Clamps a finite world-space point to this closed box.
    public func closestPoint(to point: SIMD3<Float>) throws -> SIMD3<Float> {
        guard boundsFinite(point) else { throw Failure.invalidPoint }
        let axes = basis
        let origin = boundsDouble(center)
        let extent = boundsDouble(halfSize)
        let local = axes.transpose * (boundsDouble(point) - origin)
        let clamped = simd_clamp(local, -extent, extent)
        let result = SIMD3<Float>(axes * clamped + origin)
        guard boundsFinite(result) else { throw Failure.unrepresentableResult }
        return result
    }

    public func intersects(_ sphere: BoundingSphere) -> Bool {
        let local = basis.transpose * (boundsDouble(sphere.center) - boundsDouble(center))
        let separation = simd_max(simd_abs(local) - boundsDouble(halfSize), .zero)
        return simd_length_squared(separation) <= Double(sphere.radius) * Double(sphere.radius)
    }

    public func intersects(_ box: AxisAlignedBox) -> Bool {
        let minimum = boundsDouble(box.minimum)
        let maximum = boundsDouble(box.maximum)
        return overlaps(center: (minimum + maximum) * 0.5, halfSize: (maximum - minimum) * 0.5, basis: matrix_identity_double3x3)
    }

    /// Tests all face normals and edge cross products. Touching counts as overlap.
    public func intersects(_ box: OrientedBoundingBox) -> Bool {
        overlaps(center: boundsDouble(box.center), halfSize: boundsDouble(box.halfSize), basis: box.basis)
    }

    /// Intersects the plane `dot(normal, point) + offset = 0`.
    /// The normal may have any finite, nonzero length.
    public func intersectsPlane(normal: SIMD3<Float>, offset: Float) -> Bool {
        guard boundsFinite(normal), normal != .zero, offset.isFinite else { return false }
        let direction = boundsDouble(normal)
        let reach = simd_dot(boundsDouble(halfSize), simd_abs(basis.transpose * direction))
        return abs(simd_dot(direction, boundsDouble(center)) + Double(offset)) <= reach
    }

    /// Finds the first surface hit within the ray's distance limit. A ray
    /// starting strictly inside the box hits the exit surface.
    public func intersection(ray: SceneRay) throws -> RayHit? {
        let inverse = basis.transpose
        let extent = boundsDouble(halfSize)
        let origin = inverse * (boundsDouble(ray.origin) - boundsDouble(center))
        let worldDirection = simd_normalize(boundsDouble(ray.direction))
        let direction = inverse * worldDirection
        var entry = -Double.infinity
        var exit = Double.infinity
        for axis in 0..<3 {
            if direction[axis] == 0 {
                guard abs(origin[axis]) <= extent[axis] else { return nil }
                continue
            }
            let first = (-extent[axis] - origin[axis]) / direction[axis]
            let second = (extent[axis] - origin[axis]) / direction[axis]
            entry = max(entry, min(first, second))
            exit = min(exit, max(first, second))
            guard entry <= exit else { return nil }
        }
        // A parallel ray on a face starts on the surface even if its interval
        // spans points on both sides of the ray origin.
        let onSurface = all(simd_abs(origin) .<= extent) && any(simd_abs(origin) .== extent)
        let distance = onSurface ? 0 : (entry >= 0 ? entry : exit)
        guard distance >= 0, distance <= Double(ray.maximumDistance) else { return nil }
        let rounded = Float(distance)
        let point = SIMD3<Float>(boundsDouble(ray.origin) + worldDirection * distance)
        guard rounded.isFinite, boundsFinite(point) else { throw Failure.unrepresentableResult }
        return RayHit(distance: rounded, position: point)
    }

    /// Applies a finite affine transform whose transformed box axes remain
    /// orthogonal, including signed scaling along those axes. Throws for shear
    /// or a collapsed axis. The result encloses the transformed corners after rounding.
    public func transformed(by matrix: simd_float4x4) throws -> OrientedBoundingBox {
        guard let affine = BoundsAffine(matrix) else { throw Failure.invalidTransform }
        let transformedAxes = affine.linear * basis
        let (x, y, z) = transformedAxes.columns
        guard simd_length_squared(x) > 0, simd_length_squared(y) > 0, simd_length_squared(z) > 0 else {
            throw Failure.invalidTransform
        }
        let right = simd_normalize(x)
        let up = simd_normalize(y)
        let forward = simd_normalize(z)
        guard abs(simd_dot(right, up)) <= 1e-6, abs(simd_dot(right, forward)) <= 1e-6,
              abs(simd_dot(up, forward)) <= 1e-6 else { throw Failure.invalidTransform }

        // Form a right-handed basis even when the transform reflects an axis.
        let normal = simd_normalize(simd_cross(right, up))
        let axes = simd_double3x3(columns: (right, simd_cross(normal, right), normal))
        let rotation = simd_normalize(simd_quatf(vector: SIMD4<Float>(simd_quatd(axes).vector)))
        let inverse = Self.basis(for: rotation).transpose
        let center = affine.linear * boundsDouble(center) + affine.translation
        let roundedCenter = SIMD3<Float>(center)
        let centerError = simd_abs(inverse * (center - boundsDouble(roundedCenter)))
        let radius = Self.absolute(inverse * transformedAxes) * boundsDouble(halfSize)
        let halfSize = boundsFloatAbove(radius + centerError)
        guard boundsFinite(roundedCenter), boundsFinite(halfSize) else { throw Failure.unrepresentableResult }
        return OrientedBoundingBox(validatedCenter: roundedCenter, halfSize: halfSize, rotation: rotation)
    }

    public struct RayHit: Equatable, Sendable {
        public let distance: Float
        public let position: SIMD3<Float>
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidPoint
        case invalidHalfSize
        case invalidRotation
        case invalidTransform
        case unrepresentableResult

        public var errorDescription: String? {
            switch self {
            case .invalidPoint: "An oriented box needs a finite center and finite query points."
            case .invalidHalfSize: "An oriented box needs finite, nonnegative half extents."
            case .invalidRotation: "An oriented box needs a unit rotation quaternion."
            case .invalidTransform: "Use a finite affine transform that preserves the box's orthogonal axes."
            case .unrepresentableResult: "The oriented box result exceeds the Float range."
            }
        }
    }

    var basis: simd_double3x3 { Self.basis(for: rotation) }

    var exactBounds: CollisionBounds {
        let origin = boundsDouble(center)
        let radius = Self.absolute(basis) * boundsDouble(halfSize)
        return CollisionBounds(minimum: origin - radius, maximum: origin + radius)
    }

    private func overlaps(center: SIMD3<Double>, halfSize: SIMD3<Double>, basis other: simd_double3x3) -> Bool {
        let own = basis
        let ownExtent = boundsDouble(self.halfSize)
        let offset = center - boundsDouble(self.center)
        let ownTranspose = own.transpose
        let otherTranspose = other.transpose
        func separates(_ axis: SIMD3<Double>) -> Bool {
            let reach = simd_dot(ownExtent, simd_abs(ownTranspose * axis))
                + simd_dot(halfSize, simd_abs(otherTranspose * axis))
            return abs(simd_dot(offset, axis)) > reach
        }
        for i in 0..<3 {
            if separates(own[i]) || separates(other[i]) { return false }
            for j in 0..<3 where separates(simd_cross(own[i], other[j])) { return false }
        }
        return true
    }

    private static func basis(for rotation: simd_quatf) -> simd_double3x3 {
        simd_double3x3(simd_normalize(simd_quatd(vector: SIMD4<Double>(rotation.vector))))
    }

    private static func absolute(_ matrix: simd_double3x3) -> simd_double3x3 {
        simd_double3x3(columns: (simd_abs(matrix[0]), simd_abs(matrix[1]), simd_abs(matrix[2])))
    }
}

/// A query enclosure can extend past Float even though a mesh's bounds cannot.
struct CollisionBounds {
    let minimum: SIMD3<Double>
    let maximum: SIMD3<Double>

    func overlaps(minimum: SIMD3<Float>, maximum: SIMD3<Float>) -> Bool {
        all(self.minimum .<= boundsDouble(maximum)) && all(boundsDouble(minimum) .<= self.maximum)
    }
}
