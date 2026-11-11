import Foundation
import simd

/// A closed axis-aligned box. Equal corners describe a single point.
public struct AxisAlignedBox: Equatable, Sendable {
    public let minimum: SIMD3<Float>
    public let maximum: SIMD3<Float>

    /// The corners must be finite, with `minimum` no greater than `maximum` on every axis.
    public init(minimum: SIMD3<Float>, maximum: SIMD3<Float>) throws {
        guard boundsFinite(minimum), boundsFinite(maximum) else { throw Failure.nonFiniteCorner }
        guard minimum.x <= maximum.x, minimum.y <= maximum.y, minimum.z <= maximum.z else {
            throw Failure.invertedCorners
        }
        self.init(validatedMinimum: minimum, maximum: maximum)
    }

    /// The smallest box that encloses at least one finite point.
    public init(enclosing points: some Sequence<SIMD3<Float>>) throws {
        var lower = SIMD3<Float>(repeating: .infinity)
        var upper = SIMD3<Float>(repeating: -.infinity)
        var count = 0
        for point in points {
            guard boundsFinite(point) else { throw Failure.nonFinitePoint(count) }
            lower = simd_min(lower, point)
            upper = simd_max(upper, point)
            count += 1
        }
        guard count > 0 else { throw Failure.noPoints }
        self.init(validatedMinimum: lower, maximum: upper)
    }

    init(validatedMinimum minimum: SIMD3<Float>, maximum: SIMD3<Float>) {
        self.minimum = minimum
        self.maximum = maximum
    }

    /// The midpoint of the corners.
    public var center: SIMD3<Float> { minimum * 0.5 + maximum * 0.5 }

    /// The extent on each axis. An axis wider than `Float.greatestFiniteMagnitude` reports infinity.
    public var size: SIMD3<Float> { maximum - minimum }

    /// Whether the closed box contains the point.
    public func contains(_ point: SIMD3<Float>) -> Bool {
        all(point .>= minimum) && all(point .<= maximum)
    }

    /// Whether the closed box contains both corners of the other box.
    public func contains(_ other: AxisAlignedBox) -> Bool { contains(other.minimum) && contains(other.maximum) }

    /// The nearest point in the closed box. Finite points inside are unchanged.
    public func closestPoint(to point: SIMD3<Float>) throws -> SIMD3<Float> {
        guard boundsFinite(point) else { throw Failure.nonFiniteQueryPoint }
        return simd_clamp(point, minimum, maximum)
    }

    /// Euclidean distance from the point to the closed box.
    public func distance(to point: SIMD3<Float>) throws -> Double {
        try simd_length(boundsDouble(point) - boundsDouble(closestPoint(to: point)))
    }

    /// Whether the closed boxes overlap. Touching faces, edges or corners count as overlap.
    public func intersects(_ other: AxisAlignedBox) -> Bool {
        all(minimum .<= other.maximum) && all(other.minimum .<= maximum)
    }

    /// Whether the sphere reaches any point of the closed box.
    public func intersects(_ sphere: BoundingSphere) -> Bool {
        let offset = boundsDouble(sphere.center) - boundsDouble(simd_clamp(sphere.center, minimum, maximum))
        let radius = Double(sphere.radius)
        return simd_length_squared(offset) <= radius * radius
    }

    public func intersects(_ plane: Plane3D) -> Bool { plane.intersects(self) }
    public func intersects(_ triangle: Triangle3D) -> Bool { triangle.intersects(self) }

    /// Moves both corners by the same world-space offset. The result must stay finite.
    public func translated(by offset: SIMD3<Float>) throws -> AxisAlignedBox {
        try AxisAlignedBox(minimum: minimum + offset, maximum: maximum + offset)
    }

    /// The smallest box enclosing this box after an affine transform, such as a
    /// node's local-to-world matrix. Corners are rounded outward to `Float`, so the
    /// result always encloses the exactly transformed box. Singular transforms are
    /// allowed and flatten the box. Throws for a non-affine or non-finite matrix,
    /// or a result outside the `Float` range.
    public func transformed(by matrix: simd_float4x4) throws -> AxisAlignedBox {
        guard let affine = BoundsAffine(matrix) else { throw Failure.invalidTransform }
        return try transformed(by: affine)
    }

    func transformed(by matrix: simd_double4x4) throws -> AxisAlignedBox {
        guard let affine = BoundsAffine(matrix) else { throw Failure.invalidTransform }
        return try transformed(by: affine)
    }

    private func transformed(by affine: BoundsAffine) throws -> AxisAlignedBox {
        let lower = boundsDouble(minimum)
        let upper = boundsDouble(maximum)
        let center = affine.linear * ((lower + upper) * 0.5) + affine.translation
        let halfSize = (upper - lower) * 0.5
        let absolute = simd_double3x3(columns: (
            simd_abs(affine.linear.columns.0), simd_abs(affine.linear.columns.1), simd_abs(affine.linear.columns.2)
        ))
        let extent = absolute * halfSize
        let result = AxisAlignedBox(
            validatedMinimum: boundsFloatBelow(center - extent), maximum: boundsFloatAbove(center + extent)
        )
        guard boundsFinite(result.minimum), boundsFinite(result.maximum) else { throw Failure.unrepresentableResult }
        return result
    }

    /// The sphere centered on the box that passes through its corners.
    /// The radius is rounded up, so the sphere encloses the box.
    public var boundingSphere: BoundingSphere {
        let center = center
        let corner = simd_max(
            simd_abs(boundsDouble(maximum) - boundsDouble(center)),
            simd_abs(boundsDouble(center) - boundsDouble(minimum))
        )
        return BoundingSphere(validatedCenter: center, radius: boundsFloatAbove(simd_length(corner)))
    }

    public enum Failure: LocalizedError, Equatable {
        case nonFiniteCorner
        case invertedCorners
        case noPoints
        case nonFinitePoint(Int)
        case nonFiniteQueryPoint
        case invalidTransform
        case unrepresentableResult

        public var errorDescription: String? {
            switch self {
            case .nonFiniteCorner: "A box needs finite corners."
            case .invertedCorners: "A box's minimum corner must not exceed its maximum on any axis."
            case .noPoints: "A box needs at least one point to enclose."
            case .nonFinitePoint(let index): "Point \(index) is not finite."
            case .nonFiniteQueryPoint: "A box query needs a finite point."
            case .invalidTransform: "Bounds can only be transformed by a finite affine matrix."
            case .unrepresentableResult: "The transformed bounds exceed the Float range."
            }
        }
    }
}

/// A closed sphere. A zero radius describes a single point.
public struct BoundingSphere: Equatable, Sendable {
    public let center: SIMD3<Float>
    public let radius: Float

    /// The center must be finite and the radius finite and nonnegative.
    public init(center: SIMD3<Float>, radius: Float) throws {
        guard boundsFinite(center) else { throw Failure.nonFiniteCenter }
        guard radius.isFinite, radius >= 0 else { throw Failure.invalidRadius }
        self.init(validatedCenter: center, radius: radius)
    }

    init(validatedCenter center: SIMD3<Float>, radius: Float) {
        self.center = center
        self.radius = radius
    }

    /// Whether the closed sphere contains the point.
    public func contains(_ point: SIMD3<Float>) -> Bool {
        let offset = boundsDouble(point) - boundsDouble(center)
        let radius = Double(radius)
        return simd_length_squared(offset) <= radius * radius
    }

    /// Whether the two closed spheres overlap, including external tangency.
    public func intersects(_ other: BoundingSphere) -> Bool {
        let offset = boundsDouble(center) - boundsDouble(other.center)
        let reach = Double(radius) + Double(other.radius)
        return simd_length_squared(offset) <= reach * reach
    }

    public func intersects(_ box: AxisAlignedBox) -> Bool { box.intersects(self) }
    public func intersects(_ plane: Plane3D) -> Bool { plane.intersects(self) }

    /// A sphere enclosing this one after an affine transform. The radius is scaled
    /// by the transform's largest singular value, which is exact for rotations and
    /// uniform scales and encloses the ellipsoid that nonuniform scale or shear
    /// produces. Throws for a non-affine or non-finite matrix, or a result outside
    /// the `Float` range.
    public func transformed(by matrix: simd_float4x4) throws -> BoundingSphere {
        guard let affine = BoundsAffine(matrix) else { throw Failure.invalidTransform }
        let center = affine.linear * boundsDouble(center) + affine.translation
        let roundedCenter = SIMD3(Float(center.x), Float(center.y), Float(center.z))
        let gram = affine.linear.transpose * affine.linear
        // Pad the closed-form eigenvalue by a relative margin far above Double rounding,
        // and grow the radius by however far rounding moved the center.
        let scale = largestSymmetricEigenvalue(gram).squareRoot() * (1 + 1e-9)
        let centerError = simd_length(boundsDouble(roundedCenter) - center)
        let result = BoundingSphere(
            validatedCenter: roundedCenter, radius: boundsFloatAbove(Double(radius) * scale + centerError)
        )
        guard boundsFinite(result.center), result.radius.isFinite else { throw Failure.unrepresentableResult }
        return result
    }

    public enum Failure: LocalizedError, Equatable {
        case nonFiniteCenter
        case invalidRadius
        case invalidTransform
        case unrepresentableResult

        public var errorDescription: String? {
            switch self {
            case .nonFiniteCenter: "A sphere needs a finite center."
            case .invalidRadius: "A sphere's radius must be finite and nonnegative."
            case .invalidTransform: "Bounds can only be transformed by a finite affine matrix."
            case .unrepresentableResult: "The transformed sphere exceeds the Float range."
            }
        }
    }
}

extension MeshGeometry {
    /// The smallest box enclosing every vertex position in local coordinates,
    /// including vertices that no triangle references. Computed on each access.
    public var bounds: AxisAlignedBox {
        var lower = vertices[0].position
        var upper = lower
        for vertex in vertices {
            lower = simd_min(lower, vertex.position)
            upper = simd_max(upper, vertex.position)
        }
        return AxisAlignedBox(validatedMinimum: lower, maximum: upper)
    }

    /// A sphere centered on ``bounds`` whose radius reaches the farthest vertex
    /// position, rounded up so every vertex lies inside. Usually tighter than the
    /// box's own corner sphere. Computed on each access.
    public var boundingSphere: BoundingSphere {
        let center = bounds.center
        let origin = boundsDouble(center)
        var radiusSquared = 0.0
        for vertex in vertices {
            radiusSquared = max(radiusSquared, simd_length_squared(boundsDouble(vertex.position) - origin))
        }
        return BoundingSphere(validatedCenter: center, radius: boundsFloatAbove(radiusSquared.squareRoot()))
    }
}

/// A finite affine matrix split into its linear part and translation, in Double.
struct BoundsAffine {
    let linear: simd_double3x3
    let translation: SIMD3<Double>

    init?(_ matrix: simd_float4x4) {
        self.init(simd_double4x4(
            SIMD4<Double>(matrix[0]), SIMD4<Double>(matrix[1]),
            SIMD4<Double>(matrix[2]), SIMD4<Double>(matrix[3])
        ))
    }

    init?(_ matrix: simd_double4x4) {
        let (x, y, z, w) = matrix.columns
        guard x.w == 0, y.w == 0, z.w == 0, w.w == 1 else { return nil }
        let columns = [x, y, z, w].map { SIMD3($0.x, $0.y, $0.z) }
        guard columns.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { return nil }
        linear = simd_double3x3(columns: (columns[0], columns[1], columns[2]))
        translation = columns[3]
    }
}

/// The largest eigenvalue of a symmetric positive semidefinite matrix, by the
/// closed-form trigonometric solution of its characteristic cubic.
private func largestSymmetricEigenvalue(_ matrix: simd_double3x3) -> Double {
    let (c0, c1, c2) = matrix.columns
    let offDiagonal = c1.x * c1.x + c2.x * c2.x + c2.y * c2.y
    let diagonal = SIMD3(c0.x, c1.y, c2.z)
    guard offDiagonal > 0 else { return diagonal.max() }
    let mean = diagonal.sum() / 3
    let centered = diagonal - mean
    let spread = ((simd_length_squared(centered) + 2 * offDiagonal) / 6).squareRoot()
    guard spread > 0 else { return mean }
    let shifted = simd_double3x3(columns: (
        SIMD3(centered.x, c0.y, c0.z), SIMD3(c1.x, centered.y, c1.z), SIMD3(c2.x, c2.y, centered.z)
    ))
    let half = simd_determinant(shifted) / (2 * spread * spread * spread)
    let angle = acos(min(max(half, -1), 1)) / 3
    return mean + 2 * spread * cos(angle)
}

func boundsFinite(_ value: SIMD3<Float>) -> Bool {
    value.x.isFinite && value.y.isFinite && value.z.isFinite
}

func boundsDouble(_ value: SIMD3<Float>) -> SIMD3<Double> {
    SIMD3(Double(value.x), Double(value.y), Double(value.z))
}

/// The largest `Float` no greater than the value, per component.
func boundsFloatBelow(_ value: SIMD3<Double>) -> SIMD3<Float> {
    SIMD3(boundsFloatBelow(value.x), boundsFloatBelow(value.y), boundsFloatBelow(value.z))
}

/// The smallest `Float` no less than the value, per component.
func boundsFloatAbove(_ value: SIMD3<Double>) -> SIMD3<Float> {
    SIMD3(boundsFloatAbove(value.x), boundsFloatAbove(value.y), boundsFloatAbove(value.z))
}

func boundsFloatBelow(_ value: Double) -> Float {
    let rounded = Float(value)
    return Double(rounded) > value ? rounded.nextDown : rounded
}

func boundsFloatAbove(_ value: Double) -> Float {
    let rounded = Float(value)
    return Double(rounded) < value ? rounded.nextUp : rounded
}
