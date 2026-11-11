import simd

/// The nearest point on an indexed triangle in the query's coordinate system.
public struct MeshClosestPoint: Equatable, Sendable {
    public let position: SIMD3<Float>
    /// Euclidean distance in world units, computed before rounding the position.
    public let distance: Double
    /// Zero-based triangle index, counting groups of three source indices.
    public let triangleIndex: Int
    /// Nonnegative weights of the triangle's three vertices in source index order.
    public let barycentric: SIMD3<Double>
}

struct BVHProximityFrame {
    private let affine: BoundsAffine

    init(_ matrix: simd_float4x4) throws {
        guard let affine = BoundsAffine(matrix) else { throw MeshBVH.Failure.invalidProximityTransform }
        self.affine = affine
    }

    func triangle(_ corners: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)) -> ProximityTriangle {
        func transform(_ point: SIMD3<Float>) -> SIMD3<Double> {
            affine.linear * boundsDouble(point) + affine.translation
        }
        return ProximityTriangle(a: transform(corners.0), b: transform(corners.1), c: transform(corners.2))
    }

    func bounds(lower: SIMD3<Float>, upper: SIMD3<Float>) -> CollisionBounds {
        CollisionBounds.transformed(lower: lower, upper: upper, linear: affine.linear, translation: affine.translation)
    }
}

extension CollisionBounds {
    /// Outward-rounded enclosing bounds of a box in an affine query frame.
    static func transformed(
        lower: SIMD3<Float>, upper: SIMD3<Float>, linear: simd_double3x3, translation: SIMD3<Double>
    ) -> Self {
        let (x, y, z) = linear.columns
        let low = boundsDouble(lower), high = boundsDouble(upper)
        let x0 = x * low.x, x1 = x * high.x
        let y0 = y * low.y, y1 = y * high.y
        let z0 = z * low.z, z1 = z * high.z
        let minimum = simd_min(x0, x1) + simd_min(y0, y1) + simd_min(z0, z1) + translation
        let maximum = simd_max(x0, x1) + simd_max(y0, y1) + simd_max(z0, z1) + translation

        // Widen by the absolute summands to cover Double rounding even when
        // large terms cancel in a shear or the translation.
        let scale = simd_max(simd_abs(x0), simd_abs(x1)) + simd_max(simd_abs(y0), simd_abs(y1))
            + simd_max(simd_abs(z0), simd_abs(z1)) + simd_abs(translation)
        let padding = (8 * Double.ulpOfOne) * scale
        return Self(minimum: minimum - padding, maximum: maximum + padding)
    }

    func squaredDistance(to point: SIMD3<Double>) -> Double {
        simd_length_squared(point - simd_clamp(point, minimum, maximum))
    }
}

/// Two nearest triangle witnesses in a shared world coordinate system.
public struct MeshClosestPoints: Equatable, Sendable {
    public let first: MeshClosestPoint
    public let second: MeshClosestPoint
    public var distance: Double { first.distance }
}

extension MeshClosestPoint {
    init(_ point: ProximityTriangle.Point, distance: Double, triangleIndex: Int) throws {
        let position = SIMD3<Float>(point.position)
        guard boundsFinite(position) else { throw MeshBVH.Failure.unrepresentableProximityPoint }
        self.init(position: position, distance: distance, triangleIndex: triangleIndex, barycentric: point.barycentric)
    }
}

extension CollisionBounds {
    var squaredDiagonal: Double { simd_length_squared(maximum - minimum) }

    func squaredDistance(to other: CollisionBounds) -> Double {
        let gap = simd_max(simd_max(minimum - other.maximum, other.minimum - maximum), .zero)
        return simd_length_squared(gap)
    }
}
