import Foundation
import simd

/// An infinite oriented plane in Hessian form, `dot(normal, point) + constant = 0`.
public struct Plane3D: Equatable, Sendable {
    /// The unit normal. Distances are evaluated with its double-precision representation.
    public var normal: SIMD3<Float> { SIMD3<Float>(unitNormal) }
    /// The constant of the normalized plane equation.
    public var constant: Double { planeConstant / normalLength }
    var homogeneousCoefficients: SIMD4<Double> { SIMD4(planeNormal, planeConstant) }
    private let planeNormal: SIMD3<Double>
    private let planeConstant: Double
    private let normalLength: Double
    var unitNormal: SIMD3<Double> { planeNormal / normalLength }

    /// Normalizes both coefficients of the supplied plane equation.
    public init(normal: SIMD3<Float>, constant: Double) throws {
        let vector = try Self.validatedNormal(normal)
        self.init(planeNormal: vector, planeConstant: constant)
        guard self.constant.isFinite else { throw Failure.invalidConstant }
    }

    /// Builds a plane through a point with the supplied normal's orientation.
    public init(normal: SIMD3<Float>, through point: SIMD3<Float>) throws {
        guard boundsFinite(point) else { throw Failure.nonfinitePoint }
        let vector = try Self.validatedNormal(normal)
        self.init(planeNormal: vector, planeConstant: -simd_dot(vector, boundsDouble(point)))
    }

    /// Builds a plane through three noncollinear points, with winding `(b - a) × (c - a)`.
    public init(through a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) throws {
        guard [a, b, c].allSatisfy(boundsFinite) else { throw Failure.nonfinitePoint }
        let first = boundsDouble(a)
        let cross = simd_cross(boundsDouble(b) - first, boundsDouble(c) - first)
        guard cross != .zero else { throw Failure.invalidNormal }
        self.init(planeNormal: cross, planeConstant: -simd_dot(cross, first))
    }

    fileprivate init(planeNormal: SIMD3<Double>, planeConstant: Double) {
        self.planeNormal = planeNormal
        self.planeConstant = planeConstant
        self.normalLength = simd_length(planeNormal)
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.unitNormal == rhs.unitNormal && lhs.constant == rhs.constant
    }

    public func signedDistance(to point: SIMD3<Float>) -> Double {
        signedDistance(to: boundsDouble(point))
    }

    /// The minimum oriented distance among points on the sphere.
    public func signedDistance(to sphere: BoundingSphere) -> Double {
        signedDistance(to: sphere.center) - Double(sphere.radius)
    }

    public func projectedPoint(_ point: SIMD3<Float>) throws -> SIMD3<Float> {
        guard boundsFinite(point) else { throw Failure.nonfinitePoint }
        let value = boundsDouble(point)
        return try Self.checkedPoint(value - unitNormal * signedDistance(to: value))
    }

    /// Touching the plane counts as intersection.
    public func intersects(_ sphere: BoundingSphere) -> Bool {
        abs(signedDistance(to: sphere.center)) <= Double(sphere.radius)
    }

    /// Touching the plane counts as intersection, including flat boxes.
    public func intersects(_ box: AxisAlignedBox) -> Bool {
        let lower = boundsDouble(box.minimum), upper = boundsDouble(box.maximum)
        let radius = simd_dot(simd_abs(planeNormal), (upper - lower) * 0.5)
        return abs(planeValue(at: (lower + upper) * 0.5)) <= radius
    }

    /// Intersects a closed segment. A coplanar segment returns its start.
    public func intersection(from start: SIMD3<Float>, to end: SIMD3<Float>) throws -> SIMD3<Float>? {
        guard boundsFinite(start), boundsFinite(end) else { throw Failure.nonfinitePoint }
        let origin = boundsDouble(start), step = boundsDouble(end) - origin
        guard let parameter = parameter(from: origin, along: step), (0...1).contains(parameter) else { return nil }
        return try Self.checkedPoint(origin + step * parameter)
    }

    fileprivate func parameter(from origin: SIMD3<Double>, along direction: SIMD3<Double>) -> Double? {
        let value = planeValue(at: origin)
        let slope = simd_dot(planeNormal, direction)
        if slope == 0 { return value == 0 ? 0 : nil }
        return -value / slope
    }

    private func signedDistance(to point: SIMD3<Double>) -> Double {
        planeValue(at: point) / normalLength
    }

    // Retain the original coefficients so normalization cannot move
    // exactly coplanar input points or directions across the plane.
    private func planeValue(at point: SIMD3<Double>) -> Double {
        simd_dot(planeNormal, point) + planeConstant
    }

    private static func validatedNormal(_ normal: SIMD3<Float>) throws -> SIMD3<Double> {
        guard boundsFinite(normal), normal != .zero else { throw Failure.invalidNormal }
        return boundsDouble(normal)
    }

    private static func checkedPoint(_ point: SIMD3<Double>) throws -> SIMD3<Float> {
        let result = SIMD3<Float>(point)
        guard boundsFinite(result) else { throw Failure.unrepresentablePoint }
        return result
    }

    public enum Failure: LocalizedError, Equatable {
        case nonfinitePoint
        case invalidNormal
        case invalidConstant
        case unrepresentablePoint

        public var errorDescription: String? {
            switch self {
            case .nonfinitePoint: "A plane needs finite input points."
            case .invalidNormal: "A plane needs a finite nonzero normal or three noncollinear points."
            case .invalidConstant: "The plane's normalized constant must be finite."
            case .unrepresentablePoint: "The computed point exceeds the Float coordinate range."
            }
        }
    }
}

/// Three finite points in counterclockwise front-face order. Degenerate
/// triangles have zero area and no plane, normal, or barycentric coordinates.
public struct Triangle3D: Equatable, Sendable {
    public let a: SIMD3<Float>
    public let b: SIMD3<Float>
    public let c: SIMD3<Float>

    public init(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) throws {
        guard [a, b, c].allSatisfy(boundsFinite) else { throw Failure.nonfinitePoint }
        self.a = a
        self.b = b
        self.c = c
    }

    public var area: Double { simd_length(cross) * 0.5 }

    public var centroid: SIMD3<Float> { SIMD3<Float>((boundsDouble(a) + boundsDouble(b) + boundsDouble(c)) / 3) }

    public var normal: SIMD3<Float>? { unitNormal.map(SIMD3<Float>.init) }

    public var plane: Plane3D? {
        let normal = cross
        guard normal != .zero else { return nil }
        return Plane3D(planeNormal: normal, planeConstant: -simd_dot(normal, boundsDouble(a)))
    }

    /// Weights of the point's orthogonal projection in A, B, C order.
    /// Weights may be negative outside the triangle. A degenerate triangle
    /// or nonfinite point returns nil.
    public func barycentricCoordinates(of point: SIMD3<Float>) -> SIMD3<Double>? {
        guard boundsFinite(point) else { return nil }
        let first = boundsDouble(a), fromA = boundsDouble(point) - first
        let ab = boundsDouble(b) - first, ac = boundsDouble(c) - first
        let normal = simd_cross(ab, ac)
        let denominator = simd_length_squared(normal)
        guard denominator > 0 else { return nil }
        let u = simd_dot(simd_cross(ac - ab, fromA - ab), normal)
        let v = simd_dot(simd_cross(fromA, ac), normal)
        let w = simd_dot(simd_cross(ab, fromA), normal)
        return SIMD3(u, v, w) / denominator
    }

    /// Tests the projected point against the closed triangle.
    public func containsProjection(of point: SIMD3<Float>) -> Bool {
        barycentricCoordinates(of: point).map { $0.x >= 0 && $0.y >= 0 && $0.z >= 0 } ?? false
    }

    fileprivate var cross: SIMD3<Double> { simd_cross(boundsDouble(b) - boundsDouble(a), boundsDouble(c) - boundsDouble(a)) }
    private var unitNormal: SIMD3<Double>? {
        let normal = cross
        return normal == .zero ? nil : simd_normalize(normal)
    }

    public enum Failure: LocalizedError, Equatable {
        case nonfinitePoint
        case unrepresentablePoint

        public var errorDescription: String? {
            switch self {
            case .nonfinitePoint: "A triangle query needs finite input points."
            case .unrepresentablePoint: "The computed point exceeds the Float coordinate range."
            }
        }
    }
}

/// The nearest accepted surface point on a bounded world ray.
public struct RaySurfaceHit: Equatable, Sendable {
    public let distance: Float
    public let position: SIMD3<Float>
}

/// A bounded ray hit on a triangle, with interpolants in authored vertex order.
public struct RayTriangleHit: Equatable, Sendable {
    public let distance: Float
    public let position: SIMD3<Float>
    /// The weights of vertices A, B, and C.
    public let barycentric: SIMD3<Float>
    /// The authored geometric normal, even for a back-face hit.
    public let normal: SIMD3<Float>
    public let isFrontFace: Bool
}

/// The closest pair on a bounded world ray and a closed line segment.
public struct RaySegmentProximity: Equatable, Sendable {
    public let rayPosition: SIMD3<Float>
    public let segmentPosition: SIMD3<Float>
    /// World units from the ray origin.
    public let rayDistance: Double
    /// Zero at the first segment endpoint and one at the second.
    public let segmentParameter: Double
    public let separation: Double
}

public extension SceneRay {
    /// The nearest point within the ray's closed distance interval.
    func closestPoint(to point: SIMD3<Float>) throws -> SIMD3<Float> {
        guard boundsFinite(point) else { throw ScenePickingError.nonfiniteQueryPoint }
        let start = boundsDouble(origin), step = preciseDirection
        let offset = boundsDouble(point) - start
        let along = min(max(0, simd_dot(offset, step)), Double(maximumDistance))
        let closest = SIMD3<Float>(start + step * along)
        guard boundsFinite(closest) else { throw ScenePickingError.unrepresentableResult }
        return closest
    }

    /// Euclidean distance from a point to the bounded ray.
    func distance(to point: SIMD3<Float>) throws -> Double {
        guard boundsFinite(point) else { throw ScenePickingError.nonfiniteQueryPoint }
        let offset = boundsDouble(point) - boundsDouble(origin)
        let step = preciseDirection
        let projection = simd_dot(offset, step)
        let along = min(max(0, projection), Double(maximumDistance))
        return separation(of: offset, direction: step, projection: projection, along: along)
    }

    /// Finds the closest pair including the ray origin, finite limit and both
    /// segment endpoints. Parallel overlaps choose the earliest ray position.
    func closestApproach(toSegmentFrom start: SIMD3<Float>, to end: SIMD3<Float>) throws -> RaySegmentProximity {
        guard boundsFinite(start), boundsFinite(end) else { throw ScenePickingError.nonfiniteQueryPoint }
        let rayOrigin = boundsDouble(origin), step = preciseDirection
        let segmentStart = boundsDouble(start)
        let edge = boundsDouble(end) - segmentStart
        let relativeStart = segmentStart - rayOrigin
        let limit = Double(maximumDistance)
        let parameter = raySegmentParameter(start: relativeStart, edge: edge, direction: step, maximumDistance: limit)
        let offset = relativeStart + edge * parameter
        let projection = simd_dot(offset, step)
        let along = min(max(0, projection), limit)
        let rayPosition = SIMD3<Float>(rayOrigin + step * along)
        let segmentPosition = SIMD3<Float>(segmentStart + edge * parameter)
        guard boundsFinite(rayPosition), boundsFinite(segmentPosition) else { throw ScenePickingError.unrepresentableResult }
        return RaySegmentProximity(
            rayPosition: rayPosition, segmentPosition: segmentPosition, rayDistance: along,
            segmentParameter: parameter, separation: separation(of: offset, direction: step, projection: projection, along: along)
        )
    }

    /// Finds the entry surface, or the exit when the ray starts inside.
    /// Parallel axes stay in their slabs. Touching a boundary counts.
    func intersection(with box: AxisAlignedBox) throws -> RaySurfaceHit? {
        let start = boundsDouble(origin), step = preciseDirection
        let lower = boundsDouble(box.minimum), upper = boundsDouble(box.maximum)
        var entry = -Double.infinity, exit = Double.infinity
        for axis in 0..<3 {
            let slope = step[axis]
            if slope == 0 {
                guard (lower[axis]...upper[axis]).contains(start[axis]) else { return nil }
                continue
            }
            let first = (lower[axis] - start[axis]) / slope
            let last = (upper[axis] - start[axis]) / slope
            entry = max(entry, min(first, last))
            exit = min(exit, max(first, last))
            guard entry <= exit else { return nil }
        }
        return try hit(at: entry >= 0 ? entry : exit, direction: step)
    }

    /// Finds the nearest sphere surface, using the exit when starting inside.
    /// An infinite bounding sphere has no finite surface.
    func intersection(with sphere: BoundingSphere) throws -> RaySurfaceHit? {
        let radius = Double(sphere.radius)
        guard radius.isFinite else { return nil }
        let rawDirection = boundsDouble(direction)
        let directionSquared = simd_length_squared(rawDirection)
        let toCenter = boundsDouble(sphere.center) - boundsDouble(origin)
        let radiusSquared = radius * radius
        let originGap = subtractSquaredComponents(of: toCenter, from: radiusSquared)
        if originGap == 0 {
            return RaySurfaceHit(distance: 0, position: origin)
        }
        // The cross product retains small lateral offsets to distant spheres.
        // Evaluate tangency before unit normalization rounds either term.
        let perpendicular = simd_cross(toCenter, rawDirection)
        let scaledChordSquared = subtractSquaredComponents(of: perpendicular, from: radiusSquared * directionSquared)
        guard scaledChordSquared >= 0 else { return nil }
        let step = preciseDirection
        let center = simd_dot(toCenter, step)
        if scaledChordSquared == 0 { return try hit(at: center, direction: step) }
        let halfChord = (scaledChordSquared / directionSquared).squareRoot()
        // Use the product of the roots to recover the smaller magnitude
        // without canceling a positive hit close to the ray origin.
        let farther = center + (center >= 0 ? halfChord : -halfChord)
        let nearer = -originGap / farther
        let entry = min(farther, nearer), exit = max(farther, nearer)
        return try hit(at: entry >= 0 ? entry : exit, direction: step)
    }

    /// Finds the plane surface within the ray bounds. A coplanar ray returns its origin.
    func intersection(with plane: Plane3D) throws -> RaySurfaceHit? {
        let rawDirection = boundsDouble(direction)
        guard let parameter = plane.parameter(from: boundsDouble(origin), along: rawDirection) else { return nil }
        let length = simd_length(rawDirection)
        return try hit(at: parameter * length, direction: rawDirection / length)
    }

    /// Tests a closed triangle with the same authored face-side convention as
    /// mesh picking. Degenerate and coplanar triangles have no ray hit.
    func intersection(with triangle: Triangle3D, side: MeshRenderState.Side = .double) throws -> RayTriangleHit? {
        let step = preciseDirection, start = boundsDouble(origin), normal = triangle.cross
        let facing = simd_dot(step, normal)
        guard facing != 0, facing < 0 ? side != .back : side != .front else { return nil }

        // Shear along the ray's dominant axis, as in the watertight edge
        // test of Woop, Benthin and Wald. Adjacent triangles evaluate the
        // same projected edge with opposite signs.
        let magnitude = simd_abs(step)
        let z = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0 : (magnitude.y >= magnitude.z ? 1 : 2)
        var x = (z + 1) % 3, y = (z + 2) % 3
        if step[z] < 0 { swap(&x, &y) }
        let sx = step[x] / step[z], sy = step[y] / step[z]
        func project(_ point: SIMD3<Float>) -> SIMD3<Double> {
            let delta = boundsDouble(point) - start
            return SIMD3(delta[x] - sx * delta[z], delta[y] - sy * delta[z], delta[z] / step[z])
        }
        let a = project(triangle.a), b = project(triangle.b), c = project(triangle.c)
        let u = c.x * b.y - c.y * b.x
        let v = a.x * c.y - a.y * c.x
        let w = b.x * a.y - b.y * a.x
        let weights = SIMD3(u, v, w)
        guard simd_reduce_min(weights) >= 0 || simd_reduce_max(weights) <= 0 else { return nil }
        let determinant = u + v + w
        guard determinant != 0 else { return nil }
        let parameter = (u * a.z + v * b.z + w * c.z) / determinant
        guard let surface = try hit(at: parameter, direction: step) else { return nil }
        return RayTriangleHit(
            distance: surface.distance, position: surface.position, barycentric: SIMD3<Float>(weights / determinant),
            normal: SIMD3<Float>(simd_normalize(normal)), isFrontFace: facing < 0
        )
    }

    private var preciseDirection: SIMD3<Double> { simd_normalize(boundsDouble(direction)) }

    private func separation(
        of offset: SIMD3<Double>, direction: SIMD3<Double>, projection: Double, along: Double
    ) -> Double {
        simd_length(projection == along ? simd_cross(offset, direction) : offset - direction * along)
    }

    private func hit(at parameter: Double, direction: SIMD3<Double>) throws -> RaySurfaceHit? {
        guard parameter >= 0, parameter <= Double(maximumDistance) else { return nil }
        let distance = Float(parameter)
        let position = SIMD3<Float>(boundsDouble(origin) + direction * parameter)
        guard distance.isFinite, boundsFinite(position) else { throw ScenePickingError.unrepresentableResult }
        return RaySurfaceHit(distance: distance, position: position)
    }
}

/// Selects a segment parameter for the closest pair, with the ray origin
/// subtracted from the segment so large world translations cancel once.
func raySegmentParameter(
    start: SIMD3<Double>, edge: SIMD3<Double>, direction: SIMD3<Double>, maximumDistance: Double
) -> Double {
    let lengthSquared = simd_length_squared(edge)
    guard lengthSquared > 0 else { return 0 }
    func clamp(_ value: Double) -> Double { min(max(value, 0), 1) }

    // Use cross products to remove longitudinal components without subtracting
    // nearly equal dot products when the lines are close to parallel.
    let perpendicularEdge = simd_cross(edge, direction)
    let perpendicularLengthSquared = simd_length_squared(perpendicularEdge)
    if perpendicularLengthSquared == 0 {
        let step = simd_dot(edge, direction)
        return step == 0 ? 0 : clamp(-simd_dot(start, direction) / step)
    }
    let perpendicularStart = simd_cross(start, direction)
    let parameter = clamp(-simd_dot(perpendicularStart, perpendicularEdge) / perpendicularLengthSquared)
    let projection = simd_dot(start + edge * parameter, direction)
    let along = min(max(0, projection), maximumDistance)
    // Outside the ray interval the minimizing pair lies on its nearest end.
    return projection == along ? parameter : clamp(simd_dot(direction * along - start, edge) / lengthSquared)
}

// Subtract the largest term first so a small lateral offset survives when
// the radius and a major component are equal.
private func subtractSquaredComponents(of vector: SIMD3<Double>, from value: Double) -> Double {
    let squares = vector * vector
    let lowerXY = min(squares.x, squares.y), upperXY = max(squares.x, squares.y)
    let largest = max(upperXY, squares.z)
    let middle = min(upperXY, max(lowerXY, squares.z))
    let smallest = min(lowerXY, squares.z)
    return ((value - largest) - middle) - smallest
}
