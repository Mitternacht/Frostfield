import simd

/// A contact between a closed collider and one nondegenerate mesh triangle.
public struct MeshCollisionContact: Equatable, Sendable {
    public let triangleIndex: Int
    /// The point on the triangle closest to the collider's spine or box center.
    public let position: SIMD3<Float>
    /// Unit direction in which to move the collider away from this triangle.
    public let normal: SIMD3<Float>
    /// Translation along `normal` to reach tangency with the triangle, in geometry units.
    public let depth: Float
}

public extension MeshBVH {
    /// All contacts with a capsule in geometry coordinates, including tangencies,
    /// ordered by triangle index. Degenerate triangles have no contact surface.
    /// Each contact is a separate constraint when a collider hits multiple triangles.
    func contacts(intersecting capsule: Capsule) throws -> [MeshCollisionContact] {
        try roundedContacts(spine: capsule.spine, radius: capsule.radius, bounds: capsule.exactBounds)
    }

    /// All contacts with a sphere in geometry coordinates, ordered by triangle index.
    func contacts(intersecting sphere: BoundingSphere) throws -> [MeshCollisionContact] {
        let center = boundsDouble(sphere.center)
        let reach = SIMD3<Double>(repeating: Double(sphere.radius))
        let spine = CollisionSegment(start: center, end: center)
        return try roundedContacts(
            spine: spine, radius: sphere.radius, bounds: CollisionBounds(minimum: center - reach, maximum: center + reach)
        )
    }

    /// All contacts with an oriented box in geometry coordinates, ordered by triangle index.
    func contacts(intersecting box: OrientedBoundingBox) throws -> [MeshCollisionContact] {
        try boxContacts(
            center: boundsDouble(box.center), halfSize: boundsDouble(box.halfSize),
            basis: box.basis, bounds: box.exactBounds
        )
    }

    /// All contacts with an axis-aligned box in geometry coordinates, ordered by triangle index.
    func contacts(intersecting box: AxisAlignedBox) throws -> [MeshCollisionContact] {
        let minimum = boundsDouble(box.minimum)
        let maximum = boundsDouble(box.maximum)
        return try boxContacts(
            center: (minimum + maximum) * 0.5, halfSize: (maximum - minimum) * 0.5, basis: matrix_identity_double3x3,
            bounds: CollisionBounds(minimum: minimum, maximum: maximum)
        )
    }

    private func roundedContacts(
        spine: CollisionSegment, radius: Float, bounds: CollisionBounds
    ) throws -> [MeshCollisionContact] {
        try contacts(in: bounds) { triangle in
            let closest = triangle.closestPoints(to: spine)
            let separation = closest.0 - closest.1
            let distanceSquared = simd_length_squared(separation)
            let radius = Double(radius)
            guard distanceSquared <= radius * radius else { return nil }
            if distanceSquared > 0 {
                let distance = distanceSquared.squareRoot()
                return (closest.1, separation / distance, radius - distance)
            }
            // When the spine crosses the triangle, leave its plane on whichever
            // side needs the smaller translation of the whole capsule.
            let first = simd_dot(spine.start - triangle.a, triangle.normal)
            let last = simd_dot(spine.end - triangle.a, triangle.normal)
            let front = max(0, -min(first, last))
            let back = max(0, max(first, last))
            return front <= back
                ? (closest.1, triangle.normal, radius + front)
                : (closest.1, -triangle.normal, radius + back)
        }
    }

    private func boxContacts(
        center: SIMD3<Double>, halfSize: SIMD3<Double>, basis: simd_double3x3, bounds: CollisionBounds
    ) throws -> [MeshCollisionContact] {
        let inverse = basis.transpose
        return try contacts(in: bounds) { triangle in
            let localA = inverse * (triangle.a - center)
            let localB = inverse * (triangle.b - center)
            let localC = inverse * (triangle.c - center)
            guard let translation = triangleBoxTranslation(localA, localB, localC, halfSize: halfSize) else { return nil }
            return (triangle.closestPoint(to: center), basis * translation.normal, translation.depth)
        }
    }

    private func contacts(
        in bounds: CollisionBounds,
        test: (CollisionTriangle) -> (point: SIMD3<Double>, normal: SIMD3<Double>, depth: Double)?
    ) throws -> [MeshCollisionContact] {
        var result: [MeshCollisionContact] = []
        try forEachCollisionCandidate(in: bounds) { index in
            guard let triangle = CollisionTriangle(geometry.triangleCorners(index)), let hit = test(triangle) else { return }
            let depth = Float(hit.depth)
            let point = SIMD3<Float>(hit.point)
            guard depth.isFinite, boundsFinite(point) else { throw Failure.unrepresentableContact }
            result.append(MeshCollisionContact(
                triangleIndex: index, position: point, normal: SIMD3<Float>(hit.normal), depth: depth
            ))
        }
        return result.sorted { $0.triangleIndex < $1.triangleIndex }
    }
}

private struct CollisionTriangle {
    private let triangle: ProximityTriangle
    let normal: SIMD3<Double>
    var a: SIMD3<Double> { triangle.a }
    var b: SIMD3<Double> { triangle.b }
    var c: SIMD3<Double> { triangle.c }

    init?(_ corners: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)) {
        let triangle = ProximityTriangle(a: boundsDouble(corners.0), b: boundsDouble(corners.1), c: boundsDouble(corners.2))
        guard let normal = triangle.normal else { return nil }
        self.triangle = triangle
        self.normal = normal
    }

    var edges: [CollisionSegment] {
        [CollisionSegment(start: a, end: b), CollisionSegment(start: b, end: c), CollisionSegment(start: c, end: a)]
    }

    /// Tests the projection onto the plane with oriented edge half spaces.
    func containsProjection(of point: SIMD3<Double>) -> Bool {
        simd_dot(simd_cross(b - a, point - a), normal) >= 0
            && simd_dot(simd_cross(c - b, point - b), normal) >= 0
            && simd_dot(simd_cross(a - c, point - c), normal) >= 0
    }

    func closestPoint(to point: SIMD3<Double>) -> SIMD3<Double> {
        triangle.closestPoint(to: point, normal: normal).position
    }

    /// A closest pair is a plane crossing, a spine endpoint against the triangle,
    /// or a pair of points on the spine and a triangle edge.
    func closestPoints(to segment: CollisionSegment) -> (SIMD3<Double>, SIMD3<Double>) {
        let signed = simd_dot(segment.start - a, normal)
        let slope = simd_dot(segment.direction, normal)
        if slope != 0 {
            let t = -signed / slope
            if (0...1).contains(t) {
                let point = segment.start + t * segment.direction
                if containsProjection(of: point) { return (point, point) }
            }
        }
        let candidates = [(segment.start, closestPoint(to: segment.start)), (segment.end, closestPoint(to: segment.end))]
            + edges.map { segment.closestPoints(to: $0) }
        return candidates.min {
            simd_length_squared($0.0 - $0.1) < simd_length_squared($1.0 - $1.1)
        }!
    }
}

/// The box is centered at the origin. The same separating axes that detect an
/// overlap give the shortest translation that takes the box to triangle tangency.
private func triangleBoxTranslation(
    _ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>, halfSize: SIMD3<Double>
) -> (normal: SIMD3<Double>, depth: Double)? {
    var best = (normal: SIMD3<Double>.zero, depth: Double.infinity)
    func consider(_ direction: SIMD3<Double>) -> Bool {
        guard direction != .zero else { return true }
        let axis = simd_normalize(direction)
        let p0 = simd_dot(a, axis)
        let p1 = simd_dot(b, axis)
        let p2 = simd_dot(c, axis)
        let reach = simd_dot(halfSize, simd_abs(axis))
        let positive = reach + max(p0, p1, p2)
        let negative = reach - min(p0, p1, p2)
        guard positive >= 0, negative >= 0 else { return false }
        let depth = min(positive, negative)
        if depth < best.depth { best = (positive <= negative ? axis : -axis, depth) }
        return true
    }
    let edges = [b - a, c - b, a - c]
    let axes = [SIMD3<Double>(1, 0, 0), SIMD3<Double>(0, 1, 0), SIMD3<Double>(0, 0, 1)]
    for axis in axes {
        guard consider(axis) else { return nil }
        for edge in edges where !consider(simd_cross(axis, edge)) { return nil }
    }
    return consider(simd_cross(edges[0], edges[1])) ? best : nil
}

public extension Triangle3D {
    /// Tests overlap with a closed axis-aligned box, including tangency and
    /// degenerate triangles that intersect as segments or points.
    func intersects(_ box: AxisAlignedBox) -> Bool {
        let lower = boundsDouble(box.minimum), upper = boundsDouble(box.maximum)
        let center = (lower + upper) * 0.5
        return triangleBoxTranslation(
            boundsDouble(a) - center, boundsDouble(b) - center, boundsDouble(c) - center,
            halfSize: (upper - lower) * 0.5
        ) != nil
    }

    /// The closest point on the closed triangle. A degenerate triangle uses
    /// its remaining segments or point.
    func closestPoint(to point: SIMD3<Float>) throws -> SIMD3<Float> {
        guard boundsFinite(point) else { throw Failure.nonfinitePoint }
        let triangle = ProximityTriangle(a: boundsDouble(a), b: boundsDouble(b), c: boundsDouble(c))
        let result = SIMD3<Float>(triangle.closestPoint(to: boundsDouble(point), normal: triangle.normal).position)
        guard boundsFinite(result) else { throw Failure.unrepresentablePoint }
        return result
    }
}
