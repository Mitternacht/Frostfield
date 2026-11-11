import Frostfield
import Testing
import simd

struct GeometricQueryTests {
    @Test
    func `Closed shape overlaps include tangencies and reject diagonal separating axes`() throws {
        let box = try AxisAlignedBox(minimum: SIMD3(repeating: -1), maximum: .one)
        let inner = try AxisAlignedBox(minimum: .zero, maximum: .one)
        #expect(box.contains(inner))
        #expect(!inner.contains(box))
        #expect(try box.closestPoint(to: SIMD3(2, -3, 0.5)) == SIMD3(1, -1, 0.5))
        #expect(try box.distance(to: SIMD3(4, 5, 0)) == 5)
        let onFace = try Triangle3D(SIMD3(-2, 0, 1), SIMD3(2, 0, 1), SIMD3(0, 2, 1))
        let separated = try Triangle3D(SIMD3(2, 0.25, 0), SIMD3(0.25, 2, 0), SIMD3(2, 2, 0))
        let degenerate = try Triangle3D(SIMD3(0, -2, 0), SIMD3(0, 2, 0), SIMD3(0, 2, 0))
        #expect(box.intersects(onFace))
        #expect(!box.intersects(separated))
        #expect(degenerate.intersects(box))
        let touching = try BoundingSphere(center: SIMD3(2, 0, 0), radius: 1)
        let near = try BoundingSphere(center: .zero, radius: 1)
        let distant = try BoundingSphere(center: SIMD3(3.25, 0, 0), radius: 1)
        #expect(box.intersects(touching) && touching.intersects(box))
        #expect(touching.intersects(near))
        #expect(!distant.intersects(near) && !distant.intersects(box))
    }

    @Test
    func `Ray proximity uses the inclusive finite endpoint and earliest parallel pair`() throws {
        let ray = try SceneRay(origin: .zero, direction: SIMD3(0, 0, 1), maximumDistance: 3)
        let behind = SIMD3<Float>(2, 0, -3)
        let ahead = SIMD3<Float>(3, 0, 7)
        #expect(try ray.closestPoint(to: behind) == .zero)
        #expect(try ray.closestPoint(to: ahead) == SIMD3(0, 0, 3))
        #expect(try ray.distance(to: ahead) == 5)
        let finite = try ray.closestApproach(toSegmentFrom: SIMD3(3, 0, 4), to: SIMD3(-1, 0, 6))
        #expect(finite.rayDistance == 3 && finite.segmentParameter == 0.5)
        #expect(finite.rayPosition == SIMD3(0, 0, 3))
        #expect(finite.segmentPosition == SIMD3(1, 0, 5))
        #expect(finite.separation == Double(5).squareRoot())
        let parallel = try ray.closestApproach(toSegmentFrom: SIMD3(0, 0.5, 2), to: SIMD3(0, 0.5, -1))
        #expect(parallel.rayPosition == .zero && parallel.segmentPosition == SIMD3(0, 0.5, 0))
        #expect(parallel.segmentParameter == 2.0 / 3 && parallel.separation == 0.5)
        let pointSegment = try ray.closestApproach(toSegmentFrom: ahead, to: ahead)
        #expect(pointSegment.segmentParameter == 0 && pointSegment.separation == 5)
        let unbounded = try SceneRay(origin: .zero, direction: SIMD3(0, 0, 1))
        let crossing = try unbounded.closestApproach(toSegmentFrom: SIMD3(3, 0, 4), to: SIMD3(-1, 0, 6))
        #expect(crossing.rayDistance == 5.5 && crossing.segmentParameter == 0.75)
        #expect(crossing.separation == 0)
        #expect(throws: ScenePickingError.nonfiniteQueryPoint) { try ray.distance(to: SIMD3(.nan, 0, 0)) }
    }

    @Test
    func `Plane normalizes coefficients and queries closed volumes and segments`() throws {
        let plane = try Plane3D(normal: SIMD3(0, 0, 4), constant: -8)
        #expect(plane.normal == SIMD3(0, 0, 1))
        #expect(plane.constant == -2)
        #expect(plane.signedDistance(to: SIMD3(4, 5, 7)) == 5)
        #expect(plane.signedDistance(to: SIMD3(4, 5, 1)) == -1)
        #expect(try plane.projectedPoint(SIMD3(4, 5, 7)) == SIMD3(4, 5, 2))
        #expect(try Plane3D(normal: SIMD3(0, 0, 3), through: SIMD3(2, 0, 2)) == plane)
        #expect(try Plane3D(through: SIMD3(0, 0, 2), SIMD3(1, 0, 2), SIMD3(0, 1, 2)) == plane)

        let box = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(3, 3, 2))
        let shortBox = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(3, 3, Float(2).nextDown))
        #expect(plane.intersects(box))
        #expect(!plane.intersects(shortBox))
        let sphere = try BoundingSphere(center: SIMD3(0, 0, 3), radius: 1)
        let separated = try BoundingSphere(center: SIMD3(0, 0, 3), radius: Float(1).nextDown)
        #expect(plane.signedDistance(to: sphere) == 0)
        #expect(plane.intersects(sphere))
        #expect(!plane.intersects(separated))
        #expect(try plane.intersection(from: SIMD3(2, 0, 1), to: SIMD3(4, 0, 3)) == SIMD3(3, 0, 2))
        #expect(try plane.intersection(from: SIMD3(2, 0, 2), to: SIMD3(4, 0, 2)) == SIMD3(2, 0, 2))
        #expect(try plane.intersection(from: SIMD3(2, 0, 1), to: SIMD3(4, 0, 2)) == SIMD3(4, 0, 2))
        #expect(try plane.intersection(from: SIMD3(2, 0, 1), to: SIMD3(4, 0, 1)) == nil)
        #expect(try plane.intersection(from: SIMD3(2, 0, 3), to: SIMD3(4, 0, 4)) == nil)
    }

    @Test
    func `Plane and triangle reject invalid coefficients and points`() throws {
        #expect(throws: Plane3D.Failure.invalidNormal) { try Plane3D(normal: .zero, constant: 0) }
        #expect(throws: Plane3D.Failure.invalidNormal) {
            try Plane3D(through: .zero, SIMD3(1, 1, 1), SIMD3(2, 2, 2))
        }
        #expect(throws: Plane3D.Failure.invalidConstant) { try Plane3D(normal: SIMD3(0, 0, 1), constant: .infinity) }
        #expect(throws: Plane3D.Failure.nonfinitePoint) { try Plane3D(normal: SIMD3(0, 0, 1), through: SIMD3(.nan, 0, 0)) }
        #expect(throws: Triangle3D.Failure.nonfinitePoint) { try Triangle3D(.zero, SIMD3(.infinity, 0, 0), .zero) }
        let triangle = try sampleTriangle()
        #expect(throws: Triangle3D.Failure.nonfinitePoint) { try triangle.closestPoint(to: SIMD3(.nan, 0, 0)) }
        #expect(triangle.barycentricCoordinates(of: SIMD3(.nan, 0, 0)) == nil)
    }

    @Test
    func `Triangle reports projected weights and falls back to segments when degenerate`() throws {
        let triangle = try sampleTriangle()
        #expect(triangle.area == 6)
        #expect(triangle.centroid == SIMD3(4.0 / 3, 1, 0))
        #expect(triangle.normal == SIMD3(0, 0, 1))
        #expect(triangle.plane == (try Plane3D(normal: SIMD3(0, 0, 1), constant: 0)))
        #expect(triangle.barycentricCoordinates(of: SIMD3(1, 1.5, 20)) == SIMD3(0.25, 0.25, 0.5))
        #expect(triangle.containsProjection(of: SIMD3(2, 1.5, 20)))
        #expect(!triangle.containsProjection(of: SIMD3(3, 1.5, 0)))
        #expect(try triangle.closestPoint(to: SIMD3(1, 1.5, 20)) == SIMD3(1, 1.5, 0))
        #expect(try triangle.closestPoint(to: SIMD3(1, -2, 20)) == SIMD3(1, 0, 0))
        #expect(try triangle.closestPoint(to: SIMD3(-1, -2, 20)) == .zero)

        let flat = try Triangle3D(.zero, SIMD3(2, 0, 0), SIMD3(4, 0, 0))
        #expect(flat.area == 0)
        #expect(flat.normal == nil)
        #expect(flat.plane == nil)
        #expect(flat.barycentricCoordinates(of: SIMD3(1, 0, 0)) == nil)
        #expect(try flat.closestPoint(to: SIMD3(1, 1, 0)) == SIMD3(1, 0, 0))
        #expect(try Triangle3D(.one, .one, .one).closestPoint(to: .zero) == .one)
    }

    @Test
    func `Box rays honor parallel slabs, inside exits and both inclusive bounds`() throws {
        let box = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 2))
        let forward = SIMD3<Float>(3, 0, 0)
        let exterior = SIMD3<Float>(-2, 0, 1)
        let entryRay = try SceneRay(origin: exterior, direction: forward, maximumDistance: 2)
        let entry = try #require(try entryRay.intersection(with: box))
        #expect(entry.distance == 2)
        #expect(entry.position == SIMD3(0, 0, 1))
        #expect(try SceneRay(origin: exterior, direction: forward, maximumDistance: Float(2).nextDown).intersection(with: box) == nil)
        #expect(try SceneRay(origin: .one, direction: forward, maximumDistance: 1).intersection(with: box)?.position == SIMD3(2, 1, 1))
        #expect(try SceneRay(origin: .one, direction: forward, maximumDistance: Float(1).nextDown).intersection(with: box) == nil)
        #expect(try SceneRay(origin: exterior, direction: -forward).intersection(with: box) == nil)
        #expect(try SceneRay(origin: SIMD3(-2, Float(0).nextDown, 1), direction: forward).intersection(with: box) == nil)
        #expect(try SceneRay(origin: SIMD3(0, 0, 1), direction: -forward, maximumDistance: 0).intersection(with: box)?.distance == 0)

        let largest = Float.greatestFiniteMagnitude
        let distant = try AxisAlignedBox(minimum: SIMD3(largest, 0, 0), maximum: SIMD3(largest, 0, 0))
        let longRay = try SceneRay(origin: SIMD3(-largest, 0, 0), direction: forward)
        #expect(throws: ScenePickingError.unrepresentableResult) { try longRay.intersection(with: distant) }
        #expect(try SceneRay(origin: longRay.origin, direction: forward, maximumDistance: largest).intersection(with: distant) == nil)
    }

    @Test
    func `Sphere rays hit the first bounded surface and retain distant tangent precision`() throws {
        let sphere = try BoundingSphere(center: .zero, radius: 1)
        let forward = SIMD3<Float>(9, 0, 0)
        let entryRay = try SceneRay(origin: SIMD3(-3, 0, 0), direction: forward, maximumDistance: 2)
        #expect(try entryRay.intersection(with: sphere)?.position == SIMD3(-1, 0, 0))
        #expect(try SceneRay(origin: entryRay.origin, direction: forward, maximumDistance: Float(2).nextDown).intersection(with: sphere) == nil)
        #expect(try SceneRay(origin: .zero, direction: forward, maximumDistance: 1).intersection(with: sphere)?.distance == 1)
        #expect(try SceneRay(origin: .zero, direction: forward, maximumDistance: Float(1).nextDown).intersection(with: sphere) == nil)
        #expect(try SceneRay(origin: SIMD3(2, 0, 0), direction: forward).intersection(with: sphere) == nil)
        #expect(try SceneRay(origin: SIMD3(1, 0, 0), direction: forward, maximumDistance: 0).intersection(with: sphere)?.distance == 0)
        let inward = try SceneRay(origin: SIMD3(1, 0, 0), direction: SIMD3(-3, 4, 0), maximumDistance: 0)
        #expect(try inward.intersection(with: sphere)?.distance == 0)
        let ray = try SceneRay(origin: .zero, direction: forward)
        let distantTangent = try BoundingSphere(center: SIMD3(100_000_000, 1, 0), radius: 1)
        let distantMiss = try BoundingSphere(center: SIMD3(100_000_000, 1.01, 0), radius: 1)
        let point = try BoundingSphere(center: SIMD3(2, 0, 0), radius: 0)
        #expect(try ray.intersection(with: distantTangent)?.distance == 100_000_000)
        #expect(try ray.intersection(with: distantMiss) == nil)
        #expect(try ray.intersection(with: point)?.position == point.center)
        let diagonal = try SceneRay(origin: .zero, direction: SIMD3(1, 1, 0))
        let obliqueTangent = try BoundingSphere(center: SIMD3(1, 1, 1), radius: 1)
        #expect(try diagonal.intersection(with: obliqueTangent)?.position == SIMD3(1, 1, 0))
        let largeSphere = try BoundingSphere(center: SIMD3(100_000_000, 0.1, 0), radius: 100_000_000)
        let around = try SceneRay(origin: .zero, direction: SIMD3(0, 1, 0))
        let past = try SceneRay(origin: .zero, direction: SIMD3(0, 0, 1))
        #expect(try around.intersection(with: largeSphere)?.position == SIMD3(0, 0.1, 0))
        #expect(try past.intersection(with: largeSphere) == nil)
        let contactOnly = try SceneRay(origin: .zero, direction: forward, maximumDistance: 0)
        #expect(try contactOnly.intersection(with: largeSphere) == nil)
        let nearby = try #require(try ray.intersection(with: largeSphere))
        #expect(abs(nearby.distance - 5e-11) < 1e-16)
    }

    @Test
    func `Plane rays distinguish crossings, coplanar origins and disjoint parallels`() throws {
        let plane = try Plane3D(normal: SIMD3(0, 0, 1), constant: -2)
        let below = SIMD3<Float>(1, 2, 0)
        #expect(try SceneRay(origin: below, direction: SIMD3(0, 0, 1), maximumDistance: 2).intersection(with: plane)?.position == SIMD3(1, 2, 2))
        #expect(try SceneRay(origin: below, direction: SIMD3(0, 0, 1), maximumDistance: Float(2).nextDown).intersection(with: plane) == nil)
        #expect(try SceneRay(origin: below, direction: SIMD3(0, 0, -1)).intersection(with: plane) == nil)
        #expect(try SceneRay(origin: below, direction: SIMD3(1, 0, 0)).intersection(with: plane) == nil)
        let coplanar = try SceneRay(origin: SIMD3(1, 2, 2), direction: SIMD3(1, 0, 0), maximumDistance: 0)
        #expect(try coplanar.intersection(with: plane)?.position == coplanar.origin)
        #expect(try coplanar.intersection(with: plane)?.distance == 0)
        let tilted = try Plane3D(normal: SIMD3(1, 2, 3), constant: 0)
        let tiltedStart = SIMD3<Float>(-5, 1, 1)
        let tiltedRay = try SceneRay(origin: tiltedStart, direction: SIMD3(-2, 1, 0), maximumDistance: 0)
        #expect(tilted.signedDistance(to: tiltedStart) == 0)
        #expect(try tiltedRay.intersection(with: tilted)?.position == tiltedStart)
        #expect(try tilted.intersection(from: tiltedStart, to: SIMD3(-7, 2, 1)) == tiltedStart)
    }

    @Test
    func `Triangle rays retain authored winding, weights, side and closed endpoints`() throws {
        let triangle = try sampleTriangle()
        let origin = SIMD3<Float>(1, 1.5, 2)
        let direction = SIMD3<Float>(0, 0, -8)
        let ray = try SceneRay(origin: origin, direction: direction, maximumDistance: 2)
        let hit = try #require(try ray.intersection(with: triangle, side: .front))
        #expect(hit.distance == 2)
        #expect(hit.position == SIMD3(1, 1.5, 0))
        #expect(hit.barycentric == SIMD3(0.25, 0.25, 0.5))
        #expect(hit.normal == SIMD3(0, 0, 1))
        #expect(hit.isFrontFace)
        #expect(try ray.intersection(with: triangle, side: .back) == nil)

        let rear = try SceneRay(origin: SIMD3(1, 1.5, -2), direction: -direction, maximumDistance: 2)
        #expect(try rear.intersection(with: triangle, side: .front) == nil)
        let rearHit = try #require(try rear.intersection(with: triangle, side: .back))
        #expect(rearHit.normal == hit.normal)
        #expect(!rearHit.isFrontFace)
        #expect(try rear.intersection(with: triangle) == rearHit)

        let short = try SceneRay(origin: origin, direction: direction, maximumDistance: Float(2).nextDown)
        #expect(try short.intersection(with: triangle) == nil)
        #expect(try SceneRay(origin: hit.position, direction: direction, maximumDistance: 0).intersection(with: triangle)?.distance == 0)
        #expect(try SceneRay(origin: hit.position, direction: SIMD3(1, 0, 0)).intersection(with: triangle) == nil)
        #expect(try SceneRay(origin: SIMD3(3, 2, 2), direction: direction).intersection(with: triangle) == nil)
        let degenerate = try Triangle3D(.zero, SIMD3(1, 1, 0), SIMD3(2, 2, 0))
        #expect(try ray.intersection(with: degenerate) == nil)
    }

    @Test
    func `Triangle edge tests share boundaries at large and small scales`() throws {
        let base = SIMD3<Float>(repeating: 100_000_000)
        let diagonal = base + SIMD3(32, 32, 0)
        let lower = try Triangle3D(base, base + SIMD3(32, 0, 0), diagonal)
        let upper = try Triangle3D(base, diagonal, base + SIMD3(0, 32, 0))
        let edge = try SceneRay(origin: base + SIMD3(16, 16, 64), direction: SIMD3(0, 0, -1), maximumDistance: 64)
        #expect(try edge.intersection(with: lower)?.barycentric == SIMD3(0.5, 0, 0.5))
        #expect(try edge.intersection(with: upper)?.barycentric == SIMD3(0.5, 0.5, 0))
        let oblique = try SceneRay(origin: base + SIMD3(0, 8, 32), direction: SIMD3(2, 1, -4))
        let obliqueLower = try oblique.intersection(with: lower)
        let obliqueUpper = try oblique.intersection(with: upper)
        #expect(obliqueLower != nil || obliqueUpper != nil)

        let tiny: Float = 1e-20
        let triangle = try Triangle3D(.zero, SIMD3(tiny, 0, 0), SIMD3(0, tiny, 0))
        let ray = try SceneRay(origin: SIMD3(tiny / 4, tiny / 4, tiny), direction: SIMD3(0, 0, -1))
        #expect(try ray.intersection(with: triangle)?.barycentric == SIMD3(0.5, 0.25, 0.25))
    }

    private func sampleTriangle() throws -> Triangle3D {
        try Triangle3D(.zero, SIMD3(4, 0, 0), SIMD3(0, 3, 0))
    }
}
