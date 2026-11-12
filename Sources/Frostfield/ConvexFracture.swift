import Foundation
import simd

/// Cuts a convex mesh the way Three.js `ConvexObjectBreaker` does.
///
/// `cut(_:by:)` splits one body with a world plane. Points on the normal's
/// positive side form one piece and points on the negative side form the other.
/// A piece is kept only when the cut produces more than four points, then rebuilt
/// with ``ConvexHull``. Its origin is the centroid of those points, transformed by
/// the body's rotation. Mass is halved. The piece can break again when twice the
/// largest positive offset from that centroid exceeds ``minimumBreakSize``.
/// `subdivideByImpact` repeats that cut around the impact, using the supplied
/// random values in place of `Math.random`.
public struct ConvexFracture: Sendable, Equatable {
    public var minimumBreakSize: Float
    public var planeTolerance: Float

    public init(minimumBreakSize: Float = 1.4, planeTolerance: Float = 0.0001) throws {
        guard minimumBreakSize.isFinite, minimumBreakSize > 0, planeTolerance.isFinite, planeTolerance >= 0 else {
            throw Failure.parameters
        }
        self.minimumBreakSize = minimumBreakSize
        self.planeTolerance = planeTolerance
    }

    /// A world plane `normal · x + constant = 0`, with a unit normal.
    public struct Plane: Sendable, Equatable {
        public var normal: SIMD3<Float>
        public var constant: Float

        init(normal: SIMD3<Float>, constant: Float) {
            self.normal = normal
            self.constant = constant
        }

        public init(normal: SIMD3<Float>, through point: SIMD3<Float>) throws {
            let length = simd_length(normal)
            guard length.isFinite, length > 0, point.x.isFinite, point.y.isFinite, point.z.isFinite else {
                throw Failure.plane
            }
            let unit = normal / length
            self.normal = unit
            self.constant = -simd_dot(unit, point)
        }

        func distance(to point: SIMD3<Float>) -> Float {
            simd_dot(normal, point) + constant
        }
    }

    public struct Body: Sendable {
        public var geometry: MeshGeometry
        public var position: SIMD3<Float>
        public var rotation: simd_quatf
        public var mass: Float
        public var velocity: SIMD3<Float>
        public var angularVelocity: SIMD3<Float>
        public var breakable: Bool

        public init(
            geometry: MeshGeometry, position: SIMD3<Float> = .zero,
            rotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)),
            mass: Float, velocity: SIMD3<Float> = .zero, angularVelocity: SIMD3<Float> = .zero,
            breakable: Bool = true
        ) {
            self.geometry = geometry
            self.position = position
            self.rotation = rotation
            self.mass = mass
            self.velocity = velocity
            self.angularVelocity = angularVelocity
            self.breakable = breakable
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case parameters
        case plane
        case body
        case impact

        public var errorDescription: String? {
            switch self {
            case .parameters: "The break size must be positive and the plane tolerance nonnegative."
            case .plane: "The cutting plane needs a finite nonzero normal and a finite point."
            case .body: "A breakable body needs a positive mass, a finite pose, and an indexed triangle mesh."
            case .impact: "The impact point, normal and iteration counts must be finite, and the first cut plane must have area."
            }
        }
    }

    /// Splits `body` by a world plane. Either piece is nil when that side has four points or fewer.
    public func cut(_ body: Body, by plane: Plane) throws -> (negative: Body?, positive: Body?) {
        let local = try localPlane(plane, for: body)
        var negative: [SIMD3<Float>] = []
        var positive: [SIMD3<Float>] = []
        var seen: Set<UInt64> = []
        let geometry = body.geometry
        guard geometry.indices.count.isMultiple(of: 3) else { throw Failure.body }
        for face in stride(from: 0, to: geometry.indices.count, by: 3) {
            let corners = [geometry.indices[face], geometry.indices[face + 1], geometry.indices[face + 2]]
            for segment in 0..<3 {
                let a = corners[segment]
                let b = corners[(segment + 1) % 3]
                let key = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
                guard seen.insert(key).inserted else { continue }
                classify(a, b, in: geometry, plane: local, negative: &negative, positive: &positive)
            }
        }
        return (
            try piece(from: negative, of: body),
            try piece(from: positive, of: body)
        )
    }

    /// Cuts around an impact. `random` returns values in 0..<1. Zero radial and random
    /// iterations still make the first cut, then keep both pieces.
    public func subdivideByImpact(
        _ body: Body, at impact: SIMD3<Float>, normal: SIMD3<Float>,
        radialIterations: Int = 0, randomIterations: Int = 0, random: () -> Float = { Float.random(in: 0..<1) }
    ) throws -> [Body] {
        guard impact.x.isFinite, impact.y.isFinite, impact.z.isFinite,
              normal.x.isFinite, normal.y.isFinite, normal.z.isFinite, simd_length(normal) > 0,
              radialIterations >= 0, randomIterations >= 0 else { throw Failure.impact }
        let unit = normal / simd_length(normal)
        let first = try plane(impact, body.position, impact + unit)
        var debris: [Body] = []
        try subdivide(
            body, start: 0, end: 2 * .pi, iteration: 0, root: body, impact: impact, normal: unit,
            first: first, radialIterations: radialIterations, randomIterations: randomIterations,
            random: random, debris: &debris
        )
        return debris
    }

    private func subdivide(
        _ body: Body, start: Float, end: Float, iteration: Int, root: Body, impact: SIMD3<Float>,
        normal: SIMD3<Float>, first: Plane, radialIterations: Int, randomIterations: Int,
        random: () -> Float, debris: inout [Body]
    ) throws {
        let total = radialIterations + randomIterations
        if random() < Float(iteration) * 0.05 || iteration > total {
            debris.append(body)
            return
        }
        var angle = Float.pi
        let plane: Plane
        if iteration == 0 {
            plane = first
        } else if iteration <= radialIterations {
            angle = (end - start) * (0.2 + 0.6 * random()) + start
            let rotated = impact + rotate(root.position - impact, around: normal, by: angle)
            plane = try self.plane(impact, impact + normal, rotated)
        } else {
            angle = (0.5 * Float(iteration & 1) + 0.2 * (2 - random())) * .pi
            let rotated = body.position + rotate(impact - body.position, around: normal, by: angle)
            plane = try self.plane(body.position, body.position + normal, rotated)
        }
        let pieces = try cut(body, by: plane)
        if let negative = pieces.negative {
            try subdivide(
                negative, start: start, end: angle, iteration: iteration + 1, root: root, impact: impact,
                normal: normal, first: first, radialIterations: radialIterations, randomIterations: randomIterations,
                random: random, debris: &debris
            )
        }
        if let positive = pieces.positive {
            try subdivide(
                positive, start: angle, end: end, iteration: iteration + 1, root: root, impact: impact,
                normal: normal, first: first, radialIterations: radialIterations, randomIterations: randomIterations,
                random: random, debris: &debris
            )
        }
    }

    private func localPlane(_ plane: Plane, for body: Body) throws -> Plane {
        guard body.mass.isFinite, body.mass > 0, geometryIsUsable(body.geometry),
              body.position.x.isFinite, body.position.y.isFinite, body.position.z.isFinite,
              simd_length(body.rotation.vector) > 0 || body.rotation.real.isFinite else { throw Failure.body }
        let localNormal = body.rotation.inverse.act(plane.normal)
        return Plane(normal: localNormal, constant: simd_dot(plane.normal, body.position) + plane.constant)
    }

    private func classify(
        _ a: UInt32, _ b: UInt32, in geometry: MeshGeometry, plane: Plane,
        negative: inout [SIMD3<Float>], positive: inout [SIMD3<Float>]
    ) {
        let p0 = geometry.vertices[Int(a)].position
        let p1 = geometry.vertices[Int(b)].position
        let d0 = plane.distance(to: p0)
        let d1 = plane.distance(to: p1)
        func keep(_ point: SIMD3<Float>, _ distance: Float) {
            if distance > planeTolerance {
                positive.append(point)
            } else if distance < -planeTolerance {
                negative.append(point)
            } else {
                negative.append(point)
                positive.append(point)
            }
        }
        keep(p0, d0)
        keep(p1, d1)
        let crosses = (d0 > planeTolerance && d1 < -planeTolerance) || (d1 > planeTolerance && d0 < -planeTolerance)
        if crosses, d1 != d0 {
            let t = -d0 / (d1 - d0)
            let point = p0 + (p1 - p0) * t
            negative.append(point)
            positive.append(point)
        }
    }

    private func piece(from points: [SIMD3<Float>], of body: Body) throws -> Body? {
        guard points.count > 4 else { return nil }
        var centroid = SIMD3<Float>.zero
        for point in points { centroid += point }
        centroid /= Float(points.count)
        var radius: Float = 0
        let centered = points.map { point -> SIMD3<Float> in
            let relative = point - centroid
            radius = max(radius, relative.x, relative.y, relative.z)
            return relative
        }
        let geometry: MeshGeometry
        do { geometry = try MeshGeometry.convexHull(centered) } catch { return nil }
        return Body(
            geometry: geometry, position: body.position + body.rotation.act(centroid), rotation: body.rotation,
            mass: body.mass * 0.5, velocity: body.velocity, angularVelocity: body.angularVelocity,
            breakable: 2 * radius > minimumBreakSize
        )
    }

    private func plane(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) throws -> Plane {
        let normal = simd_cross(b - a, c - a)
        guard simd_length(normal) > 1e-8 else { throw Failure.impact }
        return try Plane(normal: normal, through: a)
    }

    private func rotate(_ vector: SIMD3<Float>, around axis: SIMD3<Float>, by angle: Float) -> SIMD3<Float> {
        simd_quatf(angle: angle, axis: axis).act(vector)
    }

    private func geometryIsUsable(_ geometry: MeshGeometry) -> Bool {
        geometry.indices.count >= 3 && geometry.indices.count.isMultiple(of: 3)
            && geometry.indices.allSatisfy { Int($0) < geometry.vertices.count }
    }
}
