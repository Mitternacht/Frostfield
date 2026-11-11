import Foundation
import simd

/// The convex hull of a point set, built with Quickhull in Double precision.
///
/// The build starts from the largest tetrahedron it can find among the
/// extreme points, then repeatedly adds the point farthest outside a face,
/// replacing every face that point can see with a fan to its horizon.
/// A point counts as outside a face only when it is more than ``tolerance``
/// beyond the face's plane, so points on or within that distance of the hull
/// are not vertices and nearly coplanar faces stay convex to within the
/// tolerance. Vertices that may lie in a hull edge or face are removed only
/// when rebuilding without them still contains the input within the tolerance.
/// [Barber, Dobkin and Huhdanpaa 1996](https://doi.org/10.1145/235815.235821)
public struct ConvexHull: Sendable {
    /// The input points, which triangles index.
    public let points: [SIMD3<Float>]
    /// Outward-facing triangles that wind counterclockwise seen from outside.
    public let triangles: [SIMD3<Int>]
    /// The distance beyond a face plane at which a point counts as outside:
    /// 64 ulps of the sum of the largest absolute coordinate on each axis.
    public let tolerance: Double
    private let planes: [SIMD4<Double>]

    /// Builds the hull of at least four finite points that do not all lie on
    /// one plane. Duplicate and interior points are allowed.
    public init(points: [SIMD3<Float>]) throws {
        guard points.count >= 4 else { throw Failure.tooFewPoints }
        if let invalid = points.firstIndex(where: { !($0.x.isFinite && $0.y.isFinite && $0.z.isFinite) }) {
            throw Failure.nonFinitePoint(invalid)
        }
        let positions = points.map(SIMD3<Double>.init)
        let magnitude = positions.reduce(SIMD3<Double>.zero) { simd_max($0, simd_abs($1)) }
        let tolerance = 64 * Double.ulpOfOne * simd_reduce_add(magnitude)
        var builder = QuickhullBuilder(positions: positions, tolerance: tolerance, active: Array(positions.indices))
        try builder.run()
        var faces = builder.faces.filter(\.alive)
        for _ in 1..<4 {
            let redundant = QuickhullBuilder.nonExtremeVertices(of: faces)
            guard !redundant.isEmpty else { break }
            let active = Set(faces.flatMap { [$0.vertices.x, $0.vertices.y, $0.vertices.z] }).subtracting(redundant).sorted()
            guard active.count >= 4 else { break }
            var simplified = QuickhullBuilder(positions: positions, tolerance: tolerance, active: active)
            guard (try? simplified.run()) != nil else { break }
            let candidate = simplified.faces.filter(\.alive)
            // The normal-cone heuristic also flags very shallow genuine corners.
            guard candidate.allSatisfy({ face in
                positions.allSatisfy { simd_dot(face.normal, $0) - face.offset <= tolerance }
            }) else { break }
            faces = candidate
        }
        self.points = points
        triangles = faces.map(\.vertices)
        planes = faces.map { SIMD4($0.normal, $0.offset) }
        self.tolerance = tolerance
    }

    /// The indices of the points that are hull vertices, in ascending order.
    public var vertexIndices: [Int] {
        Set(triangles.flatMap { [$0.x, $0.y, $0.z] }).sorted()
    }

    /// The enclosed volume, from the divergence theorem over the triangles.
    public var volume: Double {
        triangles.reduce(0) { sum, triangle in
            let a = SIMD3<Double>(points[triangle.x]), b = SIMD3<Double>(points[triangle.y])
            let c = SIMD3<Double>(points[triangle.z])
            return sum + simd_dot(a, simd_cross(b, c)) / 6
        }
    }

    /// The largest signed distance from a face plane to the point: positive
    /// outside the hull, zero on it and negative inside.
    public func signedDistance(to point: SIMD3<Float>) -> Double {
        let p = SIMD3<Double>(point)
        return planes.reduce(-Double.infinity) { max($0, simd_dot(SIMD3($1.x, $1.y, $1.z), p) - $1.w) }
    }

    /// Whether the point lies inside or within `extraTolerance` plus the
    /// hull's own ``tolerance`` of its surface.
    public func contains(_ point: SIMD3<Float>, extraTolerance: Double = 0) -> Bool {
        signedDistance(to: point) <= tolerance + extraTolerance
    }

    /// Flat-shaded triangles with one normal per face. UVs project each face
    /// along its dominant axis onto the hull's bounding box, with V
    /// increasing downward on side faces, and tangents follow the UVs.
    public func geometry() throws -> MeshGeometry {
        let corners = vertexIndices.map { SIMD3<Double>(points[$0]) }
        let lower = corners.reduce(corners[0]) { simd_min($0, $1) }
        let upper = corners.reduce(corners[0]) { simd_max($0, $1) }
        let extent = simd_max(upper - lower, SIMD3(repeating: .leastNormalMagnitude))
        var vertices: [MeshVertex] = []
        vertices.reserveCapacity(triangles.count * 3)
        for (face, triangle) in triangles.enumerated() {
            let normal = SIMD3(planes[face].x, planes[face].y, planes[face].z)
            let axis = simd_abs(normal)
            let positions = [triangle.x, triangle.y, triangle.z].map { SIMD3<Double>(points[$0]) }
            let uvs = positions.map { p -> SIMD2<Double> in
                let t = (p - lower) / extent
                if axis.x >= axis.y && axis.x >= axis.z { return SIMD2(t.z, 1 - t.y) }
                if axis.y >= axis.z { return SIMD2(t.x, t.z) }
                return SIMD2(t.x, 1 - t.y)
            }
            let tangent = surfaceTangent(positions: positions, uvs: uvs, normal: normal)
            for (position, uv) in zip(positions, uvs) {
                vertices.append(MeshVertex(
                    position: SIMD3<Float>(position), normal: SIMD3<Float>(normal),
                    textureCoordinate: SIMD2<Float>(uv), tangent: tangent
                ))
            }
        }
        return try MeshGeometry(vertices: vertices, indices: (0..<UInt32(vertices.count)).map { $0 })
    }

    public enum Failure: LocalizedError, Equatable {
        case tooFewPoints
        case nonFinitePoint(Int)
        case degenerate
        case numericalFailure

        public var errorDescription: String? {
            switch self {
            case .tooFewPoints: "A convex hull needs at least four points."
            case .nonFinitePoint(let index): "Hull point \(index) is not finite."
            case .degenerate: "The points lie on one plane, line or point, so they enclose no volume."
            case .numericalFailure:
                "The hull could not be completed at Double precision; the input has points within the tolerance of many faces."
            }
        }
    }
}

extension MeshGeometry {
    /// A flat-shaded convex hull of the points. See ``ConvexHull``.
    public static func convexHull(_ points: [SIMD3<Float>]) throws -> MeshGeometry {
        try ConvexHull(points: points).geometry()
    }
}

/// A unit tangent along increasing U, orthogonal to the normal, with handedness
/// from the direction of increasing V.
func surfaceTangent(positions: [SIMD3<Double>], uvs: [SIMD2<Double>], normal: SIMD3<Double>) -> SIMD4<Float> {
    let e1 = positions[1] - positions[0], e2 = positions[2] - positions[0]
    let d1 = uvs[1] - uvs[0], d2 = uvs[2] - uvs[0]
    let determinant = d1.x * d2.y - d2.x * d1.y
    var direction = SIMD3<Double>(1, 0, 0)
    var bitangent = SIMD3<Double>(0, 1, 0)
    if determinant != 0 {
        direction = (e1 * d2.y - e2 * d1.y) / determinant
        bitangent = (e2 * d1.x - e1 * d2.x) / determinant
    }
    var tangent = direction - normal * simd_dot(direction, normal)
    if simd_length_squared(tangent) < 1e-24 {
        let helper = abs(normal.x) < 0.9 ? SIMD3<Double>(1, 0, 0) : SIMD3<Double>(0, 1, 0)
        tangent = helper - normal * simd_dot(helper, normal)
    }
    tangent = simd_normalize(tangent)
    let handedness: Float = simd_dot(simd_cross(normal, tangent), bitangent) < 0 ? -1 : 1
    return SIMD4(SIMD3<Float>(tangent), handedness)
}

private struct QuickhullBuilder {
    struct Face {
        var vertices: SIMD3<Int>
        var normal: SIMD3<Double>
        var offset: Double
        var outside: [Int] = []
        var alive = true
    }

    struct Edge: Hashable {
        let from: Int
        let to: Int
    }

    let positions: [SIMD3<Double>]
    let tolerance: Double
    /// The points this build considers.
    let active: [Int]
    var faces: [Face] = []
    var edges: [Edge: Int] = [:]

    /// Candidates for removal whose incident face normals nearly fail to span
    /// three dimensions. A nearly flat but extreme vertex can also qualify.
    static func nonExtremeVertices(of faces: [Face]) -> Set<Int> {
        var normals: [Int: [SIMD3<Double>]] = [:]
        for face in faces {
            for vertex in [face.vertices.x, face.vertices.y, face.vertices.z] { normals[vertex, default: []].append(face.normal) }
        }
        var redundant = Set<Int>()
        for (vertex, list) in normals {
            var spans = false
            search: for i in list.indices {
                for j in (i + 1)..<list.count {
                    let cross = simd_cross(list[i], list[j])
                    guard simd_length(cross) > 1e-9 else { continue }
                    for k in (j + 1)..<list.count where abs(simd_dot(cross, list[k])) > 1e-9 {
                        spans = true
                        break search
                    }
                }
            }
            if !spans { redundant.insert(vertex) }
        }
        return redundant
    }

    func distance(_ point: Int, to face: Face) -> Double {
        simd_dot(face.normal, positions[point]) - face.offset
    }

    mutating func makeFace(_ a: Int, _ b: Int, _ c: Int) throws -> Int {
        let pa = positions[a], pb = positions[b], pc = positions[c]
        let cross = simd_cross(pb - pa, pc - pa)
        let length = simd_length(cross)
        guard length > 0, length.isFinite else { throw ConvexHull.Failure.numericalFailure }
        let normal = cross / length
        // Average the corners so the plane does not favor one of them.
        let offset = simd_dot(normal, (pa + pb + pc) / 3)
        let index = faces.count
        faces.append(Face(vertices: SIMD3(a, b, c), normal: normal, offset: offset))
        for edge in [Edge(from: a, to: b), Edge(from: b, to: c), Edge(from: c, to: a)] {
            guard edges.updateValue(index, forKey: edge) == nil else { throw ConvexHull.Failure.numericalFailure }
        }
        return index
    }

    mutating func run() throws {
        let simplex = try initialSimplex()
        let used = Set(simplex)
        var created: [Int] = []
        let triples = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)]
        for (i, j, k, opposite) in triples {
            var a = simplex[i], b = simplex[j]
            let c = simplex[k]
            let pa = positions[a], normal = simd_cross(positions[b] - pa, positions[c] - pa)
            if simd_dot(normal, positions[simplex[opposite]] - pa) > 0 { swap(&a, &b) }
            created.append(try makeFace(a, b, c))
        }
        assign(active.filter { !used.contains($0) }, to: created)

        var pending = created
        while let faceIndex = pending.popLast() {
            guard faces[faceIndex].alive, !faces[faceIndex].outside.isEmpty else { continue }
            let face = faces[faceIndex]
            let v = face.vertices
            let eye = extreme(
                among: face.outside, measure: { distance($0, to: face) },
                center: (positions[v.x] + positions[v.y] + positions[v.z]) / 3
            )

            // Faces the eye sees, flood-filled from this one, and the horizon around them.
            var visible: [Int: Bool] = [faceIndex: true]
            var queue = [faceIndex]
            var horizon: [Edge] = []
            var cursor = 0
            while cursor < queue.count {
                let current = faces[queue[cursor]]
                cursor += 1
                let v = current.vertices
                for edge in [Edge(from: v.x, to: v.y), Edge(from: v.y, to: v.z), Edge(from: v.z, to: v.x)] {
                    guard let neighbor = edges[Edge(from: edge.to, to: edge.from)] else {
                        throw ConvexHull.Failure.numericalFailure
                    }
                    let sees: Bool
                    if let known = visible[neighbor] {
                        sees = known
                    } else {
                        sees = distance(eye, to: faces[neighbor]) > tolerance
                        visible[neighbor] = sees
                        if sees { queue.append(neighbor) }
                    }
                    if !sees { horizon.append(edge) }
                }
            }

            var orphans: [Int] = []
            for index in queue {
                faces[index].alive = false
                orphans += faces[index].outside
                faces[index].outside = []
                let v = faces[index].vertices
                for edge in [Edge(from: v.x, to: v.y), Edge(from: v.y, to: v.z), Edge(from: v.z, to: v.x)] {
                    edges[edge] = nil
                }
            }
            var fan: [Int] = []
            for edge in horizon {
                fan.append(try makeFace(edge.from, edge.to, eye))
            }
            assign(orphans.filter { $0 != eye }, to: fan)
            pending += fan
        }
        try validate()
    }

    /// The point with the largest measure. Points within the tolerance of the
    /// largest tie, and the one farthest from `center` wins: a strictly convex
    /// function peaks at a corner of the tied set, so a point in the middle of
    /// a hull edge or face never becomes a vertex.
    func extreme(among candidates: some Collection<Int>, measure: (Int) -> Double, center: SIMD3<Double>) -> Int {
        let largest = candidates.map(measure).max()!
        return candidates.filter { measure($0) >= largest - tolerance }.max {
            simd_distance_squared(positions[$0], center) < simd_distance_squared(positions[$1], center)
        }!
    }

    /// Gives each point to the new face it lies farthest outside, if any.
    mutating func assign(_ points: [Int], to candidates: [Int]) {
        for point in points {
            var best: Int?
            var bestDistance = tolerance
            for face in candidates {
                let d = distance(point, to: faces[face])
                if d > bestDistance {
                    bestDistance = d
                    best = face
                }
            }
            if let best { faces[best].outside.append(point) }
        }
    }

    /// The two extreme points on the widest axis, the farthest point from
    /// their line, and the farthest point from that plane.
    func initialSimplex() throws -> [Int] {
        var lower = positions[active[0]], upper = positions[active[0]]
        for p in active.map({ positions[$0] }) {
            lower = simd_min(lower, p)
            upper = simd_max(upper, p)
        }
        let extent = upper - lower
        let axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2)
        let a = extreme(among: active, measure: { -self.positions[$0][axis] }, center: (lower + upper) / 2)
        let b = extreme(among: active, measure: { self.positions[$0][axis] }, center: positions[a])
        guard positions[b][axis] - positions[a][axis] > tolerance else { throw ConvexHull.Failure.degenerate }
        let direction = simd_normalize(positions[b] - positions[a])
        let c = extreme(
            among: active, measure: { simd_length(simd_cross(self.positions[$0] - self.positions[a], direction)) },
            center: (positions[a] + positions[b]) / 2
        )
        guard simd_length(simd_cross(positions[c] - positions[a], direction)) > tolerance else {
            throw ConvexHull.Failure.degenerate
        }
        let normal = simd_normalize(simd_cross(positions[b] - positions[a], positions[c] - positions[a]))
        let d = extreme(
            among: active, measure: { abs(simd_dot(normal, self.positions[$0] - self.positions[a])) },
            center: (positions[a] + positions[b] + positions[c]) / 3
        )
        guard abs(simd_dot(normal, positions[d] - positions[a])) > tolerance else { throw ConvexHull.Failure.degenerate }
        return [a, b, c, d]
    }

    /// Every edge has one opposite twin, and every face's neighbors lie on or behind its plane.
    func validate() throws {
        for (edge, faceIndex) in edges {
            guard faces[faceIndex].alive, let twin = edges[Edge(from: edge.to, to: edge.from)], faces[twin].alive else {
                throw ConvexHull.Failure.numericalFailure
            }
            let other = faces[twin].vertices
            let apex = [other.x, other.y, other.z].first { $0 != edge.from && $0 != edge.to }!
            guard distance(apex, to: faces[faceIndex]) <= 4 * tolerance else { throw ConvexHull.Failure.numericalFailure }
        }
    }
}
