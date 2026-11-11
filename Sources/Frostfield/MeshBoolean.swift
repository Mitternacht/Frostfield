import Foundation
import simd

/// A boolean combination of two solids.
public enum MeshBooleanOperation: Sendable, Equatable {
    /// Space inside either solid.
    case union
    /// Space inside the first solid and outside the second.
    case subtraction
    /// Space inside both solids.
    case intersection
}

/// A closed mesh placed in the space where a boolean operation runs.
public struct MeshBooleanOperand: Sendable {
    public var geometry: MeshGeometry
    /// Maps the geometry's positions into the operation's space.
    public var transform: simd_float4x4

    public init(_ geometry: MeshGeometry, transform: simd_float4x4 = matrix_identity_float4x4) {
        self.geometry = geometry
        self.transform = transform
    }
}

/// Constructive solid geometry on closed, consistently oriented triangle meshes.
///
/// The evaluator follows the classic split-and-classify method. Each operand's
/// triangles are cut along the curve where the two surfaces meet, so every
/// resulting piece lies entirely inside, outside or on the other surface.
/// Pieces are grouped into patches bounded by that curve and each patch is
/// classified once: on the other surface it is *same* or *opposite* by its
/// normal, otherwise inside or outside by the generalized winding number,
/// which stays exact for closed meshes whatever a ray would graze.
/// [Laidlaw, Trumbore and Hughes 1986](https://doi.org/10.1145/15886.15904),
/// [Jacobson, Kavan and Sorkine-Hornung 2013](https://doi.org/10.1145/2461912.2461916)
///
/// Intersection points are computed once per edge and plane in Double and
/// shared by every triangle that uses them, and edges carry the same split
/// points on both sides, so the result is watertight by position. A
/// ``MeshBVH`` over the second operand limits the triangle pairs tested.
///
/// Precision: all geometry is computed in Double from Float inputs. Points
/// closer than the tolerance, ``relativeTolerance`` times the larger of the
/// operands' combined bounding-box extent and largest absolute coordinate,
/// are one point, and a vertex within that distance of a plane lies on it.
/// That is about eight Float ulps, so surfaces placed to coincide by Float
/// transforms still meet exactly. Edges shorter than the tolerance collapse
/// and their triangles are dropped; a triangle with numerically zero area is
/// reported. Inputs must be closed two-manifolds without self-intersections,
/// with outward-facing counterclockwise triangles; separate closed components
/// are allowed. A result that would touch itself along an edge or vertex is
/// reported rather than returned.
public enum MeshBoolean {
    /// The welding and coplanarity distance relative to the operands' scale.
    public static let relativeTolerance = 1e-6

    /// Combines two closed meshes. Output positions are in the operation's
    /// space. Normals, both UV sets and tangents interpolate across split
    /// triangles from each piece's source triangle; secondary UVs and tangents
    /// are kept only when every input vertex has them. Faces of the second
    /// operand kept by a subtraction are reversed, with negated normals and
    /// tangent handedness.
    public static func evaluate(
        _ operation: MeshBooleanOperation, _ first: MeshBooleanOperand, _ second: MeshBooleanOperand
    ) throws -> MeshGeometry {
        var evaluator = try BooleanEvaluator(first, second)
        try evaluator.splitSurfaces()
        return try evaluator.assemble(operation)
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidTransform(operand: Int)
        case degenerateTriangle(operand: Int, triangle: Int)
        case openBoundary(operand: Int)
        case nonManifoldEdge(operand: Int)
        case invertedOrientation(operand: Int)
        case triangulation(operand: Int, triangle: Int)
        case nonManifoldResult
        case emptyResult

        public var errorDescription: String? {
            switch self {
            case .invalidTransform(let operand): "Boolean operand \(operand) needs a finite, invertible transform."
            case .degenerateTriangle(let operand, let triangle):
                "Triangle \(triangle) of boolean operand \(operand) has three distinct corners but no area."
            case .openBoundary(let operand):
                "Boolean operand \(operand) is not closed: an edge belongs to only one triangle."
            case .nonManifoldEdge(let operand):
                "Boolean operand \(operand) is not a consistently oriented manifold: an edge is shared by more than two triangles or by two with the same direction."
            case .invertedOrientation(let operand):
                "Boolean operand \(operand) encloses negative volume; its triangles must face outward."
            case .triangulation(let operand, let triangle):
                "Triangle \(triangle) of boolean operand \(operand) could not be split along the intersection."
            case .nonManifoldResult:
                "The result is not a closed manifold: the solids touch along an edge or vertex, or the inputs intersect themselves."
            case .emptyResult: "The boolean result encloses no volume."
            }
        }
    }
}

extension MeshGeometry {
    /// The space inside this mesh or `other`. See ``MeshBoolean``.
    public func union(_ other: MeshGeometry) throws -> MeshGeometry {
        try MeshBoolean.evaluate(.union, MeshBooleanOperand(self), MeshBooleanOperand(other))
    }

    /// The space inside this mesh and outside `other`. See ``MeshBoolean``.
    public func subtracting(_ other: MeshGeometry) throws -> MeshGeometry {
        try MeshBoolean.evaluate(.subtraction, MeshBooleanOperand(self), MeshBooleanOperand(other))
    }

    /// The space inside both this mesh and `other`. See ``MeshBoolean``.
    public func intersection(_ other: MeshGeometry) throws -> MeshGeometry {
        try MeshBoolean.evaluate(.intersection, MeshBooleanOperand(self), MeshBooleanOperand(other))
    }

    /// The volume enclosed by a closed, outward-facing mesh, from the
    /// divergence theorem; negative when the triangles face inward.
    public var signedVolume: Double {
        var sum = 0.0
        for offset in stride(from: 0, to: indices.count, by: 3) {
            let a = SIMD3<Double>(vertices[Int(indices[offset])].position)
            let b = SIMD3<Double>(vertices[Int(indices[offset + 1])].position)
            let c = SIMD3<Double>(vertices[Int(indices[offset + 2])].position)
            sum += simd_dot(a, simd_cross(b, c))
        }
        return sum / 6
    }
}

private struct UndirectedEdge: Hashable {
    let low: Int
    let high: Int
    init(_ a: Int, _ b: Int) {
        low = min(a, b)
        high = max(a, b)
    }
}

private struct DirectedEdge: Hashable {
    let from: Int
    let to: Int
}

/// Positions merged within a distance, found through a uniform hash grid.
private struct WeldingPool {
    var positions: [SIMD3<Double>] = []
    var cells: [SIMD3<Int>: [Int]] = [:]
    let tolerance: Double

    func cell(_ point: SIMD3<Double>) -> SIMD3<Int> {
        SIMD3(Int((point.x / tolerance).rounded(.down)), Int((point.y / tolerance).rounded(.down)),
              Int((point.z / tolerance).rounded(.down)))
    }

    mutating func insert(_ point: SIMD3<Double>) -> Int {
        let center = cell(point)
        var best: Int?
        var bestDistance = tolerance
        for dx in -1...1 { for dy in -1...1 { for dz in -1...1 {
            for id in cells[center &+ SIMD3(dx, dy, dz)] ?? [] {
                let distance = simd_distance(positions[id], point)
                if distance <= bestDistance {
                    bestDistance = distance
                    best = id
                }
            }
        } } }
        if let best { return best }
        positions.append(point)
        cells[center, default: []].append(positions.count - 1)
        return positions.count - 1
    }
}

/// One operand with welded corners, planes and a hierarchy for box queries.
private struct BooleanSolid {
    let geometry: MeshGeometry
    let operand: Int
    /// World positions of the original vertices.
    let worldPositions: [SIMD3<Double>]
    let normalMatrix: simd_double3x3
    let linear: simd_double3x3
    let mirrors: Bool
    /// The geometry's triangle index for each kept triangle.
    var sources: [Int] = []
    /// Original vertex indices of each triangle, reordered when the transform mirrors.
    var slots: [SIMD3<Int>] = []
    /// Pool ids of each triangle's corners, in the same order as `slots`.
    var corners: [SIMD3<Int>] = []
    /// Unit normal and offset of each triangle's plane.
    var planes: [SIMD4<Double>] = []
    var bvh: MeshBVH?
}

/// The pieces of one source triangle after splitting.
private struct Piece {
    let operand: Int
    let source: Int
    let corners: SIMD3<Int>
}

private enum PatchClass {
    case inside, outside, same, opposite
}

private struct BooleanEvaluator {
    var solids: [BooleanSolid]
    var pool: WeldingPool
    let tolerance: Double
    /// Constraint segments per operand and triangle, as pool-id pairs.
    var constraints: [[[SIMD2<Int>]]]
    var edgeSplits: [UndirectedEdge: Set<Int>] = [:]
    var pieces: [[Piece]] = [[], []]
    var curveEdges: [Set<UndirectedEdge>] = [[], []]

    init(_ first: MeshBooleanOperand, _ second: MeshBooleanOperand) throws {
        var prepared: [BooleanSolid] = []
        var lower = SIMD3<Double>(repeating: .infinity), upper = SIMD3<Double>(repeating: -.infinity)
        for (operand, input) in [first, second].enumerated() {
            let m = input.transform
            let world = simd_double4x4(columns: (
                SIMD4<Double>(m.columns.0), SIMD4<Double>(m.columns.1), SIMD4<Double>(m.columns.2), SIMD4<Double>(m.columns.3)
            ))
            let linear = simd_double3x3(
                SIMD3(world.columns.0.x, world.columns.0.y, world.columns.0.z),
                SIMD3(world.columns.1.x, world.columns.1.y, world.columns.1.z),
                SIMD3(world.columns.2.x, world.columns.2.y, world.columns.2.z)
            )
            let determinant = linear.determinant
            let finite = (0..<4).allSatisfy { column in (0..<4).allSatisfy { world[column][$0].isFinite } }
            guard finite, determinant.isFinite, determinant != 0,
                  world[0][3] == 0, world[1][3] == 0, world[2][3] == 0, world[3][3] == 1 else {
                throw MeshBoolean.Failure.invalidTransform(operand: operand)
            }
            let positions = input.geometry.vertices.map { vertex -> SIMD3<Double> in
                let p = world * SIMD4(SIMD3<Double>(vertex.position), 1)
                return SIMD3(p.x, p.y, p.z)
            }
            for p in positions {
                lower = simd_min(lower, p)
                upper = simd_max(upper, p)
            }
            prepared.append(BooleanSolid(
                geometry: input.geometry, operand: operand, worldPositions: positions,
                normalMatrix: linear.inverse.transpose, linear: linear, mirrors: determinant < 0
            ))
        }
        let scale = max(simd_reduce_max(upper - lower), simd_reduce_max(simd_max(simd_abs(lower), simd_abs(upper))))
        tolerance = MeshBoolean.relativeTolerance * max(scale, .leastNormalMagnitude)
        var weld = WeldingPool(tolerance: tolerance)

        for operand in 0..<2 {
            var solid = prepared[operand]
            let ids = solid.worldPositions.map { weld.insert($0) }
            let indices = solid.geometry.indices
            var directed: [DirectedEdge: Int] = [:]
            var volume = 0.0
            for triangle in 0..<(indices.count / 3) {
                var slot = SIMD3(Int(indices[triangle * 3]), Int(indices[triangle * 3 + 1]), Int(indices[triangle * 3 + 2]))
                if solid.mirrors { slot = SIMD3(slot.x, slot.z, slot.y) }
                let corner = SIMD3(ids[slot.x], ids[slot.y], ids[slot.z])
                // A triangle whose corners weld together collapses with its edge; its neighbours close the gap.
                guard corner.x != corner.y, corner.y != corner.z, corner.z != corner.x else { continue }
                let a = weld.positions[corner.x], b = weld.positions[corner.y], c = weld.positions[corner.z]
                let cross = simd_cross(b - a, c - a)
                let longest = max(simd_distance(a, b), simd_distance(b, c), simd_distance(c, a))
                guard simd_length(cross) > 1e-9 * tolerance * longest else {
                    throw MeshBoolean.Failure.degenerateTriangle(operand: operand, triangle: triangle)
                }
                let normal = simd_normalize(cross)
                solid.sources.append(triangle)
                solid.slots.append(slot)
                solid.corners.append(corner)
                solid.planes.append(SIMD4(normal, simd_dot(normal, (a + b + c) / 3)))
                volume += simd_dot(a, simd_cross(b, c))
                for (from, to) in [(corner.x, corner.y), (corner.y, corner.z), (corner.z, corner.x)] {
                    directed[DirectedEdge(from: from, to: to), default: 0] += 1
                }
            }
            for (edge, count) in directed {
                guard count == 1 else { throw MeshBoolean.Failure.nonManifoldEdge(operand: operand) }
                guard let reverse = directed[DirectedEdge(from: edge.to, to: edge.from)] else {
                    throw MeshBoolean.Failure.openBoundary(operand: operand)
                }
                guard reverse == 1 else { throw MeshBoolean.Failure.nonManifoldEdge(operand: operand) }
            }
            guard volume > 0 else { throw MeshBoolean.Failure.invertedOrientation(operand: operand) }
            prepared[operand] = solid
        }
        pool = weld
        solids = prepared
        constraints = prepared.map { [[SIMD2<Int>]](repeating: [], count: $0.corners.count) }
        for operand in 0..<2 {
            let solid = solids[operand]
            let corners = Array(Set(solid.corners.flatMap { [$0.x, $0.y, $0.z] })).sorted()
            var local: [Int: UInt32] = [:]
            var vertices: [MeshVertex] = []
            for id in corners {
                local[id] = UInt32(vertices.count)
                vertices.append(MeshVertex(position: SIMD3<Float>(pool.positions[id]), normal: SIMD3(0, 0, 1)))
            }
            let indices = solid.corners.flatMap { [local[$0.x]!, local[$0.y]!, local[$0.z]!] }
            solids[operand].bvh = try MeshBVH(geometry: try MeshGeometry(vertices: vertices, indices: indices))
        }
    }

    // MARK: Splitting

    mutating func splitSurfaces() throws {
        let first = solids[0], second = solids[1]
        let floatPadding = Float(tolerance * 4) + Float(1e-6) * max(
            simd_reduce_max(simd_abs(first.bvh!.bounds.maximum)), simd_reduce_max(simd_abs(second.bvh!.bounds.maximum)),
            simd_reduce_max(simd_abs(first.bvh!.bounds.minimum)), simd_reduce_max(simd_abs(second.bvh!.bounds.minimum)), 1
        )
        for t in first.corners.indices {
            let c = first.corners[t]
            let points = [pool.positions[c.x], pool.positions[c.y], pool.positions[c.z]]
            let lower = SIMD3<Float>(points.reduce(points[0]) { simd_min($0, $1) }) - floatPadding
            let upper = SIMD3<Float>(points.reduce(points[0]) { simd_max($0, $1) }) + floatPadding
            let box = try AxisAlignedBox(minimum: lower, maximum: upper)
            for u in second.bvh!.triangles(intersecting: box) {
                intersect(t, u)
            }
        }
        // A point that lies on an edge splits that edge in both triangles sharing it.
        for operand in 0..<2 {
            for triangle in constraints[operand].indices where !constraints[operand][triangle].isEmpty {
                let corner = solids[operand].corners[triangle]
                let ends = Set(constraints[operand][triangle].flatMap { [$0.x, $0.y] })
                for id in ends {
                    for (a, b) in [(corner.x, corner.y), (corner.y, corner.z), (corner.z, corner.x)]
                    where id != a && id != b && liesOnSegment(id, a, b) {
                        edgeSplits[UndirectedEdge(a, b), default: []].insert(id)
                    }
                }
            }
        }
        for operand in 0..<2 {
            for triangle in solids[operand].corners.indices {
                try retriangulate(operand, triangle)
            }
        }
    }

    private func distance(_ point: SIMD3<Double>, to plane: SIMD4<Double>) -> Double {
        let d = simd_dot(SIMD3(plane.x, plane.y, plane.z), point) - plane.w
        return abs(d) <= tolerance ? 0 : d
    }

    private func liesOnSegment(_ id: Int, _ a: Int, _ b: Int) -> Bool {
        let p = pool.positions[id], start = pool.positions[a], end = pool.positions[b]
        let edge = end - start
        let length = simd_length(edge)
        let t = simd_dot(p - start, edge) / (length * length)
        guard t * length > tolerance, (1 - t) * length > tolerance else { return false }
        return simd_length(p - (start + edge * t)) <= tolerance
    }

    /// Where an edge crosses a plane, computed from the lower pool id so every triangle gets the same point.
    private mutating func crossing(_ a: Int, _ b: Int, plane: SIMD4<Double>) -> Int {
        let (low, high) = a < b ? (a, b) : (b, a)
        let p = pool.positions[low], q = pool.positions[high]
        let dp = distance(p, to: plane), dq = distance(q, to: plane)
        return pool.insert(p + (q - p) * (dp / (dp - dq)))
    }

    /// The points where a triangle meets another triangle's plane: corners on the plane and edge crossings.
    private mutating func section(_ corner: SIMD3<Int>, _ distances: SIMD3<Double>, plane: SIMD4<Double>) -> [Int] {
        var result: [Int] = []
        let ids = [corner.x, corner.y, corner.z]
        for index in 0..<3 where distances[index] == 0 { result.append(ids[index]) }
        for index in 0..<3 {
            let next = (index + 1) % 3
            if distances[index] * distances[next] < 0 {
                result.append(crossing(ids[index], ids[next], plane: plane))
            }
        }
        var unique: [Int] = []
        for id in result where !unique.contains(id) { unique.append(id) }
        return unique
    }

    private mutating func intersect(_ t: Int, _ u: Int) {
        let first = solids[0], second = solids[1]
        let tc = first.corners[t], uc = second.corners[u]
        let tPlane = first.planes[t], uPlane = second.planes[u]
        let tPoints = [tc.x, tc.y, tc.z].map { pool.positions[$0] }
        let uPoints = [uc.x, uc.y, uc.z].map { pool.positions[$0] }
        let dT = SIMD3(distance(tPoints[0], to: uPlane), distance(tPoints[1], to: uPlane), distance(tPoints[2], to: uPlane))
        let dU = SIMD3(distance(uPoints[0], to: tPlane), distance(uPoints[1], to: tPlane), distance(uPoints[2], to: tPlane))
        if all(dT .> 0) || all(dT .< 0) || all(dU .> 0) || all(dU .< 0) { return }
        if all(dT .== 0) || all(dU .== 0) {
            intersectCoplanar(t, u)
            return
        }
        let tSection = section(tc, dT, plane: uPlane)
        let uSection = section(uc, dU, plane: tPlane)
        guard !tSection.isEmpty, !uSection.isEmpty else { return }
        let line = simd_cross(SIMD3(tPlane.x, tPlane.y, tPlane.z), SIMD3(uPlane.x, uPlane.y, uPlane.z))
        guard simd_length(line) > 1e-12 else { return }
        let direction = simd_normalize(line)
        func interval(_ ids: [Int]) -> (low: (Double, Int), high: (Double, Int)) {
            let values = ids.map { (simd_dot(direction, pool.positions[$0]), $0) }
            return (values.min { $0.0 < $1.0 }!, values.max { $0.0 < $1.0 }!)
        }
        let tSpan = interval(tSection), uSpan = interval(uSection)
        let start = tSpan.low.0 >= uSpan.low.0 ? tSpan.low : uSpan.low
        let end = tSpan.high.0 <= uSpan.high.0 ? tSpan.high : uSpan.high
        guard start.0 <= end.0 + tolerance, start.1 != end.1,
              simd_distance(pool.positions[start.1], pool.positions[end.1]) > tolerance else { return }
        let segment = SIMD2(start.1, end.1)
        constraints[0][t].append(segment)
        constraints[1][u].append(segment)
    }

    /// Coplanar triangles cut each other along the parts of their edges that lie inside the other.
    private mutating func intersectCoplanar(_ t: Int, _ u: Int) {
        let tc = solids[0].corners[t], uc = solids[1].corners[u]
        let tIDs = [tc.x, tc.y, tc.z], uIDs = [uc.x, uc.y, uc.z]
        let tNormal = SIMD3(solids[0].planes[t].x, solids[0].planes[t].y, solids[0].planes[t].z)
        let uNormal = SIMD3(solids[1].planes[u].x, solids[1].planes[u].y, solids[1].planes[u].z)
        // Points along each edge, as (parameter, pool id), from crossings with the other triangle's edges.
        var onT = [[(Double, Int)]](repeating: [], count: 3)
        var onU = [[(Double, Int)]](repeating: [], count: 3)
        for i in 0..<3 {
            let a = tIDs[i], b = tIDs[(i + 1) % 3]
            for j in 0..<3 {
                let c = uIDs[j], d = uIDs[(j + 1) % 3]
                for (id, s, r) in edgeContacts(a, b, c, d, normal: tNormal) {
                    onT[i].append((s, id))
                    onU[j].append((r, id))
                }
            }
        }
        addClippedEdges(of: uIDs, along: onU, inside: tIDs, normal: tNormal, to: 0, t)
        addClippedEdges(of: tIDs, along: onT, inside: uIDs, normal: uNormal, to: 1, u)
    }

    /// Contacts between two coplanar segments: a proper crossing, or endpoints
    /// lying on the other segment. Returns pool ids with each segment's parameter.
    private mutating func edgeContacts(
        _ a: Int, _ b: Int, _ c: Int, _ d: Int, normal: SIMD3<Double>
    ) -> [(Int, Double, Double)] {
        let p = pool.positions[a], q = pool.positions[b], r = pool.positions[c], s = pool.positions[d]
        let e = q - p, f = s - r
        let lengthE = simd_length(e), lengthF = simd_length(f)
        var contacts: [(Int, Double, Double)] = []
        func parameter(_ point: SIMD3<Double>, _ origin: SIMD3<Double>, _ edge: SIMD3<Double>, _ length: Double) -> Double? {
            let t = simd_dot(point - origin, edge) / (length * length)
            guard t * length >= -tolerance, (t - 1) * length <= tolerance,
                  simd_distance(point, origin + edge * min(max(t, 0), 1)) <= tolerance else { return nil }
            return min(max(t, 0), 1)
        }
        if let t = parameter(r, p, e, lengthE) { contacts.append((c, t, 0)) }
        if let t = parameter(s, p, e, lengthE) { contacts.append((d, t, 1)) }
        if let t = parameter(p, r, f, lengthF) { contacts.append((a, 0, t)) }
        if let t = parameter(q, r, f, lengthF) { contacts.append((b, 1, t)) }
        let denominator = simd_dot(simd_cross(e, f), normal)
        if contacts.isEmpty, abs(denominator) > 1e-12 * lengthE * lengthF {
            let t = simd_dot(simd_cross(r - p, f), normal) / denominator
            let w = simd_dot(simd_cross(r - p, e), normal) / denominator
            if t * lengthE > tolerance, (1 - t) * lengthE > tolerance, w * lengthF > tolerance, (1 - w) * lengthF > tolerance {
                // Interpolate along the first operand's edge from its lower pool id.
                let (low, high, lowT) = a < b ? (p, q, t) : (q, p, 1 - t)
                let id = pool.insert(low + (high - low) * lowT)
                contacts.append((id, t, w))
            }
        }
        return contacts
    }

    private mutating func addClippedEdges(
        of ids: [Int], along contacts: [[(Double, Int)]], inside other: [Int], normal: SIMD3<Double>,
        to operand: Int, _ triangle: Int
    ) {
        for index in 0..<3 {
            let a = ids[index], b = ids[(index + 1) % 3]
            var points = contacts[index]
            if contains(pool.positions[a], triangle: other, normal: normal) { points.append((0, a)) }
            if contains(pool.positions[b], triangle: other, normal: normal) { points.append((1, b)) }
            points.sort { $0.0 < $1.0 }
            var chain: [Int] = []
            for (_, id) in points where chain.last != id && !chain.contains(id) { chain.append(id) }
            for (from, to) in zip(chain, chain.dropFirst()) {
                let middle = (pool.positions[from] + pool.positions[to]) / 2
                if contains(middle, triangle: other, normal: normal) {
                    constraints[operand][triangle].append(SIMD2(from, to))
                }
            }
        }
    }

    /// Whether a point in the triangle's plane lies inside it or within the tolerance of its edges.
    private func contains(_ point: SIMD3<Double>, triangle ids: [Int], normal: SIMD3<Double>) -> Bool {
        for index in 0..<3 {
            let a = pool.positions[ids[index]], b = pool.positions[ids[(index + 1) % 3]]
            let inward = simd_normalize(simd_cross(normal, b - a))
            if simd_dot(inward, point - a) < -tolerance { return false }
        }
        return true
    }

    /// Cuts one triangle along its constraints and edge splits, or keeps it whole.
    private mutating func retriangulate(_ operand: Int, _ triangle: Int) throws {
        let corner = solids[operand].corners[triangle]
        let cornerIDs = [corner.x, corner.y, corner.z]
        let splits = (0..<3).map { edgeSplits[UndirectedEdge(cornerIDs[$0], cornerIDs[($0 + 1) % 3])] ?? [] }
        let segments = constraints[operand][triangle]
        if segments.isEmpty && splits.allSatisfy(\.isEmpty) {
            pieces[operand].append(Piece(operand: operand, source: triangle, corners: corner))
            return
        }
        let plane = solids[operand].planes[triangle]
        var graph = PlanarGraph(normal: SIMD3(plane.x, plane.y, plane.z), tolerance: tolerance)
        for index in 0..<3 {
            let a = cornerIDs[index], b = cornerIDs[(index + 1) % 3]
            let direction = pool.positions[b] - pool.positions[a]
            let chain = [a] + splits[index].sorted {
                simd_dot(pool.positions[$0] - pool.positions[a], direction) < simd_dot(pool.positions[$1] - pool.positions[a], direction)
            } + [b]
            for (from, to) in zip(chain, chain.dropFirst()) { graph.addEdge(from, to, curve: false) }
        }
        for segment in segments where segment.x != segment.y {
            graph.addEdge(segment.x, segment.y, curve: true)
        }
        graph.resolveContacts(pool: &pool)
        let faces: [[SIMD3<Int>]]
        do {
            faces = try graph.triangulateFaces(pool: pool, start: corner.x)
        } catch {
            throw MeshBoolean.Failure.triangulation(operand: operand, triangle: solids[operand].sources[triangle])
        }
        for face in faces {
            for piece in face { pieces[operand].append(Piece(operand: operand, source: triangle, corners: piece)) }
        }
        for (edge, isCurve) in graph.edges where isCurve { curveEdges[operand].insert(edge) }
    }

    // MARK: Classification and output

    mutating func assemble(_ operation: MeshBooleanOperation) throws -> MeshGeometry {
        var kept: [(piece: Piece, flipped: Bool)] = []
        for operand in 0..<2 {
            let classes = try classify(operand)
            for (index, piece) in pieces[operand].enumerated() {
                let keep: Bool
                var flipped = false
                switch (operation, operand, classes[index]) {
                case (.union, 0, .outside), (.union, 0, .same), (.union, 1, .outside): keep = true
                case (.intersection, 0, .inside), (.intersection, 0, .same), (.intersection, 1, .inside): keep = true
                case (.subtraction, 0, .outside), (.subtraction, 0, .opposite): keep = true
                case (.subtraction, 1, .inside):
                    keep = true
                    flipped = true
                default: keep = false
                }
                if keep { kept.append((piece, flipped)) }
            }
        }
        guard !kept.isEmpty else { throw MeshBoolean.Failure.emptyResult }

        // The result must be closed: every directed edge meets exactly one reversed twin.
        var directed: [DirectedEdge: Int] = [:]
        for (piece, flipped) in kept {
            let c = flipped ? SIMD3(piece.corners.x, piece.corners.z, piece.corners.y) : piece.corners
            for (from, to) in [(c.x, c.y), (c.y, c.z), (c.z, c.x)] { directed[DirectedEdge(from: from, to: to), default: 0] += 1 }
        }
        for (edge, count) in directed where count != 1 || directed[DirectedEdge(from: edge.to, to: edge.from)] != 1 {
            throw MeshBoolean.Failure.nonManifoldResult
        }

        let keepsSecondary = solids.allSatisfy { solid in solid.geometry.vertices.allSatisfy { $0.secondaryTextureCoordinate != nil } }
        let keepsTangents = solids.allSatisfy { solid in solid.geometry.vertices.allSatisfy { $0.tangent != nil } }
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        var lookup: [BooleanVertexKey: UInt32] = [:]
        for (piece, flipped) in kept {
            let solid = solids[piece.operand]
            let order = flipped ? [piece.corners.x, piece.corners.z, piece.corners.y]
                : [piece.corners.x, piece.corners.y, piece.corners.z]
            for id in order {
                let vertex = interpolate(
                    at: pool.positions[id], in: solid, triangle: piece.source, flipped: flipped,
                    secondary: keepsSecondary, tangents: keepsTangents
                )
                let key = BooleanVertexKey(vertex)
                if let existing = lookup[key] {
                    indices.append(existing)
                } else {
                    vertices.append(vertex)
                    lookup[key] = UInt32(vertices.count - 1)
                    indices.append(UInt32(vertices.count - 1))
                }
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// Classifies each piece by flood-filling patches that do not cross the intersection curve.
    private func classify(_ operand: Int) throws -> [PatchClass] {
        let list = pieces[operand]
        var byEdge: [UndirectedEdge: [Int]] = [:]
        for (index, piece) in list.enumerated() {
            let c = piece.corners
            for (a, b) in [(c.x, c.y), (c.y, c.z), (c.z, c.x)] { byEdge[UndirectedEdge(a, b), default: []].append(index) }
        }
        var patch = [Int](repeating: -1, count: list.count)
        var patches: [[Int]] = []
        for seed in list.indices where patch[seed] < 0 {
            var members = [seed]
            patch[seed] = patches.count
            var cursor = 0
            while cursor < members.count {
                let c = list[members[cursor]].corners
                cursor += 1
                for (a, b) in [(c.x, c.y), (c.y, c.z), (c.z, c.x)] {
                    let edge = UndirectedEdge(a, b)
                    guard !curveEdges[operand].contains(edge), let neighbors = byEdge[edge], neighbors.count == 2 else { continue }
                    for neighbor in neighbors where patch[neighbor] < 0 {
                        patch[neighbor] = patches.count
                        members.append(neighbor)
                    }
                }
            }
            patches.append(members)
        }
        let other = solids[1 - operand]
        var classes = [PatchClass](repeating: .outside, count: list.count)
        for members in patches {
            let result = classifyPatch(members.map { list[$0] }, against: other)
            for member in members { classes[member] = result }
        }
        return classes
    }

    private func classifyPatch(_ members: [Piece], against other: BooleanSolid) -> PatchClass {
        func area(_ piece: Piece) -> Double {
            let c = piece.corners
            return simd_length(simd_cross(pool.positions[c.y] - pool.positions[c.x], pool.positions[c.z] - pool.positions[c.x]))
        }
        let candidates = members.sorted { area($0) > area($1) }.prefix(3)
        var best: (confidence: Double, inside: Bool)?
        for piece in candidates {
            let c = piece.corners
            let points = [pool.positions[c.x], pool.positions[c.y], pool.positions[c.z]]
            let centroid = (points[0] + points[1] + points[2]) / 3
            let plane = solids[piece.operand].planes[piece.source]
            if let facing = coplanarFacing(points, centroid: centroid, normal: SIMD3(plane.x, plane.y, plane.z), in: other) {
                return facing ? .same : .opposite
            }
            let winding = windingNumber(at: centroid, of: other)
            let confidence = abs(winding - 0.5)
            if best.map({ confidence > $0.confidence }) ?? true { best = (confidence, winding > 0.5) }
        }
        return best?.inside == true ? .inside : .outside
    }

    /// Whether a piece lies on a triangle of the other solid, and if so whether their normals agree.
    private func coplanarFacing(
        _ points: [SIMD3<Double>], centroid: SIMD3<Double>, normal: SIMD3<Double>, in other: BooleanSolid
    ) -> Bool? {
        let pad = SIMD3<Float>(repeating: Float(tolerance * 4) + 1e-6 * max(1, Float(simd_reduce_max(simd_abs(centroid)))))
        guard let box = try? AxisAlignedBox(minimum: SIMD3<Float>(centroid) - pad, maximum: SIMD3<Float>(centroid) + pad) else {
            return nil
        }
        for triangle in other.bvh!.triangles(intersecting: box) {
            let plane = other.planes[triangle]
            let otherNormal = SIMD3(plane.x, plane.y, plane.z)
            guard points.allSatisfy({ abs(simd_dot(otherNormal, $0) - plane.w) <= 2 * tolerance }) else { continue }
            let c = other.corners[triangle]
            if contains(centroid, triangle: [c.x, c.y, c.z], normal: otherNormal) {
                return simd_dot(normal, otherNormal) > 0
            }
        }
        return nil
    }

    /// The generalized winding number: about 1 inside a closed outward mesh and 0 outside.
    private func windingNumber(at point: SIMD3<Double>, of solid: BooleanSolid) -> Double {
        var total = 0.0
        for c in solid.corners {
            let a = pool.positions[c.x] - point, b = pool.positions[c.y] - point, d = pool.positions[c.z] - point
            let la = simd_length(a), lb = simd_length(b), ld = simd_length(d)
            let numerator = simd_dot(a, simd_cross(b, d))
            let denominator = la * lb * ld + simd_dot(a, b) * ld + simd_dot(b, d) * la + simd_dot(d, a) * lb
            total += 2 * atan2(numerator, denominator)
        }
        return total / (4 * .pi)
    }

    /// The source triangle's attributes at a point on it.
    private func interpolate(
        at point: SIMD3<Double>, in solid: BooleanSolid, triangle: Int, flipped: Bool, secondary: Bool, tangents: Bool
    ) -> MeshVertex {
        let slot = solid.slots[triangle]
        let ids = [slot.x, slot.y, slot.z]
        let p = ids.map { solid.worldPositions[$0] }
        let v0 = p[1] - p[0], v1 = p[2] - p[0], v2 = point - p[0]
        let d00 = simd_dot(v0, v0), d01 = simd_dot(v0, v1), d11 = simd_dot(v1, v1)
        let d20 = simd_dot(v2, v0), d21 = simd_dot(v2, v1)
        let denominator = d00 * d11 - d01 * d01
        let w1 = (d11 * d20 - d01 * d21) / denominator, w2 = (d00 * d21 - d01 * d20) / denominator
        let weights = [1 - w1 - w2, w1, w2]
        let source = ids.map { solid.geometry.vertices[$0] }
        let plane = solid.planes[triangle]
        let faceNormal = SIMD3(plane.x, plane.y, plane.z)

        var normal = SIMD3<Double>.zero
        var uv = SIMD2<Double>.zero
        var uv1 = SIMD2<Double>.zero
        var tangent = SIMD3<Double>.zero
        for (weight, vertex) in zip(weights, source) {
            normal += weight * simd_normalize(solid.normalMatrix * SIMD3<Double>(vertex.normal))
            uv += weight * SIMD2<Double>(vertex.textureCoordinate)
            if secondary, let coordinate = vertex.secondaryTextureCoordinate { uv1 += weight * SIMD2<Double>(coordinate) }
            if tangents, let t = vertex.tangent {
                tangent += weight * simd_normalize(solid.linear * SIMD3<Double>(SIMD3<Float>(t.x, t.y, t.z)))
            }
        }
        normal = simd_length_squared(normal) > 1e-20 ? simd_normalize(normal) : faceNormal
        var tangentValue: SIMD4<Float>?
        if tangents, let sourceTangent = source[0].tangent {
            var direction = tangent - normal * simd_dot(tangent, normal)
            if simd_length_squared(direction) < 1e-20 {
                let helper = abs(normal.x) < 0.9 ? SIMD3<Double>(1, 0, 0) : SIMD3<Double>(0, 1, 0)
                direction = helper - normal * simd_dot(helper, normal)
            }
            var handedness = sourceTangent.w * (solid.mirrors ? -1 : 1)
            if flipped { handedness = -handedness }
            tangentValue = SIMD4(SIMD3<Float>(simd_normalize(direction)), handedness)
        }
        if flipped { normal = -normal }
        return MeshVertex(
            position: SIMD3<Float>(point), normal: SIMD3<Float>(normal), textureCoordinate: SIMD2<Float>(uv),
            secondaryTextureCoordinate: secondary ? SIMD2<Float>(uv1) : nil, tangent: tangentValue
        )
    }
}

/// The straight-line graph of one triangle: its split boundary and the intersection curve inside it.
private struct PlanarGraph {
    let normal: SIMD3<Double>
    let tolerance: Double
    /// Each undirected edge and whether it belongs to the intersection curve.
    var edges: [UndirectedEdge: Bool] = [:]
    var vertices: Set<Int> = []

    init(normal: SIMD3<Double>, tolerance: Double) {
        self.normal = normal
        self.tolerance = tolerance
    }

    mutating func addEdge(_ a: Int, _ b: Int, curve: Bool) {
        guard a != b else { return }
        let key = UndirectedEdge(a, b)
        edges[key] = (edges[key] ?? false) || curve
        vertices.insert(a)
        vertices.insert(b)
    }

    /// The point in the triangle's plane, in 2D coordinates that keep its counterclockwise winding.
    func project(_ point: SIMD3<Double>) -> SIMD2<Double> {
        let axis = simd_abs(normal)
        if axis.z >= axis.x && axis.z >= axis.y { return normal.z > 0 ? SIMD2(point.x, point.y) : SIMD2(point.y, point.x) }
        if axis.x >= axis.y { return normal.x > 0 ? SIMD2(point.y, point.z) : SIMD2(point.z, point.y) }
        return normal.y > 0 ? SIMD2(point.z, point.x) : SIMD2(point.x, point.z)
    }

    /// Splits edges at vertices that lie on them and at crossings between edges, until none remain.
    mutating func resolveContacts(pool: inout WeldingPool) {
        var changed = true
        var rounds = 0
        while changed, rounds < 64 {
            changed = false
            rounds += 1
            outer: for (edge, curve) in edges {
                let a = pool.positions[edge.low], b = pool.positions[edge.high]
                let direction = b - a
                let length = simd_length(direction)
                for vertex in vertices where vertex != edge.low && vertex != edge.high {
                    let p = pool.positions[vertex]
                    let t = simd_dot(p - a, direction) / (length * length)
                    guard t * length > tolerance, (1 - t) * length > tolerance,
                          simd_length(p - (a + direction * t)) <= tolerance else { continue }
                    edges[edge] = nil
                    addEdge(edge.low, vertex, curve: curve)
                    addEdge(vertex, edge.high, curve: curve)
                    changed = true
                    break outer
                }
            }
            if changed { continue }
            let list = Array(edges.keys)
            search: for i in list.indices {
                for j in (i + 1)..<list.count {
                    let e = list[i], f = list[j]
                    guard e.low != f.low, e.low != f.high, e.high != f.low, e.high != f.high else { continue }
                    let p = project(pool.positions[e.low]), q = project(pool.positions[e.high])
                    let r = project(pool.positions[f.low]), s = project(pool.positions[f.high])
                    let o1 = PlanarArrangement.orientation(p, q, r), o2 = PlanarArrangement.orientation(p, q, s)
                    let o3 = PlanarArrangement.orientation(r, s, p), o4 = PlanarArrangement.orientation(r, s, q)
                    guard o1 * o2 < 0, o3 * o4 < 0 else { continue }
                    let t = o3 / (o3 - o4)
                    let a = pool.positions[e.low], b = pool.positions[e.high]
                    let id = pool.insert(a + (b - a) * t)
                    let curveE = edges[e]!, curveF = edges[f]!
                    edges[e] = nil
                    edges[f] = nil
                    addEdge(e.low, id, curve: curveE)
                    addEdge(id, e.high, curve: curveE)
                    addEdge(f.low, id, curve: curveF)
                    addEdge(id, f.high, curve: curveF)
                    changed = true
                    break search
                }
            }
        }
    }

    /// Traces the faces of the graph and triangulates each one with its holes.
    func triangulateFaces(pool: WeldingPool, start: Int) throws -> [[SIMD3<Int>]] {
        // Prune dangling curve edges, which bound no face.
        var live = edges
        var degree: [Int: Int] = [:]
        for edge in live.keys {
            degree[edge.low, default: 0] += 1
            degree[edge.high, default: 0] += 1
        }
        var pruning = true
        while pruning {
            pruning = false
            for edge in Array(live.keys) where degree[edge.low] == 1 || degree[edge.high] == 1 {
                live[edge] = nil
                degree[edge.low]! -= 1
                degree[edge.high]! -= 1
                pruning = true
            }
        }

        var planar: [Int: SIMD2<Double>] = [:]
        var neighbors: [Int: [Int]] = [:]
        for edge in live.keys {
            for id in [edge.low, edge.high] where planar[id] == nil { planar[id] = project(pool.positions[id]) }
            neighbors[edge.low, default: []].append(edge.high)
            neighbors[edge.high, default: []].append(edge.low)
        }
        for (vertex, list) in neighbors {
            let origin = planar[vertex]!
            neighbors[vertex] = list.sorted {
                let a = planar[$0]! - origin, b = planar[$1]! - origin
                return atan2(a.y, a.x) < atan2(b.y, b.x)
            }
        }

        // Union-find components so isolated loops become holes of the face around them.
        var parent: [Int: Int] = [:]
        func find(_ x: Int) -> Int {
            var x = x
            while let p = parent[x], p != x { x = p }
            return x
        }
        for id in planar.keys { parent[id] = id }
        for edge in live.keys {
            let a = find(edge.low), b = find(edge.high)
            if a != b { parent[a] = b }
        }

        var visited = Set<DirectedEdge>()
        var faces: [(ring: [Int], area: Double, component: Int)] = []
        var holes: [(ring: [Int], component: Int)] = []
        let mainComponent = find(start)
        for edge in live.keys {
            for initial in [DirectedEdge(from: edge.low, to: edge.high), DirectedEdge(from: edge.high, to: edge.low)]
            where !visited.contains(initial) {
                var ring: [Int] = []
                var current = initial
                while visited.insert(current).inserted {
                    ring.append(current.from)
                    let around = neighbors[current.to]!
                    let back = around.firstIndex(of: current.from)!
                    let next = around[(back + around.count - 1) % around.count]
                    current = DirectedEdge(from: current.to, to: next)
                }
                let area = PolygonTriangulator.doubledArea(ring.map { planar[$0]! })
                let component = find(ring[0])
                if area > 0 {
                    faces.append((ring, area, component))
                } else if component != mainComponent {
                    holes.append((ring, component))
                }
            }
        }

        var assigned = [[[Int]]](repeating: [], count: faces.count)
        for hole in holes {
            let probe = planar[hole.ring[0]]!
            var best: Int?
            for (index, face) in faces.enumerated() where face.component != hole.component {
                guard PlanarArrangement.winding(of: probe, in: face.ring.map { planar[$0]! }) != 0 else { continue }
                if best.map({ face.area < faces[$0].area }) ?? true { best = index }
            }
            if let best { assigned[best].append(hole.ring) }
        }

        var result: [[SIMD3<Int>]] = []
        for (index, face) in faces.enumerated() {
            let rings = [face.ring] + assigned[index]
            let flat = rings.flatMap { $0 }
            let triangles = try PolygonTriangulator.triangulate(
                outer: face.ring.map { planar[$0]! }, holes: assigned[index].map { $0.map { planar[$0]! } }
            )
            result.append(triangles.map { SIMD3(flat[$0.x], flat[$0.y], flat[$0.z]) })
        }
        return result
    }
}

/// Bitwise identity of an output vertex.
private struct BooleanVertexKey: Hashable {
    let bits: [UInt32]

    init(_ vertex: MeshVertex) {
        let tangent = vertex.tangent ?? .zero
        let secondary = vertex.secondaryTextureCoordinate ?? .zero
        bits = [
            vertex.position.x, vertex.position.y, vertex.position.z, vertex.normal.x, vertex.normal.y, vertex.normal.z,
            vertex.textureCoordinate.x, vertex.textureCoordinate.y, secondary.x, secondary.y,
            tangent.x, tangent.y, tangent.z, tangent.w,
        ].map(\.bitPattern)
    }
}
