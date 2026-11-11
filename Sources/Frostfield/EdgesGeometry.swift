import Foundation
import simd

public extension LineGeometry {
    /// The crease and boundary edges of a mesh as separate segments, like
    /// Three.js `EdgesGeometry`.
    ///
    /// Vertices are welded by position rounded to 10⁻⁴ with ties toward +∞, as
    /// Three.js does with `Math.round`, so a mesh with split normals or UV seams
    /// still shares its edges. An edge is kept when it bounds one triangle, or
    /// when its two faces' normals differ by at least `thresholdAngle` (radians,
    /// default 1°): their dot product is at most `cos(thresholdAngle)`. On a
    /// mesh where each edge has at most two consistently wound faces, that is
    /// the Three.js edge set. Elsewhere this differs on purpose: an edge where
    /// more than two triangles meet is always kept (Three.js pairs faces by
    /// winding), and triangles whose area is zero are ignored (Three.js gives
    /// them a zero normal, which keeps their edges). Triangles whose welded
    /// corners coincide are ignored by both. Segments keep the first triangle's
    /// winding and appear in triangle order.
    ///
    /// - Returns: The edges, or nil when no edge meets the threshold.
    static func edges(
        of mesh: MeshGeometry, thresholdAngle: Float = .pi / 180, color: SIMD4<Float>? = nil
    ) throws -> LineGeometry? {
        guard thresholdAngle.isFinite, (0...Float.pi).contains(thresholdAngle) else {
            throw PrimitiveGeometryError.invalidThresholdAngle
        }
        let threshold = cos(Double(thresholdAngle))
        let weld = PositionWeld(mesh)
        struct Edge {
            let start: Int
            let end: Int
            var normals: [SIMD3<Double>]
        }
        var edges: [Edge] = []
        var lookup: [SIMD2<Int>: Int] = [:]
        for triangle in stride(from: 0, to: mesh.indices.count, by: 3) {
            let corners = (0..<3).map { Int(mesh.indices[triangle + $0]) }
            let welded = corners.map { weld.id[$0] }
            guard welded[0] != welded[1], welded[1] != welded[2], welded[2] != welded[0] else { continue }
            let points = corners.map { SIMD3<Double>(mesh.vertices[$0].position) }
            let cross = simd_cross(points[1] - points[0], points[2] - points[0])
            let length = simd_length(cross)
            guard length > 0, length.isFinite else { continue }
            let normal = cross / length
            for side in 0..<3 {
                let a = welded[side]
                let b = welded[(side + 1) % 3]
                let key = SIMD2(min(a, b), max(a, b))
                if let index = lookup[key] {
                    edges[index].normals.append(normal)
                } else {
                    lookup[key] = edges.count
                    edges.append(Edge(start: corners[side], end: corners[(side + 1) % 3], normals: [normal]))
                }
            }
        }
        var positions: [SIMD3<Float>] = []
        for edge in edges {
            let keep = edge.normals.count != 2 || simd_dot(edge.normals[0], edge.normals[1]) <= threshold
            guard keep else { continue }
            positions.append(mesh.vertices[edge.start].position)
            positions.append(mesh.vertices[edge.end].position)
        }
        guard !positions.isEmpty else { return nil }
        return try LineGeometry(
            positions: positions, colors: color.map { Array(repeating: $0, count: positions.count) },
            topology: .segments
        )
    }

    /// Every distinct triangle edge of a mesh as separate segments, like Three.js
    /// `WireframeGeometry`. As in Three.js, edges are distinct by their exact end
    /// positions in either direction, without welding, and appear in triangle
    /// order with each triangle's winding. Edges whose ends are equal are
    /// skipped; Three.js keeps them, but such a line draws nothing in WebGL.
    ///
    /// - Returns: The edges, or nil when every edge has zero length.
    static func wireframe(of mesh: MeshGeometry, color: SIMD4<Float>? = nil) throws -> LineGeometry? {
        struct Edge: Hashable {
            let a: SIMD3<Float>
            let b: SIMD3<Float>
        }
        var seen = Set<Edge>()
        var positions: [SIMD3<Float>] = []
        for triangle in stride(from: 0, to: mesh.indices.count, by: 3) {
            for side in 0..<3 {
                let start = mesh.vertices[Int(mesh.indices[triangle + side])].position
                let end = mesh.vertices[Int(mesh.indices[triangle + (side + 1) % 3])].position
                guard start != end, !seen.contains(Edge(a: end, b: start)), seen.insert(Edge(a: start, b: end)).inserted else {
                    continue
                }
                positions += [start, end]
            }
        }
        guard !positions.isEmpty else { return nil }
        return try LineGeometry(
            positions: positions, colors: color.map { Array(repeating: $0, count: positions.count) },
            topology: .segments
        )
    }
}

/// Identifies vertices whose positions round to the same 10⁻⁴ grid point, with
/// ties toward +∞ like JavaScript's `Math.round`.
private struct PositionWeld {
    /// The welded identifier of each vertex.
    let id: [Int]

    init(_ mesh: MeshGeometry) {
        var lookup: [SIMD3<Int64>: Int] = [:]
        var ids: [Int] = []
        ids.reserveCapacity(mesh.vertices.count)
        for vertex in mesh.vertices {
            let exact = SIMD3<Double>(vertex.position) * 10_000
            var scaled = exact.rounded(.toNearestOrAwayFromZero)
            // Math.round(−2.5) is −2: negative ties round up.
            for axis in 0..<3 where scaled[axis] - exact[axis] == -0.5 { scaled[axis] += 1 }
            // Large Float coordinates exceed the Int64 range once scaled; clamp so the conversion is defined.
            let limit = Double(Int64.max / 2)
            let clamped = simd_clamp(scaled, SIMD3(repeating: -limit), SIMD3(repeating: limit))
            let key = SIMD3(Int64(clamped.x), Int64(clamped.y), Int64(clamped.z))
            if let existing = lookup[key] {
                ids.append(existing)
            } else {
                lookup[key] = lookup.count
                ids.append(lookup.count - 1)
            }
        }
        id = ids
    }
}
