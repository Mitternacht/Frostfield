import Frostfield
import simd

/// Mesh checks shared by the shape, text, hull and boolean suites.
enum ShapeCheck {
    /// Whether every edge, matched by exact position, is used by exactly two
    /// triangles in opposite directions: the mesh is closed and consistently oriented.
    static func isClosed(_ mesh: MeshGeometry) -> Bool {
        var ids: [SIMD3<Float>: Int] = [:]
        let welded = mesh.vertices.map { vertex in
            if let id = ids[vertex.position] { return id }
            ids[vertex.position] = ids.count
            return ids.count - 1
        }
        var directed: [SIMD2<Int>: Int] = [:]
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let corners = (0..<3).map { welded[Int(mesh.indices[offset + $0])] }
            for index in 0..<3 {
                let edge = SIMD2(corners[index], corners[(index + 1) % 3])
                guard edge.x != edge.y else { return false }
                directed[edge, default: 0] += 1
            }
        }
        return directed.allSatisfy { edge, count in count == 1 && directed[SIMD2(edge.y, edge.x)] == 1 }
    }

    /// The summed area of the triangles.
    static func surfaceArea(_ mesh: MeshGeometry) -> Double {
        var total = 0.0
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let (a, b, c) = corners(mesh, offset / 3)
            total += simd_length(simd_cross(b - a, c - a)) / 2
        }
        return total
    }

    static func corners(_ mesh: MeshGeometry, _ triangle: Int) -> (SIMD3<Double>, SIMD3<Double>, SIMD3<Double>) {
        let offset = triangle * 3
        return (
            SIMD3<Double>(mesh.vertices[Int(mesh.indices[offset])].position),
            SIMD3<Double>(mesh.vertices[Int(mesh.indices[offset + 1])].position),
            SIMD3<Double>(mesh.vertices[Int(mesh.indices[offset + 2])].position)
        )
    }

    static func bounds(_ points: some Sequence<SIMD3<Float>>) -> (lower: SIMD3<Float>, upper: SIMD3<Float>) {
        var lower = SIMD3<Float>(repeating: .infinity), upper = SIMD3<Float>(repeating: -.infinity)
        for point in points {
            lower = simd_min(lower, point)
            upper = simd_max(upper, point)
        }
        return (lower, upper)
    }

    /// For every triangle, the vertex tangents point along increasing U and their
    /// handedness matches the direction of increasing V, as a normal map expects.
    static func tangentsFollowTextureCoordinates(_ mesh: MeshGeometry, minimumArea: Double = 1e-9) -> Bool {
        for triangle in 0..<(mesh.indices.count / 3) {
            let vertices = (0..<3).map { mesh.vertices[Int(mesh.indices[triangle * 3 + $0])] }
            let p = vertices.map { SIMD3<Double>($0.position) }
            let uv = vertices.map { SIMD2<Double>($0.textureCoordinate) }
            let e1 = p[1] - p[0], e2 = p[2] - p[0]
            let d1 = uv[1] - uv[0], d2 = uv[2] - uv[0]
            let determinant = d1.x * d2.y - d2.x * d1.y
            guard simd_length(simd_cross(e1, e2)) / 2 > minimumArea, abs(determinant) > 1e-12 else { continue }
            let alongU = (e1 * d2.y - e2 * d1.y) / determinant
            let alongV = (e2 * d1.x - e1 * d2.x) / determinant
            for vertex in vertices {
                guard let tangent = vertex.tangent else { return false }
                let t = SIMD3<Double>(SIMD3<Float>(tangent.x, tangent.y, tangent.z))
                let n = SIMD3<Double>(vertex.normal)
                guard abs(simd_dot(t, n)) < 1e-3, simd_dot(t, alongU) > 0 else { return false }
                let bitangent = Double(tangent.w) * simd_cross(n, t)
                guard simd_dot(bitangent, alongV) > 0 else { return false }
            }
        }
        return true
    }
}

/// A deterministic SplitMix64 generator so randomized fixtures repeat exactly.
struct ShapeTestRandom: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
