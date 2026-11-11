import Foundation
import Frostfield
import Testing
import simd

private func triangleCorners(_ mesh: MeshGeometry, _ triangle: Int) -> (MeshVertex, MeshVertex, MeshVertex) {
    let start = triangle * 3
    return (
        mesh.vertices[Int(mesh.indices[start])],
        mesh.vertices[Int(mesh.indices[start + 1])],
        mesh.vertices[Int(mesh.indices[start + 2])]
    )
}

private func positionEdgeCounts(_ mesh: MeshGeometry) -> [Set<SIMD3<Float>>: Int] {
    var edges: [Set<SIMD3<Float>>: Int] = [:]
    for triangle in 0..<(mesh.indices.count / 3) {
        let (a, b, c) = triangleCorners(mesh, triangle)
        for edge in [Set([a.position, b.position]), Set([b.position, c.position]), Set([c.position, a.position])] {
            edges[edge, default: 0] += 1
        }
    }
    return edges
}

struct MeshSubdivisionTests {
    /// A cube of half size 1: vertex i has x, y, z = ±1 from bits 0, 1, 2. Faces +X, −X, +Y, −Y, +Z, −Z.
    static func cube(creases: [PolygonMesh.Crease] = []) throws -> PolygonMesh {
        let positions = (0..<8).map { i in SIMD3<Float>(i & 1 == 0 ? -1 : 1, i & 2 == 0 ? -1 : 1, i & 4 == 0 ? -1 : 1) }
        return try PolygonMesh(
            positions: positions, faceVertexCounts: Array(repeating: 4, count: 6),
            faceVertexIndices: [1, 3, 7, 5, 0, 4, 6, 2, 2, 6, 7, 3, 0, 1, 5, 4, 4, 5, 7, 6, 0, 2, 3, 1], creases: creases
        )
    }

    static let cubeEdges: [(UInt32, UInt32)] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]

    /// A unit-circumradius icosahedron with outward counterclockwise faces.
    static func icosahedron() throws -> PolygonMesh {
        let phi = (1 + sqrt(Float(5))) / 2
        var positions: [SIMD3<Float>] = []
        for a in [Float(-1), 1] {
            for b in [-phi, phi] {
                positions += [SIMD3(0, a, b), SIMD3(a, b, 0), SIMD3(b, 0, a)]
            }
        }
        positions = positions.map(simd_normalize)
        let edge = simd_distance(positions[0], positions.dropFirst().min { simd_distance(positions[0], $0) < simd_distance(positions[0], $1) }!)
        var faces: [UInt32] = []
        for i in 0..<12 {
            for j in (i + 1)..<12 where abs(simd_distance(positions[i], positions[j]) - edge) < 1e-4 {
                for k in (j + 1)..<12 where abs(simd_distance(positions[i], positions[k]) - edge) < 1e-4
                    && abs(simd_distance(positions[j], positions[k]) - edge) < 1e-4 {
                    let outward = simd_dot(simd_cross(positions[j] - positions[i], positions[k] - positions[i]), positions[i]) > 0
                    faces += outward ? [UInt32(i), UInt32(j), UInt32(k)] : [UInt32(i), UInt32(k), UInt32(j)]
                }
            }
        }
        return try PolygonMesh(positions: positions, faceVertexCounts: Array(repeating: 3, count: faces.count / 3), faceVertexIndices: faces)
    }

    @Test
    func `Catmull–Clark on a cube converges to the closed-form limit of its corner and face center`() throws {
        // Halstead's limit mask on the first level gives (n²v + 4Σe + Σf) / (n(n + 5)). For the corner
        // (n = 3): v = 5/9, edge points (0, 3/4, 3/4)…, face points (1, 0, 0)…, so each
        // coordinate is (5 + 6 + 1) / 24 = 1/2. For a face center (n = 4): z = (16 + 12 + 20/9) / 36 = 68/81.
        let corner = SIMD3<Float>(repeating: 0.5)
        let faceCenter = SIMD3<Float>(0, 0, 68.0 / 81)
        let cube = try Self.cube()
        #expect(cube.faceVertexCounts.count == 6)
        for iterations in 1...4 {
            let limit = try cube.subdividedCatmullClark(iterations: iterations, options: .init(projectsToLimit: true))
            #expect(simd_distance(limit.positions[7], corner) < 1e-6, "iterations \(iterations)")
            // Face 4 (+Z) has its face point at vertex 8 + 4 from the first level on.
            #expect(simd_distance(limit.positions[12], faceCenter) < 1e-6, "iterations \(iterations)")
        }
        var previous = Float.infinity
        var distances: [Float] = []
        for iterations in 1...6 {
            let refined = try cube.subdividedCatmullClark(iterations: iterations)
            let distance = simd_distance(refined.positions[7], corner)
            distances.append(distance)
            #expect(distance < previous)
            previous = distance
            #expect(refined.faceVertexCounts.count == 24 * Int(pow(4.0, Double(iterations - 1))))
        }
        // Measured: 0.0962 after one level, then each level cuts the distance to a sixth.
        #expect(distances[0] > 0.09 && distances[5] < 2e-5, "\(distances)")
        for level in 1..<6 { #expect(abs(distances[level] / distances[level - 1] - 1.0 / 6) < 0.01) }
    }

    @Test
    func `Loop on an icosahedron approaches a sphere-like limit with Loop's closed-form vertex limit`() throws {
        let ico = try Self.icosahedron()
        #expect(ico.faceVertexCounts.count == 20)
        // Loop's limit for valence n: (1 − nχ)v + χΣp, χ = 1 / (3 / (8β) + n). On a unit
        // icosahedron each vertex's five neighbors sum to √5 v.
        let n = 5.0
        let c = 3.0 / 8 + cos(2 * Double.pi / n) / 4
        let beta = (5.0 / 8 - c * c) / n
        let chi = 1 / (3 / (8 * beta) + n)
        let factor = Float(1 - n * chi + chi * sqrt(5.0))
        #expect(abs(factor - 0.70781) < 1e-4)
        let expected = ico.positions[0] * factor
        for iterations in 1...3 {
            let limit = try ico.subdividedLoop(iterations: iterations, options: .init(projectsToLimit: true))
            #expect(simd_distance(limit.positions[0], expected) < 1e-6, "iterations \(iterations)")
        }
        var previous = Float.infinity
        for iterations in 1...5 {
            let refined = try ico.subdividedLoop(iterations: iterations)
            let distance = simd_distance(refined.positions[0], expected)
            #expect(distance < previous)
            previous = distance
        }
        // Measured: 0.0598 after one level, 1.05e-4 after five, about a fifth per level.
        #expect(previous < 2e-4, "\(previous)")

        // The limit surface is round to 1.16% of its radius (0.6996 to 0.7078).
        let surface = try ico.subdividedLoop(iterations: 4, options: .init(projectsToLimit: true))
        let radii = surface.positions.map(simd_length)
        let mean = radii.reduce(0, +) / Float(radii.count)
        let spread = (radii.max()! - radii.min()!) / mean
        #expect(spread < 0.0125, "spread \(spread), mean \(mean)")
        let mesh = try surface.triangulated()
        #expect(mesh.indices.count == 20 * 256 * 3)
        #expect(mesh.vertices.allSatisfy { simd_dot($0.normal, simd_normalize($0.position)) > 0.999 })
        #expect(positionEdgeCounts(mesh).values.allSatisfy { $0 == 2 })
    }

    @Test
    func `Creases keep sharp edges, semi-sharp creases blend, and hard edges can become creases`() throws {
        func cornerLimit(_ sharpness: Float?) throws -> Float {
            let creases = sharpness.map { s in Self.cubeEdges.map { PolygonMesh.Crease($0.0, $0.1, sharpness: s) } } ?? []
            return try Self.cube(creases: creases).subdividedCatmullClark(iterations: 4, options: .init(projectsToLimit: true)).positions[7].x
        }
        let smooth = try cornerLimit(nil), half = try cornerLimit(0.5), one = try cornerLimit(1)
        let two = try cornerLimit(2), sharp = try cornerLimit(.infinity)
        #expect(abs(smooth - 0.5) < 1e-6)
        #expect(sharp == 1)
        #expect(smooth < half && half < one && one < two && two < sharp, "\(smooth) \(half) \(one) \(two) \(sharp)")
        // One level with sharpness 1/2 on every edge blends each rule halfway (DeRose et al.):
        // the corner goes to (5/9 + 1) / 2 = 7/9 and the edge point of the top front edge to
        // ((0, 3/4, 3/4) + (0, 1, 1)) / 2.
        let blended = try Self.cube(creases: Self.cubeEdges.map { PolygonMesh.Crease($0.0, $0.1, sharpness: 0.5) })
            .subdividedCatmullClark(iterations: 1)
        #expect(simd_distance(blended.positions[7], SIMD3(repeating: 7.0 / 9)) < 1e-6)
        #expect(blended.positions.contains { simd_distance($0, SIMD3(0, 7.0 / 8, 7.0 / 8)) < 1e-6 })
        #expect(blended.creases.isEmpty)

        let sharpCube = try Self.cube(creases: Self.cubeEdges.map { PolygonMesh.Crease($0.0, $0.1) })
            .subdividedCatmullClark(iterations: 3)
        #expect(sharpCube.positions.allSatisfy { abs(simd_reduce_max(simd_abs($0)) - 1) < 1e-6 })
        #expect(sharpCube.creases.count == 12 * 8)

        // MeshGeometry.cube() splits normals on every edge; as creases they keep the box.
        let box = try MeshGeometry.cube().subdividedLoop(iterations: 2, hardEdgesAsCreases: true)
        #expect(box.vertices.allSatisfy { abs(simd_reduce_max(simd_abs($0.position)) - 0.5) < 1e-6 })
        #expect(Set(box.vertices.map(\.normal)).count == 6)
        #expect(box.vertices.allSatisfy { simd_reduce_max(simd_abs($0.normal)) == 1 })
        let rounded = try MeshGeometry.cube().subdividedLoop(iterations: 2)
        #expect(rounded.vertices.allSatisfy { simd_reduce_max(simd_abs($0.position)) < 0.5 })
        #expect(positionEdgeCounts(rounded).values.allSatisfy { $0 == 2 })
    }

    @Test
    func `Texture coordinates refine linearly or smoothly and boundary corners can stay`() throws {
        // A planar 4 × 4 quad grid over [−1, 1]² with UV an affine function of position.
        var positions: [SIMD3<Float>] = []
        for row in 0...4 { for column in 0...4 { positions.append(SIMD3(Float(column) / 2 - 1, 1 - Float(row) / 2, 0)) } }
        var faces: [UInt32] = []
        for row in 0..<4 {
            for column in 0..<4 {
                let a = UInt32(row * 5 + column)
                faces += [a, a + 5, a + 6, a + 1]
            }
        }
        func uv(_ p: SIMD3<Float>) -> SIMD2<Float> { SIMD2(0.25 * p.x + 0.5, 0.5 - 0.4 * p.y) }
        let grid = try PolygonMesh(
            positions: positions, faceVertexCounts: Array(repeating: 4, count: 16), faceVertexIndices: faces,
            textureCoordinates: .init(values: positions.map(uv), indices: faces)
        )
        for interpolation in [SubdivisionOptions.TextureCoordinateInterpolation.linear, .smooth] {
            let refined = try grid.subdividedCatmullClark(
                iterations: 3, options: .init(textureCoordinates: interpolation, keepsBoundaryCorners: true)
            )
            let channel = try #require(refined.textureCoordinates)
            #expect(refined.positions.allSatisfy { $0.z == 0 })
            for (corner, index) in refined.faceVertexIndices.enumerated() {
                let expected = uv(refined.positions[Int(index)])
                #expect(simd_distance(channel.values[Int(channel.indices[corner])], expected) < 1e-6)
            }
            for corner in [SIMD3<Float>(-1, -1, 0), SIMD3(-1, 1, 0), SIMD3(1, -1, 0), SIMD3(1, 1, 0)] {
                #expect(refined.positions.contains(corner))
            }
        }
        // Without boundary corners, the patch corners round off along the boundary rule.
        let rounded = try grid.subdividedCatmullClark(iterations: 2)
        #expect(!rounded.positions.contains(SIMD3(1, 1, 0)))

        // A UV seam stays a seam: per-face islands on a cube keep their corners and range.
        let islands: [SIMD2<Float>] = [SIMD2(0, 1), SIMD2(1, 1), SIMD2(1, 0), SIMD2(0, 0)]
        let base = try Self.cube()
        let boxed = try PolygonMesh(
            positions: base.positions, faceVertexCounts: base.faceVertexCounts, faceVertexIndices: base.faceVertexIndices,
            textureCoordinates: .init(values: (0..<6).flatMap { _ in islands }, indices: (0..<24).map { UInt32($0) })
        )
        for interpolation in [SubdivisionOptions.TextureCoordinateInterpolation.linear, .smooth] {
            let refined = try boxed.subdividedCatmullClark(iterations: 2, options: .init(textureCoordinates: interpolation))
            let values = try #require(refined.textureCoordinates).values
            #expect(values.allSatisfy { $0.x >= 0 && $0.x <= 1 && $0.y >= 0 && $0.y <= 1 })
            #expect(values.filter { islands.contains($0) }.count == 24)
        }
    }

    @Test
    func `Polygons triangulate by ear clipping and subdivide into quads`() throws {
        // A concave L-shaped hexagon in the XY plane, area 3.
        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(2, 0, 0), SIMD3(2, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 2, 0), SIMD3(0, 2, 0)]
        let shape = try PolygonMesh(positions: positions, faceVertexCounts: [6], faceVertexIndices: [0, 1, 2, 3, 4, 5])
        let triangles = try shape.triangulated()
        #expect(triangles.indices.count == 12)
        var area: Float = 0
        for t in 0..<4 {
            let (a, b, c) = triangleCorners(triangles, t)
            let cross = simd_cross(b.position - a.position, c.position - a.position)
            #expect(cross.z > 0)
            area += cross.z / 2
        }
        #expect(abs(area - 3) < 1e-6)
        #expect(triangles.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })
        let refined = try shape.subdividedCatmullClark(iterations: 1)
        #expect(refined.faceVertexCounts == Array(repeating: 4, count: 6))

        #expect(throws: MeshGeometry.GeometryProcessingError.invalidPolygonMesh("Loop subdivision needs triangles")) {
            try shape.subdividedLoop(iterations: 1)
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidSubdivision) { try shape.subdividedCatmullClark(iterations: 0) }
        #expect(throws: MeshGeometry.GeometryProcessingError.tooManyTriangles) {
            try shape.subdividedCatmullClark(iterations: 8, options: .init(maximumTriangleCount: 1000))
        }
        #expect(throws: (any Error).self) { try PolygonMesh(positions: positions, faceVertexCounts: [2], faceVertexIndices: [0, 1]) }
        #expect(throws: (any Error).self) { try PolygonMesh(positions: positions, faceVertexCounts: [3], faceVertexIndices: [0, 1, 1]) }
        #expect(throws: (any Error).self) {
            try PolygonMesh(positions: positions, faceVertexCounts: [6], faceVertexIndices: [0, 1, 2, 3, 4, 5], creases: [.init(0, 3)])
        }
    }
    @Test
    func `A concave quad uses its interior diagonal for either winding`() throws {
        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(4, 0.75, 0), SIMD3(0, 1.5, 0), SIMD3(1, 0.75, 0)]
        for reverse in [false, true] {
            let indices: [UInt32] = reverse ? [3, 2, 1, 0] : [0, 1, 2, 3]
            let mesh = try PolygonMesh(positions: positions, faceVertexCounts: [4], faceVertexIndices: indices).triangulated()
            #expect(mesh.indices.count == 6)
            var area: Float = 0
            let orientation: Float = reverse ? -1 : 1
            for face in 0..<2 {
                let (a, b, c) = triangleCorners(mesh, face)
                let signedArea = simd_cross(b.position - a.position, c.position - a.position).z * 0.5
                #expect(signedArea * orientation > 0)
                area += signedArea
            }
            #expect(abs(area - orientation * 2.25) < 1e-6)
            #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3(0, 0, orientation) })
        }
    }

    @Test
    func `Subdivision discards degenerate input and preserves material groups and face-varying color`() throws {
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0),
            SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(0, 1, 0), SIMD3(0.5, 0, 0),
        ]
        let red = SIMD4<Float>(1, 0, 0, 0.5)
        let blue = SIMD4<Float>(0, 0, 1, 0.75)
        let vertices = positions.enumerated().map { index, position in
            let uv = SIMD2(position.x, position.y)
            return MeshVertex(
                position: position, normal: SIMD3(0, 0, 1), textureCoordinate: uv,
                secondaryTextureCoordinate: uv * 0.25 + SIMD2(0.125, 0.25),
                tangent: SIMD4(1, 0, 0, 1), color: index < 3 ? red : blue
            )
        }
        let geometry = try MeshGeometry(
            vertices: vertices, indices: [0, 1, 2, 0, 1, 3, 0, 1, 6, 3, 4, 5]
        ).withGroups([.init(start: 0, count: 9, materialIndex: 7), .init(start: 9, count: 3, materialIndex: 2)])
        let welded = try PolygonMesh(geometry, hardEdgesAsCreases: true)
        #expect(welded.faceVertexCounts == [3, 3])
        #expect(welded.faceGroups == [.init(start: 0, count: 1, materialIndex: 7), .init(start: 1, count: 1, materialIndex: 2)])
        #expect(welded.creases.isEmpty)

        let loop = try geometry.subdividedLoop(iterations: 1, options: .init(colorInterpolation: .smooth))
        let catmullClark = try geometry.subdividedCatmullClark(iterations: 1)
        for (output, trianglesPerGroup) in [(loop, 4), (catmullClark, 6)] {
            let expectedCount = trianglesPerGroup * 3
            #expect(output.groups == [
                .init(start: 0, count: expectedCount, materialIndex: 7),
                .init(start: expectedCount, count: expectedCount, materialIndex: 2),
            ])
            let first = try output.subset(output.groups[0].range)
            let second = try output.subset(output.groups[1].range)
            #expect(first.vertices.allSatisfy { $0.color == red })
            #expect(second.vertices.allSatisfy { $0.color == blue })
            #expect(!Set(first.vertices.map(\.position)).isDisjoint(with: Set(second.vertices.map(\.position))))
            for vertex in output.vertices {
                let uv1 = try #require(vertex.secondaryTextureCoordinate)
                #expect(simd_length(uv1 - (vertex.textureCoordinate * 0.25 + SIMD2(0.125, 0.25))) < 1e-6)
                #expect(vertex.tangent == nil)
            }
        }

        let missingUV = MeshVertex(position: positions[2], normal: SIMD3(0, 0, 1))
        let partialUV = try MeshGeometry(vertices: [vertices[0], vertices[1], missingUV], indices: [0, 1, 2])
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidPolygonMesh("referenced vertices must agree on whether UV1 is present")) {
            try partialUV.subdividedLoop(iterations: 1)
        }

        let allDegenerate = try MeshGeometry(vertices: vertices, indices: [0, 1, 3])
        #expect(throws: MeshGeometry.GeometryProcessingError.noTrianglesRemain) {
            try allDegenerate.subdividedLoop(iterations: 1)
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidPolygonMesh("each face needs at least three corners, and the counts must match the indices")) {
            try PolygonMesh(positions: positions, faceVertexCounts: [Int.max, 3], faceVertexIndices: [0, 1, 2])
        }
    }

}
