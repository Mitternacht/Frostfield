import Frostfield
import simd
import Testing

struct LoftGeometryTests {
    @Test("An open loft follows different 3D profiles and uses distance-based UVs")
    func openProfiles() throws {
        let profiles: [[SIMD3<Float>]] = [
            [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(3, 0, 0)],
            [SIMD3(0, 0, 1), SIMD3(2, 0, 1), SIMD3(5, 0, 1)],
            [SIMD3(0, 1, 4), SIMD3(2, 1.5, 4), SIMD3(7, 1, 5)],
        ]
        let loft = try MeshGeometry.loft(sections: profiles, closed: false)
        #expect(loft.vertices.map(\.position) == profiles.flatMap { $0 })
        #expect(loft.indices == [
            0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4,
            3, 4, 6, 4, 7, 6, 4, 5, 7, 5, 8, 7,
        ])
        let firstSpan = 1 + Double(2).squareRoot() + Double(5).squareRoot()
        let lastSpan = Double(10).squareRoot() + Double(11.25).squareRoot() + Double(21).squareRoot()
        let expectedU = Float(firstSpan / (firstSpan + lastSpan))
        #expect(loft.vertices[0].textureCoordinate == SIMD2(0, 1))
        #expect(abs(loft.vertices[1].textureCoordinate.y - 2.0 / 3.0) < 1e-6)
        #expect(simd_distance(loft.vertices[4].textureCoordinate, SIMD2(expectedU, 0.6)) < 1e-6)
        #expect(loft.vertices[8].textureCoordinate == SIMD2(1, 0))
        checkNormals(loft)
    }

    @Test("Closed profiles wrap UVs while keeping identical smooth normals at the seam")
    func closedSeam() throws {
        let square: [SIMD2<Float>] = [SIMD2(-1, -1), SIMD2(1, -1), SIMD2(1, 1), SIMD2(-1, 1)]
        let profiles = [(Float(1), Float(0)), (1.5, 1), (0.8, 4)].map { size, z in
            square.map { SIMD3($0.x * size + z * 0.05, $0.y * size, z) }
        }
        let loft = try MeshGeometry.loft(sections: profiles)
        #expect(loft.vertices.count == 15)
        #expect(loft.indices.count == 2 * 4 * 6)
        for (row, profile) in profiles.enumerated() {
            let start = loft.vertices[row * 5]
            let end = loft.vertices[row * 5 + 4]
            #expect(start.position == profile[0] && start.position == end.position)
            #expect(start.normal == end.normal)
            #expect(start.textureCoordinate.x == end.textureCoordinate.x)
            #expect(start.textureCoordinate.y == 1 && end.textureCoordinate.y == 0)
            for column in 0..<4 {
                let vertex = loft.vertices[row * 5 + column]
                #expect(simd_dot(SIMD2(vertex.normal.x, vertex.normal.y), square[column]) > 0.8)
            }
        }
        checkNormals(loft)
        let reversed = try MeshGeometry.loft(sections: profiles.map { Array($0.reversed()) })
        let inside = reversed.vertices[0].normal
        #expect(simd_dot(SIMD2(inside.x, inside.y), square[3]) < -0.8)
    }

    @Test("Concave caps preserve collinear boundary points, hard edges and closed topology")
    func cappedConcaveLoft() throws {
        let contour: [SIMD2<Float>] = [
            SIMD2(0, 0), SIMD2(2, 0), SIMD2(4, 0), SIMD2(4, 3), SIMD2(2, 1), SIMD2(0, 3),
        ]
        // Rotate the cross sections to the YZ plane, facing along +X.
        let sections = [Float(1), Float(5)].map { x in contour.map { SIMD3(x, $0.y, -$0.x) } }
        let loft = try MeshGeometry.loft(sections: sections, capStart: true, capEnd: true)
        let wallVertices = 2 * (contour.count + 1)
        let wallIndices = contour.count * 6
        let capIndexCount = (contour.count - 2) * 3
        #expect(loft.vertices.count == wallVertices + 2 * contour.count)
        #expect(loft.indices.count == wallIndices + 2 * capIndexCount)
        #expect(loft.vertices[0].normal.x == 0)
        for index in 0..<2 {
            let vertices = loft.vertices.dropFirst(wallVertices + index * contour.count).prefix(contour.count)
            let indices = loft.indices.dropFirst(wallIndices + index * capIndexCount).prefix(capIndexCount)
            let outward: Float = index == 0 ? -1 : 1
            #expect(vertices.map(\.position) == sections[index])
            #expect(vertices.allSatisfy { $0.normal == SIMD3(outward, 0, 0) })
            #expect(vertices.map { $0.textureCoordinate.x }.min() == 0 && vertices.map { $0.textureCoordinate.x }.max() == 1)
            #expect(vertices.map { $0.textureCoordinate.y }.min() == 0 && vertices.map { $0.textureCoordinate.y }.max() == 1)
            var area: Float = 0
            let triangles = Array(indices)
            for offset in stride(from: 0, to: triangles.count, by: 3) {
                let cross = triangleCross(loft, triangles[offset], triangles[offset + 1], triangles[offset + 2])
                #expect(cross.x * outward > 0)
                area += cross.x * outward / 2
            }
            #expect(abs(area - 8) < 1e-6)
        }
        checkWatertight(loft)
        let startOnly = try MeshGeometry.loft(sections: sections, capStart: true)
        #expect(startOnly.vertices.count == wallVertices + contour.count)
        #expect(startOnly.vertices[wallVertices].normal == SIMD3(-1, 0, 0))
    }

    @Test("An optional cap can triangulate a nonplanar open end while keeping authored positions")
    func openCap() throws {
        let first: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(0, 1, 0)]
        let last: [SIMD3<Float>] = [SIMD3(0, 0, 1), SIMD3(1, 0, 1.2), SIMD3(1, 1, 1), SIMD3(0, 1, 1)]
        let loft = try MeshGeometry.loft(sections: [first, last], closed: false, capEnd: true)
        #expect(loft.vertices.count == 12 && loft.indices.count == 24)
        #expect(loft.vertices.suffix(4).map(\.position) == last)
        #expect(loft.vertices.suffix(4).allSatisfy { $0.normal.z > 0.9 })
        checkNormals(loft)
    }

    @Test("Malformed loft sections and collapsed walls fail with a specific error")
    func invalidProfiles() throws {
        let ring: [SIMD3<Float>] = [.zero, SIMD3(1, 0, 0), SIMD3(1, 1, 0)]
        let upper = ring.map { $0 + SIMD3(0, 0, 1) }
        #expect(throws: MeshGeometry.LoftError.sectionCount) { try MeshGeometry.loft(sections: [ring]) }
        #expect(throws: MeshGeometry.LoftError.pointCount(1)) {
            try MeshGeometry.loft(sections: [ring, Array(upper.prefix(2))])
        }
        #expect(throws: MeshGeometry.LoftError.nonfinitePoint(section: 1, point: 2)) {
            try MeshGeometry.loft(sections: [ring, [upper[0], upper[1], SIMD3(.infinity, 1, 1)]])
        }
        #expect(throws: MeshGeometry.LoftError.duplicatePoint(section: 0, point: 3)) {
            try MeshGeometry.loft(sections: [ring + [ring[0]], upper + [upper[0]]])
        }
        #expect(throws: MeshGeometry.LoftError.degenerateWall(section: 0, edge: 0)) {
            try MeshGeometry.loft(sections: [ring, ring])
        }
        #expect(throws: MeshGeometry.LoftError.degenerateWall(section: 0, edge: 0)) {
            try MeshGeometry.loft(sections: [ring, [ring[0], upper[1], upper[2]]])
        }
        #expect(throws: MeshGeometry.LoftError.cap(0)) {
            try MeshGeometry.loft(sections: [
                [ring[0], ring[1]], [upper[0], upper[1]],
            ], closed: false, capStart: true)
        }
        let crossing: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(3, 2, 0), SIMD3(0, 2, 0), SIMD3(2, 0, 0)]
        #expect(throws: MeshGeometry.LoftError.cap(0)) {
            try MeshGeometry.loft(sections: [crossing, crossing.map { $0 + SIMD3(0, 0, 1) }], capStart: true)
        }
        let hugeProfile = Array(repeating: SIMD3<Float>.zero, count: 1_000)
        #expect(throws: MeshGeometry.LoftError.vertexBudget) {
            try MeshGeometry.loft(sections: Array(repeating: hugeProfile, count: 1_000))
        }
    }

    private func checkNormals(_ mesh: MeshGeometry) {
        #expect(mesh.vertices.allSatisfy { abs(simd_length($0.normal) - 1) < 1e-6 })
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let (a, b, c) = (mesh.indices[offset], mesh.indices[offset + 1], mesh.indices[offset + 2])
            let cross = triangleCross(mesh, a, b, c)
            let sum = mesh.vertices[Int(a)].normal + mesh.vertices[Int(b)].normal + mesh.vertices[Int(c)].normal
            #expect(simd_dot(cross, sum) > 0)
        }
    }

    private func triangleCross(_ mesh: MeshGeometry, _ a: UInt32, _ b: UInt32, _ c: UInt32) -> SIMD3<Float> {
        let origin = mesh.vertices[Int(a)].position
        return simd_cross(mesh.vertices[Int(b)].position - origin, mesh.vertices[Int(c)].position - origin)
    }

    private func checkWatertight(_ mesh: MeshGeometry) {
        var identifiers: [SIMD3<Float>: Int] = [:]
        let welded = mesh.vertices.map { vertex in
            if let index = identifiers[vertex.position] { return index }
            let index = identifiers.count
            identifiers[vertex.position] = index
            return index
        }
        var edges: [SIMD2<Int>: (count: Int, winding: Int)] = [:]
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            for corner in 0..<3 {
                let a = welded[Int(mesh.indices[offset + corner])]
                let b = welded[Int(mesh.indices[offset + (corner + 1) % 3])]
                let key = SIMD2(min(a, b), max(a, b))
                edges[key, default: (0, 0)].count += 1
                edges[key, default: (0, 0)].winding += a < b ? 1 : -1
            }
        }
        #expect(edges.values.allSatisfy { $0.count == 2 && $0.winding == 0 })
    }
}
