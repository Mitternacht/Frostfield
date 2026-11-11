import Frostfield
import Testing
import simd

struct MeshGeometryTests {
    @Test
    func `Cube is unit sized with outward winding and matching face normals`() throws {
        let cube = MeshGeometry.cube()
        try #require(cube.vertices.count == 24)
        try #require(cube.indices.count == 36)
        try #require(cube.indices.allSatisfy { Int($0) < cube.vertices.count })

        let normals = Set(cube.vertices.map(\.normal))
        #expect(normals == [
            SIMD3<Float>(1, 0, 0), SIMD3<Float>(-1, 0, 0),
            SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, -1, 0),
            SIMD3<Float>(0, 0, 1), SIMD3<Float>(0, 0, -1),
        ])
        for vertex in cube.vertices {
            #expect(abs(vertex.position.x) == 0.5)
            #expect(abs(vertex.position.y) == 0.5)
            #expect(abs(vertex.position.z) == 0.5)
            #expect(simd_dot(vertex.position, vertex.normal) == 0.5)
        }

        for start in stride(from: 0, to: cube.indices.count, by: 3) {
            let a = cube.vertices[Int(cube.indices[start])]
            let b = cube.vertices[Int(cube.indices[start + 1])]
            let c = cube.vertices[Int(cube.indices[start + 2])]
            let geometricNormal = simd_cross(b.position - a.position, c.position - a.position)

            #expect(b.normal == a.normal)
            #expect(c.normal == a.normal)
            #expect(geometricNormal == a.normal)
        }
    }

    @Test
    func `Mesh rejects malformed or out of bounds indices`() {
        let cube = MeshGeometry.cube()

        #expect(throws: MeshGeometry.ValidationError.invalidTriangleIndexCount) {
            try MeshGeometry(vertices: cube.vertices, indices: [0, 1])
        }
        #expect(throws: MeshGeometry.ValidationError.indexOutOfBounds(24)) {
            try MeshGeometry(vertices: cube.vertices, indices: [0, 1, 24])
        }
    }

    @Test
    func `Mesh reports the first invalid vertex`() {
        var vertices = MeshGeometry.cube().vertices
        vertices[1] = MeshVertex(position: .zero, normal: .zero)
        vertices[2] = MeshVertex(position: SIMD3<Float>(.infinity, 0, 0), normal: SIMD3<Float>(0, 1, 0))

        #expect(throws: MeshGeometry.ValidationError.invalidVertex(1)) {
            try MeshGeometry(vertices: vertices, indices: [0, 1, 2])
        }
        vertices[1] = MeshVertex(position: .zero, normal: SIMD3<Float>(0, 1, 0))
        #expect(throws: MeshGeometry.ValidationError.invalidVertex(2)) {
            try MeshGeometry(vertices: vertices, indices: [0, 1, 2])
        }
    }
}
