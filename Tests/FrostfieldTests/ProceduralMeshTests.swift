import Frostfield
import Testing
import simd

struct ProceduralMeshTests {
    @Test
    func `Plane subdivisions preserve top-left UV and front winding`() throws {
        let mesh = try MeshGeometry.plane(width: 4, height: 2, columns: 2, rows: 3)
        try #require(mesh.vertices.count == 12)
        try #require(mesh.indices.count == 36)
        #expect(mesh.vertices[0].position == SIMD3<Float>(-2, 1, 0))
        #expect(mesh.vertices[0].textureCoordinate == SIMD2<Float>(0, 0))
        #expect(mesh.vertices[11].position == SIMD3<Float>(2, -1, 0))
        #expect(mesh.vertices[11].textureCoordinate == SIMD2<Float>(1, 1))
        #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3<Float>(0, 0, 1) })
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let (a, b, c) = triangle(mesh, offset: offset)
            #expect(simd_cross(b - a, c - a).z > 0)
        }
    }

    @Test(arguments: [3, 12], [2, 6])
    func `Sphere winds outward with deterministic caps and a split UV seam`(segments: Int, rings: Int) throws {
        let radius: Float = 2
        let mesh = try MeshGeometry.uvSphere(radius: radius, segments: segments, rings: rings)
        let rowWidth = segments + 1
        let capStart = (rings - 1) * rowWidth
        try #require(mesh.vertices.count == capStart + 2 * segments)
        try #require(mesh.indices.count == 6 * segments * (rings - 1))
        #expect(mesh.vertices.allSatisfy { abs(simd_length($0.position) - radius) < 1e-5 })
        #expect(mesh.vertices.allSatisfy { abs(simd_length($0.normal) - 1) < 1e-5 })

        for row in 0..<(rings - 1) {
            let start = mesh.vertices[row * rowWidth]
            let end = mesh.vertices[row * rowWidth + segments]
            #expect(start.position == end.position)
            #expect(start.normal == end.normal)
            #expect(start.textureCoordinate.x == 0)
            #expect(end.textureCoordinate.x == 1)
            #expect(start.textureCoordinate.y == end.textureCoordinate.y)
        }
        for wedge in 0..<segments {
            let north = mesh.vertices[capStart + 2 * wedge]
            let south = mesh.vertices[capStart + 2 * wedge + 1]
            let u = (Float(wedge) + 0.5) / Float(segments)
            #expect(north.position == SIMD3<Float>(0, radius, 0))
            #expect(south.position == SIMD3<Float>(0, -radius, 0))
            #expect(north.textureCoordinate == SIMD2<Float>(u, 0))
            #expect(south.textureCoordinate == SIMD2<Float>(u, 1))
        }
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let (a, b, c) = triangle(mesh, offset: offset)
            #expect(simd_dot(simd_cross(b - a, c - a), a + b + c) > 0)
        }
    }

    @Test
    func `Rejects invalid sizes and unbounded segment requests`() {
        #expect(throws: MeshGeometry.ProceduralValidationError.invalidPlaneSize) {
            try MeshGeometry.plane(width: .infinity)
        }
        #expect(throws: MeshGeometry.ProceduralValidationError.invalidPlaneSubdivisions) {
            try MeshGeometry.plane(rows: Int.max)
        }
        #expect(throws: MeshGeometry.ProceduralValidationError.invalidSphereRadius) {
            try MeshGeometry.uvSphere(radius: 0)
        }
        #expect(throws: MeshGeometry.ProceduralValidationError.invalidSphereSubdivisions) {
            try MeshGeometry.uvSphere(segments: 2)
        }
        #expect(throws: MeshGeometry.ProceduralValidationError.invalidSphereSubdivisions) {
            try MeshGeometry.uvSphere(rings: Int.max)
        }
    }

    private func triangle(_ mesh: MeshGeometry, offset: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
        (
            mesh.vertices[Int(mesh.indices[offset])].position,
            mesh.vertices[Int(mesh.indices[offset + 1])].position,
            mesh.vertices[Int(mesh.indices[offset + 2])].position
        )
    }
}
