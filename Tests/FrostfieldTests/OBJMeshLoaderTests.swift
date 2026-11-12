import Frostfield
import Testing
import simd

struct OBJMeshLoaderTests {
    @Test
    func `Relative indices triangulate a quad and convert OBJ UVs`() throws {
        let mesh = try OBJMeshLoader.load("""
        # The first quad is counterclockwise.
        v 0 0 0
        v 2 0 0
        v 2 2 0
        v 0 2 0
        vt 0 0
        vt 1 0
        vt 1 1
        vt 0 1
        vn 0 0 2
        f -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1 # trailing comment
        """)

        try #require(mesh.vertices.count == 4)
        #expect(mesh.indices == [0, 1, 2, 0, 2, 3])
        #expect(mesh.vertices.map(\.textureCoordinate) == [
            SIMD2<Float>(0, 1), SIMD2<Float>(1, 1), SIMD2<Float>(1, 0), SIMD2<Float>(0, 0),
        ])
        #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3<Float>(0, 0, 1) })
    }

    @Test
    func `A UV seam keeps separate vertices while matching corners share indices`() throws {
        let mesh = try OBJMeshLoader.load("""
        v 0 0 0
        v 1 0 0
        v 0 1 0
        v -1 0 0
        vt 0 0
        vt 1 0
        vt 0 1
        vt 1 1
        vn 0 0 1
        f 1/1/1 2/2/1 3/3/1
        f 1/4/1 3/3/1 4/2/1
        """)

        try #require(mesh.vertices.count == 5)
        #expect(mesh.indices == [0, 1, 2, 3, 2, 4])
        #expect(mesh.vertices[0].position == mesh.vertices[3].position)
        #expect(mesh.vertices[0].textureCoordinate != mesh.vertices[3].textureCoordinate)
    }

    @Test
    func `Faces without normals get flat normals and retain winding`() throws {
        let mesh = try OBJMeshLoader.load("""
        v 0 0 0
        v 1 0 0
        v 0 1 0
        v 0 0 1
        f 1 2 3
        f 1 4 2
        """)

        try #require(mesh.vertices.count == 6)
        #expect(mesh.indices == [0, 1, 2, 3, 4, 5])
        #expect(mesh.vertices[0].normal == SIMD3<Float>(0, 0, 1))
        #expect(mesh.vertices[3].normal == SIMD3<Float>(0, 1, 0))
        #expect(mesh.vertices[0].position == mesh.vertices[3].position)
    }

    @Test
    func `A concave face is rejected instead of producing overlapping triangles`() {
        #expect(throws: OBJMeshLoader.Failure.invalidLine(6, "A face must be simple and strictly convex.")) {
            try OBJMeshLoader.load("""
            v 0 0 0
            v 2 0 0
            v 1 0.5 0
            v 2 2 0
            v 0 2 0
            f 1 2 3 4 5
            """)
        }
    }

    @Test
    func `Index and numeric failures include the source line`() {
        #expect(throws: OBJMeshLoader.Failure.invalidLine(5, "Position index -4 is out of range.")) {
            try OBJMeshLoader.load("""
            v 0 0 0

            v 1 0 0
            v 0 1 0
            f -4 2 3
            """)
        }
        #expect(throws: OBJMeshLoader.Failure.invalidLine(2, "Expected a finite number, found 'inf'.")) {
            try OBJMeshLoader.load("# comment\nv inf 0 0")
        }
    }

    @Test
    func `Empty geometry and oversized faces are rejected`() {
        #expect(throws: OBJMeshLoader.Failure.noFaces) {
            try OBJMeshLoader.load("v 0 0 0")
        }
        let face = "f " + Array(repeating: "1", count: 257).joined(separator: " ")
        #expect(throws: OBJMeshLoader.Failure.invalidLine(2, "A face requires 3...256 corners.")) {
            try OBJMeshLoader.load("v 0 0 0\n" + face)
        }
    }

    @Test("Lines and points import beside polygon faces")
    func linesAndPoints() throws {
        let source = """
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 0 1 1 0 0
        f 1 2 3
        l -4 -3 -2
        l 1 2 3 1
        p 4
        """
        let document = try OBJMeshLoader.loadDocument(source)
        let mesh = try #require(document.mesh)
        #expect(mesh.indices == [0, 1, 2])
        try #require(document.lines.count == 2)
        #expect(document.lines[0].topology == .strip)
        #expect(document.lines[0].positions == [SIMD3<Float>(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 1, 0)])
        #expect(document.lines[1].topology == .loop)
        #expect(document.lines[1].positions.count == 3)
        let points = try #require(document.points)
        #expect(points.positions == [SIMD3<Float>(0, 0, 1)])
        #expect(points.colors == [SIMD4<Float>(1, 0, 0, 1)])
        #expect(throws: OBJMeshLoader.Failure.noFaces) {
            try OBJMeshLoader.load("v 0 0 0\nv 1 0 0\nl 1 2")
        }
        let onlyLines = try OBJMeshLoader.loadDocument("v 0 0 0\nv 1 0 0\nl 1 2")
        #expect(onlyLines.mesh == nil)
        #expect(onlyLines.lines.count == 1)
        #expect(throws: OBJMeshLoader.Failure.noGeometry) {
            try OBJMeshLoader.loadDocument("v 0 0 0")
        }
    }
}
