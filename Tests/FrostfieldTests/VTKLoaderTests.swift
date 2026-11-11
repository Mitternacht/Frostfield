import Foundation
import Frostfield
import Testing
import simd

struct VTKLoaderTests {
    @Test("Legacy VTK reads wrapped polygon and strip topology, normals, and point colors")
    func ascii() throws {
        let mesh = try VTKLoader.load(Self.ascii("""
            POINTS 4 float
            -1 -1
            -3 1 -1 -3
            1 1 -3 -1 1 -3
            POLYGONS 1 5
            4 0 1 2 3
            TRIANGLE_STRIPS 1 5
            4 0 1 3 2
            POINT_DATA 4
            SCALARS measure float
            LOOKUP_TABLE default
            1 2 3 4
            NORMALS authored double
            0 0 1 0 0 1
            0 0 1 0 0 1
            COLOR_SCALARS paint 4
            1 0.5 0 0.25  1 0.5 0 0.5
            1 0.5 0 0.75  1 0.5 0 1
            """, crlf: true))
        #expect(mesh.vertices.map(\.position) == [SIMD3(-1, -1, -3), SIMD3(1, -1, -3), SIMD3(1, 1, -3), SIMD3(-1, 1, -3)])
        #expect(mesh.indices == [0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 3])
        #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })
        #expect(mesh.vertices.compactMap { $0.color?.w } == [0.25, 0.5, 0.75, 1])
        #expect(isNearly(mesh.vertices[2].color ?? .zero, SIMD4(1, 0.214041, 0, 0.75)))
    }

    @Test("Legacy VTK cell colors and normals follow source cells after triangulation")
    func cellAttributes() throws {
        let mesh = try VTKLoader.load(Self.ascii("""
            POINTS 4 float
            -1 -1 -3 1 -1 -3 1 1 -3 -1 1 -3
            VERTICES 1 2
            1 3
            TRIANGLE_STRIPS 1 5
            4 0 1 3 2
            POLYGONS 1 5
            4 0 1 2 3
            CELL_DATA 3
            NORMALS authored float
            0 0 -1 0 1 0 1 0 0
            COLOR_SCALARS paint 3
            0 0 1 0 1 0 0.5 0 0
            """))
        #expect(mesh.indices == Array(0..<12).map(UInt32.init))
        #expect(mesh.vertices.prefix(6).allSatisfy { isNearly($0.color ?? .zero, SIMD4(0.214041, 0, 0, 1)) })
        #expect(mesh.vertices.prefix(6).allSatisfy { $0.normal == SIMD3(1, 0, 0) })
        #expect(mesh.vertices.suffix(6).allSatisfy { $0.color == SIMD4(0, 1, 0, 1) })
        #expect(mesh.vertices.suffix(6).allSatisfy { $0.normal == SIMD3(0, 1, 0) })
    }

    @Test("Big-endian VTK points, colors, and strip indices render through Metal")
    func binaryRendering() throws {
        let data = Self.binary()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("vtk")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let mesh = try VTKLoader.load(contentsOf: url)
        #expect(mesh.indices == [0, 1, 2, 1, 3, 2])
        #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })
        #expect(mesh.vertices.allSatisfy { isNearly($0.color ?? .zero, SIMD4(0.21586, 0, 1, 1)) })
        let fixture = try PrimitiveRenderFixture()
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: mesh,
            material: .classic(ClassicMaterial(model: .basic))
        )
        let result = try fixture.render(fixture.graph([SceneNode(mesh: renderer)]))
        #expect(result.statistics.submittedTriangleCount == 2)
        #expect(isNearly(result.pixels[32 * 64 + 32], SIMD4(0.21586, 0, 1, 1)))
        #expect(result.pixels[10 * 64 + 10] == .zero)
    }

    @Test("VTK rejects incomplete arrays, excessive counts, wrong point indices, and other datasets")
    func malformed() throws {
        #expect(throws: VTKLoader.Failure.unsupported("UNSTRUCTURED_GRID")) {
            try VTKLoader.load(Self.ascii("", dataset: "UNSTRUCTURED_GRID"))
        }
        #expect(throws: VTKLoader.Failure.size) { try VTKLoader.load(Self.ascii("POINTS 40000000 float")) }
        #expect(throws: VTKLoader.Failure.truncated) { try VTKLoader.load(Self.binary().dropLast(7)) }
        #expect(throws: VTKLoader.Failure.malformed("point index")) {
            try VTKLoader.load(Self.ascii("""
                POINTS 3 float
                0 0 0  1 0 0  0 1 0
                POLYGONS 1 4
                3 0 1 4
                """))
        }
        #expect(throws: VTKLoader.Failure.malformed("connectivity")) {
            try VTKLoader.load(Self.ascii("""
                POINTS 3 float
                0 0 0  1 0 0  0 1 0
                POLYGONS 1 5
                3 0 1 2
                """))
        }
        #expect(throws: VTKLoader.Failure.malformed("number")) {
            try VTKLoader.load(Self.ascii("POINTS 1 float\n0 0 inf"))
        }
    }

    static func ascii(_ body: String, crlf: Bool = false, dataset: String = "POLYDATA") -> Data {
        let value = "# vtk DataFile Version 3.0\nfixture\nASCII\nDATASET \(dataset)\n" + body + "\n"
        return Data((crlf ? value.replacingOccurrences(of: "\n", with: "\r\n") : value).utf8)
    }

    private static func binary() -> Data {
        var result = Data("# vtk DataFile Version 4.0\r\nfixture\r\nBINARY\r\nDATASET POLYDATA\r\nPOINTS 4 double\r\n".utf8)
        let values: [Double] = [-1, -1, -3, 1, -1, -3, -1, 1, -3, 1, 1, -3]
        result += words(values.map(\.bitPattern))
        result += Data("\r\nTRIANGLE_STRIPS 1 5\r\n".utf8)
        result += words([4, 0, 1, 2, 3] as [Int32])
        result += Data("\r\nPOINT_DATA 4\r\nNORMALS authored float\r\n".utf8)
        result += words(Array(repeating: [Float(0), 0, 1], count: 4).flatMap { $0.map(\.bitPattern) })
        result += Data("\r\nCOLOR_SCALARS paint 3\r\n".utf8)
        result += Data(Array(repeating: [UInt8(128), 0, 255], count: 4).flatMap { $0 })
        result += Data("\r\n".utf8)
        return result
    }

    private static func words<T: FixedWidthInteger>(_ values: [T]) -> Data {
        values.reduce(into: Data()) { data, value in
            withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
        }
    }
}
