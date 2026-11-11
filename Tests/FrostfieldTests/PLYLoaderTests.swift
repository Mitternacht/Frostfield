import Foundation
import Frostfield
import Testing

struct PLYLoaderTests {
    @Test("All native PLY formats round-trip indexed geometry and typed vertex columns",
          arguments: PLYExporter.Format.allCases)
    func roundTrip(format: PLYExporter.Format) throws {
        let normal = SIMD3<Float>(0, 0, 1)
        let mesh = try MeshGeometry(vertices: [
            .init(position: SIMD3(-1, 0, 0), normal: normal, textureCoordinate: SIMD2(0.25, 0.125),
                  color: SIMD4(0.215_860_5, 0, 1, 1)),
            .init(position: SIMD3(1, 0, 0), normal: normal, textureCoordinate: SIMD2(0.75, 0.75)),
            .init(position: SIMD3(0, 1, 0), normal: normal, textureCoordinate: SIMD2(0.5, 0)),
        ], indices: [0, 1, 2])
        let properties: [PLYExporter.TypedScalarProperty] = [
            .init(name: "signed8", values: .int8([.min, -1, .max])),
            .init(name: "unsigned8", values: .uint8([0, 42, .max])),
            .init(name: "signed16", values: .int16([.min, 1_001, .max])),
            .init(name: "unsigned16", values: .uint16([0, 43_981, .max])),
            .init(name: "signed32", values: .int32([.min, -16_777_217, .max])),
            .init(name: "objectID", values: .uint32([.max, 16_777_217, 0])),
            .init(name: "weight", values: .float32([0.125, -0.0, 1])),
            .init(name: "confidence", values: .float64([Double(1).nextUp, -0.0, Double.leastNonzeroMagnitude])),
        ]
        let original = try PLYExporter.data(from: mesh, options: .init(format: format, typedScalarProperties: properties))
        let imported = try PLYLoader.load(original, options: .init(customVertexProperties: properties.map(\.name)))
        guard case .mesh(let decoded) = imported.geometry else {
            Issue.record("Expected an indexed PLY mesh")
            return
        }
        #expect(decoded.indices == mesh.indices)
        #expect(decoded.vertices.map(\.position) == mesh.vertices.map(\.position))
        #expect(decoded.vertices.map(\.normal) == mesh.vertices.map(\.normal))
        #expect(decoded.vertices.map(\.textureCoordinate) == mesh.vertices.map(\.textureCoordinate))
        let firstColor = try #require(decoded.vertices[0].color)
        #expect(abs(firstColor.x - 0.215_860_5) < 0.000_001)
        #expect(firstColor.y == 0 && firstColor.z == 1 && firstColor.w == 1)
        #expect(imported.sourceVertexIndices == nil)
        #expect(imported.pointNormals == nil && imported.pointTextureCoordinates == nil)
        #expect(imported.customVertexProperties.map(\.name) == properties.map(\.name))
        let reencoded = try PLYExporter.data(from: decoded, options: .init(
            format: format, typedScalarProperties: imported.customVertexProperties
        ))
        #expect(reencoded == original)
    }

    @Test("Face colors and corner UVs override vertex attributes with the documented quad split")
    func faceOverrides() throws {
        let text = """
        ply
        format ascii 1.0
        comment end_header can also appear in a comment
        obj_info Scan fixture
        element face 2
        property list uchar uint32 vertex_indices
        property list uchar float32 texcoord
        property uchar r
        property uchar g
        property uchar b
        element edge 1
        property int vertex1
        property int vertex2
        property list ushort double metadata
        element vertex 4
        property float px
        property float py
        property float altitude
        property float tx
        property float ty
        property uint32 id
        end_header
        4 0 1 2 3 8 0 0 1 0 1 1 0 1 128 0 255
        3 0 2 3 0 0 255 0
        0 1 2 0.125 -0.5
        0 0 0 0.25 0.125 4294967295
        1 0 0 0.25 0.25 16777217
        1 1 0 0.75 0.75 2
        0 1 0 0.25 0.75 3
        """
        let data = Data(text.replacingOccurrences(of: "\n", with: "\r\n").utf8)
        let options = PLYLoader.Options(propertyNameMapping: ["altitude": "pz", "id": "identifier"],
                                        customVertexProperties: ["identifier"])
        let imported = try PLYLoader.load(data, options: options)
        guard case .mesh(let mesh) = imported.geometry,
              case .uint32(let ids) = try #require(imported.customVertexProperties.first).values else {
            Issue.record("Expected a split mesh with UInt32 IDs")
            return
        }
        #expect(mesh.indices == Array(UInt32(0)..<9))
        #expect(imported.sourceVertexIndices == [0, 1, 2, 0, 2, 3, 0, 2, 3])
        #expect(ids == [.max, 16_777_217, 2, .max, 2, 3, .max, 2, 3])
        #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })
        #expect(mesh.vertices.map(\.textureCoordinate) == [
            SIMD2(0, 1), SIMD2(1, 1), SIMD2(1, 0), SIMD2(0, 1), SIMD2(1, 0), SIMD2(0, 0),
            SIMD2(0.25, 0.875), SIMD2(0.75, 0.25), SIMD2(0.25, 0.25),
        ])
        let first = try #require(mesh.vertices[0].color)
        #expect(abs(first.x - 0.215_860_5) < 0.000_001 && first.z == 1)
        #expect(mesh.vertices[6...8].allSatisfy { $0.color == SIMD4(0, 1, 0, 1) })
    }

    @Test("Face-free PLY retains authored UVs and normals beside the rendered point cloud")
    func pointCloud() throws {
        let text = """
        ply
        format ascii 1.0
        element vertex 2
        property double posx
        property int16 posy
        property float32 posz
        property float normalx
        property float normaly
        property float normalz
        property float texture_u
        property float texture_v
        property uint16 diffuse_red
        property uint16 diffuse_green
        property uint16 diffuse_blue
        element face 0
        property list uchar int vertex_index
        end_header
        1.25 -2 3 0 0 1 0.75 0.25 32768 0 65535
        -4.5 6 0 1 0 0 0.25 0.75 0 65535 0
        """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("ply")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(text.utf8).write(to: url)
        let imported = try PLYLoader.load(contentsOf: url)
        guard case .pointCloud(let cloud) = imported.geometry else {
            Issue.record("Expected a face-free point cloud")
            return
        }
        #expect(cloud.positions == [SIMD3(1.25, -2, 3), SIMD3(-4.5, 6, 0)])
        let colors = try #require(cloud.colors)
        #expect(abs(colors[0].x - 0.214_048_2) < 0.000_001)
        #expect(colors[0].z == 1 && colors[1] == SIMD4(0, 1, 0, 1))
        #expect(imported.pointNormals == [SIMD3(0, 0, 1), SIMD3(1, 0, 0)])
        #expect(imported.pointTextureCoordinates == [SIMD2(0.75, 0.75), SIMD2(0.25, 0.25)])
        #expect(imported.customVertexProperties.isEmpty && imported.sourceVertexIndices == nil)
    }

    @Test("CR-delimited binary headers preserve the first payload byte even when it is LF")
    func binaryCarriageReturn() throws {
        let header = "ply\rformat binary_little_endian 1.0\relement vertex 1\rproperty uchar x\rproperty char y\rproperty uchar z\rend_header\r"
        let asset = try PLYLoader.load(Data(header.utf8) + Data([10, 255, 1]))
        guard case .pointCloud(let cloud) = asset.geometry else {
            Issue.record("Expected one binary point")
            return
        }
        #expect(cloud.positions == [SIMD3(10, -1, 1)])
    }

    @Test("Malformed PLY input fails without returning a partial geometry")
    func invalidInputs() throws {
        let positions = "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        let rows = "0 0 0\n1 0 0\n0 1 0\n"
        func file(_ header: String, _ body: String = rows) -> Data {
            Data(("ply\nformat ascii 1.0\n" + header + "end_header\n" + body).utf8)
        }
        for invalid in [
            Data("ply\nformat ascii 2.0\n".utf8),
            Data("ply\nformat binary_big_endian 1.0\nelement vertex 9".utf8),
            file(positions + "property float x\n"),
            file(positions + "property list float int bad_count\n"),
            file(positions + "element vertex 0\n"),
            file(positions, "0 0 0\n1 0 0"),
            file(positions, rows + "99"),
            file(positions, "NaN 0 0\n1 0 0\n0 1 0\n"),
            file(positions.replacingOccurrences(of: "float x", with: "uchar x"), "-1 0 0\n1 0 0\n0 1 0\n"),
        ] {
            #expect(throws: PLYLoader.Failure.self) { try PLYLoader.load(invalid) }
        }
        #expect(throws: PLYLoader.Failure.size) {
            try PLYLoader.load(file(positions.replacingOccurrences(of: "vertex 3", with: "vertex 5000000")))
        }
        let faceHeader = positions + "element face 1\nproperty list int32 int32 vertex_index\n"
        for face in ["-1", "65537", "3 0 1", "3 0 1 -1", "3 0 1 3", "5 0 1 2 1 0"] {
            #expect(throws: PLYLoader.Failure.self) { try PLYLoader.load(file(faceHeader, rows + face)) }
        }
        let uvHeader = faceHeader + "property list uchar float texcoord\n"
        #expect(throws: PLYLoader.Failure.invalidFace(0)) {
            try PLYLoader.load(file(uvHeader, rows + "3 0 1 2 4 0 0 1 1"))
        }
        #expect(throws: PLYLoader.Failure.customProperty("missing")) {
            try PLYLoader.load(file(positions), options: .init(customVertexProperties: ["missing"]))
        }
        let binary = try PLYExporter.data(from: PointCloud(positions: [.zero]), options: .init(format: .binaryBigEndian))
        #expect(throws: PLYLoader.Failure.self) { try PLYLoader.load(binary.dropLast()) }
        #expect(throws: PLYLoader.Failure.self) { try PLYLoader.load(binary + Data([0])) }
    }
}
