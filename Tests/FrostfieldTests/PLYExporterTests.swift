import Foundation
import Frostfield
import Metal
import ModelIO
import simd
import Testing

struct PLYExporterTests {
    @Test("PLY writes exact normals, bottom-left UVs, sRGB bytes, custom columns, and face records")
    func structure() throws {
        let mesh = try triangle()
        let properties: [PLYExporter.ScalarProperty] = [
            .init(name: "confidence", values: [0.125, -0.5, 1]),
            .init(name: "temperature_2", values: [273.5, 0, 500]),
        ]
        let text = String(decoding: try PLYExporter.data(from: mesh, options: .init(scalarProperties: properties)), as: UTF8.self)
        #expect(text == """
        ply
        format ascii 1.0
        element vertex 3
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property float s
        property float t
        property uchar red
        property uchar green
        property uchar blue
        property float confidence
        property float temperature_2
        element face 1
        property list uchar int vertex_index
        end_header
        1.25 -2.5 0.125 0.0 0.0 1.0 0.25 0.875 128 0 255 0.125 273.5
        2.25 -2.5 0.125 0.0 0.0 1.0 0.0 1.0 255 255 255 -0.5 0.0
        1.25 -1.5 0.125 0.0 0.0 1.0 0.0 1.0 255 0 0 1.0 500.0
        3 0 1 2

        """)

        for format in [PLYExporter.Format.binaryLittleEndian, .binaryBigEndian] {
            let file = try PLYExporter.data(from: mesh, options: .init(format: format, scalarProperties: properties))
            let headerEnd = try #require(file.range(of: Data("end_header\n".utf8))).upperBound
            let header = String(decoding: file[..<headerEnd], as: UTF8.self)
            #expect(header.contains("format \(format.rawValue) 1.0\n"))
            let bytes = Array(file[headerEnd...])
            try #require(bytes.count == 3 * 43 + 13)
            let firstFloatBytes: [UInt8] = format == .binaryLittleEndian ? [0, 0, 160, 63] : [63, 160, 0, 0]
            #expect(Array(bytes[0..<4]) == firstFloatBytes)
            #expect(Array(bytes[32..<35]) == [128, 0, 255])
            let scalarBytes: [UInt8] = format == .binaryLittleEndian ? [0, 0, 0, 62] : [62, 0, 0, 0]
            #expect(Array(bytes[35..<39]) == scalarBytes)
            let faceBytes: [UInt8] = format == .binaryLittleEndian
                ? [3, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
                : [3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2]
            #expect(Array(bytes.suffix(13)) == faceBytes)
        }
    }

    @Test("Typed PLY columns preserve the exact scalar widths and values in every encoding")
    func typedProperties() throws {
        let cloud = try PointCloud(positions: [.zero, SIMD3(1, 2, 3)])
        var legacy = PLYExporter.ScalarProperty(name: "legacy", values: [0, -0.5])
        legacy.values[0] = 0.25
        let precise = Double(1).nextUp
        let typed: [PLYExporter.TypedScalarProperty] = [
            .init(name: "signed8", values: .int8([.min, .max])),
            .init(name: "unsigned8", values: .uint8([.max, 3])),
            .init(name: "signed16", values: .int16([.min, 4_660])),
            .init(name: "unsigned16", values: .uint16([.max, 43_981])),
            .init(name: "signed32", values: .int32([.min, -16_777_217])),
            .init(name: "unsigned32", values: .uint32([.max, 16_777_217])),
            .init(name: "single", values: .float32([1.5, Float(bitPattern: 0x80000000)])),
            .init(name: "precise", values: .float64([precise, -precise])),
        ]
        let columns = """
        element vertex 2
        property float x
        property float y
        property float z
        property float legacy
        property char signed8
        property uchar unsigned8
        property short signed16
        property ushort unsigned16
        property int signed32
        property uint unsigned32
        property float single
        property double precise
        end_header

        """
        let ascii = try PLYExporter.data(from: cloud, options: .init(
            scalarProperties: [legacy], typedScalarProperties: typed
        ))
        let text = """
        0.0 0.0 0.0 0.25 -128 255 -32768 65535 -2147483648 4294967295 1.5 1.0000000000000002
        1.0 2.0 3.0 -0.5 127 3 4660 43981 -16777217 16777217 -0.0 -1.0000000000000002

        """
        #expect(String(decoding: ascii, as: UTF8.self) == "ply\nformat ascii 1.0\n" + columns + text)

        // Per-field big-endian bytes. Reversing fields individually gives the little-endian reference.
        let records: [[UInt8]] = [
            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0x3E, 0x80, 0, 0],
            [0x80], [0xFF], [0x80, 0], [0xFF, 0xFF], [0x80, 0, 0, 0], [0xFF, 0xFF, 0xFF, 0xFF],
            [0x3F, 0xC0, 0, 0], [0x3F, 0xF0, 0, 0, 0, 0, 0, 1],
            [0x3F, 0x80, 0, 0], [0x40, 0, 0, 0], [0x40, 0x40, 0, 0], [0xBF, 0, 0, 0],
            [0x7F], [3], [0x12, 0x34], [0xAB, 0xCD], [0xFE, 0xFF, 0xFF, 0xFF], [1, 0, 0, 1],
            [0x80, 0, 0, 0], [0xBF, 0xF0, 0, 0, 0, 0, 0, 1],
        ]
        for format in [PLYExporter.Format.binaryLittleEndian, .binaryBigEndian] {
            let file = try PLYExporter.data(from: cloud, options: .init(
                format: format, scalarProperties: [legacy], typedScalarProperties: typed
            ))
            let header = "ply\nformat \(format.rawValue) 1.0\n" + columns
            let expectedBytes = records.flatMap { field in
                format == .binaryLittleEndian ? Array(field.reversed()) : field
            }
            #expect(file == Data(header.utf8) + Data(expectedBytes))
        }
    }

    @Test("Typed and legacy custom columns share validation")
    func typedPropertyValidation() throws {
        let cloud = try PointCloud(positions: [.zero, SIMD3(1, 0, 0)])
        #expect(throws: PLYExporter.Failure.propertyName("LEGACY")) {
            try PLYExporter.data(from: cloud, options: .init(
                scalarProperties: [.init(name: "legacy", values: [0, 1])],
                typedScalarProperties: [.init(name: "LEGACY", values: .uint8([1, 2]))]
            ))
        }
        #expect(throws: PLYExporter.Failure.propertyCount("label")) {
            try PLYExporter.data(from: cloud, options: .init(
                typedScalarProperties: [.init(name: "label", values: .uint32([.max]))]
            ))
        }
        for values: PLYExporter.TypedScalarProperty.Values in [.float32([0, .infinity]), .float64([0, .nan])] {
            #expect(throws: PLYExporter.Failure.propertyValue("weight", vertex: 1)) {
                try PLYExporter.data(from: cloud, options: .init(
                    typedScalarProperties: [.init(name: "weight", values: values)]
                ))
            }
        }
    }

    @Test("All PLY mesh encodings round-trip through Model I/O", arguments: PLYExporter.Format.allCases)
    func roundTrip(format: PLYExporter.Format) throws {
        let mesh = try triangle()
        let file = try PLYExporter.data(from: mesh, options: .init(
            format: format, scalarProperties: [.init(name: "confidence", values: [0.125, 0.25, 0.5])]
        ))
        try withPLY(file) { url in
            let imported = try ModelIOSceneLoader.load(contentsOf: url)
            let geometry = try #require(imported.scene.meshes.first?.primitives.first)
            #expect(Set(geometry.vertices.map(\.position)) == Set(mesh.vertices.map(\.position)))
            #expect(geometry.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })
            #expect(geometry.indices.count == 3)
            let points = geometry.indices.map { geometry.vertices[Int($0)].position }
            #expect(simd_cross(points[1] - points[0], points[2] - points[0]).z > 0)
            let native = MDLAsset(url: url)
            let nativeMesh = try #require(native.childObjects(of: MDLMesh.self).first as? MDLMesh)
            let positions = try #require(nativeMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributePosition))
            let colors = try #require(nativeMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeColor))
            try #require(positions.format == .float3 && colors.format == .float3)
            let expectedColors: [SIMD3<Float>] = [SIMD3(128.0 / 255, 0, 1), SIMD3(repeating: 1), SIMD3(1, 0, 0)]
            for index in 0..<nativeMesh.vertexCount {
                let position = float3(positions, index: index)
                let original = try #require(mesh.vertices.firstIndex { $0.position == position })
                #expect(simd_distance(float3(colors, index: index), expectedColors[original]) < 1e-6)
            }
        }
    }

    @Test("Point clouds have no face element and are readable by the native parser")
    func pointCloud() throws {
        let cloud = try PointCloud(
            positions: [SIMD3(1.25, -2.5, 0.125), SIMD3(3, 4, 5)],
            colors: [SIMD4(0, 1, 0, 1), SIMD4(1, 0, 0, 0.25)], sizes: [1.5, 0]
        )
        let sizes = try #require(cloud.sizes)
        let options = PLYExporter.Options(
            format: .binaryBigEndian, scalarProperties: [.init(name: "point_size", values: sizes)]
        )
        let file = try PLYExporter.data(from: cloud, options: options)
        let headerEnd = try #require(file.range(of: Data("end_header\n".utf8))).upperBound
        let header = String(decoding: file[..<headerEnd], as: UTF8.self)
        #expect(header.contains("element vertex 2\n"))
        #expect(header.contains("property float point_size\n"))
        #expect(!header.contains("element face") && !header.contains("property float nx") && !header.contains("property float s\n"))
        #expect(file.count - headerEnd == 2 * 19)
        try withPLY(file) { url in
            let asset = MDLAsset(url: url)
            let mesh = try #require(asset.childObjects(of: MDLMesh.self).first as? MDLMesh)
            #expect(mesh.vertexCount == 2)
            let pointElements = (mesh.submeshes ?? []).compactMap { $0 as? MDLSubmesh }
            #expect(pointElements.count == 1 && pointElements.first?.geometryType == .points)
            let positions = try #require(mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributePosition, as: .float3))
            for (index, expected) in cloud.positions.enumerated() {
                #expect(float3(positions, index: index) == expected)
            }
        }
    }

    @Test("Scene PLY combines world point rows and mirrored mesh instances", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func mixedScene() throws {
        let fixture = try PathTestFixture()
        let cloud = try PointCloud(positions: [SIMD3(1, 0, 0), SIMD3(2, 0, 0)])
        let points = try PointsRenderer(device: fixture.device, cloud: cloud, material: .init())
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: triangle(), supportsInstancing: true
        )
        let placements: [simd_float4x4] = [Float(2), Float(3)].map {
            SceneTransform(translation: SIMD3($0, 0, 0), scale: SIMD3(-1, 1, 1)).matrix
        }
        let instances = try InstanceSet(device: fixture.device, transforms: placements)
        let nodes = [
            SceneNode(transform: .init(translation: SIMD3(2, 0, 0)), primitive: .points(points)),
            SceneNode(mesh: mesh, instances: instances),
        ]
        let root = SceneNode(transform: .init(translation: SIMD3(10, -2, 3)), children: nodes)
        let graph = SceneGraph(roots: [root], camera: fixture.camera)
        let file = try PLYExporter.data(from: graph)
        let text = String(decoding: file, as: UTF8.self)
        #expect(text.contains("element vertex 8\n"))
        let body = try #require(text.components(separatedBy: "end_header\n").last)
        let rows = body.split(separator: "\n").map(String.init)
        try #require(rows.count == 10)
        #expect(rows[0] == "13.0 -2.0 3.0 0.0 0.0 0.0 0.0 0.0 255 255 255")
        #expect(rows[1].hasPrefix("14.0 -2.0 3.0 "))
        #expect(rows[2].hasPrefix("10.75 -4.5 3.125 "))
        #expect(rows[5].hasPrefix("11.75 -4.5 3.125 "))
        #expect(rows.suffix(2) == ["3 2 4 3", "3 5 7 6"])

        let verticesOnly = String(decoding: try PLYExporter.data(from: graph, options: .init(
            includeNormals: false, includeTextureCoordinates: false, includeColors: false, includeFaces: false
        )), as: UTF8.self)
        #expect(verticesOnly.contains("element vertex 8\n"))
        #expect(!verticesOnly.contains("element face") && !verticesOnly.contains("property float nx"))
        #expect(!verticesOnly.contains("property float s\n") && !verticesOnly.contains("property uchar red"))
        #expect(verticesOnly.split(separator: "\n").suffix(8).allSatisfy { $0.split(separator: " ").count == 3 })
    }

    @Test("Custom properties reject reserved names, header injection, duplicates, and invalid rows")
    func propertyValidation() throws {
        let cloud = try PointCloud(positions: [.zero])
        for name in ["NX", "foo\nend_header", "foo bar", "end_header", "α", "123", ""] {
            #expect(throws: PLYExporter.Failure.propertyName(name)) {
                try PLYExporter.data(from: cloud, options: .init(scalarProperties: [.init(name: name, values: [0])]))
            }
        }
        #expect(throws: PLYExporter.Failure.propertyName("WEIGHT")) {
            try PLYExporter.data(from: cloud, options: .init(scalarProperties: [
                .init(name: "weight", values: [0]), .init(name: "WEIGHT", values: [1]),
            ]))
        }
        #expect(throws: PLYExporter.Failure.propertyCount("weight")) {
            try PLYExporter.data(from: cloud, options: .init(scalarProperties: [.init(name: "weight", values: [])]))
        }
        #expect(throws: PLYExporter.Failure.propertyValue("weight", vertex: 0)) {
            try PLYExporter.data(from: cloud, options: .init(scalarProperties: [.init(name: "weight", values: [.infinity])]))
        }
    }

    private func triangle() throws -> MeshGeometry {
        try MeshGeometry(vertices: [
            .init(
                position: SIMD3(1.25, -2.5, 0.125), normal: SIMD3(0, 0, 1),
                textureCoordinate: SIMD2(0.25, 0.125), color: SIMD4(0.215_860_5, -0.5, 1.5, 0.25)
            ),
            .init(position: SIMD3(2.25, -2.5, 0.125), normal: SIMD3(0, 0, 1)),
            .init(position: SIMD3(1.25, -1.5, 0.125), normal: SIMD3(0, 0, 1), color: SIMD4(1, 0, 0, 1)),
        ], indices: [0, 1, 2])
    }

    private func float3(_ data: MDLVertexAttributeData, index: Int) -> SIMD3<Float> {
        let pointer = data.dataStart.advanced(by: data.stride * index)
        return SIMD3(
            pointer.loadUnaligned(fromByteOffset: 0, as: Float.self),
            pointer.loadUnaligned(fromByteOffset: 4, as: Float.self),
            pointer.loadUnaligned(fromByteOffset: 8, as: Float.self)
        )
    }

    private func withPLY(_ data: Data, check: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fixture.ply")
        try data.write(to: url)
        try check(url)
    }
}
