import Foundation
import Frostfield
import Testing
import simd

struct DracoExporterTests {
    @Test("Sequential Draco preserves exact connectivity and unquantized vertex data")
    func sequentialMesh() throws {
        let vertices = [
            MeshVertex(position: SIMD3(0, 0, 0), normal: SIMD3(0, 0, 1), secondaryTextureCoordinate: SIMD2(0.25, 0.75)),
            MeshVertex(
                position: SIMD3(1, 0, 0), normal: SIMD3(0, 0, 2), secondaryTextureCoordinate: SIMD2(0.375, 0.5),
                color: SIMD4(0.25, 0.5, 0.75, 0.125)
            ),
            MeshVertex(
                position: SIMD3(0, 1, 0), normal: SIMD3(0, 1, 1), secondaryTextureCoordinate: SIMD2(0.5, 0.25),
                color: SIMD4(1, 0, 1, 0.5)
            ),
            MeshVertex(position: SIMD3(1, 1, 0), normal: SIMD3(0, 0, 1), secondaryTextureCoordinate: SIMD2(0.75, 0.125)),
            MeshVertex(position: SIMD3(3, 4, 5), normal: SIMD3(1, 0, 0), secondaryTextureCoordinate: .zero),
        ]
        let indices: [UInt32] = [0, 1, 2, 2, 2, 3]
        let mesh = try MeshGeometry(vertices: vertices, indices: indices)
        let data = try DracoExporter.encode(
            mesh, method: .sequential,
            options: .init(encodeSpeed: 10, decodeSpeed: 0, quantization: .none, colorEncoding: .linear),
            textureCoordinateSet: .secondary
        )
        let decoded = try DracoDecoder.decode(data)
        try #require(data.count >= 9)
        #expect(Array(data.prefix(5)) == Array("DRACO".utf8))
        #expect(Array(data[7...8]) == [1, 0])
        #expect(decoded.indices == indices && decoded.pointCount == vertices.count)
        #expect(try attribute(.position, in: decoded) == vertices.flatMap { [$0.position.x, $0.position.y, $0.position.z] })
        #expect(try attribute(.normal, in: decoded) == vertices.flatMap { [$0.normal.x, $0.normal.y, $0.normal.z] })
        #expect(try attribute(.textureCoordinate, in: decoded) == vertices.flatMap {
            let uv = $0.secondaryTextureCoordinate!
            return [uv.x, uv.y]
        })
        #expect(try attribute(.color, in: decoded) == vertices.flatMap {
            let color = $0.color ?? SIMD4(repeating: 1)
            return [color.x, color.y, color.z, color.w]
        })
        #expect(throws: DracoExporter.Failure.edgeBreakerTopology) {
            try DracoExporter.encode(mesh, method: .edgeBreaker)
        }
    }

    @Test("EdgeBreaker preserves oriented cube corners, seams, UVs and sRGB colors")
    func edgeBreakerMesh() throws {
        let cube = MeshGeometry.cube()
        let vertices = cube.vertices.enumerated().map { index, vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                color: SIMD4(0.25, 0.5, 0, Float(index + 1) / 32)
            )
        }
        let mesh = try MeshGeometry(vertices: vertices, indices: cube.indices)
        let data = try DracoExporter.encode(
            mesh,
            options: .init(encodeSpeed: 3, decodeSpeed: 4, quantization: .init(normal: 12, textureCoordinate: 12, color: 12),
                           colorEncoding: .sRGB)
        )
        let decoded = try DracoDecoder.decode(data)
        try #require(data.count >= 9)
        #expect(Array(data[7...8]) == [1, 1])
        #expect(decoded.indices.count == cube.indices.count)
        let positions = try attribute(.position, in: decoded)
        let normals = try attribute(.normal, in: decoded)
        let uvs = try attribute(.textureCoordinate, in: decoded)
        let colors = try attribute(.color, in: decoded)
        var match: [UInt32] = []
        for point in 0..<decoded.pointCount {
            let position = vector(positions, at: point)
            let normal = vector(normals, at: point)
            let uv = SIMD2(uvs[point * 2], uvs[point * 2 + 1])
            let alpha = colors[point * 4 + 3]
            let vertex = try #require(vertices.firstIndex {
                simd_distance($0.position, position) < 0.0001 && simd_distance($0.normal, normal) < 0.002
                    && simd_distance($0.textureCoordinate, uv) < 0.001
            })
            #expect(abs(colors[point * 4] - 0.537_099) < 0.001)
            #expect(abs(colors[point * 4 + 1] - 0.735_357) < 0.001)
            #expect(colors[point * 4 + 2] == 0)
            #expect(abs(alpha - Float(vertex + 1) / 32) < 0.001)
            match.append(UInt32(vertex))
        }
        let decodedFaces = faces(decoded.indices.map { match[Int($0)] })
        #expect(decodedFaces == faces(cube.indices))
    }

    @Test("Quantization changes precision without changing sequential topology")
    func quantizationPrecision() throws {
        let mesh = try MeshGeometry(vertices: [
            MeshVertex(position: .zero, normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(1, 0.123_456, 0), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0.345_678, 1, 0), normal: SIMD3(0, 0, 1)),
        ], indices: [0, 1, 2])
        func bake(_ bits: Int) throws -> DracoDecoder.Mesh {
            let data = try DracoExporter.encode(
                mesh, method: .sequential, options: .init(quantization: .init(position: bits)),
                includeNormals: false, textureCoordinateSet: nil
            )
            return try DracoDecoder.decode(data)
        }
        let coarse = try bake(4)
        let fine = try bake(20)
        #expect(coarse.indices == mesh.indices && fine.indices == mesh.indices)
        #expect(coarse.attributes.map(\.semantic) == [.position])
        let coarseValue = try attribute(.position, in: coarse)[4]
        let fineValue = try attribute(.position, in: fine)[4]
        #expect(abs(coarseValue - 0.123_456) > 0.005)
        #expect(abs(fineValue - 0.123_456) < 0.000_002)
    }

    @Test("Slow EdgeBreaker encodes high-precision positions and tiny authored normal directions")
    func highPrecisionEdgeBreaker() throws {
        let plane = try MeshGeometry.plane(width: 8, height: 9, columns: 24, rows: 24)
        let mesh = try MeshGeometry(vertices: plane.vertices.map {
            MeshVertex(position: $0.position, normal: SIMD3(0, 0, 1e-25), textureCoordinate: $0.textureCoordinate)
        }, indices: plane.indices)
        let data = try DracoExporter.encode(
            mesh, options: .init(encodeSpeed: 0, decodeSpeed: 0, quantization: .init(position: 30, normal: 12))
        )
        let decoded = try DracoDecoder.decode(data)
        #expect(decoded.indices.count == mesh.indices.count)
        #expect(decoded.pointCount == mesh.vertices.count)
        let normals = try attribute(.normal, in: decoded)
        for point in 0..<decoded.pointCount {
            #expect(simd_distance(vector(normals, at: point), SIMD3(0, 0, 1)) < 0.002)
        }
    }

    @Test("Both point-cloud methods emit reproducible standalone streams")
    func pointClouds() throws {
        let cloud = try PointCloud(
            positions: [
                SIMD3(0, 0, 0), SIMD3(0.125, -0.75, 2), SIMD3(-2, 0.5, 1), SIMD3(1, 2, -0.25),
                SIMD3(0.125, -0.75, 2), SIMD3(0.125, -0.75, 2),
            ],
            colors: [
                SIMD4(0.25, 0.5, 0, 0.125), SIMD4(0, 1, 0.25, 0.25), SIMD4(0.5, 0.25, 1, 0.375),
                SIMD4(0.125, 0, 0.5, 1), SIMD4(1, 0, 0.25, 0.75), SIMD4(1, 0, 0.25, 0.75),
            ],
            sizes: [1, 2, 3, 0, 5, 1]
        )
        let lossless = DracoExporter.Options(quantization: .none, colorEncoding: .linear)
        let spatial = DracoExporter.Options(encodeSpeed: 4, decodeSpeed: 2, quantization: .init(color: 14), colorEncoding: .sRGB)
        let sequential = try DracoExporter.encode(cloud, method: .sequential, options: lossless)
        let kdTree = try DracoExporter.encode(cloud, method: .kdTree, options: spatial)
        try #require(sequential.count >= 9 && kdTree.count >= 9)
        #expect(Array(sequential[7...8]) == [0, 0])
        #expect(Array(kdTree[7...8]) == [0, 1])
        #expect(try DracoExporter.encode(cloud, method: .sequential, options: lossless) == sequential)
        #expect(try DracoExporter.encode(cloud, method: .kdTree, options: spatial) == kdTree)
        #expect(throws: DracoDecoder.Failure.unsupportedGeometry) { try DracoDecoder.decode(kdTree) }
        if let directory = ProcessInfo.processInfo.environment["FROSTFIELD_DRACO_TEST_OUTPUT"] {
            let output = URL(fileURLWithPath: directory, isDirectory: true)
            try sequential.write(to: output.appendingPathComponent("sequential-points.drc"))
            try kdTree.write(to: output.appendingPathComponent("kd-tree-points.drc"))
        }
    }

    @Test("Draco encoding rejects unsupported options, UV layouts and quantization ranges")
    func validation() throws {
        let mesh = MeshGeometry.cube()
        #expect(throws: DracoExporter.Failure.options) {
            try DracoExporter.encode(mesh, options: .init(encodeSpeed: 11))
        }
        #expect(throws: DracoExporter.Failure.options) {
            try DracoExporter.encode(mesh, options: .init(quantization: .init(normal: 1)))
        }
        #expect(throws: DracoExporter.Failure.missingUV1(vertex: 0)) {
            try DracoExporter.encode(mesh, textureCoordinateSet: .secondary)
        }
        let cloud = try PointCloud(positions: [.zero])
        #expect(throws: DracoExporter.Failure.pointCloudQuantization) {
            try DracoExporter.encode(cloud, options: .init(quantization: .none))
        }
        let tiny = try PointCloud(positions: [.zero, SIMD3(.leastNonzeroMagnitude, 0, 0)])
        #expect(throws: DracoExporter.Failure.attributeRange) {
            try DracoExporter.encode(tiny, method: .sequential)
        }
        let huge = try PointCloud(positions: [SIMD3(-.greatestFiniteMagnitude, 0, 0), SIMD3(.greatestFiniteMagnitude, 0, 0)])
        #expect(throws: DracoExporter.Failure.attributeRange) {
            try DracoExporter.encode(huge, method: .sequential)
        }
    }

    private func attribute(_ semantic: DracoDecoder.Semantic, in mesh: DracoDecoder.Mesh) throws -> [Float] {
        let attribute = try #require(mesh.attributes.first { $0.semantic == semantic })
        #expect(attribute.component == .float32)
        return attribute.data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
    }

    private func vector(_ array: [Float], at index: Int) -> SIMD3<Float> {
        SIMD3(array[3 * index], array[3 * index + 1], array[3 * index + 2])
    }

    private func faces(_ indices: [UInt32]) -> Set<SIMD3<UInt32>> {
        Set(stride(from: 0, to: indices.count, by: 3).map { offset in
            let a = indices[offset], b = indices[offset + 1], c = indices[offset + 2]
            if a <= b, a <= c { return SIMD3(a, b, c) }
            if b <= c { return SIMD3(b, c, a) }
            return SIMD3(c, a, b)
        })
    }
}
