import Foundation
import Frostfield
import Testing
import zlib

struct VTKXMLLoaderTests {
    @Test("ASCII XML PolyData keeps named normals and ordered polygons and strips")
    func asciiRendering() throws {
        let mesh = try VTKLoader.load(Self.xml("""
            <PointData Normals="authored">
              <DataArray type="Float32" Name="unused" format="ascii">0</DataArray>
              <DataArray type="Float32" Name="authored" NumberOfComponents="3" format="ascii">
                0 0 1 0 0 1 0 0 1 0 0 1
              </DataArray>
            </PointData>
            <Points>
              <DataArray type="Float32" NumberOfComponents="3" format="ascii">
                -1 -1 -3 1 -1 -3 -1 1 -3 1 1 -3
              </DataArray>
            </Points>
            <Strips>
              <DataArray type="Int32" Name="offsets">4</DataArray>
              <DataArray type="Int64" Name="connectivity">0 1 2 3</DataArray>
            </Strips>
            <Polys>
              <DataArray type="UInt32" Name="connectivity">0 1 3 2</DataArray>
              <DataArray type="UInt64" Name="offsets">4</DataArray>
            </Polys>
            """, attributes: "NumberOfPoints=\"4\" NumberOfStrips=\"1\" NumberOfPolys=\"1\""))
        #expect(mesh.vertices.map(\.normal) == Array(repeating: SIMD3(0, 0, 1), count: 4))
        #expect(mesh.indices == [0, 1, 2, 1, 3, 2, 0, 1, 3, 0, 3, 2])

        let fixture = try PrimitiveRenderFixture()
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: mesh,
            material: .classic(ClassicMaterial(model: .normal))
        )
        let result = try fixture.render(fixture.graph([SceneNode(mesh: renderer)]))
        #expect(result.statistics.submittedTriangleCount == 4)
        #expect(isNearly(result.pixels[32 * 64 + 32], SIMD4(0.5, 0.5, 1, 1)))
        #expect(result.pixels[10 * 64 + 10] == .zero)
    }

    @Test("Big-endian XML PolyData decodes UInt64 zlib headers and multiple blocks")
    func multiBlock() throws {
        let vectors: [Double] = [-1, -1, -3, 1, -1, -3, 0, 1, -3]
        let points = try Self.zlib(Self.words(vectors.map(\.bitPattern)), blockSize: 32)
        let offsets = try Self.zlib(Self.words([3] as [UInt64]), blockSize: 8)
        let connectivity = try Self.zlib(Self.words([0, 1, 2] as [Int64]), blockSize: 16)
        let document = Self.xml("""
            <Points><DataArray type="Float64" NumberOfComponents="3" format="binary">\(points)</DataArray></Points>
            <Polys>
              <DataArray type="UInt64" Name="offsets" format="binary">\(offsets)</DataArray>
              <DataArray type="Int64" Name="connectivity" format="binary">\(connectivity)</DataArray>
            </Polys>
            """, root: "byte_order=\"BigEndian\" header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\"")
        let mesh = try VTKLoader.load(document)
        #expect(mesh.vertices.map(\.position) == [SIMD3(-1, -1, -3), SIMD3(1, -1, -3), SIMD3(0, 1, -3)])
        #expect(mesh.indices == [0, 1, 2])

        let invalidHeader = Self.words([3, 32, 8, 1, 1, 1] as [UInt64]).base64EncodedString()
        let invalidPoints = document.replacingUTF8(points, with: invalidHeader + "AAAA")
        #expect(throws: VTKLoader.Failure.malformed("XML zlib block")) { try VTKLoader.load(invalidPoints) }
    }

    @Test("Appended XML offsets count raw base64 bytes across CRLF whitespace")
    func appended() throws {
        func blob<T: FixedWidthInteger>(_ values: [T]) -> String {
            var bytes = Data()
            let count = UInt32(values.count * MemoryLayout<T>.size).littleEndian
            withUnsafeBytes(of: count) { bytes.append(contentsOf: $0) }
            for value in values {
                withUnsafeBytes(of: value.littleEndian) { bytes.append(contentsOf: $0) }
            }
            return bytes.base64EncodedString()
        }
        let points = blob([Float(-1), -1, -3, 1, -1, -3, 0, 1, -3].map(\.bitPattern)) + "\r\n"
        let connectivity = blob([0, 1, 2] as [Int32])
        let offsets = blob([3] as [Int32])
        let pointOffset = 0
        let connectivityOffset = points.utf8.count
        let offsetOffset = connectivityOffset + connectivity.utf8.count
        let document = Self.xml("""
            <Points><DataArray type="Float32" NumberOfComponents="3" format="appended" offset="\(pointOffset)"/></Points>
            <Polys>
              <DataArray type="Int32" Name="connectivity" format="appended" offset="\(connectivityOffset)"/>
              <DataArray type="Int32" Name="offsets" format="appended" offset="\(offsetOffset)"/>
            </Polys>
            """, appended: "<AppendedData encoding=\"base64\">_\(points)\(connectivity)\(offsets)</AppendedData>")
        let mesh = try VTKLoader.load(document)
        #expect(mesh.vertices[2].position == SIMD3(0, 1, -3))
        #expect(mesh.indices == [0, 1, 2])
        let broken = document.replacingUTF8("offset=\"\(offsetOffset)\"", with: "offset=\"9999\"")
        #expect(throws: VTKLoader.Failure.malformed("appended offset")) { try VTKLoader.load(broken) }
    }

    @Test("XML PolyData rejects DTDs, truncated vectors, and out-of-range offsets and indices")
    func malformed() throws {
        let points = """
            <Points><DataArray type="Float32" NumberOfComponents="3">0 0 0 1 0 0 0 1 0</DataArray></Points>
            """
        let poly = """
            <Polys><DataArray type="Int32" Name="connectivity">0 1 2</DataArray>
            <DataArray type="Int32" Name="offsets">3</DataArray></Polys>
            """
        let document = Self.xml(points + poly)
        #expect(throws: VTKLoader.Failure.unsupported("XML encoding")) {
            try VTKLoader.load(Data("<?xml version=\"1.0\" encoding=\"UTF-7\"?>".utf8) + document)
        }
        #expect(throws: VTKLoader.Failure.unsupported("XML DTD")) {
            try VTKLoader.load(Data("<!DOCTYPE VTKFile [<!ENTITY value \"external\">]>".utf8) + document)
        }
        #expect(throws: VTKLoader.Failure.truncated) {
            try VTKLoader.load(Self.xml(points.replacingOccurrences(of: "0 1 0</", with: "</") + poly))
        }
        #expect(throws: VTKLoader.Failure.malformed("Polys connectivity")) {
            try VTKLoader.load(Self.xml(points + poly.replacingOccurrences(of: ">0 1 2<", with: ">0 1 3<")))
        }
        #expect(throws: VTKLoader.Failure.size) {
            try VTKLoader.load(Self.xml(points + poly.replacingOccurrences(of: ">3<", with: ">999999999<")))
        }
        #expect(throws: VTKLoader.Failure.unsupported("XML appended encoding")) {
            try VTKLoader.load(Self.xml(points + poly, appended: "<AppendedData encoding=\"raw\">_ABCD</AppendedData>"))
        }
    }

    private static func xml(
        _ body: String, attributes: String = "NumberOfPoints=\"3\" NumberOfPolys=\"1\"",
        root: String = "", appended: String = ""
    ) -> Data {
        Data("""
            <VTKFile type="PolyData" \(root)><PolyData><Piece \(attributes)>
            \(body)
            </Piece></PolyData>\(appended)</VTKFile>
            """.utf8)
    }

    private static func words<T: FixedWidthInteger>(_ values: [T]) -> Data {
        values.reduce(into: Data()) { data, value in
            withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
        }
    }

    private static func zlib(_ raw: Data, blockSize: Int) throws -> String {
        var sizes: [UInt64] = []
        var payload = Data()
        for offset in stride(from: 0, to: raw.count, by: blockSize) {
            let block = raw[offset..<min(offset + blockSize, raw.count)]
            var count = compressBound(UInt(block.count))
            var encoded = Data(count: Int(count))
            let status = block.withUnsafeBytes { input in
                encoded.withUnsafeMutableBytes { output in
                    compress2(
                        output.bindMemory(to: UInt8.self).baseAddress!, &count,
                        input.bindMemory(to: UInt8.self).baseAddress!, UInt(block.count), Z_BEST_SPEED
                    )
                }
            }
            try #require(status == Z_OK)
            sizes.append(UInt64(count))
            payload.append(encoded.prefix(Int(count)))
        }
        let header = words([UInt64(sizes.count), UInt64(blockSize), UInt64(raw.count % blockSize)] + sizes)
        return header.base64EncodedString() + payload.base64EncodedString()
    }
}

private extension Data {
    func replacingUTF8(_ old: String, with new: String) -> Data {
        Data(String(decoding: self, as: UTF8.self).replacingOccurrences(of: old, with: new).utf8)
    }
}
