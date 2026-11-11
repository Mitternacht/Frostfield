import Foundation
import Frostfield
import Testing

struct STLLoaderTests {
    @Test("Native STL round-trip preserves facet corner order and generated normals",
          arguments: [STLExporter.Format.ascii, .binary])
    func roundTrip(format: STLExporter.Format) throws {
        let source = try MeshGeometry(vertices: [
            .init(position: SIMD3(-1, 0, 0), normal: SIMD3(0, 1, 0)),
            .init(position: SIMD3(1, 0, 0), normal: SIMD3(0, 1, 0)),
            .init(position: SIMD3(0, 1, 0), normal: SIMD3(0, 1, 0)),
            .init(position: SIMD3(0, 0, 1), normal: SIMD3(0, 1, 0)),
        ], indices: [0, 1, 2, 1, 0, 3])
        let bytes = try STLExporter.data(from: source, format: format, name: "Wedge")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("stl")
        defer { try? FileManager.default.removeItem(at: url) }
        try bytes.write(to: url)
        let asset = try STLLoader.load(contentsOf: url)
        #expect(asset.geometry.vertices.map(\.position) == source.indices.map { source.vertices[Int($0)].position })
        #expect(asset.geometry.indices == [0, 1, 2, 3, 4, 5])
        #expect(asset.geometry.vertices.prefix(3).allSatisfy { $0.normal == SIMD3(0, 0, 1) })
        #expect(asset.geometry.vertices.suffix(3).allSatisfy { $0.normal == SIMD3(0, 1, 0) })
        #expect(asset.defaultColor == nil && !asset.geometry.hasVertexColors)
        if format == .ascii {
            #expect(asset.solids.map(\.name) == ["Wedge"])
            #expect(asset.geometry.groups == [.init(start: 0, count: 6)])
        } else {
            #expect(asset.solids.isEmpty && asset.geometry.groups.isEmpty)
        }
        let encoded = try STLExporter.data(from: asset.geometry, format: format, name: "Wedge")
        #expect(encoded == bytes)
    }

    @Test("ASCII STL keeps named solid ranges and accepts zero facet normals")
    func solidGroups() throws {
        let ascii = """
        SOLID First part
          FACET NORMAL 0 0 2
            OUTER LOOP
              VERTEX 0 0 0
              VERTEX 1 0 0
              VERTEX 0 1 0
            ENDLOOP
          ENDFACET
        ENDSOLID First part
        solid Empty part
        endsolid
        solid Last part
          facet normal 0 0 0
            outer loop
              vertex -1.25e1 0 0
              vertex 0 0 0
              vertex 0 0 1
            endloop
          endfacet
          facet normal 0 0 0
            outer loop
              vertex 0 0 0
              vertex 0 0 0
              vertex 0 0 0
            endloop
          endfacet
        endsolid Last part
        """
        let bytes = Data([0xEF, 0xBB, 0xBF]) + Data(ascii.replacingOccurrences(of: "\n", with: "\r\n").utf8)
        let asset = try STLLoader.load(bytes)
        #expect(asset.solids.map(\.name) == ["First part", "Empty part", "Last part"])
        #expect(asset.solids.map(\.indexRange) == [0..<3, 3..<3, 3..<9])
        #expect(asset.geometry.groups == [
            .init(start: 0, count: 3, materialIndex: 0), .init(start: 3, count: 6, materialIndex: 2),
        ])
        #expect(asset.geometry.vertices[0].normal == SIMD3(0, 0, 2))
        #expect(asset.geometry.vertices[3].position.x == -12.5)
        #expect(asset.geometry.vertices[3].normal == SIMD3(0, -1, 0))
        #expect(asset.geometry.vertices[6].normal == SIMD3(0, 1, 0))
    }

    @Test("Binary STL gives exact layout precedence and imports Magics default and facet colors")
    func magicsColors() throws {
        let sharedColor: UInt16 = 0xFFFF
        let facetColor: UInt16 = 1 | (16 << 5) | (31 << 10)
        let file = binary(attributes: [sharedColor, facetColor], color: [128, 64, 0, 129])
        let asset = try STLLoader.load(file)
        let defaultColor = try #require(asset.defaultColor)
        #expect(abs(defaultColor.x - 0.215_860_5) < 0.000_001)
        #expect(abs(defaultColor.y - 0.051_269_46) < 0.000_001)
        #expect(defaultColor.z == 0 && defaultColor.w == Float(129) / 255)
        #expect(asset.geometry.vertices.prefix(3).allSatisfy { $0.color == defaultColor })
        let custom = try #require(asset.geometry.vertices[3].color)
        #expect(abs(custom.x - 0.002_496_75) < 0.000_001)
        #expect(abs(custom.y - 0.229_274_77) < 0.000_001)
        #expect(custom.z == 1 && custom.w == defaultColor.w)
        #expect(asset.geometry.vertices.suffix(3).allSatisfy { $0.color == custom })
        #expect(asset.solids.isEmpty)
        let plain = try STLLoader.load(binary(attributes: [sharedColor, facetColor]))
        #expect(plain.defaultColor == nil && !plain.geometry.hasVertexColors)
    }

    @Test("Malformed, empty, truncated and nonfinite STL data fails")
    func malformed() throws {
        let data = binary(attributes: [0])
        #expect(throws: STLLoader.Failure.self) { try STLLoader.load(data.dropLast()) }
        #expect(throws: STLLoader.Failure.self) { try STLLoader.load(data + Data([0])) }
        #expect(throws: STLLoader.Failure.noFacets) { try STLLoader.load(binary(attributes: [])) }
        #expect(throws: STLLoader.Failure.invalidFacet(0)) {
            try STLLoader.load(binary(attributes: [0], normal: SIMD3(.infinity, 0, 0)))
        }
        #expect(throws: STLLoader.Failure.noFacets) {
            try STLLoader.load(Data("solid empty\nendsolid empty".utf8))
        }
        for ascii in [
            "solid A\nfacet normal 0 0 1 outer loop vertex 0 0 0 endloop endfacet\nendsolid A",
            "solid A\nfacet normal 0 0 1 outer loop vertex 0 0 0 vertex 1 0 0 vertex 0 1 0",
            "solid A\nfacet normal 0 0 1 outer loop vertex 0 0 0 vertex 1 0 0 vertex 0 1 0 vertex 1 1 0 endloop endfacet\nendsolid A",
            "solid A\nfacet normal 0 0 1 outer loop vertex nan 0 0 vertex 1 0 0 vertex 0 1 0 endloop endfacet\nendsolid A",
        ] {
            #expect(throws: STLLoader.Failure.self) { try STLLoader.load(Data(ascii.utf8)) }
        }
    }

    private func binary(
        attributes: [UInt16], color: [UInt8]? = nil, normal: SIMD3<Float> = SIMD3(0, 0, 1)
    ) -> Data {
        let title = "solid Binary title"
        var bytes = Data(title.utf8)
        bytes.append(Data(repeating: 0, count: 80 - bytes.count))
        if let color { bytes.replaceSubrange(70..<80, with: Array("COLOR=".utf8) + color) }
        func append<T: FixedWidthInteger>(_ integer: T) {
            var value = integer.littleEndian
            withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
        }
        append(UInt32(attributes.count))
        for attribute in attributes {
            for vector in [normal, .zero, SIMD3(1, 0, 0), SIMD3(0, 1, 0)] {
                append(vector.x.bitPattern)
                append(vector.y.bitPattern)
                append(vector.z.bitPattern)
            }
            append(attribute)
        }
        return bytes
    }
}
