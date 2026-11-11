import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PCDLoaderTests {
    @Test("ASCII PCD keeps organized dimensions and extracts packed RGB and typed point attributes")
    func ascii() throws {
        let firstColor = Float(bitPattern: 0x0080FF00)
        let secondColor = Float(bitPattern: 0x000040FF)
        let file = """
            # PCD v0.7
            VERSION .7
            FIELDS x bins y z rgb normal_x normal_y normal_z intensity label
            SIZE 4 4 8 2 4 4 4 4 2 4
            TYPE F F F I F F F F U I
            COUNT 1 3 1 1 1 1 1 1 1 1
            WIDTH 2
            HEIGHT 2
            POINTS 4
            VIEWPOINT 10 0 0 1 0 0 0
            DATA ascii
            -2 99 98 97 0 -4 \(firstColor) 0 0 1 16 -5
            2 99 98 97 0 -4 \(secondColor) 0 1 0 32 10
            -2 99 98 97 2 -4 \(firstColor) 1 0 0 64 -20
            2 99 98 97 2 -4 \(secondColor) 0 0 -1 128 1234
            """.replacingOccurrences(of: "\n", with: "\r\n")
        let cloud = try PCDLoader.load(Data(file.utf8))
        #expect(cloud.width == 2 && cloud.height == 2)
        #expect(cloud.pointCloud.positions == [
            SIMD3(-2, 0, -4), SIMD3(2, 0, -4), SIMD3(-2, 2, -4), SIMD3(2, 2, -4),
        ])
        #expect(cloud.normals == [SIMD3(0, 0, 1), SIMD3(0, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, -1)])
        #expect(cloud.intensities == [16, 32, 64, 128] && cloud.labels == [-5, 10, -20, 1234])
        let colors = try #require(cloud.pointCloud.colors)
        #expect(isNearly(colors[0], SIMD4(0.21586, 1, 0, 1), tolerance: 0.0001))
        #expect(isNearly(colors[1], SIMD4(0, 0.05127, 1, 1), tolerance: 0.0001))
    }

    @Test("Binary PCD reads heterogeneous records in either byte order and preserves RGBA alpha",
          arguments: [PCDLoader.ByteOrder.littleEndian, .bigEndian])
    func binary(_ byteOrder: PCDLoader.ByteOrder) throws {
        var bytes = Self.header(
            fields: "z _ rgba label x y intensity normal_x normal_y normal_z",
            sizes: "4 1 4 4 8 2 2 4 4 4", types: "F U U U F I U F F F",
            points: 2, encoding: "binary"
        )
        func record(_ x: Double, _ y: Int16, _ rgba: UInt32, _ label: UInt32) -> Data {
            var row = Data()
            Self.put(Float(-4).bitPattern, into: &row, order: byteOrder)
            row.append(0xDD)
            Self.put(rgba, into: &row, order: byteOrder)
            Self.put(label, into: &row, order: byteOrder)
            Self.put(x.bitPattern, into: &row, order: byteOrder)
            Self.put(y, into: &row, order: byteOrder)
            Self.put(UInt16(40_000), into: &row, order: byteOrder)
            for component: Float in [0, 1, -0.5] { Self.put(component.bitPattern, into: &row, order: byteOrder) }
            return row
        }
        bytes.append(record(-2, -3, 0x804080FF, 4_000_000_000))
        bytes.append(record(2.25, 5, 0xFF00FF00, 17))
        // Compatibility with the pinned binary fixture's whole unused records.
        bytes.append(record(17, 23, 0, 10))
        let cloud = try PCDLoader.load(bytes, byteOrder: byteOrder)
        #expect(cloud.pointCloud.positions == [SIMD3(-2, -3, -4), SIMD3(2.25, 5, -4)])
        #expect(cloud.labels == [4_000_000_000, 17])
        #expect(cloud.intensities == [40_000, 40_000])
        #expect(cloud.normals == [SIMD3(0, 1, -0.5), SIMD3(0, 1, -0.5)])
        let colors = try #require(cloud.pointCloud.colors)
        #expect(isNearly(colors[0], SIMD4(0.05127, 0.21586, 1, 128 / 255), tolerance: 0.0001))
        #expect(colors[1] == SIMD4(0, 1, 0, 1))
        #expect(throws: PCDLoader.Failure.invalidBody) {
            try PCDLoader.load(bytes + Data([1]), byteOrder: byteOrder)
        }
    }

    @Test("LZF field-major data uses COUNT widths, overlapping references, and omitted padding")
    func compressed() throws {
        let header = Self.header(
            fields: "x _ y signature z rgb label", sizes: "4 1 4 2 4 4 4", types: "F U F U F U I",
            counts: "1 2 1 3 1 1 1", points: 4, encoding: "binary_compressed"
        )
        // X is 16 zero bytes, encoded as one literal and a long overlapping back reference.
        var rest = Data()
        for y: Float in [-2, -1, 1, 2] { Self.put(y.bitPattern, into: &rest) }
        for value in UInt16(0)..<12 { Self.put(value, into: &rest) }
        for z: Float in [-2, -4, -6, -8] { Self.put(z.bitPattern, into: &rest) }
        for rgb: UInt32 in [0xFF0000, 0x00FF00, 0x0000FF, 0x000000] { Self.put(rgb, into: &rest) }
        for label: Int32 in [-42, 3, 7, 1000] { Self.put(label, into: &rest) }
        let encoded = Data([0, 0, 0xE0, 6, 0]) + Self.literals(rest)
        let cloud = try PCDLoader.load(header + Self.lzf(encoded, expandedBytes: 16 + rest.count) + Data([0, 0]))
        #expect(cloud.pointCloud.positions == [
            SIMD3(0, -2, -2), SIMD3(0, -1, -4), SIMD3(0, 1, -6), SIMD3(0, 2, -8),
        ])
        #expect(cloud.pointCloud.colors == [SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1), SIMD4(0, 0, 0, 1)])
        #expect(cloud.labels == [-42, 3, 7, 1000] && cloud.normals == nil && cloud.intensities == nil)
    }

    @Test("PCD bounds invalid schemas, records, and LZF streams")
    func malformed() {
        #expect(throws: PCDLoader.Failure.invalidHeader("FIELDS")) {
            try PCDLoader.load(Self.header(fields: "x x z", points: 1, encoding: "ascii"))
        }
        #expect(throws: PCDLoader.Failure.invalidHeader("POINTS")) {
            try PCDLoader.load(Data("FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nWIDTH 2\nPOINTS 1\nDATA ascii\n".utf8))
        }
        #expect(throws: PCDLoader.Failure.invalidHeader("COUNT")) {
            try PCDLoader.load(Self.header(counts: "1 0 1", points: 1, encoding: "ascii"))
        }
        #expect(throws: PCDLoader.Failure.invalidHeader("x, y, z")) {
            try PCDLoader.load(Self.header(fields: "x y zz", points: 1, encoding: "ascii"))
        }
        #expect(throws: PCDLoader.Failure.invalidPoint(0)) {
            try PCDLoader.load(Self.header(points: 1, encoding: "ascii") + Data("0 nan 0".utf8))
        }
        #expect(throws: PCDLoader.Failure.invalidBody) {
            try PCDLoader.load(Self.header(points: 2, encoding: "binary") + Data(repeating: 0, count: 23))
        }
        #expect(throws: PCDLoader.Failure.size) {
            try PCDLoader.load(Self.header(points: (1 << 22) + 1, encoding: "binary"))
        }
        let compressed = Self.header(points: 1, encoding: "binary_compressed")
        for malformed in [
            Self.lzf(Data([0x20, 0]), expandedBytes: 12), // reference before output
            Self.lzf(Data([0, 0, 0xE0, 10, 0]), expandedBytes: 12), // output overflow
            Self.lzf(Data([0, 0]), expandedBytes: 12), // short expansion
            Self.lzf(Data([12, 0]), expandedBytes: 12), // literal exceeds input
            Self.lzf(Data([0, 0]), expandedBytes: 8), // prefix disagrees with schema
            Self.lzf(Data([0, 0, 0xE0, 2, 0]), expandedBytes: 12) + Data([1]), // nonzero data past prefix
        ] {
            #expect(throws: PCDLoader.Failure.invalidCompression) { try PCDLoader.load(compressed + malformed) }
        }
    }

    @Test("PCD colors and positions reach the Metal point renderer")
    func render() throws {
        let input = Self.header(fields: "x y z rgb", sizes: "4 4 4 4", types: "F F F U", points: 2, encoding: "ascii")
            + Data("-2 0 -4 8453888\n2 0 -4 16639\n".utf8)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("pcd")
        try input.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let asset = try PCDLoader.load(contentsOf: file)
        let fixture = try PrimitiveRenderFixture(side: 64)
        let renderer = try PointsRenderer(
            device: fixture.device, cloud: asset.pointCloud,
            material: PointsMaterial(size: .pixels(6), color: SIMD4(1, 1, 1, 1))
        )
        let pixels = try fixture.render(fixture.graph([SceneNode(primitive: .points(renderer))])).pixels
        #expect(isNearly(pixels[32 * 64 + 16], SIMD4(0.21586, 1, 0, 1)))
        #expect(isNearly(pixels[32 * 64 + 48], SIMD4(0, 0.05127, 1, 1)))
        #expect(pixels.filter { $0.w == 1 }.count == 72)
        #expect(pixels[32 * 64 + 32].w == 0)
    }

    private static func header(
        fields: String = "x y z", sizes: String = "4 4 4", types: String = "F F F",
        counts: String? = nil, points: Int, encoding: String
    ) -> Data {
        let countLine = counts.map { "COUNT \($0)\n" } ?? ""
        return Data("VERSION 0.7\nFIELDS \(fields)\nSIZE \(sizes)\nTYPE \(types)\n\(countLine)WIDTH \(points)\nDATA \(encoding)\n".utf8)
    }

    private static func put<T: FixedWidthInteger>(
        _ value: T, into bytes: inout Data, order: PCDLoader.ByteOrder = .littleEndian
    ) {
        let encoded = order == .littleEndian ? value.littleEndian : value.bigEndian
        withUnsafeBytes(of: encoded) { bytes.append(contentsOf: $0) }
    }

    private static func lzf(_ payload: Data, expandedBytes: Int) -> Data {
        var data = Data()
        put(UInt32(payload.count), into: &data)
        put(UInt32(expandedBytes), into: &data)
        data.append(payload)
        return data
    }

    private static func literals(_ bytes: Data) -> Data {
        var encoded = Data()
        for offset in stride(from: 0, to: bytes.count, by: 32) {
            let count = min(32, bytes.count - offset)
            encoded.append(UInt8(count - 1))
            encoded.append(contentsOf: bytes[offset..<(offset + count)])
        }
        return encoded
    }
}
