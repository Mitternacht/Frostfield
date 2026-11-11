import Foundation
import Frostfield
import Testing
import simd

// Fixtures marked "reference encoder" were produced with the meshoptimizer 1.3.0 npm
// package (MeshoptEncoder.encodeVertexBufferLevel, encodeGltfBuffer, and the
// encodeFilter* functions). Expected filter outputs come from its MeshoptDecoder.
// The specification allows filtered results to differ by one unit in the last place.
struct MeshoptDecoderTests {
    @Test
    func `Decodes the specification's 4-bit sentinel example from a hand-built stream`() throws {
        // One block of 16 four-byte elements. Byte 0 uses a 4-bit group; bytes 1 to 3 are zero groups.
        let group: [UInt8] = [0x17, 0x5F, 0xF0, 0xBC, 0x77, 0xA9, 0x21, 0x00, 0x34, 0xB5]
        let tail = [UInt8](repeating: 0, count: 28) + [0, 7, 0, 0]
        let stream = Data([0xA0, 0b10] + group + [0, 0, 0] + tail)
        let decoded = try MeshoptDecoder.decode(stream, count: 16, byteStride: 4, mode: .attributes, specification: .ext)
        let deltas = [-1, -4, -3, 26, -91, 0, -6, 6, -4, -4, 5, -5, 1, -1, 0, 0]
        var running = 0
        let expected = deltas.map { delta -> UInt8 in
            running += delta
            return UInt8(truncatingIfNeeded: running)
        }
        #expect((0..<16).map { decoded[$0 * 4] } == expected)
        // The baseline in the tail carries through the zero groups.
        #expect((0..<16).allSatisfy { decoded[$0 * 4 + 1] == 7 && decoded[$0 * 4 + 2] == 0 })

        #expect(throws: MeshoptDecoder.Failure.unusedBytes(1)) {
            try MeshoptDecoder.decode(
                Data([0xA0, 0b10] + group + [0, 0, 0, 0] + tail), count: 16, byteStride: 4, mode: .attributes
            )
        }
        #expect(throws: MeshoptDecoder.Failure.truncatedStream) {
            try MeshoptDecoder.decode(
                Data([0xA0, 0b10] + group.dropLast() + tail), count: 16, byteStride: 4, mode: .attributes
            )
        }
    }

    @Test
    func `Decodes a two-block version 0 attribute stream`() throws {
        // Reference encoder, level 2, version 0: 300 elements of 4 bytes split into blocks of 256 and 44.
        let stream = try #require(Data(base64Encoded: Self.versionZero.joined()))
        let decoded = try MeshoptDecoder.decode(stream, count: 300, byteStride: 4, mode: .attributes, specification: .ext)
        var expected: [UInt8] = []
        for i in 0..<300 {
            let third: Int = i < 100 ? 0 : (i * 3) & 255
            expected += [UInt8(i & 255), UInt8((i * 37) & 255), UInt8(third), UInt8(((i * i) >> 3) & 255)]
        }
        #expect([UInt8](decoded) == expected)
    }

    @Test
    func `Decodes version 1 control and channel modes and keeps EXT streams at version 0`() throws {
        // Reference encoder, level 3, version 1. The stream uses literal, zero, and both
        // bit-width control modes, 16-bit deltas for channel 0, and rotated XOR for channel 1.
        let stream = try #require(Data(base64Encoded: Self.versionOne))
        let decoded = try MeshoptDecoder.decode(stream, count: 40, byteStride: 8, mode: .attributes)
        for i in 0..<40 {
            let base = i * 8
            let first = UInt16(decoded[base]) | UInt16(decoded[base + 1]) << 8
            let second = UInt16(decoded[base + 2]) | UInt16(decoded[base + 3]) << 8
            let bits = (0..<4).reduce(UInt32(0)) { $0 | UInt32(decoded[base + 4 + $1]) << (8 * UInt32($1)) }
            #expect(first == UInt16(truncatingIfNeeded: 1_000 + i * 300))
            #expect(second == UInt16(40_000 - i * 7))
            #expect(Float(bitPattern: bits) == Float(i) * 0.25 - 3)
        }
        #expect(throws: MeshoptDecoder.Failure.invalidHeader(found: 0xA1, expected: "0xa0")) {
            try MeshoptDecoder.decode(stream, count: 40, byteStride: 8, mode: .attributes, specification: .ext)
        }
        var reserved = stream
        reserved[reserved.count - 2] = 3
        #expect(throws: MeshoptDecoder.Failure.malformedStream("channel 0 has reserved mode 0x3")) {
            try MeshoptDecoder.decode(reserved, count: 40, byteStride: 8, mode: .attributes)
        }
    }

    @Test
    func `Decodes triangle codes, preserving each triangle's winding`() throws {
        // Hand-derived: code 0xf0 reads table entry 0 (0x00) for a new triangle (0, 1, 2);
        // code 0x10 reuses edge 1, (2, 1), with the next vertex 3. The table is the encoder's.
        let table: [UInt8] = [0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xA9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0, 0]
        let handBuilt = try MeshoptDecoder.decode(
            Data([0xE1, 0xF0, 0x10] + table), count: 6, byteStride: 2, mode: .triangles
        )
        #expect(Self.integers(handBuilt, width: 2) == [0, 1, 2, 2, 1, 3])

        // Reference encoder: a 3 × 2 quad grid followed by a triangle of free indices.
        let input: [UInt32] = [
            0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 6, 2, 6, 3, 3, 6, 7, 4, 8, 5, 5, 8, 9, 5, 9, 6, 6, 9, 10, 6, 10, 7,
            7, 10, 11, 20, 300, 41,
        ]
        let stream = try #require(Data(base64Encoded: "4f4eAB4AHp4OBB4DHv/wCP8SsASFBAB2h1ZneKmGZYlomAFpAAA="))
        let decoded = Self.integers(try MeshoptDecoder.decode(stream, count: 39, byteStride: 2, mode: .triangles), width: 2)
        // The codec may rotate a triangle's corners but keeps their cyclic order.
        #expect(decoded == [
            0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 6, 2, 6, 3, 3, 6, 7, 5, 4, 8, 5, 8, 9, 5, 9, 6, 6, 9, 10, 6, 10, 7,
            7, 10, 11, 20, 300, 41,
        ])
        for triangle in 0..<13 {
            let source = Array(input[(triangle * 3)..<(triangle * 3 + 3)])
            let result = Array(decoded[(triangle * 3)..<(triangle * 3 + 3)])
            #expect((0..<3).contains { shift in (0..<3).allSatisfy { source[($0 + shift) % 3] == result[$0] } })
        }
    }

    @Test
    func `Decodes an index sequence with two baselines`() throws {
        // Reference encoder, 4-byte indices.
        let stream = try #require(Data(base64Encoded: "0QAEBAAEBKEfBQQEBaiLEQAAAAA="))
        let expected: [UInt32] = [0, 1, 2, 2, 3, 4, 1_000, 1_001, 5, 6, 1_002, 70_000]
        #expect(Self.integers(try MeshoptDecoder.decode(stream, count: 12, byteStride: 4, mode: .indices), width: 4) == expected)
        #expect(throws: MeshoptDecoder.Failure.malformedStream("index 11 decodes to 70000, which does not fit 16 bits")) {
            try MeshoptDecoder.decode(stream, count: 12, byteStride: 2, mode: .indices)
        }
        // Six continuation bytes exceed the five-byte limit.
        #expect(throws: MeshoptDecoder.Failure.malformedStream("a variable-length integer exceeds five bytes")) {
            try MeshoptDecoder.decode(
                Data([0xD1] + Array(repeating: 0x80, count: 6) + [0, 0, 0, 0, 0]), count: 1, byteStride: 4, mode: .indices
            )
        }
    }

    @Test
    func `Applies octahedral, quaternion, exponential and color filters`() throws {
        func decoded(_ base64: String, count: Int, stride: Int, filter: MeshoptDecoder.Filter) throws -> Data {
            try MeshoptDecoder.decode(
                try #require(Data(base64Encoded: base64)), count: count, byteStride: stride, mode: .attributes, filter: filter
            )
        }
        func expectClose(_ actual: [Int], _ expected: [Int]) {
            #expect(actual.count == expected.count)
            #expect(zip(actual, expected).allSatisfy { abs($0 - $1) <= 1 }, "\(actual) vs \(expected)")
        }
        // Inputs: (0, 0, 1), (1, 0, 0), (0.6, -0.8, 0), (-0.48, 0.6, -0.64), with fourth components ±1.
        let octahedral8 = try decoded(
            "oAE/AAAA/pHuAQ8AAACRtQABPAAAAAQDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAf38=", count: 4, stride: 4, filter: .octahedral
        )
        expectClose(octahedral8.map { Int(Int8(bitPattern: $0)) }, [0, 0, 127, 127, 127, 0, 0, -127, 76, -102, 0, 127, -60, 76, -82, 127])
        let octahedral16 = try decoded(
            "oAEfAAAA3LwBPwAAAA4HEQEPAAAA3KwBDwAAAAkUAAABPAAAAAQDASQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wf/fw==",
            count: 4, stride: 8, filter: .octahedral
        )
        let normals = Self.signed16(octahedral16)
        expectClose(normals, [0, 0, 32_767, 32_767, 32_767, 0, 0, -32_767, 19_653, -26_219, 0, 32_767, -15_721, 19_658, -20_979, 32_767])
        // 12-bit octahedral storage keeps each direction within a fraction of a degree.
        let last = SIMD3(Float(normals[12]), Float(normals[13]), Float(normals[14])) / 32_767
        #expect(simd_distance(last, SIMD3(-0.48, 0.6, -0.64)) < 0.002)

        // Inputs: identity, 90° about X, and (0.5, -0.5, 0.5, 0.5), encoded with 12 bits.
        let quaternions = try decoded(
            "oAEMAAAAsgEMAAAACwEMAAAAsQEMAAAACgEcAAAArwE8AAAADgMBMAAAAAUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wc=",
            count: 3, stride: 8, filter: .quaternion
        )
        expectClose(Self.signed16(quaternions), [0, 0, 0, 32_767, 23_170, 0, 0, 23_170, 16_399, -16_378, 16_378, 16_378])

        // 24-bit mantissas represent these values exactly, so decoding is exact.
        let exponential = try decoded(
            "oAAAATAAAAA/ATAAAAANAAABMAAAABABMAAAAAMAATAAAADeATAAAAA5ATAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGDqAAC46wAEffM=",
            count: 2, stride: 12, filter: .exponential
        )
        let floats = Self.integers(exponential, width: 4).map { Float(bitPattern: $0) }
        #expect(floats == [1.5, -2.25, 1_000.125, 0.0078125, -0.5, 12_345.5])

        // Inputs: red, (0.2, 0.4, 0.6, 0.5), transparent black, and white, as 8-bit and 12-bit YCoCg.
        let color8 = try MeshoptDecoder.decode(
            try #require(Data(base64Encoded: "of8ATMsBAJxmAAB+AAAAfX/+AAAAAAAAAAAAAAAAAAAAAAAAAEB/wf8A")),
            count: 4, byteStride: 4, mode: .attributes, filter: .color
        )
        expectClose(color8.map(Int.init), [254, 1, 0, 255, 51, 102, 153, 128, 0, 0, 0, 0, 255, 255, 255, 255])
        let color16 = try MeshoptDecoder.decode(
            try #require(Data(base64Encoded: "of//AMzLAQAECx4AY2YAABUIAAABAAAACAAAAAIAAQAFBw4AAAAAAAAAAAAAAAAAAAAE/wcB/P8PAAA=")),
            count: 4, byteStride: 8, mode: .attributes, filter: .color
        )
        expectClose(
            Self.integers(color16, width: 2).map(Int.init),
            [65_519, 16, 0, 65_535, 13_107, 26_214, 39_321, 32_776, 0, 0, 0, 0, 65_535, 65_535, 65_535, 65_535]
        )
    }

    @Test
    func `Rejects invalid layouts, headers, references and filter inputs`() throws {
        let tail = Data(repeating: 0, count: 32)
        #expect(throws: MeshoptDecoder.Failure.invalidLayout("ATTRIBUTES needs a byte stride that is a multiple of 4 from 4 to 256")) {
            try MeshoptDecoder.decode(tail, count: 1, byteStride: 6, mode: .attributes)
        }
        #expect(throws: MeshoptDecoder.Failure.invalidLayout("COLOR is available only in KHR_meshopt_compression")) {
            try MeshoptDecoder.decode(tail, count: 1, byteStride: 4, mode: .attributes, filter: .color, specification: .ext)
        }
        #expect(throws: MeshoptDecoder.Failure.invalidLayout("TRIANGLES needs a count that is a multiple of 3")) {
            try MeshoptDecoder.decode(tail, count: 4, byteStride: 2, mode: .triangles)
        }
        #expect(throws: MeshoptDecoder.Failure.invalidLayout("INDICES cannot use a filter")) {
            try MeshoptDecoder.decode(tail, count: 1, byteStride: 4, mode: .indices, filter: .exponential)
        }
        #expect(throws: MeshoptDecoder.Failure.invalidHeader(found: 0, expected: "0xa0 or 0xa1")) {
            try MeshoptDecoder.decode(Data([0]) + tail, count: 1, byteStride: 4, mode: .attributes)
        }
        #expect(throws: MeshoptDecoder.Failure.invalidHeader(found: 0xE0, expected: "0xe1")) {
            try MeshoptDecoder.decode(Data([0xE0, 0xF0]) + Data(repeating: 0, count: 16), count: 3, byteStride: 4, mode: .triangles)
        }

        var table = [UInt8](repeating: 0, count: 16)
        #expect(throws: MeshoptDecoder.Failure.malformedStream("a triangle references edge FIFO entry 0 before it is written")) {
            try MeshoptDecoder.decode(Data([0xE1, 0x00] + table), count: 3, byteStride: 4, mode: .triangles)
        }
        #expect(throws: MeshoptDecoder.Failure.malformedStream("a triangle references vertex FIFO entry 0 before it is written")) {
            try MeshoptDecoder.decode(Data([0xE1, 0xFE, 0x11] + table), count: 3, byteStride: 4, mode: .triangles)
        }
        table[3] = 0xF1
        #expect(throws: MeshoptDecoder.Failure.malformedStream("code table entry 3 contains the reserved value 0xf")) {
            try MeshoptDecoder.decode(Data([0xE1, 0xF0] + table), count: 3, byteStride: 4, mode: .triangles)
        }
        #expect(throws: MeshoptDecoder.Failure.truncatedStream) {
            try MeshoptDecoder.decode(Data([0xE1, 0xF0, 0xF0]), count: 6, byteStride: 4, mode: .triangles)
        }

        // Zero groups repeat the tail's baseline element. A third component of 5 is not a
        // valid octahedral 1.0, and exponent 101 is outside the filter's range.
        let octahedral = Data([0xA0, 0, 0, 0, 0] + Array(repeating: 0, count: 28) + [0, 0, 5, 0])
        #expect(throws: MeshoptDecoder.Failure.invalidFilterInput(element: 0, "the third component must encode 1.0 for 2 to 8 bits")) {
            try MeshoptDecoder.decode(octahedral, count: 1, byteStride: 4, mode: .attributes, filter: .octahedral)
        }
        let exponent = Data([0xA0, 0, 0, 0, 0] + Array(repeating: 0, count: 28) + [1, 0, 0, 101])
        #expect(throws: MeshoptDecoder.Failure.invalidFilterInput(element: 0, "exponent 101 is outside -100 to 100")) {
            try MeshoptDecoder.decode(exponent, count: 1, byteStride: 4, mode: .attributes, filter: .exponential)
        }
    }

    @Test
    func `Loads a compressed quantized glTF mesh without reading its fallback buffer`() throws {
        // Reference encoder streams: SHORT positions (stride 8), octahedral BYTE normals,
        // SHORT normalized UVs, and a TRIANGLES index stream (0, 1, 2, 2, 1, 3).
        let streams = [
            "oAE/AAAAyMfIAAEMAAAAyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB/AA==",
            "oAEZAAAAAT8AAAD+/f4BBAAAAAEMAAAA/gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "4fAQAHaHVmd4qYZliWiYAWkAAA==",
        ].map { Data(base64Encoded: $0)! }
        var compressed = Data()
        var ranges: [(offset: Int, length: Int)] = []
        for stream in streams {
            while !compressed.count.isMultiple(of: 4) { compressed.append(0) }
            ranges.append((compressed.count, stream.count))
            compressed.append(stream)
        }
        func view(
            _ index: Int, offset: Int, length: Int, stride: Int, count: Int, mode: String, filter: String? = nil
        ) -> [String: Any] {
            var record: [String: Any] = [
                "buffer": 0, "byteOffset": ranges[index].offset, "byteLength": ranges[index].length,
                "byteStride": stride, "count": count, "mode": mode,
            ]
            if let filter { record["filter"] = filter }
            var parent: [String: Any] = [
                "buffer": 1, "byteOffset": offset, "byteLength": length, "extensions": ["EXT_meshopt_compression": record],
            ]
            // Vertex attributes declare the decoded stride on the parent view, as glTF requires.
            if mode == "ATTRIBUTES" { parent["byteStride"] = stride }
            return parent
        }
        let fallback: [String: Any] = ["byteLength": 112, "extensions": ["EXT_meshopt_compression": ["fallback": true]]]
        var document: [String: Any] = [
            "asset": ["version": "2.0"],
            "extensionsUsed": ["EXT_meshopt_compression", "KHR_mesh_quantization"],
            "extensionsRequired": ["EXT_meshopt_compression", "KHR_mesh_quantization"],
            "buffers": [
                ["byteLength": compressed.count, "uri": "data:application/octet-stream;base64,\(compressed.base64EncodedString())"],
                fallback,
            ],
            "bufferViews": [
                view(0, offset: 0, length: 32, stride: 8, count: 4, mode: "ATTRIBUTES"),
                view(1, offset: 32, length: 16, stride: 4, count: 4, mode: "ATTRIBUTES", filter: "OCTAHEDRAL"),
                view(2, offset: 48, length: 16, stride: 4, count: 4, mode: "ATTRIBUTES"),
                view(3, offset: 64, length: 12, stride: 2, count: 6, mode: "TRIANGLES"),
            ],
            "accessors": [
                ["bufferView": 0, "componentType": 5122, "type": "VEC3", "count": 4],
                ["bufferView": 1, "componentType": 5120, "normalized": true, "type": "VEC3", "count": 4],
                ["bufferView": 2, "componentType": 5122, "normalized": true, "type": "VEC2", "count": 4],
                ["bufferView": 3, "componentType": 5123, "type": "SCALAR", "count": 6],
            ],
            "meshes": [["primitives": [["attributes": ["POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2], "indices": 3]]]],
        ]
        let mesh = try GLTFMeshLoader.load(data: JSONSerialization.data(withJSONObject: document))
        #expect(mesh.vertices.map(\.position) == [.zero, SIMD3(100, 0, 0), SIMD3(0, 100, 0), SIMD3(100, 100, 0)])
        #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })
        #expect(mesh.vertices.map(\.textureCoordinate) == [.zero, SIMD2(1, 0), SIMD2(0, 1), SIMD2(1, 1)])
        #expect(mesh.indices == [0, 1, 2, 2, 1, 3])

        // An uncompressed view cannot read the fallback buffer, and layouts must agree.
        var views = try #require(document["bufferViews"] as? [[String: Any]])
        views.append(["buffer": 1, "byteLength": 12])
        document["bufferViews"] = views
        var accessors = try #require(document["accessors"] as? [[String: Any]])
        accessors[3] = ["bufferView": 4, "componentType": 5123, "type": "SCALAR", "count": 6]
        document["accessors"] = accessors
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(3, "buffer view 4 reads meshopt fallback buffer 1 without compression")) {
            try GLTFMeshLoader.load(data: JSONSerialization.data(withJSONObject: document))
        }
        views[3] = view(3, offset: 64, length: 12, stride: 2, count: 5, mode: "TRIANGLES")
        accessors[3]["bufferView"] = 3
        document["bufferViews"] = views
        document["accessors"] = accessors
        #expect(throws: GLTFMeshLoader.Failure.invalidCompressedBufferView(3, "byteStride × count must equal the buffer view byteLength")) {
            try GLTFMeshLoader.load(data: JSONSerialization.data(withJSONObject: document))
        }
        views[3] = view(3, offset: 64, length: 12, stride: 2, count: 6, mode: "TRIANGLES")
        var corrupt = compressed
        corrupt[ranges[3].offset] = 0xE0
        var buffers = try #require(document["buffers"] as? [[String: Any]])
        buffers[0]["uri"] = "data:application/octet-stream;base64,\(corrupt.base64EncodedString())"
        document["buffers"] = buffers
        document["bufferViews"] = views
        let header = MeshoptDecoder.Failure.invalidHeader(found: 0xE0, expected: "0xe1").localizedDescription
        #expect(throws: GLTFMeshLoader.Failure.invalidCompressedBufferView(3, header)) {
            try GLTFMeshLoader.load(data: JSONSerialization.data(withJSONObject: document))
        }
    }

    private static func integers(_ data: Data, width: Int) -> [UInt32] {
        stride(from: 0, to: data.count, by: width).map { offset in
            (0..<width).reduce(UInt32(0)) { $0 | UInt32(data[data.startIndex + offset + $1]) << (8 * UInt32($1)) }
        }
    }

    private static func signed16(_ data: Data) -> [Int] {
        integers(data, width: 2).map { Int(Int16(bitPattern: UInt16($0))) }
    }

    private static let versionOne =
        "oZdoAFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWBUqqqqqqqqqqqqqAAAqDd3d3d3d3d3d3d3d3d3d3d3d3d0AAAAABSAwDxcHEAAPGg4mL0xPj//4/Bgw0NAw9MTyYuJi8TEY/B7d3QAABwMPAwAAAAAAAAAAAAAAAAAA6ANAnAAAQMABUg=="

    private static let versionZero = [
        "oFVVVVUqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq/////wBK",
        "SkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpK",
        "SkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpK",
        "SkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpK",
        "SkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkoAoKqqAAD2ZmZmZmZYZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZm",
        "ZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZm+v///wACIiRERmZoiIqqrMzO7u8QEBAQEhISEhQUFBQWFhYW",
        "GBgYGBoaGhocHBwcHh4eHiAgICAiIiIiJCQkJCYmJiYoKCgoKioqKiwsLCwuLi4uMDAwMDIyMjI0NDQ0NjY2Njg4ODg6Ojo6PDw8",
        "PD4+Pj5AQEBAQkJCQkRERERGRkZGSEhISEpKSkpMTExMTk5OTlBQUFBSUlJSVFRUVFZWVlZYWFhYWlpaWlxcXFxeXl5eYGBgYGJi",
        "YmJkZGRkZmZmZmhoaGhqampqbGxsbG5ubm5wcHBwcnJycnR0dHR2dnZ2eHh4eHp6enp8fHx8fn5+foAVqqqqqqqqqqqqqqoAP0pK",
        "SkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKAAAAACpmZmZmZmZmZmZmZmZmZmZmZmZmZmZmAAA/gICA",
        "goKCgoSEhISGhoaGiIiIiIqKioqMjIyMjo6OjpCQkJCSkpKSlJSUlJYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "AAA=",
    ]
}
