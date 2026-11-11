import Foundation
import Frostfield
import Testing
import simd

struct GLTFQuantizationTests {
    private static let byte = 5120, unsignedByte = 5121, short = 5122, unsignedShort = 5123, float = 5126
    private static var declared: [String: Any] {
        ["extensionsUsed": ["KHR_mesh_quantization"], "extensionsRequired": ["KHR_mesh_quantization"]]
    }

    @Test
    func `Dequantizes positions, directions and texture coordinates with the glTF equations`() throws {
        var fixture = GLTFExtensionFixture()
        // Element strides keep every attribute aligned to four bytes, as glTF requires.
        let position = fixture.accessor(
            GLTFExtensionFixture.bytes([Int16(-3), 100, 32_767, 0, 0, 0, 0, 0, 5, -7, 9, 0]),
            component: Self.short, type: "VEC3", count: 3, stride: 8
        )
        let normal = fixture.accessor(
            GLTFExtensionFixture.bytes([Int8(127), 0, 0, 0, 90, 90, 0, 0, 0, -127, 0, 0]),
            component: Self.byte, type: "VEC3", count: 3, normalized: true, stride: 4
        )
        let tangent = fixture.accessor(
            GLTFExtensionFixture.bytes([Int16(0), 32_767, 0, 32_767, 0, 0, 32_767, -32_767, 23_170, 23_170, 0, 32_767]),
            component: Self.short, type: "VEC4", count: 3, normalized: true
        )
        let primary = fixture.accessor(
            GLTFExtensionFixture.bytes([UInt16(0), 0, 2, 0, 0, 3]), component: Self.unsignedShort, type: "VEC2", count: 3
        )
        let secondary = fixture.accessor(
            GLTFExtensionFixture.bytes([Int8(127), -127, 0, 0, -128, 0, 0, 0, 64, 0, 0, 0]),
            component: Self.byte, type: "VEC2", count: 3, normalized: true, stride: 4
        )
        let attributes = [
            "POSITION": position, "NORMAL": normal, "TANGENT": tangent, "TEXCOORD_0": primary, "TEXCOORD_1": secondary,
        ]
        let data = try fixture.document(Self.declared.merging(["meshes": [["primitives": [["attributes": attributes]]]]]) { $1 })

        let mesh = try GLTFMeshLoader.load(data: data)
        #expect(mesh.vertices.map(\.position) == [SIMD3(-3, 100, 32_767), .zero, SIMD3(5, -7, 9)])
        // Quantized directions are renormalized: (90, 90, 0) / 127 has length 1.0022.
        let diagonal = SIMD3<Float>(1, 1, 0) / Float(2).squareRoot()
        #expect(mesh.vertices[0].normal == SIMD3(1, 0, 0))
        #expect(simd_distance(mesh.vertices[1].normal, diagonal) < 1e-6)
        #expect(mesh.vertices[2].normal == SIMD3(0, -1, 0))
        #expect(mesh.vertices[0].tangent == SIMD4(0, 1, 0, 1))
        #expect(mesh.vertices[1].tangent == SIMD4(0, 0, 1, -1))
        let last = try #require(mesh.vertices[2].tangent)
        #expect(simd_distance(SIMD3(last.x, last.y, last.z), diagonal) < 1e-6 && last.w == 1)
        // Unnormalized integers keep their value; UV 2.0 repeats the texture twice.
        #expect(mesh.vertices.map(\.textureCoordinate) == [.zero, SIMD2(2, 0), SIMD2(0, 3)])
        // BYTE normalization is max(c / 127, -1), so -128 and -127 both decode to -1.
        #expect(mesh.vertices.map(\.secondaryTextureCoordinate) == [SIMD2(1, -1), SIMD2(-1, 0), SIMD2(Float(64) / 127, 0)])
    }

    @Test(arguments: [
        (5121, true, [UInt8(255), 0, 51, 0], SIMD3<Float>(1, 0, 0.2)),
        (5120, false, [UInt8(255), 2, 253, 0], SIMD3<Float>(-1, 2, -3)),
        (5120, true, [UInt8(127), 129, 0, 0], SIMD3<Float>(1, -1, 0)),
        (5121, false, [UInt8(200), 1, 0, 0], SIMD3<Float>(200, 1, 0)),
    ])
    func `Accepts every 8-bit position encoding`(
        component: Int, normalized: Bool, bytes: [UInt8], expected: SIMD3<Float>
    ) throws {
        var fixture = GLTFExtensionFixture()
        let triangle: [UInt8] = bytes + [0, 0, 0, 0] + [0, 1, 0, 0]
        let position = fixture.accessor(
            triangle, component: component, type: "VEC3", count: 3, normalized: normalized, stride: 4
        )
        let data = try fixture.document(Self.declared.merging([
            "meshes": [["primitives": [["attributes": ["POSITION": position]]]]],
        ]) { $1 })
        let mesh = try GLTFMeshLoader.load(data: data)
        let decoded = Set(mesh.vertices.map(\.position))
        #expect(decoded.contains(expected))
        #expect(decoded.contains(.zero))
    }

    @Test
    func `Requires the extension and rejects combinations it does not allow`() throws {
        func load(_ component: Int, normalized: Bool, semantic: String, type: String, declare: Bool) throws -> Data {
            var fixture = GLTFExtensionFixture()
            let position = fixture.accessor(
                GLTFExtensionFixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0]), component: Self.float, type: "VEC3", count: 3
            )
            let width = type == "VEC2" ? 2 : 3
            let attribute = fixture.accessor(
                Array(repeating: 0, count: 3 * 4), component: component, type: type, count: 3,
                normalized: normalized, stride: component == Self.short && width == 3 ? nil : 4
            )
            var sections: [String: Any] = [
                "meshes": [["primitives": [["attributes": semantic == "POSITION"
                    ? ["POSITION": attribute] : ["POSITION": position, semantic: attribute]]]]],
            ]
            if declare { sections.merge(Self.declared) { $1 } }
            return try fixture.document(sections)
        }

        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(1, "POSITION requires VEC3 with float components")) {
            try GLTFMeshLoader.load(data: load(Self.byte, normalized: false, semantic: "POSITION", type: "VEC3", declare: false))
        }
        let direction = "NORMAL requires VEC3 with floats or normalized byte or short components"
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(1, direction)) {
            try GLTFMeshLoader.load(data: load(Self.byte, normalized: false, semantic: "NORMAL", type: "VEC3", declare: true))
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(1, direction)) {
            try GLTFMeshLoader.load(data: load(Self.unsignedByte, normalized: true, semantic: "NORMAL", type: "VEC3", declare: true))
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(
            1, "TEXCOORD_0 requires VEC2 with floats or normalized unsigned byte or short components"
        )) {
            try GLTFMeshLoader.load(data: load(Self.short, normalized: true, semantic: "TEXCOORD_0", type: "VEC2", declare: false))
        }
        #expect(try GLTFMeshLoader.load(
            data: load(Self.short, normalized: true, semantic: "TEXCOORD_0", type: "VEC2", declare: true)
        ).vertices.count == 3)
    }

    @Test
    func `Dequantizes dense and sparse morph target offsets without renormalizing them`() throws {
        var fixture = GLTFExtensionFixture()
        let position = fixture.accessor(
            GLTFExtensionFixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0]), component: Self.float, type: "VEC3", count: 3
        )
        let normal = fixture.accessor(
            GLTFExtensionFixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1]), component: Self.float, type: "VEC3", count: 3
        )
        let positionOffsets = fixture.accessor(
            GLTFExtensionFixture.bytes([Int8(2), 0, 0, 0, 0, -3, 0, 0, 0, 0, 4, 0]),
            component: Self.byte, type: "VEC3", count: 3, stride: 4
        )
        let normalOffsets = fixture.accessor(
            GLTFExtensionFixture.bytes([Int16(16_384), 0, 0, 0, 0, -32_767, 0, 0, 0, 0, 0, 0]),
            component: Self.short, type: "VEC3", count: 3, normalized: true, stride: 8
        )
        let sparseIndices = fixture.view(GLTFExtensionFixture.bytes([UInt16(1)]))
        let sparseValues = fixture.view(GLTFExtensionFixture.bytes([Int16(32_767), 0, -32_767]))
        let sparse = fixture.accessor(
            view: nil, component: Self.short, type: "VEC3", count: 3, normalized: true,
            extra: ["sparse": [
                "count": 1, "indices": ["bufferView": sparseIndices, "componentType": Self.unsignedShort],
                "values": ["bufferView": sparseValues],
            ]]
        )
        let primitive: [String: Any] = [
            "attributes": ["POSITION": position, "NORMAL": normal],
            "targets": [["POSITION": positionOffsets, "NORMAL": normalOffsets], ["POSITION": sparse]],
        ]
        let data = try fixture.document(Self.declared.merging([
            "meshes": [["primitives": [primitive]]], "nodes": [["mesh": 0]], "scenes": [["nodes": [0]]],
        ]) { $1 })

        let scene = try GLTFMeshLoader.loadScene(data: data, includeMorphTargets: true)
        let targets = try #require(scene.meshes.first?.morphTargets?.primitives.first?.targets)
        try #require(targets.count == 2)
        #expect(targets[0].positionOffsets == [SIMD3(2, 0, 0), SIMD3(0, -3, 0), SIMD3(0, 0, 4)])
        #expect(targets[0].normalOffsets == [SIMD3(Float(16_384) / 32_767, 0, 0), SIMD3(0, -1, 0), .zero])
        #expect(targets[1].positionOffsets == [.zero, SIMD3(1, 0, -1), .zero])
    }
}
