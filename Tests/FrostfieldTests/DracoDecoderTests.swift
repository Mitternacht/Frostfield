import Foundation
import Frostfield
import Testing
import simd

struct DracoDecoderTests {
    // Test vectors from google/draco 1.5.9 testdata, copyright The Draco Authors,
    // under Apache 2.0. Both 2.2 streams encode cube_att.obj.
    private static let edgeBreaker = "RFJBQ08CAgEBAAAAGAwADAAABd/3ve8L/wJmQAH/AAADAAkDAAADCQIAAQEJAwACAgIDAQEBAQEDxSmRC60KDqxJGy4gR6kRfLJoOCKFAAAAAP//DwABAQEBAQMBJFUNrQ4L3PQFYjtaNiDJBocAAAAA/w8AAAADAQANAwEwIwEIAQgGAArxc22AAPg/8H8ACPwfAMD/AQAAAAACAACAAP8PAAD/BwAAAAAAAAAAAAAAAAAAAACAPxQAAAAAAAAAAAAAgD8MDA=="
    private static let sequential = "RFJBQ08CAgEAAAAMGAEAAQIAAwEEBQYEBwUICQoICwkMDQ4MDg8QERIQEhMUFRYUFhcBAwAJAwAAAwkCAAEBCQMAAgICAwABAQEBA5EjHQ9VDQ+jA6dCc5hoeARwu8H/QqoAAAAA//8PAAABAQEBA1UVVRVZFQxU7vMvAMaPQGswR4UAAAAA/w8AAAADAQANAwEwIwEIAQgGAAjxc2+AAACAgP8DIPB/AAD/BwAAgAAA/P//AP8PAAD/BwAAAAAAAAAAAAAAAAAAAACAPxQAAAAAAAAAAAAAgD8MDA=="
    private static let legacy = "RFJBQ08BAQEBAAAAAAAAAAgAAAAMAAAAAgsAAAAAAAAAKgAAAAMAAAAAAAAAX0sVAQAAAAAAAAABVQYAAADz1FMAblBVBgAAAPPUUwBuUAAAAAAAAAAAA/8AAAEBAQEAAAAACQMAAAACAQAAAAMJAgAAAAIBAAAAAQkDAAAAAwEBAAAAAAAAAAAAAAAAAACAPw4BAQIDAAAAVSUBCK0SBgAAAAAAAADM7rJWBkUAAAAA/z8AAAEBAAAAAAAAAAAAAIA/DAEBAgMAAABZHVURVREMAAAAAAAAAFgGNV0gRd8hHT6kgAAAAAD/DwAAAAIKAQALAAAAAAEwAAAAAAAAAFUFrQoGAAAAAAAAACUQOkpNgP8BAAD+BwAIIOD/fwAAAAIA/gP/AwAA/wEAAA=="
    private static let pointCloudHeader = "RFJBQ08CAgAAAAA="

    @Test(arguments: [edgeBreaker, sequential, legacy])
    func `Standalone mesh streams preserve face winding and typed vertex attributes`(fixture: String) throws {
        let data = try #require(Data(base64Encoded: fixture))
        let mesh = try DracoDecoder.decode(data)
        #expect(mesh.pointCount == 24)
        #expect(mesh.indices.count == 36)
        #expect(mesh.attributes.map(\.semantic) == [.position, .textureCoordinate, .normal])
        #expect(mesh.attributes.map(\.components) == [3, 2, 3])
        #expect(mesh.attributes.allSatisfy { $0.component == .float32 })
        let position = try #require(mesh.attributes.first { $0.semantic == .position })
        let normal = try #require(mesh.attributes.first { $0.semantic == .normal })
        let points = floats(position.data)
        let normals = floats(normal.data)
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let face = mesh.indices[offset..<(offset + 3)].map(Int.init)
            func vector(_ values: [Float], _ index: Int) -> SIMD3<Float> {
                let index = 3 * index
                return SIMD3(values[index], values[index + 1], values[index + 2])
            }
            let first = vector(points, face[0])
            let second = vector(points, face[1])
            let third = vector(points, face[2])
            let faceNormal = simd_normalize(simd_cross(second - first, third - first))
            #expect(abs(simd_length(faceNormal) - 1) < 0.001)
            #expect(face.allSatisfy { simd_dot(faceNormal, vector(normals, $0)) > 0.99 })
        }
        #expect(Set(points) == [0, 1])
    }

    @Test
    func `Rejects a truncated mesh and identifies unsupported point clouds`() throws {
        let triangle = try #require(Data(base64Encoded: Self.sequential))
        #expect(throws: DracoDecoder.Failure.invalidBitstream) {
            try DracoDecoder.decode(triangle.prefix(14))
        }
        let points = try #require(Data(base64Encoded: Self.pointCloudHeader))
        #expect(throws: DracoDecoder.Failure.unsupportedGeometry) {
            try DracoDecoder.decode(points)
        }
    }

    @Test(arguments: [false, true])
    func `glTF uses compressed topology and merges ordinary vertex attributes`(required: Bool) throws {
        let data = try document(required: required, positionCount: 24, textureID: 1)
        let geometry = try GLTFMeshLoader.load(data: data)
        let scene = try GLTFMeshLoader.loadScene(data: data)
        let sceneGeometry = try #require(scene.meshes.first?.primitives.first)
        #expect(sceneGeometry.indices == geometry.indices)
        #expect(sceneGeometry.vertices.map(\.position) == geometry.vertices.map(\.position))
        #expect(geometry.vertices.count == 24 && geometry.indices.count == 36)
        #expect(Set(geometry.vertices.map(\.position)).count == 8)
        #expect(Set(geometry.vertices.map(\.textureCoordinate)).count == 4)
        #expect(geometry.vertices.allSatisfy { abs(simd_length($0.normal) - 1) < 0.001 })
        #expect(geometry.vertices.allSatisfy { $0.color == SIMD4<Float>(1, 0, 0, 1) })
    }

    @Test
    func `glTF preserves compressed UInt16 joints and normalized byte weights`() throws {
        // Sequential triangle encoded with upstream draco3d 1.5.9, with generic
        // UInt16x4 JOINTS_0 and normalized UInt8x4 WEIGHTS_0 streams.
        let bytes = try #require(Data(base64Encoded:
            "RFJBQ08CAgEAAAABAwEAAQIBAwAJAwAABAQEAAEEAgQBAgABAQAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAABAQACAwFAAQAQAwAAAAABAAAAAAEBAAkDrSoXVRUDoHqBAf3+AAAAAAAA/wAAAA=="
        ))
        var fixture = GLTFExtensionFixture()
        let compressed = fixture.view(Array(bytes))
        let position = fixture.accessor(view: nil, component: 5126, type: "VEC3", count: 3)
        let joints = fixture.accessor(view: nil, component: 5123, type: "VEC4", count: 3)
        let weights = fixture.accessor(view: nil, component: 5121, type: "VEC4", count: 3, normalized: true)
        let indices = fixture.accessor(view: nil, component: 5123, type: "SCALAR", count: 3)
        let ext = "KHR_draco_mesh_compression"
        let primitive: [String: Any] = [
            "attributes": ["POSITION": position, "JOINTS_0": joints, "WEIGHTS_0": weights],
            "indices": indices,
            "extensions": [ext: [
                "bufferView": compressed, "attributes": ["POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2],
            ]],
        ]
        let document = try fixture.document([
            "extensionsUsed": [ext], "extensionsRequired": [ext], "meshes": [["primitives": [primitive]]],
            "nodes": [["mesh": 0, "skin": 0], ["children": [2]], [:]], "scenes": [["nodes": [0, 1]]],
            "skins": [["joints": [1, 2]]],
        ])
        let scene = try GLTFMeshLoader.loadScene(data: document, includeSkins: true)
        let mesh = try #require(scene.meshes.first)
        let influences = try #require(mesh.primitiveSkinInfluences.first ?? nil)
        #expect(influences.map(\.joints) == [SIMD4(0, 0, 0, 0), SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0)])
        #expect(influences.map(\.weights) == [SIMD4(1, 0, 0, 0), SIMD4(1, 0, 0, 0), SIMD4(128.0 / 255, 127.0 / 255, 0, 0)])
    }

    @Test
    func `glTF rejects missing compressed IDs and mismatched accessor counts`() throws {
        #expect(throws: GLTFMeshLoader.Failure.invalidDracoCompression(1, "attribute TEXCOORD_0 has no Draco data or accessor")) {
            try GLTFMeshLoader.load(data: document(required: true, positionCount: 24, textureID: 8))
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidDracoCompression(1, "attribute POSITION does not match its accessor")) {
            try GLTFMeshLoader.load(data: document(required: true, positionCount: 3, textureID: 1))
        }
    }

    private func document(required: Bool, positionCount: Int, textureID: Int) throws -> Data {
        var fixture = GLTFExtensionFixture()
        // An intentionally different uncompressed POSITION buffer establishes that
        // a supported optional extension takes precedence over its fallback.
        let fallback = fixture.view(GLTFExtensionFixture.floats(Array(repeating: 2, count: positionCount * 3)))
        let bytes = try #require(Data(base64Encoded: Self.edgeBreaker))
        let compressed = fixture.view(Array(bytes))
        let position = fixture.accessor(
            view: required ? nil : fallback, component: 5126, type: "VEC3", count: positionCount
        )
        let normal = fixture.accessor(view: nil, component: 5126, type: "VEC3", count: 24)
        let uv = fixture.accessor(view: nil, component: 5126, type: "VEC2", count: 24)
        let colors = Array(repeating: [255, 0, 0, 255] as [UInt8], count: 24).flatMap { $0 }
        let color = fixture.accessor(colors, component: 5121, type: "VEC4", count: 24, normalized: true)
        let indices = fixture.accessor(view: nil, component: 5123, type: "SCALAR", count: 36)
        let ext = "KHR_draco_mesh_compression"
        let primitive: [String: Any] = [
            "attributes": ["POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv, "COLOR_0": color],
            "indices": indices,
            "extensions": [ext: [
                "bufferView": compressed, "attributes": ["POSITION": 0, "TEXCOORD_0": textureID, "NORMAL": 2],
            ]],
        ]
        var sections: [String: Any] = [
            "extensionsUsed": [ext], "meshes": [["primitives": [primitive]]],
            "nodes": [["mesh": 0]], "scenes": [["nodes": [0]]],
        ]
        if required { sections["extensionsRequired"] = [ext] }
        return try fixture.document(sections)
    }

    private func floats(_ data: Data) -> [Float] {
        data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map { offset in
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
            }
        }
    }
}
