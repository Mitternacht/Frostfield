import Foundation
import Frostfield
import Testing
import simd

struct GLTFMeshLoaderTests {
    @Test(arguments: [5121, 5123, 5125])
    func `Reads GLB stride offsets and each unsigned index width`(_ componentType: Int) throws {
        var binary = Data(repeating: 0, count: 4)
        let positions = [SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0)]
        let coordinates = [SIMD2<Float>(0.125, 0.25), SIMD2<Float>(0.75, 0.375), SIMD2<Float>(0.375, 0.875)]
        for (position, uv) in zip(positions, coordinates) {
            append([Float(99), position.x, position.y, position.z, 0, 0, 1, uv.x, uv.y], to: &binary)
        }
        let indexStart = binary.count
        binary.append(Data(repeating: 0, count: 4))
        for index in [2, 0, 1] {
            switch componentType {
            case 5121: append(UInt8(index), to: &binary)
            case 5123: append(UInt16(index), to: &binary)
            default: append(UInt32(index), to: &binary)
            }
        }
        let attributes = ["POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2]
        let document: [String: Any] = [
            "asset": ["version": "2.0"],
            "buffers": [["byteLength": binary.count]],
            "bufferViews": [
                ["buffer": 0, "byteOffset": 4, "byteLength": 108, "byteStride": 36],
                ["buffer": 0, "byteOffset": indexStart, "byteLength": binary.count - indexStart],
            ],
            "accessors": [
                ["bufferView": 0, "byteOffset": 4, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0, 0, 0], "max": [1, 1, 0]],
                ["bufferView": 0, "byteOffset": 16, "componentType": 5126, "count": 3, "type": "VEC3"],
                ["bufferView": 0, "byteOffset": 28, "componentType": 5126, "count": 3, "type": "VEC2"],
                ["bufferView": 1, "byteOffset": 4, "componentType": componentType, "count": 3, "type": "SCALAR"],
            ],
            "meshes": [["primitives": [["attributes": attributes, "indices": 3]]]],
        ]

        let mesh = try GLTFMeshLoader.load(data: glb(document, binary: binary))
        #expect(mesh.indices == [2, 0, 1])
        #expect(mesh.vertices.map(\.position) == positions)
        #expect(mesh.vertices.map(\.normal) == Array(repeating: SIMD3<Float>(0, 0, 1), count: 3))
        #expect(mesh.vertices.map(\.textureCoordinate) == coordinates)
    }

    @Test
    func `Loads a selected unindexed primitive from a data URI`() throws {
        var document = positionDocument()
        let triangle: [String: Any] = ["attributes": ["POSITION": 0]]
        let skipped: [String: Any] = ["attributes": ["POSITION": 0], "mode": 1]
        document["meshes"] = [["primitives": [skipped]], ["primitives": [skipped, triangle]]]

        let mesh = try GLTFMeshLoader.load(data: json(document), meshIndex: 1, primitiveIndex: 1)
        #expect(mesh.indices == [0, 1, 2])
        #expect(mesh.vertices.map(\.normal) == Array(repeating: SIMD3<Float>(0, 0, 1), count: 3))
        #expect(mesh.vertices.map(\.textureCoordinate) == [.zero, .zero, .zero])
        #expect(throws: GLTFMeshLoader.Failure.missingMesh(4)) {
            try GLTFMeshLoader.load(data: json(document), meshIndex: 4)
        }
    }

    @Test
    func `Generates separate flat normals across shared corners`() throws {
        var binary = triangleBytes()
        append([Float(0), 0, 1], to: &binary)
        binary.append(contentsOf: [0, 1, 2, 0, 3, 1])
        let document: [String: Any] = [
            "asset": ["version": "2.0"],
            "buffers": [["byteLength": binary.count, "uri": "data:application/gltf-buffer;base64,\(binary.base64EncodedString())"]],
            "bufferViews": [
                ["buffer": 0, "byteLength": 48],
                ["buffer": 0, "byteOffset": 48, "byteLength": 6],
            ],
            "accessors": [
                ["bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                 "min": [0, 0, 0], "max": [1, 1, 1]],
                ["bufferView": 1, "componentType": 5121, "count": 6, "type": "SCALAR"],
            ],
            "meshes": [["primitives": [["attributes": ["POSITION": 0], "indices": 1]]]],
        ]

        let mesh = try GLTFMeshLoader.load(data: json(document))
        #expect(mesh.indices == [0, 1, 2, 3, 4, 5])
        #expect(mesh.vertices[0].position == mesh.vertices[3].position)
        #expect(mesh.vertices.map(\.normal) == [
            SIMD3<Float>(0, 0, 1), SIMD3<Float>(0, 0, 1), SIMD3<Float>(0, 0, 1),
            SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 1, 0),
        ])
    }

    @Test
    func `Rejects an overflowing accessor offset or truncated GLB`() throws {
        var document = positionDocument(uri: "mesh.bin")
        document["accessors"] = [
            ["bufferView": 0, "byteOffset": Int.max, "componentType": 5126, "count": 3, "type": "VEC3"],
        ]
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(0, "unaligned access or out of bounds")) {
            try GLTFMeshLoader.load(data: json(document))
        }

        document["buffers"] = [["byteLength": 36]]
        let binary = try glb(document, binary: triangleBytes())
        #expect(throws: GLTFMeshLoader.Failure.invalidGLB("incorrect total length")) {
            try GLTFMeshLoader.load(data: Data(binary.dropLast()))
        }
    }

    @Test
    func `Rejects reserved and out-of-bounds indices`() throws {
        for invalid in [UInt8(3), UInt8.max] {
            var bytes = triangleBytes()
            bytes.append(contentsOf: [0, 1, invalid])
            let uri = "data:application/octet-stream;base64,\(bytes.base64EncodedString())"
            var document = positionDocument(uri: uri)
            document["buffers"] = [["byteLength": bytes.count, "uri": uri]] as [[String: Any]]
            document["bufferViews"] = [
                ["buffer": 0, "byteLength": 36],
                ["buffer": 0, "byteOffset": 36, "byteLength": 3],
            ]
            document["accessors"] = [
                ["bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"],
                ["bufferView": 1, "componentType": 5121, "count": 3, "type": "SCALAR"],
            ]
            document["meshes"] = [["primitives": [["attributes": ["POSITION": 0], "indices": 1]]]]
            #expect(throws: GLTFMeshLoader.Failure.invalidIndex(UInt32(invalid))) {
                try GLTFMeshLoader.load(data: json(document))
            }
        }
    }

    @Test
    func `Reports unsupported geometry features before reading buffers`() throws {
        var document = positionDocument()
        document["extensionsRequired"] = ["KHR_texture_transform"]
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_texture_transform")) {
            try GLTFMeshLoader.load(data: json(document))
        }
        document.removeValue(forKey: "extensionsRequired")
        document["meshes"] = [["primitives": [["attributes": ["POSITION": 0], "mode": 5]]]]
        #expect(throws: GLTFMeshLoader.Failure.unsupportedMode(5)) {
            try GLTFMeshLoader.load(data: json(document))
        }
        let draco: [String: Any] = [
            "attributes": ["POSITION": 0], "extensions": ["KHR_draco_mesh_compression": ["bufferView": 99, "attributes": ["POSITION": 0]]],
        ]
        document["meshes"] = [["primitives": [draco]]]
        document["extensionsRequired"] = ["KHR_draco_mesh_compression"]
        #expect(throws: GLTFMeshLoader.Failure.invalidDracoCompression(99, "missing buffer view")) {
            try GLTFMeshLoader.load(data: json(document))
        }
        document.removeValue(forKey: "extensionsRequired")
        document["meshes"] = [["primitives": [["attributes": ["POSITION": 0]]]]]
        document["accessors"] = [
            ["bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "sparse": ["count": 1]],
        ]
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("sparse accessor 0")) {
            try GLTFMeshLoader.load(data: json(document))
        }
    }

    @Test
    func `URL loading confines buffers to adjacent regular files`() throws {
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent("frostfield-gltf-\(UUID().uuidString)")
        let directory = root.appendingPathComponent("assets")
        try files.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? files.removeItem(at: root) }
        let documentURL = directory.appendingPathComponent("mesh.gltf")
        let outside = root.appendingPathComponent("outside.bin")
        try triangleBytes().write(to: outside)
        try triangleBytes().write(to: directory.appendingPathComponent("mesh data.bin"))
        try files.createSymbolicLink(at: directory.appendingPathComponent("link.bin"), withDestinationURL: outside)

        let adjacent = try json(positionDocument(uri: "mesh%20data.bin"))
        try adjacent.write(to: documentURL)
        #expect(try GLTFMeshLoader.load(contentsOf: documentURL).indices == [0, 1, 2])
        #expect(throws: GLTFMeshLoader.Failure.externalBufferRequiresURL("mesh%20data.bin")) {
            try GLTFMeshLoader.load(data: adjacent)
        }

        let traversal = "%2E%2E%2Foutside.bin"
        try json(positionDocument(uri: traversal)).write(to: documentURL)
        #expect(throws: GLTFMeshLoader.Failure.unsafeBufferURI(traversal)) {
            try GLTFMeshLoader.load(contentsOf: documentURL)
        }

        try json(positionDocument(uri: "link.bin")).write(to: documentURL)
        #expect(throws: GLTFMeshLoader.Failure.unreadableFile("link.bin")) {
            try GLTFMeshLoader.load(contentsOf: documentURL)
        }
    }

    @Test
    func `Rejects a resource path through an intermediate symlink`() throws {
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent("frostfield-gltf-\(UUID().uuidString)")
        let directory = root.appendingPathComponent("assets")
        let outside = root.appendingPathComponent("outside")
        try files.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? files.removeItem(at: root) }
        try files.createDirectory(at: outside, withIntermediateDirectories: false)
        try triangleBytes().write(to: outside.appendingPathComponent("\u{0301}payload.bin"))
        try files.createSymbolicLink(at: directory.appendingPathComponent("link"), withDestinationURL: outside)

        // The slash and combining accent are a single Swift Character.
        let uri = "link/%CC%81payload.bin"
        let documentURL = directory.appendingPathComponent("mesh.gltf")
        try json(positionDocument(uri: uri)).write(to: documentURL)
        #expect(throws: GLTFMeshLoader.Failure.unsafeBufferURI(uri)) {
            try GLTFMeshLoader.load(contentsOf: documentURL)
        }
    }

    @Test
    func `Imports the default hierarchy and reuses meshes with all their primitives`() throws {
        var document = positionDocument()
        let triangle: [String: Any] = ["attributes": ["POSITION": 0]]
        let skipped: [String: Any] = ["attributes": ["POSITION": 0], "mode": 1]
        let nodes: [[String: Any]] = [
            ["name": "Unselected", "mesh": 0],
            ["name": "Group", "children": [3, 2], "translation": [1, 2, 3],
             "rotation": [0, 0, 0.70710677, 0.70710677], "scale": [1, 2, 1]],
            ["name": "Right", "mesh": 1],
            ["name": "Left", "mesh": 1,
             "matrix": [0, -2, 0, 0, -3, 0, 0, 0, 0, 0, 4, 0, 5, 6, 7, 1]],
        ]
        let scenes: [[String: Any]] = [
            ["name": "Unselected", "nodes": [0]],
            ["name": "Imported", "nodes": [1]],
            ["name": "Shared root", "nodes": [1]],
        ]
        let meshes: [[String: Any]] = [
            ["primitives": [skipped]],
            ["name": "Repeated mesh", "primitives": [triangle, triangle]],
        ]
        document["scene"] = 1
        document["scenes"] = scenes
        document["nodes"] = nodes
        document["meshes"] = meshes

        let bytes = try json(document)
        let scene = try GLTFMeshLoader.loadScene(data: bytes)
        #expect(scene.sourceIndex == 1)
        #expect(scene.name == "Imported")
        #expect(scene.rootNodeIndices == [1])
        try #require(scene.nodes.map(\.sourceIndex) == [1, 2, 3])
        #expect(scene.nodes.map(\.parentIndex) == [nil, 1, 1])
        #expect(scene.nodes.map(\.meshIndex) == [nil, 1, 1])
        #expect(scene.nodes[0].childIndices == [3, 2])
        #expect(scene.nodes[2].name == "Left")

        let group = scene.nodes[0].localTransform.matrix
        #expect(simd_length(group.columns.0 - SIMD4<Float>(0, 1, 0, 0)) < 1e-5)
        #expect(simd_length(group.columns.1 - SIMD4<Float>(-2, 0, 0, 0)) < 1e-5)
        #expect(group.columns.3 == SIMD4<Float>(1, 2, 3, 1))

        let left = scene.nodes[2].localTransform.matrix
        #expect(simd_length(left.columns.0 - SIMD4<Float>(0, -2, 0, 0)) < 1e-5)
        #expect(simd_length(left.columns.1 - SIMD4<Float>(-3, 0, 0, 0)) < 1e-5)
        #expect(simd_length(left.columns.2 - SIMD4<Float>(0, 0, 4, 0)) < 1e-5)
        #expect(left.columns.3 == SIMD4<Float>(5, 6, 7, 1))

        try #require(scene.meshes.count == 1)
        #expect(scene.meshes[0].sourceIndex == 1)
        #expect(scene.meshes[0].name == "Repeated mesh")
        #expect(scene.meshes[0].primitives.map(\.indices) == [[0, 1, 2], [0, 1, 2]])

        let shared = try GLTFMeshLoader.loadScene(data: bytes, sceneIndex: 2)
        #expect(shared.name == "Shared root")
        #expect(shared.rootNodeIndices == scene.rootNodeIndices)
        #expect(throws: GLTFMeshLoader.Failure.unsupportedMode(1)) {
            try GLTFMeshLoader.loadScene(data: bytes, sceneIndex: 0)
        }
    }

    @Test
    func `Loads a GLB scene from a local URL`() throws {
        var document = positionDocument()
        document["buffers"] = [["byteLength": 36]]
        document["nodes"] = [["mesh": 0]]
        document["scenes"] = [["nodes": [0]]]
        let bytes = try glb(document, binary: triangleBytes())
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).glb")
        try bytes.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let scene = try GLTFMeshLoader.loadScene(contentsOf: url)
        #expect(scene.rootNodeIndices == [0])
        #expect(scene.nodes[0].localTransform.matrix.columns.3 == SIMD4<Float>(0, 0, 0, 1))
        #expect(scene.meshes[0].primitives[0].indices == [0, 1, 2])
    }

    @Test
    func `Rejects cycles duplicate links and roots with parents`() throws {
        #expect(throws: GLTFMeshLoader.Failure.nodeCycle(0)) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["children": [1]], ["children": [0]]]))
        }
        #expect(throws: GLTFMeshLoader.Failure.multipleParents(2)) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["children": [2]], ["children": [2]], [:]]))
        }
        #expect(throws: GLTFMeshLoader.Failure.duplicateNodeReference(1)) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["children": [1, 1]], [:]]))
        }
        #expect(throws: GLTFMeshLoader.Failure.duplicateNodeReference(0)) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [[:]], roots: [0, 0]))
        }
        #expect(throws: GLTFMeshLoader.Failure.sceneRootHasParent(1)) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["children": [1]], [:]], roots: [1]))
        }
        #expect(throws: GLTFMeshLoader.Failure.missingNode(7)) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["children": [7]]]))
        }
    }

    @Test
    func `Rejects transforms the scene cannot represent and skinned nodes`() throws {
        let identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        let mixed: [String: Any] = ["matrix": identity, "translation": [0, 0, 0]]
        #expect(throws: GLTFMeshLoader.Failure.invalidNodeTransform(0, "matrix and TRS cannot be combined")) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [mixed]))
        }

        let shear = [1.0, 0, 0, 0, 0.25, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        #expect(throws: GLTFMeshLoader.Failure.invalidNodeTransform(0, "matrix cannot be decomposed into TRS without shear")) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["matrix": shear]]))
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidNodeTransform(0, "transform must be finite and invertible")) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["scale": [1, 0, 1]]]))
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("skin on node 0")) {
            try GLTFMeshLoader.loadScene(data: sceneData(nodes: [["skin": 0]]))
        }
    }

    @Test
    func `Keeps single primitive loading independent of the scene hierarchy`() throws {
        var document = positionDocument()
        let library = try json(document)
        #expect(try GLTFMeshLoader.load(data: library).indices == [0, 1, 2])
        #expect(throws: GLTFMeshLoader.Failure.missingScene(0)) {
            try GLTFMeshLoader.loadScene(data: library)
        }

        document["nodes"] = [["children": [0]]]
        document["scenes"] = [["nodes": [0]]]
        let invalidHierarchy = try json(document)
        #expect(try GLTFMeshLoader.load(data: invalidHierarchy).indices == [0, 1, 2])
        #expect(throws: GLTFMeshLoader.Failure.nodeCycle(0)) {
            try GLTFMeshLoader.loadScene(data: invalidHierarchy)
        }
    }

    private func sceneData(nodes: [[String: Any]], roots: [Int] = [0]) throws -> Data {
        try json(["asset": ["version": "2.0"], "nodes": nodes, "scenes": [["nodes": roots]]])
    }

    private func positionDocument(uri: String? = nil) -> [String: Any] {
        let uri = uri ?? "data:application/octet-stream;base64,\(triangleBytes().base64EncodedString())"
        return [
            "asset": ["version": "2.0"],
            "buffers": [["byteLength": 36, "uri": uri]],
            "bufferViews": [["buffer": 0, "byteLength": 36]],
            "accessors": [
                ["bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0, 0, 0], "max": [1, 1, 0]],
            ],
            "meshes": [["primitives": [["attributes": ["POSITION": 0]]]]],
        ]
    }

    private func triangleBytes() -> Data {
        var bytes = Data()
        append([Float(0), 0, 0, 1, 0, 0, 0, 1, 0], to: &bytes)
        return bytes
    }

    private func json(_ document: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: document)
    }

    private func glb(_ document: [String: Any], binary: Data) throws -> Data {
        var json = try json(document)
        while !json.count.isMultiple(of: 4) { json.append(0x20) }
        var binary = binary
        while !binary.count.isMultiple(of: 4) { binary.append(0) }
        var data = Data()
        append(UInt32(0x4654_6C67), to: &data)
        append(UInt32(2), to: &data)
        append(UInt32(12 + 8 + json.count + 8 + binary.count), to: &data)
        append(UInt32(json.count), to: &data)
        append(UInt32(0x4E4F_534A), to: &data)
        data.append(json)
        append(UInt32(binary.count), to: &data)
        append(UInt32(0x004E_4942), to: &data)
        data.append(binary)
        return data
    }

    private func append(_ floats: [Float], to data: inout Data) {
        for value in floats { append(value.bitPattern, to: &data) }
    }

    private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var encoded = value.littleEndian
        withUnsafeBytes(of: &encoded) { data.append(contentsOf: $0) }
    }
}
