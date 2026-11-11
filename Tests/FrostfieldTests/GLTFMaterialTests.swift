import Foundation
import Frostfield
import Metal
import Testing

struct GLTFMaterialTests {
    @Test
    func `Loads used material factors and embedded PNG with primitive identity`() throws {
        let png = try #require(Data(base64Encoded: Self.png))
        let image: [String: Any] = ["name": "Tint map", "uri": "data:image/png;base64,\(Self.png)"]
        var document = fixture(image: image)
        let first: [String: Any] = ["name": "Defaults"]
        let unused: [String: Any] = ["name": "Unused", "alphaMode": "BLEND"]
        let surface: [String: Any] = [
            "baseColorFactor": [0.25, 0.5, 0.75, 1], "metallicFactor": 0.125,
            "roughnessFactor": 0.625, "baseColorTexture": ["index": 0],
        ]
        let tinted: [String: Any] = ["name": "Tinted", "pbrMetallicRoughness": surface]
        document["materials"] = [first, unused, tinted]
        let attributes = ["POSITION": 0, "TEXCOORD_0": 1]
        let primitives: [[String: Any]] = [
            ["attributes": attributes, "material": 2],
            ["attributes": attributes, "material": 0],
            ["attributes": attributes],
        ]
        document["meshes"] = [["primitives": primitives]]

        let scene = try GLTFMeshLoader.loadScene(data: json(document), includeMaterials: true)
        try #require(scene.meshes.count == 1)
        #expect(scene.meshes[0].primitiveMaterialIndices == [2, 0, nil])
        try #require(scene.materials.map(\.sourceIndex) == [0, 2])
        #expect(scene.materials[0].baseColorFactor == SIMD4<Float>(repeating: 1))
        #expect(scene.materials[0].metallicFactor == 1)
        #expect(scene.materials[0].roughnessFactor == 1)
        #expect(scene.materials[0].baseColorImage == nil)

        let material = scene.materials[1]
        #expect(material.name == "Tinted")
        #expect(material.baseColorFactor == SIMD4<Float>(0.25, 0.5, 0.75, 1))
        #expect(material.metallicFactor == 0.125)
        #expect(material.roughnessFactor == 0.625)
        let decoded = try #require(material.baseColorImage)
        #expect(decoded.sourceIndex == 0)
        #expect(decoded.name == "Tint map")
        #expect(decoded.mimeType == "image/png")
        #expect(decoded.data == png)
    }

    @Test
    func `Reads GLB image bytes and checks the declared MIME type`() throws {
        let png = try #require(Data(base64Encoded: Self.png))
        var binary = geometry()
        let offset = binary.count
        binary.append(png)
        var document = fixture(image: ["bufferView": 2, "mimeType": "image/png"])
        document["buffers"] = [["byteLength": binary.count]]
        document["bufferViews"] = [
            ["buffer": 0, "byteLength": 36],
            ["buffer": 0, "byteOffset": 36, "byteLength": 24],
            ["buffer": 0, "byteOffset": offset, "byteLength": png.count],
        ]

        let scene = try GLTFMeshLoader.loadScene(data: glb(document, binary: binary), includeMaterials: true)
        let material = try #require(scene.materials.first)
        #expect(material.baseColorImage?.data == png)
        document["images"] = [["bufferView": 2, "mimeType": "image/jpeg"]] as [[String: Any]]
        #expect(throws: GLTFMeshLoader.Failure.invalidImage(0, "the MIME type does not match the image signature")) {
            try GLTFMeshLoader.loadScene(data: glb(document, binary: binary), includeMaterials: true)
        }
    }

    @Test
    func `URL import reads an adjacent JPEG but rejects a traversal URI`() throws {
        let jpeg = try #require(Data(base64Encoded: Self.jpeg))
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent("frostfield-material-\(UUID().uuidString)")
        try files.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? files.removeItem(at: root) }

        let file = root.appendingPathComponent("model.gltf")
        try jpeg.write(to: root.appendingPathComponent("base color.jpg"))
        try json(fixture(image: ["uri": "base%20color.jpg"])).write(to: file)
        let scene = try GLTFMeshLoader.loadScene(contentsOf: file, includeMaterials: true)
        let image = try #require(scene.materials.first?.baseColorImage)
        #expect(image.mimeType == "image/jpeg")
        #expect(image.data == jpeg)

        let traversal = "%2e%2e%2fbase.jpg"
        try json(fixture(image: ["uri": traversal])).write(to: file)
        #expect(throws: GLTFMeshLoader.Failure.unsafeBufferURI(traversal)) {
            try GLTFMeshLoader.loadScene(contentsOf: file, includeMaterials: true)
        }
    }

    @Test
    func `Geometry-only scene loading remains independent of unsupported material fields`() throws {
        var document = fixture(image: ["uri": "data:image/png;base64,\(Self.png)"])
        document["materials"] = [["alphaMode": "ADDITIVE"]]
        let data = try json(document)
        let scene = try GLTFMeshLoader.loadScene(data: data)
        #expect(scene.materials.isEmpty)
        #expect(scene.meshes.first?.primitiveMaterialIndices == [0])
        #expect(try GLTFMeshLoader.load(data: data).indices == [0, 1, 2])
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("alpha mode ADDITIVE on material 0")) {
            try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        }
    }

    @Test
    func `Textured materials require primary UV and validate explicit samplers`() throws {
        var document = fixture(image: ["uri": "data:image/png;base64,\(Self.png)"])
        let primitive: [String: Any] = ["attributes": ["POSITION": 0], "material": 0]
        document["meshes"] = [["primitives": [primitive]]]
        #expect(throws: GLTFMeshLoader.Failure.invalidMaterial(0, "each primitive using the base color texture needs TEXCOORD_0")) {
            try GLTFMeshLoader.loadScene(data: json(document), includeMaterials: true)
        }

        document = fixture(image: ["uri": "data:image/png;base64,\(Self.png)"])
        document["textures"] = [["source": 0, "sampler": 0]]
        #expect(throws: GLTFMeshLoader.Failure.invalidTexture(0, "missing sampler 0")) {
            try GLTFMeshLoader.loadScene(data: json(document), includeMaterials: true)
        }
        document["samplers"] = [["minFilter": 9986, "magFilter": 9728, "wrapS": 33071, "wrapT": 33648]]
        let loaded = try GLTFMeshLoader.loadScene(data: json(document), includeMaterials: true)
        let sampling = try #require(loaded.materials.first?.baseColorTexture?.sampling)
        #expect(sampling.minification == .nearestMipmapLinear)
        #expect(sampling.magnification == .nearest)
        #expect(sampling.wrapS == .clampToEdge)
        #expect(sampling.wrapT == .mirroredRepeat)
    }

    @Test
    func `Reads all core maps and cutout material factors`() throws {
        var document = fixture(image: ["uri": "data:image/png;base64,\(Self.png)"])
        let map = ["index": 0]
        document["materials"] = [[
            "pbrMetallicRoughness": ["baseColorTexture": map, "metallicRoughnessTexture": map],
            "normalTexture": ["index": 0, "scale": 0.4],
            "occlusionTexture": ["index": 0, "strength": 0.7],
            "emissiveTexture": map,
            "emissiveFactor": [0.1, 0.4, 1],
            "alphaMode": "MASK", "alphaCutoff": 0.35, "doubleSided": true,
        ]]
        let scene = try GLTFMeshLoader.loadScene(data: json(document), includeMaterials: true)
        let material = try #require(scene.materials.first)
        let indices = [
            material.baseColorTexture, material.metallicRoughnessTexture, material.normalTexture,
            material.occlusionTexture, material.emissiveTexture,
        ].compactMap { $0?.sourceIndex }
        #expect(indices == [0, 0, 0, 0, 0])
        #expect(material.normalScale == 0.4)
        #expect(material.occlusionStrength == 0.7)
        #expect(material.emissiveFactor == SIMD3(0.1, 0.4, 1))
        #expect(material.alphaMode == .mask(cutoff: 0.35))
        #expect(material.isDoubleSided)
        if let device = MTLCreateSystemDefaultDevice() {
            let prepared = try material.prepare(device: device)
            #expect(prepared.baseColorTexture?.pixelFormat == .rgba8Unorm_srgb)
            #expect(prepared.alphaMode == .mask(cutoff: 0.35))
            let geometry = try #require(scene.meshes.first?.primitives.first)
            _ = try MeshRenderer(
                device: device, colorPixelFormat: .bgra8Unorm_srgb,
                geometry: geometry, material: .physicallyBased(prepared)
            )
        }
    }

    private static let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="
    private static let jpeg = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAABv/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AJYA4FH/2Q=="

    private func fixture(image: [String: Any]) -> [String: Any] {
        let bytes = geometry()
        let primitive: [String: Any] = ["attributes": ["POSITION": 0, "TEXCOORD_0": 1], "material": 0]
        let surface = ["baseColorTexture": ["index": 0]]
        return [
            "asset": ["version": "2.0"],
            "buffers": [["byteLength": bytes.count, "uri": "data:application/octet-stream;base64,\(bytes.base64EncodedString())"]],
            "bufferViews": [["buffer": 0, "byteLength": 36], ["buffer": 0, "byteOffset": 36, "byteLength": 24]],
            "accessors": [
                ["bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0, 0, 0], "max": [1, 1, 0]],
                ["bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"],
            ],
            "images": [image],
            "textures": [["source": 0]],
            "materials": [["pbrMetallicRoughness": surface]],
            "meshes": [["primitives": [primitive]]],
            "nodes": [["mesh": 0]],
            "scenes": [["nodes": [0]]],
        ]
    }

    private func geometry() -> Data {
        let scalars: [Float] = [
            0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1,
        ]
        var data = Data()
        for scalar in scalars { append(scalar.bitPattern, to: &data) }
        return data
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
        append(0x4654_6C67, to: &data)
        append(2, to: &data)
        append(UInt32(12 + 8 + json.count + 8 + binary.count), to: &data)
        append(UInt32(json.count), to: &data)
        append(0x4E4F_534A, to: &data)
        data.append(json)
        append(UInt32(binary.count), to: &data)
        append(0x004E_4942, to: &data)
        data.append(binary)
        return data
    }

    private func append(_ word: UInt32, to data: inout Data) {
        var value = word.littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
}
