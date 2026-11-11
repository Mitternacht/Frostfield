import Foundation
import Frostfield
import Testing
import simd

struct GLTFAnimationTests {
    @Test
    func `Samples independent channels and preserves the source graph`() throws {
        var fixture = GLTFMotionFixture()
        let moveTimes = fixture.floats([1, 3], type: "SCALAR", count: 2, bounds: [1, 3])
        let move = fixture.floats([2, 0, 0, 6, 0, 0], type: "VEC3", count: 2)
        let scaleTimes = fixture.floats([0, 3, 4], type: "SCALAR", count: 3, bounds: [0, 4])
        let scale = fixture.floats([1, 1, 1, 2, 2, 2, 3, 3, 3], type: "VEC3", count: 3)
        let spinTimes = fixture.floats([0, 4], type: "SCALAR", count: 2, bounds: [0, 4])
        let spin = fixture.floats([
            0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0.5, 0,
            0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 0,
        ], type: "VEC4", count: 6)
        let animation: [String: Any] = [
            "name": "Different clocks",
            "channels": [channel(0, "translation", 0), channel(0, "scale", 1), channel(0, "rotation", 2)],
            "samplers": [
                ["input": moveTimes, "output": move],
                ["input": scaleTimes, "output": scale, "interpolation": "STEP"],
                ["input": spinTimes, "output": spin, "interpolation": "CUBICSPLINE"],
            ],
        ]
        let data = try fixture.document(["animations": [animation]])
        #expect(try GLTFMeshLoader.loadScene(data: data).animations.isEmpty)
        let scene = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true)
        let clip = try #require(scene.animations.first)
        #expect(clip.sourceIndex == 0)
        #expect(clip.name == "Different clocks")
        #expect(clip.duration == 4)
        #expect(clip.channels.map(\.interpolation) == [.linear, .step, .cubicSpline])

        let child = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 1, 0)), materialParameters: SIMD4(1, 0, 1, 1)
        )
        let root = SceneNode(transform: SceneTransform(translation: SIMD3(1, 0, 0)), children: [child])
        let graph = try graphWith(root)
        let sample = try clip.applying(to: graph, nodeIDs: [0: root.id], at: 2)
        let transformed = try #require(sample.roots.first)
        #expect(transformed.transform.translation == SIMD3(4, 0, 0))
        #expect(transformed.transform.scale == SIMD3(repeating: 1))
        #expect(abs(transformed.transform.rotation.vector.z - 0.25 / sqrt(1.0625)) < 0.00001)
        #expect(abs(simd_length(transformed.transform.rotation.vector) - 1) < 0.00001)
        #expect(transformed.children.first?.id == child.id)
        #expect(transformed.children.first?.transform.translation == child.transform.translation)
        #expect(transformed.children.first?.materialParameters == child.materialParameters)
        #expect(graph.roots.first?.transform.translation == SIMD3(1, 0, 0))

        let loop = try clip.applying(to: graph, nodeIDs: [0: root.id], at: -1, looping: true)
        #expect(loop.roots.first?.transform.translation == SIMD3(6, 0, 0))
        #expect(loop.roots.first?.transform.scale == SIMD3(repeating: 2))
        let clamped = try clip.applying(to: graph, nodeIDs: [0: root.id], at: 20)
        #expect(clamped.roots.first?.transform.scale == SIMD3(repeating: 3))
        #expect(throws: GLTFAnimationClip.SamplingError.missingNodeMapping(0)) {
            try clip.applying(to: graph, nodeIDs: [:], at: 2)
        }
    }

    @Test
    func `Decodes normalized rotation keys and uses the shortest arc`() throws {
        var fixture = GLTFMotionFixture()
        let times = fixture.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let output = fixture.bytes(
            [0, 0, 0, 127, 0, 0, UInt8(bitPattern: -90), UInt8(bitPattern: -90)],
            type: "VEC4", count: 2, component: 5120, normalized: true
        )
        let animation: [String: Any] = [
            "channels": [channel(0, "rotation", 0)],
            "samplers": [["input": times, "output": output]],
        ]
        let data = try fixture.document(["animations": [animation]])
        let scene = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true)
        let clip = try #require(scene.animations.first)
        let root = SceneNode(transform: SceneTransform(translation: SIMD3(1, 2, 3), scale: SIMD3(repeating: 2)))
        let sample = try clip.applying(to: graphWith(root), nodeIDs: [0: root.id], at: 1)
        let transform = try #require(sample.roots.first?.transform)
        #expect(transform.translation == root.transform.translation)
        #expect(transform.scale == root.transform.scale)
        let vector = transform.rotation.act(SIMD3(1, 0, 0))
        #expect(abs(vector.x - sqrt(0.5)) < 0.00001)
        #expect(abs(vector.y - sqrt(0.5)) < 0.00001)
    }

    @Test
    func `Rejects invalid input times, output counts and duplicate targets`() throws {
        var fixture = GLTFMotionFixture()
        let repeated = fixture.floats([0, 0], type: "SCALAR", count: 2, bounds: [0, 0])
        let times = fixture.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let output = fixture.floats([1, 0, 0, 2, 0, 0], type: "VEC3", count: 2)
        let single = fixture.floats([1, 0, 0], type: "VEC3", count: 1)

        func data(input: Int, output: Int, duplicate: Bool = false) throws -> Data {
            let channels = Array(repeating: channel(0, "translation", 0), count: duplicate ? 2 : 1)
            let animation: [String: Any] = ["channels": channels, "samplers": [["input": input, "output": output]]]
            return try fixture.document(["animations": [animation]])
        }

        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(repeated, "animation times must be nonnegative and strictly increasing")) {
            try GLTFMeshLoader.loadScene(data: data(input: repeated, output: output), includeAnimations: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(single, "animation output count does not match the sampler")) {
            try GLTFMeshLoader.loadScene(data: data(input: times, output: single), includeAnimations: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidAnimation(0, "node 0 has duplicate translation channels")) {
            try GLTFMeshLoader.loadScene(data: data(input: times, output: output, duplicate: true), includeAnimations: true)
        }
    }

    private func graphWith(_ root: SceneNode) throws -> SceneGraph {
        SceneGraph(roots: [root], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
    }

    private func channel(_ node: Int, _ path: String, _ sampler: Int) -> [String: Any] {
        ["target": ["node": node, "path": path], "sampler": sampler]
    }
}

struct GLTFMotionFixture {
    private var buffer = Data()
    private var views: [[String: Int]] = []
    private var accessors: [[String: Any]] = []

    mutating func floats(_ values: [Float], type: String, count: Int, bounds: [Float]? = nil) -> Int {
        let data = values.flatMap { value in
            withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) }
        }
        return bytes(data, type: type, count: count, component: 5126, bounds: bounds)
    }

    mutating func bytes(
        _ data: [UInt8], type: String, count: Int, component: Int, normalized: Bool = false, bounds: [Float]? = nil
    ) -> Int {
        while !buffer.count.isMultiple(of: 4) { buffer.append(0) }
        views.append(["buffer": 0, "byteOffset": buffer.count, "byteLength": data.count])
        buffer.append(contentsOf: data)
        var accessor: [String: Any] = [
            "bufferView": views.count - 1, "componentType": component, "type": type, "count": count,
        ]
        if normalized { accessor["normalized"] = true }
        if let bounds {
            accessor["min"] = [bounds[0]]
            accessor["max"] = [bounds[1]]
        }
        accessors.append(accessor)
        return accessors.count - 1
    }

    func document(_ sections: [String: Any]) throws -> Data {
        var document: [String: Any] = [
            "asset": ["version": "2.0"], "scenes": [["nodes": [0]]], "nodes": [[:] as [String: Int]],
            "buffers": [["byteLength": buffer.count, "uri": "data:application/octet-stream;base64,\(buffer.base64EncodedString())"]],
            "bufferViews": views, "accessors": accessors,
        ]
        for (key, value) in sections { document[key] = value }
        return try JSONSerialization.data(withJSONObject: document)
    }
}
