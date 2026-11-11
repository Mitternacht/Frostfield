import Foundation
import Frostfield
import Testing
import simd

struct GLTFMorphTargetTests {
    @Test("Clip optimization keeps imported morph channels on the mixer weight path")
    func optimizedMorphClip() throws {
        var asset = MorphAsset()
        let primitive = asset.triangle(targets: asset.standardTargets())
        let times = asset.floats([0, 1, 2, 3, 4, 5], type: "SCALAR", count: 6, bounds: [0, 5])
        let weights = asset.floats([0, 0.25, 0, 0.25, 0, 0.25, 1, 0.75, 1, 0.75, 1, 0.75], type: "SCALAR", count: 12)
        let animation: [String: Any] = [
            "channels": [["target": ["node": 0, "path": "weights"], "sampler": 0]],
            "samplers": [["input": times, "output": weights]],
        ]
        let data = try asset.data(primitives: [primitive], animations: [animation])
        let scene = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true, includeMorphTargets: true)
        let node = UUID()
        let source = try AnimationClip(try #require(scene.animations.first), nodeIDs: [0: node]).makingAdditive()
        let reduced = source.optimized()
        #expect(try reduced.typedTracks().map(\.times) == [[0, 2, 3, 5]])
        #expect(reduced.tracks.isEmpty && reduced.blendMode == .additive)

        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: reduced, timeScale: 0)
        try mixer.play(action)
        try mixer.setTime(2.5, for: action)
        let rest: [Float] = [0.25, 0.25]
        #expect(try mixer.morphWeights(for: node, rest: rest) == [0.75, 0.5])
        #expect(try reduced.sample(at: 2.5) == source.sample(at: 2.5))
    }

    @Test
    func `Imports morph offsets, default weights, node weights and target names`() throws {
        var asset = MorphAsset()
        let primitive = asset.triangle(targets: asset.standardTargets())
        let nodes: [[String: Any]] = [["mesh": 0, "weights": [0.5, 1]], ["mesh": 0]]
        let data = try asset.data(
            primitives: [primitive], meshWeights: [0.25, 0.5], names: ["Stretch", "Lift"], nodes: nodes
        )

        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("morph weights on node 0")) {
            try GLTFMeshLoader.loadScene(data: data)
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("morph targets")) {
            try GLTFMeshLoader.load(data: data)
        }
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMorphTargets: true)
        let morph = try #require(scene.meshes.first?.morphTargets)
        #expect(morph.defaultWeights == [0.25, 0.5])
        #expect(morph.names == ["Stretch", "Lift"])
        #expect(scene.nodes.map(\.morphWeights) == [[0.5, 1], [0.25, 0.5]])
        let targets = try #require(morph.primitives.first)
        #expect(targets.vertexCount == 3)
        #expect(targets.targets.count == 2)
        #expect(targets.targets[0].positionOffsets == [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 0)])
        #expect(targets.targets[0].normalOffsets == Array(repeating: SIMD3(1, 0, 0), count: 3))
        #expect(targets.targets[0].tangentOffsets == nil)
        #expect(targets.targets[1].tangentOffsets == Array(repeating: SIMD3(0, 1, 0), count: 3))

        let base = try #require(scene.meshes.first?.primitives.first)
        let morphed = try targets.apply(weights: try #require(scene.nodes[0].morphWeights), to: base)
        expectStandardMorph(morphed)
    }

    @Test
    func `Applies two targets on the CPU with renormalized normals and tangents`() throws {
        let base = try MeshGeometry(
            vertices: [SIMD3<Float>(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0)].map {
                MeshVertex(position: $0, normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2($0.x, $0.y), tangent: SIMD4(1, 0, 0, 1))
            },
            indices: [0, 1, 2]
        )
        let targets = try MeshMorphTargets(vertexCount: 3, targets: [
            .init(positionOffsets: [.zero, SIMD3(1, 0, 0), .zero], normalOffsets: Array(repeating: SIMD3(1, 0, 0), count: 3)),
            .init(positionOffsets: Array(repeating: SIMD3(0, 0, 1), count: 3), tangentOffsets: Array(repeating: SIMD3(0, 1, 0), count: 3)),
        ])
        let morphed = try targets.apply(weights: [0.5, 1], to: base)
        expectStandardMorph(morphed)
        #expect(morphed.vertices.map(\.textureCoordinate) == base.vertices.map(\.textureCoordinate))
        #expect(morphed.indices == base.indices)

        let unchanged = try targets.apply(weights: [0, 0], to: base)
        #expect(unchanged.vertices.map(\.position) == base.vertices.map(\.position))
        let partial = try targets.apply(weights: [1, 0], to: base)
        #expect(partial.vertices[1].position == SIMD3(2, 0, 0))
        #expect(partial.vertices[1].tangent == SIMD4(1, 0, 0, 1))
    }

    @Test
    func `Reads sparse morph offsets over zeros and over dense values`() throws {
        var asset = MorphAsset()
        let lift = asset.sparse(count: 3, base: nil, indices: [2], values: [0, 0, 2])
        let tilt = asset.sparse(count: 3, base: [1, 0, 0, 1, 0, 0, 1, 0, 0], indices: [0, 2], values: [0, 1, 0, 0, 2, 0])
        let primitive = asset.triangle(targets: [["POSITION": lift, "NORMAL": tilt]])
        let scene = try GLTFMeshLoader.loadScene(data: asset.data(primitives: [primitive]), includeMorphTargets: true)
        let target = try #require(scene.meshes.first?.morphTargets?.primitives.first?.targets.first)
        #expect(target.positionOffsets == [.zero, .zero, SIMD3(0, 0, 2)])
        #expect(target.normalOffsets == [SIMD3(0, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 2, 0)])
        #expect(scene.nodes.first?.morphWeights == [0])

        for indices: [UInt8] in [[2, 1], [1, 3]] {
            var invalid = MorphAsset()
            let offsets = invalid.sparse(count: 3, base: nil, indices: indices, values: [0, 0, 1, 0, 0, 2])
            let primitive = invalid.triangle(targets: [["POSITION": offsets]])
            let data = try invalid.data(primitives: [primitive])
            #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(offsets, "sparse indices must increase strictly within the accessor")) {
                try GLTFMeshLoader.loadScene(data: data, includeMorphTargets: true)
            }
        }
    }

    @Test
    func `Samples weights channels with linear, step, cubic and normalized keys`() throws {
        var asset = MorphAsset()
        let primitive = asset.triangle(targets: asset.standardTargets())
        let linearTimes = asset.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let unitTimes = asset.floats([0, 1], type: "SCALAR", count: 2, bounds: [0, 1])
        let linear = asset.floats([0, 0, 1, 0.5], type: "SCALAR", count: 4)
        let step = asset.floats([0.2, 0.4, 0.6, 0.8], type: "SCALAR", count: 4)
        let cubic = asset.floats([0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0], type: "SCALAR", count: 12)
        let normalized = asset.bytes([0, 51, 255, 102], type: "SCALAR", count: 4, component: 5121, normalized: true)
        func animation(_ input: Int, _ output: Int, _ interpolation: String) -> [String: Any] {
            [
                "channels": [["target": ["node": 0, "path": "weights"], "sampler": 0]],
                "samplers": [["input": input, "output": output, "interpolation": interpolation]],
            ]
        }
        let data = try asset.data(primitives: [primitive], animations: [
            animation(linearTimes, linear, "LINEAR"), animation(unitTimes, step, "STEP"),
            animation(unitTimes, cubic, "CUBICSPLINE"), animation(unitTimes, normalized, "LINEAR"),
        ])
        let scene = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true, includeMorphTargets: true)
        #expect(scene.animations.count == 4)
        #expect(scene.animations[0].channels.first?.target == .weights)
        #expect(scene.animations[0].duration == 2)

        let id = UUID()
        func weights(_ clip: Int, at time: Float) throws -> [Float]? {
            try scene.animations[clip].sample(at: time, nodeIDs: [0: id])[id]?.morphWeights
        }
        #expect(try weights(0, at: 1) == [0.5, 0.25])
        #expect(try weights(1, at: 0.5) == [0.2, 0.4])
        #expect(try weights(1, at: 1) == [0.6, 0.8])
        let spline = try #require(try weights(2, at: 0.5))
        #expect(abs(spline[0] - 0.75) < 1e-6 && abs(spline[1] - 0.5) < 1e-6)
        let bytes = try #require(try weights(3, at: 0.5))
        #expect(abs(bytes[0] - 0.5) < 1e-6 && abs(bytes[1] - 0.3) < 1e-6)

        // A graph has no morph state, so applying the clip leaves transforms alone.
        let node = SceneNode(id: id, transform: .init(translation: SIMD3(1, 2, 3)))
        let graph = SceneGraph(roots: [node], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let applied = try scene.animations[0].applying(to: graph, nodeIDs: [0: id], at: 1)
        #expect(applied.roots[0].transform.translation == node.transform.translation)

        var mixer = AnimationMixer()
        let clip = try AnimationClip(scene.animations[0], nodeIDs: [0: id])
        let action = try mixer.addAction(for: clip)
        try mixer.play(action)
        _ = try mixer.update(deltaTime: 1)
        let rest: [Float] = [0.5, 1]
        #expect(try mixer.morphWeights(for: id, rest: rest) == [0.5, 0.25])
        try mixer.setWeight(0.5, for: action)
        #expect(try mixer.morphWeights(for: id, rest: rest) == [0.5, 0.625])
        #expect(try mixer.apply(to: graph).roots[0].transform.translation == node.transform.translation)
        #expect(throws: AnimationMixer.Failure.morphWeightCountMismatch(id, expected: 3, actual: 2)) {
            try mixer.morphWeights(for: id, rest: [0, 0, 0])
        }

        try mixer.stop(action)
        let layer = try mixer.addAction(for: clip.makingAdditive(), timeScale: 0)
        try mixer.setTime(1, for: layer)
        try mixer.play(layer)
        #expect(try mixer.morphWeights(for: id, rest: rest) == [1, 1.25])
    }

    @Test
    func `Rejects weights channels without morph targets or with mismatched counts`() throws {
        var fixture = GLTFMotionFixture()
        let times = fixture.floats([0, 1], type: "SCALAR", count: 2, bounds: [0, 1])
        let values = fixture.floats([0, 1], type: "SCALAR", count: 2)
        let animation: [String: Any] = [
            "channels": [["target": ["node": 0, "path": "weights"], "sampler": 0]],
            "samplers": [["input": times, "output": values]],
        ]
        let meshless = try fixture.document(["animations": [animation]])
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("animation target weights")) {
            try GLTFMeshLoader.loadScene(data: meshless, includeAnimations: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidAnimation(0, "node 0 has no morph targets for its weights channel")) {
            try GLTFMeshLoader.loadScene(data: meshless, includeAnimations: true, includeMorphTargets: true)
        }

        var asset = MorphAsset()
        let primitive = asset.triangle(targets: asset.standardTargets())
        let unitTimes = asset.floats([0, 1], type: "SCALAR", count: 2, bounds: [0, 1])
        let short = asset.floats([0, 0, 1], type: "SCALAR", count: 3)
        let data = try asset.data(primitives: [primitive], animations: [[
            "channels": [["target": ["node": 0, "path": "weights"], "sampler": 0]],
            "samplers": [["input": unitTimes, "output": short]],
        ]])
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(short, "animation output count does not match the sampler")) {
            try GLTFMeshLoader.loadScene(data: data, includeAnimations: true, includeMorphTargets: true)
        }
    }

    @Test
    func `Rejects morph targets the importer cannot represent`() throws {
        func load(_ build: (inout MorphAsset) throws -> Data) throws -> GLTFScene {
            var asset = MorphAsset()
            return try GLTFMeshLoader.loadScene(data: build(&asset), includeMorphTargets: true)
        }

        #expect(throws: GLTFMeshLoader.Failure.invalidMorphTarget(0, "target attribute TANGENT is missing from the base primitive")) {
            try load { asset in
                let offsets = asset.floats(Array(repeating: 0, count: 9), type: "VEC3", count: 3)
                let primitive = asset.triangle(tangents: false, targets: [["TANGENT": offsets]])
                return try asset.data(primitives: [primitive])
            }
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("morph target attribute TEXCOORD_0")) {
            try load { asset in
                let offsets = asset.floats(Array(repeating: 0, count: 6), type: "VEC2", count: 3)
                let primitive = asset.triangle(targets: [["TEXCOORD_0": offsets]])
                return try asset.data(primitives: [primitive])
            }
        }
        // The triangle's base attributes use accessors 0 through 2.
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(3, "morph target offsets need one value per vertex")) {
            try load { asset in
                var primitive = asset.triangle(targets: [])
                primitive["targets"] = [["POSITION": asset.floats(Array(repeating: 0, count: 6), type: "VEC3", count: 2)]]
                return try asset.data(primitives: [primitive])
            }
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidMorphWeights(0, "weights need one finite value per morph target of the node's mesh")) {
            try load { asset in
                let primitive = asset.triangle(targets: asset.standardTargets())
                return try asset.data(primitives: [primitive], nodes: [["mesh": 0, "weights": [1]]])
            }
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidMorphTarget(0, "default weights need one finite value per morph target")) {
            try load { asset in
                let primitive = asset.triangle(targets: asset.standardTargets())
                return try asset.data(primitives: [primitive], meshWeights: [1])
            }
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidMorphTarget(0, "every primitive needs the same number of morph targets")) {
            try load { asset in
                let targets = asset.standardTargets()
                let primitives = [asset.triangle(targets: targets), asset.triangle(targets: [targets[0]])]
                return try asset.data(primitives: primitives)
            }
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("morph targets on mesh 0 without vertex normals")) {
            try load { asset in
                var primitive = asset.triangle(targets: asset.standardTargets())
                var attributes = try #require(primitive["attributes"] as? [String: Int])
                attributes["NORMAL"] = nil
                attributes["TANGENT"] = nil
                primitive["attributes"] = attributes
                primitive["targets"] = [["POSITION": asset.floats(Array(repeating: 0, count: 9), type: "VEC3", count: 3)]]
                return try asset.data(primitives: [primitive])
            }
        }
    }

    @Test
    func `Rejects invalid morph target values and weights`() throws {
        let offsets = Array(repeating: SIMD3<Float>(0, 0, 1), count: 3)
        #expect(throws: MeshMorphTargets.ValidationError.invalidVertexCount) { try MeshMorphTargets(vertexCount: 0, targets: [.init(positionOffsets: [])]) }
        #expect(throws: MeshMorphTargets.ValidationError.emptyTargets) { try MeshMorphTargets(vertexCount: 3, targets: []) }
        #expect(throws: MeshMorphTargets.ValidationError.emptyTarget(0)) { try MeshMorphTargets(vertexCount: 3, targets: [.init()]) }
        #expect(throws: MeshMorphTargets.ValidationError.vertexCountMismatch(target: 1)) {
            try MeshMorphTargets(vertexCount: 3, targets: [.init(positionOffsets: offsets), .init(normalOffsets: Array(offsets.prefix(2)))])
        }
        #expect(throws: MeshMorphTargets.ValidationError.nonfiniteOffset(target: 0)) {
            try MeshMorphTargets(vertexCount: 3, targets: [.init(positionOffsets: [.zero, .zero, SIMD3(.nan, 0, 0)])])
        }

        let base = try MeshGeometry(
            vertices: [SIMD3<Float>(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0)].map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) },
            indices: [0, 1, 2]
        )
        let targets = try MeshMorphTargets(vertexCount: 3, targets: [
            .init(normalOffsets: offsets.map { -$0 }), .init(tangentOffsets: offsets),
        ])
        #expect(throws: MeshMorphTargets.ValidationError.weightCount(expected: 2, actual: 1)) { try targets.apply(weights: [1], to: base) }
        #expect(throws: MeshMorphTargets.ValidationError.nonfiniteWeight) { try targets.apply(weights: [.infinity, 0], to: base) }
        #expect(throws: MeshMorphTargets.ValidationError.geometryVertexCount(expected: 3, actual: 24)) {
            try targets.apply(weights: [0, 0], to: .cube())
        }
        #expect(throws: MeshMorphTargets.ValidationError.missingBaseTangents) { try targets.apply(weights: [0, 1], to: base) }
        #expect(throws: MeshMorphTargets.ValidationError.degenerateNormal(vertex: 0)) { try targets.apply(weights: [1, 0], to: base) }
    }

    /// Checks the standard triangle morphed with weights (0.5, 1).
    private func expectStandardMorph(_ geometry: MeshGeometry) {
        #expect(geometry.vertices.map(\.position) == [SIMD3(0, 0, 1), SIMD3(1.5, 0, 1), SIMD3(0, 1, 1)])
        let normal = simd_normalize(SIMD3<Float>(0.5, 0, 1))
        let tangent = SIMD3<Float>(1, 1, 0) / Float(2).squareRoot()
        for vertex in geometry.vertices {
            #expect(simd_length(vertex.normal - normal) < 1e-6)
            let morphedTangent = vertex.tangent ?? .zero
            #expect(simd_length(SIMD3(morphedTangent.x, morphedTangent.y, morphedTangent.z) - tangent) < 1e-6)
            #expect(morphedTangent.w == 1)
        }
    }
}

/// Builds a one-mesh glTF document with base64 buffers. Add every accessor
/// before calling `data`, which serializes the fixture's current value.
private struct MorphAsset {
    private var buffer = Data()
    private var views: [[String: Int]] = []
    private var accessors: [[String: Any]] = []

    mutating func floats(_ values: [Float], type: String, count: Int, bounds: [Float]? = nil) -> Int {
        var accessor: [String: Any] = [
            "bufferView": view(Self.bytes(values)), "componentType": 5126, "type": type, "count": count,
        ]
        if let bounds {
            accessor["min"] = [bounds[0]]
            accessor["max"] = [bounds[1]]
        }
        accessors.append(accessor)
        return accessors.count - 1
    }

    mutating func bytes(_ data: [UInt8], type: String, count: Int, component: Int, normalized: Bool) -> Int {
        accessors.append([
            "bufferView": view(data), "componentType": component, "type": type, "count": count, "normalized": normalized,
        ])
        return accessors.count - 1
    }

    /// Float VEC3 offsets whose listed vertices are replaced through sparse storage.
    mutating func sparse(count: Int, base: [Float]?, indices: [UInt8], values: [Float]) -> Int {
        var accessor: [String: Any] = [
            "componentType": 5126, "type": "VEC3", "count": count,
            "sparse": [
                "count": indices.count,
                "indices": ["bufferView": view(indices), "componentType": 5121],
                "values": ["bufferView": view(Self.bytes(values))],
            ],
        ]
        if let base { accessor["bufferView"] = view(Self.bytes(base)) }
        accessors.append(accessor)
        return accessors.count - 1
    }

    /// A triangle in the XY plane facing +Z, with +X tangents unless disabled.
    mutating func triangle(tangents: Bool = true, targets: [[String: Int]]) -> [String: Any] {
        var attributes = [
            "POSITION": floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3),
            "NORMAL": floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3),
        ]
        if tangents { attributes["TANGENT"] = floats([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], type: "VEC4", count: 3) }
        var primitive: [String: Any] = ["attributes": attributes]
        if !targets.isEmpty { primitive["targets"] = targets }
        return primitive
    }

    /// A stretch of vertex 1 that tilts normals toward +X, and a lift along +Z that turns tangents toward +Y.
    mutating func standardTargets() -> [[String: Int]] {
        [
            [
                "POSITION": floats([0, 0, 0, 1, 0, 0, 0, 0, 0], type: "VEC3", count: 3),
                "NORMAL": floats([1, 0, 0, 1, 0, 0, 1, 0, 0], type: "VEC3", count: 3),
            ],
            [
                "POSITION": floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3),
                "TANGENT": floats([0, 1, 0, 0, 1, 0, 0, 1, 0], type: "VEC3", count: 3),
            ],
        ]
    }

    func data(
        primitives: [[String: Any]], meshWeights: [Float]? = nil, names: [String]? = nil,
        nodes: [[String: Any]] = [["mesh": 0]], animations: [[String: Any]] = []
    ) throws -> Data {
        var mesh: [String: Any] = ["primitives": primitives]
        if let meshWeights { mesh["weights"] = meshWeights }
        if let names { mesh["extras"] = ["targetNames": names] }
        var document: [String: Any] = [
            "asset": ["version": "2.0"], "scenes": [["nodes": Array(nodes.indices)]], "nodes": nodes, "meshes": [mesh],
            "buffers": [["byteLength": buffer.count, "uri": "data:application/octet-stream;base64,\(buffer.base64EncodedString())"]],
            "bufferViews": views, "accessors": accessors,
        ]
        if !animations.isEmpty { document["animations"] = animations }
        return try JSONSerialization.data(withJSONObject: document)
    }

    private mutating func view(_ data: [UInt8]) -> Int {
        while !buffer.count.isMultiple(of: 4) { buffer.append(0) }
        views.append(["buffer": 0, "byteOffset": buffer.count, "byteLength": data.count])
        buffer.append(contentsOf: data)
        return views.count - 1
    }

    private static func bytes(_ values: [Float]) -> [UInt8] {
        values.flatMap { value in withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) } }
    }
}
