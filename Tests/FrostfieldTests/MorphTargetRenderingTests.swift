import Foundation
import Frostfield
import Metal
import Testing
import simd

struct MorphTargetLayoutTests {
    @Test("Packed morph records match the Metal layouts")
    func layout() {
        typealias Layout = MeshRenderer.PackedMorphLayout
        typealias Weight = MeshRenderer.PackedMorphWeight
        #expect(MemoryLayout<Layout>.size == 20)
        #expect(MemoryLayout<Layout>.stride == 20)
        #expect(MemoryLayout<Layout>.offset(of: \.activeCount) == 0)
        #expect(MemoryLayout<Layout>.offset(of: \.vertexCount) == 4)
        #expect(MemoryLayout<Layout>.offset(of: \.positionBase) == 8)
        #expect(MemoryLayout<Layout>.offset(of: \.normalBase) == 12)
        #expect(MemoryLayout<Layout>.offset(of: \.tangentBase) == 16)
        #expect(MemoryLayout<Weight>.size == 8)
        #expect(MemoryLayout<Weight>.stride == 8)
        #expect(MemoryLayout<Weight>.offset(of: \.target) == 0)
        #expect(MemoryLayout<Weight>.offset(of: \.weight) == 4)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MorphTargetRenderingTests {
    private let fixture: GeometryRenderFixture
    private let base: MeshGeometry
    private let targets: MeshMorphTargets

    init() throws {
        fixture = try GeometryRenderFixture()
        base = try GeometryRenderFixture.square(0.25)
        // Target 0 stretches the right edge and tilts normals toward +X; target 1
        // lifts the square toward the camera and turns tangents toward +Y. Every
        // value is exact in Float, so GPU and CPU positions agree bit for bit.
        targets = try MeshMorphTargets(vertexCount: 4, targets: [
            .init(
                positionOffsets: [.zero, SIMD3(0.5, 0, 0), SIMD3(0.5, 0, 0), .zero],
                normalOffsets: Array(repeating: SIMD3(0.5, 0, 0), count: 4)
            ),
            .init(
                positionOffsets: Array(repeating: SIMD3(0, 0, 0.5), count: 4),
                tangentOffsets: Array(repeating: SIMD3(0, 0.5, 0), count: 4)
            ),
        ])
    }

    @Test("Per-draw GPU weights match the CPU morph in color and picking", arguments: [
        [Float(0), 0], [1, 0], [0.5, 1], [-0.5, 0.25],
    ])
    func matchesCPU(weights: [Float]) throws {
        let morphed = try fixture.unlit(base, morphTargets: targets)
        let plain = try fixture.unlit(try targets.apply(weights: weights, to: base))
        // Two draws share the mesh with different weights; the second uses the base pose.
        let graph = SceneGraph(roots: [
            SceneNode(transform: SceneTransform(translation: SIMD3(-0.5, 0, -2)), mesh: morphed, morphWeights: weights),
            SceneNode(transform: SceneTransform(translation: SIMD3(0.75, 0, -2)), mesh: morphed),
        ], camera: fixture.camera)
        let reference = SceneGraph(roots: [
            SceneNode(transform: SceneTransform(translation: SIMD3(-0.5, 0, -2)), mesh: plain),
            SceneNode(transform: SceneTransform(translation: SIMD3(0.75, 0, -2)), mesh: try fixture.unlit(base)),
        ], camera: fixture.camera)
        let gpu = try fixture.render(graph)
        let cpu = try fixture.render(reference)
        #expect(gpu.statistics.drawnMeshCount == 2)
        #expect(zip(gpu.pixels, cpu.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.002) })
        #expect(gpu.pixels.contains { $0.w == 1 })

        // Picking morphs the same triangles on the CPU.
        let rightEdge = -0.5 + 0.25 + 0.5 * weights[0]
        let depth = 2 - 0.5 * weights[1]
        let inside = (rightEdge - 0.1) / depth * 0.5 + 0.5
        let hit = try #require(try graph.pick(at: SIMD2(inside, 0.5), aspectRatio: 1))
        #expect(hit.nodeID == graph.roots[0].id)
        #expect(abs(hit.intersection.position.z - -depth) < 1e-5)
        let outside = (rightEdge + 0.05) / depth * 0.5 + 0.5
        #expect(try graph.pick(at: SIMD2(outside, 0.5), aspectRatio: 1)?.nodeID != graph.roots[0].id)
    }

    @Test("Vertex-stage normals and tangents match the CPU morph, with and without skinning")
    func normalsAndTangents() throws {
        let library = try fixture.device.makeLibrary(source: Self.frameSource, options: nil)
        let material = MeshMaterial.custom(library: library, fragment: "frostfield_test_frame")
        let skin = try MeshSkin(
            jointCount: 1,
            influences: Array(repeating: VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0)), count: 4)
        )
        // A quarter turn about Z with a stretch, applied after morphing.
        let pose = SceneTransform(
            rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1)), scale: SIMD3(1.5, 1, 1)
        ).matrix
        let palette = try JointPalette(matrices: [pose])
        let weights: [Float] = [0.5, 1]
        let expected = try targets.apply(weights: weights, to: base)
        for skinned in [false, true] {
            let mesh = try fixture.unlit(base, skin: skinned ? skin : nil, morphTargets: targets, material: material)
            let reference = skinned ? try skin.deform(expected, using: palette) : expected
            for (selector, value) in [(Float(0), reference.vertices[0].normal), (1, tangentDirection(reference.vertices[0]))] {
                let node = SceneNode(
                    transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh,
                    materialParameters: SIMD4(selector, 0, 0, 1), jointPalette: skinned ? palette : nil,
                    morphWeights: weights
                )
                let pixels = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera)).pixels
                let sample = fixture.pixel(pixels, x: 32, y: 32)
                #expect(simd_distance(SIMD3(sample.x, sample.y, sample.z), value) < 0.003, "skinned \(skinned), selector \(selector)")
            }
        }
    }

    @Test("Morph targets blend in the bind pose before skinning in color and picking")
    func morphThenSkin() throws {
        let skin = try MeshSkin(
            jointCount: 1,
            influences: Array(repeating: VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0)), count: 4)
        )
        // Doubling X after the stretch puts the right edge at 2 × (0.25 + 0.5) − 0.5 = 1.
        // Skinning first would leave it at 2 × 0.25 − 0.5 + 0.5 = 0.5.
        let pose = try JointPalette(matrices: [translationMatrix(-0.5) * SceneTransform(scale: SIMD3(2, 1, 1)).matrix])
        let weights: [Float] = [1, 0]
        let mesh = try fixture.unlit(base, skin: skin, morphTargets: targets)
        let node = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh, jointPalette: pose, morphWeights: weights
        )
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let reference = try skin.deform(try targets.apply(weights: weights, to: base), using: pose)
        let expected = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: try fixture.unlit(reference))],
            camera: fixture.camera
        )
        let gpu = try fixture.render(graph).pixels
        let cpu = try fixture.render(expected).pixels
        #expect(zip(gpu, cpu).allSatisfy { isNearly($0, $1, tolerance: 0.002) })
        let column = fixture.column(x: 0.9, depth: 2)
        #expect(fixture.pixel(gpu, x: column, y: 32).w == 1)
        #expect(fixture.pixel(gpu, x: fixture.column(x: 1.1, depth: 2), y: 32).w == 0)
        #expect(try graph.pick(at: SIMD2(0.9 / 2 * 0.5 + 0.5, 0.5), aspectRatio: 1)?.nodeID == node.id)
    }

    @Test("Morphed casters shadow directional and punctual receivers from their blended positions")
    func morphedShadows() throws {
        // Directional: the stretch extends a blocker over the light path to x = −0.14 on the receiver.
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let blocker = try fixture.unlit(base, morphTargets: targets)
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        func receiverValue(_ weights: [Float]) throws -> Float {
            let node = SceneNode(transform: SceneTransform(translation: SIMD3(0.25, 0, -2)), mesh: blocker, morphWeights: weights)
            let pixels = try fixture.render(
                SceneGraph(roots: [receiver, node], camera: fixture.camera), shadowMap: shadowMap
            ).pixels
            return fixture.pixel(pixels, x: fixture.column(x: -0.15, depth: 3), y: 32).x
        }
        #expect(abs(try receiverValue([0, 0]) - 0.832) < 0.01)
        #expect(abs(try receiverValue([1, 0]) - 0.16) < 0.01)

        // Punctual: a morph slides a cube 2 m along X under a spot light, moving its shadow across.
        let lighting = try PunctualLightingFixture()
        let cube = MeshGeometry.cube()
        let slide = try MeshMorphTargets(vertexCount: cube.vertices.count, targets: [
            .init(positionOffsets: Array(repeating: SIMD3(4, 0, 0), count: cube.vertices.count)),
        ])
        let mesh = try MeshRenderer(
            device: lighting.device, colorPixelFormat: .rgba16Float, geometry: cube, morphTargets: slide
        )
        let floor = SceneNode(mesh: try lighting.floor(halfExtent: 4))
        let spot = SceneNode(
            transform: PunctualLightingFixture.aim(SIMD3(0, 3, 0), at: .zero),
            light: try PunctualLight(
                kind: .spot(innerConeAngle: 0.7, outerConeAngle: 0.87), intensity: 20, shadow: PunctualLight.Shadow()
            )
        )
        let maps = try PunctualShadowMaps(queue: lighting.queue, resolution: 512, spotCapacity: 1, pointCapacity: 0)
        let unshadowed = try lighting.render([floor, spot], maps: maps)
        for (weight, dark, bright) in [(Float(0), Float(-2.2), Float(2.2)), (1, 2.2, -2.2)] {
            let occluder = SceneNode(
                transform: SceneTransform(translation: SIMD3(-1, 1.5, 0), scale: SIMD3(repeating: 0.5)),
                mesh: mesh, morphWeights: [weight]
            )
            let pixels = try lighting.render([floor, spot, occluder], maps: maps)
            let shadowed = lighting.pixel(dark, 0)
            let lit = lighting.pixel(bright, 0)
            #expect(unshadowed[shadowed].x > 0.2)
            #expect(pixels[shadowed].x < 0.02 * unshadowed[shadowed].x, "weight \(weight)")
            #expect(abs(pixels[lit].x - unshadowed[lit].x) <= 0.005 * unshadowed[lit].x, "weight \(weight)")
        }
    }

    @Test("Imported glTF morph targets render, follow weights channels, and take mixer weights per node")
    func preparedGLTF() throws {
        var asset = MorphQuadAsset()
        let scene = try GLTFMeshLoader.loadScene(data: try asset.document(), includeAnimations: true, includeMorphTargets: true)
        let prepared = try PreparedGLTFScene(device: fixture.device, colorPixelFormat: .rgba16Float, scene: scene)
        let id = try #require(prepared.nodeIDs[0])
        #expect(prepared.restMorphWeights == [id: [0]])

        func coverage(_ graph: SceneGraph) throws -> [Bool] {
            let pixels = try fixture.render(graph).pixels
            return [Float(0), 0.5, 1].map { fixture.pixel(pixels, x: fixture.column(x: $0, depth: 2), y: 32).w == 1 }
        }
        let rest = try prepared.graph(camera: fixture.camera)
        #expect(rest.roots.first?.morphWeights == [0])
        #expect(try coverage(rest) == [true, false, false])
        // The clip's weights channel reaches 1 at one second, sliding the quad to x = 1.
        #expect(try coverage(prepared.graph(camera: fixture.camera, animationIndex: 0, time: 1)) == [false, false, true])

        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: try AnimationClip(scene.animations[0], nodeIDs: prepared.nodeIDs))
        try mixer.play(action)
        _ = try mixer.update(deltaTime: 0.5)
        var weights: [UUID: [Float]] = [:]
        for (node, restWeights) in prepared.restMorphWeights {
            weights[node] = try mixer.morphWeights(for: node, rest: restWeights)
        }
        #expect(weights[id] == [0.5])
        let mixed = try prepared.applyingMorphWeights(weights, to: try mixer.apply(to: rest))
        #expect(try coverage(mixed) == [false, true, false])
        #expect(try mixed.pick(at: SIMD2(0.5 / 2 * 0.5 + 0.5, 0.5), aspectRatio: 1)?.nodeID == id)

        let unknown = UUID()
        #expect(throws: PreparedGLTFScene.Failure.missingMorphNode(unknown)) {
            try prepared.applyingMorphWeights([unknown: [1]], to: rest)
        }
        #expect(throws: MeshMorphTargets.ValidationError.weightCount(expected: 1, actual: 2)) {
            try prepared.applyingMorphWeights([id: [1, 0]], to: rest)
        }
        #expect(throws: PreparedGLTFScene.Failure.missingMorphNode(id)) {
            try prepared.applyingMorphWeights([id: [1]], to: SceneGraph(roots: [], camera: fixture.camera))
        }
    }

    @Test("Morph targets and weights are validated before preparing or encoding")
    func validation() throws {
        let device = fixture.device
        let three = try MeshMorphTargets(vertexCount: 3, targets: [.init(positionOffsets: Array(repeating: .zero, count: 3))])
        #expect(throws: MeshMorphTargets.ValidationError.geometryVertexCount(expected: 3, actual: 4)) {
            try fixture.unlit(base, morphTargets: three)
        }
        let many = try MeshMorphTargets(vertexCount: 4, targets: Array(
            repeating: .init(positionOffsets: Array(repeating: .zero, count: 4)), count: MeshRenderer.maximumMorphTargetCount + 1
        ))
        #expect(throws: MeshRenderer.Failure.morphTargetCount) { try fixture.unlit(base, morphTargets: many) }
        let untangented = try MeshGeometry(
            vertices: base.vertices.map { MeshVertex(position: $0.position, normal: $0.normal) }, indices: base.indices
        )
        #expect(throws: MeshMorphTargets.ValidationError.missingBaseTangents) {
            try fixture.unlit(untangented, morphTargets: targets)
        }
        let lift = try SurfaceShaderGraph { s in .init(positionOffset: s.constant(SIMD3<Float>(0, 0, 0.1))) }
        #expect(throws: MeshRenderer.Failure.morphedDeformation) {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: base,
                material: .physicallyBased(try PhysicallyBasedMaterial(alphaMode: .opaque, surfaceGraph: lift)),
                morphTargets: targets
            )
        }

        let morphed = try fixture.unlit(base, morphTargets: targets)
        let plain = try fixture.unlit(base)
        func graph(_ mesh: MeshRenderer, _ weights: [Float]?) -> SceneGraph {
            SceneGraph(
                roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh, morphWeights: weights)],
                camera: fixture.camera
            )
        }
        #expect(throws: MeshMorphTargets.ValidationError.weightCount(expected: 2, actual: 1)) {
            try fixture.render(graph(morphed, [1]))
        }
        #expect(throws: MeshMorphTargets.ValidationError.nonfiniteWeight) {
            try fixture.render(graph(morphed, [.nan, 0]))
        }
        #expect(throws: MeshRenderer.Failure.unexpectedMorphWeights) { try fixture.render(graph(plain, [1])) }
        #expect(throws: MeshMorphTargets.ValidationError.weightCount(expected: 2, actual: 1)) {
            try graph(morphed, [1]).pick(at: SIMD2(0.5, 0.5), aspectRatio: 1)
        }
        // A single-mesh scene takes weights too.
        let target = try fixture.makeTarget()
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        var single = MeshScene(camera: fixture.camera, transform: SceneTransform(translation: SIMD3(0, 0, -2)))
        single.morphWeights = [1, 0]
        try morphed.encode(single, pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer)
        try fixture.finish(buffer)
        let pixels = try fixture.pixels(of: target.color)
        #expect(fixture.pixel(pixels, x: fixture.column(x: 0.6, depth: 2), y: 32).w == 1)
    }

    private func tangentDirection(_ vertex: MeshVertex) -> SIMD3<Float> {
        let tangent = vertex.tangent ?? .zero
        return SIMD3(tangent.x, tangent.y, tangent.z)
    }

    /// Writes the unit normal, or the tangent direction when parameter x is one.
    private static let frameSource = """
    #include <metal_stdlib>
    using namespace metal;
    struct FrameInput {
        float4 position [[position]];
        float3 normal [[user(frostfield_normal)]];
        float4 tangent [[user(frostfield_tangent)]];
    };
    fragment float4 frostfield_test_frame(FrameInput input [[stage_in]], constant float4& selector [[buffer(0)]]) {
        float3 value = selector.x > 0.5 ? input.tangent.xyz : normalize(input.normal);
        return float4(value, selector.w);
    }
    """
}

/// A glTF quad at z = −2 whose single morph target slides it 1 m along +X, with a
/// weights clip that goes from 0 to 1 over one second.
private struct MorphQuadAsset {
    private var buffer = Data()
    private var views: [[String: Int]] = []
    private var accessors: [[String: Any]] = []

    mutating func document() throws -> Data {
        let h: Float = 0.25
        let corners: [SIMD3<Float>] = [
            SIMD3(-h, -h, 0), SIMD3(h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, h, 0),
        ]
        let positions = floats(corners.flatMap { [$0.x, $0.y, $0.z] }, type: "VEC3", count: 6)
        let normals = floats(Array(repeating: [Float(0), 0, 1], count: 6).flatMap { $0 }, type: "VEC3", count: 6)
        let slide = floats(Array(repeating: [Float(1), 0, 0], count: 6).flatMap { $0 }, type: "VEC3", count: 6)
        let times = floats([0, 1], type: "SCALAR", count: 2, bounds: [0, 1])
        let values = floats([0, 1], type: "SCALAR", count: 2)
        let document: [String: Any] = [
            "asset": ["version": "2.0"], "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0, "translation": [0, 0, -2]]],
            "meshes": [[
                "primitives": [["attributes": ["POSITION": positions, "NORMAL": normals], "targets": [["POSITION": slide]]]],
                "weights": [0],
            ]],
            "animations": [[
                "channels": [["target": ["node": 0, "path": "weights"], "sampler": 0]],
                "samplers": [["input": times, "output": values, "interpolation": "LINEAR"]],
            ]],
            "buffers": [["byteLength": buffer.count, "uri": "data:application/octet-stream;base64,\(buffer.base64EncodedString())"]],
            "bufferViews": views, "accessors": accessors,
        ]
        return try JSONSerialization.data(withJSONObject: document)
    }

    private mutating func floats(_ values: [Float], type: String, count: Int, bounds: [Float]? = nil) -> Int {
        while !buffer.count.isMultiple(of: 4) { buffer.append(0) }
        let bytes = values.flatMap { value in withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) } }
        views.append(["buffer": 0, "byteOffset": buffer.count, "byteLength": bytes.count])
        buffer.append(contentsOf: bytes)
        var accessor: [String: Any] = ["bufferView": views.count - 1, "componentType": 5126, "type": type, "count": count]
        if let bounds {
            accessor["min"] = [bounds[0]]
            accessor["max"] = [bounds[1]]
        }
        accessors.append(accessor)
        return accessors.count - 1
    }
}
