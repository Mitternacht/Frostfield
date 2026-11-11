import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct AnimationPathHelperTests {
    private let gpu: RasterLightingFixture

    init() throws {
        gpu = try RasterLightingFixture(width: 64, height: 64)
    }

    @Test("A cubic translation keeps its curve and authored keys under a sheared parent snapshot")
    func parentSnapshot() throws {
        let target = SceneNode(transform: .init(translation: SIMD3(1, -1, 0)), name: "Target")
        let parent = SceneNode(
            transform: .init(rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1)), scale: SIMD3(1, 2, 1)),
            children: [target]
        )
        let root = SceneNode(
            transform: .init(translation: SIMD3(0.1, -0.2, -3), scale: SIMD3(2, 1, 1)),
            children: [parent], pivot: SIMD3(0.3, 0, 0)
        )
        var graph = try scene(root)
        let live = try #require(PhysicallyBasedMaterial(roughness: 0.3).withLiveFactors().liveFactors)
        let oldFactors = live.current
        let translation = try AnimationTrack(
            property: .node(target.id, .translation), times: [0, 2], values: [
                .vector3(.zero), .vector3(.zero), .vector3(SIMD3(0, 4, 0)),
                .vector3(SIMD3(0, -4, 0)), .vector3(SIMD3(2, 0, 0)), .vector3(.zero),
            ], interpolation: .cubicSpline
        )
        let clip = try AnimationClip(tracks: [
            translation,
            .vector(.node(parent.id, .translation), times: [0, 2], values: [.zero, SIMD3(20, 30, 40)]),
            .number(.material(live, .roughness), times: [0, 2], values: [0.1, 0.9]),
        ])
        let helpers = DebugHelpers(device: gpu.device)
        let helper = try helpers.animationPath(for: target.id, in: graph, clip: clip, options: .init(divisions: 4))
        let (line, markers) = try primitives(helper)
        #expect(line.geometry.positions.count == 5)
        #expect(markers?.cloud.positions.count == 2)
        let world = try graph.worldTransform(of: parent.id)
        let first = world * SIMD4<Float>(0, 0, 0, 1)
        let middle = world * SIMD4<Float>(1, 2, 0, 1)
        let end = world * SIMD4<Float>(2, 0, 0, 1)
        #expect(line.geometry.positions[0] == xyz(first))
        #expect(simd_distance(line.geometry.positions[2], xyz(middle)) < 0.000_01)
        #expect(line.geometry.positions[4] == xyz(end))
        #expect(markers?.cloud.positions == [xyz(first), xyz(end)])
        #expect(graph.roots.count == 1 && graph.node(withID: target.id)?.transform == target.transform)
        #expect(graph.node(withID: parent.id)?.transform == parent.transform)
        #expect(live.current == oldFactors)

        let legacy = try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: target.id, keyframes: [
            .init(time: 1, transform: .init(translation: SIMD3(-2, 1, -4))),
            .init(time: 3, transform: .init(translation: SIMD3(2, 1, -4))),
        ])])
        let legacyNode = try helpers.animationPath(
            for: target.id, in: graph, clip: AnimationClip(legacy), options: .init(divisions: 3)
        )
        let (legacyLine, legacyMarkers) = try primitives(legacyNode)
        // r187 AnimationPathHelper world positions with the same parent hierarchy.
        let reference: [SIMD3<Float>] = [
            SIMD3(-5.8568542, -0.2, -7), SIMD3(-5.8568542, -0.2, -7),
            SIMD3(-3.0284271, 1.2142136, -7), SIMD3(-0.2, 2.6284271, -7),
        ]
        #expect(legacyLine.geometry.positions.count == reference.count)
        for (sample, expected) in zip(legacyLine.geometry.positions, reference) {
            #expect(simd_distance(sample, expected) < 0.000_01)
        }
        #expect(legacyMarkers?.cloud.positions == [legacyLine.geometry.positions[0], legacyLine.geometry.positions[3]])

        graph.roots[0].transform.translation.x += 2
        let moved = try helpers.animationPath(for: target.id, in: graph, clip: clip, options: .init(divisions: 4))
        let replacement = try primitives(moved).0.geometry.positions
        for (before, after) in zip(line.geometry.positions, replacement) {
            #expect(simd_distance(after - before, SIMD3(2, 0, 0)) < 0.000_01)
        }
    }

    @Test("An imported glTF STEP track holds its previous value between keys")
    func gltfStep() throws {
        let node = SceneNode()
        let graph = try scene(node)
        let helpers = DebugHelpers(device: gpu.device)
        var fixture = GLTFMotionFixture()
        let times = fixture.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let values = fixture.floats([-1, 0, -4, 1, 0, -4], type: "VEC3", count: 2)
        let document = try fixture.document(["animations": [[
            "channels": [["target": ["node": 0, "path": "translation"], "sampler": 0]],
            "samplers": [["input": times, "output": values, "interpolation": "STEP"]],
        ]]])
        let imported = try GLTFMeshLoader.loadScene(data: document, includeAnimations: true)
        let source = try #require(imported.animations.first)
        let clip = try AnimationClip(source, nodeIDs: [0: node.id])
        let helper = try helpers.animationPath(
            for: node.id, in: graph, clip: clip, options: .init(divisions: 4, showsMarkers: false)
        )
        let (line, markers) = try primitives(helper)
        #expect(line.geometry.positions.map(\.x) == [-1, -1, -1, -1, 1])
        #expect(markers == nil)
    }

    @Test("Raster pixels show a line and fixed-size key markers and honor helper visibility")
    func rasterVisibility() throws {
        let hidden = SceneNode(isVisible: false)
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4), scale: SIMD3(2, 1, 1)), children: [hidden]
        )
        var graph = try scene(parent)
        let clip = try AnimationClip(tracks: [
            .vector(.node(hidden.id, .translation), times: [0, 1], values: [SIMD3(-0.5, 0, 0), SIMD3(0.5, 0, 0)]),
        ])
        let helpers = DebugHelpers(device: gpu.device, lineMaterial: .init(width: .pixels(3), blending: .opaque))
        let helper = try helpers.animationPath(
            for: hidden.id, in: graph, clip: clip, options: .init(divisions: 4, markerSize: 9)
        )
        graph.roots.append(helper)
        let visible = try gpu.pixels(of: gpu.render(graph).color)
        #expect(visible[32 * 64 + 32] == SIMD4(0, 1, 0, 1))
        #expect(visible[29 * 64 + 18] == SIMD4(1, 0, 0, 1))
        #expect(visible[29 * 64 + 46] == SIMD4(1, 0, 0, 1))
        #expect(visible[21 * 64 + 32] == .zero)

        try graph.updateNode(withID: helper.id) { $0.isVisible = false }
        let invisible = try gpu.pixels(of: gpu.render(graph).color)
        #expect(invisible.allSatisfy { $0 == .zero })
        #expect(graph.roots[0].children[0].isVisible == false)
    }

    @Test("An attached path and its key markers follow animated shear without rebuilding renderers")
    func animatedParent() throws {
        let target = SceneNode(isVisible: false)
        let axis = SIMD3<Float>(0, 0, 1)
        let start = SceneTransform(
            translation: SIMD3(-0.55, -0.2, 0), rotation: simd_quatf(angle: .pi / 6, axis: axis),
            scale: SIMD3(1.2, 1.5, 1)
        )
        let end = SceneTransform(
            translation: SIMD3(0.55, 0.2, 0), rotation: simd_quatf(angle: -.pi / 7, axis: axis),
            scale: start.scale
        )
        let parent = SceneNode(transform: start, children: [target])
        let root = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4), scale: SIMD3(1.6, 0.8, 1)), children: [parent]
        )
        var graph = try scene(root)
        let first = SIMD3<Float>(-0.38, 0, 0), last = SIMD3<Float>(0.38, 0, 0)
        let animation = try SceneAnimation(tracks: [
            SceneAnimationTrack(nodeID: target.id, keyframes: [
                .init(time: 0, transform: .init(translation: first)),
                .init(time: 1, transform: .init(translation: last)),
            ]),
            SceneAnimationTrack(nodeID: parent.id, keyframes: [
                .init(time: 0, transform: start),
                .init(time: 1, transform: end),
            ]),
        ])
        let helpers = DebugHelpers(device: gpu.device, lineMaterial: .init(width: .pixels(3), blending: .opaque))
        let id = try helpers.addAnimationPath(
            for: target.id, to: &graph, clip: AnimationClip(animation), options: .init(divisions: 4, markerSize: 9)
        )
        #expect(graph.parentID(of: id) == parent.id)
        #expect(graph.node(withID: parent.id)?.children.map(\.id) == [target.id, id])
        #expect(graph.node(withID: target.id)?.transform == target.transform)

        let (line, pointRenderer) = try primitives(try #require(graph.node(withID: id)))
        let markers = try #require(pointRenderer)
        #expect(line.geometry.positions.first == first && line.geometry.positions.last == last)
        #expect(markers.cloud.positions == [first, last])
        let matrix = try graph.worldTransform(of: parent.id)
        #expect(abs(simd_dot(xyz(matrix.columns.0), xyz(matrix.columns.1))) > 0.1)

        let projection = try graph.camera.projection(aspectRatio: 1)
        func pixelIndex(_ position: SIMD3<Float>, in pose: SceneGraph) throws -> Int {
            let world = try pose.localToWorld(position, in: parent.id)
            let clip = projection * SIMD4(world, 1)
            let x = Int((0.5 + 0.5 * clip.x / clip.w) * Float(gpu.width))
            let y = Int((0.5 - 0.5 * clip.y / clip.w) * Float(gpu.height))
            try #require((0..<gpu.width).contains(x) && (0..<gpu.height).contains(y))
            return y * gpu.width + x
        }
        let before = try gpu.pixels(of: gpu.render(graph).color)
        let oldCenter = try pixelIndex(.zero, in: graph)
        let green = SIMD4<Float>(0, 1, 0, 1), red = SIMD4<Float>(1, 0, 0, 1)
        #expect(before[oldCenter] == green)
        for key in [first, last] {
            #expect(before[try pixelIndex(key, in: graph)] == red)
        }

        var posed = try animation.applying(to: graph, at: 1)
        let (posedLine, posedMarkers) = try primitives(try #require(posed.node(withID: id)))
        #expect(posedLine === line && posedMarkers === markers)
        let after = try gpu.pixels(of: gpu.render(posed).color)
        #expect(after[oldCenter] == .zero)
        #expect(after[try pixelIndex(.zero, in: posed)] == green)
        for key in [first, last] {
            #expect(after[try pixelIndex(key, in: posed)] == red)
        }

        try posed.updateNode(withID: parent.id) { $0.isVisible = false }
        #expect(try gpu.pixels(of: gpu.render(posed).color).allSatisfy { $0 == .zero })
    }

    @Test("Missing bindings and invalid divisions leave the source scene intact")
    func validation() throws {
        let node = SceneNode()
        var graph = try scene(node)
        let clip = try AnimationClip(tracks: [
            .quaternion(.node(node.id, .rotation), times: [0], values: [simd_quatf(real: 1, imag: .zero)]),
        ])
        let helpers = DebugHelpers(device: gpu.device)
        #expect(throws: DebugHelpers.AnimationPathFailure.missingTranslationTrack(node.id)) {
            try helpers.animationPath(for: node.id, in: graph, clip: clip)
        }
        #expect(throws: DebugHelpers.AnimationPathFailure.divisions) {
            try helpers.animationPath(for: node.id, in: graph, clip: clip, options: .init(divisions: 0))
        }
        let missing = UUID()
        #expect(throws: SceneGraph.EditingError.missingNode(missing)) {
            try helpers.animationPath(for: missing, in: graph, clip: clip)
        }
        #expect(throws: DebugHelpers.AnimationPathFailure.missingTranslationTrack(node.id)) {
            try helpers.addAnimationPath(for: node.id, to: &graph, clip: clip)
        }
        let translation = try AnimationClip(tracks: [
            .vector(.node(node.id, .translation), times: [0, 1], values: [.zero, SIMD3(1, 0, 0)]),
        ])
        #expect(throws: PrimitiveRendererError.size) {
            try helpers.addAnimationPath(for: node.id, to: &graph, clip: translation, options: .init(markerSize: 0))
        }
        #expect(graph.roots.map(\.id) == [node.id] && graph.roots[0].transform == node.transform)
        let helper = try helpers.addAnimationPath(
            for: node.id, to: &graph, clip: translation, options: .init(showsMarkers: false)
        )
        #expect(graph.roots.map(\.id) == [node.id, helper])
    }

    private func scene(_ node: SceneNode) throws -> SceneGraph {
        SceneGraph(roots: [node], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20))
    }

    private func primitives(_ node: SceneNode) throws -> (LineRenderer, PointsRenderer?) {
        guard case .lines(let line) = node.children.first?.primitive else {
            throw DebugHelpers.AnimationPathFailure.missingTranslationTrack(node.id)
        }
        if node.children.count == 1 { return (line, nil) }
        guard case .points(let points) = node.children[1].primitive else {
            throw DebugHelpers.AnimationPathFailure.missingTranslationTrack(node.id)
        }
        return (line, points)
    }

    private func xyz(_ point: SIMD4<Float>) -> SIMD3<Float> { SIMD3(point.x, point.y, point.z) }
}
