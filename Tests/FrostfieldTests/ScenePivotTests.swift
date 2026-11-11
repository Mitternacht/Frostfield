import Foundation
import Frostfield
import Metal
import ModelIO
import Testing
import simd

struct ScenePivotTests {
    private let z = SIMD3<Float>(0, 0, 1)
    private var camera: PerspectiveCamera {
        get throws { try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20) }
    }

    @Test("Local scale, rotation and child coordinates share the pivoted transform")
    func composition() throws {
        let child = SceneNode(transform: .init(translation: SIMD3(1, 1, 0)))
        let pivot = SIMD3<Float>(1, 0, 0)
        let node = SceneNode(transform: .init(
            translation: SIMD3(2, 1, -3), rotation: simd_quatf(angle: .pi / 2, axis: z), scale: SIMD3(2, 3, 1)
        ), children: [child], pivot: pivot)
        let parent = SceneNode(
            transform: .init(translation: SIMD3(10, 0, 0), scale: SIMD3(2, 1, 1)), children: [node],
            pivot: SIMD3(-1, 0, 0)
        )
        let graph = SceneGraph(roots: [parent], camera: try camera)
        #expect(close(try graph.worldPosition(of: node.id), SIMD3(17, -1, -3)))
        #expect(close(try graph.worldPosition(of: child.id), SIMD3(11, 1, -3)))
        #expect(close(try graph.localToWorld(pivot, in: node.id), SIMD3(17, 1, -3)))
        #expect(close(try graph.worldToLocal(SIMD3(11, 1, -3), in: node.id), SIMD3(1, 1, 0)))
        #expect(try graph.resolve()[2].worldTransform == graph.worldTransform(of: child.id))

        var zero = node
        zero.pivot = .zero
        #expect(zero.localTransformMatrix == zero.transform.matrix)
        zero.pivot = SIMD3(.infinity, 0, 0)
        let invalid = SceneGraph(roots: [zero], camera: try camera)
        #expect(throws: SceneGraph.ValidationError.invalidLocalTransform(zero.id)) { try invalid.resolve() }
    }

    @Test("Attach preserves a pivoted subtree's world pose and its editable pivot")
    func attach() throws {
        let child = SceneNode(transform: .init(translation: SIMD3(0, 2, 1)))
        let node = SceneNode(
            transform: .init(translation: SIMD3(2, 1, -3), rotation: simd_quatf(angle: 0.8, axis: z), scale: SIMD3(-2, 3, 1)),
            children: [child], pivot: SIMD3(-0.4, 0.8, 0.6)
        )
        let target = SceneNode(
            transform: .init(translation: SIMD3(-1, 0, -1), rotation: simd_quatf(angle: 0.7, axis: z), scale: SIMD3(repeating: 2)),
            pivot: SIMD3(0.3, -0.4, 0)
        )
        var graph = SceneGraph(roots: [node, target], camera: try camera)
        let before = try graph.worldTransform(of: node.id), beforeChild = try graph.worldTransform(of: child.id)
        try graph.attach(node.id, to: target.id)
        #expect(graph.node(withID: node.id)?.pivot == node.pivot)
        #expect(graph.parentID(of: node.id) == target.id)
        expectClose(try graph.worldTransform(of: node.id), before)
        expectClose(try graph.worldTransform(of: child.id), beforeChild)
    }

    @Test("Look-at uses the current object origin and keeps the authored translation and pivot")
    func lookAt() throws {
        let pivot = SIMD3<Float>(1, 0, 0)
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4), rotation: simd_quatf(angle: .pi / 2, axis: z)),
            pivot: pivot
        )
        var graph = SceneGraph(roots: [node], camera: try camera)
        let anchor = try graph.localToWorld(pivot, in: node.id)
        let before = try graph.worldPosition(of: node.id)
        let right = SIMD3<Float>(1, 0, 0)
        try graph.lookAt(node.id, worldTarget: before + right * 2, forward: .positiveZ)
        let result = try #require(graph.node(withID: node.id))
        #expect(close(result.transform.rotation.act(z), right))
        #expect(result.transform.translation == node.transform.translation)
        #expect(result.pivot == pivot)
        #expect(close(try graph.localToWorld(pivot, in: node.id), anchor))
    }

    @Test("Gizmo rotation and CCD joints turn around the authored pivot")
    func tools() throws {
        let tip = SceneNode(transform: .init(translation: SIMD3(2, 0, 0)))
        let pivot = SIMD3<Float>(1, 0, 0)
        let arm = SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), children: [tip], pivot: pivot)
        var graph = SceneGraph(roots: [arm], camera: try camera)
        let gizmo = try TransformGizmo(mode: .rotate)
        let frame = try #require(try gizmo.frame(for: arm.id, in: graph, drawableHeight: 600))
        let anchor = SIMD3<Float>(1, 0, -5)
        #expect(close(frame.position, anchor))
        let start = try SceneRay(origin: anchor + SIMD3(frame.size, 0, 2), direction: -z)
        var drag = try #require(frame.beginDrag(.axis(.z), ray: start))
        let end = try SceneRay(origin: anchor + SIMD3(0, frame.size, 2), direction: -z)
        #expect(try drag.update(ray: end, in: &graph))
        let target = SIMD3<Float>(1, 1, -5)
        #expect(close(try graph.worldPosition(of: tip.id), target))
        #expect(close(try graph.localToWorld(pivot, in: arm.id), anchor))
        try drag.cancel(in: &graph)
        let chain = try CCDIKChain(effectorID: tip.id, links: [.init(nodeID: arm.id)], iterations: 20, tolerance: 0.000_01)
        let solved = try chain.solve(in: graph, target: target)
        #expect(solved.distance < 0.000_1)
        #expect(close(try solved.graph.worldPosition(of: tip.id), target))
    }

    @Test("Scene archives preserve pivots and keep the old schema for pivot-free graphs",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func archive() throws {
        let gpu = try PathTestFixture()
        let node = SceneNode(
            transform: .init(rotation: simd_quatf(angle: 0.5, axis: z), scale: SIMD3(2, 1, 1)), pivot: SIMD3(1, 2, 3)
        )
        var graph = SceneGraph(roots: [node], camera: try camera)
        let captured = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        let manifest = try #require(JSONSerialization.jsonObject(with: captured.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 8)
        let restored = try captured.load(device: gpu.device)
        #expect(restored.graph.roots[0].pivot == node.pivot)
        #expect(restored.graph.roots[0].transform == node.transform)
        try expectClose(restored.graph.worldTransform(of: node.id), graph.worldTransform(of: node.id))

        var invalid = manifest
        invalid["version"] = 7
        #expect(throws: SceneArchive.Failure.decoding("node pivots require archive version 8")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: invalid), binary: captured.binary)
        }

        graph.roots[0].pivot = nil
        let previous = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        let oldManifest = try #require(JSONSerialization.jsonObject(with: previous.manifest) as? [String: Any])
        #expect(oldManifest["version"] as? Int == 1)
        #expect(try previous.load(device: gpu.device).graph.roots[0].pivot == nil)
    }

    @Test("glTF represents animated pivots with portable TRS nodes and preserves in-between poses",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func gltfAnimation() throws {
        let gpu = try PathTestFixture()
        let mesh = try gpu.pbr(.cube())
        let child = SceneNode(transform: .init(translation: SIMD3(0, 0.8, 0.2)), name: "Child")
        let node = SceneNode(
            transform: .init(translation: SIMD3(0.1, -0.3, -4), scale: SIMD3(1, 2, 1)),
            mesh: mesh, children: [child], name: "Hinge", pivot: SIMD3(1, 0.25, 0)
        )
        let graph = SceneGraph(roots: [node], camera: try camera)
        let track = try SceneAnimationTrack(nodeID: node.id, keyframes: [
            .init(time: 0, transform: node.transform),
            .init(time: 1, transform: .init(
                translation: SIMD3(-0.2, 0.7, -4), rotation: simd_quatf(angle: .pi * 0.8, axis: z), scale: SIMD3(2, 1, 1)
            )),
        ])
        let animation = try SceneAnimation(tracks: [track])
        let output = try GLTFExporter(commandQueue: gpu.queue).export(graph, animations: [.init(animation: animation)])
        let imported = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true, includeAnimations: true)
        let index = try #require(output.nodeIndices[node.id])
        let container = try #require(imported.nodes.first { $0.sourceIndex == index })
        let payloadIndex = try #require(container.childIndices.first)
        let payload = try #require(imported.nodes.first { $0.sourceIndex == payloadIndex })
        #expect(container.meshIndex == nil && payload.meshIndex != nil)
        let pivot = try #require(node.pivot)
        let childIndex = try #require(output.nodeIndices[child.id])
        #expect(payload.localTransform.translation == -pivot)
        #expect(payload.childIndices == [childIndex])
        #expect(imported.animations.first?.channels.allSatisfy { $0.sourceNodeIndex == index } == true)

        let prepared = try PreparedGLTFScene(device: gpu.device, colorPixelFormat: .rgba16Float, scene: imported)
        for time: Float in [0, 0.27, 0.61, 1] {
            let expected = try animation.applying(to: graph, at: time)
            let actual = try prepared.graph(camera: camera, animationIndex: 0, time: time)
            let hinge = try #require(actual.node(named: "Hinge"))
            let importedChild = try #require(actual.node(named: "Child"))
            let expectedNode = try #require(expected.node(withID: node.id))
            #expect(expectedNode.pivot == node.pivot)
            try expectClose(actual.worldTransform(of: hinge.children[0].id), expected.worldTransform(of: node.id))
            try expectClose(actual.worldTransform(of: importedChild.id), expected.worldTransform(of: child.id))
        }
    }

    @Test("USDZ preserves static and animated pivot transform stacks through ModelIO",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"), arguments: [false, true])
    func usdz(animated: Bool) throws {
        let gpu = try PathTestFixture()
        let node = SceneNode(
            transform: .init(translation: SIMD3(-0.3, 0.5, -4), rotation: simd_quatf(angle: 0.6, axis: z),
                             scale: SIMD3(1, 2, 1)),
            mesh: try gpu.pbr(.cube()), pivot: SIMD3(1, 0.25, 0.125)
        )
        let graph = SceneGraph(roots: [node], camera: try camera)
        let clip = try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: node.id, keyframes: [
            .init(time: 0, transform: node.transform),
            .init(time: 1, transform: .init(
                translation: SIMD3(-0.2, 0.7, -4), rotation: simd_quatf(angle: 1.4, axis: z), scale: SIMD3(2, 1, 1)
            )),
        ])])
        let output = try USDZExporter(commandQueue: gpu.queue).export(
            graph, animation: animated ? clip : nil, includesCamera: false
        )
        #expect(output.rootLayer.contains("xformOp:translate:pivot = (1, 0.25, 0.125)"))
        #expect(output.rootLayer.contains(
            #"xformOpOrder = ["xformOp:translate", "xformOp:translate:pivot", "xformOp:orient", "xformOp:scale", "!invert!xformOp:translate:pivot"]"#
        ))
        let path = try #require(output.nodePaths[node.id])
        let asset = MDLAsset(url: try temporaryFile("pivot.usdz", output.data))
        let object = try #require(flatten(asset)[path])
        for time: Float in animated ? [0, 1] : [0] {
            let posed = animated ? try clip.applying(to: graph, at: time) : graph
            let matrix = try #require(object.transform?.localTransform?(atTime: Double(time)))
            try expectClose(matrix, posed.worldTransform(of: node.id))
        }
    }

    @Test("An authored pivot moves raster, picking and a refit path-tracing scene together",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func rendering() throws {
        let gpu = try PathTestFixture()
        let size = 48, y = 24, left = 16, right = 31
        let radiance = SIMD3<Float>(0.75, 0.25, 0.5)
        let mesh = try gpu.pbr(.plane(width: 0.68, height: 0.68), color: .zero, emission: radiance)
        let node = SceneNode(transform: .init(translation: SIMD3(-0.55, 0, -3)), mesh: mesh, pivot: SIMD3(0.55, 0, 0))
        var graph = SceneGraph(roots: [node], camera: gpu.camera)
        let path = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: gpu.queue, scene: path)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 11)
        let leftRay = try SceneRay(origin: SIMD3(-0.55, 0, 0), direction: -z)
        let rightRay = try SceneRay(origin: SIMD3(0.55, 0, 0), direction: -z)
        func trace() throws -> [SIMD4<Float>] {
            try gpu.trace(tracer, width: size, height: size, samples: 1, settings: settings)
        }
        let rasterBefore = try gpu.rasterize(graph, width: size, height: size)
        let pathBefore = try trace()
        #expect(try graph.pick(ray: leftRay)?.nodeID == node.id)
        #expect(try graph.pick(ray: rightRay) == nil)
        graph.roots[0].transform.rotation = simd_quatf(angle: .pi, axis: z)
        #expect(try graph.pick(ray: rightRay)?.nodeID == node.id)
        #expect(try graph.pick(ray: leftRay) == nil)
        #expect(try path.updateTransforms(graph))
        #expect(path.statistics.bottomLevelBuildCount == 1)
        let rasterAfter = try gpu.rasterize(graph, width: size, height: size)
        let pathAfter = try trace()
        for (before, after) in [(rasterBefore, rasterAfter), (pathBefore, pathAfter)] {
            #expect(before[y * size + left].x == radiance.x)
            #expect(before[y * size + right].x == 0)
            #expect(after[y * size + left].x == 0)
            #expect(after[y * size + right].x == radiance.x)
        }
    }

    private func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Bool {
        simd_distance(a, b) < 0.000_1
    }

    private func expectClose(_ a: simd_float4x4, _ b: simd_float4x4) {
        for column in 0..<4 { #expect(simd_distance(a[column], b[column]) < 0.000_1) }
    }
}
