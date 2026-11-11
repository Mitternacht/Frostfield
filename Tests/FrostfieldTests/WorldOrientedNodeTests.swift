import Frostfield
import Metal
import Testing
import simd

struct WorldOrientedNodeTests {
    @Test("Mirrored scales, local pivot and nonuniform parent match Three.js r187 Gyroscope")
    func reference() throws {
        let child = SceneNode(transform: .init(translation: SIMD3(0.5, 0, -0.2)))
        let oriented = SceneNode(
            transform: .init(
                translation: SIMD3(0.3, -0.7, 0.4),
                rotation: simd_quatf(vector: SIMD4(0.051274211, 0.214523675, -0.198232796, 0.955015344)),
                scale: SIMD3(1.1, -0.8, 1.3)
            ),
            children: [child], pivot: SIMD3(0.1, 0.2, -0.15), orientationMode: .world
        )
        let parent = SceneNode(
            transform: .init(
                translation: SIMD3(2, -1, -4),
                rotation: simd_quatf(vector: SIMD4(0.066236133, -0.201315249, 0.267998680, 0.939820011)),
                scale: SIMD3(-2, 3, 0.75)
            ), children: [oriented]
        )
        var graph = try scene([parent])
        let expected = simd_float4x4(columns: (
            SIMD4(-1.8246328, 0.7846009, 0.9461791, 0),
            SIMD4(-0.9175638, -2.0982602, -0.0295084, 0),
            SIMD4(0.5917441, -0.2780589, 1.3717075, 0),
            SIMD4(1.7385443, -2.4007148, -4.0644250, 1)
        ))
        let resolved = try graph.resolve()
        expectClose(resolved[1].worldTransform, expected)
        try expectClose(graph.worldTransform(of: oriented.id), expected)
        let childPosition = SIMD3<Float>(0.7078791, -1.9528026, -3.8656770)
        #expect(simd_distance(try graph.worldPosition(of: child.id), childPosition) < 0.000_01)
        #expect(simd_distance(try graph.worldToLocal(childPosition, in: oriented.id), child.transform.translation) < 0.000_01)
        let skeleton = try #require(try LineGeometry.skeleton(in: graph, joints: [oriented.id, child.id]))
        #expect(simd_distance(skeleton.positions[0], childPosition) < 0.000_01)

        graph.roots[0].transform.scale.x = 2
        var mirrored = expected
        mirrored.columns.0 = -mirrored.columns.0
        mirrored.columns.3 = SIMD4(3.1303806, -1.5442612, -3.3213755, 1)
        try expectClose(graph.worldTransform(of: oriented.id), mirrored)
        #expect(try simd_determinant(graph.worldTransform(of: oriented.id)) < 0)

        let root = try scene([oriented])
        #expect(try root.worldTransform(of: oriented.id) == oriented.localTransformMatrix)
    }

    @Test("World-preserving attachment solves orientation, scale and pivot under a reflected parent")
    func reparent() throws {
        let child = SceneNode(transform: .init(translation: SIMD3(0.25, 0.5, 0)))
        let node = SceneNode(
            transform: .init(
                translation: SIMD3(0.6, -0.3, 0.7), rotation: simd_quatf(angle: 0.7, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(1.1, -0.8, 1.4)
            ), children: [child], pivot: SIMD3(0.1, 0.2, 0.3), orientationMode: .world
        )
        let originalParent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -5), rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 0, 1)),
                             scale: SIMD3(1.5, 0.75, 1.2)), children: [node]
        )
        let newParent = SceneNode(
            transform: .init(translation: SIMD3(0.3, 0.7, -1), rotation: simd_quatf(angle: 1, axis: SIMD3(0, 1, 0)),
                             scale: SIMD3(-0.8, 1.1, 2))
        )
        var graph = try scene([originalParent, newParent])
        let world = try graph.worldTransform(of: node.id), worldChild = try graph.worldTransform(of: child.id)
        try graph.attach(node.id, to: newParent.id)
        #expect(graph.parentID(of: node.id) == newParent.id)
        #expect(graph.node(withID: node.id)?.pivot == node.pivot)
        #expect(graph.node(withID: node.id)?.orientationMode == .world)
        try expectClose(graph.worldTransform(of: node.id), world)
        try expectClose(graph.worldTransform(of: child.id), worldChild)
        try graph.attach(node.id, to: nil)
        try expectClose(graph.worldTransform(of: node.id), world)
        try expectClose(graph.worldTransform(of: child.id), worldChild)
    }

    @Test("Look-at and a local gizmo use the child’s world orientation")
    func tools() throws {
        let node = SceneNode(transform: .init(translation: SIMD3(0.3, 0, 0)), orientationMode: .world)
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -5), rotation: simd_quatf(angle: 0.6, axis: SIMD3(0, 0, 1)),
                             scale: SIMD3(2, 1, 0.75)), children: [node]
        )
        var graph = try scene([parent])
        let eye = try graph.worldPosition(of: node.id)
        try graph.lookAt(node.id, worldTarget: eye + SIMD3(2, 0, 0), forward: .positiveZ)
        let basis = try graph.worldTransform(of: node.id).columns.2
        #expect(simd_distance(simd_normalize(xyz(basis)), SIMD3(1, 0, 0)) < 0.000_01)
        let gizmo = try TransformGizmo(space: .local)
        let frame = try #require(try gizmo.frame(for: node.id, in: graph, drawableHeight: 64))
        #expect(simd_distance(frame.position, eye) < 0.000_01)
        #expect(simd_distance(frame.rotation.act(SIMD3(0, 0, 1)), SIMD3(1, 0, 0)) < 0.000_01)
    }

    @Test("Raster, world picking and path refits share the moved but world-oriented pose",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func rendering() throws {
        let gpu = try PathTestFixture()
        let size = 64
        let glow = SIMD3<Float>(0.75, 0.25, 0.5)
        let mesh = try gpu.pbr(.plane(width: 1.4, height: 0.36), color: .zero, emission: glow)
        let child = SceneNode(transform: .init(translation: SIMD3(0.45, 0, 0)), mesh: mesh, orientationMode: .world)
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4), scale: SIMD3(1.2, 0.8, 1)), children: [child]
        )
        var graph = SceneGraph(roots: [parent], camera: gpu.camera)
        let path = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: gpu.queue, scene: path)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 11)
        func trace() throws -> [SIMD4<Float>] {
            try gpu.trace(tracer, width: size, height: size, samples: 1, settings: settings)
        }
        func ray(_ point: SIMD3<Float>) throws -> SceneRay {
            try SceneRay(origin: SIMD3(point.x, point.y, 0), direction: SIMD3(0, 0, -1))
        }
        let projection = try gpu.camera.projection(aspectRatio: 1)
        func pixel(_ point: SIMD3<Float>) -> Int {
            let clip = projection * SIMD4(point, 1)
            let x = Int((0.5 + 0.5 * clip.x / clip.w) * Float(size))
            let y = Int((0.5 - 0.5 * clip.y / clip.w) * Float(size))
            return y * size + x
        }

        let original = SIMD3<Float>(0.54, 0, -4)
        let movedHorizontal = SIMD3<Float>(0.58, 0.54, -4)
        let missedVertical = SIMD3<Float>(0, 0.96, -4)
        let beforeRaster = try gpu.rasterize(graph, width: size, height: size)
        let beforePath = try trace()
        #expect(try graph.pick(ray: ray(original))?.nodeID == child.id)
        #expect(try graph.pick(ray: ray(movedHorizontal)) == nil)

        graph.roots[0].transform.rotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))
        #expect(try graph.pick(ray: ray(movedHorizontal))?.nodeID == child.id)
        #expect(try graph.pick(ray: ray(missedVertical)) == nil)
        #expect(try graph.pick(ray: ray(original)) == nil)
        #expect(try path.updateTransforms(graph))
        #expect(path.statistics.bottomLevelBuildCount == 1)
        let bounds = try #require(try graph.worldBounds(of: child.id))
        #expect(abs(bounds.maximum.x - bounds.minimum.x - 1.68) < 0.000_01)
        #expect(abs(bounds.maximum.y - bounds.minimum.y - 0.288) < 0.000_01)
        let afterRaster = try gpu.rasterize(graph, width: size, height: size)
        let afterPath = try trace()
        for (before, after) in [(beforeRaster, afterRaster), (beforePath, afterPath)] {
            #expect(before[pixel(original)].x == glow.x)
            #expect(before[pixel(movedHorizontal)].x == 0)
            #expect(after[pixel(movedHorizontal)].x == glow.x)
            #expect(after[pixel(missedVertical)].x == 0)
            #expect(after[pixel(original)].x == 0)
        }
    }

    @Test("Interchange exporters and CCD reject a world-oriented binding they cannot preserve")
    func unsupportedBindings() throws {
        let tip = SceneNode(transform: .init(translation: SIMD3(1, 0, 0)))
        let node = SceneNode(children: [tip], orientationMode: .world)
        let graph = try scene([SceneNode(children: [node])])
        let ik = try CCDIKChain(effectorID: tip.id, links: [.init(nodeID: node.id)])
        #expect(throws: CCDIKChain.Failure.worldOrientedNode(node.id)) {
            try ik.solve(in: graph, target: SIMD3(0, 1, 0))
        }
        #expect(throws: GLTFExporter.Failure.invalidData("Node \(node.id) uses world orientation, which glTF cannot retain.")) {
            try GLTFExporter().export(graph)
        }
        #expect(throws: USDZExporter.Failure.invalidData("Node \(node.id) uses world orientation, which USDZ export cannot retain.")) {
            try USDZExporter().export(graph)
        }
    }

    private func scene(_ roots: [SceneNode]) throws -> SceneGraph {
        SceneGraph(roots: roots, camera: try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20))
    }

    private func xyz(_ value: SIMD4<Float>) -> SIMD3<Float> { SIMD3(value.x, value.y, value.z) }

    private func expectClose(_ a: simd_float4x4, _ b: simd_float4x4) {
        for column in 0..<4 { #expect(simd_distance(a[column], b[column]) < 0.000_02) }
    }
}
