import Frostfield
import Metal
import Testing
import simd

struct TransformGizmoTests {
    private func camera() throws -> PerspectiveCamera {
        try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 40)
    }

    private func ray(at point: SIMD3<Float>, toward direction: SIMD3<Float> = SIMD3(0, 0, -1)) throws -> SceneRay {
        try SceneRay(origin: point - direction * 3, direction: direction)
    }

    private func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Bool {
        simd_length(a - b) < 0.0001
    }

    @Test("World translation snaps from the starting point through a reflected nonuniform parent")
    func worldTranslation() throws {
        let z = SIMD3<Float>(0, 0, 1)
        let x = SIMD3<Float>(1, 0, 0)
        let node = SceneNode(transform: .init(translation: SIMD3(0.1, -0.2, 0.2)))
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -6), rotation: simd_quatf(angle: .pi / 2, axis: z), scale: SIMD3(-2, 3, 0.5)),
            children: [node]
        )
        var graph = SceneGraph(roots: [parent], camera: try camera())
        let gizmo = try TransformGizmo(axes: [.x], showsScreenHandle: false, snap: 0.5)
        let frame = try #require(try gizmo.frame(for: node.id, in: graph, drawableHeight: 600))
        #expect(frame.visibleHandles == [.axis(.x)])
        let pivot = try graph.worldPosition(of: node.id)
        let initial = try ray(at: pivot + x * frame.size)
        var drag = try #require(frame.beginDrag(.axis(.x), ray: initial))
        #expect(try !drag.update(ray: initial, in: &graph))
        let moving = try ray(at: pivot + x * (frame.size + 0.26))
        #expect(try drag.update(ray: moving, in: &graph))
        #expect(close(try graph.worldPosition(of: node.id), pivot + x * 0.5))
        #expect(try !drag.update(ray: moving, in: &graph))
        let parallel = try SceneRay(origin: pivot + z, direction: x)
        #expect(try !drag.update(ray: parallel, in: &graph))
        #expect(close(try graph.worldPosition(of: node.id), pivot + x * 0.5))
        try drag.cancel(in: &graph)
        #expect(!drag.isActive)
        #expect(graph.node(withID: node.id)?.transform == node.transform)
    }

    @Test("Local translation follows rotated axes instead of world axes")
    func localTranslation() throws {
        let z = SIMD3<Float>(0, 0, 1)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -5), rotation: simd_quatf(angle: .pi / 2, axis: z)))
        var graph = SceneGraph(roots: [node], camera: try camera())
        let gizmo = try TransformGizmo(space: .local)
        let frame = try #require(try gizmo.frame(for: node.id, in: graph, drawableHeight: 400))
        let y = SIMD3<Float>(0, 1, 0)
        #expect(close(frame.rotation.act(SIMD3(1, 0, 0)), y))
        var drag = try #require(frame.beginDrag(.axis(.x), ray: ray(at: frame.position + y * frame.size)))
        #expect(try drag.update(ray: ray(at: frame.position + y * (frame.size + 0.7)), in: &graph))
        #expect(close(try graph.worldPosition(of: node.id), frame.position + y * 0.7))
    }

    @Test("Plane handles translate in local space and scale only their two axes")
    func planeHandles() throws {
        let node = SceneNode(transform: .init(
            translation: SIMD3(0, 0, -5), rotation: simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1)),
            scale: SIMD3(-1.2, 2.3, 4)
        ))
        var graph = SceneGraph(roots: [node], camera: try camera())
        let handle = TransformGizmo.Handle.plane(.xy)
        let translate = try TransformGizmo(space: .local, snap: 0.25)
        let move = try #require(try translate.frame(for: node.id, in: graph, drawableHeight: 600))
        #expect(move.visibleHandles.contains(handle))
        #expect(!move.visibleHandles.contains(.plane(.yz)))
        let u = move.rotation.act(SIMD3(1, 0, 0))
        let v = move.rotation.act(SIMD3(0, 1, 0))
        let start = 0.35 * move.size
        let point = move.position + (u + v) * start
        var moving = try #require(move.beginDrag(handle, ray: ray(at: point)))
        #expect(try moving.update(ray: ray(at: point + u * 0.26 - v * 0.37), in: &graph))
        #expect(close(try graph.worldPosition(of: node.id), move.position + (u - v) * 0.25))
        try moving.cancel(in: &graph)

        let scale = try TransformGizmo(mode: .scale, snap: 0.25, snapReference: .absolute)
        let resized = try #require(try scale.frame(for: node.id, in: graph, drawableHeight: 600))
        var sizing = try #require(resized.beginDrag(handle, ray: ray(at: point)))
        let newPoint = resized.position + u * (start * 1.4) + v * (start * 0.7)
        #expect(try sizing.update(ray: ray(at: newPoint), in: &graph))
        #expect(graph.node(withID: node.id)?.transform.scale == SIMD3(-1.75, 1.5, 4))
        try sizing.cancel(in: &graph)

        var uniform = try #require(resized.beginDrag(.screen, ray: ray(at: resized.position)))
        #expect(try uniform.update(ray: ray(at: resized.position + SIMD3(0, resized.size * 0.14, 0)), in: &graph))
        #expect(close(try #require(graph.node(withID: node.id)).transform.scale, node.transform.scale * 1.125))
    }

    @Test("Absolute world snapping and parent-space bounds keep the world-axis constraint")
    func absoluteTranslationBounds() throws {
        let node = SceneNode(transform: .init(translation: SIMD3(0.08, 0.1, 0)))
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -6), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3<Float>(0, 0, 1)),
                             scale: SIMD3(2, 1, 0.5)),
            children: [node]
        )
        var graph = SceneGraph(roots: [parent], camera: try camera())
        let bounds = try AxisAlignedBox(minimum: SIMD3(-0.5, -0.2, -0.5), maximum: SIMD3(repeating: 0.5))
        let tool = try TransformGizmo(axes: [.x], snap: 0.25, snapReference: .absolute, translationBounds: bounds)
        let frame = try #require(try tool.frame(for: node.id, in: graph, drawableHeight: 600))
        let start = frame.position + SIMD3(frame.size, 0, 0)
        var drag = try #require(frame.beginDrag(.axis(.x), ray: ray(at: start)))
        #expect(try drag.update(ray: ray(at: start + SIMD3(0.29, 0, 0)), in: &graph))
        let snapped = try graph.worldPosition(of: node.id)
        #expect(close(snapped, SIMD3(0.25, frame.position.y, frame.position.z)))
        #expect(try drag.update(ray: ray(at: start + SIMD3(2, 0, 0)), in: &graph))
        let stopped = try #require(graph.node(withID: node.id)).transform.translation
        #expect(bounds.contains(stopped))
        #expect(abs(stopped.y - bounds.minimum.y) < 0.00001)
        let world = try graph.worldPosition(of: node.id)
        #expect(close(world, SIMD3(frame.position.x + 0.3 * sqrt(2), frame.position.y, frame.position.z)))
        try drag.cancel(in: &graph)
        try graph.updateNode(withID: node.id) { $0.transform.translation.y = -1 }
        #expect(throws: TransformGizmo.Failure.outsideTranslationBounds) {
            try tool.frame(for: node.id, in: graph, drawableHeight: 600)
        }
    }

    @Test("Local and world rotations use the starting axis and snap through multiple turns")
    func rotation() throws {
        let x = SIMD3<Float>(1, 0, 0), y = SIMD3<Float>(0, 1, 0), z = SIMD3<Float>(0, 0, 1)
        let node = SceneNode(transform: .init(rotation: simd_quatf(angle: 0.3, axis: x)))
        let parentRotation = simd_quatf(angle: 0.45, axis: y)
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -7), rotation: parentRotation, scale: SIMD3(repeating: 2)),
            children: [node]
        )
        for space in [TransformGizmo.Space.local, .world] {
            var graph = SceneGraph(roots: [parent], camera: try camera())
            let gizmo = try TransformGizmo(mode: .rotate, space: space, axes: [.z], snap: 0.4)
            let frame = try #require(try gizmo.frame(for: node.id, in: graph, drawableHeight: 600))
            let axis = frame.rotation.act(z)
            let arm = frame.rotation.act(x) * frame.size
            func sample(_ angle: Float) throws -> SceneRay {
                try ray(at: frame.position + simd_quatf(angle: angle, axis: axis).act(arm), toward: -axis)
            }
            var drag = try #require(frame.beginDrag(.axis(.z), ray: sample(0)))
            #expect(try !drag.update(ray: sample(0.1), in: &graph))
            #expect(try !drag.update(ray: ray(at: frame.position, toward: -axis), in: &graph))
            for angle: Float in [0.7, 2.2, 3.9, 5.5, 7.08] {
                #expect(try drag.update(ray: sample(angle), in: &graph))
            }
            let rotated = try #require(graph.node(withID: node.id)?.transform.rotation)
            let target = simd_quatf(angle: 7.2, axis: parentRotation.inverse.act(axis)) * node.transform.rotation
            #expect(close(rotated.act(x), target.act(x)))
            #expect(close(rotated.act(y), target.act(y)))
            try drag.cancel(in: &graph)
            #expect(graph.node(withID: node.id)?.transform == node.transform)
        }
    }

    @Test("View rotation uses the line to an off-center camera rather than the node axes")
    func viewRotation() throws {
        let node = SceneNode(transform: .init(rotation: simd_quatf(angle: 0.3, axis: SIMD3<Float>(0, 1, 0))))
        var view = SceneTransform(translation: SIMD3(3, 2, 6))
        try view.look(at: SIMD3(0.4, 0.2, 0))
        var graph = SceneGraph(roots: [node], camera: try camera(), cameraTransform: view)
        let tool = try TransformGizmo(mode: .rotate, space: .local, axes: [])
        let frame = try #require(try tool.frame(for: node.id, in: graph, drawableHeight: 600))
        #expect(frame.visibleHandles == [.viewRotation])
        let axis = simd_normalize(view.translation - frame.position)
        let first = simd_normalize(simd_cross(axis, SIMD3(0, 1, 0))) * frame.size
        let second = simd_cross(axis, first)
        var drag = try #require(frame.beginDrag(.viewRotation, ray: ray(at: frame.position + first, toward: -axis)))
        #expect(try drag.update(ray: ray(at: frame.position + second, toward: -axis), in: &graph))
        let actual = try #require(graph.node(withID: node.id)).transform.rotation
        let expected = simd_quatf(angle: .pi / 2, axis: axis) * node.transform.rotation
        #expect(close(actual.act(SIMD3(1, 0, 0)), expected.act(SIMD3(1, 0, 0))))
    }

    @Test("Scaling snaps one local component or all components and preserves reflections")
    func scale() throws {
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -5), scale: SIMD3(-2, 3, 4)))
        var graph = SceneGraph(roots: [node], camera: try camera())
        let gizmo = try TransformGizmo(mode: .scale, snap: 0.25)
        let frame = try #require(try gizmo.frame(for: node.id, in: graph, drawableHeight: 600))
        let x = SIMD3<Float>(1, 0, 0), y = SIMD3<Float>(0, 1, 0)
        #expect(!frame.visibleHandles.contains(.axis(.z)))
        var drag = try #require(frame.beginDrag(.axis(.x), ray: ray(at: frame.position + x * frame.size)))
        #expect(try drag.update(ray: ray(at: frame.position + x * frame.size * 1.24), in: &graph))
        #expect(graph.node(withID: node.id)?.transform.scale == SIMD3(-2.5, 3, 4))
        #expect(try drag.update(ray: ray(at: frame.position - x * frame.size), in: &graph))
        #expect(close(try #require(graph.node(withID: node.id)).transform.scale, SIMD3(-0.0002, 3, 4)))
        try drag.cancel(in: &graph)
        var uniform = try #require(frame.beginDrag(.screen, ray: ray(at: frame.position)))
        #expect(try uniform.update(ray: ray(at: frame.position + y * frame.size * 0.49), in: &graph))
        #expect(graph.node(withID: node.id)?.transform.scale == SIMD3(-3, 4.5, 6))
        uniform.finish()
        try uniform.cancel(in: &graph)
        #expect(graph.node(withID: node.id)?.transform.scale == SIMD3(-3, 4.5, 6))
    }

    @Test("Visibility, camera scaling, and stale-target failures are explicit")
    func validation() throws {
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -6)))
        let parent = SceneNode(children: [node])
        var graph = SceneGraph(roots: [parent], camera: try camera())
        let gizmo = try TransformGizmo()
        let full = try #require(try gizmo.frame(for: node.id, in: graph, drawableHeight: 600))
        let half = try #require(try gizmo.frame(for: node.id, in: graph, drawableHeight: 1200))
        #expect(abs(full.size - half.size * 2) < 0.00001)
        var drag = try #require(full.beginDrag(.screen, ray: ray(at: full.position)))
        try graph.updateNode(withID: parent.id) { $0.transform.translation.x = 1 }
        #expect(throws: TransformGizmo.Failure.staleDrag) { try drag.cancel(in: &graph) }
        #expect(throws: TransformGizmo.Failure.staleDrag) { try drag.update(ray: ray(at: full.position), in: &graph) }
        #expect(graph.node(withID: node.id)?.transform == node.transform)
        try graph.updateNode(withID: parent.id) { $0.isVisible = false }
        #expect(try gizmo.frame(for: node.id, in: graph, drawableHeight: 600) == nil)
        try graph.updateNode(withID: parent.id) {
            $0.isVisible = true
            $0.transform.scale.x = 2
        }
        let rotation = try TransformGizmo(mode: .rotate)
        #expect(throws: TransformGizmo.Failure.unsupportedParentTransform) {
            try rotation.frame(for: node.id, in: graph, drawableHeight: 600)
        }
        #expect(throws: TransformGizmo.Failure.invalidOptions) { try TransformGizmo(snap: .nan) }
    }

    @Test("Rendered arrow and ring pick and transform the same graph node",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func renderedInteraction() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let size = 128
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false)
        colorDescriptor.usage = [.renderTarget]
        colorDescriptor.storageMode = .shared
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.depthPixelFormat, width: size, height: size, mipmapped: false
        )
        depthDescriptor.usage = [.renderTarget]
        depthDescriptor.storageMode = .private
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let overlay = try TransformGizmoOverlay(device: device, colorPixelFormat: color.pixelFormat)
        let mesh = try MeshRenderer(device: device, colorPixelFormat: color.pixelFormat, geometry: .cube())
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -6), scale: SIMD3(2.5, 1.4, 1)),
            mesh: mesh, materialParameters: SIMD4(0.3, 0.3, 0.3, 1)
        )
        var graph = SceneGraph(roots: [node], camera: try camera())

        func pixel(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
            var value = SIMD4<UInt8>.zero
            color.getBytes(&value, bytesPerRow: 4, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
            return value
        }
        func draw(_ frame: TransformGizmo.Frame, highlighting: TransformGizmo.Handle? = nil) throws {
            var preview = graph
            preview.roots += overlay.nodes(for: frame, highlighting: highlighting)
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = 1
            let commands = try #require(queue.makeCommandBuffer())
            try preview.encode(pass: pass, commandBuffer: commands)
            commands.commit()
            commands.waitUntilCompleted()
            #expect(commands.status == .completed)
        }
        func cameraRay(_ x: Int, _ y: Int) throws -> SceneRay {
            try graph.camera.ray(at: (SIMD2(Float(x), Float(y)) + SIMD2(repeating: 0.5)) / Float(size), aspectRatio: 1)
        }

        let translation = try TransformGizmo(sizeInPixels: 20)
        let frame = try #require(try translation.frame(for: node.id, in: graph, drawableHeight: Float(size)))
        let handle = TransformGizmo.Handle.axis(.x)
        let pointer = try cameraRay(77, 64)
        #expect(try overlay.hitTest(pointer, in: frame) == handle)
        #expect(try overlay.hitTest(cameraRay(5, 5), in: frame) == nil)
        try draw(frame)
        let red = pixel(77, 64)
        #expect(Int(red.x) > 2 * Int(max(red.y, red.z)))
        try draw(frame, highlighting: handle)
        #expect(pixel(77, 64).y > red.y)
        #expect(overlay.nodes(for: frame).map(\.id) == overlay.nodes(for: frame, highlighting: handle).map(\.id))

        let planeHandle = TransformGizmo.Handle.plane(.xy)
        #expect(try overlay.hitTest(cameraRay(71, 57), in: frame) == planeHandle)
        try draw(frame, highlighting: planeHandle)
        let planar = pixel(71, 57)
        #expect(planar.x > planar.z && planar.y > planar.z)

        var drag = try #require(frame.beginDrag(handle, ray: pointer))
        #expect(try drag.update(ray: cameraRay(91, 64), in: &graph))
        drag.finish()
        let moved = try #require(try translation.frame(for: node.id, in: graph, drawableHeight: Float(size)))
        #expect(moved.position.x > 0.7)
        #expect(try overlay.hitTest(pointer, in: moved) == .screen)

        let centerRay = try cameraRay(77, 64)
        let point = SIMD2(77.5, 64.5) / Float(size)
        #expect(try graph.pick(at: point, aspectRatio: 1)?.nodeID == node.id)
        var direct = try #require(moved.beginDrag(.screen, ray: centerRay))
        #expect(try direct.update(ray: cameraRay(71, 60), in: &graph))
        #expect((try graph.worldPosition(of: node.id)).x < moved.position.x)
        #expect((try graph.worldPosition(of: node.id)).y > moved.position.y)
        try direct.cancel(in: &graph)

        let rotation = try TransformGizmo(mode: .rotate, axes: [.z], sizeInPixels: 20)
        let ring = try #require(try rotation.frame(for: node.id, in: graph, drawableHeight: Float(size)))
        let ringRay = try cameraRay(77, 44)
        #expect(try overlay.hitTest(ringRay, in: ring) == .axis(.z))
        try draw(ring)
        let blue = pixel(77, 44)
        #expect(Int(blue.z) > 2 * Int(blue.x))
        var turning = try #require(ring.beginDrag(.axis(.z), ray: ringRay))
        #expect(try turning.update(ray: cameraRay(58, 64), in: &graph))
        #expect(abs((try #require(graph.node(withID: node.id))).transform.rotation.angle - .pi / 2) < 0.09)
        try turning.cancel(in: &graph)

        let viewRing = try #require(try rotation.frame(for: node.id, in: graph, drawableHeight: Float(size)))
        let viewRay = try cameraRay(77, 39)
        #expect(try overlay.hitTest(viewRay, in: viewRing) == .viewRotation)
        try draw(viewRing, highlighting: .viewRotation)
        let selected = pixel(77, 39)
        #expect(selected.x > 20 && selected.y > 20)
    }
}
