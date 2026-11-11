import Frostfield
import Metal
import Testing
import simd

struct SceneDebugHelperTests {
    @Test(arguments: [MeshDepthMode.forward, .reversed])
    func `Explicit orthographic camera uses the real world origin and custom segment colors`(_ mode: MeshDepthMode) throws {
        let pose = SceneTransform(
            translation: SIMD3(3, 2, -6), rotation: simd_quatf(angle: 0.6, axis: SIMD3(0, 1, 0))
        )
        let projection = orthographic(depthMode: mode)
        let camera = try ShaderGraphCamera(viewMatrix: pose.matrix.inverse, projectionMatrix: projection, near: 1, far: 3)
        let style = LineGeometry.CameraFrustumColors(
            frustum: SIMD4(0, 1, 0, 1), cone: SIMD4(1, 0, 1, 1),
            up: SIMD4(0, 0, 1, 1), target: SIMD4(1, 1, 0, 1), cross: SIMD4(0, 1, 1, 1)
        )
        let world = try LineGeometry.cameraFrustum(camera: camera, depthMode: mode, colors: style)
        #expect(world.segments.count == 25)
        let colorGroups = Array(repeating: style.frustum, count: 12) + Array(repeating: style.cone, count: 4)
            + Array(repeating: style.up, count: 3) + [style.target] + Array(repeating: style.cross, count: 5)
        #expect(world.segments.map { world.colors?[$0.start] } == colorGroups)
        for segmentIndex in [12, 13, 14, 15, 20] {
            #expect(simd_distance(world.positions[world.segments[segmentIndex].start], pose.translation) < 1e-5)
        }
        let nearCorner = pose.matrix * SIMD4<Float>(-0.6, -0.5, -1, 1)
        let farCorner = pose.matrix * SIMD4<Float>(-0.6, -0.5, -3, 1)
        #expect(simd_distance(world.positions[0], SIMD3(nearCorner.x, nearCorner.y, nearCorner.z)) < 1e-5)
        #expect(simd_distance(world.positions[8], SIMD3(farCorner.x, farCorner.y, farCorner.z)) < 1e-5)
    }

    @Test
    func `Off-axis helper follows a real framed view and rejects infinite far bounds`() throws {
        let eye = SIMD3<Float>(1, 1, 1)
        let lower = SIMD3<Float>(-1, -1, -2)
        let right = SIMD3<Float>(3, -1, -2)
        let top = SIMD3<Float>(-1, 2, -2)
        let view = try CameraUtils.frameCorners(
            eye: eye, bottomLeft: lower, bottomRight: right, topLeft: top, near: 3, far: 8, depthMode: .reversed
        )
        let geometry = try LineGeometry.cameraFrustum(camera: view.graphCamera, depthMode: view.depthMode)
        #expect(simd_distance(geometry.positions[0], lower) < 1e-5)
        #expect(simd_distance(geometry.positions[1], right) < 1e-5)
        #expect(simd_distance(geometry.positions[5], top) < 1e-5)
        #expect(simd_distance(geometry.positions[8], eye + (lower - eye) * (8.0 / 3)) < 1e-5)
        #expect(simd_distance(geometry.positions[24], eye) < 1e-5)

        let infinite = try CameraUtils.frameCorners(
            eye: eye, bottomLeft: lower, bottomRight: right, topLeft: top, near: 3, far: .infinity
        )
        #expect(throws: PrimitiveGeometryError.invalidHelperParameter("the projection needs a finite far plane")) {
            try LineGeometry.cameraFrustum(camera: infinite.graphCamera, depthMode: infinite.depthMode)
        }
    }

    @Test
    func `Skeleton can follow direct joints or bridge transform groups and hide inherited branches`() throws {
        let leaf = SceneNode(transform: SceneTransform(translation: SIMD3(0, 1, 0)))
        let bridged = SceneNode(transform: SceneTransform(translation: SIMD3(0, -1, 0)))
        let group = SceneNode(transform: SceneTransform(translation: SIMD3(1, 0, 0)), children: [bridged])
        let hiddenLeaf = SceneNode(transform: SceneTransform(translation: SIMD3(2, 0, 0)))
        let hiddenJoint = SceneNode(children: [hiddenLeaf], isVisible: false)
        let joint = SceneNode(
            transform: SceneTransform(translation: SIMD3(1, 0, -5), scale: SIMD3(1, 2, 1)),
            children: [leaf, group, hiddenJoint]
        )
        let graph = SceneGraph(roots: [joint], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20))
        let joints = [hiddenLeaf.id, leaf.id, bridged.id, hiddenJoint.id, joint.id]
        let raw = try #require(try LineGeometry.skeleton(in: graph, joints: joints, connections: .directJointParent))
        #expect(raw.positions.count == 6)
        let direct = try #require(try LineGeometry.skeleton(
            in: graph, joints: joints, connections: .directJointParent, visibleOnly: true
        ))
        #expect(direct.positions == [SIMD3(1, 2, -5), SIMD3(1, 0, -5)])
        #expect(direct.colors == [SIMD4(0, 0, 1, 1), SIMD4(0, 1, 0, 1)])
        let bridgedGeometry = try #require(try LineGeometry.skeleton(
            in: graph, joints: joints, connections: .nearestJointAncestor, visibleOnly: true
        ))
        #expect(bridgedGeometry.positions == direct.positions + [SIMD3(2, -2, -5), SIMD3(1, 0, -5)])
    }

    @Test
    func `Hemisphere helper has twelve octahedron edges and opposing colors`() throws {
        let axis = SIMD3<Float>(3, -4, 0)
        let sky = SIMD3<Float>(1, 0.25, 0)
        let ground = SIMD3<Float>(0, 0.25, 1)
        let geometry = try LineGeometry.hemisphereLight(skyColor: sky, groundColor: ground, direction: axis, size: 2)
        #expect(geometry.segments.count == 12)
        #expect(Set(geometry.positions).count == 6)
        #expect(geometry.positions.allSatisfy { abs(simd_length($0) - 2) < 1e-6 })
        #expect(simd_distance(geometry.positions[0], simd_normalize(axis) * 2) < 1e-6)
        #expect(geometry.positions[8] == -geometry.positions[0])
        #expect(geometry.colors == Array(repeating: SIMD4(sky, 1), count: 8)
            + Array(repeating: SIMD4(ground, 1), count: 16))
        #expect(throws: PrimitiveGeometryError.invalidHelperParameter("the hemisphere helper direction must be finite and nonzero")) {
            try LineGeometry.hemisphereLight(skyColor: sky, groundColor: ground, direction: .zero)
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Hemisphere helper renders both colors in the authored direction at its world placement`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(2.5), depthTest: false))
        let origin = SIMD3<Float>(0.5, 0, -5)
        let node = try helpers.hemisphereLight(
            skyColor: SIMD3(1, 0.25, 0), groundColor: SIMD3(0, 0.25, 1),
            direction: SIMD3(1, 0, 0), origin: origin, size: 1.6
        )
        #expect(node.transform.translation == origin)
        let pixels = try fixture.render(fixture.graph([node])).pixels
        let center = Int(fixture.pixel(origin).x)
        let skyPixels = pixels.indices.filter { pixels[$0].x > 0.9 && pixels[$0].z < 0.05 }
        let groundPixels = pixels.indices.filter { pixels[$0].z > 0.9 && pixels[$0].x < 0.05 }
        #expect(skyPixels.count > 10)
        #expect(groundPixels.count > 10)
        #expect(skyPixels.contains { $0 % 128 > center + 8 })
        #expect(groundPixels.contains { $0 % 128 < center - 8 })
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `An attached frustum follows ancestor transforms and refreshes its cropped projection`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(2), depthTest: false))
        let source = SceneNode(transform: SceneTransform(translation: SIMD3(-1, 0, 0)))
        let parent = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -6), scale: SIMD3(1.25, 0.8, 1)),
            children: [source]
        )
        var graph = fixture.graph([parent])
        let projection = orthographic()
        var attachment = try helpers.addCameraFrustum(projection: projection, toCamera: source.id, in: &graph)
        let initial = try line(attachment.nodeID, in: graph)
        #expect(graph.parentID(of: attachment.nodeID) == source.id)
        let initialWorld = try graph.worldTransform(of: attachment.nodeID)
        #expect(initialWorld == (try graph.worldTransform(of: source.id)))
        let left = try fixture.render(graph).pixels

        try graph.updateNode(withID: parent.id) {
            $0.transform.translation.x = 2
            $0.transform.rotation = simd_quatf(angle: 0.1, axis: SIMD3(0, 0, 1))
        }
        let moved = try fixture.render(graph).pixels
        #expect(try line(attachment.nodeID, in: graph) === initial)
        #expect(try centerX(moved) - centerX(left) > 10)
        let changedWorld = try graph.worldTransform(of: attachment.nodeID)
        #expect(changedWorld == (try graph.worldTransform(of: source.id)))
        #expect(changedWorld != initialWorld)

        let window = try CameraViewOffset(fullWidth: 128, fullHeight: 128, x: 64, y: 0, width: 64, height: 128)
        let crop = try window.applying(to: projection)
        attachment.colors.frustum = SIMD4(0, 1, 0, 1)
        try attachment.update(projection: crop, in: &graph)
        let refreshed = try line(attachment.nodeID, in: graph)
        #expect(refreshed !== initial)
        #expect(abs(refreshed.geometry.positions[0].x) < 1e-6)
        #expect(refreshed.geometry.colors?.first == SIMD4(0, 1, 0, 1))
        let cropPixels = try fixture.render(graph).pixels
        #expect(cropPixels.contains { $0.y > 0.95 && $0.x < 0.05 && $0.z < 0.05 })
        try attachment.update(projection: crop, in: &graph)
        #expect(try line(attachment.nodeID, in: graph) === refreshed)
        let singular = simd_float4x4(columns: (.zero, .zero, .zero, .zero))
        #expect(throws: PrimitiveGeometryError.invalidHelperParameter("the projection must be finite and invertible")) {
            try attachment.update(projection: singular, in: &graph)
        }
        #expect(try line(attachment.nodeID, in: graph) === refreshed)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Skeleton refresh keeps old encoded joints and renders the new pose on its stable node`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(3)))
        let leaf = SceneNode(transform: SceneTransform(translation: SIMD3(0, 1, 0)))
        let joint = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -5)), children: [leaf])
        var graph = fixture.graph([joint])
        var attachment = try helpers.addSkeleton(joints: [joint.id, leaf.id], to: &graph, visibleOnly: true)
        let oldRenderer = try line(attachment.nodeID, in: graph)
        #expect(!oldRenderer.material.depthTest)
        let first = try fixture.base.makeTarget()
        let buffer = try #require(fixture.base.queue.makeCommandBuffer())
        try graph.encode(pass: fixture.base.pass(color: first.color, depth: first.depth), commandBuffer: buffer)

        try graph.updateNode(withID: leaf.id) { $0.transform.translation = SIMD3(1, 0, 0) }
        try attachment.update(in: &graph)
        let newRenderer = try line(attachment.nodeID, in: graph)
        #expect(newRenderer !== oldRenderer)
        #expect(newRenderer.geometry.positions == [SIMD3(1, 0, -5), SIMD3(0, 0, -5)])
        #expect(graph.parentID(of: attachment.nodeID) == nil)
        try fixture.base.finish(buffer)
        let prior = try fixture.base.pixels(of: first.color)
        let current = try fixture.render(graph).pixels
        #expect(prior[57 * 128 + 64].w > 0.9)
        #expect(current[57 * 128 + 64].w == 0)
        #expect(prior[64 * 128 + 71].w == 0)
        #expect(current[64 * 128 + 71].w > 0.9)
        #expect(current[64 * 128 + 71].y > 0.1 && current[64 * 128 + 71].z > 0.1)

        try graph.updateNode(withID: joint.id) { $0.isVisible = false }
        try attachment.update(in: &graph)
        let helper = try #require(graph.node(withID: attachment.nodeID))
        #expect(helper.primitive == nil)
        #expect(try fixture.render(graph).pixels.allSatisfy { $0.w == 0 })
        try graph.updateNode(withID: joint.id) { $0.isVisible = true }
        attachment.colors = .init(joint: SIMD4(1, 0, 0, 1), parent: SIMD4(1, 1, 0, 1))
        try attachment.update(in: &graph)
        #expect(try line(attachment.nodeID, in: graph).geometry.positions == newRenderer.geometry.positions)
        let recolored = try fixture.render(graph).pixels[64 * 128 + 71]
        #expect(recolored.x > 0.95 && recolored.y > 0.1 && recolored.z == 0)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Render-time skeleton updates follow both camera paths without changing the source or archive`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(3)))
        let leaf = SceneNode(transform: SceneTransform(translation: SIMD3(0, 1, 0)))
        let joint = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -5)), children: [leaf])
        var graph = fixture.graph([joint])
        let attachment = try helpers.addSkeleton(joints: [joint.id, leaf.id], to: &graph)
        let saved = try line(attachment.nodeID, in: graph)
        let right = try fixture.base.makeTarget()
        let left = try fixture.base.makeTarget()
        let first = try #require(fixture.base.queue.makeCommandBuffer())
        let second = try #require(fixture.base.queue.makeCommandBuffer())
        let rightPass = fixture.base.pass(color: right.color, depth: right.depth)
        let leftPass = fixture.base.pass(color: left.color, depth: left.depth)

        try graph.updateNode(withID: leaf.id) { $0.transform.translation = SIMD3(1, 0, 0) }
        let before = try SceneArchive(capturing: graph)
        let unretained = try #require(fixture.base.queue.makeCommandBufferWithUnretainedReferences())
        #expect(throws: DebugHelpers.SkeletonRefreshFailure.commandBuffer) {
            try graph.encode(pass: rightPass, commandBuffer: unretained, updatingSkeletons: [attachment])
        }
        #expect(unretained.status == .notEnqueued)
        let traversal = SceneRenderOrdering(sortObjects: false)
        try graph.encode(pass: rightPass, commandBuffer: first, ordering: traversal, updatingSkeletons: [attachment])
        #expect(try line(attachment.nodeID, in: graph) === saved)
        #expect(try SceneArchive(capturing: graph) == before)

        try graph.updateNode(withID: leaf.id) { $0.transform.translation = SIMD3(-1, 0, 0) }
        try graph.encode(
            pass: leftPass, commandBuffer: second,
            clipFromWorld: graph.camera.projection(aspectRatio: 1), cameraPosition: .zero,
            ordering: traversal, updatingSkeletons: [attachment]
        )
        #expect(try line(attachment.nodeID, in: graph) === saved)
        #expect(saved.geometry.positions == [SIMD3(0, 1, -5), SIMD3(0, 0, -5)])
        try fixture.base.finish(first)
        try fixture.base.finish(second)
        let rightPixels = try fixture.base.pixels(of: right.color)
        let leftPixels = try fixture.base.pixels(of: left.color)
        #expect(rightPixels[64 * 128 + 71].w > 0.9)
        #expect(rightPixels[64 * 128 + 56].w == 0)
        #expect(leftPixels[64 * 128 + 71].w == 0)
        #expect(leftPixels[64 * 128 + 56].w > 0.9)
    }

    private func orthographic(depthMode: MeshDepthMode = .forward) -> simd_float4x4 {
        let reversed = depthMode == .reversed
        return simd_float4x4(columns: (
            SIMD4(1 / 0.6, 0, 0, 0), SIMD4(0, 2, 0, 0),
            SIMD4(0, 0, reversed ? 0.5 : -0.5, 0), SIMD4(0, 0, reversed ? 1.5 : -0.5, 1)
        ))
    }

    private func line(_ id: UUID, in graph: SceneGraph) throws -> LineRenderer {
        let node = try #require(graph.node(withID: id))
        guard case .lines(let renderer) = node.primitive else {
            Issue.record("The helper has no prepared lines.")
            throw SceneGraph.EditingError.missingNode(id)
        }
        return renderer
    }

    private func centerX(_ pixels: [SIMD4<Float>]) throws -> Float {
        let columns = pixels.indices.filter { pixels[$0].w > 0.5 }.map { $0 % 128 }
        try #require(!columns.isEmpty)
        return Float(columns.reduce(0, +)) / Float(columns.count)
    }
}
