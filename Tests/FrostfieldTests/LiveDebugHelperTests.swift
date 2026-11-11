import Foundation
import Frostfield
import Metal
import Testing
import simd

struct LiveDebugHelperTests {
    @Test
    func `Axes accept independent solid linear colors without changing their endpoints`() throws {
        let palette = LineGeometry.AxesColors(
            x: SIMD4(0.2, 0.3, 0.4, 1), y: SIMD4(0.4, 0.5, 0.6, 0.8), z: SIMD4(0.6, 0.7, 0.8, 0.5)
        )
        let styled = try LineGeometry.axes(size: 2, colors: palette)
        #expect(styled.positions == (try LineGeometry.axes(size: 2)).positions)
        #expect(styled.colors == [palette.x, palette.x, palette.y, palette.y, palette.z, palette.z])
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Live box tracks a transformed subtree without including helpers or rewriting a pending draw`() throws {
        let fixture = try GeometryRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(3), depthTest: false))
        let mesh = try fixture.unlit(.cube())
        let model = SceneNode(
            transform: .init(translation: SIMD3(0, 0.1, 0.1)), mesh: mesh, materialParameters: SIMD4(0, 0, 1, 1)
        )
        var axes = try helpers.node(.axes(size: 50))
        axes.isVisible = false
        let source = SceneNode(children: [model, axes])
        let parent = SceneNode(transform: .init(translation: SIMD3(-1.5, 0, -4)), children: [source])
        var graph = SceneGraph(roots: [parent], camera: fixture.camera)
        let box = try helpers.addBox(of: source.id, to: &graph, parent: source.id, color: SIMD4(1, 0, 0, 1))
        #expect(graph.parentID(of: box.nodeID) == source.id)
        #expect(graph.node(withID: box.nodeID)?.excludesFromDebugBounds == true)
        let initial = try line(box.nodeID, graph)
        try expectEdges(of: initial.geometry, node: box.nodeID, mesh: mesh, model: model.id, in: graph)
        let oldTarget = try fixture.makeTarget()
        let oldCommand = try #require(fixture.queue.makeCommandBuffer())
        try graph.encode(pass: fixture.pass(color: oldTarget.color, depth: oldTarget.depth), commandBuffer: oldCommand)

        try graph.updateNode(withID: parent.id) {
            $0.transform = SceneTransform(
                translation: SIMD3(1.5, 0, -4), rotation: simd_quatf(angle: 0.34, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(-1.7, 0.8, 1)
            )
        }
        let authored = try SceneArchive(capturing: graph)
        let target = try fixture.makeTarget()
        let command = try #require(fixture.queue.makeCommandBuffer())
        try graph.encode(
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command, updatingBoxes: [box]
        )
        #expect(try line(box.nodeID, graph) === initial)
        #expect(try SceneArchive(capturing: graph) == authored)
        let restored = try authored.load(device: fixture.device).graph
        #expect(restored.node(withID: axes.id)?.excludesFromDebugBounds == true)
        #expect(restored.node(withID: box.nodeID)?.excludesFromDebugBounds == true)

        try fixture.finish(oldCommand)
        try fixture.finish(command)
        let oldRed = try columns(of: fixture.pixels(of: oldTarget.color), side: fixture.side, channel: 0)
        let newRed = try columns(of: fixture.pixels(of: target.color), side: fixture.side, channel: 0)
        #expect(oldRed.upperBound < 56)
        #expect(newRed.lowerBound > 67)
        #expect(newRed.upperBound - newRed.lowerBound > oldRed.upperBound - oldRed.lowerBound + 10)

        try box.update(in: &graph)
        let updated = try line(box.nodeID, graph)
        #expect(updated !== initial)
        try expectEdges(of: updated.geometry, node: box.nodeID, mesh: mesh, model: model.id, in: graph)
        try box.update(in: &graph)
        #expect(try line(box.nodeID, graph) === updated)
        graph.removeNode(withID: model.id)
        try box.update(in: &graph)
        let empty = try #require(graph.node(withID: box.nodeID))
        #expect(empty.primitive == nil)
        #expect(try fixture.render(graph).pixels.allSatisfy { $0.w == 0 })
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Box can retarget to a batch and follows edits to its geometry placements`() throws {
        let fixture = try GeometryRenderFixture()
        let helpers = DebugHelpers(device: fixture.device)
        let batch = try BatchedMesh(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [.cube()], capacity: 1
        )
        let instance = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(1), isVisible: false)
        let source = SceneNode(transform: .init(translation: SIMD3(3, 1, -4)), batch: batch)
        let empty = SceneNode()
        var graph = SceneGraph(roots: [source, empty], camera: fixture.camera)
        var box = try helpers.addBox(of: empty.id, to: &graph)
        #expect(graph.node(withID: box.nodeID)?.primitive == nil)
        box.sourceID = source.id
        try box.update(in: &graph)
        let initial = try AxisAlignedBox(enclosing: line(box.nodeID, graph).geometry.positions)
        #expect(initial.center == SIMD3(4, 1, -4))
        #expect(initial.size == SIMD3(repeating: 1))
        try batch.update(instance, transform: translationMatrix(-2, 1, 0))
        try box.update(in: &graph)
        let updated = try AxisAlignedBox(enclosing: line(box.nodeID, graph).geometry.positions)
        #expect(updated.center == SIMD3(1, 2, -4))
        #expect(updated.size == initial.size)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Live plane updates its equation and size in an explicit view with stable prepared geometry`() throws {
        let fixture = try GeometryRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(3), depthTest: false))
        let parent = SceneNode(transform: .init(translation: SIMD3(-1, 0, -4)))
        var graph = SceneGraph(roots: [parent], camera: fixture.camera)
        var plane = try helpers.addPlane(
            Plane3D(normal: SIMD3(0, 0, 1), constant: 0), to: &graph, parent: parent.id,
            size: 0.75, color: SIMD3(0, 1, 0)
        )
        let initial = try #require(graph.node(withID: plane.nodeID))
        let outline = try line(plane.nodeID, graph)
        let fill = try #require(initial.children.first?.mesh)
        let first = try render(graph, with: fixture)

        plane.plane = try Plane3D(normal: SIMD3(0, 1, 1), through: SIMD3(0, 0, 0.2))
        plane.size = 1.6
        try graph.updateNode(withID: parent.id) {
            $0.transform = SceneTransform(translation: SIMD3(1.3, 0, -4), scale: SIMD3(1.5, 0.8, 1))
        }
        let moved = try render(graph, with: fixture, planes: [plane])
        #expect(graph.node(withID: plane.nodeID)?.transform == initial.transform)
        #expect(try line(plane.nodeID, graph) === outline)
        let oldGreen = try columns(of: first, side: fixture.side, channel: 1)
        let newGreen = try columns(of: moved, side: fixture.side, channel: 1)
        #expect(oldGreen.upperBound < 58)
        #expect(newGreen.upperBound > 100)
        #expect(newGreen.upperBound - newGreen.lowerBound > oldGreen.upperBound - oldGreen.lowerBound + 20)
        let centerFill = fixture.pixel(moved, x: 85, y: 65)
        #expect(abs(centerFill.w - 0.2) < 0.005 && abs(centerFill.y - 0.2) < 0.005)

        try plane.update(in: &graph)
        let updated = try #require(graph.node(withID: plane.nodeID))
        #expect(try line(plane.nodeID, graph) === outline)
        #expect(updated.children.first?.mesh === fill)
        for corner in outline.geometry.positions {
            let world = try graph.localToWorld(corner, in: plane.nodeID)
            let parentLocal = try graph.worldToLocal(world, in: parent.id)
            #expect(abs(plane.plane.signedDistance(to: parentLocal)) < 1e-5)
        }
        #expect(try render(graph, with: fixture) == moved)

        plane.size = 0
        #expect(throws: PrimitiveGeometryError.invalidHelperParameter("plane helper size must be finite and positive")) {
            try plane.update(in: &graph)
        }
        #expect(graph.node(withID: plane.nodeID)?.transform == updated.transform)
        let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let target = try fixture.makeTarget()
        #expect(throws: DebugHelpers.LiveHelperRefreshFailure.commandBuffer) {
            try graph.encode(
                pass: fixture.pass(color: target.color, depth: target.depth),
                commandBuffer: command, updatingPlanes: [plane]
            )
        }
        #expect(command.status == .notEnqueued)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Arrow changes direction length and color while preserving all three scene nodes`() throws {
        let fixture = try GeometryRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(3), depthTest: false))
        let parent = SceneNode(transform: .init(translation: SIMD3(0, 0, -4)))
        var graph = SceneGraph(roots: [parent], camera: fixture.camera)
        var arrow = try helpers.addArrow(
            direction: SIMD3(0, 1, 0), origin: SIMD3(-1, 0, 0), length: 1.5, color: SIMD3(1, 0, 0),
            headWidth: 0.45, to: &graph, parent: parent.id
        )
        let root = try #require(graph.node(withID: arrow.nodeID))
        let shaftID = try #require(root.children.first?.id)
        let headID = try #require(root.children.last?.id)
        let shaft = try line(shaftID, graph)
        let head = try #require(graph.node(withID: headID)?.mesh)
        #expect(root.excludesFromDebugBounds)
        #expect(graph.parentID(of: root.id) == parent.id)
        let vertical = try render(graph, with: fixture)
        #expect(fixture.pixel(vertical, x: 48, y: 55).x > 0.85)
        #expect(fixture.pixel(vertical, x: 75, y: 64).w == 0)

        arrow.origin = SIMD3(0.3, 0, 0)
        arrow.direction = SIMD3(9, 0, 0)
        arrow.length = 2
        arrow.headLength = 0.8
        arrow.headWidth = 0.55
        try arrow.update(in: &graph)
        #expect(graph.node(withID: root.id)?.children.map(\.id) == [shaftID, headID])
        #expect(try line(shaftID, graph) === shaft)
        #expect(graph.node(withID: headID)?.mesh === head)
        let tip = try graph.localToWorld(SIMD3(0, 1, 0), in: headID)
        #expect(simd_distance(tip, SIMD3(2.3, 0, -4)) < 1e-5)
        let horizontal = try render(graph, with: fixture)
        #expect(fixture.pixel(horizontal, x: 48, y: 55).w == 0)
        #expect(fixture.pixel(horizontal, x: 75, y: 64).x > 0.85)
        #expect(fixture.pixel(horizontal, x: 94, y: 64).x > 0.85)

        arrow.color = SIMD3(0, 0, 1)
        try arrow.update(in: &graph)
        #expect(graph.node(withID: headID)?.mesh === head)
        #expect(try line(shaftID, graph) !== shaft)
        let blue = try render(graph, with: fixture)
        #expect(fixture.pixel(blue, x: 75, y: 64).z > 0.85)
        #expect(fixture.pixel(blue, x: 94, y: 64).z > 0.85)
        #expect(fixture.pixel(blue, x: 94, y: 64).x < 0.05)

        arrow.headLength = arrow.length
        try arrow.update(in: &graph)
        #expect(graph.node(withID: shaftID)?.isVisible == false)
        #expect(graph.node(withID: headID)?.mesh === head)
        let archived = try SceneArchive(capturing: graph)
        arrow.direction = .zero
        #expect(throws: PrimitiveGeometryError.invalidHelperParameter(
            "the arrow needs a finite origin, a nonzero direction and color in 0 to 1"
        )) {
            try arrow.update(in: &graph)
        }
        #expect(try SceneArchive(capturing: graph) == archived)
    }

    private func line(_ id: UUID, _ graph: SceneGraph) throws -> LineRenderer {
        guard case .lines(let renderer) = graph.node(withID: id)?.primitive else {
            Issue.record("The helper has no prepared lines.")
            throw SceneGraph.EditingError.missingNode(id)
        }
        return renderer
    }

    private func columns(of pixels: [SIMD4<Float>], side: Int, channel: Int) throws -> ClosedRange<Int> {
        let columns = pixels.indices.filter { pixels[$0][channel] > 0.85 }.map { $0 % side }
        let first = try #require(columns.min())
        let last = try #require(columns.max())
        return first...last
    }

    private func expectEdges(
        of geometry: LineGeometry, node: UUID, mesh: MeshRenderer, model: UUID, in graph: SceneGraph
    ) throws {
        let world = try geometry.positions.map { try graph.localToWorld($0, in: node) }
        let bounds = try AxisAlignedBox(enclosing: world)
        let expected = try mesh.bounds.transformed(by: graph.worldTransform(of: model))
        #expect(geometry.segments.count == 12)
        #expect(simd_distance(bounds.minimum, expected.minimum) < 1e-5)
        #expect(simd_distance(bounds.maximum, expected.maximum) < 1e-5)
        for edge in geometry.segments {
            let extent = simd_abs(world[edge.start] - world[edge.end])
            #expect((0..<3).filter { extent[$0] > 1e-5 }.count == 1)
        }
    }

    private func render(
        _ graph: SceneGraph, with fixture: GeometryRenderFixture, planes: [DebugHelpers.PlaneAttachment] = []
    ) throws -> [SIMD4<Float>] {
        let target = try fixture.makeTarget()
        let command = try #require(fixture.queue.makeCommandBuffer())
        try graph.encode(
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command,
            clipFromWorld: simd_float4x4(diagonal: SIMD4(0.25, 0.25, -0.1, 1)), cameraPosition: .zero,
            directionToCamera: SIMD3(0, 0, 1), updatingPlanes: planes
        )
        try fixture.finish(command)
        return try fixture.pixels(of: target.color)
    }
}
