import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PrimitiveDrawRangeTests {
    @Test("Point ranges offset colored GPU draws and preserve source indices in both pickers")
    func points() throws {
        let fixture = try GeometryRenderFixture()
        let cloud = try PointCloud(
            positions: [SIMD3(-1, 0, -3), SIMD3(0, 0, -3), SIMD3(1, 0, -3)],
            colors: [SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1)]
        )
        let renderer = try PointsRenderer(device: fixture.device, cloud: cloud, material: .init(size: .pixels(7)))
        let graph = SceneGraph(roots: [SceneNode(primitive: .points(renderer))], camera: fixture.camera)
        renderer.drawRange = try GeometryDrawRange(start: 2, count: Int.max)
        let render = try fixture.render(graph)
        #expect(render.statistics.submittedPointCount == 1)
        let right = fixture.column(x: 1, depth: 3)
        let left = fixture.column(x: -1, depth: 3)
        #expect(fixture.pixel(render.pixels, x: right, y: 32) == SIMD4(0, 0, 1, 1))
        #expect(fixture.pixel(render.pixels, x: left, y: 32) == .zero)

        let ray = try SceneRay(origin: SIMD3(1, 0, 0), direction: SIMD3(0, 0, -1))
        let hit = try #require(try graph.raycast(ray: ray, pointThreshold: 0.01).first)
        guard case .primitive(let primitive) = hit else { Issue.record("Expected a point hit"); return }
        #expect(primitive.element == .point(2))
        #expect(try graph.pickPrimitive(at: SIMD2(2.0 / 3, 0.5), drawableSize: SIMD2(64, 64))?.element == .point(2))
        #expect(try graph.pickPrimitive(at: SIMD2(1.0 / 3, 0.5), drawableSize: SIMD2(64, 64)) == nil)
        renderer.drawRange = try GeometryDrawRange(start: Int.max)
        let empty = try fixture.render(graph)
        #expect(empty.statistics.submittedPointCount == 0)
        #expect(empty.pixels.allSatisfy { $0 == .zero })
        #expect(try graph.raycast(ray: ray, pointThreshold: 0.01).isEmpty)
    }

    @Test("Narrowed strips rebuild endpoint joins and retain encoded segment buffers until completion")
    func strip() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try LineGeometry(
            positions: [SIMD3(-1.2, 0, -3), SIMD3(-0.4, 0, -3), SIMD3(0.4, 0, -3), SIMD3(1.2, 0, -3)]
        )
        let renderer = try LineRenderer(
            device: fixture.device, geometry: geometry, material: .init(width: .pixels(5), cap: .round)
        )
        let graph = SceneGraph(roots: [SceneNode(primitive: .lines(renderer))], camera: fixture.camera)
        try renderer.setDrawRange(.init(start: 1, count: 2))
        let segment = try #require(renderer.drawnSegments.first)
        #expect(renderer.drawnSegments.count == 1)
        #expect(segment.start == 1 && segment.end == 2)
        #expect(segment.previous == nil && segment.next == nil)
        #expect(segment.startDistance == geometry.segments[1].startDistance)
        let removed = try SceneRay(origin: SIMD3(-0.9, 0, 0), direction: SIMD3(0, 0, -1))
        #expect(try graph.raycast(ray: removed, lineThreshold: 0.01).isEmpty)
        #expect(try graph.pickPrimitive(at: SIMD2(0.5, 0.5), drawableSize: SIMD2(64, 64))?.element == .segment(0))

        let target = try fixture.makeTarget()
        let buffer = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let old = try graph.encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer)
        try renderer.setDrawRange(.init(start: 0, count: 2))
        let replacement = try fixture.render(graph)
        try withExtendedLifetime(target.depth) { try fixture.finish(buffer) }
        let oldPixels = try fixture.pixels(of: target.color)
        let outer = fixture.column(x: -0.9, depth: 3)
        #expect(old.submittedLineSegmentCount == 1)
        #expect(replacement.statistics.submittedLineSegmentCount == 1)
        #expect(fixture.pixel(oldPixels, x: 32, y: 32).w == 1)
        #expect(fixture.pixel(oldPixels, x: outer, y: 32) == .zero)
        #expect(fixture.pixel(replacement.pixels, x: 32, y: 32) == .zero)
        #expect(fixture.pixel(replacement.pixels, x: outer, y: 32).w == 1)
    }

    @Test("Loop and shifted pair ranges persist in native archives alongside original positions")
    func archive() throws {
        let fixture = try GeometryRenderFixture()
        let positions = [SIMD3<Float>(-1, -1, -3), SIMD3(1, -1, -3), SIMD3(1, 1, -3), SIMD3(-1, 1, -3)]
        let points = try PointsRenderer(device: fixture.device, cloud: PointCloud(positions: positions), material: .init())
        let loop = try LineRenderer(device: fixture.device, geometry: LineGeometry(positions: positions, topology: .loop))
        let pairs = try LineRenderer(device: fixture.device, geometry: LineGeometry(positions: positions, topology: .segments))
        points.drawRange = try GeometryDrawRange(start: 1, count: 2)
        try loop.setDrawRange(.init(start: 1))
        try pairs.setDrawRange(.init(start: 1, count: 3))
        #expect(loop.drawnSegments.map(\.start) == [1, 2, 3])
        #expect(loop.drawnSegments.map(\.end) == [2, 3, 1])
        #expect(pairs.drawnSegments.map(\.start) == [1])
        #expect(pairs.drawnSegments.map(\.end) == [2])
        let graph = SceneGraph(
            roots: [.init(primitive: .points(points)), .init(primitive: .lines(loop)), .init(primitive: .lines(pairs))],
            camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: graph, commandQueue: fixture.queue)
        var manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 11)
        let restored = try archive.load(device: fixture.device).graph
        guard case .points(let restoredPoints)? = restored.roots[0].primitive,
              case .lines(let restoredLoop)? = restored.roots[1].primitive,
              case .lines(let restoredPairs)? = restored.roots[2].primitive else {
            Issue.record("Expected points, loop and segment pairs")
            return
        }
        #expect(restoredPoints.cloud.positions == positions && restoredPoints.drawRange == points.drawRange)
        #expect(restoredLoop.geometry.positions == positions && restoredLoop.drawnSegments == loop.drawnSegments)
        #expect(restoredPairs.geometry.positions == positions && restoredPairs.drawnSegments == pairs.drawnSegments)
        let recaptured = try SceneArchive(capturing: restored, commandQueue: fixture.queue)
        #expect(recaptured.manifest == archive.manifest)
        #expect(recaptured.binary == archive.binary)
        manifest["version"] = 10
        let downgrade = try JSONSerialization.data(withJSONObject: manifest)
        #expect(throws: SceneArchive.Failure.decoding("primitive draw ranges require archive version 11")) {
            try SceneArchive(manifest: downgrade, binary: archive.binary)
        }
    }
}
