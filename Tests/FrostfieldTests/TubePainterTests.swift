import Frostfield
import Metal
import Testing
import simd

struct TubePainterTests {
    private static let red = SIMD4<Float>(1, 0, 0, 1)
    private static let green = SIMD4<Float>(0, 1, 0, 1)

    @Test("Incremental painting keeps earlier rings and moves the rounded tip with radius and color")
    func incrementalStroke() throws {
        var painter = try TubePainter()
        #expect(try painter.mesh() == nil)
        #expect(painter.radialSegments == 15 && painter.radius == 0.01)
        try painter.move(to: .zero)
        #expect(try !painter.line(to: .zero))
        try painter.setColor(Self.red)
        #expect(try painter.line(to: SIMD3(0.5, 0, 0)))
        let first = try #require(try painter.mesh())
        #expect(first.vertices.count == 192 && first.indices.count == 240 * 3)
        #expect(ShapeCheck.isClosed(first))
        #expect(first.vertices.allSatisfy { $0.color == Self.red })
        #expect(first.vertices[0].position == SIMD3(0, 0.01, 0))
        let originalBounds = ShapeCheck.bounds(first.vertices.map(\.position))
        #expect(abs(originalBounds.lower.x + 0.01) < 1e-6)
        #expect(abs(originalBounds.upper.x - 0.51) < 1e-6)

        try painter.setRadius(0.02)
        try painter.setColor(Self.green)
        let destination = SIMD3<Float>(1, 0.1, 0)
        try painter.line(to: destination)
        let second = try #require(try painter.mesh())
        #expect(second.vertices.count == first.vertices.count + 16)
        #expect(second.indices.count == first.indices.count + 90)
        #expect(second.indices.prefix(first.indices.count).elementsEqual(first.indices))
        #expect(ShapeCheck.isClosed(second))
        #expect(zip(first.vertices.prefix(112), second.vertices.prefix(112)).allSatisfy(sameVertex))
        #expect(second.vertices[112..<192].allSatisfy { $0.color == Self.green })
        for vertex in second.vertices.suffix(16) {
            #expect(vertex.color == Self.green)
            #expect(abs(simd_distance(vertex.position, destination) - 0.02) < 1e-6)
        }
        let direction = simd_normalize(destination - SIMD3(0.5, 0, 0))
        let tip = destination + 0.02 * direction
        #expect(second.vertices[176..<192].allSatisfy { simd_distance($0.position, tip) < 1e-6 })
        #expect(first.vertices[176..<192].allSatisfy { $0.position == SIMD3(0.51, 0, 0) && $0.color == Self.red })
    }

    @Test("Moving the brush starts a separate closed stroke and retains the previous cap")
    func multipleStrokes() throws {
        var painter = try TubePainter(radialSegments: 8)
        try painter.move(to: .zero)
        try painter.line(to: SIMD3(0, 0, 0.3))
        let first = try #require(try painter.mesh())
        try painter.move(to: SIMD3(0, 1, 0))
        #expect(try painter.mesh()?.indices == first.indices)
        try painter.setRadius(0.05)
        try painter.setColor(Self.green)
        try painter.line(to: SIMD3(0, 1.3, 0))
        let result = try #require(try painter.mesh())
        #expect(result.vertices.count == 2 * first.vertices.count)
        #expect(result.indices.count == 2 * first.indices.count)
        #expect(zip(first.vertices, result.vertices.prefix(first.vertices.count)).allSatisfy(sameVertex))
        #expect(ShapeCheck.isClosed(result))
    }

    @Test("Brush validation and failed appends preserve prior geometry and position")
    func validation() throws {
        let limit = try MeshRenderer.DynamicGeometryCapacity(vertices: 192, triangles: 240)
        var painter = try TubePainter(capacity: limit)
        #expect(throws: TubePainter.Failure.missingMove) { try painter.line(to: SIMD3(1, 0, 0)) }
        #expect(throws: TubePainter.Failure.radius) { try painter.setRadius(.nan) }
        #expect(throws: TubePainter.Failure.color) { try painter.setColor(SIMD4(1, 0, 0, 2)) }
        #expect(throws: TubePainter.Failure.point) { try painter.move(to: SIMD3(.infinity, 0, 0)) }
        try painter.move(to: .zero)
        try painter.line(to: SIMD3(1, 0, 0))
        let baseline = try #require(try painter.mesh())
        #expect(throws: TubePainter.Failure.reversingDirection) { try painter.line(to: .zero) }
        #expect(throws: TubePainter.Failure.capacity) { try painter.line(to: SIMD3(2, 0, 0)) }
        #expect(try !painter.line(to: SIMD3(1, 0, 0)))
        let unchanged = try #require(try painter.mesh())
        #expect(unchanged.indices == baseline.indices)
        #expect(zip(baseline.vertices, unchanged.vertices).allSatisfy(sameVertex))

        var tooFar = try TubePainter()
        try tooFar.move(to: SIMD3(1e20, 1e20, 0))
        #expect(throws: TubePainter.Failure.foldedSurface) { try tooFar.line(to: SIMD3(1e20, 1e20, 1)) }
        #expect(try tooFar.mesh() == nil)
    }

    @Test("One dynamic Metal renderer publishes the live brush extension and changed radius",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func liveRender() throws {
        let gpu = try GeometryRenderFixture(side: 128)
        var painter = try TubePainter()
        try painter.move(to: SIMD3(-0.85, 0, -2))
        try painter.setRadius(0.1)
        try painter.setColor(Self.red)
        try painter.line(to: SIMD3(-0.15, 0, -2))
        let mesh = try #require(try painter.mesh())
        let renderer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, dynamicGeometry: mesh, capacity: painter.capacity,
            material: .classic(ClassicMaterial(model: .basic))
        )
        let graph = SceneGraph(roots: [SceneNode(mesh: renderer)], camera: gpu.camera)
        let first = try gpu.render(graph)
        let left = gpu.column(x: -0.5, depth: 2), right = gpu.column(x: 0.62, depth: 2), center = gpu.side / 2
        #expect(gpu.pixel(first.pixels, x: left, y: center) == Self.red)
        #expect(gpu.pixel(first.pixels, x: right, y: center) == .zero)
        #expect(first.statistics.submittedTriangleCount == 240)

        try painter.setRadius(0.24)
        try painter.setColor(Self.green)
        try painter.line(to: SIMD3(0.75, 0, -2))
        try renderer.updateGeometry(try #require(try painter.mesh()))
        let second = try gpu.render(graph)
        #expect(renderer.geometryRevision == 1)
        #expect(second.statistics.submittedTriangleCount == 270)
        #expect(gpu.pixel(second.pixels, x: left, y: center) == Self.red)
        #expect(gpu.pixel(second.pixels, x: right, y: center).y > 0.7)
        #expect(second.pixels.contains(Self.green))
        let leftWidth = (0..<gpu.side).filter { gpu.pixel(second.pixels, x: left, y: $0).w > 0 }.count
        let rightWidth = (0..<gpu.side).filter { gpu.pixel(second.pixels, x: right, y: $0).w > 0 }.count
        #expect(rightWidth > leftWidth * 2)
    }

    private func sameVertex(_ a: MeshVertex, _ b: MeshVertex) -> Bool {
        a.position == b.position && a.normal == b.normal && a.textureCoordinate == b.textureCoordinate
            && a.tangent == b.tangent && a.color == b.color
    }
}
