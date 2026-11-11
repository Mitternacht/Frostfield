import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CurveFlowTests {
    @Test("Independent offsets update rendering, bounds and picking while in-flight draws retain old vertices", arguments: [CurveFlow.Execution.cpu, .metal])
    func updatesWithoutRepreparing(_ execution: CurveFlow.Execution) throws {
        let fixture = try GeometryRenderFixture()
        let lower = try spine(y: -0.35)
        let upper = try spine(y: 0.35)
        let source = try MeshGeometry.plane(width: 0.25, height: 0.25)
        let flow = try CurveFlow(
            device: fixture.device, mesh: source,
            placements: [.init(curve: lower, offset: 0.5), .init(curve: upper, offset: 1.5)], execution: execution
        )
        let renderer = try makeRenderer(flow)
        let graph = graph(renderer, fixture: fixture)
        let beforeTarget = try fixture.makeTarget()
        let before = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let oldBounds = renderer.bounds
        #expect(try graph.pick(at: viewport(x: -0.5, y: -0.35), aspectRatio: 1) != nil)
        let oldStats = try graph.encode(pass: fixture.pass(color: beforeTarget.color, depth: beforeTarget.depth), commandBuffer: before)
        #expect(oldStats.drawnMeshCount == 1)

        try flow.move(along: 0.75, at: 0)
        #expect(flow.revision == 1)
        #expect(flow.placements.map(\.offset) == [1.25, 1.5])
        #expect(renderer.bounds == flow.bounds && renderer.bounds.minimum.x > oldBounds.minimum.x)
        #expect(renderer.geometry.vertices.map(\.position) == flow.geometry.vertices.map(\.position))
        #expect(try graph.pick(at: viewport(x: -0.5, y: -0.35), aspectRatio: 1) == nil)
        #expect(try graph.pick(at: viewport(x: 0.25, y: -0.35), aspectRatio: 1) != nil)
        #expect(try graph.pick(at: viewport(x: 0.5, y: 0.35), aspectRatio: 1) != nil)

        let after = try fixture.render(graph)
        #expect(after.statistics.drawnMeshCount == 1)
        try fixture.finish(before)
        let beforePixels = try fixture.pixels(of: beforeTarget.color)
        #expect(pixel(beforePixels, x: -0.5, y: -0.35, fixture: fixture) == tint)
        #expect(pixel(beforePixels, x: 0.25, y: -0.35, fixture: fixture) == .zero)
        #expect(pixel(after.pixels, x: -0.5, y: -0.35, fixture: fixture) == .zero)
        #expect(pixel(after.pixels, x: 0.25, y: -0.35, fixture: fixture) == tint)
        #expect(pixel(beforePixels, x: 0.5, y: 0.35, fixture: fixture) == tint)
        #expect(pixel(after.pixels, x: 0.5, y: 0.35, fixture: fixture) == tint)
    }

    @Test("Surface attributes follow a curve flow that moves back into the frustum", arguments: [CurveFlow.Execution.cpu, .metal])
    func surfaceAttributes(_ execution: CurveFlow.Execution) throws {
        let fixture = try GeometryRenderFixture()
        let plain = try MeshGeometry.plane(width: 0.4, height: 0.4)
        let tilted = try MeshGeometry(vertices: plain.vertices.map {
            MeshVertex(position: $0.position, normal: simd_normalize(SIMD3(0, 1, 1)), textureCoordinate: $0.textureCoordinate)
        }, indices: plain.indices)
        let path = try spine(y: 0)
        let flow = try CurveFlow(device: fixture.device, mesh: tilted, placements: [.init(curve: path, offset: 10)], execution: execution)
        let renderer = try makeRenderer(flow)
        let graph = graph(renderer, fixture: fixture)
        #expect(try fixture.render(graph).statistics.culledMeshCount == 1)
        try flow.setOffset(1, at: 0)

        let targets = try fixture.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: fixture.side, height: fixture.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let attributes = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let command = try #require(fixture.queue.makeCommandBuffer())
        let draw = try graph.encodeSurfaceAttributes(
            pass: fixture.pass(color: attributes, depth: targets.depth), commandBuffer: command
        )
        #expect(draw.drawnMeshCount == 1 && draw.culledMeshCount == 0)
        try fixture.finish(command)
        var normal = SIMD2<UInt32>.zero
        attributes.getBytes(&normal, bytesPerRow: MemoryLayout<SIMD2<UInt32>>.stride,
                            from: MTLRegionMake2D(32, 32, 1, 1), mipmapLevel: 0)
        #expect(normal.x != 0)
    }

    @Test("The vertex pool bounds distinct in-flight revisions and can retry after completion", arguments: [CurveFlow.Execution.cpu, .metal])
    func bufferReuse(_ execution: CurveFlow.Execution) throws {
        let fixture = try GeometryRenderFixture()
        let path = try spine(y: 0)
        let flow = try CurveFlow(
            device: fixture.device, mesh: .plane(width: 0.3, height: 0.3),
            placements: [.init(curve: path, offset: 0.5)], execution: execution
        )
        let renderer = try makeRenderer(flow)
        let scene = MeshScene(camera: fixture.camera, transform: .init(translation: SIMD3(0, 0, -2)))
        var pending: [any MTLCommandBuffer] = []
        for index in 0..<3 {
            try flow.setOffset(0.5 + Double(index) * 0.25, at: 0)
            let target = try fixture.makeTarget()
            let command = try #require(fixture.queue.makeCommandBuffer())
            try renderer.encode(scene, pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command)
            pending.append(command)
        }
        try flow.setOffset(1.25, at: 0)
        let target = try fixture.makeTarget()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let pass = fixture.pass(color: target.color, depth: target.depth)
        #expect(throws: CurveFlow.Failure.buffersBusy) {
            try renderer.encode(scene, pass: pass, commandBuffer: command)
        }
        try fixture.finish(pending.removeFirst())
        try renderer.encode(scene, pass: pass, commandBuffer: command)
        try fixture.finish(command)
        #expect(try pixel(fixture.pixels(of: target.color), x: 0.25, y: 0, fixture: fixture) == tint)
        for command in pending { try fixture.finish(command) }
    }

    @Test("Discarding an uncommitted draw releases its curve flow vertex lease", arguments: [CurveFlow.Execution.cpu, .metal])
    func abandonedCommand(_ execution: CurveFlow.Execution) throws {
        let fixture = try GeometryRenderFixture()
        let path = try spine(y: 0)
        let flow = try CurveFlow(
            device: fixture.device, mesh: .plane(width: 0.25, height: 0.25),
            placements: [.init(curve: path, offset: 0.5)], execution: execution
        )
        let renderer = try makeRenderer(flow)
        let scene = MeshScene(camera: fixture.camera, transform: .init(translation: SIMD3(0, 0, -2)))
        let target = try fixture.makeTarget()
        for index in 0..<5 {
            try flow.setOffset(0.5 + Double(index) * 0.1, at: 0)
            try autoreleasepool {
                let command = try #require(fixture.queue.makeCommandBuffer())
                try renderer.encode(
                    scene, pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command
                )
            }
        }
        let command = try #require(fixture.queue.makeCommandBuffer())
        try renderer.encode(scene, pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command)
        try fixture.finish(command)
        #expect(try pixel(fixture.pixels(of: target.color), x: -0.1, y: 0, fixture: fixture) == tint)
    }

    @Test("Invalid placement edits preserve the complete current revision")
    func validation() throws {
        let fixture = try GeometryRenderFixture()
        let path = try spine(y: 0)
        let flow = try CurveFlow(
            device: fixture.device, mesh: .plane(), placements: [.init(curve: path, offset: 0.5)]
        )
        let original = flow.geometry.vertices.map(\.position)
        #expect(throws: CurveFlow.Failure.index(1)) { try flow.setOffset(1, at: 1) }
        #expect(throws: CurveFlow.Failure.placementCount) { try flow.replacePlacements([]) }
        #expect(throws: CurveDeformer.Failure.placement) { try flow.setOffset(.infinity, at: 0) }
        #expect(flow.revision == 0)
        #expect(flow.placements[0].offset == 0.5)
        #expect(flow.geometry.vertices.map(\.position) == original)
        try flow.setOffset(0.5, at: 0)
        #expect(flow.revision == 0)
        let newCurve = try spine(y: 0.5)
        try flow.setPlacement(.init(curve: newCurve, offset: 1), at: 0)
        #expect(flow.revision == 1)
        #expect(flow.bounds.center.y == 0.5)
    }

    @Test("Metal frame tables reproduce the CPU attributes on curved, wrapped and extended placements")
    func metalVertices() throws {
        let fixture = try GeometryRenderFixture()
        let open = try CurveDeformer(curve: Curve3D(position: { parameter in
            let angle = parameter * Double.pi * 0.5
            return SIMD3(2 * sin(angle), 2 * (1 - cos(angle)), 0)
        }), subdivisions: 96)
        let loop = try CurveDeformer(curve: Curve3D(position: { parameter in
            let angle = 2 * Double.pi * parameter
            let radius = 2 + 0.15 * cos(3 * angle)
            return SIMD3(radius * cos(angle), radius * sin(angle), 0.15 * sin(3 * angle))
        }), subdivisions: 128, boundary: .closed)
        let normal = simd_normalize(SIMD3<Float>(1, -1, 1))
        let tangent = SIMD4(simd_normalize(SIMD3<Float>(1, 1, 0)), -1)
        let positions: [SIMD3<Float>] = [SIMD3(1.9, 0, 0), SIMD3(2.15, 0.15, 0), SIMD3(2.05, 0.15, 0.1)]
        let mesh = try MeshGeometry(vertices: positions.enumerated().map { index, position in
            MeshVertex(
                position: position, normal: normal, textureCoordinate: SIMD2(Float(index) * 0.25, 0),
                secondaryTextureCoordinate: SIMD2(1, Float(index) * 0.25), tangent: tangent
            )
        }, indices: [0, 1, 2])
        let flow = try CurveFlow(
            device: fixture.device, mesh: mesh,
            placements: [.init(curve: open, offset: 1), .init(curve: loop, offset: -0.05)],
            sourceXOrigin: 2, distanceScale: 1.3, execution: .metal
        )
        try expectMetalMatches(flow, fixture: fixture)

        try flow.replacePlacements([.init(curve: loop, offset: loop.length * 15), .init(curve: open, offset: open.length + 0.05)])
        try expectMetalMatches(flow, fixture: fixture)
        let original = flow.placements.map(\.offset)
        #expect(throws: CurveFlow.Failure.metalDomain) { try flow.setOffset(1e7 + 1, at: 1) }
        #expect(throws: CurveFlow.Failure.metalSpine) { try flow.setPlacement(.init(curve: spine(y: 0), offset: 1), at: 0) }
        #expect(flow.placements.map(\.offset) == original && flow.revision == 1)
    }

    private func expectMetalMatches(_ flow: CurveFlow, fixture: GeometryRenderFixture) throws {
        let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let buffer = try flow.acquireVertices(for: command)
        #expect(try flow.acquireVertices(for: command) === buffer)
        try fixture.finish(command)
        let vertices = buffer.contents().bindMemory(to: PackedMeshVertex.self, capacity: flow.geometry.vertices.count)
        for (index, vertex) in flow.geometry.vertices.enumerated() {
            let expected = PackedMeshVertex(vertex)
            #expect(simd_distance(vertices[index].position, expected.position) < 1e-5)
            #expect(simd_distance(vertices[index].normal, expected.normal) < 1e-5)
            #expect(simd_distance(vertices[index].tangent, expected.tangent) < 1e-5)
            #expect(vertices[index].uv == expected.uv)
        }
        let following = try #require(fixture.queue.makeCommandBuffer())
        #expect(try flow.acquireVertices(for: following) === buffer)
        try fixture.finish(following)
    }

    private var tint: SIMD4<Float> { SIMD4(0.75, 0.25, 0.125, 1) }

    private func spine(y: Double) throws -> CurveDeformer {
        try CurveDeformer(curve: Curve3D(position: { SIMD3(-1 + 2 * $0, y, 0) }), subdivisions: 8)
    }

    private func makeRenderer(_ flow: CurveFlow) throws -> MeshRenderer {
        try MeshRenderer(
            curveFlow: flow, colorPixelFormat: .rgba16Float,
            material: .classic(.init(model: .basic, color: SIMD3(tint.x, tint.y, tint.z)))
        )
    }

    private func graph(_ mesh: MeshRenderer, fixture: GeometryRenderFixture) -> SceneGraph {
        SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: fixture.camera
        )
    }

    private func viewport(x: Float, y: Float) -> SIMD2<Float> { SIMD2(0.5 + x / 4, 0.5 - y / 4) }

    private func pixel(_ pixels: [SIMD4<Float>], x: Float, y: Float, fixture: GeometryRenderFixture) -> SIMD4<Float> {
        fixture.pixel(pixels, x: fixture.column(x: x, depth: 2), y: fixture.row(y: y, depth: 2))
    }
}
