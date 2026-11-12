import Frostfield
import Metal
import Testing
import simd

private var supportsPrimitiveIndirectDraws: Bool {
    MTLCreateSystemDefaultDevice().map { PrimitiveIndirectDraws.isSupported(on: $0) } ?? false
}

@Suite(.enabled(if: supportsPrimitiveIndirectDraws, "Requires primitive indirect draws"))
struct PrimitiveIndirectDrawTests {
    private let fixture: PrimitiveRenderFixture

    init() throws {
        fixture = try PrimitiveRenderFixture(side: 64)
    }

    @Test("GPU arguments draw points, hairlines and sprites, and a bad vertex count draws nothing")
    func primitives() throws {
        try expectIndirect(
            node: SceneNode(primitive: .points(try PointsRenderer(
                device: fixture.device,
                cloud: try PointCloud(positions: [SIMD3(0, 0, -4)]),
                material: PointsMaterial(size: .pixels(16), color: SIMD4(1, 0, 0, 1))
            ))),
            vertexCount: 6, instanceCount: 1
        )
        try expectIndirect(
            node: SceneNode(primitive: .lines(try LineRenderer(
                device: fixture.device,
                geometry: LineGeometry(positions: [SIMD3(-1, 0, -4), SIMD3(1, 0, -4)]),
                material: LineMaterial(width: .hairline, color: SIMD4(0, 1, 0, 1), blending: .opaque)
            ))),
            vertexCount: 2, instanceCount: 1
        )
        try expectIndirect(
            node: SceneNode(primitive: .sprites(try SpriteRenderer(
                device: fixture.device,
                sprites: [Sprite(position: SIMD3(0, 0, -4), size: SIMD2(repeating: 16))],
                material: SpriteMaterial(color: SIMD4(0, 0, 1, 1), sizeAttenuation: false, blending: .opaque)
            ))),
            vertexCount: 6, instanceCount: 1
        )
    }

    private func expectIndirect(node: SceneNode, vertexCount: UInt32, instanceCount: UInt32) throws {
        let direct = try fixture.render(fixture.graph([node]))
        let directCoverage = covered(direct.pixels)
        #expect(directCoverage > 0)
        #expect(direct.statistics.indirectArgumentDrawCount == 0)

        var valid = fixture.graph([node])
        valid.primitiveIndirectDraws[node.id] = try arguments(vertexCount: vertexCount, instanceCount: instanceCount)
        let drawn = try fixture.render(valid)
        #expect(covered(drawn.pixels) == directCoverage)
        #expect(drawn.statistics.indirectArgumentDrawCount == 1)
        #expect(drawn.statistics.submittedPointCount == 0)
        #expect(drawn.statistics.submittedLineSegmentCount == 0)
        #expect(drawn.statistics.submittedSpriteCount == 0)

        var invalid = fixture.graph([node])
        invalid.primitiveIndirectDraws[node.id] = try arguments(vertexCount: vertexCount + 3, instanceCount: instanceCount)
        let rejected = try fixture.render(invalid)
        #expect(covered(rejected.pixels) == 0)
        #expect(rejected.statistics.indirectArgumentDrawCount == 1)
    }

    @Test("Primitive indirect arguments reject a mesh node and a missing node")
    func targets() throws {
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1, height: 1),
            material: .physicallyBased(try PhysicallyBasedMaterial())
        )
        let node = SceneNode(mesh: mesh)
        var graph = fixture.graph([node])
        graph.primitiveIndirectDraws[node.id] = try arguments(vertexCount: 6, instanceCount: 1)
        #expect(throws: PrimitiveIndirectDraws.Failure.incompatibleNode(node.id)) {
            try fixture.render(graph)
        }
        var missing = fixture.graph([])
        let id = UUID()
        missing.primitiveIndirectDraws[id] = try arguments(vertexCount: 6, instanceCount: 1)
        #expect(throws: PrimitiveIndirectDraws.Failure.unknownNode(id)) {
            try fixture.render(missing)
        }
    }

    private func covered(_ pixels: [SIMD4<Float>]) -> Int {
        pixels.filter { $0.w > 0.5 }.count
    }

    private func arguments(vertexCount: UInt32, instanceCount: UInt32) throws -> PrimitiveIndirectDraws {
        var record = MTLDrawPrimitivesIndirectArguments(
            vertexCount: vertexCount, instanceCount: instanceCount, vertexStart: 0, baseInstance: 0
        )
        let buffer = try #require(fixture.device.makeBuffer(
            bytes: &record, length: MemoryLayout<MTLDrawPrimitivesIndirectArguments>.stride, options: .storageModeShared
        ))
        return try PrimitiveIndirectDraws(buffer: buffer)
    }
}
