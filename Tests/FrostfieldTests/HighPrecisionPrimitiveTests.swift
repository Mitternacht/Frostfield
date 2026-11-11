import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionPrimitiveTests {
    @Test("Points, lines and rotated sprites keep their pixels, world padding and culling at a 2⁶⁰ origin")
    func placementAndCulling() throws {
        let gpu = try GeometryRenderFixture(side: 96)
        let red = SIMD4<Float>(1, 0, 0, 1)
        let green = SIMD4<Float>(0, 1, 0, 1)
        let blue = SIMD4<Float>(0, 0, 1, 1)
        let white = SIMD4<Float>(repeating: 1)
        let cloud = try PointCloud(positions: [.zero])
        let points = ScenePrimitive.points(try PointsRenderer(
            device: gpu.device, cloud: cloud, material: .init(size: .pixels(7), color: red)
        ))
        let edgePoints = ScenePrimitive.points(try PointsRenderer(
            device: gpu.device, cloud: cloud, material: .init(size: .world(0.5), shape: .circle, color: white)
        ))
        let line = ScenePrimitive.lines(try LineRenderer(
            device: gpu.device, geometry: LineGeometry(positions: [SIMD3(-0.45, -0.2, 0), SIMD3(0.45, 0.2, 0)]),
            material: .init(width: .world(0.22), color: green)
        ))
        let sprite = ScenePrimitive.sprites(try SpriteRenderer(
            device: gpu.device, sprites: [Sprite(position: .zero, size: SIMD2(0.9, 0.32), rotation: .pi / 2)],
            material: .init(color: blue, blending: .opaque)
        ))
        let localOverride = try AxisAlignedBox(minimum: SIMD3(3.5, -0.1, 0), maximum: SIMD3(4, 0.1, 0))
        let nodes = [
            node(points, at: SIMD3(-1.4, 0.5, -3)),
            node(line, at: SIMD3(0, -0.6, -3)),
            node(sprite, at: SIMD3(1.3, 0.5, -3)),
            node(edgePoints, at: SIMD3(-3.12, 1.4, -3)),
            node(points, at: SIMD3(4.5, 0, -3)),
            node(line, at: SIMD3(4.5, 0, -3)),
            node(sprite, at: SIMD3(4.5, 0, -3)),
            SceneNode(
                transform: .init(translation: SIMD3(0, 1.5, -3)), primitive: points, cullingBounds: localOverride
            ),
        ]
        let reference = try gpu.render(scene(gpu, origin: .zero, nodes: nodes, highPrecision: false))
        let translated = scene(gpu, origin: SIMD3(0x1p60, 0, 0), nodes: nodes, highPrecision: true)
        let precise = try gpu.render(translated)
        #expect(zip(reference.pixels, precise.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.001) })
        #expect(precise.statistics.drawnPrimitiveCount == 4)
        #expect(precise.statistics.culledPrimitiveCount == 4)
        #expect(precise.statistics.submittedPointCount == 2)
        #expect(precise.statistics.submittedLineSegmentCount == 1)
        #expect(precise.statistics.submittedSpriteCount == 1)

        #expect(gpu.pixel(precise.pixels, x: 25, y: 40) == red)
        #expect(gpu.pixel(precise.pixels, x: 31, y: 40) == .zero)
        #expect(isNearly(gpu.pixel(precise.pixels, x: 48, y: 58), green, tolerance: 0.01))
        #expect(gpu.pixel(precise.pixels, x: 69, y: 34) == blue)
        #expect(gpu.pixel(precise.pixels, x: 75, y: 40) == .zero)
        #expect(gpu.pixel(precise.pixels, x: 0, y: 25) == white)

        var legacy = translated
        legacy.highPrecision = false
        #expect(try gpu.render(legacy).pixels != reference.pixels)
    }

    @Test("Transparent sprite ordering uses camera-relative depth at a 2⁶⁰ depth origin")
    func transparentDepth() throws {
        let gpu = try GeometryRenderFixture()
        func sprite(color: SIMD4<Float>, depth: Float) throws -> SceneNode {
            let renderer = try SpriteRenderer(
                device: gpu.device, sprites: [Sprite(position: .zero, size: SIMD2(8, 8))],
                material: .init(color: color, sizeAttenuation: false)
            )
            return node(.sprites(renderer), at: SIMD3(0, 0, -depth))
        }
        let near = try sprite(color: SIMD4(1, 0, 0, 0.5), depth: 4)
        let far = try sprite(color: SIMD4(0, 0, 1, 0.5), depth: 8)
        let ordinary = try gpu.render(scene(gpu, origin: .zero, nodes: [near, far], highPrecision: false))
        let precise = try gpu.render(scene(gpu, origin: SIMD3(0, 0, 0x1p60), nodes: [near, far], highPrecision: true))
        #expect(precise.pixels == ordinary.pixels)
        #expect(gpu.pixel(precise.pixels, x: 32, y: 32) == SIMD4(0.5, 0, 0.25, 0.75))
        #expect(precise.statistics.drawnPrimitiveCount == 2)
        #expect(precise.statistics.culledPrimitiveCount == 0)
    }

    private func node(_ primitive: ScenePrimitive, at position: SIMD3<Float>) -> SceneNode {
        SceneNode(transform: .init(translation: position), primitive: primitive)
    }

    private func scene(
        _ gpu: GeometryRenderFixture, origin: SIMD3<Float>, nodes: [SceneNode], highPrecision: Bool
    ) -> SceneGraph {
        let root = SceneNode(transform: .init(translation: origin), children: nodes)
        var graph = SceneGraph(roots: [root], camera: gpu.camera, cameraTransform: .init(translation: origin))
        graph.highPrecision = highPrecision
        return graph
    }
}
