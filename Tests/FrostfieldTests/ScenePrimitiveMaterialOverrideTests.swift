import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ScenePrimitiveMaterialOverrideTests {
    @Test("Point and line overrides keep live ranges and honor exclusions in both depth modes")
    func rangesAndExclusions() throws {
        let gpu = try PrimitiveRenderFixture(side: 64)
        let red = SIMD4<Float>(1, 0, 0, 1)
        let points = try PointsRenderer(
            device: gpu.device,
            cloud: PointCloud(positions: [gpu.point(pixel: SIMD2(8, 16), depth: 4),
                                          gpu.point(pixel: SIMD2(24, 16), depth: 4)]),
            material: PointsMaterial(color: red)
        )
        let lines = try LineRenderer(
            device: gpu.device,
            geometry: LineGeometry(positions: [8, 24, 40, 56].map {
                gpu.point(pixel: SIMD2(Float($0), 32.5), depth: 4)
            }, topology: .segments),
            material: LineMaterial(width: .hairline, color: red, blending: .opaque)
        )
        let excluded = SceneNode(
            transform: .init(translation: SIMD3(0, -2, 0)), primitive: .lines(lines)
        )
        var scene = gpu.graph([SceneNode(primitive: .points(points)), SceneNode(primitive: .lines(lines)), excluded])
        let override = try ScenePrimitiveMaterialOverride(
            points: PointsMaterial(size: .pixels(8), color: SIMD4(0, 1, 0, 1)),
            lines: LineMaterial(width: .pixels(6), cap: .butt, color: SIMD4(0, 0, 1, 1), blending: .opaque),
            excluding: [excluded.id]
        )
        try override.prepare(in: scene, colorPixelFormat: .rgba16Float)
        points.drawRange = try .init(start: 1, count: 1)
        try lines.setDrawRange(.init(start: 2, count: 2))
        let original = try gpu.render(scene).pixels
        scene.primitiveMaterialOverride = override
        let pixels = try gpu.render(scene).pixels
        #expect(pixels[16 * 64 + 8] == .zero)
        #expect(pixels[16 * 64 + 27] == SIMD4(0, 1, 0, 1))
        #expect(pixels[32 * 64 + 16] == .zero)
        #expect(pixels[34 * 64 + 48] == SIMD4(0, 0, 1, 1))
        #expect(original[48 * 64 + 48] == red && pixels[48 * 64 + 48] == red)
        #expect(try gpu.render(scene, depthMode: .reversed) == pixels)
        scene.primitiveMaterialOverride = nil
        #expect(try gpu.render(scene).pixels == original)
    }

    @Test("World-sized overrides expand culling bounds at the frustum edge")
    func cullingUsesOverrideSize() throws {
        let gpu = try PrimitiveRenderFixture(side: 64)
        let renderer = try PointsRenderer(
            device: gpu.device, cloud: PointCloud(positions: [gpu.point(pixel: SIMD2(-2, 32), depth: 4)]),
            material: PointsMaterial()
        )
        var scene = gpu.graph([SceneNode(primitive: .points(renderer))])
        #expect(try gpu.render(scene).pixels.allSatisfy { $0 == .zero })
        scene.primitiveMaterialOverride = try ScenePrimitiveMaterialOverride(points: PointsMaterial(size: .world(1)))
        #expect(try gpu.render(scene).pixels[32 * 64] == SIMD4(repeating: 1))
    }

    @Test("An overridden sprite changes draw phase and retains its texture for an unretained command")
    func unretainedSprite() throws {
        let gpu = try PrimitiveRenderFixture(side: 64)
        let mesh = try gpu.base.unlit(.plane(width: 2, height: 2))
        let plane = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh,
            materialParameters: SIMD4(0, 0, 1, 1)
        )
        let sprites = try SpriteRenderer(
            device: gpu.device, sprites: [.init(position: SIMD3(0, 0, -4), size: SIMD2(1, 1))],
            material: SpriteMaterial(blending: .opaque)
        )
        var scene = gpu.graph([SceneNode(primitive: .sprites(sprites)), plane])
        let center = 32 * 64 + 32
        #expect(try gpu.render(scene).pixels[center] == SIMD4(0, 0, 1, 1))
        let target = try gpu.base.makeTarget()
        let command = try #require(gpu.base.queue.makeCommandBufferWithUnretainedReferences())
        do {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
            desc.usage = .shaderRead
            let texture = try #require(gpu.device.makeTexture(descriptor: desc))
            let red: [UInt8] = [255, 0, 0, 255]
            texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: red, bytesPerRow: 4)
            scene.primitiveMaterialOverride = try ScenePrimitiveMaterialOverride(
                sprites: SpriteMaterial(map: texture, blending: .alpha, depthTest: false),
                parameters: SIMD4(1, 1, 1, 0.5)
            )
            try scene.encode(pass: gpu.base.pass(color: target.color, depth: target.depth), commandBuffer: command)
            scene.primitiveMaterialOverride = nil
        }
        try gpu.base.finish(command)
        #expect(try gpu.base.pixels(of: target.color)[center] == SIMD4(0.5, 0, 0.5, 1))
    }

    @Test("A primitive override rejects feedback before encoding any color draw")
    func feedback() throws {
        let gpu = try PrimitiveRenderFixture(side: 64)
        let point = try PointsRenderer(
            device: gpu.device, cloud: PointCloud(positions: [SIMD3(0, 0, -4)]), material: PointsMaterial()
        )
        var scene = gpu.graph([SceneNode(primitive: .points(point))])
        let target = try gpu.base.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 64, height: 64, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead]
        let color = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let pass = gpu.base.pass(color: color, depth: target.depth)
        let command = try #require(gpu.base.queue.makeCommandBuffer())
        scene.primitiveMaterialOverride = try ScenePrimitiveMaterialOverride(points: PointsMaterial(map: color))
        #expect(throws: PrimitiveRendererError.mapAliasing) {
            try scene.encode(pass: pass, commandBuffer: command)
        }
        scene.primitiveMaterialOverride = nil
        try scene.encode(pass: pass, commandBuffer: command)
        try gpu.base.finish(command)
        #expect(try gpu.base.pixels(of: color)[32 * 64 + 32] == SIMD4(repeating: 1))
    }
}
