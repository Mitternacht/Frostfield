import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PointsAndSpritesTests {
    private func coveredPixels(_ pixels: [SIMD4<Float>], side: Int, columns: Range<Int>) -> [SIMD2<Int>] {
        pixels.indices.compactMap { index in
            let x = index % side
            return columns.contains(x) && pixels[index].w > 0 ? SIMD2(x, index / side) : nil
        }
    }

    private func extent(_ pixels: [SIMD2<Int>]) -> (minimum: SIMD2<Int>, maximum: SIMD2<Int>) {
        (pixels.reduce(SIMD2(Int.max, Int.max)) { simd_min($0, $1) }, pixels.reduce(SIMD2(Int.min, Int.min)) { simd_max($0, $1) })
    }

    @Test
    func `World-sized points shrink as one over distance like an object of that width`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // Unit-wide points at depths 4 and 8: 32 × 1 / 4 = 8 and 4 pixels, centered on pixel corners.
        func node(_ position: SIMD3<Float>, _ size: PointsMaterial.Size) throws -> SceneNode {
            let renderer = try PointsRenderer(
                device: fixture.device, cloud: try PointCloud(positions: [position]),
                material: PointsMaterial(size: size, color: SIMD4(1, 0.5, 0.25, 1))
            )
            return SceneNode(primitive: .points(renderer))
        }
        let attenuated = try fixture.render(fixture.graph([
            try node(SIMD3(-2, 0, -4), .world(1)), try node(SIMD3(4, 0, -8), .world(1)),
        ]), name: "points-attenuated").pixels
        let near = coveredPixels(attenuated, side: 64, columns: 0..<32)
        let far = coveredPixels(attenuated, side: 64, columns: 32..<64)
        #expect(near.count == 64)
        #expect(extent(near) == (SIMD2(12, 28), SIMD2(19, 35)))
        #expect(far.count == 16)
        #expect(extent(far) == (SIMD2(46, 30), SIMD2(49, 33)))
        #expect(isNearly(attenuated[30 * 64 + 14], SIMD4(1, 0.5, 0.25, 1)))

        // Pixel sizes ignore depth.
        let constant = try fixture.render(fixture.graph([
            try node(SIMD3(-2, 0, -4), .pixels(6)), try node(SIMD3(4, 0, -8), .pixels(6)),
        ])).pixels
        #expect(coveredPixels(constant, side: 64, columns: 0..<32).count == 36)
        #expect(coveredPixels(constant, side: 64, columns: 32..<64).count == 36)
    }

    @Test
    func `Per-point sizes and colors multiply the material`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        let cloud = try PointCloud(
            positions: [SIMD3(-2, 0, -4), SIMD3(4, 0, -8)],
            colors: [SIMD4(0, 1, 0, 1), SIMD4(1, 0, 1, 1)],
            sizes: [0.5, 2]
        )
        let renderer = try PointsRenderer(
            device: fixture.device, cloud: cloud, material: PointsMaterial(size: .world(1), color: SIMD4(1, 1, 1, 1))
        )
        let pixels = try fixture.render(fixture.graph([SceneNode(primitive: .points(renderer))])).pixels
        // 0.5 units at depth 4 is 4 pixels; 2 units at depth 8 is 8 pixels.
        #expect(coveredPixels(pixels, side: 64, columns: 0..<32).count == 16)
        #expect(coveredPixels(pixels, side: 64, columns: 32..<64).count == 64)
        #expect(isNearly(pixels[32 * 64 + 16], SIMD4(0, 1, 0, 1)))
        #expect(isNearly(pixels[32 * 64 + 48], SIMD4(1, 0, 1, 1)))
    }

    @Test
    func `Antialiased circular points cover their disc area`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        // 0.5 units at depth 2 is a 16-pixel disc, centered off the pixel grid.
        let center = fixture.point(pixel: SIMD2(60.3, 70.6), depth: 2)
        let renderer = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [center]),
            material: PointsMaterial(size: .world(0.5), shape: .circle, blending: .alpha)
        )
        let pixels = try fixture.render(fixture.graph([SceneNode(primitive: .points(renderer))]), name: "points-disc").pixels
        let area = pixels.reduce(Float(0)) { $0 + $1.w }
        #expect(abs(area - .pi * 64) / (.pi * 64) < 0.01, "disc area \(area)")
        // The corners of the bounding square stay empty.
        #expect(fixture.alpha(pixels, x: 53, y: 63) == 0)
        #expect(fixture.alpha(pixels, x: 60, y: 70) == 1)
    }

    @Test
    func `Alpha test discards map texels below the cutoff`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // A 2 × 2 map whose top row is transparent.
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 2, mipmapped: false)
        descriptor.usage = .shaderRead
        let map = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels: [UInt8] = [255, 255, 255, 0, 255, 255, 255, 0, 255, 0, 0, 255, 255, 0, 0, 255]
        map.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: texels, bytesPerRow: 8)
        let renderer = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0, 0, -4)]),
            material: PointsMaterial(size: .pixels(16), map: map, alphaTest: 0.5)
        )
        let pixels = try fixture.render(fixture.graph([SceneNode(primitive: .points(renderer))])).pixels
        // The square spans pixels 24...39; linear filtering crosses 0.5 alpha at its middle row.
        #expect(fixture.alpha(pixels, x: 32, y: 26) == 0)
        #expect(isNearly(pixels[37 * 64 + 32], SIMD4(1, 0, 0, 1)))
        #expect(coveredPixels(pixels, side: 64, columns: 0..<64).count == 16 * 8)
    }

    @Test
    func `Sprites stay parallel to the image plane for any camera orientation`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        let orientation = simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3<Float>(0.3, 1, -0.2)))
        let cameraTransform = SceneTransform(translation: SIMD3(1, 2, 3), rotation: orientation)
        let forward = orientation.act(SIMD3<Float>(0, 0, -1))
        let anchor = cameraTransform.translation + forward * 5
        func render(_ sprite: Sprite, attenuated: Bool = true, name: String? = nil) throws -> [SIMD2<Int>] {
            let renderer = try SpriteRenderer(
                device: fixture.device, sprites: [sprite],
                material: SpriteMaterial(sizeAttenuation: attenuated, blending: .opaque)
            )
            let pixels = try fixture.render(
                fixture.graph([SceneNode(primitive: .sprites(renderer))], cameraTransform: cameraTransform), name: name
            ).pixels
            return coveredPixels(pixels, side: 64, columns: 0..<64)
        }
        // 2.5 × 1.25 units at distance 5 is 16 × 8 pixels about the image center.
        let centered = try render(Sprite(position: anchor, size: SIMD2(2.5, 1.25)), name: "sprite-centered")
        #expect(centered.count == 128)
        #expect(extent(centered) == (SIMD2(24, 28), SIMD2(39, 35)))
        // A quarter turn swaps the extents.
        let turned = try render(Sprite(position: anchor, size: SIMD2(2.5, 1.25), rotation: .pi / 2))
        #expect(turned.count == 128)
        #expect(extent(turned) == (SIMD2(28, 24), SIMD2(35, 39)))
        // A top-left anchor hangs the sprite right and down from its position.
        let anchored = try render(Sprite(position: anchor, size: SIMD2(2.5, 1.25), center: .zero))
        #expect(extent(anchored) == (SIMD2(32, 32), SIMD2(47, 39)))
        // Pixel-sized sprites keep their size at any distance.
        let near = try render(Sprite(position: cameraTransform.translation + forward * 2, size: SIMD2(10, 6)), attenuated: false)
        let far = try render(Sprite(position: cameraTransform.translation + forward * 20, size: SIMD2(10, 6)), attenuated: false)
        #expect(near.count == 60 && far.count == 60)
        #expect(extent(near) == (SIMD2(27, 29), SIMD2(36, 34)))
    }

    @Test
    func `Sprite regions select texels from an atlas`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 1, mipmapped: false)
        descriptor.usage = .shaderRead
        let atlas = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels: [UInt8] = [255, 0, 0, 255, 0, 0, 255, 255]
        atlas.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: texels, bytesPerRow: 8)
        // The right texel, sampled at its center so linear filtering returns it exactly.
        let sprite = Sprite(position: SIMD3(0, 0, -4), size: SIMD2(1, 1), textureRegion: SIMD4(0.75, 0.5, 0, 0))
        let renderer = try SpriteRenderer(device: fixture.device, sprites: [sprite], material: SpriteMaterial(map: atlas))
        let pixels = try fixture.render(fixture.graph([SceneNode(primitive: .sprites(renderer))])).pixels
        #expect(isNearly(pixels[32 * 64 + 32], SIMD4(0, 0, 1, 1)))
    }

    @Test
    func `A zero per-point size hides the point in every mode and from picking`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        let hidden = SIMD3<Float>(-1, 0, -4)
        let shown = SIMD3<Float>(1, 0, -4)
        let cloud = try PointCloud(positions: [hidden, shown], sizes: [0, 1])
        let materials = [
            PointsMaterial(size: .pixels(4)),
            PointsMaterial(size: .pixels(4), shape: .circle),
            PointsMaterial(size: .pixels(4), blending: .alpha),
            PointsMaterial(size: .world(0.5), shape: .circle, blending: .additive),
        ]
        for material in materials {
            let renderer = try PointsRenderer(device: fixture.device, cloud: cloud, material: material)
            let graph = fixture.graph([SceneNode(primitive: .points(renderer))])
            let pixels = try fixture.render(graph).pixels
            #expect(coveredPixels(pixels, side: 64, columns: 0..<32).isEmpty, "\(material.size) \(material.blending)")
            #expect(!coveredPixels(pixels, side: 64, columns: 32..<64).isEmpty)
            func pick(_ point: SIMD3<Float>) throws -> PrimitiveHit? {
                try graph.pickPrimitive(at: fixture.pixel(point) / 64, drawableSize: SIMD2(64, 64), tolerance: .pixels(2))
            }
            #expect(try pick(hidden) == nil)
            #expect(try pick(shown)?.element == .point(1))
        }
    }

    @Test
    func `Additive sprites add their premultiplied colors`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // Two 8-pixel sprites 4 pixels apart overlap over the middle four columns.
        func render(_ blending: PrimitiveBlending) throws -> [SIMD4<Float>] {
            let renderer = try SpriteRenderer(
                device: fixture.device,
                sprites: [Sprite(position: SIMD3(-0.25, 0, -4), size: SIMD2(1, 1)), Sprite(position: SIMD3(0.25, 0, -4), size: SIMD2(1, 1))],
                material: SpriteMaterial(color: SIMD4(0.4, 0.2, 0.1, 0.5), blending: blending)
            )
            return try fixture.render(fixture.graph([SceneNode(primitive: .sprites(renderer))])).pixels
        }
        let additive = try render(.additive)
        // Each sprite adds (0.4, 0.2, 0.1) × 0.5; alpha still composites as source-over.
        #expect(isNearly(additive[32 * 64 + 32], SIMD4(0.4, 0.2, 0.1, 0.75)), "\(additive[32 * 64 + 32])")
        #expect(isNearly(additive[32 * 64 + 27], SIMD4(0.2, 0.1, 0.05, 0.5)))
        // Source-over darkens the first sprite under the second instead.
        let over = try render(.alpha)
        #expect(isNearly(over[32 * 64 + 32], SIMD4(0.3, 0.15, 0.075, 0.75)), "\(over[32 * 64 + 32])")
    }

    @Test
    func `Sprite alpha test discards map texels below the cutoff`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // A 2 × 1 map: a quarter-opaque white texel, then an opaque red one.
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 1, mipmapped: false)
        descriptor.usage = .shaderRead
        let map = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels: [UInt8] = [255, 255, 255, 64, 255, 0, 0, 255]
        map.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: texels, bytesPerRow: 8)
        func render(alphaTest: Float) throws -> [SIMD4<Float>] {
            let renderer = try SpriteRenderer(
                device: fixture.device, sprites: [Sprite(position: SIMD3(0, 0, -4), size: SIMD2(16, 8))],
                material: SpriteMaterial(map: map, sizeAttenuation: false, alphaTest: alphaTest)
            )
            return try fixture.render(fixture.graph([SceneNode(primitive: .sprites(renderer))])).pixels
        }
        // The sprite spans columns 24…39. Linear filtering ramps alpha from 0.25 at
        // u = 0.25 to 1 at u = 0.75, crossing 0.5 at u = 5/12, between columns 30 and 31.
        let tested = try render(alphaTest: 0.5)
        #expect(coveredPixels(tested, side: 64, columns: 0..<64).map(\.x).min() == 31)
        #expect(coveredPixels(tested, side: 64, columns: 0..<64).count == 9 * 8)
        #expect(isNearly(tested[32 * 64 + 38], SIMD4(1, 0, 0, 1)))
        let untested = try render(alphaTest: 0)
        #expect(abs(untested[32 * 64 + 25].w - 64 / 255) < 0.004)
    }

    @Test
    func `Sprite rotation turns counterclockwise on screen`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // A 20 × 2 pixel bar centered on the image, turned +45°.
        let renderer = try SpriteRenderer(
            device: fixture.device,
            sprites: [Sprite(position: SIMD3(0, 0, -4), size: SIMD2(20, 2), rotation: .pi / 4)],
            material: SpriteMaterial(sizeAttenuation: false, blending: .opaque)
        )
        let pixels = try fixture.render(fixture.graph([SceneNode(primitive: .sprites(renderer))]), name: "sprite-rotated").pixels
        // Counterclockwise with +y up lays the bar from lower left to upper right; rows grow downward.
        #expect(fixture.alpha(pixels, x: 36, y: 27) == 1 && fixture.alpha(pixels, x: 27, y: 36) == 1)
        #expect(fixture.alpha(pixels, x: 36, y: 36) == 0 && fixture.alpha(pixels, x: 27, y: 27) == 0)
    }

    @Test
    func `Points and sprites reject invalid input`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cloud = try PointCloud(positions: [.zero])
        #expect(throws: PrimitiveRendererError.size) {
            try PointsRenderer(device: device, cloud: cloud, material: PointsMaterial(size: .world(-1)))
        }
        #expect(throws: PrimitiveRendererError.alphaTest) {
            try PointsRenderer(device: device, cloud: cloud, material: PointsMaterial(alphaTest: 2))
        }
        #expect(throws: PrimitiveGeometryError.pointCount) { try PointCloud(positions: []) }
        #expect(throws: PrimitiveGeometryError.invalidSize(0)) { try PointCloud(positions: [.zero], sizes: [-1]) }
        #expect(throws: PrimitiveGeometryError.invalidColor(0)) {
            try PointCloud(positions: [.zero], colors: [SIMD4(1, 1, 1, 1.5)])
        }
        #expect(throws: PrimitiveRendererError.spriteCount) { try SpriteRenderer(device: device, sprites: []) }
        #expect(throws: PrimitiveRendererError.invalidSprite(0)) {
            try SpriteRenderer(device: device, sprites: [Sprite(position: .zero, size: SIMD2(-1, 1))])
        }
    }

    @Test
    func `A color graph replaces point, sprite, and line color without a new pipeline`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        let graph = try SurfaceShaderGraph { s in
            .init(baseColor: s.uniform("Tint", SIMD3(0, 1, 0)))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let points = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(-2, 0, -4)]),
            material: PointsMaterial(
                size: .pixels(8), color: SIMD4(1, 0, 0, 1), colorGraph: graph, colorUniforms: uniforms
            )
        )
        let sprites = try SpriteRenderer(
            device: fixture.device, sprites: [Sprite(position: SIMD3(2, 0, -4), size: SIMD2(repeating: 1))],
            material: SpriteMaterial(color: SIMD4(1, 0, 0, 1), colorGraph: graph, colorUniforms: uniforms)
        )
        let line = try LineRenderer(
            device: fixture.device,
            geometry: try LineGeometry(positions: [SIMD3(-1, 0, -4), SIMD3(1, 0, -4)]),
            material: LineMaterial(
                width: .pixels(4), cap: .butt, color: SIMD4(1, 0, 0, 1), blending: .opaque,
                colorGraph: graph, colorUniforms: uniforms
            )
        )
        let pixels = try fixture.render(fixture.graph([
            SceneNode(primitive: .points(points)),
            SceneNode(primitive: .sprites(sprites)),
            SceneNode(primitive: .lines(line)),
        ])).pixels
        #expect(isNearly(pixels[32 * 64 + 16], SIMD4(0, 1, 0, 1)))
        #expect(isNearly(pixels[32 * 64 + 48], SIMD4(0, 1, 0, 1)))
        #expect(isNearly(pixels[32 * 64 + 32], SIMD4(0, 1, 0, 1)))
        try uniforms.set("Tint", to: SIMD3(0, 0, 1))
        let edited = try fixture.render(fixture.graph([
            SceneNode(primitive: .points(points)),
            SceneNode(primitive: .sprites(sprites)),
            SceneNode(primitive: .lines(line)),
        ])).pixels
        #expect(isNearly(edited[32 * 64 + 16], SIMD4(0, 0, 1, 1)))
        #expect(isNearly(edited[32 * 64 + 48], SIMD4(0, 0, 1, 1)))
        #expect(isNearly(edited[32 * 64 + 32], SIMD4(0, 0, 1, 1)))
        #expect(throws: PrimitiveRendererError.colorGraph("A primitive color graph reads UV0, the incoming color, and uniforms.")) {
            _ = try PointsRenderer(
                device: fixture.device, cloud: try PointCloud(positions: [.zero]),
                material: PointsMaterial(
                    colorGraph: try SurfaceShaderGraph { s in .init(baseColor: s.worldPosition) }
                )
            )
        }
    }

    @Test
    func `A point graph replaces the material size and edits without a new pipeline`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        let graph = try SurfaceShaderGraph { s in
            .init(pointSize: s.uniform("Size", Float(6)))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let renderer = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(-2, 0, -4)]),
            material: PointsMaterial(
                size: .pixels(2), color: SIMD4(1, 0, 0, 1), colorGraph: graph, colorUniforms: uniforms
            )
        )
        let node = SceneNode(primitive: .points(renderer))
        let six = try fixture.render(fixture.graph([node])).pixels
        #expect(coveredPixels(six, side: 64, columns: 0..<64).count == 36)
        try uniforms.set("Size", to: 4)
        let four = try fixture.render(fixture.graph([node])).pixels
        #expect(coveredPixels(four, side: 64, columns: 0..<64).count == 16)
        try uniforms.set("Size", to: 0)
        let hidden = try fixture.render(fixture.graph([node])).pixels
        #expect(coveredPixels(hidden, side: 64, columns: 0..<64).isEmpty)
        #expect(throws: PhysicallyBasedMaterial.Failure.pointSizeOutput) {
            try PhysicallyBasedMaterial(surfaceGraph: graph)
        }
        #expect(throws: PrimitiveRendererError.colorGraph("A primitive color graph reads base color and opacity.")) {
            _ = try SpriteRenderer(
                device: fixture.device, sprites: [Sprite(position: .zero, size: SIMD2(1, 1))],
                material: SpriteMaterial(colorGraph: graph)
            )
        }
    }
}
