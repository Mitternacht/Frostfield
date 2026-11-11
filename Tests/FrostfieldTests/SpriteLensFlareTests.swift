import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SpriteLensFlareTests {
    @Test func `Authored sprites keep their axis positions, pixel sizes, UV orientation and HDR blend`() throws {
        let gpu = try PostProcessFixture()
        let width = 64, height = 32
        let base = SIMD4<Float>(0.125, 0.25, 0.5, 0.625)
        let source = try gpu.color(width: width, height: height) { _, _ in base }
        let depth = try gpu.depth(width: width, height: height) { _, _ in 1 }
        let target = try gpu.color(width: width, height: height)
        let hot = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(8, 2, 0, 0.5) }
        let split = try gpu.color(width: 2, height: 2) { _, y in y == 0 ? SIMD4(0, 0, 8, 0.25) : SIMD4(0, 4, 0, 0.25) }
        let green = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(0, 4, 0, 1) }
        let srgb = try srgbPixel(device: gpu.device)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 30)
        let light = try #require(try LensFlare.Light(
            worldPosition: SIMD3(-3.875, 1.875, -4), camera: camera, cameraTransform: .init(), aspectRatio: 2, intensity: 1.5
        ))
        #expect(simd_distance(light.position, SIMD2(16.5 / 64, 8.5 / 32)) < 0.000001)
        let effect = try SpriteLensFlare(device: gpu.device, elements: [
            .init(texture: hot, size: 8, color: SIMD3(1, 0.5, 1)),
            .init(texture: split, size: 8, distance: 0.5),
            .init(texture: green, size: 4, distance: 1),
            .init(texture: srgb, size: 4, color: SIMD3(2, 1, 1))
        ], light: light)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(effect)])
        try gpu.run { try chain.encode(source: source, depth: depth, destination: target, commandBuffer: $0) }
        let pixels = try gpu.read(target)
        let decoded = pow((Float(128) / 255 + 0.055) / 1.055, 2.4)
        let overlap = base + SIMD4(6 + decoded * (128.0 / 255) * 3, 0.75, 0, 0)
        #expect(PostProcessFixture.close(pixels[8 * width + 16], overlap, tolerance: 0.006))
        #expect(pixels[8 * width + 19] == base + SIMD4(6, 0.75, 0, 0))
        #expect(pixels[8 * width + 21] == base)
        #expect(pixels[13 * width + 32] == base + SIMD4(0, 0, 3, 0))
        #expect(pixels[18 * width + 32] == base + SIMD4(0, 1.5, 0, 0))
        #expect(pixels[23 * width + 47] == base + SIMD4(0, 6, 0, 0))
        #expect(pixels.allSatisfy { $0.w == base.w })
    }

    @Test(arguments: [MeshDepthMode.forward, .reversed])
    func `Actual scene depth hides nearer flares and scales partial visibility`(mode: MeshDepthMode) throws {
        let gpu = try PostProcessFixture()
        let size = 24
        let source = try gpu.color(width: size, height: size) { _, _ in SIMD4(0.25, 0.5, 0.75, 0.5) }
        let target = try gpu.color(width: size, height: size)
        let dummy = try gpu.color(width: size, height: size)
        let texture = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(8, 0, 0, 0.5) }
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 30)
        let light = try #require(try LensFlare.Light(
            worldPosition: SIMD3(0, 0, -4), camera: camera, cameraTransform: .init(), aspectRatio: 1, depthMode: mode
        ))
        let effect = try SpriteLensFlare(
            device: gpu.device, elements: [.init(texture: texture, size: 6)], light: light,
            depthMode: mode, occlusionRadius: 4, occlusionSamples: 2
        )
        let wall = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 16, height: 16), depthMode: mode
        )
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: size, height: size, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        let depth = try #require(gpu.device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = dummy
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .dontCare
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = mode == .forward ? 1 : 0
        func composite(wallX: Float, wallZ: Float) throws -> SIMD4<Float> {
            let blocker = SceneNode(transform: .init(translation: SIMD3(wallX, 0, wallZ)), mesh: wall)
            let scene = SceneGraph(roots: [blocker], camera: camera)
            try gpu.run { command in
                try scene.encode(pass: pass, commandBuffer: command, depthMode: mode)
                try effect.encode(source: source, depth: depth, destination: target, commandBuffer: command)
            }
            return try gpu.read(target)[12 * size + 12]
        }
        #expect(try composite(wallX: 0, wallZ: -2) == SIMD4(0.25, 0.5, 0.75, 0.5))
        #expect(try composite(wallX: 0, wallZ: -6) == SIMD4(4.25, 0.5, 0.75, 0.5))
        #expect(try composite(wallX: -8, wallZ: -2) == SIMD4(2.25, 0.5, 0.75, 0.5))
    }

    @Test func `Sprite targets cannot alias textures and a light behind the camera bypasses exactly`() throws {
        let gpu = try PostProcessFixture()
        let source = try gpu.color(width: 8, height: 8) { _, _ in SIMD4(2, 1, 0.5, 0.75) }
        let target = try gpu.color(width: 8, height: 8)
        let depth = try gpu.depth(width: 8, height: 8) { _, _ in 1 }
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let behind = try LensFlare.Light(worldPosition: SIMD3(0, 0, 2), camera: camera, cameraTransform: .init(), aspectRatio: 1)
        #expect(behind == nil)
        let effect = try SpriteLensFlare(device: gpu.device, elements: [.init(texture: source)], light: behind)
        try effect.prepare(pixelFormat: .rgba16Float)
        try gpu.run { try effect.encode(source: source, depth: depth, destination: target, commandBuffer: $0) }
        #expect(try gpu.read(source) == gpu.read(target))

        let alias = try SpriteLensFlare(device: gpu.device, elements: [.init(texture: target)], light: .init(position: SIMD2(0.5, 0.5)))
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.destination) {
            try alias.encode(source: source, depth: depth, destination: target, commandBuffer: command)
        }
        #expect(throws: PostProcessEffectError.missingDepth) {
            try effect.encode(source: source, depth: nil, destination: target, commandBuffer: command)
        }
    }

    @Test func `A one-pixel sprite samples the authored minification mip`() throws {
        let gpu = try PostProcessFixture()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 8, height: 8, mipmapped: true)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let sprite = try #require(gpu.device.makeTexture(descriptor: descriptor))
        for mip in 0..<sprite.mipmapLevelCount {
            let side = 8 >> mip
            let value: SIMD4<Float16> = mip == 3 ? SIMD4(0, 0, 5, 0.5) : SIMD4(5, 0, 0, 1)
            let texels = Array(repeating: value, count: side * side)
            texels.withUnsafeBytes { bytes in
                sprite.replace(region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: mip,
                               withBytes: bytes.baseAddress!, bytesPerRow: side * MemoryLayout<SIMD4<Float16>>.stride)
            }
        }
        let source = try gpu.color(width: 8, height: 8) { _, _ in SIMD4(0, 0, 0, 1) }
        let depth = try gpu.depth(width: 8, height: 8) { _, _ in 1 }
        let target = try gpu.color(width: 8, height: 8)
        let effect = try SpriteLensFlare(device: gpu.device, elements: [.init(texture: sprite)],
                                         light: .init(position: SIMD2(repeating: 4.5 / 8)), occlusionRadius: 0)
        try gpu.run { try effect.encode(source: source, depth: depth, destination: target, commandBuffer: $0) }
        let pixels = try gpu.read(target)
        #expect(pixels[4 * 8 + 4] == SIMD4(0, 0, 2.5, 1))
        #expect(pixels[4 * 8 + 3] == SIMD4(0, 0, 0, 1))
    }

    private func srgbPixel(device: any MTLDevice) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let rgba: [UInt8] = [128, 0, 0, 128]
        rgba.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 4)
        }
        return texture
    }
}
