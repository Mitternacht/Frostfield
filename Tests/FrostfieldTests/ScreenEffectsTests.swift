import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ScreenEffectsTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let effects: ScreenEffects

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        self.device = device
        self.queue = queue
        effects = try ScreenEffects(queue: queue, destinationPixelFormat: .rgba16Float)
    }

    @Test("Threshold bloom spreads across both axes and keeps in-flight scratch through resize")
    func bloomAndResize() throws {
        let small = try impulse(size: 9)
        let large = try impulse(size: 17)
        let firstTarget = try texture(size: 9)
        let secondTarget = try texture(size: 17)
        let first = try #require(queue.makeCommandBuffer())
        let second = try #require(queue.makeCommandBuffer())
        let settings = ScreenEffects.Settings(bloomThreshold: 1, bloomStrength: 1, bloomRadius: 4)

        // Resize the cached scratch before the first buffer is even submitted.
        try effects.encode(source: small, destination: firstTarget, settings: settings, commandBuffer: first)
        try effects.encode(source: large, destination: secondTarget, settings: settings, commandBuffer: second)
        first.commit()
        second.commit()
        first.waitUntilCompleted()
        second.waitUntilCompleted()
        try #require(first.status == .completed)
        try #require(second.status == .completed)

        for target in [firstTarget, secondTarget] {
            let size = target.width
            let pixels = try read(target)
            let center = size / 2
            #expect(channel(pixels, size: size, x: center, y: center) > 4)
            #expect(channel(pixels, size: size, x: center - 2, y: center) > 0.01)
            #expect(channel(pixels, size: size, x: center, y: center - 2) > 0.01)
            #expect(channel(pixels, size: size, x: center, y: center - 2, component: 3) == 1)
        }

        let bypass = try render(small, settings: .init())
        let belowThreshold = try render(
            small, settings: .init(bloomThreshold: 5, bloomStrength: 1, bloomRadius: 4)
        )
        #expect(bypass == belowThreshold)
        #expect(channel(bypass, size: 9, x: 4, y: 4) == 4)
        #expect(channel(bypass, size: 9, x: 2, y: 4) == 0)
    }

    @Test("Color fringing displaces red and blue, and grain repeats by seed")
    func colorAndGrain() throws {
        let size = 16
        let ramp = try texture(size: size) { x, _ in
            let fraction = Float(x) / Float(size - 1)
            return SIMD4(fraction, 0.4, 1 - fraction, 1)
        }
        let fringing = try render(ramp, settings: .init(chromaticAberration: 4))
        #expect(channel(fringing, size: size, x: 13, y: 8) > 0.94)
        #expect(channel(fringing, size: size, x: 13, y: 8, component: 2) > 0.23)
        #expect(abs(channel(fringing, size: size, x: 13, y: 8, component: 1) - 0.4) < 0.001)
        #expect(channel(fringing, size: size, x: 13, y: 8, component: 3) == 1)

        let flat = try texture(size: size) { _, _ in SIMD4(0.2, 0.2, 0.2, 0.5) }
        let grain = try render(flat, settings: .init(grainStrength: 0.12, grainSeed: 1))
        let repeated = try render(flat, settings: .init(grainStrength: 0.12, grainSeed: 1))
        let different = try render(flat, settings: .init(grainStrength: 0.12, grainSeed: 2))
        #expect(grain == repeated)
        #expect(grain != different)
        #expect(channel(grain, size: size, x: 8, y: 8, component: 3) == 0.5)
    }

    @Test("Encoding rejects the wrong queue, aliased textures, and nonfinite settings")
    func rejectsUnsafeInput() throws {
        let source = try impulse(size: 9)
        let destination = try texture(size: 9)
        let otherQueue = try #require(device.makeCommandQueue())
        let wrongBuffer = try #require(otherQueue.makeCommandBuffer())
        #expect(throws: ScreenEffects.Failure.commandBuffer) {
            try effects.encode(source: source, destination: destination, commandBuffer: wrongBuffer)
        }
        let buffer = try #require(queue.makeCommandBuffer())
        #expect(throws: ScreenEffects.Failure.incompatibleTextures) {
            try effects.encode(source: source, destination: source, commandBuffer: buffer)
        }
        #expect(throws: ScreenEffects.Failure.invalidSettings) {
            try effects.encode(
                source: source, destination: destination,
                settings: .init(grainStrength: .nan), commandBuffer: buffer
            )
        }
    }

    private func impulse(size: Int) throws -> any MTLTexture {
        try texture(size: size) { x, y in
            x == size / 2 && y == size / 2 ? SIMD4(4, 2, 1, 1) : SIMD4(0, 0, 0, 1)
        }
    }

    private func texture(size: Int, pixel: ((Int, Int) -> SIMD4<Float>)? = nil) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .renderTarget]
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        if let pixel {
            var pixels: [Float16] = []
            pixels.reserveCapacity(size * size * 4)
            for y in 0..<size {
                for x in 0..<size {
                    let rgba = pixel(x, y)
                    pixels += [Float16(rgba.x), Float16(rgba.y), Float16(rgba.z), Float16(rgba.w)]
                }
            }
            try pixels.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0,
                    withBytes: try #require(bytes.baseAddress), bytesPerRow: size * 8
                )
            }
        }
        return texture
    }

    private func render(_ source: any MTLTexture, settings: ScreenEffects.Settings) throws -> [Float16] {
        let destination = try texture(size: source.width)
        let buffer = try #require(queue.makeCommandBuffer())
        try effects.encode(source: source, destination: destination, settings: settings, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        return try read(destination)
    }

    private func read(_ texture: any MTLTexture) throws -> [Float16] {
        var pixels = Array(repeating: Float16.zero, count: texture.width * texture.height * 4)
        try pixels.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 8,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return pixels
    }

    private func channel(_ pixels: [Float16], size: Int, x: Int, y: Int, component: Int = 0) -> Float {
        Float(pixels[(y * size + x) * 4 + component])
    }
}
