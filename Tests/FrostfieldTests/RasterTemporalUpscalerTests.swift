import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RasterTemporalUpscalerTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Native temporal scaling follows moving premultiplied color and preserves disocclusion")
    func nativeMotionAndAlpha() throws {
        let size = SIMD2(8, 4)
        let bright = SIMD4<Float>(3, 1.5, 0.75, 0.75)
        let dark = SIMD4<Float>(1, 0.5, 0.25, 0.25)
        let old = try gpu.color(width: size.x, height: size.y) { x, _ in (2...3).contains(x) ? bright : .zero }
        let new = try gpu.color(width: size.x, height: size.y) { x, _ in
            x == 4 || x == 7 ? bright : (5...6).contains(x) ? dark : .zero
        }
        let oldDepth = try gpu.depth(width: size.x, height: size.y) { x, _ in (2...3).contains(x) ? 0.25 : 0.75 }
        let newDepth = try gpu.depth(width: size.x, height: size.y) { x, _ in (4...7).contains(x) ? 0.25 : 0.75 }
        let noHistory = try gpu.color(width: size.x, height: size.y) { _, _ in .zero }
        let vectors = try gpu.color(width: size.x, height: size.y) { x, _ in
            (5...6).contains(x) ? SIMD4(-3, 0, 0.25, 1) : SIMD4(0, 0, 0.75, 1)
        }
        let upscaler = try RasterTemporalUpscaler(
            queue: gpu.queue, inputWidth: size.x, inputHeight: size.y, outputWidth: 16, outputHeight: 8
        )
        #expect(upscaler.backend == .native)
        #expect(upscaler.outputTexture.usage.contains(.shaderRead))
        let first = try render(upscaler, source: old, depth: oldDepth, motion: noHistory)
        #expect(!first.frame.reusedHistory)
        let second = try render(upscaler, source: new, depth: newDepth, motion: vectors)
        #expect(second.frame.reusedHistory)
        #expect(second.pixels[4 * 16 + 11] == SIMD4(2, 1, 0.5, 0.5))
        #expect(second.pixels[4 * 16 + 5] == .zero)
        upscaler.reset()
        #expect(upscaler.nextJitter == TemporalAntiAliasing.jitter(at: 0))
        let reset = try render(upscaler, source: new, depth: newDepth, motion: vectors)
        #expect(!reset.frame.reusedHistory)
        #expect(reset.pixels[4 * 16 + 11] == dark)
    }

    @Test("Opaque MetalFX upscales moving HDR geometry when supported and otherwise selects native")
    func hardwareOrFallback() throws {
        let width = 48, height = 32
        func make(_ backend: RasterTemporalUpscaler.Backend? = nil) throws -> RasterTemporalUpscaler {
            try RasterTemporalUpscaler(
                queue: gpu.queue, inputWidth: width, inputHeight: height, outputWidth: width * 2, outputHeight: height * 2,
                colorMode: .opaque, preferredBackend: backend
            )
        }
        guard RasterTemporalUpscaler.isMetalFXSupported(device: gpu.device) else {
            #expect(try make().backend == .native)
            #expect(throws: RasterTemporalUpscaler.Failure.unsupported) { try make(.metalFX) }
            return
        }
        let tiny = try RasterTemporalUpscaler(
            queue: gpu.queue, inputWidth: 1, inputHeight: 1, outputWidth: 2, outputHeight: 2, colorMode: .opaque
        )
        #expect(tiny.backend == .native)
        #expect(throws: RasterTemporalUpscaler.Failure.scaler) {
            try RasterTemporalUpscaler(
                queue: gpu.queue, inputWidth: 1, inputHeight: 1, outputWidth: 2, outputHeight: 2,
                colorMode: .opaque, preferredBackend: .metalFX
            )
        }
        let upscaler = try make()
        #expect(upscaler.backend == .metalFX)
        #expect(upscaler.outputTexture.pixelFormat == .rgba16Float)
        let bright = SIMD4<Float>(2, 0.5, 0.25, 1)
        let background = SIMD4<Float>(0.125, 0.125, 0.125, 1)
        let prior = try gpu.color(width: width, height: height) { x, _ in (8...17).contains(x) ? bright : background }
        let moved = try gpu.color(width: width, height: height) { x, _ in (26...35).contains(x) ? bright : background }
        let priorDepth = try gpu.depth(width: width, height: height) { x, _ in (8...17).contains(x) ? 0.25 : 0.75 }
        let movedDepth = try gpu.depth(width: width, height: height) { x, _ in (26...35).contains(x) ? 0.25 : 0.75 }
        let absent = try gpu.color(width: width, height: height) { _, _ in .zero }
        let firstJitter = upscaler.nextJitter
        let first = try render(upscaler, source: prior, depth: priorDepth, motion: absent, jitter: firstJitter)
        #expect(!first.frame.reusedHistory)
        let nextJitter = upscaler.nextJitter
        let jitterDelta = firstJitter - nextJitter
        let vectors = try gpu.color(width: width, height: height) { x, _ in
            (26...35).contains(x) ? SIMD4(-18 + jitterDelta.x, jitterDelta.y, 0.25, 1) : SIMD4(jitterDelta.x, jitterDelta.y, 0.75, 1)
        }
        let second = try render(upscaler, source: moved, depth: movedDepth, motion: vectors, jitter: nextJitter)
        #expect(second.frame.reusedHistory)
        let oldCenter = 32 * 96 + 24, newCenter = 32 * 96 + 60
        #expect(first.pixels[oldCenter].x > 1.75 && first.pixels[newCenter].x < 0.25)
        #expect(second.pixels[oldCenter].x < 0.25 && second.pixels[newCenter].x > 1.75)
        #expect(second.pixels.allSatisfy { $0.x.isFinite && $0.w == 1 })

        upscaler.reset()
        let restarted = try render(upscaler, source: prior, depth: priorDepth, motion: absent, jitter: firstJitter)
        #expect(!restarted.frame.reusedHistory)
        #expect(PostProcessFixture.close(restarted.pixels[oldCenter], first.pixels[oldCenter], tolerance: 0.02))
    }

    @Test("Upscaling validates resource ownership and discards only its outstanding unsubmitted frame")
    func validationAndDiscard() throws {
        let upscaler = try RasterTemporalUpscaler(queue: gpu.queue, inputWidth: 5, inputHeight: 3, outputWidth: 5, outputHeight: 3)
        let color = try gpu.color(width: 5, height: 3) { _, _ in SIMD4(0.25, 0.5, 1, 0.5) }
        let motion = try gpu.color(width: 5, height: 3) { _, _ in .zero }
        let depth = try gpu.depth(width: 5, height: 3) { _, _ in 0.5 }
        let wrongMotion = try gpu.color(width: 5, height: 3, pixelFormat: .rgba8Unorm)
        let wrongSize = try gpu.color(width: 6, height: 3)
        let foreignQueue = try #require(gpu.device.makeCommandQueue())
        let foreign = try #require(foreignQueue.makeCommandBuffer())
        let pending = try #require(gpu.queue.makeCommandBuffer())
        let second = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: RasterTemporalUpscaler.Failure.dimensions) {
            try RasterTemporalUpscaler(queue: gpu.queue, inputWidth: 5, inputHeight: 3, outputWidth: 10, outputHeight: 7)
        }
        #expect(throws: RasterTemporalUpscaler.Failure.colorMode) {
            try RasterTemporalUpscaler(
                queue: gpu.queue, inputWidth: 5, inputHeight: 3, outputWidth: 10, outputHeight: 6, preferredBackend: .metalFX
            )
        }
        func encode(
            _ command: any MTLCommandBuffer, source: (any MTLTexture)? = nil,
            vectors: (any MTLTexture)? = nil, jitter: SIMD2<Float> = .zero
        ) throws -> RasterTemporalUpscaler.Frame {
            try upscaler.encode(source: source ?? color, depth: depth, motionTexture: vectors ?? motion, jitter: jitter, commandBuffer: command)
        }
        #expect(throws: RasterTemporalUpscaler.Failure.commandBuffer) { try encode(foreign) }
        #expect(throws: RasterTemporalUpscaler.Failure.motion) { try encode(pending, vectors: wrongMotion) }
        #expect(throws: RasterTemporalUpscaler.Failure.dimensions) { try encode(pending, vectors: wrongSize) }
        #expect(throws: RasterTemporalUpscaler.Failure.jitter) { try encode(pending, jitter: SIMD2(2, 0)) }
        #expect(throws: RasterTemporalUpscaler.Failure.alias) { try encode(pending, source: upscaler.outputTexture) }
        #expect(try !encode(pending).reusedHistory)
        #expect(throws: RasterTemporalUpscaler.Failure.frameInFlight) { try encode(second) }
        #expect(throws: RasterTemporalUpscaler.Failure.discard) { try upscaler.discardUnsubmittedFrame(second) }
        try upscaler.discardUnsubmittedFrame(pending)
        let rendered = try render(upscaler, source: color, depth: depth, motion: motion)
        #expect(!rendered.frame.reusedHistory)
        #expect(try rendered.pixels == gpu.read(color))
    }

    private func render(
        _ scaler: RasterTemporalUpscaler, source: any MTLTexture, depth: any MTLTexture,
        motion: any MTLTexture, jitter: SIMD2<Float> = .zero
    ) throws -> (frame: RasterTemporalUpscaler.Frame, pixels: [SIMD4<Float>]) {
        let output = scaler.outputTexture
        let copy = try gpu.color(width: output.width, height: output.height)
        var frame: RasterTemporalUpscaler.Frame?
        try gpu.run { command in
            frame = try scaler.encode(source: source, depth: depth, motionTexture: motion, jitter: jitter, commandBuffer: command)
            let blit = try #require(command.makeBlitCommandEncoder())
            blit.copy(
                from: output, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: output.width, height: output.height, depth: 1),
                to: copy, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin()
            )
            blit.endEncoding()
        }
        return (try #require(frame), try gpu.read(copy))
    }
}
