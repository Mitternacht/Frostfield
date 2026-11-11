import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessChainTests {
    private let gpu: PostProcessFixture
    private let swizzle: PostProcessGraph
    private let gain: PostProcessGraph

    init() throws {
        gpu = try PostProcessFixture()
        swizzle = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).swizzle("bgra", as: SIMD4<Float>.self) }
        gain = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * p.uniform("Gain", Float(1)) }
    }

    private static func frame(_ x: Int, _ y: Int) -> SIMD4<Float> {
        SIMD4(Float(x) / 8, Float(y) / 8, 0.5 + Float(x + y) / 16, (x + y) % 2 == 0 ? 1 : 0.75)
    }

    private func expectFrame(
        _ pixels: [SIMD4<Float>], width: Int, tolerance: Float = 1e-3,
        sourceLocation: SourceLocation = #_sourceLocation, _ expected: (SIMD4<Float>) -> SIMD4<Float>
    ) {
        for (index, pixel) in pixels.enumerated() {
            let want = expected(Self.frame(index % width, index / width))
            #expect(PostProcessFixture.close(pixel, want, tolerance: tolerance), "pixel \(index): \(pixel) ≠ \(want)",
                    sourceLocation: sourceLocation)
        }
    }

    @Test("A two-stage chain swizzles channels and then applies a uniform gain")
    func swizzleThenGain() throws {
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(swizzle), .graph(gain)])
        #expect(chain.stageCount == 2)
        #expect(try chain.uniforms(at: 0)?.declarations.isEmpty == true)
        try #require(try chain.uniforms(at: 1)).set("Gain", to: 2)
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }

        expectFrame(try gpu.read(destination), width: 4) { SIMD4($0.z * 2, $0.y * 2, $0.x * 2, $0.w) }
    }

    @Test("Composer edits preserve stage settings and do not affect already encoded passes")
    func liveStageEdits() throws {
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(gain), .graph(swizzle)])
        let uniforms = try #require(try chain.uniforms(at: 0))
        try uniforms.set("Gain", to: 2)
        try chain.setEnabled(false, at: 1)
        let redGain = try PostProcessGraph { p in
            p.sceneColor(at: p.screenUV) * p.constant(SIMD4<Float>(2, 1, 1, 1))
        }
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4)
        let saved = try gpu.color(width: 4, height: 4)
        try chain.insertEffect(.graph(redGain), at: 0)
        try chain.insertEffect(.save(to: saved), at: chain.stageCount)
        #expect(chain.stageCount == 4)
        #expect(try chain.uniforms(at: 1) === uniforms)
        #expect(try chain.isEnabled(at: 2) == false)

        try gpu.run { command in
            try chain.encode(source: source, destination: destination, commandBuffer: command)
            try chain.removeEffect(at: 0)
            try chain.removeEffect(at: 2)
        }
        let first = try gpu.read(destination)
        expectFrame(first, width: 4) { SIMD4($0.x * 4, $0.y * 2, $0.z * 2, $0.w) }
        #expect(try gpu.read(saved) == first)
        #expect(chain.stageCount == 2)
        #expect(try chain.uniforms(at: 0) === uniforms)
        #expect(try chain.isEnabled(at: 1) == false)

        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        expectFrame(try gpu.read(destination), width: 4) { SIMD4($0.x * 2, $0.y * 2, $0.z * 2, $0.w) }
        #expect(try gpu.read(saved) == first)
    }

    @Test("Failed stage edits preserve the existing composer")
    func stageEditValidation() throws {
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(gain)])
        let uniforms = try #require(try chain.uniforms(at: 0))
        #expect(throws: PostProcessChain.Failure.stageIndex(-1)) { try chain.insertEffect(.graph(swizzle), at: -1) }
        #expect(throws: PostProcessChain.Failure.stageIndex(2)) { try chain.insertEffect(.graph(swizzle), at: 2) }
        #expect(throws: PostProcessChain.Failure.stageIndex(1)) { try chain.removeEffect(at: 1) }
        #expect(throws: ColorGrade.Failure.invalidStrength) {
            try chain.insertEffect(.colorGrade(strength: 2), at: 0)
        }
        #expect(chain.stageCount == 1)
        #expect(try chain.uniforms(at: 0) === uniforms)
        let limit = PostProcessChain.maximumStageCount
        let full = try PostProcessChain(queue: gpu.queue, effects: Array(repeating: .graph(gain), count: limit))
        #expect(throws: PostProcessChain.Failure.tooManyStages(limit: limit)) {
            try full.insertEffect(.graph(swizzle), at: limit)
        }
        #expect(full.stageCount == limit)
    }

    @Test("Bypassed stages pass their input through, and a fully bypassed chain copies the source")
    func bypass() throws {
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(swizzle), .graph(gain)])
        try #require(try chain.uniforms(at: 1)).set("Gain", to: 2)
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4)

        try chain.setEnabled(false, at: 0)
        #expect(try !chain.isEnabled(at: 0) && chain.isEnabled(at: 1))
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        expectFrame(try gpu.read(destination), width: 4) { SIMD4($0.x * 2, $0.y * 2, $0.z * 2, $0.w) }

        try chain.setEnabled(true, at: 0)
        try chain.setEnabled(false, at: 1)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        expectFrame(try gpu.read(destination), width: 4) { SIMD4($0.z, $0.y, $0.x, $0.w) }

        try chain.setEnabled(false, at: 0)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination) == gpu.read(source))
        // A single enabled stage writes the destination directly.
        #expect(chain.intermediateAllocationCount == 0)

        let display = try PostProcessChain(queue: gpu.queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.graph(gain)])
        try display.setEnabled(false, at: 0)
        let displayTarget = try gpu.color(width: 4, height: 4, pixelFormat: .bgra8Unorm_srgb)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessChain.Failure.bypassCopy) {
            try display.encode(source: source, destination: displayTarget, commandBuffer: buffer)
        }
    }

    @Test("Ping-pong intermediates are reused across encodes and reallocated when the size changes")
    func intermediateReuse() throws {
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(swizzle), .graph(gain), .graph(swizzle)])
        let small = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let smallTarget = try gpu.color(width: 4, height: 4)
        let large = try gpu.color(width: 8, height: 6, pixel: Self.frame)
        let largeTarget = try gpu.color(width: 8, height: 6)
        #expect(chain.intermediateAllocationCount == 0)

        for _ in 0..<3 {
            try gpu.run { try chain.encode(source: small, destination: smallTarget, commandBuffer: $0) }
        }
        #expect(chain.intermediateAllocationCount == 2)
        try gpu.run { try chain.encode(source: large, destination: largeTarget, commandBuffer: $0) }
        #expect(chain.intermediateAllocationCount == 4)
        // Fewer enabled stages reuse the existing intermediates.
        try chain.setEnabled(false, at: 1)
        try gpu.run { try chain.encode(source: large, destination: largeTarget, commandBuffer: $0) }
        #expect(chain.intermediateAllocationCount == 4)
        // Two swizzles cancel.
        expectFrame(try gpu.read(largeTarget), width: 8) { $0 }

        // Both sizes encoded into one buffer before either is submitted stay valid.
        try chain.setEnabled(true, at: 1)
        try #require(try chain.uniforms(at: 1)).set("Gain", to: 0.5)
        try gpu.run { buffer in
            try chain.encode(source: small, destination: smallTarget, commandBuffer: buffer)
            try chain.encode(source: large, destination: largeTarget, commandBuffer: buffer)
        }
        #expect(chain.intermediateAllocationCount == 8)
        expectFrame(try gpu.read(smallTarget), width: 4) { SIMD4($0.x * 0.5, $0.y * 0.5, $0.z * 0.5, $0.w) }
        expectFrame(try gpu.read(largeTarget), width: 8) { SIMD4($0.x * 0.5, $0.y * 0.5, $0.z * 0.5, $0.w) }
    }

    @Test("Every graph stage receives the scene depth")
    func depthForwarding() throws {
        let writeDepth = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            return p.vector(p.sceneDepth(at: p.screenUV), color.g, color.b, color.a)
        }
        let addDepth = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            return p.vector(color.r, color.g + p.sceneDepth(at: p.screenUV), color.b, color.a)
        }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(writeDepth), .graph(gain), .graph(addDepth)])
        func stored(_ x: Int, _ y: Int) -> Float { Float(y * 4 + x) / 16 }
        let depth = try gpu.depth(width: 4, height: 4, value: stored)
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4)
        try gpu.run { try chain.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for (index, pixel) in pixels.enumerated() {
            let (x, y) = (index % 4, index / 4)
            let input = Self.frame(x, y)
            let expected = SIMD4(stored(x, y), input.y + stored(x, y), input.z, input.w)
            #expect(PostProcessFixture.close(pixel, expected), "(\(x), \(y)): \(pixel)")
        }

        try chain.setEnabled(false, at: 0)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessChain.Failure.missingDepth(stage: 2)) {
            try chain.encode(source: source, destination: destination, commandBuffer: buffer)
        }
    }

    @Test("Color grade and screen effects stages match their standalone passes")
    func builtInStages() throws {
        let screenSettings = ScreenEffects.Settings(chromaticAberration: 2, grainStrength: 0.05, grainSeed: 7)
        let chain = try PostProcessChain(
            queue: gpu.queue, effects: [.graph(swizzle), .colorGrade(strength: 0.75), .screenEffects(screenSettings)]
        )
        #expect(try chain.uniforms(at: 1) == nil)
        let source = try gpu.color(width: 6, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 6, height: 4)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }

        let swizzled = try gpu.color(width: 6, height: 4)
        let graded = try gpu.color(width: 6, height: 4)
        let reference = try gpu.color(width: 6, height: 4)
        let pass = try PostProcessPass(graph: swizzle, device: gpu.device)
        let grade = try ColorGrade(device: gpu.device, destinationPixelFormat: .rgba16Float)
        let effects = try ScreenEffects(queue: gpu.queue, destinationPixelFormat: .rgba16Float)
        try gpu.run { buffer in
            try pass.encode(source: source, destination: swizzled, commandBuffer: buffer)
            try grade.encode(source: swizzled, destination: graded, strength: 0.75, commandBuffer: buffer)
            try effects.encode(source: graded, destination: reference, settings: screenSettings, commandBuffer: buffer)
        }
        let result = try gpu.read(destination)
        #expect(result == (try gpu.read(reference)))

        // Changing a built-in stage's settings keeps its prepared pipelines.
        try chain.replaceEffect(at: 1, with: .colorGrade(strength: 0))
        try chain.replaceEffect(at: 2, with: .screenEffects(.init()))
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        expectFrame(try gpu.read(destination), width: 6) { SIMD4($0.z, $0.y, $0.x, $0.w) }

        #expect(throws: ColorGrade.Failure.invalidStrength) {
            try chain.replaceEffect(at: 1, with: .colorGrade(strength: 2))
        }
        #expect(throws: ScreenEffects.Failure.invalidSettings) {
            try chain.replaceEffect(at: 2, with: .screenEffects(.init(grainStrength: .nan)))
        }
    }

    @Test("The last enabled stage writes an sRGB display destination")
    func displayDestination() throws {
        let chain = try PostProcessChain(
            queue: gpu.queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.graph(gain), .graph(swizzle)]
        )
        try #require(try chain.uniforms(at: 0)).set("Gain", to: 0.5)
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4, pixelFormat: .bgra8Unorm_srgb)

        func decode(_ bytes: [UInt8], at index: Int) -> SIMD4<Float> {
            func linear(_ byte: UInt8) -> Float {
                let value = Float(byte) / 255
                return value <= 0.04045 ? value / 12.92 : powf((value + 0.055) / 1.055, 2.4)
            }
            let base = index * 4
            return SIMD4(linear(bytes[base + 2]), linear(bytes[base + 1]), linear(bytes[base]), Float(bytes[base + 3]) / 255)
        }
        func render() throws -> [UInt8] {
            try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
            var bytes = [UInt8](repeating: 0, count: 4 * 4 * 4)
            try bytes.withUnsafeMutableBytes { raw in
                destination.getBytes(try #require(raw.baseAddress), bytesPerRow: 16,
                                     from: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0)
            }
            return bytes
        }

        // Swizzle last, then gain last: the final format pipeline follows whichever stage is last.
        for gainIsLast in [false, true] {
            try chain.setEnabled(!gainIsLast, at: 1)
            let bytes = try render()
            for index in 0..<16 {
                let input = Self.frame(index % 4, index / 4)
                let expected = gainIsLast
                    ? SIMD4(input.x * 0.5, input.y * 0.5, input.z * 0.5, input.w)
                    : SIMD4(input.z * 0.5, input.y * 0.5, input.x * 0.5, input.w)
                #expect(PostProcessFixture.close(decode(bytes, at: index), expected, tolerance: 0.01),
                        "pixel \(index): \(decode(bytes, at: index)) ≠ \(expected)")
            }
        }
    }

    @Test("Preparation and encoding reject invalid stages, buffers and textures before encoding")
    func validation() throws {
        let textured = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * p.texture("Mask").sample(p.screenUV).rgb }
        #expect(throws: PostProcessPass.Failure.missingTexture("Mask")) {
            try PostProcessChain(queue: gpu.queue, effects: [.graph(swizzle), .graph(textured)])
        }
        #expect(throws: PostProcessChain.Failure.destinationFormat) {
            try PostProcessChain(queue: gpu.queue, destinationPixelFormat: .r32Float, effects: [])
        }
        #expect(throws: ColorGrade.Failure.invalidStrength) {
            try PostProcessChain(queue: gpu.queue, effects: [.colorGrade(strength: -1)])
        }
        let wrongUniforms = try ShaderUniforms([ShaderUniform("Gain", SIMD2<Float>(1, 1))])
        #expect(throws: PostProcessPass.Failure.uniformDeclarations) {
            try PostProcessChain(queue: gpu.queue, effects: [.graph(gain, uniforms: wrongUniforms)])
        }

        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(swizzle), .colorGrade(strength: 0.5)])
        #expect(throws: PostProcessChain.Failure.stageIndex(2)) { try chain.setEnabled(false, at: 2) }
        #expect(throws: PostProcessChain.Failure.stageIndex(-1)) { try chain.uniforms(at: -1) }
        // A failed replacement leaves the stage as it was.
        #expect(throws: PostProcessPass.Failure.missingTexture("Mask")) {
            try chain.replaceEffect(at: 0, with: .graph(textured))
        }
        #expect(try chain.uniforms(at: 0)?.declarations.isEmpty == true)

        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4)
        let large = try gpu.color(width: 8, height: 4)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessChain.Failure.sizeMismatch) {
            try chain.encode(source: source, destination: large, commandBuffer: buffer)
        }
        #expect(throws: PostProcessChain.Failure.destination) {
            try chain.encode(source: source, destination: source, commandBuffer: buffer)
        }
        let wrongDepth = try gpu.depth(width: 8, height: 4) { _, _ in 1 }
        #expect(throws: PostProcessChain.Failure.sizeMismatch) {
            try chain.encode(source: source, depth: wrongDepth, destination: destination, commandBuffer: buffer)
        }
        let otherQueue = try #require(gpu.device.makeCommandQueue())
        let foreign = try #require(otherQueue.makeCommandBuffer())
        #expect(throws: PostProcessChain.Failure.commandBuffer) {
            try chain.encode(source: source, destination: destination, commandBuffer: foreign)
        }
        // The color grade (the last stage) rejects a texture view only after the graph stage
        // validated, so the chain throws before encoding either stage.
        let view = try #require(destination.makeTextureView(pixelFormat: .rgba16Float))
        #expect(throws: ColorGrade.Failure.incompatibleTextures) {
            try chain.encode(source: source, destination: view, commandBuffer: buffer)
        }
    }
}
