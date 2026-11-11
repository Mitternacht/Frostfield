import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MotionBlurTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Native pixel motion produces horizontal and vertical streaks with the r187 tap offsets")
    func directionAndEnergy() throws {
        let size = 8
        let impulse = SIMD4<Float>(4, 2, 1, 0.5)
        let source = try gpu.color(width: size, height: size) { x, y in x == 4 && y == 3 ? impulse : .zero }
        let output = try gpu.color(width: size, height: size)
        let horizontal = try gpu.color(width: size, height: size) { _, _ in SIMD4(4, 0, 0.25, 1) }
        let vertical = try gpu.color(width: size, height: size) { _, _ in SIMD4(0, -4, 0.75, 1) }
        let effect = try MotionBlur(device: gpu.device, motionTexture: horizontal, settings: .init(sampleCount: 3))
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(effect)])
        let prepared = effect.builtPipelineCount

        try gpu.run { try chain.encode(source: source, destination: output, commandBuffer: $0) }
        var result = try gpu.read(output)
        #expect(result[3 * size] == impulse / 4)
        #expect(result[3 * size + 2] == impulse / 4)
        #expect(result[3 * size + 4] == impulse / 2)
        #expect(result[3 * size + 6] == .zero)
        #expect(result[5 * size + 4] == .zero)

        effect.motionTexture = vertical
        try gpu.run { try chain.encode(source: source, destination: output, commandBuffer: $0) }
        result = try gpu.read(output)
        #expect(result[7 * size + 4] == impulse / 4)
        #expect(result[5 * size + 4] == impulse / 4)
        #expect(result[3 * size + 4] == impulse / 2)
        #expect(result[3 * size + 2] == .zero)
        #expect(result[1 * size + 4] == .zero)

        let constant = try gpu.color(width: size, height: size) { _, _ in impulse }
        effect.settings.sampleCount = 64
        try gpu.run { try chain.encode(source: constant, destination: output, commandBuffer: $0) }
        #expect(try gpu.read(output).allSatisfy { $0 == impulse })
        #expect(effect.builtPipelineCount == prepared)
    }

    @Test("Fractional vectors match bilinear HDR averaging and invalid vectors are exact copies")
    func referenceAndInvalidVectors() throws {
        let width = 13, height = 9
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = [0, 0.25, 0.5, 1][(x + 2 * y) % 4]
            let rgb = SIMD3(Float(x + 2) / 4, Float(y + 1) / 9, -0.5)
            return SIMD4(rgb * alpha, alpha)
        }
        let motion = try gpu.color(width: width, height: height) { x, y in
            let previousDepth = Float(y) / Float(height)
            switch (x + 2 * y) % 7 {
            case 0: return SIMD4(0, 0, previousDepth, 1)
            case 1: return SIMD4(20, -30, previousDepth, 0)
            case 2: return SIMD4(.nan, 1, previousDepth, 1)
            case 3: return SIMD4(1, .infinity, previousDepth, 1)
            case 4: return SIMD4(-3, 2, previousDepth, .nan)
            case 5: return SIMD4(3.5, -2.25, previousDepth, 1)
            default: return SIMD4(-5, 3.25, previousDepth, 1)
            }
        }
        let output = try gpu.color(width: width, height: height)
        let input = try gpu.read(source)
        let vectors = try gpu.read(motion)
        let effect = try MotionBlur(device: gpu.device, motionTexture: motion, settings: .init(sampleCount: 5, shutterScale: 1.25))
        try gpu.run { try effect.encode(source: source, destination: output, commandBuffer: $0) }
        let result = try gpu.read(output)
        for (index, actual) in result.enumerated() {
            let v = vectors[index]
            let delta = SIMD2(v.x, v.y)
            if v.w != 1 || !v.x.isFinite || !v.y.isFinite || delta == .zero {
                #expect(actual == input[index])
                continue
            }
            let center = SIMD2(Float(index % width), Float(index / width))
            let samples = effect.settings.sampleCount
            let expected = (1...samples).reduce(input[index]) { sum, tap in
                let weight = Float(tap) / Float(samples - 1) - 0.5
                let position = center + delta * (weight * effect.settings.shutterScale)
                return sum + bilinear(input, width: width, height: height, position: position)
            } / Float(samples + 1)
            #expect(PostProcessFixture.close(actual, expected, tolerance: 0.006),
                    "pixel \(index): \(actual) vs \(expected)")
            #expect((0...1).contains(actual.w))
        }
        #expect(result.contains { $0.x > 1 && $0.z < 0 })
        #expect(zip(input, result).contains { $0.0.w == 0 && $0.1.w > 0 })

        effect.settings.shutterScale = 0
        try gpu.run { try effect.encode(source: source, destination: output, commandBuffer: $0) }
        #expect(try gpu.read(output) == input)
    }

    @Test("Seeded blue-noise jitter moves both tap axes and preserves invalid and constant HDR pixels")
    func seededJitter() throws {
        let width = 17, height = 13
        let source = try gpu.color(width: width, height: height) { x, y in
            SIMD4(2 + Float(x) / 4, 0.25 + Float(y) / 8, -0.5, 0.75)
        }
        let vectors = try gpu.color(width: width, height: height) { x, _ in
            switch x {
            case 0: SIMD4(4, -3, 0, 0)
            case 1: SIMD4(.nan, -3, 0, 1)
            case 2: SIMD4(0, 0, 0, 1)
            default: SIMD4(4, -3, 0, 1)
            }
        }
        let destination = try gpu.color(width: width, height: height)
        let effect = try MotionBlur(device: gpu.device, motionTexture: vectors, settings: .init(sampleCount: 5))
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(effect)])
        let prepared = effect.builtPipelineCount

        func render(_ color: any MTLTexture) throws -> [SIMD4<Float>] {
            try gpu.run { try chain.encode(source: color, destination: destination, commandBuffer: $0) }
            return try gpu.read(destination)
        }

        let regular = try render(source)
        effect.settings.frameSeed = .max
        #expect(try render(source) == regular)

        effect.settings.frameSeed = 0
        effect.settings.jitter = 0.65
        let jittered = try render(source)
        #expect(try render(source) == jittered)
        let input = try gpu.read(source)
        let center = SIMD2<Float>(8, 6)
        let index = 6 * width + 8
        // Independent values at the two unshifted phase-zero rank coordinates.
        let rank = SIMD2<Float>(2388, 2666) / 4095
        let displacement = SIMD2<Float>(4, -3)
        var expected = input[index]
        for tap in 1...5 {
            let offset = Float(tap) / 4 - 0.5
            let position = center + displacement * (SIMD2(repeating: offset) + 0.65 * rank)
            expected += bilinear(input, width: width, height: height, position: position)
        }
        expected /= 6
        #expect(PostProcessFixture.close(jittered[index], expected, tolerance: 0.003))
        #expect(jittered[index].x > regular[index].x + 0.1)
        #expect(jittered[index].y < regular[index].y - 0.05)
        #expect(jittered[index].x > 1 && jittered[index].z == -0.5 && jittered[index].w == 0.75)
        for y in 0..<height {
            for x in 0..<3 {
                let pixel = y * width + x
                #expect(jittered[pixel] == input[pixel])
            }
        }

        effect.settings.frameSeed = 0x7a31_f420
        let first = try render(source)
        effect.settings.frameSeed = 0x8d44_f420
        let second = try render(source)
        #expect(zip(first, second).contains { abs($0.0.x - $0.1.x) > 0.05 })

        let hdr = SIMD4<Float>(4, 2, -0.5, 0.5)
        let constant = try gpu.color(width: width, height: height) { _, _ in hdr }
        #expect(try render(constant).allSatisfy { $0 == hdr })
        effect.settings.shutterScale = 0
        #expect(try render(source) == input)
        #expect(effect.builtPipelineCount == prepared + 1)
    }

    @Test("Motion textures, retained command buffers and destination aliases are validated")
    func validation() throws {
        let source = try gpu.color(width: 5, height: 3)
        let output = try gpu.color(width: 5, height: 3)
        let motion = try gpu.color(width: 5, height: 3)
        let effect = try MotionBlur(device: gpu.device, motionTexture: motion)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let unretained = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        #expect(throws: PostProcessEffectError.commandBuffer) {
            try effect.encode(source: source, destination: output, commandBuffer: unretained)
        }
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: source, destination: source, commandBuffer: command)
        }
        effect.motionTexture = output
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: source, destination: output, commandBuffer: command)
        }
        effect.motionTexture = try gpu.color(width: 4, height: 3)
        #expect(throws: PostProcessEffectError.sizeMismatch) {
            try effect.encode(source: source, destination: output, commandBuffer: command)
        }
        effect.motionTexture = try gpu.color(width: 5, height: 3, pixelFormat: .rgba8Unorm)
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, destination: output, commandBuffer: command)
        }
        effect.motionTexture = motion
        effect.settings.sampleCount = 1
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, destination: output, commandBuffer: command)
        }
        effect.settings.sampleCount = 16
        for invalid in [Float.nan, -0.01, 1.01] {
            effect.settings.jitter = invalid
            #expect(throws: PostProcessEffectError.self) {
                try effect.encode(source: source, destination: output, commandBuffer: command)
            }
        }
    }

    private func bilinear(_ input: [SIMD4<Float>], width: Int, height: Int, position: SIMD2<Float>) -> SIMD4<Float> {
        let clamped = simd_clamp(position, .zero, SIMD2(Float(width - 1), Float(height - 1)))
        let low = floor(clamped)
        let x = Int(low.x), y = Int(low.y)
        let nextX = min(x + 1, width - 1), nextY = min(y + 1, height - 1)
        let fraction = clamped - low
        let top = input[y * width + x] * (1 - fraction.x) + input[y * width + nextX] * fraction.x
        let bottom = input[nextY * width + x] * (1 - fraction.x) + input[nextY * width + nextX] * fraction.x
        return top * (1 - fraction.y) + bottom * fraction.y
    }
}
