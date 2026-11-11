import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RadialBlurTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Off-center radial streaks match the pinned weighted reference in premultiplied HDR")
    func weightedReference() throws {
        let width = 21, height = 13
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = [0, 0.125, 0.375, 0.75, 1][(2 * x + 3 * y) % 5]
            let rgb = SIMD3(Float(x) / 9, Float(y) / 8, -Float(x + y) / 32)
            return SIMD4(rgb * alpha, alpha)
        }
        let destination = try gpu.color(width: width, height: height)
        let input = try gpu.read(source)
        let effect = try RadialBlur(device: gpu.device)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(effect)])
        let prepared = effect.builtPipelineCount
        for settings: RadialBlur.Settings in [
            .init(center: SIMD2(0.19, 0.72)),
            .init(center: SIMD2(0.79, 0.24), radius: 2.7, sampleCount: 64, weight: 0.8, decay: 0.98, exposure: 1.7)
        ] {
            effect.settings = settings
            try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
            let result = try gpu.read(destination)
            for (index, actual) in result.enumerated() {
                let expected = reference(input, width: width, height: height, index: index, settings: settings)
                #expect(PostProcessFixture.close(actual, expected, tolerance: 0.006),
                        "pixel \(index), center \(settings.center): \(actual) vs \(expected)")
            }
            #expect(result.contains { $0.x > 1 && $0.z < 0 })
            #expect(zip(input, result).contains { $0.0.w == 0 && $0.1.w > 0 && $0.1.w < 1 })
            #expect(result.allSatisfy { (0...1).contains($0.w) })
        }
        #expect(effect.builtPipelineCount == prepared)
    }

    @Test("Overshooting rays clamp to the authored corner, and zero settings bypass exactly")
    func clampAndIdentity() throws {
        let width = 5, height = 3
        let corner = SIMD4<Float>(2, 0.5, 0.125, 0.5)
        let source = try gpu.color(width: width, height: height) { x, y in
            x == 0 && y == 0 ? corner : .zero
        }
        let destination = try gpu.color(width: width, height: height)
        let settings = RadialBlur.Settings(center: .zero, radius: 4, sampleCount: 1, weight: 1, decay: 1, exposure: 2)
        let effect = try RadialBlur(device: gpu.device, settings: settings)
        try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
        let result = try gpu.read(destination)
        #expect(result[0] == 2 * corner)
        #expect(result.dropFirst().allSatisfy { $0 == corner })
        let input = try gpu.read(source)
        for disabled: RadialBlur.Settings in [.init(radius: 0), .init(weight: 0), .init(exposure: 0)] {
            effect.settings = disabled
            try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
            #expect(try gpu.read(destination) == input)
        }

        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: source, destination: source, commandBuffer: command)
        }
        effect.settings.sampleCount = 65
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, destination: destination, commandBuffer: command)
        }
        #expect(throws: PostProcessEffectError.self) {
            try RadialBlur(device: gpu.device, settings: .init(center: SIMD2(.nan, 0)))
        }
    }

    private func reference(
        _ input: [SIMD4<Float>], width: Int, height: Int, index: Int, settings: RadialBlur.Settings
    ) -> SIMD4<Float> {
        let fragment = SIMD2(Float(index % width) + 0.5, Float(index / width) + 0.5)
        let size = SIMD2(Float(width), Float(height))
        let uv = fragment / size
        func fract(_ x: Float) -> Float { x - floor(x) }
        let phase = fract(52.9829189 * fract(0.06711056 * fragment.x + 0.00583715 * fragment.y))
        let ray = (settings.center - uv) * settings.radius
        var sum = SIMD4<Float>.zero
        for tap in 1...settings.sampleCount {
            let location = uv + ray * ((Float(tap) + phase) / Float(settings.sampleCount))
            let last = size - 1
            let clamped = simd_clamp(location * size - 0.5, .zero, last)
            let low = floor(clamped)
            let x = Int(low.x), y = Int(low.y)
            let nextX = min(x + 1, width - 1), nextY = min(y + 1, height - 1)
            let fraction = clamped - low
            let top = input[y * width + x] * (1 - fraction.x) + input[y * width + nextX] * fraction.x
            let bottom = input[nextY * width + x] * (1 - fraction.x) + input[nextY * width + nextX] * fraction.x
            sum += (top * (1 - fraction.y) + bottom * fraction.y) * settings.weight * pow(settings.decay, Float(tap - 1))
        }
        var color = input[index] + sum * (settings.exposure / (2 * Float(settings.sampleCount)))
        color.w = min(color.w, 1)
        return color
    }
}
