import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TiltShiftTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Each full-resolution iteration follows the rotated pmndrs falloff and four-tap reference")
    func variableKawaseReference() throws {
        let width = 13, height = 9
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = [0, 0.25, 0.75, 1][(2 * x + y) % 4]
            let straight = SIMD3<Float>(Float(x % 4) + 0.25, Float(y % 3), Float((x + 2 * y) % 7) / 4)
            return SIMD4(straight * alpha, alpha)
        }
        let destination = try gpu.color(width: width, height: height)
        let original = try gpu.read(source)
        let settings = TiltShift.Settings(
            rotation: 0.46, offset: 0.21, focusArea: 0.72, feather: 0.48,
            blur: .init(offsets: [0.5, 1.25], resolutionScale: 1)
        )
        let effect = try TiltShift(device: gpu.device, settings: settings)
        try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
        let output = try gpu.read(destination)
        let reference = cpuReference(original, width: width, height: height, settings: settings, fractionalSteps: nil)
        // Metal can round subtexel sample coordinates. Check a separate reference
        // with 1/256-texel increments, keeping the same comparison bound.
        let quantized = cpuReference(original, width: width, height: height, settings: settings, fractionalSteps: 256)
        for index in output.indices {
            let actual = output[index]
            #expect(PostProcessFixture.close(actual, reference[index], tolerance: 0.008)
                    || PostProcessFixture.close(actual, quantized[index], tolerance: 0.008),
                    "pixel \(index): \(actual), exact \(reference[index]), quantized \(quantized[index])")
        }
        #expect(output.contains { $0.x > 1 })
        #expect(zip(original, output).contains { $0.0.w == 0 && $0.1.w > 0 })
    }

    @Test("A shifted vertical band preserves exact sharp pixels through a reduced chain stage")
    func reducedChainAndAlpha() throws {
        let width = 41, height = 21
        let hue = SIMD3<Float>(4, 2, 0.5)
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = (x + y).isMultiple(of: 2) ? 1 : 0
            return SIMD4(hue * alpha, alpha)
        }
        let destination = try gpu.color(width: width, height: height)
        let settings = TiltShift.Settings(
            rotation: .pi / 2, offset: 14 / Float(height), focusArea: 0.42, feather: 0.2,
            blur: .init(offsets: [0, 1, 2], resolutionScale: 0.5)
        )
        let effect = try TiltShift(device: gpu.device, settings: settings)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(effect)])
        #expect(!effect.readsDepth)
        let prepared = effect.builtPipelineCount
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        let original = try gpu.read(source)
        let output = try gpu.read(destination)
        for y in 0..<height {
            for x in 25...29 { #expect(output[y * width + x] == original[y * width + x]) }
        }
        let blurred = output[(height / 2) * width + 5]
        #expect(blurred.w > 0.3 && blurred.w < 0.7)
        for sample in output {
            let expected = SIMD4(hue * sample.w, sample.w)
            #expect(PostProcessFixture.close(sample, expected, tolerance: 0.004))
        }
        #expect(effect.builtPipelineCount == prepared)

        // With the band outside the image every pixel uses exactly the native blur.
        effect.settings.offset = 8
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        let kawase = try KawaseBlur(device: gpu.device, settings: settings.blur)
        let blurredOnly = try gpu.color(width: width, height: height)
        try gpu.run { try kawase.encode(source: source, destination: blurredOnly, commandBuffer: $0) }
        #expect(try gpu.read(destination) == gpu.read(blurredOnly))
        #expect(effect.builtPipelineCount == prepared)
    }

    @Test("A hard focus band, invalid parameters and overlapping output are handled explicitly")
    func hardBandAndValidation() throws {
        let size = 9
        let source = try gpu.color(width: size, height: size) { x, _ in
            SIMD4(x.isMultiple(of: 2) ? 2 : 0, 0, 0, 1)
        }
        let destination = try gpu.color(width: size, height: size)
        let effect = try TiltShift(
            device: gpu.device, settings: .init(focusArea: 0.1, feather: 0, blur: .init(offsets: [0], resolutionScale: 1))
        )
        try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
        let result = try gpu.read(destination)
        #expect(result[4 * size + 4] == SIMD4(2, 0, 0, 1))
        #expect(result[3 * size + 4] == SIMD4(1, 0, 0, 1))

        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: source, destination: source, commandBuffer: command)
        }
        effect.settings.rotation = .nan
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, destination: destination, commandBuffer: command)
        }
    }

    private func cpuReference(
        _ original: [SIMD4<Float>], width: Int, height: Int, settings: TiltShift.Settings, fractionalSteps: Float?
    ) -> [SIMD4<Float>] {
        let inner = max(0, settings.focusArea - settings.feather)
        let sine = sin(settings.rotation), cosine = cos(settings.rotation)
        var input = original
        for (iteration, kernel) in settings.blur.offsets.enumerated() {
            let final = iteration == settings.blur.offsets.count - 1
            input = original.indices.map { index in
                let x = Float(index % width), y = Float(index / width)
                let normalX = (x + 0.5 - Float(width) / 2) * 2 / Float(height)
                let normalY = (y + 0.5 - Float(height) / 2) * 2 / Float(height)
                let distance = abs(sine * normalX + cosine * normalY - settings.offset)
                if final && distance <= inner { return original[index] }
                let t = min(max((distance - inner) / (settings.focusArea - inner), 0), 1)
                let delta = (kernel + 0.5) * t * t * (3 - 2 * t)
                var average = SIMD4<Float>.zero
                for dy: Float in [-delta, delta] {
                    for dx: Float in [-delta, delta] {
                        average += sample(
                            input, width: width, height: height, x: x + dx, y: y + dy, fractionalSteps: fractionalSteps
                        ) / 4
                    }
                }
                return SIMD4<Float>(SIMD4<Float16>(average))
            }
        }
        return input
    }

    private func sample(
        _ image: [SIMD4<Float>], width: Int, height: Int, x: Float, y: Float, fractionalSteps: Float?
    ) -> SIMD4<Float> {
        let left = Int(floor(x)), top = Int(floor(y))
        var tx = x - Float(left), ty = y - Float(top)
        if let fractionalSteps {
            tx = (tx * fractionalSteps).rounded(.toNearestOrEven) / fractionalSteps
            ty = (ty * fractionalSteps).rounded(.toNearestOrEven) / fractionalSteps
        }
        func texel(_ x: Int, _ y: Int) -> SIMD4<Float> {
            image[min(max(y, 0), height - 1) * width + min(max(x, 0), width - 1)]
        }
        return (texel(left, top) * (1 - tx) + texel(left + 1, top) * tx) * (1 - ty)
            + (texel(left, top + 1) * (1 - tx) + texel(left + 1, top + 1) * tx) * ty
    }
}
