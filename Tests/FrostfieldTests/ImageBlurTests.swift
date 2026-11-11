import Foundation
import Frostfield
import Metal
import Testing

@Suite("Gaussian kernel")
struct GaussianKernelTests {
    @Test("Weights sum to one, are symmetric, span three sigma and match the integrated Gaussian")
    func weights() throws {
        for sigma: Float in [0.5, 1, 2.5, 7.3, 64] {
            let weights = try GaussianBlur.kernel(sigma: sigma)
            let radius = Int((3 * sigma).rounded(.up))
            #expect(weights.count == 2 * radius + 1)
            #expect(abs(weights.reduce(0, +) - 1) < 1e-5, "sigma \(sigma)")
            for offset in 0...radius {
                #expect(weights[radius + offset] == weights[radius - offset])
            }
            for offset in 0..<radius {
                #expect(weights[radius + offset] > weights[radius + offset + 1])
            }
        }
        // Sigma 1: mass of [−½, ½] over the mass of [−3½, 3½].
        let weights = try GaussianBlur.kernel(sigma: 1)
        let center = erf(0.5 / 2.0.squareRoot()) / erf(3.5 / 2.0.squareRoot())
        #expect(abs(Double(weights[3]) - center) < 1e-6)
        #expect(try GaussianBlur.kernel(sigma: 0) == [1])
        #expect(throws: PostProcessEffectError.self) { try GaussianBlur.kernel(sigma: 65) }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ImageBlurTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Blurring a delta reproduces the separable kernel")
    func gaussianDelta() throws {
        let sigma: Float = 2.5
        let size = 41
        let center = size / 2
        let blur = try GaussianBlur(device: gpu.device, settings: .init(sigma: sigma))
        let source = try gpu.color(width: size, height: size) { x, y in x == center && y == center ? SIMD4(1, 2, 4, 1) : .zero }
        let destination = try gpu.color(width: size, height: size)
        try gpu.run { try blur.encode(source: source, destination: destination, commandBuffer: $0) }
        let pixels = try gpu.read(destination)
        let weights = try GaussianBlur.kernel(sigma: sigma)
        let radius = weights.count / 2
        for y in 0..<size {
            for x in 0..<size {
                let (dx, dy) = (x - center, y - center)
                let weight = abs(dx) <= radius && abs(dy) <= radius ? weights[dx + radius] * weights[dy + radius] : 0
                let expected = SIMD4<Float>(1, 2, 4, 1) * weight
                let tolerance = 2e-5 + 3e-3 * expected.max()
                #expect(PostProcessFixture.close(pixels[y * size + x], expected, tolerance: tolerance), "(\(x), \(y))")
            }
        }
        let total = sum(pixels)
        #expect(abs(total.w - 1) < 2e-3)
    }

    @Test("One Kawase iteration averages the four texel corners at the offset, clamped to the edge")
    func kawaseReference() throws {
        let width = 9
        let height = 7
        func input(_ x: Int, _ y: Int) -> SIMD4<Float> {
            SIMD4(Float((x * 7 + y * 3) % 10) / 10, Float((x + y * 5) % 6) / 6, Float(x) / 8, 1)
        }
        func reference(_ image: (Int, Int) -> SIMD4<Float>, offset: Int) -> [SIMD4<Float>] {
            func texel(_ x: Int, _ y: Int) -> SIMD4<Float> { image(min(max(x, 0), width - 1), min(max(y, 0), height - 1)) }
            var result: [SIMD4<Float>] = []
            for y in 0..<height {
                for x in 0..<width {
                    var sum = SIMD4<Float>.zero
                    for sy in [-1, 1] {
                        for sx in [-1, 1] {
                            // The corner (x + ½ ± (k + ½)) lies between texels c − 1 and c.
                            let cx = x + (sx > 0 ? offset + 1 : -offset)
                            let cy = y + (sy > 0 ? offset + 1 : -offset)
                            sum += (texel(cx - 1, cy - 1) + texel(cx, cy - 1) + texel(cx - 1, cy) + texel(cx, cy)) / 4
                        }
                    }
                    result.append(sum / 4)
                }
            }
            return result
        }
        let blur = try KawaseBlur(device: gpu.device, settings: .init(offsets: [0, 1], resolutionScale: 1))
        let source = try gpu.color(width: width, height: height, pixel: input)
        let destination = try gpu.color(width: width, height: height)
        try gpu.run { try blur.encode(source: source, destination: destination, commandBuffer: $0) }
        let first = reference(input, offset: 0)
        let expected = reference({ first[$1 * width + $0] }, offset: 1)
        for (index, pixel) in try gpu.read(destination).enumerated() {
            #expect(PostProcessFixture.close(pixel, expected[index], tolerance: 3e-3), "pixel \(index)")
        }
    }

    @Test("Kawase iterations conserve energy away from the frame edge")
    func kawaseEnergy() throws {
        let size = 96
        let blur = try KawaseBlur(device: gpu.device, settings: .init(kernelSize: .small, resolutionScale: 1))
        let source = try gpu.color(width: size, height: size) { x, y in
            abs(x - 48) < 3 && abs(y - 47) < 2 ? SIMD4(3, 1, 0.5, 1) : .zero
        }
        let destination = try gpu.color(width: size, height: size)
        try gpu.run { try blur.encode(source: source, destination: destination, commandBuffer: $0) }
        let input = sum(try gpu.read(source))
        let output = try gpu.read(destination)
        let total = sum(output)
        #expect(abs(total.x / input.x - 1) < 3e-3)
        #expect(abs(total.w / input.w - 1) < 3e-3)
        #expect(output[47 * size + 48].x < 3 * 0.25)
        #expect(output[47 * size + 62].x > 0)
    }

    @Test("A reduced Kawase grid keeps a single highlight's energy wherever it falls in a grid texel",
          arguments: [Float(0.5), 0.25, 0.125])
    func kawaseReducedEnergy(scale: Float) throws {
        let size = 128
        let blur = try KawaseBlur(device: gpu.device, settings: .init(offsets: [0, 1], resolutionScale: scale))
        let block = Int((1 / scale).rounded())
        for dy in 0..<block {
            for dx in 0..<block {
                let (px, py) = (64 + dx, 56 + dy)
                let source = try gpu.color(width: size, height: size) { x, y in x == px && y == py ? SIMD4(64, 32, 16, 1) : .zero }
                let destination = try gpu.color(width: size, height: size)
                try gpu.run { try blur.encode(source: source, destination: destination, commandBuffer: $0) }
                let total = sum(try gpu.read(destination))
                #expect(abs(total.x / 64 - 1) < 5e-3, "scale \(scale), offset (\(dx), \(dy)): \(total.x)")
                #expect(abs(total.y / 32 - 1) < 5e-3 && abs(total.w - 1) < 5e-3)
            }
        }
    }

    @Test("The default half-resolution Kawase blur conserves energy and spreads symmetrically")
    func kawaseDefault() throws {
        let size = 384
        let blur = try KawaseBlur(device: gpu.device)
        let source = try gpu.color(width: size, height: size) { x, y in
            abs(x - 192) < 3 && abs(y - 192) < 3 ? SIMD4(3, 1, 0.5, 1) : .zero
        }
        let destination = try gpu.color(width: size, height: size)
        try gpu.run { try blur.encode(source: source, destination: destination, commandBuffer: $0) }
        let input = sum(try gpu.read(source))
        let output = try gpu.read(destination)
        let total = sum(output)
        #expect(abs(total.x / input.x - 1) < 3e-3)
        let left = output[192 * size + 150].x
        let right = output[192 * size + 234].x
        #expect(right > 0 && abs(left - right) < 0.05 * right)
    }

    @Test("The bilateral color weight keeps colors apart without depth")
    func bilateralColor() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100)
        let blur = try BilateralBlur(
            device: gpu.device, camera: camera, settings: .init(sigma: 3, depthTolerance: 0, colorSigma: 0.05)
        )
        #expect(!blur.readsDepth)
        let source = try gpu.color(width: 32, height: 8) { x, _ in x < 16 ? SIMD4(1, 0, 0, 1) : SIMD4(0, 0, 1, 1) }
        let destination = try gpu.color(width: 32, height: 8)
        try gpu.run { try blur.encode(source: source, depth: nil, destination: destination, commandBuffer: $0) }
        let output = try gpu.read(destination)
        // exp(−½ (√2 / 0.05)²) underflows: no tap of the other color contributes.
        for y in 0..<8 {
            for x in 0..<32 {
                let expected: SIMD4<Float> = x < 16 ? SIMD4(1, 0, 0, 1) : SIMD4(0, 0, 1, 1)
                let pixel = output[y * 32 + x]
                #expect(PostProcessFixture.close(pixel, expected), "(\(x), \(y)) \(pixel)")
                #expect(x < 16 ? pixel.z == 0 : pixel.x == 0, "(\(x), \(y)) mixed colors")
            }
        }
        blur.settings.colorSigma = 0
        try gpu.run { try blur.encode(source: source, depth: nil, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination)[4 * 32 + 15].z > 0.1)
    }

    @Test("The bilateral blur does not cross a depth discontinuity that a Gaussian blurs across")
    func bilateralEdge() throws {
        let width = 32
        let height = 12
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100)
        let depth = try gpu.depth(width: width, height: height) { x, _ in
            storedDepth(distance: x < 16 ? 2 : 8, near: 0.1, far: 100)
        }
        let source = try gpu.color(width: width, height: height) { x, y in
            let shade = 0.5 + 0.4 * Float((x + 2 * y) % 3 - 1)
            return x < 16 ? SIMD4(shade, 0, 0, 1) : SIMD4(0, 0, shade, 1)
        }
        let destination = try gpu.color(width: width, height: height)
        let bilateral = try BilateralBlur(device: gpu.device, camera: camera, settings: .init(sigma: 3, depthTolerance: 0.05))
        try gpu.run { try bilateral.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
        let edgePreserved = try gpu.read(destination)
        for y in 0..<height {
            #expect(edgePreserved[y * width + 15].z == 0, "row \(y) left of the edge")
            #expect(edgePreserved[y * width + 16].x == 0, "row \(y) right of the edge")
            // Within a layer the texture is smoothed toward its mean.
            #expect(abs(edgePreserved[y * width + 8].x - 0.5) < 0.12)
        }

        bilateral.settings.depthTolerance = 0
        #expect(!bilateral.readsDepth)
        try gpu.run { try bilateral.encode(source: source, depth: nil, destination: destination, commandBuffer: $0) }
        let blurred = try gpu.read(destination)
        #expect(blurred[5 * width + 15].z > 0.1)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MipmapBloomTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    private func bloom(
        _ settings: MipmapBloom.Settings, size: Int = 192, pixel: @escaping (Int, Int) -> SIMD4<Float>
    ) throws -> (input: [SIMD4<Float>], output: [SIMD4<Float>]) {
        let effect = try MipmapBloom(device: gpu.device, settings: settings)
        let source = try gpu.color(width: size, height: size, pixel: pixel)
        let destination = try gpu.color(width: size, height: size)
        try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
        return (try gpu.read(source), try gpu.read(destination))
    }

    private static func highlights(_ x: Int, _ y: Int) -> SIMD4<Float> {
        if abs(x - 96) < 2 && abs(y - 90) < 3 { return SIMD4(40, 12, 3, 1) }
        if abs(x - 80) < 6 && abs(y - 104) < 4 { return SIMD4(0.5, 2, 1, 1) }
        return SIMD4(0, 0, 0, 1)
    }

    @Test("Threshold-free energy-conserving bloom preserves total light", arguments: [Float(0.5), 1])
    func energyConservation(strength: Float) throws {
        let settings = MipmapBloom.Settings(
            strength: strength, radius: 0.8, levels: 4, threshold: nil, fireflySuppression: false, blending: .energyConserving
        )
        let (input, output) = try bloom(settings, pixel: Self.highlights)
        let before = sum(input)
        let after = sum(output)
        for channel in 0..<3 {
            #expect(abs(after[channel] / before[channel] - 1) < 5e-3, "channel \(channel): \(after[channel]) vs \(before[channel])")
        }
        #expect(after.w == before.w)
        // Light moved: the peak dimmed and a halo appeared well outside the source.
        #expect(output[90 * 192 + 96].x < input[90 * 192 + 96].x)
        #expect(output[90 * 192 + 112].x > 0.01)
        gpu.savePNG(output, width: 192, height: 192, name: "bloom-energy-\(strength)")
    }

    @Test("Additive bloom adds exactly strength times the scene's light")
    func additive() throws {
        let settings = MipmapBloom.Settings(
            strength: 0.25, radius: 0.6, levels: 5, threshold: nil, fireflySuppression: false, blending: .additive
        )
        let (input, output) = try bloom(settings, pixel: Self.highlights)
        let before = sum(input)
        let after = sum(output)
        #expect(abs(after.x / (before.x * 1.25) - 1) < 5e-3)
        #expect(abs(after.y / (before.y * 1.25) - 1) < 5e-3)
    }

    @Test("Light below the threshold does not bloom; light above it spreads")
    func threshold() throws {
        let settings = MipmapBloom.Settings(strength: 1, radius: 0.8, levels: 4, threshold: 1, knee: 0.5, blending: .additive)
        let dim = try bloom(settings, size: 64) { _, _ in SIMD4(0.45, 0.4, 0.3, 1) }
        #expect(dim.output == dim.input)
        let bright = try bloom(settings, size: 64) { x, y in
            x == 32 && y == 32 ? SIMD4(30, 30, 30, 1) : SIMD4(0.2, 0.2, 0.2, 1)
        }
        #expect(bright.output[32 * 64 + 40].x > 0.2 + 1e-3)
        // Left, right, up and down neighbors at equal distance receive equal glow.
        let right = bright.output[32 * 64 + 36].x
        let left = bright.output[32 * 64 + 28].x
        let up = bright.output[28 * 64 + 32].x
        let down = bright.output[36 * 64 + 32].x
        #expect(abs(right - left) < 0.02 * right && abs(up - down) < 0.02 * up && abs(right - up) < 0.02 * right)
    }

    @Test("Energy-conserving bloom with a threshold leaves dim frames unchanged and conserves total light")
    func thresholdEnergy() throws {
        let settings = MipmapBloom.Settings(
            strength: 0.5, radius: 0.8, levels: 4, threshold: 1, knee: 0.5, fireflySuppression: false
        )
        #expect(settings.blending == .energyConserving)
        let dim = try bloom(settings, size: 64) { _, _ in SIMD4(0.45, 0.4, 0.3, 1) }
        #expect(dim.output == dim.input)

        let (input, output) = try bloom(settings, pixel: Self.highlights)
        let before = sum(input)
        let after = sum(output)
        for channel in 0..<3 {
            #expect(abs(after[channel] / before[channel] - 1) < 5e-3, "channel \(channel): \(after[channel]) vs \(before[channel])")
        }
        // The bright core gives up part of its light; dark pixels nearby receive it.
        #expect(output[90 * 192 + 96].x < input[90 * 192 + 96].x)
        #expect(output[90 * 192 + 112].x > 0.01)
        // Light below the knee gives nothing away, so a dim background only gains glow.
        let lit = try bloom(settings) { x, y in Self.highlights(x, y) + SIMD4(0.3, 0.3, 0.3, 0) }
        for (index, pixel) in lit.input.enumerated() where pixel == SIMD4(0.3, 0.3, 0.3, 1) {
            let result = lit.output[index]
            #expect(result.x >= pixel.x && result.y >= pixel.y && result.z >= pixel.z && result.w == 1, "pixel \(index)")
        }
        #expect(abs(sum(lit.output).x / sum(lit.input).x - 1) < 5e-3)
    }

    @Test("Invalid settings throw before encoding")
    func invalidSettings() throws {
        let effect = try MipmapBloom(device: gpu.device, settings: .init(strength: 2, blending: .energyConserving))
        let source = try gpu.color(width: 8, height: 8)
        let destination = try gpu.color(width: 8, height: 8)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, destination: destination, commandBuffer: buffer)
        }
    }
}
