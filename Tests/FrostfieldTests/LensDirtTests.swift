import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LensDirtTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Lens dirt uses a center crop, multiplies HDR bloom per channel and preserves coverage")
    func pinnedComposition() throws {
        let dirtSize = SIMD2(5, 4)
        let bloomSize = SIMD2(3, 4)
        func dirtPixel(_ x: Int, _ y: Int) -> SIMD4<Float> {
            // RGB defines the mask. The independent alpha channel must not gate it.
            SIMD4(Float(x) / 4, Float(y) / 3, Float(x + y) / 8, (x + y).isMultiple(of: 2) ? 0.25 : 1)
        }
        func bloomPixel(_ x: Int, _ y: Int) -> SIMD4<Float> {
            SIMD4(Float(x + 1) * 1.25, Float(y + 1) * 0.5, Float(x + y) * 0.25, 1)
        }
        func sourcePixel(_ x: Int, _ y: Int) -> SIMD4<Float> {
            let alpha: Float = [0, 0.25, 1][(x + y) % 3]
            return SIMD4(SIMD3<Float>(0.5, 0.25, 0.125) * alpha, alpha)
        }
        let dirt = try gpu.color(width: dirtSize.x, height: dirtSize.y, pixel: dirtPixel)
        let bloom = try gpu.color(width: bloomSize.x, height: bloomSize.y, pixel: bloomPixel)
        let pass = try PostProcessPass(
            graph: .lensDirt(dirtSize: dirtSize, intensity: 1.75), device: gpu.device, textures: ["Bloom": bloom, "Dirt": dirt]
        )
        // Portrait and landscape exercise both branches of the aspect correction.
        for size in [SIMD2(7, 9), SIMD2(11, 5)] {
            let source = try gpu.color(width: size.x, height: size.y, pixel: sourcePixel)
            let destination = try gpu.color(width: size.x, height: size.y)
            let aspect = (Float(size.x) / Float(size.y)) / (Float(dirtSize.x) / Float(dirtSize.y))
            let crop = SIMD2(min(aspect, 1), min(1 / aspect, 1))
            for intensity: Float in [1.75, 0] {
                try pass.uniforms.set("Intensity", to: intensity)
                try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
                let pixels = try gpu.read(destination)
                for (index, actual) in pixels.enumerated() {
                    let x = index % size.x, y = index / size.x
                    let uv = (SIMD2(Float(x), Float(y)) + 0.5) / SIMD2<Float>(size)
                    let base = sourcePixel(x, y)
                    let glow = sampleRGB(uv, size: bloomSize, pixel: bloomPixel)
                    let mask = sampleRGB((uv - 0.5) * crop + 0.5, size: dirtSize, pixel: dirtPixel)
                    let expected = SIMD4(SIMD3(base.x, base.y, base.z) + glow * (1 + mask * intensity) * base.w, base.w)
                    // Metal's bilinear filtering and RGBA16Float writeback
                    // contribute rounding. Bound HDR results in output ULPs.
                    for channel in 0..<3 {
                        let tolerance = max(0.005, 2 * Float(Float16(expected[channel]).ulp))
                        #expect(abs(actual[channel] - expected[channel]) <= tolerance,
                                "channel \(channel), \(actual) vs \(expected)")
                    }
                    #expect(actual.w == base.w)
                    if base.w == 0 { #expect(actual == .zero) }
                }
            }
        }
    }

    @Test("The native bloom pyramid feeds the graph in one command and keeps transparent edges clean")
    func nativeBloomField() throws {
        let width = 17, height = 13
        let center = height / 2 * width + width / 2
        let source = try gpu.color(width: width, height: height) { x, y in
            if x == width / 2 && y == height / 2 { return SIMD4(12, 6, 3, 1) }
            return x < width / 2 ? .zero : SIMD4(0.125, 0.0625, 0.03125, 0.25)
        }
        let glow = try gpu.color(width: width, height: height)
        let ordinary = try gpu.color(width: width, height: height)
        let composed = try gpu.color(width: width, height: height)
        let dirt = try gpu.color(width: 1, height: 1, pixel: { _, _ in SIMD4(0.5, 0.25, 1, 1) })
        let bloom = try MipmapBloom(
            device: gpu.device, settings: .init(
                strength: 2, radius: 0.6, levels: 3, threshold: 1, knee: 0, fireflySuppression: false, blending: .additive
            )
        )
        let pass = try PostProcessPass(
            graph: .lensDirt(dirtSize: SIMD2(1, 1), intensity: 2), device: gpu.device,
            textures: ["Bloom": glow, "Dirt": dirt]
        )
        try gpu.run { command in
            try bloom.encode(source: source, destination: ordinary, commandBuffer: command)
            try bloom.encodeGlow(source: source, destination: glow, commandBuffer: command)
            try pass.encode(source: source, destination: composed, commandBuffer: command)
        }

        let base = try gpu.read(source)
        let field = try gpu.read(glow)
        let oldComposite = try gpu.read(ordinary)
        let output = try gpu.read(composed)
        #expect(field[center - 1].x > 0 && output[center - 1] == .zero)
        #expect(output[center].x > oldComposite[center].x && oldComposite[center].x > base[center].x)
        for index in field.indices {
            let color = base[index]
            let scattered = field[index]
            let rgb = SIMD3(scattered.x, scattered.y, scattered.z)
            let sourceRGB = SIMD3(color.x, color.y, color.z)
            #expect(scattered.w == 1)
            #expect(PostProcessFixture.close(oldComposite[index], SIMD4(sourceRGB + rgb, color.w), tolerance: 0.01))
            let expected = SIMD4(sourceRGB + rgb * SIMD3(2, 1.5, 3) * color.w, color.w)
            #expect(PostProcessFixture.close(output[index], expected, tolerance: 0.01))
        }
    }

    @Test("Lens dirt rejects invalid dimensions and intensity, and requires both named textures")
    func inputValidation() throws {
        for size in [SIMD2(0, 1), SIMD2(1, -1)] {
            #expect(throws: PostProcessEffectError.self) { try PostProcessGraph.lensDirt(dirtSize: size) }
        }
        for intensity: Float in [-1, .nan, .infinity] {
            #expect(throws: PostProcessEffectError.self) {
                try PostProcessGraph.lensDirt(dirtSize: SIMD2(1, 1), intensity: intensity)
            }
        }
        let graph = try PostProcessGraph.lensDirt(dirtSize: SIMD2(1, 1))
        #expect(throws: PostProcessPass.Failure.missingTexture("Bloom")) { try PostProcessPass(graph: graph, device: gpu.device) }
        let bloom = try gpu.color(width: 3, height: 3)
        #expect(throws: PostProcessPass.Failure.missingTexture("Dirt")) {
            try PostProcessPass(graph: graph, device: gpu.device, textures: ["Bloom": bloom])
        }
    }

    private func sampleRGB(_ uv: SIMD2<Float>, size: SIMD2<Int>, pixel: (Int, Int) -> SIMD4<Float>) -> SIMD3<Float> {
        let position = uv * SIMD2<Float>(size) - 0.5
        let floorPosition = floor(position)
        let index = SIMD2<Int>(floorPosition)
        let fraction = position - floorPosition
        func read(_ x: Int, _ y: Int) -> SIMD3<Float> {
            let texel = pixel(min(max(x, 0), size.x - 1), min(max(y, 0), size.y - 1))
            return SIMD3(Float(Float16(texel.x)), Float(Float16(texel.y)), Float(Float16(texel.z)))
        }
        let top = read(index.x, index.y) * (1 - fraction.x) + read(index.x + 1, index.y) * fraction.x
        let bottom = read(index.x, index.y + 1) * (1 - fraction.x) + read(index.x + 1, index.y + 1) * fraction.x
        return top * (1 - fraction.y) + bottom * fraction.y
    }
}
