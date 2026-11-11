import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RetroCRTTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Retro CRT matches the pinned image operators across animated, premultiplied HDR frames")
    func combinedReference() throws {
        let width = 19, height = 13, block = 3
        let curvature: Float = 0.24
        let levels: Float = 7, dither: Float = 0.75
        let intensity: Float = 0.55, density: Float = 0.8, speed: Float = 0.17
        let vignette: Float = 0.42, softness: Float = 0.65
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = [0, 0.25, 0.5, 1][(x + 2 * y) % 4]
            let straight = SIMD3<Float>(Float(x + 2) / 7.3, Float(y + 1) / 11.1, Float(x + y) / 13.7)
            return SIMD4(straight * alpha, alpha)
        }
        let destination = try gpu.color(width: width, height: height)
        let graph = try PostProcessGraph.retroCRT(
            pixelSize: block, curvature: curvature, colorLevels: Int(levels), ditherStrength: dither,
            scanlineIntensity: intensity, scanlineDensity: density, scanlineSpeed: speed,
            vignetteIntensity: vignette, vignetteSmoothness: softness
        )
        let pass = try PostProcessPass(graph: graph, device: gpu.device)
        let input = try gpu.read(source)
        var frames: [[SIMD4<Float>]] = []
        for time: Float in [0, 0.375] {
            try pass.uniforms.setFrame(ShaderFrame(time: Double(time)))
            try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
            let pixels = try gpu.read(destination)
            frames.append(pixels)
            #expect(pixels.contains { $0.x > 1 })
            for (index, actual) in pixels.enumerated() {
                let x = index % width, y = index / width
                let uv = (SIMD2(Float(x), Float(y)) + 0.5) / SIMD2(Float(width), Float(height))
                let screen = 2 * uv - 1
                let radiusSquared = simd_length_squared(screen)
                let warp = 0.5 + screen / (1 - radiusSquared * curvature) * (1 - 2 * curvature) * 0.5
                let sx = min(Int(floor(warp.x * Float(width) / Float(block))) * block + block / 2, width - 1)
                let sy = min(Int(floor(warp.y * Float(height) / Float(block))) * block + block / 2, height - 1)
                let color = input[sy * width + sx]
                if color.w == 0 {
                    #expect(actual == .zero)
                    continue
                }
                let bayer = Float(((x % 4 + 1) * (y % 4 + 1) * 17) % 16) / 16 - 0.5
                let rgb = SIMD3(color.x, color.y, color.z) / color.w
                let reduced = simd_max(floor((rgb + bayer / levels * dither) * levels) / levels, .zero)
                let fade = 1 - vignette * smoothstep(1.42 * (1 - softness), 1.42, sqrt(radiusSquared))
                let scan = 1 - intensity * (sin((uv.y - time * speed) * Float(height) * density) * 0.5 + 0.5)
                let expected = SIMD4(reduced * (fade * scan * color.w), color.w)
                #expect(PostProcessFixture.close(actual, expected, tolerance: 0.004),
                        "(\(x), \(y)) at \(time): \(actual) vs \(expected)")
                #expect(actual.w == color.w)
            }
        }
        #expect(frames[0] != frames[1])
    }

    @Test("Barrel-only CRT samples the pinned warped coordinates and can be disabled exactly")
    func barrelCoordinates() throws {
        let width = 15, height = 9
        let source = try gpu.color(width: width, height: height) { x, y in
            SIMD4(Float(x) * 0.125, Float(y) * 0.125, 0.5, 1)
        }
        let destination = try gpu.color(width: width, height: height)
        func render(_ curvature: Float) throws -> [SIMD4<Float>] {
            let pass = try PostProcessPass(
                graph: .retroCRT(
                    pixelSize: 1, curvature: curvature, colorLevels: 0, scanlineIntensity: 0, vignetteIntensity: 0
                ), device: gpu.device
            )
            try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
            return try gpu.read(destination)
        }
        #expect(try render(0) == gpu.read(source))
        for curvature: Float in [0.12, 0.49] {
            let warped = try render(curvature)
            for (x, y) in [(0, 0), (7, 0), (2, 2), (7, 4), (14, 4), (13, 8)] {
                let centered = (SIMD2(Float(x), Float(y)) + 0.5) / SIMD2(Float(width), Float(height)) * 2 - 1
                let warpedUV = 0.5 + centered * (1 - curvature * 2) / (1 - simd_length_squared(centered) * curvature) * 0.5
                let pixel = warpedUV * SIMD2(Float(width), Float(height)) - 0.5
                let expected = SIMD4(
                    min(max(pixel.x, 0), Float(width - 1)) / 8,
                    min(max(pixel.y, 0), Float(height - 1)) / 8, 0.5, 1
                )
                #expect(PostProcessFixture.close(warped[y * width + x], expected, tolerance: 0.002))
            }
        }
        #expect(throws: PostProcessEffectError.self) { try PostProcessGraph.retroCRT(curvature: 0.5) }
        #expect(throws: PostProcessEffectError.self) { try PostProcessGraph.retroCRT(colorLevels: -1) }
    }

    private func smoothstep(_ a: Float, _ b: Float, _ value: Float) -> Float {
        let t = min(max((value - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }
}
