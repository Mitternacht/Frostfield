import Foundation
import Frostfield
import Metal
import simd
import Testing

struct ToonShaderTests {
    @Test("The example toon shaders match the r187 fragment formulas on the CPU and GPU")
    func formulas() throws {
        let lighting = ToonShaders.Lighting(
            direction: SIMD3(0, 1, 0), color: SIMD3(repeating: 0.2), ambient: SIMD3(repeating: 0.05)
        )
        let directional = ToonShaders.directional(
            normal: SIMD3(0, 0, 1), refraction: SIMD3(0, 0, -1), base: SIMD3(0.8, 0.2, 0.1), lighting: lighting
        )
        #expect(abs(directional.x - 0.5685055) < 1e-5)
        #expect(abs(directional.y - 0.14212638) < 1e-5)
        #expect(abs(directional.z - 0.07106319) < 1e-5)

        let dim = ToonShaders.Lighting(direction: SIMD3(0, 0, 1), color: SIMD3(repeating: 1), ambient: SIMD3(0.1, 0, 0))
        let lines = (SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1), SIMD3<Float>(1, 1, 0))
        let hatched = ToonShaders.hatching(
            normal: SIMD3(1, 0, 0), fragment: SIMD2(5, 0), base: SIMD3(repeating: 1), lines: lines, lighting: dim
        )
        #expect(hatched == SIMD3(1, 1, 0))
        let plain = ToonShaders.hatching(
            normal: SIMD3(0, 0, 1), fragment: SIMD2(1, 1), base: SIMD3(repeating: 1), lines: lines, lighting: dim
        )
        #expect(plain == SIMD3(repeating: 1))

        let dotted = ToonShaders.dotted(
            normal: SIMD3(1, 0, 0), fragment: SIMD2(3.5, 3.5), base: SIMD3(repeating: 1), line: .zero, lighting: dim
        )
        #expect(dotted == .zero)
        let open = ToonShaders.dotted(
            normal: SIMD3(0, 0, 1), fragment: SIMD2(0.5, 0.5), base: SIMD3(repeating: 1), line: .zero, lighting: dim
        )
        #expect(open == SIMD3(repeating: 1))

        let device = try #require(MTLCreateSystemDefaultDevice())
        let samples = [
            ToonShaders.GPUSample(
                kind: .directional, normal: SIMD3(0, 0, 1), lightDirection: lighting.direction,
                lightColor: lighting.color, ambient: lighting.ambient, base: SIMD3(0.8, 0.2, 0.1),
                refraction: SIMD3(0, 0, -1)
            ),
            ToonShaders.GPUSample(
                kind: .hatching, normal: SIMD3(1, 0, 0), lightDirection: dim.direction, lightColor: dim.color,
                ambient: dim.ambient, base: SIMD3(repeating: 1), fragment: SIMD2(5, 0),
                line1: lines.0, line2: lines.1, line3: lines.2, line4: lines.3
            ),
            ToonShaders.GPUSample(
                kind: .dotted, normal: SIMD3(1, 0, 0), lightDirection: dim.direction, lightColor: dim.color,
                ambient: dim.ambient, base: SIMD3(repeating: 1), fragment: SIMD2(3.5, 3.5)
            ),
        ]
        let gpu = try ToonShaders.evaluateOnGPU(samples, device: device)
        #expect(simd_distance(gpu[0], directional) < 1e-4)
        #expect(simd_distance(gpu[1], hatched) < 1e-4)
        #expect(simd_distance(gpu[2], dotted) < 1e-4)
    }
}
