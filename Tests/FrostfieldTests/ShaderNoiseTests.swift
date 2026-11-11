import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderNoiseTests {
    /// Evaluates every noise node at the points in field 0 and returns fields 1–3.
    private func gpuNoise(_ points: [SIMD3<Float>]) throws -> [[SIMD4<Float>]] {
        let graph = try BufferShaderGraph(fieldCount: 4) { b in
            let p = b.elementField(0).xyz
            let cellular = b.cellularNoise(p)
            return [
                1: b.vector(
                    b.noise(p, kind: .value), b.noise(p, kind: .gradient),
                    b.fractalNoise(p, kind: .value, octaves: 4, lacunarity: 2.17, gain: 0.55),
                    b.fractalNoise(p, kind: .gradient, octaves: 6, lacunarity: 1.93, gain: 0.5)
                ),
                2: b.vector(cellular.x, cellular.y, p.hash(), b.interleavedGradientNoise(p.xy * 37)),
                3: b.vector(p.x.hash(), p.xy.hash(), b.elementField(0).hash(), b.constant(Float(0))),
            ]
        }
        let elements = points.map { [SIMD4($0, $0.x - $0.z), .zero, .zero, .zero] }
        return try GraphProbe().run(graph, elements: elements).map { Array($0[1...3]) }
    }

    @Test("CPU noise, hash and dithering equal the GPU nodes within 1e-6 at 1,000 random points")
    func cpuMatchesGPU() throws {
        var random = SeededGenerator(seed: 0x5EED_0F_4015E)
        var points: [SIMD3<Float>] = (0..<1_000).map { _ in
            SIMD3(Float.random(in: -60...60, using: &random), Float.random(in: -60...60, using: &random),
                  Float.random(in: -60...60, using: &random))
        }
        // Lattice corners, cell boundaries, negative zero and the clamp range.
        points[0] = SIMD3(0, 0, 0)
        points[1] = SIMD3(-0.0, 1, -1)
        points[2] = SIMD3(1e7, -3e6, 0.5)
        points[3] = SIMD3(0.999_999_94, -0.000_000_1, 2)
        let gpu = try gpuNoise(points)
        var worst: Float = 0
        var exactHashes = 0
        for (point, result) in zip(points, gpu) {
            let w = point.x - point.z
            let cellular = ShaderNoise.cellular(point)
            let cpu: [SIMD4<Float>] = [
                SIMD4(
                    ShaderNoise.value(point), ShaderNoise.gradient(point),
                    ShaderNoise.fractal(point, kind: .value, octaves: 4, lacunarity: 2.17, gain: 0.55),
                    ShaderNoise.fractal(point, kind: .gradient, octaves: 6, lacunarity: 1.93, gain: 0.5)
                ),
                SIMD4(
                    cellular.x, cellular.y, ShaderNoise.hash(point),
                    ShaderNoise.interleavedGradient(SIMD2(point.x, point.y) * 37)
                ),
                SIMD4(ShaderNoise.hash(point.x), ShaderNoise.hash(SIMD2(point.x, point.y)),
                      ShaderNoise.hash(SIMD4(point, w)), 0),
            ]
            for field in 0..<3 {
                let difference = GraphProbe.maximumDifference(result[field], cpu[field])
                worst = max(worst, difference)
                #expect(difference <= 1e-6, "\(point) field \(field): GPU \(result[field]) CPU \(cpu[field])")
            }
            // Hashes depend only on integer arithmetic, so they match exactly.
            if result[1].z == cpu[1].z, result[2].x == cpu[2].x, result[2].y == cpu[2].y, result[2].z == cpu[2].z {
                exactHashes += 1
            }
        }
        #expect(exactHashes == points.count)
        print("ShaderNoise largest CPU–GPU difference over \(points.count) points: \(worst)")
    }

    @Test("Noise stays in range, hashes ignore the sign of zero, and nonfinite points read as the origin")
    func ranges() throws {
        #expect(ShaderNoise.hash(Float(0)) == ShaderNoise.hash(Float(-0.0)))
        #expect(ShaderNoise.hash(SIMD2<Float>(1, 2)) != ShaderNoise.hash(SIMD2<Float>(2, 1)))
        #expect(ShaderNoise.value(SIMD3(.nan, 0, 0)) == ShaderNoise.value(.zero))
        var random = SeededGenerator(seed: 7)
        for _ in 0..<2_000 {
            let p = SIMD3(Float.random(in: -500...500, using: &random), Float.random(in: -500...500, using: &random),
                          Float.random(in: -500...500, using: &random))
            for value in [ShaderNoise.value(p), ShaderNoise.gradient(p), ShaderNoise.fractal(p), ShaderNoise.hash(p)] {
                #expect((0...1).contains(value))
            }
            let cell = ShaderNoise.cellular(p)
            #expect(cell.x >= 0 && cell.x <= Float(3).squareRoot() && (0..<1).contains(cell.y))
            #expect((0..<1).contains(ShaderNoise.interleavedGradient(SIMD2(p.x, p.y))))
        }
    }
}
