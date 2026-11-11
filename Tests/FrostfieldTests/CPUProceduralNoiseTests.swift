import Frostfield
import Testing

struct CPUProceduralNoiseTests {
    @Test("Improved Perlin and seeded simplex match the pinned r187 CPU reference")
    func r187Reference() throws {
        let table = (0..<256).map { UInt8((73 * $0 + 19) & 255) }
        let simplex = try SimplexNoise(table: table)
        let cases: [(SIMD4<Double>, SIMD4<Double>)] = [
            (.zero, .zero),
            (SIMD4(0.125, 0.25, 0.5, 0.75),
             SIMD4(0.34486936032772064, 0.24634863722264994, -0.38333843237480503, -0.0825209253226482)),
            (SIMD4(-0.625, 1.125, -2.25, 4.5),
             SIMD4(-0.053058378398418427, 0.296574308414657, -0.4916267123752164, 0.04425657365654824)),
            (SIMD4(repeating: 0.125),
             SIMD4(0.22147832005862256, 0.42029948831583114, -0.024160768890380852, 0.2513179643706)),
            (SIMD4(repeating: -4.75),
             SIMD4(0.031990673393011093, -0.4281330117314901, -0.4632503906249999, 0.7068411679269062)),
            (SIMD4(2.125, -17.75, 11.0625, 3.4375),
             SIMD4(-0.068254196997187933, -0.34502496595050924, 0.17794769744215849, 0.06362659456178431)),
            (SIMD4(256.125, -256.75, 512.5, -512.25),
             SIMD4(0.058535732328891754, 0.35135389856046662, 0.31159430147807748, -0.02660557538823548)),
            (SIMD4(100001.125, -200008.5, 999.0625, -1201.25),
             SIMD4(-0.12355582006784971, -0.2946415567590267, -0.34450388181064839, -0.19340861576991064)),
            (SIMD4(16777216.25, -16777216.125, 0.75, -0.375),
             SIMD4(-0.0085898555116727948, 0.61755690214288717, -0.066328550946735898, 0.051198768143754049)),
        ]
        for (point, expected) in cases {
            let xyz = SIMD3(point.x, point.y, point.z)
            let actual = SIMD4(
                ShaderNoise.improvedPerlin(xyz), simplex.noise(SIMD2(point.x, point.y)),
                simplex.noise(xyz), simplex.noise(point)
            )
            for algorithm in 0..<4 {
                #expect(abs(actual[algorithm] - expected[algorithm]) < 1e-12,
                        "point \(point), algorithm \(algorithm)")
            }
        }
    }

    @Test("Noise handles seeded repeats, wrapped cells and invalid coordinates")
    func repeatabilityAndBounds() throws {
        let point = SIMD3<Double>(-0.875, 0.25, 3.125)
        #expect(ShaderNoise.improvedPerlin(point) == ShaderNoise.improvedPerlin(point + SIMD3(256, -512, 256)))
        #expect(ShaderNoise.improvedPerlin(SIMD3(1e100, 0, 0)) == 0)
        let first = SimplexNoise(seed: 95)
        #expect(first.noise(point) == SimplexNoise(seed: 95).noise(point))
        #expect(first.noise(point) != SimplexNoise(seed: 96).noise(point))
        #expect(first.noise(SIMD2(.infinity, 0)).isNaN)
        #expect(first.noise(SIMD4<Double>(repeating: .greatestFiniteMagnitude)).isNaN)
        #expect(ShaderNoise.improvedPerlin(SIMD3(.nan, 0, 0)).isNaN)
        #expect(throws: SimplexNoise.Failure.tableSize(255)) {
            try SimplexNoise(table: [UInt8](repeating: 0, count: 255))
        }
        let repeated = try SimplexNoise(table: [UInt8](repeating: 0, count: 256))
        #expect(repeated.noise(point).isFinite)
    }
}
