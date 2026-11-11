import Frostfield
import Metal
import Testing
import simd

struct ScalarColorLookupTableTests {
    private typealias Table = ScalarColorLookupTable

    @Test("Built-in colors match sampled Three.js r187 palettes on a nonunit domain")
    func upstreamColors() throws {
        // Node reference run against THREE.REVISION === "187", new Lut(palette, 7).setMin(-3).setMax(11).
        let inputs: [Double] = [-4, -2, 2, 4, 8, 12]
        let reference: [(Table.Palette, [SIMD3<Float>])] = [
            (.rainbow, [
                SIMD3(0, 0, 1), SIMD3(0, 0.714285714, 1), SIMD3(0, 1, 0.238095238),
                SIMD3(0.238095238, 1, 0), SIMD3(1, 0.714285714, 0), SIMD3(1, 0, 0),
            ]),
            (.coolToWarm, [
                SIMD3(0.235294118, 0.305882353, 0.760784314), SIMD3(0.501400560, 0.614005602, 0.931652661),
                SIMD3(0.802054155, 0.832866480, 0.895424837), SIMD3(0.887021475, 0.809523810, 0.781512605),
                SIMD3(0.890756303, 0.461064426, 0.415126050), SIMD3(0.705882353, 0.015686275, 0.149019608),
            ]),
            (.blackbody, [
                .zero, SIMD3(0.336134454, 0, 0), SIMD3(0.799253035, 0.149393091, 0),
                SIMD3(0.925303455, 0.387488329, 0), SIMD3(1, 1, 0.285714286), .one,
            ]),
            (.grayscale, [
                .zero, SIMD3(repeating: 0.179271709), SIMD3(0.439215686, 0.439215686, 0.442203548),
                SIMD3(0.557796452, 0.557796452, 0.560784314), SIMD3(repeating: 0.820728291), .one,
            ]),
        ]
        for (palette, colors) in reference {
            let table = try Table(palette: palette, intervalCount: 7, domain: -3...11)
            #expect(table.samples.count == 8)
            for (input, expected) in zip(inputs, colors) {
                let actual = try #require(table.color(at: input))
                #expect(simd_distance(actual, expected) < 0.000001, "\(palette) at \(input)")
            }
        }
        #expect(try Table().samples.count == 33)
    }

    @Test("Custom stops preserve endpoints and round halfway toward the larger index")
    func customAndDomain() throws {
        let stops = try [
            Table.Stop(0, linearRGBHex: 0x204080), Table.Stop(0.25, linearRGBHex: 0x804020),
            Table.Stop(1, linearRGBHex: 0xFFFFFF),
        ]
        var table = try Table(stops: stops, intervalCount: 4)
        let original = table.samples
        #expect(table.color(at: -.infinity) == stops[0].linearRGB)
        #expect(table.color(at: 0.125) == stops[1].linearRGB)
        #expect(table.color(at: 0.37499) == stops[1].linearRGB)
        let between = try #require(table.color(at: 0.375))
        #expect(simd_distance(between, SIMD3(0.667973856, 0.500653595, 0.416993464)) < 0.000001)
        #expect(table.color(at: 0.875) == .one)
        #expect(table.color(at: .infinity) == .one)
        #expect(table.color(at: .nan) == nil)

        try table.setDomain(-2...2)
        #expect(table.samples == original)
        #expect(table.color(at: -1.5) == stops[1].linearRGB)
        #expect(table.color(at: 2) == .one)
        #expect(throws: Table.Failure.domain) { try table.setDomain(4...4) }
        #expect(table.domain == -2...2)
        let endpoints = try Table(palette: .blackbody, intervalCount: 1)
        #expect(endpoints.samples == [.zero, .one])
        #expect(endpoints.color(at: 0.49999) == .zero)
        #expect(endpoints.color(at: 0.5) == .one)

        let tiny = SIMD3<Float>(.leastNormalMagnitude, .leastNonzeroMagnitude, 0)
        let contrast = try Table(stops: [.init(0, linearRGB: SIMD3(repeating: 65_504)), .init(1, linearRGB: tiny)])
        #expect(contrast.color(at: 1) == tiny)
    }

    @Test("Colormaps reject invalid interval counts, domains, stops and colors")
    func validation() throws {
        let first = Table.Stop(0, linearRGB: .zero), last = Table.Stop(1, linearRGB: .one)
        for count in [0, -1, Table.maximumIntervalCount + 1] {
            #expect(throws: Table.Failure.intervalCount) { try Table(intervalCount: count) }
        }
        #expect(throws: Table.Failure.domain) { try Table(domain: 0 ... .infinity) }
        #expect(throws: Table.Failure.domain) {
            try Table(domain: -Double.greatestFiniteMagnitude ... Double.greatestFiniteMagnitude)
        }
        for invalid in [[], [first], [first, first, last], [last, first], [first, .init(.nan, linearRGB: .one), last],
                        [first, .init(0.8, linearRGB: .one)]] {
            #expect(throws: Table.Failure.stops) { try Table(stops: invalid) }
        }
        for color in [SIMD3<Float>(-0.25, 0, 0), SIMD3(0, .nan, 0), SIMD3(0, 0, 65_505)] {
            #expect(throws: Table.Failure.stopColor) { try Table(stops: [.init(0, linearRGB: color), last]) }
        }
        #expect(throws: Table.Failure.hexColor) { try Table.Stop(0, linearRGBHex: 0x1FF_FFFF) }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ScalarColorLookupTextureTests {
    private typealias Table = ScalarColorLookupTable

    @Test("A shader maps scalar input through the same horizontal samples as Swift")
    func sampledTexture() throws {
        let fixture = try PostProcessFixture()
        let table = try Table(palette: .coolToWarm, intervalCount: 4, domain: -1...1)
        let ramp = try table.makeTexture(device: fixture.device)
        #expect(ramp.width == 5 && ramp.height == 1 && ramp.pixelFormat == .rgba16Float)
        let lower = Float(table.domain.lowerBound)
        let span = Float(table.domain.upperBound - table.domain.lowerBound)
        let intervals = Float(table.intervalCount)
        let count = Float(table.samples.count)
        let graph = try PostProcessGraph { p in
            let value = p.sceneColor(at: p.screenUV).r
            let normalized = ((value - lower) / span).clamped(0, 1)
            let index = (normalized * intervals + 0.5).floor()
            let uv = p.vector((index + 0.5) / count, p.constant(Float(0.5)))
            return p.texture("Ramp", addressing: .clampToEdge, filtering: .nearest).sample(uv)
        }
        let pass = try PostProcessPass(graph: graph, device: fixture.device, textures: ["Ramp": ramp])
        let values: [Float] = [-2, -1, -0.875, -0.625, -0.125, 0.375, 0.875, 1, 2]
        let source = try fixture.color(width: values.count, height: 1) { x, _ in SIMD4(values[x], 0, 0, 1) }
        let target = try fixture.color(width: values.count, height: 1)
        try fixture.run { try pass.encode(source: source, destination: target, commandBuffer: $0) }
        let result = try fixture.read(target)
        for (value, pixel) in zip(values, result) {
            let rgb = try #require(table.color(at: Double(value)))
            #expect(pixel == SIMD4<Float>(SIMD4<Float16>(SIMD4(rgb, 1))))
        }
    }

    @Test("A vertical legend shows both endpoints, retains HDR, and updates in place")
    func legend() throws {
        let fixture = try RasterLightingFixture(width: 2, height: 3)
        let first = try Table(palette: .rainbow, intervalCount: 2)
        let second = try Table(stops: [.init(0, linearRGB: .zero), .init(1, linearRGB: SIMD3(2, 1, 0.5))], intervalCount: 2)
        let legend = try first.makeTexture(device: fixture.device, layout: .legend)
        let target = try fixture.colorTarget()
        let viewer = try TextureDebugPass(device: fixture.device)
        #expect(legend.width == 1 && legend.height == 3)
        func render() throws -> [SIMD4<Float>] {
            try fixture.run { try viewer.encode(source: .color(legend), destination: target, commandBuffer: $0) }
            return try fixture.pixels(of: target)
        }
        let expected: [SIMD4<Float>] = [SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1)]
        #expect(try render() == expected.flatMap { [$0, $0] })
        #expect(throws: Table.Failure.texture) { try second.write(to: legend) }
        #expect(try render() == expected.flatMap { [$0, $0] })
        try second.write(to: legend, layout: .legend)
        let updated: [SIMD4<Float>] = [SIMD4(2, 1, 0.5, 1), SIMD4(1, 0.5, 0.25, 1), SIMD4(0, 0, 0, 1)]
        #expect(try render() == updated.flatMap { [$0, $0] })
    }
}
