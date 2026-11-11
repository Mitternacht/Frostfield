import Foundation
import Frostfield
import Metal
import simd
import Testing

/// Oscillators, coordinate utilities, triplanar mapping, parallax occlusion, bit
/// operations, dithering, matrices and derivatives, each evaluated on the GPU and
/// compared with a CPU reference or a closed form.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderPatternTests {
    private let probe: GraphProbe

    init() throws {
        probe = try GraphProbe()
    }

    private static func fract(_ x: Float) -> Float { min(x - x.rounded(.down), 0x1.fffffep-1) }

    @Test("Oscillators match their closed forms at sample points, componentwise for vectors")
    func oscillators() throws {
        let times: [Float] = [-1.3, -0.25, 0, 0.1, 0.25, 0.4999, 0.5, 0.75, 1, 1.25, 2.6, 17.125]
        let scalar = try probe.map(times.map { SIMD4($0, 0, 0, 0) }) { b, field in
            let t = field.x
            return b.vector(
                b.oscillator(.sine, t), b.oscillator(.square, t), b.oscillator(.triangle, t), b.oscillator(.sawtooth, t)
            )
        }
        let vector = try probe.map(times.map { SIMD4($0, $0 * 0.5, 0, 0) }) { b, field in
            let t = field.xy
            return b.vector(t.oscillated(.sine), t.oscillated(.triangle))
        }
        // The sine reference uses double precision; a float phase of 2πt carries about
        // |t| × 2⁻²² radians of rounding, which bounds the tolerance for |t| < 20.
        func sine(_ t: Float) -> Float { Float(0.5 - 0.5 * cos(2 * Double.pi * Double(t))) }
        for ((t, result), pair) in zip(zip(times, scalar), vector) {
            let expected = SIMD4<Float>(
                sine(t), Self.fract(t) >= 0.5 ? 1 : 0, abs(Self.fract(t + 0.5) * 2 - 1), Self.fract(t)
            )
            #expect(GraphProbe.maximumDifference(result, expected) <= 1e-5, "t = \(t): \(result) vs \(expected)")
            #expect(SIMD3(result.y, result.z, result.w) == SIMD3(expected.y, expected.z, expected.w), "t = \(t)")
            let half = t * 0.5
            let expectedPair = SIMD4<Float>(
                sine(t), sine(half), abs(Self.fract(t + 0.5) * 2 - 1), abs(Self.fract(half + 0.5) * 2 - 1)
            )
            #expect(GraphProbe.maximumDifference(pair, expectedPair) <= 1e-5, "t = \(t): \(pair)")
        }
        // Each wave starts at zero, as Three.js oscillators do.
        let quarter = zip(times, scalar).filter { [0, 0.25, 0.5, 0.75].contains($0.0) }.map(\.1)
        let expectedQuarters: [SIMD4<Float>] = [SIMD4(0, 0, 0, 0), SIMD4(0.5, 0, 0.5, 0.25), SIMD4(1, 1, 1, 0.5),
                                                SIMD4(0.5, 1, 0.5, 0.75)]
        for (result, expected) in zip(quarter, expectedQuarters) {
            #expect(GraphProbe.maximumDifference(result, expected) <= 1e-6, "\(result) vs \(expected)")
        }
    }

    @Test("rotateUV, spherizeUV, polar and checker match closed forms")
    func coordinates() throws {
        let points: [SIMD2<Float>] = [SIMD2(1, 0.5), SIMD2(0.5, 1), SIMD2(0, 0.5), SIMD2(0.5, 0), SIMD2(0.5, 0.5),
                                      SIMD2(0.2, 0.9), SIMD2(-0.3, 1.7), SIMD2(2.25, -1.75)]
        let inputs = points.map { SIMD4($0.x, $0.y, 0, 0) }
        let transformed = try probe.map(inputs) { b, field in
            let uv = field.xy
            return b.vector(
                b.rotateUV(uv, angle: b.constant(Float.pi / 2)).x, b.rotateUV(uv, angle: b.constant(Float(0.7))).y,
                b.spherizeUV(uv, strength: b.constant(Float(2))).x, b.spherizeUV(uv, strength: b.constant(Float(2))).y
            )
        }
        let polar = try probe.map(inputs) { b, field in
            let uv = field.xy
            let p = b.polar(uv)
            return b.vector(p.x, p.y, b.checker(uv * 4), b.checker(uv))
        }
        for ((uv, rotated), polarResult) in zip(zip(points, transformed), polar) {
            let d = uv - 0.5
            let quarter = SIMD2<Float>(0.5, 0.5) + SIMD2(cos(Float.pi / 2) * d.x - sin(Float.pi / 2) * d.y,
                                                         sin(Float.pi / 2) * d.x + cos(Float.pi / 2) * d.y)
            let tilted = 0.5 + sin(Float(0.7)) * d.x + cos(Float(0.7)) * d.y
            let length2 = simd_dot(d, d)
            let spherized = uv + d * (length2 * length2 * 2)
            let expected = SIMD4(quarter.x, tilted, spherized.x, spherized.y)
            #expect(GraphProbe.maximumDifference(rotated, expected) <= 1e-6, "\(uv): \(rotated) vs \(expected)")
            func checker(_ p: SIMD2<Float>) -> Float {
                let sum = p.x.rounded(.down) + p.y.rounded(.down)
                return sum - 2 * (sum * 0.5).rounded(.down)
            }
            let turns = Self.fract(atan2(d.y, d.x) / (2 * .pi))
            let expectedPolar = SIMD4(simd_length(d), turns, checker(uv * 4), checker(uv))
            #expect(GraphProbe.maximumDifference(polarResult, expectedPolar) <= 1e-6, "\(uv): \(polarResult)")
        }
        // A quarter turn carries (1, 0.5) to (0.5, 1); polar angles in turns follow atan2.
        #expect(GraphProbe.maximumDifference(transformed[0], SIMD4(0.5, transformed[0].y, transformed[0].z, transformed[0].w)) < 1e-6)
        #expect(abs(polar[0].y) < 1e-6 && abs(polar[1].y - 0.25) < 1e-6 && abs(polar[2].y - 0.5) < 1e-6
                    && abs(polar[3].y - 0.75) < 1e-6)
    }

    @Test("Triplanar weights sum to 1, follow |n|^k, and blend the yz, zx and xy projections")
    func triplanar() throws {
        var random = SeededGenerator(seed: 11)
        var normals: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), .zero, SIMD3(1, 1, 1)]
        for _ in 0..<300 {
            normals.append(SIMD3(Float.random(in: -1...1, using: &random), Float.random(in: -1...1, using: &random),
                                 Float.random(in: -1...1, using: &random)))
        }
        for sharpness: Float in [0, 1, 4, 8.5] {
            let weights = try probe.map(normals.map { SIMD4($0, 0) }) { b, field in
                b.vector(b.triplanarWeights(normal: field.xyz, sharpness: sharpness), b.constant(Float(0)))
            }
            for (normal, w) in zip(normals, weights) {
                #expect(abs(w.x + w.y + w.z - 1) <= 1e-6, "k \(sharpness), n \(normal): \(w)")
                let raised = SIMD3(pow(abs(normal.x), sharpness), pow(abs(normal.y), sharpness), pow(abs(normal.z), sharpness))
                let total = raised.x + raised.y + raised.z
                let expected = total > 0 ? raised / total : SIMD3(repeating: 1.0 / 3.0)
                #expect(simd_reduce_max(simd_abs(SIMD3(w.x, w.y, w.z) - expected)) <= 2e-5, "k \(sharpness), n \(normal): \(w)")
            }
            if sharpness > 0 {
                #expect(weights[0] == SIMD4(1, 0, 0, 0) && weights[1] == SIMD4(0, 1, 0, 0) && weights[2] == SIMD4(0, 0, 1, 0))
            }
            #expect(GraphProbe.maximumDifference(weights[3], SIMD4(1.0 / 3, 1.0 / 3, 1.0 / 3, 0)) <= 1e-7)
        }

        // High sharpness and short normals, where every |n|^k underflows in float: the
        // weights depend only on the direction, so compare with a double-precision ratio.
        let extreme: [(normal: SIMD3<Float>, sharpness: Float)] = [
            (simd_normalize(SIMD3(1, 0.98, 0.97)), 200), (SIMD3(0.001, 0.002, 0.003), 16),
            (SIMD3(-2e-20, 1e-20, 3e-20), 4), (simd_normalize(SIMD3(0.2, -1, 0.5)), 1_000), (SIMD3(1e-30, 0, 0), 2),
        ]
        for (normal, sharpness) in extreme {
            let result = try probe.map([SIMD4(normal, 0)]) { b, field in
                b.vector(b.triplanarWeights(normal: field.xyz, sharpness: sharpness), b.constant(Float(0)))
            }[0]
            let a = SIMD3<Double>(abs(Double(normal.x)), abs(Double(normal.y)), abs(Double(normal.z)))
            let ratio = a / a.max()
            let raised = SIMD3(pow(ratio.x, Double(sharpness)), pow(ratio.y, Double(sharpness)), pow(ratio.z, Double(sharpness)))
            let expected = SIMD3<Float>(raised / (raised.x + raised.y + raised.z))
            #expect(simd_reduce_max(simd_abs(SIMD3(result.x, result.y, result.z) - expected)) <= 2e-5,
                    "k \(sharpness), n \(normal): \(result) vs \(expected)")
        }

        // Three 8 × 8 textures whose texels record their column, row and projection.
        func plane(_ id: Float) throws -> any MTLTexture {
            var texels: [SIMD4<Float>] = []
            for row in 0..<8 {
                for column in 0..<8 { texels.append(SIMD4(Float(column) / 8, Float(row) / 8, id, 1)) }
            }
            return try probe.texture(width: 8, height: 8, texels: texels)
        }
        func lookup(_ uv: SIMD2<Float>, _ id: Float) -> SIMD4<Float> {
            let column = Float(Int(Self.fract(uv.x) * 8)), row = Float(Int(Self.fract(uv.y) * 8))
            return SIMD4(column / 8, row / 8, id, 1)
        }
        // Positions at texel centers of every projection, with normals of mixed orientation.
        let samples: [(SIMD3<Float>, SIMD3<Float>)] = (0..<40).map { index in
            let p = SIMD3(Float(index % 8), Float((index * 3) % 8), Float((index * 5) % 8)) / 8 + 1.0 / 16
            return (p, simd_normalize(SIMD3(Float.random(in: -1...1, using: &random), Float.random(in: -1...1, using: &random),
                                            Float.random(in: 0.1...1, using: &random))))
        }
        let textures = ["X": try plane(0.25), "Y": try plane(0.5), "Z": try plane(0.75)]
        let graph = try BufferShaderGraph(fieldCount: 4) { b in
            let x = b.texture("X", content: .data, filtering: .nearest)
            let y = b.texture("Y", content: .data, filtering: .nearest)
            let z = b.texture("Z", content: .data, filtering: .nearest)
            let p = b.elementField(0).xyz
            let n = b.elementField(1).xyz
            return [2: b.triplanar(x: x, y: y, z: z, position: p, normal: n, sharpness: 2),
                    3: x.triplanar(position: p, normal: n, sharpness: 2)]
        }
        let results = try probe.run(graph, elements: samples.map { [SIMD4($0.0, 0), SIMD4($0.1, 0), .zero, .zero] },
                                    textures: textures)
        for ((p, n), result) in zip(samples, results) {
            let squared = n * n
            let w = squared / (squared.x + squared.y + squared.z)
            let yz = SIMD2(p.y, p.z), zx = SIMD2(p.z, p.x), xy = SIMD2(p.x, p.y)
            let blended = lookup(yz, 0.25) * w.x + lookup(zx, 0.5) * w.y + lookup(xy, 0.75) * w.z
            let single = lookup(yz, 0.25) * w.x + lookup(zx, 0.25) * w.y + lookup(xy, 0.25) * w.z
            #expect(GraphProbe.maximumDifference(result[2], blended) <= 2e-6, "\(p) \(n): \(result[2]) vs \(blended)")
            #expect(GraphProbe.maximumDifference(result[3], single) <= 2e-6, "\(p) \(n): \(result[3]) vs \(single)")
        }
    }

    /// The parallax march on the CPU, sampling a 1-row height texture with nearest
    /// filtering and clamped addressing, as the compute stage does.
    private static func parallax(
        uv: SIMD2<Float>, view: SIMD3<Float>, depth: Float, steps: Int, heights: [Float]
    ) -> (uv: SIMD2<Float>, depth: Float) {
        func surfaceDepth(_ coordinates: SIMD2<Float>) -> Float {
            let index = min(max(Int((coordinates.x * Float(heights.count)).rounded(.down)), 0), heights.count - 1)
            return 1 - heights[index]
        }
        let v = simd_normalize(view)
        let shift = -SIMD2(v.x, v.y) / max(v.z, 0.05) * depth
        var previousDepth: Float = 0
        var previousGap = -surfaceDepth(uv)
        var hit: Float = 0
        var found = previousGap >= 0
        for layer in 1...steps {
            let layerDepth = Float(layer) / Float(steps)
            let gap = layerDepth - surfaceDepth(uv + shift * layerDepth)
            if !found && gap >= 0 {
                hit = previousDepth + (layerDepth - previousDepth) * (previousGap / (previousGap - gap))
                found = true
            }
            previousDepth = layerDepth
            previousGap = gap
        }
        return (uv + shift * hit, hit)
    }

    private func parallaxGraph(steps: Int) throws -> BufferShaderGraph {
        try BufferShaderGraph(fieldCount: 3) { b in
            let height = b.texture("Height", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let input = b.elementField(0)
            let uv = height.parallaxOcclusion(input.xy, view: b.elementField(1).xyz, depth: input.z, steps: steps)
            return [2: b.vector(uv, b.constant(Float(0)), b.constant(Float(0)))]
        }
    }

    @Test("Parallax occlusion offsets planar height fields exactly and finds a height step within one layer")
    func parallaxOcclusion() throws {
        let views: [SIMD3<Float>] = [SIMD3(0, 0, 1), SIMD3(0.3, 0.1, 0.9), SIMD3(-0.5, 0.4, 0.6), SIMD3(0.9, 0, 0.1),
                                     SIMD3(0.99, 0.05, 0.02), SIMD3(0, 0, -1), SIMD3(0.2, -0.7, 0.4)]
        let graph = try parallaxGraph(steps: 8)
        for height: Float in [1, 0.75, 0.5, 0.2, 0] {
            let texture = try probe.texture(width: 4, height: 4, texels: Array(repeating: SIMD4(height, 0, 0, 1), count: 16))
            let elements = views.map { [SIMD4<Float>(0.4, 0.6, 0.1, 0), SIMD4($0, 0), .zero] }
            let results = try probe.run(graph, elements: elements, textures: ["Height": texture])
            for (view, result) in zip(views, results) {
                let v = simd_normalize(view)
                // A planar field at depth (1 − h) · scale shifts the coordinates by −(v.xy / v.z) times that depth.
                let offset = v.z > 0 ? -SIMD2(v.x, v.y) / max(v.z, 0.05) * 0.1 * (1 - height) : .zero
                let expected = SIMD2<Float>(0.4, 0.6) + offset
                #expect(simd_reduce_max(simd_abs(SIMD2(result[2].x, result[2].y) - expected)) <= 1e-6,
                        "h \(height), v \(view): \(result[2]) vs \(expected)")
            }
        }

        // Invalid depths and coordinates return the coordinates unchanged.
        let flat = try probe.texture(width: 4, height: 4, texels: Array(repeating: SIMD4(0, 0, 0, 1), count: 16))
        let invalid: [SIMD4<Float>] = [SIMD4(0.4, 0.6, -0.1, 0), SIMD4(0.4, 0.6, 0, 0), SIMD4(0.4, 0.6, .nan, 0),
                                       SIMD4(0.4, 0.6, .infinity, 0)]
        let unchanged = try probe.run(graph, elements: invalid.map { [$0, SIMD4(0.3, 0.1, 0.9, 0), .zero] },
                                      textures: ["Height": flat])
        for (input, result) in zip(invalid, unchanged) {
            #expect(SIMD2(result[2].x, result[2].y) == SIMD2(0.4, 0.6), "depth \(input.z): \(result[2])")
        }
        // The helper always runs its fixed march, so every pixel of a quad takes every sample:
        // its only return is the last statement.
        let source = try graph.metalFunction(named: "march")
        let helper = try #require(source.components(separatedBy: "// parallax occlusion").last)
        let body = try #require(helper.components(separatedBy: "\n}\n").first)
        #expect(body.components(separatedBy: "return ").count == 2, "\(body)")
        #expect(body.contains("return valid ? uv + shift * hitDepth : uv;"), "\(body)")

        // A 16-texel step: the left half is the surface, the right half lies a full scale below.
        let step = (0..<16).map { $0 < 8 ? Float(1) : 0 }
        let stepTexture = try probe.texture(width: 16, height: 1, texels: step.map { SIMD4($0, 0, 0, 1) })
        let cases: [(uv: SIMD2<Float>, view: SIMD3<Float>, wall: Float?)] = [
            (SIMD2(0.55, 0.5), SIMD3(0.5, 0, 1), 0.5),     // marches left into the wall at u = 0.5
            (SIMD2(0.53, 0.5), SIMD3(1, 0.2, 1), 0.15),    // reaches the wall early
            (SIMD2(0.75, 0.5), SIMD3(-0.5, 0, 1), nil),    // marches right onto the floor
            (SIMD2(0.25, 0.5), SIMD3(-0.5, 0, 1), nil),    // starts on the top surface
        ]
        for steps in [8, 32] {
            let graph = try parallaxGraph(steps: steps)
            let elements = cases.map { [SIMD4($0.uv.x, $0.uv.y, 0.2, 0), SIMD4($0.view, 0), .zero] }
            let results = try probe.run(graph, elements: elements, textures: ["Height": stepTexture])
            for (item, result) in zip(cases, results) {
                let reference = Self.parallax(uv: item.uv, view: item.view, depth: 0.2, steps: steps, heights: step)
                #expect(simd_reduce_max(simd_abs(SIMD2(result[2].x, result[2].y) - reference.uv)) <= 1e-6,
                        "\(steps) layers from \(item.uv): \(result[2]) vs \(reference.uv)")
                let v = simd_normalize(item.view)
                let shift = -v.x / v.z * 0.2
                if let wall = item.wall {
                    // The march brackets the wall to one layer; the true hit lies at the wall depth.
                    #expect(abs(reference.depth - wall) <= 1 / Float(steps), "\(steps) layers: \(reference.depth) vs \(wall)")
                    let wallU: Float = item.uv.x + shift * wall
                    #expect(abs(wallU - 0.5) <= 1e-6)
                } else if item.uv.x < 0.5 {
                    #expect(reference.depth == 0)
                } else {
                    #expect(abs(reference.depth - 1) <= 1e-6)
                }
            }
        }
    }

    @Test("Bit operations act on clamped 24-bit integers and hashes ignore the sign of zero")
    func bitsAndHashes() throws {
        let pairs: [(Float, Float)] = [(12, 10), (3, 4), (48, 4), (5.9, 2.2), (16_777_215, 1), (20_000_000, 3),
                                       (-4, 7), (.nan, 1), (1, 24), (255, 8)]
        let results = try probe.map(pairs.map { SIMD4($0.0, $0.1, -0.0, 0) }) { b, field in
            let a = field.x, c = field.y
            let shifted = a.bits(.shiftLeft, c).bits(.shiftRight, c)
            return b.vector(
                a.bits(.and, c), a.bits(.or, c), a.bits(.xor, c),
                shifted + (field.z.hash() - b.constant(Float(0)).hash()) * 1_000
            )
        }
        func integer(_ x: Float) -> UInt32 { x.isFinite ? UInt32(min(max(x, 0), 16_777_215)) : 0 }
        for ((a, c), result) in zip(pairs, results) {
            let x = integer(a), y = integer(c)
            let roundTrip = y < 24 ? ((x << y) & 0xFF_FFFF) >> y : 0
            let expected = SIMD4(Float(x & y), Float(x | y), Float(x ^ y), Float(roundTrip))
            #expect(result == expected, "\(a), \(c): \(result) vs \(expected)")
        }
        #expect(results[0] == SIMD4(8, 14, 6, 12))
    }

    @Test("Ordered RGB dithering matches the r187 phase at every cell and wraps negative coordinates")
    func bayerDithering() throws {
        let input = (-4..<8).flatMap { y in
            (-4..<8).map { x in SIMD4(Float(x) + 0.25, Float(y) + 0.75, 64, 0) }
        }
        let color = SIMD3<Float>(0.125, 0.375, 0.75)
        let output = try probe.map(input) { builder, field in
            let base = builder.constant(color)
            let defaultResult = builder.bayerDither(base, pixel: field.xy)
            let authored = builder.bayerDither(base, pixel: field.xy, levels: field.z)
            return builder.vector(defaultResult, authored.x)
        }
        for (sample, actual) in zip(input, output) {
            let column = (Int(sample.x.rounded(.down)) + 4) % 4
            let row = (Int(sample.y.rounded(.down)) + 4) % 4
            let phase = Float(((column + 1) * (row + 1) * 17) % 16) / 16 - 0.5
            let expected = SIMD4(color + phase / 32, color.x + phase / 64)
            #expect(actual == expected)
        }
    }

    @Test("Simplex, curl and animated Voronoi match the r187 procedural patterns")
    func proceduralPatterns() throws {
        let inputs: [SIMD4<Float>] = [
            SIMD4(0.125, 0.25, 0.5, 0), SIMD4(-0.625, 1.125, -2.25, 1),
            SIMD4(0.125, 0.125, 0.125, 2), SIMD4(1.125, -0.75, 2.5, 0.5), .zero,
            SIMD4(0.125, 0.25, 0.5, 1.75), SIMD4(-2.25, -0.125, 1.25, 4),
        ]
        // Scalar float32 evaluation of the r187 TSL expressions, including ties
        // at simplex boundaries and two times at the same spatial coordinate.
        let expectedFields: [SIMD4<Float>] = [
            SIMD4(0.316197455, 2.804707289, -0.182127953, 0.374003649),
            SIMD4(0.458693892, -1.479346156, 2.569208622, -0.783197880),
            SIMD4(0.183348626, -3.300053596, 2.847882271, 3.703796864),
            SIMD4(0.686005354, -4.708547115, 0.693045855, -2.260531425),
            SIMD4(0.271711051, -2.350281477, 4.515526772, 1.848301768),
            SIMD4(0.316197455, 2.804707289, -0.182127953, 0.374003649),
            SIMD4(0.707455754, -2.152749062, 1.926661015, -0.611466467),
        ]
        let expectedDistances: [SIMD2<Float>] = [
            SIMD2(0.203125, 0.203125), SIMD2(0.160151184, 0.266208529),
            SIMD2(0.683998048, 0.768521845), SIMD2(0.506718278, 0.184859514),
            SIMD2(0.5, 0.083618090), SIMD2(0.613865376, 0.714444697), SIMD2(0.087208942, 0.150731742),
        ]
        let fields = try probe.map(inputs) { builder, field in
            let position = builder.vector(field.x, field.y, field.z)
            let curl = builder.curlNoise(position)
            return builder.vector(builder.simplexNoise(position), curl.x, curl.y, curl.z)
        }
        let distances = try probe.map(inputs) { builder, field in
            let position = builder.vector(field.x, field.y, field.z)
            let planar = builder.animatedVoronoi(field.xy, time: field.w)
            let spatial = builder.animatedVoronoi(position, time: field.w)
            return builder.vector(planar, spatial, builder.constant(Float(0)), builder.constant(Float(0)))
        }
        for i in inputs.indices {
            #expect(abs(fields[i].x - expectedFields[i].x) <= 4e-5, "\(inputs[i]): \(fields[i]) vs \(expectedFields[i])")
            #expect(GraphProbe.maximumDifference(fields[i], expectedFields[i]) <= 0.001,
                    "\(inputs[i]): \(fields[i]) vs \(expectedFields[i])")
            #expect(abs(distances[i].x - expectedDistances[i].x) <= 0.002, "\(inputs[i]): \(distances[i])")
            #expect(abs(distances[i].y - expectedDistances[i].y) <= 0.002, "\(inputs[i]): \(distances[i])")
        }
    }

    @Test("Dithering adds an interleaved-gradient offset of at most half a quantization step")
    func dithering() throws {
        let pixels: [SIMD2<Float>] = (0..<64).map { SIMD2(Float($0 % 8) + 0.5, Float($0 / 8) + 0.5) }
        let results = try probe.map(pixels.map { SIMD4($0.x, $0.y, 0.5, 0.25) }) { b, field in
            let pixel = field.xy
            let gray = b.dither(field.z, pixel: pixel, levels: 255)
            let color = b.dither(b.vector(field.z, field.w, field.z), pixel: pixel, levels: 15)
            return b.vector(gray, color.x, color.y, b.interleavedGradientNoise(pixel))
        }
        var distinct = Set<Float>()
        for (pixel, result) in zip(pixels, results) {
            let noise = ShaderNoise.interleavedGradient(pixel)
            distinct.insert(noise)
            let expected = SIMD4(0.5 + (noise - 0.5) / 255, 0.5 + (noise - 0.5) / 15, 0.25 + (noise - 0.5) / 15, noise)
            #expect(GraphProbe.maximumDifference(result, expected) <= 1e-6, "\(pixel): \(result) vs \(expected)")
            #expect(abs(result.x - 0.5) <= 0.5 / 255)
        }
        #expect(distinct.count == pixels.count, "an 8 × 8 block has no repeated noise values")
    }

    @Test("Matrix nodes build, multiply, transpose, invert and slice like simd, and reject scalar operations")
    func matrices() throws {
        let model = SceneTransform(
            translation: SIMD3(1, -2, 3), rotation: simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3(1, 2, 3))),
            scale: SIMD3(2, 0.5, 1.5)
        ).matrix
        let graph = try BufferShaderGraph(fieldCount: 4) { b in
            let m = b.matrix(b.elementField(0), b.elementField(1), b.elementField(2), b.elementField(3))
            let constant = b.constant(model)
            let v = b.vector(b.constant(SIMD3<Float>(0.5, -1, 2)), b.constant(Float(1)))
            let inverse = m.inverse()
            let product = (constant * inverse) * v
            let upper = m.upperLeft
            return [
                0: m * v,
                1: b.vector(m.transposed().column(2).xyz, m.determinant()),
                2: product,
                3: b.vector(upper.inverse().transposed() * b.constant(SIMD3<Float>(0, 0, 1)), upper.determinant()),
            ]
        }
        let columns = [model.columns.0, model.columns.1, model.columns.2, model.columns.3]
        let result = try probe.run(graph, elements: [columns])[0]
        let v = SIMD4<Float>(0.5, -1, 2, 1)
        let upper = simd_float3x3(SIMD3(model.columns.0.x, model.columns.0.y, model.columns.0.z),
                                  SIMD3(model.columns.1.x, model.columns.1.y, model.columns.1.z),
                                  SIMD3(model.columns.2.x, model.columns.2.y, model.columns.2.z))
        let normal = simd_transpose(simd_inverse(upper)) * SIMD3(0, 0, 1)
        let expected: [SIMD4<Float>] = [
            model * v,
            SIMD4(simd_transpose(model).columns.2.x, simd_transpose(model).columns.2.y, simd_transpose(model).columns.2.z,
                  simd_determinant(model)),
            v,
            SIMD4(normal, simd_determinant(upper)),
        ]
        for (actual, reference) in zip(result, expected) {
            #expect(GraphProbe.maximumDifference(actual, reference) <= 1e-5, "\(actual) vs \(reference)")
        }

        #expect(throws: MaterialGraph.Failure.matrixOperand(node: 2, input: 1)) {
            try BufferShaderGraph(nodes: [.input(.elementField(0)), .matrix([0, 0, 0, 0]), .add(1, 1), .swizzle(2, "x")],
                                  fieldCount: 1, outputs: [0: 0])
        }
        #expect(throws: MaterialGraph.Failure.invalidMatrix(3)) {
            try BufferShaderGraph(nodes: [.input(.elementField(0)), .matrix([0, 0, 0, 0]), .swizzle(0, "xyz"),
                                          .matrixMultiply(1, 2)], fieldCount: 1, outputs: [0: 0])
        }
        #expect(throws: MaterialGraph.Failure.invalidMatrix(2)) {
            try BufferShaderGraph(nodes: [.input(.elementField(0)), .matrix([0, 0, 0, 0]), .matrixColumn(1, 4)],
                                  fieldCount: 1, outputs: [0: 0])
        }
        #expect(throws: TextureShaderGraph.Failure.outputType(node: 1, actual: .matrix4)) {
            try TextureShaderGraph(nodes: [.input(.gridUV), .input(.viewMatrix)], output: 1)
        }
    }

    @Test("Matrix uniforms use one slot per column, leave scalar slots unchanged, and match simd on the GPU")
    func matrixUniforms() throws {
        let marker = "matrix-uniform-slots-9c2d"
        let model = simd_float4x4(columns: (
            SIMD4(1.5, 0.25, -0.5, 0), SIMD4(0, 0.5, 0.125, 0), SIMD4(0.25, 0, 2, 0), SIMD4(0.5, -1, 0.25, 1)
        ))
        let basis = simd_float3x3(SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 0, 0))
        let graph = try BufferShaderGraph(fieldCount: 2) { b in
            let transform = b.uniform("Model", simd_float4x4(1))
            let turned = b.uniform("Basis", simd_float3x3(1))
            let gain = b.uniform("Gain", Float(1))
            let point = b.elementField(0)
            let moved = (transform * point).xyz * gain
            return [1: b.vector(moved, (turned * point.xyz).y).named(marker)]
        }
        let source = try graph.metalFunction(named: "matrix_uniforms")
        #expect(source.contains(
            "float4x4(frostfield_graph_uniform(resources, 0u), frostfield_graph_uniform(resources, 1u), "
                + "frostfield_graph_uniform(resources, 2u), frostfield_graph_uniform(resources, 3u))"
        ), "\(source)")
        #expect(source.contains(
            "float3x3(frostfield_graph_uniform(resources, 4u).xyz, frostfield_graph_uniform(resources, 5u).xyz, "
                + "frostfield_graph_uniform(resources, 6u).xyz)"
        ), "\(source)")
        #expect(source.contains("frostfield_graph_uniform(resources, 7u).x"), "\(source)")

        let scalar = try BufferShaderGraph(fieldCount: 1) { b in
            [0: b.elementField(0) * b.uniform("Gain", Float(2)) + b.vector(b.time, b.constant(0), b.constant(0), b.constant(0))]
        }
        let scalarSource = try scalar.metalFunction(named: "scalar_uniform_layout")
        #expect(scalarSource.contains("frostfield_graph_uniform(resources, 0u).x"), "\(scalarSource)")
        #expect(scalarSource.contains("frostfield_graph_uniform(resources, 1u).x"), "\(scalarSource)")
        #expect(!scalarSource.contains("frostfield_graph_uniform(resources, 2u)"), "\(scalarSource)")

        let points: [SIMD4<Float>] = [
            SIMD4(0.5, -1, 2, 1), SIMD4(0, 0, 0, 1), SIMD4(-2, 0.25, 0.5, 1), SIMD4(1, 1, 1, 0),
        ]
        ShaderLibraryCache.countCompilations(containing: marker)
        let identity = try probe.run(graph, elements: points.map { [$0, .zero] }).map { $0[1] }
        for (point, value) in zip(points, identity) {
            let expected = SIMD4(SIMD3(point.x, point.y, point.z), point.y)
            #expect(GraphProbe.maximumDifference(value, expected) <= 1e-5, "identity \(point): \(value)")
        }
        let uniforms = try ShaderUniforms(graph.uniforms)
        try uniforms.set("Model", to: model)
        try uniforms.set("Basis", to: basis)
        try uniforms.set("Gain", to: 0.5)
        let transformed = try probe.run(graph, elements: points.map { [$0, .zero] }, uniforms: uniforms).map { $0[1] }
        for (point, value) in zip(points, transformed) {
            let moved = (model * point) * 0.5
            let turned = basis * SIMD3(point.x, point.y, point.z)
            let expected = SIMD4(SIMD3(moved.x, moved.y, moved.z), turned.y)
            #expect(GraphProbe.maximumDifference(value, expected) <= 1e-5, "uniform \(point): \(value) vs \(expected)")
        }
        #expect(ShaderLibraryCache.compilationCount(containing: marker) == 1)
    }

    @Test("Integers and bools bitcast, count bits, divide toward zero, and edit without a new pipeline")
    func integersAndBools() throws {
        let marker = "integer-bool-probe-5e1a"
        let graph = try BufferShaderGraph(fieldCount: 2) { b in
            let sample = b.elementField(0).x
            let roundTrip = sample.bitcast().bitcast()
            let ones = sample.countOneBits().toFloat()
            let quotient = (b.constant(Int32(-7)) / b.constant(Int32(2))).toFloat()
            let scaled = (b.uniform("K", Int32(4)) * b.constant(Int32(2)) + b.constant(Int32(3))).toFloat()
            let picked = b.branch(b.uniform("On", false)) { b.constant(Float(1)) } else: { b.constant(Float(0)) }
            let allOnes = b.constant(Int32(-1)).countOneBits().toFloat()
            return [
                0: b.vector(roundTrip, ones, quotient, scaled).named(marker),
                1: b.vector(picked, allOnes, sample.toInt().toFloat(), b.constant(Float(0))),
            ]
        }
        let source = try graph.metalFunction(named: "integer_probe")
        #expect(source.contains("as_type<int>(frostfield_graph_uniform(resources, 0u).x)"), "\(source)")
        #expect(source.contains("frostfield_graph_uniform(resources, 1u).x != 0.0f"), "\(source)")
        #expect(source.contains("int(popcount(as_type<uint>("), "\(source)")
        let floats = try BufferShaderGraph(fieldCount: 1) { b in
            [0: b.elementField(0) * b.uniform("Gain", Float(2))]
        }
        let floatSource = try floats.metalFunction(named: "float_uniform_bits")
        #expect(floatSource.contains("frostfield_graph_uniform(resources, 0u).x"), "\(floatSource)")
        #expect(!floatSource.contains("as_type<int>"), "\(floatSource)")

        let samples: [Float] = [1, -2.5, 0, 0.1, 16777216]
        ShaderLibraryCache.countCompilations(containing: marker)
        let uniforms = try ShaderUniforms(graph.uniforms)
        #expect(uniforms.integer(named: "K") == 4)
        #expect(uniforms.boolean(named: "On") == false)
        let first = try probe.run(graph, elements: samples.map { [SIMD4($0, 0, 0, 0), .zero] }, uniforms: uniforms)
        for (sample, fields) in zip(samples, first) {
            #expect(fields[0].x.bitPattern == sample.bitPattern, "round trip \(sample)")
            #expect(fields[0].y == Float(sample.bitPattern.nonzeroBitCount), "bits \(sample): \(fields[0].y)")
            #expect(fields[0].z == -3, "division \(fields[0].z)")
            #expect(fields[0].w == 11, "scaled \(fields[0].w)")
            #expect(fields[1].x == 0, "bool \(fields[1].x)")
            #expect(fields[1].y == 32, "all ones \(fields[1].y)")
            let truncated = sample.isFinite ? Float(Int(sample)) : 0
            #expect(fields[1].z == truncated, "truncation \(sample): \(fields[1].z)")
        }
        try uniforms.setInteger("K", Int32(5))
        try uniforms.setBoolean("On", true)
        let edited = try probe.run(graph, elements: [[SIMD4<Float>(1, 0, 0, 0), .zero]], uniforms: uniforms)[0]
        #expect(edited[0].w == 13)
        #expect(edited[1].x == 1)
        #expect(uniforms.integer(named: "K") == 5)
        try uniforms.setInteger("K", Int32(-2))
        let negative = try probe.run(graph, elements: [[SIMD4<Float>(1, 0, 0, 0), .zero]], uniforms: uniforms)[0]
        #expect(negative[0].w == -1, "negative uniform \(negative[0].w)")
        #expect(uniforms.integer(named: "K") == -2)
        #expect(ShaderLibraryCache.compilationCount(containing: marker) == 1)
        let encoded = try JSONEncoder().encode(ShaderUniform("K", Int32(-2)))
        #expect(try JSONDecoder().decode(ShaderUniform.self, from: encoded) == ShaderUniform("K", Int32(-2)))
        #expect(throws: MaterialGraph.Failure.typeMismatch(node: 2, input: 1, expected: .scalar, actual: .int)) {
            try BufferShaderGraph(
                nodes: [.input(.elementField(0)), .integer(1), .unary(.sin, 1)], fieldCount: 1, outputs: [0: 0]
            )
        }
        #expect(throws: MaterialGraph.Failure.typeMismatch(node: 3, input: 2, expected: .int, actual: .scalar)) {
            try BufferShaderGraph(
                nodes: [.input(.elementField(0)), .integer(1), .scalar(2), .multiply(1, 2)], fieldCount: 1, outputs: [0: 0]
            )
        }
        #expect(throws: MaterialGraph.Failure.typeMismatch(node: 2, input: 1, expected: .scalar, actual: .bool)) {
            try BufferShaderGraph(
                nodes: [.input(.elementField(0)), .boolean(true), .add(1, 1)], fieldCount: 1, outputs: [0: 0]
            )
        }
    }

    @Test("Derivatives measure per-pixel change in post-processing and are rejected outside fragment stages")
    func derivatives() throws {
        let fixture = try PostProcessFixture()
        let graph = try PostProcessGraph { p in
            let uv = p.screenUV
            let size = p.viewportSize
            return p.vector(
                uv.x.dFdx() * size.x, uv.y.dFdy() * size.y, (uv.x + uv.y * 2).fwidth() * size.x, size.y / 16
            )
        }
        let pass = try PostProcessPass(graph: graph, device: fixture.device, pixelFormat: .rgba16Float)
        let source = try fixture.color(width: 16, height: 16) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try fixture.color(width: 16, height: 16)
        try fixture.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        let pixels = try fixture.read(destination)
        for (index, pixel) in pixels.enumerated() {
            #expect(PostProcessFixture.close(pixel, SIMD4(1, 1, 3, 1)), "pixel \(index): \(pixel)")
        }

        #expect(throws: MaterialGraph.Failure.unavailable(node: 1, stage: .compute)) {
            try TextureShaderGraph(nodes: [.input(.gridUV), .derivative(.dx, 0), .swizzle(1, "x")], output: 2)
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 1, stage: .compute)) {
            try BufferShaderGraph(nodes: [.input(.elementField(0)), .derivative(.width, 0)], fieldCount: 1, outputs: [0: 1])
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 1, stage: .vertex)) {
            try SurfaceShaderGraph(nodes: [.input(.objectPosition), .derivative(.dy, 0)], outputs: .init(positionOffset: 1))
        }
    }

    @Test("Structs nest, branch, and round-trip their fields on the GPU")
    func structs() throws {
        let marker = "struct-fields-7c4e"
        let graph = try BufferShaderGraph(fieldCount: 1) { b in
            let sample = b.elementField(0)
            let point = b.defineStruct("Point", [
                ShaderStruct.Field("x", .scalar), ShaderStruct.Field("y", .scalar),
            ])
            let here = b.make(point, ["x": sample.x, "y": sample.y])
            let wrapped = b.defineStruct("Sample", [
                ShaderStruct.Field("point", structure: point.index),
                ShaderStruct.Field("scale", .scalar),
            ])
            let packed = b.make(wrapped, ["point": here, "scale": b.constant(Float(2))])
            let other = b.make(wrapped, [
                "point": b.make(point, ["x": b.constant(Float(0)), "y": b.constant(Float(1))]),
                "scale": b.constant(Float(3)),
            ])
            let chosen = b.branch(sample.x) { packed } else: { other }
            let scaled = packed.field("point", as: point).field("x", as: Float.self) * packed.field("scale", as: Float.self)
            return [0: b.vector(
                scaled,
                chosen.field("point", as: point).field("y", as: Float.self),
                chosen.field("scale", as: Float.self),
                b.constant(Float(0))
            ).named(marker)]
        }
        let source = try graph.metalFunction(named: "struct_probe")
        #expect(source.contains("struct frostfield_struct_0_Point"))
        #expect(source.contains("struct frostfield_struct_1_Sample"))
        #expect(source.contains("frostfield_struct_0_Point point;"))
        #expect(!source.contains("void n"))
        let samples = [SIMD2<Float>(1, 4), SIMD2(0, 9), SIMD2(0.2, -3)]
        let result = try probe.run(graph, elements: samples.map { [SIMD4($0.x, $0.y, 0, 0)] })
        for (sample, fields) in zip(samples, result) {
            let first = sample.x > 0.5
            #expect(fields[0].x == sample.x * 2, "scaled \(sample)")
            #expect(fields[0].y == (first ? sample.y : 1), "nested \(sample): \(fields[0].y)")
            #expect(fields[0].z == (first ? 2 : 3), "arm \(sample): \(fields[0].z)")
        }
        #expect(try JSONDecoder().decode(BufferShaderGraph.self, from: JSONEncoder().encode(graph)) == graph)

        #expect(throws: MaterialGraph.Failure.invalidDeclaration("Struct Point is missing field y.")) {
            try BufferShaderGraph(fieldCount: 1) { b in
                let point = b.defineStruct("Point", [
                    ShaderStruct.Field("x", .scalar), ShaderStruct.Field("y", .scalar),
                ])
                _ = b.make(point, ["x": b.constant(Float(1))])
                return [0: b.elementField(0)]
            }
        }
        #expect(throws: MaterialGraph.Failure.invalidDeclaration("Field x is scalar, not vector3.")) {
            try BufferShaderGraph(fieldCount: 1) { b in
                let point = b.defineStruct("Point", [ShaderStruct.Field("x", .scalar)])
                let value = b.make(point, ["x": b.constant(Float(1))])
                return [0: b.vector(value.field("x", as: SIMD3<Float>.self), b.constant(Float(0)))]
            }
        }
        #expect(throws: MaterialGraph.Failure.invalidDeclaration("The arms of a struct branch must be the same struct.")) {
            try BufferShaderGraph(fieldCount: 1) { b in
                let point = b.defineStruct("Point", [ShaderStruct.Field("x", .scalar)])
                let wrapped = b.defineStruct("Sample", [ShaderStruct.Field("point", structure: point.index)])
                let here = b.make(point, ["x": b.constant(Float(1))])
                let there = b.make(wrapped, ["point": here])
                _ = b.branch(b.constant(Float(1))) { here } else: { there }
                return [0: b.elementField(0)]
            }
        }
        let negativeDefault = try BufferShaderGraph(fieldCount: 1) { b in
            [0: b.vector(b.uniform("K", Int32(-2)).toFloat(), b.constant(Float(0)), b.constant(Float(0)), b.constant(Float(0)))]
        }
        #expect(try ShaderUniforms(negativeDefault.uniforms).integer(named: "K") == -2)
    }

    @Test("An explicit texture level selects that mip and clamps past the last one")
    func textureLevel() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: 4, height: 4, mipmapped: true
        )
        descriptor.mipmapLevelCount = 3
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        func fill(_ level: Int, _ color: SIMD4<Float>) {
            let size = 4 >> level
            let texels = Array(repeating: color, count: size * size)
            texels.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: level,
                    withBytes: bytes.baseAddress!, bytesPerRow: size * 16
                )
            }
        }
        fill(0, SIMD4(1, 0, 0, 1))
        fill(1, SIMD4(0, 1, 0, 1))
        fill(2, SIMD4(0, 0, 1, 1))
        let graph = try BufferShaderGraph(fieldCount: 1) { b in
            let map = b.texture("Map", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let sample = map.sample(b.constant(SIMD2<Float>(0.5, 0.5)), level: b.uniform("Lod", Float(1)))
            return [0: sample]
        }
        let uniforms = try ShaderUniforms(graph.uniforms)
        func read(_ lod: Float) throws -> SIMD4<Float> {
            try uniforms.set("Lod", to: lod)
            return try probe.run(
                graph, elements: [[SIMD4<Float>(0, 0, 0, 0)]], uniforms: uniforms, textures: ["Map": texture]
            )[0][0]
        }
        let base = try read(0)
        let middle = try read(1)
        let last = try read(8)
        #expect(base == SIMD4(1, 0, 0, 1))
        #expect(middle == SIMD4(0, 1, 0, 1))
        #expect(last == SIMD4(0, 0, 1, 1))
        let encoded = try JSONEncoder().encode(graph)
        #expect(try JSONDecoder().decode(BufferShaderGraph.self, from: encoded) == graph)
    }

    @Test("Explicit UV derivatives select the mip from the larger texel footprint")
    func textureGradient() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: 4, height: 4, mipmapped: true
        )
        descriptor.mipmapLevelCount = 3
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        func fill(_ level: Int, _ color: SIMD4<Float>) {
            let size = 4 >> level
            let texels = Array(repeating: color, count: size * size)
            texels.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: level,
                    withBytes: bytes.baseAddress!, bytesPerRow: size * 16
                )
            }
        }
        fill(0, SIMD4(1, 0, 0, 1))
        fill(1, SIMD4(0, 1, 0, 1))
        fill(2, SIMD4(0, 0, 1, 1))
        let graph = try BufferShaderGraph(fieldCount: 1) { b in
            let map = b.texture("Map", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let scale = b.uniform("Scale", Float(1))
            let step = scale * Float(0.25)
            let sample = map.sample(
                b.constant(SIMD2<Float>(0.5, 0.5)),
                ddx: b.vector(step, b.constant(Float(0))),
                ddy: b.vector(b.constant(Float(0)), step)
            )
            return [0: sample]
        }
        let uniforms = try ShaderUniforms(graph.uniforms)
        func read(_ scale: Float) throws -> SIMD4<Float> {
            try uniforms.set("Scale", to: scale)
            return try probe.run(
                graph, elements: [[SIMD4<Float>(0, 0, 0, 0)]], uniforms: uniforms, textures: ["Map": texture]
            )[0][0]
        }
        #expect(try read(0) == SIMD4(1, 0, 0, 1))
        #expect(try read(1) == SIMD4(1, 0, 0, 1))
        #expect(try read(2) == SIMD4(0, 1, 0, 1))
        #expect(try read(4) == SIMD4(0, 0, 1, 1))
        #expect(try read(8) == SIMD4(0, 0, 1, 1))
        let encoded = try JSONEncoder().encode(graph)
        #expect(try JSONDecoder().decode(BufferShaderGraph.self, from: encoded) == graph)

        let overflow = try BufferShaderGraph(fieldCount: 1) { b in
            let map = b.texture("Map", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let huge = b.constant(Float(1e20))
            let sample = map.sample(
                b.constant(SIMD2<Float>(0.5, 0.5)),
                ddx: b.vector(huge * huge, b.constant(Float(0))),
                ddy: b.constant(SIMD2<Float>(0, 0.25))
            )
            return [0: sample]
        }
        let broken = try probe.run(overflow, elements: [[SIMD4<Float>(0, 0, 0, 0)]], textures: ["Map": texture])[0][0]
        #expect(broken == SIMD4(1, 0, 0, 1))

        let branched = try BufferShaderGraph(fieldCount: 1) { b in
            let map = b.texture("Map", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let sample = map.sample(
                b.constant(SIMD2<Float>(0.5, 0.5)),
                ddx: b.constant(SIMD2<Float>(0.5, 0)),
                ddy: b.constant(SIMD2<Float>(0, 0.5))
            )
            return [0: b.branch(b.constant(Float(1))) { sample } else: { b.constant(SIMD4<Float>(0, 0, 0, 0)) }]
        }
        let source = try branched.metalFunction(named: "gradient_arm")
        let sampleAt = try #require(source.range(of: "frostfield_graph_sample"))
        let branchAt = try #require(source.range(of: "if ("))
        let elseAt = try #require(source.range(of: "} else {"))
        #expect(branchAt.lowerBound < sampleAt.lowerBound && sampleAt.lowerBound < elseAt.lowerBound)
        let chosen = try probe.run(branched, elements: [[SIMD4<Float>(0, 0, 0, 0)]], textures: ["Map": texture])[0][0]
        #expect(chosen == SIMD4(0, 1, 0, 1))
    }
}
