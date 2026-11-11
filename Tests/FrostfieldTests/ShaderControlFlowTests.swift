import Foundation
import Frostfield
import Metal
import Testing

struct ShaderControlFlowTests {
    @Test("Branches run only the chosen arm, chain as else-if, and match a CPU reference on the GPU",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func branches() throws {
        let graph = try BufferShaderGraph(fieldCount: 2) { b in
            let x = b.elementField(0).x
            let noisy = b.branch(x.greaterThan(b.constant(0.5))) {
                b.fractalNoise(b.vector(x, x * 2, x * 3) * 4, kind: .value, octaves: 3).named("only_true")
            } else: {
                x * 2
            }
            let piecewise = b.branch([
                (condition: x.lessThan(b.constant(0.25)), value: x + 10),
                (condition: x.lessThan(b.constant(0.5)), value: x + 20),
            ], otherwise: x + 30)
            return [1: b.vector(noisy, piecewise, x, x)]
        }
        let source = try graph.metalFunction(named: "branch_probe")
        let lines = source.components(separatedBy: "\n")
        // The fractal noise only the true arm reads is generated inside the first if block.
        let ifLine = try #require(lines.firstIndex { $0.contains("> 0.5f) {") })
        let elseLine = try #require(lines.firstIndex { $0.contains("} else {") })
        let noiseLine = try #require(lines.firstIndex { $0.contains("only_true") })
        #expect(ifLine < noiseLine && noiseLine < elseLine, "\(source)")
        #expect(lines[noiseLine].hasPrefix("        float n"), "\(lines[noiseLine])")
        #expect(source.contains("branch"))

        let inputs: [Float] = [-1, 0, 0.1, 0.249, 0.25, 0.3, 0.4999, 0.5, 0.5001, 0.75, 1.3, 2.7]
        let output = try GraphProbe().run(graph, elements: inputs.map { [SIMD4($0, 0, 0, 0), .zero] }).map { $0[1] }
        for (x, value) in zip(inputs, output) {
            let noisy = x > 0.5 ? ShaderNoise.fractal(SIMD3(x, x * 2, x * 3) * 4, kind: .value, octaves: 3) : x * 2
            let piecewise = x < 0.25 ? x + 10 : x < 0.5 ? x + 20 : x + 30
            #expect(abs(value.x - noisy) <= 1e-6, "branch at \(x): \(value.x) vs \(noisy)")
            #expect(value.y == piecewise, "else-if at \(x): \(value.y) vs \(piecewise)")
        }
    }

    @Test("Loops accumulate from their index and value, hoist invariant work, nest, and stop at their exit",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func loops() throws {
        let graph = try BufferShaderGraph(fieldCount: 2) { b in
            let x = b.elementField(0).x
            let scaled = (x * 3).named("invariant")
            // Σ_{i<8} (i + 1) · 3x = 108 x.
            let sum = b.loop(count: 8, initial: b.constant(Float(0))) { i, total in total + (i + 1) * scaled }
            // Doubles from 1 until the value exceeds |x| + 1.
            let power = b.loop(count: 32, initial: b.constant(Float(1))) { _, value in
                let next = value * 2
                return ShaderLoopStep(next: next, exit: next.greaterThan(x.abs() + 1))
            }
            // Σ_{i<3} Σ_{j<4} (4i + j) = 66, starting from x.
            let nested = b.loop(count: 3, initial: x) { i, outer in
                b.loop(count: 4, initial: outer) { j, inner in inner + i * 4 + j }
            }
            // Σ_{i<4} (i, i², x) = (6, 14, 4x).
            let vector = b.loop(count: 4, initial: b.constant(SIMD3<Float>(0, 0, 0))) { i, v in v + b.vector(i, i * i, x) }
            return [1: b.vector(sum, power, nested, vector.y + vector.z)]
        }
        let source = try graph.metalFunction(named: "loop_probe")
        let invariant = try #require(source.range(of: "invariant"))
        let firstLoop = try #require(source.range(of: "for (int n"))
        #expect(invariant.upperBound < firstLoop.lowerBound, "loop-invariant work precedes the loop: \(source)")
        #expect(source.contains("loop 8") && source.contains("loop 32") && source.contains("{ break; }"))

        let inputs: [Float] = [-3, -0.5, 0, 0.25, 1, 2.5, 7, 100]
        let output = try GraphProbe().run(graph, elements: inputs.map { [SIMD4($0, 0, 0, 0), .zero] }).map { $0[1] }
        for (x, value) in zip(inputs, output) {
            var power: Float = 1
            for _ in 0..<32 {
                power *= 2
                if power > abs(x) + 1 { break }
            }
            var sum: Float = 0
            for i in 0..<8 { sum += Float(i + 1) * (x * 3) }
            #expect(value.x == sum, "sum at \(x): \(value.x) vs \(sum)")
            #expect(value.y == power, "exit at \(x): \(value.y) vs \(power)")
            #expect(value.z == x + 66, "nested at \(x): \(value.z)")
            #expect(value.w == 14 + 4 * x, "vector at \(x): \(value.w)")
        }
    }

    @Test("Control flow rejects escaping loop values, bad counts, non-loops, deep nesting and divergent samples")
    func validation() throws {
        // A loop index read by the output outside its loop.
        #expect(throws: MaterialGraph.Failure.loopValueEscapes(node: 5, loop: 2)) {
            try SurfaceShaderGraph(nodes: [
                .scalar(0), .loopIndex(loop: 2), .loop(count: 4, initial: 0, next: 3, exit: nil),
                .add(4, 1), .loopValue(loop: 2), .add(1, 0),
            ], outputs: .init(roughness: 5))
        }
        for count in [0, MaterialGraph.maximumLoopIterations + 1] {
            #expect(throws: MaterialGraph.Failure.invalidLoopCount(node: 1, count: count)) {
                try SurfaceShaderGraph(nodes: [.scalar(0), .loop(count: count, initial: 0, next: 0, exit: nil)],
                                       outputs: .init(roughness: 1))
            }
        }
        #expect(throws: MaterialGraph.Failure.notALoop(node: 1, loop: 0)) {
            try SurfaceShaderGraph(nodes: [.scalar(1), .loopIndex(loop: 0)], outputs: .init(roughness: 1))
        }
        // The initial value reads the loop's own iteration.
        #expect(throws: MaterialGraph.Failure.loopValueEscapes(node: 0, loop: 1)) {
            try SurfaceShaderGraph(nodes: [
                .loopIndex(loop: 1), .loop(count: 2, initial: 0, next: 2, exit: nil), .loopValue(loop: 1),
            ], outputs: .init(roughness: 1))
        }
        // Mixing the iterations of two sibling loops.
        let mixing = #expect(throws: MaterialGraph.Failure.self) {
            try SurfaceShaderGraph(nodes: [
                .scalar(0), .loop(count: 2, initial: 0, next: 3, exit: nil), .loop(count: 2, initial: 0, next: 4, exit: nil),
                .add(5, 6), .loopValue(loop: 2), .loopIndex(loop: 1), .loopIndex(loop: 2), .add(1, 2),
            ], outputs: .init(roughness: 7))
        }
        if case .loopValueEscapes = mixing {} else { Issue.record("Unexpected failure \(String(describing: mixing))") }

        // Sixteen nested branches compile; seventeen exceed the nesting limit.
        func nested(_ depth: Int) throws -> SurfaceShaderGraph {
            try SurfaceShaderGraph { s in
                func nest(_ level: Int) -> ShaderValue<Float> {
                    guard level > 0 else { return s.roughness * 0.5 }
                    return s.branch(s.metallic.greaterThan(s.constant(Float(level) / 20))) {
                        nest(level - 1)
                    } else: {
                        s.constant(Float(level))
                    }
                }
                return .init(roughness: nest(depth))
            }
        }
        let deepest = try nested(MaterialGraph.maximumNestingDepth).metalFunction(named: "nested")
        #expect(deepest.contains(String(repeating: "    ", count: MaterialGraph.maximumNestingDepth + 1) + "float n"))
        let tooDeep = #expect(throws: MaterialGraph.Failure.self) { try nested(MaterialGraph.maximumNestingDepth + 1) }
        if case .nestingTooDeep = tooDeep {} else { Issue.record("Unexpected failure \(String(describing: tooDeep))") }

        // A texture sample inside a loop that can exit early has undefined derivatives.
        func sampling(exits: Bool) throws -> SurfaceShaderGraph {
            try SurfaceShaderGraph { s in
                let height = s.texture("Height", content: .data)
                let total: ShaderValue<Float>
                if exits {
                    total = s.loop(count: 8, initial: s.constant(Float(0))) { i, value in
                        let h = height.sample(s.uv0 + s.vector(i, i) * 0.01).x
                        return ShaderLoopStep(next: value + h, exit: h.greaterThan(s.constant(0.5)))
                    }
                } else {
                    total = s.loop(count: 8, initial: s.constant(Float(0))) { i, value in
                        value + height.sample(s.uv0 + s.vector(i, i) * 0.01).x
                    }
                }
                return .init(roughness: total / 8)
            }
        }
        let divergent = #expect(throws: MaterialGraph.Failure.self) { try sampling(exits: true) }
        if case .divergentDerivative = divergent {} else { Issue.record("Unexpected failure \(String(describing: divergent))") }
        let uniform = try sampling(exits: false).metalFunction(named: "fixed_loop")
        let loopStart = try #require(uniform.range(of: "for (int n"))
        let sample = try #require(uniform.range(of: "frostfield_graph_sample("))
        #expect(loopStart.upperBound < sample.lowerBound, "a fixed-count loop keeps its samples: \(uniform)")

        // A sample that only one arm reads is still taken outside the branch.
        let hoisted = try SurfaceShaderGraph { s in
            let albedo = s.texture("Albedo")
            let color = s.branch(s.uv0.x.greaterThan(s.constant(0.5))) {
                albedo.sample(s.uv0).rgb
            } else: {
                s.baseColor
            }
            return .init(baseColor: color)
        }.metalFunction(named: "hoisted")
        let branchStart = try #require(hoisted.range(of: "if (n"))
        let sampled = try #require(hoisted.range(of: "frostfield_graph_sample("))
        #expect(sampled.upperBound < branchStart.lowerBound, "\(hoisted)")
    }

    @Test("Declared functions that take derivatives stay outside branch arms; others run only in their arm")
    func functionPlacement() throws {
        let source = try SurfaceShaderGraph { s in
            let edge = s.function("edge", returns: Float.self, parameters: [.init("x", .scalar)], body: "return fwidth(x);")
            let twice = s.function("twice", returns: Float.self, parameters: [.init("x", .scalar)], body: "return x * 2.0f;")
            // A function that calls the derivative-taking one by its generated name.
            let indirect = s.function(
                "indirect", returns: Float.self, parameters: [.init("x", .scalar)], body: "return placement_fn_edge(x) * 0.5f;"
            )
            let condition = s.uv0.x.greaterThan(s.constant(0.5))
            let roughness = s.branch(condition) { edge(s.uv0.y).named("edge_call") } else: { s.roughness }
            let metallic = s.branch(condition) { twice(s.uv0.y).named("twice_call") } else: { s.metallic }
            let occlusion = s.branch(condition) { indirect(s.uv0.x).named("indirect_call") } else: { s.constant(Float(1)) }
            return .init(metallic: metallic, roughness: roughness, occlusion: occlusion)
        }.metalFunction(named: "placement")
        // Statements in the function body are indented four spaces; those in an arm, eight.
        let lines = source.components(separatedBy: "\n")
        func line(_ label: String) throws -> String { try #require(lines.first { $0.contains(label) }) }
        for label in ["edge_call", "indirect_call"] {
            let statement = try line(label)
            #expect(statement.hasPrefix("    float n") && !statement.hasPrefix("     "),
                    "a derivative inside a function runs in uniform control flow: \(statement)")
        }
        #expect(try line("twice_call").hasPrefix("        float n"), "a plain function call runs only in its arm: \(source)")
    }

    @Test("One evaluation runs at most maximumTotalIterations loop iterations and parallax layers")
    func iterationBudget() throws {
        // Nested counts multiply: 256 outer iterations plus 256 × 255 inner ones fit exactly.
        func nested(_ outer: Int, _ inner: Int) throws -> SurfaceShaderGraph {
            try SurfaceShaderGraph { s in
                .init(roughness: s.loop(count: outer, initial: s.constant(Float(0))) { i, total in
                    s.loop(count: inner, initial: total) { j, sum in sum + i * j * 1e-9 }
                })
            }
        }
        #expect(MaterialGraph.maximumTotalIterations == 256 + 256 * 255)
        _ = try nested(256, 255)
        let over = #expect(throws: MaterialGraph.Failure.self) { try nested(256, 256) }
        guard case .tooManyIterations(_, let iterations) = over else {
            Issue.record("Unexpected failure \(String(describing: over))")
            return
        }
        #expect(iterations == 256 + 256 * 256)
        #expect(over?.errorDescription?.contains("65536") == true, "\(String(describing: over?.errorDescription))")
        // Two maximal loops side by side already exceed the budget at the second.
        #expect(throws: MaterialGraph.Failure.self) {
            try BufferShaderGraph(fieldCount: 1) { b in
                let x = b.elementField(0).x
                let first = b.loop(count: 1_024, initial: x) { i, v in v + i }
                let rows = (0..<64).map { k in b.loop(count: 1_024, initial: first) { i, v in v + i * Float(k) } }
                return [0: b.vector(rows.reduce(first, +), x, x, x)]
            }
        }
        // Parallax layers inside a loop count once per iteration: 508 × (127 + 1) + 508 = 65,532 fits; 509 does not.
        func marching(_ count: Int) throws -> SurfaceShaderGraph {
            try SurfaceShaderGraph { s in
                let height = s.texture("Height", content: .data)
                let view = s.tangentSpace(s.viewDirection)
                return .init(roughness: s.loop(count: count, initial: s.constant(Float(0))) { i, total in
                    total + height.parallaxOcclusion(
                        s.uv0 + s.vector(i, i) * 1e-4, view: view, depth: s.constant(Float(0.05)), steps: 127
                    ).x
                })
            }
        }
        _ = try marching(508)
        let parallax = #expect(throws: MaterialGraph.Failure.self) { try marching(509) }
        if case .tooManyIterations = parallax {} else { Issue.record("Unexpected failure \(String(describing: parallax))") }
    }
}
