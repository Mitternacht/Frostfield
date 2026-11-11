import Foundation
import Frostfield
import Metal
import simd
import Testing

struct ShaderGraphLanguageTests {
    @Test("The Swift builder records typed nodes, shared literals, declarations and labels")
    func builderGraph() throws {
        let graph = try SurfaceShaderGraph { s in
            let tint = s.uniform("Tint", SIMD3<Float>(0.8, 0.4, 0.2))
            let scale = s.uniform("Scale", Float(4))
            let albedo = s.texture("Albedo")
            let uv = s.uv0 * scale
            let grain = s.fractalNoise(s.worldPosition * 3, octaves: 4).named("grain")
            let color = albedo.sample(uv).rgb * tint
            return .init(
                baseColor: color.mix(s.baseColor, by: grain),
                roughness: grain.remap(0, 1, 0.25, 0.75),
                emission: s.vector(grain, grain, grain) * 0.5
            )
        }
        #expect(graph.uniforms.map(\.name) == ["Tint", "Scale"])
        #expect(graph.uniforms[0].defaultValue == SIMD4(0.8, 0.4, 0.2, 0))
        #expect(graph.uniforms[1].type == .scalar)
        #expect(graph.textures.map(\.name) == ["Albedo"])
        #expect(graph.hasSurfaceOutputs && !graph.hasVertexOutputs)

        let source = try graph.metalFunction(named: "builder_surface")
        #expect(source.contains("frostfield_graph_uniform(resources, 0u).xyz"))
        #expect(source.contains("frostfield_graph_uniform(resources, 1u)"))
        #expect(source.contains("frostfield_graph_sample(resources, 0u, 0u, 0u,"))
        #expect(source.contains("frostfield_graph_fbm("))
        #expect(source.contains("frostfield_graph_remap("))
        #expect(source.contains("grain"))
        #expect(try graph.metalVertexFunction(named: "builder_vertex") == nil)
        // The literal 0.5 and the world position input are each emitted once.
        #expect(graph.nodes.filter { $0 == .scalar(0.5) }.count == 1)
        #expect(graph.nodes.filter { $0 == .input(.worldPosition) }.count == 1)
    }

    @Test("Vertex offsets compile to a separate object-space function")
    func vertexOffset() throws {
        let graph = try SurfaceShaderGraph { s in
            let wave = (s.objectPosition.x * 6 + s.uniform("Time", Float(0))).sin() * 0.1
            return .init(positionOffset: s.displacement(wave))
        }
        #expect(!graph.hasSurfaceOutputs && graph.hasVertexOutputs)
        let vertex = try #require(try graph.metalVertexFunction(named: "wave_offset"))
        #expect(vertex.contains("static inline float3 wave_offset("))
        #expect(vertex.contains("thread const FrostfieldVertexInput& input"))
        #expect(vertex.contains("input.objectPosition"))
        #expect(vertex.contains("float3(-1e6f), float3(1e6f)"))
    }

    @Test("Stages reject inputs and resources they cannot provide")
    func stageAvailability() {
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .vertex)) {
            try SurfaceShaderGraph(nodes: [.input(.viewDirection)], outputs: .init(positionOffset: 0))
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .surface)) {
            try SurfaceShaderGraph(nodes: [.input(.objectPosition)], outputs: .init(baseColor: 0))
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 1, stage: .vertex)) {
            try SurfaceShaderGraph(nodes: [.input(.uv0), .sampleScene(coordinates: 0), .swizzle(1, "rgb")],
                                   outputs: .init(positionOffset: 2))
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 1, stage: .surface)) {
            try MaterialGraph(nodes: [.input(.uv0), .sampleScene(coordinates: 0), .swizzle(1, "rgb")], output: 2)
        }
        // An unlit material graph only receives UV0, the normal, world position, and parameters.
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .surface)) {
            try MaterialGraph(nodes: [.input(.viewDirection)], output: 0)
        }
    }

    @Test("Declarations, swizzles, combinations and calls are checked")
    func declarationValidation() {
        #expect(throws: MaterialGraph.Failure.invalidUniform(node: 0, index: 0)) {
            try SurfaceShaderGraph(nodes: [.uniform(0)], outputs: .init(roughness: 0))
        }
        #expect(throws: MaterialGraph.Failure.invalidTexture(node: 1, slot: 2)) {
            try SurfaceShaderGraph(nodes: [.input(.uv0), .sampleTexture(slot: 2, coordinates: 0), .swizzle(1, "x")],
                                   outputs: .init(roughness: 2))
        }
        #expect(throws: MaterialGraph.Failure.invalidSwizzle(node: 1, components: "xyz")) {
            try SurfaceShaderGraph(nodes: [.input(.uv0), .swizzle(0, "xyz")], outputs: .init(baseColor: 1))
        }
        #expect(throws: MaterialGraph.Failure.invalidSwizzle(node: 1, components: "xg")) {
            try SurfaceShaderGraph(nodes: [.color(.one), .swizzle(0, "xg")], outputs: .init(baseColor: 0))
        }
        #expect(throws: MaterialGraph.Failure.invalidCombine(node: 2, components: 5)) {
            try SurfaceShaderGraph(nodes: [.color(.one), .uv(.u), .combine([0, 1, 1])], outputs: .init(baseColor: 0))
        }
        #expect(throws: MaterialGraph.Failure.invalidNoise(1)) {
            try SurfaceShaderGraph(nodes: [.color(.one), .fractalNoise(.value, position: 0, octaves: 9, lacunarity: 2, gain: 0.5)],
                                   outputs: .init(roughness: 1))
        }
        #expect(throws: MaterialGraph.Failure.typeMismatch(node: 1, input: 0, expected: .vector3, actual: .vector2)) {
            try SurfaceShaderGraph(nodes: [.input(.uv0), .binary(.cross, 0, 0)], outputs: .init(baseColor: 1))
        }
        let square = ShaderFunction(name: "square", parameters: [.init("x", .scalar)], returns: .scalar, body: "return x * x;")
        #expect(throws: MaterialGraph.Failure.argumentCount(node: 1, expected: 1, actual: 2)) {
            try SurfaceShaderGraph(nodes: [.scalar(2), .function(0, arguments: [0, 0])], functions: [square],
                                   outputs: .init(roughness: 1))
        }
        #expect(throws: MaterialGraph.Failure.invalidFunction(node: 1, index: 1)) {
            try SurfaceShaderGraph(nodes: [.scalar(2), .function(1, arguments: [0])], functions: [square],
                                   outputs: .init(roughness: 1))
        }
        let duplicate = [ShaderUniform("A", Float(1)), ShaderUniform("A", Float(2))]
        #expect(throws: MaterialGraph.Failure.self) {
            try SurfaceShaderGraph(nodes: [.scalar(1)], uniforms: duplicate, outputs: .init(roughness: 0))
        }
        for body in ["return x; }", "#include <x>\nreturn x;", ""] {
            let function = ShaderFunction(name: "bad", parameters: [.init("x", .scalar)], returns: .scalar, body: body)
            #expect(throws: MaterialGraph.Failure.self) {
                try SurfaceShaderGraph(nodes: [.scalar(1)], functions: [function], outputs: .init(roughness: 0))
            }
        }
        let keyword = ShaderFunction(name: "float3", parameters: [], returns: .scalar, body: "return 1.0f;")
        #expect(throws: MaterialGraph.Failure.self) {
            try SurfaceShaderGraph(nodes: [.scalar(1)], functions: [keyword], outputs: .init(roughness: 0))
        }
    }

    @Test("Live uniforms type-check edits and advance their revision only on change")
    func uniforms() throws {
        let uniforms = try ShaderUniforms([ShaderUniform("Gain", Float(1)), ShaderUniform("Tint", SIMD3<Float>(1, 0, 0))])
        #expect(uniforms.revision == 0)
        try uniforms.set("Gain", to: 1)
        #expect(uniforms.revision == 0)
        try uniforms.set("Gain", to: 2.5)
        #expect(uniforms.revision == 1)
        #expect(uniforms.value(named: "Gain") == SIMD4(2.5, 0, 0, 0))
        #expect(throws: ShaderUniforms.Failure.type(name: "Tint", expected: .vector3, actual: .scalar)) {
            try uniforms.set("Tint", to: 1)
        }
        #expect(throws: ShaderUniforms.Failure.unknownName("Missing")) { try uniforms.set("Missing", to: 1) }
        #expect(throws: ShaderUniforms.Failure.nonfinite("Gain")) { try uniforms.set("Gain", to: .nan) }
        #expect(uniforms.revision == 1)
        let snapshot = uniforms.snapshot()
        #expect(snapshot.values == [SIMD4(2.5, 0, 0, 0), SIMD4(1, 0, 0, 0)])
        uniforms.reset()
        #expect(uniforms.value(named: "Gain") == SIMD4(1, 0, 0, 0))
        #expect(uniforms.revision == 2)
        #expect(throws: ShaderUniforms.Failure.duplicateName("A")) {
            try ShaderUniforms([ShaderUniform("A", Float(0)), ShaderUniform("A", Float(1))])
        }
    }

    @Test("Generated Metal evaluates every operation family on the GPU",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func gpuEvaluation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        // Each channel probes a different family. The harness supplies known inputs,
        // two uniforms, and a 2×1 texture with distinct texels.
        let graph = try SurfaceShaderGraph { s in
            let offset = s.uniform("Offset", SIMD2<Float>(0.25, 0))
            let gain = s.uniform("Gain", Float(0.5))
            let texel = s.texture("Lookup", addressing: .clampToEdge, filtering: .nearest).sample(s.uv0 + offset)
            let mixed = texel.rgb * gain                                    // (0.5, 0.25, 0) from the right texel
            let hsv = s.constant(SIMD3<Float>(0.2, 0.6, 0.9)).rgbToHSV().hsvToRGB()  // round trip
            let srgb = s.constant(SIMD3<Float>(0.5, 0.25, 0.75)).srgbToLinear().linearToSRGB()
            let wrapped = s.constant(Float(-0.25)).mod(1)                 // 0.75
            let swapped = s.vector(s.uv0.y, s.uv0.x)                         // (0.5, 0.6)
            let selected = s.select(s.constant(Float(1)), swapped.x, s.constant(Float(0)))  // 0.5
            let square = s.function("square", returns: Float.self, parameters: [.init("x", .scalar)], body: "return x * x;")
            let cubic = square(selected) * selected                          // 0.125
            let reflected = s.constant(SIMD3<Float>(1, -1, 0)).reflect(s.constant(SIMD3<Float>(0, 1, 0)))
            let checks = s.vector(wrapped, cubic, reflected.y)               // (0.75, 0.125, 1)
            let rough = s.constant(Float(0.9)).clamped(0.2, 0.4)             // 0.4
            return .init(
                baseColor: mixed, metallic: (hsv - s.constant(SIMD3<Float>(0.2, 0.6, 0.9))).length(),
                roughness: rough, emission: checks,
                transmissionMask: (srgb - s.constant(SIMD3<Float>(0.5, 0.25, 0.75))).length(),
                occlusion: s.noise(s.worldPosition, kind: .gradient)
            )
        }
        let function = try graph.metalFunction(named: "probe_surface")
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        \(SurfaceShaderGraph.supportSource)
        struct FrostfieldGraphResources {
            constant float4* uniforms;
            texture2d<float> lookup;
        };
        static inline float4 frostfield_graph_uniform(thread const FrostfieldGraphResources& r, uint index) {
            return r.uniforms[index];
        }
        static inline float4 frostfield_graph_sample(
            thread const FrostfieldGraphResources& r, uint slot, uint address, uint filter, float2 uv
        ) {
            constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest);
            return r.lookup.sample(nearest, uv);
        }
        \(function)
        kernel void probe(
            constant float4* uniforms [[buffer(0)]],
            device float4* output [[buffer(1)]],
            texture2d<float> lookup [[texture(0)]]
        ) {
            FrostfieldSurfaceInput input = {};
            input.uv0 = float2(0.6f, 0.5f);
            input.worldPosition = float3(0.3f, 1.7f, -2.2f);
            input.normal = float3(0, 0, 1);
            FrostfieldGraphResources resources = { uniforms, lookup };
            FrostfieldSurfaceOutput surface = { float3(0), 0, 0, float3(0), 1, 1, float3(0, 0, 1), false };
            probe_surface(input, resources, surface);
            output[0] = float4(surface.baseColor, surface.metallic);
            output[1] = float4(surface.emission, surface.roughness);
            output[2] = float4(surface.transmissionMask, surface.occlusion, 0, 0);
        }
        """
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: try #require(library.makeFunction(name: "probe")))

        let uniforms = try ShaderUniforms(graph: graph)
        let values = uniforms.snapshot().values
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 2, height: 1, mipmapped: false)
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let texels: [SIMD4<Float>] = [SIMD4(0.1, 0.2, 0.3, 1), SIMD4(1, 0.5, 0, 1)]
        texels.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: 32)
        }
        let output = try #require(device.makeBuffer(length: 48, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        values.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setTexture(texture, index: 0)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)

        let result = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: 3))
        func near(_ a: SIMD4<Float>, _ b: SIMD4<Float>, _ tolerance: Float = 1e-3) -> Bool {
            simd_reduce_max(simd_abs(a - b)) <= tolerance
        }
        #expect(near(result[0], SIMD4(0.5, 0.25, 0, 0)), "texture × uniform and HSV round trip: \(result[0])")
        #expect(near(result[1], SIMD4(0.75, 0.125, 1, 0.4)), "mod, select, function, reflect, clamp: \(result[1])")
        #expect(result[2].x < 1e-3, "sRGB round trip: \(result[2].x)")
        #expect(result[2].y > 0 && result[2].y < 1, "gradient noise stays in range: \(result[2].y)")
    }

    @Test("Raster preparation reports compiler errors with the generated line",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func compilerDiagnostics() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let broken = ShaderFunction(name: "broken", parameters: [.init("x", .scalar)], returns: .scalar,
                                    body: "return undefined_symbol * x;")
        let graph = try SurfaceShaderGraph(nodes: [.scalar(1), .function(0, arguments: [0])], functions: [broken],
                                           outputs: .init(roughness: 1))
        do {
            _ = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: .cube(),
                material: .physicallyBased(PhysicallyBasedMaterial(surfaceGraph: graph))
            )
            Issue.record("A graph with an undefined symbol compiled.")
        } catch {
            let message = error.localizedDescription
            #expect(message.contains("undefined_symbol"), "\(message)")
            #expect(message.contains("Line "), "\(message)")
        }
    }

    @Test("Graphs at the node limit validate and generate on a 512 KB stack, and still report cycles and missing nodes")
    func deepChain() throws {
        // Each node doubles the one before it, so validation and emission walk a
        // dependency chain as deep as the graph allows.
        let count = MaterialGraph.maximumNodeCount
        let chain: [MaterialGraph.Node] = [.input(.metallic)] + (1..<count).map { .add($0 - 1, $0 - 1) }
        // Node 1 also reads the last node, closing a loop through the whole chain,
        // or reads a node past the end.
        let looped = [chain[0], .add(0, count - 1)] + chain[2...]
        let dangling = [chain[0], .add(0, count + 5)] + chain[2...]
        let outcome = try Self.onSmallStack { () -> (String, MaterialGraph.Failure?, MaterialGraph.Failure?) in
            let graph = try SurfaceShaderGraph(nodes: chain, outputs: .init(metallic: count - 1))
            func failure(_ nodes: [MaterialGraph.Node]) -> MaterialGraph.Failure? {
                do {
                    _ = try SurfaceShaderGraph(nodes: nodes, outputs: .init(metallic: count - 1))
                    return nil
                } catch {
                    return error as? MaterialGraph.Failure
                }
            }
            return (try graph.metalFunction(named: "deep_chain"), failure(looped), failure(dangling))
        }
        let (function, cycle, missing) = outcome
        #expect(cycle == .cycle(count - 1))
        #expect(missing == .invalidReference(node: 1, referenced: count + 5))
        // Statements follow dependency order.
        let first = try #require(function.range(of: "float n1 = n0 + n0;"))
        let last = try #require(function.range(of: "float n\(count - 1) = n\(count - 2) + n\(count - 2);"))
        #expect(first.upperBound < last.lowerBound)

        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let source = Self.probeSource(function) + """
        kernel void probe(constant float4* inputs [[buffer(0)]], device float4* output [[buffer(1)]]) {
            FrostfieldGraphResources resources = { inputs };
            FrostfieldSurfaceInput input = {};
            input.metallic = 0.0f;
            FrostfieldSurfaceOutput surface = { float3(0), 0.5f, 0.5f, float3(0), 1, 1, float3(0, 0, 1), false };
            deep_chain(input, resources, surface);
            output[0] = float4(surface.metallic);
        }
        """
        let result = try Self.runProbe(source, inputs: [.zero], outputCount: 1, device: device)
        #expect(result[0].x == 0, "\(result[0])")
    }

    @Test("A label ending in a backslash stays inside its comment")
    func backslashLabel() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(
                roughness: s.roughness.named("rough\\"),
                positionOffset: s.displacement(s.constant(Float(0.1))).named("wave\\")
            )
        }
        let surface = try graph.metalFunction(named: "probe_surface")
        let vertex = try #require(try graph.metalVertexFunction(named: "probe_offset"))
        for function in [surface, vertex] {
            #expect(!function.contains("\\\n"), "\(function)")
        }
        #expect(vertex.contains("wave/"))

        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let source = Self.probeSource(surface + "\n\n" + vertex) + """
        kernel void probe(constant float4* inputs [[buffer(0)]], device float4* output [[buffer(1)]]) {
            FrostfieldGraphResources resources = { inputs };
            FrostfieldVertexInput vertexInput = {};
            vertexInput.objectNormal = float3(0, 1, 0);
            output[0] = float4(probe_offset(vertexInput, resources), 0);
            FrostfieldSurfaceInput input = {};
            input.roughness = 0.4f;
            FrostfieldSurfaceOutput surface = { float3(0), 0, 0.9f, float3(0), 1, 1, float3(0, 0, 1), false };
            probe_surface(input, resources, surface);
            output[1] = float4(surface.roughness);
        }
        """
        let result = try Self.runProbe(source, inputs: [.zero], outputCount: 2, device: device)
        #expect(simd_reduce_max(simd_abs(result[0] - SIMD4(0, 0.1, 0, 0))) < 1e-6, "vertex offset: \(result[0])")
        #expect(abs(result[1].x - 0.4) < 1e-6, "roughness: \(result[1])")
    }

    @Test("Operands from another builder are rejected rather than read as local nodes")
    func foreignOperands() throws {
        var leaked: ShaderValue<Float>?
        var leakedUV: ShaderValue<SIMD2<Float>>?
        _ = try SurfaceShaderGraph { s in
            leaked = s.uniform("Wetness", Float(0.25))
            leakedUV = s.uv0
            return .init(roughness: leaked)
        }
        let wetness = try #require(leaked)
        let uv = try #require(leakedUV)
        let foreign = MaterialGraph.Failure.invalidDeclaration("A graph value belongs to a different builder.")
        #expect(throws: foreign) {
            try SurfaceShaderGraph { s in .init(roughness: s.roughness * wetness) }
        }
        #expect(throws: foreign) {
            try SurfaceShaderGraph { s in .init(roughness: s.select(wetness, s.roughness, s.metallic)) }
        }
        #expect(throws: foreign) {
            try SurfaceShaderGraph { s in .init(roughness: s.texture("Albedo").sample(uv).x) }
        }
        #expect(throws: foreign) {
            try SurfaceShaderGraph { s in
                let square = s.function("square", returns: Float.self, parameters: [.init("x", .scalar)], body: "return x * x;")
                return .init(roughness: square(wetness))
            }
        }
        // A foreign operand fails the graph even when its node is not connected to an output.
        #expect(throws: foreign) {
            try SurfaceShaderGraph { s in
                _ = s.metallic.mix(s.roughness, by: wetness)
                return .init(roughness: s.roughness)
            }
        }
        #expect(throws: foreign) {
            try TextureShaderGraph { t in t.gridUV.x + wetness }
        }
        #expect(throws: foreign) {
            try PostProcessGraph { p in p.sceneColor(at: uv) }
        }
    }

    @Test("Normalizing a vector with a NaN or infinite component yields zero on the GPU",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func nonfiniteNormalize() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let source = Self.probeSource("") + """
        kernel void probe(constant float4* inputs [[buffer(0)]], device float4* output [[buffer(1)]]) {
            output[0] = float4(frostfield_graph_unit(inputs[0].xyz), 0);
            output[1] = float4(frostfield_graph_unit(inputs[1].xyz), 0);
            output[2] = frostfield_graph_unit(inputs[2]);
            output[3] = float4(frostfield_graph_unit(inputs[1].xy), 0, 0);
            output[4] = float4(frostfield_graph_unit(inputs[3].xyz), 0);
            output[5] = float4(frostfield_graph_unit(inputs[4].xyz), 0);
        }
        """
        let inputs: [SIMD4<Float>] = [
            SIMD4(1, .nan, 0, 0), SIMD4(.nan, 1, 0, 0), SIMD4(1, 0, 0, .nan), SIMD4(3, 4, 0, 0), SIMD4(.infinity, 1, 0, 0),
        ]
        let result = try Self.runProbe(source, inputs: inputs, outputCount: 6, device: device)
        for index in [0, 1, 2, 3, 5] {
            #expect(result[index] == .zero, "output \(index): \(result[index])")
        }
        #expect(simd_reduce_max(simd_abs(result[4] - SIMD4(0.6, 0.8, 0, 0))) < 1e-6, "\(result[4])")
    }

    @Test("Metal predefined macro names are not accepted as generated identifiers")
    func macroNames() throws {
        let macros = [
            "FLT_MAX", "FLT_MIN", "FLT_EPSILON", "HALF_MAX", "M_PI_F", "M_PI_2_F", "M_E_F", "M_SQRT2_F", "M_PI_H",
            "INFINITY", "NAN", "MAXFLOAT", "HUGE_VALF", "INT_MAX", "UINT_MAX", "DBL_MAX", "CHAR_BIT", "assert",
            "__METAL_VERSION__", "__HAVE_MESH__",
        ]
        let graph = try SurfaceShaderGraph(nodes: [.scalar(0.5)], outputs: .init(roughness: 0))
        for name in macros {
            let parameter = ShaderFunction(name: "probe", parameters: [.init(name, .scalar)], returns: .scalar, body: "return 1.0f;")
            #expect(throws: MaterialGraph.Failure.invalidDeclaration("Function probe has an invalid parameter name \"\(name)\".")) {
                try SurfaceShaderGraph(nodes: [.scalar(1)], functions: [parameter], outputs: .init(roughness: 0))
            }
            let function = ShaderFunction(name: name, parameters: [], returns: .scalar, body: "return 1.0f;")
            #expect(throws: MaterialGraph.Failure.invalidDeclaration("Function names must be unique Metal identifiers: \"\(name)\".")) {
                try SurfaceShaderGraph(nodes: [.scalar(1)], functions: [function], outputs: .init(roughness: 0))
            }
            #expect(throws: SurfaceShaderGraph.Failure.invalidFunctionName(name)) {
                try graph.metalFunction(named: name)
            }
        }

        // Short all-caps parameter names remain valid, and uniform names are labels,
        // not Metal identifiers, so a macro name there compiles.
        let scaled = try SurfaceShaderGraph { s in
            let gain = s.uniform("FLT_MAX", Float(0.5))
            let function = s.function(
                "scaled", returns: Float.self, parameters: [.init("N", .scalar), .init("UV", .scalar)], body: "return N * UV;"
            )
            return .init(roughness: function(gain, s.roughness))
        }
        let function = try scaled.metalFunction(named: "M_scaled")
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let source = Self.probeSource(function) + """
        kernel void probe(constant float4* inputs [[buffer(0)]], device float4* output [[buffer(1)]]) {
            FrostfieldGraphResources resources = { inputs };
            FrostfieldSurfaceInput input = {};
            input.roughness = 0.4f;
            FrostfieldSurfaceOutput surface = { float3(0), 0, 0.9f, float3(0), 1, 1, float3(0, 0, 1), false };
            M_scaled(input, resources, surface);
            output[0] = float4(surface.roughness);
        }
        """
        let uniforms = try ShaderUniforms(graph: scaled).snapshot().values
        let result = try Self.runProbe(source, inputs: uniforms, outputCount: 1, device: device)
        #expect(abs(result[0].x - 0.2) < 1e-6, "\(result[0])")
    }

    // MARK: - Helpers

    /// Runs work on a thread with a 512 KB stack, the size of Swift concurrency and
    /// Dispatch worker threads, and returns its result.
    private static func onSmallStack<T>(_ work: @escaping @Sendable () throws -> T) throws -> T {
        let box = ResultBox<T>()
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            box.result = Result { try work() }
            done.signal()
        }
        thread.stackSize = 512 * 1024
        thread.start()
        done.wait()
        return try box.result!.get()
    }

    /// The graph support source, a resource struct whose uniforms are the probe's
    /// input buffer, and the given generated functions.
    private static func probeSource(_ functions: String) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;
        \(SurfaceShaderGraph.supportSource)
        struct FrostfieldGraphResources {
            constant float4* uniforms;
        };
        static inline float4 frostfield_graph_uniform(thread const FrostfieldGraphResources& r, uint index) {
            return r.uniforms[index];
        }
        \(functions)

        """
    }

    /// Compiles a source with a `probe` kernel using safe math, as the renderers do,
    /// and runs one thread with the inputs at buffer 0 and the outputs at buffer 1.
    private static func runProbe(
        _ source: String, inputs: [SIMD4<Float>], outputCount: Int, device: any MTLDevice
    ) throws -> [SIMD4<Float>] {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: try #require(library.makeFunction(name: "probe")))
        let input = try #require(inputs.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        let output = try #require(device.makeBuffer(length: outputCount * 16, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: outputCount))
    }
}

/// Carries a result across a thread hand-off that a semaphore orders.
private final class ResultBox<T>: @unchecked Sendable {
    var result: Result<T, any Error>?
}
