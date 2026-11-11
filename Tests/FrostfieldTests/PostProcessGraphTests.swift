import Frostfield
import Metal
import Testing
import simd

struct PostProcessGraphTests {
    /// Premultiplied pixels with exact half-float values and alternating alpha.
    private static func frame(_ x: Int, _ y: Int) -> SIMD4<Float> {
        let alpha: Float = (x + y) % 2 == 0 ? 1 : 0.5
        return SIMD4(Float(x) / 8, Float(y) / 8, Float(x + y) / 16, alpha)
    }

    @Test("The generated function has the post-process signature and sanitizes its result")
    func generatedSource() throws {
        let rgba = try PostProcessGraph { p in p.sceneColor(at: p.screenUV) }
        let rgb = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * p.uniform("Gain", Float(2)) }
        #expect(rgba.outputType == .vector4 && rgb.outputType == .vector3)
        #expect(!rgba.readsDepth)

        let source = try rgb.metalFunction(named: "gain_post")
        #expect(source.contains("static inline float4 gain_post("))
        #expect(source.contains("thread const FrostfieldPostInput& input"))
        #expect(source.contains("thread const FrostfieldGraphResources& resources"))
        #expect(source.contains("frostfield_graph_scene(resources, n"))
        #expect(source.contains("frostfield_graph_uniform(resources, 0u)"))
        #expect(source.contains("frostfield_graph_finite("))
        // A vector3 result takes the source alpha at the pixel.
        #expect(source.contains("frostfield_graph_scene(resources, input.screenUV).a"))
        #expect(!(try rgba.metalFunction(named: "copy_post")).contains("input.screenUV).a"))
        #expect(throws: PostProcessGraph.Failure.invalidFunctionName("9post")) {
            try rgba.metalFunction(named: "9post")
        }
    }

    @Test("Graphs reject surface inputs, non-color outputs and values from other builders")
    func graphValidation() throws {
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .postProcess)) {
            try PostProcessGraph { p in p.worldPosition }
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .postProcess)) {
            try PostProcessGraph(nodes: [.input(.uv0), .vector2(.zero), .add(0, 1), .sampleScene(coordinates: 2)], output: 3)
        }
        #expect(throws: PostProcessGraph.Failure.outputType(node: 0, actual: .scalar)) {
            try PostProcessGraph(nodes: [.scalar(1)], output: 0)
        }
        var foreign: ShaderValue<SIMD4<Float>>?
        _ = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            foreign = color
            return color
        }
        let leaked = try #require(foreign)
        #expect(throws: MaterialGraph.Failure.self) {
            try PostProcessGraph { _ in leaked }
        }
    }

    @Test("An inverting graph inverts a known premultiplied 4×4 frame",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func inversion() throws {
        let gpu = try PostProcessFixture()
        let invert = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            return p.vector(color.a.xxx - color.rgb, color.a)
        }
        let pass = try PostProcessPass(graph: invert, device: gpu.device)
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<4 {
            for x in 0..<4 {
                let input = Self.frame(x, y)
                let expected = SIMD4(input.w - input.x, input.w - input.y, input.w - input.z, input.w)
                #expect(PostProcessFixture.close(pixels[y * 4 + x], expected, tolerance: 1e-3), "(\(x), \(y)): \(pixels[y * 4 + x])")
            }
        }
    }

    @Test("A vector3 output keeps the source alpha without premultiplying again",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func rgbOutputKeepsAlpha() throws {
        let gpu = try PostProcessFixture()
        let double = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * 2 }
        let pass = try PostProcessPass(graph: double, device: gpu.device)
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let destination = try gpu.color(width: 4, height: 4)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<4 {
            for x in 0..<4 {
                let input = Self.frame(x, y)
                let expected = SIMD4(input.x * 2, input.y * 2, input.z * 2, input.w)
                #expect(PostProcessFixture.close(pixels[y * 4 + x], expected, tolerance: 1e-3), "(\(x), \(y)): \(pixels[y * 4 + x])")
            }
        }
    }

    @Test("Screen coordinates are output pixel centers and texel size is one over the output size",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func screenCoordinates() throws {
        let gpu = try PostProcessFixture()
        let coordinates = try PostProcessGraph { p in
            p.vector(p.screenUV.x, p.screenUV.y, p.texelSize.x, p.texelSize.y)
        }
        let pass = try PostProcessPass(graph: coordinates, device: gpu.device)
        let source = try gpu.color(width: 4, height: 2) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try gpu.color(width: 4, height: 2)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<2 {
            for x in 0..<4 {
                let expected = SIMD4((Float(x) + 0.5) / 4, (Float(y) + 0.5) / 2, 0.25, 0.5)
                #expect(PostProcessFixture.close(pixels[y * 4 + x], expected, tolerance: 1e-3), "(\(x), \(y)): \(pixels[y * 4 + x])")
            }
        }
    }

    @Test("Depth sampling returns the stored Metal depth at each pixel",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func depthSampling() throws {
        let gpu = try PostProcessFixture()
        let graph = try PostProcessGraph { p in
            let depth = p.sceneDepth(at: p.screenUV)
            return p.vector(depth, 1 - depth, depth * depth, p.constant(Float(1)))
        }
        #expect(graph.readsDepth)
        let pass = try PostProcessPass(graph: graph, device: gpu.device)
        func stored(_ x: Int, _ y: Int) -> Float { Float(y * 4 + x + 1) / 20 }
        let depth = try gpu.depth(width: 4, height: 4, value: stored)
        let source = try gpu.color(width: 4, height: 4) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try gpu.color(width: 4, height: 4)
        try gpu.run { try pass.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<4 {
            for x in 0..<4 {
                let d = stored(x, y)
                #expect(PostProcessFixture.close(pixels[y * 4 + x], SIMD4(d, 1 - d, d * d, 1), tolerance: 1e-3),
                        "(\(x), \(y)): \(pixels[y * 4 + x])")
            }
        }
        #expect(throws: PostProcessPass.Failure.missingDepth) {
            try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        }
    }

    @Test("Editing a uniform changes the output while the pass keeps its pipeline and shared library",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func uniformEditing() throws {
        let gpu = try PostProcessFixture()
        let gain = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * p.uniform("Gain", Float(1)) }
        let pass = try PostProcessPass(graph: gain, device: gpu.device)
        let pipeline = pass.pipelineState
        let library = pass.library
        let source = try gpu.color(width: 4, height: 4, pixel: Self.frame)
        let first = try gpu.color(width: 4, height: 4)
        let second = try gpu.color(width: 4, height: 4)

        try gpu.run { try pass.encode(source: source, destination: first, commandBuffer: $0) }
        try pass.uniforms.set("Gain", to: 3)
        #expect(pass.uniforms.revision == 1)
        try gpu.run { try pass.encode(source: source, destination: second, commandBuffer: $0) }

        let before = try gpu.read(first)
        let after = try gpu.read(second)
        for index in before.indices {
            let input = Self.frame(index % 4, index / 4)
            #expect(PostProcessFixture.close(before[index], input, tolerance: 1e-3))
            #expect(PostProcessFixture.close(after[index], SIMD4(input.x * 3, input.y * 3, input.z * 3, input.w), tolerance: 1e-3))
        }
        #expect(pass.pipelineState === pipeline)
        #expect(pass.library === library)

        // The same graph on the same device reuses the compiled library, even for another output format.
        let display = try PostProcessPass(graph: gain, device: gpu.device, pixelFormat: .bgra8Unorm_srgb, uniforms: pass.uniforms)
        #expect(display.library === library)
        #expect(display.pipelineState !== pipeline)
        #expect(display.uniforms === pass.uniforms)
        let other = try PostProcessPass(graph: try .vignette(), device: gpu.device)
        #expect(other.library !== library)
    }

    @Test("Declared textures bind by name to their slots",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func textureBinding() throws {
        let gpu = try PostProcessFixture()
        let graph = try PostProcessGraph { p in
            let tint = p.texture("Tint").sample(p.screenUV).rgb
            let ramp = p.texture("Ramp", content: .data, addressing: .clampToEdge, filtering: .nearest).sample(p.screenUV)
            return p.sceneColor(at: p.screenUV).rgb * tint + ramp.rgb * 0.125
        }
        let tint = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(0.5, 0.25, 1, 1) }
        let ramp = try gpu.color(width: 2, height: 1) { x, _ in x == 0 ? SIMD4(0, 0, 0, 1) : SIMD4(1, 2, 4, 1) }
        let pass = try PostProcessPass(graph: graph, device: gpu.device, textures: ["Ramp": ramp, "Tint": tint])
        let source = try gpu.color(width: 4, height: 2) { _, _ in SIMD4(1, 1, 1, 1) }
        let destination = try gpu.color(width: 4, height: 2)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<2 {
            for x in 0..<4 {
                let expected = x < 2 ? SIMD4<Float>(0.5, 0.25, 1, 1) : SIMD4<Float>(0.625, 0.5, 1.5, 1)
                #expect(PostProcessFixture.close(pixels[y * 4 + x], expected, tolerance: 1e-3), "(\(x), \(y)): \(pixels[y * 4 + x])")
            }
        }
    }

    @Test("Preparation and encoding reject missing bindings, mismatched uniforms and sizes",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func passValidation() throws {
        let gpu = try PostProcessFixture()
        let textured = try PostProcessGraph { p in
            p.sceneColor(at: p.screenUV).rgb * p.texture("Tint").sample(p.screenUV).rgb * p.uniform("Gain", Float(1))
        }
        let tint = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(1, 1, 1, 1) }
        #expect(throws: PostProcessPass.Failure.missingTexture("Tint")) {
            try PostProcessPass(graph: textured, device: gpu.device)
        }
        #expect(throws: PostProcessPass.Failure.unknownTexture("Extra")) {
            try PostProcessPass(graph: textured, device: gpu.device, textures: ["Tint": tint, "Extra": tint])
        }
        let depth = try gpu.depth(width: 1, height: 1) { _, _ in 0 }
        #expect(throws: PostProcessPass.Failure.incompatibleTexture("Tint")) {
            try PostProcessPass(graph: textured, device: gpu.device, textures: ["Tint": depth])
        }
        let wrongType = try ShaderUniforms([ShaderUniform("Gain", SIMD3<Float>(1, 1, 1))])
        #expect(throws: PostProcessPass.Failure.uniformDeclarations) {
            try PostProcessPass(graph: textured, device: gpu.device, uniforms: wrongType, textures: ["Tint": tint])
        }
        let wrongName = try ShaderUniforms([ShaderUniform("Strength", Float(1))])
        #expect(throws: PostProcessPass.Failure.uniformDeclarations) {
            try PostProcessPass(graph: textured, device: gpu.device, uniforms: wrongName, textures: ["Tint": tint])
        }
        #expect(throws: PostProcessPass.Failure.pixelFormat) {
            try PostProcessPass(graph: textured, device: gpu.device, pixelFormat: .bgra8Unorm, textures: ["Tint": tint])
        }

        let pass = try PostProcessPass(graph: textured, device: gpu.device, textures: ["Tint": tint])
        let source = try gpu.color(width: 4, height: 4)
        let small = try gpu.color(width: 4, height: 4)
        let large = try gpu.color(width: 8, height: 4)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessPass.Failure.sizeMismatch) {
            try pass.encode(source: source, destination: large, commandBuffer: buffer)
        }
        #expect(throws: PostProcessPass.Failure.destination) {
            try pass.encode(source: source, destination: source, commandBuffer: buffer)
        }
        #expect(throws: PostProcessPass.Failure.destination) {
            try pass.encode(source: source, destination: tint, commandBuffer: buffer)
        }
        let display = try gpu.color(width: 4, height: 4, pixelFormat: .bgra8Unorm_srgb)
        #expect(throws: PostProcessPass.Failure.destination) {
            try pass.encode(source: source, destination: display, commandBuffer: buffer)
        }
        #expect(throws: PostProcessPass.Failure.source) {
            try pass.encode(source: display, destination: small, commandBuffer: buffer)
        }
        let unretained = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        #expect(throws: PostProcessPass.Failure.commandBuffer) {
            try pass.encode(source: source, destination: small, commandBuffer: unretained)
        }
    }

    // MARK: - Presets

    @Test("The vignette preset darkens toward the corners by its documented falloff",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func vignette() throws {
        let gpu = try PostProcessFixture()
        let pass = try PostProcessPass(graph: try .vignette(strength: 0.8), device: gpu.device)
        let source = try gpu.color(width: 8, height: 6) { _, _ in SIMD4(1, 0.5, 0.25, 1) }
        let destination = try gpu.color(width: 8, height: 6)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<6 {
            for x in 0..<8 {
                let centered = SIMD2((Float(x) + 0.5) / 8, (Float(y) + 0.5) / 6) * 2 - 1
                let edge = min(max((centered * centered).sum() * 0.5, 0), 1)
                let factor = 1 - 0.8 * edge * edge * (3 - 2 * edge)
                let expected = SIMD4(factor, 0.5 * factor, 0.25 * factor, 1)
                #expect(PostProcessFixture.close(pixels[y * 8 + x], expected), "(\(x), \(y)): \(pixels[y * 8 + x])")
            }
        }
        #expect(pixels[0].x < pixels[3 * 8 + 3].x)
    }

    @Test("The box blur preset spreads an impulse evenly over a 3×3 block",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func boxBlur() throws {
        let gpu = try PostProcessFixture()
        let pass = try PostProcessPass(graph: try .boxBlur(), device: gpu.device)
        let source = try gpu.color(width: 5, height: 5) { x, y in
            x == 2 && y == 2 ? SIMD4(9, 4.5, 0, 1) : SIMD4(0, 0, 0, 1)
        }
        let destination = try gpu.color(width: 5, height: 5)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<5 {
            for x in 0..<5 {
                let inside = abs(x - 2) <= 1 && abs(y - 2) <= 1
                let expected = inside ? SIMD4<Float>(1, 0.5, 0, 1) : SIMD4<Float>(0, 0, 0, 1)
                #expect(PostProcessFixture.close(pixels[y * 5 + x], expected), "(\(x), \(y)): \(pixels[y * 5 + x])")
            }
        }
    }

    @Test("The sharpen preset applies its five-tap kernel",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func sharpen() throws {
        let gpu = try PostProcessFixture()
        let pass = try PostProcessPass(graph: try .sharpen(amount: 0.25), device: gpu.device)
        let source = try gpu.color(width: 5, height: 5) { x, y in
            x == 2 && y == 2 ? SIMD4(1, 1, 1, 1) : SIMD4(0.25, 0.25, 0.25, 1)
        }
        let destination = try gpu.color(width: 5, height: 5)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        // Center: 1 + 0.25 · (4 − 1). Cross neighbors: 0.25 + 0.25 · (1 − 1.75). Elsewhere flat.
        #expect(PostProcessFixture.close(pixels[2 * 5 + 2], SIMD4(1.75, 1.75, 1.75, 1)))
        #expect(PostProcessFixture.close(pixels[1 * 5 + 2], SIMD4(0.0625, 0.0625, 0.0625, 1)))
        #expect(PostProcessFixture.close(pixels[2 * 5 + 3], SIMD4(0.0625, 0.0625, 0.0625, 1)))
        #expect(PostProcessFixture.close(pixels[1 * 5 + 1], SIMD4(0.25, 0.25, 0.25, 1)))
        #expect(PostProcessFixture.close(pixels[4 * 5 + 0], SIMD4(0.25, 0.25, 0.25, 1)))
    }

    @Test("The chromatic aberration preset displaces red and blue radially",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func chromaticAberration() throws {
        let gpu = try PostProcessFixture()
        let pass = try PostProcessPass(graph: try .chromaticAberration(amount: 2), device: gpu.device)
        let source = try gpu.color(width: 16, height: 4) { x, _ in
            SIMD4(Float(x) / 15, 0.4, 1 - Float(x) / 15, 1)
        }
        let destination = try gpu.color(width: 16, height: 4)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }

        let pixels = try gpu.read(destination)
        for y in 0..<4 {
            for x in 0..<16 {
                let shift = ((Float(x) + 0.5) / 16 * 2 - 1) * 2
                let red = min(max(Float(x) + shift, 0), 15) / 15
                let blue = 1 - min(max(Float(x) - shift, 0), 15) / 15
                let expected = SIMD4(red, 0.4, blue, 1)
                #expect(PostProcessFixture.close(pixels[y * 16 + x], expected), "(\(x), \(y)): \(pixels[y * 16 + x])")
            }
        }
    }

    @Test("The depth fog preset follows view distance for forward and reversed depth",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func depthFog() throws {
        let gpu = try PostProcessFixture()
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 1, far: 10)
        let distances: [Float] = [1, 2, 4, 6, 10]
        let factors: [Float] = [0, 0, 0.5, 1, 1]
        let source = try gpu.color(width: 5, height: 1) { _, _ in SIMD4(0, 0.5, 0, 1) }
        for mode in [MeshDepthMode.forward, .reversed] {
            let graph = try PostProcessGraph.depthFog(
                color: SIMD3(1, 0, 0), near: 2, far: 6, camera: camera, depthMode: mode
            )
            let pass = try PostProcessPass(graph: graph, device: gpu.device)
            let depth = try gpu.depth(width: 5, height: 1) { x, _ in
                let forward = 10 * (distances[x] - 1) / (9 * distances[x])
                return mode == .forward ? forward : 1 - forward
            }
            let destination = try gpu.color(width: 5, height: 1)
            try gpu.run { try pass.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }

            let pixels = try gpu.read(destination)
            for (x, factor) in factors.enumerated() {
                let expected = SIMD4(factor, 0.5 * (1 - factor), 0, 1)
                #expect(PostProcessFixture.close(pixels[x], expected), "\(mode) z = \(distances[x]): \(pixels[x])")
            }
        }
    }
    @Test("Exponential and height fog use perspective depth and preserve premultiplied alpha",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func exponentialFog() throws {
        let gpu = try PostProcessFixture()
        let width = 4
        let height = 3
        let camera = try PerspectiveCamera(verticalFieldOfView: 1.1, near: 0.5, far: 10)
        let inverseProjection = try camera.projection(aspectRatio: Float(width) / Float(height)).inverse
        let pose = SceneTransform(
            translation: SIMD3(0, 1.3, 0),
            rotation: simd_quatf(angle: 0.24, axis: simd_normalize(SIMD3<Float>(1, 0, 1)))
        )
        let distances: [Float] = [2, 4, 7, 10]
        let density: Float = 0.16
        let top: Float = 1.5
        let fogColor = SIMD3<Float>(1, 0, 0.25)
        let alphas: [Float] = [0.5, 1, 0.5, 0]
        let source = try gpu.color(width: width, height: height) { x, _ in
            SIMD4(0, alphas[x] * 0.5, 0, alphas[x])
        }
        #expect(throws: PostProcessGraph.Failure.fogDensity) {
            try PostProcessGraph.exponentialDepthFog(color: fogColor, density: -1, camera: camera)
        }

        for mode in [MeshDepthMode.forward, .reversed] {
            for usesHeight in [false, true] {
                let graph = try usesHeight
                    ? PostProcessGraph.exponentialHeightFog(
                        color: fogColor, density: density, height: top, camera: camera,
                        cameraTransform: pose, depthMode: mode
                    )
                    : PostProcessGraph.exponentialDepthFog(color: fogColor, density: density, camera: camera, depthMode: mode)
                let pass = try PostProcessPass(graph: graph, device: gpu.device)
                let forwardDepth = distances.map { distance in
                    camera.far * (distance - camera.near) / ((camera.far - camera.near) * distance)
                }
                let depth = try gpu.depth(width: width, height: height) { x, _ in
                    mode == .forward ? forwardDepth[x] : 1 - forwardDepth[x]
                }
                let destination = try gpu.color(width: width, height: height)
                try gpu.run { try pass.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
                let pixels = try gpu.read(destination)
                var factors: [Float] = []

                for y in 0..<height {
                    for x in 0..<width {
                        var distance = distances[x]
                        if usesHeight {
                            let ndcX = 2 * (Float(x) + 0.5) / Float(width) - 1
                            let ndcY = 1 - 2 * (Float(y) + 0.5) / Float(height)
                            let view = inverseProjection * SIMD4(ndcX, ndcY, forwardDepth[x], 1)
                            let world = pose.matrix * (view / view.w)
                            distance *= max(top - world.y, 0)
                        }
                        let opticalDepth = density * distance
                        let factor = 1 - exp(-opticalDepth * opticalDepth)
                        let alpha = alphas[x]
                        if alpha > 0 { factors.append(factor) }
                        let expected = SIMD4(
                            factor * alpha, 0.5 * (1 - factor) * alpha, 0.25 * factor * alpha, alpha
                        )
                        #expect(PostProcessFixture.close(pixels[y * width + x], expected),
                                "\(mode), height \(usesHeight), (\(x), \(y)): \(pixels[y * width + x])")
                    }
                }
                if usesHeight {
                    #expect(factors.contains(0))
                    #expect(factors.contains { $0 > 0.2 })
                }
            }
        }
    }

}
