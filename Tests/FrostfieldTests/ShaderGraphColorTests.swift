import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderGraphColorTests {
    @Test("Compute blends, ACES and posterize match r187 on signed, SDR and HDR colors")
    func blendToneAndPosterize() throws {
        let graph = try BufferShaderGraph(fieldCount: 4) { b in
            let base = b.elementField(0).rgb
            let blend = b.elementField(1)
            let steps = b.elementField(2).rgb
            let one = b.constant(Float(1))
            return [
                0: b.vector(b.blendScreen(base, blend.rgb), one),
                1: b.vector(b.blendOverlay(base, blend.rgb), one),
                2: b.vector(b.acesFilmicToneMapping(base, exposure: blend.a), one),
                3: b.vector(b.posterize(base, steps: steps), one),
            ]
        }
        let source: [[SIMD4<Float>]] = [
            [SIMD4(0.1, 0.5, 0.8, 0), SIMD4(0.6, 0.25, 0.1, 1), SIMD4(4, 8, 16, 0), .zero],
            [SIMD4(4, 0.5, 0.2, 0), SIMD4(0.1, 0.8, 0.6, 0.5), SIMD4(2, 4, 8, 0), .zero],
            [SIMD4(-0.6, 1.4, 0.2, 0), SIMD4(0.7, 0.25, 1.5, 1.5), SIMD4(4, 4, 4, 0), .zero],
            [.zero, SIMD4(1, 0, 0.5, 2), SIMD4(4, 4, 0, 0), .zero],
            [SIMD4(120, 0.5, 2, 0), SIMD4(0.2, 0.6, 0.9, 2), SIMD4(4, 4, 4, 0), .zero],
        ]
        // The expected RGBs come from evaluating the pinned TSL bodies with
        // r187 Matrix3 and ColorManagement. Frostfield defines zero steps as zero.
        let expected: [[SIMD4<Float>]] = [
            [SIMD4(0.64, 0.625, 0.82, 1), SIMD4(0.12, 0.25, 0.64, 1),
             SIMD4(0.216532518, 0.558042180, 0.695510596, 1), SIMD4(0, 0.5, 0.75, 1)],
            [SIMD4(3.7, 0.9, 0.68, 1), SIMD4(6.4, 0.8, 0.24, 1),
             SIMD4(1, 0.425035968, 0.183609382, 1), SIMD4(4, 0.5, 0.125, 1)],
            [SIMD4(0.52, 1.3, 1.4, 1), SIMD4(-0.84, 1.6, 0.6, 1),
             SIMD4(0, 0.940140995, 0.540357207, 1), SIMD4(-0.75, 1.25, 0, 1)],
            [SIMD4(1, 0, 0.5, 1), SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0, 1), SIMD4(0, 0, 0, 1)],
            [SIMD4(96.2, 0.8, 1.1, 1), SIMD4(191.4, 0.6, 1.2, 1),
             SIMD4(1, 1, 0.990832232, 1), SIMD4(120, 0.5, 2, 1)],
        ]
        try compare(graph, source: source, expected: expected)
    }

    @Test("CDL preserves alpha and uses original luma with default or supplied coefficients")
    func cdlAndScalarSteps() throws {
        let graph = try BufferShaderGraph(fieldCount: 4) { b in
            let color = b.elementField(0)
            let slope = b.elementField(1)
            let offset = b.elementField(2)
            let power = b.elementField(3).rgb
            return [
                0: b.cdl(color, slope: slope.rgb, offset: offset.rgb, power: power, saturation: slope.a),
                1: b.cdl(color, slope: slope.rgb, offset: offset.rgb, power: power, saturation: slope.a,
                         luminanceCoefficients: b.constant(SIMD3<Float>(0.25, 0.5, 0.25))),
                2: b.cdl(color),
                3: b.vector(b.posterize(color.r, steps: b.constant(SIMD3<Float>(2, 4, 8))),
                            b.posterize(color.b, steps: offset.a)),
            ]
        }
        let source: [[SIMD4<Float>]] = [
            [SIMD4(0.2, 0.4, 0.6, 0.125), SIMD4(2, 1, 0.5, 0.75), SIMD4(-0.1, 0.1, 0.05, 4), SIMD4(2, 0.5, 1.5, 0)],
            [SIMD4(1.2, 0.1, -0.1, 0.625), SIMD4(1, 1, 1, 1.8), SIMD4(0.5, -1, -0.1, 4), SIMD4(1.25, 0.5, 2, 0)],
            [SIMD4(0.1, 0.5, 0.9, 0.42), SIMD4(3, 2, 1, 0), SIMD4(0.15, 0, -0.1, 0), SIMD4(0.5, 0.5, 0.5, 0)],
            [SIMD4(-0.4, 0, 0.8, -0.1), SIMD4(1, 1, 1, 0.4), SIMD4(-0.3, 0, 0.2, 8), SIMD4(1, 2, 0.5, 0)],
        ]
        let expected: [[SIMD4<Float>]] = [
            [SIMD4(0.16048, 0.623310086, 0.248277094, 0.125), SIMD4(0.1675, 0.630330086, 0.255297094, 0.125),
             SIMD4(0.2, 0.4, 0.6, 0.125), SIMD4(0, 0, 0.125, 0.5)],
            [SIMD4(3.238550537, 0, 0, 0.625), SIMD4(3.234086537, 0, 0, 0.625),
             SIMD4(1.2, 0.1, 0, 0.625), SIMD4(1, 1, 1.125, -0.25)],
            [SIMD4(0.44384, 0.44384, 0.44384, 0.42), SIMD4(0.5, 0.5, 0.5, 0.42),
             SIMD4(0.1, 0.5, 0.9, 0.42), .zero],
            [SIMD4(0, 0, 0.383632, -0.1), SIMD4(0.06, 0.06, 0.46, -0.1),
             SIMD4(0, 0, 0.8, -0.1), SIMD4(-0.5, -0.5, -0.5, 0.75)],
        ]
        try compare(graph, source: source, expected: expected)
    }

    @Test("A material composes all five operators with live exposure and retains straight CDL alpha")
    func surfaceComposition() throws {
        let gpu = try GeometryRenderFixture(side: 17)
        let graph = try SurfaceShaderGraph { b in
            let overlay = b.blendOverlay(b.parameters.rgb, b.constant(SIMD3<Float>(0.1, 0.8, 0.6)))
            let screened = b.blendScreen(overlay, b.constant(SIMD3<Float>(0.2, 0.1, 0.35)))
            let grade = b.cdl(
                b.vector(screened, b.constant(Float(0.625))),
                slope: b.constant(SIMD3<Float>(0.75, 1, 1.3)),
                offset: b.constant(SIMD3<Float>(-0.1, 0.05, -0.1)),
                power: b.constant(SIMD3<Float>(1.2, 1, 0.8)),
                saturation: b.constant(Float(0.7))
            )
            let color = b.acesFilmicToneMapping(grade.rgb, exposure: b.uniform("Exposure", Float(1)))
            return .init(baseColor: b.posterize(color, steps: b.constant(Float(16))), opacity: grade.a)
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let material = try PhysicallyBasedMaterial(
            alphaMode: .blend, surfaceGraph: graph, surfaceUniforms: uniforms, shadingModel: .unlit
        )
        let mesh = try gpu.unlit(.plane(width: 2, height: 2), material: .physicallyBased(material))
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh, materialParameters: SIMD4(4, 0.5, 0.2, 1)
        )
        let scene = SceneGraph(roots: [node], camera: gpu.camera)
        for (exposure, expected): (Float, SIMD4<Float>) in [
            (0.25, SIMD4(0.5078125, 0.234375, 0.1953125, 0.625)),
            (1, SIMD4(0.5859375, 0.5078125, 0.46875, 0.625)),
        ] {
            try uniforms.set("Exposure", to: exposure)
            let rendered = try gpu.render(scene).pixels
            #expect(isNearly(gpu.pixel(rendered, x: 8, y: 8), expected, tolerance: 0.001))
        }
    }

    @Test("Post graph ACES agrees with the native display curve at matching exposure")
    func displayCurve() throws {
        let gpu = try PostProcessFixture()
        let colors: [SIMD3<Float>] = [.zero, SIMD3(0.125, 0.5, 2), SIMD3(120, 0.5, 2), SIMD3(1, 0.25, 0.75)]
        let source = try gpu.color(width: colors.count, height: 1) { x, _ in SIMD4(colors[x], 1) }
        let graphTarget = try gpu.color(width: colors.count, height: 1)
        let displayTarget = try gpu.color(width: colors.count, height: 1)
        let display = try HDRDisplay(device: gpu.device, destinationPixelFormat: .rgba16Float)
        for stops: Float in [0, 1] {
            let graph = try PostProcessGraph { b in
                let color = b.sceneColor(at: b.screenUV)
                let exposure = stops == 0 ? nil : b.constant(Float(2))
                return b.vector(b.acesFilmicToneMapping(color.rgb, exposure: exposure), color.a)
            }
            let pass = try PostProcessPass(graph: graph, device: gpu.device)
            try gpu.run { command in
                try pass.encode(source: source, destination: graphTarget, commandBuffer: command)
                try display.encode(
                    source: source, destination: displayTarget,
                    settings: .init(exposureStops: stops, toneMapping: .acesFilmic), commandBuffer: command
                )
            }
            for (graphPixel, displayPixel) in try zip(gpu.read(graphTarget), gpu.read(displayTarget)) {
                #expect(isNearly(graphPixel, displayPixel, tolerance: 0.001))
            }
        }
    }

    private func compare(
        _ graph: BufferShaderGraph, source: [[SIMD4<Float>]], expected: [[SIMD4<Float>]]
    ) throws {
        let gpu = try GeometryRenderFixture()
        let records = source.flatMap { $0 }
        let size = records.count * MemoryLayout<SIMD4<Float>>.stride
        let input = try #require(gpu.device.makeBuffer(bytes: records, length: size, options: .storageModeShared))
        let output = try #require(gpu.device.makeBuffer(length: size, options: .storageModeShared))
        let command = try #require(gpu.queue.makeCommandBuffer())
        let kernel = try ShaderBufferKernel(graph: graph, device: gpu.device)
        try kernel.encode(elementCount: source.count, from: input, to: output, commandBuffer: command)
        try gpu.finish(command)
        let pixels = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: records.count)
        for (index, reference) in expected.flatMap({ $0 }).enumerated() {
            #expect(isNearly(pixels[index], reference, tolerance: 0.00003), "Field \(index): \(pixels[index])")
        }
    }
}
