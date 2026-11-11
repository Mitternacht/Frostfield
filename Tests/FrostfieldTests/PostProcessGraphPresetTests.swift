import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessGraphPresetTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    private static func frame(_ x: Int, _ y: Int) -> SIMD4<Float> {
        let alpha: Float = (x + 2 * y) % 4 == 0 ? 0.5 : 1
        let straight = SIMD3(Float(x % 5) / 4, Float(y % 4) / 3, Float((x + y) % 7) / 6) * 0.9 + 0.05
        return SIMD4(straight * alpha, alpha)
    }

    private func apply(
        _ graph: PostProcessGraph, width: Int = 10, height: Int = 8, configure: (ShaderUniforms) throws -> Void = { _ in },
        pixel: @escaping (Int, Int) -> SIMD4<Float> = PostProcessGraphPresetTests.frame
    ) throws -> [SIMD4<Float>] {
        let pass = try PostProcessPass(graph: graph, device: gpu.device)
        try configure(pass.uniforms)
        let source = try gpu.color(width: width, height: height, pixel: pixel)
        let destination = try gpu.color(width: width, height: height)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        return try gpu.read(destination)
    }

    private func expectEach(
        _ pixels: [SIMD4<Float>], width: Int = 10, tolerance: Float = 2e-3, sourceLocation: SourceLocation = #_sourceLocation,
        _ expected: (Int, Int, SIMD4<Float>) -> SIMD4<Float>
    ) {
        for (index, pixel) in pixels.enumerated() {
            let (x, y) = (index % width, index / width)
            let want = expected(x, y, Self.frame(x, y))
            #expect(PostProcessFixture.close(pixel, simd_clamp(want, .zero, SIMD4(repeating: 65504)), tolerance: tolerance),
                    "(\(x), \(y)): \(pixel) ≠ \(want)", sourceLocation: sourceLocation)
        }
    }

    private static func straight(_ c: SIMD4<Float>) -> SIMD3<Float> { SIMD3(c.x, c.y, c.z) / c.w }

    @Test("Sepia applies the Three.js matrix with straight channels clamped to one")
    func sepia() throws {
        let pixels = try apply(try .sepia(amount: 0.8))
        expectEach(pixels) { _, _, c in
            let a: Float = 0.8
            let rgb = SIMD3(c.x, c.y, c.z)
            let r = simd_dot(rgb, SIMD3(1 - 0.607 * a, 0.769 * a, 0.189 * a))
            let g = simd_dot(rgb, SIMD3(0.349 * a, 1 - 0.314 * a, 0.168 * a))
            let b = simd_dot(rgb, SIMD3(0.272 * a, 0.534 * a, 1 - 0.869 * a))
            return SIMD4(min(r, c.w), min(g, c.w), min(b, c.w), c.w)
        }
    }

    @Test("Hue rotation and saturation follow HueSaturationShader", arguments: [
        SIMD2<Float>(0.3, 0.4), SIMD2(-0.6, -0.5), SIMD2(1, 0),
    ])
    func hueSaturation(values: SIMD2<Float>) throws {
        let pixels = try apply(try .hueSaturation(hue: values.x, saturation: values.y))
        expectEach(pixels) { _, _, c in
            let angle = values.x * .pi
            let (s, co) = (sin(angle), cos(angle))
            let w = (SIMD3(2 * co, -Float(3).squareRoot() * s - co, Float(3).squareRoot() * s - co) + 1) / 3
            let rgb = SIMD3(c.x, c.y, c.z)
            var rotated = SIMD3(simd_dot(rgb, w), simd_dot(rgb, SIMD3(w.z, w.x, w.y)), simd_dot(rgb, SIMD3(w.y, w.z, w.x)))
            let average = (rotated.x + rotated.y + rotated.z) / 3
            let factor = values.y > 0 ? 1 - 1 / (1.001 - values.y) : -values.y
            rotated += (SIMD3(repeating: average) - rotated) * factor
            return SIMD4(rotated, c.w)
        }
    }

    @Test("Brightness, contrast, color correction and vignette act on straight color")
    func straightColorFilters() throws {
        expectEach(try apply(try .brightnessContrast(brightness: 0.1, contrast: 0.3))) { _, _, c in
            let s = (Self.straight(c) + 0.1 - 0.5) / 0.7 + 0.5
            return SIMD4(s * c.w, c.w)
        }
        expectEach(try apply(try .brightnessContrast(brightness: -0.05, contrast: -0.4))) { _, _, c in
            let s = (Self.straight(c) - 0.05 - 0.5) * 0.6 + 0.5
            return SIMD4(s * c.w, c.w)
        }
        let power = SIMD3<Float>(1.5, 2, 0.8)
        let multiply = SIMD3<Float>(1.1, 0.9, 1)
        let add = SIMD3<Float>(0.05, 0, 0.1)
        expectEach(try apply(try .colorCorrection(power: power, multiply: multiply, add: add))) { _, _, c in
            let s = Self.straight(c) + add
            return SIMD4(multiply * SIMD3(pow(s.x, power.x), pow(s.y, power.y), pow(s.z, power.z)) * c.w, c.w)
        }
        expectEach(try apply(try .vignette(offset: 1.2, darkness: 0.7))) { x, y, c in
            let uv = (SIMD2(Float(x) + 0.5, Float(y) + 0.5) / SIMD2(10, 8) - 0.5) * 1.2
            let s = simd_mix(Self.straight(c), SIMD3(repeating: 0.3), SIMD3(repeating: simd_dot(uv, uv)))
            return SIMD4(s * c.w, c.w)
        }
    }

    @Test("RGB shift reads red and blue whole texels apart, and pixelate repeats each block's center texel")
    func shiftAndPixelate() throws {
        // Two texels right on screen: amount 0.2 of a 10-pixel frame at angle 0.
        expectEach(try apply(try .rgbShift(amount: 0.2, angle: 0))) { x, y, c in
            let red = Self.frame(min(x + 2, 9), y)
            let blue = Self.frame(max(x - 2, 0), y)
            return SIMD4(red.x, c.y, blue.z, c.w)
        }
        // Angle π/2 moves red up on screen, as in Three.js.
        expectEach(try apply(try .rgbShift(amount: 0.25, angle: .pi / 2))) { x, y, c in
            SIMD4(Self.frame(x, max(y - 2, 0)).x, c.y, Self.frame(x, min(y + 2, 7)).z, c.w)
        }
        let pixelated = try apply(try .pixelate(size: 3))
        expectEach(pixelated) { x, y, _ in Self.frame(min((x / 3) * 3 + 1, 9), min((y / 3) * 3 + 1, 7)) }
    }

    @Test("Sobel returns four times the luminance step on both sides of an edge")
    func sobel() throws {
        let pixels = try apply(try .sobel(), pixel: { x, _ in x < 5 ? SIMD4(0.2, 0.2, 0.2, 1) : SIMD4(0.7, 0.7, 0.7, 1) })
        for y in 0..<8 {
            for x in 0..<10 {
                let expected: Float = x == 4 || x == 5 ? 4 * 0.5 : 0
                #expect(PostProcessFixture.close(pixels[y * 10 + x], SIMD4(expected, expected, expected, 1), tolerance: 3e-3))
            }
        }
    }

    @Test("Frei–Chen returns opaque black on uniform bright and zero-energy inputs")
    func freiChenConstant() throws {
        let graph = try PostProcessGraph.freiChen()
        let colors: [SIMD4<Float>] = [.zero, SIMD4(0.002, 0.004, 0.001, 0.25), SIMD4(3, 6, 9, 1)]
        for color in colors {
            let pixels = try apply(graph, width: 7, height: 5, pixel: { _, _ in color })
            #expect(pixels.allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
        }
        let onePixel = try apply(graph, width: 1, height: 1, pixel: { _, _ in SIMD4(2, 4, 3, 1) })
        #expect(onePixel == [SIMD4(0, 0, 0, 1)])
    }

    @Test("Frei–Chen matches the nine pinned masks on horizontal, vertical, diagonal and chromatic edges")
    func freiChenReference() throws {
        let graph = try PostProcessGraph.freiChen()
        let width = 11, height = 7
        let dark = SIMD4<Float>(0.125, 0, 0.0625, 0.5)
        let bright = SIMD4<Float>(0.625, 1.25, 0.25, 1)
        let frames: [(Int, Int) -> SIMD4<Float>] = [
            { x, _ in x < 5 ? dark : bright },
            { _, y in y < 3 ? dark : bright },
            { x, y in 2 * x < y + 4 ? dark : bright },
            Self.frame
        ]
        for frame in frames {
            let pixels = try apply(graph, width: width, height: height, pixel: frame)
            #expect(pixels.contains { $0.x > 0.3 })
            for (index, actual) in pixels.enumerated() {
                let x = index % width, y = index / width
                let edge = Self.referenceFreiChen(x: x, y: y, width: width, height: height, pixel: frame)
                let expected = SIMD4(edge, edge, edge, 1)
                #expect(PostProcessFixture.close(actual, expected, tolerance: 1e-3),
                        "(\(x), \(y)): \(actual) ≠ \(expected)")
            }
        }

        // Equal RGB lengths on either side of a hue change do not create a
        // response. This would produce an edge if the pass used luminance.
        let equalLength = try apply(graph, width: width, height: height, pixel: { x, _ in
            x < 5 ? SIMD4(0.5, 0, 0, 0.5) : SIMD4(0, 0.5, 0, 1)
        })
        #expect(equalLength.allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
    }

    /// r187's column-major GLSL reference, using every mask in the denominator.
    private static func referenceFreiChen(
        x: Int, y: Int, width: Int, height: Int, pixel: (Int, Int) -> SIMD4<Float>
    ) -> Float {
        let a = 0.3535533845424652
        let sixth = 0.1666666716337204
        let third = 0.3333333432674408
        let twoThirds = 0.6666666865348816
        let masks: [[Double]] = [
            [a, 0, -a, 0.5, 0, -0.5, a, 0, -a],
            [a, 0.5, a, 0, 0, 0, -a, -0.5, -a],
            [0, a, -0.5, -a, 0, a, 0.5, -a, 0],
            [0.5, -a, 0, -a, 0, a, 0, a, -0.5],
            [0, -0.5, 0, 0.5, 0, 0.5, 0, -0.5, 0],
            [-0.5, 0, 0.5, 0, 0, 0, 0.5, 0, -0.5],
            [sixth, -third, sixth, -third, twoThirds, -third, sixth, -third, sixth],
            [-third, sixth, -third, sixth, twoThirds, sixth, -third, sixth, -third],
            Array(repeating: third, count: 9)
        ]
        var neighborhood: [Double] = []
        for dx in -1...1 {
            for dy in -1...1 {
                let value = pixel(min(max(x + dx, 0), width - 1), min(max(y - dy, 0), height - 1))
                let rgb = SIMD3(Double(Float16(value.x)), Double(Float16(value.y)), Double(Float16(value.z)))
                neighborhood.append(simd_length(rgb))
            }
        }
        let response = masks.map { mask in
            let convolution = zip(mask, neighborhood).reduce(0) { $0 + $1.0 * $1.1 }
            return convolution * convolution
        }
        let energy = response.reduce(0, +)
        return energy > 0 ? Float(sqrt(response.prefix(4).reduce(0, +) / energy)) : 0
    }

    @Test("Dot screen matches DotScreenShader's pattern over a bottom-left texture grid")
    func dotScreen() throws {
        let pixels = try apply(try .dotScreen(center: SIMD2(0.5, 0.5), angle: 1.57, scale: 1))
        expectEach(pixels, tolerance: 4e-3) { x, y, c in
            let uv = SIMD2((Float(x) + 0.5) / 10, 1 - (Float(y) + 0.5) / 8)
            let texel = uv * 256 - 0.5
            let (s, co) = (sin(Float(1.57)), cos(Float(1.57)))
            let point = SIMD2(co * texel.x - s * texel.y, s * texel.x + co * texel.y)
            let pattern = sin(point.x) * sin(point.y) * 4
            // DotScreenShader works on straight color; the result is premultiplied by alpha.
            let gray = ((c.x + c.y + c.z) / (3 * c.w) * 10 - 5 + pattern) * c.w
            return SIMD4(max(gray, 0), max(gray, 0), max(gray, 0), c.w)
        }
    }

    @Test("Halftone dots cover each channel's value as a fraction of the cell area")
    func halftone() throws {
        let width = 96
        let pixels = try apply(try .halftone(radius: 4), width: width, height: width, pixel: { _, _ in SIMD4(0.25, 0.5, 0.1, 1) })
        gpu.savePNG(pixels, width: width, height: width, name: "halftone")
        let mean = sum(pixels) / Double(pixels.count)
        #expect(abs(mean.x - 0.25) < 0.02 && abs(mean.y - 0.5) < 0.03 && abs(mean.z - 0.1) < 0.02, "\(mean)")
        #expect(pixels.contains { $0.x > 0.99 } && pixels.contains { $0.x < 0.01 })

        // Dots follow the straight color and are premultiplied once by the pixel's alpha.
        let translucent = try apply(
            try .halftone(radius: 4), width: width, height: width, pixel: { _, _ in SIMD4(0.125, 0.25, 0.05, 0.5) }
        )
        let translucentMean = sum(translucent) / Double(translucent.count)
        #expect(abs(translucentMean.x - 0.125) < 0.01 && abs(translucentMean.y - 0.25) < 0.015
                    && abs(translucentMean.z - 0.05) < 0.01, "\(translucentMean)")
        #expect(translucent.allSatisfy { $0.w == 0.5 && $0.x <= 0.5 + 1e-3 })
        #expect(translucent.contains { $0.x > 0.499 })
    }

    @Test("Film grain with zero intensity passes through; grayscale uses Three.js luminance")
    func film() throws {
        expectEach(try apply(try .film(intensity: 0))) { _, _, c in c }
        expectEach(try apply(try .film(intensity: 0, grayscale: true))) { _, _, c in
            let l = simd_dot(SIMD3(c.x, c.y, c.z), SIMD3(0.2125, 0.7154, 0.0721))
            return SIMD4(l, l, l, c.w)
        }
        let grained = try apply(try .film(intensity: 1), pixel: { _, _ in SIMD4(0.5, 0.5, 0.5, 1) })
        let lowest: Float = 0.55 - 2e-3
        let highest: Float = 1 + 2e-3
        #expect(grained.allSatisfy { (pixel: SIMD4<Float>) in pixel.x >= lowest && pixel.x <= highest })
        #expect(Set(grained.map { (pixel: SIMD4<Float>) in pixel.x }).count > 10)
    }

    @Test("Glitch passes through when bypassed or still, and is deterministic for given uniforms")
    func glitch() throws {
        let graph = try PostProcessGraph.glitch()
        expectEach(try apply(graph) { try $0.set("Bypass", to: Float(1)) }) { _, _, c in c }
        expectEach(try apply(graph) { u in
            for name in ["Amount", "SeedX", "SeedY", "ColumnSize"] { try u.set(name, to: Float(0)) }
        }) { _, _, c in c }
        var randomizer = GlitchRandomizer(seed: 42, goWild: true)
        let uniforms = try ShaderUniforms(graph: graph)
        try randomizer.advance(uniforms)
        let values = uniforms.snapshot().values
        let first = try apply(graph) { u in
            var replay = GlitchRandomizer(seed: 42, goWild: true)
            try replay.advance(u)
        }
        let second = try apply(graph) { u in
            var replay = GlitchRandomizer(seed: 42, goWild: true)
            try replay.advance(u)
            #expect(u.snapshot().values == values)
        }
        #expect(first == second)
        #expect(first != (0..<80).map { Self.frame($0 % 10, $0 / 10) })
    }

    @Test("The luminosity high pass keeps only light above the threshold")
    func highPass() throws {
        expectEach(try apply(try .luminosityHighPass(threshold: 0.3, smoothWidth: 0.2))) { _, _, c in
            let l = simd_dot(SIMD3(c.x, c.y, c.z), SIMD3(0.2126, 0.7152, 0.0722))
            let t = min(max((l - 0.3) / 0.2, 0), 1)
            return c * (t * t * (3 - 2 * t))
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessChainEffectTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Effect stages compose with graph stages and match the standalone passes")
    func composition() throws {
        let blur = try GaussianBlur(device: gpu.device, settings: .init(sigma: 1.5))
        let gain = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * p.uniform("Gain", Float(2)) }
        let chain = try PostProcessChain(
            queue: gpu.queue, destinationPixelFormat: .rgba16Float, effects: [.pass(blur), .graph(gain)]
        )
        let source = try gpu.color(width: 12, height: 9) { x, y in SIMD4(Float(x) / 11, Float(y) / 8, 0.5, 1) }
        let destination = try gpu.color(width: 12, height: 9)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }

        let blurred = try gpu.color(width: 12, height: 9)
        let gained = try gpu.color(width: 12, height: 9)
        let standaloneGain = try PostProcessPass(graph: gain, device: gpu.device)
        try gpu.run { buffer in
            try blur.encode(source: source, destination: blurred, commandBuffer: buffer)
            try standaloneGain.encode(source: blurred, destination: gained, commandBuffer: buffer)
        }
        #expect(try gpu.read(destination) == gpu.read(gained))

        // An sRGB display destination prepares the effect for that format too.
        let display = try PostProcessChain(queue: gpu.queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.pass(blur)])
        let target = try gpu.color(width: 12, height: 9, pixelFormat: .bgra8Unorm_srgb)
        try gpu.run { try display.encode(source: source, destination: target, commandBuffer: $0) }
    }

    @Test("Depth-reading effects get the chain's depth or report their stage")
    func depthForwarding() throws {
        let rays = try GodRays(device: gpu.device, settings: .init(
            lightPosition: SIMD2(0.5, 0.5), color: .one, intensity: 1, source: .sky(threshold: 0)
        ))
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(try .vignette(strength: 0)), .pass(rays)])
        let source = try gpu.color(width: 8, height: 8) { _, _ in SIMD4(0.25, 0.25, 0.25, 1) }
        let destination = try gpu.color(width: 8, height: 8)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessChain.Failure.missingDepth(stage: 1)) {
            try chain.encode(source: source, destination: destination, commandBuffer: buffer)
        }
        let depth = try gpu.depth(width: 8, height: 8) { _, _ in 1 }
        try gpu.run { try chain.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination).allSatisfy { PostProcessFixture.close($0, SIMD4(0.5, 0.5, 0.5, 1)) })
        #expect(throws: PostProcessChain.Failure.scaledDepth(stage: 1)) { try chain.setResolutionScale(0.5, at: 1) }
    }

    @Test("A scaled stage runs on a box-filtered half-size copy and is upsampled for the next stage")
    func resolutionScale() throws {
        let gain = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * p.uniform("Gain", Float(3)) }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(gain), .graph(try .vignette(strength: 0))])
        try chain.setResolutionScale(0.5, at: 0)
        #expect(try chain.resolutionScale(at: 0) == 0.5)
        #expect(throws: PostProcessChain.Failure.resolutionScale) { try chain.setResolutionScale(0, at: 0) }
        // Two-pixel stripes average to their mean at half size; uniform rows stay exact.
        let source = try gpu.color(width: 16, height: 8) { x, _ in x % 2 == 0 ? SIMD4(0.1, 0.2, 0.3, 1) : SIMD4(0.3, 0.2, 0.1, 1) }
        let destination = try gpu.color(width: 16, height: 8)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination).allSatisfy { PostProcessFixture.close($0, SIMD4(0.6, 0.6, 0.6, 1)) })
        // One full-size intermediate plus the scaled pair.
        #expect(chain.intermediateAllocationCount == 3)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(chain.intermediateAllocationCount == 3)
    }

    @Test("Scaled stages average each reduced texel's exact footprint at any scale")
    func resolutionScaleFootprint() throws {
        let copy = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).rgb * p.uniform("Gain", Float(1)) }
        // At 1/8, a single highlight keeps its energy wherever it falls in its 8 × 8 block.
        let eighth = try PostProcessChain(queue: gpu.queue, effects: [.graph(copy)])
        try eighth.setResolutionScale(0.125, at: 0)
        let destination = try gpu.color(width: 64, height: 64)
        for (dx, dy) in [(0, 0), (3, 5), (7, 7), (1, 6), (5, 2)] {
            let source = try gpu.color(width: 64, height: 64) { x, y in x == 24 + dx && y == 32 + dy ? SIMD4(64, 64, 64, 1) : .zero }
            try gpu.run { try eighth.encode(source: source, destination: destination, commandBuffer: $0) }
            let total = sum(try gpu.read(destination))
            #expect(abs(total.x - 64) < 0.2, "offset (\(dx), \(dy)): \(total.x)")
        }
        // Stripes that a sparse tap pattern would miss average to their exact mean.
        let stripes = try gpu.color(width: 64, height: 8) { x, _ in
            [0, 3, 4, 7].contains(x % 8) ? SIMD4(1, 1, 1, 1) : SIMD4(0, 0, 0, 1)
        }
        let striped = try gpu.color(width: 64, height: 8)
        try gpu.run { try eighth.encode(source: stripes, destination: striped, commandBuffer: $0) }
        #expect(try gpu.read(striped).allSatisfy { $0 == SIMD4(0.5, 0.5, 0.5, 1) })

        // A non-power-of-two scale matches a CPU area average followed by bilinear upsampling.
        let (width, height) = (40, 24)
        let scaled = try PostProcessChain(queue: gpu.queue, effects: [.graph(copy)])
        try scaled.setResolutionScale(0.3, at: 0)
        func value(_ x: Int, _ y: Int) -> Float { Float((x * 7919 + y * 104_729) % 13 < 6 ? 1 : 0) }
        let noise = try gpu.color(width: width, height: height) { x, y in SIMD4(value(x, y), 0, 0, 1) }
        let output = try gpu.color(width: width, height: height)
        try gpu.run { try scaled.encode(source: noise, destination: output, commandBuffer: $0) }
        let (lowWidth, lowHeight) = (12, 7)  // round(40 · 0.3), round(24 · 0.3)
        let sx = Double(width) / Double(lowWidth)
        let sy = Double(height) / Double(lowHeight)
        var low = [Double](repeating: 0, count: lowWidth * lowHeight)
        for j in 0..<lowHeight {
            for i in 0..<lowWidth {
                var total = 0.0
                for y in 0..<height {
                    let wy = max(0, min(Double(j + 1) * sy, Double(y + 1)) - max(Double(j) * sy, Double(y)))
                    guard wy > 0 else { continue }
                    for x in 0..<width {
                        let wx = max(0, min(Double(i + 1) * sx, Double(x + 1)) - max(Double(i) * sx, Double(x)))
                        total += wx * wy * Double(value(x, y))
                    }
                }
                low[j * lowWidth + i] = total / (sx * sy)
            }
        }
        func texel(_ i: Int, _ j: Int) -> Double { low[min(max(j, 0), lowHeight - 1) * lowWidth + min(max(i, 0), lowWidth - 1)] }
        let pixels = try gpu.read(output)
        var worst = 0.0
        for y in 0..<height {
            for x in 0..<width {
                let u = (Double(x) + 0.5) / sx - 0.5
                let v = (Double(y) + 0.5) / sy - 0.5
                let (i, j) = (Int(u.rounded(.down)), Int(v.rounded(.down)))
                let (fu, fv) = (u - Double(i), v - Double(j))
                let top = texel(i, j) * (1 - fu) + texel(i + 1, j) * fu
                let bottom = texel(i, j + 1) * (1 - fu) + texel(i + 1, j + 1) * fu
                worst = max(worst, abs(Double(pixels[y * width + x].x) - (top * (1 - fv) + bottom * fv)))
            }
        }
        #expect(worst < 8e-3, "largest difference \(worst)")
    }
}
