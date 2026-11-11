import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct FSR1UpscalerTests {
    private let gpu: PostProcessFixture

    init() throws { gpu = try PostProcessFixture() }

    @Test("EASU matches a 12-tap HDR diagonal-edge reference with clamped frame boundaries")
    func easuReference() throws {
        let size = SIMD2(7, 5), outputSize = SIMD2(13, 9)
        let dark = SIMD4<Float>(0.125, 0.25, 0.375, 0.25)
        let bright = SIMD4<Float>(2.5, 1.25, 0.625, 0.75)
        let source = try gpu.color(width: size.x, height: size.y) { x, y in
            if x == 0 && y == 0 { return .zero }
            return x <= y + 1 ? dark : bright
        }
        let destination = try gpu.color(width: outputSize.x, height: outputSize.y)
        let fsr = try FSR1Upscaler(queue: gpu.queue, sharpening: nil)
        try gpu.run { try fsr.encode(source: source, destination: destination, commandBuffer: $0) }
        let input = try gpu.read(source)
        let actual = try gpu.read(destination)
        for (index, pixel) in actual.enumerated() {
            let xy = SIMD2(index % outputSize.x, index / outputSize.x)
            let expected = easu(input, inputSize: size, outputSize: outputSize, pixel: xy)
            #expect(PostProcessFixture.close(pixel, expected, tolerance: 0.004), "pixel \(xy): \(pixel) vs \(expected)")
        }
        #expect(actual.contains { $0.x > 1 })
        #expect(actual.contains { $0.w > dark.w && $0.w < bright.w })
    }

    @Test("Standalone RCAS follows SDR and HDR lobe limits and retains coverage")
    func rcasReference() throws {
        let size = SIMD2(8, 5)
        let source = try gpu.color(width: size.x, height: size.y) { x, y in
            let alpha: Float = (x == 0 && y == 0) ? 0 : (x + y).isMultiple(of: 3) ? 0.5 : 1
            var signal = 0.32 + Float((x * 3 + y * 2) % 5) / 16
            if x == 3 && y == 2 { signal = 0.7 }
            if x >= 5 { signal *= 5 }
            return SIMD4(signal * alpha, signal * 0.8 * alpha, signal * 0.5 * alpha, alpha)
        }
        let destination = try gpu.color(width: size.x, height: size.y)
        let input = try gpu.read(source)
        let pass = try RCASSharpen(device: gpu.device)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(pass)])
        let prepared = pass.builtPipelineCount
        var center: [Float] = []
        for settings: RCASSharpen.Settings in [.init(sharpness: 0), .init(sharpness: 0, denoise: true), .init(sharpness: 2)] {
            pass.settings = settings
            try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
            let actual = try gpu.read(destination)
            for (index, pixel) in actual.enumerated() {
                let xy = SIMD2(index % size.x, index / size.x)
                let expected = rcas(input, size: size, pixel: xy, settings: settings)
                #expect(PostProcessFixture.close(pixel, expected, tolerance: 0.004), "pixel \(xy): \(pixel) vs \(expected)")
                #expect(pixel.w == input[index].w)
            }
            #expect(actual[0] == .zero)
            #expect(actual.contains { $0.x > 1 })
            center.append(actual[2 * size.x + 3].x)
        }
        #expect(center[0] > center[1] + 0.001)
        #expect(center[0] > center[2] + 0.005)
        #expect(pass.builtPipelineCount == prepared)
    }

    @Test("Combined and standalone stages agree and constant HDR remains exact across resize")
    func compositionAndValidation() throws {
        let constant = SIMD4<Float>(6, 3, 1.5, 0.5)
        let source = try gpu.color(width: 5, height: 3) { _, _ in constant }
        let first = try gpu.color(width: 10, height: 6)
        let second = try gpu.color(width: 11, height: 7)
        let fsr = try FSR1Upscaler(queue: gpu.queue)
        try gpu.run {
            try fsr.encode(source: source, destination: first, commandBuffer: $0)
            try fsr.encode(source: source, destination: second, commandBuffer: $0)
        }
        #expect(try gpu.read(first).allSatisfy { $0 == constant })
        #expect(try gpu.read(second).allSatisfy { $0 == constant })

        let varied = try gpu.color(width: 5, height: 3) { x, y in SIMD4(Float(x + y + 1) / 8, 0.25, 0.125, 1) }
        let staged = try gpu.color(width: 10, height: 6)
        let expected = try gpu.color(width: 10, height: 6)
        let rcas = try RCASSharpen(device: gpu.device)
        try gpu.run {
            try fsr.encode(source: varied, destination: first, commandBuffer: $0)
            fsr.sharpening = nil
            try fsr.encode(source: varied, destination: staged, commandBuffer: $0)
            try rcas.encode(source: staged, destination: expected, commandBuffer: $0)
        }
        #expect(try gpu.read(first) == gpu.read(expected))
        let copy = try gpu.color(width: 5, height: 3)
        try gpu.run { try fsr.encode(source: varied, destination: copy, commandBuffer: $0) }
        #expect(try gpu.read(copy) == gpu.read(varied))

        let command = try #require(gpu.queue.makeCommandBuffer())
        let foreignQueue = try #require(gpu.device.makeCommandQueue())
        let foreign = try #require(foreignQueue.makeCommandBuffer())
        let tooSmall = try gpu.color(width: 4, height: 3)
        #expect(throws: PostProcessEffectError.commandBuffer) {
            try fsr.encode(source: source, destination: first, commandBuffer: foreign)
        }
        #expect(throws: PostProcessEffectError.destination) {
            try fsr.encode(source: source, destination: source, commandBuffer: command)
        }
        #expect(throws: PostProcessEffectError.sizeMismatch) {
            try fsr.encode(source: source, destination: tooSmall, commandBuffer: command)
        }
        fsr.sharpening = .init(sharpness: .nan)
        #expect(throws: PostProcessEffectError.self) {
            try fsr.encode(source: source, destination: first, commandBuffer: command)
        }
    }

    @Test("Spatial FSR consumes a transparent HDR path frame on the trace command buffer")
    func pathFrame() throws {
        let inputSize = SIMD2(17, 13), outputSize = SIMD2(31, 25)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let card = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1.45, height: 1.1),
            material: .physicallyBased(.init(
                emissiveFactor: SIMD3(1, 0.5, 0.25), alphaMode: .opaque, emissiveStrength: 4
            ))
        )
        let placement = SceneTransform(
            translation: SIMD3(0, 0, -2), rotation: simd_quatf(angle: 0.23, axis: SIMD3(0, 0, 1))
        )
        let graph = SceneGraph(roots: [SceneNode(transform: placement, mesh: card)], camera: camera)
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: .init(environment: .zero, backgroundVisible: false))
        let history = try tracer.makeAccumulation(width: inputSize.x, height: inputSize.y)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: inputSize.x, height: inputSize.y, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.hazardTrackingMode = .tracked
        let trace = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let spatial = try gpu.color(width: outputSize.x, height: outputSize.y)
        let sharpened = try gpu.color(width: outputSize.x, height: outputSize.y)
        let fsr = try FSR1Upscaler(queue: gpu.queue, sharpening: nil)
        try gpu.run { command in
            _ = try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: trace, accumulation: history,
                commandBuffer: command, samplesPerPixel: 8, settings: .init(maxBounces: 1, seed: 8)
            )
            try fsr.encode(source: trace, destination: spatial, commandBuffer: command)
            fsr.sharpening = .init()
            try fsr.encode(source: trace, destination: sharpened, commandBuffer: command)
        }
        let source = try gpu.read(trace)
        let output = try gpu.read(spatial)
        let detail = try gpu.read(sharpened)
        #expect(source.contains { $0.w > 0 && $0.w < 1 })
        #expect(output.first == .zero && detail.first == .zero)
        #expect(output[output.count / 2] == SIMD4(4, 2, 1, 1))
        #expect(detail[detail.count / 2] == output[output.count / 2])
        for (index, pixel) in output.enumerated() {
            let xy = SIMD2(index % outputSize.x, index / outputSize.x)
            let expected = easu(source, inputSize: inputSize, outputSize: outputSize, pixel: xy)
            #expect(PostProcessFixture.close(pixel, expected, tolerance: 0.008), "pixel \(xy): \(pixel) vs \(expected)")
            #expect(detail[index].w == pixel.w)
        }
    }

    private func easu(
        _ input: [SIMD4<Float>], inputSize: SIMD2<Int>, outputSize: SIMD2<Int>, pixel: SIMD2<Int>
    ) -> SIMD4<Float> {
        let p = (SIMD2(Float(pixel.x), Float(pixel.y)) + 0.5)
            * SIMD2(Float(inputSize.x), Float(inputSize.y)) / SIMD2(Float(outputSize.x), Float(outputSize.y)) - 0.5
        let base = floor(p)
        let fraction = p - base
        func at(_ x: Int, _ y: Int) -> SIMD4<Float> {
            texel(input, size: inputSize, pixel: SIMD2(Int(base.x) + x, Int(base.y) + y))
        }
        var direction = SIMD2<Float>.zero
        var edge: Float = 0
        for y in 0...1 {
            for x in 0...1 {
                let weight = (x == 0 ? 1 - fraction.x : fraction.x) * (y == 0 ? 1 - fraction.y : fraction.y)
                let center = luma(at(x, y))
                let left = luma(at(x - 1, y)), right = luma(at(x + 1, y))
                let up = luma(at(x, y - 1)), down = luma(at(x, y + 1))
                let dx = right - left, dy = down - up
                let lx = min(abs(dx) / max(abs(right - center), abs(center - left), 1 / 65536), 1)
                let ly = min(abs(dy) / max(abs(down - center), abs(center - up), 1 / 65536), 1)
                direction += SIMD2(dx, dy) * weight
                edge += weight * (lx * lx + ly * ly)
            }
        }
        let d2 = simd_length_squared(direction)
        direction = d2 < 1 / 32768 ? SIMD2(1, direction.y) : direction / sqrt(d2)
        edge = pow(edge * 0.5, 2)
        let stretch = simd_length_squared(direction) / max(abs(direction.x), abs(direction.y))
        let scaleX = 1 + (stretch - 1) * edge, scaleY = 1 - 0.5 * edge
        let lobe = 0.5 - 0.29 * edge
        var color = SIMD4<Float>.zero
        var total: Float = 0
        for y in -1...2 {
            for x in -1...2 where (0...1).contains(x) || (0...1).contains(y) {
                let ox = Float(x) - fraction.x, oy = Float(y) - fraction.y
                let rx = (ox * direction.x + oy * direction.y) * scaleX
                let ry = (-ox * direction.y + oy * direction.x) * scaleY
                let distance = min(rx * rx + ry * ry, 1 / lobe)
                let weight = (pow(0.4 * distance - 1, 2) * 1.5625 - 0.5625) * pow(lobe * distance - 1, 2)
                color += at(x, y) * weight
                total += weight
            }
        }
        let core = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)]
        let bounded = simd_clamp(color / total, core.reduce(core[0], simd_min), core.reduce(core[0], simd_max))
        return bounded.w > 0 ? bounded : .zero
    }

    private func rcas(
        _ input: [SIMD4<Float>], size: SIMD2<Int>, pixel: SIMD2<Int>, settings: RCASSharpen.Settings
    ) -> SIMD4<Float> {
        let center = texel(input, size: size, pixel: pixel)
        guard center.w > 0 else { return .zero }
        let rgb = SIMD3(center.x, center.y, center.z) / center.w
        let ring = [SIMD2(0, -1), SIMD2(-1, 0), SIMD2(1, 0), SIMD2(0, 1)].map { offset in
            let c = texel(input, size: size, pixel: pixel &+ offset)
            return c.w > 0 ? SIMD3(c.x, c.y, c.z) / c.w : rgb
        }
        let samples = ring + [rgb]
        let isHDR = samples.contains { simd_reduce_max($0) > 1 }
        let probe = samples.map { isHDR ? $0 / (1 + simd_reduce_max($0)) : $0 }
        var limit: Float = -0.1875
        for channel in 0..<3 {
            let values = probe.prefix(4).map { $0[channel] }
            let low = values.min()!, high = values.max()!, middle = probe[4][channel]
            let bottom = high > 0 ? -min(low, middle) / (4 * high) : -0.25
            let top = low < 1 ? (1 - max(high, middle)) / (4 * low - 4) : -0.25
            limit = max(limit, bottom, top)
        }
        var weight = min(limit, 0) * exp2(-settings.sharpness)
        if settings.denoise {
            let levels = probe.map { 0.5 * $0.x + $0.y + 0.5 * $0.z }
            let noise = abs(levels.prefix(4).reduce(0, +) / 4 - levels[4])
            weight *= 1 - 0.5 * min(noise / max(levels.max()! - levels.min()!, 1 / 65536), 1)
        }
        let result = (rgb + ring.reduce(.zero, +) * weight) / (1 + 4 * weight)
        return SIMD4(simd_max(result, .zero) * center.w, center.w)
    }

    private func texel(_ input: [SIMD4<Float>], size: SIMD2<Int>, pixel: SIMD2<Int>) -> SIMD4<Float> {
        let x = min(max(pixel.x, 0), size.x - 1), y = min(max(pixel.y, 0), size.y - 1)
        return input[y * size.x + x]
    }

    private func luma(_ color: SIMD4<Float>) -> Float { 0.5 * color.x + color.y + 0.5 * color.z }
}
