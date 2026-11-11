import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessTexturePassTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Texture replacement and HDR source-over preserve the authored alpha and orientation")
    func replaceAndBlend() throws {
        let incoming = SIMD4<Float>(0.5, 0.25, 1, 0.5)
        let overlay: [SIMD4<Float>] = [.zero, SIMD4(4, 0, 0, 0.5), SIMD4(0, 2, 0, 1), SIMD4(0, 0, 0.75, 0.25)]
        let source = try gpu.color(width: 2, height: 2) { _, _ in incoming }
        let layer = try gpu.color(width: 2, height: 2) { x, y in overlay[y * 2 + x] }
        let destination = try gpu.color(width: 2, height: 2)
        let depth = try gpu.depth(width: 2, height: 2) { x, y in x == y ? 0 : 1 }
        let pass = try PostProcessTexturePass(device: gpu.device, texture: layer)
        #expect(!pass.readsDepth)
        for opacity: Float in [1, 0.5, 0] {
            pass.opacity = opacity
            try gpu.run { try pass.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
            let expected = overlay.map { Self.compose($0, over: incoming, opacity: opacity) }
            #expect(try gpu.read(destination) == expected)
        }
    }

    @Test("Straight-alpha sRGB texels are linearized and premultiplied before bilinear resizing")
    func resampling() throws {
        let layer = try gpu.color(width: 2, height: 2, pixelFormat: .rgba8Unorm_srgb)
        let texels: [UInt8] = [
            255, 0, 0, 0, 0, 128, 0, 255,
            0, 0, 255, 255, 255, 0, 0, 0,
        ]
        texels.withUnsafeBytes { bytes in
            layer.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: 8)
        }
        let source = try gpu.color(width: 4, height: 4)
        let destination = try gpu.color(width: 4, height: 4)
        let pass = try PostProcessTexturePass(device: gpu.device, texture: layer, alphaMode: .straight)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        let result = try gpu.read(destination)
        let green = pow((Float(128) / 255 + 0.055) / 1.055, 2.4)
        let weights: [Float] = [0, 0.25, 0.75, 1]
        for y in 0..<4 {
            for x in 0..<4 {
                let top = weights[x] * (1 - weights[y])
                let bottom = (1 - weights[x]) * weights[y]
                let expected = SIMD4<Float>(0, green * top, bottom, top + bottom)
                #expect(PostProcessFixture.close(result[y * 4 + x], expected, tolerance: 0.0005))
            }
        }
    }

    @Test("A masked texture stage saves the full result and leaves depth available to later stages",
          arguments: [Float(1), 0.5])
    func maskAndSave(opacity: Float) throws {
        let width = 4, height = 2
        let incoming = SIMD4<Float>(2, 0.125, 0, 0.75)
        let overlay = SIMD4<Float>(0, 1.5, 0.25, 0.5)
        let source = try gpu.color(width: width, height: height) { _, _ in incoming }
        let layer = try gpu.color(width: 1, height: 1) { _, _ in overlay }
        let mask = try gpu.color(width: width, height: height) { x, y in
            SIMD4((x + y).isMultiple(of: 2) ? 1 : 0, 0, 0, 0)
        }
        let depth = try gpu.depth(width: width, height: height) { x, _ in x < 2 ? 0.25 : 0.75 }
        let saved = try gpu.color(width: width, height: height)
        let destination = try gpu.color(width: width, height: height)
        let pass = try PostProcessTexturePass(device: gpu.device, texture: layer, opacity: opacity)
        let depthGraph = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            return p.vector(color.rgb * p.sceneDepth(at: p.screenUV), color.a)
        }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(pass), .save(to: saved), .graph(depthGraph)])
        try chain.setMask(.init(texture: mask), stages: 0..<2)
        try gpu.run { try chain.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
        let selected = Self.compose(overlay, over: incoming, opacity: opacity)
        let expectedSave = (0..<(width * height)).map { index in
            (index % width + index / width).isMultiple(of: 2) ? selected : incoming
        }
        let expectedOutput = expectedSave.enumerated().map { index, pixel in
            let factor: Float = index % width < 2 ? 0.25 : 0.75
            return pixel * SIMD4(factor, factor, factor, 1)
        }
        #expect(try gpu.read(saved) == expectedSave)
        #expect(try gpu.read(destination) == expectedOutput)
    }

    @Test("Texture copy permits shared inputs and rejects destination aliases and invalid opacity")
    func aliasValidation() throws {
        let original = SIMD4<Float>(0.5, 1.25, 0.25, 0.5)
        let source = try gpu.color(width: 4, height: 4) { _, _ in original }
        let destination = try gpu.color(width: 4, height: 4)
        let pass = try PostProcessTexturePass(device: gpu.device, texture: source)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination) == Array(repeating: original, count: 16))

        let buffer = try #require(gpu.queue.makeCommandBuffer())
        pass.texture = destination
        #expect(throws: PostProcessEffectError.destination) {
            try pass.encode(source: source, destination: destination, commandBuffer: buffer)
        }
        let view = try #require(destination.makeTextureView(pixelFormat: .rgba16Float))
        pass.texture = view
        #expect(throws: PostProcessEffectError.destination) {
            try pass.encode(source: source, destination: destination, commandBuffer: buffer)
        }
        pass.texture = source
        pass.opacity = .infinity
        #expect(throws: PostProcessEffectError.invalidSettings("Texture pass opacity must be from 0 to 1.")) {
            try pass.encode(source: source, destination: destination, commandBuffer: buffer)
        }
    }

    private static func compose(_ overlay: SIMD4<Float>, over source: SIMD4<Float>, opacity: Float) -> SIMD4<Float> {
        if opacity == 1 { return overlay }
        let layer = overlay * opacity
        return layer + source * (1 - layer.w)
    }
}
