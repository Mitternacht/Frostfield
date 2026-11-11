import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessHistoryTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    private static let lit = SIMD4<Float>(1, 0.5, 0.05, 1)

    private func frames(_ effect: any PostProcessEffect, count: Int) throws -> [SIMD4<Float>] {
        let first = try gpu.color(width: 4, height: 4) { x, y in x == 1 && y == 2 ? Self.lit : .zero }
        let empty = try gpu.color(width: 4, height: 4) { _, _ in .zero }
        let destination = try gpu.color(width: 4, height: 4)
        var results: [SIMD4<Float>] = []
        for frame in 0..<count {
            try gpu.run { try effect.encode(source: frame == 0 ? first : empty, depth: nil, destination: destination, commandBuffer: $0) }
            results.append(try gpu.read(destination)[2 * 4 + 1])
        }
        return results
    }

    @Test("Afterimage reproduces Three.js AfterimageShader: damped where above 0.1, then the maximum")
    func threeJSMaximum() throws {
        let effect = try Afterimage(device: gpu.device, mode: .maximum(damp: 0.96))
        let results = try frames(effect, count: 3)
        #expect(PostProcessFixture.close(results[0], Self.lit))
        // Blue 0.05 is not above 0.1, so it drops to zero; the rest decays by 0.96 per frame.
        #expect(PostProcessFixture.close(results[1], SIMD4(0.96, 0.48, 0, 0.96)))
        #expect(PostProcessFixture.close(results[2], SIMD4(0.9216, 0.4608, 0, 0.9216)))
    }

    @Test("Blend mode mixes with the previous output by the persistence factor, and reset starts over")
    func blend() throws {
        let effect = try Afterimage(device: gpu.device, mode: .blend(persistence: 0.25))
        let results = try frames(effect, count: 3)
        #expect(PostProcessFixture.close(results[0], Self.lit))
        #expect(PostProcessFixture.close(results[1], Self.lit * 0.25))
        #expect(PostProcessFixture.close(results[2], Self.lit * 0.0625))
        effect.reset()
        let empty = try gpu.color(width: 4, height: 4) { _, _ in .zero }
        let destination = try gpu.color(width: 4, height: 4)
        try gpu.run { try effect.encode(source: empty, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination).allSatisfy { $0 == .zero })
    }

    @Test("A history stage feeds the previous frame to a graph in the same chain")
    func graphHistory() throws {
        let history = try PostProcessHistory(queue: gpu.queue, width: 4, height: 4)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [
            .graph(try .motionTrail(persistence: 0.5), textures: ["Previous": history.texture]),
            .pass(history),
        ])
        let first = try gpu.color(width: 4, height: 4) { x, _ in x == 3 ? SIMD4(2, 1, 0.5, 1) : .zero }
        let empty = try gpu.color(width: 4, height: 4) { _, _ in .zero }
        let destination = try gpu.color(width: 4, height: 4)
        var trail: [SIMD4<Float>] = []
        for frame in 0..<3 {
            try gpu.run { try chain.encode(source: frame == 0 ? first : empty, destination: destination, commandBuffer: $0) }
            trail.append(try gpu.read(destination)[3])
        }
        #expect(PostProcessFixture.close(trail[0], SIMD4(1, 0.5, 0.25, 0.5)))
        #expect(PostProcessFixture.close(trail[1], SIMD4(0.5, 0.25, 0.125, 0.25)))
        #expect(PostProcessFixture.close(trail[2], SIMD4(0.25, 0.125, 0.0625, 0.125)))
        #expect(history.capturedFrameCount == 3)
        // Several frames in one command buffer see each other in order.
        try history.reset()
        try gpu.run { buffer in
            try chain.encode(source: first, destination: destination, commandBuffer: buffer)
            try chain.encode(source: empty, destination: destination, commandBuffer: buffer)
        }
        #expect(PostProcessFixture.close(try gpu.read(destination)[3], SIMD4(0.5, 0.25, 0.125, 0.25)))
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SupersampleAccumulatorTests {
    @Test("Jitter patterns are the standard positions, centered and inside the pixel")
    func patterns() throws {
        #expect(try SupersampleAccumulator.jitterOffsets(sampleCount: 4) == [
            SIMD2(-2, -6) / 16, SIMD2(6, -2) / 16, SIMD2(-6, 2) / 16, SIMD2(2, 6) / 16,
        ])
        for count in [1, 2, 3, 8, 16, 32, 100] {
            let offsets = try SupersampleAccumulator.jitterOffsets(sampleCount: count)
            #expect(offsets.count == count)
            #expect(offsets.allSatisfy { simd_reduce_max(abs($0)) <= 0.5 })
            #expect(Set(offsets.map { "\($0)" }).count == count)
        }
        #expect(throws: PostProcessEffectError.self) { try SupersampleAccumulator.jitterOffsets(sampleCount: 0) }
    }

    @Test("Accumulated jittered subframes give each edge pixel its exact sample coverage", arguments: [4, 16, 32])
    func coverage(sampleCount: Int) throws {
        let gpu = try PostProcessFixture()
        let width = 16
        let height = 8
        let library = try gpu.device.makeLibrary(source: Self.source, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "test_ssaa_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "test_ssaa_fragment")
        descriptor.colorAttachments[0].pixelFormat = .rgba16Float
        let pipeline = try gpu.device.makeRenderPipelineState(descriptor: descriptor)
        // Covered region: pixel x < 7.3 and pixel y < 2.6, as clip-space half planes.
        let edgeX = Float(7.3) / Float(width) * 2 - 1
        let edgeY = 1 - Float(2.6) / Float(height) * 2
        let accumulator = try SupersampleAccumulator(device: gpu.device)
        let destination = try gpu.color(width: width, height: height)
        try gpu.run { buffer in
            try accumulator.encode(sampleCount: sampleCount, destination: destination, commandBuffer: buffer) { _, jitter, target in
                var clip = SupersampleAccumulator.jittered(matrix_identity_float4x4, offset: jitter, width: width, height: height)
                var edges = SIMD2(edgeX, edgeY)
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = target
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
                pass.colorAttachments[0].storeAction = .store
                let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
                encoder.setRenderPipelineState(pipeline)
                encoder.setVertexBytes(&clip, length: MemoryLayout<simd_float4x4>.size, index: 0)
                encoder.setVertexBytes(&edges, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                encoder.endEncoding()
            }
        }
        let pixels = try gpu.read(destination)
        let offsets = try SupersampleAccumulator.jitterOffsets(sampleCount: sampleCount)
        for y in 0..<height {
            for x in 0..<width {
                let covered = offsets.filter { Float(x) + 0.5 + $0.x < 7.3 && Float(y) + 0.5 + $0.y < 2.6 }.count
                let expected = Float(covered) / Float(sampleCount)
                #expect(abs(pixels[y * width + x].x - expected) < 2e-3, "(\(x), \(y)) \(pixels[y * width + x].x) vs \(expected)")
            }
        }
    }

    @Test("The mean is exact at every sample count, not darkened by half-float running sums",
          arguments: [7, 16, 64, 100, 256])
    func exactMean(sampleCount: Int) throws {
        let gpu = try PostProcessFixture()
        let accumulator = try SupersampleAccumulator(device: gpu.device)
        let destination = try gpu.color(width: 4, height: 4)
        func clear(_ target: any MTLTexture, _ color: SIMD4<Double>, on buffer: any MTLCommandBuffer) throws {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: color.x, green: color.y, blue: color.z, alpha: color.w)
            pass.colorAttachments[0].storeAction = .store
            let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
            encoder.endEncoding()
        }
        // Identical subframes: the mean is the subframe itself.
        let constant = SIMD4<Double>(0.7002, 0.3, 3.7, 0.05)
        try gpu.run { buffer in
            try accumulator.encode(sampleCount: sampleCount, destination: destination, commandBuffer: buffer) { _, _, target in
                try clear(target, constant, on: buffer)
            }
        }
        let expected = SIMD4<Float>(
            Float(Float16(constant.x)), Float(Float16(constant.y)), Float(Float16(constant.z)), Float(Float16(constant.w))
        )
        let means = try gpu.read(destination)
        #expect(means.allSatisfy { $0 == expected }, "\(means[0]) vs \(expected)")

        // Varying subframes: the half-float rounding of the exact mean, within one unit in the last place.
        func subframe(_ index: Int) -> SIMD4<Double> { SIMD4(0.1 + 0.013 * Double(index % 37), 2.5, 0.01 * Double(index % 5), 1) }
        try gpu.run { buffer in
            try accumulator.encode(sampleCount: sampleCount, destination: destination, commandBuffer: buffer) { index, _, target in
                try clear(target, subframe(index), on: buffer)
            }
        }
        var mean = SIMD4<Double>.zero
        for index in 0..<sampleCount {
            let value = subframe(index)
            mean += SIMD4(Double(Float16(value.x)), Double(Float16(value.y)), Double(Float16(value.z)), Double(Float16(value.w)))
        }
        mean /= Double(sampleCount)
        let pixel = try gpu.read(destination)[5]
        for channel in 0..<4 {
            let want = Float16(mean[channel])
            #expect(abs(Float(pixel[channel]) - Float(want)) <= Float(want.ulp), "channel \(channel): \(pixel[channel]) vs \(mean[channel])")
        }
    }

    /// Two triangles fill the rectangle left of `edges.x` and above `edges.y` in clip space.
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 test_ssaa_vertex(uint id [[vertex_id]], constant float4x4& clip [[buffer(0)]], constant float2& edges [[buffer(1)]]) {
        const float2 corners[] = {float2(-3, edges.y), float2(edges.x, edges.y), float2(edges.x, 3),
                                  float2(-3, edges.y), float2(edges.x, 3), float2(-3, 3)};
        return clip * float4(corners[id], 0.5, 1);
    }
    fragment float4 test_ssaa_fragment() { return float4(1); }
    """
}
