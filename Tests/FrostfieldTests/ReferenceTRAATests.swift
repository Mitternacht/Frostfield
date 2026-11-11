import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ReferenceTRAATests {
    private let gpu: PostProcessFixture

    init() throws { gpu = try PostProcessFixture() }

    @Test("TRAA velocity, subpixel, variance and luminance terms match the pinned equations")
    func referenceWeights() throws {
        let width = 9, height = 5
        let prior = try gpu.color(width: width, height: height) { _, _ in Self.gray(8) }
        let current = try gpu.color(width: width, height: height) { x, y in
            if x == 4 && y == 2 { return Self.gray(1) }
            return Self.gray((x + y).isMultiple(of: 2) ? 0 : 4)
        }
        let depth = try gpu.depth(width: width, height: height) { _, _ in 0.5 }
        let noHistory = try gpu.color(width: width, height: height) { _, _ in .zero }
        let oldJitter = SIMD2<Float>(0.25, 0.125), newJitter = SIMD2<Float>(-0.25, 0.375)
        let displacement = SIMD2<Float>(2.5, -1.5)
        let jitteredMotion = displacement + oldJitter - newJitter
        let motion = try gpu.color(width: width, height: height) { _, _ in
            SIMD4(jitteredMotion.x, jitteredMotion.y, 0.5, 1)
        }
        let output = try gpu.color(width: width, height: height)

        var results: [Float] = []
        for config: TemporalAntiAliasing.TRAASettings in [
            .init(maxVelocityLength: 32, useSubpixelCorrection: false),
            .init(maxVelocityLength: 32, useSubpixelCorrection: true),
            .init(maxVelocityLength: 8, useSubpixelCorrection: false),
            .init(maxVelocityLength: 0.25, useSubpixelCorrection: false)
        ] {
            var settings = TemporalAntiAliasing.Settings.referenceTRAA
            settings.traa = config
            let taa = try TemporalAntiAliasing(device: gpu.device, settings: settings)
            _ = try frame(taa, source: prior, depth: depth, motion: noHistory, output: output, jitter: oldJitter)
            let pixels = try frame(taa, source: current, depth: depth, motion: motion, output: output, jitter: newJitter)
            let actual = pixels[2 * width + 4]
            let motionFactor = min(simd_length(displacement) / config.maxVelocityLength, 1)
            let currentWeight = min(0.05 + motionFactor + (config.useSubpixelCorrection ? 0.25 : 0), 1)
            let mean: Float = 17 / 9
            let sigma = sqrt(65.0 / 9.0 - mean * mean)
            let varianceGamma = 0.5 + 0.5 * pow(1 - motionFactor, 2)
            let clipped = mean + sigma * varianceGamma
            let a = currentWeight / 1.5
            let b = (1 - currentWeight) / (1 + clipped / (1 + clipped))
            let reference = (a + clipped * b) / (a + b)
            #expect(PostProcessFixture.close(actual, Self.gray(reference), tolerance: 0.004),
                    "velocity \(config.maxVelocityLength), correction \(config.useSubpixelCorrection): \(actual)")
            results.append(actual.x)
        }
        #expect(results[0] > results[1] + 0.2)
        #expect(results[0] > results[2] + 0.2)
        #expect(results[3] == 1)
    }

    @Test("Depth edge policy recovers matching footprint history and rejects unrelated geometry",
          arguments: [MeshDepthMode.forward, .reversed])
    func depthEdgePolicy(depthMode: MeshDepthMode) throws {
        let width = 8, height = 4
        let prior = try gpu.color(width: width, height: height) { x, _ in Self.gray(x == 2 ? 4 : 0.125) }
        let current = try gpu.color(width: width, height: height) { x, _ in Self.gray(x == 4 ? 4 : 1) }
        func stored(_ value: Float) -> Float { depthMode == .reversed ? 1 - value : value }
        let oldDepth = try gpu.depth(width: width, height: height) { x, _ in stored(x == 2 ? 0.25 : 0.75) }
        let newDepth = try gpu.depth(width: width, height: height) { x, _ in stored(x == 5 || x == 6 ? 0.25 : 0.75) }
        let output = try gpu.color(width: width, height: height)
        let noHistory = try gpu.color(width: width, height: height) { _, _ in .zero }
        let vectors = try gpu.color(width: width, height: height) { x, y in
            guard x == 5 else { return .zero }
            // At x5, the raw nearest pixel is x3 (background). The footprint
            // also touches x2, which is the only eligible previous foreground tap.
            if y == 0 { return SIMD4(-2.25, 0, stored(0.25), 0) }
            if y == 3 { return SIMD4(-2.25, 0, stored(0.125), 1) }
            return SIMD4(-2.25, 0, stored(0.25), 1)
        }
        for threshold: Float in [0.001, 0.9] {
            var settings = TemporalAntiAliasing.Settings.referenceTRAA
            settings.traa?.edgeDepthDifference = threshold
            let taa = try TemporalAntiAliasing(device: gpu.device, settings: settings)
            _ = try frame(taa, source: prior, depth: oldDepth, motion: noHistory, output: output, depthMode: depthMode)
            let pixels = try frame(taa, source: current, depth: newDepth, motion: vectors, output: output, depthMode: depthMode)
            if threshold < 0.01 {
                #expect(pixels[width + 5].x > 2.4)
            } else {
                #expect(pixels[width + 5] == Self.gray(1))
            }
            #expect(pixels[5] == Self.gray(1))
            #expect(pixels[3 * width + 5] == Self.gray(1))
        }
    }

    @Test("Reference settings leave the camera-only resolve and its clipping behavior intact")
    func cameraOnlyAndValidation() throws {
        let size = SIMD2(7, 5)
        let prior = try gpu.color(width: size.x, height: size.y) { _, _ in Self.gray(3) }
        let current = try gpu.color(width: size.x, height: size.y) { x, _ in Self.gray(x.isMultiple(of: 2) ? 4 : 1) }
        let depth = try gpu.depth(width: size.x, height: size.y) { _, _ in 0.5 }
        let output = try gpu.color(width: size.x, height: size.y)
        let cameraOnly = try TemporalAntiAliasing(device: gpu.device, settings: .init(jitterSequenceLength: 32))
        let reference = try TemporalAntiAliasing(device: gpu.device, settings: .referenceTRAA)
        var results: [[SIMD4<Float>]] = []
        for taa in [cameraOnly, reference] {
            _ = try frame(taa, source: prior, depth: depth, motion: nil, output: output)
            results.append(try frame(taa, source: current, depth: depth, motion: nil, output: output))
        }
        #expect(results[0] == results[1])
        let motion = try gpu.color(width: size.x, height: size.y) { _, _ in SIMD4(0, 0, 0.5, 1) }
        let firstMotion = try frame(reference, source: current, depth: depth, motion: motion, output: output)
        #expect(try firstMotion == gpu.read(current))
        #expect(reference.accumulatedFrameCount == 1)

        reference.settings.traa?.maxVelocityLength = 0
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: TemporalAntiAliasing.Failure.invalidSettings) {
            try reference.encode(
                source: current, depth: depth, clipFromWorld: matrix_identity_float4x4, jitter: .zero,
                motionTexture: motion, destination: output, commandBuffer: buffer
            )
        }
        #expect(reference.accumulatedFrameCount == 1)
    }

    private func frame(
        _ taa: TemporalAntiAliasing, source: any MTLTexture, depth: any MTLTexture, motion: (any MTLTexture)?,
        output: any MTLTexture, jitter: SIMD2<Float> = .zero, depthMode: MeshDepthMode = .forward
    ) throws -> [SIMD4<Float>] {
        try gpu.run {
            try taa.encode(
                source: source, depth: depth, clipFromWorld: matrix_identity_float4x4, jitter: jitter,
                depthMode: depthMode, motionTexture: motion, destination: output, commandBuffer: $0
            )
        }
        return try gpu.read(output)
    }

    private static func gray(_ value: Float) -> SIMD4<Float> { SIMD4(value, value, value, 1) }
}
