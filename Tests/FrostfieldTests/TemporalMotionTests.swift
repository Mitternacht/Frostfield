import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TemporalMotionTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Motion follows a foreground across pixels and rejects newly revealed depth", arguments: [MeshDepthMode.forward, .reversed])
    func movingForeground(mode: MeshDepthMode) throws {
        let width = 8, height = 4
        let previous = try gpu.color(width: width, height: height) { x, _ in
            Self.gray(x == 2 ? 4 : x == 3 ? 8 : 2)
        }
        let current = try gpu.color(width: width, height: height) { _, _ in Self.gray(1) }
        let output = try gpu.color(width: width, height: height)
        func stored(_ forward: Float) -> Float { mode == .reversed ? 1 - forward : forward }
        let previousDepth = try gpu.depth(width: width, height: height) { x, _ in stored(x == 2 || x == 3 ? 0.25 : 0.75) }
        let currentDepth = try gpu.depth(width: width, height: height) { x, _ in
            stored(x == 7 ? 1 : x == 5 || x == 6 ? 0.25 : 0.75)
        }
        let noHistory = try gpu.color(width: width, height: height) { _, _ in .zero }
        let motion = try gpu.color(width: width, height: height) { x, _ in
            switch x {
            case 1, 2: return SIMD4(0.75, 0, stored(0.75), 1) // depth lookup hits old foreground
            case 3: return SIMD4(0, 0, .nan, 1)
            case 4: return SIMD4(-99, 0, stored(0.25), 1)
            case 5: return SIMD4(-2.25, 0, stored(0.25), 1) // raw x3 and corrected color x2
            case 6: return SIMD4(-3.25, 0, stored(0.25), 0)
            default: return SIMD4(0.75, 0, stored(0.75), 1)
            }
        }
        let taa = try TemporalAntiAliasing(device: gpu.device, settings: .accumulation)
        try gpu.run {
            try taa.encode(
                source: previous, depth: previousDepth, clipFromWorld: matrix_identity_float4x4,
                jitter: SIMD2(0.25, 0), depthMode: mode, motionTexture: noHistory,
                destination: output, commandBuffer: $0
            )
        }
        #expect(try gpu.read(output) == gpu.read(previous))
        try gpu.run {
            try taa.encode(
                source: current, depth: currentDepth, clipFromWorld: matrix_identity_float4x4,
                jitter: SIMD2(-0.5, 0), depthMode: mode, motionTexture: motion,
                destination: output, commandBuffer: $0
            )
        }
        let result = try gpu.read(output)
        for y in 0..<height {
            for x in 0..<width {
                let expected: Float = x == 5 ? 2.5 : x == 0 ? 1.5 : 1
                #expect(result[y * width + x] == Self.gray(expected), "\(mode) at \(x), \(y)")
            }
        }
        #expect(taa.accumulatedFrameCount == 2)
    }

    @Test("Depth-gated color taps ignore old neighbors and motion mode seeds depth after camera-only use")
    func filteredDepthBoundary() throws {
        let width = 8, height = 4
        let discardedColor = try gpu.color(width: width, height: height) { _, _ in Self.gray(7) }
        let previous = try gpu.color(width: width, height: height) { x, _ in Self.gray(x == 2 ? 4 : 0) }
        let current = try gpu.color(width: width, height: height) { x, _ in Self.gray(x == 5 ? 0 : 4) }
        let output = try gpu.color(width: width, height: height)
        let previousDepth = try gpu.depth(width: width, height: height) { x, _ in x == 2 ? 0.25 : 0.75 }
        let currentDepth = try gpu.depth(width: width, height: height) { x, _ in x == 5 ? 0.25 : 0.75 }
        let motion = try gpu.color(width: width, height: height) { x, _ in
            x == 5 ? SIMD4(-2.75, 0, 0.25, 1) : .zero
        }
        let taa = try TemporalAntiAliasing(device: gpu.device)
        func frame(_ color: any MTLTexture, depth: any MTLTexture, motion: (any MTLTexture)?) throws {
            try gpu.run {
                try taa.encode(
                    source: color, depth: depth, clipFromWorld: matrix_identity_float4x4, jitter: .zero,
                    motionTexture: motion, destination: output, commandBuffer: $0
                )
            }
        }
        try frame(discardedColor, depth: previousDepth, motion: nil)
        try frame(previous, depth: previousDepth, motion: motion)
        #expect(try gpu.read(output) == gpu.read(previous))
        #expect(taa.accumulatedFrameCount == 1)
        try frame(current, depth: currentDepth, motion: motion)
        let result = try gpu.read(output)
        // Previous position x=2.75. The color at x=3 has weight 0.25 but belongs to old background.
        #expect(result[width + 5] == Self.gray(2))
        #expect(result[width + 4] == Self.gray(4))

        taa.reset()
        try frame(current, depth: currentDepth, motion: motion)
        #expect(try gpu.read(output) == gpu.read(current))
    }

    @Test("Motion texture format, dimensions and destination aliasing are validated before advancing history")
    func validation() throws {
        let source = try gpu.color(width: 5, height: 3)
        let output = try gpu.color(width: 5, height: 3)
        let depth = try gpu.depth(width: 5, height: 3) { _, _ in 0.25 }
        let wrongSize = try gpu.color(width: 4, height: 3)
        let wrongFormat = try gpu.color(width: 5, height: 3, pixelFormat: .rgba8Unorm)
        let taa = try TemporalAntiAliasing(device: gpu.device)
        let command = try #require(gpu.queue.makeCommandBuffer())
        func encode(_ motion: any MTLTexture) throws {
            try taa.encode(
                source: source, depth: depth, clipFromWorld: matrix_identity_float4x4, jitter: .zero,
                motionTexture: motion, destination: output, commandBuffer: command
            )
        }
        #expect(throws: TemporalAntiAliasing.Failure.motion) { try encode(wrongFormat) }
        #expect(throws: TemporalAntiAliasing.Failure.sizeMismatch) { try encode(wrongSize) }
        #expect(throws: TemporalAntiAliasing.Failure.destination) { try encode(output) }
        #expect(taa.frameIndex == 0 && taa.accumulatedFrameCount == 0)
    }

    private static func gray(_ value: Float) -> SIMD4<Float> { SIMD4(value, value, value, 1) }
}
