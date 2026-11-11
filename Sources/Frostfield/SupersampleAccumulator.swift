import Foundation
import Metal
import simd

/// Supersampling by jittered accumulation, like Three.js `SSAARenderPass`.
///
/// `encode` asks the caller to render `sampleCount` subframes, each with the
/// projection offset by a subpixel jitter from `jitterOffsets(sampleCount:)`, and
/// writes their mean to the destination. Use `jittered(_:offset:width:height:)` on
/// the scene's world-to-clip matrix and the
/// `SceneGraph.encode(pass:commandBuffer:clipFromWorld:cameraPosition:...)` overload.
///
/// Subframes are summed in single precision and the destination is written once, as
/// `sum / sampleCount`, so the result is the RGBA16Float rounding of the exact mean at
/// every sample count. (Adding `subframe / sampleCount` to a half-float target
/// instead rounds the running sum at every step and darkens the result by up to 5 %
/// at 256 samples.) The accumulator owns one RGBA16Float subframe target and two
/// RGBA32Float running sums, and reuses them until the size changes.
public final class SupersampleAccumulator {
    public static let maximumSampleCount = 256

    public let device: any MTLDevice
    private let shaders: PostProcessShaders
    private var subframe: (any MTLTexture)?
    private var sums: [any MTLTexture] = []

    public init(device: any MTLDevice) throws {
        self.device = device
        shaders = try PostProcessShaders(device: device)
        _ = try shaders.render("frostfield_accumulate_fragment", formats: [.rgba32Float])
        _ = try shaders.render("frostfield_accumulate_fragment", formats: [.rgba16Float])
    }

    /// Sample offsets in pixels within `[-½, ½)`, positive x right and positive y down.
    ///
    /// Counts 1, 2, 4, 8 and 16 use the Direct3D standard multisample positions, as
    /// Three.js does. Other counts up to 256 use the Halton (2, 3) sequence, which
    /// stratifies any count.
    public static func jitterOffsets(sampleCount: Int) throws -> [SIMD2<Float>] {
        guard (1...maximumSampleCount).contains(sampleCount) else {
            throw PostProcessEffectError.invalidSettings("Supersampling takes 1 to \(maximumSampleCount) samples.")
        }
        let standard: [Int: [(Int, Int)]] = [
            1: [(0, 0)],
            2: [(4, 4), (-4, -4)],
            4: [(-2, -6), (6, -2), (-6, 2), (2, 6)],
            8: [(1, -3), (-1, 3), (5, 1), (-3, -5), (-5, 5), (-7, -1), (3, 7), (7, -7)],
            16: [(1, 1), (-1, -3), (-3, 2), (4, -1), (-5, -2), (2, 5), (5, 3), (3, -5),
                 (-2, 6), (0, -7), (-4, -6), (-6, 4), (-8, 0), (7, -4), (6, 7), (-7, -8)],
        ]
        if let positions = standard[sampleCount] {
            return positions.map { SIMD2(Float($0.0), Float($0.1)) / 16 }
        }
        func halton(_ index: Int, _ base: Int) -> Float {
            var result: Float = 0
            var fraction: Float = 1
            var i = index
            while i > 0 {
                fraction /= Float(base)
                result += fraction * Float(i % base)
                i /= base
            }
            return result
        }
        return (1...sampleCount).map { SIMD2(halton($0, 2), halton($0, 3)) - 0.5 }
    }

    /// Offsets a world-to-clip matrix so the pixel centers of a `width` × `height`
    /// target sample the scene at `offset` pixels from their centers.
    public static func jittered(_ clipFromWorld: simd_float4x4, offset: SIMD2<Float>, width: Int, height: Int) -> simd_float4x4 {
        var jitter = matrix_identity_float4x4
        jitter.columns.3 = SIMD4(-2 * offset.x / Float(width), 2 * offset.y / Float(height), 0, 1)
        return jitter * clipFromWorld
    }

    /// Renders `sampleCount` jittered subframes and writes their mean into
    /// `destination`, a tracked RGBA16Float render target, on the caller's retained
    /// command buffer.
    ///
    /// `renderSubframe` receives the subframe index, its jitter in pixels and the
    /// RGBA16Float target to fill completely, and encodes into the same command buffer.
    public func encode(
        sampleCount: Int,
        destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer,
        renderSubframe: (_ index: Int, _ jitter: SIMD2<Float>, _ target: any MTLTexture) throws -> Void
    ) throws {
        let offsets = try Self.jitterOffsets(sampleCount: sampleCount)
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw PostProcessEffectError.commandBuffer
        }
        guard commandBuffer.device === device, destination.device === device else { throw PostProcessEffectError.wrongDevice }
        guard destination.pixelFormat == .rgba16Float, destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              destination.width > 0, destination.height > 0 else {
            throw PostProcessEffectError.destination
        }
        if subframe?.width != destination.width || subframe?.height != destination.height {
            subframe = try PostProcessShaders.scratch(
                device: device, width: destination.width, height: destination.height,
                usage: [.renderTarget, .shaderRead], label: "Frostfield supersample subframe"
            )
        }
        guard let subframe else { throw PostProcessEffectError.allocation }
        let sumCount = min(2, sampleCount - 1)
        if sums.first.map({ $0.width != destination.width || $0.height != destination.height }) ?? false {
            sums = []
        }
        while sums.count < sumCount {
            sums.append(try PostProcessShaders.scratch(
                device: device, width: destination.width, height: destination.height, pixelFormat: .rgba32Float,
                usage: [.renderTarget, .shaderRead], label: "Frostfield supersample sum \(sums.count)"
            ))
        }
        let sumPipeline = try shaders.render("frostfield_accumulate_fragment", formats: [.rgba32Float])
        let meanPipeline = try shaders.render("frostfield_accumulate_fragment", formats: [.rgba16Float])
        for (index, offset) in offsets.enumerated() {
            try renderSubframe(index, offset, subframe)
            let isLast = index == offsets.count - 1
            // Subframe i adds the running sum of the earlier ones; the last writes the mean.
            let previous = index == 0 ? subframe : sums[(index - 1) % 2]
            var parameters = SIMD2<Float>(isLast ? 1 / Float(sampleCount) : 1, index == 0 ? 0 : 1)
            try PostProcessShaders.drawFullScreen(
                into: isLast ? destination : sums[index % 2], label: "Frostfield supersample accumulate \(index)",
                commandBuffer: commandBuffer, pipeline: isLast ? meanPipeline : sumPipeline
            ) { encoder in
                encoder.setFragmentTexture(subframe, index: 0)
                encoder.setFragmentTexture(previous, index: 1)
                encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
            }
        }
    }
}
