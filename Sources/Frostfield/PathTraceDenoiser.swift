import Foundation
import Metal

/// Filters a linear HDR path-traced image using its first-hit surface guides.
public final class PathTraceDenoiser {
    private let pipeline: any MTLComputePipelineState

    public init(device: any MTLDevice) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        guard let shader = library.makeFunction(name: "frostfield_path_denoise") else { throw Failure.shader }
        pipeline = try device.makeComputePipelineState(function: shader)
    }

    /// The guides use signed normal RGB and ray distance A, and linear albedo RGB
    /// and roughness A. Depth zero denotes a miss. All textures are standalone,
    /// tracked RGBA16Float at the same size. Read inputs use `.shaderRead`, the
    /// destination uses `.shaderWrite`, and an optional scratch uses both.
    /// Encode after tracing on a retained command buffer. Reuse scratch only on
    /// the same queue in submission order or after the prior user completes.
    public func encode(
        source: any MTLTexture, normalDepth: any MTLTexture, albedoRoughness: any MTLTexture,
        destination: any MTLTexture, scratch: (any MTLTexture)? = nil, passes: Int = 3,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard (1...4).contains(passes) else { throw Failure.passes }
        guard commandBuffer.device === pipeline.device, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }

        func valid(_ texture: any MTLTexture, usage: MTLTextureUsage) -> Bool {
            texture.device === pipeline.device && texture.pixelFormat == .rgba16Float && texture.textureType == .type2D
                && texture.sampleCount == 1 && texture.width == source.width && texture.height == source.height
                && texture.hazardTrackingMode != .untracked && texture.storageMode != .memoryless
                && texture.parent == nil && texture.buffer == nil && texture.heap == nil && texture.iosurface == nil
                && !texture.isFramebufferOnly && texture.usage.contains(usage)
        }

        let input = [source, normalDepth, albedoRoughness]
        let outputUsage: MTLTextureUsage = passes > 2 ? [.shaderRead, .shaderWrite] : .shaderWrite
        guard input.allSatisfy({ valid($0, usage: .shaderRead) }), valid(destination, usage: outputUsage),
              input.allSatisfy({ $0 !== destination }) else { throw Failure.textures }
        let alternate: any MTLTexture
        if let scratch {
            guard valid(scratch, usage: [.shaderRead, .shaderWrite]), scratch !== destination,
                  input.allSatisfy({ $0 !== scratch }) else { throw Failure.textures }
            alternate = scratch
        } else if passes > 1 {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: source.width, height: source.height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = [.shaderRead, .shaderWrite]
            guard let texture = pipeline.device.makeTexture(descriptor: descriptor) else { throw Failure.resource }
            alternate = texture
        } else {
            alternate = destination
        }

        let groupWidth = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        let groupHeight = min(8, pipeline.maxTotalThreadsPerThreadgroup / groupWidth)
        let groups = MTLSize(
            width: (source.width + groupWidth - 1) / groupWidth,
            height: (source.height + groupHeight - 1) / groupHeight, depth: 1
        )
        let threads = MTLSize(width: groupWidth, height: groupHeight, depth: 1)
        var current = source
        for pass in 0..<passes {
            let output = (passes - pass).isMultiple(of: 2) ? alternate : destination
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
            encoder.label = "Frostfield guided path denoise"
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(current, index: 0)
            encoder.setTexture(normalDepth, index: 1)
            encoder.setTexture(albedoRoughness, index: 2)
            encoder.setTexture(output, index: 3)
            var spacing: UInt32 = 1 << pass
            encoder.setBytes(&spacing, length: MemoryLayout<UInt32>.stride, index: 0)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
            encoder.endEncoding()
            current = output
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case shader, passes, commandBuffer, textures, resource, encoder

        public var errorDescription: String? {
            switch self {
            case .shader: "Frostfield's path denoiser shader is missing."
            case .passes: "Use one to four denoising passes."
            case .commandBuffer: "Use an uncommitted command buffer on the denoiser device that retains resources."
            case .textures: "Denoising requires separate, tracked RGBA16Float textures with matching dimensions and compatible usage."
            case .resource: "Metal could not allocate the denoiser scratch texture."
            case .encoder: "Metal could not create the denoiser compute encoder."
            }
        }
    }
}
