import Foundation
import Metal

/// Expands a native block-compressed 2D Metal texture into a shader-readable texture.
/// Metal supplies the decompressed samples, including any source texture swizzle.
public final class TextureDecompressor: Sendable {
    public let device: any MTLDevice
    private let pipelines: [MTLPixelFormat: any MTLRenderPipelineState]

    public init(device: any MTLDevice) throws {
        self.device = device
        do {
            let shaders = try PostProcessShaders(device: device)
            let formats: [MTLPixelFormat] = [.rgba8Unorm, .rgba8Unorm_srgb, .rgba16Float]
            var pipelines: [MTLPixelFormat: any MTLRenderPipelineState] = [:]
            for format in formats {
                pipelines[format] = try shaders.render("frostfield_decompress_texture_fragment", formats: [format])
            }
            self.pipelines = pipelines
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Returns a completed, private-storage uncompressed texture.
    ///
    /// Supply the producer queue to order the copy after committed, tracked
    /// source writes. Finish source writes first when using another queue or an
    /// untracked source. Populate each requested mip before calling.
    public func decompress(
        _ source: any MTLTexture, maximumSize: Int? = nil, includeMipmaps: Bool = true,
        commandQueue: (any MTLCommandQueue)? = nil
    ) throws -> any MTLTexture {
        if let commandQueue, commandQueue.device !== device { throw Failure.wrongDevice }
        guard let queue = commandQueue ?? device.makeCommandQueue() else { throw Failure.resource("command queue") }
        guard let command = queue.makeCommandBuffer() else { throw Failure.resource("command buffer") }
        command.label = "Frostfield texture decompression"
        let texture = try encode(source, maximumSize: maximumSize, includeMipmaps: includeMipmaps, commandBuffer: command)
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw Failure.execution(command.error?.localizedDescription ?? "The command buffer did not complete.")
        }
        return texture
    }

    /// Encodes a copy into a new texture on an uncommitted, retaining command buffer.
    ///
    /// LDR color and unsigned 8-bit data expand to RGBA8 with the source transfer.
    /// Signed formats, 11-bit EAC, BC6H and HDR ASTC expand to linear RGBA16Float.
    /// Samples go directly to the destination format with their original
    /// orientation, source swizzles, and alpha association.
    ///
    /// By default every allocated mip is copied. Set `includeMipmaps: false` for
    /// only level zero. A positive `maximumSize` caps width and height separately.
    /// When this changes dimensions, bilinear sampling scales each copied level,
    /// keeping only levels that fit the output. The output budget is 8,192 pixels
    /// per axis and 16,777,216 base texels.
    ///
    /// This method returns before GPU execution. Use the result after the caller's
    /// command buffer completes, or in subsequent ordered passes on the same queue.
    /// The caller supplies synchronization for any untracked or cross-queue source.
    public func encode(
        _ source: any MTLTexture, maximumSize: Int? = nil, includeMipmaps: Bool = true,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        guard source.device === device, commandBuffer.device === device else { throw Failure.wrongDevice }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard source.textureType == .type2D, source.sampleCount == 1, !source.isFramebufferOnly,
              source.storageMode != .memoryless, source.isSparse != true,
              source.usage == .unknown || source.usage.contains(.shaderRead) else { throw Failure.source }
        guard let format = Self.outputFormat(for: source.pixelFormat), let pipeline = pipelines[format] else {
            throw Failure.pixelFormat(source.pixelFormat)
        }

        let cap = maximumSize ?? Int.max
        let width = min(source.width, cap), height = min(source.height, cap)
        guard (try? TextureLoader.validate(width: width, height: height)) != nil else { throw Failure.size }
        let fullMipCount = Int.bitWidth - max(width, height).leadingZeroBitCount
        let levels = includeMipmaps ? min(fullMipCount, source.mipmapLevelCount) : 1
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.mipmapLevelCount = levels
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.shaderRead, .renderTarget]
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.resource("output texture") }
        texture.label = source.label

        for mip in 0..<levels {
            let mipWidth = max(1, width >> mip), mipHeight = max(1, height >> mip)
            let resizes = mipWidth != max(1, source.width >> mip) || mipHeight != max(1, source.height >> mip)
            let pass = MTLRenderPassDescriptor()
            let attachment = pass.colorAttachments[0]!
            attachment.texture = texture
            attachment.level = mip
            attachment.loadAction = .dontCare
            attachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
                throw Failure.resource("render encoder")
            }
            var options = SIMD4<UInt32>(UInt32(mipWidth), UInt32(mipHeight), UInt32(mip), resizes ? 1 : 0)
            encoder.label = "Frostfield decompress mip \(mip)"
            encoder.setRenderPipelineState(pipeline)
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentBytes(&options, length: MemoryLayout.size(ofValue: options), index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
        }
        return texture
    }

    private static func outputFormat(for format: MTLPixelFormat) -> MTLPixelFormat? {
        switch format {
        case .bc1_rgba, .bc2_rgba, .bc3_rgba, .bc4_rUnorm, .bc5_rgUnorm, .bc7_rgbaUnorm,
             .eac_rgba8, .etc2_rgb8, .etc2_rgb8a1, .pvrtc_rgb_2bpp, .pvrtc_rgb_4bpp, .pvrtc_rgba_2bpp, .pvrtc_rgba_4bpp:
            return .rgba8Unorm
        case .bc1_rgba_srgb, .bc2_rgba_srgb, .bc3_rgba_srgb, .bc7_rgbaUnorm_srgb,
             .eac_rgba8_srgb, .etc2_rgb8_srgb, .etc2_rgb8a1_srgb,
             .pvrtc_rgb_2bpp_srgb, .pvrtc_rgb_4bpp_srgb, .pvrtc_rgba_2bpp_srgb, .pvrtc_rgba_4bpp_srgb:
            return .rgba8Unorm_srgb
        case .bc4_rSnorm, .bc5_rgSnorm, .bc6H_rgbFloat, .bc6H_rgbuFloat,
             .eac_r11Unorm, .eac_r11Snorm, .eac_rg11Unorm, .eac_rg11Snorm:
            return .rgba16Float
        default:
            // Each ASTC profile has a consecutive range of footprint formats
            // in Metal. Reserved enum slots cannot be used by a source texture.
            switch format.rawValue {
            case MTLPixelFormat.astc_4x4_ldr.rawValue...MTLPixelFormat.astc_12x12_ldr.rawValue: return .rgba8Unorm
            case MTLPixelFormat.astc_4x4_srgb.rawValue...MTLPixelFormat.astc_12x12_srgb.rawValue: return .rgba8Unorm_srgb
            case MTLPixelFormat.astc_4x4_hdr.rawValue...MTLPixelFormat.astc_12x12_hdr.rawValue: return .rgba16Float
            default: return nil
            }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case pipeline(String)
        case wrongDevice
        case commandBuffer
        case source
        case pixelFormat(MTLPixelFormat)
        case size
        case resource(String)
        case execution(String)

        public var errorDescription: String? {
            switch self {
            case .pipeline(let detail): "Could not prepare Metal texture decompression. \(detail)"
            case .wrongDevice: "Use the decompressor's Metal device for the source and commands."
            case .commandBuffer: "Use an uncommitted command buffer that retains references."
            case .source: "Use a shader-readable, nonsparse, single-sample 2D texture."
            case .pixelFormat(let format): "Metal format \(format.rawValue) is outside the supported block formats."
            case .size: "Use a positive size within 8,192 pixels per axis and 16,777,216 base texels."
            case .resource(let name): "Metal could not allocate the decompression \(name)."
            case .execution(let detail): "Metal texture decompression failed. \(detail)"
            }
        }
    }
}
