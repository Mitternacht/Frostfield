import Foundation
import Metal
import simd
import Synchronization

/// An opaque scene depth snapshot for soft particle intersections.
/// Supply the same full-size view and depth mapping as the particle draw.
/// Store the snapshot before drawing particles and keep its storage separate
/// from the active color and depth attachments. A cleared sample leaves
/// particle opacity unchanged.
public struct SoftParticleDepth {
    /// A view-space fade distance and contrast. Points and sprites take these
    /// values from their material and read depth from the scene viewport.
    public struct Fade: Equatable, Sendable {
        public let distance: Float
        public let contrast: Float

        public init(distance: Float = 1, contrast: Float = 2) throws {
            guard distance.isNormal, distance > 0, contrast.isNormal, contrast > 0 else { throw Failure.settings }
            self.distance = distance
            self.contrast = contrast
        }
    }

    public let texture: any MTLTexture
    /// View-space distance over which the particle reaches its normal opacity.
    public let distance: Float
    /// Symmetric fade curve power. One gives a linear fade.
    public let contrast: Float

    public init(texture: any MTLTexture, distance: Float = 1, contrast: Float = 2) throws {
        let fade = try Fade(distance: distance, contrast: contrast)
        guard PostProcessPass.isReadable(texture, format: .depth32Float), !texture.isAliasable() else {
            throw Failure.texture
        }
        self.texture = texture
        self.distance = fade.distance
        self.contrast = fade.contrast
    }

    public enum Failure: LocalizedError, Equatable {
        case settings, texture, attachments, storageOverlap, projection, commandBuffer

        public var errorDescription: String? {
            switch self {
            case .settings: "Soft particle distance and contrast must be positive, finite normal numbers."
            case .texture: "Use a tracked, shader-readable, single-sample Depth32Float snapshot on the particle device."
            case .attachments: "Soft particles need a full-size single-sample scene pass that matches the depth snapshot."
            case .storageOverlap: "The sampled soft particle depth must have storage separate from the draw attachments."
            case .projection: "Use a perspective or orthographic Metal projection that matches the stored depth mapping."
            case .commandBuffer: "Use an uncommitted command buffer that retains its references."
            }
        }
    }

    struct Prepared {
        let texture: any MTLTexture
        let projection: SIMD4<Float>
        let settings: SIMD4<Float>

        func bind(into encoder: any MTLRenderCommandEncoder) {
            var constants = (projection, settings)
            encoder.setFragmentTexture(texture, index: 0)
            encoder.setFragmentBytes(&constants, length: MemoryLayout.size(ofValue: constants), index: 1)
        }
    }

    func prepare(
        pass: MTLRenderPassDescriptor, clipFromView: simd_float4x4, depthMode: MeshDepthMode,
        device: any MTLDevice, commandBuffer: any MTLCommandBuffer
    ) throws -> Prepared {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else { throw Failure.commandBuffer }
        guard texture.device === device, commandBuffer.device === device, !texture.isAliasable() else { throw Failure.texture }
        let colorAttachment = pass.colorAttachments[0]!
        let depthAttachment = pass.depthAttachment!
        func matches(_ attachment: MTLRenderPassAttachmentDescriptor, texture attachmentTexture: any MTLTexture) -> Bool {
            attachment.level == 0 && attachment.slice == 0 && attachment.depthPlane == 0
                && attachment.resolveTexture == nil
                && attachmentTexture.device === device && attachmentTexture.textureType == .type2D
                && attachmentTexture.sampleCount == 1 && attachmentTexture.width == texture.width
                && attachmentTexture.height == texture.height
        }
        guard let color = colorAttachment.texture, matches(colorAttachment, texture: color),
              depthAttachment.texture.map({ matches(depthAttachment, texture: $0) }) ?? (depthAttachment.resolveTexture == nil),
              pass.renderTargetWidth == 0 || pass.renderTargetWidth == texture.width,
              pass.renderTargetHeight == 0 || pass.renderTargetHeight == texture.height,
              pass.renderTargetArrayLength <= 1, pass.rasterizationRateMap == nil,
              pass.stencilAttachment.texture == nil, pass.stencilAttachment.resolveTexture == nil,
              (1..<8).allSatisfy({ pass.colorAttachments[$0].texture == nil && pass.colorAttachments[$0].resolveTexture == nil })
        else { throw Failure.attachments }

        guard !TextureTransfer.sharesStorage(texture, color),
              depthAttachment.texture.map({ !TextureTransfer.sharesStorage(texture, $0) }) ?? true else {
            throw Failure.storageOverlap
        }

        let coefficients = try Self.projectionCoefficients(clipFromView, depthMode: depthMode)
        return Prepared(
            texture: texture, projection: coefficients,
            settings: SIMD4(distance, contrast, depthMode == .reversed ? 0 : 1, 0)
        )
    }

    static func projectionCoefficients(_ clipFromView: simd_float4x4, depthMode: MeshDepthMode) throws -> SIMD4<Float> {
        // Metal depth is (a · viewZ + b) / (c · viewZ + d). For these
        // projections, X and Y do not enter either clip Z or clip W.
        let (x, y, z, w) = clipFromView.columns
        let a = z.z, b = w.z, c = z.w, d = w.w
        let derivative = a * d - b * c
        guard x.z == 0, x.w == 0, y.z == 0, y.w == 0,
              (c < 0 && d == 0) || (c == 0 && d > 0),
              derivative.isFinite, depthMode == .reversed ? derivative > 0 : derivative < 0 else {
            throw Failure.projection
        }
        return SIMD4(a, b, c, d)
    }
}

/// Both particle renderers use the same fragment resource specialization.
final class ParticleDrawPipelines {
    private let library: any MTLLibrary
    private let descriptor: MTLRenderPipelineDescriptor
    private let fragment: String
    private let hard: any MTLRenderPipelineState
    private let soft = Mutex<(any MTLRenderPipelineState)?>(nil)

    init(library: any MTLLibrary, descriptor: MTLRenderPipelineDescriptor, fragment: String) throws {
        self.library = library
        self.descriptor = descriptor
        self.fragment = fragment
        hard = try Self.make(library: library, descriptor: descriptor, fragment: fragment, soft: false)
    }

    func state(withSoftDepth enabled: Bool) throws -> any MTLRenderPipelineState {
        guard enabled else { return hard }
        return try soft.withLock { cached in
            if let cached { return cached }
            let prepared = try Self.make(library: library, descriptor: descriptor, fragment: fragment, soft: true)
            cached = prepared
            return prepared
        }
    }

    private static func make(
        library: any MTLLibrary, descriptor: MTLRenderPipelineDescriptor, fragment: String, soft: Bool
    ) throws -> any MTLRenderPipelineState {
        let constants = MTLFunctionConstantValues()
        var soft = soft
        constants.setConstantValue(&soft, type: .bool, index: 56)
        descriptor.fragmentFunction = try library.makeFunction(name: fragment, constantValues: constants)
        return try library.device.makeRenderPipelineState(descriptor: descriptor)
    }
}
