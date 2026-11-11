import Foundation
import Metal

/// Color or depth sampled by raster surface graphs. Either supply a finished
/// render or have the scene capture opaque draws before shading the overlay.
/// Order all producer and consumer passes on the same Metal queue.
public struct SceneViewportSnapshot {
    public enum Source: Equatable, Sendable {
        /// Caller-produced textures, with normalized sampling even at reduced size.
        case stored
        /// Copy the current scene's opaque color and depth before its overlays.
        /// This requires a single-sample color pass with matching-size textures.
        case opaquePass
    }

    public let source: Source
    /// Premultiplied linear RGBA16Float. Surface graphs sample mip zero bilinearly.
    public let color: (any MTLTexture)?
    /// Depth32Float containing the producer's stored 0...1 values, sampled nearest.
    public let depth: (any MTLTexture)?

    /// Supply at least one tracked, shader-readable single-sample texture.
    /// When both are present they must share a device and dimensions. An
    /// opaque capture generates the color mip levels when the texture has them.
    public init(
        color: (any MTLTexture)? = nil, depth: (any MTLTexture)? = nil, source: Source = .stored
    ) throws {
        guard color != nil || depth != nil else { throw Failure.empty }
        if let color, !PostProcessPass.isReadable(color, format: .rgba16Float) { throw Failure.color }
        if let depth, !PostProcessPass.isReadable(depth, format: .depth32Float) { throw Failure.depth }
        if let color, let depth {
            guard color.device === depth.device else { throw Failure.device }
            guard color.width == depth.width, color.height == depth.height else { throw Failure.size }
        }
        self.color = color
        self.depth = depth
        self.source = source
    }

    struct OpaqueCapture {
        let opaque: MTLRenderPassDescriptor
        let overlay: MTLRenderPassDescriptor
        private let copies: [(source: any MTLTexture, destination: any MTLTexture)]

        init(
            snapshot: SceneViewportSnapshot, pass: MTLRenderPassDescriptor,
            transmissionBackdrop: (any MTLTexture)?
        ) throws {
            let color = pass.colorAttachments[0]!
            let depth = pass.depthAttachment!
            guard let colorTarget = color.texture, let depthTarget = depth.texture,
                  pass.stencilAttachment.texture == nil, pass.stencilAttachment.resolveTexture == nil,
                  pass.renderTargetArrayLength <= 1,
                  (pass.renderTargetWidth == 0 || pass.renderTargetWidth == colorTarget.width),
                  (pass.renderTargetHeight == 0 || pass.renderTargetHeight == colorTarget.height),
                  pass.rasterizationRateMap == nil,
                  (1..<8).allSatisfy({
                      pass.colorAttachments[$0].texture == nil && pass.colorAttachments[$0].resolveTexture == nil
                  }),
                  Self.canSplit(color, texture: colorTarget), Self.canSplit(depth, texture: depthTarget),
                  let opaque = pass.copy() as? MTLRenderPassDescriptor,
                  let overlay = pass.copy() as? MTLRenderPassDescriptor else { throw Failure.capturePass }
            var copies: [(source: any MTLTexture, destination: any MTLTexture)] = []
            if let colorCopy = snapshot.color {
                try Self.validateCopy(from: colorTarget, to: colorCopy)
                if let transmissionBackdrop, TextureTransfer.sharesStorage(colorCopy, transmissionBackdrop) {
                    guard colorCopy === transmissionBackdrop else { throw Failure.feedback }
                } else {
                    copies.append((colorTarget, colorCopy))
                }
            }
            if let depthCopy = snapshot.depth {
                try Self.validateCopy(from: depthTarget, to: depthCopy)
                copies.append((depthTarget, depthCopy))
            }
            opaque.colorAttachments[0].storeAction = .store
            opaque.depthAttachment.storeAction = .store
            overlay.colorAttachments[0].loadAction = .load
            overlay.depthAttachment.loadAction = .load
            self.opaque = opaque
            self.overlay = overlay
            self.copies = copies
        }

        func encodeCopy(commandBuffer: any MTLCommandBuffer) throws {
            guard !copies.isEmpty else { return }
            guard let encoder = commandBuffer.makeBlitCommandEncoder() else { throw Failure.captureEncoder }
            encoder.label = "Frostfield opaque viewport snapshot"
            for (source, destination) in copies {
                encoder.copy(
                    from: source, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                    sourceSize: MTLSize(width: source.width, height: source.height, depth: 1),
                    to: destination, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin()
                )
                if destination.pixelFormat == .rgba16Float, destination.mipmapLevelCount > 1 {
                    encoder.generateMipmaps(for: destination)
                }
            }
            encoder.endEncoding()
        }

        private static func canSplit(_ attachment: MTLRenderPassAttachmentDescriptor, texture: any MTLTexture) -> Bool {
            texture.textureType == .type2D && texture.sampleCount == 1
                && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
                && attachment.level == 0 && attachment.slice == 0 && attachment.depthPlane == 0
                && attachment.resolveTexture == nil
                && (attachment.loadAction == .clear || attachment.loadAction == .load)
                && (attachment.storeAction == .store || attachment.storeAction == .dontCare)
        }

        private static func validateCopy(from source: any MTLTexture, to destination: any MTLTexture) throws {
            guard !source.isFramebufferOnly, source.pixelFormat == destination.pixelFormat,
                  [source, destination].allSatisfy({
                      $0.parent == nil && $0.buffer == nil && !$0.isAliasable() && $0.isSparse != true
                          && ($0.heap.map { $0.hazardTrackingMode == .tracked } ?? true)
                  }) else { throw Failure.capturePass }
            guard source.width == destination.width, source.height == destination.height else { throw Failure.size }
        }
    }

    func validate(pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        var attachments: [MTLRenderPassAttachmentDescriptor] = (0..<8).map { pass.colorAttachments[$0]! }
        attachments.append(pass.depthAttachment)
        attachments.append(pass.stencilAttachment)
        for texture in [color, depth].compactMap({ $0 }) {
            guard texture.device === commandBuffer.device else { throw Failure.device }
            for attachment in attachments {
                if [attachment.texture, attachment.resolveTexture].contains(where: {
                    $0.map { TextureTransfer.sharesStorage(texture, $0) } ?? false
                }) {
                    throw Failure.feedback
                }
            }
        }
    }

    struct Prepared {
        let arguments: any MTLBuffer
        let textures: [any MTLTexture]

        func bind(into encoder: any MTLRenderCommandEncoder) {
            for texture in textures { encoder.useResource(texture, usage: .read, stages: .fragment) }
            encoder.setFragmentBuffer(arguments, offset: 0, index: 30)
        }
    }

    func prepare(device: any MTLDevice) throws -> Prepared {
        // Let Metal choose the argument-buffer layout, including Tier 1's
        // private layout. Both the color and depth fields are 2D textures.
        let descriptors = (0..<2).map { index in
            let descriptor = MTLArgumentDescriptor()
            descriptor.index = index
            descriptor.dataType = .texture
            descriptor.textureType = .type2D
            descriptor.access = .readOnly
            return descriptor
        }
        guard let encoder = device.makeArgumentEncoder(arguments: descriptors),
              let arguments = device.makeBuffer(length: encoder.encodedLength, options: .storageModeShared) else {
            throw Failure.allocation
        }
        arguments.label = "Frostfield viewport snapshot arguments"
        encoder.setArgumentBuffer(arguments, offset: 0)
        encoder.setTexture(color, index: 0)
        encoder.setTexture(depth, index: 1)
        return Prepared(arguments: arguments, textures: [color, depth].compactMap { $0 })
    }

    public enum Failure: LocalizedError, Equatable {
        case empty, color, depth, device, size, feedback, commandBuffer
        case missingColor, missingDepth, allocation
        case capturePass, captureReader, captureEncoder

        public var errorDescription: String? {
            switch self {
            case .empty: "Provide a viewport color or depth texture."
            case .color: "Viewport color must be a tracked shader-readable 2D single-sample RGBA16Float texture."
            case .depth: "Viewport depth must be a tracked shader-readable 2D single-sample Depth32Float texture."
            case .device: "Viewport textures must use the render pass's Metal device."
            case .size: "Viewport color and depth must have the same dimensions."
            case .feedback: "Viewport textures must use separate storage from the pass's render and resolve attachments."
            case .commandBuffer: "Use an uncommitted command buffer that retains its references for viewport sampling."
            case .missingColor: "A visible surface graph reads scene color. Supply a stored viewport color texture."
            case .missingDepth: "A visible draw reads scene depth. Supply a viewport depth texture."
            case .allocation: "Metal could not allocate the surface viewport texture bindings."
            case .capturePass: "Opaque viewport capture requires a single-sample scene color pass with copyable base-level attachments."
            case .captureReader: "A surface graph that reads the opaque viewport must draw in the transparent or transmission pass."
            case .captureEncoder: "Metal could not create the opaque viewport blit encoder."
            }
        }
    }
}

struct SurfaceViewportInputs: OptionSet {
    let rawValue: UInt8

    static let color = Self(rawValue: 1)
    static let depth = Self(rawValue: 2)

    func validate(in snapshot: SceneViewportSnapshot?) throws {
        if contains(.color), snapshot?.color == nil { throw SceneViewportSnapshot.Failure.missingColor }
        if contains(.depth), snapshot?.depth == nil { throw SceneViewportSnapshot.Failure.missingDepth }
    }
}
