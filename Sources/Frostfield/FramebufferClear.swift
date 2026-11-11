import Foundation
import Metal
import simd

/// Independent framebuffer clears for caller-owned Metal render passes.
public struct FramebufferClear: Equatable, Sendable {
    /// Premultiplied linear RGBA, with alpha in 0...1. Applies to every active
    /// color attachment. The texture format determines stored precision and range.
    public let color: SIMD4<Double>?
    /// Metal window depth in 0...1. Use one for forward depth or zero for reversed depth.
    public let depth: Double?
    /// The stored eight-bit stencil value.
    public let stencil: UInt32?

    public init(color: SIMD4<Double>? = nil, depth: Double? = nil, stencil: UInt32? = nil) throws {
        if let color {
            guard (0..<4).allSatisfy({ Float(color[$0]).isFinite }), (0...1).contains(color.w) else {
                throw Failure.color
            }
        }
        if let depth, !(0...1).contains(depth) { throw Failure.depth }
        if let stencil, stencil > UInt8.max { throw Failure.stencil }
        self.color = color
        self.depth = depth
        self.stencil = stencil
    }

    /// Sets the selected load actions and clear values. Other actions, mip
    /// levels and layers stay as authored by the caller. A missing selected
    /// attachment throws before the descriptor changes.
    public func apply(to pass: MTLRenderPassDescriptor) throws {
        if color != nil, !(0..<8).contains(where: { pass.colorAttachments[$0].texture != nil }) {
            throw Failure.missingAttachment("color")
        }
        if depth != nil, pass.depthAttachment.texture == nil { throw Failure.missingAttachment("depth") }
        if stencil != nil, pass.stencilAttachment.texture == nil { throw Failure.missingAttachment("stencil") }

        if let color {
            let value = MTLClearColor(red: color.x, green: color.y, blue: color.z, alpha: color.w)
            for index in 0..<8 where pass.colorAttachments[index].texture != nil {
                let attachment = pass.colorAttachments[index]!
                attachment.clearColor = value
                attachment.loadAction = .clear
            }
        }
        if let depth {
            pass.depthAttachment.clearDepth = depth
            pass.depthAttachment.loadAction = .clear
        }
        if let stencil {
            pass.stencilAttachment.clearStencil = stencil
            pass.stencilAttachment.loadAction = .clear
        }
    }

    /// Encodes a standalone clear. The descriptor remains available for later
    /// draws. Only selected attachments participate, and their store actions
    /// and resolve targets apply. Use an uncommitted retaining command buffer.
    public func encode(
        in targets: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard color != nil || depth != nil || stencil != nil else { return }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard let pass = targets.copy() as? MTLRenderPassDescriptor else { throw Failure.encoder }
        try apply(to: pass)
        pass.visibilityResultBuffer = nil
        func omit(_ attachment: MTLRenderPassAttachmentDescriptor) {
            attachment.texture = nil
            attachment.resolveTexture = nil
        }
        if color == nil { for index in 0..<8 { omit(pass.colorAttachments[index]) } }
        if depth == nil { omit(pass.depthAttachment) }
        if stencil == nil { omit(pass.stencilAttachment) }

        let attachments: [MTLRenderPassAttachmentDescriptor] =
            (0..<8).map { pass.colorAttachments[$0]! } + [pass.depthAttachment, pass.stencilAttachment]
        for attachment in attachments {
            if let texture = attachment.texture, texture.device !== commandBuffer.device { throw Failure.device }
            if let resolve = attachment.resolveTexture, resolve.device !== commandBuffer.device { throw Failure.device }
        }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield framebuffer clear"
        encoder.endEncoding()
    }

    public enum Failure: LocalizedError, Equatable {
        case color
        case depth
        case stencil
        case missingAttachment(String)
        case commandBuffer
        case device
        case encoder

        public var errorDescription: String? {
            switch self {
            case .color: "Use finite Float32-representable clear color components and alpha in 0...1."
            case .depth: "Use a clear depth in 0...1."
            case .stencil: "Use a clear stencil value in 0...255."
            case .missingAttachment(let kind): "The pass has no \(kind) attachment to clear."
            case .commandBuffer: "Use an uncommitted retaining command buffer for the clear."
            case .device: "The clear attachments must use the command buffer's Metal device."
            case .encoder: "Metal could not create the framebuffer clear pass."
            }
        }
    }
}
