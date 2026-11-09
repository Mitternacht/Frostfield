import Metal

/// The frame loop shared by Frostfield's renderers: one command queue, a
/// bounded set of frame slots, and one uniform buffer per slot.
///
/// ``submit(label:target:encode:)`` claims a free slot without waiting, asks
/// for the target only then, checks that the target matches the renderer's
/// pipelines, lets the renderer encode into the slot's uniform buffer, and
/// commits. The slot stays busy until Metal reports the command buffer
/// complete, so a slot's buffer is never written while the GPU may still read
/// it.
final class FrameSubmitter {
    /// The renderer's type name, used in precondition messages.
    let rendererName: String
    let colorPixelFormat: MTLPixelFormat
    /// The depth format the renderer's pipelines test against, or `nil` when
    /// they draw without a depth attachment.
    let depthPixelFormat: MTLPixelFormat?
    let slots: FrameSlots
    /// One uniform buffer per frame slot, indexed by slot.
    let uniformBuffers: [any MTLBuffer]

    private let commandQueue: any MTLCommandQueue
    private let onFrameCompleted: @Sendable (FrameCompletion) -> Void

    /// Creates the command queue and the per-slot uniform buffers, and counts
    /// the buffers in `resources`.
    init(
        device: any MTLDevice,
        rendererName: String,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat?,
        maxFramesInFlight: Int,
        uniformBufferLength: Int,
        resources: inout ResourceStatistics,
        onFrameCompleted: @escaping @Sendable (FrameCompletion) -> Void
    ) throws(RendererError) {
        precondition((1...3).contains(maxFramesInFlight), "maxFramesInFlight must be 1, 2, or 3")
        let label = "Frostfield.\(rendererName)"
        guard let commandQueue = device.makeCommandQueue() else { throw .commandQueueUnavailable }
        commandQueue.label = label

        var uniformBuffers: [any MTLBuffer] = []
        for slot in 0..<maxFramesInFlight {
            guard let buffer = device.makeBuffer(length: uniformBufferLength, options: .storageModeShared) else {
                throw .bufferAllocationFailed
            }
            buffer.label = "\(label).Uniforms[\(slot)]"
            resources.buffersAllocated += 1
            resources.bufferBytesAllocated += buffer.length
            uniformBuffers.append(buffer)
        }

        self.rendererName = rendererName
        self.colorPixelFormat = colorPixelFormat
        self.depthPixelFormat = depthPixelFormat
        self.commandQueue = commandQueue
        self.uniformBuffers = uniformBuffers
        self.slots = FrameSlots(capacity: maxFramesInFlight)
        self.onFrameCompleted = onFrameCompleted
    }

    /// Claims a slot, asks for the target, lets `encode` record the frame,
    /// and commits.
    ///
    /// A `nil` target, or one without a non-empty color texture, releases the
    /// slot unused. A target the renderer's pipelines cannot draw into stops
    /// on a precondition before any command buffer exists (see
    /// ``checkCompatibility(of:colorTexture:)``). `encode` receives the open
    /// encoder, the slot's uniform buffer, the target's pass, and its color
    /// texture; it must not end the encoder.
    func submit(
        label: String,
        target: () -> FrameTarget?,
        encode: (
            _ encoder: any MTLRenderCommandEncoder,
            _ uniformBuffer: any MTLBuffer,
            _ pass: MTLRenderPassDescriptor,
            _ colorTexture: any MTLTexture
        ) -> Void
    ) -> FrameOutcome {
        guard let slot = slots.acquire() else { return .skippedNoSlot }

        guard let target = target(),
              let colorTexture = target.passDescriptor.colorAttachments[0].texture,
              colorTexture.width > 0, colorTexture.height > 0
        else {
            slots.releaseUnusedForMissingTarget(slot)
            return .skippedNoTarget
        }
        checkCompatibility(of: target.passDescriptor, colorTexture: colorTexture)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: target.passDescriptor)
        else {
            slots.releaseUnusedForEncodingFailure(slot)
            return .encodingFailed
        }
        commandBuffer.label = label

        // The slot's previous frame has completed, so the GPU no longer reads
        // this buffer.
        encode(encoder, uniformBuffers[slot], target.passDescriptor, colorTexture)
        encoder.endEncoding()

        if let drawable = target.drawable {
            commandBuffer.present(drawable)
        }

        let frame = slots.markSubmitted(slot)
        let slots = self.slots
        let onFrameCompleted = self.onFrameCompleted
        commandBuffer.addCompletedHandler { buffer in
            let succeeded = buffer.status == .completed
            slots.complete(slot, failed: !succeeded)
            onFrameCompleted(FrameCompletion(frame: frame, succeeded: succeeded))
        }
        commandBuffer.commit()
        return .submitted(frame: frame)
    }

    /// Stops with a message that says how to fix the target when its
    /// attachments do not match what the renderer's pipelines were built for:
    /// the color format, a single sample per pixel, the depth attachment (its
    /// format, or none for a renderer without depth), and no stencil
    /// attachment. Metal API validation aborts on each of these mismatches,
    /// and without validation the result is undefined, so they are caught
    /// before encoding.
    func checkCompatibility(of pass: MTLRenderPassDescriptor, colorTexture: any MTLTexture) {
        precondition(
            colorTexture.pixelFormat == colorPixelFormat,
            "Target pixel format \(colorTexture.pixelFormat.rawValue) does not match the pipeline's \(colorPixelFormat.rawValue)"
        )
        precondition(
            colorTexture.sampleCount == 1,
            "\(rendererName) renders single-sample targets, but the target's color texture has sampleCount "
                + "\(colorTexture.sampleCount). For an MTKView, leave sampleCount at 1."
        )
        let depthFormat = pass.depthAttachment.texture?.pixelFormat
        if let depthPixelFormat {
            precondition(
                depthFormat == depthPixelFormat,
                "\(rendererName) needs a depth attachment of pixel format \(depthPixelFormat.rawValue), but the target has "
                    + (depthFormat.map { "format \($0.rawValue)" } ?? "none")
                    + ". For an MTKView, set depthStencilPixelFormat to the renderer's depthPixelFormat."
            )
        } else {
            precondition(
                depthFormat == nil,
                "\(rendererName) draws without depth, but the target has a depth attachment of pixel format "
                    + "\(depthFormat?.rawValue ?? 0). For an MTKView, set depthStencilPixelFormat to .invalid."
            )
        }
        let stencilFormat = pass.stencilAttachment.texture?.pixelFormat
        precondition(
            stencilFormat == nil,
            "\(rendererName) draws without a stencil attachment, but the target has one of pixel format "
                + "\(stencilFormat?.rawValue ?? 0). For an MTKView, use a depthStencilPixelFormat without stencil, "
                + "such as .depth32Float, or .invalid for a renderer without depth."
        )
    }
}
