import Metal

/// The frame loop shared by Frostfield's renderers: one command queue, a
/// bounded set of frame slots, and one uniform buffer per slot.
///
/// ``submit(label:target:encode:)`` claims a free slot without waiting, asks
/// for the target only then, lets the renderer encode into the slot's uniform
/// buffer, and commits. The slot stays busy until Metal reports the command
/// buffer complete, so a slot's buffer is never written while the GPU may
/// still read it.
final class FrameSubmitter {
    let colorPixelFormat: MTLPixelFormat
    let slots: FrameSlots
    /// One uniform buffer per frame slot, indexed by slot.
    let uniformBuffers: [any MTLBuffer]

    private let commandQueue: any MTLCommandQueue
    private let onFrameCompleted: @Sendable (FrameCompletion) -> Void

    /// Creates the command queue and the per-slot uniform buffers, and counts
    /// the buffers in `resources`.
    init(
        device: any MTLDevice,
        label: String,
        colorPixelFormat: MTLPixelFormat,
        maxFramesInFlight: Int,
        uniformBufferLength: Int,
        resources: inout ResourceStatistics,
        onFrameCompleted: @escaping @Sendable (FrameCompletion) -> Void
    ) throws(RendererError) {
        precondition((1...3).contains(maxFramesInFlight), "maxFramesInFlight must be 1, 2, or 3")
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

        self.colorPixelFormat = colorPixelFormat
        self.commandQueue = commandQueue
        self.uniformBuffers = uniformBuffers
        self.slots = FrameSlots(capacity: maxFramesInFlight)
        self.onFrameCompleted = onFrameCompleted
    }

    /// Claims a slot, asks for the target, lets `encode` record the frame,
    /// and commits.
    ///
    /// A `nil` target, or one without a non-empty color texture, releases the
    /// slot unused. `encode` receives the open encoder, the slot's uniform
    /// buffer, the target's pass, and its color texture; it must not end the
    /// encoder.
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
        precondition(
            colorTexture.pixelFormat == colorPixelFormat,
            "Target pixel format \(colorTexture.pixelFormat.rawValue) does not match the pipeline's \(colorPixelFormat.rawValue)"
        )

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
}
