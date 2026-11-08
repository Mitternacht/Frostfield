import Metal

/// The conventional (`MTLCommandQueue`) path of the comparison.
///
/// It mirrors Frostfield's `TriangleRenderer` (same shader, pipeline settings,
/// slot policy, and uniform record) and additionally reports each frame's GPU
/// interval, which the shipping renderer does not expose. The pixel-identity
/// tests check that both produce the same image.
final class ConventionalTriangle {
    let device: any MTLDevice
    let colorPixelFormat: MTLPixelFormat
    var statistics: ExperimentStatistics { slots.statistics }

    private let queue: any MTLCommandQueue
    private let pipeline: any MTLRenderPipelineState
    private let uniformBuffers: [any MTLBuffer]
    private let slots: SlotPool
    private let onFrameCompleted: @Sendable (ExperimentFrameCompletion) -> Void

    init(
        device: any MTLDevice,
        library: any MTLLibrary,
        colorPixelFormat: MTLPixelFormat,
        maxFramesInFlight: Int = 2,
        onFrameCompleted: @escaping @Sendable (ExperimentFrameCompletion) -> Void
    ) throws(ExperimentError) {
        guard let queue = device.makeCommandQueue() else { throw .queueUnavailable }
        queue.label = "Experiment.Conventional"
        guard let vertex = library.makeFunction(name: TriangleFunctions.vertex) else {
            throw .shaderFunctionMissing(name: TriangleFunctions.vertex)
        }
        guard let fragment = library.makeFunction(name: TriangleFunctions.fragment) else {
            throw .shaderFunctionMissing(name: TriangleFunctions.fragment)
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Experiment.Conventional.Triangle"
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        do {
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw .pipelineCreationFailed(reason: String(describing: error))
        }
        var buffers: [any MTLBuffer] = []
        for slot in 0..<maxFramesInFlight {
            guard let buffer = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared) else {
                throw .bufferAllocationFailed
            }
            buffer.label = "Experiment.Conventional.Uniforms[\(slot)]"
            buffers.append(buffer)
        }
        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.queue = queue
        self.uniformBuffers = buffers
        self.slots = SlotPool(capacity: maxFramesInFlight)
        self.onFrameCompleted = onFrameCompleted
    }

    /// Encodes and commits one frame into the texture `target(slot)` returns.
    /// The slot index lets a host give every in-flight frame its own target.
    func drawFrame(rotation: Float, target: (_ slot: Int) -> (any MTLTexture)?) -> ExperimentFrameOutcome {
        guard let slot = slots.acquire() else { return .skippedNoSlot }
        guard let texture = target(slot), texture.width > 0, texture.height > 0 else {
            slots.releaseUnused(slot, missingTarget: true)
            return .skippedNoTarget
        }
        precondition(texture.pixelFormat == colorPixelFormat)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)
        else {
            slots.releaseUnused(slot, missingTarget: false)
            return .encodingFailed
        }
        commandBuffer.label = "Experiment.Conventional.Frame"

        let buffer = uniformBuffers[slot]
        buffer.contents().storeBytes(
            of: triangleUniforms(width: texture.width, height: texture.height, rotation: rotation),
            as: SIMD4<Float>.self
        )
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(buffer, offset: 0, index: TriangleFunctions.uniformsBufferIndex)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        let frame = slots.markSubmitted(slot)
        let slots = self.slots
        let onFrameCompleted = self.onFrameCompleted
        // A conventional command buffer retains the pipeline, buffer, and
        // texture it references until it completes; nothing else is needed to
        // keep them alive.
        commandBuffer.addCompletedHandler { buffer in
            let succeeded = buffer.status == .completed
            slots.complete(slot, failed: !succeeded)
            onFrameCompleted(ExperimentFrameCompletion(
                frame: frame,
                succeeded: succeeded,
                gpuSeconds: buffer.gpuEndTime - buffer.gpuStartTime
            ))
        }
        commandBuffer.commit()
        return .submitted(frame: frame)
    }
}
