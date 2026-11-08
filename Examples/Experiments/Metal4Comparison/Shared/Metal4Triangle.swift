// The iOS Simulator SDK declares no Metal 4 types (its MTL4 headers are
// empty), so this path exists only in device and macOS builds. A shipping
// backend would need the same compile-time exclusion plus the runtime family
// check in `TriangleBackend.select`.
#if !targetEnvironment(simulator)
import Metal

/// The Metal 4 path of the comparison: the same triangle, shader source,
/// pipeline settings, slot policy, and uniform record as `ConventionalTriangle`,
/// submitted through `MTL4CommandQueue`.
///
/// What Metal 4 moves from Metal to the app, and how this type handles it:
/// - **Command memory.** One `MTL4CommandAllocator` per frame slot. An
///   allocator is reset only when `SlotPool` hands its slot out again, which
///   happens only after the commit feedback for the slot's previous frame.
/// - **Command buffer.** One `MTL4CommandBuffer`, reused for every frame; Metal
///   allows reuse immediately after commit because the allocator holds the
///   encoded commands.
/// - **Bindings.** One `MTL4ArgumentTable` holding the GPU address of the
///   slot's uniform buffer. Metal snapshots the table at each draw, so the
///   table can be rewritten while earlier frames are in flight.
/// - **Residency.** Nothing is resident implicitly. The uniform buffers live in
///   a queue-level residency set; render targets must be added through
///   `makeResident(_:)` before a frame uses them.
/// - **Lifetime.** `MTL4CommandBuffer` does not retain the resources it uses.
///   The allocators, buffers, and targets must outlive every in-flight frame.
///   Swift 6 cannot capture these non-`Sendable` Metal objects in the
///   `@Sendable` commit-feedback handler, so this type cannot extend their
///   lifetime from the handler; releasing it with frames in flight is a
///   programming error and traps in `deinit`. Hosts drain first.
/// - **Synchronization.** Completion and GPU timing come from the commit
///   feedback handler. No CPU wait: a frame without a free slot is skipped.
///   There is no automatic hazard tracking between passes, so each in-flight
///   frame renders into its own target (`target(slot)`).
final class Metal4Triangle {
    let device: any MTLDevice
    let colorPixelFormat: MTLPixelFormat
    var statistics: ExperimentStatistics { slots.statistics }

    /// Total bytes the command allocators currently reserve for encoding.
    var allocatorBytes: UInt64 { allocators.reduce(0) { $0 + $1.allocatedSize() } }

    private let queue: any MTL4CommandQueue
    private let commandBuffer: any MTL4CommandBuffer
    private let allocators: [any MTL4CommandAllocator]
    private let uniformBuffers: [any MTLBuffer]
    private let argumentTable: any MTL4ArgumentTable
    private let bufferResidency: any MTLResidencySet
    private let targetResidency: any MTLResidencySet
    private let pipeline: any MTLRenderPipelineState
    private let slots: SlotPool
    private let onFrameCompleted: @Sendable (ExperimentFrameCompletion) -> Void

    init(
        device: any MTLDevice,
        library: any MTLLibrary,
        colorPixelFormat: MTLPixelFormat,
        maxFramesInFlight: Int = 2,
        onFrameCompleted: @escaping @Sendable (ExperimentFrameCompletion) -> Void
    ) throws(ExperimentError) {
        guard TriangleBackend.select(device: device) == .metal4 else { throw .metal4Unsupported }
        guard let queue = device.makeMTL4CommandQueue() else { throw .queueUnavailable }
        guard let commandBuffer = device.makeCommandBuffer() else { throw .commandBufferUnavailable }

        var allocators: [any MTL4CommandAllocator] = []
        var buffers: [any MTLBuffer] = []
        for slot in 0..<maxFramesInFlight {
            guard let allocator = device.makeCommandAllocator() else { throw .allocatorUnavailable }
            allocators.append(allocator)
            guard let buffer = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared) else {
                throw .bufferAllocationFailed
            }
            buffer.label = "Experiment.Metal4.Uniforms[\(slot)]"
            buffers.append(buffer)
        }

        // Metal 4 compiles pipelines through an explicit compiler object and
        // names functions with descriptors. A missing function would surface
        // as a generic compiler error, so check names first to keep the same
        // typed failure as the conventional path.
        for name in [TriangleFunctions.vertex, TriangleFunctions.fragment] where !library.functionNames.contains(name) {
            throw .shaderFunctionMissing(name: name)
        }
        let vertex = MTL4LibraryFunctionDescriptor()
        vertex.library = library
        vertex.name = TriangleFunctions.vertex
        let fragment = MTL4LibraryFunctionDescriptor()
        fragment.library = library
        fragment.name = TriangleFunctions.fragment
        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.label = "Experiment.Metal4.Triangle"
        pipelineDescriptor.vertexFunctionDescriptor = vertex
        pipelineDescriptor.fragmentFunctionDescriptor = fragment
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        do {
            let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
            pipeline = try compiler.makeRenderPipelineState(descriptor: pipelineDescriptor, compilerTaskOptions: nil)
        } catch {
            throw .pipelineCreationFailed(reason: String(describing: error))
        }

        let tableDescriptor = MTL4ArgumentTableDescriptor()
        tableDescriptor.maxBufferBindCount = TriangleFunctions.uniformsBufferIndex + 1
        tableDescriptor.label = "Experiment.Metal4.Arguments"
        do {
            argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)
        } catch {
            throw .argumentTableUnavailable(reason: String(describing: error))
        }

        do {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "Experiment.Metal4.Buffers"
            bufferResidency = try device.makeResidencySet(descriptor: descriptor)
            descriptor.label = "Experiment.Metal4.Targets"
            targetResidency = try device.makeResidencySet(descriptor: descriptor)
        } catch {
            throw .residencySetUnavailable(reason: String(describing: error))
        }
        for buffer in buffers { bufferResidency.addAllocation(buffer) }
        bufferResidency.commit()
        targetResidency.commit()
        queue.addResidencySet(bufferResidency)
        queue.addResidencySet(targetResidency)

        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.queue = queue
        self.commandBuffer = commandBuffer
        self.allocators = allocators
        self.uniformBuffers = buffers
        self.slots = SlotPool(capacity: maxFramesInFlight)
        self.onFrameCompleted = onFrameCompleted
    }

    deinit {
        // See "Lifetime" above: the GPU may still read memory this object
        // owns. Crashing here is the honest outcome for that misuse; a
        // shipping backend needs a deferred-release owner instead.
        precondition(
            slots.statistics.inFlight == 0,
            "Metal4Triangle released with \(slots.statistics.inFlight) frames in flight; MTL4 command buffers do not retain resources"
        )
    }

    /// Adds a render target to the queue's residency. A Metal 4 frame may use
    /// only resident allocations; call this before the first frame that
    /// renders into `texture`, not per frame.
    func makeResident(_ texture: any MTLTexture) {
        guard !targetResidency.containsAllocation(texture) else { return }
        targetResidency.addAllocation(texture)
        targetResidency.commit()
    }

    /// Removes render targets from residency. Only valid once no in-flight
    /// frame renders into them; this experiment requires an idle renderer.
    func evictTargets() {
        precondition(statistics.inFlight == 0, "Evicting targets while frames are in flight")
        targetResidency.removeAllAllocations()
        targetResidency.commit()
    }

    /// Encodes and commits one frame into the texture `target(slot)` returns.
    func drawFrame(rotation: Float, target: (_ slot: Int) -> (any MTLTexture)?) -> ExperimentFrameOutcome {
        guard let slot = slots.acquire() else { return .skippedNoSlot }
        guard let texture = target(slot), texture.width > 0, texture.height > 0 else {
            slots.releaseUnused(slot, missingTarget: true)
            return .skippedNoTarget
        }
        precondition(texture.pixelFormat == colorPixelFormat)
        precondition(targetResidency.containsAllocation(texture), "Render target is not resident; call makeResident(_:) first")

        // The slot's previous frame has completed (SlotPool only hands out
        // completed slots), so no command buffer still uses this memory.
        let allocator = allocators[slot]
        allocator.reset()
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.label = "Experiment.Metal4.Frame"

        let pass = MTL4RenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass, options: []) else {
            commandBuffer.endCommandBuffer()
            slots.releaseUnused(slot, missingTarget: false)
            return .encodingFailed
        }

        let buffer = uniformBuffers[slot]
        buffer.contents().storeBytes(
            of: triangleUniforms(width: texture.width, height: texture.height, rotation: rotation),
            as: SIMD4<Float>.self
        )
        argumentTable.setAddress(buffer.gpuAddress, index: TriangleFunctions.uniformsBufferIndex)
        encoder.setRenderPipelineState(pipeline)
        encoder.setArgumentTable(argumentTable, stages: .vertex)
        encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()

        let frame = slots.markSubmitted(slot)
        let slots = self.slots
        let onFrameCompleted = self.onFrameCompleted
        let options = MTL4CommitOptions()
        options.addFeedbackHandler { feedback in
            let succeeded = feedback.error == nil
            slots.complete(slot, failed: !succeeded)
            onFrameCompleted(ExperimentFrameCompletion(
                frame: frame,
                succeeded: succeeded,
                gpuSeconds: feedback.gpuEndTime - feedback.gpuStartTime
            ))
        }
        queue.commit([commandBuffer], options: options)
        return .submitted(frame: frame)
    }
}
#endif
