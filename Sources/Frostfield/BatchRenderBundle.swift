import Metal

/// Indexed commands refer to a palette's immutable index buffer. The parent
/// encoder binds the view, instance records, material, and render pipeline.
struct BatchRenderBundle {
    static let maximumRuns = 16_384

    private struct Span {
        let start: Int
        var count: Int
        let mirrored: Bool
    }

    private let commands: any MTLIndirectCommandBuffer
    private let indices: any MTLBuffer
    private let spans: [Span]
    var executionCount: Int { spans.count }

    static func supported(on device: any MTLDevice) -> Bool {
        guard device.supportsFamily(.apple3) || device.supportsFamily(.apple7) else { return false }
        return device.makeIndirectCommandBuffer(descriptor: descriptor(), maxCommandCount: 1, options: []) != nil
    }

    private static func descriptor() -> MTLIndirectCommandBufferDescriptor {
        let descriptor = MTLIndirectCommandBufferDescriptor()
        descriptor.commandTypes = .drawIndexed
        descriptor.inheritBuffers = true
        descriptor.inheritPipelineState = true
        return descriptor
    }

    init(indices: any MTLBuffer, runs: [BatchedMesh.Run]) throws {
        guard (1...Self.maximumRuns).contains(runs.count) else { throw BatchedMesh.Failure.indirectCommandBudget }
        guard let commands = indices.device.makeIndirectCommandBuffer(
            descriptor: Self.descriptor(), maxCommandCount: runs.count, options: []
        ) else { throw BatchedMesh.Failure.indirectResource }
        commands.label = "Frostfield batched indexed commands"
        var spans: [Span] = []
        for (offset, run) in runs.enumerated() {
            let command = commands.indirectRenderCommandAt(offset)
            command.drawIndexedPrimitives(
                .triangle, indexCount: run.indices.count, indexType: .uint32,
                indexBuffer: indices, indexBufferOffset: run.indices.lowerBound * MemoryLayout<UInt32>.stride,
                instanceCount: run.instanceCount, baseVertex: 0, baseInstance: run.firstInstance
            )
            // Keep the existing order across winding changes, including blends.
            if let last = spans.last, last.mirrored == run.mirrored {
                spans[spans.count - 1].count += 1
            } else {
                spans.append(Span(start: offset, count: 1, mirrored: run.mirrored))
            }
        }
        self.commands = commands
        self.indices = indices
        self.spans = spans
    }

    func encode(into encoder: any MTLRenderCommandEncoder) {
        encoder.useResource(indices, usage: .read, stages: .vertex)
        for span in spans {
            encoder.setFrontFacing(span.mirrored ? .clockwise : .counterClockwise)
            encoder.executeCommandsInBuffer(commands, range: span.start..<(span.start + span.count))
        }
    }
}

/// Metal indirect pipelines access the textured mesh's resources through an
/// argument buffer. The renderer retains the texture and sampler for the ICB.
struct BatchTextureArguments {
    private let buffer: any MTLBuffer
    private let texture: any MTLTexture
    private let sampler: any MTLSamplerState

    func recordResources(in inventory: inout SceneResourceInventory) {
        inventory.recordBuffer(buffer)
        inventory.recordTexture(texture)
    }

    init(texture: any MTLTexture, fragment: any MTLFunction) throws {
        let device = texture.device
        let arguments = fragment.makeArgumentEncoder(bufferIndex: 1)
        guard let buffer = device.makeBuffer(length: arguments.encodedLength, options: .storageModeShared) else {
            throw MeshRenderer.Failure.resource("indirect texture arguments")
        }
        buffer.label = "Frostfield indirect texture arguments"
        let sampler = try TextureSampling(supportsArgumentBuffers: true).makeSamplerState(device: device)
        arguments.setArgumentBuffer(buffer, offset: 0)
        arguments.setTexture(texture, index: 0)
        arguments.setSamplerState(sampler, index: 1)
        self.buffer = buffer
        self.texture = texture
        self.sampler = sampler
    }

    func encode(into encoder: any MTLRenderCommandEncoder) {
        encoder.useResource(texture, usage: .read, stages: .fragment)
        encoder.setFragmentBuffer(buffer, offset: 0, index: 1)
    }
}
