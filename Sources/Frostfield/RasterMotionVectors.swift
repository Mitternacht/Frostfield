import Foundation
import Metal
import simd

/// Records indexed mesh motion from the transforms and poses used by raster color.
/// The RGBA16Float output stores previous minus current in top-left pixels, the
/// previous stored depth in B, and validity in A. Pixels without history are zero.
public final class RasterMotionVectors {
    public static let pixelFormat: MTLPixelFormat = .rgba16Float
    /// Total retained GPU vertex bytes per recorded view. Shared meshes count once.
    public static let maximumVertexHistoryBytes = 256 * 1024 * 1024
    public let queue: any MTLCommandQueue
    private let library: any MTLLibrary
    private let fragment: any MTLFunction
    private let depthState: any MTLDepthStencilState
    private var pipelines: [MeshRenderer.VertexSpecialization: any MTLRenderPipelineState] = [:]

    public init(queue: any MTLCommandQueue) throws {
        self.queue = queue
        let device = queue.device
        library = try device.makeDefaultLibrary(bundle: .module)
        guard let fragment = library.makeFunction(name: "frostfield_raster_motion_fragment") else { throw Failure.pipeline }
        self.fragment = fragment
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = .equal
        guard let state = device.makeDepthStencilState(descriptor: depth) else { throw Failure.pipeline }
        depthState = state
        _ = try pipeline(for: .init(skinned: false, morphed: false, instanced: false))
        _ = try pipeline(for: .init(skinned: false, morphed: false, instanced: true))
    }

    private func pipeline(for specialization: MeshRenderer.VertexSpecialization) throws -> any MTLRenderPipelineState {
        if let cached = pipelines[specialization] { return cached }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield raster motion vectors"
        descriptor.vertexFunction = try specialization.function("frostfield_raster_motion_vertex", in: library)
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = Self.pixelFormat
        descriptor.depthAttachmentPixelFormat = .depth32Float
        let prepared = try queue.device.makeRenderPipelineState(descriptor: descriptor)
        pipelines[specialization] = prepared
        return prepared
    }

    /// Retains the prior view and the geometry identities used for its vectors.
    /// Keep one history per view. Submit GPU vertex captures on the same queue
    /// in history order and discard the history of any failed or abandoned frame.
    public struct History {
        fileprivate let device: any MTLDevice
        fileprivate let vertexQueue: (any MTLCommandQueue)?
        fileprivate let size: SIMD2<Int>
        fileprivate let viewport: SIMD4<Float>
        fileprivate let depthRange: SIMD2<Float>
        fileprivate let entries: [UUID: Entry]
    }

    /// A single scene encode's motion destination and optional previous frame.
    /// `history` becomes available after the color and motion work has been encoded.
    public final class Recording {
        public let destination: any MTLTexture
        public let previous: History?
        public fileprivate(set) var history: History?
        fileprivate let owner: RasterMotionVectors

        fileprivate init(owner: RasterMotionVectors, destination: any MTLTexture, previous: History?) {
            self.owner = owner
            self.destination = destination
            self.previous = previous
        }

        func validate(
            pass: MTLRenderPassDescriptor, size: SIMD2<Int>, depthMode: MeshDepthMode, commandBuffer: any MTLCommandBuffer
        ) throws {
            guard history == nil else { throw Failure.usedRecording }
            guard commandBuffer.commandQueue === owner.queue, commandBuffer.retainedReferences,
                  commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
                throw Failure.commandBuffer
            }
            guard destination.device === owner.queue.device,
                  PostProcessPass.isReadable(destination, format: RasterMotionVectors.pixelFormat),
                  destination.usage.contains(.renderTarget), destination.width == size.x, destination.height == size.y else {
                throw Failure.destination
            }
            let depth = pass.depthAttachment!
            guard depthMode != .logarithmic, pass.rasterizationRateMap == nil, let texture = depth.texture,
                  depth.level == 0, depth.slice == 0, depth.depthPlane == 0, depth.resolveTexture == nil,
                  depth.storeAction == .store, texture.device === owner.queue.device, texture.textureType == .type2D,
                  texture.sampleCount == 1, texture.pixelFormat == .depth32Float,
                  texture.storageMode != .memoryless, texture.hazardTrackingMode != .untracked,
                  texture.width == size.x, texture.height == size.y else { throw Failure.depth }
            var attachments: [MTLRenderPassAttachmentDescriptor] = (0..<8).map { pass.colorAttachments[$0]! }
            attachments.append(depth)
            attachments.append(pass.stencilAttachment)
            guard attachments.allSatisfy({ attachment in
                [attachment.texture, attachment.resolveTexture].compactMap { $0 }.allSatisfy {
                    !TextureTransfer.sharesStorage(destination, $0)
                }
            }) else { throw Failure.destination }
            if let previous {
                guard previous.device === owner.queue.device, previous.size == size,
                      previous.vertexQueue.map({ $0 === owner.queue }) ?? true else { throw Failure.history }
            }
        }

        func prepare(draws: [Draw], size: SIMD2<Int>, viewport: MTLViewport?, depth: any MTLTexture) throws -> Prepared {
            let (entries, vertexCopies) = try vertexHistory(for: draws)
            let frame = History(
                device: owner.queue.device, vertexQueue: vertexCopies.isEmpty ? nil : owner.queue, size: size,
                viewport: viewport.map { SIMD4(Float($0.originX), Float($0.originY), Float($0.width), Float($0.height)) }
                    ?? SIMD4(0, 0, Float(size.x), Float(size.y)),
                depthRange: viewport.map { SIMD2(Float($0.znear), Float($0.zfar)) } ?? SIMD2(0, 1),
                entries: entries
            )
            var packets: [Packet] = []
            if let previous {
                for draw in draws {
                    guard let prior = previous.entries[draw.id], prior.renderer === draw.entry.renderer,
                          prior.indexRange == draw.entry.indexRange,
                          prior.topologyRevision == draw.entry.topologyRevision else { continue }
                    var instances: (buffer: any MTLBuffer, previousOffset: Int)?
                    switch (draw.entry.instances, prior.instances) {
                    case (nil, nil): break
                    case (let current?, let old?) where current.owner === old.owner && current.layout == old.layout:
                        guard current.transforms.count == old.transforms.count else { continue }
                        instances = try Self.packInstances(current: current.transforms, previous: old.transforms, device: owner.queue.device)
                    default: continue
                    }
                    let specialization = MeshRenderer.VertexSpecialization(
                        skinned: draw.skinInfluences != nil, morphed: draw.morphOffsets != nil, instanced: instances != nil
                    )
                    packets.append(Packet(
                        draw: draw, previous: prior, instances: instances, pipeline: try owner.pipeline(for: specialization)
                    ))
                }
            }
            let descriptor = MTLRenderPassDescriptor()
            descriptor.colorAttachments[0].texture = destination
            descriptor.colorAttachments[0].loadAction = .clear
            descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
            descriptor.colorAttachments[0].storeAction = .store
            descriptor.depthAttachment.texture = depth
            descriptor.depthAttachment.loadAction = .load
            descriptor.depthAttachment.storeAction = .store
            return Prepared(
                recording: self, frame: frame, viewport: viewport, descriptor: descriptor,
                vertexCopies: vertexCopies, packets: packets
            )
        }

        private func vertexHistory(for draws: [Draw]) throws -> ([UUID: Entry], [VertexCopy]) {
            var entries: [UUID: Entry] = [:]
            entries.reserveCapacity(draws.count)
            var copies: [VertexCopy] = []
            var snapshots: [ObjectIdentifier: any MTLBuffer] = [:]
            var remainingBytes = RasterMotionVectors.maximumVertexHistoryBytes
            for draw in draws {
                var entry = draw.entry
                if let length = draw.vertexSnapshotLength {
                    let source = draw.vertices
                    let key = ObjectIdentifier(source)
                    if let cached = snapshots[key] {
                        entry.vertexSnapshot = cached
                    } else {
                        guard length <= remainingBytes else { throw Failure.vertexHistoryBudget }
                        guard let snapshot = owner.queue.device.makeBuffer(length: length, options: .storageModePrivate)
                        else { throw Failure.vertexHistoryBuffer }
                        snapshot.label = "Frostfield raster motion vertex history"
                        remainingBytes -= length
                        snapshots[key] = snapshot
                        copies.append(VertexCopy(source: source, snapshot: snapshot))
                        entry.vertexSnapshot = snapshot
                    }
                }
                entries[draw.id] = entry
            }
            return (entries, copies)
        }

        private static func packInstances(
            current: [simd_float4x4], previous: [simd_float4x4], device: any MTLDevice
        ) throws -> (buffer: any MTLBuffer, previousOffset: Int) {
            let offset = current.count * MemoryLayout<InstanceSet.PackedInstance>.stride
            guard offset > 0, offset <= device.maxBufferLength / 2,
                  let buffer = device.makeBuffer(length: offset * 2, options: .storageModeShared) else {
                throw Failure.instanceBuffer
            }
            buffer.label = "Frostfield current and previous raster motion instances"
            let records = buffer.contents().bindMemory(to: InstanceSet.PackedInstance.self, capacity: current.count * 2)
            for index in current.indices {
                records[index] = .init(transform: current[index], tint: SIMD4(repeating: 1))
                records[current.count + index] = .init(transform: previous[index], tint: SIMD4(repeating: 1))
            }
            return (buffer, offset)
        }
    }

    public func recording(to destination: any MTLTexture, previous: History? = nil) -> Recording {
        Recording(owner: self, destination: destination, previous: previous)
    }

    struct Instances {
        let owner: InstanceSet
        let layout: UUID
        let transforms: [simd_float4x4]
    }

    struct Entry {
        let renderer: MeshRenderer
        let clip: simd_float4x4
        let indexRange: Range<Int>
        let topologyRevision: UInt64?
        let instances: Instances?
        let palette: MeshRenderer.PreparedPalette?
        let morph: MeshRenderer.MorphDraw?
        var vertexSnapshot: (any MTLBuffer)? = nil
    }

    struct Draw {
        let id: UUID
        let entry: Entry
        let vertices: any MTLBuffer
        let vertexSnapshotLength: Int?
        let indices: any MTLBuffer
        let skinInfluences: (any MTLBuffer)?
        let morphOffsets: (any MTLBuffer)?
        let winding: MTLWinding
        let cull: MTLCullMode
    }

    fileprivate struct VertexCopy {
        let source: any MTLBuffer
        let snapshot: any MTLBuffer
    }

    fileprivate struct Packet {
        let draw: Draw
        let previous: Entry
        let instances: (buffer: any MTLBuffer, previousOffset: Int)?
        let pipeline: any MTLRenderPipelineState
    }

    struct Prepared {
        fileprivate let recording: Recording
        fileprivate let frame: History
        fileprivate let viewport: MTLViewport?
        fileprivate let descriptor: MTLRenderPassDescriptor
        fileprivate let vertexCopies: [VertexCopy]
        fileprivate let packets: [Packet]

        func encode(commandBuffer: any MTLCommandBuffer, scissor: MTLScissorRect?) throws {
            if !vertexCopies.isEmpty {
                guard let copyEncoder = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
                copyEncoder.label = "Frostfield raster motion vertex history"
                for copy in vertexCopies {
                    copyEncoder.copy(
                        from: copy.source, sourceOffset: 0, to: copy.snapshot, destinationOffset: 0, size: copy.snapshot.length
                    )
                }
                copyEncoder.endEncoding()
            }
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { throw Failure.encoder }
            encoder.label = "Frostfield raster motion vectors"
            if let viewport { encoder.setViewport(viewport) }
            if let scissor, scissor.width > 0, scissor.height > 0 { encoder.setScissorRect(scissor) }
            encoder.setDepthStencilState(recording.owner.depthState)
            if let previous = recording.previous {
                var viewport = previous.viewport
                var depthRange = previous.depthRange
                encoder.setFragmentBytes(&viewport, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                encoder.setFragmentBytes(&depthRange, length: MemoryLayout<SIMD2<Float>>.stride, index: 1)
            }
            for packet in packets {
                let draw = packet.draw
                encoder.setRenderPipelineState(packet.pipeline)
                encoder.setCullMode(draw.cull)
                encoder.setFrontFacing(draw.winding)
                encoder.setVertexBuffer(draw.vertices, offset: 0, index: 0)
                encoder.setVertexBuffer(packet.previous.vertexSnapshot ?? draw.vertices, offset: 0, index: 15)
                var transforms = (draw.entry.clip, packet.previous.clip)
                encoder.setVertexBytes(&transforms, length: MemoryLayout.size(ofValue: transforms), index: 1)
                if let influences = draw.skinInfluences {
                    encoder.setVertexBuffer(influences, offset: 0, index: 4)
                    draw.entry.palette?.bind(into: encoder)
                    packet.previous.palette?.bind(into: encoder, index: 11)
                }
                if let instances = packet.instances {
                    encoder.setVertexBuffer(instances.buffer, offset: 0, index: 6)
                    encoder.setVertexBuffer(instances.buffer, offset: instances.previousOffset, index: 12)
                }
                if let offsets = draw.morphOffsets {
                    encoder.setVertexBuffer(offsets, offset: 0, index: 7)
                    draw.entry.morph?.bind(into: encoder)
                    packet.previous.morph?.bind(into: encoder, weightsIndex: 13, layoutIndex: 14)
                }
                let range = draw.entry.indexRange
                encoder.drawIndexedPrimitives(
                    type: .triangle, indexCount: range.count, indexType: .uint32,
                    indexBuffer: draw.indices, indexBufferOffset: range.lowerBound * MemoryLayout<UInt32>.stride,
                    instanceCount: draw.entry.instances?.transforms.count ?? 1
                )
            }
            encoder.endEncoding()
            recording.history = frame
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case pipeline, usedRecording, commandBuffer, destination, depth, history, instanceBuffer, encoder
        case vertexHistoryBudget, vertexHistoryBuffer

        public var errorDescription: String? {
            switch self {
            case .pipeline: "Metal could not prepare the raster motion pipeline."
            case .usedRecording: "Create a fresh motion recording for each scene encode."
            case .commandBuffer: "Use an uncommitted, retaining command buffer on the motion pass's queue."
            case .destination: "Use a separate tracked RGBA16Float motion target matching the scene extent."
            case .depth: "Store the scene's matching single-sample forward or reversed Depth32Float attachment."
            case .history: "Motion history needs the current attachment extent and Metal device. Vertex history also needs the same command queue."
            case .instanceBuffer: "Metal could not allocate raster motion instance transforms."
            case .encoder: "Metal could not create a raster motion or history copy encoder."
            case .vertexHistoryBudget: "Raster motion vertex history supports up to 256 MiB per recorded view."
            case .vertexHistoryBuffer: "Metal could not allocate the raster motion vertex history."
            }
        }
    }
}
