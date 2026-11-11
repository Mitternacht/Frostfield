import Foundation
import Metal
import Synchronization
import simd

/// A fixed-topology batch of independently movable, curve-deformed mesh copies.
///
/// A successful edit updates the CPU geometry and bounds. A renderer made with
/// `MeshRenderer.init(curveFlow:colorPixelFormat:material:depthMode:supportsInstancing:renderState:sampleCount:usesStencilAttachment:)`
/// updates changed vertices without rebuilding its indices or material pipeline.
/// Use the flow and its renderers on one thread at a time. Encoded draws keep
/// their own vertex snapshot until their command buffer completes.
public final class CurveFlow {
    public enum Execution: Equatable, Sendable {
        /// Bake on the CPU and upload changed render vertices.
        case cpu
        /// Evaluate prepared frames in Metal. CPU geometry stays available
        /// for bounds, picking and export. Placements use the initial spines.
        case metal
    }

    public let execution: Execution
    public let device: any MTLDevice
    public let source: MeshGeometry
    public let sourceXOrigin: Double
    public let distanceScale: Double
    public let indexRanges: [Range<Int>]
    public private(set) var placements: [CurveDeformer.Placement]
    public private(set) var geometry: MeshGeometry
    public private(set) var bounds: AxisAlignedBox
    public private(set) var revision: UInt64 = 0

    private static let maximumBuffers = 3
    private var slots: [Slot] = []
    private var pickingBVH: MeshBVH?
    private let metal: CurveFlowMetal?
    private var metalPlacements: [CurveFlowMetal.Placement]

    public init(
        device: any MTLDevice, mesh: MeshGeometry, placements: [CurveDeformer.Placement],
        sourceXOrigin: Double = 0, distanceScale: Double = 1, execution: Execution = .cpu
    ) throws {
        let result = try CurveDeformer.deform(
            mesh, placements: placements, sourceXOrigin: sourceXOrigin, distanceScale: distanceScale
        )
        guard result.geometry.vertices.count <= device.maxBufferLength / MemoryLayout<PackedMeshVertex>.stride else {
            throw Failure.resource
        }
        let metal = execution == .metal ? try CurveFlowMetal(
            device: device, mesh: mesh, placements: placements,
            sourceXOrigin: sourceXOrigin, distanceScale: distanceScale
        ) : nil
        self.metal = metal
        self.metalPlacements = try metal.map { writer in try placements.map(writer.packed) } ?? []
        self.execution = execution
        self.device = device
        self.source = mesh
        self.placements = placements
        self.sourceXOrigin = sourceXOrigin
        self.distanceScale = distanceScale
        self.indexRanges = result.indexRanges
        self.geometry = result.geometry
        self.bounds = result.geometry.bounds
    }

    /// Moves a copy to an absolute signed offset on its current spine.
    public func setOffset(_ offset: Double, at index: Int) throws {
        guard placements.indices.contains(index) else { throw Failure.index(index) }
        guard placements[index].offset != offset else { return }
        var placement = placements[index]
        placement.offset = offset
        try setPlacement(placement, at: index)
    }

    /// Advances a copy by a signed distance on its current spine.
    public func move(along distance: Double, at index: Int) throws {
        guard placements.indices.contains(index) else { throw Failure.index(index) }
        try setOffset(placements[index].offset + distance, at: index)
    }

    /// Changes a copy's curve and offset together, preserving its topology.
    /// Only the selected copy's curve deformation is recalculated.
    public func setPlacement(_ placement: CurveDeformer.Placement, at index: Int) throws {
        guard placements.indices.contains(index) else { throw Failure.index(index) }
        let copy = try placement.curve.deform(
            source, offset: placement.offset, sourceXOrigin: sourceXOrigin, distanceScale: distanceScale
        )
        let first = index * source.vertices.count
        var vertices = geometry.vertices
        vertices.replaceSubrange(first..<(first + copy.vertices.count), with: copy.vertices)
        let next = try MeshGeometry(vertices: vertices, indices: geometry.indices).withGroups(geometry.groups)
        if let metal { metalPlacements[index] = try metal.packed(placement) }
        placements[index] = placement
        publish(next)
    }

    /// Changes all placements atomically. The prepared copy count stays fixed.
    public func replacePlacements(_ placements: [CurveDeformer.Placement]) throws {
        guard placements.count == self.placements.count else { throw Failure.placementCount }
        let next = try CurveDeformer.deform(
            source, placements: placements, sourceXOrigin: sourceXOrigin, distanceScale: distanceScale
        )
        if let metal { metalPlacements = try placements.map(metal.packed) }
        self.placements = placements
        publish(next.geometry)
    }

    private func publish(_ next: MeshGeometry) {
        geometry = next
        bounds = next.bounds
        pickingBVH = nil
        revision &+= 1
    }

    var cullingBounds: AxisAlignedBox {
        metal?.cullingBounds(bounds, placements: metalPlacements) ?? bounds
    }

    func pickingAcceleration() throws -> MeshBVH {
        let bvh = try pickingBVH ?? MeshBVH(geometry: geometry)
        pickingBVH = bvh
        return bvh
    }

    /// The first renderer has already allocated a buffer of the right size.
    func adoptInitialBuffer(_ buffer: any MTLBuffer) {
        guard slots.isEmpty else { return }
        slots.append(Slot(buffer: buffer, content: metal == nil ? .ready(revision) : .empty))
    }

    func acquireVertices(for commandBuffer: any MTLCommandBuffer) throws -> any MTLBuffer {
        guard commandBuffer.device === device,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        let command = ObjectIdentifier(commandBuffer)
        let revision = self.revision
        let ready = Content.ready(revision)
        let pending = Content.pending(revision, command)
        let slot: Slot
        let writeContent = metal == nil ? ready : pending
        let writes: Bool
        if let current = slots.first(where: { $0.retainMatching(ready: ready, pending: pending) }) {
            slot = current
            writes = false
        } else {
            writes = true
            if let idle = slots.first(where: { $0.claimIfIdle(as: writeContent) }) {
                slot = idle
            } else {
                guard slots.count < Self.maximumBuffers else { throw Failure.buffersBusy }
                let length = geometry.vertices.count * MemoryLayout<PackedMeshVertex>.stride
                guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else { throw Failure.resource }
                buffer.label = "Frostfield curve flow vertices"
                slot = Slot(buffer: buffer, content: writeContent, readers: 1)
                slots.append(slot)
            }
        }
        let lease = Lease(slot)
        if writes, let metal {
            try metal.encode(
                placements: metalPlacements, distanceScale: Float(distanceScale),
                output: slot.buffer, commandBuffer: commandBuffer
            )
            commandBuffer.addCompletedHandler { buffer in
                if buffer.status == .completed { slot.publish(pending: pending, as: ready) }
                lease.release()
            }
        } else {
            if writes {
                let packed = slot.buffer.contents().bindMemory(to: PackedMeshVertex.self, capacity: geometry.vertices.count)
                for (index, vertex) in geometry.vertices.enumerated() {
                    packed[index] = PackedMeshVertex(vertex)
                }
            }
            commandBuffer.addCompletedHandler { _ in lease.release() }
        }
        return slot.buffer
    }

    private enum Content: Equatable {
        case empty
        case ready(UInt64)
        case pending(UInt64, ObjectIdentifier)
    }

    private final class Slot: @unchecked Sendable {
        let buffer: any MTLBuffer
        private let state: Mutex<State>

        private struct State {
            var content: Content
            var readers: Int
        }

        init(buffer: any MTLBuffer, content: Content, readers: Int = 0) {
            self.buffer = buffer
            self.state = Mutex(State(content: content, readers: readers))
        }

        func retainMatching(ready: Content, pending: Content) -> Bool {
            state.withLock { state in
                guard state.content == ready || state.content == pending else { return false }
                state.readers += 1
                return true
            }
        }

        func claimIfIdle(as content: Content) -> Bool {
            state.withLock { state in
                guard state.readers == 0 else { return false }
                state = State(content: content, readers: 1)
                return true
            }
        }

        func publish(pending: Content, as ready: Content) {
            state.withLock { state in
                if state.content == pending { state.content = ready }
            }
        }

        func release() {
            state.withLock { state in
                state.readers -= 1
                if state.readers == 0, case .pending = state.content { state.content = .empty }
            }
        }
    }

    /// Also keeps the buffer alive for command buffers with unretained references.
    private final class Lease: Sendable {
        private let slot: Slot
        private let active = Mutex(true)

        init(_ slot: Slot) { self.slot = slot }
        deinit { release() }

        func release() {
            active.withLock { active in
                if active {
                    slot.release()
                    active = false
                }
            }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case index(Int)
        case placementCount
        case vertexGraph
        case commandBuffer
        case resource
        case buffersBusy
        case metalDomain
        case metalSpine
        case compute(String)

        public var errorDescription: String? {
            switch self {
            case .index(let index): "Curve flow placement \(index) is out of range."
            case .placementCount: "Replace the same number of placements as the prepared curve flow."
            case .vertexGraph: "A curve flow material cannot also deform vertices with a surface graph."
            case .commandBuffer: "Use an uncommitted command buffer from the curve flow's device."
            case .resource: "Metal could not allocate a curve flow buffer."
            case .buffersBusy: "Complete or discard an earlier command buffer before encoding another curve flow revision."
            case .metalDomain: "The curve flow exceeds the Metal precision, interval or bend limits. Use CPU execution for this geometry."
            case .metalSpine: "Metal curve flow can only use spines prepared at initialization."
            case .compute(let reason): "Curve flow compute failed: \(reason)"
            }
        }
    }
}

/// Matches FrostfieldMeshVertex in Scene.metal and the graph deformation kernels.
struct PackedMeshVertex {
    let position: SIMD4<Float>
    let normal: SIMD4<Float>
    let uv: SIMD4<Float>
    let tangent: SIMD4<Float>

    init(_ vertex: MeshVertex) {
        self.init(vertex, packedPosition: vertex.position, packedNormal: vertex.normal)
    }

    init(_ vertex: MeshVertex, packedPosition: SIMD3<Float>, packedNormal: SIMD3<Float>) {
        let uv1 = vertex.secondaryTextureCoordinate ?? .zero
        position = SIMD4(packedPosition, 1)
        normal = SIMD4(packedNormal, 0)
        uv = SIMD4(vertex.textureCoordinate.x, vertex.textureCoordinate.y, uv1.x, uv1.y)
        tangent = vertex.tangent ?? .zero
    }
}
