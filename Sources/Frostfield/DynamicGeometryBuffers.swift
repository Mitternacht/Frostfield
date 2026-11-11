import Metal
import simd
import Synchronization

extension MeshRenderer {
    /// Fixed capacities for a renderer that replaces its indexed geometry.
    public struct DynamicGeometryCapacity: Hashable, Sendable {
        public static let maximumVertices = 524_288
        public static let maximumTriangles = 262_144
        public let vertices: Int
        public let triangles: Int

        public init(vertices: Int, triangles: Int) throws {
            guard (1...Self.maximumVertices).contains(vertices), (1...Self.maximumTriangles).contains(triangles) else {
                throw Failure.dynamicGeometryCapacity
            }
            self.vertices = vertices
            self.triangles = triangles
        }

        func validate(_ geometry: MeshGeometry) throws {
            guard geometry.vertices.count <= vertices, geometry.indices.count / 3 <= triangles else {
                throw Failure.dynamicGeometryCapacity
            }
        }
    }
}

/// CPU-written, revisioned storage. Encoding and edits are serialized by the
/// caller. Only leases are released from Metal's callback thread.
final class DynamicGeometryBuffers {
    struct Buffers {
        let vertices: any MTLBuffer
        let indices: any MTLBuffer
        let colors: (any MTLBuffer)?
    }

    struct Revision {
        let number: UInt64
        let topologyRevision: UInt64
        let geometry: MeshGeometry
        let bounds: AxisAlignedBox
    }

    private static let maximumSlots = 3
    let capacity: MeshRenderer.DynamicGeometryCapacity
    private let device: any MTLDevice
    private let colored: Bool
    private var slots: [Slot]
    var initial: Buffers { slots[0].buffers }

    init(device: any MTLDevice, capacity: MeshRenderer.DynamicGeometryCapacity, geometry: MeshGeometry, colored: Bool) throws {
        self.device = device
        self.capacity = capacity
        self.colored = colored
        let buffers = try Self.allocate(device: device, capacity: capacity, colored: colored)
        slots = [Slot(buffers, revision: 0, readers: 0)]
        write(geometry, into: buffers)
    }

    func acquire(
        _ revision: Revision, for commandBuffer: any MTLCommandBuffer, requiringColors: Bool = false
    ) throws -> Buffers {
        guard commandBuffer.device === device else { throw MeshRenderer.Failure.wrongDevice }
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw MeshRenderer.Failure.dynamicGeometryCommandBuffer
        }
        let slot: Slot
        if let matching = slots.first(where: { $0.retain(ifRevision: revision.number) }) {
            slot = matching
        } else {
            if let idle = slots.first(where: { $0.claim(ifIdle: revision.number) }) {
                slot = idle
            } else {
                guard slots.count < Self.maximumSlots else { throw MeshRenderer.Failure.dynamicGeometryBuffersBusy }
                let buffers = try Self.allocate(device: device, capacity: capacity, colored: colored)
                slot = Slot(buffers, revision: revision.number, readers: 1)
                slots.append(slot)
            }
            write(revision.geometry, into: slot.buffers)
        }
        let lease = Lease(slot)
        if requiringColors, slot.buffers.colors == nil {
            // The existing vertex and index buffers may already be in flight.
            // Add independent storage, then retain it alongside those buffers.
            let colors = try Self.colorBuffer(device: device, capacity: capacity)
            writeColors(revision.geometry, into: colors)
            slot.buffers = Buffers(vertices: slot.buffers.vertices, indices: slot.buffers.indices, colors: colors)
        }
        commandBuffer.addCompletedHandler { _ in lease.release() }
        return slot.buffers
    }

    func recordResources(in inventory: inout SceneResourceInventory) {
        for slot in slots {
            inventory.recordBuffer(slot.buffers.vertices)
            inventory.recordBuffer(slot.buffers.indices)
            inventory.recordBuffer(slot.buffers.colors)
        }
    }

    private static func allocate(
        device: any MTLDevice, capacity: MeshRenderer.DynamicGeometryCapacity, colored: Bool
    ) throws -> Buffers {
        func buffer(length: Int, label: String) throws -> any MTLBuffer {
            guard length <= device.maxBufferLength,
                  let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw MeshRenderer.Failure.resource(label)
            }
            buffer.label = label
            return buffer
        }
        return Buffers(
            vertices: try buffer(length: capacity.vertices * MemoryLayout<PackedMeshVertex>.stride,
                                 label: "Frostfield dynamic mesh vertices"),
            indices: try buffer(length: capacity.triangles * 3 * MemoryLayout<UInt32>.stride,
                                label: "Frostfield dynamic mesh indices"),
            colors: colored ? try colorBuffer(device: device, capacity: capacity) : nil
        )
    }

    private static func colorBuffer(
        device: any MTLDevice, capacity: MeshRenderer.DynamicGeometryCapacity
    ) throws -> any MTLBuffer {
        let length = capacity.vertices * MemoryLayout<SIMD4<Float>>.stride
        let label = "Frostfield dynamic mesh colors"
        guard length <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw MeshRenderer.Failure.resource(label)
        }
        buffer.label = label
        return buffer
    }

    private func write(_ geometry: MeshGeometry, into buffers: Buffers) {
        let vertices = buffers.vertices.contents().bindMemory(to: PackedMeshVertex.self, capacity: capacity.vertices)
        for (index, vertex) in geometry.vertices.enumerated() {
            vertices[index] = PackedMeshVertex(vertex)
        }
        if let colors = buffers.colors { writeColors(geometry, into: colors) }
        geometry.indices.withUnsafeBytes {
            buffers.indices.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
    }

    private func writeColors(_ geometry: MeshGeometry, into buffer: any MTLBuffer) {
        let colors = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity.vertices)
        for (index, vertex) in geometry.vertices.enumerated() {
            colors[index] = vertex.color ?? SIMD4(repeating: 1)
        }
    }

    private final class Slot: @unchecked Sendable {
        // Only the render owner reads or changes these bindings. Completion
        // handlers release the reader count in state.
        var buffers: Buffers
        private struct State {
            var revision: UInt64
            var readers: Int
        }
        private let state: Mutex<State>

        init(_ buffers: Buffers, revision: UInt64, readers: Int) {
            self.buffers = buffers
            state = Mutex(State(revision: revision, readers: readers))
        }

        func retain(ifRevision revision: UInt64) -> Bool {
            state.withLock {
                guard $0.revision == revision else { return false }
                $0.readers += 1
                return true
            }
        }

        func claim(ifIdle revision: UInt64) -> Bool {
            state.withLock {
                guard $0.readers == 0 else { return false }
                $0 = State(revision: revision, readers: 1)
                return true
            }
        }

        func release() { state.withLock { $0.readers -= 1 } }
    }

    /// Keeps all slot buffers alive even when the command has unretained
    /// references. Discarding an uncommitted command destroys its handler.
    private final class Lease: Sendable {
        private let slot: Slot
        private let active = Mutex(true)

        init(_ slot: Slot) { self.slot = slot }
        deinit { release() }

        func release() {
            active.withLock {
                if $0 {
                    slot.release()
                    $0 = false
                }
            }
        }
    }
}
