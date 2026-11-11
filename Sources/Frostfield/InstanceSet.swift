import Foundation
import Metal
import simd
import Synchronization

/// Per-instance transforms and linear RGBA tints that draw one scene node's mesh
/// many times in a single instanced draw, like Three.js `InstancedMesh` with its
/// `instanceMatrix` and `instanceColor`.
///
/// Each instance places the mesh in the node's local space: the world transform
/// of instance `i` is `node world × transforms[i]`, applied after morphing and
/// skinning. Its tint multiplies the draw's material parameters in the same role:
/// PBR base color, and the RGBA tint of unlit, textured and shadowed Lambert
/// materials. Custom fragments can read it as `float4 [[user(frostfield_instance_tint)]]`.
/// Lambert parameters describe a light, so Lambert ignores tints.
///
/// The set is a reference type, so every graph that holds it sees edits, as with
/// a Three.js attribute. Edits change only CPU memory. The next encode copies
/// the records into one of three GPU buffers that no submitted command buffer is
/// still reading; later draws reuse that buffer until the next edit. Each encode
/// marks its buffer busy until its command buffer completes, so a CPU write
/// never races a GPU read. When all three are busy, the encode copies into a
/// transient buffer instead. A command buffer that is encoded but never
/// committed keeps its buffer busy, leaving two for later frames. Prepare the
/// mesh with `supportsInstancing: true`.
///
/// Use the set from one thread at a time, like the scene graph that holds it.
public final class InstanceSet {
    /// The largest capacity, which needs 3 × 16 MiB of GPU buffers.
    public static let maximumCapacity = 262_144
    /// The number of persistent GPU buffers a set rotates through.
    public static let bufferCount = 3

    public let device: any MTLDevice
    /// The fixed maximum number of instances.
    public let capacity: Int
    /// Mesh-to-node affine transforms, one per instance.
    public private(set) var transforms: [simd_float4x4]
    /// Linear RGBA tints aligned with `transforms`. One when not supplied.
    public private(set) var tints: [SIMD4<Float>]
    /// Increments after every successful edit.
    public private(set) var revision: UInt64 = 0
    /// A bulk replacement can reorder or replace slots, even at the same count.
    private(set) var layoutID = UUID()

    /// The number of instances drawn. Zero skips the node's draws.
    public var count: Int { transforms.count }

    /// Whether every transform reverses handedness. Instances must agree, since
    /// one draw has one front-face winding.
    public var isMirrored: Bool { mirroredCount > 0 }

    private var mirroredCount = 0
    private var sorting = false
    private let buffers: [any MTLBuffer]
    private var bufferRevisions: [UInt64?]
    private var currentBuffer: Int?
    private let busy: BusyBuffers
    private var cachedBounds: (revision: UInt64, mesh: AxisAlignedBox, enclosing: AxisAlignedBox)?

    func recordResources(in inventory: inout SceneResourceInventory) {
        for buffer in buffers { inventory.recordBuffer(buffer) }
    }

    /// Creates a set and its three GPU buffers of `capacity × 64` bytes each.
    /// Capacity defaults to the initial count and must be in `1...maximumCapacity`.
    /// Transforms must be finite, affine and invertible, all with the same
    /// handedness; tints must be finite and match the transform count.
    public init(
        device: any MTLDevice, transforms: [simd_float4x4], tints: [SIMD4<Float>]? = nil, capacity: Int? = nil
    ) throws {
        let capacity = capacity ?? transforms.count
        guard (1...Self.maximumCapacity).contains(capacity) else { throw Failure.capacity }
        let length = capacity * MemoryLayout<PackedInstance>.stride
        let buffers = try (0..<Self.bufferCount).map { index in
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw Failure.resource
            }
            buffer.label = "Frostfield instances \(index)"
            return buffer
        }
        self.device = device
        self.capacity = capacity
        self.buffers = buffers
        bufferRevisions = Array(repeating: nil, count: Self.bufferCount)
        busy = BusyBuffers(count: Self.bufferCount)
        self.transforms = []
        self.tints = []
        try replaceAll(transforms: transforms, tints: tints)
    }

    /// Replaces every instance. The count may change within the capacity. On
    /// failure the set is unchanged.
    public func replaceAll(transforms: [simd_float4x4], tints: [SIMD4<Float>]? = nil) throws {
        guard !sorting else { throw Failure.reentrantSort }
        guard transforms.count <= capacity else { throw Failure.count }
        if let tints, tints.count != transforms.count { throw Failure.tintCount }
        var mirrored = 0
        for (index, transform) in transforms.enumerated() {
            guard let determinant = Self.validatedDeterminant(transform) else { throw Failure.transform(index) }
            if determinant < 0 { mirrored += 1 }
        }
        guard mirrored == 0 || mirrored == transforms.count else { throw Failure.mixedHandedness }
        if let tints, let index = tints.firstIndex(where: { !Self.isFinite($0) }) { throw Failure.tint(index) }
        self.transforms = transforms
        self.tints = tints ?? Array(repeating: SIMD4(repeating: 1), count: transforms.count)
        mirroredCount = mirrored
        layoutID = UUID()
        revision &+= 1
    }

    /// Replaces one transform. The new transform must keep the set's handedness
    /// unless it is the only instance.
    public func setTransform(_ transform: simd_float4x4, at index: Int) throws {
        guard !sorting else { throw Failure.reentrantSort }
        guard transforms.indices.contains(index) else { throw Failure.index(index) }
        guard let determinant = Self.validatedDeterminant(transform) else { throw Failure.transform(index) }
        let wasMirrored = simd_determinant(transforms[index]) < 0
        let mirrored = mirroredCount - (wasMirrored ? 1 : 0) + (determinant < 0 ? 1 : 0)
        guard mirrored == 0 || mirrored == transforms.count else { throw Failure.mixedHandedness }
        transforms[index] = transform
        mirroredCount = mirrored
        revision &+= 1
    }

    /// Replaces one tint.
    public func setTint(_ tint: SIMD4<Float>, at index: Int) throws {
        guard !sorting else { throw Failure.reentrantSort }
        guard tints.indices.contains(index) else { throw Failure.index(index) }
        guard Self.isFinite(tint) else { throw Failure.tint(index) }
        tints[index] = tint
        revision &+= 1
    }

    /// Stably sorts source indices using `compare`, then applies the same
    /// permutation to transforms and tints. Returns the previous index for
    /// each new slot. An identity order leaves the revision unchanged. The
    /// comparator can read the set. Attempts to edit it during comparison throw.
    @discardableResult
    public func sort(by compare: (Int, Int) throws -> Bool) throws -> [Int] {
        guard !sorting else { throw Failure.reentrantSort }
        let order: [Int]
        do {
            sorting = true
            defer { sorting = false }
            order = try transforms.indices.sorted(by: compare)
        }
        if !order.elementsEqual(transforms.indices) {
            try replaceAll(transforms: order.map { transforms[$0] }, tints: order.map { tints[$0] })
        }
        return order
    }

    /// The node-space box around every instance of a mesh with these local
    /// bounds, cached until the next edit. Nil when the set is empty.
    package func bounds(enclosing mesh: AxisAlignedBox) throws -> AxisAlignedBox? {
        guard !transforms.isEmpty else { return nil }
        if let cachedBounds, cachedBounds.revision == revision, cachedBounds.mesh == mesh {
            return cachedBounds.enclosing
        }
        var lower = SIMD3<Float>(repeating: .infinity)
        var upper = SIMD3<Float>(repeating: -.infinity)
        for transform in transforms {
            let box = try mesh.transformed(by: transform)
            lower = simd_min(lower, box.minimum)
            upper = simd_max(upper, box.maximum)
        }
        let enclosing = try AxisAlignedBox(minimum: lower, maximum: upper)
        cachedBounds = (revision, mesh, enclosing)
        return enclosing
    }

    /// Returns a GPU buffer holding the current records for a draw in this command
    /// buffer, and keeps it from being rewritten until the command buffer completes.
    /// Call only after every other check for the draw has passed.
    package func acquireBuffer(for commandBuffer: any MTLCommandBuffer) throws -> any MTLBuffer {
        guard commandBuffer.device === device else { throw Failure.wrongDevice }
        let slot: Int
        if let current = currentBuffer, bufferRevisions[current] == revision {
            slot = current
            busy.retain(slot)
        } else if let free = busy.retainFree(preferring: currentBuffer) {
            slot = free
            write(into: buffers[slot])
            bufferRevisions[slot] = revision
            currentBuffer = slot
        } else {
            let length = max(transforms.count, 1) * MemoryLayout<PackedInstance>.stride
            guard let transient = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw Failure.resource
            }
            transient.label = "Frostfield transient instances"
            write(into: transient)
            if !commandBuffer.retainedReferences {
                let retained = RetainedInstanceBuffer(buffer: transient)
                commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
            }
            return transient
        }
        let busy = busy
        commandBuffer.addCompletedHandler { _ in busy.release(slot) }
        if !commandBuffer.retainedReferences {
            // The set may be released before the GPU finishes; keep the buffer alive.
            let retained = RetainedInstanceBuffer(buffer: buffers[slot])
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
        }
        return buffers[slot]
    }

    private func write(into buffer: any MTLBuffer) {
        let records = buffer.contents().bindMemory(to: PackedInstance.self, capacity: max(transforms.count, 1))
        for index in transforms.indices {
            records[index] = PackedInstance(transform: transforms[index], tint: tints[index])
        }
    }

    /// The determinant of a finite, affine, invertible matrix with a finite inverse, or nil.
    static func validatedDeterminant(_ matrix: simd_float4x4) -> Float? {
        let (x, y, z, w) = matrix.columns
        guard isFinite(x), isFinite(y), isFinite(z), isFinite(w),
              x.w == 0, y.w == 0, z.w == 0, w.w == 1 else { return nil }
        let determinant = simd_determinant(matrix)
        guard determinant.isFinite, determinant != 0 else { return nil }
        let (a, b, c, d) = simd_inverse(matrix).columns
        guard isFinite(a), isFinite(b), isFinite(c), isFinite(d) else { return nil }
        return determinant
    }

    private static func isFinite(_ value: SIMD4<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
    }

    /// One instance record, matching `FrostfieldInstance` in Scene.metal: the
    /// transform's first three rows, then the tint. 64-byte stride, no padding.
    package struct PackedInstance {
        package var row0: SIMD4<Float>
        package var row1: SIMD4<Float>
        package var row2: SIMD4<Float>
        package var tint: SIMD4<Float>

        package init(transform: simd_float4x4, tint: SIMD4<Float>) {
            let rows = simd_transpose(transform).columns
            row0 = rows.0
            row1 = rows.1
            row2 = rows.2
            self.tint = tint
        }

        /// The record a draw without instances binds.
        package static let identity = PackedInstance(transform: matrix_identity_float4x4, tint: SIMD4(repeating: 1))
    }

    /// Counts the submitted command buffers reading each persistent buffer.
    /// Completion handlers release their count from Metal's callback thread.
    private final class BusyBuffers: Sendable {
        private let readers: Mutex<[Int]>

        init(count: Int) {
            readers = Mutex(Array(repeating: 0, count: count))
        }

        func retain(_ slot: Int) {
            readers.withLock { $0[slot] += 1 }
        }

        func release(_ slot: Int) {
            readers.withLock { $0[slot] -= 1 }
        }

        /// Marks and returns an idle buffer, trying the preferred one first.
        func retainFree(preferring preferred: Int?) -> Int? {
            readers.withLock { readers in
                let order = (preferred.map { [$0] } ?? []) + readers.indices.filter { $0 != preferred }
                guard let slot = order.first(where: { readers[$0] == 0 }) else { return nil }
                readers[slot] += 1
                return slot
            }
        }
    }

    // The buffer's contents never change after the GPU starts reading it; the handler only owns its lifetime.
    private struct RetainedInstanceBuffer: @unchecked Sendable {
        let buffer: any MTLBuffer
    }

    public enum Failure: LocalizedError, Equatable {
        case capacity
        case count
        case index(Int)
        case transform(Int)
        case mixedHandedness
        case tintCount
        case tint(Int)
        case wrongDevice
        case reentrantSort
        case resource

        public var errorDescription: String? {
            switch self {
            case .capacity: "An instance set needs a capacity of 1 to \(InstanceSet.maximumCapacity)."
            case .count: "The instance count exceeds the set's capacity."
            case .index(let index): "Instance \(index) is outside the set."
            case .transform(let index): "Instance \(index) needs a finite, affine, invertible transform."
            case .mixedHandedness: "Every instance transform needs the same handedness (determinant sign)."
            case .tintCount: "Supply one tint per instance transform."
            case .tint(let index): "Instance \(index) needs a finite tint."
            case .wrongDevice: "The instance set belongs to a different Metal device."
            case .reentrantSort: "The instance set cannot be edited from its sort comparator."
            case .resource: "Metal could not allocate the instance buffers."
            }
        }
    }
}
