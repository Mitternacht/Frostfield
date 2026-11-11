import Foundation
import Metal
import Synchronization
import simd

/// Independently placed indexed geometries that share one material and GPU geometry storage.
/// A scene node can draw the batch in any color or depth view of the raster renderer.
///
/// Instance and palette edits can run between encodes while earlier draws are in flight.
/// Use this object and its scene graphs on one thread at a time.
public final class BatchedMesh {
    /// A stable handle for the lifetime of an instance. Removing it invalidates the
    /// handle, even if a later instance occupies the same slot.
    public struct InstanceID: Hashable, Sendable {
        public let index: Int
        fileprivate let token: UUID
    }

    /// A stable handle to a live geometry slot. Replacing its geometry preserves the
    /// handle. Removing the slot invalidates it, including after numeric slot reuse.
    public struct GeometryID: Hashable, Sendable {
        public let index: Int
        fileprivate let token: UUID
    }

    public struct Instance: Sendable {
        public let id: InstanceID
        public let geometryID: GeometryID
        public var geometryIndex: Int { geometryID.index }
        public let transform: simd_float4x4
        public let tint: SIMD4<Float>
        public let isVisible: Bool
    }

    public enum Submission: Equatable, Sendable {
        case direct
        /// Uses reusable CPU-encoded Metal indirect command buffers.
        case indirectCommandBuffer
    }

    /// Checks GPU family support and allocates a one-command probe buffer.
    public static func supportsIndirectSubmission(on device: any MTLDevice) -> Bool {
        BatchRenderBundle.supported(on: device)
    }

    public enum SortPass: Equatable, Sendable {
        case color
        case surfaceAttributes
        case directionalShadow(cascade: Int)
        case punctualShadow(slice: Int)
        case selectionDepth
    }

    public struct SortContext: Sendable {
        public let pass: SortPass
        public let clipFromWorld: simd_float4x4
        public let isBlended: Bool
    }

    public struct SortItem: Sendable {
        public let instance: Instance
        public let worldTransform: simd_float4x4
        public let worldBounds: AxisAlignedBox?
    }

    public let device: any MTLDevice
    public let capacity: Int
    public let submission: Submission
    public let vertexCapacity: Int
    public let indexCapacity: Int
    /// Live geometry values in slot order, omitting vacant slots.
    /// Their handles appear in the same order in `geometryIDs`.
    public var geometries: [MeshGeometry] { geometrySlots.compactMap { $0?.mesh } }
    public var geometryIDs: [GeometryID] { geometrySlots.compactMap { $0?.id } }
    public private(set) var count = 0
    public var perObjectFrustumCulled = true
    /// Opaque items sort near-to-far, blended items far-to-near. Disabling
    /// sorting uses slot order. Equal depths also use slot order.
    public var sortsObjects = true
    /// Returns the complete ordering of the supplied slots for this pass, after
    /// visibility and frustum filtering. Omit this callback to use view depth.
    /// Do not mutate or encode this batch from the callback.
    public var customSort: (([SortItem], SortContext) throws -> [InstanceID])?

    var renderer: MeshRenderer? { palette?.renderer }
    package var indirectBundleBuildCount: Int { palette?.indirectBundleBuildCount ?? 0 }
    var activeInstances: some Sequence<Instance> { slots.lazy.compactMap { $0 } }
    var hasVisibleInstances: Bool { activeInstances.contains(where: \.isVisible) }

    private let configuration: RenderConfiguration
    private var palette: Palette?
    private var retired: [RetiredPalette] = []
    private var geometrySlots: [GeometrySlot?] = []
    private var slots: [Instance?] = []
    private var freeSlots: [Int] = []
    private var pickingTrees: [Int: MeshBVH] = [:]
    private var sorting = false

    private struct GeometrySlot {
        let id: GeometryID
        let mesh: MeshGeometry
        let bounds: AxisAlignedBox

        init(id: GeometryID, mesh: MeshGeometry) {
            self.id = id
            self.mesh = mesh
            self.bounds = mesh.bounds
        }
    }

    private struct RenderConfiguration {
        let format: MTLPixelFormat
        let material: MeshMaterial
        let depthMode: MeshDepthMode
        let renderState: MeshRenderState?
        let sampleCount: Int
    }

    fileprivate final class Palette {
        let renderer: MeshRenderer
        let ranges: [Int: Range<Int>]
        private var bundles: [(runs: [Run], bundle: BatchRenderBundle)] = []
        private(set) var indirectBundleBuildCount = 0

        init(renderer: MeshRenderer, ranges: [Int: Range<Int>]) {
            self.renderer = renderer
            self.ranges = ranges
        }

        func bundle(for runs: [Run]) throws -> BatchRenderBundle {
            if let cached = bundles.first(where: { $0.runs == runs }) { return cached.bundle }
            let bundle = try renderer.prepareBatchBundle(runs)
            if bundles.count == 4 { bundles.removeFirst() }
            bundles.append((runs, bundle))
            indirectBundleBuildCount += 1
            return bundle
        }
    }

    private final class RetiredPalette {
        weak var palette: Palette?
        init(_ palette: Palette) { self.palette = palette }
    }

    /// Prepares indexed triangle geometry that shares one material and render state.
    /// An empty batch prepares its material when the first geometry is added.
    /// The vertex and index capacities limit the combined live geometry.
    public init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat,
        geometries: [MeshGeometry] = [], capacity: Int, material: MeshMaterial = .unlit,
        depthMode: MeshDepthMode = .forward, renderState: MeshRenderState? = nil, sampleCount: Int = 1,
        vertexCapacity: Int? = nil, indexCapacity: Int? = nil, submission: Submission = .direct
    ) throws {
        guard (1...InstanceSet.maximumCapacity).contains(capacity),
              capacity <= device.maxBufferLength / MemoryLayout<InstanceSet.PackedInstance>.stride else {
            throw Failure.capacity
        }
        let vertexLimit = min(1_048_576, device.maxBufferLength / MemoryLayout<PackedMeshVertex>.stride)
        let indexLimit = min(6_291_456, device.maxBufferLength / MemoryLayout<UInt32>.stride)
        let vertexCapacity = vertexCapacity ?? vertexLimit
        let indexCapacity = indexCapacity ?? indexLimit
        guard vertexCapacity > 0, vertexCapacity <= vertexLimit, indexCapacity > 0, indexCapacity <= indexLimit else {
            throw Failure.geometryBudget
        }
        guard RendererCapabilities.supportsRasterSampleCount(sampleCount, on: device) else {
            throw MeshRenderer.Failure.sampleCount
        }
        guard renderState?.stencil == nil else { throw MeshRenderer.Failure.stencilPass }
        if case .physicallyBased(let material) = material {
            if material.surfaceGraph?.hasVertexOutputs == true { throw Failure.vertexDeformation }
            if material.hasTransmission { throw Failure.transmission }
        }
        if submission == .indirectCommandBuffer, !Self.supportsIndirectSubmission(on: device) {
            throw Failure.indirectUnsupported
        }
        self.device = device
        self.capacity = capacity
        self.submission = submission
        self.vertexCapacity = vertexCapacity
        self.indexCapacity = indexCapacity
        configuration = RenderConfiguration(
            format: colorPixelFormat, material: material, depthMode: depthMode,
            renderState: renderState, sampleCount: sampleCount
        )
        guard geometries.count <= 4_096 else { throw Failure.geometryBudget }
        let initial = geometries.enumerated().map {
            GeometrySlot(id: GeometryID(index: $0.offset, token: UUID()), mesh: $0.element)
        }
        try publishGeometry(initial.map(Optional.some))
    }

    public func geometryID(at index: Int) throws -> GeometryID {
        guard geometrySlots.indices.contains(index), let slot = geometrySlots[index] else { throw Failure.geometry(index) }
        return slot.id
    }

    public func geometry(for id: GeometryID) throws -> MeshGeometry {
        try slot(for: id).mesh
    }

    @discardableResult
    public func addGeometry(_ geometry: MeshGeometry) throws -> GeometryID {
        try checkEditing()
        var next = geometrySlots
        let index = next.firstIndex(where: { $0 == nil }) ?? next.count
        guard index < 4_096 else { throw Failure.geometryBudget }
        let id = GeometryID(index: index, token: UUID())
        let entry = GeometrySlot(id: id, mesh: geometry)
        if index == next.count { next.append(entry) } else { next[index] = entry }
        try publishGeometry(next)
        return id
    }

    /// Replaces the geometry for all instances of this handle.
    public func replaceGeometry(_ id: GeometryID, with geometry: MeshGeometry) throws {
        try checkEditing()
        _ = try slot(for: id)
        var next = geometrySlots
        next[id.index] = GeometrySlot(id: id, mesh: geometry)
        try publishGeometry(next)
        pickingTrees[id.index] = nil
    }

    /// Removes the geometry and its dependent instances in one edit.
    public func removeGeometry(_ id: GeometryID) throws {
        try checkEditing()
        _ = try slot(for: id)
        var next = geometrySlots
        next[id.index] = nil
        try publishGeometry(next)
        pickingTrees[id.index] = nil
        for index in slots.indices where slots[index]?.geometryID == id { removeSlot(index) }
    }

    private func slot(for id: GeometryID) throws -> GeometrySlot {
        guard geometrySlots.indices.contains(id.index), let slot = geometrySlots[id.index], slot.id == id else {
            throw Failure.geometryHandle
        }
        return slot
    }

    private func publishGeometry(_ next: [GeometrySlot?]) throws {
        let occupied = next.compactMap { $0 }
        var vertices = 0, indices = 0
        for geometry in occupied {
            guard geometry.mesh.vertices.count <= vertexCapacity - vertices,
                  geometry.mesh.indices.count <= indexCapacity - indices else { throw Failure.geometryBudget }
            vertices += geometry.mesh.vertices.count
            indices += geometry.mesh.indices.count
        }
        retired.removeAll { $0.palette == nil }
        let prepared: Palette?
        if occupied.isEmpty {
            prepared = nil
        } else {
            // At most three palette generations can be outstanding. Encodes retain
            // the version with their original index ranges through GPU completion.
            guard retired.count + (palette == nil ? 0 : 1) < 3 else { throw Failure.paletteBusy }
            let combined = try MeshGeometry.merged(occupied.map { MeshGeometry.MergePart($0.mesh) })
            let config = configuration
            let renderer = try MeshRenderer(
                device: device, colorPixelFormat: config.format, geometry: combined.geometry,
                material: config.material, depthMode: config.depthMode, supportsInstancing: true,
                renderState: config.renderState, sampleCount: config.sampleCount,
                indirectCommands: submission == .indirectCommandBuffer
            )
            let ranges = Dictionary(uniqueKeysWithValues: zip(occupied.map { $0.id.index }, combined.indexRanges))
            prepared = Palette(renderer: renderer, ranges: ranges)
        }
        if let palette { retired.append(RetiredPalette(palette)) }
        geometrySlots = next
        palette = prepared
    }

    @discardableResult
    public func addInstance(
        geometryIndex: Int, transform: simd_float4x4 = matrix_identity_float4x4,
        tint: SIMD4<Float> = SIMD4(repeating: 1), isVisible: Bool = true
    ) throws -> InstanceID {
        try addInstance(geometryID: geometryID(at: geometryIndex), transform: transform, tint: tint, isVisible: isVisible)
    }

    @discardableResult
    public func addInstance(
        geometryID: GeometryID, transform: simd_float4x4 = matrix_identity_float4x4,
        tint: SIMD4<Float> = SIMD4(repeating: 1), isVisible: Bool = true
    ) throws -> InstanceID {
        try checkEditing()
        guard count < capacity else { throw Failure.capacity }
        _ = try slot(for: geometryID)
        try validate(transform: transform, tint: tint)
        let index = freeSlots.popLast() ?? slots.count
        let id = InstanceID(index: index, token: UUID())
        let instance = Instance(id: id, geometryID: geometryID, transform: transform, tint: tint, isVisible: isVisible)
        if index == slots.count { slots.append(instance) } else { slots[index] = instance }
        count += 1
        return id
    }

    public func instance(for id: InstanceID) throws -> Instance {
        guard slots.indices.contains(id.index), let instance = slots[id.index], instance.id == id else {
            throw Failure.instance
        }
        return instance
    }

    /// Updates the supplied fields together. An invalid edit leaves the instance unchanged.
    public func update(
        _ id: InstanceID, geometryIndex: Int? = nil, transform: simd_float4x4? = nil,
        tint: SIMD4<Float>? = nil, isVisible: Bool? = nil
    ) throws {
        let old = try instance(for: id)
        let geometryID = try geometryIndex.map(geometryID(at:)) ?? old.geometryID
        try update(id, geometryID: geometryID, transform: transform, tint: tint, isVisible: isVisible)
    }

    public func update(
        _ id: InstanceID, geometryID: GeometryID, transform: simd_float4x4? = nil,
        tint: SIMD4<Float>? = nil, isVisible: Bool? = nil
    ) throws {
        try checkEditing()
        let old = try instance(for: id)
        _ = try slot(for: geometryID)
        let transform = transform ?? old.transform
        let tint = tint ?? old.tint
        try validate(transform: transform, tint: tint)
        slots[id.index] = Instance(
            id: id, geometryID: geometryID, transform: transform, tint: tint, isVisible: isVisible ?? old.isVisible
        )
    }

    public func remove(_ id: InstanceID) throws {
        try checkEditing()
        _ = try instance(for: id)
        removeSlot(id.index)
    }

    private func removeSlot(_ index: Int) {
        slots[index] = nil
        freeSlots.append(index)
        count -= 1
    }

    private func checkEditing() throws {
        guard !sorting else { throw Failure.reentrantSort }
    }

    private func validate(transform: simd_float4x4, tint: SIMD4<Float>) throws {
        guard InstanceSet.validatedDeterminant(transform) != nil else { throw Failure.transform }
        guard tint.x.isFinite, tint.y.isFinite, tint.z.isFinite, tint.w.isFinite else { throw Failure.tint }
    }

    /// Encloses the active geometry in batch-local coordinates.
    public func bounds(visibleOnly: Bool = false) throws -> AxisAlignedBox? {
        try bounds(world: matrix_identity_float4x4, visibleOnly: visibleOnly)
    }

    func bounds(world: simd_float4x4, visibleOnly: Bool) throws -> AxisAlignedBox? {
        var enclosing: AxisAlignedBox?
        for instance in activeInstances where !visibleOnly || instance.isVisible {
            let box = try slot(for: instance.geometryID).bounds.transformed(by: world * instance.transform)
            enclosing = enclosing.map {
                AxisAlignedBox(validatedMinimum: simd_min($0.minimum, box.minimum), maximum: simd_max($0.maximum, box.maximum))
            } ?? box
        }
        return enclosing
    }

    struct Object {
        let instance: Instance
        let world: simd_float4x4
        let bounds: AxisAlignedBox?
        let mirrored: Bool
    }

    func objects(world: simd_float4x4, visibleOnly: Bool = true) throws -> [Object] {
        var objects: [Object] = []
        objects.reserveCapacity(count)
        for instance in activeInstances where !visibleOnly || instance.isVisible {
            let transform = world * instance.transform
            guard let determinant = InstanceSet.validatedDeterminant(transform) else { throw Failure.transform }
            let bounds = try slot(for: instance.geometryID).bounds
            objects.append(Object(
                instance: instance, world: transform, bounds: try? bounds.transformed(by: transform),
                mirrored: determinant < 0
            ))
        }
        return objects
    }

    func enclosing(_ snapshot: DrawSnapshot, cameraRelative: Bool = false) -> AxisAlignedBox? {
        guard !snapshot.objects.isEmpty else { return nil }
        var lower = SIMD3<Float>(repeating: .infinity)
        var upper = SIMD3<Float>(repeating: -.infinity)
        for object in snapshot.objects {
            guard let bounds = snapshot.cullingBounds(of: object, cameraRelative: cameraRelative) else { return nil }
            lower = simd_min(lower, bounds.minimum)
            upper = simd_max(upper, bounds.maximum)
        }
        return AxisAlignedBox(validatedMinimum: lower, maximum: upper)
    }

    fileprivate struct RelativeObject {
        let bounds: AxisAlignedBox?
        let center: SIMD3<Double>
    }

    struct DrawSnapshot {
        fileprivate let palette: Palette
        let objects: [Object]
        let perObjectFrustumCulled: Bool
        fileprivate let sortsObjects: Bool
        fileprivate let customSort: (([SortItem], SortContext) throws -> [InstanceID])?
        fileprivate let relativeObjects: [InstanceID: RelativeObject]?

        fileprivate func cullingBounds(of object: Object, cameraRelative: Bool) -> AxisAlignedBox? {
            cameraRelative ? relativeObjects?[object.instance.id]?.bounds : object.bounds
        }
    }

    func drawSnapshot(world: simd_float4x4, relativeWorld: simd_double4x4? = nil) throws -> DrawSnapshot? {
        guard !sorting else { throw Failure.reentrantSort }
        guard let palette else { return nil }
        let objects = try objects(world: world)
        let relativeObjects = try relativeWorld.map { relative in
            try Dictionary(uniqueKeysWithValues: objects.map { object in
                let instance = object.instance
                let transform = instance.transform
                let relativeTransform = relative * simd_double4x4(
                    SIMD4<Double>(transform[0]), SIMD4<Double>(transform[1]),
                    SIMD4<Double>(transform[2]), SIMD4<Double>(transform[3])
                )
                let local = try slot(for: instance.geometryID).bounds
                let center = relativeTransform * SIMD4(SIMD3<Double>(local.center), 1)
                return (instance.id, RelativeObject(
                    bounds: try? local.transformed(by: relativeTransform), center: SIMD3(center.x, center.y, center.z)
                ))
            })
        }
        return DrawSnapshot(
            palette: palette, objects: objects, perObjectFrustumCulled: perObjectFrustumCulled,
            sortsObjects: sortsObjects, customSort: customSort, relativeObjects: relativeObjects
        )
    }

    struct Run: Equatable {
        let indices: Range<Int>
        let firstInstance: Int
        var instanceCount: Int
        let mirrored: Bool
    }

    struct Plan {
        fileprivate let palette: Palette
        let objects: [Object]
        let runs: [Run]
        let culledCount: Int
    }

    func plan(
        _ snapshot: DrawSnapshot, frustum: ViewFrustum?,
        context: SortContext, sortingBack: SIMD3<Float>? = nil, cameraRelative: Bool = false
    ) throws -> Plan {
        guard !sorting else { throw Failure.reentrantSort }
        let visible = snapshot.objects.filter { object in
            guard let frustum, let bounds = snapshot.cullingBounds(of: object, cameraRelative: cameraRelative) else { return true }
            return frustum.classify(box: bounds) != .outside
        }
        let ordered: [Object]
        if snapshot.sortsObjects, let customSort = snapshot.customSort, !visible.isEmpty {
            sorting = true
            defer { sorting = false }
            let items = visible.map { SortItem(instance: $0.instance, worldTransform: $0.world, worldBounds: $0.bounds) }
            let order = try customSort(items, context)
            guard order.count == visible.count else { throw Failure.customOrder }
            var remaining = Dictionary(uniqueKeysWithValues: visible.enumerated().map { ($0.element.instance.id, $0.offset) })
            ordered = try order.map { id in
                guard let index = remaining.removeValue(forKey: id) else { throw Failure.customOrder }
                return visible[index]
            }
        } else if snapshot.sortsObjects, let sortingBack {
            // Projection onto the backward view axis differs from camera-relative
            // depth by a common constant, so it produces the same order.
            let axis = SIMD3<Double>(sortingBack)
            ordered = visible.map { object in
                if cameraRelative, let relative = snapshot.relativeObjects?[object.instance.id] {
                    return (object: object, depth: simd_dot(relative.center, axis))
                }
                let column = object.world.columns.3
                let center = object.bounds?.center ?? SIMD3(column.x, column.y, column.z)
                return (object: object, depth: simd_dot(SIMD3<Double>(center), axis))
            }.sorted { a, b in
                if a.depth == b.depth { return a.object.instance.id.index < b.object.instance.id.index }
                return context.isBlended ? a.depth < b.depth : a.depth > b.depth
            }.map(\.object)
        } else {
            ordered = visible
        }

        var runs: [Run] = []
        for (offset, object) in ordered.enumerated() {
            guard let range = snapshot.palette.ranges[object.instance.geometryIndex] else { throw Failure.geometryHandle }
            if let last = runs.last, last.indices == range, last.mirrored == object.mirrored {
                runs[runs.count - 1].instanceCount += 1
            } else {
                runs.append(Run(indices: range, firstInstance: offset, instanceCount: 1, mirrored: object.mirrored))
            }
        }
        return Plan(
            palette: snapshot.palette, objects: ordered, runs: runs, culledCount: snapshot.objects.count - ordered.count
        )
    }

    struct Packet {
        let buffer: any MTLBuffer
        let runs: [Run]
        let instanceCount: Int
        let bundle: BatchRenderBundle?
    }

    func acquire(
        _ plan: Plan, commandBuffer: any MTLCommandBuffer, bytesUsed: inout Int, useIndirect: Bool = true
    ) throws -> Packet? {
        guard !plan.runs.isEmpty else { return nil }
        let length = plan.objects.count * MemoryLayout<InstanceSet.PackedInstance>.stride
        guard length <= 64 * 1_024 * 1_024 - bytesUsed else { throw Failure.drawBudget }
        let bundle = useIndirect && submission == .indirectCommandBuffer ? try plan.palette.bundle(for: plan.runs) : nil
        // The immutable packet belongs to this pass and may be used from any queue.
        // Mutating a slot after encoding never rewrites the old Metal bytes.
        let records = plan.objects.map {
            InstanceSet.PackedInstance(transform: $0.instance.transform, tint: $0.instance.tint)
        }
        let buffer = try records.withUnsafeBytes { bytes in
            guard let address = bytes.baseAddress,
                  let buffer = device.makeBuffer(bytes: address, length: bytes.count, options: .storageModeShared) else {
                throw Failure.resource
            }
            return buffer
        }
        buffer.label = "Frostfield batched mesh instances"
        bytesUsed += length
        let retained = Mutex<Retained?>(Retained(palette: plan.palette, buffer: buffer, bundle: bundle))
        commandBuffer.addCompletedHandler { _ in retained.withLock { $0 = nil } }
        return Packet(buffer: buffer, runs: plan.runs, instanceCount: records.count, bundle: bundle)
    }

    private struct Retained: @unchecked Sendable {
        let palette: Palette
        let buffer: any MTLBuffer
        let bundle: BatchRenderBundle?
    }

    func appendHits(ray: SceneRay, node: ResolvedSceneNode, to result: inout [SceneHit]) throws {
        guard let renderer else { return }
        for object in try objects(world: node.worldTransform) {
            let tree = try pickingTree(object.instance.geometryID)
            let hits = try tree.intersections(ray: ray, localToWorld: object.world, side: renderer.renderState.side)
            result.append(contentsOf: hits.map {
                SceneHit(nodeID: node.id, intersection: $0, instanceIndex: nil, batchInstanceID: object.instance.id)
            })
        }
    }

    func nearestHit(ray: SceneRay, node: ResolvedSceneNode) throws -> SceneHit? {
        guard let renderer else { return nil }
        var nearest: SceneHit?
        for object in try objects(world: node.worldTransform) {
            let tree = try pickingTree(object.instance.geometryID)
            guard let hit = try tree.intersection(ray: ray, localToWorld: object.world, side: renderer.renderState.side),
                  nearest.map({ hit.distance < $0.intersection.distance }) != false else { continue }
            nearest = SceneHit(nodeID: node.id, intersection: hit, instanceIndex: nil, batchInstanceID: object.instance.id)
        }
        return nearest
    }

    private func pickingTree(_ id: GeometryID) throws -> MeshBVH {
        if let tree = pickingTrees[id.index] { return tree }
        let tree = try MeshBVH(geometry: slot(for: id).mesh)
        pickingTrees[id.index] = tree
        return tree
    }

    public enum Failure: LocalizedError, Equatable {
        case capacity, geometryBudget, instance, transform, tint, vertexDeformation, transmission, resource, drawBudget
        case geometry(Int), geometryHandle, paletteBusy, customOrder, reentrantSort
        case indirectUnsupported, indirectResource, indirectCommandBudget

        public var errorDescription: String? {
            switch self {
            case .capacity: "The batched mesh instance capacity is invalid or full."
            case .geometryBudget: "The batched mesh geometry palette exceeds the supported limits."
            case .instance: "The batched mesh instance handle is stale or belongs to a different batch."
            case .transform: "Batched mesh transforms must be finite, affine and invertible."
            case .tint: "Batched mesh tints must be finite linear RGBA."
            case .vertexDeformation: "A batched mesh material cannot deform its vertices with a surface graph."
            case .transmission: "The batched mesh material cannot sample the transmission backdrop."
            case .drawBudget: "Batched meshes exceeded the scene encode's instance snapshot budget."
            case .resource: "Metal could not allocate the batched mesh draw records."
            case .geometryHandle: "The batched mesh geometry handle is stale or belongs to a different batch."
            case .paletteBusy: "Complete or discard earlier batched mesh commands before changing the palette."
            case .customOrder: "Return every visible batch instance handle exactly once from the custom sort."
            case .reentrantSort: "Finish the custom sort before editing or encoding the batch."
            case .indirectUnsupported: "This Metal device cannot submit the batched mesh's indirect commands."
            case .indirectResource: "Metal could not allocate the batched mesh's indirect command buffer."
            case .indirectCommandBudget: "One batched mesh pass supports up to 16,384 indirect commands."
            case .geometry(let index): "Batched mesh geometry index \(index) does not refer to a live slot."
            }
        }
    }
}
