import Foundation
import Metal
import simd

// Bottom-level geometry, the software BVH builder and Metal acceleration structures for
// `PathTracingScene`.
extension PathTracingScene {
    // MARK: - Records

    /// One bottom-level vertex, matching PTVertex in PathTracingShading.h (48 bytes):
    /// object-space position, unit normal, then UV0 in xy and UV1 in zw. Colored geometry
    /// uses the position and normal w lanes for half-packed RG and BA. Metal builds read
    /// the first 12 bytes as the triangle position.
    package struct Vertex: Sendable {
        package var position: SIMD4<Float>
        package var normal: SIMD4<Float>
        package var uv: SIMD4<Float>
    }

    /// The work of one bottom-level build: object-space geometry after skinning.
    struct LevelJob: Sendable {
        let key: GeometryKey
        let nodeID: UUID
        let geometry: MeshGeometry
        let tracksPoseMotion: Bool
        var triangleCount: Int { geometry.indices.count / 3 }
    }

    /// A built bottom level. The buffers and structures are written once, before the level is
    /// published, and only read afterwards, by the GPU; Metal resources are safe to retain and
    /// release on any thread, so a finished level may cross isolation boundaries.
    package struct BottomLevel: @unchecked Sendable {
        let vertices: any MTLBuffer
        /// Three vertex indices per triangle: in software BVH leaf order, or in source order.
        let indices: any MTLBuffer
        /// For the software BVH: each stored triangle's source number, the primitive ID Metal reports.
        let primitives: (any MTLBuffer)?
        let nodes: (any MTLBuffer)?
        let structure: (any MTLAccelerationStructure)?
        let triangleCount: Int
        let nodeCount: Int
        let hasVertexColors: Bool
        /// The renderer's original vertex IDs still address its posed vertices.
        let tracksPoseMotion: Bool
        let minimum: SIMD3<Float>
        let maximum: SIMD3<Float>

        var buffers: [any MTLBuffer] { [vertices, indices] + [primitives, nodes].compactMap { $0 } }
    }

    /// Progress of the bottom-level builds of a preparation.
    public struct PreparationProgress: Equatable, Sendable {
        public let completedLevelCount: Int
        public let levelCount: Int
        /// Triangles in finished bottom levels.
        public let completedTriangleCount: Int
        public let triangleCount: Int

        /// Finished triangles over all triangles, from 0 to 1.
        public var fractionCompleted: Double {
            triangleCount == 0 ? 1 : Double(completedTriangleCount) / Double(triangleCount)
        }
    }

    // MARK: - Bottom levels

    /// Converts, validates and builds every job's level. Software levels build their BVHs on
    /// the CPU; Metal levels build and compact primitive acceleration structures on the GPU in
    /// batches, waiting for each. With `cancellable`, throws `CancellationError` between levels
    /// when the current task is cancelled.
    static func buildLevels(
        _ jobs: [LevelJob], queue: any MTLCommandQueue, metal: Bool, cancellable: Bool = false,
        progress: (@Sendable (PreparationProgress) -> Void)?
    ) throws -> [GeometryKey: BottomLevel] {
        let device = queue.device
        let total = jobs.reduce(0) { $0 + $1.triangleCount }
        var completedTriangles = 0
        var levels: [GeometryKey: BottomLevel] = [:]
        // Keep only the GPU copies and the metadata while a batch waits to build.
        var pending: [(
            key: GeometryKey, vertices: any MTLBuffer, indices: any MTLBuffer, triangleCount: Int,
            minimum: SIMD3<Float>, maximum: SIMD3<Float>, hasVertexColors: Bool, tracksPoseMotion: Bool
        )] = []
        var pendingTriangles = 0

        func report() {
            progress?(PreparationProgress(
                completedLevelCount: levels.count, levelCount: jobs.count,
                completedTriangleCount: completedTriangles, triangleCount: total
            ))
        }
        func flush() throws {
            guard !pending.isEmpty else { return }
            let structures = try buildPrimitiveStructures(
                pending.map { ($0.vertices, $0.indices, $0.triangleCount) }, queue: queue
            )
            for (entry, structure) in zip(pending, structures) {
                levels[entry.key] = BottomLevel(
                    vertices: entry.vertices, indices: entry.indices, primitives: nil, nodes: nil, structure: structure,
                    triangleCount: entry.triangleCount, nodeCount: 0, hasVertexColors: entry.hasVertexColors,
                    tracksPoseMotion: entry.tracksPoseMotion,
                    minimum: entry.minimum, maximum: entry.maximum
                )
                completedTriangles += entry.triangleCount
            }
            pending.removeAll()
            pendingTriangles = 0
            report()
        }

        func build(_ job: LevelJob) throws {
            let geometry = try levelGeometry(job.geometry, nodeID: job.nodeID)
            let hasVertexColors = job.geometry.hasVertexColors
            if metal {
                let vertices = try makeBuffer(geometry.vertices, device: device, label: "path bottom-level vertices")
                let indices = try makeBuffer(geometry.indices, device: device, label: "path bottom-level indices")
                pending.append((
                    job.key, vertices, indices, job.triangleCount, geometry.minimum, geometry.maximum, hasVertexColors,
                    job.tracksPoseMotion
                ))
                pendingTriangles += job.triangleCount
                // About 1.5 GB of uncompacted structures and scratch per batch at roughly 100 bytes a triangle.
                if pendingTriangles >= 16_000_000 { try flush() }
                return
            }
            let tree = try buildBVH(
                vertices: geometry.vertices.map { SIMD3($0.position.x, $0.position.y, $0.position.z) }, indices: geometry.indices
            )
            levels[job.key] = BottomLevel(
                vertices: try makeBuffer(geometry.vertices, device: device, label: "path bottom-level vertices"),
                indices: try makeBuffer(tree.indices, device: device, label: "path bottom-level indices"),
                primitives: try makeBuffer(tree.primitives, device: device, label: "path bottom-level primitive numbers"),
                nodes: try makeBuffer(tree.nodes, device: device, label: "path bottom-level BVH"),
                structure: nil, triangleCount: tree.primitives.count, nodeCount: tree.nodes.count,
                hasVertexColors: hasVertexColors, tracksPoseMotion: job.tracksPoseMotion,
                minimum: geometry.minimum, maximum: geometry.maximum
            )
            completedTriangles += tree.primitives.count
            report()
        }

        // Metal returns some new resources autoreleased; each level drains its own pool.
        for job in jobs {
            if cancellable { try Task.checkCancellation() }
            try autoreleasepool { try build(job) }
        }
        if cancellable { try Task.checkCancellation() }
        try autoreleasepool { try flush() }
        return levels
    }

    struct LevelGeometry {
        let vertices: [Vertex]
        let indices: [UInt32]
        let minimum: SIMD3<Float>
        let maximum: SIMD3<Float>
    }

    /// Packs vertices, rejecting non-finite or out-of-range positions, zero normals and
    /// zero-area triangles, and bounds the positions that triangles use.
    static func levelGeometry(_ source: MeshGeometry, nodeID: UUID) throws -> LevelGeometry {
        let count = source.indices.count / 3
        guard count > 0 else { throw Failure.emptyScene }
        guard count <= maximumTriangleCount, source.vertices.count <= maximumTriangleCount * 3 else {
            throw Failure.triangleLimit
        }
        let hasVertexColors = source.hasVertexColors
        func pack(_ low: Float, _ high: Float) -> Float {
            Float(bitPattern: UInt32(Float16(low).bitPattern) | UInt32(Float16(high).bitPattern) << 16)
        }
        var vertices: [Vertex] = []
        vertices.reserveCapacity(source.vertices.count)
        for vertex in source.vertices {
            let normal = vertex.normal
            let magnitude = max(abs(normal.x), max(abs(normal.y), abs(normal.z)))
            let position = vertex.position
            guard position.x.isFinite, position.y.isFinite, position.z.isFinite,
                  simd_reduce_max(abs(position)) <= 1e12, magnitude.isFinite, magnitude > 0 else {
                throw Failure.invalidGeometry(nodeID)
            }
            let second = vertex.secondaryTextureCoordinate ?? .zero
            var lanes = SIMD2<Float>(1, 0)
            if hasVertexColors {
                // Keep half values finite so the packed float bit patterns also stay finite.
                let color = simd_clamp(vertex.color ?? SIMD4(repeating: 1), SIMD4(repeating: -65_504), SIMD4(repeating: 65_504))
                lanes = SIMD2(pack(color.x, color.y), pack(color.z, color.w))
            }
            vertices.append(Vertex(
                position: SIMD4(position, lanes.x), normal: SIMD4(simd_normalize(normal / magnitude), lanes.y),
                uv: SIMD4(vertex.textureCoordinate.x, vertex.textureCoordinate.y, second.x, second.y)
            ))
        }
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = -minimum
        try source.indices.withUnsafeBufferPointer { indices in
            for triangle in 0..<count {
                let a = source.vertices[Int(indices[3 * triangle])].position
                let b = source.vertices[Int(indices[3 * triangle + 1])].position
                let c = source.vertices[Int(indices[3 * triangle + 2])].position
                guard simd_cross(b - a, c - a) != .zero else { throw Failure.degenerateTriangle(nodeID, triangle) }
                minimum = simd_min(minimum, simd_min(a, simd_min(b, c)))
                maximum = simd_max(maximum, simd_max(a, simd_max(b, c)))
            }
        }
        return LevelGeometry(vertices: vertices, indices: source.indices, minimum: minimum, maximum: maximum)
    }

    /// Builds a bottom level whose leaves hold up to four triangles: the index triples in leaf
    /// order, each stored triangle's source number, and the nodes.
    package static func buildBVH(
        vertices: [SIMD3<Float>], indices: [UInt32]
    ) throws -> (indices: [UInt32], primitives: [UInt32], nodes: [BVHNode]) {
        let count = indices.count / 3
        guard count > 0 else { throw Failure.emptyScene }
        guard count <= maximumTriangleCount else { throw Failure.triangleLimit }
        var minimum = [SIMD3<Float>](repeating: .zero, count: count)
        var maximum = [SIMD3<Float>](repeating: .zero, count: count)
        for triangle in 0..<count {
            let a = vertices[Int(indices[3 * triangle])]
            let b = vertices[Int(indices[3 * triangle + 1])]
            let c = vertices[Int(indices[3 * triangle + 2])]
            minimum[triangle] = simd_min(a, simd_min(b, c))
            maximum[triangle] = simd_max(a, simd_max(b, c))
        }
        let tree = try buildTree(minimum: minimum, maximum: maximum, leafSize: 4)
        var ordered = [UInt32](repeating: 0, count: 3 * count)
        for (position, triangle) in tree.order.enumerated() {
            let source = 3 * Int(triangle)
            ordered[3 * position] = indices[source]
            ordered[3 * position + 1] = indices[source + 1]
            ordered[3 * position + 2] = indices[source + 2]
        }
        return (ordered, tree.order, tree.nodes)
    }

    // MARK: - Software BVH

    /// Builds a binary tree over primitive bounds with a binned surface area heuristic (16 bins
    /// per axis). Leaves store ranges of `order` of at most `leafSize` primitives; children
    /// always follow their parent, so a reverse pass refits the tree. A node falls back to an
    /// object-median split along its widest centroid axis when the heuristic finds no split or
    /// when its deeper child could exceed `maximumBVHDepth`, so depth stays within the
    /// traversal stack: a median split needs at most ⌈log₂(n / leafSize)⌉ further levels.
    static func buildTree(
        minimum: [SIMD3<Float>], maximum: [SIMD3<Float>], leafSize: Int
    ) throws -> (order: [UInt32], nodes: [BVHNode]) {
        let count = minimum.count
        guard count > 0 else { throw Failure.emptyScene }
        func levelsNeeded(_ primitives: Int) -> Int {
            var leaves = (primitives + leafSize - 1) / leafSize
            var levels = 0
            while leaves > 1 {
                leaves = (leaves + 1) / 2
                levels += 1
            }
            return levels
        }
        guard 1 + levelsNeeded(count) <= maximumBVHDepth else { throw Failure.bvhDepth }

        let centroids = zip(minimum, maximum).map { ($0 + $1) * 0.5 }
        var order = (0..<count).map { UInt32($0) }
        var nodes: [BVHNode] = []
        nodes.reserveCapacity(2 * ((count + leafSize - 1) / leafSize))
        // Pending ranges with their depth and the parent slot to point at them.
        var pending: [(start: Int, end: Int, depth: Int, parent: Int, right: Bool)] = [(0, count, 1, -1, false)]
        let binCount = 16
        // Bin bounds, counts and suffix sums, reused by every node.
        var binLow = [SIMD3<Float>](repeating: .zero, count: binCount)
        var binHigh = [SIMD3<Float>](repeating: .zero, count: binCount)
        var binSize = [Int](repeating: 0, count: binCount)
        var rightArea = [Float](repeating: 0, count: binCount)
        var rightSize = [Int](repeating: 0, count: binCount)

        try minimum.withUnsafeBufferPointer { lows in
            try maximum.withUnsafeBufferPointer { highs in
                try centroids.withUnsafeBufferPointer { centers in
                    try order.withUnsafeMutableBufferPointer { order in
                        while let task = pending.popLast() {
                            let nodeIndex = nodes.count
                            if task.parent >= 0 {
                                if task.right { nodes[task.parent].meta.y = UInt32(nodeIndex) } else {
                                    nodes[task.parent].meta.x = UInt32(nodeIndex)
                                }
                            }
                            var low = lows[Int(order[task.start])]
                            var high = highs[Int(order[task.start])]
                            var centerLow = centers[Int(order[task.start])]
                            var centerHigh = centerLow
                            for position in (task.start + 1)..<max(task.start + 1, task.end) {
                                let primitive = Int(order[position])
                                low = simd_min(low, lows[primitive])
                                high = simd_max(high, highs[primitive])
                                centerLow = simd_min(centerLow, centers[primitive])
                                centerHigh = simd_max(centerHigh, centers[primitive])
                            }
                            nodes.append(BVHNode(minimum: SIMD4(low, 0), maximum: SIMD4(high, 0), meta: .zero))
                            let size = task.end - task.start
                            if size <= leafSize {
                                nodes[nodeIndex].meta = SIMD4(0, 0, UInt32(task.start), UInt32(size))
                                continue
                            }
                            guard task.depth < maximumBVHDepth else { throw Failure.bvhDepth }
                            let extent = centerHigh - centerLow
                            var middle = -1
                            // Binned SAH over every axis with a centroid extent.
                            var bestCost = Float.infinity
                            var bestAxis = -1
                            var bestBin = 0
                            for axis in 0..<3 where extent[axis] > 0 {
                                for bin in 0..<binCount {
                                    binLow[bin] = SIMD3(repeating: .greatestFiniteMagnitude)
                                    binHigh[bin] = SIMD3(repeating: -.greatestFiniteMagnitude)
                                    binSize[bin] = 0
                                }
                                let scale = Float(binCount) / extent[axis]
                                // An extremely narrow axis can overflow the reciprocal. Skip it
                                // and allow another axis or the median split to separate the leaves.
                                guard scale.isFinite else { continue }
                                for position in task.start..<task.end {
                                    let primitive = Int(order[position])
                                    let bin = min(binCount - 1, Int((centers[primitive][axis] - centerLow[axis]) * scale))
                                    binLow[bin] = simd_min(binLow[bin], lows[primitive])
                                    binHigh[bin] = simd_max(binHigh[bin], highs[primitive])
                                    binSize[bin] += 1
                                }
                                var accumulatedLow = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
                                var accumulatedHigh = -accumulatedLow
                                var accumulated = 0
                                for bin in stride(from: binCount - 1, to: 0, by: -1) {
                                    accumulatedLow = simd_min(accumulatedLow, binLow[bin])
                                    accumulatedHigh = simd_max(accumulatedHigh, binHigh[bin])
                                    accumulated += binSize[bin]
                                    rightArea[bin] = Self.area(accumulatedLow, accumulatedHigh)
                                    rightSize[bin] = accumulated
                                }
                                accumulatedLow = SIMD3(repeating: .greatestFiniteMagnitude)
                                accumulatedHigh = -accumulatedLow
                                accumulated = 0
                                for bin in 1..<binCount {
                                    accumulatedLow = simd_min(accumulatedLow, binLow[bin - 1])
                                    accumulatedHigh = simd_max(accumulatedHigh, binHigh[bin - 1])
                                    accumulated += binSize[bin - 1]
                                    guard accumulated > 0, rightSize[bin] > 0 else { continue }
                                    let cost = Self.area(accumulatedLow, accumulatedHigh) * Float(accumulated)
                                        + rightArea[bin] * Float(rightSize[bin])
                                    // Both children must still fit within the depth bound.
                                    let deeper = max(accumulated, rightSize[bin])
                                    if cost < bestCost, task.depth + 1 + levelsNeeded(deeper) <= maximumBVHDepth {
                                        bestCost = cost
                                        bestAxis = axis
                                        bestBin = bin
                                    }
                                }
                            }
                            if bestAxis >= 0 {
                                let scale = Float(binCount) / extent[bestAxis]
                                var left = task.start
                                var right = task.end - 1
                                while left <= right {
                                    let primitive = Int(order[left])
                                    let bin = min(binCount - 1, Int((centers[primitive][bestAxis] - centerLow[bestAxis]) * scale))
                                    if bin < bestBin {
                                        left += 1
                                    } else {
                                        order.swapAt(left, right)
                                        right -= 1
                                    }
                                }
                                if left > task.start && left < task.end { middle = left }
                            }
                            if middle < 0 {
                                // Object median along the widest centroid axis, ties broken by primitive number.
                                let axis = extent.x >= max(extent.y, extent.z) ? 0 : (extent.y >= extent.z ? 1 : 2)
                                var range = UnsafeMutableBufferPointer(rebasing: order[task.start..<task.end])
                                range.sort { a, b in
                                    let x = centers[Int(a)][axis]
                                    let y = centers[Int(b)][axis]
                                    return x == y ? a < b : x < y
                                }
                                middle = task.start + size / 2
                            }
                            // The right range is pushed first, so the left child follows its parent directly.
                            pending.append((middle, task.end, task.depth + 1, nodeIndex, true))
                            pending.append((task.start, middle, task.depth + 1, nodeIndex, false))
                        }
                    }
                }
            }
        }
        return (order, nodes)
    }

    private static func area(_ low: SIMD3<Float>, _ high: SIMD3<Float>) -> Float {
        let size = simd_max(high - low, .zero)
        return size.x * size.y + size.y * size.z + size.z * size.x
    }

    // MARK: - Metal acceleration structures

    /// Builds one compacted primitive acceleration structure per level on the GPU. Each
    /// level is one opaque triangle geometry over its vertex positions and indices; alpha
    /// tests come from per-instance opacity. Waits for the builds and for compaction.
    static func buildPrimitiveStructures(
        _ levels: [(vertices: any MTLBuffer, indices: any MTLBuffer, triangleCount: Int)], queue: any MTLCommandQueue
    ) throws -> [any MTLAccelerationStructure] {
        let device = queue.device
        let descriptors = levels.map { level -> MTLPrimitiveAccelerationStructureDescriptor in
            let geometry = MTLAccelerationStructureTriangleGeometryDescriptor()
            geometry.vertexBuffer = level.vertices
            geometry.vertexBufferOffset = 0
            geometry.vertexStride = MemoryLayout<Vertex>.stride
            geometry.vertexFormat = .float3
            geometry.indexBuffer = level.indices
            geometry.indexType = .uint32
            geometry.triangleCount = level.triangleCount
            geometry.opaque = true
            // Stochastic coverage tests each candidate once.
            geometry.allowDuplicateIntersectionFunctionInvocation = false
            let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = [geometry]
            return descriptor
        }
        var built: [any MTLAccelerationStructure] = []
        var scratch: [any MTLBuffer] = []
        for descriptor in descriptors {
            let sizes = device.accelerationStructureSizes(descriptor: descriptor)
            guard let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let buffer = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16), options: .storageModePrivate) else {
                throw Failure.resource("path primitive acceleration structure")
            }
            structure.label = "Frostfield path bottom level"
            built.append(structure)
            scratch.append(buffer)
        }
        guard let sizes = device.makeBuffer(length: 8 * levels.count, options: .storageModeShared),
              let buffer = queue.makeCommandBuffer(), let encoder = buffer.makeAccelerationStructureCommandEncoder() else {
            throw Failure.resource("path acceleration structure encoder")
        }
        buffer.label = "Frostfield path bottom-level builds"
        for index in descriptors.indices {
            encoder.build(
                accelerationStructure: built[index], descriptor: descriptors[index], scratchBuffer: scratch[index],
                scratchBufferOffset: 0
            )
        }
        encoder.endEncoding()
        // Compacted sizes are written after every build in the previous pass completes.
        guard let measure = buffer.makeAccelerationStructureCommandEncoder() else {
            throw Failure.resource("path acceleration structure encoder")
        }
        for index in descriptors.indices {
            measure.writeCompactedSize(accelerationStructure: built[index], buffer: sizes, offset: 8 * index, sizeDataType: .ulong)
        }
        measure.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else { throw Failure.accelerationStructure(buffer.error?.localizedDescription) }
        scratch.removeAll()

        let compactedSizes = (0..<levels.count).map { sizes.contents().load(fromByteOffset: 8 * $0, as: UInt64.self) }
        var compacted: [any MTLAccelerationStructure] = []
        for size in compactedSizes {
            guard let structure = device.makeAccelerationStructure(size: Int(size)) else {
                throw Failure.resource("path compacted acceleration structure")
            }
            structure.label = "Frostfield path bottom level"
            compacted.append(structure)
        }
        guard let copy = queue.makeCommandBuffer(), let encoder = copy.makeAccelerationStructureCommandEncoder() else {
            throw Failure.resource("path acceleration structure encoder")
        }
        copy.label = "Frostfield path bottom-level compaction"
        for index in built.indices {
            encoder.copyAndCompact(sourceAccelerationStructure: built[index], destinationAccelerationStructure: compacted[index])
        }
        encoder.endEncoding()
        copy.commit()
        copy.waitUntilCompleted()
        guard copy.status == .completed else { throw Failure.accelerationStructure(copy.error?.localizedDescription) }
        return compacted
    }

    /// The instance acceleration structure over every instance, in instance order, and the
    /// descriptor that a refit reuses.
    struct MetalTopLevel {
        let structure: any MTLAccelerationStructure
        let descriptor: MTLInstanceAccelerationStructureDescriptor
        let descriptorBuffer: any MTLBuffer
        /// The bottom-level structures in descriptor index order.
        let levels: [any MTLAccelerationStructure]
    }

    /// Writes one Metal instance descriptor per instance. Instances with an alpha test are
    /// non-opaque; double-sided ones disable culling.
    static func metalInstanceDescriptors(
        _ instances: [Instance], levelIndices: [GeometryKey: UInt32], device: any MTLDevice
    ) throws -> any MTLBuffer {
        let stride = MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride
        guard let buffer = device.makeBuffer(
            length: max(stride * instances.count, stride), options: [.storageModeShared, .hazardTrackingModeTracked]
        ) else { throw Failure.resource("path instance descriptors") }
        buffer.label = "Frostfield path instance descriptors"
        let descriptors = buffer.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: instances.count)
        for (index, instance) in instances.enumerated() {
            guard let level = levelIndices[instance.geometry] else { throw Failure.resource("path bottom level") }
            let world = instance.world
            func column(_ value: SIMD4<Float>) -> MTLPackedFloat3 { MTLPackedFloat3Make(value.x, value.y, value.z) }
            var descriptor = MTLAccelerationStructureInstanceDescriptor()
            descriptor.transformationMatrix = MTLPackedFloat4x3(columns: (
                column(world.columns.0), column(world.columns.1), column(world.columns.2), column(world.columns.3)
            ))
            var options: MTLAccelerationStructureInstanceOptions = instance.alphaTested ? .nonOpaque : .opaque
            if instance.cullsBackFaces == 0 || instance.renderer.renderState.shadowSide != nil {
                options.insert(.disableTriangleCulling)
            }
            descriptor.options = options
            descriptor.mask = 0xff
            descriptor.intersectionFunctionTableOffset = 0
            descriptor.accelerationStructureIndex = level
            descriptors[index] = descriptor
        }
        return buffer
    }

    /// Builds the instance structure, or refits `previous` into a new structure when given,
    /// so encoded command buffers keep reading the previous one. Waits for completion
    /// before the scene publishes the new structure.
    static func metalTopLevel(
        instances: [Instance], bottomLevels: [GeometryKey: BottomLevel], refitting previous: MetalTopLevel?,
        queue: any MTLCommandQueue
    ) throws -> MetalTopLevel {
        let device = queue.device
        let levels: [any MTLAccelerationStructure]
        var levelIndices: [GeometryKey: UInt32] = [:]
        if let previous {
            levels = previous.levels
            // A refit keeps the bottom-level order of the previous build.
            for (key, level) in bottomLevels {
                guard let structure = level.structure,
                      let index = levels.firstIndex(where: { $0 === structure }) else { throw Failure.resource("path bottom level") }
                levelIndices[key] = UInt32(index)
            }
        } else {
            var ordered: [any MTLAccelerationStructure] = []
            for (key, level) in bottomLevels {
                guard let structure = level.structure else { throw Failure.resource("path bottom level") }
                levelIndices[key] = UInt32(ordered.count)
                ordered.append(structure)
            }
            levels = ordered
        }
        let descriptorBuffer = try metalInstanceDescriptors(instances, levelIndices: levelIndices, device: device)
        let descriptor = MTLInstanceAccelerationStructureDescriptor()
        descriptor.instancedAccelerationStructures = levels
        descriptor.instanceCount = instances.count
        descriptor.instanceDescriptorBuffer = descriptorBuffer
        descriptor.instanceDescriptorType = .default
        descriptor.usage = .refit
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        let scratchSize = previous == nil ? sizes.buildScratchBufferSize : sizes.refitScratchBufferSize
        guard let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: max(scratchSize, 16), options: .storageModePrivate),
              let buffer = queue.makeCommandBuffer(), let encoder = buffer.makeAccelerationStructureCommandEncoder() else {
            throw Failure.resource("path instance acceleration structure")
        }
        structure.label = "Frostfield path top level"
        buffer.label = previous == nil ? "Frostfield path top-level build" : "Frostfield path top-level refit"
        if let previous {
            encoder.refit(
                sourceAccelerationStructure: previous.structure, descriptor: descriptor,
                destinationAccelerationStructure: structure, scratchBuffer: scratch, scratchBufferOffset: 0
            )
        } else {
            encoder.build(accelerationStructure: structure, descriptor: descriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
        }
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else { throw Failure.accelerationStructure(buffer.error?.localizedDescription) }
        return MetalTopLevel(structure: structure, descriptor: descriptor, descriptorBuffer: descriptorBuffer, levels: levels)
    }

    static func makeBuffer<T>(_ values: [T], device: any MTLDevice, label: String) throws -> any MTLBuffer {
        let buffer = values.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap {
                device.makeBuffer(bytes: $0, length: bytes.count, options: [.storageModeShared, .hazardTrackingModeTracked])
            }
        }
        guard let buffer else { throw Failure.resource(label) }
        buffer.label = "Frostfield \(label)"
        return buffer
    }
}
