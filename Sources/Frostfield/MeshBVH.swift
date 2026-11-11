import Foundation
import Metal
import simd
import Synchronization

/// A CPU bounding-volume hierarchy over one mesh's triangles for picking,
/// proximity, and collision queries.
///
/// The build bins triangle centroids into 16 buckets per axis and splits where the
/// surface area heuristic is cheapest. It stops at one triangle, at
/// ``maximumDepth``, when a leaf of up to ``preferredLeafTriangleCount`` triangles
/// is cheaper than splitting, or when centroids coincide, as with duplicate or
/// degenerate triangles. Leaves can therefore exceed the preferred size.
///
/// Memory is 48 bytes per node and 4 bytes per triangle, in addition to the
/// geometry, whose arrays the hierarchy shares and keeps alive. A build makes at
/// most `2 × triangleCount − 1` nodes; leaves average about two triangles, so a
/// typical mesh needs about 52 bytes per triangle. A 65,024-triangle sphere builds
/// 63,943 nodes, 19 levels deep.
///
/// Ray queries return exactly what ``MeshGeometry`` returns for the same ray:
/// both use one triangle test, culling rule, distance and tie rule. Node bounds
/// are widened per query by a relative tolerance of 1e-6 of the query's
/// coordinate scale, so rounding at triangle edges cannot discard a hit the
/// linear scan reports. Only a ray parallel to a triangle's plane to within
/// rounding, where the triangle test itself is ill-conditioned, could differ.
public struct MeshBVH: Sendable {
    /// The deepest leaf a build creates, which bounds every traversal stack.
    public static let maximumDepth = 64
    /// The largest leaf the surface area heuristic chooses over a split.
    public static let preferredLeafTriangleCount = 4
    fileprivate static let binCount = 16
    fileprivate static let boundsTolerance = 1e-6

    /// The indexed triangles in this immutable hierarchy snapshot.
    public let geometry: MeshGeometry
    /// The deepest leaf, counting the root as depth one.
    public let depth: Int
    private let nodes: [Node]
    /// Original triangle indices in leaf order.
    private let order: [UInt32]
    private let refitTopology: SelectiveRefitTopology?

    public var nodeCount: Int { nodes.count }
    public var triangleCount: Int { order.count }
    /// The box enclosing every triangle in the geometry's local coordinates.
    public var bounds: AxisAlignedBox { AxisAlignedBox(validatedMinimum: nodes[0].lower, maximum: nodes[0].upper) }

    struct Node {
        var lower: SIMD3<Float>
        var upper: SIMD3<Float>
        /// A leaf's first slot in `order`, or an interior node's right child.
        var start: UInt32
        /// A leaf's triangle count, or zero for an interior node, whose left child follows it.
        var count: UInt32
    }

    /// Builds the hierarchy in `O(n log n)` time for `n` triangles. Throws when
    /// the geometry has more triangles than a 32-bit triangle index can address.
    public init(geometry: MeshGeometry) throws {
        try self.init(geometry: geometry, cancellable: false, progress: nil)
    }

    /// Work completed by a concurrent build. Bounds and hierarchy indexing
    /// contribute equally to the fraction, independent of elapsed time.
    public struct BuildProgress: Equatable, Sendable {
        public let completedBoundsCount: Int
        public let indexedTriangleCount: Int
        public let triangleCount: Int

        public var fractionCompleted: Double {
            Double(completedBoundsCount + indexedTriangleCount) / Double(2 * triangleCount)
        }
    }

    /// Builds the same immutable hierarchy as the synchronous initializer,
    /// off the caller's actor. The original geometry remains available during
    /// the build. Progress callbacks run on the build task's executor.
    /// Cancelling the task throws CancellationError before a tree is returned.
    @concurrent public static func build(
        geometry: MeshGeometry, progress: (@Sendable (BuildProgress) -> Void)? = nil
    ) async throws -> MeshBVH {
        try Task.checkCancellation()
        let result = try MeshBVH(geometry: geometry, cancellable: true, progress: progress)
        try Task.checkCancellation()
        let count = result.triangleCount
        progress?(BuildProgress(completedBoundsCount: count, indexedTriangleCount: count, triangleCount: count))
        try Task.checkCancellation()
        return result
    }

    private init(
        geometry: MeshGeometry, cancellable: Bool, progress: (@Sendable (BuildProgress) -> Void)?
    ) throws {
        let triangleCount = geometry.indices.count / 3
        guard triangleCount <= Int(UInt32.max) else { throw Failure.triangleLimit }
        var builder = try Builder(
            geometry: geometry, triangleCount: triangleCount, cancellable: cancellable, progress: progress
        )
        _ = try builder.build(0..<triangleCount, depth: 1)
        self.init(geometry: geometry, nodes: builder.nodes, order: builder.order, depth: builder.deepest, refitTopology: nil)
    }

    private init(
        geometry: MeshGeometry, nodes: [Node], order: [UInt32], depth: Int, refitTopology: SelectiveRefitTopology?
    ) {
        self.geometry = geometry
        self.nodes = nodes
        self.order = order
        self.depth = depth
        self.refitTopology = refitTopology
    }

    /// Returns a new hierarchy with this tree's topology and bounds updated
    /// to the supplied mesh. Vertex slots must correspond, with the same
    /// vertex count and identical indices in the same order. Positions and
    /// other vertex attributes may change.
    ///
    /// Refit takes linear time in triangles and nodes and preserves the
    /// original hierarchy for concurrent queries. A fresh build may prune
    /// more effectively after large changes in relative vertex placement.
    public func refitted(to geometry: MeshGeometry) throws -> MeshBVH {
        guard self.geometry.vertices.count == geometry.vertices.count, self.geometry.indices == geometry.indices else {
            throw Failure.refitTopologyMismatch
        }
        var updated = nodes
        // Children follow their parent in the packed tree, so their new
        // bounds are ready by the time the parent is visited.
        for index in nodes.indices.reversed() {
            refitBounds(at: index, in: &updated, geometry: geometry)
        }
        return MeshBVH(geometry: geometry, nodes: updated, order: order, depth: depth, refitTopology: refitTopology)
    }

    /// Replaces positions at the given source vertex indices and returns a
    /// new hierarchy. Every leaf containing an edited vertex and its ancestors
    /// are refitted. Indices, other vertex attributes, material groups and the
    /// original snapshot are preserved. Empty or unchanged updates return
    /// the same snapshot.
    ///
    /// The first edit builds reverse adjacency, which later refits of the
    /// returned snapshot reuse. The positions and nodes use copy-on-write
    /// storage. An unreferenced vertex changes the geometry without changing
    /// any node bounds.
    public func refitted(updatingPositions positions: [Int: SIMD3<Float>]) throws -> MeshBVH {
        var changes: [(index: Int, position: SIMD3<Float>)] = []
        changes.reserveCapacity(positions.count)
        for (index, position) in positions.sorted(by: { $0.key < $1.key }) {
            guard geometry.vertices.indices.contains(index) else { throw Failure.refitVertexIndex(index) }
            guard boundsFinite(position) else { throw Failure.nonfiniteRefitPosition(index) }
            if position != geometry.vertices[index].position { changes.append((index, position)) }
        }
        guard !changes.isEmpty else { return self }

        let topology = refitTopology ?? SelectiveRefitTopology(geometry: geometry, nodes: nodes, order: order)
        let revised = geometry.replacingValidatedPositions(changes)
        var dirty = Set<Int>()
        for change in changes {
            topology.insertAffectedNodes(forVertex: change.index, into: &dirty)
        }
        var updated = nodes
        for index in dirty.sorted(by: >) {
            refitBounds(at: index, in: &updated, geometry: revised)
        }
        return MeshBVH(geometry: revised, nodes: updated, order: order, depth: depth, refitTopology: topology)
    }

    private func refitBounds(at index: Int, in updated: inout [Node], geometry: MeshGeometry) {
        let original = nodes[index]
        var revised = original
        if original.count > 0 {
            revised.lower = SIMD3(repeating: .infinity)
            revised.upper = SIMD3(repeating: -.infinity)
            let start = Int(original.start)
            for slot in start..<(start + Int(original.count)) {
                let triangle = geometry.triangleBounds(Int(order[slot]))
                revised.lower = simd_min(revised.lower, triangle.lower)
                revised.upper = simd_max(revised.upper, triangle.upper)
            }
        } else {
            let left = updated[index + 1], right = updated[Int(original.start)]
            revised.lower = simd_min(left.lower, right.lower)
            revised.upper = simd_max(left.upper, right.upper)
        }
        if revised.lower != original.lower || revised.upper != original.upper {
            updated[index] = revised
        }
    }

    /// Packs this immutable snapshot for local-space ray queries in a Metal shader.
    public func makeShaderBuffer(device: any MTLDevice) throws -> MeshBVHShaderBuffer {
        try MeshBVHShaderBuffer(device: device, geometry: geometry, nodes: nodes, order: order)
    }

    /// Finds the nearest indexed triangle exactly as
    /// ``MeshGeometry/intersection(ray:localToWorld:cullsBackFaces:)`` does.
    public func intersection(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, cullsBackFaces: Bool = true
    ) throws -> MeshRayHit? {
        try intersection(ray: ray, localToWorld: localToWorld, side: cullsBackFaces ? .front : .double)
    }

    /// Finds the nearest triangle on the supplied material side.
    public func intersection(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, side: MeshRenderState.Side
    ) throws -> MeshRayHit? {
        try intersection(ray: ray, localToWorld: localToWorld, side: side, reporting: geometry)
    }

    fileprivate func intersection(
        ray: SceneRay, localToWorld: simd_float4x4, side: MeshRenderState.Side, reporting attributes: MeshGeometry
    ) throws -> MeshRayHit? {
        let transform = try PickingAffine(localToWorld)
        let query = PickingQuery(ray: ray, transform: transform, side: side)
        guard let hit = nearestTriangle(query, ignoringTriangle: nil) else { return nil }
        return try query.hit(distance: hit.distance, triangleIndex: hit.triangle, in: attributes, transform: transform)
    }

    /// Called only for geometrically valid candidates in an optional masked traversal.
    typealias HitFilter = (PickingQuery, Int) -> Bool

    /// Keeps baked transport in Double and excludes the triangle that launched the ray.
    func nearestTriangle(
        origin: SIMD3<Double>, unitDirection: SIMD3<Double>, ignoringTriangle: Int?, acceptsHit: HitFilter? = nil
    ) -> (distance: Double, triangle: Int)? {
        nearestTriangle(
            PickingQuery(origin: origin, unitDirection: unitDirection),
            ignoringTriangle: ignoringTriangle, acceptsHit: acceptsHit
        )
    }

    private func nearestTriangle(
        _ query: PickingQuery, ignoringTriangle: Int?, acceptsHit: HitFilter? = nil
    ) -> (distance: Double, triangle: Int)? {
        let slab = Slab(query: query, root: nodes[0])
        var closestDistance = query.maximumDistance
        var closestTriangle: Int?
        guard let rootEntry = slab.entry(nodes[0], limit: closestDistance) else { return nil }
        // At most one unvisited sibling per level, plus the root.
        var stack = InlineArray<64, (node: Int, entry: Double)>(repeating: (0, 0))
        stack[0] = (0, rootEntry)
        var pending = 1

        while pending > 0 {
            pending -= 1
            let (start, entry) = stack[pending]
            guard entry <= closestDistance else { continue }
            var index = start
            descent: while true {
                let node = nodes[index]
                if node.count > 0 {
                    for slot in Int(node.start)..<Int(node.start + node.count) {
                        let triangle = Int(order[slot])
                        guard triangle != ignoringTriangle,
                              let distance = query.distance(to: triangle, in: geometry) else { continue }
                        // The linear scan keeps the lowest triangle index among equal distances.
                        if let current = closestTriangle,
                           distance > closestDistance || (distance == closestDistance && triangle > current) {
                            continue
                        }
                        guard acceptsHit?(query, triangle) != false else { continue }
                        closestDistance = distance
                        closestTriangle = triangle
                    }
                    break descent
                }
                let left = index + 1
                let right = Int(node.start)
                let leftEntry = slab.entry(nodes[left], limit: closestDistance)
                let rightEntry = slab.entry(nodes[right], limit: closestDistance)
                switch (leftEntry, rightEntry) {
                case let (near?, far?):
                    if near <= far {
                        stack[pending] = (right, far)
                        index = left
                    } else {
                        stack[pending] = (left, near)
                        index = right
                    }
                    pending += 1
                case (_?, nil): index = left
                case (nil, _?): index = right
                case (nil, nil): break descent
                }
            }
        }

        guard let closestTriangle else { return nil }
        return (closestDistance, closestTriangle)
    }

    /// Early-out visibility query for geometry already in world coordinates.
    func occludes(
        origin: SIMD3<Double>, unitDirection: SIMD3<Double>, ignoringTriangle: Int?, maximumDistance: Double = .infinity,
        acceptsHit: HitFilter? = nil
    ) -> Bool {
        let query = PickingQuery(origin: origin, unitDirection: unitDirection, maximumDistance: maximumDistance)
        let slab = Slab(query: query, root: nodes[0])
        var stack = InlineArray<64, Int>(repeating: 0)
        var count = 1
        while count > 0 {
            count -= 1
            let index = stack[count]
            let node = nodes[index]
            guard slab.entry(node, limit: query.maximumDistance) != nil else { continue }
            if node.count > 0 {
                for slot in Int(node.start)..<Int(node.start + node.count) {
                    let triangle = Int(order[slot])
                    if triangle != ignoringTriangle, query.distance(to: triangle, in: geometry) != nil,
                       acceptsHit?(query, triangle) != false { return true }
                }
            } else {
                stack[count] = Int(node.start)
                stack[count + 1] = index + 1
                count += 2
            }
        }
        return false
    }

    /// Finds every triangle the ray crosses exactly as
    /// ``MeshGeometry/intersections(ray:localToWorld:cullsBackFaces:)`` does, nearest first.
    public func intersections(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, cullsBackFaces: Bool = true
    ) throws -> [MeshRayHit] {
        try intersections(ray: ray, localToWorld: localToWorld, side: cullsBackFaces ? .front : .double)
    }

    /// Finds every triangle on the supplied material side, nearest first.
    public func intersections(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, side: MeshRenderState.Side
    ) throws -> [MeshRayHit] {
        try intersections(ray: ray, localToWorld: localToWorld, side: side, reporting: geometry)
    }

    fileprivate func intersections(
        ray: SceneRay, localToWorld: simd_float4x4, side: MeshRenderState.Side, reporting attributes: MeshGeometry
    ) throws -> [MeshRayHit] {
        let transform = try PickingAffine(localToWorld)
        let query = PickingQuery(ray: ray, transform: transform, side: side)
        let slab = Slab(query: query, root: nodes[0])
        var hits: [(distance: Double, triangle: Int)] = []
        var stack = [0]
        stack.reserveCapacity(Self.maximumDepth)
        while let index = stack.popLast() {
            let node = nodes[index]
            guard slab.entry(node, limit: query.maximumDistance) != nil else { continue }
            if node.count > 0 {
                for slot in Int(node.start)..<Int(node.start + node.count) {
                    let triangle = Int(order[slot])
                    if let distance = query.distance(to: triangle, in: geometry) { hits.append((distance, triangle)) }
                }
            } else {
                stack.append(Int(node.start))
                stack.append(index + 1)
            }
        }
        return try query.sortedHits(hits, in: attributes, transform: transform)
    }

    /// Finds the nearest point on the closed mesh triangles to a world point.
    /// The distance limit is inclusive and infinity is unbounded. Equal
    /// distances prefer the lowest source triangle index. Degenerate triangles
    /// contribute their segments or point. A finite affine transform may
    /// include nonuniform scale, shear, reflection, or axis collapse.
    public func closestPoint(
        to point: SIMD3<Float>, localToWorld: simd_float4x4 = matrix_identity_float4x4,
        maximumDistance: Double = .infinity
    ) throws -> MeshClosestPoint? {
        guard boundsFinite(point) else { throw Failure.nonfiniteProximityPoint }
        guard maximumDistance >= 0 else { throw Failure.invalidProximityDistance }
        let frame = try BVHProximityFrame(localToWorld)
        let query = boundsDouble(point)
        func lowerBound(_ index: Int) -> Double {
            let node = nodes[index]
            return frame.bounds(lower: node.lower, upper: node.upper).squaredDistance(to: query)
        }

        let limit = maximumDistance.nextUp * maximumDistance.nextUp
        var best: (point: ProximityTriangle.Point, distanceSquared: Double, triangle: Int)?
        var stack = InlineArray<64, (node: Int, lowerBound: Double)>(repeating: (0, 0))
        stack[0] = (0, lowerBound(0))
        var pending = 1
        while pending > 0 {
            pending -= 1
            let (index, lower) = stack[pending]
            guard lower <= (best?.distanceSquared ?? limit) else { continue }
            let node = nodes[index]
            if node.count > 0 {
                for slot in Int(node.start)..<(Int(node.start) + Int(node.count)) {
                    let triangle = Int(order[slot])
                    let surface = frame.triangle(geometry.triangleCorners(triangle))
                    let candidate = surface.closestPoint(to: query, normal: surface.normal)
                    let squared = simd_length_squared(candidate.position - query)
                    guard squared <= limit else { continue }
                    if let best, squared > best.distanceSquared
                        || (squared == best.distanceSquared && triangle >= best.triangle) { continue }
                    best = (candidate, squared, triangle)
                }
            } else {
                let left = index + 1, right = Int(node.start)
                let leftBound = lowerBound(left), rightBound = lowerBound(right)
                let leftNearer = leftBound <= rightBound
                stack[pending] = leftNearer ? (right, rightBound) : (left, leftBound)
                stack[pending + 1] = leftNearer ? (left, leftBound) : (right, rightBound)
                pending += 2
            }
        }
        guard let best else { return nil }
        let distance = best.distanceSquared.squareRoot()
        guard distance <= maximumDistance else { return nil }
        return try MeshClosestPoint(best.point, distance: distance, triangleIndex: best.triangle)
    }

    /// Finds the closest surface points on two static BVHs in a shared world
    /// coordinate system. Distance and both witness positions use that frame.
    /// Equal distances prefer the lowest pair of source triangle indices, in
    /// receiver then argument order. Intersecting surfaces have zero distance.
    public func closestPoints(
        to other: MeshBVH,
        localToWorld: simd_float4x4 = matrix_identity_float4x4,
        otherLocalToWorld: simd_float4x4 = matrix_identity_float4x4,
        maximumDistance: Double = .infinity
    ) throws -> MeshClosestPoints? {
        guard maximumDistance >= 0 else { throw Failure.invalidProximityDistance }
        let firstFrame = try BVHProximityFrame(localToWorld)
        let secondFrame = try BVHProximityFrame(otherLocalToWorld)
        let limit = maximumDistance.nextUp * maximumDistance.nextUp
        var best: (pair: ProximityTriangle.Pair, first: Int, second: Int)?
        _ = visitProximityPairs(
            with: other, firstFrame: firstFrame, secondFrame: secondFrame,
            distanceLimit: { best?.pair.squaredDistance ?? limit }
        ) { firstIndex, first, secondIndex, second in
            let candidate = first.closestPoints(to: second)
            let squared = candidate.squaredDistance
            guard squared <= limit else { return false }
            if let best, squared > best.pair.squaredDistance
                || (squared == best.pair.squaredDistance && (firstIndex, secondIndex) >= (best.first, best.second)) {
                return false
            }
            best = (candidate, firstIndex, secondIndex)
            return false
        }
        guard let best else { return nil }
        let distance = best.pair.squaredDistance.squareRoot()
        guard distance <= maximumDistance else { return nil }
        return try MeshClosestPoints(
            first: MeshClosestPoint(best.pair.first, distance: distance, triangleIndex: best.first),
            second: MeshClosestPoint(best.pair.second, distance: distance, triangleIndex: best.second)
        )
    }

    /// Visits pairs whose transformed triangle AABBs overlap, including
    /// tangencies, and passes their original indices. A callback may perform a
    /// more specific intersection test and return true to stop. Returns true
    /// only if a callback stops traversal. Pair order follows the hierarchy.
    public func forEachOverlappingTrianglePair(
        with other: MeshBVH,
        localToWorld: simd_float4x4 = matrix_identity_float4x4,
        otherLocalToWorld: simd_float4x4 = matrix_identity_float4x4,
        _ visit: (Int, Int) throws -> Bool
    ) throws -> Bool {
        let firstFrame = try BVHProximityFrame(localToWorld)
        let secondFrame = try BVHProximityFrame(otherLocalToWorld)
        return try visitProximityPairs(
            with: other, firstFrame: firstFrame, secondFrame: secondFrame, distanceLimit: { 0 }
        ) { firstIndex, first, secondIndex, second in
            guard first.bounds.squaredDistance(to: second.bounds) == 0 else { return false }
            return try visit(firstIndex, secondIndex)
        }
    }

    private func visitProximityPairs(
        with other: MeshBVH, firstFrame: BVHProximityFrame, secondFrame: BVHProximityFrame,
        distanceLimit: () -> Double,
        visit: (Int, ProximityTriangle, Int, ProximityTriangle) throws -> Bool
    ) rethrows -> Bool {
        typealias Branch = (index: Int, bounds: CollisionBounds)
        typealias BranchPair = (first: Branch, second: Branch, lowerBound: Double)
        func branch(_ index: Int, in bvh: MeshBVH, frame: BVHProximityFrame) -> Branch {
            let node = bvh.nodes[index]
            return (index, frame.bounds(lower: node.lower, upper: node.upper))
        }
        func pair(_ first: Branch, _ second: Branch) -> BranchPair {
            (first, second, first.bounds.squaredDistance(to: second.bounds))
        }

        var stack = [pair(branch(0, in: self, frame: firstFrame), branch(0, in: other, frame: secondFrame))]
        stack.reserveCapacity(2 * Self.maximumDepth - 1)
        while let (first, second, lower) = stack.popLast() {
            guard lower <= distanceLimit() else { continue }
            let left = nodes[first.index], right = other.nodes[second.index]
            if left.count > 0, right.count > 0 {
                for slot in Int(left.start)..<(Int(left.start) + Int(left.count)) {
                    let firstIndex = Int(order[slot])
                    let firstTriangle = firstFrame.triangle(geometry.triangleCorners(firstIndex))
                    for otherSlot in Int(right.start)..<(Int(right.start) + Int(right.count)) {
                        let secondIndex = Int(other.order[otherSlot])
                        let secondTriangle = secondFrame.triangle(other.geometry.triangleCorners(secondIndex))
                        if try visit(firstIndex, firstTriangle, secondIndex, secondTriangle) { return true }
                    }
                }
                continue
            }

            // Descend only one tree per step to keep the stack bounded by the
            // sum of the two tree depths. Split the larger box when both can split.
            let children: (BranchPair, BranchPair)
            if right.count > 0 || (left.count == 0 && first.bounds.squaredDiagonal >= second.bounds.squaredDiagonal) {
                children = (
                    pair(branch(first.index + 1, in: self, frame: firstFrame), second),
                    pair(branch(Int(left.start), in: self, frame: firstFrame), second)
                )
            } else {
                children = (
                    pair(first, branch(second.index + 1, in: other, frame: secondFrame)),
                    pair(first, branch(Int(right.start), in: other, frame: secondFrame))
                )
            }
            if children.0.lowerBound <= children.1.lowerBound {
                stack.append(children.1)
                stack.append(children.0)
            } else {
                stack.append(children.0)
                stack.append(children.1)
            }
        }
        return false
    }

    /// Traverses the tree against a caller-defined shape in mesh coordinates.
    /// The classifier visits the root and may skip or contain whole subtrees.
    /// A contained subtree supplies one range without classifying descendants.
    ///
    /// A range callback runs before any triangle callback. Both receive the
    /// same node and containment flag for that range. Returning true from
    /// either callback ends traversal and returns true. Errors propagate.
    ///
    /// Optional scores order siblings from low to high. A NaN score is
    /// treated as positive infinity. Equal scores or an absent scorer retain
    /// tree order. The scorer can run before a node's classifier.
    public func shapecast(
        orderBounds: ((ShapeCastNode) throws -> Double)? = nil,
        classifyBounds: (ShapeCastNode, Double?) throws -> ShapeCastClassification,
        visitRange: ((ShapeCastNode, ShapeCastRange, Bool) throws -> Bool)? = nil,
        visitTriangle: ((Triangle3D, Int, ShapeCastNode, Bool) throws -> Bool)? = nil
    ) throws -> Bool {
        typealias Entry = (node: ShapeCastNode, score: Double?)
        func entry(_ index: Int, depth: Int) throws -> Entry {
            let node = nodes[index]
            let info = ShapeCastNode(
                bounds: AxisAlignedBox(validatedMinimum: node.lower, maximum: node.upper),
                isLeaf: node.count > 0, depth: depth, index: index
            )
            let score = try orderBounds?(info)
            return (info, score?.isNaN == true ? .infinity : score)
        }

        var stack = [try entry(0, depth: 1)]
        stack.reserveCapacity(depth)
        while let (node, score) = stack.popLast() {
            let contained: Bool
            switch try classifyBounds(node, score) {
            case .outside: continue
            case .intersecting: contained = false
            case .contained: contained = true
            }
            if !contained, !node.isLeaf {
                let left = try entry(node.index + 1, depth: node.depth + 1)
                let right = try entry(Int(nodes[node.index].start), depth: node.depth + 1)
                let rightFirst = (right.score ?? 0) < (left.score ?? 0)
                stack.append(rightFirst ? left : right)
                stack.append(rightFirst ? right : left)
                continue
            }
            guard visitRange != nil || visitTriangle != nil else { continue }
            let triangles = ShapeCastRange(order[triangleSlots(below: node.index)])
            if try visitRange?(node, triangles, contained) == true { return true }
            if let visitTriangle {
                for index in triangles {
                    let (a, b, c) = geometry.triangleCorners(index)
                    if try visitTriangle(Triangle3D(a, b, c), index, node, contained) { return true }
                }
            }
        }
        return false
    }

    private func triangleSlots(below index: Int) -> Range<Int> {
        var left = index, right = index
        while nodes[left].count == 0 { left += 1 }
        while nodes[right].count == 0 { right = Int(nodes[right].start) }
        return Int(nodes[left].start)..<(Int(nodes[right].start) + Int(nodes[right].count))
    }

    /// The indices of triangles that touch a closed box in the geometry's local
    /// coordinates, in ascending order. The separating-axis test runs in Double,
    /// and degenerate triangles are tested as their segment or point.
    public func triangles(intersecting box: AxisAlignedBox) -> [Int] {
        let center = (boundsDouble(box.minimum) + boundsDouble(box.maximum)) * 0.5
        let halfSize = (boundsDouble(box.maximum) - boundsDouble(box.minimum)) * 0.5
        var found: [Int] = []
        var stack = [0]
        stack.reserveCapacity(Self.maximumDepth)
        while let index = stack.popLast() {
            let node = nodes[index]
            guard all(node.lower .<= box.maximum), all(box.minimum .<= node.upper) else { continue }
            if node.count > 0 {
                for slot in Int(node.start)..<Int(node.start + node.count) {
                    let triangle = Int(order[slot])
                    let corners = geometry.triangleCorners(triangle)
                    if Self.triangle(corners, overlapsBoxAt: center, halfSize: halfSize) { found.append(triangle) }
                }
            } else {
                stack.append(Int(node.start))
                stack.append(index + 1)
            }
        }
        return found.sorted()
    }

    func forEachCollisionCandidate(in bounds: CollisionBounds, visit: (Int) throws -> Void) rethrows {
        var stack = [0]
        stack.reserveCapacity(Self.maximumDepth)
        while let index = stack.popLast() {
            let node = nodes[index]
            guard bounds.overlaps(minimum: node.lower, maximum: node.upper) else { continue }
            if node.count > 0 {
                for slot in Int(node.start)..<Int(node.start + node.count) { try visit(Int(order[slot])) }
            } else {
                stack.append(Int(node.start))
                stack.append(index + 1)
            }
        }
    }

    /// Whether this hierarchy indexes the same positions and triangles. Copies of
    /// one geometry share storage and compare in constant time.
    func isBuilt(from other: MeshGeometry) -> Bool {
        let mine = geometry
        guard mine.indices.count == other.indices.count, mine.vertices.count == other.vertices.count else {
            return false
        }
        let sharedIndices = mine.indices.withUnsafeBufferPointer { lhs in
            other.indices.withUnsafeBufferPointer { lhs.baseAddress == $0.baseAddress }
        }
        let sharedVertices = mine.vertices.withUnsafeBufferPointer { lhs in
            other.vertices.withUnsafeBufferPointer { lhs.baseAddress == $0.baseAddress }
        }
        if sharedIndices, sharedVertices { return true }
        return mine.indices == other.indices
            && zip(mine.vertices, other.vertices).allSatisfy { $0.position == $1.position }
    }

    private static func triangle(
        _ corners: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>),
        overlapsBoxAt center: SIMD3<Double>, halfSize: SIMD3<Double>
    ) -> Bool {
        let points = [boundsDouble(corners.0) - center, boundsDouble(corners.1) - center, boundsDouble(corners.2) - center]
        func separated(along axis: SIMD3<Double>) -> Bool {
            let projections = points.map { simd_dot($0, axis) }
            let reach = simd_dot(halfSize, simd_abs(axis))
            return projections.min()! > reach || projections.max()! < -reach
        }

        // Box face normals.
        for axis in 0..<3 {
            let values = points.map { $0[axis] }
            if values.min()! > halfSize[axis] || values.max()! < -halfSize[axis] { return false }
        }
        // Triangle edges crossed with the box axes.
        let edges = [points[1] - points[0], points[2] - points[1], points[0] - points[2]]
        let units = [SIMD3<Double>(1, 0, 0), SIMD3<Double>(0, 1, 0), SIMD3<Double>(0, 0, 1)]
        for edge in edges {
            for unit in units where separated(along: simd_cross(unit, edge)) { return false }
        }
        // The triangle's plane.
        return !separated(along: simd_cross(edges[0], edges[1]))
    }

    public enum Failure: LocalizedError, Equatable {
        case triangleLimit
        case geometryMismatch
        case refitTopologyMismatch
        case refitVertexIndex(Int)
        case nonfiniteRefitPosition(Int)
        case unrepresentableContact
        case nonfiniteProximityPoint
        case invalidProximityTransform
        case invalidProximityDistance
        case unrepresentableProximityPoint

        public var errorDescription: String? {
            switch self {
            case .triangleLimit: "A mesh BVH addresses at most 4,294,967,295 triangles."
            case .geometryMismatch: "The BVH was built from different mesh positions or triangles."
            case .refitTopologyMismatch: "BVH refit needs the same vertex count and source triangle indices."
            case .refitVertexIndex(let index): "BVH refit vertex index \(index) is outside the geometry."
            case .nonfiniteRefitPosition(let index): "BVH refit vertex \(index) needs a finite position."
            case .unrepresentableContact: "A mesh contact is outside the representable Float range."
            case .nonfiniteProximityPoint: "A mesh proximity query needs a finite point."
            case .invalidProximityTransform: "Mesh proximity needs a finite affine transform."
            case .invalidProximityDistance: "A mesh proximity distance limit must be nonnegative."
            case .unrepresentableProximityPoint: "A mesh proximity point is outside the representable Float range."
            }
        }
    }
}

extension MeshGeometry {
    /// Finds the nearest indexed triangle through a hierarchy built from this
    /// geometry, with results identical to the linear scan. A hierarchy built from
    /// this value or a copy is accepted in constant time; one built from separately
    /// created geometry is compared position by position first. Throws
    /// ``MeshBVH/Failure/geometryMismatch`` when the positions or triangles differ.
    public func intersection(
        ray: SceneRay, localToWorld: simd_float4x4, cullsBackFaces: Bool, bvh: MeshBVH
    ) throws -> MeshRayHit? {
        try intersection(ray: ray, localToWorld: localToWorld, side: cullsBackFaces ? .front : .double, bvh: bvh)
    }

    /// Finds the nearest triangle on the supplied material side. The hit uses
    /// this geometry's attributes, allowing a BVH to be reused after UV or normal edits.
    public func intersection(
        ray: SceneRay, localToWorld: simd_float4x4, side: MeshRenderState.Side, bvh: MeshBVH
    ) throws -> MeshRayHit? {
        guard bvh.isBuilt(from: self) else { throw MeshBVH.Failure.geometryMismatch }
        return try bvh.intersection(ray: ray, localToWorld: localToWorld, side: side, reporting: self)
    }

    /// Finds all triangle hits using the supplied hierarchy and this geometry's attributes.
    public func intersections(
        ray: SceneRay, localToWorld: simd_float4x4, side: MeshRenderState.Side, bvh: MeshBVH
    ) throws -> [MeshRayHit] {
        guard bvh.isBuilt(from: self) else { throw MeshBVH.Failure.geometryMismatch }
        return try bvh.intersections(ray: ray, localToWorld: localToWorld, side: side, reporting: self)
    }

    fileprivate func triangleBounds(_ triangle: Int) -> (lower: SIMD3<Float>, upper: SIMD3<Float>) {
        let (a, b, c) = triangleCorners(triangle)
        return (simd_min(a, simd_min(b, c)), simd_max(a, simd_max(b, c)))
    }

    func triangleCorners(_ triangle: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
        let offset = triangle * 3
        return (
            vertices[Int(indices[offset])].position,
            vertices[Int(indices[offset + 1])].position,
            vertices[Int(indices[offset + 2])].position
        )
    }
}

/// A ray's slab test against padded node bounds, in the geometry's local space.
private struct Slab {
    let origin: SIMD3<Double>
    /// Infinite on axes whose direction is too small to invert, where the ray stays in one slab.
    let inverseDirection: SIMD3<Double>
    let padding: Double

    init(query: PickingQuery, root: MeshBVH.Node) {
        origin = query.localOrigin
        inverseDirection = 1 / query.localDirection
        // Rounding in the triangle test grows with the coordinates it subtracts.
        let scale = max(
            simd_reduce_max(simd_abs(origin)),
            Double(simd_reduce_max(simd_max(simd_abs(root.lower), simd_abs(root.upper))))
        )
        padding = scale * MeshBVH.boundsTolerance + .leastNormalMagnitude
    }

    /// The parameter where the ray enters the padded box, clamped to zero, or nil
    /// when it misses the box within `0...limit`.
    @inline(__always)
    func entry(_ node: MeshBVH.Node, limit: Double) -> Double? {
        var near = 0.0
        var far = limit
        for axis in 0..<3 {
            let lower = Double(node.lower[axis]) - padding
            let upper = Double(node.upper[axis]) + padding
            guard inverseDirection[axis].isFinite else {
                guard origin[axis] >= lower, origin[axis] <= upper else { return nil }
                continue
            }
            var first = (lower - origin[axis]) * inverseDirection[axis]
            var second = (upper - origin[axis]) * inverseDirection[axis]
            if first > second { swap(&first, &second) }
            near = max(near, first)
            far = min(far, second)
            guard near <= far else { return nil }
        }
        return near
    }
}

/// Recursive binned-SAH construction over precomputed triangle bounds.
private struct Builder {
    private static let checkpointInterval = 1_024

    var nodes: [MeshBVH.Node] = []
    var order: [UInt32]
    var deepest = 0
    let lowers: [SIMD3<Float>]
    let uppers: [SIMD3<Float>]
    let centroids: [SIMD3<Double>]
    private let cancellable: Bool
    private let progress: (@Sendable (MeshBVH.BuildProgress) -> Void)?
    private var indexedTriangles = 0
    private var lastReportedIndexCount = 0

    init(
        geometry: MeshGeometry, triangleCount: Int, cancellable: Bool,
        progress: (@Sendable (MeshBVH.BuildProgress) -> Void)?
    ) throws {
        var lowers: [SIMD3<Float>] = []
        var uppers: [SIMD3<Float>] = []
        var centroids: [SIMD3<Double>] = []
        var order: [UInt32] = []
        lowers.reserveCapacity(triangleCount)
        uppers.reserveCapacity(triangleCount)
        centroids.reserveCapacity(triangleCount)
        order.reserveCapacity(triangleCount)
        for triangle in 0..<triangleCount {
            try Self.checkCancellation(cancellable, iteration: triangle)
            let (lower, upper) = geometry.triangleBounds(triangle)
            lowers.append(lower)
            uppers.append(upper)
            centroids.append((boundsDouble(lower) + boundsDouble(upper)) * 0.5)
            order.append(UInt32(triangle))

            let completed = triangle + 1
            if let progress, completed == triangleCount || completed.isMultiple(of: Self.checkpointInterval) {
                progress(.init(completedBoundsCount: completed, indexedTriangleCount: 0, triangleCount: triangleCount))
            }
        }
        self.lowers = lowers
        self.uppers = uppers
        self.centroids = centroids
        self.order = order
        self.cancellable = cancellable
        self.progress = progress
        nodes.reserveCapacity(max(1, triangleCount))
    }

    @inline(__always)
    private static func checkCancellation(_ cancellable: Bool, iteration: Int = 0) throws {
        if cancellable, iteration.isMultiple(of: checkpointInterval) {
            try Task.checkCancellation()
        }
    }

    private mutating func finishLeaf(_ index: Int, triangleCount: Int) -> Int {
        if let progress {
            indexedTriangles += triangleCount
            let total = order.count
            if indexedTriangles < total, indexedTriangles - lastReportedIndexCount >= Self.checkpointInterval {
                lastReportedIndexCount = indexedTriangles
                progress(.init(completedBoundsCount: total, indexedTriangleCount: indexedTriangles, triangleCount: total))
            }
        }
        return index
    }

    private struct Bin {
        var lower = SIMD3<Float>(repeating: .infinity)
        var upper = SIMD3<Float>(repeating: -.infinity)
        var count = 0

        mutating func include(lower: SIMD3<Float>, upper: SIMD3<Float>, count: Int = 1) {
            self.lower = simd_min(self.lower, lower)
            self.upper = simd_max(self.upper, upper)
            self.count += count
        }

        mutating func include(_ other: Bin) {
            include(lower: other.lower, upper: other.upper, count: other.count)
        }

        /// Half the box's surface area, zero for an empty bin.
        var halfArea: Double {
            guard count > 0 else { return 0 }
            let size = boundsDouble(upper) - boundsDouble(lower)
            return size.x * size.y + size.y * size.z + size.z * size.x
        }
    }

    private struct Split {
        let axis: Int
        let bin: Int
        let cost: Double
        let imbalance: Int
        let minimum: Double
        let scale: Double
    }

    mutating func build(_ range: Range<Int>, depth: Int) throws -> Int {
        try Self.checkCancellation(cancellable)
        deepest = max(deepest, depth)
        var enclosing = Bin()
        var centroidLower = SIMD3<Double>(repeating: .infinity)
        var centroidUpper = SIMD3<Double>(repeating: -.infinity)
        for slot in range {
            try Self.checkCancellation(cancellable, iteration: slot)
            let triangle = Int(order[slot])
            enclosing.include(lower: lowers[triangle], upper: uppers[triangle])
            centroidLower = simd_min(centroidLower, centroids[triangle])
            centroidUpper = simd_max(centroidUpper, centroids[triangle])
        }

        let index = nodes.count
        nodes.append(MeshBVH.Node(
            lower: enclosing.lower, upper: enclosing.upper, start: UInt32(range.lowerBound), count: UInt32(range.count)
        ))
        guard range.count > 1, depth < MeshBVH.maximumDepth,
              let split = try bestSplit(range, centroidLower: centroidLower, centroidUpper: centroidUpper) else {
            return finishLeaf(index, triangleCount: range.count)
        }
        // Splitting costs one node visit plus the area-weighted child triangles.
        let parentArea = enclosing.halfArea
        let splitIsCheaper = split.cost < parentArea * Double(range.count - 1)
        guard splitIsCheaper || range.count > MeshBVH.preferredLeafTriangleCount else {
            return finishLeaf(index, triangleCount: range.count)
        }

        let middle = try partition(range, by: split)
        _ = try build(range.lowerBound..<middle, depth: depth + 1)
        let right = try build(middle..<range.upperBound, depth: depth + 1)
        nodes[index].start = UInt32(right)
        nodes[index].count = 0
        return index
    }

    private func bin(of triangle: Int, axis: Int, minimum: Double, scale: Double) -> Int {
        let position = (centroids[triangle][axis] - minimum) * scale
        return min(max(Int(position), 0), MeshBVH.binCount - 1)
    }

    /// The cheapest binned split; ties prefer balanced children, then lower axes and planes.
    private func bestSplit(
        _ range: Range<Int>, centroidLower: SIMD3<Double>, centroidUpper: SIMD3<Double>
    ) throws -> Split? {
        var best: Split?
        for axis in 0..<3 {
            try Self.checkCancellation(cancellable)
            let extent = centroidUpper[axis] - centroidLower[axis]
            guard extent > 0 else { continue }
            let scale = Double(MeshBVH.binCount) / extent
            var bins = [Bin](repeating: Bin(), count: MeshBVH.binCount)
            for slot in range {
                try Self.checkCancellation(cancellable, iteration: slot)
                let triangle = Int(order[slot])
                let index = bin(of: triangle, axis: axis, minimum: centroidLower[axis], scale: scale)
                bins[index].include(lower: lowers[triangle], upper: uppers[triangle])
            }

            var rightCosts = [Double](repeating: 0, count: MeshBVH.binCount)
            var rightCounts = [Int](repeating: 0, count: MeshBVH.binCount)
            var accumulated = Bin()
            for plane in stride(from: MeshBVH.binCount - 1, to: 0, by: -1) {
                accumulated.include(bins[plane])
                rightCosts[plane] = accumulated.halfArea * Double(accumulated.count)
                rightCounts[plane] = accumulated.count
            }
            accumulated = Bin()
            for plane in 1..<MeshBVH.binCount {
                accumulated.include(bins[plane - 1])
                guard accumulated.count > 0, rightCounts[plane] > 0 else { continue }
                let candidate = Split(
                    axis: axis, bin: plane,
                    cost: accumulated.halfArea * Double(accumulated.count) + rightCosts[plane],
                    imbalance: abs(accumulated.count - rightCounts[plane]),
                    minimum: centroidLower[axis], scale: scale
                )
                if let current = best,
                   candidate.cost > current.cost
                    || (candidate.cost == current.cost && candidate.imbalance >= current.imbalance) {
                    continue
                }
                best = candidate
            }
        }
        return best
    }

    private mutating func partition(_ range: Range<Int>, by split: Split) throws -> Int {
        try Self.checkCancellation(cancellable)
        var lower = range.lowerBound
        var upper = range.upperBound - 1
        while lower <= upper {
            try Self.checkCancellation(cancellable, iteration: upper - lower)
            let triangle = Int(order[lower])
            if bin(of: triangle, axis: split.axis, minimum: split.minimum, scale: split.scale) < split.bin {
                lower += 1
            } else {
                order.swapAt(lower, upper)
                upper -= 1
            }
        }
        return lower
    }
}

/// The active hierarchy for one renderer's geometry and shader pose.
final class MeshBVHCache: Sendable {
    private struct Entry: Sendable {
        let revision: ShaderUniformRevision?
        let hierarchy: MeshBVH
    }

    private let state = Mutex<Entry?>(nil)

    func prepared(revision: ShaderUniformRevision? = nil) -> MeshBVH? {
        state.withLock { cached in
            guard let cached, cached.revision == revision else { return nil }
            return cached.hierarchy
        }
    }

    func installIfAbsent(_ hierarchy: MeshBVH, revision: ShaderUniformRevision? = nil) -> MeshBVH {
        state.withLock { cached in
            if let cached, cached.revision == revision { return cached.hierarchy }
            cached = Entry(revision: revision, hierarchy: hierarchy)
            return hierarchy
        }
    }

    func bvh(for geometry: @autoclosure () throws -> MeshGeometry, revision: ShaderUniformRevision? = nil) throws -> MeshBVH {
        try state.withLock { cached in
            if let cached, cached.revision == revision { return cached.hierarchy }
            let surface = try geometry()
            // The renderer replaces this cache when its geometry or draw range
            // changes. A new shader pose therefore keeps the same topology.
            let built = try cached?.hierarchy.refitted(to: surface) ?? MeshBVH(geometry: surface)
            cached = Entry(revision: revision, hierarchy: built)
            return built
        }
    }
}
