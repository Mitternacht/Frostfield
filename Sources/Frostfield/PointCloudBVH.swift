import simd

/// An immutable CPU hierarchy over a point cloud's local positions.
/// Original point indices remain stable. A query can select a draw range
/// without rebuilding the hierarchy.
public struct PointCloudBVH: Sendable {
    public let cloud: PointCloud
    public var nodeCount: Int { nodes.count }
    private let nodes: [Node]
    private let order: [UInt32]

    private struct Node {
        let bounds: AxisAlignedBox
        let sourceFirst: UInt32
        let sourceLast: UInt32
        /// A leaf's first ordered point or a branch's right child.
        var start: UInt32
        /// Zero for a branch, whose left child follows it.
        var count: UInt32
    }

    /// Builds a spatial hierarchy that shares the cloud's immutable storage.
    public init(cloud: PointCloud) {
        var builder = Builder(positions: cloud.positions)
        _ = builder.build(0..<cloud.positions.count)
        self.cloud = cloud
        self.nodes = builder.nodes
        self.order = builder.order
    }

    /// Returns point hits with the same inclusive world threshold, ray
    /// distance limit and size visibility as the linear point-cloud query.
    /// Hits are sorted by distance, then by source point index.
    public func intersections(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, threshold: Float = 1,
        drawRange: GeometryDrawRange = .all
    ) throws -> [PrimitiveRayHit] {
        let query = try PrimitiveWorldRay(ray, threshold: threshold, localToWorld: localToWorld)
        return try intersections(query, range: drawRange.resolve(elementCount: cloud.positions.count))
    }

    func intersections(_ query: PrimitiveWorldRay, range: Range<Int>) throws -> [PrimitiveRayHit] {
        var hits: [(index: Int, hit: PrimitiveRayHit)] = []
        try forEachRayCandidate(query, range: range) { index in
            if let hit = try query.pointHit(in: cloud, index: index) { hits.append((index, hit)) }
        }
        return hits.sorted {
            $0.hit.distance == $1.hit.distance ? $0.index < $1.index : $0.hit.distance < $1.hit.distance
        }.map(\.hit)
    }

    func forEachRayCandidate(
        _ query: PrimitiveWorldRay, range: Range<Int>, visit: (Int) throws -> Void
    ) rethrows {
        guard !range.isEmpty else { return }
        var stack = [0]
        while let index = stack.popLast() {
            let node = nodes[index]
            guard Int(node.sourceFirst) < range.upperBound, Int(node.sourceLast) >= range.lowerBound,
                  query.overlapsPointBounds(node.bounds) else { continue }
            if node.count > 0 {
                let start = Int(node.start)
                for slot in start..<(start + Int(node.count)) {
                    let source = Int(order[slot])
                    if range.contains(source) { try visit(source) }
                }
            } else {
                stack.append(Int(node.start))
                stack.append(index + 1)
            }
        }
    }

    private struct Builder {
        private static let targetLeafSize = 16
        let positions: [SIMD3<Float>]
        var order: [UInt32]
        var nodes: [Node] = []

        init(positions: [SIMD3<Float>]) {
            self.positions = positions
            self.order = positions.indices.map(UInt32.init)
        }

        mutating func build(_ range: Range<Int>) -> Int {
            var lower = SIMD3<Float>(repeating: .infinity)
            var upper = SIMD3<Float>(repeating: -.infinity)
            var sourceFirst = UInt32.max
            var sourceLast: UInt32 = 0
            for slot in range {
                let index = order[slot]
                let point = positions[Int(index)]
                lower = simd_min(lower, point)
                upper = simd_max(upper, point)
                sourceFirst = min(sourceFirst, index)
                sourceLast = max(sourceLast, index)
            }

            let index = nodes.count
            nodes.append(Node(
                bounds: AxisAlignedBox(validatedMinimum: lower, maximum: upper),
                sourceFirst: sourceFirst, sourceLast: sourceLast,
                start: UInt32(range.lowerBound), count: UInt32(range.count)
            ))
            let size = boundsDouble(upper) - boundsDouble(lower)
            guard range.count > Self.targetLeafSize, size != .zero else { return index }
            let axis = size.x >= max(size.y, size.z) ? 0 : (size.y >= size.z ? 1 : 2)
            let midpoint = Double(lower[axis]) + size[axis] * 0.5
            var split = range.lowerBound
            for slot in range where Double(positions[Int(order[slot])][axis]) < midpoint {
                order.swapAt(split, slot)
                split += 1
            }
            let minimumChildCount = range.count / 4
            if split - range.lowerBound < minimumChildCount || range.upperBound - split < minimumChildCount {
                // A median fallback bounds recursion for highly uneven clouds.
                let positions = self.positions
                order[range].sort {
                    let left = positions[Int($0)][axis], right = positions[Int($1)][axis]
                    return left == right ? $0 < $1 : left < right
                }
                split = range.lowerBound + range.count / 2
            }
            _ = build(range.lowerBound..<split)
            let right = build(split..<range.upperBound)
            nodes[index].start = UInt32(right)
            nodes[index].count = 0
            return index
        }
    }
}
