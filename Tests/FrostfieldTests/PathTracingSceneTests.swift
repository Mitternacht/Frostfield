import Frostfield
import simd
import Testing

struct PathTracingSceneTests {
    @Test("Path scene records match the Metal buffer strides")
    func layout() {
        #expect(MemoryLayout<PathTracingScene.Vertex>.stride == 48)
        #expect(MemoryLayout<PathTracingScene.Vertex>.alignment == 16)
        #expect(MemoryLayout<PathTracingScene.BVHNode>.stride == 48)
        #expect(MemoryLayout<PathTracingScene.BVHNode>.alignment == 16)
        // 144 bytes of core fields, the 592-byte KHR extension block, then the 384-byte lobe block.
        #expect(MemoryLayout<PathTracingScene.Material>.stride == 1_120)
        #expect(MemoryLayout<PathTracingScene.Material>.offset(of: \.extensions) == 144)
        #expect(MemoryLayout<PathTracingScene.Material>.offset(of: \.lobes) == 736)
        #expect(MemoryLayout<PathTracingScene.Material>.alignment == 16)
    }

    @Test("The SAH BVH covers every triangle exactly once with contiguous leaf ranges and bounded depth")
    func boundedTree() throws {
        // Shared centroids and mixed input order exercise tie breaking.
        let xs = (0..<1_025).map { Float(($0 * 137) % 19) }
        let tree = try PathTracingScene.buildBVH(vertices: positions(xs), indices: (0..<(3 * 1_025)).map(UInt32.init))
        try checkTree(tree, triangleCount: 1_025, positions: positions(xs))
    }

    @Test("Degenerate clusters keep the BVH within its traversal depth")
    func depthBound() throws {
        // Exponentially spaced triangles make every SAH split peel off one triangle; the depth
        // guard switches to median splits before the 32-level stack would overflow.
        let xs = (0..<4_096).map { Float(pow(1.02, Double($0))) }
        let tree = try PathTracingScene.buildBVH(vertices: positions(xs), indices: (0..<(3 * 4_096)).map(UInt32.init))
        try checkTree(tree, triangleCount: 4_096, positions: positions(xs))
    }

    @Test("The SAH tree is cheaper to traverse than a median split on clustered triangles")
    func surfaceAreaQuality() throws {
        // Two dense clusters far apart: SAH separates them at the root, a median split does not.
        let xs = (0..<2_000).map { $0 < 1_000 ? Float($0) * 0.001 : 1_000 + Float($0) * 0.001 }
        let tree = try PathTracingScene.buildBVH(vertices: positions(xs), indices: (0..<(3 * 2_000)).map(UInt32.init))
        let root = tree.nodes[0]
        let left = tree.nodes[Int(root.meta.x)]
        let right = tree.nodes[Int(root.meta.y)]
        #expect(left.maximum.x < 2 && right.minimum.x > 1_000)
    }

    private func positions(_ xs: [Float]) -> [SIMD3<Float>] {
        xs.flatMap { x in [SIMD3(x, 0, 0), SIMD3(x + 0.25, 0, 0), SIMD3(x, 0.25, 0)] }
    }

    private func checkTree(
        _ tree: (indices: [UInt32], primitives: [UInt32], nodes: [PathTracingScene.BVHNode]), triangleCount: Int,
        positions: [SIMD3<Float>]
    ) throws {
        #expect(tree.indices.count == 3 * triangleCount && tree.primitives.count == triangleCount)
        #expect(Set(tree.primitives) == Set((0..<triangleCount).map(UInt32.init)))
        var visited = Set<Int>()
        var covered = Set<Int>()
        var pending: [(index: Int, depth: Int)] = [(0, 1)]
        var maximumDepth = 0
        while let (index, depth) = pending.popLast() {
            try #require(tree.nodes.indices.contains(index))
            try #require(visited.insert(index).inserted)
            maximumDepth = max(maximumDepth, depth)
            let node = tree.nodes[index]
            if node.meta.w == 0 {
                for childIndex in [Int(node.meta.x), Int(node.meta.y)] {
                    try #require(tree.nodes.indices.contains(childIndex))
                    #expect(childIndex > index)
                    let child = tree.nodes[childIndex]
                    #expect(contains(node, point: child.minimum))
                    #expect(contains(node, point: child.maximum))
                    pending.append((childIndex, depth + 1))
                }
            } else {
                #expect(node.meta.w <= 4)
                let start = Int(node.meta.z)
                for stored in start..<(start + Int(node.meta.w)) {
                    try #require(tree.primitives.indices.contains(stored))
                    #expect(covered.insert(stored).inserted)
                    // The stored triangle is the source triangle its primitive number names.
                    let source = Int(tree.primitives[stored])
                    for corner in 0..<3 {
                        #expect(tree.indices[3 * stored + corner] == UInt32(3 * source + corner))
                        #expect(contains(node, point: SIMD4(positions[3 * source + corner], 1)))
                    }
                }
            }
        }
        #expect(visited.count == tree.nodes.count)
        #expect(covered.count == triangleCount)
        #expect(maximumDepth <= PathTracingScene.maximumBVHDepth)
    }

    @Test("Empty BVH input fails before allocating GPU resources")
    func empty() {
        #expect(throws: PathTracingScene.Failure.emptyScene) { try PathTracingScene.buildBVH(vertices: [], indices: []) }
    }

    private func contains(_ node: PathTracingScene.BVHNode, point: SIMD4<Float>) -> Bool {
        (0..<3).allSatisfy { node.minimum[$0] <= point[$0] && node.maximum[$0] >= point[$0] }
    }
}
