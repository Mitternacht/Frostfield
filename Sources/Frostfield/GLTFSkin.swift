import simd

extension GLTFScene {
    /// Source joints and inverse binds in the same order as the vertex joint indices.
    public struct Skin {
        public let sourceIndex: Int
        public let name: String?
        public let jointNodeIndices: [Int]
        public let inverseBindMatrices: [simd_float4x4]
        public let skeletonRootNodeIndex: Int?
    }
}

struct GLTFSkinDocument: Decodable {
    private typealias Failure = GLTFMeshLoader.Failure

    private let skins: [Record]?

    private struct Record: Decodable {
        let name: String?
        let joints: [Int]
        let inverseBindMatrices: Int?
        let skeleton: Int?
    }

    private struct NodeSpan {
        let root: Int
        let subtree: Range<Int>
    }

    func resolve(
        nodes: [GLTFScene.Node], roots: [Int], maximumElements: Int,
        readMatrices: (Int, Int) throws -> [simd_float4x4]
    ) throws -> [GLTFScene.Skin] {
        let records = skins ?? []
        let used = Set(nodes.compactMap(\.skinIndex)).sorted()
        let hierarchy = nodeSpans(nodes, roots: roots)
        var remaining = maximumElements
        var result: [GLTFScene.Skin] = []
        for index in used {
            guard records.indices.contains(index) else { throw Failure.missingSkin(index) }
            let skin = records[index]
            let count = skin.joints.count
            guard (1...MeshSkin.maximumJointCount).contains(count), Set(skin.joints).count == count else {
                throw Failure.invalidSkin(index, "a skin requires one to 256 distinct joints")
            }
            guard count <= remaining else { throw Failure.sceneSizeLimit }
            remaining -= count

            let skeleton: NodeSpan?
            if let node = skin.skeleton {
                guard let span = hierarchy[node] else {
                    throw Failure.invalidSkin(index, "skeleton root node \(node) is outside the selected scene")
                }
                skeleton = span
            } else {
                skeleton = nil
            }

            var commonRoot: Int?
            for joint in skin.joints {
                guard let span = hierarchy[joint] else {
                    throw Failure.invalidSkin(index, "joint node \(joint) is outside the selected scene")
                }
                if let root = commonRoot, span.root != root {
                    throw Failure.invalidSkin(index, "joints must share a root in the selected scene")
                }
                if let skeleton, !skeleton.subtree.contains(span.subtree.lowerBound) {
                    throw Failure.invalidSkin(index, "the skeleton root must contain every joint")
                }
                commonRoot = span.root
            }

            let matrices: [simd_float4x4]
            if let accessor = skin.inverseBindMatrices {
                matrices = try readMatrices(accessor, count)
            } else {
                matrices = Array(repeating: matrix_identity_float4x4, count: count)
            }
            result.append(.init(
                sourceIndex: index, name: skin.name, jointNodeIndices: skin.joints,
                inverseBindMatrices: matrices, skeletonRootNodeIndex: skin.skeleton
            ))
        }
        return result
    }

    private func nodeSpans(_ nodes: [GLTFScene.Node], roots: [Int]) -> [Int: NodeSpan] {
        let byIndex = Dictionary(uniqueKeysWithValues: nodes.map { ($0.sourceIndex, $0) })
        var spans: [Int: NodeSpan] = [:]
        var pending = roots.reversed().map { (index: $0, root: $0, entry: Int?.none) }
        var counter = 0

        // One depth-first traversal gives every node a constant-time ancestry check.
        while let (index, root, entry) = pending.popLast() {
            if let entry {
                spans[index] = NodeSpan(root: root, subtree: entry..<counter)
                continue
            }
            guard let node = byIndex[index] else { continue }
            pending.append((index, root, counter))
            counter += 1
            for child in node.childIndices.reversed() {
                pending.append((child, root, nil))
            }
        }
        return spans
    }
}
