/// Packed vertex-to-leaf adjacency and parents, shared by refitted snapshots.
struct SelectiveRefitTopology: Sendable {
    private let offsets: [Int]
    private let leaves: [UInt32]
    private let parents: [UInt32]

    init(geometry: MeshGeometry, nodes: [MeshBVH.Node], order: [UInt32]) {
        let indices = geometry.indices
        var offsets = [Int](repeating: 0, count: geometry.vertices.count + 1)
        for index in indices { offsets[Int(index)] += 1 }

        // Begin with each vertex's exclusive end. Filling backward converts
        // those cursors to start offsets without keeping a second array.
        var end = 0
        for vertex in geometry.vertices.indices {
            end += offsets[vertex]
            offsets[vertex] = end
        }
        offsets[geometry.vertices.count] = end

        var leaves = [UInt32](repeating: 0, count: indices.count)
        var parents = [UInt32](repeating: 0, count: nodes.count)
        for (index, node) in nodes.enumerated() {
            if node.count > 0 {
                let start = Int(node.start)
                for slot in start..<(start + Int(node.count)) {
                    let firstCorner = Int(order[slot]) * 3
                    for corner in firstCorner..<(firstCorner + 3) {
                        let vertex = Int(indices[corner])
                        offsets[vertex] -= 1
                        leaves[offsets[vertex]] = UInt32(index)
                    }
                }
            } else {
                parents[index + 1] = UInt32(index)
                parents[Int(node.start)] = UInt32(index)
            }
        }
        self.offsets = offsets
        self.leaves = leaves
        self.parents = parents
    }

    func insertAffectedNodes(forVertex vertex: Int, into dirty: inout Set<Int>) {
        for slot in offsets[vertex]..<offsets[vertex + 1] {
            var index = Int(leaves[slot])
            // A previous corner may already have marked this path to the root.
            while dirty.insert(index).inserted, index != 0 {
                index = Int(parents[index])
            }
        }
    }
}
