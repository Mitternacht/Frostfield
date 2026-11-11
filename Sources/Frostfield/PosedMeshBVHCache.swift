import Synchronization

/// Picking snapshots for one renderer's immutable skin and morph source.
final class PosedMeshBVHCache: Sendable {
    private struct Pose: Equatable, Sendable {
        let palette: JointPalette?
        let weights: [Float]?
        let range: Range<Int>
    }

    private struct Entry: Sendable {
        let pose: Pose
        let bvh: MeshBVH
    }

    // Keep two recent poses for a renderer shared by multiple scene nodes.
    // Readers keep immutable snapshots even when the cache evicts them.
    private let recent = Mutex<[Entry]>([])

    func bvh(
        palette: JointPalette?, weights: [Float]?, range: Range<Int>,
        geometry: () throws -> MeshGeometry
    ) throws -> MeshBVH {
        let pose = Pose(palette: palette, weights: weights, range: range)
        return try recent.withLock { entries in
            if let index = entries.firstIndex(where: { $0.pose == pose }) {
                let cached = entries[index].bvh
                if index != 0 { entries.swapAt(0, index) }
                return cached
            }

            let posed = try geometry().restrictingIndices(to: range)
            let previous = entries.first { $0.pose.range == range }
            let snapshot = try previous?.bvh.refitted(to: posed) ?? MeshBVH(geometry: posed)
            entries.insert(Entry(pose: pose, bvh: snapshot), at: 0)
            if entries.count > 2 { entries.removeLast() }
            return snapshot
        }
    }
}
