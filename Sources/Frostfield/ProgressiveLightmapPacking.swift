import simd

extension ProgressiveLightmapBaker {
    /// Rebuilds display meshes in their authored local space. The atlas input
    /// and transport BVH both keep world-space triangles in the same order.
    static func packedBindings(
        _ atlas: MeshGeometry.UVAtlas, ranges: [Range<Int>], sources: [Surface]
    ) throws -> [ReceiverBinding] {
        var vertexOffset = 0
        return try zip(ranges, sources).map { range, surface in
            let original = surface.geometry
            let packedIndices = atlas.geometry.indices
            // Merging reverses mirrored faces. The atlas validates that it keeps
            // every corner of that merged mesh, so this detects the reversal.
            let mergedSecond = atlas.sourceVertexIndices[Int(packedIndices[range.lowerBound + 1])] - vertexOffset
            let corners = mergedSecond == Int(original.indices[1]) ? [0, 1, 2] : [0, 2, 1]
            var localIndex: [UInt32: UInt32] = [:]
            var vertices: [MeshVertex] = []
            var indices: [UInt32] = []
            var sourceIndices: [Int] = []
            indices.reserveCapacity(original.indices.count)
            for start in stride(from: range.lowerBound, to: range.upperBound, by: 3) {
                for (localCorner, packedCorner) in corners.enumerated() {
                    let packed = packedIndices[start + packedCorner]
                    let sourceIndex = atlas.sourceVertexIndices[Int(packed)] - vertexOffset
                    guard sourceIndex == Int(original.indices[start - range.lowerBound + localCorner]) else {
                        throw MeshGeometry.UVAtlasError.invalidParameterization
                    }
                    if let index = localIndex[packed] {
                        indices.append(index)
                        continue
                    }
                    let index = UInt32(vertices.count)
                    localIndex[packed] = index
                    indices.append(index)
                    sourceIndices.append(sourceIndex)
                    let source = original.vertices[sourceIndex]
                    vertices.append(MeshVertex(
                        position: source.position, normal: source.normal, textureCoordinate: source.textureCoordinate,
                        secondaryTextureCoordinate: atlas.geometry.vertices[Int(packed)].secondaryTextureCoordinate,
                        tangent: source.tangent, color: source.color
                    ))
                }
            }
            vertexOffset += original.vertices.count
            let geometry = try MeshGeometry(vertices: vertices, indices: indices).withGroups(original.groups)
            return ReceiverBinding(geometry: geometry, worldFromObject: surface.worldFromObject, sourceVertexIndices: sourceIndices)
        }
    }
}
