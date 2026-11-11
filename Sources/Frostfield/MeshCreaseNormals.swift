import Foundation
import simd

extension MeshGeometry {
    /// Recomputes normals with hard edges where adjacent faces meet at more than
    /// `creaseAngle` radians, like Three.js `BufferGeometryUtils.toCreasedNormals`.
    ///
    /// Connectivity comes from exact vertex positions, so shading stays smooth across UV
    /// seams. Around each position, triangles joined by smooth edges form one group, and
    /// the group's normal is the corner-angle-weighted mean of its face normals. Open
    /// edges, edges shared by more than two triangles, and edges whose triangles wind in
    /// opposite directions are always hard. A vertex used by several groups is split into
    /// one vertex per group. Original vertices keep their index and split copies
    /// are appended. Tangents on referenced vertices are dropped because they depend
    /// on the old normals. Use `generatingTangents` afterwards if needed.
    public func creasedNormals(creaseAngle: Float = .pi / 3) throws -> MeshGeometry {
        try creasedNormalsWithRemap(creaseAngle: creaseAngle).geometry
    }

    /// Recomputes normals and returns the source index for each original or split vertex.
    public func creasedNormalsWithRemap(creaseAngle: Float = .pi / 3) throws -> VertexRemap {
        try smoothingGroups(creaseAngle: creaseAngle).remapped(keepingNormalsOfUnsplitVertices: false)
    }

    /// Splits vertices along edges sharper than `creaseAngle` radians, like Three.js
    /// `EdgeSplitModifier`. With `keepingNormals`, vertices that end up in a single
    /// group keep their existing normals and tangents, and only split vertices get new
    /// normals (and lose tangents); otherwise this matches `creasedNormals(creaseAngle:)`.
    public func splittingEdges(creaseAngle: Float, keepingNormals: Bool = true) throws -> MeshGeometry {
        try splittingEdgesWithRemap(creaseAngle: creaseAngle, keepingNormals: keepingNormals).geometry
    }

    /// Splits edges and returns the source index for each original or split vertex.
    public func splittingEdgesWithRemap(creaseAngle: Float, keepingNormals: Bool = true) throws -> VertexRemap {
        try smoothingGroups(creaseAngle: creaseAngle).remapped(keepingNormalsOfUnsplitVertices: keepingNormals)
    }

    private func smoothingGroups(creaseAngle: Float) throws -> SmoothingGroups {
        guard creaseAngle.isFinite, (0...Float.pi).contains(creaseAngle) else {
            throw GeometryProcessingError.invalidCreaseAngle
        }
        let topology = EdgeTopology(self)
        let limit = max(Double(creaseAngle), 8 * Double.ulpOfOne)
        let faceNormals = topology.faceNormals
        return SmoothingGroups(geometry: self, topology: topology) { a, b in
            let na = faceNormals[a / 3], nb = faceNormals[b / 3]
            // A degenerate face cannot join two otherwise unrelated smoothing groups.
            if na == .zero || nb == .zero { return true }
            return atan2(simd_length(simd_cross(na, nb)), simd_dot(na, nb)) > limit
        }
    }
}

/// Exact-position connectivity of an indexed triangle mesh.
struct EdgeTopology {
    static let boundary: Int = -1
    static let nonManifold: Int = -2

    /// Per vertex, the first vertex with the same position (−0 equal to +0).
    let positionID: [UInt32]
    /// Per corner `c`, for the edge from corner `c` to the next corner of its triangle: the
    /// corner that starts the same edge in the adjacent triangle, `boundary`, or
    /// `nonManifold` for edges shared by three or more triangles.
    let neighbor: [Int]
    /// Per triangle, the unit face normal in double precision, or zero when degenerate.
    let faceNormals: [SIMD3<Double>]
    let indices: [UInt32]

    init(_ geometry: MeshGeometry) {
        indices = geometry.indices
        var first: [SIMD3<UInt32>: UInt32] = [:]
        first.reserveCapacity(geometry.vertices.count)
        var ids = [UInt32](repeating: 0, count: geometry.vertices.count)
        for (index, vertex) in geometry.vertices.enumerated() {
            let key = vertex.position.exactKey
            if let existing = first[key] {
                ids[index] = existing
            } else {
                first[key] = UInt32(index)
                ids[index] = UInt32(index)
            }
        }
        positionID = ids

        let cornerCount = geometry.indices.count
        var edges: [UInt64: (first: Int, second: Int, count: Int)] = [:]
        edges.reserveCapacity(cornerCount)
        for corner in 0..<cornerCount {
            let a = ids[Int(geometry.indices[corner])]
            let b = ids[Int(geometry.indices[Self.next(corner)])]
            let key = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
            if var entry = edges[key] {
                if entry.count == 1 { entry.second = corner }
                entry.count += 1
                edges[key] = entry
            } else {
                edges[key] = (corner, -1, 1)
            }
        }
        var neighbor = [Int](repeating: Self.boundary, count: cornerCount)
        for corner in 0..<cornerCount {
            let a = ids[Int(geometry.indices[corner])]
            let b = ids[Int(geometry.indices[Self.next(corner)])]
            guard a != b, let entry = edges[UInt64(min(a, b)) << 32 | UInt64(max(a, b))] else { continue }
            switch entry.count {
            case 1: neighbor[corner] = Self.boundary
            case 2: neighbor[corner] = entry.first == corner ? entry.second : entry.first
            default: neighbor[corner] = Self.nonManifold
            }
        }
        self.neighbor = neighbor

        faceNormals = stride(from: 0, to: cornerCount, by: 3).map { start in
            let p0 = SIMD3<Double>(geometry.vertices[Int(geometry.indices[start])].position)
            let p1 = SIMD3<Double>(geometry.vertices[Int(geometry.indices[start + 1])].position)
            let p2 = SIMD3<Double>(geometry.vertices[Int(geometry.indices[start + 2])].position)
            let normal = simd_cross(p1 - p0, p2 - p0)
            let length = simd_length(normal)
            return length > 0 && length.isFinite ? normal / length : .zero
        }
    }

    static func next(_ corner: Int) -> Int { corner % 3 == 2 ? corner - 2 : corner + 1 }
    static func previous(_ corner: Int) -> Int { corner % 3 == 0 ? corner + 2 : corner - 1 }

    func position(_ corner: Int) -> UInt32 { positionID[Int(indices[corner])] }

    /// Whether corner `c`'s edge has exactly one neighbor that traverses it in the opposite direction.
    func isConsistentManifold(_ corner: Int) -> Bool {
        let other = neighbor[corner]
        guard other >= 0 else { return false }
        return position(other) == position(Self.next(corner))
    }
}

/// Per-corner smoothing groups: corners at one position joined across smooth edges.
struct SmoothingGroups {
    let geometry: MeshGeometry
    let topology: EdgeTopology
    /// Per corner, the root corner of its group.
    let group: [Int]

    /// `isSharp` receives the two corners that start a shared edge in its two triangles.
    init(geometry: MeshGeometry, topology: EdgeTopology, isSharp: (Int, Int) -> Bool) {
        self.geometry = geometry
        self.topology = topology
        var parent = Array(geometry.indices.indices)
        func find(_ x: Int) -> Int {
            var root = x
            while parent[root] != root { root = parent[root] }
            var node = x
            while parent[node] != root {
                let next = parent[node]
                parent[node] = root
                node = next
            }
            return root
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
        }
        for corner in 0..<geometry.indices.count where topology.isConsistentManifold(corner) {
            let other = topology.neighbor[corner]
            guard corner < other, !isSharp(corner, other) else { continue }
            // Edge A→B in this triangle runs B→A in the other.
            union(corner, EdgeTopology.next(other))
            union(EdgeTopology.next(corner), other)
        }
        group = geometry.indices.indices.map(find)
    }

    /// Group normals indexed by root corner: corner-angle-weighted sums of unit face normals.
    func groupNormals() -> [SIMD3<Double>] {
        var sums = [SIMD3<Double>](repeating: .zero, count: geometry.indices.count)
        let vertices = geometry.vertices
        let indices = geometry.indices
        for corner in 0..<indices.count {
            let face = topology.faceNormals[corner / 3]
            guard face != .zero else { continue }
            let p = SIMD3<Double>(vertices[Int(indices[corner])].position)
            let a = SIMD3<Double>(vertices[Int(indices[EdgeTopology.next(corner)])].position) - p
            let b = SIMD3<Double>(vertices[Int(indices[EdgeTopology.previous(corner)])].position) - p
            let angle = atan2(simd_length(simd_cross(a, b)), simd_dot(a, b))
            sums[group[corner]] += face * angle
        }
        return sums
    }

    /// Keep original indices, and append one copy per additional smoothing group.
    func remapped(keepingNormalsOfUnsplitVertices keep: Bool) throws -> MeshGeometry.VertexRemap {
        let sums = groupNormals()
        let indices = geometry.indices
        var firstGroup = [Int](repeating: -1, count: geometry.vertices.count)
        var split = [Bool](repeating: false, count: geometry.vertices.count)
        for corner in indices.indices {
            let source = Int(indices[corner])
            if firstGroup[source] == -1 {
                firstGroup[source] = group[corner]
            } else if firstGroup[source] != group[corner] {
                split[source] = true
            }
        }

        func replacement(_ source: Int, _ root: Int) -> MeshVertex {
            let vertex = geometry.vertices[source]
            if root == -1 || (keep && !split[source]) { return vertex }
            let sum = sums[root]
            let normal = sum == .zero ? vertex.normal : SIMD3<Float>(simd_normalize(sum))
            return MeshVertex(
                position: vertex.position, normal: normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.secondaryTextureCoordinate, color: vertex.color
            )
        }
        var outputVertices = geometry.vertices.indices.map { replacement($0, firstGroup[$0]) }
        var sourceVertexIndices = Array(geometry.vertices.indices)
        var outputIndices = indices
        var extra: [SIMD2<Int>: UInt32] = [:]
        for corner in indices.indices {
            let source = Int(indices[corner])
            let root = group[corner]
            guard firstGroup[source] != root else { continue }
            let key = SIMD2(source, root)
            if let existing = extra[key] {
                outputIndices[corner] = existing
                continue
            }
            guard outputVertices.count < Int(UInt32.max) else { throw MeshGeometry.GeometryProcessingError.tooManyVertices }
            let output = UInt32(outputVertices.count)
            extra[key] = output
            outputIndices[corner] = output
            outputVertices.append(replacement(source, root))
            sourceVertexIndices.append(source)
        }
        let output = try MeshGeometry(vertices: outputVertices, indices: outputIndices).withGroups(geometry.groups)
        return MeshGeometry.VertexRemap(geometry: output, sourceVertexIndices: sourceVertexIndices)
    }
}
