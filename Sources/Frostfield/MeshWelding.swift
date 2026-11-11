import Foundation
import simd

extension MeshGeometry {
    /// How close two vertices must be in each attribute for `welded(tolerance:)` to merge them.
    /// A nil attribute tolerance ignores that attribute; the merged vertex keeps the
    /// attributes of the lowest-indexed vertex in its cluster.
    public struct WeldTolerance: Sendable, Equatable {
        /// The largest Euclidean distance between merged positions, finite and nonnegative.
        public var position: Float
        /// The largest angle in radians between merged normals, in 0...π, or nil to ignore normals.
        public var normalAngle: Float?
        /// The largest per-component difference between merged texture coordinates in both
        /// sets, finite and nonnegative, or nil to ignore them. A vertex with a secondary set
        /// never merges with one without.
        public var textureCoordinate: Float?
        /// The largest angle in radians between merged tangents, in 0...π, or nil to ignore
        /// tangents. Merged tangents also need the same handedness, and a vertex with a
        /// tangent never merges with one without.
        public var tangentAngle: Float?
        /// The largest per-channel linear RGBA difference, or nil to ignore color.
        /// The default preserves color boundaries. Missing colors compare as white.
        public var color: Float?

        public init(
            position: Float = 1e-4, normalAngle: Float? = 1e-3, textureCoordinate: Float? = 1e-4,
            tangentAngle: Float? = 1e-3, color: Float? = 0
        ) {
            self.position = position
            self.normalAngle = normalAngle
            self.textureCoordinate = textureCoordinate
            self.tangentAngle = tangentAngle
            self.color = color
        }

        /// Merges by position alone, for connectivity, within a distance.
        public static func position(_ distance: Float) -> WeldTolerance {
            WeldTolerance(position: distance, normalAngle: nil, textureCoordinate: nil, tangentAngle: nil, color: nil)
        }

        func validate() throws {
            guard position.isFinite, position >= 0,
                  normalAngle.map({ $0.isFinite && (0...Float.pi).contains($0) }) != false,
                  textureCoordinate.map({ $0.isFinite && $0 >= 0 }) != false,
                  color.map({ $0.isFinite && $0 >= 0 }) != false,
                  tangentAngle.map({ $0.isFinite && (0...Float.pi).contains($0) }) != false else {
                throw GeometryProcessingError.invalidTolerance
            }
        }
    }

    /// Geometry with one source vertex index for each output vertex.
    public struct VertexRemap: Sendable {
        public let geometry: MeshGeometry
        public let sourceVertexIndices: [Int]
    }

    /// Merges referenced vertices that agree within the supplied tolerances.
    /// Each joins the lowest-indexed qualifying representative. Material ranges
    /// follow surviving triangles and newly collapsed triangles are dropped by default.
    /// Use ``weldedWithRemap(tolerance:removingDegenerateTriangles:canMerge:)``
    /// to carry separately stored per-vertex channels.
    public func welded(
        tolerance: WeldTolerance = WeldTolerance(), removingDegenerateTriangles: Bool = true
    ) throws -> MeshGeometry {
        try weldedWithRemap(tolerance: tolerance, removingDegenerateTriangles: removingDegenerateTriangles).geometry
    }

    /// Returns the source representative for each output vertex. Use `canMerge`
    /// to require equality of external channels, such as skin weights or morph deltas,
    /// before welding. It receives the representative and the candidate source indices.
    public func weldedWithRemap(
        tolerance: WeldTolerance = WeldTolerance(), removingDegenerateTriangles: Bool = true,
        canMerge: (Int, Int) -> Bool = { _, _ in true }
    ) throws -> VertexRemap {
        try tolerance.validate()
        let representatives = weldRepresentatives(tolerance, canMerge: canMerge)
        var remapped: [UInt32] = []
        remapped.reserveCapacity(indices.count)
        var groupOffsets: [Int] = [0]
        for start in stride(from: 0, to: indices.count, by: 3) {
            let a = representatives[Int(indices[start])]
            let b = representatives[Int(indices[start + 1])]
            let c = representatives[Int(indices[start + 2])]
            if !removingDegenerateTriangles || (a != b && b != c && a != c) {
                remapped += [a, b, c]
            }
            if !groups.isEmpty { groupOffsets.append(remapped.count) }
        }
        guard !remapped.isEmpty else { throw GeometryProcessingError.noTrianglesRemain }

        var newIndex = [UInt32](repeating: .max, count: vertices.count)
        for index in remapped { newIndex[Int(index)] = 0 }
        var kept: [MeshVertex] = []
        var sources: [Int] = []
        for index in vertices.indices where newIndex[index] == 0 {
            newIndex[index] = UInt32(kept.count)
            kept.append(vertices[index])
            sources.append(index)
        }
        let result = try MeshGeometry(vertices: kept, indices: remapped.map { newIndex[Int($0)] })
            .withGroups(Self.remappedGroups(groups, offsets: groupOffsets))
        return VertexRemap(geometry: result, sourceVertexIndices: sources)
    }

    /// For each vertex, the index of its cluster's representative.
    private func weldRepresentatives(_ tolerance: WeldTolerance, canMerge: (Int, Int) -> Bool) -> [UInt32] {
        var used = [Bool](repeating: false, count: vertices.count)
        for index in indices { used[Int(index)] = true }
        let distanceSquared = Double(tolerance.position) * Double(tolerance.position)

        func angleMatches(_ a: SIMD3<Double>, _ b: SIMD3<Double>, limit: Float) -> Bool {
            atan2(simd_length(simd_cross(a, b)), simd_dot(a, b)) <= Double(limit)
        }

        func matches(_ representative: UInt32, _ candidate: Int) -> Bool {
            let a = vertices[Int(representative)]
            let b = vertices[candidate]
            let delta = SIMD3<Double>(a.position) - SIMD3<Double>(b.position)
            guard simd_length_squared(delta) <= distanceSquared else { return false }
            if let limit = tolerance.normalAngle,
               !angleMatches(SIMD3<Double>(a.normal), SIMD3<Double>(b.normal), limit: limit) { return false }
            if let limit = tolerance.textureCoordinate {
                guard simd_reduce_max(simd_abs(a.textureCoordinate - b.textureCoordinate)) <= limit else { return false }
                switch (a.secondaryTextureCoordinate, b.secondaryTextureCoordinate) {
                case (nil, nil): break
                case (let x?, let y?): guard simd_reduce_max(simd_abs(x - y)) <= limit else { return false }
                default: return false
                }
            }
            if let limit = tolerance.tangentAngle {
                switch (a.tangent, b.tangent) {
                case (nil, nil): break
                case (let x?, let y?):
                    let tx = SIMD3<Double>(Double(x.x), Double(x.y), Double(x.z))
                    let ty = SIMD3<Double>(Double(y.x), Double(y.y), Double(y.z))
                    guard x.w == y.w, angleMatches(tx, ty, limit: limit) else { return false }
                default: return false
                }
            }
            if let limit = tolerance.color {
                let white = SIMD4<Float>(repeating: 1)
                guard simd_reduce_max(simd_abs((a.color ?? white) - (b.color ?? white))) <= limit else { return false }
            }
            return canMerge(Int(representative), candidate)
        }

        var representative = [UInt32](repeating: 0, count: vertices.count)
        if tolerance.position == 0 {
            // Exact positions: one bucket per position bit pattern, with −0 as +0.
            var buckets: [SIMD3<UInt32>: [UInt32]] = [:]
            for (index, vertex) in vertices.enumerated() where used[index] {
                let key = vertex.position.exactKey
                if let found = buckets[key]?.first(where: { matches($0, index) }) {
                    representative[index] = found
                } else {
                    buckets[key, default: []].append(UInt32(index))
                    representative[index] = UInt32(index)
                }
            }
            return representative
        }

        // Cells twice the tolerance wide, so a tolerance ball overlaps at most two cells per axis.
        let cell = 2 * Double(tolerance.position)
        var grid: [SIMD3<Int64>: [UInt32]] = [:]
        grid.reserveCapacity(vertices.count)
        func cellIndex(_ value: Double) -> Int64 {
            let scaled = (value / cell).rounded(.down)
            return Int64(max(min(scaled, 4e18), -4e18))
        }
        for (index, vertex) in vertices.enumerated() where used[index] {
            let p = SIMD3<Double>(vertex.position)
            let home = SIMD3(cellIndex(p.x), cellIndex(p.y), cellIndex(p.z))
            var low = home, high = home
            for axis in 0..<3 {
                low[axis] = cellIndex(p[axis] - Double(tolerance.position))
                high[axis] = cellIndex(p[axis] + Double(tolerance.position))
            }
            var best = UInt32.max
            for x in low.x...high.x {
                for y in low.y...high.y {
                    for z in low.z...high.z {
                        guard let candidates = grid[SIMD3(x, y, z)] else { continue }
                        for candidate in candidates where candidate < best && matches(candidate, index) {
                            best = candidate
                        }
                    }
                }
            }
            if best != .max {
                representative[index] = best
            } else {
                grid[home, default: []].append(UInt32(index))
                representative[index] = UInt32(index)
            }
        }
        return representative
    }

    static func remappedGroups(_ groups: [MeshGeometryGroup], offsets: [Int]) throws -> [MeshGeometryGroup] {
        let output = groups.compactMap { group -> MeshGeometryGroup? in
            let start = offsets[group.start / 3]
            let count = offsets[group.range.upperBound / 3] - start
            guard count > 0 else { return nil }
            return MeshGeometryGroup(start: start, count: count, materialIndex: group.materialIndex)
        }
        // An empty group list would make SceneNode.grouped draw every surviving triangle.
        guard groups.isEmpty || !output.isEmpty else { throw GeometryProcessingError.noMaterialGroupsRemain }
        return output
    }

    /// Bytes for packed base mesh vertices, optional vertex colors, and UInt32 indices.
    /// Excludes skinning, morph targets, instances, and GPU allocation overhead.
    public var estimatedGPUByteCount: Int {
        let vectorStride = MemoryLayout<SIMD4<Float>>.stride
        let vectorsPerVertex = hasVertexColors ? 5 : 4
        return vertices.count * vectorsPerVertex * vectorStride + indices.count * MemoryLayout<UInt32>.stride
    }

    /// Tessellated geometry and the edges used to generate its appended vertices.
    public struct Tessellation: Sendable {
        public let geometry: MeshGeometry
        /// Each entry gives the two endpoint indices for one appended midpoint.
        /// Endpoints can refer to earlier midpoints, so callers can carry their own
        /// channels by appending an interpolated value for each entry in order.
        public let midpointSources: [SIMD2<UInt32>]
    }

    /// Splits triangles until no edge is longer than `maximumEdgeLength`, like Three.js
    /// `TessellateModifier`, but without cracks: each long edge is split at its midpoint
    /// in every triangle that shares it, including across UV and normal seams.
    ///
    /// Each pass splits every edge longer than the limit. A triangle with one split edge
    /// becomes two triangles, with two becomes three (cutting the remaining quad along its
    /// shorter diagonal), and with three becomes four. Midpoint vertices interpolate
    /// texture coordinates linearly and normalize interpolated normals. Interpolated
    /// tangents are kept when both ends have one with the same handedness and a nonzero
    /// sum. Interpolates linear colors and preserves material groups. Edges may still
    /// exceed the limit when the pass budget is reached. Throws if the triangle budget
    /// is exceeded or if a long edge has no representable midpoint.
    public func tessellated(
        maximumEdgeLength: Float, maximumPasses: Int = 8, maximumTriangleCount: Int = 16_777_216
    ) throws -> MeshGeometry {
        try tessellatedWithRemap(
            maximumEdgeLength: maximumEdgeLength, maximumPasses: maximumPasses,
            maximumTriangleCount: maximumTriangleCount
        ).geometry
    }

    /// Tessellates and records how each new vertex interpolates two earlier vertices.
    public func tessellatedWithRemap(
        maximumEdgeLength: Float, maximumPasses: Int = 8, maximumTriangleCount: Int = 16_777_216
    ) throws -> Tessellation {
        guard maximumEdgeLength.isFinite, maximumEdgeLength > 0, (1...32).contains(maximumPasses),
              maximumTriangleCount > 0 else {
            throw GeometryProcessingError.invalidTessellation
        }
        guard indices.count / 3 <= maximumTriangleCount else { throw GeometryProcessingError.tooManyTriangles }
        let limitSquared = Double(maximumEdgeLength) * Double(maximumEdgeLength)
        var vertices = self.vertices
        var indices = self.indices
        var groups = self.groups
        var midpointSources: [SIMD2<UInt32>] = []
        for _ in 0..<maximumPasses {
            var midpoints: [UInt64: UInt32] = [:]
            var output: [UInt32] = []
            output.reserveCapacity(indices.count)
            var groupOffsets: [Int] = [0]
            var triangleCount = indices.count / 3
            func midpoint(_ a: UInt32, _ b: UInt32) throws -> UInt32 {
                let key = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
                if let existing = midpoints[key] { return existing }
                guard vertices.count < Int(UInt32.max) else { throw GeometryProcessingError.tooManyVertices }
                let created = UInt32(vertices.count)
                let low = min(a, b), high = max(a, b)
                let first = vertices[Int(low)], second = vertices[Int(high)]
                let vertex = Self.interpolatedVertex(first, second)
                guard vertex.position != first.position, vertex.position != second.position else {
                    throw GeometryProcessingError.unrepresentableMidpoint
                }
                vertices.append(vertex)
                midpointSources.append(SIMD2(low, high))
                midpoints[key] = created
                return created
            }
            for start in stride(from: 0, to: indices.count, by: 3) {
                let corner = [indices[start], indices[start + 1], indices[start + 2]]
                let long = (0..<3).map { edge in
                    // Symmetric in the two ends, so both sides of a seam decide alike.
                    let a = SIMD3<Double>(vertices[Int(corner[edge])].position)
                    let b = SIMD3<Double>(vertices[Int(corner[(edge + 1) % 3])].position)
                    return simd_distance_squared(a, b) > limitSquared
                }
                let count = long.count(where: { $0 })
                guard count <= maximumTriangleCount - triangleCount else {
                    throw GeometryProcessingError.tooManyTriangles
                }
                triangleCount += count
                if count == 0 {
                    output += corner
                } else if count == 3 {
                    let m01 = try midpoint(corner[0], corner[1])
                    let m12 = try midpoint(corner[1], corner[2])
                    let m20 = try midpoint(corner[2], corner[0])
                    output += [corner[0], m01, m20, m01, corner[1], m12, m20, m12, corner[2], m01, m12, m20]
                } else if count == 1 {
                    let edge = long.firstIndex(of: true)!
                    let a = corner[edge], b = corner[(edge + 1) % 3], c = corner[(edge + 2) % 3]
                    let m = try midpoint(a, b)
                    output += [a, m, c, m, b, c]
                } else {
                    // The unsplit edge runs from a to b; c is the apex between the two split edges.
                    let edge = long.firstIndex(of: false)!
                    let a = corner[edge], b = corner[(edge + 1) % 3], c = corner[(edge + 2) % 3]
                    let mbc = try midpoint(b, c)
                    let mca = try midpoint(c, a)
                    output += [mca, mbc, c]
                    let pa = SIMD3<Double>(vertices[Int(a)].position)
                    let pb = SIMD3<Double>(vertices[Int(b)].position)
                    let pm1 = SIMD3<Double>(vertices[Int(mbc)].position)
                    let pm2 = SIMD3<Double>(vertices[Int(mca)].position)
                    if simd_distance_squared(pa, pm1) <= simd_distance_squared(pb, pm2) {
                        output += [a, b, mbc, a, mbc, mca]
                    } else {
                        output += [a, b, mca, b, mbc, mca]
                    }
                }
                if !groups.isEmpty { groupOffsets.append(output.count) }
            }
            if midpoints.isEmpty { break }
            groups = try Self.remappedGroups(groups, offsets: groupOffsets)
            indices = output
        }
        let result = try MeshGeometry(vertices: vertices, indices: indices).withGroups(groups)
        return Tessellation(geometry: result, midpointSources: midpointSources)
    }

    private static func interpolatedVertex(_ a: MeshVertex, _ b: MeshVertex) -> MeshVertex {
        let sum = SIMD3<Double>(a.normal) + SIMD3<Double>(b.normal)
        let normal = sum == .zero ? a.normal : SIMD3<Float>(simd_normalize(sum))
        var tangent: SIMD4<Float>?
        if let ta = a.tangent, let tb = b.tangent, ta.w == tb.w {
            let sum = SIMD3(ta.x + tb.x, ta.y + tb.y, ta.z + tb.z)
            if simd_length_squared(sum) > 1e-12 { tangent = SIMD4(simd_normalize(sum), ta.w) }
        }
        let secondary = a.secondaryTextureCoordinate.flatMap { aUV in
            b.secondaryTextureCoordinate.map { bUV in SIMD2<Float>((SIMD2<Double>(aUV) + SIMD2<Double>(bUV)) * 0.5) }
        }
        var color: SIMD4<Float>?
        if a.color != nil || b.color != nil {
            let white = SIMD4<Float>(repeating: 1)
            color = SIMD4<Float>((SIMD4<Double>(a.color ?? white) + SIMD4<Double>(b.color ?? white)) * 0.5)
        }
        return MeshVertex(
            position: SIMD3<Float>((SIMD3<Double>(a.position) + SIMD3<Double>(b.position)) * 0.5), normal: normal,
            textureCoordinate: SIMD2<Float>((SIMD2<Double>(a.textureCoordinate) + SIMD2<Double>(b.textureCoordinate)) * 0.5),
            secondaryTextureCoordinate: secondary, tangent: tangent, color: color
        )
    }

    public enum GeometryProcessingError: LocalizedError, Equatable {
        case invalidTolerance
        case noTrianglesRemain
        case noMaterialGroupsRemain
        case tooManyVertices
        case tooManyTriangles
        case invalidTessellation
        case unrepresentableMidpoint
        case invalidCreaseAngle
        case invalidSimplification
        case invalidLevelOfDetail
        case unrepresentableLevelOfDetailDistance
        case invalidPolygonMesh(String)
        case invalidSubdivision

        public var errorDescription: String? {
            switch self {
            case .invalidTolerance:
                "Weld tolerances must be finite and nonnegative, with angles from zero to π."
            case .noTrianglesRemain:
                "The operation would leave no triangles."
            case .noMaterialGroupsRemain:
                "The operation removed every material group while ungrouped triangles remain."
            case .tooManyVertices:
                "The result would exceed the 32-bit vertex index range."
            case .tooManyTriangles:
                "The result would exceed the requested triangle limit."
            case .invalidTessellation:
                "Tessellation needs a positive finite edge length, 1 to 32 passes, and a positive triangle limit."
            case .unrepresentableMidpoint:
                "The midpoint of a long edge rounds to an endpoint in Float precision."
            case .invalidCreaseAngle:
                "The crease angle must be from zero to π radians."
            case .invalidSimplification:
                "Simplification needs a nonnegative target and error budget, finite nonnegative attribute weights, and a ratio from zero to one."
            case .invalidLevelOfDetail:
                "LOD generation needs 1 to \(LevelOfDetail.maximumLevelCount) levels, a ratio between zero and one, a valid perspective pixel budget, and an invertible affine world transform."
            case .unrepresentableLevelOfDetailDistance:
                "The estimated LOD switch distance exceeds the finite world distance range."
            case .invalidPolygonMesh(let reason):
                "Invalid polygon mesh: \(reason)."
            case .invalidSubdivision:
                "Subdivision needs 1 to 8 iterations, a positive triangle limit, and finite output vertices."
            }
        }
    }
}

extension SIMD3 where Scalar == Float {
    /// Bit patterns with −0 folded into +0, so equal positions share a key.
    var exactKey: SIMD3<UInt32> {
        SIMD3<UInt32>((x == 0 ? Float(0) : x).bitPattern, (y == 0 ? Float(0) : y).bitPattern, (z == 0 ? Float(0) : z).bitPattern)
    }
}
