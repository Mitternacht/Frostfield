import Foundation
import simd

/// Subdivision options shared by the Loop and Catmull–Clark schemes.
public struct SubdivisionOptions: Sendable, Equatable {
    /// How face-varying texture coordinates refine.
    public enum TextureCoordinateInterpolation: Sendable, Equatable {
        /// Coordinates stay linear on each original face, so every texel keeps its place
        /// on the control cage. Like OpenSubdiv's `FVAR_LINEAR_ALL`.
        case linear
        /// Coordinates refine with the same rules as positions, treating UV seams as sharp
        /// boundaries and keeping corners where a seam vertex has one face. Like OpenSubdiv's
        /// `FVAR_LINEAR_CORNERS_ONLY`, it avoids the texture swimming of linear coordinates
        /// on curved results.
        case smooth
    }

    public var textureCoordinates: TextureCoordinateInterpolation
    /// Refines linear RGBA independently of the UV sets.
    public var colorInterpolation: TextureCoordinateInterpolation
    /// Moves the final vertices to their limit positions on the subdivision surface:
    /// Halstead et al.'s masks for Catmull–Clark and Loop's for Loop. Positions on semi-sharp
    /// creases that are still sharp after the last level use the crease mask.
    public var projectsToLimit: Bool
    /// Keeps open-boundary vertices that belong to a single face fixed, such as the
    /// corners of a patch, like OpenSubdiv's `VTX_BOUNDARY_EDGE_AND_CORNER`. Otherwise they
    /// follow the boundary crease rule and round off (`VTX_BOUNDARY_EDGE_ONLY`).
    public var keepsBoundaryCorners: Bool
    /// The largest number of triangles to produce. Positive.
    public var maximumTriangleCount: Int

    public init(
        textureCoordinates: TextureCoordinateInterpolation = .linear, colorInterpolation: TextureCoordinateInterpolation = .linear,
        projectsToLimit: Bool = false, keepsBoundaryCorners: Bool = false, maximumTriangleCount: Int = 16_777_216
    ) {
        self.textureCoordinates = textureCoordinates
        self.colorInterpolation = colorInterpolation
        self.projectsToLimit = projectsToLimit
        self.keepsBoundaryCorners = keepsBoundaryCorners
        self.maximumTriangleCount = maximumTriangleCount
    }
}

extension PolygonMesh {
    /// Catmull–Clark subdivision (1978) for any polygons, with boundaries, creases,
    /// semi-sharp creases and corners. Each level turns an n-sided face into n quads.
    /// Boundary edges use the crease rule, and a vertex with three or more sharp edges is a
    /// corner. Iterations run from 1 to 8.
    public func subdividedCatmullClark(iterations: Int, options: SubdivisionOptions = SubdivisionOptions()) throws -> PolygonMesh {
        guard (1...8).contains(iterations), options.maximumTriangleCount > 0 else {
            throw MeshGeometry.GeometryProcessingError.invalidSubdivision
        }
        // Each level makes one quad per face corner, then four per quad.
        let quads = Double(faceVertexIndices.count) * pow(4, Double(iterations - 1))
        guard quads * 2 <= Double(options.maximumTriangleCount), quads * 4 <= Double(Int32.max) else {
            throw MeshGeometry.GeometryProcessingError.tooManyTriangles
        }
        return try subdivided(.catmullClark, iterations: iterations, options: options)
    }

    /// Loop subdivision (1987) for triangle faces, with boundaries, creases, semi-sharp
    /// creases and corners, using Loop's original vertex weights. Throws for other polygons.
    public func subdividedLoop(iterations: Int, options: SubdivisionOptions = SubdivisionOptions()) throws -> PolygonMesh {
        guard (1...8).contains(iterations), options.maximumTriangleCount > 0 else {
            throw MeshGeometry.GeometryProcessingError.invalidSubdivision
        }
        guard faceVertexCounts.allSatisfy({ $0 == 3 }) else {
            throw MeshGeometry.GeometryProcessingError.invalidPolygonMesh("Loop subdivision needs triangles")
        }
        let triangles = Double(faceVertexCounts.count) * pow(4, Double(iterations))
        guard triangles <= Double(options.maximumTriangleCount), triangles * 3 <= Double(Int32.max) else {
            throw MeshGeometry.GeometryProcessingError.tooManyTriangles
        }
        return try subdivided(.loop, iterations: iterations, options: options)
    }

    private func subdivided(_ scheme: RefinementScheme, iterations: Int, options: SubdivisionOptions) throws -> PolygonMesh {
        var creaseTable: [UInt64: Double] = [:]
        for crease in creases {
            let key = Self.edgeKey(crease.first, crease.second)
            creaseTable[key] = max(creaseTable[key] ?? 0, Double(crease.sharpness))
        }
        var mesh = RefinementMesh(
            points: positions.map { SIMD4(SIMD3<Double>($0), 0) }, faceStarts: faceStarts,
            faceVertices: faceVertexIndices.map { Int32($0) }, creases: creaseTable, corners: Set(corners.map { Int32($0) }),
            mode: .positions, keepsBoundaryCorners: options.keepsBoundaryCorners
        )
        let fvarMode: RefinementMesh.Mode = options.textureCoordinates == .linear ? .linear : .faceVarying
        func faceVarying(
            _ values: [SIMD4<Double>], indices: [UInt32], mode: RefinementMesh.Mode
        ) -> RefinementMesh {
            // A single attribute value can be reused at unrelated positions. Preserve
            // that value while giving each topological vertex its own refinement index.
            var known: [SIMD2<UInt32>: Int32] = [:]
            var points: [SIMD4<Double>] = []
            let faceVertices = indices.enumerated().map { corner, index in
                let key = SIMD2(faceVertexIndices[corner], index)
                if let existing = known[key] { return existing }
                let vertex = Int32(points.count)
                known[key] = vertex
                points.append(values[Int(index)])
                return vertex
            }
            return RefinementMesh(
                points: points, faceStarts: mesh.faceStarts, faceVertices: faceVertices,
                creases: [:], corners: [], mode: mode, keepsBoundaryCorners: true
            )
        }
        var channels = [textureCoordinates, secondaryTextureCoordinates].map { channel in
            channel.map {
                faceVarying($0.values.map { SIMD4(Double($0.x), Double($0.y), 0, 0) }, indices: $0.indices, mode: fvarMode)
            }
        }
        let colorMode: RefinementMesh.Mode = options.colorInterpolation == .linear ? .linear : .faceVarying
        channels.append(colors.map { faceVarying($0.values.map(SIMD4<Double>.init), indices: $0.indices, mode: colorMode) })
        for _ in 0..<iterations {
            mesh = try mesh.refined(scheme)
            for index in channels.indices { channels[index] = try channels[index]?.refined(scheme) }
        }
        if options.projectsToLimit {
            mesh.points = mesh.limitPoints(scheme)
            for index in channels.indices {
                if var channel = channels[index], channel.mode == .faceVarying {
                    channel.points = channel.limitPoints(scheme)
                    channels[index] = channel
                }
            }
        }
        func uvChannel(_ refined: RefinementMesh?) -> FaceVaryingChannel? {
            refined.map {
                FaceVaryingChannel(
                    values: $0.points.map { SIMD2(Float($0.x), Float($0.y)) }, indices: $0.faceVertices.map(UInt32.init)
                )
            }
        }
        let rgbaChannel = channels[2].map {
            ColorChannel(values: $0.points.map(SIMD4<Float>.init), indices: $0.faceVertices.map(UInt32.init))
        }
        let points = mesh.points.map { SIMD3(Float($0.x), Float($0.y), Float($0.z)) }
        guard points.allSatisfy(\.isFiniteVector) else { throw MeshGeometry.GeometryProcessingError.invalidSubdivision }
        var outputCreases: [Crease] = []
        for (key, sharpness) in mesh.creases.sorted(by: { $0.key < $1.key }) where sharpness > 0 {
            outputCreases.append(Crease(UInt32(key >> 32), UInt32(key & 0xFFFF_FFFF), sharpness: Float(sharpness)))
        }
        let starts = mesh.faceStarts
        let childGroup: (Int) -> Int
        if scheme == .loop {
            let stride = 1 << (2 * iterations)
            childGroup = { $0 * stride }
        } else {
            let sourceStarts = faceStarts
            let stride = 1 << (2 * (iterations - 1))
            childGroup = { sourceStarts[$0] * stride }
        }
        let groups = faceGroups.map { group in
            let start = childGroup(group.start)
            let count = childGroup(group.start + group.count) - start
            return FaceGroup(start: start, count: count, materialIndex: group.materialIndex)
        }
        return try PolygonMesh(
            positions: points, faceVertexCounts: (0..<(starts.count - 1)).map { starts[$0 + 1] - starts[$0] },
            faceVertexIndices: mesh.faceVertices.map(UInt32.init), textureCoordinates: uvChannel(channels[0]),
            secondaryTextureCoordinates: uvChannel(channels[1]), colors: rgbaChannel, faceGroups: groups,
            creases: outputCreases, corners: corners
        )
    }
}

extension MeshGeometry {
    /// Loop subdivision of the triangles, like the third-party three-subdivide
    /// `LoopSubdivision.modify`. Positions are welded and zero-area faces discarded.
    /// UV and color seams remain face varying. With `hardEdgesAsCreases`, edges where
    /// the input splits normals become sharp. Normals are recomputed, with splits along
    /// sharp creases. Generate tangents after subdivision for normal mapping.
    public func subdividedLoop(
        iterations: Int, hardEdgesAsCreases: Bool = false, options: SubdivisionOptions = SubdivisionOptions()
    ) throws -> MeshGeometry {
        try PolygonMesh(self, hardEdgesAsCreases: hardEdgesAsCreases)
            .subdividedLoop(iterations: iterations, options: options).triangulated()
    }

    /// Catmull–Clark subdivision of the triangles as polygons, triangulated afterwards.
    /// For quad-dominant input, build a `PolygonMesh` with quads instead: Catmull–Clark on
    /// triangles gives a lumpier surface. Welding, creases, normals and tangents behave as
    /// in `subdividedLoop`.
    public func subdividedCatmullClark(
        iterations: Int, hardEdgesAsCreases: Bool = false, options: SubdivisionOptions = SubdivisionOptions()
    ) throws -> MeshGeometry {
        try PolygonMesh(self, hardEdgesAsCreases: hardEdgesAsCreases)
            .subdividedCatmullClark(iterations: iterations, options: options).triangulated()
    }
}

private enum RefinementScheme {
    case loop, catmullClark
}

/// One subdivision level: points, faces, crease sharpness and corners, in double precision.
private struct RefinementMesh {
    enum Mode {
        /// Positions: creases, boundaries and corners as given.
        case positions
        /// Smooth face-varying values: open edges (UV seams) are sharp.
        case faceVarying
        /// Linear face-varying values: every edge sharp and every vertex a corner.
        case linear
    }

    var points: [SIMD4<Double>]
    var faceStarts: [Int]
    var faceVertices: [Int32]
    /// Sharpness of explicit creases, keyed by the unordered vertex pair.
    var creases: [UInt64: Double]
    var corners: Set<Int32>
    let mode: Mode
    /// Whether open-boundary vertices with one face are corners.
    let keepsBoundaryCorners: Bool

    var faceCount: Int { faceStarts.count - 1 }

    struct Adjacency {
        /// Per face corner, the edge from that corner to the next.
        var cornerEdge: [Int32]
        var ends: [(Int32, Int32)]
        var faceCount: [Int32]
        /// Effective sharpness: infinite on open and non-manifold edges.
        var sharpness: [Double]
    }

    func adjacency() -> Adjacency {
        var lookup: [UInt64: Int32] = [:]
        lookup.reserveCapacity(faceVertices.count)
        var cornerEdge = [Int32](repeating: 0, count: faceVertices.count)
        var ends: [(Int32, Int32)] = []
        var count: [Int32] = []
        for face in 0..<faceCount {
            let start = faceStarts[face], end = faceStarts[face + 1]
            for corner in start..<end {
                let a = faceVertices[corner], b = faceVertices[corner + 1 == end ? start : corner + 1]
                let key = PolygonMesh.edgeKey(UInt32(a), UInt32(b))
                if let edge = lookup[key] {
                    cornerEdge[corner] = edge
                    count[Int(edge)] += 1
                } else {
                    let edge = Int32(ends.count)
                    lookup[key] = edge
                    cornerEdge[corner] = edge
                    ends.append((min(a, b), max(a, b)))
                    count.append(1)
                }
            }
        }
        let sharpness = ends.indices.map { edge -> Double in
            if mode == .linear || count[edge] != 2 { return .infinity }
            return creases[PolygonMesh.edgeKey(UInt32(ends[edge].0), UInt32(ends[edge].1))] ?? 0
        }
        return Adjacency(cornerEdge: cornerEdge, ends: ends, faceCount: count, sharpness: sharpness)
    }

    /// Per-vertex sums over incident edges and faces.
    private struct VertexSums {
        var valence: [Int32]
        var neighborSum: [SIMD4<Double>]
        var sharpCount: [Int32]
        var sharpNeighborSum: [SIMD4<Double>]
        var finiteSharpness: [Double]
        var infinitelySharp: [Bool]
        var faces: [Int32]
        var facePointSum: [SIMD4<Double>]
        /// For quads, the sum of the corner opposite each vertex.
        var diagonalSum: [SIMD4<Double>]
    }

    private func vertexSums(_ adjacency: Adjacency, facePoints: [SIMD4<Double>]) -> VertexSums {
        let n = points.count
        var sums = VertexSums(
            valence: Array(repeating: 0, count: n), neighborSum: Array(repeating: .zero, count: n),
            sharpCount: Array(repeating: 0, count: n), sharpNeighborSum: Array(repeating: .zero, count: n),
            finiteSharpness: Array(repeating: 0, count: n), infinitelySharp: Array(repeating: false, count: n),
            faces: Array(repeating: 0, count: n), facePointSum: Array(repeating: .zero, count: n),
            diagonalSum: Array(repeating: .zero, count: n)
        )
        for (edge, (a, b)) in adjacency.ends.enumerated() {
            let pa = points[Int(a)], pb = points[Int(b)]
            sums.valence[Int(a)] += 1
            sums.valence[Int(b)] += 1
            sums.neighborSum[Int(a)] += pb
            sums.neighborSum[Int(b)] += pa
            let s = adjacency.sharpness[edge]
            guard s > 0 else { continue }
            for (v, other) in [(a, pb), (b, pa)] {
                sums.sharpCount[Int(v)] += 1
                sums.sharpNeighborSum[Int(v)] += other
                if s.isInfinite { sums.infinitelySharp[Int(v)] = true } else { sums.finiteSharpness[Int(v)] += s }
            }
        }
        for face in 0..<faceCount {
            let start = faceStarts[face], count = faceStarts[face + 1] - start
            for k in 0..<count {
                let v = Int(faceVertices[start + k])
                sums.faces[v] += 1
                if !facePoints.isEmpty { sums.facePointSum[v] += facePoints[face] }
                if count == 4 { sums.diagonalSum[v] += points[Int(faceVertices[start + (k + 2) % 4])] }
            }
        }
        return sums
    }

    /// The vertex rule's sharp alternative and its blend weight, or nil for the smooth rule.
    private func sharpRule(
        _ v: Int, sums: VertexSums, crease: (SIMD4<Double>, SIMD4<Double>) -> SIMD4<Double>
    ) -> (point: SIMD4<Double>, weight: Double)? {
        let p = points[v]
        let k = Int(sums.sharpCount[v])
        let explicitCorner = mode == .linear || corners.contains(Int32(v))
            || (keepsBoundaryCorners && sums.faces[v] == 1 && sums.infinitelySharp[v] && k >= 2)
        let weight = explicitCorner || sums.infinitelySharp[v] ? Double.infinity
            : (k > 0 ? sums.finiteSharpness[v] / Double(k) : 0)
        if explicitCorner || k > 2 { return (p, weight) }
        if k == 2 { return (crease(p, sums.sharpNeighborSum[v]), weight) }
        return nil
    }

    private static func blend(_ smooth: SIMD4<Double>, _ rule: (point: SIMD4<Double>, weight: Double)?) -> SIMD4<Double> {
        guard let rule else { return smooth }
        if rule.weight >= 1 { return rule.point }
        return smooth + (rule.point - smooth) * rule.weight
    }

    /// Loop's β for a smooth vertex of valence n.
    static func loopBeta(_ n: Int) -> Double {
        let c = 3.0 / 8 + cos(2 * Double.pi / Double(n)) / 4
        return (5.0 / 8 - c * c) / Double(n)
    }

    func refined(_ scheme: RefinementScheme) throws -> RefinementMesh {
        let adjacency = adjacency()
        let vertexCount = points.count
        let edgeCount = adjacency.ends.count

        var facePoints: [SIMD4<Double>] = []
        if scheme == .catmullClark {
            facePoints.reserveCapacity(faceCount)
            for face in 0..<faceCount {
                let start = faceStarts[face], end = faceStarts[face + 1]
                var sum = SIMD4<Double>.zero
                for corner in start..<end { sum += points[Int(faceVertices[corner])] }
                facePoints.append(sum / Double(end - start))
            }
        }

        // Per edge: face points (Catmull–Clark) or opposite vertices (Loop) of its faces.
        var edgeFaceSum = [SIMD4<Double>](repeating: .zero, count: edgeCount)
        for face in 0..<faceCount {
            let start = faceStarts[face], end = faceStarts[face + 1]
            for corner in start..<end {
                let edge = Int(adjacency.cornerEdge[corner])
                switch scheme {
                case .catmullClark:
                    edgeFaceSum[edge] += facePoints[face]
                case .loop:
                    let opposite = corner + 2 < end ? corner + 2 : corner + 2 - (end - start)
                    edgeFaceSum[edge] += points[Int(faceVertices[opposite])]
                }
            }
        }

        let faceOffset = scheme == .catmullClark ? faceCount : 0
        guard vertexCount + faceOffset + edgeCount <= Int(Int32.max) else {
            throw MeshGeometry.GeometryProcessingError.tooManyVertices
        }
        var newPoints = [SIMD4<Double>](repeating: .zero, count: vertexCount + faceOffset + edgeCount)
        for face in 0..<faceOffset { newPoints[vertexCount + face] = facePoints[face] }
        for edge in 0..<edgeCount {
            let (a, b) = adjacency.ends[edge]
            let pa = points[Int(a)], pb = points[Int(b)]
            let midpoint = (pa + pb) * 0.5
            let s = adjacency.sharpness[edge]
            let point: SIMD4<Double>
            if s >= 1 {
                point = midpoint
            } else {
                let smooth = scheme == .catmullClark
                    ? (pa + pb + edgeFaceSum[edge]) * 0.25
                    : (pa + pb) * (3.0 / 8) + edgeFaceSum[edge] * (1.0 / 8)
                point = smooth + (midpoint - smooth) * s
            }
            newPoints[vertexCount + faceOffset + edge] = point
        }

        let sums = vertexSums(adjacency, facePoints: facePoints)
        for v in 0..<vertexCount {
            let p = points[v]
            let n = Double(sums.valence[v])
            guard n > 0 else {
                newPoints[v] = p
                continue
            }
            let smooth: SIMD4<Double>
            switch scheme {
            case .catmullClark:
                let q = sums.facePointSum[v] / Double(max(sums.faces[v], 1))
                let r = (p * n + sums.neighborSum[v]) / (2 * n)
                smooth = (q + 2 * r + (n - 3) * p) / n
            case .loop:
                let beta = Self.loopBeta(Int(n))
                smooth = p * (1 - n * beta) + sums.neighborSum[v] * beta
            }
            let rule = sharpRule(v, sums: sums) { p, ends in
                scheme == .catmullClark ? (ends + 6 * p) / 8 : p * 0.75 + ends * 0.125
            }
            newPoints[v] = Self.blend(smooth, rule)
        }

        var starts = [0]
        var vertices: [Int32] = []
        switch scheme {
        case .catmullClark:
            starts.reserveCapacity(faceVertices.count + 1)
            vertices.reserveCapacity(faceVertices.count * 4)
            let edgeBase = Int32(vertexCount + faceOffset)
            for face in 0..<faceCount {
                let start = faceStarts[face], end = faceStarts[face + 1], count = end - start
                for k in 0..<count {
                    let corner = start + k
                    let previous = start + (k + count - 1) % count
                    vertices += [
                        faceVertices[corner], edgeBase + adjacency.cornerEdge[corner],
                        Int32(vertexCount + face), edgeBase + adjacency.cornerEdge[previous],
                    ]
                    starts.append(vertices.count)
                }
            }
        case .loop:
            starts.reserveCapacity(faceCount * 4 + 1)
            vertices.reserveCapacity(faceCount * 12)
            let edgeBase = Int32(vertexCount)
            for face in 0..<faceCount {
                let start = faceStarts[face]
                let a = faceVertices[start], b = faceVertices[start + 1], c = faceVertices[start + 2]
                let ab = edgeBase + adjacency.cornerEdge[start]
                let bc = edgeBase + adjacency.cornerEdge[start + 1]
                let ca = edgeBase + adjacency.cornerEdge[start + 2]
                vertices += [a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]
                for _ in 0..<4 { starts.append(starts[starts.count - 1] + 3) }
            }
        }

        // Each half of a crease keeps the parent's sharpness less one (infinite stays infinite).
        var childCreases: [UInt64: Double] = [:]
        if mode == .positions {
            for (edge, (a, b)) in adjacency.ends.enumerated() {
                guard let s = creases[PolygonMesh.edgeKey(UInt32(a), UInt32(b))] else { continue }
                let child = s.isInfinite ? s : s - 1
                guard child > 0 else { continue }
                let middle = UInt32(vertexCount + faceOffset + edge)
                childCreases[PolygonMesh.edgeKey(UInt32(a), middle)] = child
                childCreases[PolygonMesh.edgeKey(middle, UInt32(b))] = child
            }
        }
        return RefinementMesh(
            points: newPoints, faceStarts: starts, faceVertices: vertices, creases: childCreases,
            corners: corners, mode: mode, keepsBoundaryCorners: keepsBoundaryCorners
        )
    }

    /// Limit positions of the current vertices.
    func limitPoints(_ scheme: RefinementScheme) -> [SIMD4<Double>] {
        let adjacency = adjacency()
        let sums = vertexSums(adjacency, facePoints: [])
        return points.indices.map { v in
            let p = points[v]
            let n = Double(sums.valence[v])
            guard n > 0 else { return p }
            let smooth: SIMD4<Double>
            switch scheme {
            case .catmullClark:
                // Halstead, Kass and DeRose 1993, for vertices whose faces are all quads.
                smooth = (n * n * p + 4 * sums.neighborSum[v] + sums.diagonalSum[v]) / (n * (n + 5))
            case .loop:
                let beta = Self.loopBeta(Int(n))
                let chi = 1 / (3 / (8 * beta) + n)
                smooth = p * (1 - n * chi) + sums.neighborSum[v] * chi
            }
            let rule = sharpRule(v, sums: sums) { p, ends in (ends + 4 * p) / 6 }
            return Self.blend(smooth, rule)
        }
    }
}
