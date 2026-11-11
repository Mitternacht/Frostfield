import Foundation
import simd

extension MeshGeometry {
    /// Error weights and constraints for `simplified(targetTriangleCount:options:)`.
    public struct SimplificationOptions: Sendable, Equatable {
        /// The largest collapse error to accept, in mesh units; collapses stop before
        /// exceeding it even when the target count has not been reached.
        public var maximumError: Float
        /// Scales texture-coordinate differences in the error. At weight 1, a UV difference
        /// of 0.01 costs as much as moving the surface by 1% of the mesh's largest dimension.
        /// Zero ignores texture coordinates in the error; seams are still preserved.
        public var textureCoordinateWeight: Float
        /// Scales unit-normal differences in the error, like `textureCoordinateWeight`. Keep
        /// it small: on curved surfaces normals vary much faster than positions, and large
        /// weights trade geometric accuracy for smoother shading.
        public var normalWeight: Float
        /// Weights linear RGBA differences in the error. Missing colors sample as opaque white.
        /// Zero keeps each representative vertex's authored color.
        public var colorWeight: Float
        /// Keeps every vertex on an open boundary in place, so meshes that tile still meet.
        public var locksBoundary: Bool
        /// Where a collapsed edge's vertex goes.
        public var vertexPlacement: VertexPlacement

        /// Where a collapse puts the merged vertex.
        public enum VertexPlacement: Sendable, Equatable {
            /// At the position and attributes that minimize the combined quadric, when both
            /// ends are interior vertices with a single set of attributes and the minimum lies
            /// within one edge length of the edge's midpoint; otherwise at the better endpoint.
            /// Moved vertices get least-squares texture coordinates, normals, and colors from
            /// the attribute quadrics, or the surviving endpoint's when that weight is zero.
            /// Tangents are re-orthogonalized to the new normal.
            case optimal
            /// Always at an endpoint, so every output vertex is an unchanged input vertex.
            case endpoints
        }

        public init(
            maximumError: Float = .infinity, textureCoordinateWeight: Float = 0.5, normalWeight: Float = 0.05,
            colorWeight: Float = 0.25, locksBoundary: Bool = false, vertexPlacement: VertexPlacement = .optimal
        ) {
            self.maximumError = maximumError
            self.textureCoordinateWeight = textureCoordinateWeight
            self.normalWeight = normalWeight
            self.colorWeight = colorWeight
            self.locksBoundary = locksBoundary
            self.vertexPlacement = vertexPlacement
        }

        func validate() throws {
            guard !maximumError.isNaN, maximumError >= 0,
                  [textureCoordinateWeight, normalWeight, colorWeight].allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw GeometryProcessingError.invalidSimplification
            }
        }
    }

    /// A simplified mesh and how far it moved from the input.
    public struct Simplification: Sendable {
        public let geometry: MeshGeometry
        /// The largest accepted collapse error, in mesh units: the square root of the
        /// area-weighted mean squared distance, in position and weighted attribute space,
        /// from the planes of the input triangles around the merged vertex. Zero when
        /// nothing collapsed or every collapse was exact to 10⁻⁶ of the mesh's extent.
        public let error: Float
        /// For each output vertex, the input vertex it copies or, under optimal placement, the
        /// input vertex whose attributes it updates. Use `.endpoints` and this remap to
        /// copy external vertex channels such as skin weights or morph deltas. The remap
        /// does not estimate how the new topology deforms under those external channels.
        public let sourceVertexIndices: [Int]
    }

    /// Reduces the triangle count with quadric error metrics (Garland and Heckbert 1997),
    /// extended with attribute quadrics for texture coordinates, normals and colors (Hoppe 1999),
    /// like `meshopt_simplifyWithAttributes` and Three.js `SimplifyModifier`.
    ///
    /// Each step collapses an edge. With `.endpoints` placement the merged vertex is one of
    /// the ends, so output vertices are a subset of input vertices with their attributes
    /// unchanged; `.optimal` placement moves interior vertices to the quadric minimum and
    /// keeps the endpoint rule for seams and boundaries. Vertices with the same
    /// position and different attributes form a seam; a collapse may move a seam vertex
    /// only along the seam, carrying each side's attributes with it. Boundary vertices
    /// move only along the boundary, and edge quadrics keep seams and boundaries in
    /// shape. Material group boundaries are fixed and each group keeps at least one triangle.
    /// Collapses that would fold a triangle past about 75°, flip or flatten its primary UV
    /// orientation, or change local manifold connectivity are rejected. The deterministic
    /// result may stop above the target when the error limit or topology prevents further
    /// collapses. The reported quadric error is an estimate, separate from surface distance.
    public func simplified(
        targetTriangleCount: Int, options: SimplificationOptions = SimplificationOptions()
    ) throws -> Simplification {
        guard targetTriangleCount >= 0 else { throw GeometryProcessingError.invalidSimplification }
        try options.validate()
        let count = indices.count / 3
        guard targetTriangleCount < count else {
            return Simplification(geometry: self, error: 0, sourceVertexIndices: Array(vertices.indices))
        }
        try validateSimplificationCapacity()
        var simplifier = QuadricSimplifier(self, options: options)
        simplifier.simplify(toTriangleCount: max(1, targetTriangleCount))
        return try simplifier.result()
    }

    /// Simplifies to `ratio` (0...1) of the current triangle count, rounded to nearest.
    public func simplified(
        ratio: Float, options: SimplificationOptions = SimplificationOptions()
    ) throws -> Simplification {
        guard ratio.isFinite, (0...1).contains(ratio) else { throw GeometryProcessingError.invalidSimplification }
        return try simplified(
            targetTriangleCount: Int((Double(indices.count / 3) * Double(ratio)).rounded()), options: options
        )
    }
    func validateSimplificationCapacity() throws {
        guard vertices.count < Int(UInt32.max - 1) else { throw GeometryProcessingError.tooManyVertices }
        guard indices.count <= Int(Int32.max) else { throw GeometryProcessingError.tooManyTriangles }
    }

}

/// A symmetric quadratic form (pᵀAp + 2b·p + c) with its accumulated area weight.
private struct Quadric {
    var a00 = 0.0, a01 = 0.0, a02 = 0.0, a11 = 0.0, a12 = 0.0, a22 = 0.0
    var b0 = 0.0, b1 = 0.0, b2 = 0.0, c = 0.0, w = 0.0

    /// Adds `weight · (g·p + d)²` without changing the weight total.
    mutating func addSquare(_ g: SIMD3<Double>, _ d: Double, weight: Double) {
        a00 += weight * g.x * g.x
        a01 += weight * g.x * g.y
        a02 += weight * g.x * g.z
        a11 += weight * g.y * g.y
        a12 += weight * g.y * g.z
        a22 += weight * g.z * g.z
        b0 += weight * g.x * d
        b1 += weight * g.y * d
        b2 += weight * g.z * d
        c += weight * d * d
    }

    mutating func add(_ other: Quadric) {
        a00 += other.a00; a01 += other.a01; a02 += other.a02
        a11 += other.a11; a12 += other.a12; a22 += other.a22
        b0 += other.b0; b1 += other.b1; b2 += other.b2
        c += other.c; w += other.w
    }

    func evaluate(_ p: SIMD3<Double>) -> Double {
        let x = p.x, y = p.y, z = p.z
        let ax = a00 * x + a01 * y + a02 * z
        let ay = a01 * x + a11 * y + a12 * z
        let az = a02 * x + a12 * y + a22 * z
        return x * ax + y * ay + z * az + 2 * (b0 * x + b1 * y + b2 * z) + c
    }

    static func + (a: Quadric, b: Quadric) -> Quadric {
        var sum = a
        sum.add(b)
        return sum
    }
}

/// Pass-based half-edge-collapse simplification, as in meshoptimizer: each pass ranks
/// every edge's cheaper valid direction, then applies collapses in order, at most one
/// per vertex, re-checking topology and flips against the collapses already applied.
struct QuadricSimplifier {
    private let source: MeshGeometry
    private let options: MeshGeometry.SimplificationOptions
    /// Positions scaled to a unit largest dimension, in double precision. Indexed by wedge;
    /// geometric code reads the entry of a position's first wedge.
    private var positions: [SIMD3<Double>]
    private let scale: Double
    private let center: SIMD3<Double>
    /// Wedges whose position and attributes optimal placement changed.
    private var moved: Set<UInt32> = []
    /// Per wedge (input vertex), the first wedge with the same position.
    private let positionOf: [UInt32]
    private let attributeCount: Int
    /// Per wedge, `attributeCount` weighted attribute values.
    private var attributes: [Double]
    /// Per wedge, the primary texture coordinates, for the texture fold check.
    private var textureCoordinates: [SIMD2<Double>]
    private let hasSecondaryAttributes: Bool
    private let hasColorAttributes: Bool
    /// Per position (indexed by its first wedge): plane and edge quadrics.
    private var positionQuadrics: [Quadric]
    /// Per wedge: the attribute quadric and, per attribute, the weighted gradient sum and offset sum.
    private var attributeQuadrics: [Quadric]
    private var gradients: [Double]
    /// Current triangles as wedge triples.
    private var triangles: [UInt32]
    /// Material group slot per current triangle, with −1 for ungrouped faces.
    private var triangleGroups: [Int] = []
    private var error: Double = 0
    private var hardLocked: [Bool]

    init(_ geometry: MeshGeometry, options: MeshGeometry.SimplificationOptions) {
        source = geometry
        self.options = options
        let vertexCount = geometry.vertices.count

        var lower = SIMD3<Double>(repeating: .infinity), upper = SIMD3<Double>(repeating: -.infinity)
        for vertex in geometry.vertices {
            lower = simd_min(lower, SIMD3<Double>(vertex.position))
            upper = simd_max(upper, SIMD3<Double>(vertex.position))
        }
        let extent = simd_reduce_max(upper - lower)
        scale = extent > 0 ? extent : 1
        let center = (lower + upper) * 0.5
        self.center = center
        positions = geometry.vertices.map { (SIMD3<Double>($0.position) - center) / (extent > 0 ? extent : 1) }

        let topology = EdgeTopology(geometry)
        positionOf = topology.positionID
        triangles = geometry.indices
        if !geometry.groups.isEmpty {
            triangleGroups = Array(repeating: -1, count: geometry.indices.count / 3)
            for (slot, group) in geometry.groups.enumerated() {
                for face in (group.start / 3)..<(group.range.upperBound / 3) {
                    triangleGroups[face] = slot
                }
            }
        }

        // Attribute vectors: UV0, UV1 when every vertex has one, normals, and linear RGBA.
        let uvWeight = Double(options.textureCoordinateWeight)
        let normalWeight = Double(options.normalWeight)
        let colorWeight = Double(options.colorWeight)
        let hasColor = geometry.hasVertexColors && colorWeight > 0
        let hasSecondary = geometry.vertices.allSatisfy { $0.secondaryTextureCoordinate != nil }
        var count = 0
        if uvWeight > 0 { count += hasSecondary ? 4 : 2 }
        if normalWeight > 0 { count += 3 }
        if hasColor { count += 4 }
        attributeCount = count
        hasSecondaryAttributes = hasSecondary && uvWeight > 0
        hasColorAttributes = hasColor
        var values: [Double] = []
        values.reserveCapacity(vertexCount * count)
        for vertex in geometry.vertices {
            if uvWeight > 0 {
                values.append(Double(vertex.textureCoordinate.x) * uvWeight)
                values.append(Double(vertex.textureCoordinate.y) * uvWeight)
                if hasSecondary, let secondary = vertex.secondaryTextureCoordinate {
                    values.append(Double(secondary.x) * uvWeight)
                    values.append(Double(secondary.y) * uvWeight)
                }
            }
            if normalWeight > 0 {
                let n = simd_normalize(SIMD3<Double>(vertex.normal))
                values += [n.x * normalWeight, n.y * normalWeight, n.z * normalWeight]
            }
            if hasColor {
                let color = SIMD4<Double>(vertex.color ?? SIMD4(repeating: 1)) * colorWeight
                values += [color.x, color.y, color.z, color.w]
            }
        }
        attributes = values
        textureCoordinates = geometry.vertices.map { SIMD2(Double($0.textureCoordinate.x), Double($0.textureCoordinate.y)) }

        positionQuadrics = Array(repeating: Quadric(), count: vertexCount)
        attributeQuadrics = Array(repeating: Quadric(), count: vertexCount)
        gradients = Array(repeating: 0, count: vertexCount * count * 4)
        hardLocked = Array(repeating: false, count: vertexCount)

        // Edge quadrics hold boundaries and seams in shape. Their weight is relative to the
        // area weight of the triangle planes, scaled by the squared edge length.
        let edgeWeight = 4.0
        for corner in 0..<triangles.count {
            let neighbor = topology.neighbor[corner]
            let next = EdgeTopology.next(corner)
            var constrained = neighbor < 0
            let face = topology.faceNormals[corner / 3]
            let crossesGroup = !triangleGroups.isEmpty && neighbor >= 0
                && triangleGroups[corner / 3] != triangleGroups[neighbor / 3]
            if neighbor == EdgeTopology.nonManifold || face == .zero || crossesGroup
                || (neighbor >= 0 && !topology.isConsistentManifold(corner)) {
                hardLocked[Int(topology.position(corner))] = true
                hardLocked[Int(topology.position(next))] = true
                constrained = true
            } else if neighbor >= 0 {
                constrained = triangles[corner] != triangles[EdgeTopology.next(neighbor)]
                    || triangles[next] != triangles[neighbor]
            }
            guard constrained else { continue }
            guard face != .zero else { continue }
            let a = positions[Int(triangles[corner])], b = positions[Int(triangles[next])]
            let edge = b - a
            let lengthSquared = simd_length_squared(edge)
            guard lengthSquared > 0 else { continue }
            let normal = simd_normalize(simd_cross(edge, face))
            var quadric = Quadric()
            quadric.addSquare(normal, -simd_dot(normal, a), weight: lengthSquared * edgeWeight)
            // Edge quadrics add no area weight, so they do not dilute the per-area error.
            positionQuadrics[Int(positionOf[Int(triangles[corner])])].add(quadric)
            positionQuadrics[Int(positionOf[Int(triangles[next])])].add(quadric)
        }

        for start in stride(from: 0, to: triangles.count, by: 3) {
            let w0 = Int(triangles[start]), w1 = Int(triangles[start + 1]), w2 = Int(triangles[start + 2])
            let p0 = positions[w0], p1 = positions[w1], p2 = positions[w2]
            let e1 = p1 - p0, e2 = p2 - p0
            let cross = simd_cross(e1, e2)
            let doubleArea = simd_length(cross)
            guard doubleArea > 0, doubleArea.isFinite else { continue }
            let area = doubleArea * 0.5
            let normal = cross / doubleArea
            var plane = Quadric()
            plane.addSquare(normal, -simd_dot(normal, p0), weight: area)
            plane.w = area
            for wedge in [w0, w1, w2] { positionQuadrics[Int(positionOf[wedge])].add(plane) }

            guard count > 0 else { continue }
            // Each attribute is linear over the triangle: s(p) = g·p + d, with g in its plane.
            let lengthSquared = doubleArea * doubleArea
            let towardsE1 = simd_cross(e2, cross) / lengthSquared
            let towardsE2 = simd_cross(cross, e1) / lengthSquared
            var quadric = Quadric()
            quadric.w = area
            var linear: [(SIMD3<Double>, Double)] = []
            linear.reserveCapacity(count)
            for j in 0..<count {
                let s0 = attributes[w0 * count + j], s1 = attributes[w1 * count + j], s2 = attributes[w2 * count + j]
                let g = towardsE1 * (s1 - s0) + towardsE2 * (s2 - s0)
                let d = s0 - simd_dot(g, p0)
                quadric.addSquare(g, d, weight: area)
                linear.append((g, d))
            }
            for wedge in [w0, w1, w2] {
                attributeQuadrics[wedge].add(quadric)
                for j in 0..<count {
                    let base = (wedge * count + j) * 4
                    let (g, d) = linear[j]
                    gradients[base] += area * g.x
                    gradients[base + 1] += area * g.y
                    gradients[base + 2] += area * g.z
                    gradients[base + 3] += area * d
                }
            }
        }
    }

    var triangleCount: Int { triangles.count / 3 }

    mutating func simplify(toTriangleCount target: Int) {
        let limit = options.maximumError.isInfinite ? Double.infinity : pow(Double(options.maximumError) / scale, 2)
        while triangleCount > target {
            if performPass(target: target, errorLimit: limit) == 0 { break }
        }
    }

    func result() throws -> MeshGeometry.Simplification {
        guard triangles.count < source.indices.count else {
            return MeshGeometry.Simplification(geometry: source, error: 0, sourceVertexIndices: Array(source.vertices.indices))
        }
        var newIndex = [UInt32](repeating: .max, count: source.vertices.count)
        var sources: [Int] = []
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        indices.reserveCapacity(triangles.count)
        for wedge in triangles {
            if newIndex[Int(wedge)] == .max {
                newIndex[Int(wedge)] = UInt32(vertices.count)
                vertices.append(moved.contains(wedge) ? movedVertex(wedge) : source.vertices[Int(wedge)])
                sources.append(Int(wedge))
            }
            indices.append(newIndex[Int(wedge)])
        }
        guard !indices.isEmpty else { throw MeshGeometry.GeometryProcessingError.noTrianglesRemain }
        var groups: [MeshGeometryGroup] = []
        if !triangleGroups.isEmpty {
            var start = 0
            for end in 1...triangleGroups.count where end == triangleGroups.count || triangleGroups[end] != triangleGroups[start] {
                let slot = triangleGroups[start]
                if slot >= 0 {
                    groups.append(MeshGeometryGroup(
                        start: start * 3, count: (end - start) * 3, materialIndex: source.groups[slot].materialIndex
                    ))
                }
                start = end
            }
        }
        return MeshGeometry.Simplification(
            geometry: try MeshGeometry(vertices: vertices, indices: indices).withGroups(groups),
            // Squared errors below 10⁻¹² of the extent are rounding and count as zero.
            error: error > 1e-12 ? Float(error.squareRoot() * scale) : 0, sourceVertexIndices: sources
        )
    }

    /// A moved wedge's vertex from its updated position and weighted attributes.
    private func movedVertex(_ wedge: UInt32) -> MeshVertex {
        let original = source.vertices[Int(wedge)]
        let position = SIMD3<Float>(positions[Int(positionOf[Int(wedge)])] * scale + center)
        let count = attributeCount
        let base = Int(wedge) * count
        var index = 0
        var uv = original.textureCoordinate
        var secondary = original.secondaryTextureCoordinate
        var normal = original.normal
        let uvWeight = Double(options.textureCoordinateWeight)
        if uvWeight > 0 {
            uv = SIMD2(Float(attributes[base] / uvWeight), Float(attributes[base + 1] / uvWeight))
            index = 2
            if hasSecondaryAttributes {
                secondary = SIMD2(Float(attributes[base + 2] / uvWeight), Float(attributes[base + 3] / uvWeight))
                index = 4
            }
        }
        if options.normalWeight > 0 {
            let n = SIMD3(attributes[base + index], attributes[base + index + 1], attributes[base + index + 2])
            if simd_length_squared(n) > 0 { normal = SIMD3<Float>(simd_normalize(n)) }
            index += 3
        }
        var color = original.color
        if hasColorAttributes {
            let weight = Double(options.colorWeight)
            color = SIMD4(
                Float(attributes[base + index] / weight), Float(attributes[base + index + 1] / weight),
                Float(attributes[base + index + 2] / weight), Float(attributes[base + index + 3] / weight)
            )
        }
        var tangent: SIMD4<Float>?
        if let t = original.tangent {
            let direction = SIMD3(t.x, t.y, t.z)
            let unitNormal = SIMD3<Float>(simd_normalize(SIMD3<Double>(normal)))
            var orthogonal = direction - unitNormal * simd_dot(unitNormal, direction)
            if simd_length_squared(orthogonal) <= 1e-8 {
                let axis: SIMD3<Float> = abs(unitNormal.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
                orthogonal = simd_cross(unitNormal, axis)
            }
            tangent = SIMD4(simd_normalize(orthogonal), t.w)
        }
        return MeshVertex(
            position: position, normal: normal, textureCoordinate: uv, secondaryTextureCoordinate: secondary,
            tangent: tangent, color: color
        )
    }

    private struct Candidate {
        var cost: Double
        var from: UInt32
        var to: UInt32
        /// The optimal position, or nil to keep `to` where it is.
        var placement: SIMD3<Double>?
    }

    private enum Kind: UInt8 {
        case interior, boundary, locked
    }

    /// One pass. Returns the number of collapses applied.
    private mutating func performPass(target: Int, errorLimit: Double) -> Int {
        let vertexCount = positions.count
        // Incident triangles per position.
        var offsets = [Int32](repeating: 0, count: vertexCount + 1)
        for wedge in triangles { offsets[Int(positionOf[Int(wedge)]) + 1] += 1 }
        for index in 0..<vertexCount { offsets[index + 1] += offsets[index] }
        var fill = offsets
        var incident = [Int32](repeating: 0, count: triangles.count)
        for corner in 0..<triangles.count {
            let position = Int(positionOf[Int(triangles[corner])])
            incident[Int(fill[position])] = Int32(corner / 3)
            fill[position] += 1
        }

        // Classify positions and list edges once each.
        var kind = [Kind](repeating: .interior, count: vertexCount)
        // The one wedge every corner at a position uses, or UInt32.max for several.
        var soleWedge = [UInt32](repeating: .max, count: vertexCount)
        var edges: [(UInt32, UInt32, Bool)] = []
        edges.reserveCapacity(triangles.count / 2 + 16)
        var neighbors: [(UInt32, Int32)] = []
        for u in 0..<vertexCount where offsets[u + 1] > offsets[u] {
            neighbors.removeAll(keepingCapacity: true)
            var wedge = UInt32.max - 1
            for slot in Int(offsets[u])..<Int(offsets[u + 1]) {
                let t = Int(incident[slot])
                for k in 0..<3 {
                    let x = positionOf[Int(triangles[3 * t + k])]
                    if x == UInt32(u) {
                        let w = triangles[3 * t + k]
                        wedge = wedge == UInt32.max - 1 || wedge == w ? w : .max
                    }
                    guard x != UInt32(u) else { continue }
                    if let found = neighbors.firstIndex(where: { $0.0 == x }) {
                        neighbors[found].1 += 1
                    } else {
                        neighbors.append((x, 1))
                    }
                }
            }
            var boundaryEdges = 0
            var nonManifold = false
            for (x, count) in neighbors {
                if count == 1 { boundaryEdges += 1 }
                if count > 2 { nonManifold = true }
                if UInt32(u) < x { edges.append((UInt32(u), x, count == 1)) }
            }
            soleWedge[u] = wedge
            if nonManifold || hardLocked[u] || !(boundaryEdges == 0 || boundaryEdges == 2) {
                kind[u] = .locked
            } else if boundaryEdges == 2 {
                kind[u] = options.locksBoundary ? .locked : .boundary
            }
        }

        var candidates: [Candidate] = []
        candidates.reserveCapacity(edges.count)
        var mapping: [(UInt32, UInt32)] = []
        for (a, b, isBoundary) in edges {
            var best: Candidate?
            for reversed in [false, true] {
                let u = reversed ? b : a, v = reversed ? a : b
                guard kind[Int(u)] != .locked else { continue }
                if kind[Int(u)] == .boundary, !isBoundary { continue }
                guard wedgeMapping(from: u, to: v, offsets: offsets, incident: incident, into: &mapping) else { continue }
                let cost = collapseCost(from: u, to: v, mapping: mapping)
                if cost.isFinite, best.map({ cost < $0.cost }) ?? true { best = Candidate(cost: cost, from: u, to: v) }
            }
            if options.vertexPlacement == .optimal, kind[Int(a)] == .interior, kind[Int(b)] == .interior,
               soleWedge[Int(a)] < UInt32.max - 1, soleWedge[Int(b)] < UInt32.max - 1,
               let optimal = optimalPlacement(from: a, to: b, wedges: (soleWedge[Int(a)], soleWedge[Int(b)])),
               best.map({ optimal.cost < $0.cost }) ?? true {
                best = Candidate(cost: optimal.cost, from: a, to: b, placement: optimal.point)
            }
            if let best { candidates.append(best) }
        }
        guard !candidates.isEmpty else { return 0 }
        candidates.sort { $0.cost < $1.cost || ($0.cost == $1.cost && ($0.from, $0.to) < ($1.from, $1.to)) }

        // Each collapse removes about two triangles. Accept up to 1.5 times the cost of the
        // collapse that would reach the goal, since locking skips some cheaper ones.
        let needed = triangleCount - target
        let goalIndex = max(0, needed / 2)
        let passLimit = min(errorLimit, goalIndex < candidates.count ? 1.5 * candidates[goalIndex].cost : .infinity)

        var wedgeRemap = [UInt32](0..<UInt32(vertexCount))
        var locked = [Bool](repeating: false, count: vertexCount)
        var mark = [UInt32](repeating: 0, count: vertexCount)
        var token: UInt32 = 0
        var removed = 0
        var applied = 0
        var groupCounts = Array(repeating: 0, count: source.groups.count)
        for slot in triangleGroups where slot >= 0 { groupCounts[slot] += 1 }
        for candidate in candidates {
            if candidate.cost > passLimit || removed >= needed { break }
            let u = candidate.from, v = candidate.to
            if locked[Int(u)] || locked[Int(v)] { continue }
            guard wedgeMapping(from: u, to: v, offsets: offsets, incident: incident, into: &mapping) else { continue }

            // Link condition on the current mesh: the common neighbors of u and v must be
            // exactly the apexes of the triangles on edge uv. Marks: token for neighbors of
            // u, token + 1 for apexes, token + 2 once counted as a neighbor of v.
            func current(_ t: Int) -> (UInt32, UInt32, UInt32)? {
                let a = positionOf[Int(wedgeRemap[Int(triangles[3 * t])])]
                let b = positionOf[Int(wedgeRemap[Int(triangles[3 * t + 1])])]
                let c = positionOf[Int(wedgeRemap[Int(triangles[3 * t + 2])])]
                return a == b || b == c || a == c ? nil : (a, b, c)
            }
            if token >= UInt32.max - 3 {
                mark = [UInt32](repeating: 0, count: vertexCount)
                token = 0
            }
            token += 3
            var apexCount = 0
            var apexes = SIMD2<UInt32>(repeating: .max)
            var removedFaces = 0
            var hasSurvivingFace = false
            var groupsRemoved: [Int] = []
            for slot in Int(offsets[Int(u)])..<Int(offsets[Int(u) + 1]) {
                let face = Int(incident[slot])
                guard let (a, b, c) = current(face) else { continue }
                let onEdge = a == v || b == v || c == v
                if onEdge {
                    removedFaces += 1
                    if !triangleGroups.isEmpty, triangleGroups[face] >= 0 {
                        groupsRemoved.append(triangleGroups[face])
                    }
                } else {
                    hasSurvivingFace = true
                }
                for x in [a, b, c] where x != u && x != v {
                    if onEdge {
                        if mark[Int(x)] != token + 1 {
                            if apexCount < 2 { apexes[apexCount] = x }
                            apexCount += 1
                            mark[Int(x)] = token + 1
                        }
                    } else if mark[Int(x)] != token + 1 {
                        mark[Int(x)] = token
                    }
                }
            }
            var commonCount = 0
            var vHasApexFace = false
            for slot in Int(offsets[Int(v)])..<Int(offsets[Int(v) + 1]) {
                guard let (a, b, c) = current(Int(incident[slot])) else { continue }
                if a != u, b != u, c != u {
                    hasSurvivingFace = true
                    if apexCount == 2 {
                        vHasApexFace = vHasApexFace || Self.contains(apexes, in: (a, b, c))
                    }
                }
                for x in [a, b, c] where x != u && x != v && (mark[Int(x)] == token || mark[Int(x)] == token + 1) {
                    commonCount += 1
                    mark[Int(x)] = token + 2
                }
            }
            if !hasSurvivingFace || apexCount == 0 || removedFaces != apexCount || commonCount != apexCount {
                continue
            }
            if groupsRemoved.contains(where: { group in groupCounts[group] <= groupsRemoved.count(where: { $0 == group }) }) {
                continue
            }
            // Common link edges also forbid a collapse. Without this check a tetrahedron
            // could leave two coincident triangles after one edge collapses.
            if vHasApexFace, (Int(offsets[Int(u)])..<Int(offsets[Int(u) + 1])).contains(where: { slot in
                guard let face = current(Int(incident[slot])) else { return false }
                return Self.contains(apexes, in: face)
            }) {
                continue
            }

            // Reject folds: every surviving triangle that moves keeps its orientation within 75°.
            let destination = candidate.placement ?? positions[Int(v)]
            var folds = false
            for (moving, other) in candidate.placement == nil ? [(u, v)] : [(u, v), (v, u)] where !folds {
                let origin = positions[Int(moving)]
                for slot in Int(offsets[Int(moving)])..<Int(offsets[Int(moving) + 1]) {
                    guard let (a, b, c) = current(Int(incident[slot])), a != other, b != other, c != other else { continue }
                    let (next, last) = a == moving ? (b, c) : (b == moving ? (c, a) : (a, b))
                    let pb = positions[Int(next)], pc = positions[Int(last)]
                    let before = simd_cross(pb - origin, pc - origin)
                    let after = simd_cross(pb - destination, pc - destination)
                    if !(simd_dot(before, after) > 0.25 * (simd_length_squared(before) * simd_length_squared(after)).squareRoot()) {
                        folds = true
                        break
                    }
                }
            }
            if folds { continue }

            // Least-squares attributes at an optimal position, from the unmerged quadrics.
            var placedAttributes: [Double] = []
            var placedCoordinate: SIMD2<Double>?
            if let placement = candidate.placement, let (wu, wv) = mapping.first {
                let count = attributeCount
                let weight = attributeQuadrics[Int(wu)].w + attributeQuadrics[Int(wv)].w
                if count > 0, weight > 0 {
                    for j in 0..<count {
                        let a = (Int(wu) * count + j) * 4, b = (Int(wv) * count + j) * 4
                        let g = SIMD3(gradients[a] + gradients[b], gradients[a + 1] + gradients[b + 1], gradients[a + 2] + gradients[b + 2])
                        placedAttributes.append((simd_dot(g, placement) + gradients[a + 3] + gradients[b + 3]) / weight)
                    }
                    guard placementAttributesAreRepresentable(placedAttributes) else { continue }
                    if options.textureCoordinateWeight > 0 {
                        let uvWeight = Double(options.textureCoordinateWeight)
                        let stored = SIMD2(Float(placedAttributes[0] / uvWeight), Float(placedAttributes[1] / uvWeight))
                        placedCoordinate = SIMD2<Double>(stored)
                    }
                }
            }

            // Reject texture folds: no surviving triangle may flip or flatten its primary UV
            // orientation, whatever the attribute weights.
            var textureFolds = false
            for (moving, other) in candidate.placement == nil ? [(u, v)] : [(u, v), (v, u)] where !textureFolds {
                for slot in Int(offsets[Int(moving)])..<Int(offsets[Int(moving) + 1]) {
                    let t = Int(incident[slot])
                    let w = SIMD3(wedgeRemap[Int(triangles[3 * t])], wedgeRemap[Int(triangles[3 * t + 1])], wedgeRemap[Int(triangles[3 * t + 2])])
                    let p = SIMD3(positionOf[Int(w.x)], positionOf[Int(w.y)], positionOf[Int(w.z)])
                    if p.x == p.y || p.y == p.z || p.x == p.z || p.x == other || p.y == other || p.z == other { continue }
                    let k = p.x == moving ? 0 : (p.y == moving ? 1 : 2)
                    var corners = (textureCoordinates[Int(w.x)], textureCoordinates[Int(w.y)], textureCoordinates[Int(w.z)])
                    let before = Self.signedArea(corners.0, corners.1, corners.2)
                    guard before != 0 else { continue }
                    let replacement: SIMD2<Double>
                    if moving == u {
                        guard let target = mapping.first(where: { $0.0 == w[k] })?.1 else { continue }
                        replacement = placedCoordinate ?? textureCoordinates[Int(target)]
                    } else {
                        guard let coordinate = placedCoordinate else { continue }
                        replacement = coordinate
                    }
                    switch k {
                    case 0: corners.0 = replacement
                    case 1: corners.1 = replacement
                    default: corners.2 = replacement
                    }
                    if !(Self.signedArea(corners.0, corners.1, corners.2) * before > 0) {
                        textureFolds = true
                        break
                    }
                }
            }
            if textureFolds { continue }

            if let placement = candidate.placement, let (_, wv) = mapping.first {
                let count = attributeCount
                for (j, value) in placedAttributes.enumerated() { attributes[Int(wv) * count + j] = value }
                if let coordinate = placedCoordinate { textureCoordinates[Int(wv)] = coordinate }
                positions[Int(v)] = placement
                moved.insert(wv)
            }
            for (from, to) in mapping { wedgeRemap[Int(from)] = to }
            positionQuadrics[Int(v)].add(positionQuadrics[Int(u)])
            for (from, to) in mapping {
                attributeQuadrics[Int(to)].add(attributeQuadrics[Int(from)])
                let count = attributeCount * 4
                for j in 0..<count { gradients[Int(to) * count + j] += gradients[Int(from) * count + j] }
            }
            locked[Int(u)] = true
            locked[Int(v)] = true
            for group in groupsRemoved { groupCounts[group] -= 1 }
            removed += removedFaces
            error = max(error, candidate.cost)
            applied += 1
        }
        guard applied > 0 else { return 0 }

        var kept: [UInt32] = []
        kept.reserveCapacity(triangles.count)
        var keptGroups: [Int] = []
        keptGroups.reserveCapacity(triangleGroups.count)
        for start in stride(from: 0, to: triangles.count, by: 3) {
            let a = wedgeRemap[Int(triangles[start])], b = wedgeRemap[Int(triangles[start + 1])], c = wedgeRemap[Int(triangles[start + 2])]
            let pa = positionOf[Int(a)], pb = positionOf[Int(b)], pc = positionOf[Int(c)]
            if pa == pb || pb == pc || pa == pc {
                let original = SIMD3(
                    positionOf[Int(triangles[start])], positionOf[Int(triangles[start + 1])], positionOf[Int(triangles[start + 2])]
                )
                // The simplifier leaves source degeneracies in place. Only collapsed faces disappear.
                if original.x != original.y, original.y != original.z, original.x != original.z { continue }
            }
            kept += [a, b, c]
            if !triangleGroups.isEmpty { keptGroups.append(triangleGroups[start / 3]) }
        }
        triangles = kept
        triangleGroups = keptGroups
        return applied
    }

    /// The wedge each wedge at `u` becomes when `u` collapses onto `v`, from the triangles
    /// on edge uv. False when a wedge at u has no single counterpart: the collapse would
    /// tear or blend an attribute seam.
    private func wedgeMapping(
        from u: UInt32, to v: UInt32, offsets: [Int32], incident: [Int32], into mapping: inout [(UInt32, UInt32)]
    ) -> Bool {
        mapping.removeAll(keepingCapacity: true)
        let range = Int(offsets[Int(u)])..<Int(offsets[Int(u) + 1])
        for slot in range {
            let t = Int(incident[slot])
            var wu = UInt32.max, wv = UInt32.max
            for k in 0..<3 {
                let wedge = triangles[3 * t + k]
                let position = positionOf[Int(wedge)]
                if position == u { wu = wedge } else if position == v { wv = wedge }
            }
            guard wu != .max, wv != .max else { continue }
            if let existing = mapping.first(where: { $0.0 == wu }) {
                if existing.1 != wv { return false }
            } else {
                guard !mapping.contains(where: { $0.1 == wv }) else { return false }
                mapping.append((wu, wv))
            }
        }
        guard !mapping.isEmpty else { return false }
        for slot in range {
            let t = Int(incident[slot])
            for k in 0..<3 {
                let wedge = triangles[3 * t + k]
                if positionOf[Int(wedge)] == u, !mapping.contains(where: { $0.0 == wedge }) { return false }
            }
        }
        return true
    }

    private static func contains(_ apexes: SIMD2<UInt32>, in triangle: (UInt32, UInt32, UInt32)) -> Bool {
        let (a, b, c) = triangle
        return (a == apexes.x || b == apexes.x || c == apexes.x)
            && (a == apexes.y || b == apexes.y || c == apexes.y)
    }

    private func placementAttributesAreRepresentable(_ values: [Double]) -> Bool {
        var index = 0
        for (count, weight) in [
            (hasSecondaryAttributes ? 4 : 2, options.textureCoordinateWeight),
            (3, options.normalWeight),
            (hasColorAttributes ? 4 : 0, options.colorWeight),
        ] where weight > 0 {
            let end = index + count
            guard values[index..<end].allSatisfy({ Float($0 / Double(weight)).isFinite }) else { return false }
            index = end
        }
        return true
    }

    static func signedArea(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>) -> Double {
        let e1 = b - a, e2 = c - a
        return e1.x * e2.y - e1.y * e2.x
    }

    /// The point minimizing the merged position and attribute quadrics of interior
    /// single-wedge vertices u and v, with attributes eliminated (Hoppe 1999), and its cost.
    /// Nil when the system is singular or the minimum lies more than an edge length from
    /// the edge's midpoint.
    private func optimalPlacement(
        from u: UInt32, to v: UInt32, wedges: (UInt32, UInt32)
    ) -> (point: SIMD3<Double>, cost: Double)? {
        var q = positionQuadrics[Int(u)] + positionQuadrics[Int(v)]
        let positionWeight = q.w
        let count = attributeCount
        if count > 0 {
            let attribute = attributeQuadrics[Int(wedges.0)] + attributeQuadrics[Int(wedges.1)]
            let weight = attribute.w
            q.add(attribute)
            if weight > 0 {
                for j in 0..<count {
                    let a = (Int(wedges.0) * count + j) * 4, b = (Int(wedges.1) * count + j) * 4
                    let g = SIMD3(gradients[a] + gradients[b], gradients[a + 1] + gradients[b + 1], gradients[a + 2] + gradients[b + 2])
                    let d = gradients[a + 3] + gradients[b + 3]
                    // Minimizing over the attribute s = (g·p + d) / W subtracts (g·p + d)² / W.
                    q.addSquare(g, d, weight: -1 / weight)
                }
            }
        }
        let matrix = simd_double3x3(
            SIMD3(q.a00, q.a01, q.a02), SIMD3(q.a01, q.a11, q.a12), SIMD3(q.a02, q.a12, q.a22)
        )
        let determinant = matrix.determinant
        let trace = q.a00 + q.a11 + q.a22
        guard trace > 0, abs(determinant) > 1e-9 * trace * trace * trace else { return nil }
        let point = matrix.inverse * SIMD3(-q.b0, -q.b1, -q.b2)
        let pu = positions[Int(u)], pv = positions[Int(v)]
        guard point.x.isFinite, point.y.isFinite, point.z.isFinite,
              simd_distance(point, (pu + pv) * 0.5) <= simd_distance(pu, pv) else { return nil }
        let worldPoint = SIMD3<Float>(point * scale + center)
        guard worldPoint.x.isFinite, worldPoint.y.isFinite, worldPoint.z.isFinite else { return nil }
        let storedPoint = (SIMD3<Double>(worldPoint) - center) / scale
        let value = abs(q.evaluate(storedPoint))
        let cost = positionWeight > 0 ? value / positionWeight : value
        return cost.isFinite ? (storedPoint, cost) : nil
    }

    /// Position and attribute error of moving u onto v, per unit of accumulated area.
    private func collapseCost(from u: UInt32, to v: UInt32, mapping: [(UInt32, UInt32)]) -> Double {
        let p = positions[Int(v)]
        let positionQuadric = positionQuadrics[Int(u)] + positionQuadrics[Int(v)]
        var total = abs(positionQuadric.evaluate(p))
        let count = attributeCount
        if count > 0 {
            for (from, to) in mapping {
                let quadric = attributeQuadrics[Int(from)] + attributeQuadrics[Int(to)]
                var value = quadric.evaluate(p)
                for j in 0..<count {
                    let s = attributes[Int(to) * count + j]
                    let a = (Int(from) * count + j) * 4, b = (Int(to) * count + j) * 4
                    let gx = gradients[a] + gradients[b], gy = gradients[a + 1] + gradients[b + 1]
                    let gz = gradients[a + 2] + gradients[b + 2], gd = gradients[a + 3] + gradients[b + 3]
                    value += -2 * s * (gx * p.x + gy * p.y + gz * p.z + gd) + s * s * quadric.w
                }
                total += abs(value)
            }
        }
        let weight = positionQuadric.w
        return weight > 0 ? total / weight : total
    }
}
