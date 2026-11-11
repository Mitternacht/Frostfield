import Foundation
import simd

/// A reusable distance and frame table for bending indexed geometry along a 3D curve.
/// Mesh +X follows the spine. +Y starts along the projected up direction and +Z
/// completes a right-handed frame. Distances and frames are sampled approximations.
public struct CurveDeformer: Sendable {
    public enum Boundary: Equatable, Sendable {
        /// Extends in a straight line along each endpoint tangent.
        case open
        /// Wraps signed distances around a loop and joins the frames at its seam.
        case closed
    }

    /// A copy of the source mesh at a distance on this curve.
    public struct Placement: Sendable {
        public let curve: CurveDeformer
        public var offset: Double

        public init(curve: CurveDeformer, offset: Double) {
            self.curve = curve
            self.offset = offset
        }
    }

    public static let maximumSubdivisions = 4_096
    public static let maximumPlacementCount = 4_096
    public static let maximumVertexCount = 1_048_576
    public static let maximumIndexCount = 6_291_456

    public let boundary: Boundary
    /// The sum of the sampled chord lengths, in curve coordinate units.
    public let length: Double

    let frameTableID = UUID()
    private let knots: [Knot]
    private let turns: [Turn]

    /// Prepares parallel-transported frames. Closed curves must return to their
    /// starting point. A parallel initial up direction uses the least-aligned
    /// coordinate axis instead. Use more subdivisions for tighter bends.
    public init(
        curve: Curve3D, subdivisions: Int = 512, boundary: Boundary = .open,
        up: SIMD3<Float> = SIMD3(0, 1, 0)
    ) throws {
        let minimum = boundary == .closed ? 3 : 1
        guard (minimum...Self.maximumSubdivisions).contains(subdivisions) else { throw Failure.resolution }
        guard let up = Self.unit(SIMD3<Double>(up)) else { throw Failure.upDirection }
        var centers = try curve.points(segments: subdivisions).map(SIMD3<Double>.init)
        if boundary == .closed {
            let roughLength = zip(centers, centers.dropFirst()).reduce(0.0) {
                $0 + simd_distance($1.0, $1.1)
            }
            guard simd_distance(centers[0], centers[subdivisions]) <= roughLength * 1e-6 else {
                throw Failure.openLoop
            }
            centers[subdivisions] = centers[0]
        }

        var distances = [0.0]
        var edges: [SIMD3<Double>] = []
        for index in 0..<subdivisions {
            let difference = centers[index + 1] - centers[index]
            let span = simd_length(difference)
            let distance = distances[index] + span
            guard distance > distances[index] else { throw Failure.coincidentSample(index + 1) }
            distances.append(distance)
            edges.append(difference / span)
        }
        let length = distances[subdivisions]

        func bisector(_ a: SIMD3<Double>, _ b: SIMD3<Double>, at index: Int) throws -> SIMD3<Double> {
            let sum = a + b
            guard simd_length_squared(sum) > 1e-12 else { throw Failure.reversingSample(index) }
            return simd_normalize(sum)
        }
        let firstTangent = boundary == .closed
            ? try bisector(edges[subdivisions - 1], edges[0], at: 0)
            : edges[0]
        var tangents = [firstTangent]
        for index in 1..<subdivisions {
            tangents.append(try bisector(edges[index - 1], edges[index], at: index))
        }
        tangents.append(boundary == .closed ? firstTangent : edges[subdivisions - 1])

        let projectedUp = up - firstTangent * simd_dot(up, firstTangent)
        let firstReference = simd_length_squared(projectedUp) > 1e-12
            ? simd_normalize(projectedUp) : sweepInitialReference(tangent: firstTangent)
        var references = [firstReference]
        for index in 1...subdivisions {
            guard let next = sweepTransportReference(references[index - 1], from: tangents[index - 1], to: tangents[index]) else {
                throw Failure.reversingSample(index)
            }
            references.append(next)
        }

        if boundary == .closed {
            let lastReference = references[subdivisions]
            let seamTwist = atan2(
                simd_dot(firstTangent, simd_cross(lastReference, firstReference)),
                simd_dot(lastReference, firstReference)
            )
            for index in 1..<subdivisions {
                let roll = simd_quatd(angle: seamTwist * distances[index] / length, axis: tangents[index])
                references[index] = roll.act(references[index])
            }
            references[subdivisions] = firstReference
        }

        let knots = centers.indices.map { index in
            let tangent = tangents[index]
            let normal = references[index]
            let basis = simd_double3x3(columns: (tangent, normal, simd_cross(tangent, normal)))
            return Knot(center: centers[index], distance: distances[index], rotation: simd_quatd(basis))
        }
        self.knots = knots
        self.turns = zip(knots, knots.dropFirst()).map { Turn(from: $0.rotation, to: $1.rotation) }
        self.boundary = boundary
        self.length = length
    }

    /// Returns the rigid local-to-curve transform for a distance, for example
    /// as a transform in an `InstanceSet` when bending is unnecessary.
    public func transform(atDistance distance: Double) throws -> simd_float4x4 {
        guard distance.isFinite else { throw Failure.placement }
        let frame = sample(at: distance)
        let position = SIMD3<Float>(frame.center)
        guard boundsFinite(position) else { throw Failure.unrepresentablePosition }
        let basis = frame.basis
        return simd_float4x4(columns: (
            SIMD4(SIMD3<Float>(basis[0]), 0), SIMD4(SIMD3<Float>(basis[1]), 0),
            SIMD4(SIMD3<Float>(basis[2]), 0), SIMD4(position, 1)
        ))
    }

    /// Bends a mesh at `offset + (sourceX − sourceXOrigin) × distanceScale`.
    /// The scale must be positive. Indices, material groups, UVs and colors
    /// are preserved. The local deformation derivative transforms normals and
    /// tangents. Use a sufficiently tessellated mesh to follow tight bends.
    public func deform(
        _ mesh: MeshGeometry, offset: Double = 0, sourceXOrigin: Double = 0, distanceScale: Double = 1
    ) throws -> MeshGeometry {
        try Self.validateBudget(mesh, copies: 1)
        let vertices = try deformedVertices(mesh, offset: offset, sourceXOrigin: sourceXOrigin, distanceScale: distanceScale)
        return try MeshGeometry(vertices: vertices, indices: mesh.indices).withGroups(mesh.groups)
    }

    /// Bakes several independently placed copies into one indexed mesh and
    /// returns each copy's index range. Material group assignments carry over.
    public static func deform(
        _ mesh: MeshGeometry, placements: [Placement], sourceXOrigin: Double = 0, distanceScale: Double = 1
    ) throws -> MeshGeometry.Merged {
        try validateBudget(mesh, copies: placements.count)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        var groups: [MeshGeometryGroup] = []
        var ranges: [Range<Int>] = []
        vertices.reserveCapacity(mesh.vertices.count * placements.count)
        indices.reserveCapacity(mesh.indices.count * placements.count)
        for placement in placements {
            let firstIndex = indices.count
            let baseVertex = UInt32(vertices.count)
            vertices += try placement.curve.deformedVertices(
                mesh, offset: placement.offset, sourceXOrigin: sourceXOrigin, distanceScale: distanceScale
            )
            indices += mesh.indices.map { baseVertex + $0 }
            ranges.append(firstIndex..<indices.count)
            groups += mesh.groups.map {
                MeshGeometryGroup(start: firstIndex + $0.start, count: $0.count, materialIndex: $0.materialIndex)
            }
        }
        let geometry = try MeshGeometry(vertices: vertices, indices: indices).withGroups(groups)
        return MeshGeometry.Merged(geometry: geometry, indexRanges: ranges)
    }

    private func deformedVertices(
        _ mesh: MeshGeometry, offset: Double, sourceXOrigin: Double, distanceScale: Double
    ) throws -> [MeshVertex] {
        guard offset.isFinite, sourceXOrigin.isFinite, distanceScale.isFinite, distanceScale > 0 else {
            throw Failure.placement
        }
        let vertices = try mesh.vertices.enumerated().map { index, vertex in
            let source = SIMD3<Double>(vertex.position)
            let distance = offset + (source.x - sourceXOrigin) * distanceScale
            guard distance.isFinite else { throw Failure.unrepresentableVertex(index) }
            let frame = sample(at: distance)
            let basis = frame.basis
            let side = source.y * basis[1] + source.z * basis[2]
            let position = SIMD3<Float>(frame.center + side)
            guard boundsFinite(position) else { throw Failure.unrepresentableVertex(index) }

            // The longitudinal derivative includes frame rotation at the vertex's
            // cross-section. The cofactor matrix gives inverse-transpose normals
            // up to a positive scalar while avoiding division by the determinant.
            let along = distanceScale * (frame.centerDerivative + simd_cross(frame.angularVelocity, side))
            guard let alongDirection = Self.unit(along), simd_dot(alongDirection, basis[0]) > 1e-8 else {
                throw Failure.foldedVertex(index)
            }
            let originalNormal = SIMD3<Double>(vertex.normal)
            let cofactor = simd_double3x3(columns: (
                basis[0], simd_cross(basis[2], along), simd_cross(along, basis[1])
            ))
            guard let normal = Self.unit(cofactor * originalNormal) else { throw Failure.unrepresentableVertex(index) }
            let tangent: SIMD4<Float>?
            if let original = vertex.tangent {
                let direction = Double(original.x) * along + Double(original.y) * basis[1] + Double(original.z) * basis[2]
                guard let unit = Self.unit(direction),
                      let orthogonal = Self.unit(unit - normal * simd_dot(unit, normal)) else {
                    throw Failure.unrepresentableVertex(index)
                }
                tangent = SIMD4(SIMD3<Float>(orthogonal), original.w)
            } else {
                tangent = nil
            }
            return MeshVertex(
                position: position, normal: SIMD3<Float>(normal), textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.secondaryTextureCoordinate, tangent: tangent, color: vertex.color
            )
        }
        for start in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = Int(mesh.indices[start]), b = Int(mesh.indices[start + 1]), c = Int(mesh.indices[start + 2])
            if Self.faceNormal(vertices[a], vertices[b], vertices[c]) == .zero,
               Self.faceNormal(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]) != .zero {
                throw Failure.collapsedTriangle(start / 3)
            }
        }
        return vertices
    }

    private static func faceNormal(_ a: MeshVertex, _ b: MeshVertex, _ c: MeshVertex) -> SIMD3<Double> {
        let origin = SIMD3<Double>(a.position)
        return simd_cross(SIMD3<Double>(b.position) - origin, SIMD3<Double>(c.position) - origin)
    }

    private static func unit(_ vector: SIMD3<Double>) -> SIMD3<Double>? {
        guard vector.x.isFinite, vector.y.isFinite, vector.z.isFinite else { return nil }
        let scale = simd_reduce_max(simd_abs(vector))
        guard scale > 0 else { return nil }
        return simd_normalize(vector / scale)
    }

    private static func validateBudget(_ mesh: MeshGeometry, copies: Int) throws {
        guard (1...maximumPlacementCount).contains(copies),
              mesh.vertices.count <= maximumVertexCount / copies, mesh.indices.count <= maximumIndexCount / copies else {
            throw Failure.meshBudget
        }
    }

    /// The curve data shared by the CPU evaluator and the Metal vertex writer.
    /// Checks every interval because offsets can move vertices anywhere along the spine.
    func metalIntervals(crossSectionRadius: Double) throws -> [CurveFlowMetal.Interval] {
        guard length <= CurveFlowMetal.coordinateLimit else { throw CurveFlow.Failure.metalDomain }
        return try turns.indices.map { index in
            let a = knots[index], b = knots[index + 1], turn = turns[index]
            let span = b.distance - a.distance
            let delta = b.center - a.center
            let tangentA = a.tangent, tangentB = b.tangent
            let start = Float(a.distance), end = Float(b.distance)
            guard simd_reduce_max(simd_abs(a.center)) <= CurveFlowMetal.coordinateLimit,
                  simd_reduce_max(simd_abs(b.center)) <= CurveFlowMetal.coordinateLimit,
                  span >= 1e-8, abs(Double(end - start) - span) <= 0.01 * span else {
                throw CurveFlow.Failure.metalDomain
            }

            // The Hermite derivative is a quadratic Bézier with these three
            // control vectors. Bound its projection along the rotating frame
            // tangent, then subtract the largest contribution from frame rotation.
            let middle = 3 * (delta / span) - tangentA - tangentB
            let tangentSwing = 2 * sin(turn.angle * 0.5)
            let projection = min(1, simd_dot(tangentA, middle), simd_dot(tangentA, tangentB))
            let minimum = projection - tangentSwing * max(1, simd_length(middle))
                - (turn.angle / span) * crossSectionRadius
            guard minimum > 0.01 else { throw CurveFlow.Failure.metalDomain }

            return CurveFlowMetal.Interval(
                centerStart: SIMD4(SIMD3<Float>(a.center), start),
                deltaEnd: SIMD4(SIMD3<Float>(delta), end),
                tangentSpan: SIMD4(SIMD3<Float>(tangentA), Float(span)),
                nextTangent: SIMD4(SIMD3<Float>(tangentB), 0),
                rotation: SIMD4<Float>(a.rotation.vector),
                axisAngle: SIMD4(SIMD3<Float>(turn.axis), Float(turn.angle))
            )
        }
    }

    private struct Knot: Sendable {
        let center: SIMD3<Double>
        let distance: Double
        let rotation: simd_quatd
        var tangent: SIMD3<Double> { rotation.act(SIMD3(1, 0, 0)) }
    }

    private struct Turn: Sendable {
        let axis: SIMD3<Double>
        let angle: Double

        init(from first: simd_quatd, to next: simd_quatd) {
            let relative = simd_normalize(next * first.inverse)
            let imaginary = relative.real < 0 ? -relative.imag : relative.imag
            let magnitude = simd_length(imaginary)
            if magnitude > 1e-14 {
                axis = imaginary / magnitude
                angle = 2 * atan2(magnitude, abs(relative.real))
            } else {
                axis = SIMD3(1, 0, 0)
                angle = 0
            }
        }
    }

    private struct Frame {
        let center: SIMD3<Double>
        let basis: simd_double3x3
        let centerDerivative: SIMD3<Double>
        let angularVelocity: SIMD3<Double>
    }

    private func sample(at requested: Double) -> Frame {
        let distance: Double
        if boundary == .closed {
            let wrapped = requested.truncatingRemainder(dividingBy: length)
            distance = wrapped < 0 ? wrapped + length : wrapped
        } else if requested < 0 || requested > length {
            let knot = requested < 0 ? knots[0] : knots[knots.count - 1]
            let tangent = knot.tangent
            return Frame(
                center: knot.center + (requested - knot.distance) * tangent, basis: simd_double3x3(knot.rotation),
                centerDerivative: tangent, angularVelocity: .zero
            )
        } else {
            distance = requested
        }

        var low = 1
        var high = knots.count - 1
        while low < high {
            let middle = (low + high) / 2
            if knots[middle].distance < distance { low = middle + 1 } else { high = middle }
        }
        let a = knots[low - 1], b = knots[low]
        let span = b.distance - a.distance
        let t = (distance - a.distance) / span
        let t2 = t * t
        let delta = b.center - a.center
        let tangentA = a.tangent, tangentB = b.tangent
        // Cubic Hermite between centers, with unit endpoint derivatives along distance.
        let center = a.center + (3 * t2 - 2 * t2 * t) * delta
            + (t2 * t - 2 * t2 + t) * span * tangentA + (t2 * t - t2) * span * tangentB
        let derivative = (6 * t - 6 * t2) * (delta / span)
            + (3 * t2 - 4 * t + 1) * tangentA + (3 * t2 - 2 * t) * tangentB
        let turn = turns[low - 1]
        let rotation = simd_quatd(angle: turn.angle * t, axis: turn.axis) * a.rotation
        return Frame(
            center: center, basis: simd_double3x3(rotation), centerDerivative: derivative,
            angularVelocity: (turn.angle / span) * turn.axis
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case resolution
        case upDirection
        case openLoop
        case coincidentSample(Int)
        case reversingSample(Int)
        case placement
        case meshBudget
        case unrepresentablePosition
        case unrepresentableVertex(Int)
        case foldedVertex(Int)
        case collapsedTriangle(Int)

        public var errorDescription: String? {
            switch self {
            case .resolution: "Use up to 4,096 subdivisions, with at least one for open paths and three for loops."
            case .upDirection: "Use a finite, nonzero initial up direction."
            case .openLoop: "A closed deformation needs coincident curve endpoints."
            case .coincidentSample(let index): "Curve sample \(index) does not advance along the spine."
            case .reversingSample(let index): "Curve sample \(index) reverses direction too sharply to frame."
            case .placement: "Use finite placement values and a positive distance scale."
            case .meshBudget: "The placements, vertices or indices exceed the curve deformation budget."
            case .unrepresentablePosition: "The curve placement exceeds the Float coordinate range."
            case .unrepresentableVertex(let index): "Deformed vertex \(index) cannot be represented."
            case .foldedVertex(let index): "Deformed vertex \(index) crosses the inside of the bend."
            case .collapsedTriangle(let index): "Deformed triangle \(index) collapses in the output mesh."
            }
        }
    }
}
