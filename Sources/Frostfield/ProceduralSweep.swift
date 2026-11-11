import Foundation
import simd

extension MeshGeometry {
    /// A circular tube centered along an open 3D polyline. Rings use a
    /// parallel-transported frame with the tangent bisecting each interior turn.
    /// U wraps around the tube. V increases with normalized path length.
    /// Optional caps have flat normals and separate disk UVs.
    ///
    /// Supply 2 to 4,097 finite, distinct consecutive points and 3 to 128
    /// radial segments. Choose a radius that clears bends and nonadjacent
    /// segments. Locally folded or collapsed triangles are rejected.
    public static func tube(
        along path: [SIMD3<Float>], radius: Float, radialSegments: Int = 12, capped: Bool = true
    ) throws -> MeshGeometry {
        guard (2...4_097).contains(path.count) else { throw SweepError.invalidPathCount }
        guard radius.isNormal, radius > 0 else { throw SweepError.invalidRadius }
        guard (3...128).contains(radialSegments) else { throw SweepError.invalidRadialSegments }
        let ringStride = radialSegments + 1
        let vertexCount = (path.count + (capped ? 2 : 0)) * ringStride
        guard vertexCount <= 262_144 else { throw SweepError.vertexBudget }
        if let invalid = path.firstIndex(where: { !sweepFinite($0) }) {
            throw SweepError.invalidPoint(invalid)
        }

        let centers = path.map(SIMD3<Double>.init)
        var directions: [SIMD3<Double>] = []
        directions.reserveCapacity(path.count - 1)
        var lengths = [0.0]
        lengths.reserveCapacity(path.count)
        for index in 1..<centers.count {
            let edge = centers[index] - centers[index - 1]
            let length = simd_length(edge)
            guard length > 0 else { throw SweepError.coincidentPoints(index) }
            directions.append(edge / length)
            lengths.append(lengths[index - 1] + length)
        }

        var tangents = [directions[0]]
        tangents.reserveCapacity(path.count)
        for index in 1..<directions.count {
            let bisector = directions[index - 1] + directions[index]
            guard simd_length_squared(bisector) > 1e-12 else { throw SweepError.reversingPath(index) }
            tangents.append(simd_normalize(bisector))
        }
        tangents.append(directions[directions.count - 1])

        let circle = (0..<radialSegments).map { index in
            let angle = 2 * Double.pi * Double(index) / Double(radialSegments)
            return SIMD2(cos(angle), sin(angle))
        }
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(vertexCount)
        indices.reserveCapacity((path.count - 1 + (capped ? 1 : 0)) * radialSegments * 6)

        var reference = sweepInitialReference(tangent: tangents[0])
        for index in centers.indices {
            let tangent = tangents[index]
            if index > 0 {
                guard let transported = sweepTransportReference(reference, from: tangents[index - 1], to: tangent) else {
                    throw SweepError.reversingPath(index)
                }
                reference = transported
            }
            let binormal = simd_cross(tangent, reference)
            let v = Float(lengths[index] / lengths[lengths.count - 1])
            for column in 0...radialSegments {
                let xy = circle[column % radialSegments]
                let normal = reference * xy.x + binormal * xy.y
                let position = SIMD3<Float>(centers[index] + normal * Double(radius))
                guard sweepFinite(position) else { throw SweepError.unrepresentableSurface }
                let uTangent = binormal * xy.x - reference * xy.y
                vertices.append(MeshVertex(
                    position: position, normal: SIMD3<Float>(normal),
                    textureCoordinate: SIMD2(Float(column) / Float(radialSegments), v),
                    tangent: SIMD4(SIMD3<Float>(uTangent), 1)
                ))
            }
        }

        for row in 0..<(path.count - 1) {
            for column in 0..<radialSegments {
                let a = row * ringStride + column
                let b = a + ringStride
                try appendSweepTriangle(a, a + 1, b, vertices: vertices, indices: &indices)
                try appendSweepTriangle(a + 1, b + 1, b, vertices: vertices, indices: &indices)
            }
        }
        if capped {
            for end in [0, path.count - 1] {
                let isStart = end == 0
                let normal = SIMD3<Float>(tangents[end]) * (isStart ? -1 : 1)
                let sourceRing = end * ringStride
                let tangent = SIMD4(vertices[sourceRing].normal, isStart ? 1 : -1)
                let center = vertices.count
                vertices.append(MeshVertex(
                    position: path[end], normal: normal, textureCoordinate: SIMD2(repeating: 0.5), tangent: tangent
                ))
                for column in 0..<radialSegments {
                    let xy = circle[column]
                    vertices.append(MeshVertex(
                        position: vertices[sourceRing + column].position, normal: normal,
                        textureCoordinate: SIMD2(Float(0.5 + 0.5 * xy.x), Float(0.5 - 0.5 * xy.y)),
                        tangent: tangent
                    ))
                }
                for column in 0..<radialSegments {
                    let a = center + 1 + column
                    let b = center + 1 + (column + 1) % radialSegments
                    try appendSweepTriangle(center, isStart ? b : a, isStart ? a : b, vertices: vertices, indices: &indices)
                }
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// A circular tube along a spatial curve, sampled at approximate equal
    /// arc lengths with parallel-transported frames. U follows path distance,
    /// and top-left V wraps around each ring. Open tubes can have flat caps
    /// with independent disk UVs. Closed tubes duplicate the first ring at
    /// U = 1 and require matching path endpoints and tangents.
    ///
    /// Supply 1...4,096 tubular segments (at least three if closed), 3...128
    /// radial segments and 1...65,536 arc-length sampling intervals. The
    /// mesh is limited to 262,144 vertices. Choose a radius that clears bends
    /// and nonadjacent segments. Locally folded or collapsed faces are rejected.
    public static func tube(
        along path: Curve3D, radius: Float, tubularSegments: Int = 64, radialSegments: Int = 8,
        closed: Bool = false, capped: Bool = false, arcLengthSubdivisions: Int = 200
    ) throws -> MeshGeometry {
        guard ((closed ? 3 : 1)...4_096).contains(tubularSegments) else { throw CurveTubeError.invalidTubularSegments }
        guard radius.isNormal, radius > 0 else { throw SweepError.invalidRadius }
        guard (3...128).contains(radialSegments) else { throw SweepError.invalidRadialSegments }
        guard !closed || !capped else { throw CurveTubeError.closedWithCaps }
        let ringStride = radialSegments + 1
        let vertexCount = (tubularSegments + (capped ? 3 : 1)) * ringStride
        guard vertexCount <= 262_144 else { throw SweepError.vertexBudget }

        let sampled = try path.arcSamples(segments: tubularSegments, subdivisions: arcLengthSubdivisions)
        let frames: [CurveSweepFrame]
        do {
            frames = try CurveSweepFrame.make(samples: sampled.samples, length: sampled.length, closed: closed)
        } catch {
            switch error {
            case .invalidClosedPath: throw CurveTubeError.invalidClosedPath
            case .invalidSample(let index): throw CurveTubeError.invalidPathFrame(index)
            }
        }
        let circle = (0..<radialSegments).map { index in
            let angle = 2 * Double.pi * Double(index) / Double(radialSegments)
            return SIMD2(-cos(angle), sin(angle))
        }
        let extent = Double(radius)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(vertexCount)
        indices.reserveCapacity((tubularSegments + (capped ? 1 : 0)) * radialSegments * 6)
        for row in frames.indices {
            let frame = frames[row]
            let atSeam = closed && (row == 0 || row == tubularSegments)
            let before = frames[atSeam ? tubularSegments - 1 : max(0, row - 1)]
            let after = frames[atSeam ? 1 : min(tubularSegments, row + 1)]
            let u = Float(row) / Float(tubularSegments)
            for column in 0...radialSegments {
                let xy = circle[column % radialSegments]
                let normal = frame.map(xy)
                let position = SIMD3<Float>(frame.center + extent * normal)
                guard sweepFinite(position) else { throw SweepError.unrepresentableSurface }
                // The sampled surface derivative includes the roll distributed
                // along a closed path. A coarse chord can turn past the local
                // tangent, in which case use the curve derivative at the ring.
                let chord = (after.center - before.center) + extent * (after.map(xy) - before.map(xy))
                let projected = chord - simd_dot(chord, normal) * normal
                let tangent = simd_dot(projected, frame.tangent) > 0 ? simd_normalize(projected) : frame.tangent
                let v = 1 - Float(column) / Float(radialSegments)
                vertices.append(MeshVertex(
                    position: position, normal: SIMD3<Float>(normal),
                    textureCoordinate: SIMD2(u, v), tangent: SIMD4(SIMD3<Float>(tangent), -1)
                ))
            }
        }
        for row in 0..<tubularSegments {
            for column in 0..<radialSegments {
                let a = row * ringStride + column
                let b = a + ringStride
                try appendSweepTriangle(a, b, a + 1, vertices: vertices, indices: &indices)
                try appendSweepTriangle(b, b + 1, a + 1, vertices: vertices, indices: &indices)
            }
        }
        if capped {
            for end in [0, tubularSegments] {
                let frame = frames[end]
                let isStart = end == 0
                let normal = SIMD3<Float>(frame.tangent) * (isStart ? -1 : 1)
                let tangent = SIMD4(SIMD3<Float>(frame.normal), isStart ? 1 : -1)
                let center = vertices.count
                vertices.append(MeshVertex(
                    position: SIMD3<Float>(frame.center), normal: normal,
                    textureCoordinate: SIMD2(repeating: 0.5), tangent: tangent
                ))
                for (column, xy) in circle.enumerated() {
                    vertices.append(MeshVertex(
                        position: vertices[end * ringStride + column].position, normal: normal,
                        textureCoordinate: SIMD2(Float(0.5 + 0.5 * xy.x), Float(0.5 - 0.5 * xy.y)), tangent: tangent
                    ))
                }
                for column in 0..<radialSegments {
                    let a = center + 1 + column
                    let b = center + 1 + (column + 1) % radialSegments
                    try appendSweepTriangle(center, isStart ? a : b, isStart ? b : a, vertices: vertices, indices: &indices)
                }
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// Curve topology errors. Limits and surface errors shared with the
    /// polyline overload use `SweepError`.
    public enum CurveTubeError: LocalizedError, Equatable {
        case invalidTubularSegments
        case closedWithCaps
        case invalidClosedPath
        case invalidPathFrame(Int)

        public var errorDescription: String? {
            switch self {
            case .invalidTubularSegments: "A curve tube needs 1 to 4,096 tubular segments, with at least three for a closed tube."
            case .closedWithCaps: "A closed curve tube cannot have endpoint caps."
            case .invalidClosedPath: "A closed curve tube needs matching endpoint positions and tangents."
            case .invalidPathFrame(let index): "Tube curve sample \(index) coincides with its neighbor or reverses its frame."
            }
        }
    }

    public enum SweepError: LocalizedError, Equatable {
        case invalidPathCount
        case invalidPoint(Int)
        case coincidentPoints(Int)
        case reversingPath(Int)
        case invalidRadius
        case invalidRadialSegments
        case vertexBudget
        case unrepresentableSurface

        public var errorDescription: String? {
            switch self {
            case .invalidPathCount: "A tube path needs 2 to 4,097 points."
            case .invalidPoint(let index): "Tube path point \(index) must be finite."
            case .coincidentPoints(let index): "Tube path point \(index) coincides with the previous point."
            case .reversingPath(let index): "Tube path point \(index) reverses direction too sharply to frame."
            case .invalidRadius: "Tube radius must be a positive normal number."
            case .invalidRadialSegments: "A tube needs 3 to 128 radial segments."
            case .vertexBudget: "A tube may have at most 262,144 vertices including its caps."
            case .unrepresentableSurface: "The tube folds at a bend or exceeds position precision at this radius."
            }
        }
    }
}

private func appendSweepTriangle(
    _ a: Int, _ b: Int, _ c: Int, vertices: [MeshVertex], indices: inout [UInt32]
) throws {
    let first = vertices[a], second = vertices[b], third = vertices[c]
    let origin = SIMD3<Double>(first.position)
    let edgeA = SIMD3<Double>(second.position) - origin
    let edgeB = SIMD3<Double>(third.position) - origin
    let expected = SIMD3<Double>(first.normal + second.normal + third.normal)
    guard simd_dot(simd_cross(edgeA, edgeB), expected) > 0 else {
        throw MeshGeometry.SweepError.unrepresentableSurface
    }
    indices.append(UInt32(a))
    indices.append(UInt32(b))
    indices.append(UInt32(c))
}

/// Sampling helpers for constructing paths for procedural meshes.
public enum ProceduralPath {
    /// Samples a cubic Bézier at uniform parameter intervals, including both
    /// endpoints. Supply 1 to 4,096 segments and finite control points.
    public static func cubicBezier(
        from start: SIMD3<Float>, control1: SIMD3<Float>, control2: SIMD3<Float>,
        to end: SIMD3<Float>, segments: Int = 24
    ) throws -> [SIMD3<Float>] {
        guard (1...4_096).contains(segments) else { throw ValidationError.invalidSegments }
        guard sweepFinite(start), sweepFinite(control1), sweepFinite(control2), sweepFinite(end) else {
            throw ValidationError.invalidControlPoint
        }
        return try Curve3D.cubicBezier(from: start, control1: control1, control2: control2, to: end)
            .points(segments: segments)
    }

    public enum ValidationError: LocalizedError, Equatable {
        case invalidSegments
        case invalidControlPoint

        public var errorDescription: String? {
            switch self {
            case .invalidSegments: "A Bézier path needs 1 to 4,096 segments."
            case .invalidControlPoint: "Bézier control points must be finite."
            }
        }
    }
}

func sweepInitialReference(tangent: SIMD3<Double>) -> SIMD3<Double> {
    let alignment = abs(tangent)
    let axis: SIMD3<Double>
    if alignment.x <= alignment.y && alignment.x <= alignment.z {
        axis = SIMD3(1, 0, 0)
    } else if alignment.y <= alignment.z {
        axis = SIMD3(0, 1, 0)
    } else {
        axis = SIMD3(0, 0, 1)
    }
    return simd_normalize(axis - tangent * simd_dot(axis, tangent))
}

/// Shortest-arc Rodrigues rotation, followed by re-orthogonalization.
func sweepTransportReference(
    _ reference: SIMD3<Double>, from previous: SIMD3<Double>, to tangent: SIMD3<Double>
) -> SIMD3<Double>? {
    let denominator = 1 + simd_dot(previous, tangent)
    guard denominator > 1e-12 else { return nil }
    let axis = simd_cross(previous, tangent)
    let cross = simd_cross(axis, reference)
    let transported = reference + cross + simd_cross(axis, cross) / denominator
    return simd_normalize(transported - tangent * simd_dot(tangent, transported))
}

private func sweepFinite(_ point: SIMD3<Float>) -> Bool {
    point.x.isFinite && point.y.isFinite && point.z.isFinite
}
