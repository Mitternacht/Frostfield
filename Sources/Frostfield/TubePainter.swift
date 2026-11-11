import Foundation
import simd

/// An incremental, colored 3D brush with parallel-transported tube walls
/// and rounded caps. Feed mesh snapshots to a dynamic `MeshRenderer` using
/// this painter's capacity. The caller controls when to publish a revision.
public struct TubePainter: Sendable {
    public let radialSegments: Int
    public let capacity: MeshRenderer.DynamicGeometryCapacity
    public private(set) var radius: Float = 0.01
    public private(set) var color = SIMD4<Float>(repeating: 1)

    private static let capIntervals = 4
    private let circle: [SIMD2<Double>]
    private var vertices: [MeshVertex] = []
    private var indices: [UInt32] = []
    private var cursor: SIMD3<Double>?
    private var tail: Tail?

    public init(radialSegments: Int = 15, capacity: MeshRenderer.DynamicGeometryCapacity? = nil) throws {
        guard (3...128).contains(radialSegments) else { throw Failure.radialSegments }
        let capacity = try capacity ?? .init(vertices: 65_536, triangles: 65_536)
        guard capacity.vertices >= (2 * Self.capIntervals + 4) * (radialSegments + 1),
              capacity.triangles >= 4 * Self.capIntervals * radialSegments else { throw Failure.capacity }
        self.radialSegments = radialSegments
        self.capacity = capacity
        let points = (0..<radialSegments).map { column in
            let angle = 2 * Double.pi * Double(column) / Double(radialSegments)
            return SIMD2(cos(angle), sin(angle))
        }
        circle = points + [points[0]]
    }

    /// Sets a world-space radius for the next sample. At the start of a
    /// stroke the radius applies to both ends of the first segment.
    public mutating func setRadius(_ value: Float) throws {
        guard value.isNormal, value > 0 else { throw Failure.radius }
        radius = value
    }

    /// Sets linear RGBA for the next sample. A segment interpolates between
    /// its two endpoint colors.
    public mutating func setColor(_ value: SIMD4<Float>) throws {
        let rgb = SIMD3(value.x, value.y, value.z)
        guard boundsFinite(rgb), simd_reduce_min(rgb) >= 0, (0...1).contains(value.w) else { throw Failure.color }
        color = value
    }

    /// Starts a new stroke at a finite position. Earlier strokes remain in
    /// the mesh with their current caps.
    public mutating func move(to point: SIMD3<Float>) throws {
        guard boundsFinite(point) else { throw Failure.point }
        cursor = SIMD3<Double>(point)
        tail = nil
    }

    /// Appends one brush sample. A repeated point returns false. Invalid or
    /// over-capacity samples leave the stroke at its previous position.
    @discardableResult
    public mutating func line(to point: SIMD3<Float>) throws -> Bool {
        guard boundsFinite(point) else { throw Failure.point }
        guard let cursor else { throw Failure.missingMove }
        let destination = SIMD3<Double>(point)
        let delta = destination - cursor
        let length = simd_length(delta)
        guard length > 0 else { return false }
        let direction = delta / length
        let reference: SIMD3<Double>
        if let tail {
            guard let transported = sweepTransportReference(tail.frame.reference, from: tail.frame.direction, to: direction) else {
                throw Failure.reversingDirection
            }
            reference = transported
        } else {
            let axis: SIMD3<Double> = abs(direction.y) < 0.99 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
            reference = simd_normalize(axis - simd_dot(axis, direction) * direction)
        }
        let frame = Frame(direction: direction, reference: reference, side: simd_cross(direction, reference))
        let distance = (tail?.distance ?? 0) + length
        let stride = circle.count
        let capVertices = (Self.capIntervals + 1) * stride
        let vertexIncrease = tail == nil ? 2 * (stride + capVertices) : stride
        let triangleIncrease = tail == nil ? 4 * Self.capIntervals * radialSegments : 2 * radialSegments
        guard vertexIncrease <= capacity.vertices - vertices.count,
              triangleIncrease <= capacity.triangles - indices.count / 3 else { throw Failure.capacity }

        let nextRing = try ring(center: destination, frame: frame, distance: distance)
        let ringStart: Int
        let capStart: Int
        var addedIndices: [UInt32] = []
        addedIndices.reserveCapacity(triangleIncrease * 3)
        if let tail {
            ringStart = vertices.count
            capStart = tail.capStart
            try appendStrip(
                vertices[tail.ringStart..<tail.ringStart + stride], nextRing[...],
                start: tail.ringStart, end: ringStart, into: &addedIndices
            )
            let endCap = try cap(center: destination, frame: frame, distance: distance, start: capStart, outward: 1)
            vertices.replaceSubrange(capStart..<capStart + capVertices, with: endCap.vertices)
            vertices.append(contentsOf: nextRing)
        } else {
            let firstStart = vertices.count
            ringStart = firstStart + stride
            let firstCapStart = ringStart + stride
            capStart = firstCapStart + capVertices
            let firstRing = try ring(center: cursor, frame: frame, distance: 0)
            let firstCap = try cap(center: cursor, frame: frame, distance: 0, start: firstCapStart, outward: -1)
            let endCap = try cap(center: destination, frame: frame, distance: distance, start: capStart, outward: 1)
            try appendStrip(firstRing[...], nextRing[...], start: firstStart, end: ringStart, into: &addedIndices)
            addedIndices.append(contentsOf: firstCap.indices)
            addedIndices.append(contentsOf: endCap.indices)
            vertices.append(contentsOf: firstRing)
            vertices.append(contentsOf: nextRing)
            vertices.append(contentsOf: firstCap.vertices)
            vertices.append(contentsOf: endCap.vertices)
        }
        indices.append(contentsOf: addedIndices)
        self.cursor = destination
        tail = Tail(frame: frame, distance: distance, ringStart: ringStart, capStart: capStart)
        return true
    }

    /// Returns the current indexed stroke mesh, or nil before any segment
    /// has been drawn. U wraps each ring and V follows stroke length in
    /// world units. Earlier snapshots retain their own geometry.
    public func mesh() throws -> MeshGeometry? {
        guard !indices.isEmpty else { return nil }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    public enum Failure: LocalizedError, Equatable {
        case radialSegments
        case radius
        case color
        case point
        case missingMove
        case reversingDirection
        case capacity
        case foldedSurface

        public var errorDescription: String? {
            switch self {
            case .radialSegments: "Use 3 to 128 radial segments for the tube brush."
            case .radius: "The brush radius must be a positive normal number."
            case .color: "Use finite nonnegative linear RGB and alpha in 0...1."
            case .point: "Brush points must be finite."
            case .missingMove: "Move the tube brush to the start of a stroke."
            case .reversingDirection: "Use a new stroke when the brush reverses direction."
            case .capacity: "The tube stroke exceeds the painter's vertex or triangle capacity."
            case .foldedSurface: "The brush folds or collapses a face at this radius or sample spacing."
            }
        }
    }

    private struct Frame: Sendable {
        let direction: SIMD3<Double>
        let reference: SIMD3<Double>
        let side: SIMD3<Double>
    }

    private struct Tail: Sendable {
        let frame: Frame
        let distance: Double
        let ringStart: Int
        let capStart: Int
    }

    private func ring(
        center: SIMD3<Double>, frame: Frame, distance: Double, latitude: Double = 0, outward: Double = 1
    ) throws -> [MeshVertex] {
        let atPole = latitude == .pi / 2
        let circumferential = atPole ? 0 : cos(latitude)
        let axial = outward * (atPole ? 1 : sin(latitude))
        let v = Float(distance + outward * Double(radius) * latitude)
        guard v.isFinite else { throw Failure.foldedSurface }
        return try circle.enumerated().map { column, xy in
            let radial = xy.x * frame.reference + xy.y * frame.side
            let normal = circumferential * radial + axial * frame.direction
            let position = SIMD3<Float>(center + Double(radius) * normal)
            guard boundsFinite(position) else { throw Failure.foldedSurface }
            let tangent = SIMD3<Float>(-xy.y * frame.reference + xy.x * frame.side)
            return MeshVertex(
                position: position, normal: SIMD3<Float>(normal), textureCoordinate: SIMD2(Float(column) / Float(radialSegments), v),
                tangent: SIMD4(tangent, 1), color: color
            )
        }
    }

    private func cap(
        center: SIMD3<Double>, frame: Frame, distance: Double, start: Int, outward: Double
    ) throws -> (vertices: [MeshVertex], indices: [UInt32]) {
        let stride = circle.count
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity((Self.capIntervals + 1) * stride)
        for row in 0...Self.capIntervals {
            let latitude = (.pi / 2) * Double(row) / Double(Self.capIntervals)
            vertices.append(contentsOf: try ring(center: center, frame: frame, distance: distance, latitude: latitude, outward: outward))
        }
        for row in 0..<Self.capIntervals {
            let a = row * stride, b = a + stride
            try appendStrip(vertices[a..<b], vertices[b..<b + stride], start: start + a, end: start + b,
                            reversed: outward < 0, endsAtPole: row == Self.capIntervals - 1, into: &indices)
        }
        return (vertices, indices)
    }

    private func appendStrip(
        _ first: ArraySlice<MeshVertex>, _ second: ArraySlice<MeshVertex>, start: Int, end: Int,
        reversed: Bool = false, endsAtPole: Bool = false, into indices: inout [UInt32]
    ) throws {
        func triangle(
            _ a: MeshVertex, _ b: MeshVertex, _ c: MeshVertex, _ i: Int, _ j: Int, _ k: Int
        ) throws {
            let origin = SIMD3<Double>(a.position)
            let face = simd_cross(SIMD3<Double>(b.position) - origin, SIMD3<Double>(c.position) - origin)
            let expected = SIMD3<Double>(a.normal + b.normal + c.normal)
            let facing = simd_dot(face, expected)
            guard (reversed ? -facing : facing) > 0 else { throw Failure.foldedSurface }
            indices.append(UInt32(i))
            indices.append(UInt32(reversed ? k : j))
            indices.append(UInt32(reversed ? j : k))
        }
        for column in 0..<radialSegments {
            let a = first.startIndex + column, b = second.startIndex + column
            try triangle(first[a], first[a + 1], second[b], start + column, start + column + 1, end + column)
            if !endsAtPole {
                try triangle(first[a + 1], second[b + 1], second[b], start + column + 1, end + column + 1, end + column)
            }
        }
    }
}
