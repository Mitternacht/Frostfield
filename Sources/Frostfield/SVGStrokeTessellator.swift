import CoreGraphics
import simd

/// A ribbon retains UVs at its joins and seams. The default clipped outline
/// instead unions the same polygons before triangulating with planar UVs.
struct SVGStrokeTessellator {
    private typealias Point = SIMD2<Double>
    private typealias Stroke = SVGLoader.Stroke

    private struct Vertex {
        let position: Point
        let uv: SIMD2<Float>

        init(_ position: Point, _ u: Float, _ v: Float) {
            self.position = position
            uv = SIMD2(u, v)
        }
    }

    private struct Section {
        var left: Point
        var right: Point

        init(center: Point, offset: Point) {
            left = center - offset
            right = center + offset
        }

        mutating func set(_ point: Point, leftSide: Bool) {
            if leftSide { left = point } else { right = point }
        }
    }

    private let polygons: [[Vertex]]

    init(path: CGPath, transform: CGAffineTransform, stroke: SVGLoader.Stroke, tolerance: Float, yUp: Bool) throws {
        guard tolerance.isFinite, tolerance > 0 else { throw PlanarShape.Failure.invalidTolerance }
        if case .unresolved(let name) = stroke.join { throw SVGLoader.Failure.strokeJoin(name) }
        // The Frobenius norm bounds how much the document transform stretches
        // a local flattening error in any direction.
        let stretch = hypot(hypot(transform.a, transform.b), hypot(transform.c, transform.d))
        guard stretch > 0 else { polygons = []; return }
        let localTolerance = Double(tolerance) / max(Double(stretch), .leastNormalMagnitude)
        let subpaths: [PlanarShape.FlattenedSubpath]
        do {
            subpaths = try PlanarShape.flattenedSubpaths(path, tolerance: localTolerance, maximumPoints: 4_096)
        } catch is PlanarComplexityLimit {
            throw SVGLoader.Failure.size
        }

        var pieces: [[Vertex]] = []
        var remainingVertices = 65_536
        for subpath in subpaths {
            var points: [Point] = []
            for point in subpath.points where point != points.last {
                points.append(point)
            }
            if subpath.closed, points.last == points.first { points.removeLast() }
            guard points.count >= 2 else { continue }
            try Self.appendRibbon(
                points, closed: subpath.closed, stroke: stroke, tolerance: localTolerance
            ) { polygon in
                guard polygon.count <= remainingVertices else { throw SVGLoader.Failure.size }
                remainingVertices -= polygon.count
                let transformed = try polygon.map { vertex in
                    let position = CGPoint(x: vertex.position.x, y: vertex.position.y).applying(transform)
                    guard abs(position.x) <= SVGLoader.coordinateLimit,
                          abs(position.y) <= SVGLoader.coordinateLimit else { throw SVGLoader.Failure.geometry }
                    // Round once to the precision used by the planar arrangement and mesh.
                    let point = Point(SIMD2<Float>(Float(position.x), Float(yUp ? -position.y : position.y)))
                    return Vertex(point, vertex.uv.x, vertex.uv.y)
                }
                pieces.append(transformed)
            }
        }
        polygons = pieces
    }

    func outline() throws -> [PlanarShape] {
        let rings = polygons.compactMap { polygon -> [SIMD2<Float>]? in
            let points = polygon.map(\.position)
            let origin = points[0]
            let area = (1..<(points.count - 1)).reduce(0.0) { area, index in
                area + Self.cross(points[index] - origin, points[index + 1] - origin)
            }
            guard area != 0 else { return nil }
            return (area > 0 ? points : Array(points.reversed())).map(SIMD2<Float>.init)
        }
        guard rings.reduce(0, { $0 + $1.count }) <= 8_192 else { throw SVGLoader.Failure.size }
        do {
            return try PlanarShape.shapes(rings: rings, fillRule: .winding, maximumEdges: 8_192)
        } catch is PlanarComplexityLimit {
            throw SVGLoader.Failure.size
        }
    }

    func geometry() throws -> MeshGeometry? {
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for polygon in polygons {
            let base = UInt32(vertices.count)
            let origin = polygon[0].position
            var faces: [UInt32] = []
            for index in 1..<(polygon.count - 1) {
                let area = Self.cross(polygon[index].position - origin, polygon[index + 1].position - origin)
                guard area != 0 else { continue }
                let b = base + UInt32(index), c = b + 1
                faces += area > 0 ? [base, b, c] : [base, c, b]
            }
            guard !faces.isEmpty else { continue }
            vertices += polygon.map { vertex in
                MeshVertex(
                    position: SIMD3(Float(vertex.position.x), Float(vertex.position.y), 0),
                    normal: SIMD3(0, 0, 1), textureCoordinate: vertex.uv
                )
            }
            indices += faces
        }
        return indices.isEmpty ? nil : try MeshGeometry(vertices: vertices, indices: indices)
    }

    private static func appendRibbon(
        _ points: [Point], closed: Bool, stroke: Stroke, tolerance: Double,
        emit: ([Vertex]) throws -> Void
    ) throws {
        let count = points.count
        let segmentCount = closed ? count : count - 1
        let halfWidth = Double(stroke.width) * 0.5
        var lengths: [Double] = []
        var directions: [Point] = []
        var starts: [Section] = []
        var ends: [Section] = []
        for index in 0..<segmentCount {
            let a = points[index], b = points[(index + 1) % count]
            let length = simd_length(b - a)
            let direction = (b - a) / length
            let offset = normal(direction) * halfWidth
            lengths.append(length)
            directions.append(direction)
            starts.append(Section(center: a, offset: offset))
            ends.append(Section(center: b, offset: offset))
        }
        func u(_ index: Int) -> Float { Float(index) / Float(segmentCount) }

        let corners = closed ? 0..<count : 1..<(count - 1)
        for index in corners {
            let previous = (index + segmentCount - 1) % segmentCount
            let incoming = directions[previous], outgoing = directions[index]
            let turn = cross(incoming, outgoing)
            guard abs(turn) > 1e-12 else { continue }

            let center = points[index]
            let cornerU = index == 0 ? 1 : u(index)
            let outerIsLeft = turn > 0
            let side: Double = outerIsLeft ? -1 : 1
            let edgeV: Float = outerIsLeft ? 0 : 1
            let previousOffset = normal(incoming) * (halfWidth * side)
            let nextOffset = normal(outgoing) * (halfWidth * side)
            let previousOuter = center + previousOffset
            let nextOuter = center + nextOffset

            let tipOffset = incoming - outgoing
            let miterOffset = tipOffset * (halfWidth / abs(turn))
            let tip = center + miterOffset
            let fraction = halfWidth * Double(stroke.miterLimit) / simd_length(miterOffset)
            let inner = center - miterOffset

            // A shared inner intersection gives both segments the same edge.
            // Very short segments can overlap, but neither adjacent corner
            // can trim past the midpoint and fold the segment's triangles.
            let trim = abs(simd_dot(miterOffset, incoming))
            let canTrimInner = trim < min(lengths[previous], lengths[index]) * 0.5
            if canTrimInner {
                ends[previous].set(inner, leftSide: !outerIsLeft)
                starts[index].set(inner, leftSide: !outerIsLeft)
            }

            if (stroke.join == .miter || stroke.join == .miterClip), fraction >= 1, canTrimInner {
                ends[previous].set(tip, leftSide: outerIsLeft)
                starts[index].set(tip, leftSide: outerIsLeft)
                continue
            }

            let pivot = Vertex(center, cornerU, 0.5)
            let previousEdge = Vertex(previousOuter, cornerU, edgeV)
            let nextEdge = Vertex(nextOuter, cornerU, edgeV)
            if canTrimInner {
                let innerEdge = Vertex(inner, cornerU, 1 - edgeV)
                try emit([previousEdge, pivot, innerEdge])
                try emit([pivot, nextEdge, innerEdge])
            }

            switch stroke.join {
            case .round:
                let angle = atan2(cross(previousOffset, nextOffset), simd_dot(previousOffset, nextOffset))
                try emit(arc(
                    from: previousOffset, to: nextOffset, angle: angle, center: center, u: cornerU, edgeV: edgeV,
                    radius: halfWidth, tolerance: tolerance
                ))
            case .miterClip where fraction < 1:
                // The pinned pointsToStroke interpolates along both outer
                // offsets by the ratio of limited to unrestricted length.
                let a = previousOuter + (tip - previousOuter) * fraction
                let b = nextOuter + (tip - nextOuter) * fraction
                try emit([pivot, previousEdge, Vertex(a, cornerU, edgeV), Vertex(b, cornerU, edgeV), nextEdge])
            case .miter, .miterClip:
                if fraction >= 1 {
                    try emit([pivot, previousEdge, Vertex(tip, cornerU, edgeV), nextEdge])
                } else {
                    try emit([pivot, previousEdge, nextEdge])
                }
            case .bevel:
                try emit([pivot, previousEdge, nextEdge])
            case .unresolved:
                break // Rejected before flattening.
            }
        }

        if !closed {
            let last = segmentCount - 1
            switch stroke.cap {
            case .butt: break
            case .square:
                starts[0].left -= directions[0] * halfWidth
                starts[0].right -= directions[0] * halfWidth
                ends[last].left += directions[last] * halfWidth
                ends[last].right += directions[last] * halfWidth
            case .round:
                let firstOffset = normal(directions[0]) * halfWidth
                let lastOffset = normal(directions[last]) * halfWidth
                try emit(arc(
                    from: firstOffset, to: -firstOffset, angle: .pi, center: points[0],
                    u: 0, edgeV: 0.5, radius: halfWidth, tolerance: tolerance
                ))
                try emit(arc(
                    from: -lastOffset, to: lastOffset, angle: .pi, center: points[count - 1],
                    u: 1, edgeV: 0.5, radius: halfWidth, tolerance: tolerance
                ))
            }
        }
        for index in 0..<segmentCount {
            let start = starts[index], end = ends[index]
            try emit([
                Vertex(start.right, u(index), 1), Vertex(start.left, u(index), 0),
                Vertex(end.left, u(index + 1), 0), Vertex(end.right, u(index + 1), 1)
            ])
        }
    }

    private static func arc(
        from start: Point, to end: Point, angle: Double, center: Point, u: Float, edgeV: Float,
        radius: Double, tolerance: Double
    ) -> [Vertex] {
        let step = 2 * acos(max(-1, 1 - tolerance / radius))
        let divisions = Int(min(64, max(12, ceil(abs(angle) / max(step, .leastNormalMagnitude)))))
        var polygon = [Vertex(center, u, 0.5), Vertex(center + start, u, edgeV)]
        for index in 1..<divisions {
            let phi = angle * Double(index) / Double(divisions)
            let rotated = start * cos(phi) + normal(start) * sin(phi)
            polygon.append(Vertex(center + rotated, u, edgeV))
        }
        polygon.append(Vertex(center + end, u, edgeV))
        return polygon
    }

    private static func normal(_ v: Point) -> Point { Point(-v.y, v.x) }
    private static func cross(_ a: Point, _ b: Point) -> Double { a.x * b.y - a.y * b.x }
}
