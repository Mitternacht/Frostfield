import Foundation
import simd

extension MeshGeometry {
    /// Extrudes one simple XY contour with planar caps at ±depth/2. The 3 to
    /// 256 vertices may wind either way. Omit the duplicated closing vertex.
    /// The contour may be concave and include intermediate collinear vertices.
    /// Each wall has its own outward normal and uses perimeter-length U and
    /// depth V. The first input vertex is the wall UV seam.
    public static func extrude(contour: [SIMD2<Float>], depth: Float) throws -> MeshGeometry {
        guard (3...256).contains(contour.count) else { throw ExtrusionError.invalidVertexCount }
        let halfDepth = depth / 2
        guard halfDepth.isNormal, halfDepth > 0 else { throw ExtrusionError.invalidDepth }

        var seen = Set<SIMD2<Float>>()
        for (index, point) in contour.enumerated() {
            guard point.x.isFinite, point.y.isFinite else { throw ExtrusionError.invalidPoint(index) }
            guard seen.insert(point).inserted else { throw ExtrusionError.duplicatePoint(index) }
        }
        var points = contour.map(SIMD2<Double>.init)
        try validateContourEdges(points)
        let origin = points[0]
        var signedArea = 0.0
        var unsignedArea = 0.0
        for index in 1..<(points.count - 1) {
            let area = contourTurn(origin, points[index], points[index + 1])
            signedArea += area
            unsignedArea += abs(area)
        }
        guard abs(signedArea) > Double(points.count) * Double.ulpOfOne * unsignedArea else {
            throw ExtrusionError.degenerateContour
        }
        if signedArea < 0 {
            points = [origin] + points.dropFirst().reversed()
        }
        let cap = try triangulateContour(points)
        let count = points.count
        let lower = points.reduce(origin) { simd_min($0, $1) }
        let upper = points.reduce(origin) { simd_max($0, $1) }
        let extent = upper - lower
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(6 * count)
        indices.reserveCapacity(12 * count - 12)

        for sign: Float in [1, -1] {
            for point in points {
                vertices.append(MeshVertex(
                    position: SIMD3(Float(point.x), Float(point.y), sign * halfDepth),
                    normal: SIMD3(0, 0, sign),
                    textureCoordinate: SIMD2(Float((point.x - lower.x) / extent.x), Float((upper.y - point.y) / extent.y)),
                    tangent: SIMD4(1, 0, 0, -sign)
                ))
            }
        }
        for triangle in cap {
            let a = UInt32(triangle.x), b = UInt32(triangle.y), c = UInt32(triangle.z)
            let back = UInt32(count)
            indices += [a, b, c, back + a, back + c, back + b]
        }

        let edgeLengths = points.indices.map { simd_length(points[($0 + 1) % count] - points[$0]) }
        let perimeter = edgeLengths.reduce(0, +)
        var distance = 0.0
        for index in points.indices {
            let a = points[index], b = points[(index + 1) % count]
            let length = edgeLengths[index]
            let edge = SIMD2<Float>((b - a) / length)
            let outward = SIMD3(edge.y, -edge.x, 0)
            let tangent = SIMD4(edge.x, edge.y, 0, -1)
            let startU = Float(distance / perimeter)
            distance += length
            let endU = Float(distance / perimeter)
            let base = UInt32(vertices.count)
            for (point, u, z, v) in [(a, startU, halfDepth, 0), (b, endU, halfDepth, 0),
                                     (a, startU, -halfDepth, 1), (b, endU, -halfDepth, 1)] {
                vertices.append(MeshVertex(
                    position: SIMD3(Float(point.x), Float(point.y), z), normal: outward,
                    textureCoordinate: SIMD2(u, Float(v)), tangent: tangent
                ))
            }
            indices += [base, base + 2, base + 1, base + 1, base + 2, base + 3]
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    public enum ExtrusionError: LocalizedError, Equatable {
        case invalidVertexCount
        case invalidDepth
        case invalidPoint(Int)
        case duplicatePoint(Int)
        case selfIntersection
        case degenerateContour
        case triangulation

        public var errorDescription: String? {
            switch self {
            case .invalidVertexCount: "An extrusion contour needs 3 to 256 vertices."
            case .invalidDepth: "Extrusion depth must produce finite, positive half-depth at normal Float precision."
            case .invalidPoint(let index): "Extrusion contour vertex \(index) must be finite."
            case .duplicatePoint(let index): "Extrusion contour vertex \(index) duplicates another vertex."
            case .selfIntersection: "Extrusion contour edges must form a simple loop without crossing, touching or backtracking."
            case .degenerateContour: "Extrusion contour must enclose an area at the supplied position precision."
            case .triangulation: "The extrusion contour could not be triangulated at the supplied position precision."
            }
        }
    }

    private static func validateContourEdges(_ points: [SIMD2<Double>]) throws {
        let count = points.count
        for index in points.indices {
            let next = (index + 1) % count
            let a = points[index], b = points[next], previous = points[(index + count - 1) % count]
            if contourTurn(previous, a, b) == 0, simd_dot(previous - a, b - a) > 0 {
                throw ExtrusionError.selfIntersection
            }
            for other in (index + 1)..<count {
                let otherNext = (other + 1) % count
                guard other != next, otherNext != index else { continue }
                if contourEdgesIntersect(a, b, points[other], points[otherNext]) {
                    throw ExtrusionError.selfIntersection
                }
            }
        }
    }

    private static func triangulateContour(_ points: [SIMD2<Double>]) throws -> [SIMD3<Int>] {
        // Remove convex ears whose triangles contain no remaining boundary point.
        // https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
        var remaining = Array(points.indices)
        var triangles: [SIMD3<Int>] = []
        triangles.reserveCapacity(points.count - 2)
        while remaining.count > 3 {
            let count = remaining.count
            var ear: Int?
            for index in remaining.indices {
                let previous = remaining[(index + count - 1) % count]
                let current = remaining[index]
                let next = remaining[(index + 1) % count]
                let a = points[previous], b = points[current], c = points[next]
                guard contourTurn(a, b, c) > 0 else { continue }
                let lower = simd_min(a, simd_min(b, c)), upper = simd_max(a, simd_max(b, c))
                let blocked = remaining.contains { vertex in
                    guard vertex != previous, vertex != current, vertex != next else { return false }
                    let point = points[vertex]
                    return contourBoundsContain(point, lower: lower, upper: upper)
                        && contourTurn(a, b, point) >= 0 && contourTurn(b, c, point) >= 0 && contourTurn(c, a, point) >= 0
                }
                guard !blocked else { continue }
                triangles.append(SIMD3(previous, current, next))
                ear = index
                break
            }
            guard let ear else { throw ExtrusionError.triangulation }
            remaining.remove(at: ear)
        }
        let a = remaining[0], b = remaining[1], c = remaining[2]
        guard contourTurn(points[a], points[b], points[c]) > 0 else { throw ExtrusionError.triangulation }
        triangles.append(SIMD3(a, b, c))
        return triangles
    }

    private static func contourEdgesIntersect(
        _ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>, _ d: SIMD2<Double>
    ) -> Bool {
        let firstLower = simd_min(a, b), firstUpper = simd_max(a, b)
        let secondLower = simd_min(c, d), secondUpper = simd_max(c, d)
        guard firstLower.x <= secondUpper.x, secondLower.x <= firstUpper.x,
              firstLower.y <= secondUpper.y, secondLower.y <= firstUpper.y else { return false }
        let firstC = contourTurn(a, b, c), firstD = contourTurn(a, b, d)
        let secondA = contourTurn(c, d, a), secondB = contourTurn(c, d, b)
        if firstC == 0 && contourBoundsContain(c, lower: firstLower, upper: firstUpper) { return true }
        if firstD == 0 && contourBoundsContain(d, lower: firstLower, upper: firstUpper) { return true }
        if secondA == 0 && contourBoundsContain(a, lower: secondLower, upper: secondUpper) { return true }
        if secondB == 0 && contourBoundsContain(b, lower: secondLower, upper: secondUpper) { return true }
        return firstC * firstD < 0 && secondA * secondB < 0
    }

    private static func contourBoundsContain(_ p: SIMD2<Double>, lower: SIMD2<Double>, upper: SIMD2<Double>) -> Bool {
        p.x >= lower.x && p.x <= upper.x && p.y >= lower.y && p.y <= upper.y
    }

    private static func contourTurn(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>) -> Double {
        let ab = b - a, ac = c - a
        let left = ab.x * ac.y, right = ab.y * ac.x
        let determinant = left - right
        let tolerance = 8 * Double.ulpOfOne * (abs(left) + abs(right))
        return abs(determinant) <= tolerance ? 0 : determinant
    }
}
