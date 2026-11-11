import simd

/// Refines an already validated fill using original Float-representable points.
/// Keeping the constrained edges avoids bridging an outside point into the fill.
enum PolygonSteinerInsertion {
    private static let maximumTriangleVisits = 8_388_608

    static func apply(
        points: [SIMD2<Double>], boundaryCount: Int, to triangles: inout [SIMD3<Int>]
    ) throws {
        var occupied = Set(points.prefix(boundaryCount))
        let distinct = points.indices.dropFirst(boundaryCount).filter { occupied.insert(points[$0]).inserted }
        guard !distinct.isEmpty else { return }
        // Each insertion can add at most two triangles. Bound the sum of
        // candidate visits before searching, with division to avoid overflow.
        let averageBudget = maximumTriangleVisits / distinct.count
        guard triangles.count <= averageBudget, distinct.count - 1 <= averageBudget - triangles.count else {
            throw PlanarShape.TriangulationFailure.steinerWorkLimit
        }
        for index in distinct {
            try insert(index, points: points, triangles: &triangles)
        }
    }

    private static func insert(
        _ index: Int, points: [SIMD2<Double>], triangles: inout [SIMD3<Int>]
    ) throws {
        let point = points[index]
        var edgeFaces = 0
        // Search only the triangles present before insertion. If the point
        // lies on a shared edge, both incident triangles must be split.
        for triangleIndex in 0..<triangles.count {
            let triangle = triangles[triangleIndex]
            let a = points[triangle.x], b = points[triangle.y], c = points[triangle.z]
            let lower = simd_min(a, simd_min(b, c)), upper = simd_max(a, simd_max(b, c))
            guard point.x >= lower.x, point.x <= upper.x, point.y >= lower.y, point.y <= upper.y else { continue }
            let sides = SIMD3(orientation(a, b, point), orientation(b, c, point), orientation(c, a, point))
            guard simd_reduce_min(sides) >= 0 else { continue }
            if let edge = (0..<3).first(where: { sides[$0] == 0 }) {
                let start = triangle[edge], end = triangle[(edge + 1) % 3], opposite = triangle[(edge + 2) % 3]
                edgeFaces += 1
                guard edgeFaces <= 2 else { throw PlanarShape.Failure.unresolvedBoundary }
                triangles[triangleIndex] = SIMD3(start, index, opposite)
                triangles.append(SIMD3(index, end, opposite))
            } else {
                guard edgeFaces == 0 else { throw PlanarShape.Failure.unresolvedBoundary }
                triangles[triangleIndex] = SIMD3(triangle.x, triangle.y, index)
                triangles.append(SIMD3(triangle.y, triangle.z, index))
                triangles.append(SIMD3(triangle.z, triangle.x, index))
                return
            }
        }
    }

    private static func orientation(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ point: SIMD2<Double>) -> Double {
        let approximate = PlanarArrangement.orientation(a, b, point)
        if approximate != 0 || a.x == b.x || a.y == b.y { return approximate }

        // Each product of two binary32 coordinates is exact in Double.
        // Sum the six products of the determinant with error-free additions
        // when the fast sign is ambiguous. The last nonzero term gives the
        // exact sign, so a close point is never snapped onto a constraint.
        // https://www.cs.cmu.edu/~quake/robust.html
        let terms = [a.x * b.y, -a.y * b.x, b.x * point.y, -b.y * point.x, point.x * a.y, -point.y * a.x]
        var expansion: [Double] = []
        expansion.reserveCapacity(terms.count)
        for term in terms {
            var carry = term
            for index in expansion.indices {
                let value = expansion[index]
                let sum = carry + value
                let recovered = sum - carry
                expansion[index] = (carry - (sum - recovered)) + (value - recovered)
                carry = sum
            }
            expansion.append(carry)
        }
        return expansion.last { $0 != 0 } ?? 0
    }
}
