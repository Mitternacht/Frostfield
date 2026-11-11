import Foundation
import simd

public extension MeshGeometry {
    /// Connects corresponding points of successive 3D cross sections with triangles.
    ///
    /// Supply at least two sections with the same number of points. A closed
    /// section needs at least three distinct points, and an open section needs
    /// two. Omit a repeated closing point. For outward walls, wind the sections
    /// counterclockwise as seen from the end of the loft looking toward its start.
    ///
    /// U follows accumulated mean distances between corresponding points of
    /// neighboring sections. V follows each profile's perimeter from one to
    /// zero in top-left UV coordinates. Closed profiles duplicate the seam so
    /// both sides have separate UVs and a shared smooth normal.
    ///
    /// Caps have separate vertices and planar UVs. A capped section needs at
    /// least three points and a simple projection onto its Newell plane. On an
    /// open strip the cap closes the endpoint polygon while the strip stays open.
    /// A loft may contain up to one million vertices and 4,096 points per cap.
    static func loft(
        sections: [[SIMD3<Float>]], closed: Bool = true, capStart: Bool = false, capEnd: Bool = false
    ) throws -> MeshGeometry {
        guard sections.count >= 2 else { throw LoftError.sectionCount }
        let columns = sections[0].count
        guard columns >= (closed ? 3 : 2) else { throw LoftError.pointCount(0) }
        let capSections = (capStart ? [0] : []) + (capEnd ? [sections.count - 1] : [])
        if let cap = capSections.first, !(3...4_096).contains(columns) { throw LoftError.cap(cap) }
        guard columns <= 1_000_000 else { throw LoftError.vertexBudget }
        let stride = columns + (closed ? 1 : 0)
        let capVertices = capSections.count * columns
        guard capVertices < 1_000_000, sections.count <= (1_000_000 - capVertices) / stride else {
            throw LoftError.vertexBudget
        }

        var rowDistance = [0.0]
        rowDistance.reserveCapacity(sections.count)
        for (row, section) in sections.enumerated() {
            guard section.count == columns else { throw LoftError.pointCount(row) }
            var seen = Set<SIMD3<Float>>()
            for (index, point) in section.enumerated() {
                guard point.x.isFinite, point.y.isFinite, point.z.isFinite else {
                    throw LoftError.nonfinitePoint(section: row, point: index)
                }
                guard seen.insert(point).inserted else { throw LoftError.duplicatePoint(section: row, point: index) }
            }
            if row > 0 {
                let distance = zip(sections[row - 1], section).reduce(0.0) {
                    $0 + simd_distance(SIMD3<Double>($1.0), SIMD3<Double>($1.1))
                }
                guard distance > 0 else { throw LoftError.degenerateWall(section: row - 1, edge: 0) }
                rowDistance.append(rowDistance[row - 1] + distance / Double(columns))
            }
        }

        let vertexCount = sections.count * stride + capVertices
        let indexCount = (sections.count - 1) * (stride - 1) * 6 + capSections.count * (columns - 2) * 3
        var surface = LoftSurface(vertexCount: vertexCount, indexCount: indexCount)
        let totalDistance = rowDistance[sections.count - 1]
        for (row, section) in sections.enumerated() {
            let points = section.map(SIMD3<Double>.init)
            var perimeter = [0.0]
            perimeter.reserveCapacity(stride)
            for column in 1..<stride {
                let distance = simd_distance(points[column - 1], points[column % columns])
                perimeter.append(perimeter[column - 1] + distance)
            }
            let u = Float(rowDistance[row] / totalDistance)
            let totalPerimeter = perimeter[stride - 1]
            for column in 0..<stride {
                let v = Float(1 - perimeter[column] / totalPerimeter)
                surface.appendVertex(points[column % columns], uv: SIMD2(u, v))
            }
        }

        for row in 0..<(sections.count - 1) {
            for column in 0..<(stride - 1) {
                let a = row * stride + column
                let d = a + stride
                guard surface.appendTriangle(a, a + 1, d), surface.appendTriangle(a + 1, d + 1, d) else {
                    throw LoftError.degenerateWall(section: row, edge: column)
                }
            }
        }
        if closed {
            for row in sections.indices {
                let first = row * stride
                let last = first + columns
                // The wall on each side of the UV split supplies one normal.
                let averaged = try surface.unitNormal(at: first) + surface.unitNormal(at: last)
                surface.normalSums[first] = averaged
                surface.normalSums[last] = averaged
            }
        }
        for section in capSections {
            let neighbor = section == 0 ? 1 : section - 1
            try surface.appendCap(sections[section], nextTo: sections[neighbor], sectionIndex: section)
        }
        return try surface.geometry()
    }

    enum LoftError: LocalizedError, Equatable {
        case sectionCount
        case pointCount(Int)
        case vertexBudget
        case nonfinitePoint(section: Int, point: Int)
        case duplicatePoint(section: Int, point: Int)
        case degenerateWall(section: Int, edge: Int)
        case undefinedNormal(Int)
        case cap(Int)

        public var errorDescription: String? {
            switch self {
            case .sectionCount: "A loft needs at least two sections."
            case .pointCount(let section): "Loft section \(section) needs the same number of points as the first, at least two for strips or three for rings."
            case .vertexBudget: "A loft may have at most one million vertices."
            case .nonfinitePoint(let section, let point): "Loft section \(section), point \(point) must be finite."
            case .duplicatePoint(let section, let point): "Loft section \(section), point \(point) duplicates another point in that section."
            case .degenerateWall(let section, let edge): "The loft wall after section \(section) collapses at edge \(edge)."
            case .undefinedNormal(let vertex): "Loft vertex \(vertex) has no smooth normal."
            case .cap(let section): "Loft section \(section) needs 3 to 4096 points whose projection forms a simple polygon."
            }
        }
    }
}

private struct LoftSurface {
    private var points: [SIMD3<Double>] = []
    private var uvs: [SIMD2<Float>] = []
    var normalSums: [SIMD3<Double>] = []
    private var indices: [UInt32] = []

    init(vertexCount: Int, indexCount: Int) {
        points.reserveCapacity(vertexCount)
        uvs.reserveCapacity(vertexCount)
        normalSums.reserveCapacity(vertexCount)
        indices.reserveCapacity(indexCount)
    }

    mutating func appendVertex(_ point: SIMD3<Double>, uv: SIMD2<Float>) {
        points.append(point)
        uvs.append(uv)
        normalSums.append(.zero)
    }

    /// Each cross product adds its triangle's area-weighted normal to its corners.
    mutating func appendTriangle(_ a: Int, _ b: Int, _ c: Int) -> Bool {
        let normal = simd_cross(points[b] - points[a], points[c] - points[a])
        guard normal != .zero else { return false }
        normalSums[a] += normal
        normalSums[b] += normal
        normalSums[c] += normal
        indices.append(UInt32(a))
        indices.append(UInt32(b))
        indices.append(UInt32(c))
        return true
    }

    func unitNormal(at index: Int) throws -> SIMD3<Double> {
        let sum = normalSums[index]
        guard sum != .zero else { throw MeshGeometry.LoftError.undefinedNormal(index) }
        return simd_normalize(sum)
    }

    mutating func appendCap(
        _ section: [SIMD3<Float>], nextTo neighbor: [SIMD3<Float>], sectionIndex: Int
    ) throws {
        let failure = MeshGeometry.LoftError.cap(sectionIndex)
        let cap = section.map(SIMD3<Double>.init)
        let count = Double(cap.count)
        let centroid = cap.reduce(.zero, +) / count
        let neighborCenter = neighbor.reduce(SIMD3<Double>.zero) { $0 + SIMD3<Double>($1) } / count
        let centered = cap.map { $0 - centroid }
        var normal = SIMD3<Double>.zero
        for index in centered.indices {
            normal += simd_cross(centered[index], centered[(index + 1) % centered.count])
        }
        guard normal != .zero else { throw failure }
        normal = simd_normalize(normal)
        let towardInterior = simd_dot(normal, neighborCenter - centroid)
        if towardInterior > 0 || (towardInterior == 0 && sectionIndex == 0) { normal = -normal }

        let reference: SIMD3<Double> = abs(normal.x) > 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let bitangent = simd_normalize(simd_cross(normal, reference))
        let tangent = simd_cross(bitangent, normal)
        let contour = centered.map { SIMD2(simd_dot($0, tangent), simd_dot($0, bitangent)) }
        guard Set(contour).count == contour.count, !PlanarArrangement.hasCrossing([contour]) else { throw failure }
        let triangles: [SIMD3<Int>]
        do {
            triangles = try PolygonTriangulator.triangulate(outer: contour)
        } catch {
            throw failure
        }
        guard triangles.count == cap.count - 2 else { throw failure }

        let low = contour.reduce(contour[0], simd_min)
        let high = contour.reduce(contour[0], simd_max)
        let extent = high - low
        guard extent.x > 0, extent.y > 0 else { throw failure }
        let firstVertex = points.count
        for (point, xy) in zip(cap, contour) {
            let uv = SIMD2((xy.x - low.x) / extent.x, (high.y - xy.y) / extent.y)
            appendVertex(point, uv: SIMD2<Float>(uv))
        }
        for triangle in triangles {
            guard appendTriangle(firstVertex + triangle.x, firstVertex + triangle.y, firstVertex + triangle.z) else {
                throw failure
            }
        }
    }

    func geometry() throws -> MeshGeometry {
        let vertices = try points.indices.map { index in
            MeshVertex(
                position: SIMD3<Float>(points[index]), normal: SIMD3<Float>(try unitNormal(at: index)),
                textureCoordinate: uvs[index]
            )
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }
}
