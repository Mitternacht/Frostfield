import simd

struct MarchingCubeExtractor {
    private struct Face {
        let corners: [Int]
        let edges: [Int]
        let axis: Int
        let high: Bool
        let reversed: Bool
    }

    // Neighboring cells enumerate the same world face in the same order.
    // Reversed faces point inward in that order, so their contours reverse.
    private static let faces = [
        Face(corners: [0, 1, 2, 3], edges: [0, 1, 2, 3], axis: 2, high: false, reversed: true),
        Face(corners: [4, 5, 6, 7], edges: [4, 5, 6, 7], axis: 2, high: true, reversed: false),
        Face(corners: [0, 1, 5, 4], edges: [0, 9, 4, 8], axis: 1, high: false, reversed: false),
        Face(corners: [3, 2, 6, 7], edges: [2, 10, 6, 11], axis: 1, high: true, reversed: true),
        Face(corners: [0, 3, 7, 4], edges: [3, 11, 7, 8], axis: 0, high: false, reversed: true),
        Face(corners: [1, 2, 6, 5], edges: [1, 10, 5, 9], axis: 0, high: true, reversed: false),
    ]
    private static let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    private struct Vertex {
        let position: SIMD3<Float>
        let gradient: SIMD3<Double>
        let boundary: UInt8
        var faceNormal: SIMD3<Double> = .zero
    }

    private let grid: MarchingCubes
    private let level: Double
    private let maximumTriangles: Int
    private let strides: SIMD3<Int>
    private let cornerOffsets: [Int]
    private var vertices: [Vertex] = []
    private var vertexColors: [SIMD3<Float>]?
    private var indices: [UInt32] = []
    private var edgeVertices: [UInt64: UInt32] = [:]

    init(grid: MarchingCubes, level: Double, maximumTriangles: Int) {
        self.grid = grid
        self.level = level
        self.maximumTriangles = maximumTriangles
        vertexColors = grid.palette.map { _ in [] }
        let row = grid.resolution.x, plane = row * grid.resolution.y
        strides = SIMD3(1, row, plane)
        cornerOffsets = [0, 1, row + 1, row, plane, plane + 1, plane + row + 1, plane + row]
    }

    mutating func mesh(boundary: MarchingCubes.Boundary) throws -> MeshGeometry? {
        let resolution = grid.resolution
        if boundary == .requireClosed {
            for index in grid.values.indices where Double(grid.values[index]) > level {
                guard boundaryMask(coordinates(index)) == 0 else { throw MarchingCubes.Failure.openSurface }
            }
        }
        for z in 0..<(resolution.z - 1) {
            for y in 0..<(resolution.y - 1) {
                for x in 0..<(resolution.x - 1) {
                    let origin = SIMD3(x, y, z)
                    let base = index(origin)
                    let corners = cornerOffsets.map { base + $0 }
                    let values = corners.map { Double(grid.values[$0]) - level }
                    guard values.contains(where: { $0 > 0 }), values.contains(where: { $0 <= 0 }) else { continue }
                    try polygonize(origin: origin, corners: corners, values: values)
                }
            }
        }
        guard !indices.isEmpty else { return nil }
        try validateTopology(boundary: boundary)
        return try geometry()
    }

    private mutating func polygonize(origin: SIMD3<Int>, corners: [Int], values: [Double]) throws {
        let inside = values.map { $0 > 0 }
        var next = [Int](repeating: -1, count: 12)
        for face in Self.faces {
            let crossings = face.corners.indices.filter { inside[face.corners[$0]] != inside[face.corners[($0 + 1) % 4]] }
            func connect(_ a: Int, _ b: Int) {
                let forward = inside[face.corners[a]] == face.reversed
                next[face.edges[forward ? a : b]] = face.edges[forward ? b : a]
            }
            if crossings.count == 2 {
                connect(crossings[0], crossings[1])
            } else if crossings.count == 4 {
                let c = face.corners
                // Nielson and Hamann's bilinear face decider. The canonical
                // face order also breaks an exact saddle tie consistently.
                let determinant = values[c[0]] * values[c[2]] - values[c[1]] * values[c[3]]
                if determinant >= 0 {
                    connect(0, 1)
                    connect(2, 3)
                } else {
                    connect(3, 0)
                    connect(1, 2)
                }
            } else if face.corners.allSatisfy({ values[$0] == 0 }) {
                // Above-level samples on both sides of an exact-level face
                // can make two coincident sheets. Reject that singular case.
                let axis = face.axis
                let hasNeighbor = face.high ? origin[axis] + 2 < grid.resolution[axis] : origin[axis] > 0
                if hasNeighbor {
                    let offset = face.high ? strides[axis] : -strides[axis]
                    if face.corners.contains(where: { Double(grid.values[corners[$0] + offset]) > level }) {
                        throw MarchingCubes.Failure.singularSurface
                    }
                }
            }
        }

        for start in next.indices where next[start] >= 0 {
            var loop: [UInt32] = []
            var edge = start
            repeat {
                guard next[edge] >= 0 else { throw MarchingCubes.Failure.singularSurface }
                let endpoints = Self.edges[edge]
                let vertex = try crossing(corners[endpoints.0], corners[endpoints.1], values[endpoints.0], values[endpoints.1])
                if loop.last != vertex { loop.append(vertex) }
                let following = next[edge]
                next[edge] = -1
                edge = following
            } while edge != start
            if loop.count > 1, loop.first == loop.last { loop.removeLast() }
            try emit(loop)
        }
    }

    private mutating func crossing(_ a: Int, _ b: Int, _ first: Double, _ second: Double) throws -> UInt32 {
        let start = second == 0 ? b : a
        let end = first == 0 ? a : b
        let key = Self.edgeKey(UInt32(start), UInt32(end))
        if let cached = edgeVertices[key] { return cached }
        let low = coordinates(start), high = coordinates(end)
        let t = start == end ? 0 : first / (first - second)
        let p = position(low) * (1 - t) + position(high) * t
        let gradient = outwardGradient(low) * (1 - t) + outwardGradient(high) * t
        let color = grid.palette.map {
            SIMD3<Float>(SIMD3<Double>($0[start]) * (1 - t) + SIMD3<Double>($0[end]) * t)
        }
        let vertex = try append(
            Vertex(position: SIMD3<Float>(p), gradient: gradient, boundary: boundaryMask(low) & boundaryMask(high)),
            color: color
        )
        edgeVertices[key] = vertex
        return vertex
    }

    private mutating func emit(_ loop: [UInt32]) throws {
        guard loop.count >= 3 else { return }
        guard Set(loop).count == loop.count else { throw MarchingCubes.Failure.singularSurface }
        if loop.count == 3 {
            try triangle(loop[0], loop[1], loop[2])
            return
        }
        // One disk per contour loop. The center stays in the cell and the
        // triangulation does not insert extra edges on its shared faces.
        var position = SIMD3<Double>.zero, gradient = SIMD3<Double>.zero
        var color = SIMD3<Double>.zero
        var boundary = UInt8.max
        for index in loop {
            let vertex = vertices[Int(index)]
            position += SIMD3<Double>(vertex.position)
            gradient += vertex.gradient
            boundary &= vertex.boundary
            if let rgb = vertexColors?[Int(index)] { color += SIMD3<Double>(rgb) }
        }
        let count = Double(loop.count)
        let center = try append(
            Vertex(position: SIMD3<Float>(position / count), gradient: gradient, boundary: boundary),
            color: vertexColors.map { _ in SIMD3<Float>(color / count) }
        )
        for index in loop.indices {
            try triangle(loop[index], loop[(index + 1) % loop.count], center)
        }
    }

    private mutating func append(_ vertex: Vertex, color: SIMD3<Float>?) throws -> UInt32 {
        // Bound unused vertices from collapsed contours as well as emitted triangles.
        guard vertices.count < maximumTriangles * 3 + 12 else { throw MarchingCubes.Failure.triangleLimit }
        let index = UInt32(vertices.count)
        vertices.append(vertex)
        if let color { vertexColors?.append(color) }
        return index
    }

    private mutating func triangle(_ a: UInt32, _ b: UInt32, _ c: UInt32) throws {
        let start = SIMD3<Double>(vertices[Int(a)].position)
        let area = simd_cross(SIMD3<Double>(vertices[Int(b)].position) - start, SIMD3<Double>(vertices[Int(c)].position) - start)
        guard area != .zero else { return }
        guard indices.count / 3 < maximumTriangles else { throw MarchingCubes.Failure.triangleLimit }
        indices.append(a)
        indices.append(b)
        indices.append(c)
        vertices[Int(a)].faceNormal += area
        vertices[Int(b)].faceNormal += area
        vertices[Int(c)].faceNormal += area
    }

    private func coordinates(_ index: Int) -> SIMD3<Int> {
        SIMD3(index % strides.y, (index / strides.y) % grid.resolution.y, index / strides.z)
    }

    private func index(_ coordinates: SIMD3<Int>) -> Int {
        coordinates.x + coordinates.y * strides.y + coordinates.z * strides.z
    }

    private func position(_ coordinates: SIMD3<Int>) -> SIMD3<Double> {
        SIMD3(grid.axes[0][coordinates.x], grid.axes[1][coordinates.y], grid.axes[2][coordinates.z])
    }

    private func outwardGradient(_ coordinates: SIMD3<Int>) -> SIMD3<Double> {
        var result = SIMD3<Double>.zero
        for axis in 0..<3 {
            var low = coordinates, high = coordinates
            low[axis] = max(0, low[axis] - 1)
            high[axis] = min(grid.resolution[axis] - 1, high[axis] + 1)
            result[axis] = (Double(grid.values[index(low)]) - Double(grid.values[index(high)]))
                / (grid.axes[axis][high[axis]] - grid.axes[axis][low[axis]])
        }
        return result
    }

    private func boundaryMask(_ coordinates: SIMD3<Int>) -> UInt8 {
        var mask: UInt8 = 0
        for axis in 0..<3 {
            if coordinates[axis] == 0 { mask |= 1 << (2 * axis) }
            if coordinates[axis] == grid.resolution[axis] - 1 { mask |= 1 << (2 * axis + 1) }
        }
        return mask
    }

    private static func edgeKey(_ a: UInt32, _ b: UInt32) -> UInt64 {
        (UInt64(min(a, b)) << 32) | UInt64(max(a, b))
    }

    private struct EdgeUse {
        let corner: Int
        var paired = false
    }

    private func validateTopology(boundary: MarchingCubes.Boundary) throws {
        var edges: [UInt64: EdgeUse] = [:]
        var links = CornerLinks(count: indices.count)
        func following(_ corner: Int) -> Int { corner % 3 == 2 ? corner - 2 : corner + 1 }
        for corner in indices.indices {
            let next = following(corner)
            let edge = Self.edgeKey(indices[corner], indices[next])
            if var use = edges[edge] {
                guard !use.paired, indices[use.corner] == indices[next] else { throw MarchingCubes.Failure.singularSurface }
                links.join(corner, following(use.corner))
                links.join(next, use.corner)
                use.paired = true
                edges[edge] = use
            } else {
                edges[edge] = EdgeUse(corner: corner)
            }
        }
        for use in edges.values where !use.paired {
            let a = vertices[Int(indices[use.corner])].boundary
            let b = vertices[Int(indices[following(use.corner)])].boundary
            guard a & b != 0 else { throw MarchingCubes.Failure.singularSurface }
            guard boundary == .open else { throw MarchingCubes.Failure.openSurface }
        }
        // A vertex's incident triangles must be a single fan, even if every
        // individual edge has a valid number of incident triangles.
        var fans = [Int](repeating: -1, count: vertices.count)
        for corner in indices.indices {
            let vertex = Int(indices[corner])
            let fan = links.root(corner)
            guard fans[vertex] == -1 || fans[vertex] == fan else { throw MarchingCubes.Failure.singularSurface }
            fans[vertex] = fan
        }
    }

    private struct CornerLinks {
        var parents: [Int]

        init(count: Int) { parents = Array(0..<count) }

        mutating func root(_ index: Int) -> Int {
            var node = index
            while parents[node] != node {
                parents[node] = parents[parents[node]]
                node = parents[node]
            }
            return node
        }

        mutating func join(_ a: Int, _ b: Int) {
            let first = root(a), second = root(b)
            parents[max(first, second)] = min(first, second)
        }
    }

    private func geometry() throws -> MeshGeometry {
        let origin = SIMD3<Double>(grid.bounds.minimum)
        let extent = SIMD3<Double>(grid.bounds.maximum) - origin
        var remap = [UInt32](repeating: .max, count: vertices.count)
        var meshVertices: [MeshVertex] = []
        let meshIndices = try indices.map { index in
            let slot = Int(index)
            if remap[slot] == .max {
                let vertex = vertices[slot]
                let direction = vertex.gradient == .zero ? vertex.faceNormal : vertex.gradient
                guard direction != .zero else { throw MarchingCubes.Failure.singularSurface }
                let p = (SIMD3<Double>(vertex.position) - origin) / extent
                remap[slot] = UInt32(meshVertices.count)
                meshVertices.append(MeshVertex(
                    position: vertex.position, normal: SIMD3<Float>(simd_normalize(direction)),
                    textureCoordinate: SIMD2(Float(p.x), Float(p.z)),
                    color: vertexColors.map { SIMD4($0[slot], 1) }
                ))
            }
            return remap[slot]
        }
        return try MeshGeometry(vertices: meshVertices, indices: meshIndices)
    }
}
