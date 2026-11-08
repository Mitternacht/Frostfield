import Testing
import simd
@testable import Frostfield

/// The procedural cube: counts, extent, normals, and winding.
struct CubeGeometryTests {
    let cube = IndexedGeometry.cube(size: 2)

    @Test func `The cube has four vertices per face and two triangles per face`() {
        #expect(cube.vertexCount == 24)
        #expect(cube.triangleCount == 12)
        #expect(cube.indices.count == 36)
    }

    @Test func `The cube spans its size around the origin`() {
        let minimum = cube.positions.reduce(SIMD3<Float>(repeating: .infinity)) { simd_min($0, $1) }
        let maximum = cube.positions.reduce(SIMD3<Float>(repeating: -.infinity)) { simd_max($0, $1) }
        #expect(minimum == SIMD3(repeating: -1))
        #expect(maximum == SIMD3(repeating: 1))
    }

    /// Each of the six axis directions is the normal of exactly one face, and
    /// every vertex lies on the face its normal names.
    @Test func `Normals are outward unit axes, one direction per face`() {
        var counts: [SIMD3<Float>: Int] = [:]
        for (position, normal) in zip(cube.positions, cube.normals) {
            #expect(simd_length(normal) == 1)
            #expect(simd_dot(position, normal) == 1, "vertex \(position) is not on the face with normal \(normal)")
            counts[normal, default: 0] += 1
        }
        #expect(counts.count == 6)
        #expect(counts.values.allSatisfy { $0 == 4 })
    }

    /// Seen from outside, each triangle runs counter-clockwise: the right-hand
    /// cross product of its edges points the same way as its vertex normals,
    /// away from the cube's centre.
    @Test func `Every triangle winds counter-clockwise seen from outside`() {
        for triangle in 0..<cube.triangleCount {
            let corners = (0..<3).map { Int(cube.indices[triangle * 3 + $0]) }
            let a = cube.positions[corners[0]], b = cube.positions[corners[1]], c = cube.positions[corners[2]]
            let faceNormal = simd_normalize(simd_cross(b - a, c - a))
            for corner in corners {
                #expect(simd_distance(faceNormal, cube.normals[corner]) < 1e-6, "triangle \(triangle)")
            }
            #expect(simd_dot(faceNormal, (a + b + c) / 3) > 0, "triangle \(triangle) faces inward")
        }
    }

    @Test(arguments: [Float(0.5), 1, 3])
    func `The size scales every position`(size: Float) {
        let scaled = IndexedGeometry.cube(size: size)
        #expect(scaled.positions == IndexedGeometry.cube(size: 1).positions.map { $0 * size })
        #expect(scaled.indices == cube.indices)
        #expect(scaled.normals == cube.normals)
    }
}

/// Invalid vertex or index data is rejected when the geometry is created,
/// with a diagnostic naming the offending element, before any GPU buffer
/// exists.
struct GeometryValidationTests {
    let triangle: [SIMD3<Float>] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    let up: [SIMD3<Float>] = Array(repeating: [0, 0, 1], count: 3)

    @Test func `Valid geometry keeps its data`() throws {
        let geometry = try IndexedGeometry(positions: triangle, normals: up, indices: [0, 1, 2])
        #expect(geometry.positions == triangle)
        #expect(geometry.normals == up)
        #expect(geometry.indices == [0, 1, 2])
        #expect(geometry.triangleCount == 1)
    }

    @Test func `Geometry without vertices or indices is rejected`() {
        #expect(throws: GeometryError.noVertices) {
            try IndexedGeometry(positions: [], normals: [], indices: [0, 1, 2])
        }
        #expect(throws: GeometryError.noTriangles) {
            try IndexedGeometry(positions: triangle, normals: up, indices: [])
        }
    }

    @Test func `An incomplete triangle is rejected`() {
        let error = #expect(throws: GeometryError.incompleteTriangle(indexCount: 4)) {
            try IndexedGeometry(positions: triangle, normals: up, indices: [0, 1, 2, 0])
        }
        #expect(error?.description.contains("only 1 of its 3 indices") == true)
    }

    /// The first index that names a missing vertex is reported with its
    /// position, its triangle, and the valid range.
    @Test(arguments: [
        ([UInt32(0), 1, 2, 0, 2, 3], 5, UInt32(3)),
        ([UInt32(0), 1, 2, 2, 1, .max], 5, UInt32.max),
        ([UInt32(7), 1, 2], 0, UInt32(7)),
    ])
    func `An index outside the vertices is rejected`(indices: [UInt32], position: Int, index: UInt32) {
        let error = #expect(throws: GeometryError.indexOutOfRange(position: position, index: index, vertexCount: 3)) {
            try IndexedGeometry(positions: triangle, normals: up, indices: indices)
        }
        #expect(error?.description.contains("triangle \(position / 3)") == true)
        #expect(error?.description.contains("must be less than 3") == true)
    }

    @Test(arguments: [0, 2, 4])
    func `A normal count that differs from the vertex count is rejected`(normalCount: Int) {
        let normals = Array(repeating: SIMD3<Float>(0, 0, 1), count: normalCount)
        let error = #expect(throws: GeometryError.attributeCountMismatch(attribute: .normal, count: normalCount, vertexCount: 3)) {
            try IndexedGeometry(positions: triangle, normals: normals, indices: [0, 1, 2])
        }
        #expect(error?.description.contains("3 positions but \(normalCount) normals") == true)
    }

    @Test func `Non-finite positions and normals are rejected`() {
        var positions = triangle
        positions[1].y = .nan
        #expect(throws: GeometryError.nonFiniteAttribute(attribute: .position, vertex: 1)) {
            try IndexedGeometry(positions: positions, normals: up, indices: [0, 1, 2])
        }
        var normals = up
        normals[2].x = .infinity
        #expect(throws: GeometryError.nonFiniteAttribute(attribute: .normal, vertex: 2)) {
            try IndexedGeometry(positions: triangle, normals: normals, indices: [0, 1, 2])
        }
    }

    @Test func `A zero-length normal is rejected`() {
        var normals = up
        normals[0] = .zero
        let error = #expect(throws: GeometryError.zeroLengthNormal(vertex: 0)) {
            try IndexedGeometry(positions: triangle, normals: normals, indices: [0, 1, 2])
        }
        #expect(error?.description.contains("vertex 0") == true)
    }
}
