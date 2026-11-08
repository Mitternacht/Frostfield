import simd

/// A per-vertex attribute of ``IndexedGeometry``, named in diagnostics.
public enum VertexAttribute: String, Sendable, Equatable {
    case position
    case normal
}

/// Why vertex and index data cannot form an ``IndexedGeometry``.
///
/// Each case carries the offending count, index, or vertex so the description
/// can say exactly what to fix. Geometry is checked when it is created, so a
/// renderer never uploads or draws data that could read outside its buffers.
public enum GeometryError: Error, Equatable, Sendable, CustomStringConvertible {
    /// There are no positions.
    case noVertices
    /// There are no indices.
    case noTriangles
    /// The index count is not a multiple of three.
    case incompleteTriangle(indexCount: Int)
    /// The index at `position` in the index array refers to a vertex that
    /// does not exist.
    case indexOutOfRange(position: Int, index: UInt32, vertexCount: Int)
    /// An attribute array does not have one element per vertex.
    case attributeCountMismatch(attribute: VertexAttribute, count: Int, vertexCount: Int)
    /// An attribute of a vertex contains NaN or infinity.
    case nonFiniteAttribute(attribute: VertexAttribute, vertex: Int)
    /// A vertex normal has zero length, so it has no direction.
    case zeroLengthNormal(vertex: Int)

    public var description: String {
        switch self {
        case .noVertices:
            "The geometry has no vertices. Supply at least three positions."
        case .noTriangles:
            "The geometry has no indices. Supply three indices per triangle."
        case .incompleteTriangle(let indexCount):
            "The index count (\(indexCount)) is not a multiple of 3: the last triangle has only "
                + "\(indexCount % 3) of its 3 indices."
        case .indexOutOfRange(let position, let index, let vertexCount):
            "Index \(index) at position \(position) (triangle \(position / 3)) is out of range: "
                + "the geometry has \(vertexCount) vertices, so every index must be less than \(vertexCount)."
        case .attributeCountMismatch(let attribute, let count, let vertexCount):
            "The geometry has \(vertexCount) positions but \(count) \(attribute)s. "
                + "Every vertex needs exactly one \(attribute)."
        case .nonFiniteAttribute(let attribute, let vertex):
            "The \(attribute) of vertex \(vertex) contains NaN or infinity."
        case .zeroLengthNormal(let vertex):
            "The normal of vertex \(vertex) has zero length. A normal must point away from the surface."
        }
    }
}

/// Indexed triangle geometry: model-space positions and normals with one
/// element per vertex, and three `UInt32` indices per triangle.
///
/// The initializer checks every index and attribute, so an
/// `IndexedGeometry` value is always safe to upload and draw. Triangles are
/// front-facing when their indices run counter-clockwise as seen from the side
/// their normals point to (see `docs/coordinates.md`).
///
/// The vertex layout is fixed: a position stream and a normal stream of
/// `SIMD3<Float>`. Mismatched attribute counts are the only way that layout
/// can be incompatible, and they are rejected here.
public struct IndexedGeometry: Sendable, Equatable {
    /// Model-space vertex positions.
    public let positions: [SIMD3<Float>]
    /// Model-space vertex normals, one per position. Shaders normalize them.
    public let normals: [SIMD3<Float>]
    /// Vertex indices, three per triangle.
    public let indices: [UInt32]

    /// The number of vertices.
    public var vertexCount: Int { positions.count }
    /// The number of triangles.
    public var triangleCount: Int { indices.count / 3 }

    /// - Throws: ``GeometryError`` describing the first problem found:
    ///   missing data, an incomplete triangle, an attribute count that does
    ///   not match the vertex count, a non-finite value, a zero-length normal,
    ///   or an index outside the vertices.
    public init(positions: [SIMD3<Float>], normals: [SIMD3<Float>], indices: [UInt32]) throws(GeometryError) {
        guard !positions.isEmpty else { throw .noVertices }
        guard !indices.isEmpty else { throw .noTriangles }
        guard indices.count.isMultiple(of: 3) else { throw .incompleteTriangle(indexCount: indices.count) }
        guard normals.count == positions.count else {
            throw .attributeCountMismatch(attribute: .normal, count: normals.count, vertexCount: positions.count)
        }
        for (vertex, position) in positions.enumerated() where !Self.isFinite(position) {
            throw .nonFiniteAttribute(attribute: .position, vertex: vertex)
        }
        for (vertex, normal) in normals.enumerated() {
            guard Self.isFinite(normal) else { throw .nonFiniteAttribute(attribute: .normal, vertex: vertex) }
            guard simd_length_squared(normal) > 0 else { throw .zeroLengthNormal(vertex: vertex) }
        }
        let vertexCount = positions.count
        if let position = indices.firstIndex(where: { Int($0) >= vertexCount }) {
            throw .indexOutOfRange(position: position, index: indices[position], vertexCount: vertexCount)
        }
        self.positions = positions
        self.normals = normals
        self.indices = indices
    }

    private static func isFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }
}

extension IndexedGeometry {
    /// An axis-aligned cube centred on the origin with edges of length
    /// `size`.
    ///
    /// Each face has its own four vertices, so every vertex carries its
    /// face's outward normal (24 vertices, 12 triangles). Every triangle runs
    /// counter-clockwise seen from outside the cube.
    ///
    /// - Precondition: `size` is finite and greater than zero.
    public static func cube(size: Float = 1) -> IndexedGeometry {
        precondition(size.isFinite && size > 0, "A cube needs a finite, positive size; got \(size)")
        let half = size / 2
        // For each face: its outward normal n and two in-face axes u and v
        // with u × v = n, so (-u-v, u-v, u+v, -u+v) runs counter-clockwise
        // seen from outside.
        let faces: [(n: SIMD3<Float>, u: SIMD3<Float>, v: SIMD3<Float>)] = [
            ([1, 0, 0], [0, 0, -1], [0, 1, 0]),
            ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
            ([0, 1, 0], [1, 0, 0], [0, 0, -1]),
            ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
            ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
            ([0, 0, -1], [-1, 0, 0], [0, 1, 0]),
        ]
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        positions.reserveCapacity(24)
        normals.reserveCapacity(24)
        indices.reserveCapacity(36)
        for face in faces {
            let first = UInt32(positions.count)
            let centre = face.n * half
            for (a, b) in [(-1, -1), (1, -1), (1, 1), (-1, 1)] as [(Float, Float)] {
                positions.append(centre + (a * face.u + b * face.v) * half)
                normals.append(face.n)
            }
            indices += [first, first + 1, first + 2, first, first + 2, first + 3]
        }
        do {
            return try IndexedGeometry(positions: positions, normals: normals, indices: indices)
        } catch {
            preconditionFailure("The generated cube is invalid: \(error)")
        }
    }
}
