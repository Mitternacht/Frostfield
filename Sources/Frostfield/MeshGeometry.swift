import Foundation
import simd

/// A shader texture's source coordinates on a mesh.
public enum MeshTextureCoordinateSet: UInt32, Hashable, Sendable {
    case primary = 0
    case secondary = 1
}

/// A position, surface normal, and normalized top-left texture coordinates.
public struct MeshVertex: Sendable {
    public let position: SIMD3<Float>
    public let normal: SIMD3<Float>
    public let textureCoordinate: SIMD2<Float>
    public let secondaryTextureCoordinate: SIMD2<Float>?
    /// An optional unit tangent and a handedness sign of −1 or +1.
    public let tangent: SIMD4<Float>?
    /// An optional finite linear RGBA color, like glTF `COLOR_0` and Three.js
    /// `vertexColors`. Renderers multiply it into the base color and alpha. When any
    /// vertex of a mesh has a color, vertices without one use opaque white.
    public let color: SIMD4<Float>?

    public init(
        position: SIMD3<Float>, normal: SIMD3<Float>, textureCoordinate: SIMD2<Float> = .zero,
        secondaryTextureCoordinate: SIMD2<Float>? = nil, tangent: SIMD4<Float>? = nil,
        color: SIMD4<Float>? = nil
    ) {
        self.position = position
        self.normal = normal
        self.textureCoordinate = textureCoordinate
        self.secondaryTextureCoordinate = secondaryTextureCoordinate
        self.tangent = tangent
        self.color = color
    }
}

/// A contiguous range of a mesh's triangles that draws with one material, like a
/// Three.js `BufferGeometry` group. Ranges count indices and start on a triangle.
public struct MeshGeometryGroup: Hashable, Sendable {
    /// The first index of the group; a multiple of three.
    public let start: Int
    /// The number of indices; a positive multiple of three.
    public let count: Int
    /// The material slot that draws the group.
    public let materialIndex: Int

    public init(start: Int, count: Int, materialIndex: Int = 0) {
        self.start = start
        self.count = count
        self.materialIndex = materialIndex
    }

    /// The group's index range. A group that ``MeshGeometry/withGroups(_:)`` would
    /// reject still yields a range: empty when `count` is not positive, and ending at
    /// `Int.max` when the sum overflows, so ``MeshGeometry/subset(_:)`` throws instead
    /// of trapping.
    public var range: Range<Int> {
        guard count > 0 else { return start..<start }
        let (end, overflow) = start.addingReportingOverflow(count)
        return start..<(overflow ? Int.max : end)
    }
}

/// Indexed triangles in a right-handed coordinate system.
public struct MeshGeometry: Sendable {
    public let vertices: [MeshVertex]
    /// Each group of three indices is a counterclockwise triangle viewed from its front.
    public let indices: [UInt32]
    /// Material groups, in index order. Empty means one material draws every triangle.
    /// Renderers draw all triangles with their single material; use
    /// ``SceneNode/grouped(_:materials:device:colorPixelFormat:depthMode:sampleCount:renderStates:id:transform:)``
    /// to draw each group with its own material.
    public private(set) var groups: [MeshGeometryGroup] = []
    /// Whether any vertex carries a color.
    public var hasVertexColors: Bool { vertices.contains { $0.color != nil } }

    public init(vertices: [MeshVertex], indices: [UInt32]) throws {
        guard !vertices.isEmpty else { throw ValidationError.emptyVertices }
        guard !indices.isEmpty, indices.count.isMultiple(of: 3) else {
            throw ValidationError.invalidTriangleIndexCount
        }

        for (index, vertex) in vertices.enumerated() {
            guard Self.isFinite(vertex.position), Self.isFinite(vertex.normal), vertex.normal != .zero,
                  vertex.textureCoordinate.x.isFinite, vertex.textureCoordinate.y.isFinite,
                  vertex.secondaryTextureCoordinate.map({ $0.x.isFinite && $0.y.isFinite }) != false,
                  vertex.tangent.map(Self.isValidTangent) != false,
                  vertex.color.map({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) != false else {
                throw ValidationError.invalidVertex(index)
            }
        }

        if let invalidIndex = indices.first(where: { Int($0) >= vertices.count }) {
            throw ValidationError.indexOutOfBounds(invalidIndex)
        }

        self.init(validatedVertices: vertices, indices: indices)
    }

    private init(validatedVertices: [MeshVertex], indices: [UInt32]) {
        vertices = validatedVertices
        self.indices = indices
    }

    func replacingVertices(_ vertices: [MeshVertex]) throws -> MeshGeometry {
        var geometry = try MeshGeometry(vertices: vertices, indices: indices)
        geometry.groups = groups
        return geometry
    }

    func replacingValidatedPositions(_ changes: [(index: Int, position: SIMD3<Float>)]) -> MeshGeometry {
        var updated = vertices
        for (index, position) in changes {
            let vertex = vertices[index]
            updated[index] = MeshVertex(
                position: position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.secondaryTextureCoordinate, tangent: vertex.tangent,
                color: vertex.color
            )
        }
        var result = MeshGeometry(validatedVertices: updated, indices: indices)
        result.groups = groups
        return result
    }

    /// Retains vertex identity for skins and morphs when selecting draw indices.
    func restrictingIndices(to range: Range<Int>) -> MeshGeometry {
        range.count == indices.count ? self : MeshGeometry(validatedVertices: vertices, indices: Array(indices[range]))
    }

    /// Rebuilds smooth normals by summing each incident face's area-weighted
    /// normal. Only shared indices smooth together. Unused vertices and zero
    /// sums keep the direction of the authored normal.
    ///
    /// Vertex identities, index order, groups, colors and UVs are preserved.
    /// Tangents are cleared so they can be regenerated against the new normals.
    public func generatingVertexNormals() -> MeshGeometry {
        var sums = [SIMD3<Double>](repeating: .zero, count: vertices.count)
        for offset in stride(from: 0, to: indices.count, by: 3) {
            let a = Int(indices[offset]), b = Int(indices[offset + 1]), c = Int(indices[offset + 2])
            let origin = SIMD3<Double>(vertices[a].position)
            let edge1 = SIMD3<Double>(vertices[b].position) - origin
            let edge2 = SIMD3<Double>(vertices[c].position) - origin
            let face = simd_cross(edge1, edge2)
            sums[a] += face
            sums[b] += face
            sums[c] += face
        }
        let updated = zip(vertices, sums).map { vertex, sum in
            let normal = simd_normalize(sum == .zero ? SIMD3<Double>(vertex.normal) : sum)
            return MeshVertex(
                position: vertex.position, normal: SIMD3<Float>(normal), textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.secondaryTextureCoordinate, color: vertex.color
            )
        }
        var result = MeshGeometry(validatedVertices: updated, indices: indices)
        result.groups = groups
        return result
    }

    /// The same geometry with material groups. Groups must start on a triangle, hold
    /// whole triangles, stay inside the index array, use nonnegative material indices,
    /// and not overlap. Triangles outside every group are not drawn by grouped nodes.
    public func withGroups(_ groups: [MeshGeometryGroup]) throws -> MeshGeometry {
        var end = 0
        // Check in start order; errors name the group's position in the supplied array.
        for index in groups.indices.sorted(by: { groups[$0].start < groups[$1].start }) {
            let group = groups[index]
            guard group.start >= end, group.start.isMultiple(of: 3), group.count > 0, group.count.isMultiple(of: 3),
                  group.count <= indices.count - group.start, group.materialIndex >= 0 else {
                throw ValidationError.invalidGroup(index)
            }
            end = group.start + group.count
        }
        var copy = self
        copy.groups = groups.sorted { $0.start < $1.start }
        return copy
    }

    /// A compact geometry holding the triangles in an index range, with only the
    /// vertices they use, in first-use order. The range must hold whole triangles.
    public func subset(_ range: Range<Int>) throws -> MeshGeometry {
        guard range.lowerBound >= 0, range.upperBound <= indices.count, !range.isEmpty,
              range.lowerBound.isMultiple(of: 3), range.count.isMultiple(of: 3) else {
            throw ValidationError.invalidTriangleIndexCount
        }
        var remap = [UInt32: UInt32]()
        var compactVertices: [MeshVertex] = []
        var compactIndices: [UInt32] = []
        compactIndices.reserveCapacity(range.count)
        for index in indices[range] {
            if let mapped = remap[index] {
                compactIndices.append(mapped)
            } else {
                let mapped = UInt32(compactVertices.count)
                remap[index] = mapped
                compactVertices.append(vertices[Int(index)])
                compactIndices.append(mapped)
            }
        }
        return MeshGeometry(validatedVertices: compactVertices, indices: compactIndices)
    }

    /// A unit cube centered at the origin with a separate normal for each face.
    public static func cube() -> MeshGeometry {
        // Each right/up pair has a cross product that points out of the cube.
        let faces: [(normal: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>)] = [
            (SIMD3(1, 0, 0), SIMD3(0, 0, -1), SIMD3(0, 1, 0)),
            (SIMD3(-1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 0)),
            (SIMD3(0, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, -1)),
            (SIMD3(0, -1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 1)),
            (SIMD3(0, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 1, 0)),
            (SIMD3(0, 0, -1), SIMD3(-1, 0, 0), SIMD3(0, 1, 0)),
        ]

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for (normal, right, up) in faces {
            let center = normal * 0.5
            let halfRight = right * 0.5
            let halfUp = up * 0.5
            let base = UInt32(vertices.count)

            vertices.append(MeshVertex(position: center - halfRight - halfUp, normal: normal, textureCoordinate: SIMD2(0, 1)))
            vertices.append(MeshVertex(position: center + halfRight - halfUp, normal: normal, textureCoordinate: SIMD2(1, 1)))
            vertices.append(MeshVertex(position: center + halfRight + halfUp, normal: normal, textureCoordinate: SIMD2(1, 0)))
            vertices.append(MeshVertex(position: center - halfRight + halfUp, normal: normal, textureCoordinate: SIMD2(0, 0)))
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
        }

        return MeshGeometry(validatedVertices: vertices, indices: indices)
    }

    static func isValidTangent(_ tangent: SIMD4<Float>) -> Bool {
        let direction = SIMD3(tangent.x, tangent.y, tangent.z)
        return (tangent.w == 1 || tangent.w == -1) && abs(simd_length_squared(direction) - 1) <= 0.0001
    }

    private static func isFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    public enum ValidationError: LocalizedError, Equatable {
        case emptyVertices
        case invalidTriangleIndexCount
        case invalidVertex(Int)
        case indexOutOfBounds(UInt32)
        case invalidGroup(Int)

        public var errorDescription: String? {
            switch self {
            case .emptyVertices:
                "A mesh must contain at least one vertex."
            case .invalidTriangleIndexCount:
                "A mesh must contain at least one triangle and three indices per triangle."
            case .invalidVertex(let index):
                "Mesh vertex \(index) needs a finite position and texture coordinates, a finite nonzero normal, and a valid tangent and finite color when supplied."
            case .indexOutOfBounds(let index):
                "Mesh index \(index) is outside the vertex array."
            case .invalidGroup(let index):
                "Mesh group \(index) must hold whole triangles inside the index array, use a nonnegative material index, and not overlap another group."
            }
        }
    }
}
