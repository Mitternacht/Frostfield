import Foundation
import simd

extension MeshGeometry {
    /// Bakes a finite, invertible affine matrix into the mesh. Normals use the
    /// inverse transpose. Mirrored transforms preserve front faces by reversing
    /// winding and flipping tangent handedness. Material groups remain intact.
    public func transformed(by transform: simd_float4x4) throws -> MeshGeometry {
        let transformed = try Self.merged([.init(self, transform: transform)]).geometry
        return try transformed.withGroups(groups)
    }

    /// Moves the center of the geometry's axis-aligned bounds to the origin.
    public func centered() throws -> MeshGeometry {
        var translation = matrix_identity_float4x4
        translation.columns.3 = SIMD4(-bounds.center, 1)
        return try transformed(by: translation)
    }

    /// One geometry and the transform that places it in a merged mesh.
    public struct MergePart: Sendable {
        public let geometry: MeshGeometry
        /// An affine transform with a nonzero determinant.
        public let transform: simd_float4x4

        public init(_ geometry: MeshGeometry, transform: simd_float4x4 = matrix_identity_float4x4) {
            self.geometry = geometry
            self.transform = transform
        }
    }

    /// A merged mesh and, for each part, the range of `geometry.indices` that holds its triangles.
    public struct Merged: Sendable {
        public let geometry: MeshGeometry
        public let indexRanges: [Range<Int>]
    }

    /// Bakes each part's transform into its vertices and concatenates the parts in order,
    /// like Three.js `BufferGeometryUtils.mergeGeometries` with groups.
    ///
    /// Normals use the inverse transpose and are renormalized. Tangent directions use the
    /// transform and are renormalized. A mirroring transform (negative determinant)
    /// reverses each triangle's winding and negates tangent handedness, so front faces
    /// and normal maps survive. UVs and colors are copied. Input material
    /// groups are flattened. When `useGroups` is true, each part contributes
    /// one group whose material slot is its index in `parts`.
    public static func merged(_ parts: [MergePart], useGroups: Bool = false) throws -> Merged {
        guard !parts.isEmpty else { throw MergeFailure.noParts }
        var vertexCount = 0
        var indexCount = 0
        for part in parts {
            guard part.geometry.vertices.count <= Int(UInt32.max) - vertexCount else {
                throw MergeFailure.tooManyVertices
            }
            let (combinedIndexCount, overflow) = indexCount.addingReportingOverflow(part.geometry.indices.count)
            guard !overflow else { throw MergeFailure.indexCountOverflow }
            vertexCount += part.geometry.vertices.count
            indexCount = combinedIndexCount
        }

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        var ranges: [Range<Int>] = []
        vertices.reserveCapacity(vertexCount)
        indices.reserveCapacity(indexCount)
        ranges.reserveCapacity(parts.count)
        for (partIndex, part) in parts.enumerated() {
            let m = part.transform
            let row3 = SIMD4(m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w)
            let linear = simd_double3x3(
                SIMD3<Double>(Double(m.columns.0.x), Double(m.columns.0.y), Double(m.columns.0.z)),
                SIMD3<Double>(Double(m.columns.1.x), Double(m.columns.1.y), Double(m.columns.1.z)),
                SIMD3<Double>(Double(m.columns.2.x), Double(m.columns.2.y), Double(m.columns.2.z))
            )
            let translation = SIMD3<Double>(Double(m.columns.3.x), Double(m.columns.3.y), Double(m.columns.3.z))
            let determinant = linear.determinant
            guard row3 == SIMD4(0, 0, 0, 1), determinant.isFinite, determinant != 0,
                  (0..<4).allSatisfy({ column in (0..<4).allSatisfy { m[column][$0].isFinite } }) else {
                throw MergeFailure.invalidTransform(part: partIndex)
            }
            let normalMatrix = linear.inverse.transpose
            let mirrored = determinant < 0
            let offset = UInt32(vertices.count)
            for vertex in part.geometry.vertices {
                let position = SIMD3<Float>(linear * SIMD3<Double>(vertex.position) + translation)
                let normal = normalMatrix * SIMD3<Double>(vertex.normal)
                let tangent = vertex.tangent.map { t -> SIMD4<Float> in
                    let direction = linear * SIMD3<Double>(Double(t.x), Double(t.y), Double(t.z))
                    return SIMD4(SIMD3<Float>(simd_normalize(direction)), mirrored ? -t.w : t.w)
                }
                let unitNormal = SIMD3<Float>(simd_normalize(normal))
                guard position.isFiniteVector, unitNormal.isFiniteVector,
                      tangent.map({ SIMD3($0.x, $0.y, $0.z).isFiniteVector }) != false else {
                    throw MergeFailure.invalidTransform(part: partIndex)
                }
                vertices.append(MeshVertex(
                    position: position, normal: unitNormal, textureCoordinate: vertex.textureCoordinate,
                    secondaryTextureCoordinate: vertex.secondaryTextureCoordinate, tangent: tangent, color: vertex.color
                ))
            }
            let start = indices.count
            let source = part.geometry.indices
            for triangle in stride(from: 0, to: source.count, by: 3) {
                if mirrored {
                    indices += [source[triangle] + offset, source[triangle + 2] + offset, source[triangle + 1] + offset]
                } else {
                    indices += [source[triangle] + offset, source[triangle + 1] + offset, source[triangle + 2] + offset]
                }
            }
            ranges.append(start..<indices.count)
        }
        var geometry = try MeshGeometry(vertices: vertices, indices: indices)
        if useGroups {
            geometry = try geometry.withGroups(ranges.enumerated().map { material, range in
                MeshGeometryGroup(start: range.lowerBound, count: range.count, materialIndex: material)
            })
        }
        return Merged(geometry: geometry, indexRanges: ranges)
    }

    public enum MergeFailure: LocalizedError, Equatable {
        case noParts
        case invalidTransform(part: Int)
        case tooManyVertices
        case indexCountOverflow

        public var errorDescription: String? {
            switch self {
            case .noParts:
                "Merging needs at least one geometry."
            case .invalidTransform(let part):
                "Part \(part) needs a finite affine transform with a nonzero determinant whose results stay finite."
            case .tooManyVertices:
                "The result would exceed the 32-bit vertex index range."
            case .indexCountOverflow:
                "The combined number of indices would exceed the supported array size."
            }
        }
    }
}

extension SIMD3 where Scalar == Float {
    var isFiniteVector: Bool { x.isFinite && y.isFinite && z.isFinite }
}
