import Foundation
import simd

/// Per-vertex offsets that blend onto a base mesh, in the base mesh's units and
/// vertex order. A morphed vertex is `base + Σ weightᵢ · offsetᵢ`.
public struct MeshMorphTargets: Sendable {
    /// One target's offsets. Nil attributes leave the base values unchanged.
    public struct Target: Sendable {
        public let positionOffsets: [SIMD3<Float>]?
        public let normalOffsets: [SIMD3<Float>]?
        /// Offsets to the tangent direction. Handedness is not morphed.
        public let tangentOffsets: [SIMD3<Float>]?

        public init(
            positionOffsets: [SIMD3<Float>]? = nil, normalOffsets: [SIMD3<Float>]? = nil,
            tangentOffsets: [SIMD3<Float>]? = nil
        ) {
            self.positionOffsets = positionOffsets
            self.normalOffsets = normalOffsets
            self.tangentOffsets = tangentOffsets
        }
    }

    public let vertexCount: Int
    public let targets: [Target]

    /// Requires at least one target, at least one attribute per target, one
    /// finite offset per vertex in every attribute, and a positive vertex count.
    public init(vertexCount: Int, targets: [Target]) throws {
        guard vertexCount > 0 else { throw ValidationError.invalidVertexCount }
        guard !targets.isEmpty else { throw ValidationError.emptyTargets }
        for (index, target) in targets.enumerated() {
            let attributes = [target.positionOffsets, target.normalOffsets, target.tangentOffsets].compactMap { $0 }
            guard !attributes.isEmpty else { throw ValidationError.emptyTarget(index) }
            guard attributes.allSatisfy({ $0.count == vertexCount }) else {
                throw ValidationError.vertexCountMismatch(target: index)
            }
            guard attributes.allSatisfy({ $0.allSatisfy(Self.isFinite) }) else {
                throw ValidationError.nonfiniteOffset(target: index)
            }
        }
        self.vertexCount = vertexCount
        self.targets = targets
    }

    /// Returns the geometry with weighted offsets added on the CPU. Normals and
    /// tangent directions are renormalized; tangent handedness, texture
    /// coordinates, and indices are unchanged. Supply one finite weight per target;
    /// zero weights skip their target. Throws when the geometry's vertex count
    /// differs, when tangent offsets apply to vertices without tangents, or when a
    /// morphed normal or tangent has zero length.
    public func apply(weights: [Float], to geometry: MeshGeometry) throws -> MeshGeometry {
        guard weights.count == targets.count else {
            throw ValidationError.weightCount(expected: targets.count, actual: weights.count)
        }
        guard weights.allSatisfy(\.isFinite) else { throw ValidationError.nonfiniteWeight }
        guard geometry.vertices.count == vertexCount else {
            throw ValidationError.geometryVertexCount(expected: vertexCount, actual: geometry.vertices.count)
        }
        let active = zip(targets, weights).filter { $0.1 != 0 }.map { (target: $0.0, weight: Double($0.1)) }
        guard !active.isEmpty else { return geometry }
        if active.contains(where: { $0.target.tangentOffsets != nil }), geometry.vertices.contains(where: { $0.tangent == nil }) {
            throw ValidationError.missingBaseTangents
        }

        var vertices: [MeshVertex] = []
        vertices.reserveCapacity(vertexCount)
        for (index, vertex) in geometry.vertices.enumerated() {
            var position = SIMD3<Double>(vertex.position)
            var normal = SIMD3<Double>(vertex.normal)
            var tangent = vertex.tangent.map { SIMD3<Double>(SIMD3<Float>($0.x, $0.y, $0.z)) }
            for (target, weight) in active {
                if let offsets = target.positionOffsets { position += SIMD3<Double>(offsets[index]) * weight }
                if let offsets = target.normalOffsets { normal += SIMD3<Double>(offsets[index]) * weight }
                if let offsets = target.tangentOffsets { tangent? += SIMD3<Double>(offsets[index]) * weight }
            }
            let normalLength = simd_length(normal)
            guard normalLength > 0, normalLength.isFinite else { throw ValidationError.degenerateNormal(vertex: index) }
            var morphedTangent: SIMD4<Float>?
            if let tangent, let handedness = vertex.tangent?.w {
                let length = simd_length(tangent)
                guard length > 0, length.isFinite else { throw ValidationError.degenerateTangent(vertex: index) }
                morphedTangent = SIMD4(SIMD3<Float>(tangent / length), handedness)
            }
            vertices.append(MeshVertex(
                position: SIMD3<Float>(position), normal: SIMD3<Float>(normal / normalLength),
                textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.secondaryTextureCoordinate, tangent: morphedTangent,
                color: vertex.color
            ))
        }
        return try MeshGeometry(vertices: vertices, indices: geometry.indices)
    }

    private static func isFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    public enum ValidationError: LocalizedError, Equatable {
        case invalidVertexCount
        case emptyTargets
        case emptyTarget(Int)
        case vertexCountMismatch(target: Int)
        case nonfiniteOffset(target: Int)
        case weightCount(expected: Int, actual: Int)
        case nonfiniteWeight
        case geometryVertexCount(expected: Int, actual: Int)
        case missingBaseTangents
        case degenerateNormal(vertex: Int)
        case degenerateTangent(vertex: Int)

        public var errorDescription: String? {
            switch self {
            case .invalidVertexCount: "Morph targets need a positive vertex count."
            case .emptyTargets: "Morph targets need at least one target."
            case .emptyTarget(let index): "Morph target \(index) needs position, normal, or tangent offsets."
            case .vertexCountMismatch(let index): "Every attribute of morph target \(index) needs one offset per vertex."
            case .nonfiniteOffset(let index): "Morph target \(index) contains a nonfinite offset."
            case .weightCount(let expected, let actual): "Expected \(expected) morph weights but received \(actual)."
            case .nonfiniteWeight: "Morph weights must be finite."
            case .geometryVertexCount(let expected, let actual):
                "The morph targets need a mesh with \(expected) vertices, but it has \(actual)."
            case .missingBaseTangents: "Tangent offsets require a mesh whose vertices all have tangents."
            case .degenerateNormal(let vertex): "The morphed normal of vertex \(vertex) has zero length."
            case .degenerateTangent(let vertex): "The morphed tangent of vertex \(vertex) has zero length."
            }
        }
    }
}
