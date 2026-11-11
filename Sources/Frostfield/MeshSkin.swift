import Foundation
import simd

/// Four indices and corresponding weights for one vertex in its bind pose.
public struct VertexSkinInfluence: Equatable, Sendable {
    public let joints: SIMD4<UInt16>
    public let weights: SIMD4<Float>

    public init(joints: SIMD4<UInt16>, weights: SIMD4<Float>) {
        self.joints = joints
        self.weights = weights
    }
}

/// Validated per-vertex joint influences, in the same order as a mesh's vertices.
public struct MeshSkin: Sendable {
    public static let maximumJointCount = 256

    public let jointCount: Int
    public let influences: [VertexSkinInfluence]

    /// Every joint index must be in range. Finite nonnegative weights with a
    /// positive sum are normalized when the skin is created.
    public init(jointCount: Int, influences: [VertexSkinInfluence]) throws {
        guard (1...Self.maximumJointCount).contains(jointCount) else { throw SkinningError.jointCount }
        guard !influences.isEmpty else { throw SkinningError.influenceCount }
        self.jointCount = jointCount
        self.influences = try influences.enumerated().map { index, influence in
            let joints = influence.joints
            guard (0..<4).allSatisfy({ Int(joints[$0]) < jointCount }) else {
                throw SkinningError.jointIndex(vertex: index)
            }
            let weights = influence.weights
            guard (0..<4).allSatisfy({ weights[$0].isFinite && weights[$0] >= 0 }) else {
                throw SkinningError.weights(vertex: index)
            }
            let sum = Double(weights.x) + Double(weights.y) + Double(weights.z) + Double(weights.w)
            guard sum > 0 else { throw SkinningError.weights(vertex: index) }
            let normalized = SIMD4(
                Float(Double(weights.x) / sum), Float(Double(weights.y) / sum),
                Float(Double(weights.z) / sum), Float(Double(weights.w) / sum)
            )
            return VertexSkinInfluence(joints: joints, weights: normalized)
        }
    }

    /// Evaluates the skin on the CPU for queries such as geometric picking.
    /// Rendering applies the same palette in the vertex shader.
    public func deform(_ geometry: MeshGeometry, using palette: JointPalette) throws -> MeshGeometry {
        guard influences.count == geometry.vertices.count else { throw SkinningError.influenceCount }
        guard jointCount == palette.matrices.count else { throw SkinningError.paletteCount }
        let vertices = zip(geometry.vertices, influences).map { vertex, influence in
            let joints = influence.joints
            let weights = influence.weights
            let matrices = palette.matrices
            let blend = matrices[Int(joints.x)] * weights.x + matrices[Int(joints.y)] * weights.y
                + matrices[Int(joints.z)] * weights.z + matrices[Int(joints.w)] * weights.w
            let position = blend * SIMD4(vertex.position, 1)
            return MeshVertex(
                position: xyz(position), normal: Self.transformedNormal(vertex.normal, by: blend),
                textureCoordinate: vertex.textureCoordinate, secondaryTextureCoordinate: vertex.secondaryTextureCoordinate,
                tangent: vertex.tangent.flatMap { Self.transformedTangent($0, by: blend) },
                color: vertex.color
            )
        }
        return try MeshGeometry(vertices: vertices, indices: geometry.indices)
    }

    private static func transformedTangent(_ tangent: SIMD4<Float>, by matrix: simd_float4x4) -> SIMD4<Float>? {
        let determinant = simd_dot(xyz(matrix.columns.0), simd_cross(xyz(matrix.columns.1), xyz(matrix.columns.2)))
        let direction = xyz(matrix * SIMD4(tangent.x, tangent.y, tangent.z, 0))
        let scale = max(abs(direction.x), abs(direction.y), abs(direction.z))
        guard determinant.isFinite, determinant != 0, scale.isFinite, scale > 0 else { return nil }
        return SIMD4(simd_normalize(direction / scale), determinant < 0 ? -tangent.w : tangent.w)
    }

    private static func transformedNormal(_ normal: SIMD3<Float>, by matrix: simd_float4x4) -> SIMD3<Float> {
        let x = xyz(matrix.columns.0)
        let y = xyz(matrix.columns.1)
        let z = xyz(matrix.columns.2)
        let cofactorX = simd_cross(y, z)
        let determinant = simd_dot(x, cofactorX)
        let cofactors = simd_float3x3(columns: (cofactorX, simd_cross(z, x), simd_cross(x, y)))
        let transformed = (cofactors * normal) * (determinant < 0 ? -1 : 1)
        let scale = max(abs(transformed.x), max(abs(transformed.y), abs(transformed.z)))
        guard determinant.isFinite, determinant != 0, scale.isFinite, scale > 0 else { return normal }
        return simd_normalize(transformed / scale)
    }
}

/// A pose in mesh-local space. Matrix order matches the skin's joint indices.
public struct JointPalette: Equatable, Sendable {
    public let matrices: [simd_float4x4]

    /// Each matrix must be a finite, invertible affine transform.
    public init(matrices: [simd_float4x4]) throws {
        guard (1...MeshSkin.maximumJointCount).contains(matrices.count) else { throw SkinningError.jointCount }
        for (index, matrix) in matrices.enumerated() where affineInverse(matrix) == nil {
            throw SkinningError.jointTransform(index)
        }
        self.matrices = matrices
    }

    /// Computes inverse(meshWorld) × jointWorld × inverseBind for each joint.
    public init(
        meshWorld: simd_float4x4,
        jointWorld: [simd_float4x4],
        inverseBindMatrices: [simd_float4x4]
    ) throws {
        guard jointWorld.count == inverseBindMatrices.count else { throw SkinningError.paletteCount }
        guard (1...MeshSkin.maximumJointCount).contains(jointWorld.count) else { throw SkinningError.jointCount }
        guard let worldToMesh = affineInverse(meshWorld) else { throw SkinningError.meshTransform }
        let matrices = try zip(jointWorld, inverseBindMatrices).enumerated().map { index, pair in
            let (world, inverseBind) = pair
            guard affineInverse(world) != nil, affineInverse(inverseBind) != nil else {
                throw SkinningError.jointTransform(index)
            }
            return worldToMesh * world * inverseBind
        }
        try self.init(matrices: matrices)
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.matrices.elementsEqual(rhs.matrices) { a, b in
            a.columns.0 == b.columns.0 && a.columns.1 == b.columns.1
                && a.columns.2 == b.columns.2 && a.columns.3 == b.columns.3
        }
    }
}

public enum SkinningError: LocalizedError, Equatable {
    case jointCount
    case influenceCount
    case jointIndex(vertex: Int)
    case weights(vertex: Int)
    case paletteCount
    case jointTransform(Int)
    case meshTransform
    case missingPalette
    case unexpectedPalette

    public var errorDescription: String? {
        switch self {
        case .jointCount: "A skin needs between 1 and \(MeshSkin.maximumJointCount) joints."
        case .influenceCount: "Provide one set of joint influences for every mesh vertex."
        case .jointIndex(let vertex): "Vertex \(vertex) refers to a joint outside the skin."
        case .weights(let vertex): "Vertex \(vertex) needs finite nonnegative weights with a positive sum."
        case .paletteCount: "The pose must provide one transform per skin joint."
        case .jointTransform(let joint): "Joint \(joint) needs a finite, invertible affine transform."
        case .meshTransform: "Skinning needs a finite, invertible affine mesh transform."
        case .missingPalette: "A skinned mesh needs a joint palette."
        case .unexpectedPalette: "A rigid mesh cannot use a joint palette."
        }
    }
}

private func xyz(_ vector: SIMD4<Float>) -> SIMD3<Float> {
    SIMD3(vector.x, vector.y, vector.z)
}

private func affineInverse(_ matrix: simd_float4x4) -> simd_float4x4? {
    let (x, y, z, w) = matrix.columns
    guard x.w == 0, y.w == 0, z.w == 0, w.w == 1,
          [x, y, z, w].allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
        return nil
    }
    let linear = simd_float3x3(columns: (xyz(x), xyz(y), xyz(z)))
    let determinant = simd_determinant(linear)
    guard determinant.isFinite, determinant != 0 else { return nil }
    let inverse = simd_inverse(linear)
    let translation = -(inverse * xyz(w))
    let columns = (SIMD4(inverse.columns.0, 0), SIMD4(inverse.columns.1, 0),
                   SIMD4(inverse.columns.2, 0), SIMD4(translation, 1))
    guard [columns.0, columns.1, columns.2, columns.3].allSatisfy({
        $0.x.isFinite && $0.y.isFinite && $0.z.isFinite
    }) else { return nil }
    return simd_float4x4(columns: columns)
}
