import simd

extension SceneNode {
    /// Controls how a child uses its authored rotation in the hierarchy.
    public enum OrientationMode: String, Codable, Hashable, Sendable {
        /// Compose all local components with the parent's world matrix.
        case inherited
        /// Use the local orientation in world space, like Three.js `Gyroscope`.
        /// The composed position and signed axis lengths still follow the parent.
        case world
    }

    func resolvedWorldTransform(
        relativeTo parent: simd_float4x4?, local preparedLocal: simd_float4x4? = nil
    ) -> simd_float4x4 {
        let local = preparedLocal ?? localTransformMatrix
        guard let parent else { return local }
        let composed = parent * local
        guard orientationMode == .world, let scale = Self.signedAxisLengths(composed) else { return composed }
        var world = SceneTransform(rotation: worldOrientedRotation, scale: SIMD3<Float>(scale)).matrix
        world.columns.3 = composed.columns.3
        return world
    }

    func resolvedDoubleWorldTransform(relativeTo parent: simd_double4x4?, local: simd_double4x4) -> simd_double4x4 {
        guard let parent else { return local }
        let composed = parent * local
        guard orientationMode == .world, let scale = Self.signedAxisLengths(
            SIMD3(composed[0].x, composed[0].y, composed[0].z),
            SIMD3(composed[1].x, composed[1].y, composed[1].z),
            SIMD3(composed[2].x, composed[2].y, composed[2].z)
        ) else { return composed }
        var world = simd_double4x4(simd_quatd(vector: SIMD4<Double>(worldOrientedRotation.vector)))
        world[0] *= scale.x
        world[1] *= scale.y
        world[2] *= scale.z
        world[3] = composed[3]
        return world
    }

    /// Matrix decomposition puts any mirror on X. Negative local Y or Z
    /// scales therefore contribute a half-turn to the canonical orientation.
    var worldOrientationCorrection: simd_quatf {
        let negativeY = transform.scale.y < 0, negativeZ = transform.scale.z < 0
        switch (negativeY, negativeZ) {
        case (false, false): return simd_quatf(real: 1, imag: .zero)
        case (true, true): return simd_quatf(real: 0, imag: SIMD3(1, 0, 0))
        case (false, true): return simd_quatf(real: 0, imag: SIMD3(0, 1, 0))
        case (true, false): return simd_quatf(real: 0, imag: SIMD3(0, 0, 1))
        }
    }

    var worldOrientedRotation: simd_quatf { transform.rotation * worldOrientationCorrection }

    /// Column lengths from the affine part. Unlike TRS decomposition, this
    /// intentionally accepts shear. Its determinant only supplies handedness.
    private static func signedAxisLengths(_ matrix: simd_float4x4) -> SIMD3<Double>? {
        func vector(_ column: SIMD4<Float>) -> SIMD3<Double> {
            SIMD3(Double(column.x), Double(column.y), Double(column.z))
        }
        return signedAxisLengths(vector(matrix.columns.0), vector(matrix.columns.1), vector(matrix.columns.2))
    }

    private static func signedAxisLengths(_ x: SIMD3<Double>, _ y: SIMD3<Double>, _ z: SIMD3<Double>) -> SIMD3<Double>? {
        let determinant = simd_dot(simd_cross(x, y), z)
        guard determinant.isFinite, determinant != 0 else { return nil }
        return SIMD3((determinant < 0 ? -1 : 1) * simd_length(x), simd_length(y), simd_length(z))
    }

    /// The local pose that yields the requested world pose under a new parent
    /// while using world orientation. Solves each local scale independently
    /// from the length of the new parent's transform of that oriented axis.
    static func worldOrientedLocalPose(
        preserving world: simd_float4x4, under parent: simd_float4x4, localOrigin: SIMD3<Float>
    ) throws -> SceneTransform {
        let pose: SceneTransform = try SceneTransform(decomposing: world)
        let basis = SceneTransform(rotation: pose.rotation).matrix
        guard let inherited = signedAxisLengths(parent * basis) else {
            throw SceneGraph.EditingError.unrepresentableTransform
        }
        let scale = SIMD3<Float>(SIMD3<Double>(pose.scale) / inherited)
        guard localOrigin.isFiniteVector, scale.isFiniteVector, scale.x != 0, scale.y != 0, scale.z != 0 else {
            throw SceneGraph.EditingError.unrepresentableTransform
        }
        return SceneTransform(translation: localOrigin, rotation: pose.rotation, scale: scale)
    }
}
