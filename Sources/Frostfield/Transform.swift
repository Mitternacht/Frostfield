import simd

// MARK: Coordinate conventions
//
// Frostfield uses one convention everywhere; `docs/coordinates.md` describes it
// with examples.
//
// - World and model space are right-handed: +x right, +y up, +z toward the
//   viewer. A camera with the identity orientation looks along -z.
// - Matrices are `simd_float4x4`, column-major, and multiply column vectors on
//   the right: `p' = M * p`. In a product, the rightmost matrix applies first,
//   so `projection * view * model * position`.
// - A positive rotation angle turns counter-clockwise when viewed from the
//   positive end of the rotation axis looking toward the origin.
// - Clip space is Metal's: after the perspective divide, x and y run from -1
//   to 1 with +y up, and visible depth runs from 0 at the near plane to 1 at
//   the far plane.

/// The placement of one object: scale, then rotate, then translate.
///
/// ``matrix`` applies the parts in that fixed order to a column vector, so a
/// model-space point `p` lands at `translation + rotation(scale * p)`.
///
/// A transform is a plain value. Changing it and drawing again updates the
/// per-frame uniforms only; no GPU resource is created.
public struct Transform: Sendable, Equatable {
    /// Moves the object, in world units, after scaling and rotating it.
    public var translation: SIMD3<Float>
    /// Turns the object about its own origin, after scaling it. The matrix
    /// uses the normalized quaternion, so only its direction matters.
    public var rotation: simd_quatf
    /// Scales the object along its own x, y, and z axes. Negative factors
    /// mirror the object, which also reverses its triangles' winding.
    public var scale: SIMD3<Float>

    /// No scale, rotation, or translation.
    public static let identity = Transform()

    public init(
        translation: SIMD3<Float> = .zero,
        rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        scale: SIMD3<Float> = .one
    ) {
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
    }

    /// `T * R * S`: the model-to-world matrix for column vectors.
    public var matrix: simd_float4x4 {
        let r = simd_float3x3(simd_normalize(rotation))
        // Columns of R * S are R's columns scaled by the matching factor;
        // the translation fills the last column.
        return simd_float4x4(columns: (
            SIMD4(r.columns.0 * scale.x, 0),
            SIMD4(r.columns.1 * scale.y, 0),
            SIMD4(r.columns.2 * scale.z, 0),
            SIMD4(translation, 1)
        ))
    }
}
