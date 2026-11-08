import simd

/// Why a perspective projection cannot be built from the given values.
///
/// Each description names the parameter, the value received, and what a valid
/// value looks like.
public enum ProjectionError: Error, Equatable, Sendable, CustomStringConvertible {
    /// The vertical field of view is not inside (0, π) radians.
    case invalidFieldOfView(Float)
    /// The near-plane distance is not finite and greater than zero.
    case invalidNearPlane(Float)
    /// The far-plane distance is not finite and greater than the near plane.
    case invalidFarPlane(near: Float, far: Float)
    /// The aspect ratio (width / height) is not finite and greater than zero.
    case invalidAspectRatio(Float)

    public var description: String {
        switch self {
        case .invalidFieldOfView(let value):
            let hint = value.isFinite && value > .pi && value < 180
                ? " It looks like degrees; convert with `degrees * .pi / 180`." : ""
            return "The vertical field of view must be greater than 0 and less than π radians (180°), "
                + "but is \(value) radians.\(hint)"
        case .invalidNearPlane(let value):
            return "The near-plane distance must be finite and greater than 0, but is \(value). "
                + "Use a small positive distance in front of the camera, such as 0.1; "
                + "0 or a negative value would put the near plane at or behind the eye."
        case .invalidFarPlane(let near, let far):
            return "The far-plane distance must be finite and greater than the near-plane distance (\(near)), "
                + "but is \(far)."
        case .invalidAspectRatio(let value):
            return "The aspect ratio (width / height) must be finite and greater than 0, but is \(value). "
                + "Compute it from the render target's pixel size, and do not draw while the target has no area."
        }
    }
}

/// A symmetric perspective projection into Metal clip space.
///
/// The projection is right-handed: it sees points in front of the camera,
/// which have negative view-space z. After the perspective divide, a point at
/// distance ``near`` maps to depth 0 and a point at distance ``far`` maps to
/// depth 1, Metal's visible depth range. Points nearer than ``near`` or
/// farther than ``far`` are clipped.
///
/// The initializer validates its values, so a projection that exists is always
/// usable; only the aspect ratio, which comes from the render target, is
/// checked when the matrix is built.
public struct PerspectiveProjection: Sendable, Equatable {
    /// The angle between the top and bottom clip planes, in radians.
    public let verticalFieldOfView: Float
    /// The distance from the eye to the near clip plane, in world units.
    public let near: Float
    /// The distance from the eye to the far clip plane, in world units.
    public let far: Float

    /// - Parameters:
    ///   - verticalFieldOfView: Radians, greater than 0 and less than π.
    ///   - near: Finite and greater than 0.
    ///   - far: Finite and greater than `near`.
    /// - Throws: ``ProjectionError`` naming the first invalid value.
    public init(verticalFieldOfView: Float, near: Float, far: Float) throws(ProjectionError) {
        guard verticalFieldOfView > 0, verticalFieldOfView < .pi else {
            throw .invalidFieldOfView(verticalFieldOfView)
        }
        guard near.isFinite, near > 0 else { throw .invalidNearPlane(near) }
        guard far.isFinite, far > near else { throw .invalidFarPlane(near: near, far: far) }
        self.verticalFieldOfView = verticalFieldOfView
        self.near = near
        self.far = far
    }

    /// The view-to-clip matrix for a target with this width-to-height ratio.
    ///
    /// - Throws: ``ProjectionError/invalidAspectRatio(_:)`` when the ratio is
    ///   not finite and positive.
    public func matrix(aspectRatio: Float) throws(ProjectionError) -> simd_float4x4 {
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw .invalidAspectRatio(aspectRatio) }
        return uncheckedMatrix(aspectRatio: aspectRatio)
    }

    /// ``matrix(aspectRatio:)`` for a ratio the caller has already validated,
    /// such as one computed from a non-empty render target.
    func uncheckedMatrix(aspectRatio: Float) -> simd_float4x4 {
        let yScale = 1 / tan(verticalFieldOfView / 2)
        let xScale = yScale / aspectRatio
        // z_clip = zScale * (z + near) and w_clip = -z, so depth z_clip / w_clip
        // is 0 at z = -near and 1 at z = -far.
        let zScale = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4(xScale, 0, 0, 0),
            SIMD4(0, yScale, 0, 0),
            SIMD4(0, 0, zScale, -1),
            SIMD4(0, 0, zScale * near, 0)
        ))
    }
}

/// A perspective camera placed in world space.
///
/// With the identity ``orientation`` the camera looks along -z with +y up.
/// ``orientation`` rotates that view direction into the world. The camera is
/// a plain value: changing it and drawing again updates the per-frame uniforms
/// only.
public struct Camera: Sendable, Equatable {
    /// The eye position in world space.
    public var position: SIMD3<Float>
    /// The rotation from camera space to world space. The view matrix uses the
    /// normalized quaternion.
    public var orientation: simd_quatf
    /// The lens: field of view and clip-plane distances.
    public var projection: PerspectiveProjection

    public init(
        position: SIMD3<Float>,
        orientation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        projection: PerspectiveProjection
    ) {
        self.position = position
        self.orientation = orientation
        self.projection = projection
    }

    /// The world-to-view matrix: the inverse of the camera's rigid placement.
    public var viewMatrix: simd_float4x4 {
        let inverseRotation = simd_float3x3(simd_normalize(orientation).inverse)
        let translation = -(inverseRotation * position)
        return simd_float4x4(columns: (
            SIMD4(inverseRotation.columns.0, 0),
            SIMD4(inverseRotation.columns.1, 0),
            SIMD4(inverseRotation.columns.2, 0),
            SIMD4(translation, 1)
        ))
    }

    /// `projection * view` for a target with this width-to-height ratio.
    ///
    /// - Throws: ``ProjectionError/invalidAspectRatio(_:)`` when the ratio is
    ///   not finite and positive.
    public func viewProjectionMatrix(aspectRatio: Float) throws(ProjectionError) -> simd_float4x4 {
        try projection.matrix(aspectRatio: aspectRatio) * viewMatrix
    }

    /// ``viewProjectionMatrix(aspectRatio:)`` for an already validated ratio.
    func uncheckedViewProjectionMatrix(aspectRatio: Float) -> simd_float4x4 {
        projection.uncheckedMatrix(aspectRatio: aspectRatio) * viewMatrix
    }
}
