import Foundation
import simd

/// A local-to-world transform for right-handed coordinates and column vectors.
public struct SceneTransform: Equatable, Sendable {
    public var translation: SIMD3<Float>
    /// The unit quaternion that rotates the scaled point.
    public var rotation: simd_quatf
    public var scale: SIMD3<Float>

    public init(
        translation: SIMD3<Float> = .zero,
        rotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0)),
        scale: SIMD3<Float> = SIMD3<Float>(repeating: 1)
    ) {
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
    }

    /// A column-major matrix that applies scale, rotation, then translation.
    public var matrix: simd_float4x4 {
        var result = simd_float4x4(rotation)
        result.columns.0 *= scale.x
        result.columns.1 *= scale.y
        result.columns.2 *= scale.z
        result.columns.3 = SIMD4<Float>(translation, 1)
        return result
    }
}

/// A right-handed perspective camera with a negative Z viewing direction.
public struct PerspectiveCamera {
    public let verticalFieldOfView: Float
    public let near: Float
    public let far: Float

    /// The field of view is in radians. Clip distances are positive world units.
    public init(verticalFieldOfView: Float, near: Float, far: Float) throws {
        guard verticalFieldOfView.isFinite, verticalFieldOfView > 0, verticalFieldOfView < .pi else {
            throw ValidationError.invalidFieldOfView
        }
        guard near.isFinite, far.isFinite, near > 0, far > near else {
            throw ValidationError.invalidClippingPlanes
        }

        self.verticalFieldOfView = verticalFieldOfView
        self.near = near
        self.far = far
    }

    /// Maps the camera-space clip planes to Metal depth values zero and one.
    /// The aspect ratio is the drawable width divided by its height.
    public func projection(aspectRatio: Float) throws -> simd_float4x4 {
        try projection(aspectRatio: aspectRatio, depthMode: .forward)
    }

    /// Maps near and far to 0 and 1 in forward depth, or to 1 and 0 in reversed depth.
    /// Logarithmic mode uses the forward clip matrix and writes its depth in the fragment stage.
    public func projection(aspectRatio: Float, depthMode: MeshDepthMode) throws -> simd_float4x4 {
        guard aspectRatio.isFinite, aspectRatio > 0 else {
            throw ValidationError.invalidAspectRatio
        }

        let verticalScale = 1 / tan(Double(verticalFieldOfView) / 2)
        let range = Double(far) - Double(near)
        let depthScale = depthMode == .reversed ? Double(near) / range : -Double(far) / range
        let depthOffset = depthMode == .reversed ? Double(far) * depthScale : Double(near) * depthScale
        let x = Float(verticalScale / Double(aspectRatio))
        let y = Float(verticalScale)
        let z = Float(depthScale)
        let offset = Float(depthOffset)

        guard x.isFinite, y.isFinite, z.isFinite, offset.isFinite, x > 0, y > 0 else {
            throw ValidationError.unrepresentableProjection
        }

        return simd_float4x4(columns: (
            SIMD4<Float>(x, 0, 0, 0),
            SIMD4<Float>(0, y, 0, 0),
            SIMD4<Float>(0, 0, z, -1),
            SIMD4<Float>(0, 0, offset, 0)
        ))
    }

    /// Projects a pixel window of the full image. The field of view and aspect
    /// ratio describe the full image, and the window fills the destination viewport.
    public func projection(viewOffset: CameraViewOffset) throws -> simd_float4x4 {
        try projection(viewOffset: viewOffset, depthMode: .forward)
    }

    /// Projects a pixel window with the selected near/far depth mapping.
    public func projection(viewOffset: CameraViewOffset, depthMode: MeshDepthMode) throws -> simd_float4x4 {
        try viewOffset.applying(to: projection(aspectRatio: viewOffset.fullAspectRatio, depthMode: depthMode))
    }

    public enum ValidationError: LocalizedError, Equatable {
        case invalidFieldOfView
        case invalidClippingPlanes
        case invalidAspectRatio
        case unrepresentableProjection

        public var errorDescription: String? {
            switch self {
            case .invalidFieldOfView:
                "The vertical field of view must be between zero and pi radians."
            case .invalidClippingPlanes:
                "Clip distances must be finite, with zero < near < far."
            case .invalidAspectRatio:
                "The drawable aspect ratio must be positive and finite."
            case .unrepresentableProjection:
                "The camera and aspect ratio exceed the precision of a Float projection matrix."
            }
        }
    }
}

/// A right-handed orthographic camera that looks down its local −Z axis.
///
/// The vertical half extent is in world units. The horizontal half extent is
/// that height times the drawable aspect ratio, so a square window is square
/// in world space. Clip distances are positive world units along the view
/// direction. The projection matches an imported glTF orthographic camera.
public struct OrthographicCamera: Equatable, Sendable {
    public var halfHeight: Float
    public var near: Float
    public var far: Float

    public init(halfHeight: Float, near: Float, far: Float) throws {
        try Self.validate(halfHeight: halfHeight, near: near, far: far)
        self.halfHeight = halfHeight
        self.near = near
        self.far = far
    }

    /// Column-major clip matrix. `x` grows right and `y` grows up.
    /// Forward depth maps near to 0 and far to 1. Reversed depth maps near to 1 and far to 0.
    public func projection(aspectRatio: Float, depthMode: MeshDepthMode = .forward) throws -> simd_float4x4 {
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw PerspectiveCamera.ValidationError.invalidAspectRatio }
        try Self.validate(halfHeight: halfHeight, near: near, far: far)
        let halfWidth = Double(halfHeight) * Double(aspectRatio)
        let span = Double(far) - Double(near)
        let reversed = depthMode == .reversed
        let x = 1 / halfWidth
        let y = 1 / Double(halfHeight)
        let z = (reversed ? 1 : -1) / span
        let offset = (reversed ? Double(far) : Double(near)) * z
        let scales = SIMD4(Float(x), Float(y), Float(z), Float(offset))
        let determinant = scales.x * scales.y * scales.z
        guard [scales.x, scales.y, scales.z, scales.w, determinant].allSatisfy(\.isFinite), determinant != 0 else {
            throw PerspectiveCamera.ValidationError.unrepresentableProjection
        }
        return simd_float4x4(columns: (
            SIMD4(scales.x, 0, 0, 0), SIMD4(0, scales.y, 0, 0),
            SIMD4(0, 0, scales.z, 0), SIMD4(0, 0, scales.w, 1)
        ))
    }

    private static func validate(halfHeight: Float, near: Float, far: Float) throws {
        guard halfHeight.isFinite, halfHeight > 0 else { throw ValidationError.invalidHalfHeight }
        guard near.isFinite, far.isFinite, near > 0, far > near else {
            throw PerspectiveCamera.ValidationError.invalidClippingPlanes
        }
    }

    public enum ValidationError: LocalizedError, Equatable {
        case invalidHalfHeight

        public var errorDescription: String? {
            switch self {
            case .invalidHalfHeight:
                "The orthographic half height must be finite and positive."
            }
        }
    }
}

/// A pixel window into a larger camera image, such as one tile of a still or
/// one display in a tiled setup. The origin is measured from the top left.
public struct CameraViewOffset: Equatable, Sendable {
    public let fullWidth: Int
    public let fullHeight: Int
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int

    /// The image and window dimensions must be positive. The window can extend
    /// outside the image for overscan.
    public init(fullWidth: Int, fullHeight: Int, x: Int, y: Int, width: Int, height: Int) throws {
        guard fullWidth > 0, fullHeight > 0, width > 0, height > 0 else {
            throw Failure.invalidWindow
        }
        self.fullWidth = fullWidth
        self.fullHeight = fullHeight
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    /// Width divided by height of the uncropped image.
    public var fullAspectRatio: Float { Float(Double(fullWidth) / Double(fullHeight)) }

    /// Crops a full-image Metal projection and expands the window to fill the
    /// viewport. Works with perspective and orthographic projections, including
    /// shifted and reversed-depth views. The clip-space Z and W are preserved.
    public func applying(to projection: simd_float4x4) throws -> simd_float4x4 {
        let scaleX = Double(fullWidth) / Double(width)
        let scaleY = Double(fullHeight) / Double(height)
        let offsetX = (Double(fullWidth) - 2 * Double(x) - Double(width)) / Double(width)
        let offsetY = (2 * Double(y) + Double(height) - Double(fullHeight)) / Double(height)
        var result = projection
        for index in 0..<4 {
            let column = projection[index]
            let croppedX = Float(scaleX * Double(column.x) + offsetX * Double(column.w))
            let croppedY = Float(scaleY * Double(column.y) + offsetY * Double(column.w))
            guard croppedX.isFinite, croppedY.isFinite, column.z.isFinite, column.w.isFinite else {
                throw Failure.unrepresentableProjection
            }
            result[index] = SIMD4(croppedX, croppedY, column.z, column.w)
        }
        return result
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidWindow
        case unrepresentableProjection

        public var errorDescription: String? {
            switch self {
            case .invalidWindow: "The full image and camera view window need positive pixel dimensions."
            case .unrepresentableProjection: "The cropped camera projection has nonfinite matrix elements."
            }
        }
    }
}

public extension ShaderGraphCamera {
    /// Uses the window's projection for both rendering and queries while
    /// keeping the camera pose and near/far clip distances.
    func cropped(to viewOffset: CameraViewOffset) throws -> Self {
        try Self(
            viewMatrix: viewMatrix, projectionMatrix: viewOffset.applying(to: projectionMatrix), near: near, far: far
        )
    }
}
