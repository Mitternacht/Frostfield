import Foundation
import simd

/// Utilities for explicit scene camera views.
public enum CameraUtils {
    public struct FramedView: Sendable {
        public let clipFromWorld: simd_float4x4
        public let cameraPosition: SIMD3<Float>
        /// Match the viewport's aspect ratio to this value for undistorted display.
        public let aspectRatio: Float
        public let depthMode: MeshDepthMode
        /// The view and projection separately, for graph inputs and picking.
        public let graphCamera: ShaderGraphCamera
    }

    public enum Failure: LocalizedError, Equatable {
        case nonfinitePoint
        case invalidRectangle
        case eyeBehindScreen
        case invalidClippingPlanes
        case unrepresentableView

        public var errorDescription: String? {
            switch self {
            case .nonfinitePoint: "The camera eye and screen corners must be finite."
            case .invalidRectangle: "Supply three distinct corners with perpendicular bottom and left edges."
            case .eyeBehindScreen: "The eye must be in front of the screen, on the side its right × up normal points toward."
            case .invalidClippingPlanes: "Use a positive finite near distance and a greater far distance, which can be infinity."
            case .unrepresentableView: "The framed camera exceeds the precision of a Float view and projection."
            }
        }
    }

    /// Frames a planar rectangular screen with an off-axis perspective view,
    /// like Three.js `CameraUtils.frameCorners`. The inputs are world-space
    /// points in right-handed coordinates. Looking from the eye, specify the
    /// bottom-left, bottom-right, then top-left corner. The fourth corner is
    /// implied. Right × up must point toward the eye. The screen need not be
    /// centered on the eye's axis or coincide with either clip plane.
    ///
    /// The resulting Metal projection uses a negative camera-space Z viewing
    /// direction and depth zero to one, or one to zero for reversed depth.
    /// Pass the returned values to the scene's explicit-view raster encode.
    public static func frameCorners(
        eye: SIMD3<Float>, bottomLeft: SIMD3<Float>, bottomRight: SIMD3<Float>, topLeft: SIMD3<Float>,
        near: Float, far: Float, depthMode: MeshDepthMode = .forward
    ) throws -> FramedView {
        guard [eye, bottomLeft, bottomRight, topLeft].allSatisfy(\.isFiniteVector) else { throw Failure.nonfinitePoint }
        guard near.isFinite, near > 0, far > near else { throw Failure.invalidClippingPlanes }

        let origin = SIMD3<Double>(bottomLeft), position = SIMD3<Double>(eye)
        let bottom = SIMD3<Double>(bottomRight) - origin
        let left = SIMD3<Double>(topLeft) - origin
        let width = simd_length(bottom), leftLength = simd_length(left)
        guard width > 0, leftLength > 0 else { throw Failure.invalidRectangle }
        let right = bottom / width
        // Allow roundoff in Float input corners while limiting the resulting
        // screen-edge error even for strongly elongated rectangles.
        guard abs(simd_dot(right, left)) <= 1e-5 * min(width, leftLength) else { throw Failure.invalidRectangle }
        let back = simd_normalize(simd_cross(right, left))
        let up = simd_cross(back, right)
        let height = simd_dot(up, left)
        let lowerLeft = origin - position
        let distance = -simd_dot(back, lowerLeft)
        guard distance > 0 else { throw Failure.eyeBehindScreen }

        let view = simd_float4x4(columns: (
            SIMD4(Float(right.x), Float(up.x), Float(back.x), 0),
            SIMD4(Float(right.y), Float(up.y), Float(back.y), 0),
            SIMD4(Float(right.z), Float(up.z), Float(back.z), 0),
            SIMD4(Float(-simd_dot(right, position)), Float(-simd_dot(up, position)), Float(-simd_dot(back, position)), 1)
        ))
        let reversed = depthMode == .reversed
        let z: Double
        let offset: Double
        if far.isInfinite {
            z = reversed ? 0 : -1
            offset = Double(near) * (reversed ? 1 : -1)
        } else {
            let span = Double(far) - Double(near)
            z = reversed ? Double(near) / span : -Double(far) / span
            offset = Double(near) * Double(far) / span * (reversed ? 1 : -1)
        }
        let depthSlope = Float(z)
        // A finite far plane must remain distinct from the infinite projection
        // after conversion to the matrix precision used by raster and picking.
        guard far.isInfinite || (reversed ? depthSlope > 0 : depthSlope < -1) else {
            throw Failure.unrepresentableView
        }
        // Near-plane extents are the screen extents scaled by near/distance.
        // The common near factor cancels from the X/Y scale and shift.
        let projection = simd_float4x4(columns: (
            SIMD4(Float(2 * distance / width), 0, 0, 0),
            SIMD4(0, Float(2 * distance / height), 0, 0),
            SIMD4(Float(1 + 2 * simd_dot(right, lowerLeft) / width),
                  Float(1 + 2 * simd_dot(up, lowerLeft) / height), depthSlope, -1),
            SIMD4(0, 0, Float(offset), 0)
        ))
        let aspect = Float(width / height)
        let clip = projection * view
        let clipDeterminant = simd_determinant(clip)
        let topRight = SIMD3<Float>(origin + bottom + left)
        let corners: [(SIMD3<Float>, SIMD2<Float>)] = [
            (bottomLeft, SIMD2(-1, -1)), (bottomRight, SIMD2(1, -1)),
            (topLeft, SIMD2(-1, 1)), (topRight, SIMD2(1, 1)),
        ]
        guard aspect.isFinite, aspect > 0, clipDeterminant.isFinite, clipDeterminant != 0,
              let camera = try? ShaderGraphCamera(viewMatrix: view, projectionMatrix: projection, near: near, far: far),
              corners.allSatisfy({ point, expected in
                  let projected = clip * SIMD4(point, 1)
                  return projected.w.isFinite && projected.w > 0 && projected.z.isFinite
                      && simd_distance(SIMD2(projected.x, projected.y) / projected.w, expected) < 0.000_1
              }) else { throw Failure.unrepresentableView }
        return FramedView(
            clipFromWorld: clip, cameraPosition: eye, aspectRatio: aspect, depthMode: depthMode, graphCamera: camera
        )
    }
}
