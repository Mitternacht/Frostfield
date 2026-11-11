import Foundation
import simd

public extension GLTFScene {
    /// An authored glTF camera. Its projection is independent of its node pose.
    struct Camera: Sendable {
        public let sourceIndex: Int
        public let name: String?
        public let projection: Projection

        public enum Projection: Sendable, Equatable {
            case perspective(verticalFieldOfView: Float, aspectRatio: Float?, near: Float, far: Float?)
            case orthographic(halfWidth: Float, halfHeight: Float, near: Float, far: Float)
        }

        public struct View {
            public let clipFromWorld: simd_float4x4
            public let cameraPosition: SIMD3<Float>
            /// Fit the viewport to this aspect ratio to preserve the authored framing.
            public let aspectRatio: Float
            /// A fixed viewing direction for orthographic shading. Perspective views use the camera position.
            public let directionToCamera: SIMD3<Float>?
            public let depthMode: MeshDepthMode
            /// The separate view and projection matrices and clip distances, for surface
            /// graphs that read them. Pass it to `SceneGraph.encode(...graphCamera:)`.
            public let graphCamera: ShaderGraphCamera
        }

        /// Uses the viewport aspect when the asset omits the perspective aspect ratio.
        /// Orthographic half-extents are preserved exactly as authored.
        public func metalProjection(viewportAspectRatio: Float, depthMode: MeshDepthMode = .forward) throws -> simd_float4x4 {
            let reversed = depthMode == .reversed
            let x: Double, y: Double, z: Double, offset: Double
            let w: Float
            switch projection {
            case .perspective(let fieldOfView, let aspectRatio, let near, let far):
                let aspect = aspectRatio ?? viewportAspectRatio
                guard aspect.isFinite, aspect > 0 else { throw PerspectiveCamera.ValidationError.invalidAspectRatio }
                y = 1 / tan(Double(fieldOfView) / 2)
                x = y / Double(aspect)
                w = -1
                if let far {
                    let span = Double(far) - Double(near)
                    z = reversed ? Double(near) / span : -Double(far) / span
                    offset = Double(near) * Double(far) / span * (reversed ? 1 : -1)
                } else {
                    z = reversed ? 0 : -1
                    offset = Double(near) * (reversed ? 1 : -1)
                }
            case .orthographic(let halfWidth, let halfHeight, let near, let far):
                x = 1 / Double(halfWidth)
                y = 1 / Double(halfHeight)
                z = (reversed ? 1 : -1) / (Double(far) - Double(near))
                offset = (reversed ? Double(far) : Double(near)) * z
                w = 0
            }
            let scales = SIMD4(Float(x), Float(y), Float(z), Float(offset))
            let determinant = w == 0 ? scales.x * scales.y * scales.z : scales.x * scales.y * scales.w
            guard [scales.x, scales.y, scales.z, scales.w, determinant].allSatisfy(\.isFinite),
                  determinant != 0 else { throw PerspectiveCamera.ValidationError.unrepresentableProjection }
            return simd_float4x4(columns: (
                SIMD4(scales.x, 0, 0, 0), SIMD4(0, scales.y, 0, 0),
                SIMD4(0, 0, scales.z, w), SIMD4(0, 0, scales.w, 1 + w)
            ))
        }
    }
}

struct GLTFCameraDocument: Decodable {
    let cameras: [CameraRecord]?
    let nodes: [NodeRecord]?

    struct NodeRecord: Decodable {
        let camera: Int?
    }

    struct CameraRecord: Decodable {
        let name: String?
        let type: String
        let perspective: Perspective?
        let orthographic: Orthographic?

        struct Perspective: Decodable {
            let aspectRatio: Float?
            let yfov: Float
            let znear: Float
            let zfar: Float?
        }

        struct Orthographic: Decodable {
            let xmag: Float
            let ymag: Float
            let znear: Float
            let zfar: Float
        }
    }

    var nodeCameraIndices: [Int: Int] {
        Dictionary(uniqueKeysWithValues: (nodes ?? []).enumerated().compactMap { index, node in
            node.camera.map { (index, $0) }
        })
    }

    func resolve(nodes: [GLTFScene.Node]) throws -> [GLTFScene.Camera] {
        let records = cameras ?? []
        return try Set(nodes.compactMap(\.cameraIndex)).sorted().map { index in
            guard records.indices.contains(index) else { throw GLTFMeshLoader.Failure.missingCamera(index) }
            let record = records[index]
            let projection: GLTFScene.Camera.Projection
            switch (record.type, record.perspective, record.orthographic) {
            case ("perspective", let parameters?, nil):
                guard parameters.yfov.isFinite, parameters.yfov > 0, parameters.yfov < .pi,
                      parameters.znear.isFinite, parameters.znear > 0,
                      parameters.zfar.map({ $0.isFinite && $0 > parameters.znear }) ?? true,
                      parameters.aspectRatio.map({ $0.isFinite && $0 > 0 }) ?? true else {
                    throw GLTFMeshLoader.Failure.invalidCamera(index, "invalid perspective field of view, aspect, or clipping planes")
                }
                projection = .perspective(
                    verticalFieldOfView: parameters.yfov, aspectRatio: parameters.aspectRatio,
                    near: parameters.znear, far: parameters.zfar
                )
            case ("orthographic", nil, let parameters?):
                guard [parameters.xmag, parameters.ymag, parameters.znear, parameters.zfar].allSatisfy(\.isFinite),
                      parameters.xmag > 0, parameters.ymag > 0, parameters.znear >= 0, parameters.zfar > parameters.znear else {
                    throw GLTFMeshLoader.Failure.invalidCamera(index, "invalid orthographic magnification or clipping planes")
                }
                projection = .orthographic(
                    halfWidth: parameters.xmag, halfHeight: parameters.ymag,
                    near: parameters.znear, far: parameters.zfar
                )
            default:
                throw GLTFMeshLoader.Failure.invalidCamera(index, "camera type and projection object must agree")
            }
            return GLTFScene.Camera(sourceIndex: index, name: record.name, projection: projection)
        }
    }
}
