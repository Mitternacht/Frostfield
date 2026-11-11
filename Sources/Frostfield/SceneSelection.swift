import Foundation
import simd

/// The rectangle between two points in normalized drawable coordinates, with
/// (0, 0) at the top-left. Either drag direction works. Edges are inclusive.
public struct SceneSelectionRectangle: Equatable, Sendable {
    public let minimum: SIMD2<Float>
    public let maximum: SIMD2<Float>

    public init(from start: SIMD2<Float>, to end: SIMD2<Float>) throws {
        guard [start.x, start.y, end.x, end.y].allSatisfy({ (0...1).contains($0) }) else {
            throw ScenePickingError.invalidNormalizedPoint
        }
        minimum = simd_min(start, end)
        maximum = simd_max(start, end)
    }
}

/// A selected node or a selected slot in its instanced mesh or batch.
public struct SceneSelection: Equatable, Sendable {
    public let nodeID: UUID
    public let instanceIndex: Int?
    public let batchInstanceID: BatchedMesh.InstanceID?

    init(nodeID: UUID, instanceIndex: Int?, batchInstanceID: BatchedMesh.InstanceID? = nil) {
        self.nodeID = nodeID
        self.instanceIndex = instanceIndex
        self.batchInstanceID = batchInstanceID
    }
}

public extension SceneGraph {
    /// Selects visible meshes, points, lines and sprites within the graph
    /// camera's near and far planes, using the unjittered projection.
    func select(in rectangle: SceneSelectionRectangle, aspectRatio: Float) throws -> [SceneSelection] {
        try select(in: rectangle, aspectRatio: aspectRatio, perspective: camera)
    }

    /// Overrides the selection's far plane without changing the render camera.
    /// Depth is positive along the camera's local negative Z axis.
    func select(
        in rectangle: SceneSelectionRectangle, aspectRatio: Float, maximumDepth: Float
    ) throws -> [SceneSelection] {
        let perspective = try PerspectiveCamera(verticalFieldOfView: camera.verticalFieldOfView, near: camera.near, far: maximumDepth)
        return try select(in: rectangle, aspectRatio: aspectRatio, perspective: perspective)
    }

    private func select(
        in rectangle: SceneSelectionRectangle, aspectRatio: Float, perspective: PerspectiveCamera
    ) throws -> [SceneSelection] {
        let projection = try perspective.projection(aspectRatio: aspectRatio)
        let view = simd_inverse(simdPickingMatrix(cameraTransform.matrix))
        return try select(
            in: rectangle, view: view, projection: projection,
            cameraPosition: cameraTransform.translation
        )
    }

    /// Selects through an explicit render camera, including orthographic and
    /// reversed-depth cameras. The view's world position also controls LODs.
    ///
    /// A mesh or primitive is selected by the center of its local geometry
    /// bounds transformed to world space. Instances use their individual
    /// transformed origins. Selection includes active batch slots even when
    /// their individual visibility is off, matching Three.js SelectionBox.
    /// Results follow depth-first node and instance index order. A node with
    /// both a primitive and a mesh is selected once for their combined result,
    /// or once for the primitive after any selected instance slots. Hidden
    /// subtrees and off-layer nodes are skipped.
    func select(in rectangle: SceneSelectionRectangle, camera: ShaderGraphCamera) throws -> [SceneSelection] {
        try select(
            in: rectangle, view: simdPickingMatrix(camera.viewMatrix), projection: camera.projectionMatrix,
            cameraPosition: camera.pickingPosition()
        )
    }

    private func select(
        in rectangle: SceneSelectionRectangle, view: simd_double4x4,
        projection: simd_float4x4, cameraPosition: SIMD3<Float>
    ) throws -> [SceneSelection] {
        let clipFromWorld = simdPickingMatrix(projection) * view
        guard (0..<4).allSatisfy({ index in (0..<4).allSatisfy { clipFromWorld[index][$0].isFinite } }) else {
            throw ScenePickingError.invalidTransform
        }
        let left = 2 * Double(rectangle.minimum.x) - 1
        let right = 2 * Double(rectangle.maximum.x) - 1
        let bottom = 1 - 2 * Double(rectangle.maximum.y)
        let top = 1 - 2 * Double(rectangle.minimum.y)

        func contains(_ point: SIMD4<Double>) -> Bool {
            let clip = clipFromWorld * point
            // A Float projection can round the authored near or far plane
            // outward. Keep boundary centers within one clip-space Float ulp.
            let depthRounding = Double(Float.ulpOfOne) * clip.w
            return clip.w > 0 && clip.z >= -depthRounding && clip.z <= clip.w + depthRounding
                && clip.x >= left * clip.w && clip.x <= right * clip.w
                && clip.y >= bottom * clip.w && clip.y <= top * clip.w
        }

        var selections: [SceneSelection] = []
        for node in try resolve(cameraPosition: cameraPosition) where renders(node) {
            let world = simdPickingMatrix(node.worldTransform)
            var selectsNode = false
            if let batch = node.batch {
                for instance in batch.activeInstances where contains(world * SIMD4<Double>(instance.transform.columns.3)) {
                    selections.append(SceneSelection(nodeID: node.id, instanceIndex: nil, batchInstanceID: instance.id))
                }
            } else if let mesh = node.mesh {
                if let instances = node.instances {
                    for (index, transform) in instances.transforms.enumerated()
                        where contains(world * SIMD4<Double>(transform.columns.3)) {
                        selections.append(SceneSelection(nodeID: node.id, instanceIndex: index))
                    }
                } else {
                    let center = try mesh.selectionCenter(node)
                    selectsNode = contains(world * SIMD4(SIMD3<Double>(center), 1))
                }
            }
            if let primitive = node.primitive,
               contains(world * SIMD4(SIMD3<Double>(primitive.localBounds.center), 1)) {
                selectsNode = true
            }
            if selectsNode {
                selections.append(SceneSelection(nodeID: node.id, instanceIndex: nil))
            }
        }
        return selections
    }
}
