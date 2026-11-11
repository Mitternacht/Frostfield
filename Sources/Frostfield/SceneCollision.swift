import simd

public extension SceneGraph {
    /// Bakes a static, world-space triangle hierarchy for repeated collision and
    /// ray queries. Use the hierarchy's default identity ray transform.
    ///
    /// The collision layer mask is independent of the render camera. Hidden
    /// objects participate by default. Set `visibleOnly` to skip hidden subtrees.
    /// Each instance contributes its current transform and each LOD contributes
    /// its currently selected mesh. Position-deformed meshes may wait for Metal.
    /// Rebuild after changing the collision geometry, pose, or transforms. Returns
    /// nil if the selection contains no mesh triangles.
    func collisionBVH(layers: SceneLayers = .default, visibleOnly: Bool = false) throws -> MeshBVH? {
        var parts: [MeshGeometry.MergePart] = []
        for node in try resolve() {
            guard node.layers.intersects(layers), !visibleOnly || node.isVisible else { continue }
            if let batch = node.batch {
                for object in try batch.objects(world: node.worldTransform, visibleOnly: visibleOnly) {
                    parts.append(.init(try batch.geometry(for: object.instance.geometryID), transform: object.world))
                }
                continue
            }
            guard let renderer = node.mesh, renderer.hasActiveTriangles, node.instances?.count != 0 else { continue }
            let posed = try renderer.pickingGeometry(jointPalette: node.jointPalette, morphWeights: node.morphWeights)
            guard let geometry = renderer.geometryInDrawRange(posed) else { continue }
            if let instances = node.instances {
                for transform in instances.transforms {
                    parts.append(.init(geometry, transform: node.worldTransform * transform))
                }
            } else {
                parts.append(.init(geometry, transform: node.worldTransform))
            }
        }
        guard !parts.isEmpty else { return nil }
        return try MeshBVH(geometry: MeshGeometry.merged(parts).geometry)
    }
}
