import Foundation
import simd

public extension SceneGraph {
    /// Bounds a subtree or the entire graph in world coordinates, like Three.js
    /// `Box3.setFromObject`. Nil means there is no mesh or primitive geometry.
    /// An unknown ID throws `EditingError.missingNode`.
    ///
    /// By default, hidden and off-layer objects are included. Set `visibleOnly`
    /// to apply the graph camera's visibility and layer filters. Each LOD uses
    /// the mesh selected for that camera. Pixel-sized points, strokes and
    /// sprites contribute their anchor geometry. World-sized primitives also
    /// include conservative padding for their nominal extents.
    ///
    /// The default uses transformed boxes. With `precise`, non-instanced
    /// meshes use their posed vertex positions, and points, lines, and sprites
    /// use their transformed anchors. Instanced and batched meshes use conservative bounds
    /// in both modes. If a graph-deformed mesh needs posed vertices, this
    /// query waits for its Metal deformation pass.
    func worldBounds(
        of root: UUID? = nil, precise: Bool = false, visibleOnly: Bool = false
    ) throws -> AxisAlignedBox? {
        let selected: Set<UUID>?
        if let root {
            guard let node = node(withID: root) else { throw EditingError.missingNode(root) }
            var pending = [node]
            var ids = Set<UUID>()
            while let node = pending.popLast() {
                guard ids.insert(node.id).inserted else { throw ValidationError.duplicateID(node.id) }
                pending.append(contentsOf: node.children)
            }
            selected = ids
        } else {
            selected = nil
        }

        var result: AxisAlignedBox?
        func include(_ box: AxisAlignedBox) {
            result = result.map {
                AxisAlignedBox(validatedMinimum: simd_min($0.minimum, box.minimum), maximum: simd_max($0.maximum, box.maximum))
            } ?? box
        }
        for node in try resolve() {
            guard selected?.contains(node.id) != false, !visibleOnly || renders(node) else { continue }
            if let mesh = node.mesh, node.instances?.count != 0,
               let box = try Self.meshBounds(mesh, node: node, precise: precise) {
                include(box)
            }
            if let batch = node.batch, let box = try batch.bounds(world: node.worldTransform, visibleOnly: visibleOnly) {
                include(box)
            }
            if let primitive = node.primitive {
                let box = try Self.primitiveBounds(primitive, node: node, precise: precise)
                include(box)
            }
        }
        return result
    }

    private static func meshBounds(
        _ mesh: MeshRenderer, node: ResolvedSceneNode, precise: Bool
    ) throws -> AxisAlignedBox? {
        let usesVertices = precise && node.instances == nil
        if !usesVertices, let box = try mesh.cullingBounds(of: node) { return box }

        let geometry = try mesh.pickingGeometry(jointPalette: node.jointPalette, morphWeights: node.morphWeights)
        if usesVertices { return try worldBox(geometry.vertices.lazy.map(\.position), transform: node.worldTransform) }

        let local = geometry.bounds
        if let instances = node.instances {
            return try instances.bounds(enclosing: local)?.transformed(by: node.worldTransform)
        }
        return try local.transformed(by: node.worldTransform)
    }

    private static func primitiveBounds(
        _ primitive: ScenePrimitive, node: ResolvedSceneNode, precise: Bool
    ) throws -> AxisAlignedBox {
        if !precise, let local = node.cullingBounds {
            return try local.transformed(by: node.worldTransform)
        }
        let anchors: AxisAlignedBox
        if precise {
            anchors = switch primitive {
            case .points(let renderer):
                try worldBox(renderer.cloud.positions, transform: node.worldTransform)
            case .lines(let renderer):
                try worldBox(renderer.geometry.positions, transform: node.worldTransform)
            case .sprites(let renderer):
                try worldBox(renderer.sprites.lazy.map(\.position), transform: node.worldTransform)
            }
        } else {
            anchors = try primitive.localBounds.transformed(by: node.worldTransform)
        }
        let padding = SIMD3<Double>(repeating: Double(primitive.worldPadding))
        return try AxisAlignedBox(
            minimum: boundsFloatBelow(boundsDouble(anchors.minimum) - padding),
            maximum: boundsFloatAbove(boundsDouble(anchors.maximum) + padding)
        )
    }

    private static func worldBox(
        _ positions: some Sequence<SIMD3<Float>>, transform: simd_float4x4
    ) throws -> AxisAlignedBox {
        guard let affine = BoundsAffine(transform) else { throw AxisAlignedBox.Failure.invalidTransform }
        var lower = SIMD3<Double>(repeating: .infinity)
        var upper = SIMD3<Double>(repeating: -.infinity)
        for position in positions {
            let transformed = affine.linear * boundsDouble(position) + affine.translation
            lower = simd_min(lower, transformed)
            upper = simd_max(upper, transformed)
        }
        return try AxisAlignedBox(minimum: boundsFloatBelow(lower), maximum: boundsFloatAbove(upper))
    }
}
