import Foundation
import Metal
import simd

/// Scene conversion and inspection utilities corresponding to Three.js `SceneUtils`.
public enum SceneUtils {
    public enum Failure: LocalizedError, Equatable {
        case notInstancedMesh
        case notBatchedMesh
        case unrepresentableTransform(instance: Int)
        case unsupportedTint(instance: Int)
        case nonfiniteTint(instance: Int)
        case nonfiniteVertex

        public var errorDescription: String? {
            switch self {
            case .notInstancedMesh: "Expand a scene node with an instancing-capable mesh and an instance set on the same device."
            case .notBatchedMesh: "Expand a scene node with a batch and no competing mesh payload."
            case .unrepresentableTransform(let index): "Instance \(index) needs a transform representable as translation, rotation and scale."
            case .unsupportedTint(let index): "Instance \(index) uses an independent shader tint that an ordinary mesh draw cannot carry."
            case .nonfiniteTint(let index): "Instance \(index) has a tint and material parameter product outside the Float range."
            case .nonfiniteVertex: "A world-space vertex exceeds the Float range."
            }
        }
    }

    /// Expands one instanced draw into editable child nodes in instance order.
    /// The children share the prepared mesh and copy the current tint and pose.
    /// The returned wrapper keeps the original node ID, transform, visibility,
    /// light, primitive and child subtrees. Replace the original node with it
    /// rather than adding a second copy of the same identity.
    public static func expandedInstances(of node: SceneNode) throws -> SceneNode {
        guard let mesh = node.mesh, let instances = node.instances, mesh.supportsInstancing,
              mesh.device === instances.device, node.batch == nil, node.levelsOfDetail == nil else {
            throw Failure.notInstancedMesh
        }
        let children = try instances.transforms.indices.map { index in
            let transform = try pose(instances.transforms[index], instance: index)
            let parameters = try parameters(node.materialParameters, tint: instances.tints[index], material: mesh.material, instance: index)
            return SceneNode(
                transform: transform, mesh: mesh, materialParameters: parameters,
                castsShadow: node.castsShadow, jointPalette: node.jointPalette, morphWeights: node.morphWeights,
                frustumCulled: node.frustumCulled, name: childName(node, index), layers: node.layers, renderOrder: node.renderOrder
            )
        }
        return wrapper(node, around: children)
    }

    /// Expands the batch's active slots, including hidden instances, in slot
    /// order. Children using the same geometry share one prepared mesh. The
    /// resulting ordinary draws use scene ordering and their own geometry bounds.
    public static func expandedBatch(of node: SceneNode) throws -> SceneNode {
        guard let batch = node.batch, node.mesh == nil, node.instances == nil, node.levelsOfDetail == nil,
              node.jointPalette == nil, node.morphWeights == nil else { throw Failure.notBatchedMesh }
        guard let source = batch.renderer else { return wrapper(node, around: []) }
        var renderers: [BatchedMesh.GeometryID: MeshRenderer] = [:]
        var children: [SceneNode] = []
        children.reserveCapacity(batch.count)
        for instance in batch.activeInstances {
            let index = instance.id.index
            let transform = try pose(instance.transform, instance: index)
            let tint = try parameters(node.materialParameters, tint: instance.tint, material: source.material, instance: index)
            let renderer: MeshRenderer
            if let existing = renderers[instance.geometryID] {
                renderer = existing
            } else {
                renderer = try MeshRenderer(
                    device: source.device, colorPixelFormat: source.colorPixelFormat,
                    geometry: batch.geometry(for: instance.geometryID), material: source.material,
                    depthMode: source.depthMode, renderState: source.renderState, sampleCount: source.sampleCount
                )
                renderers[instance.geometryID] = renderer
            }
            children.append(SceneNode(
                transform: transform, mesh: renderer, materialParameters: tint, castsShadow: node.castsShadow,
                frustumCulled: node.frustumCulled && batch.perObjectFrustumCulled, name: childName(node, index),
                isVisible: instance.isVisible, layers: node.layers, renderOrder: node.renderOrder
            ))
        }
        return wrapper(node, around: children)
    }

    /// Draws the complete geometry once per material, in material order.
    /// Use `SceneNode.grouped` for per-triangle material groups.
    public static func multiMaterialObject(
        _ geometry: MeshGeometry, materials: [MeshMaterial], device: any MTLDevice,
        colorPixelFormat: MTLPixelFormat = .rgba16Float, depthMode: MeshDepthMode = .forward, sampleCount: Int = 1
    ) throws -> SceneNode {
        guard !materials.isEmpty else { throw GroupedMeshError.materialCount }
        return SceneNode(children: try materials.map { material in
            SceneNode(mesh: try MeshRenderer(
                device: device, colorPixelFormat: colorPixelFormat, geometry: geometry,
                material: material, depthMode: depthMode, sampleCount: sampleCount
            ))
        })
    }

    /// Reduces world-space positions of the visible subtree, or all visible
    /// roots when `nodeID` is nil. Visits posed mesh vertices, each visible
    /// instance or batch slot, and the authored positions of points and lines.
    /// Visibility follows the hierarchy independently of camera layers. Camera-
    /// facing sprite quads have no authored vertices in this reduction.
    /// Keep the scene and shared instance sets stable during the reduction.
    public static func reduceVertices<Result>(
        in graph: SceneGraph, from nodeID: UUID? = nil, into initial: Result,
        _ update: (inout Result, SIMD3<Float>) throws -> Void
    ) throws -> Result {
        let selection: Set<UUID>?
        if let nodeID {
            guard let root = graph.node(withID: nodeID) else { throw SceneGraph.EditingError.missingNode(nodeID) }
            var pending = [root]
            var ids = Set<UUID>()
            while let node = pending.popLast() {
                ids.insert(node.id)
                pending.append(contentsOf: node.children)
            }
            selection = ids
        } else {
            selection = nil
        }

        var result = initial
        func visit(_ vertices: some Sequence<SIMD3<Float>>, in transform: simd_float4x4) throws {
            for vertex in vertices {
                guard let world = try? MeshFileSnapshot.position(vertex, in: transform) else { throw Failure.nonfiniteVertex }
                try update(&result, world)
            }
        }
        for node in try graph.resolve() where node.isVisible && selection?.contains(node.id) != false {
            if let batch = node.batch {
                for object in try batch.objects(world: node.worldTransform) {
                    let geometry = try batch.geometry(for: object.instance.geometryID)
                    try visit(geometry.vertices.lazy.map(\.position), in: object.world)
                }
            }
            if let mesh = node.mesh, node.instances?.count != 0 {
                let geometry = try mesh.pickingGeometry(jointPalette: node.jointPalette, morphWeights: node.morphWeights)
                if let instances = node.instances {
                    for transform in instances.transforms {
                        try visit(geometry.vertices.lazy.map(\.position), in: node.worldTransform * transform)
                    }
                } else {
                    try visit(geometry.vertices.lazy.map(\.position), in: node.worldTransform)
                }
            }
            switch node.primitive {
            case .points(let points): try visit(points.cloud.positions, in: node.worldTransform)
            case .lines(let lines): try visit(lines.geometry.positions, in: node.worldTransform)
            case .sprites, nil: break
            }
        }
        return result
    }

    private static func pose(_ transform: simd_float4x4, instance index: Int) throws -> SceneTransform {
        do {
            let pose: SceneTransform = try SceneTransform(decomposing: transform)
            return pose
        } catch {
            throw Failure.unrepresentableTransform(instance: index)
        }
    }

    private static func parameters(
        _ parameters: SIMD4<Float>, tint: SIMD4<Float>, material: MeshMaterial, instance index: Int
    ) throws -> SIMD4<Float> {
        if tint == SIMD4(repeating: 1) { return parameters }
        switch material {
        case .lambert: return parameters
        case .custom:
            throw Failure.unsupportedTint(instance: index)
        case .physicallyBased(let material) where material.surfaceGraph != nil:
            throw Failure.unsupportedTint(instance: index)
        default:
            let product = parameters * tint
            guard (0..<4).allSatisfy({ product[$0].isFinite }) else { throw Failure.nonfiniteTint(instance: index) }
            return product
        }
    }

    private static func childName(_ node: SceneNode, _ index: Int) -> String {
        "\(node.name ?? "Instance")_\(index)"
    }

    private static func wrapper(_ node: SceneNode, around meshes: [SceneNode]) -> SceneNode {
        var wrapper = node
        wrapper.mesh = nil
        wrapper.batch = nil
        wrapper.instances = nil
        wrapper.jointPalette = nil
        wrapper.morphWeights = nil
        // The source's draw order is now on the generated meshes. Keep an
        // existing primitive's order without turning the wrapper into a group
        // that changes the order of the original child subtree.
        if wrapper.primitive == nil {
            wrapper.renderOrder = 0
            wrapper.cullingBounds = nil
        }
        wrapper.children = meshes + node.children
        return wrapper
    }
}
