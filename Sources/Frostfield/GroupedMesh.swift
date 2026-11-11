import Foundation
import Metal

extension SceneNode {
    /// A node that draws each material group of `geometry` with its own material,
    /// like a Three.js `Mesh` with a material array. The node gets one child per
    /// group, in group order, holding a compact copy of that group's triangles. A
    /// geometry without groups draws every triangle with `materials[0]`.
    /// Supply a source-index `drawRange` to limit the prepared children, or
    /// keep a `GroupedMesh` when the range needs to change later.
    ///
    /// `renderStates`, when supplied, holds one optional state per material slot.
    /// Children share the node's transform and inherit nothing else; set material
    /// parameters and shadows on the children afterward if needed. A render order on
    /// the returned node orders all its children as a group, like on a Three.js `Group`.
    public static func grouped(
        _ geometry: MeshGeometry,
        materials: [MeshMaterial],
        device: any MTLDevice,
        colorPixelFormat: MTLPixelFormat,
        depthMode: MeshDepthMode = .forward,
        sampleCount: Int = 1,
        renderStates: [MeshRenderState?]? = nil,
        drawRange: GeometryDrawRange = .all,
        id: UUID = UUID(),
        transform: SceneTransform = SceneTransform()
    ) throws -> SceneNode {
        try GroupedMesh(
            geometry, materials: materials, device: device, colorPixelFormat: colorPixelFormat,
            depthMode: depthMode, sampleCount: sampleCount, renderStates: renderStates, drawRange: drawRange,
            id: id, transform: transform
        ).node
    }
}

/// Prepared material groups that share one draw range in the original index order.
public final class GroupedMesh {
    /// A scene node with one prepared mesh child per material group. Copies
    /// placed in a graph share the prepared renderers and receive later range edits.
    public let node: SceneNode

    /// The window in the source index buffer. Set it between encodes. Each
    /// group submits complete triangles from its intersection with the window.
    public var drawRange: GeometryDrawRange {
        didSet { applyDrawRange() }
    }

    private let draws: [(group: MeshGeometryGroup, renderer: MeshRenderer)]
    private let indexCount: Int

    public init(
        _ geometry: MeshGeometry,
        materials: [MeshMaterial],
        device: any MTLDevice,
        colorPixelFormat: MTLPixelFormat,
        depthMode: MeshDepthMode = .forward,
        sampleCount: Int = 1,
        renderStates: [MeshRenderState?]? = nil,
        drawRange: GeometryDrawRange = .all,
        id: UUID = UUID(),
        transform: SceneTransform = SceneTransform()
    ) throws {
        guard !materials.isEmpty, renderStates.map({ $0.count == materials.count }) ?? true else {
            throw GroupedMeshError.materialCount
        }
        let groups = geometry.groups.isEmpty
            ? [MeshGeometryGroup(start: 0, count: geometry.indices.count, materialIndex: 0)]
            : geometry.groups
        let draws = try groups.map { group -> (group: MeshGeometryGroup, renderer: MeshRenderer) in
            guard materials.indices.contains(group.materialIndex) else {
                throw GroupedMeshError.missingMaterial(group.materialIndex)
            }
            let part = groups.count == 1 && group.range == geometry.indices.indices
                ? geometry : try geometry.subset(group.range)
            let renderer = try MeshRenderer(
                device: device, colorPixelFormat: colorPixelFormat, geometry: part,
                material: materials[group.materialIndex], depthMode: depthMode,
                renderState: renderStates?[group.materialIndex] ?? nil, sampleCount: sampleCount
            )
            return (group, renderer)
        }
        self.draws = draws
        indexCount = geometry.indices.count
        node = SceneNode(id: id, transform: transform, children: draws.map { SceneNode(mesh: $0.renderer) })
        self.drawRange = drawRange
        applyDrawRange()
    }

    private func applyDrawRange() {
        let selected = drawRange.resolve(elementCount: indexCount)
        for (group, renderer) in draws {
            renderer.drawRange = GeometryDrawRange.intersection(of: selected, in: group.range)
        }
    }
}

public enum GroupedMeshError: LocalizedError, Equatable {
    case materialCount
    case missingMaterial(Int)

    public var errorDescription: String? {
        switch self {
        case .materialCount: "Supply at least one material, and one render state per material when render states are given."
        case .missingMaterial(let index): "A mesh group uses material slot \(index), which the material array does not have."
        }
    }
}
