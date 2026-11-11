import Foundation
import simd

/// Ordering for the visible draws of one scene color encode.
public struct SceneRenderOrdering {
    public enum Kind: Equatable, Sendable {
        case mesh, batch, points, lines, sprites
    }

    public struct Item: Equatable, Sendable {
        public let nodeID: UUID
        public let kind: Kind
        public let groupOrder: Int
        public let renderOrder: Int
        /// Projected depth in the camera's forward convention, where nearer is
        /// smaller. Nil when the draw's sorting point has no valid projection.
        public let projectedDepth: Double?
    }

    /// Return ascending when the first draw belongs earlier in the pass.
    /// Equal items retain graph order. Keep the comparison consistent and the
    /// scene and prepared resources stable for the duration of the encode.
    public typealias Comparator = (Item, Item) -> ComparisonResult

    public let sortObjects: Bool
    public let opaque: Comparator?
    public let transparent: Comparator?

    /// With sorting disabled, traversal order is used within each phase.
    /// Custom comparators replace the default order in their phase.
    /// Opaque, transparent and depth-disabled overlay phases remain separate.
    public init(sortObjects: Bool = true, opaque: Comparator? = nil, transparent: Comparator? = nil) {
        self.sortObjects = sortObjects
        self.opaque = opaque
        self.transparent = transparent
    }

    var recordsOpaqueDepth: Bool { sortObjects && opaque != nil }

    func precedes(
        _ lhs: SceneColorOrder, _ rhs: SceneColorOrder, nodes: [ResolvedSceneNode], depthMode: MeshDepthMode
    ) -> Bool {
        guard lhs.phase == rhs.phase else { return lhs.phase.rawValue < rhs.phase.rawValue }
        guard sortObjects else { return lhs.nodeIndex < rhs.nodeIndex }
        let comparator = switch lhs.phase {
        case .opaque: opaque
        case .blended: transparent
        case .overlay: nil as Comparator?
        }
        guard let comparator else { return lhs < rhs }
        let first = lhs.item(nodeID: nodes[lhs.nodeIndex].id, depthMode: depthMode)
        let second = rhs.item(nodeID: nodes[rhs.nodeIndex].id, depthMode: depthMode)
        switch comparator(first, second) {
        case .orderedAscending: return true
        case .orderedDescending: return false
        case .orderedSame: return lhs.nodeIndex < rhs.nodeIndex
        }
    }
}

/// The shared color order for meshes and primitives.
struct SceneColorOrder: Comparable {
    enum Phase: Int {
        case opaque, blended, overlay
    }

    let phase: Phase
    let kind: SceneRenderOrdering.Kind
    let groupOrder: Int
    let renderOrder: Int
    let depth: Double
    let hasDepth: Bool
    let nodeIndex: Int

    var needsSorting: Bool { phase != .opaque || groupOrder != 0 || renderOrder != 0 }

    init(
        phase: Phase, kind: SceneRenderOrdering.Kind, node: ResolvedSceneNode, nodeIndex: Int,
        clipPosition: SIMD4<Float>, depthMode: MeshDepthMode, recordsOpaqueDepth: Bool,
        logarithmicDepth: LogarithmicRasterDepth? = nil
    ) {
        self.phase = phase
        self.kind = kind
        self.nodeIndex = nodeIndex
        groupOrder = phase == .overlay ? 0 : node.groupOrder
        renderOrder = phase == .overlay ? 0 : node.renderOrder
        hasDepth = (phase == .blended || recordsOpaqueDepth && phase == .opaque)
            && clipPosition.w.isFinite && clipPosition.w > 0 && clipPosition.z.isFinite
        if hasDepth {
            let normalizedDepth = logarithmicDepth?.projectedDepth(at: clipPosition)
                ?? Double(clipPosition.z) / Double(clipPosition.w)
            depth = depthMode == .reversed ? normalizedDepth : -normalizedDepth
        } else {
            depth = 0
        }
    }

    func item(nodeID: UUID, depthMode: MeshDepthMode) -> SceneRenderOrdering.Item {
        .init(nodeID: nodeID, kind: kind, groupOrder: groupOrder, renderOrder: renderOrder,
              projectedDepth: hasDepth ? (depthMode == .reversed ? 1 - depth : -depth) : nil)
    }

    static func < (lhs: Self, rhs: Self) -> Bool {
        (lhs.phase.rawValue, lhs.groupOrder, lhs.renderOrder, lhs.depth, lhs.nodeIndex)
            < (rhs.phase.rawValue, rhs.groupOrder, rhs.renderOrder, rhs.depth, rhs.nodeIndex)
    }
}
