import Foundation
import Metal
import Synchronization
import simd

/// What a render hook learns about one node's draw. It is a plain value: it holds no
/// renderer, no encoder and no scene reference, so a hook can read it from any thread
/// but cannot change the draw it describes.
///
/// A context is built only for a draw that is actually encoded, and only when its node has
/// a hook or the graph has an inspector. Hidden nodes, nodes outside the graph's
/// `cameraLayers` or an array-camera view's layers, nodes outside the camera frustum,
/// instanced nodes with no instances, empty draws and every draw of an encode with an empty
/// scissor never reach a hook. Three.js r187 applies the same rule to hidden, filtered and
/// frustum-culled objects, which `projectObject` never gives to `renderObject`. The count and
/// scissor rules are Frostfield's own. A throw while a draw encodes skips that node's after hook.
public struct SceneRenderObjectContext: Equatable, Sendable {
    /// The scene pass that encodes the draw.
    public enum Pass: Equatable, Sendable {
        /// The scene color pass, including the opaque and the transmission overlay passes
        /// that one color encode may split into.
        case color
        /// The `SceneGraph.encodeSurfaceAttributes` prepass.
        case surfaceAttributes
    }

    /// The drawn node's stable ID.
    public let nodeID: UUID
    /// The kind of draw: an ordinary or instanced mesh (`.mesh`), a batched mesh (`.batch`),
    /// or points, lines or sprites. A skinned mesh is a `.mesh`. A node that carries both a
    /// mesh and a primitive reports two contexts, one per draw.
    public let kind: SceneRenderOrdering.Kind
    public let pass: Pass
    /// The draw's position in this encode's draw order, counting from 0 across every draw
    /// of the encode, whether or not its node has a hook. Mesh and primitive draws share the
    /// count, and an encode that splits into opaque and overlay passes continues it. It
    /// restarts at 0 for each view of an array-camera encode.
    public let drawIndex: Int
    /// The index of the array-camera view being encoded, or nil outside an array-camera encode.
    public let viewIndex: Int?
    /// The number of views in the array-camera encode, or nil outside one.
    public let viewCount: Int?
    public let depthMode: MeshDepthMode
    /// The node's world transform, resolved from the graph exactly as the draw uses it.
    /// With `SceneGraph.highPrecision` the GPU draws from a camera-relative Double model
    /// matrix, and this is the Float world matrix those are derived from.
    public let worldTransform: simd_float4x4
    /// The material parameters the draw is encoded with: the node's `materialParameters`, or
    /// the `SceneMaterialOverride`'s (or `ScenePrimitiveMaterialOverride`'s) parameters when an
    /// override replaced the node's material. Per-instance tints and vertex colors are not folded in.
    public let materialParameters: SIMD4<Float>
    /// Whether a scene material override replaced the node's authored material for this draw.
    public let usesMaterialOverride: Bool
    /// Whether the draw is in the blended (transparent) phase.
    public let isBlended: Bool
    /// The number of instances of an instanced mesh, or nil for any other draw.
    public let instanceCount: Int?
}

/// Identifies one scene encode to a ``SceneEncodeInspector``.
public struct SceneEncodeInfo: Equatable, Sendable {
    /// Distinct for every observed encode in this process; 0 is never used. It correlates
    /// the begin, draw and finish callbacks of one encode, like the `uid` that Three.js r187
    /// passes to `Inspector.beginRender` and `finishRender`.
    public let id: UInt64
    public let pass: SceneRenderObjectContext.Pass
    /// The index of the array-camera view being encoded, or nil outside an array-camera encode.
    public let viewIndex: Int?
    public let viewCount: Int?
    public let depthMode: MeshDepthMode

    private static let lastID = Atomic<UInt64>(0)

    static func nextID() -> UInt64 {
        lastID.add(1, ordering: .relaxed).newValue
    }
}

/// An opt-in observer of scene encodes, like the inspector hooks of Three.js r187's
/// `Renderer.inspector` (`beginRender` and `finishRender`) with a per-draw callback added.
/// Set it on `SceneGraph.encodeInspector`; with none set, encoding does no inspector work.
///
/// Every callback runs synchronously on the thread that encodes, while the scene's render
/// encoder is open. Do not encode into the same command buffer from a callback, and keep
/// callbacks short. The default implementations do nothing, so an inspector overrides only
/// the events it needs.
public protocol SceneEncodeInspector: AnyObject {
    /// An encode of `graph` is starting. It is called before the encode validates its input
    /// and after the graph's debug helpers are refreshed, so a failed encode still calls
    /// `sceneEncodeDidFail`. Array-camera preflight validation encodes nothing and is not reported.
    func sceneEncodeDidBegin(_ info: SceneEncodeInfo, graph: SceneGraph)

    /// One draw's commands were just encoded. `statistics` holds only that draw's counts, in
    /// the fields `SceneDrawStatistics` documents for the color or attribute pass; adding every
    /// draw's `statistics` gives the pass's `drawnMeshCount`, `drawnPrimitiveCount`,
    /// `drawnInstanceCount`, `submittedTriangleCount`, point, line and sprite counts and indirect
    /// counts. A toon outline hull drawn with a node adds its triangles to the encode total
    /// but not to this draw's `submittedTriangleCount`. Culling, shadow and toon-outline-after-scene
    /// counters belong to the encode total only.
    func sceneEncode(_ info: SceneEncodeInfo, didEncode draw: SceneRenderObjectContext, statistics: SceneDrawStatistics)

    /// The encode succeeded; `statistics` is exactly what the encode returns.
    func sceneEncodeDidFinish(_ info: SceneEncodeInfo, statistics: SceneDrawStatistics)

    /// The encode threw after `sceneEncodeDidBegin`. Discard the command buffer, as documented
    /// for the encode that failed.
    func sceneEncodeDidFail(_ info: SceneEncodeInfo, error: any Error)
}

public extension SceneEncodeInspector {
    func sceneEncodeDidBegin(_ info: SceneEncodeInfo, graph: SceneGraph) {}
    func sceneEncode(_ info: SceneEncodeInfo, didEncode draw: SceneRenderObjectContext, statistics: SceneDrawStatistics) {}
    func sceneEncodeDidFinish(_ info: SceneEncodeInfo, statistics: SceneDrawStatistics) {}
    func sceneEncodeDidFail(_ info: SceneEncodeInfo, error: any Error) {}
}
