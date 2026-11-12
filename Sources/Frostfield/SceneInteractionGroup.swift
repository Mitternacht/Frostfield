import Foundation
import simd

/// Routes native pointer and controller input to the nearest visible drawable
/// among a scene node's direct children. The host supplies events and owns
/// its gesture recognizers or controller subscriptions.
public struct SceneInteractionGroup {
    public enum Phase: Sendable {
        case down, move, up, click
    }

    public struct Event {
        public let phase: Phase
        public let hit: SceneHit
        /// The interpolated first UV set with a top-left origin.
        public var textureCoordinate: SIMD2<Float> { hit.intersection.textureCoordinate }
    }

    public struct DrawableEvent {
        public let phase: Phase
        public let hit: SceneRaycastHit
        /// The interpolated mesh UV or sprite texture coordinate, from the top left.
        public var textureCoordinate: SIMD2<Float>? { hit.textureCoordinate }
    }

    private enum Handler {
        case mesh((Event) -> Void)
        case drawable((DrawableEvent) -> Void)
    }

    public let nodeID: UUID
    private var handlers: [UUID: Handler] = [:]

    public init(nodeID: UUID) {
        self.nodeID = nodeID
    }

    /// Installs or removes a child's handler. Only the winning child receives
    /// the event. A winning child without a handler still stops the ray.
    public mutating func setHandler(for childID: UUID, _ handler: ((Event) -> Void)?) {
        handlers[childID] = handler.map(Handler.mesh)
    }

    /// Sets the child's mesh and primitive handler in place of any existing
    /// handler. Pass nil to remove it.
    public mutating func setDrawableHandler(for childID: UUID, _ handler: ((DrawableEvent) -> Void)?) {
        handlers[childID] = handler.map(Handler.drawable)
    }

    /// Routes a top-left normalized pointer event through the graph camera.
    @discardableResult
    public func dispatch(
        _ phase: Phase, at normalized: SIMD2<Float>, aspectRatio: Float, in graph: SceneGraph
    ) throws -> SceneHit? {
        let ray = try graph.cameraRay(at: normalized, aspectRatio: aspectRatio)
        return try dispatch(phase, ray: ray, in: graph)
    }

    /// Routes a pointer through an explicit render camera.
    @discardableResult
    public func dispatch(
        _ phase: Phase, at normalized: SIMD2<Float>, camera: ShaderGraphCamera,
        depthMode: MeshDepthMode = .forward, in graph: SceneGraph
    ) throws -> SceneHit? {
        deliver(phase, to: try graph.pick(at: normalized, camera: camera, depthMode: depthMode, in: nodeID))
    }

    /// Routes a pointer using the viewport's pixel dimensions so that points,
    /// lines and sprites can compete with meshes. Uses the graph camera unless
    /// an explicit camera and its depth mode are supplied.
    @discardableResult
    public func dispatch(
        _ phase: Phase, at normalized: SIMD2<Float>, drawableSize: SIMD2<Float>,
        tolerance: PrimitivePickingTolerance = .pixels(0), camera: ShaderGraphCamera? = nil,
        depthMode: MeshDepthMode = .forward, in graph: SceneGraph
    ) throws -> SceneRaycastHit? {
        guard let hit = try graph.pick(
            at: normalized, drawableSize: drawableSize, tolerance: tolerance, camera: camera,
            depthMode: depthMode, in: nodeID
        ) else { return nil }
        deliver(phase, to: hit)
        return hit
    }

    /// Routes a world-space ray, such as one supplied by a spatial input
    /// device. Use a render camera's `ray(at:depthMode:)` for a custom viewport.
    /// Throws `SceneGraph.EditingError.missingNode` if the group no longer exists.
    @discardableResult
    public func dispatch(_ phase: Phase, ray: SceneRay, in graph: SceneGraph) throws -> SceneHit? {
        deliver(phase, to: try graph.pick(ray: ray, in: nodeID))
    }

    /// Routes a world-space controller ray to any drawable. The render view
    /// orients sprites and the thresholds measure points and lines in world units.
    @discardableResult
    public func dispatch(
        _ phase: Phase, ray: SceneRay, camera: ShaderGraphCamera, drawableSize: SIMD2<Float>,
        pointThreshold: Float = 1, lineThreshold: Float = 1, in graph: SceneGraph
    ) throws -> SceneRaycastHit? {
        guard let hit = try graph.raycast(
            ray: ray, camera: camera, drawableSize: drawableSize,
            pointThreshold: pointThreshold, lineThreshold: lineThreshold, in: nodeID
        ).first else { return nil }
        deliver(phase, to: hit)
        return hit
    }

    private func deliver(_ phase: Phase, to hit: SceneHit?) -> SceneHit? {
        guard let hit else { return nil }
        deliver(phase, to: .mesh(hit))
        return hit
    }

    private func deliver(_ phase: Phase, to hit: SceneRaycastHit) {
        switch handlers[hit.nodeID] {
        case .drawable(let handler): handler(DrawableEvent(phase: phase, hit: hit))
        case .mesh(let handler):
            if case .mesh(let mesh) = hit { handler(Event(phase: phase, hit: mesh)) }
        case nil: break
        }
    }
}
