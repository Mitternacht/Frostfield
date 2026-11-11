import Foundation
import Metal
import simd

public extension SceneNode {
    /// Excludes this node and its children when a debug box measures a subtree.
    /// DebugHelpers sets this for its own nodes. The marker is stored in the
    /// node's metadata under `frostfield.debugBoundsExcluded`.
    var excludesFromDebugBounds: Bool {
        get { metadata["frostfield.debugBoundsExcluded"] == .bool(true) }
        set { metadata["frostfield.debugBoundsExcluded"] = newValue ? .bool(true) : nil }
    }
}

public extension DebugHelpers {
    /// A camera-local line node that inherits its source node's pose and visibility.
    struct CameraFrustumAttachment {
        public let nodeID: UUID
        /// Used the next time `update(projection:depthMode:in:)` is called.
        public var colors: LineGeometry.CameraFrustumColors
        fileprivate let helpers: DebugHelpers

        /// Refreshes a projection after lens, view offset or clip-plane changes.
        /// Keep the attachment under the camera node it visualizes.
        public func update(
            projection: simd_float4x4, depthMode: MeshDepthMode = .forward, in graph: inout SceneGraph
        ) throws {
            let geometry = try LineGeometry.cameraFrustum(projection: projection, depthMode: depthMode, colors: colors)
            try helpers.replaceLines(geometry, on: nodeID, in: &graph)
        }
    }

    /// A stable root helper whose line geometry is sampled from a graph's joints.
    struct SkeletonAttachment {
        public let nodeID: UUID
        public let joints: [UUID]
        public let connections: LineGeometry.SkeletonConnections
        public let visibleOnly: Bool
        /// The endpoint colors to use on the next pose refresh.
        public var colors: LineGeometry.SkeletonColors
        fileprivate let helpers: DebugHelpers

        /// Samples the graph after its pose or visibility changes. Keep the
        /// helper at the root with an identity transform. A helper with no
        /// matching links stays in the graph and draws nothing.
        public func update(in graph: inout SceneGraph) throws {
            let geometry = try LineGeometry.skeleton(
                in: graph, joints: joints, connections: connections, visibleOnly: visibleOnly, colors: colors
            )
            try helpers.replaceLines(geometry, on: nodeID, in: &graph)
        }
    }

    /// A stable line node measuring a source subtree's world-aligned box.
    struct BoxAttachment {
        public let nodeID: UUID
        /// The subtree to measure on the next refresh.
        public var sourceID: UUID
        public var color: SIMD4<Float>
        fileprivate let helpers: DebugHelpers

        /// Recomputes bounds after the source or its ancestors move. A missing
        /// or empty source leaves the attachment in the graph with no lines.
        public func update(in graph: inout SceneGraph) throws {
            guard isValidPrimitiveColor(color) else { throw PrimitiveGeometryError.invalidColor(0) }
            let geometry: LineGeometry?
            if let bounds = try graph.subtreeWorldBounds(of: sourceID) {
                // Keep world alignment even when the helper is parented under
                // a rotated, reflected or nonuniformly scaled source.
                let world = try graph.worldTransform(of: nodeID)
                guard let local = try? PickingAffine(world) else { throw SceneGraph.EditingError.invalidTransform(nodeID) }
                let positions = LineGeometry.boxEdgePositions(bounds).map { point in
                    SIMD3<Float>(local.inverse * (SIMD3<Double>(point) - local.translation))
                }
                geometry = try LineGeometry(
                    positions: positions, colors: Array(repeating: color, count: positions.count), topology: .segments
                )
            } else {
                geometry = nil
            }
            try helpers.replaceLines(geometry, on: nodeID, in: &graph)
        }
    }

    /// A plane outline and fill with reusable unit geometry. The equation and
    /// size are expressed in the helper parent's coordinates, or world space
    /// when the helper is at the root.
    struct PlaneAttachment {
        public let nodeID: UUID
        public var plane: Plane3D
        public var size: Float

        public func update(in graph: inout SceneGraph) throws {
            let transform = try Self.transform(for: plane, size: size)
            try graph.updateNode(withID: nodeID) { $0.transform = transform }
        }

        fileprivate static func transform(for plane: Plane3D, size: Float) throws -> SceneTransform {
            guard size.isFinite, size > 0 else {
                throw PrimitiveGeometryError.invalidHelperParameter("plane helper size must be finite and positive")
            }
            let frame = try PlaneFrame(normal: plane.normal, constant: 0)
            return SceneTransform(
                translation: try plane.projectedPoint(.zero),
                rotation: simd_quatf(simd_float3x3(columns: (frame.u, frame.v, frame.normal))),
                scale: SIMD3(size, size, 1)
            )
        }
    }

    /// An editable arrow in its parent's coordinates. The shaft and head use
    /// reusable unit geometry, with dimensions applied through node transforms.
    struct ArrowAttachment {
        public let nodeID: UUID
        public var direction: SIMD3<Float>
        public var origin: SIMD3<Float>
        public var length: Float
        public var color: SIMD3<Float>
        /// A nil length follows 20% of the current arrow length.
        public var headLength: Float?
        /// A nil width follows 20% of the current resolved head length.
        public var headWidth: Float?
        fileprivate let shaftID: UUID
        fileprivate let headID: UUID
        fileprivate let helpers: DebugHelpers

        /// Validates and applies the changes together, keeping all three node
        /// identities. Keep the shaft and cone children attached to this helper.
        public func update(in graph: inout SceneGraph) throws {
            let layout = try ArrowLayout(
                direction: direction, origin: origin, length: length, color: color,
                headLength: headLength, headWidth: headWidth
            )
            var snapshot = graph
            try snapshot.updateNode(withID: nodeID) { $0.transform = layout.pose }
            try snapshot.updateNode(withID: shaftID) {
                $0.transform = layout.shaftTransform
                $0.isVisible = layout.shaftLength > 0
            }
            try snapshot.updateNode(withID: headID) {
                $0.transform = layout.headTransform
                $0.materialParameters = SIMD4(color, 1)
            }
            let shaft = try ArrowLayout.shaftGeometry(length: 1, color: color)
            try helpers.replaceLines(shaft, on: shaftID, in: &snapshot)
            graph = snapshot
        }
    }

    enum LiveHelperRefreshFailure: LocalizedError, Equatable {
        case commandBuffer

        public var errorDescription: String? {
            "Render-time box and plane refresh needs an uncommitted Metal command buffer with retained references."
        }
    }

    enum SkeletonRefreshFailure: LocalizedError, Equatable {
        case commandBuffer

        public var errorDescription: String? {
            "Render-time skeleton refresh needs an uncommitted Metal command buffer with retained references."
        }
    }

    /// Inserts a frustum as a child of an authored camera pose node. It follows
    /// the current world matrix, including parent scale, pivot and orientation mode.
    /// Supply the Metal projection in the camera's local coordinates.
    @discardableResult
    func addCameraFrustum(
        projection: simd_float4x4, toCamera cameraID: UUID, in graph: inout SceneGraph,
        depthMode: MeshDepthMode = .forward, colors: LineGeometry.CameraFrustumColors = .init()
    ) throws -> CameraFrustumAttachment {
        var helper = try node(.cameraFrustum(projection: projection, depthMode: depthMode, colors: colors))
        helper.castsShadow = false
        helper.name = "Camera frustum helper"
        try graph.add(helper, toParent: cameraID)
        return CameraFrustumAttachment(nodeID: helper.id, colors: colors, helpers: self)
    }

    /// Inserts a world-space skeleton helper at the scene root. By default a
    /// link requires two directly related joints, as in Three.js. Use
    /// `visibleOnly` to skip source joints hidden by an ancestor. The helper
    /// draws over other geometry unless `depthTest` is enabled.
    @discardableResult
    func addSkeleton(
        joints: [UUID], to graph: inout SceneGraph,
        connections: LineGeometry.SkeletonConnections = .directJointParent, visibleOnly: Bool = false,
        colors: LineGeometry.SkeletonColors = .init(), depthTest: Bool = false
    ) throws -> SkeletonAttachment {
        let geometry = try LineGeometry.skeleton(
            in: graph, joints: joints, connections: connections, visibleOnly: visibleOnly, colors: colors
        )
        var material = lineMaterial
        material.depthTest = depthTest
        let helpers = DebugHelpers(
            device: device, colorPixelFormat: colorPixelFormat, lineMaterial: material, depthMode: depthMode
        )
        var helper = SceneNode(castsShadow: false, name: "Skeleton helper")
        helper.excludesFromDebugBounds = true
        if let geometry { helper.primitive = try helpers.node(geometry).primitive }
        try graph.add(helper)
        return SkeletonAttachment(nodeID: helper.id, joints: joints, connections: connections,
                                  visibleOnly: visibleOnly, colors: colors, helpers: helpers)
    }

    /// Attaches a live box at the root or under a parent. It can be a child
    /// of the measured object. Refresh manually or pass it to the color encode.
    @discardableResult
    func addBox(
        of sourceID: UUID, to graph: inout SceneGraph, parent: UUID? = nil,
        color: SIMD4<Float> = LineGeometry.helperColor(0xffff00)
    ) throws -> BoxAttachment {
        var helper = SceneNode(castsShadow: false, name: "Box helper")
        helper.excludesFromDebugBounds = true
        let attachment = BoxAttachment(nodeID: helper.id, sourceID: sourceID, color: color, helpers: self)
        var snapshot = graph
        try snapshot.add(helper, toParent: parent)
        try attachment.update(in: &snapshot)
        graph = snapshot
        return attachment
    }

    /// Attaches a plane visualizer. Changes to the equation and size update its
    /// transform. Supply a parent to keep the plane in that node's coordinates.
    @discardableResult
    func addPlane(
        _ plane: Plane3D, to graph: inout SceneGraph, parent: UUID? = nil,
        size: Float = 1, color: SIMD3<Float> = SIMD3(1, 1, 0)
    ) throws -> PlaneAttachment {
        let transform = try PlaneAttachment.transform(for: plane, size: size)
        var helper = try self.plane(normal: SIMD3(0, 0, 1), constant: 0, color: color)
        helper.transform = transform
        helper.castsShadow = false
        helper.name = "Plane helper"
        try graph.add(helper, toParent: parent)
        return PlaneAttachment(nodeID: helper.id, plane: plane, size: size)
    }

    /// Attaches an arrow whose direction, length, color and head dimensions
    /// can be edited without replacing its nodes. Update after editing the attachment.
    @discardableResult
    func addArrow(
        direction: SIMD3<Float>, origin: SIMD3<Float> = .zero, length: Float = 1,
        color: SIMD3<Float> = SIMD3(1, 1, 0), headLength: Float? = nil, headWidth: Float? = nil,
        to graph: inout SceneGraph, parent: UUID? = nil
    ) throws -> ArrowAttachment {
        let layout = try ArrowLayout(
            direction: direction, origin: origin, length: length, color: color,
            headLength: headLength, headWidth: headWidth
        )
        var shaft = try node(ArrowLayout.shaftGeometry(length: 1, color: color), transform: layout.shaftTransform)
        shaft.name = "Arrow shaft"
        shaft.isVisible = layout.shaftLength > 0
        var head = try arrowHead(base: 0, tip: 1, width: 1, color: SIMD3(repeating: 1))
        head.name = "Arrow head"
        head.transform = layout.headTransform
        head.materialParameters = SIMD4(color, 1)
        var arrow = SceneNode(transform: layout.pose, children: [shaft, head], castsShadow: false, name: "Arrow helper")
        arrow.excludesFromDebugBounds = true
        try graph.add(arrow, toParent: parent)
        return ArrowAttachment(
            nodeID: arrow.id, direction: direction, origin: origin, length: length, color: color,
            headLength: headLength, headWidth: headWidth, shaftID: shaft.id, headID: head.id, helpers: self
        )
    }

    private func replaceLines(_ geometry: LineGeometry?, on id: UUID, in graph: inout SceneGraph) throws {
        try graph.updateNode(withID: id) { helper in
            guard let geometry else {
                helper.primitive = nil
                return
            }
            if case .lines(let current) = helper.primitive {
                let old = current.geometry
                guard old.positions != geometry.positions || old.colors != geometry.colors || old.topology != geometry.topology else { return }
                helper.primitive = .lines(try current.replacingGeometry(geometry))
            } else {
                helper.primitive = try node(geometry).primitive
            }
        }
    }
}

extension SceneGraph {
    func refreshingDebugHelpers(
        skeletons: [DebugHelpers.SkeletonAttachment], boxes: [DebugHelpers.BoxAttachment],
        planes: [DebugHelpers.PlaneAttachment], commandBuffer: any MTLCommandBuffer
    ) throws -> SceneGraph {
        let hasLiveHelpers = !boxes.isEmpty || !planes.isEmpty
        guard hasLiveHelpers || !skeletons.isEmpty else { return self }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            if hasLiveHelpers { throw DebugHelpers.LiveHelperRefreshFailure.commandBuffer }
            throw DebugHelpers.SkeletonRefreshFailure.commandBuffer
        }
        // This value keeps the caller's graph and archived resources untouched.
        // Metal retains any buffers used after the temporary graph is released.
        var snapshot = self
        for attachment in skeletons { try attachment.update(in: &snapshot) }
        for attachment in boxes { try attachment.update(in: &snapshot) }
        for attachment in planes { try attachment.update(in: &snapshot) }
        return snapshot
    }
}
