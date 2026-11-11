import Foundation
import Metal
import simd

public extension DebugHelpers {
    struct AnimationPathOptions: Hashable, Sendable {
        /// Number of equal time intervals across the clip.
        public var divisions: Int
        /// The path line's linear RGBA.
        public var color: SIMD4<Float>
        public var showsMarkers: Bool
        /// The keyframe markers' linear RGBA.
        public var markerColor: SIMD4<Float>
        /// Screen-space marker diameter in pixels.
        public var markerSize: Float

        public init(
            divisions: Int = 100, color: SIMD4<Float> = SIMD4(0, 1, 0, 1),
            showsMarkers: Bool = true, markerColor: SIMD4<Float> = SIMD4(1, 0, 0, 1), markerSize: Float = 5
        ) {
            self.divisions = divisions
            self.color = color
            self.showsMarkers = showsMarkers
            self.markerColor = markerColor
            self.markerSize = markerSize
        }
    }

    enum AnimationPathFailure: LocalizedError, Equatable {
        case divisions
        case missingTranslationTrack(UUID)

        public var errorDescription: String? {
            switch self {
            case .divisions: "An animation path needs between 1 and 65,536 divisions."
            case .missingTranslationTrack(let id): "The clip has no translation track for scene node \(id)."
            }
        }
    }

    /// Prepares a snapshot of one node's authored translation curve, like
    /// Three.js `AnimationPathHelper`. Use `AnimationClip(animation)` with a
    /// `SceneAnimation`. The line samples the clip from zero to its duration
    /// and the optional markers identify every authored translation key.
    ///
    /// Add the returned node at the scene root. Its geometry is in world space
    /// under the parent's current transform, including scale and shear. Prepare
    /// another snapshot if the parent pose changes. Other tracks and mixer
    /// weights are outside this translation visualization. The input graph and
    /// its live resources are never edited.
    func animationPath(
        for nodeID: UUID, in graph: SceneGraph, clip: AnimationClip, options: AnimationPathOptions = .init()
    ) throws -> SceneNode {
        try prepareAnimationPath(for: nodeID, in: graph, clip: clip, options: options, bakesParentTransform: true).node
    }

    /// Adds a helper as a sibling of the selected node and returns its ID.
    /// The line and markers stay in the parent's local coordinates, so every
    /// draw follows the current parent and ancestor transforms without rebuilding
    /// either renderer. A root target gets a root helper. The helper inherits
    /// ancestor visibility and can be hidden independently of its target.
    ///
    /// If the target changes parents, remove the helper and add it under the
    /// new parent to keep its geometry local, or prepare a new helper. Preparation
    /// and insertion errors leave the graph unchanged.
    @discardableResult
    func addAnimationPath(
        for nodeID: UUID, to graph: inout SceneGraph, clip: AnimationClip, options: AnimationPathOptions = .init()
    ) throws -> UUID {
        let helper = try prepareAnimationPath(
            for: nodeID, in: graph, clip: clip, options: options, bakesParentTransform: false
        )
        try graph.add(helper.node, toParent: helper.parentID)
        return helper.node.id
    }

    private func prepareAnimationPath(
        for nodeID: UUID, in graph: SceneGraph, clip: AnimationClip, options: AnimationPathOptions, bakesParentTransform: Bool
    ) throws -> (node: SceneNode, parentID: UUID?) {
        guard (1...65_536).contains(options.divisions) else { throw AnimationPathFailure.divisions }
        guard let ancestors = graph.ancestors(of: nodeID) else { throw SceneGraph.EditingError.missingNode(nodeID) }
        let track = try clip.pathTranslation(of: nodeID)
        let parentID = ancestors.first?.id
        let bakedParent = bakesParentTransform ? parentID : nil
        let transform = try bakedParent.map { try graph.worldTransform(of: $0) } ?? matrix_identity_float4x4
        func sample(_ time: Double) throws -> SIMD3<Float> {
            let point = transform * SIMD4(try track.sample(time), 1)
            return SIMD3(point.x, point.y, point.z)
        }
        let divisions = options.divisions
        let duration = Double(clip.duration)
        let line = try LineGeometry(positions: (0...divisions).map {
            try sample(Double($0) / Double(divisions) * duration)
        })
        let points = try options.showsMarkers ? PointCloud(positions: track.times.map { try sample(Double($0)) }) : nil

        var material = lineMaterial
        material.color = options.color
        var lineNode = SceneNode(castsShadow: false, name: "Animation path")
        lineNode.primitive = .lines(try LineRenderer(
            device: device, colorPixelFormat: colorPixelFormat, geometry: line, material: material
        ))
        var children = [lineNode]
        if let points {
            let markers = PointsMaterial(size: .pixels(options.markerSize), color: options.markerColor,
                                         depthTest: material.depthTest)
            var markerNode = SceneNode(castsShadow: false, name: "Translation keys")
            markerNode.primitive = .points(try PointsRenderer(
                device: device, colorPixelFormat: colorPixelFormat, cloud: points, material: markers
            ))
            children.append(markerNode)
        }
        return (SceneNode(children: children, castsShadow: false, name: "Animation path helper"), parentID)
    }
}

private extension AnimationClip {
    /// Selects and samples only the relevant track, without converting or
    /// evaluating the remaining clip (which can include live render resources).
    func pathTranslation(of id: UUID) throws -> (times: [Float], sample: (Double) throws -> SIMD3<Float>) {
        let missing = DebugHelpers.AnimationPathFailure.missingTranslationTrack(id)
        if let index = trackIndex(of: .node(id, .translation)) {
            let track = tracks[index]
            return (track.times, { time in
                guard let position = try track.sample(at: time).vector3Value else { throw missing }
                return position
            })
        }
        guard let index = targetIndex(of: id) else { throw missing }
        let target = targets[index]
        if let transform = target.transform {
            return (transform.keyframes.map(\.time), { transform.sample(at: $0).translation })
        }
        if let channel = target.channels.first(where: { $0.target == .translation }) {
            return (channel.times, { time in
                var pose = AnimationNodePose()
                try channel.sample(at: time, into: &pose)
                guard let position = pose.translation else { throw missing }
                return position
            })
        }
        throw missing
    }
}
