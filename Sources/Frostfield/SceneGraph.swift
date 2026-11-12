import Foundation
import simd

/// One object or transform group in a scene hierarchy.
public struct SceneNode: Identifiable {
    public private(set) var id: UUID
    public var transform: SceneTransform
    /// How a child inherits rotation. Root nodes always use their local matrix.
    public var orientationMode: OrientationMode
    /// The object-space point around which scale and rotation act. Nil uses
    /// the local origin. The pivot maps to `transform.translation + pivot`
    /// in the parent's space.
    public var pivot: SIMD3<Float>?
    /// The node's matrix in its parent's space, including its authored pivot.
    public var localTransformMatrix: simd_float4x4 { transform.matrix(around: pivot) }
    public var mesh: MeshRenderer?
    /// Drawn into shadow maps instead of `mesh`, like Three.js `customDepthMaterial`.
    /// The color pass still uses `mesh`. Nil uses `mesh` for both.
    public var depthMesh: MeshRenderer? = nil
    /// Several independently placed meshes using one prepared material.
    /// Use either this, `mesh`, or `levelsOfDetail`.
    public var batch: BatchedMesh?
    public var materialParameters: SIMD4<Float>
    public var children: [SceneNode]
    public var castsShadow: Bool
    /// When false, this node is drawn fully lit even if its material samples shadow maps.
    /// Children do not inherit it. Defaults to true.
    public var receivesShadow: Bool
    /// Required when this node uses a skinned mesh.
    public var jointPalette: JointPalette?
    /// An optional light that contributes to PBR surfaces in this scene.
    public var light: PunctualLight?
    /// Raster diffuse irradiance. Position and orientation do not affect this light.
    public var ambientLight: AmbientLight?
    /// Draws the mesh once per instance in one instanced draw, like Three.js
    /// `InstancedMesh`. The mesh must be prepared with `supportsInstancing`.
    /// An empty set draws nothing. Ignored on nodes without a mesh.
    public var instances: InstanceSet?
    /// One finite weight per morph target of the mesh, like Three.js
    /// `morphTargetInfluences`. Nil draws the base mesh, as all-zero weights do.
    /// Must be nil for meshes without morph targets.
    public var morphWeights: [Float]?
    /// Whether scene graph encoding may skip the color draw when the node's
    /// bounds are outside the camera frustum, like Three.js `Object3D.frustumCulled`.
    public var frustumCulled: Bool
    /// A box in the node's local space that encloses everything the node draws,
    /// in every pose and instance. Culling uses it instead of the mesh's bounds.
    /// Skinned meshes and meshes with surface-graph position offsets are culled
    /// only when this is set.
    public var cullingBounds: AxisAlignedBox?
    /// Meshes to choose between by camera distance. Use either this or `mesh`.
    public var levelsOfDetail: LevelOfDetail?
    /// Draw order within the opaque and transparent groups, like Three.js
    /// `Object3D.renderOrder`: lower values draw first, and ties keep graph order
    /// (opaque) or sort back to front (transparent). It does not affect shadows.
    ///
    /// On a node without a mesh, levels of detail, or primitive, a nonzero value
    /// orders its descendants before their own render order. The nearest such
    /// ancestor wins. A node with render order 0 passes its ancestors' order through.
    public var renderOrder: Int
    /// Points, lines or sprites drawn with the node's transform, alongside any mesh.
    public var primitive: ScenePrimitive? = nil
    /// A label for lookup and tools, like Three.js `Object3D.name`. Names need not be unique.
    public var name: String?
    /// Whether this node and its descendants take part in rendering, like Three.js
    /// `Object3D.visible`. A hidden node's whole subtree is left out of color and
    /// shadow passes, punctual lighting, picking and path-tracing snapshots.
    public var isVisible: Bool
    /// The layers this node belongs to, like Three.js `Object3D.layers`. A node renders,
    /// lights, casts shadows and is picked only when its layers intersect the graph's
    /// `cameraLayers`. Unlike visibility, layers are not inherited by children.
    public var layers: SceneLayers
    /// Application data like Three.js `Object3D.userData`, as typed JSON-compatible values
    /// that `SceneArchive` serializes with the node.
    public var metadata: [String: SceneMetadataValue]
    /// Runs on the CPU just before this node's draw is encoded into the scene pass, like the
    /// CPU side of Three.js `Object3D.onBeforeRender`. It is called once for each draw of the
    /// node that is encoded, for every encode and every array-camera view, and never for a hidden,
    /// layer-filtered or frustum-culled node. The context carries the node's final world transform
    /// and material parameters. It receives no renderer, geometry or material objects and cannot
    /// change the draw. The GPU executes the encoded commands later, so a write to a buffer
    /// that the shader reads reaches every draw of the command buffer, not only this one.
    /// `SceneArchive` does not store these closures. See ``SceneRenderObjectContext``.
    public var onBeforeEncode: ((SceneRenderObjectContext) -> Void)? = nil
    /// Runs on the CPU right after this node's draw commands were encoded, like the CPU side of
    /// `Object3D.onAfterRender`. The same rules as `onBeforeEncode` apply.
    public var onAfterEncode: ((SceneRenderObjectContext) -> Void)? = nil

    public init(
        id: UUID = UUID(),
        transform: SceneTransform = SceneTransform(),
        mesh: MeshRenderer? = nil,
        materialParameters: SIMD4<Float> = SIMD4(repeating: 1),
        children: [SceneNode] = [],
        castsShadow: Bool = true,
        receivesShadow: Bool = true,
        jointPalette: JointPalette? = nil,
        light: PunctualLight? = nil,
        ambientLight: AmbientLight? = nil,
        instances: InstanceSet? = nil,
        morphWeights: [Float]? = nil,
        frustumCulled: Bool = true,
        cullingBounds: AxisAlignedBox? = nil,
        levelsOfDetail: LevelOfDetail? = nil,
        name: String? = nil,
        isVisible: Bool = true,
        layers: SceneLayers = .default,
        renderOrder: Int = 0,
        metadata: [String: SceneMetadataValue] = [:],
        batch: BatchedMesh? = nil,
        pivot: SIMD3<Float>? = nil,
        orientationMode: OrientationMode = .inherited
    ) {
        self.id = id
        self.transform = transform
        self.orientationMode = orientationMode
        self.pivot = pivot
        self.mesh = mesh
        self.batch = batch
        self.materialParameters = materialParameters
        self.children = children
        self.castsShadow = castsShadow
        self.receivesShadow = receivesShadow
        self.jointPalette = jointPalette
        self.light = light
        self.ambientLight = ambientLight
        self.instances = instances
        self.morphWeights = morphWeights
        self.frustumCulled = frustumCulled
        self.cullingBounds = cullingBounds
        self.levelsOfDetail = levelsOfDetail
        self.name = name
        self.isVisible = isVisible
        self.layers = layers
        self.renderOrder = renderOrder
        self.metadata = metadata
    }

    mutating func assignCloneID(_ id: UUID) {
        self.id = id
    }
}

extension SceneTransform {
    /// T(translation) × T(pivot) × R × S × T(-pivot).
    func matrix(around pivot: SIMD3<Float>?) -> simd_float4x4 {
        var local = matrix
        if let pivot {
            local.columns.3 += SIMD4(pivot, 0) - local * SIMD4(pivot, 0)
        }
        return local
    }
}

/// A node's resolved draw data, including transform-only groups.
public struct ResolvedSceneNode: Identifiable {
    public let id: UUID
    public let worldTransform: simd_float4x4
    /// The node's mesh, or the selected level's mesh for a level-of-detail node.
    public let mesh: MeshRenderer?
    /// The mesh that casts this node's shadow. Nil means `mesh`.
    public let depthMesh: MeshRenderer?
    public let batch: BatchedMesh?
    public let materialParameters: SIMD4<Float>
    public let castsShadow: Bool
    public let receivesShadow: Bool
    public let jointPalette: JointPalette?
    public let light: ResolvedPunctualLight?
    public let ambientLight: AmbientLight?
    /// The node's instances, morph weights, culling opt-in and culling bounds, as authored.
    public let instances: InstanceSet?
    public let morphWeights: [Float]?
    public let frustumCulled: Bool
    public let cullingBounds: AxisAlignedBox?
    /// The selected level index when the node has levels of detail.
    public let levelOfDetail: Int?
    /// The node's authored render order.
    public let renderOrder: Int
    /// The render order of the nearest non-drawing ancestor that sets a nonzero one, or 0.
    /// Draws sort by it before `renderOrder`.
    public let groupOrder: Int
    /// The node's points, lines or sprites.
    public let primitive: ScenePrimitive?
    public let name: String?
    /// False when this node or any ancestor is hidden.
    public let isVisible: Bool
    /// The node's own layers, which children do not inherit.
    public let layers: SceneLayers
    /// The node's encode hooks, which the scene encode calls around each of its draws.
    let onBeforeEncode: ((SceneRenderObjectContext) -> Void)?
    let onAfterEncode: ((SceneRenderObjectContext) -> Void)?
}

/// A forest of scene nodes and the camera used to view it.
public struct SceneGraph {
    public static let maximumPunctualLightCount = 16

    public var roots: [SceneNode]
    public var camera: PerspectiveCamera
    /// When set, raster encode, picking, post, outlines and shadow fitting use this
    /// orthographic camera instead of `camera`. The perspective camera stays available
    /// for field-of-view animation and archives that predate the orthographic view.
    public var orthographicCamera: OrthographicCamera?
    public var cameraTransform: SceneTransform
    /// An alternate material for this graph's raster color and surface-attribute
    /// draws. Meshes, LODs and batches keep their geometry and authored shadows.
    public var overrideMaterial: SceneMaterialOverride? = nil
    /// Alternate materials for point, line and sprite color draws.
    public var primitiveMaterialOverride: ScenePrimitiveMaterialOverride? = nil
    /// Render textures for surface graphs that read scene color or depth.
    /// Use a stored render or capture opaque draws before transparent materials.
    /// Set these per pass. Scene archives do not store the render snapshot.
    public var viewportSnapshot: SceneViewportSnapshot? = nil
    /// Use CPU Double model-view composition for meshes, points, lines and
    /// sprites. Local offsets survive a large shared world origin. Explicit
    /// views need a split `ShaderGraphCamera` and the camera's world position.
    /// Raster culling and LOD use the same camera-relative transforms, including
    /// native batch object bounds. Unskinned meshes and instance groups also use
    /// rebased coordinates for directional, spot, point, and rectangle proxy
    /// shadows. Batch shadows and authored world-space shader inputs use Float.
    public var highPrecision: Bool = false
    /// Shared raster image-based light for PBR materials without an authored environment.
    /// The source and render command buffer must use the same Metal device.
    public var environment: EnvironmentLighting?
    /// Finite, nonnegative intensity of the scene environment. With no scene source,
    /// this multiplies each authored PBR environment's intensity.
    public var environmentIntensity: Float = 1
    /// World +Y rotation in radians, added to `environmentEulerRotation.y`.
    /// With no scene source, this adds to each authored PBR environment's yaw.
    public var environmentRotation: Float = 0
    /// Euler angles in radians for the shared environment. The authored rotation is
    /// `Rx * Ry * Rz`, matching Three.js's default XYZ order. Lighting samples its inverse.
    public var environmentEulerRotation: SIMD3<Float> = .zero

    var effectiveEnvironmentEulerRotation: SIMD3<Float> {
        environmentEulerRotation + SIMD3(0, environmentRotation, 0)
    }
    /// Additive distant diffuse light in world space for PBR and lit classic materials.
    /// Probe intensity is independent of the scene's cubemap lighting and yaw.
    public var lightProbes: [LightProbe] = []
    /// Local raster diffuse lighting. The prepared volumes must share the render device.
    /// The highest-priority volume on a camera-visible layer contributes per fragment.
    public var lightProbeGridLighting: LightProbeGridLighting?
    /// A sub-pixel offset in pixels (x right, y down) that encoding applies to the
    /// view's clip matrix, for temporal anti-aliasing; see `TemporalAntiAliasing.nextJitter`.
    /// Zero by default. Each component must be finite and within -1...1. It moves the
    /// image without changing depth, shading or level-of-detail selection.
    public var projectionJitter: SIMD2<Float> = .zero
    /// The layers the camera sees, like Three.js `Camera.layers`. Encoding, picking,
    /// lighting and path-tracing snapshots use nodes whose layers intersect this set.
    public var cameraLayers: SceneLayers = .default
    /// An observer of this graph's scene encodes, like Three.js r187's `Renderer.inspector`.
    /// Nil by default, which costs one nil check per encode. A strong reference. An encode that
    /// another feature runs on a graph, such as a planar reflection or cube capture, reports to
    /// the inspector of the graph it encodes, as a separate encode. See ``SceneEncodeInspector``.
    public var encodeInspector: (any SceneEncodeInspector)? = nil
    /// GPU-written draw arguments for point, line and sprite nodes, keyed by node ID.
    /// An empty dictionary leaves those draws CPU-counted. See ``PrimitiveIndirectDraws``.
    public var primitiveIndirectDraws: [UUID: PrimitiveIndirectDraws] = [:]

    public init(
        roots: [SceneNode],
        camera: PerspectiveCamera,
        cameraTransform: SceneTransform = SceneTransform()
    ) {
        self.roots = roots
        self.camera = camera
        self.cameraTransform = cameraTransform
    }

    /// The clip matrix of the orthographic camera when one is set, otherwise the perspective camera.
    public func activeProjection(aspectRatio: Float, depthMode: MeshDepthMode) throws -> simd_float4x4 {
        if let orthographicCamera {
            return try orthographicCamera.projection(aspectRatio: aspectRatio, depthMode: depthMode)
        }
        return try camera.projection(aspectRatio: aspectRatio, depthMode: depthMode)
    }

    /// Clip distances of the camera that ``activeProjection(aspectRatio:depthMode:)`` uses.
    public var activeClipDistances: (near: Float, far: Float) {
        if let orthographicCamera { return (orthographicCamera.near, orthographicCamera.far) }
        return (camera.near, camera.far)
    }

    /// The view used by scene passes. Parallel when ``orthographicCamera`` is set.
    public func shaderGraphCamera(aspectRatio: Float, depthMode: MeshDepthMode) throws -> ShaderGraphCamera {
        let clips = activeClipDistances
        return try ShaderGraphCamera(
            viewMatrix: simd_inverse(cameraTransform.matrix),
            projectionMatrix: activeProjection(aspectRatio: aspectRatio, depthMode: depthMode),
            near: clips.near, far: clips.far
        )
    }

    /// A ray through the camera that the default encode uses. Orthographic rays stay parallel.
    public func cameraRay(
        at normalized: SIMD2<Float>, aspectRatio: Float, depthMode: MeshDepthMode = .forward
    ) throws -> SceneRay {
        if orthographicCamera != nil {
            return try shaderGraphCamera(aspectRatio: aspectRatio, depthMode: depthMode)
                .ray(at: normalized, depthMode: depthMode)
        }
        return try camera.ray(at: normalized, aspectRatio: aspectRatio, cameraTransform: cameraTransform)
    }

    /// Direction from the scene back toward the camera when the default view is orthographic.
    /// Shading treats a w = 0 camera vector as this direction, and point sizing uses it as the view axis.
    var orthographicDirectionToCamera: SIMD3<Float>? {
        guard orthographicCamera != nil else { return nil }
        let column = cameraTransform.matrix.columns.2
        let towardCamera = SIMD3(column.x, column.y, column.z)
        let length = simd_length(towardCamera)
        guard length.isFinite, length > 0 else { return nil }
        return towardCamera / length
    }

    /// Whether a resolved node is drawn, lit, shadowed and picked for this graph's camera:
    /// it and its ancestors are visible and its layers intersect `cameraLayers`.
    public func renders(_ node: ResolvedSceneNode) -> Bool {
        node.isVisible && node.layers.intersects(cameraLayers)
    }

    /// Resolves parents before their children and preserves the supplied sibling order.
    /// Hidden nodes are included with `isVisible` false; use `renders(_:)` to filter.
    /// Level-of-detail nodes select their level for the graph camera's translation.
    public func resolve() throws -> [ResolvedSceneNode] {
        try resolve(cameraPosition: cameraTransform.translation)
    }

    /// Resolves the graph, selecting level-of-detail meshes for a camera at this
    /// world position. The position must be finite when the graph has such nodes.
    public func resolve(cameraPosition: SIMD3<Float>) throws -> [ResolvedSceneNode] {
        try resolve(cameraPosition: cameraPosition, excluding: [])
    }

    func resolve(
        cameraPosition: SIMD3<Float>, excluding: Set<UUID>, relativeWorldTransforms: [UUID: simd_double4x4]? = nil
    ) throws -> [ResolvedSceneNode] {
        var pending = roots.reversed().map { (node: $0, parent: simd_float4x4?.none, groupOrder: 0, visible: true) }
        var seen = Set<UUID>()
        var resolved: [ResolvedSceneNode] = []
        var lightCount = 0

        while let (node, parent, groupOrder, parentVisible) = pending.popLast() {
            guard !excluding.contains(node.id) else { continue }
            guard seen.insert(node.id).inserted else { throw ValidationError.duplicateID(node.id) }
            let rotationLength = simd_length_squared(node.transform.rotation.vector)
            let local = node.localTransformMatrix
            guard abs(rotationLength - 1) <= 0.0001, Self.isInvertible(local) else {
                throw ValidationError.invalidLocalTransform(node.id)
            }

            let world = node.resolvedWorldTransform(relativeTo: parent, local: local)
            guard Self.isInvertible(world) else { throw ValidationError.invalidWorldTransform(node.id) }
            let light = node.light.map { ResolvedPunctualLight(light: $0, world: world) }
            let isVisible = parentVisible && node.isVisible
            // Only lights that can shine count toward the limit; hidden ones stay in the graph.
            if light != nil, isVisible, node.layers.intersects(cameraLayers) {
                lightCount += 1
                guard lightCount <= Self.maximumPunctualLightCount else {
                    throw ValidationError.tooManyLights(maximum: Self.maximumPunctualLightCount)
                }
            }
            if node.batch != nil,
               node.mesh != nil || node.levelsOfDetail != nil || node.instances != nil
                || node.jointPalette != nil || node.morphWeights != nil {
                throw ValidationError.batchPayload(node.id)
            }
            var mesh = node.mesh
            var level: Int?
            if let levels = node.levelsOfDetail {
                guard node.mesh == nil else { throw ValidationError.meshAndLevelsOfDetail(node.id) }
                let distance: Double
                if let origin = relativeWorldTransforms?[node.id]?.columns.3 {
                    distance = simd_length(SIMD3(origin.x, origin.y, origin.z))
                } else {
                    let position = SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
                    distance = simd_distance(SIMD3<Double>(cameraPosition), SIMD3<Double>(position))
                }
                guard distance.isFinite else { throw ValidationError.invalidCameraPosition }
                let selected = levels.level(forDistance: Float(distance))
                mesh = levels.levels[selected].mesh
                level = selected
            }
            resolved.append(ResolvedSceneNode(
                id: node.id,
                worldTransform: world,
                mesh: mesh,
                depthMesh: node.depthMesh,
                batch: node.batch,
                materialParameters: node.materialParameters,
                castsShadow: node.castsShadow,
                receivesShadow: node.receivesShadow,
                jointPalette: node.jointPalette,
                light: light,
                ambientLight: node.ambientLight,
                instances: node.instances,
                morphWeights: node.morphWeights,
                frustumCulled: node.frustumCulled,
                cullingBounds: node.cullingBounds,
                levelOfDetail: level,
                renderOrder: node.renderOrder,
                groupOrder: groupOrder,
                primitive: node.primitive,
                name: node.name,
                isVisible: isVisible,
                layers: node.layers,
                onBeforeEncode: node.onBeforeEncode,
                onAfterEncode: node.onAfterEncode
            ))

            let isGroup = node.mesh == nil && node.batch == nil && node.levelsOfDetail == nil && node.primitive == nil && node.renderOrder != 0
            let childGroupOrder = isGroup ? node.renderOrder : groupOrder
            for child in node.children.reversed() {
                pending.append((node: child, parent: world, groupOrder: childGroupOrder, visible: isVisible))
            }
        }

        return resolved
    }

    private static func isInvertible(_ matrix: simd_float4x4) -> Bool {
        guard isFinite(matrix) else { return false }
        let determinant = simd_determinant(matrix)
        return determinant.isFinite && determinant != 0 && isFinite(simd_inverse(matrix))
    }

    private static func isFinite(_ matrix: simd_float4x4) -> Bool {
        func isFinite(_ column: SIMD4<Float>) -> Bool {
            column.x.isFinite && column.y.isFinite && column.z.isFinite && column.w.isFinite
        }

        let (x, y, z, w) = matrix.columns
        return isFinite(x) && isFinite(y) && isFinite(z) && isFinite(w)
    }

    public enum ValidationError: LocalizedError, Equatable {
        case duplicateID(UUID)
        case invalidLocalTransform(UUID)
        case invalidWorldTransform(UUID)
        case tooManyLights(maximum: Int)
        case meshAndLevelsOfDetail(UUID)
        case batchPayload(UUID)
        case invalidCameraPosition

        public var errorDescription: String? {
            switch self {
            case .duplicateID(let id):
                "Scene node \(id) appears more than once."
            case .invalidLocalTransform(let id):
                "Scene node \(id) needs a finite, invertible local transform with a unit rotation."
            case .invalidWorldTransform(let id):
                "The composed world transform for scene node \(id) must be finite and invertible."
            case .tooManyLights(let maximum):
                "A scene graph supports up to \(maximum) visible punctual lights in the camera layers."
            case .meshAndLevelsOfDetail(let id):
                "Scene node \(id) needs either a mesh or levels of detail, not both."
            case .batchPayload(let id):
                "Scene node \(id) combines a batched mesh with another mesh payload or deformation."
            case .invalidCameraPosition:
                "Selecting levels of detail needs a finite camera position."
            }
        }
    }
}
