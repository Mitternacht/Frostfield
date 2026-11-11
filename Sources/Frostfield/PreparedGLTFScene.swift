import Foundation
import Metal
import simd

/// Reuses prepared Metal meshes while creating graph snapshots of an imported glTF scene.
public struct PreparedGLTFScene {
    /// Stable within this prepared scene, including transform-only source nodes.
    public let nodeIDs: [Int: UUID]
    /// Maps both original graph nodes and generated primitive draw nodes back to glTF.
    public let sourceNodeIndicesByID: [UUID: Int]
    /// The imported morph weights of each node whose mesh has morph targets, keyed
    /// by the node's ID in `nodeIDs`: the node's `weights`, else its mesh's. Pass
    /// them as the rest weights of `AnimationMixer.morphWeights(for:rest:)`.
    public let restMorphWeights: [UUID: [Float]]
    /// Live factors of every imported material that a `KHR_animation_pointer`
    /// channel animates and a primitive of the scene draws, keyed by glTF material
    /// index. Renderers prepared with the material share them; `AnimationClip(_:in:)`
    /// binds pointer channels to them.
    public let materialFactors: [Int: PhysicallyBasedMaterial.LiveFactors]
    /// The graph nodes carrying each imported light definition, keyed by glTF light index.
    public let lightNodeIDs: [Int: [UUID]]

    private let roots: [SceneNode]
    private let animations: [Int: GLTFAnimationClip]
    private let cameraNodes: [Int: GLTFScene.Camera]
    private let depthMode: MeshDepthMode
    private let skinnedInstances: [SkinInstance]
    /// The draw nodes of every primitive of a morphed source node, keyed by its ID.
    private let morphDrawIDs: [UUID: [UUID]]
    /// Every imported material's source index, drawn or not.
    let importedMaterialIndices: Set<Int>

    private struct MeshKey: Hashable {
        let sourceIndex: Int
        let skinIndex: Int?
    }

    private struct SkinInstance {
        let drawIDs: [UUID]
        let jointIDs: [UUID]
        let inverseBindMatrices: [simd_float4x4]
    }

    /// Prepares one renderer per unique source mesh, primitive, and skin.
    /// Primitives without loaded material metadata use the supplied default
    /// or the glTF default material. Authored lights replace the fallback direct light.
    /// Meshes imported with morph targets blend them on the GPU; their draw nodes
    /// start with the node's imported weights.
    ///
    /// A node with `EXT_mesh_gpu_instancing` transforms gets one child draw node per
    /// instance that shares the node's renderers. Each instance is a separate draw,
    /// not a GPU-instanced one, and its draw nodes map back to the source node.
    public init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, scene: GLTFScene,
        depthMode: MeshDepthMode = .forward,
        directionToLight: SIMD3<Float> = SIMD3(0, 1, 1), lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12, receivesShadows: Bool = false,
        defaultMaterial: MeshMaterial? = nil, environment: PhysicallyBasedMaterial.Environment? = nil
    ) throws {
        try self.init(
            device: device, colorPixelFormat: colorPixelFormat, scene: scene, depthMode: depthMode,
            directionToLight: directionToLight, lightIntensity: lightIntensity, ambientIntensity: ambientIntensity,
            receivesShadows: receivesShadows, defaultMaterial: defaultMaterial, environment: environment,
            basisTextureStorage: .rgba8
        )
    }

    /// Prepares the scene with the selected Basis storage policy for imported materials.
    public init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, scene: GLTFScene,
        depthMode: MeshDepthMode = .forward,
        directionToLight: SIMD3<Float> = SIMD3(0, 1, 1), lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12, receivesShadows: Bool = false,
        defaultMaterial: MeshMaterial? = nil, environment: PhysicallyBasedMaterial.Environment? = nil,
        basisTextureStorage: GLTFScene.Material.BasisTextureStorage
    ) throws {
        let ids = Dictionary(uniqueKeysWithValues: scene.nodes.map { ($0.sourceIndex, UUID()) })
        var sourceIndices = Dictionary(uniqueKeysWithValues: ids.map { ($0.value, $0.key) })
        let sourceMeshes = Dictionary(uniqueKeysWithValues: scene.meshes.map { ($0.sourceIndex, $0) })
        let sourceMaterials = Dictionary(uniqueKeysWithValues: scene.materials.map { ($0.sourceIndex, $0) })
        let sourceSkins = Dictionary(uniqueKeysWithValues: scene.skins.map { ($0.sourceIndex, $0) })
        let sourceLights = Dictionary(uniqueKeysWithValues: scene.lights.map { ($0.sourceIndex, $0.light) })
        let sourceCameras = Dictionary(uniqueKeysWithValues: scene.cameras.map { ($0.sourceIndex, $0) })
        let fallbackIntensity = sourceLights.isEmpty ? lightIntensity : 0
        let fallbackMaterial = try defaultMaterial ?? .physicallyBased(PhysicallyBasedMaterial(
            metallic: 1, roughness: 1, directionToLight: directionToLight,
            lightIntensity: fallbackIntensity, ambientIntensity: ambientIntensity, receivesShadows: receivesShadows,
            alphaMode: .opaque, environment: environment
        ))

        var materials: [Int: PhysicallyBasedMaterial] = [:]
        let animatedMaterials = Set(scene.animations.flatMap { clip in
            clip.pointerChannels.compactMap { channel -> Int? in
                if case .material(let index, _) = channel.target { index } else { nil }
            }
        })
        var lightNodes: [Int: [UUID]] = [:]
        var meshes: [MeshKey: [MeshRenderer]] = [:]
        var sourceNodes: [Int: SceneNode] = [:]
        var skinnedRoots: [SceneNode] = []
        var instances: [SkinInstance] = []
        var restWeights: [UUID: [Float]] = [:]
        var morphDraws: [UUID: [UUID]] = [:]
        var remaining = 1_000_000 - scene.nodes.count
        var remainingVertices = 1_000_000
        var remainingIndices = 1_000_000

        for node in scene.nodes {
            guard let id = ids[node.sourceIndex] else { throw Failure.missingSourceNode(node.sourceIndex) }
            let light = node.lightIndex.flatMap { sourceLights[$0] }
            if let index = node.lightIndex, light != nil { lightNodes[index, default: []].append(id) }
            var graphNode = SceneNode(id: id, transform: node.localTransform, light: light, name: node.name)
            defer { sourceNodes[node.sourceIndex] = graphNode }
            guard let meshIndex = node.meshIndex else { continue }
            guard let mesh = sourceMeshes[meshIndex] else { throw GLTFMeshLoader.Failure.missingMesh(meshIndex) }
            let skin = node.skinIndex.flatMap { sourceSkins[$0] }
            if let index = node.skinIndex, skin == nil { throw GLTFMeshLoader.Failure.missingSkin(index) }
            let draws = mesh.primitives.count
            let joints = skin?.jointNodeIndices.count ?? 0
            let (drawNodes, overflow) = draws.multipliedReportingOverflow(by: node.instances?.count ?? 1)
            guard !overflow, drawNodes <= remaining, joints <= remaining - drawNodes else { throw Failure.sceneSizeLimit }
            remaining -= drawNodes + joints

            let key = MeshKey(sourceIndex: meshIndex, skinIndex: node.skinIndex)
            let renderers: [MeshRenderer]
            if let prepared = meshes[key] {
                renderers = prepared
            } else {
                renderers = try mesh.primitives.enumerated().map { index, geometry in
                    guard geometry.vertices.count <= remainingVertices, geometry.indices.count <= remainingIndices else {
                        throw Failure.sceneSizeLimit
                    }
                    remainingVertices -= geometry.vertices.count
                    remainingIndices -= geometry.indices.count
                    let material: MeshMaterial
                    if let sourceIndex = mesh.primitiveMaterialIndices[index], let source = sourceMaterials[sourceIndex] {
                        if let prepared = materials[sourceIndex] {
                            material = .physicallyBased(prepared)
                        } else {
                            var prepared = try source.prepare(
                                device: device, directionToLight: directionToLight, lightIntensity: fallbackIntensity,
                                ambientIntensity: ambientIntensity, receivesShadows: receivesShadows,
                                environment: environment, basisTextureStorage: basisTextureStorage
                            )
                            if animatedMaterials.contains(sourceIndex) { prepared = prepared.withLiveFactors() }
                            materials[sourceIndex] = prepared
                            material = .physicallyBased(prepared)
                        }
                    } else {
                        material = fallbackMaterial
                    }
                    let meshSkin = try skin.map { source in
                        guard let influences = mesh.primitiveSkinInfluences[index] else {
                            throw GLTFMeshLoader.Failure.invalidSkinPrimitive("prepared mesh is missing skin influences")
                        }
                        return try MeshSkin(jointCount: source.jointNodeIndices.count, influences: influences)
                    }
                    return try MeshRenderer(
                        device: device, colorPixelFormat: colorPixelFormat, geometry: geometry,
                        material: material, depthMode: depthMode, skin: meshSkin,
                        morphTargets: mesh.morphTargets?.primitives[index]
                    )
                }
                meshes[key] = renderers
            }

            let rest = mesh.morphTargets == nil ? nil : node.morphWeights
            func primitiveNode(_ renderer: MeshRenderer) -> SceneNode {
                let draw = SceneNode(mesh: renderer, morphWeights: rest)
                sourceIndices[draw.id] = node.sourceIndex
                return draw
            }

            if let skin {
                // Skinning uses joint transforms. Keeping draw nodes at world identity
                // also prevents the source mesh transform from changing face culling.
                let drawNodes = renderers.map(primitiveNode)
                skinnedRoots.append(contentsOf: drawNodes)
                if let rest {
                    restWeights[id] = rest
                    morphDraws[id] = drawNodes.map(\.id)
                }
                let jointIDs = try skin.jointNodeIndices.map { index in
                    guard let id = ids[index] else { throw Failure.missingSourceNode(index) }
                    return id
                }
                instances.append(SkinInstance(
                    drawIDs: drawNodes.map(\.id), jointIDs: jointIDs, inverseBindMatrices: skin.inverseBindMatrices
                ))
            } else if let instances = node.instances, let first = renderers.first {
                graphNode.children = instances.map { transform in
                    var instance = primitiveNode(first)
                    instance.transform = transform
                    instance.children = renderers.dropFirst().map(primitiveNode)
                    return instance
                }
            } else {
                graphNode.mesh = renderers.first
                graphNode.morphWeights = rest
                graphNode.children = renderers.dropFirst().map(primitiveNode)
                if let rest {
                    restWeights[id] = rest
                    morphDraws[id] = [id] + graphNode.children.map(\.id)
                }
            }
        }

        self.nodeIDs = ids
        sourceNodeIndicesByID = sourceIndices
        roots = try Self.hierarchy(scene: scene, preparedNodes: sourceNodes) + skinnedRoots
        animations = Dictionary(uniqueKeysWithValues: scene.animations.map { ($0.sourceIndex, $0) })
        cameraNodes = Dictionary(uniqueKeysWithValues: scene.nodes.compactMap { node in
            node.cameraIndex.flatMap { sourceCameras[$0] }.map { (node.sourceIndex, $0) }
        })
        self.depthMode = depthMode
        skinnedInstances = instances
        restMorphWeights = restWeights
        morphDrawIDs = morphDraws
        materialFactors = materials.compactMapValues(\.liveFactors)
        importedMaterialIndices = Set(scene.materials.map(\.sourceIndex))
        lightNodeIDs = lightNodes
    }

    /// Copies the prepared hierarchy and remaps its joints, draw nodes and animation
    /// targets to fresh scene IDs. The copy shares prepared meshes, textures and
    /// live material factors. Pose the original and clone separately, or place
    /// both graphs' roots in one scene and call each prepared scene's `posing`.
    public func cloned() -> PreparedGLTFScene {
        PreparedGLTFScene(cloning: self)
    }

    private init(cloning source: PreparedGLTFScene) {
        var identities: [UUID: UUID] = [:]
        func cloneID(_ original: UUID) -> UUID {
            if let existing = identities[original] { return existing }
            let id = UUID()
            identities[original] = id
            return id
        }
        roots = SceneNode.updatingForest(source.roots) { node in
            node.assignCloneID(cloneID(node.id))
        }
        nodeIDs = source.nodeIDs.mapValues(cloneID)
        sourceNodeIndicesByID = Dictionary(uniqueKeysWithValues: source.sourceNodeIndicesByID.map {
            (cloneID($0.key), $0.value)
        })
        restMorphWeights = Dictionary(uniqueKeysWithValues: source.restMorphWeights.map {
            (cloneID($0.key), $0.value)
        })
        morphDrawIDs = Dictionary(uniqueKeysWithValues: source.morphDrawIDs.map {
            (cloneID($0.key), $0.value.map(cloneID))
        })
        skinnedInstances = source.skinnedInstances.map { instance in
            SkinInstance(
                drawIDs: instance.drawIDs.map(cloneID), jointIDs: instance.jointIDs.map(cloneID),
                inverseBindMatrices: instance.inverseBindMatrices
            )
        }
        lightNodeIDs = source.lightNodeIDs.mapValues { $0.map(cloneID) }
        animations = source.animations
        cameraNodes = source.cameraNodes
        depthMode = source.depthMode
        materialFactors = source.materialFactors
        importedMaterialIndices = source.importedMaterialIndices
    }

    /// The imported clip with this source index, bound to this scene with
    /// `AnimationClip(_:in:)`, including its `KHR_animation_pointer` channels.
    public func animationClip(_ animationIndex: Int) throws -> AnimationClip {
        guard let animation = animations[animationIndex] else { throw Failure.missingAnimation(animationIndex) }
        return try AnimationClip(animation, in: self)
    }

    /// Returns the authored pose when animationIndex is nil. Otherwise the index
    /// refers to a clip's source glTF index. Time is measured in seconds. A clip's
    /// weights channels set the morph weights of their nodes' primitive draws.
    /// `KHR_animation_pointer` channels for materials, lights and cameras are not
    /// applied here; play `animationClip(_:)` with an `AnimationMixer` for them.
    public func graph(
        camera: PerspectiveCamera, cameraTransform: SceneTransform = .init(),
        animationIndex: Int? = nil, time: Float = 0, looping: Bool = false
    ) throws -> SceneGraph {
        var graph = SceneGraph(roots: roots, camera: camera, cameraTransform: cameraTransform)
        if let animationIndex {
            guard let animation = animations[animationIndex] else { throw Failure.missingAnimation(animationIndex) }
            graph = try animation.applying(to: graph, nodeIDs: nodeIDs, at: time, looping: looping)
            if !morphDrawIDs.isEmpty {
                let poses = try animation.sample(at: time, nodeIDs: nodeIDs, looping: looping)
                let weights = poses.compactMapValues(\.morphWeights).filter { morphDrawIDs[$0.key] != nil }
                if !weights.isEmpty { graph = try applyingMorphWeights(weights, to: graph) }
            }
            if skinnedInstances.isEmpty { return graph }
        }
        return try posing(graph)
    }

    /// Recomputes every skinned draw's joint palette from the node transforms in a
    /// graph built from this scene, such as the result of `AnimationMixer.apply(to:)`.
    /// Wrapping the roots in a common parent keeps the palettes valid.
    public func posing(_ graph: SceneGraph) throws -> SceneGraph {
        if skinnedInstances.isEmpty { return graph }
        let resolved = try graph.resolve()
        let transforms = Dictionary(uniqueKeysWithValues: resolved.map { ($0.id, $0.worldTransform) })
        var palettes: [UUID: JointPalette] = [:]
        for instance in skinnedInstances {
            guard let first = instance.drawIDs.first, let meshWorld = transforms[first] else {
                throw Failure.missingSkinDraw
            }
            let jointWorld = try instance.jointIDs.map { id in
                guard let world = transforms[id] else { throw Failure.missingSkinDraw }
                return world
            }
            let palette = try JointPalette(
                meshWorld: meshWorld, jointWorld: jointWorld, inverseBindMatrices: instance.inverseBindMatrices
            )
            for id in instance.drawIDs { palettes[id] = palette }
        }

        return graph.updatingNodes { node in
            if let palette = palettes[node.id] { node.jointPalette = palette }
        }
    }

    /// Writes morph weights for imported nodes, keyed by their IDs in `nodeIDs`, to
    /// the draw nodes of every primitive of their meshes. Each array needs one finite
    /// weight per morph target. To drive weights with a mixer, pass
    /// `mixer.morphWeights(for: id, rest: restMorphWeights[id])` for each key of
    /// `restMorphWeights`. Throws when a key has no morph targets or a draw node is
    /// missing from the graph.
    public func applyingMorphWeights(_ weights: [UUID: [Float]], to graph: SceneGraph) throws -> SceneGraph {
        var drawWeights: [UUID: [Float]] = [:]
        for (id, values) in weights {
            guard let draws = morphDrawIDs[id], let rest = restMorphWeights[id] else { throw Failure.missingMorphNode(id) }
            guard values.count == rest.count else {
                throw MeshMorphTargets.ValidationError.weightCount(expected: rest.count, actual: values.count)
            }
            guard values.allSatisfy(\.isFinite) else { throw MeshMorphTargets.ValidationError.nonfiniteWeight }
            for draw in draws { drawWeights[draw] = values }
        }
        var pending = drawWeights
        let result = graph.updatingNodes { node in
            if let values = pending.removeValue(forKey: node.id) { node.morphWeights = values }
        }
        if let missing = pending.keys.first {
            throw Failure.missingMorphNode(sourceNodeIndicesByID[missing].flatMap { nodeIDs[$0] } ?? missing)
        }
        return result
    }

    /// Resolves a camera's projection and current pose in a graph from this scene.
    /// Supply the animated graph to use an animated camera or its animated parents.
    /// The source index identifies the node carrying the camera, so shared camera
    /// definitions can be used at several distinct poses.
    public func cameraView(
        nodeIndex: Int, in graph: SceneGraph, viewportAspectRatio: Float
    ) throws -> GLTFScene.Camera.View {
        guard let camera = cameraNodes[nodeIndex] else { throw Failure.missingCameraNode(nodeIndex) }
        return try view(of: camera, nodeIndex: nodeIndex, in: graph, viewportAspectRatio: viewportAspectRatio)
    }

    /// Like `cameraView(nodeIndex:in:viewportAspectRatio:)`, with the camera's
    /// projection parameters blended by a mixer's `.gltfCamera` tracks, such as
    /// `KHR_animation_pointer` channels bound by `AnimationClip(_:in:)`. Throws
    /// `AnimationMixer.Failure.unanimatableProperty` for an animated parameter the
    /// camera lacks, such as an omitted aspect ratio or far plane, and
    /// `invalidPropertyValue` for a blended projection that is not valid.
    public func cameraView(
        nodeIndex: Int, in graph: SceneGraph, viewportAspectRatio: Float, animatedBy mixer: AnimationMixer
    ) throws -> GLTFScene.Camera.View {
        guard let camera = cameraNodes[nodeIndex] else { throw Failure.missingCameraNode(nodeIndex) }
        let index = camera.sourceIndex
        func blended(_ property: AnimationProperty.GLTFCamera, _ rest: Float?) throws -> Float? {
            let key = AnimationProperty.gltfCamera(index, property)
            guard let rest else {
                let animated = try mixer.value(of: key, rest: .number(.nan))
                guard animated.floatValue?.isNaN != false else {
                    throw AnimationMixer.Failure.unanimatableProperty(key.description, "the camera does not define it")
                }
                return nil
            }
            return try mixer.value(of: key, rest: .number(rest)).floatValue
        }
        let projection: GLTFScene.Camera.Projection
        switch camera.projection {
        case .perspective(let fieldOfView, let aspectRatio, let near, let far):
            let values = (
                try blended(.verticalFieldOfView, fieldOfView)!, try blended(.aspectRatio, aspectRatio),
                try blended(.near, near)!, try blended(.far, far)
            )
            guard values.0.isFinite, values.0 > 0, values.0 < .pi, values.2.isFinite, values.2 > 0,
                  values.1.map({ $0.isFinite && $0 > 0 }) ?? true, values.3.map({ $0.isFinite && $0 > values.2 }) ?? true else {
                throw AnimationMixer.Failure.invalidPropertyValue("glTF camera \(index) projection")
            }
            projection = .perspective(verticalFieldOfView: values.0, aspectRatio: values.1, near: values.2, far: values.3)
        case .orthographic(let halfWidth, let halfHeight, let near, let far):
            let values = (
                try blended(.halfWidth, halfWidth)!, try blended(.halfHeight, halfHeight)!,
                try blended(.near, near)!, try blended(.far, far)!
            )
            guard [values.0, values.1, values.2, values.3].allSatisfy(\.isFinite), values.0 != 0, values.1 != 0,
                  values.2 >= 0, values.3 > values.2 else {
                throw AnimationMixer.Failure.invalidPropertyValue("glTF camera \(index) projection")
            }
            projection = .orthographic(halfWidth: values.0, halfHeight: values.1, near: values.2, far: values.3)
        }
        let animated = GLTFScene.Camera(sourceIndex: camera.sourceIndex, name: camera.name, projection: projection)
        return try view(of: animated, nodeIndex: nodeIndex, in: graph, viewportAspectRatio: viewportAspectRatio)
    }

    private func view(
        of camera: GLTFScene.Camera, nodeIndex: Int, in graph: SceneGraph, viewportAspectRatio: Float
    ) throws -> GLTFScene.Camera.View {
        guard let id = nodeIDs[nodeIndex] else { throw Failure.missingCameraNode(nodeIndex) }
        let identity = simd_quatf(real: 1, imag: .zero)
        var pending = graph.roots.reversed().map { (node: $0, world: simd_float4x4?.none, rotation: identity) }
        while let (node, parentWorld, parentRotation) = pending.popLast() {
            let world = node.resolvedWorldTransform(relativeTo: parentWorld)
            let rotation = node.orientationMode == .world && parentWorld != nil
                ? node.worldOrientedRotation : parentRotation * node.transform.rotation
            if node.id == id {
                let position = SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
                guard [position.x, position.y, position.z].allSatisfy(\.isFinite),
                      abs(simd_length_squared(rotation.vector) - 1) <= 0.0001 else {
                    throw SceneGraph.ValidationError.invalidWorldTransform(id)
                }
                let pose = SceneTransform(translation: position, rotation: simd_normalize(rotation))
                let projection = try camera.metalProjection(viewportAspectRatio: viewportAspectRatio, depthMode: depthMode)
                let aspectRatio = projection.columns.1.y / projection.columns.0.x
                guard aspectRatio.isFinite, aspectRatio > 0 else {
                    throw PerspectiveCamera.ValidationError.unrepresentableProjection
                }
                let direction: SIMD3<Float>?
                switch camera.projection {
                case .perspective: direction = nil
                case .orthographic: direction = pose.rotation.act(SIMD3(0, 0, 1))
                }
                let view = simd_inverse(pose.matrix)
                let clipRange: (near: Float, far: Float)
                switch camera.projection {
                case .perspective(_, _, let near, let far): clipRange = (near, far ?? .infinity)
                case .orthographic(_, _, let near, let far): clipRange = (near, far)
                }
                return GLTFScene.Camera.View(
                    clipFromWorld: projection * view, cameraPosition: position,
                    aspectRatio: aspectRatio, directionToCamera: direction, depthMode: depthMode,
                    graphCamera: try ShaderGraphCamera(
                        viewMatrix: view, projectionMatrix: projection, near: clipRange.near, far: clipRange.far
                    )
                )
            }
            for child in node.children.reversed() {
                pending.append((child, world, rotation))
            }
        }
        throw Failure.missingCameraNode(nodeIndex)
    }

    private static func hierarchy(scene: GLTFScene, preparedNodes: [Int: SceneNode]) throws -> [SceneNode] {
        let source = Dictionary(uniqueKeysWithValues: scene.nodes.map { ($0.sourceIndex, $0.childIndices) })
        var pending = scene.rootNodeIndices.reversed().map { (index: $0, depth: 1, visited: false) }
        var rebuilt: [SceneNode] = []
        while let (index, depth, visited) = pending.popLast() {
            guard depth <= 512 else { throw Failure.hierarchyTooDeep }
            guard let children = source[index], var node = preparedNodes[index] else {
                throw Failure.missingSourceNode(index)
            }
            if !visited, !children.isEmpty {
                pending.append((index, depth, true))
                for child in children.reversed() { pending.append((child, depth + 1, false)) }
                continue
            }
            node.children.append(contentsOf: rebuilt.suffix(children.count))
            rebuilt.removeLast(children.count)
            rebuilt.append(node)
        }
        return rebuilt
    }

    public enum Failure: LocalizedError, Equatable {
        case missingSourceNode(Int)
        case missingAnimation(Int)
        case missingCameraNode(Int)
        case missingSkinDraw
        case sceneSizeLimit
        case hierarchyTooDeep
        case missingMorphNode(UUID)

        public var errorDescription: String? {
            switch self {
            case .missingSourceNode(let index): "Could not find source glTF node \(index) in the selected scene."
            case .missingAnimation(let index): "The prepared glTF scene has no imported animation \(index)."
            case .missingCameraNode(let index): "The glTF graph has no imported camera on node \(index)."
            case .missingSkinDraw: "The prepared glTF graph is missing a skinned draw or joint transform."
            case .sceneSizeLimit: "The prepared glTF scene exceeds the geometry or instance limit."
            case .hierarchyTooDeep: "The prepared glTF hierarchy exceeds 512 levels."
            case .missingMorphNode(let id): "Node \(id) has no morph targets or its primitive draws are missing from the graph."
            }
        }
    }
}
