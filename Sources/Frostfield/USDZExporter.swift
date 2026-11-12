import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

/// Writes USDZ packages for AR Quick Look, RealityKit, and other USD readers, like
/// Three.js `USDZExporter`.
///
/// The package is an uncompressed zip whose first file is the USDA root layer
/// `scene.usda`. Each distinct triangle mesh is a separate layer under `geometries/`
/// that the scene references, so shared meshes and expanded instances store their
/// vertices once. Textures are PNG or JPEG files under `textures/`. Every file's data
/// starts on a 64-byte boundary, as the USDZ specification requires.
///
/// The stage declares `upAxis = "Y"` and `metersPerUnit` from the options, so Frostfield
/// and glTF content in meters keeps its size. Node transforms are written as
/// `xformOp:translate`, `xformOp:orient`, and `xformOp:scale`. Meshes carry `normals`,
/// the `st` and `st1` texture coordinates flipped into USD's bottom-left space, and
/// tangents as the `float4` primvar `tangents`, all with vertex interpolation and
/// `subdivisionScheme = "none"`.
///
/// Materials become UsdPreviewSurface networks with base color, opacity and
/// opacity threshold, metallic, roughness, normal, occlusion, emissive, clearcoat,
/// clearcoat roughness, and index of refraction, each from a factor or a UsdUVTexture.
/// Texture factors use UsdUVTexture `scale` and `bias`, texture transforms use
/// UsdTransform2d, and wrap modes use the texture's `wrapS` and `wrapT`. Features
/// UsdPreviewSurface cannot express are listed in `Output.warnings`.
///
/// Export runs synchronously. Reading private textures and evaluating GPU-deformed
/// meshes waits for the GPU, so call it off the main thread for large scenes.
public struct USDZExporter {
    public struct Options: Equatable, Sendable {
        /// The length of one scene unit in meters. Frostfield and glTF use meters.
        public var metersPerUnit: Double
        /// The width-to-height ratio of the exported camera apertures. USD cameras store an
        /// aperture rectangle, while Frostfield and glTF perspective cameras may leave it open.
        public var cameraAspectRatio: Float

        /// Both values must be finite and positive.
        public init(metersPerUnit: Double = 1, cameraAspectRatio: Float = 1) {
            self.metersPerUnit = metersPerUnit
            self.cameraAspectRatio = cameraAspectRatio
        }
    }

    /// Material features UsdPreviewSurface has no input for.
    public enum MaterialFeature: String, Equatable, Sendable {
        case transmission, volume, specular, sheen, anisotropy, iridescence, dispersion
        case clearcoatNormal, surfaceGraph
        /// UsdUVTexture can scale a normal map, but USD's compliance rules require the
        /// standard `(2, 2, 2, 1)` scale for 8-bit normal maps, so the map is written at full strength.
        case normalScale
    }

    /// A lossy conversion the exporter performed. Paths are USD prim paths in the root layer.
    public enum Warning: Equatable, Sendable, CustomStringConvertible {
        /// The material's named feature was dropped; the rest of the material was exported.
        case materialFeatureDropped(material: String, feature: MaterialFeature)
        /// Unlit shading was written as emission of the base color over a black diffuse response.
        case unlitApproximated(material: String)
        /// The Frostfield material-alpha mode has no USD equivalent and the alpha is below one.
        /// It was exported as opaque.
        case alphaExportedAsOpaque(material: String)
        /// Metal sampler states cannot be inspected, so the material's maps repeat.
        case samplerUnavailable(material: String)
        /// UsdUVTexture has no filter input, so nearest-neighbor filtering was dropped.
        case textureFilteringDropped(material: String)
        /// The renderer has no `.physicallyBased` material, so the mesh uses a default
        /// gray UsdPreviewSurface.
        case unsupportedMaterial(mesh: String)
        /// A surface graph's vertex offset was evaluated once and baked into the mesh.
        case deformationBaked(mesh: String)
        /// The mesh was written in its bind pose without skinning.
        case skinDropped(mesh: String)
        /// The mesh was written without morph targets.
        case morphTargetsDropped(mesh: String)
        /// Only some vertices carry the attribute, so the primvar was omitted.
        case attributeDropped(mesh: String, attribute: String)
        /// Instance tints have no UsdPreviewSurface equivalent and were dropped.
        case instanceTintsDropped(node: String)
        /// Only the finest of the node's levels of detail was exported.
        case levelsOfDetailReduced(node: String, levelCount: Int)
        /// Punctual and area lights were omitted. AR Quick Look and RealityKit light USDZ
        /// content with their own environment.
        case lightOmitted(node: String)
        /// The camera's infinite far plane was written as a finite clipping range.
        case infiniteFarPlaneClamped(camera: String, far: Float)
        /// A USD stage has one timeline, so clips other than the exported one were omitted.
        case animationsOmitted(count: Int)
        /// Morph target weight channels were omitted from the exported clip.
        case morphWeightAnimationOmitted
        /// A WebP or AVIF image was decoded and stored as PNG, which USDZ requires.
        case imageTranscoded(image: String)
        /// The texture's image is a KTX2 payload without a PNG or JPEG fallback, so the map was omitted.
        case textureOmitted(material: String)

        public var description: String {
            switch self {
            case .materialFeatureDropped(let material, let feature):
                "Material \(material) dropped \(feature.rawValue), which UsdPreviewSurface cannot express."
            case .unlitApproximated(let material): "Material \(material) writes its unlit base color as emission."
            case .alphaExportedAsOpaque(let material): "Material \(material) was exported as opaque from Frostfield's material-alpha mode."
            case .samplerUnavailable(let material): "Material \(material) uses repeat wrapping in place of a custom sampler."
            case .textureFilteringDropped(let material): "Material \(material) dropped nearest-neighbor texture filtering."
            case .unsupportedMaterial(let mesh): "Mesh \(mesh) has a non-PBR material and uses a default gray surface."
            case .deformationBaked(let mesh): "Mesh \(mesh) contains a surface graph's baked vertex offset."
            case .skinDropped(let mesh): "Mesh \(mesh) was exported in its bind pose without its skin."
            case .morphTargetsDropped(let mesh): "Mesh \(mesh) was exported without its morph targets."
            case .attributeDropped(let mesh, let attribute): "Mesh \(mesh) omits \(attribute), which only some vertices carry."
            case .instanceTintsDropped(let node): "Node \(node) was exported without its instance tints."
            case .levelsOfDetailReduced(let node, let count): "Node \(node) was exported with the finest of its \(count) levels of detail."
            case .lightOmitted(let node): "The light on \(node) was omitted."
            case .infiniteFarPlaneClamped(let camera, let far): "Camera \(camera) has an infinite far plane, written as \(far)."
            case .animationsOmitted(let count): "\(count) animation clips were omitted; a USD stage has one timeline."
            case .morphWeightAnimationOmitted: "Morph target weight channels were omitted."
            case .imageTranscoded(let image): "Image \(image) was transcoded to PNG."
            case .textureOmitted(let material): "Material \(material) omitted a KTX2 texture without a PNG or JPEG fallback."
            }
        }
    }

    /// An encoded package and its export report.
    public struct Output: Sendable {
        public let data: Data
        public let warnings: [Warning]
        /// The USDA text of `scene.usda`, the package's root layer.
        public let rootLayer: String
        /// The prim path of each scene graph node, keyed by node ID. Empty for `GLTFScene` exports.
        public let nodePaths: [UUID: String]
        /// The prim path of each source node index. Empty for scene graph exports.
        public let sourceNodePaths: [Int: String]
        /// The exported camera prim, when the scene graph's camera was included.
        public let cameraPath: String?
    }

    public enum Failure: LocalizedError, Equatable {
        /// Options, transforms, cameras, or materials hold non-finite or out-of-range values.
        case invalidData(String)
        /// A `GLTFScene` field refers to a node, mesh, material, or camera it does not contain.
        case missingReference(String)
        /// A mesh has more vertices than a USD `int` index can address.
        case meshTooLarge(String)
        /// ImageIO could not transcode an image to PNG.
        case imageEncoding(String)

        public var errorDescription: String? {
            switch self {
            case .invalidData(let reason): "Cannot export invalid USD data. \(reason)"
            case .missingReference(let reason): "The scene is missing a referenced object. \(reason)"
            case .meshTooLarge(let mesh): "Mesh \(mesh) has more vertices than USD int indices can address."
            case .imageEncoding(let image): "ImageIO could not encode image \(image) as PNG."
            }
        }
    }

    public var options: Options
    /// Copies private and managed textures and evaluates GPU-deformed meshes. It must use the
    /// device of those textures and renderers. Without a queue, the exporter creates one per device.
    public var commandQueue: (any MTLCommandQueue)?

    public init(options: Options = .init(), commandQueue: (any MTLCommandQueue)? = nil) {
        self.options = options
        self.commandQueue = commandQueue
    }

    /// Writes a scene graph's hierarchy under the default prim `/Root`.
    ///
    /// Each node becomes an `Xform` named from `nodeNames` (made USD-safe and unique among
    /// its siblings) or `Node_<index>` in depth-first order. A node's mesh is a child `Mesh`
    /// prim that references its geometry layer and binds its material. Instance sets expand
    /// into one child `Xform` per instance with its matrix as `xformOp:transform`. A node with
    /// levels of detail exports its finest level.
    ///
    /// `.physicallyBased` materials export as in `GLTFExporter`: without a surface graph,
    /// the node's material parameters tint the base color factor. Textures become PNG files,
    /// color maps sRGB-encoded and data maps linear, so one texture used both ways is stored twice.
    ///
    /// The labels follow `GLTFExporter.export(_:sceneName:nodeNames:animations:includesCamera:)`.
    ///
    /// - Parameters:
    ///   - animation: Transform tracks written as `xformOp` time samples at each keyframe, at
    ///     `timeCodesPerSecond = 60`. USD interpolates translation and scale linearly and
    ///     orientation spherically, as `SceneAnimation` does, and AR Quick Look plays the
    ///     timeline. Every track must target a node in the graph.
    ///   - includesCamera: Adds `/Root/Camera`, a USD camera with the graph's vertical field
    ///     of view, clipping range, and camera transform.
    public func export(
        _ graph: SceneGraph, nodeNames: [UUID: String] = [:], animation: SceneAnimation? = nil, includesCamera: Bool = true
    ) throws -> Output {
        var lowering = try USDZSceneLowering(exporter: self)
        return try lowering.export(graph, nodeNames: nodeNames, includesCamera: includesCamera, animation: animation)
    }

    /// Writes an imported scene's node hierarchy, names, mesh primitives, loaded materials,
    /// original PNG and JPEG images, sampler wrap modes, texture transforms, and cameras.
    /// KTX2 images use their PNG or JPEG fallback; WebP and AVIF images are transcoded to PNG.
    /// Skins, morph targets, and lights are reported and omitted.
    ///
    /// One clip becomes the stage's timeline: the clip with source index `animationIndex`, or
    /// the first clip when it is nil. Its node transforms are sampled at every key time, plus
    /// every 1/60 s between keys of STEP and CUBICSPLINE channels, and written as `xformOp`
    /// time samples. Weight channels and the other clips are reported and omitted.
    public func export(_ scene: GLTFScene, animationIndex: Int? = nil) throws -> Output {
        var lowering = try USDZSceneLowering(exporter: self)
        return try lowering.export(scene, animationIndex: animationIndex)
    }
}

/// Lowers scene content into a USDZ package.
struct USDZSceneLowering {
    private typealias Failure = USDZExporter.Failure
    private typealias Warning = USDZExporter.Warning
    private typealias Description = USDMaterialDescription

    private let options: USDZExporter.Options
    private var reader: GLTFTextureReader
    private var warnings: [Warning] = []
    private var writer = USDAWriter()
    private var images: [(path: String, data: Data)] = []
    private var imageKeys: [AnyHashable: Int] = [:]
    private var geometries: [(path: String, text: String)] = []
    private var geometryKeys: [AnyHashable: Int] = [:]
    private var materials: [Description] = []
    private var materialNames: [String] = []
    private var materialKeys: [Description: Int] = [:]
    private var materialAllocator = USDNameAllocator()
    /// The timeline length in seconds when an animation is exported.
    private var duration: Double?
    private static let timeCodesPerSecond = 60.0

    private static let root = "/Root"
    private static let materialsPath = "/Root/Materials"
    private static let defaultMaterial: Description = {
        var material = Description()
        material.baseColor = SIMD4(0.8, 0.8, 0.8, 1)
        material.metallic = 0
        material.roughness = 0.5
        return material
    }()

    init(exporter: USDZExporter) throws {
        let options = exporter.options
        guard options.metersPerUnit.isFinite, options.metersPerUnit > 0,
              options.cameraAspectRatio.isFinite, options.cameraAspectRatio > 0 else {
            throw Failure.invalidData("metersPerUnit and cameraAspectRatio must be finite and positive.")
        }
        self.options = options
        reader = GLTFTextureReader(queue: exporter.commandQueue)
    }

    // MARK: Scene graphs

    mutating func export(
        _ graph: SceneGraph, nodeNames: [UUID: String], includesCamera: Bool, animation: SceneAnimation?
    ) throws -> USDZExporter.Output {
        // Rejects duplicate IDs, invalid transforms, and too many lights before writing anything.
        let resolved = try graph.resolve()
        if let batchNode = resolved.first(where: { $0.batch != nil }) {
            throw Failure.invalidData("Node \(batchNode.id) uses a batched mesh.")
        }
        var samples: [UUID: [(time: Double, transform: SceneTransform)]] = [:]
        if let animation {
            let ids = Set(resolved.map(\.id))
            for track in animation.tracks {
                guard ids.contains(track.nodeID) else {
                    throw Failure.invalidData("An animation track targets node \(track.nodeID), which is not in the graph.")
                }
                samples[track.nodeID] = track.keyframes.map { (Double($0.time), $0.transform) }
            }
            duration = Double(animation.duration)
        }
        var paths: [UUID: String] = [:]
        var rootNames = USDNameAllocator(reserved: ["Materials"])
        var cameraPath: String?
        var index = 0
        var body = USDAWriter()
        func write(_ node: SceneNode, parent: String, names: inout USDNameAllocator) throws {
            guard node.orientationMode == .inherited else {
                throw Failure.invalidData("Node \(node.id) uses world orientation, which USDZ export cannot retain.")
            }
            let name = names.name(nodeNames[node.id], fallback: "Node_\(index)")
            index += 1
            let path = "\(parent)/\(name)"
            paths[node.id] = path
            body.open("def Xform \"\(name)\"")
            if let keys = samples[node.id] {
                try Self.writeAnimatedTransform(keys, pivot: node.pivot, context: path, into: &body)
            } else {
                Self.writeTransform(node.transform, pivot: node.pivot, into: &body)
            }
            var children = USDNameAllocator()
            var renderer = node.mesh
            if renderer == nil, let levels = node.levelsOfDetail {
                renderer = levels.levels[0].mesh
                if levels.levels.count > 1 { warnings.append(.levelsOfDetailReduced(node: path, levelCount: levels.levels.count)) }
            }
            if node.light != nil { warnings.append(.lightOmitted(node: path)) }
            if let renderer {
                let mesh = try sceneGraphMesh(renderer, parameters: node.materialParameters, path: path)
                if let set = node.instances {
                    if set.tints.contains(where: { $0 != SIMD4(repeating: 1) }) { warnings.append(.instanceTintsDropped(node: path)) }
                    for (instance, matrix) in set.transforms.enumerated() {
                        let instanceName = children.name("Instance_\(instance)", fallback: "Instance")
                        body.open("def Xform \"\(instanceName)\"")
                        try Self.writeMatrix(matrix, context: "\(path) instance \(instance)", into: &body)
                        writeMeshPrim(name: "Mesh", geometry: mesh.geometry, material: mesh.material, doubleSided: mesh.doubleSided, into: &body)
                        body.close()
                    }
                } else {
                    let meshName = children.name("Mesh", fallback: "Mesh")
                    writeMeshPrim(name: meshName, geometry: mesh.geometry, material: mesh.material, doubleSided: mesh.doubleSided, into: &body)
                }
            }
            for child in node.children {
                try write(child, parent: path, names: &children)
            }
            body.close()
        }
        for node in graph.roots {
            try write(node, parent: Self.root, names: &rootNames)
        }
        if includesCamera {
            let name = rootNames.name("Camera", fallback: "Camera")
            cameraPath = "\(Self.root)/\(name)"
            body.open("def Camera \"\(name)\"")
            Self.writeTransform(graph.cameraTransform, into: &body)
            let projection: GLTFScene.Camera.Projection
            if let orthographic = graph.orthographicCamera {
                projection = .orthographic(
                    halfWidth: orthographic.halfHeight, halfHeight: orthographic.halfHeight,
                    near: orthographic.near, far: orthographic.far
                )
            } else {
                projection = .perspective(
                    verticalFieldOfView: graph.camera.verticalFieldOfView, aspectRatio: nil,
                    near: graph.camera.near, far: graph.camera.far
                )
            }
            try writeCamera(projection, path: cameraPath!, into: &body)
            body.close()
        }
        return try finish(body: body, nodePaths: paths, sourceNodePaths: [:], cameraPath: cameraPath)
    }

    private struct ExportedMesh {
        let geometry: Int
        let material: Int
        let doubleSided: Bool
    }

    private mutating func sceneGraphMesh(_ renderer: MeshRenderer, parameters: SIMD4<Float>, path: String) throws -> ExportedMesh {
        let material = renderer.pathTracingMaterial
        let deforms = material?.surfaceGraph?.hasVertexOutputs == true
        let key = AnyHashable(ObjectIdentifier(renderer))
        let geometryIndex: Int
        if let existing = geometryKeys[key] {
            geometryIndex = existing
        } else {
            let geometry = deforms
                ? try renderer.deformedGeometry(queue: reader.queue(for: renderer.device)) : renderer.geometry
            geometryIndex = try addGeometry(geometry, key: key, context: path)
            if deforms { warnings.append(.deformationBaked(mesh: path)) }
            if renderer.skin != nil { warnings.append(.skinDropped(mesh: path)) }
            if renderer.morphTargets != nil { warnings.append(.morphTargetsDropped(mesh: path)) }
        }
        let materialIndex: Int
        if let material {
            let (description, local) = try describe(material, parameters: parameters)
            materialIndex = addMaterial(description, warnings: local)
        } else {
            warnings.append(.unsupportedMaterial(mesh: path))
            materialIndex = addMaterial(Self.defaultMaterial, warnings: [])
        }
        return ExportedMesh(geometry: geometryIndex, material: materialIndex, doubleSided: material?.isDoubleSided ?? false)
    }

    /// Mirrors `GLTFSceneGraphExport.material`: the per-draw tint multiplies the base color
    /// factor without a surface graph, as the raster shader does.
    private mutating func describe(
        _ material: PhysicallyBasedMaterial, parameters: SIMD4<Float>
    ) throws -> (Description, [Warning]) {
        guard (0..<4).allSatisfy({ parameters[$0].isFinite }) else {
            throw Failure.invalidData("Scene graph material parameters must be finite.")
        }
        var local: [Warning] = []
        var customSampler = false
        func texture(_ binding: PhysicallyBasedMaterial.TextureBinding?, color: Bool) throws -> Description.Texture? {
            guard let binding else { return nil }
            if binding.sampler != nil { customSampler = true }
            let key = AnyHashable(SceneTextureKey(texture: ObjectIdentifier(binding.texture), color: color))
            let image: Int
            if let existing = imageKeys[key] {
                image = existing
            } else {
                image = addImage(try reader.png(binding.texture, color: color), extension: "png", key: key)
            }
            return .init(
                image: image, coordinates: binding.textureCoordinateSet, transform: binding.transform,
                wrapS: "repeat", wrapT: "repeat"
            )
        }

        var description = Description()
        description.baseColor = material.surfaceGraph == nil
            ? simd_clamp(parameters * material.baseColorFactor, .zero, SIMD4(repeating: 1)) : material.baseColorFactor
        guard !material.hasActiveBumpMap else {
            throw Failure.invalidData("Convert bump maps to tangent-space normal maps before USDZ export.")
        }
        let maps = material.coreMaps
        guard material.objectNormalMaps == 0 else {
            throw Failure.invalidData("Convert object-space normal maps to tangent space before USDZ export.")
        }
        guard material.baseNormalEncoding == .rgb else {
            throw Failure.invalidData("Convert packed tangent-space normals to RGB before USDZ export.")
        }
        description.baseColorTexture = try texture(maps.bindings[0], color: true)
        description.metallic = material.settings.y
        description.roughness = material.settings.x
        description.metallicRoughnessTexture = try texture(maps.bindings[1], color: false)
        description.normalTexture = try texture(maps.bindings[2], color: false)
        if maps.bindings[2] != nil, maps.normalScale != 1 { local.append(feature: .normalScale) }
        description.occlusionTexture = try texture(maps.bindings[3], color: false)
        description.occlusionStrength = maps.bindings[3] == nil ? 1 : maps.occlusionStrength
        description.emissiveTexture = try texture(maps.bindings[4], color: true)
        description.emissive = material.pathTracingEmission
        switch material.alphaMode {
        case .materialAlpha, .opaque: description.alpha = .opaque
        case .mask(let cutoff): description.alpha = cutoff > 0 ? .mask(cutoff) : .opaque
        case .blend: description.alpha = .blend
        }
        let isPBR = material.shadingModel == .physicallyBased
        description.isUnlit = !isPBR
        if isPBR, let coat = material.clearcoat, coat.factor > 0 {
            description.clearcoat = coat.factor
            description.clearcoatRoughness = coat.roughness
            description.clearcoatTexture = try texture(coat.factorTexture, color: false)
            description.clearcoatRoughnessTexture = try texture(coat.roughnessTexture, color: false)
            if coat.normalTexture != nil { local.append(feature: .clearcoatNormal) }
        }
        if let optical = material.transmission {
            if optical.indexOfRefraction >= 1 { description.indexOfRefraction = optical.indexOfRefraction }
            if isPBR, optical.factor > 0 || optical.factorTexture != nil { local.append(feature: .transmission) }
            if isPBR, optical.thickness > 0 || optical.thicknessTexture != nil || optical.attenuationDistance != nil {
                local.append(feature: .volume)
            }
            if isPBR, optical.dispersion > 0 { local.append(feature: .dispersion) }
        }
        if isPBR {
            if material.specular != nil { local.append(feature: .specular) }
            if material.hasSheen { local.append(feature: .sheen) }
            if material.hasAnisotropy { local.append(feature: .anisotropy) }
            if material.hasIridescence { local.append(feature: .iridescence) }
        }
        if let graph = material.surfaceGraph, graph.outputs.connected.contains(where: { $0.0 != .positionOffset }) {
            local.append(feature: .surfaceGraph)
        }
        if customSampler { local.append(.samplerUnavailable(material: "")) }
        if material.alphaMode == .materialAlpha, description.baseColor.w < 1 { local.append(.alphaExportedAsOpaque(material: "")) }
        if description.isUnlit { local.append(.unlitApproximated(material: "")) }
        return (description, local)
    }

    private struct SceneTextureKey: Hashable {
        let texture: ObjectIdentifier
        let color: Bool
    }

    // MARK: Imported scenes

    mutating func export(_ scene: GLTFScene, animationIndex: Int?) throws -> USDZExporter.Output {
        let nodes = Dictionary(uniqueKeysWithValues: scene.nodes.map { ($0.sourceIndex, $0) })
        let meshes = Dictionary(uniqueKeysWithValues: scene.meshes.map { ($0.sourceIndex, $0) })
        let materials = Dictionary(uniqueKeysWithValues: scene.materials.map { ($0.sourceIndex, $0) })
        let cameras = Dictionary(uniqueKeysWithValues: scene.cameras.map { ($0.sourceIndex, $0) })
        var described: [Int: Int] = [:]
        var paths: [Int: String] = [:]
        var body = USDAWriter()
        var rootNames = USDNameAllocator(reserved: ["Materials"])
        var samples: [Int: [(time: Double, transform: SceneTransform)]] = [:]
        if !scene.animations.isEmpty {
            let clip: GLTFAnimationClip
            if let animationIndex {
                guard let match = scene.animations.first(where: { $0.sourceIndex == animationIndex }) else {
                    throw Failure.missingReference("Animation \(animationIndex) is not in the scene.")
                }
                clip = match
            } else {
                clip = scene.animations[0]
            }
            if scene.animations.count > 1 { warnings.append(.animationsOmitted(count: scene.animations.count - 1)) }
            samples = try Self.sample(clip, nodes: nodes)
            duration = Double(clip.duration)
            if clip.channels.contains(where: { $0.target == .weights }) { warnings.append(.morphWeightAnimationOmitted) }
        }

        func materialIndex(_ source: Int?, mesh: String) throws -> Int {
            guard let source else { return addMaterial(Self.gltfDefaultMaterial, warnings: []) }
            if let existing = described[source] { return existing }
            guard let material = materials[source] else {
                throw Failure.missingReference("Material \(source) used by \(mesh) is not in the scene.")
            }
            let (description, local) = try describe(material)
            let index = addMaterial(description, warnings: local)
            described[source] = index
            return index
        }

        var visited = Set<Int>()
        func write(_ sourceIndex: Int, parent: String, names: inout USDNameAllocator) throws {
            guard let node = nodes[sourceIndex] else { throw Failure.missingReference("Node \(sourceIndex) is not in the scene.") }
            guard visited.insert(sourceIndex).inserted else {
                throw Failure.invalidData("Node \(sourceIndex) appears more than once in the hierarchy.")
            }
            let name = names.name(node.name, fallback: "Node_\(sourceIndex)")
            let path = "\(parent)/\(name)"
            paths[sourceIndex] = path
            body.open("def Xform \"\(name)\"")
            try Self.validate(node.localTransform, context: "node \(sourceIndex)")
            if let keys = samples[sourceIndex] {
                try Self.writeAnimatedTransform(keys, context: path, into: &body)
            } else {
                Self.writeTransform(node.localTransform, into: &body)
            }
            var children = USDNameAllocator()
            if node.lightIndex != nil { warnings.append(.lightOmitted(node: path)) }
            if let meshIndex = node.meshIndex {
                guard let mesh = meshes[meshIndex] else { throw Failure.missingReference("Mesh \(meshIndex) is not in the scene.") }
                if node.skinIndex != nil { warnings.append(.skinDropped(mesh: path)) }
                if mesh.morphTargets != nil { warnings.append(.morphTargetsDropped(mesh: path)) }
                var prims: [(name: String, geometry: Int, material: Int, doubleSided: Bool)] = []
                for (primitive, geometry) in mesh.primitives.enumerated() {
                    let key = AnyHashable(SourcePrimitiveKey(mesh: meshIndex, primitive: primitive))
                    let geometryIndex = try geometryKeys[key] ?? addGeometry(geometry, key: key, context: path)
                    let source = primitive < mesh.primitiveMaterialIndices.count ? mesh.primitiveMaterialIndices[primitive] : nil
                    let material = try materialIndex(source, mesh: path)
                    let doubleSided = source.flatMap { materials[$0]?.isDoubleSided } ?? false
                    prims.append((mesh.primitives.count == 1 ? "Mesh" : "Mesh_\(primitive)", geometryIndex, material, doubleSided))
                }
                if let instances = node.instances {
                    for (instance, transform) in instances.enumerated() {
                        let instanceName = children.name("Instance_\(instance)", fallback: "Instance")
                        body.open("def Xform \"\(instanceName)\"")
                        try Self.validate(transform, context: "node \(sourceIndex) instance \(instance)")
                        Self.writeTransform(transform, into: &body)
                        var instanceNames = USDNameAllocator()
                        for prim in prims {
                            let primName = instanceNames.name(prim.name, fallback: "Mesh")
                            writeMeshPrim(name: primName, geometry: prim.geometry, material: prim.material, doubleSided: prim.doubleSided, into: &body)
                        }
                        body.close()
                    }
                } else {
                    for prim in prims {
                        let primName = children.name(prim.name, fallback: "Mesh")
                        writeMeshPrim(name: primName, geometry: prim.geometry, material: prim.material, doubleSided: prim.doubleSided, into: &body)
                    }
                }
            }
            if let cameraIndex = node.cameraIndex {
                guard let camera = cameras[cameraIndex] else { throw Failure.missingReference("Camera \(cameraIndex) is not in the scene.") }
                let cameraName = children.name(camera.name ?? "Camera", fallback: "Camera")
                body.open("def Camera \"\(cameraName)\"")
                try writeCamera(camera.projection, path: "\(path)/\(cameraName)", into: &body)
                body.close()
            }
            for child in node.childIndices {
                try write(child, parent: path, names: &children)
            }
            body.close()
        }
        for root in scene.rootNodeIndices {
            try write(root, parent: Self.root, names: &rootNames)
        }
        return try finish(body: body, nodePaths: [:], sourceNodePaths: paths, cameraPath: nil)
    }

    private struct SourcePrimitiveKey: Hashable {
        let mesh: Int
        let primitive: Int
    }

    private struct SourceImageKey: Hashable {
        let image: Int
    }

    /// glTF's default material: white, fully metallic, fully rough.
    private static let gltfDefaultMaterial = Description()

    private mutating func describe(_ material: GLTFScene.Material) throws -> (Description, [Warning]) {
        guard !material.hasActiveBumpMap else {
            throw Failure.invalidData("Convert bump maps to tangent-space normal maps before USDZ export.")
        }
        var local: [Warning] = []
        var nearest = false
        func texture(_ source: GLTFScene.Texture?) throws -> Description.Texture? {
            guard let source else { return nil }
            if source.sampling.magnification == .nearest
                || [.nearest, .nearestMipmapNearest, .nearestMipmapLinear].contains(source.sampling.minification) {
                nearest = true
            }
            guard let image = try packageImage(source) else {
                local.append(.textureOmitted(material: ""))
                return nil
            }
            func wrap(_ mode: GLTFScene.Sampling.Wrap) -> String {
                switch mode {
                case .repeat: "repeat"
                case .mirroredRepeat: "mirror"
                case .clampToEdge: "clamp"
                }
            }
            return .init(
                image: image, coordinates: source.textureCoordinateSet, transform: source.transform,
                wrapS: wrap(source.sampling.wrapS), wrapT: wrap(source.sampling.wrapT)
            )
        }

        var description = Description()
        description.baseColor = material.baseColorFactor
        description.baseColorTexture = try texture(material.baseColorTexture)
        description.metallic = material.metallicFactor
        description.roughness = material.roughnessFactor
        description.metallicRoughnessTexture = try texture(material.metallicRoughnessTexture)
        description.normalTexture = try texture(material.normalTexture)
        if material.normalTexture != nil, material.normalScale != 1 { local.append(feature: .normalScale) }
        description.occlusionTexture = try texture(material.occlusionTexture)
        description.occlusionStrength = material.occlusionStrength
        description.emissiveTexture = try texture(material.emissiveTexture)
        description.emissive = material.emissiveFactor * material.emissiveStrength
        switch material.alphaMode {
        case .opaque: description.alpha = .opaque
        case .mask(let cutoff): description.alpha = cutoff > 0 ? .mask(cutoff) : .opaque
        case .blend: description.alpha = .blend
        }
        description.isUnlit = material.isUnlit
        if !material.isUnlit {
            if let coat = material.clearcoat, coat.factor > 0 || coat.factorTexture != nil {
                description.clearcoat = coat.factor
                description.clearcoatRoughness = coat.roughness
                description.clearcoatTexture = try texture(coat.factorTexture)
                description.clearcoatRoughnessTexture = try texture(coat.roughnessTexture)
                if coat.normalTexture != nil { local.append(feature: .clearcoatNormal) }
            }
            if material.indexOfRefraction >= 1 { description.indexOfRefraction = material.indexOfRefraction }
            if let transmission = material.transmission, transmission.factor > 0 || transmission.factorTexture != nil {
                local.append(feature: .transmission)
            }
            if material.volume != nil { local.append(feature: .volume) }
            if material.specular != nil { local.append(feature: .specular) }
            if let sheen = material.sheen, sheen.colorFactor != .zero || sheen.colorTexture != nil { local.append(feature: .sheen) }
            if let anisotropy = material.anisotropy, anisotropy.strength > 0 { local.append(feature: .anisotropy) }
            if let iridescence = material.iridescence, iridescence.factor > 0 { local.append(feature: .iridescence) }
            if material.dispersion > 0 { local.append(feature: .dispersion) }
        } else {
            local.append(.unlitApproximated(material: ""))
        }
        if nearest { local.append(.textureFilteringDropped(material: "")) }
        return (description, local)
    }

    /// Stores a texture's image once, choosing a PNG or JPEG form USDZ accepts.
    private mutating func packageImage(_ texture: GLTFScene.Texture) throws -> Int? {
        let key = AnyHashable(SourceImageKey(image: texture.image.sourceIndex))
        if let existing = imageKeys[key] { return existing }
        var image = texture.image
        if image.mimeType == "image/ktx2" {
            guard let fallback = texture.fallbackImage, ["image/png", "image/jpeg"].contains(fallback.mimeType) else { return nil }
            image = fallback
        }
        switch image.mimeType {
        case "image/png": return addImage(image.data, extension: "png", key: key)
        case "image/jpeg": return addImage(image.data, extension: "jpg", key: key)
        default:
            let index = addImage(try Self.transcodeToPNG(image.data, name: image.name ?? "\(image.sourceIndex)"), extension: "png", key: key)
            warnings.append(.imageTranscoded(image: images[index].path))
            return index
        }
    }

    private static func transcodeToPNG(_ data: Data, name: String) throws -> Data {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw Failure.imageEncoding(name) }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, "public.png" as CFString, 1, nil) else {
            throw Failure.imageEncoding(name)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw Failure.imageEncoding(name) }
        return output as Data
    }

    // MARK: Shared writing

    private mutating func addImage(_ data: Data, extension fileExtension: String, key: AnyHashable) -> Int {
        let index = images.count
        images.append(("textures/Texture_\(index).\(fileExtension)", data))
        imageKeys[key] = index
        return index
    }

    private mutating func addGeometry(_ geometry: MeshGeometry, key: AnyHashable, context: String) throws -> Int {
        guard geometry.vertices.count <= Int(Int32.max) else { throw Failure.meshTooLarge(context) }
        let secondary = geometry.vertices.filter { $0.secondaryTextureCoordinate != nil }.count
        let tangents = geometry.vertices.filter { $0.tangent != nil }.count
        if secondary > 0, secondary < geometry.vertices.count { warnings.append(.attributeDropped(mesh: context, attribute: "st1")) }
        if tangents > 0, tangents < geometry.vertices.count { warnings.append(.attributeDropped(mesh: context, attribute: "tangents")) }
        let index = geometries.count
        let text = USDGeometryLayer.text(
            geometry, metersPerUnit: options.metersPerUnit,
            includeSecondary: secondary == geometry.vertices.count, includeTangents: tangents == geometry.vertices.count
        )
        geometries.append(("geometries/Geometry_\(index).usda", text))
        geometryKeys[key] = index
        return index
    }

    private mutating func addMaterial(_ description: Description, warnings local: [Warning]) -> Int {
        if let existing = materialKeys[description] { return existing }
        let index = materials.count
        materials.append(description)
        let name = materialAllocator.name("Material_\(index)", fallback: "Material")
        materialNames.append(name)
        materialKeys[description] = index
        let path = "\(Self.materialsPath)/\(name)"
        warnings += local.map { $0.withMaterialPath(path) }
        return index
    }

    private func writeMeshPrim(name: String, geometry: Int, material: Int, doubleSided: Bool, into body: inout USDAWriter) {
        body.open("def Mesh \"\(name)\"", metadata: [
            "prepend apiSchemas = [\"MaterialBindingAPI\"]",
            "prepend references = @\(geometries[geometry].path)@</Geometry>",
        ])
        body.line("rel material:binding = <\(Self.materialsPath)/\(materialNames[material])>")
        body.line("uniform bool doubleSided = \(doubleSided ? 1 : 0)")
        body.close()
    }

    private mutating func writeCamera(_ projection: GLTFScene.Camera.Projection, path: String, into body: inout USDAWriter) throws {
        let verticalAperture: Float = 24
        switch projection {
        case .perspective(let fieldOfView, let aspectRatio, let near, var far):
            let aspect = aspectRatio ?? options.cameraAspectRatio
            guard fieldOfView.isFinite, fieldOfView > 0, fieldOfView < .pi, near.isFinite, near > 0, aspect.isFinite, aspect > 0 else {
                throw Failure.invalidData("Camera \(path) needs a field of view in (0, π), a positive near plane, and a positive aspect ratio.")
            }
            if far == nil || far?.isFinite == false {
                far = max(near * 1e6, 1e6)
                warnings.append(.infiniteFarPlaneClamped(camera: path, far: far!))
            }
            guard let far, far > near else { throw Failure.invalidData("Camera \(path) needs a far plane beyond its near plane.") }
            body.line("float focalLength = \(USDAWriter.number(verticalAperture / 2 / tan(fieldOfView / 2)))")
            body.line("float horizontalAperture = \(USDAWriter.number(verticalAperture * aspect))")
            body.line("float verticalAperture = \(USDAWriter.number(verticalAperture))")
            body.line("float2 clippingRange = \(USDAWriter.tuple(near, far))")
            body.line("token projection = \"perspective\"")
        case .orthographic(let halfWidth, let halfHeight, let near, let far):
            guard [halfWidth, halfHeight, near, far].allSatisfy(\.isFinite), halfWidth > 0, halfHeight > 0, near >= 0, far > near else {
                throw Failure.invalidData("Camera \(path) needs positive orthographic extents and near < far.")
            }
            // USD apertures are in tenths of a scene unit.
            body.line("float horizontalAperture = \(USDAWriter.number(halfWidth * 20))")
            body.line("float verticalAperture = \(USDAWriter.number(halfHeight * 20))")
            body.line("float2 clippingRange = \(USDAWriter.tuple(near, far))")
            body.line("token projection = \"orthographic\"")
        }
    }

    private static func validate(_ transform: SceneTransform, context: String) throws {
        let values = [transform.translation.x, transform.translation.y, transform.translation.z,
                      transform.rotation.vector.x, transform.rotation.vector.y, transform.rotation.vector.z,
                      transform.rotation.vector.w, transform.scale.x, transform.scale.y, transform.scale.z]
        guard values.allSatisfy(\.isFinite), abs(simd_length_squared(transform.rotation.vector) - 1) <= 0.0001 else {
            throw Failure.invalidData("The \(context) transform needs finite values and a unit rotation.")
        }
    }

    /// Writes only the non-identity transform operations in stack order.
    private static func writeTransform(
        _ transform: SceneTransform, pivot: SIMD3<Float>? = nil, into body: inout USDAWriter
    ) {
        var order: [String] = []
        let t = transform.translation, q = transform.rotation.vector, s = transform.scale
        if t != .zero {
            body.line("double3 xformOp:translate = \(USDAWriter.tuple(t.x, t.y, t.z))")
            order.append("xformOp:translate")
        }
        if q != SIMD4(0, 0, 0, 1) {
            // GfQuatf text lists the real part first.
            body.line("quatf xformOp:orient = \(USDAWriter.tuple(q.w, q.x, q.y, q.z))")
            order.append("xformOp:orient")
        }
        if s != SIMD3(repeating: 1) {
            body.line("float3 xformOp:scale = \(USDAWriter.tuple(s.x, s.y, s.z))")
            order.append("xformOp:scale")
        }
        writeTransformOrder(order, pivot: pivot, into: &body)
    }

    /// Writes translate, orient, and scale time samples. Consecutive quaternions stay in one
    /// hemisphere so USD's spherical interpolation takes the short arc, as `simd_slerp` does.
    private static func writeAnimatedTransform(
        _ keys: [(time: Double, transform: SceneTransform)], pivot: SIMD3<Float>? = nil,
        context: String, into body: inout USDAWriter
    ) throws {
        var rotations: [SIMD4<Float>] = []
        for key in keys {
            try validate(key.transform, context: "\(context) at \(key.time) s")
            var q = key.transform.rotation.vector
            if let previous = rotations.last, simd_dot(previous, q) < 0 { q = -q }
            rotations.append(q)
        }
        func samples(_ values: [String]) -> String {
            "{ " + zip(keys, values).map { "\(USDAWriter.number($0.time * timeCodesPerSecond)): \($1)" }.joined(separator: ", ") + " }"
        }
        body.line("double3 xformOp:translate.timeSamples = " + samples(keys.map {
            USDAWriter.tuple($0.transform.translation.x, $0.transform.translation.y, $0.transform.translation.z)
        }))
        body.line("quatf xformOp:orient.timeSamples = " + samples(rotations.map { USDAWriter.tuple($0.w, $0.x, $0.y, $0.z) }))
        body.line("float3 xformOp:scale.timeSamples = " + samples(keys.map {
            USDAWriter.tuple($0.transform.scale.x, $0.transform.scale.y, $0.transform.scale.z)
        }))
        writeTransformOrder(["xformOp:translate", "xformOp:orient", "xformOp:scale"], pivot: pivot, into: &body)
    }

    private static func writeTransformOrder(
        _ order: [String], pivot: SIMD3<Float>?, into body: inout USDAWriter
    ) {
        var ops = order
        if let pivot {
            let name = "xformOp:translate:pivot"
            body.line("double3 \(name) = \(USDAWriter.tuple(pivot.x, pivot.y, pivot.z))")
            ops.insert(name, at: order.first == "xformOp:translate" ? 1 : 0)
            ops.append("!invert!" + name)
        }
        if !ops.isEmpty {
            let tokens = ops.map { "\"\($0)\"" }.joined(separator: ", ")
            body.line("uniform token[] xformOpOrder = [\(tokens)]")
        }
    }

    /// Samples each animated node's full local transform at the clip's key times, adding
    /// 1/60 s steps between keys of STEP and CUBICSPLINE channels that USD cannot interpolate.
    private static func sample(
        _ clip: GLTFAnimationClip, nodes: [Int: GLTFScene.Node]
    ) throws -> [Int: [(time: Double, transform: SceneTransform)]] {
        var times: [Int: Set<Float>] = [:]
        for channel in clip.channels where channel.target != .weights {
            guard nodes[channel.sourceNodeIndex] != nil else {
                throw Failure.missingReference("Animated node \(channel.sourceNodeIndex) is not in the scene.")
            }
            var keys = Set(channel.times)
            if channel.interpolation != .linear, let first = channel.times.first, let last = channel.times.last, last > first {
                let steps = Int(((last - first) * Float(timeCodesPerSecond)).rounded(.up))
                for step in 0...steps { keys.insert(min(last, first + Float(step) / Float(timeCodesPerSecond))) }
            }
            times[channel.sourceNodeIndex, default: []].formUnion(keys)
        }
        // Each node evaluates only its own channels, so the cost grows with the node's keys
        // rather than with every node's channels at every key of the clip.
        let channels = Dictionary(grouping: clip.channels.filter { $0.target != .weights }, by: \.sourceNodeIndex)
        var result: [Int: [(time: Double, transform: SceneTransform)]] = [:]
        for (node, keys) in times {
            guard let rest = nodes[node]?.localTransform, let own = channels[node] else { continue }
            result[node] = try keys.sorted().map { time in
                var pose = AnimationNodePose()
                // Key times lie inside the clip, so the clip's clamp to [0, duration] is not needed.
                for channel in own { try channel.sample(at: Double(time), into: &pose) }
                return (Double(time), pose.applied(to: rest))
            }
        }
        return result
    }

    /// USD matrices use row vectors, so each Frostfield column is written as a row.
    private static func writeMatrix(_ matrix: simd_float4x4, context: String, into body: inout USDAWriter) throws {
        let columns = [matrix.columns.0, matrix.columns.1, matrix.columns.2, matrix.columns.3]
        guard columns.allSatisfy({ column in (0..<4).allSatisfy { column[$0].isFinite } }) else {
            throw Failure.invalidData("The \(context) matrix must be finite.")
        }
        let rows = columns.map { USDAWriter.tuple($0.x, $0.y, $0.z, $0.w) }.joined(separator: ", ")
        body.line("matrix4d xformOp:transform = (\(rows))")
        body.line("uniform token[] xformOpOrder = [\"xformOp:transform\"]")
    }

    private mutating func finish(
        body: USDAWriter, nodePaths: [UUID: String], sourceNodePaths: [Int: String], cameraPath: String?
    ) throws -> USDZExporter.Output {
        var layer = USDAWriter()
        layer.open("def Xform \"Root\"", metadata: ["kind = \"component\""])
        layer.open("def Scope \"Materials\"")
        let imagePaths = images.map(\.path)
        for (index, material) in materials.enumerated() {
            if index > 0 { layer.line() }
            USDMaterialNetwork.write(
                material, name: materialNames[index], parentPath: Self.materialsPath, imagePaths: imagePaths, into: &layer
            )
        }
        layer.close()
        for line in body.text.split(separator: "\n", omittingEmptySubsequences: false).dropLast() {
            layer.line(String(line))
        }
        layer.close()
        var metadata = ["customLayerData = {", "    string creator = \"Frostfield USDZExporter\"", "}"]
        if let duration {
            metadata += [
                "endTimeCode = \(USDAWriter.number(duration * Self.timeCodesPerSecond))", "startTimeCode = 0",
                "timeCodesPerSecond = \(USDAWriter.number(Self.timeCodesPerSecond))",
            ]
        }
        let rootLayer = USDAWriter.layerHeader(defaultPrim: "Root", metersPerUnit: options.metersPerUnit, extra: metadata)
            + layer.text

        var entries = [USDZArchive.Entry(path: "scene.usda", data: Data(rootLayer.utf8))]
        entries += geometries.map { USDZArchive.Entry(path: $0.path, data: Data($0.text.utf8)) }
        entries += images.map { USDZArchive.Entry(path: $0.path, data: $0.data) }
        return USDZExporter.Output(
            data: try USDZArchive.write(entries), warnings: warnings, rootLayer: rootLayer,
            nodePaths: nodePaths, sourceNodePaths: sourceNodePaths, cameraPath: cameraPath
        )
    }
}

private extension Array where Element == USDZExporter.Warning {
    mutating func append(feature: USDZExporter.MaterialFeature) {
        append(.materialFeatureDropped(material: "", feature: feature))
    }
}

private extension USDZExporter.Warning {
    /// Material warnings are collected before the material has a prim path.
    func withMaterialPath(_ path: String) -> Self {
        switch self {
        case .materialFeatureDropped(_, let feature): .materialFeatureDropped(material: path, feature: feature)
        case .unlitApproximated: .unlitApproximated(material: path)
        case .alphaExportedAsOpaque: .alphaExportedAsOpaque(material: path)
        case .samplerUnavailable: .samplerUnavailable(material: path)
        case .textureFilteringDropped: .textureFilteringDropped(material: path)
        case .textureOmitted: .textureOmitted(material: path)
        default: self
        }
    }
}
