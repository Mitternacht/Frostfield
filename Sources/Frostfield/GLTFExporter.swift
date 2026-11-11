import Foundation
import Metal
import simd

/// Writes self-contained glTF 2.0 assets from an imported `GLTFScene` or a Frostfield `SceneGraph`.
///
/// A `.glb` stores every buffer and image in one binary chunk. A `.gltf` stores the
/// same data as base64 data URIs. Coordinates stay right-handed and Y-up in meters,
/// texture coordinates keep their top-left origin, and tangents keep their `w`
/// handedness. Float values are written so they decode to the same bits.
///
/// Export runs synchronously. Reading private or managed textures and evaluating
/// GPU-deformed meshes waits for the GPU, so call it off the main thread for large scenes.
/// Finish GPU writes to exported shared textures before exporting.
///
/// The output follows the glTF 2.0 structural rules: four-byte aligned buffer views,
/// POSITION bounds, matching byte lengths, and padded GLB chunks. `GLTFMeshLoader`
/// reads assets up to 64 MiB. Larger outputs are still written and add a warning.
public struct GLTFExporter {
    public enum Format: String, Sendable, Equatable {
        /// A binary container. The raw value is the file extension.
        case glb
        /// glTF JSON whose buffer and images are embedded as base64 data URIs.
        case gltf
    }

    /// A named transform clip targeting scene graph node IDs. Exported channels use LINEAR interpolation.
    public struct Animation {
        public let name: String?
        public let animation: SceneAnimation

        public init(name: String? = nil, animation: SceneAnimation) {
            self.name = name
            self.animation = animation
        }
    }

    /// A lossy conversion the exporter performed, or output that readers other than
    /// `GLTFMeshLoader` may reject. Indices refer to the exported glTF arrays.
    public enum Warning: Equatable, Sendable, CustomStringConvertible {
        /// glTF cannot represent surface shader graphs. The material's authored factors and
        /// maps were exported, and these graph-driven channels, uniforms, textures, and
        /// per-draw graph parameters were dropped.
        case surfaceGraphChannelsDropped(material: Int, channels: [SurfaceShaderGraph.Channel])
        /// A surface graph's vertex position offset was evaluated once and baked into the mesh.
        case deformationBaked(mesh: Int)
        /// The renderer does not use a `.physicallyBased` material, so the primitive uses glTF's default material.
        case unsupportedMaterial(mesh: Int)
        /// A scene graph skin has no joint node references, so the mesh was exported in its bind pose.
        case skinDropped(mesh: Int)
        /// Metal sampler states cannot be inspected, and a map's custom sampler has no
        /// `TextureSampling` description. The material's undescribed maps use repeat wrapping
        /// with linear, mipmapped filtering, which matches Frostfield's default sampler.
        case samplerUnavailable(material: Int)
        /// The primitive's source material was not loaded, so it uses glTF's default material.
        case materialNotLoaded(mesh: Int, primitive: Int, sourceMaterial: Int)
        /// Only some vertices carry this attribute, so it was omitted from the primitive.
        case attributeDropped(mesh: Int, primitive: Int, attribute: String)
        /// The material-alpha draw mode has no glTF equivalent, and the material has an alpha below one.
        /// It was exported as OPAQUE.
        case alphaExportedAsOpaque(material: Int)
        /// The asset exceeds the 64 MiB that `GLTFMeshLoader` reads.
        case exceedsImporterLimit(bytes: Int)
        /// KHR_lights_punctual has no area lights, so a rectangle light was omitted.
        case rectangleLightDropped
        /// glTF has no IES profiles or light cookies, so a light was exported without them.
        case lightProfileDropped
        /// The image is a KTX2 payload in a GPU block or uncompressed format, written through
        /// KHR_texture_basisu as `GLTFMeshLoader` read it. That extension specifies Basis Universal
        /// payloads, so other readers may use the texture's PNG or JPEG fallback or reject the image.
        case nonBasisKTX2(image: Int)
        /// glTF stores no selected variant. Primitives keep their default materials, and the
        /// selected KHR_materials_variants variant remains selectable by its index and name.
        case materialVariantSelectionNotStored(variant: Int)
        /// The mesh was exported without its morph targets. Node weights and weight animation channels were dropped.
        case morphTargetsDropped(mesh: Int)
        /// EXT_mesh_gpu_instancing has no standard per-instance color, so the node's instance tints were dropped.
        case instanceTintsDropped(node: Int)
        /// The node's instance set is empty, so it draws nothing. It was exported without its mesh.
        case emptyInstancesDropped(node: Int)
        /// Only the finest of the node's levels of detail was exported, as its mesh.
        case levelsOfDetailReduced(node: Int, levelCount: Int)
        /// The `KHR_animation_pointer` channels for materials, lights and cameras of
        /// the imported animation with this source index were not exported; its node
        /// channels were, and an animation with no node channels was omitted.
        case animationPointersDropped(sourceAnimation: Int, pointers: [String])
        /// An animation track targets a hidden scene node, which was not exported, so the track was dropped.
        case hiddenAnimationTargetDropped(UUID)
        /// The material was exported without layers that have no equivalent in this glTF writer,
        /// such as Frostfield's subsurface scattering.
        case materialLayersDropped(material: Int, layers: [String])

        public var description: String {
            switch self {
            case .surfaceGraphChannelsDropped(let material, let channels):
                "Material \(material) exported its base material. Surface graph channels were dropped: "
                    + channels.map(\.rawValue).joined(separator: ", ") + "."
            case .deformationBaked(let mesh): "Mesh \(mesh) contains a surface graph's baked vertex offset."
            case .unsupportedMaterial(let mesh): "Mesh \(mesh) has a non-PBR material and uses glTF's default material."
            case .skinDropped(let mesh): "Mesh \(mesh) was exported in its bind pose without its skin."
            case .samplerUnavailable(let material): "Material \(material) uses glTF's default repeat sampler in place of a custom sampler."
            case .materialNotLoaded(let mesh, let primitive, let source):
                "Mesh \(mesh) primitive \(primitive) uses glTF's default material because source material \(source) was not loaded."
            case .attributeDropped(let mesh, let primitive, let attribute):
                "Mesh \(mesh) primitive \(primitive) omits \(attribute), which only some vertices carry."
            case .alphaExportedAsOpaque(let material): "Material \(material) was exported as OPAQUE from Frostfield's material-alpha mode."
            case .exceedsImporterLimit(let bytes): "The \(bytes)-byte asset exceeds GLTFMeshLoader's 64 MiB limit."
            case .rectangleLightDropped: "A rectangle light was omitted because glTF punctual lights have no area lights."
            case .lightProfileDropped: "A light was exported without its IES profile or cookie, which glTF cannot represent."
            case .nonBasisKTX2(let image):
                "Image \(image) is a KTX2 payload that is not Basis Universal. Readers other than GLTFMeshLoader may need its PNG or JPEG fallback."
            case .materialVariantSelectionNotStored(let variant):
                "Selected material variant \(variant) is not stored. Primitives use their default materials."
            case .morphTargetsDropped(let mesh): "Mesh \(mesh) was exported without its morph targets, weights, and weight animation."
            case .instanceTintsDropped(let node): "Node \(node) was exported without its instance tints."
            case .emptyInstancesDropped(let node): "Node \(node) has no instances and was exported without its mesh."
            case .levelsOfDetailReduced(let node, let count):
                "Node \(node) was exported with the finest of its \(count) levels of detail."
            case .animationPointersDropped(let animation, let pointers):
                "Source animation \(animation) was exported without its KHR_animation_pointer channels: "
                    + pointers.joined(separator: ", ") + "."
            case .hiddenAnimationTargetDropped(let id):
                "An animation track for scene node \(id) was dropped because hidden nodes are not exported."
            case .materialLayersDropped(let material, let layers):
                "Material \(material) was exported without " + layers.joined(separator: ", ") + "."
            }
        }
    }

    /// An encoded asset and its export report.
    public struct Output: Sendable {
        public let data: Data
        public let format: Format
        /// Lossy conversions, in the order they were made. Empty for a lossless export.
        public let warnings: [Warning]
        /// The exported glTF node index for each scene graph node ID. Empty for `GLTFScene` exports.
        /// A pivoted node maps to its animated TRS container. The container's
        /// child holds the mesh, light, and authored children.
        public let nodeIndices: [UUID: Int]
        /// The exported node index for each source node index. Empty for scene graph exports.
        public let sourceNodeIndices: [Int: Int]
        /// The node carrying the scene graph's camera, when it was exported.
        public let cameraNodeIndex: Int?
    }

    public enum Failure: LocalizedError, Equatable {
        /// Vertex, transform, light, camera, material, or animation data is non-finite, empty, or out of range.
        case invalidData(String)
        /// A triangle index refers past the end of its primitive's vertices.
        case indexOutOfBounds(mesh: Int, primitive: Int, index: UInt32)
        /// A transform needs finite TRS values, a unit rotation, and an invertible matrix.
        case invalidTransform(String)
        /// A `GLTFScene` field refers to a node, mesh, material, skin, light, or camera it does not contain.
        case missingReference(String)
        /// An animation track targets a node ID that is not in the exported graph.
        case missingAnimationTarget(UUID)
        /// PNG export reads RGBA8, BGRA8, their sRGB forms, RGBA16Float, R8, RG8, R16Float, and RG16Float.
        case unsupportedTextureFormat(MTLPixelFormat)
        /// PNG export reads level 0 of a readable, single-sample 2D texture.
        case unsupportedTexture(String)
        /// The command queue belongs to a different device than the texture or renderer.
        case wrongDevice
        /// The blit that copies a private or managed texture did not complete.
        case textureReadback
        /// ImageIO could not create or finalize the PNG.
        case imageEncoding
        /// Metal could not create the named command queue, buffer, or encoder.
        case resource(String)
        /// A GLB and its chunks have 32-bit lengths, so an asset must stay below 4 GiB.
        case sizeLimit

        public var errorDescription: String? {
            switch self {
            case .invalidData(let reason): "Cannot export invalid glTF data. \(reason)"
            case .indexOutOfBounds(let mesh, let primitive, let index):
                "Index \(index) in mesh \(mesh) primitive \(primitive) is outside the vertex array."
            case .invalidTransform(let context): "The \(context) transform needs finite TRS values, a unit rotation, and an invertible matrix."
            case .missingReference(let reason): "The glTF scene is missing a referenced object. \(reason)"
            case .missingAnimationTarget(let id): "An exported animation targets node \(id), which is not in the scene graph."
            case .unsupportedTextureFormat(let format):
                "Pixel format \(format.rawValue) cannot be exported. Use RGBA8, BGRA8, sRGB RGBA8 or BGRA8, RGBA16Float, R8, RG8, R16Float, or RG16Float."
            case .unsupportedTexture(let reason): "The texture cannot be exported. \(reason)"
            case .wrongDevice: "The export command queue belongs to a different Metal device than the texture."
            case .textureReadback: "Metal could not copy the texture for export."
            case .imageEncoding: "ImageIO could not encode the texture as PNG."
            case .resource(let name): "Could not create the Metal \(name) needed for export."
            case .sizeLimit: "The glTF asset exceeds the 4 GiB limit of a GLB container."
            }
        }
    }

    public var format: Format
    /// Copies private and managed textures and evaluates GPU-deformed meshes. It must use
    /// the device of those textures and renderers. Without a queue, the exporter creates one per device.
    public var commandQueue: (any MTLCommandQueue)?

    public init(format: Format = .glb, commandQueue: (any MTLCommandQueue)? = nil) {
        self.format = format
        self.commandQueue = commandQueue
    }

    /// Writes an imported scene with its node order, names, meshes, loaded materials, original
    /// encoded images, samplers, skins, lights, cameras, and animation interpolation.
    /// Node transforms are written as TRS. Primitives that referred to an unloaded material
    /// use glTF's default material and add a warning.
    ///
    /// Material extensions, texture transforms, material variants, and GPU instances are
    /// written with their glTF extensions and listed in `extensionsUsed`:
    ///
    /// - Every textureInfo with a non-identity transform carries KHR_texture_transform.
    /// - Clearcoat, transmission, volume, IOR, emissive strength, specular, sheen,
    ///   anisotropy, iridescence, dispersion, and unlit use their KHR_materials extensions.
    /// - PNG and JPEG images are a texture's core `source`. KTX2, WebP, and AVIF images
    ///   are referenced from KHR_texture_basisu, EXT_texture_webp, or EXT_texture_avif.
    ///   A KTX2 texture's PNG or JPEG `fallbackImage` becomes its core `source`. A texture
    ///   without such a fallback omits `source`, and its extension is also listed in
    ///   `extensionsRequired`, as those extension specifications prescribe.
    /// - KHR_materials_variants writes the variant names and each primitive's mappings.
    ///   Primitives keep their default materials; a selected variant adds a warning.
    /// - Node instances become EXT_mesh_gpu_instancing TRANSLATION, ROTATION, and SCALE
    ///   float accessors. Load them back with `includeGPUInstancing`.
    ///
    /// Morph targets are not exported. A mesh with them adds a warning, and weight animation
    /// channels are skipped.
    public func export(_ scene: GLTFScene) throws -> Output {
        var builder = GLTFExportBuilder(format: format)
        let nodes = Dictionary(uniqueKeysWithValues: scene.nodes.enumerated().map { ($0.element.sourceIndex, $0.offset) })
        let meshes = Dictionary(uniqueKeysWithValues: scene.meshes.enumerated().map { ($0.element.sourceIndex, $0.offset) })
        let materials = Dictionary(uniqueKeysWithValues: scene.materials.enumerated().map { ($0.element.sourceIndex, $0.offset) })
        let skins = Dictionary(uniqueKeysWithValues: scene.skins.enumerated().map { ($0.element.sourceIndex, $0.offset) })
        let lights = Dictionary(uniqueKeysWithValues: scene.lights.enumerated().map { ($0.element.sourceIndex, $0.offset) })
        let cameras = Dictionary(uniqueKeysWithValues: scene.cameras.enumerated().map { ($0.element.sourceIndex, $0.offset) })
        func lookup(_ table: [Int: Int], _ index: Int, _ kind: String) throws -> Int {
            guard let mapped = table[index] else { throw Failure.missingReference("\(kind) \(index) is not in the scene.") }
            return mapped
        }

        var images: [Int: Int] = [:]
        func image(_ source: GLTFScene.Image) throws -> Int {
            if let existing = images[source.sourceIndex] { return existing }
            let index = try builder.image(source.data, mimeType: source.mimeType, name: source.name)
            images[source.sourceIndex] = index
            return index
        }
        func texture(_ source: GLTFScene.Texture?) throws -> GLTFExportMaterial.TextureReference? {
            guard let source else { return nil }
            let core: Int?
            var payload: (name: String, image: Int)?
            if let name = try GLTFExportBuilder.textureExtension(forMIMEType: source.image.mimeType) {
                let index = try image(source.image)
                payload = (name, index)
                if name == GLTFMaterialDocument.basisuExtension, !GLTFExportBuilder.isBasisUniversal(source.image.data) {
                    builder.warn(.nonBasisKTX2(image: index))
                }
                core = try source.fallbackImage.map { fallback in
                    guard try GLTFExportBuilder.textureExtension(forMIMEType: fallback.mimeType) == nil else {
                        throw Failure.invalidData("Texture \(source.sourceIndex) needs a PNG or JPEG fallback image, not \(fallback.mimeType).")
                    }
                    return try image(fallback)
                }
            } else {
                core = try image(source.image)
            }
            let index = builder.texture(source: core, extensionImage: payload, sampler: builder.sampler(source.sampling))
            return .init(index: index, coordinates: source.textureCoordinateSet, transform: source.transform)
        }

        for material in scene.materials {
            guard !material.hasActiveBumpMap else {
                throw Failure.unsupportedTexture("Convert bump maps to tangent-space normal maps before glTF export.")
            }
            let alpha: GLTFExportMaterial.Alpha
            switch material.alphaMode {
            case .opaque: alpha = .opaque
            case .mask(let cutoff): alpha = .mask(cutoff)
            case .blend: alpha = .blend
            }
            _ = try builder.material(GLTFExportMaterial(
                name: material.name, baseColorFactor: material.baseColorFactor,
                metallicFactor: material.metallicFactor, roughnessFactor: material.roughnessFactor,
                baseColorTexture: try texture(material.baseColorTexture),
                metallicRoughnessTexture: try texture(material.metallicRoughnessTexture),
                normalTexture: try texture(material.normalTexture), normalScale: material.normalScale,
                occlusionTexture: try texture(material.occlusionTexture), occlusionStrength: material.occlusionStrength,
                emissiveTexture: try texture(material.emissiveTexture), emissiveFactor: material.emissiveFactor,
                alpha: alpha, isDoubleSided: material.isDoubleSided,
                clearcoat: try material.clearcoat.map {
                    .init(
                        factor: $0.factor, roughness: $0.roughness, factorTexture: try texture($0.factorTexture),
                        roughnessTexture: try texture($0.roughnessTexture), normalTexture: try texture($0.normalTexture),
                        normalScale: $0.normalScale
                    )
                },
                transmission: try material.transmission.map { .init(factor: $0.factor, texture: try texture($0.factorTexture)) },
                volume: try material.volume.map {
                    .init(
                        thickness: $0.thickness, thicknessTexture: try texture($0.thicknessTexture),
                        attenuationDistance: $0.attenuationDistance.isFinite ? $0.attenuationDistance : nil,
                        attenuationColor: $0.attenuationColor
                    )
                },
                indexOfRefraction: material.indexOfRefraction,
                emissiveStrength: material.emissiveStrength,
                specular: try material.specular.map {
                    .init(
                        factor: $0.factor, factorTexture: try texture($0.factorTexture),
                        colorFactor: $0.colorFactor, colorTexture: try texture($0.colorTexture)
                    )
                },
                sheen: try material.sheen.map {
                    .init(
                        colorFactor: $0.colorFactor, colorTexture: try texture($0.colorTexture),
                        roughness: $0.roughness, roughnessTexture: try texture($0.roughnessTexture)
                    )
                },
                anisotropy: try material.anisotropy.map {
                    .init(strength: $0.strength, rotation: $0.rotation, texture: try texture($0.texture))
                },
                iridescence: try material.iridescence.map {
                    .init(
                        factor: $0.factor, factorTexture: try texture($0.factorTexture),
                        indexOfRefraction: $0.indexOfRefraction, minimumThickness: $0.minimumThickness,
                        maximumThickness: $0.maximumThickness, thicknessTexture: try texture($0.thicknessTexture)
                    )
                },
                dispersion: material.dispersion, isUnlit: material.isUnlit,
                diffuseTransmission: try material.diffuseTransmission.map {
                    .init(
                        factor: $0.factor, factorTexture: try texture($0.factorTexture),
                        colorFactor: $0.colorFactor, colorTexture: try texture($0.colorTexture)
                    )
                },
                diffuseRoughness: try material.diffuseRoughness.map {
                    .init(factor: $0.factor, texture: try texture($0.texture))
                },
                volumeScatter: try material.volumeScatter.map {
                    .init(
                        colorFactor: $0.multiscatterColorFactor, colorTexture: try texture($0.multiscatterColorTexture),
                        anisotropy: $0.scatterAnisotropy
                    )
                }
            ), deduplicate: false)
        }

        builder.setMaterialVariants(scene.materialVariants)
        if let selected = scene.selectedMaterialVariant { builder.warn(.materialVariantSelectionNotStored(variant: selected)) }
        for (meshIndex, mesh) in scene.meshes.enumerated() {
            // The authored materials, which a selected variant may have replaced.
            let defaults = mesh.defaultPrimitiveMaterialIndices ?? mesh.primitiveMaterialIndices
            guard defaults.count == mesh.primitives.count,
                  mesh.primitiveSkinInfluences.count == mesh.primitives.count,
                  mesh.primitiveMaterialVariants.isEmpty || mesh.primitiveMaterialVariants.count == mesh.primitives.count else {
                throw Failure.invalidData("Mesh \(mesh.sourceIndex) has mismatched primitive arrays.")
            }
            if mesh.morphTargets != nil { builder.warn(.morphTargetsDropped(mesh: meshIndex)) }
            var primitives: [[String: Any]] = []
            for (primitiveIndex, geometry) in mesh.primitives.enumerated() {
                var primitive = try builder.primitive(
                    geometry, influences: mesh.primitiveSkinInfluences[primitiveIndex], mesh: meshIndex, primitive: primitiveIndex
                )
                if let source = defaults[primitiveIndex] {
                    if let material = materials[source] {
                        primitive["material"] = material
                    } else {
                        builder.warn(.materialNotLoaded(mesh: meshIndex, primitive: primitiveIndex, sourceMaterial: source))
                    }
                }
                if !mesh.primitiveMaterialVariants.isEmpty {
                    let mapped = try mesh.primitiveMaterialVariants[primitiveIndex].mapValues {
                        try lookup(materials, $0, "Variant material")
                    }
                    primitive["extensions"] = try builder.variantMappings(mapped)
                }
                primitives.append(primitive)
            }
            builder.addMesh(name: mesh.name, primitives: primitives)
        }

        for skin in scene.skins {
            try builder.addSkin(
                name: skin.name, joints: skin.jointNodeIndices.map { try lookup(nodes, $0, "Skin joint node") },
                inverseBindMatrices: skin.inverseBindMatrices,
                skeleton: try skin.skeletonRootNodeIndex.map { try lookup(nodes, $0, "Skeleton node") }
            )
        }
        for light in scene.lights { try builder.addLight(light.light, name: light.name) }
        for camera in scene.cameras { try builder.addCamera(camera.projection, name: camera.name) }

        for node in scene.nodes {
            try builder.addNode(
                name: node.name, transform: node.localTransform, context: "node \(node.sourceIndex)",
                children: node.childIndices.map { try lookup(nodes, $0, "Child node") },
                mesh: try node.meshIndex.map { try lookup(meshes, $0, "Mesh") },
                skin: try node.skinIndex.map { try lookup(skins, $0, "Skin") },
                light: try node.lightIndex.map { try lookup(lights, $0, "Light") },
                camera: try node.cameraIndex.map { try lookup(cameras, $0, "Camera") },
                instances: node.instances
            )
        }

        for clip in scene.animations {
            if !clip.pointerChannels.isEmpty {
                builder.warn(.animationPointersDropped(sourceAnimation: clip.sourceIndex, pointers: clip.pointerChannels.map(\.pointer)))
            }
            var channels: [GLTFExportBuilder.Channel] = []
            // Weight channels animate morph targets, which are not exported; their meshes add a warning.
            for channel in clip.channels where channel.target != .weights {
                let width = channel.target == .rotation ? 4 : 3
                let values = channel.values.flatMap { value in (0..<width).map { value[$0] } }
                channels.append(.init(
                    node: try lookup(nodes, channel.sourceNodeIndex, "Animated node"), path: channel.target.rawValue,
                    times: channel.times, values: values, components: width, interpolation: channel.interpolation.rawValue
                ))
            }
            try builder.addAnimation(name: clip.name, channels: channels)
        }

        let roots = try scene.rootNodeIndices.map { try lookup(nodes, $0, "Root node") }
        let data = try builder.encode(sceneName: scene.name, roots: roots)
        return Output(
            data: data, format: format, warnings: builder.warnings, nodeIndices: [:], sourceNodeIndices: nodes,
            cameraNodeIndex: nil
        )
    }

    /// Writes a scene graph's hierarchy as glTF nodes in depth-first order.
    ///
    /// Each node's mesh is exported from the geometry its renderer draws, including a surface
    /// graph's vertex offset evaluated with the current uniforms. `.physicallyBased` materials
    /// export their metallic-roughness factors and maps, alpha mode, double-sidedness, and
    /// clearcoat, transmission, volume, IOR, emissive strength, specular, sheen, anisotropy,
    /// iridescence, dispersion, and unlit extensions. Map transforms become KHR_texture_transform.
    /// Without a surface graph, a node's material parameters tint the base color factor exactly
    /// as the raster shader does, so one renderer drawn with several tints produces several
    /// materials. Textures become PNG images: color maps sRGB-encoded and data maps linear,
    /// matching `GLTFMeshLoader`.
    ///
    /// A node's `InstanceSet` becomes EXT_mesh_gpu_instancing TRS accessors; tints are dropped
    /// with a warning, and a sheared instance matrix throws `Failure.invalidTransform`. A node
    /// with levels of detail exports its finest level. Morph targets are not exported.
    ///
    /// Renderer-only settings such as the material's built-in light, environment, shadow
    /// flags, `castsShadow`, and culling have no glTF form and are not exported.
    ///
    /// - Parameters:
    ///   - nodeNames: Optional glTF node names keyed by node ID; nodes not listed use `SceneNode.name`.
    ///   - animations: Transform clips. Every track must target a node in the graph.
    ///   - includesCamera: Adds a root node named "Camera" carrying the graph's perspective
    ///     camera at its camera transform, with no fixed aspect ratio.
    ///   - onlyVisible: Skips nodes whose `isVisible` is false, with their subtrees, like
    ///     Three.js `GLTFExporter`'s `onlyVisible` option; animation tracks that target them are
    ///     dropped with a warning. Layers are a camera filter and do not affect export.
    /// - Throws: `SceneGraph.ValidationError` for an invalid graph, `MeshRenderer.Failure` when a
    ///   deformation cannot run, or `GLTFExporter.Failure`.
    public func export(
        _ graph: SceneGraph, sceneName: String? = nil, nodeNames: [UUID: String] = [:],
        animations: [Animation] = [], includesCamera: Bool = true, onlyVisible: Bool = true
    ) throws -> Output {
        var lowering = GLTFSceneGraphExport(exporter: self)
        return try lowering.export(
            graph, sceneName: sceneName, nodeNames: nodeNames, animations: animations, includesCamera: includesCamera,
            onlyVisible: onlyVisible
        )
    }
}
