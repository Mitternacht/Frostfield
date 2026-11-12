import Foundation
import Metal
import simd

/// Lowers one scene graph into glTF records, sharing meshes, materials, images, and lights.
struct GLTFSceneGraphExport {
    private typealias Failure = GLTFExporter.Failure
    private typealias TextureReference = GLTFExportMaterial.TextureReference

    /// Frostfield's default PBR sampler: linear filtering, linear mip blending, and repeat wrapping.
    private static let defaultSampling = GLTFScene.Sampling(
        magnification: .linear, minification: .linearMipmapLinear, wrapS: .repeat, wrapT: .repeat
    )

    private struct ImageKey: Hashable {
        let texture: ObjectIdentifier
        let color: Bool
    }

    private struct MeshKey: Hashable {
        let renderer: ObjectIdentifier
        let material: Int?
    }

    private let format: GLTFExporter.Format
    private var builder: GLTFExportBuilder
    private var reader: GLTFTextureReader
    private var images: [ImageKey: Int] = [:]
    private var primitives: [ObjectIdentifier: [String: Any]] = [:]
    private var meshes: [MeshKey: Int] = [:]
    private var lights: [(light: PunctualLight, index: Int)] = []

    init(exporter: GLTFExporter) {
        format = exporter.format
        builder = GLTFExportBuilder(format: exporter.format)
        reader = GLTFTextureReader(queue: exporter.commandQueue)
    }

    mutating func export(
        _ graph: SceneGraph, sceneName: String?, nodeNames: [UUID: String],
        animations: [GLTFExporter.Animation], includesCamera: Bool, onlyVisible: Bool = true
    ) throws -> GLTFExporter.Output {
        // Rejects duplicate IDs, invalid transforms, and too many lights before writing anything.
        _ = try graph.resolve()

        // Number nodes depth-first so parents precede their children, keeping sibling order.
        var order: [(node: SceneNode, index: Int, payload: Int)] = []
        var children: [[Int]] = []
        var roots: [Int] = []
        // Hidden subtrees are skipped when exporting only visible nodes, like Three.js.
        func exported(_ node: SceneNode) -> Bool { !onlyVisible || node.isVisible }
        var pending = graph.roots.reversed().filter(exported).map { (node: $0, parent: Int?.none) }
        while let (node, parent) = pending.popLast() {
            let index = children.count
            let payload = node.pivot == nil ? index : index + 1
            order.append((node, index, payload))
            if payload != index { children.append([payload]) }
            children.append([])
            if let parent { children[parent].append(index) } else { roots.append(index) }
            pending.append(contentsOf: node.children.reversed().filter(exported).map { ($0, payload) })
        }

        var nodeIndices: [UUID: Int] = [:]
        var pivots: [UUID: SIMD3<Float>] = [:]
        for (node, index, payload) in order {
            guard node.batch == nil else { throw Failure.invalidData("Node \(node.id) uses a batched mesh.") }
            guard node.orientationMode == .inherited else {
                throw Failure.invalidData("Node \(node.id) uses world orientation, which glTF cannot retain.")
            }
            nodeIndices[node.id] = index
            var renderer = node.mesh
            if renderer == nil, let levels = node.levelsOfDetail {
                renderer = levels.levels[0].mesh
                if levels.levels.count > 1 { builder.warn(.levelsOfDetailReduced(node: payload, levelCount: levels.levels.count)) }
            }
            var instances: [SceneTransform]?
            if renderer != nil, let set = node.instances {
                if set.count == 0 {
                    // An empty set draws nothing, and EXT_mesh_gpu_instancing needs at least one instance.
                    builder.warn(.emptyInstancesDropped(node: payload))
                    renderer = nil
                } else {
                    instances = try set.transforms.enumerated().map { instance, matrix in
                        guard let transform = Self.decompose(matrix) else {
                            throw Failure.invalidTransform("node \(node.id) instance \(instance)")
                        }
                        return transform
                    }
                    if set.tints.contains(where: { $0 != SIMD4(repeating: 1) }) { builder.warn(.instanceTintsDropped(node: payload)) }
                }
            }
            let mesh = try renderer.map { try self.mesh($0, parameters: node.materialParameters) }
            let light = try node.light.flatMap { try self.light($0) }
            let name = nodeNames[node.id] ?? node.name
            let context = "node \(node.id)"
            if let pivot = node.pivot {
                // T(position + pivot) × R × S × T(-pivot). Animations target
                // the container, so interpolation cannot move the rotation center.
                pivots[node.id] = pivot
                var container = node.transform
                container.translation += pivot
                try builder.addNode(name: name, transform: container, context: context, children: children[index])
                try builder.addNode(
                    name: nil, transform: .init(translation: -pivot), context: context,
                    children: children[payload], mesh: mesh, light: light, instances: instances
                )
            } else {
                try builder.addNode(
                    name: name, transform: node.transform, context: context,
                    children: children[index], mesh: mesh, light: light, instances: instances
                )
            }
        }

        var cameraNode: Int?
        if includesCamera {
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
            let camera = try builder.addCamera(projection, name: nil)
            let index = try builder.addNode(name: "Camera", transform: graph.cameraTransform, context: "camera", camera: camera)
            roots.append(index)
            cameraNode = index
        }

        for clip in animations {
            var channels: [GLTFExportBuilder.Channel] = []
            for track in clip.animation.tracks {
                guard let node = nodeIndices[track.nodeID] else {
                    guard onlyVisible, graph.node(withID: track.nodeID) != nil else {
                        throw Failure.missingAnimationTarget(track.nodeID)
                    }
                    builder.warn(.hiddenAnimationTargetDropped(track.nodeID))
                    continue
                }
                let times = track.keyframes.map(\.time)
                let poses = track.keyframes.map(\.transform)
                let pivot = pivots[track.nodeID]
                channels += [
                    .init(
                        node: node, path: "translation", times: times,
                        values: poses.flatMap { pose in
                            let translation = pivot.map { pose.translation + $0 } ?? pose.translation
                            return [translation.x, translation.y, translation.z]
                        },
                        components: 3, interpolation: "LINEAR"
                    ),
                    .init(
                        node: node, path: "rotation", times: times,
                        values: poses.flatMap { [$0.rotation.vector.x, $0.rotation.vector.y, $0.rotation.vector.z, $0.rotation.vector.w] },
                        components: 4, interpolation: "LINEAR"
                    ),
                    .init(
                        node: node, path: "scale", times: times,
                        values: poses.flatMap { [$0.scale.x, $0.scale.y, $0.scale.z] },
                        components: 3, interpolation: "LINEAR"
                    ),
                ]
            }
            try builder.addAnimation(name: clip.name, channels: channels)
        }

        let data = try builder.encode(sceneName: sceneName, roots: roots)
        return GLTFExporter.Output(
            data: data, format: format, warnings: builder.warnings, nodeIndices: nodeIndices,
            sourceNodeIndices: [:], cameraNodeIndex: cameraNode
        )
    }

    /// One glTF mesh per renderer and exported material. Tinted copies share vertex data.
    private mutating func mesh(_ renderer: MeshRenderer, parameters: SIMD4<Float>) throws -> Int {
        let material = renderer.pathTracingMaterial
        let materialIndex = try material.map { try self.material($0, parameters: parameters) }
        let key = MeshKey(renderer: ObjectIdentifier(renderer), material: materialIndex)
        if let existing = meshes[key] { return existing }

        let meshIndex = builder.meshCount
        let deforms = material?.surfaceGraph?.hasVertexOutputs == true
        var primitive: [String: Any]
        if let cached = primitives[ObjectIdentifier(renderer)] {
            primitive = cached
        } else {
            // Export what draws use: a vertex graph's offset is evaluated with its current uniforms.
            let geometry = deforms ? try renderer.deformedGeometry(queue: reader.queue(for: renderer.device)) : renderer.geometry
            primitive = try builder.primitive(geometry, influences: nil, mesh: meshIndex, primitive: 0)
            primitives[ObjectIdentifier(renderer)] = primitive
        }
        if material == nil { builder.warn(.unsupportedMaterial(mesh: meshIndex)) }
        if deforms { builder.warn(.deformationBaked(mesh: meshIndex)) }
        if renderer.skin != nil { builder.warn(.skinDropped(mesh: meshIndex)) }
        if renderer.morphTargets != nil { builder.warn(.morphTargetsDropped(mesh: meshIndex)) }
        primitive["material"] = materialIndex
        builder.addMesh(name: nil, primitives: [primitive])
        meshes[key] = meshIndex
        return meshIndex
    }

    private mutating func material(_ material: PhysicallyBasedMaterial, parameters: SIMD4<Float>) throws -> Int {
        guard (0..<4).allSatisfy({ parameters[$0].isFinite }) else {
            throw Failure.invalidData("Scene graph material parameters must be finite.")
        }
        let graph = material.surfaceGraph
        // Without a graph, the raster shader multiplies the per-draw tint into the base color factor.
        let baseColor = graph == nil
            ? simd_clamp(parameters * material.baseColorFactor, .zero, SIMD4(repeating: 1)) : material.baseColorFactor
        guard !material.hasActiveBumpMap else {
            throw Failure.unsupportedTexture("Convert bump maps to tangent-space normal maps before glTF export.")
        }
        let maps = material.coreMaps
        guard material.objectNormalMaps == 0 else {
            throw Failure.unsupportedTexture("Convert object-space normal maps to tangent space before glTF export.")
        }
        guard material.baseNormalEncoding == .rgb, material.coatNormalEncoding == .rgb else {
            throw Failure.unsupportedTexture("Convert packed tangent-space normals to RGB before glTF export.")
        }
        var customSampler = false
        func reference(_ binding: PhysicallyBasedMaterial.TextureBinding?, color: Bool) throws -> TextureReference? {
            guard let binding else { return nil }
            if binding.sampler != nil, binding.sampling == nil { customSampler = true }
            return try texture(binding, color: color)
        }

        let alpha: GLTFExportMaterial.Alpha
        switch material.alphaMode {
        case .materialAlpha, .opaque: alpha = .opaque
        case .mask(let cutoff): alpha = .mask(cutoff)
        case .blend: alpha = .blend
        }

        var transmission: GLTFExportMaterial.Transmission?
        var volume: GLTFExportMaterial.Volume?
        if let optical = material.transmission {
            let hasVolume = optical.thickness > 0 || optical.thicknessTexture != nil
                || optical.attenuationDistance != nil || optical.attenuationColor != SIMD3(repeating: 1)
            // A zero factor with no volume leaves only the IOR, which KHR_materials_ior carries alone.
            if optical.factor > 0 || optical.factorTexture != nil || (hasVolume && material.diffuseTransmission == nil) {
                transmission = .init(factor: optical.factor, texture: try reference(optical.factorTexture, color: false))
            }
            if hasVolume {
                volume = .init(
                    thickness: optical.thickness, thicknessTexture: try reference(optical.thicknessTexture, color: false),
                    attenuationDistance: optical.attenuationDistance, attenuationColor: optical.attenuationColor
                )
            }
        }

        // The material stores the emissive factor multiplied by its strength.
        let strength = material.emissiveStrength
        let emission = material.pathTracingEmission
        let emissiveFactor = strength == 1 || strength == 0
            ? emission : simd_clamp(emission / strength, .zero, SIMD3(repeating: 1))

        let exported = GLTFExportMaterial(
            name: nil, baseColorFactor: baseColor,
            metallicFactor: material.settings.y, roughnessFactor: material.settings.x,
            baseColorTexture: try reference(maps.bindings[0], color: true),
            metallicRoughnessTexture: try reference(maps.bindings[1], color: false),
            normalTexture: try reference(maps.bindings[2], color: false),
            normalScale: maps.bindings[2] == nil ? 1 : maps.normalScale,
            occlusionTexture: try reference(maps.bindings[3], color: false),
            occlusionStrength: maps.bindings[3] == nil ? 1 : maps.occlusionStrength,
            emissiveTexture: try reference(maps.bindings[4], color: true),
            emissiveFactor: emissiveFactor,
            alpha: alpha, isDoubleSided: material.isDoubleSided,
            clearcoat: try material.clearcoat.map { coat in
                .init(
                    factor: coat.factor, roughness: coat.roughness,
                    factorTexture: try reference(coat.factorTexture, color: false),
                    roughnessTexture: try reference(coat.roughnessTexture, color: false),
                    normalTexture: try reference(coat.normalTexture, color: false),
                    normalScale: coat.normalTexture == nil ? 1 : coat.normalScale
                )
            },
            transmission: transmission, volume: volume,
            indexOfRefraction: material.transmission?.indexOfRefraction ?? 1.5,
            emissiveStrength: strength,
            specular: try material.specular.map { layer in
                .init(
                    factor: layer.factor, factorTexture: try reference(layer.factorTexture, color: false),
                    colorFactor: layer.colorFactor, colorTexture: try reference(layer.colorTexture, color: true)
                )
            },
            sheen: try material.sheen.map { layer in
                .init(
                    colorFactor: layer.colorFactor, colorTexture: try reference(layer.colorTexture, color: true),
                    roughness: layer.roughness, roughnessTexture: try reference(layer.roughnessTexture, color: false)
                )
            },
            anisotropy: try material.anisotropy.map { layer in
                .init(strength: layer.strength, rotation: layer.rotation, texture: try reference(layer.texture, color: false))
            },
            iridescence: try material.iridescence.map { layer in
                .init(
                    factor: layer.factor, factorTexture: try reference(layer.factorTexture, color: false),
                    indexOfRefraction: layer.indexOfRefraction, minimumThickness: layer.minimumThickness,
                    maximumThickness: layer.maximumThickness, thicknessTexture: try reference(layer.thicknessTexture, color: false)
                )
            },
            dispersion: material.transmission?.dispersion ?? 0,
            isUnlit: material.shadingModel == .unlit,
            diffuseTransmission: try material.diffuseTransmission.map { layer in
                .init(
                    factor: layer.factor, factorTexture: try reference(layer.factorTexture, color: false),
                    colorFactor: layer.colorFactor, colorTexture: try reference(layer.colorTexture, color: true)
                )
            },
            diffuseRoughness: try material.diffuseRoughness.map { layer in
                .init(factor: layer.factor, texture: try reference(layer.texture, color: false))
            },
            volumeScatter: try material.volumeScatter.map { layer in
                .init(
                    colorFactor: layer.multiscatterColor, colorTexture: try reference(layer.multiscatterColorTexture, color: true),
                    anisotropy: layer.anisotropy
                )
            }
        )
        let index = try builder.material(exported, deduplicate: true)
        if let graph {
            let dropped = graph.outputs.connected.map(\.0).filter { $0 != .positionOffset }
            if !dropped.isEmpty { builder.warn(.surfaceGraphChannelsDropped(material: index, channels: dropped)) }
        }
        if customSampler { builder.warn(.samplerUnavailable(material: index)) }
        var droppedLayers: [String] = []
        if material.hasSubsurface { droppedLayers.append("subsurface scattering") }
        if material.lightMap != nil { droppedLayers.append("light map") }
        if material.hasRetroreflection { droppedLayers.append("retroreflection") }
        if !droppedLayers.isEmpty { builder.warn(.materialLayersDropped(material: index, layers: droppedLayers)) }
        if material.alphaMode == .materialAlpha, baseColor.w < 1 { builder.warn(.alphaExportedAsOpaque(material: index)) }
        return index
    }

    /// Encodes each texture once per color convention and shares identical PNG bytes.
    private mutating func texture(_ binding: PhysicallyBasedMaterial.TextureBinding, color: Bool) throws -> TextureReference {
        let key = ImageKey(texture: ObjectIdentifier(binding.texture), color: color)
        let image: Int
        if let existing = images[key] {
            image = existing
        } else {
            let data = try reader.png(binding.texture, color: color)
            image = try builder.image(data, mimeType: "image/png", name: binding.texture.label)
            images[key] = image
        }
        let sampler = builder.sampler(binding.sampling.map(Self.sampling) ?? Self.defaultSampling)
        return TextureReference(
            index: builder.texture(image: image, sampler: sampler), coordinates: binding.textureCoordinateSet,
            transform: binding.transform
        )
    }

    /// The glTF sampler for a binding's sampling description.
    private static func sampling(_ description: PhysicallyBasedMaterial.TextureSampling) -> GLTFScene.Sampling {
        func wrap(_ mode: PhysicallyBasedMaterial.TextureSampling.Addressing) -> GLTFScene.Sampling.Wrap {
            switch mode {
            case .repeat: .repeat
            case .clampToEdge: .clampToEdge
            case .mirroredRepeat: .mirroredRepeat
            }
        }
        let minification: GLTFScene.Sampling.Minification = switch (description.minification, description.mipFilter) {
        case (.nearest, .notMipmapped): .nearest
        case (.linear, .notMipmapped): .linear
        case (.nearest, .nearest): .nearestMipmapNearest
        case (.linear, .nearest): .linearMipmapNearest
        case (.nearest, .linear): .nearestMipmapLinear
        case (.linear, .linear): .linearMipmapLinear
        }
        return GLTFScene.Sampling(
            magnification: description.magnification == .nearest ? .nearest : .linear, minification: minification,
            wrapS: wrap(description.addressU), wrapT: wrap(description.addressV)
        )
    }

    /// Splits an affine instance matrix into TRS, as `GLTFMeshLoader` does for node matrices.
    /// A mirrored matrix gets a negative X scale. Nil for a sheared or projective matrix,
    /// which EXT_mesh_gpu_instancing cannot represent.
    private static func decompose(_ matrix: simd_float4x4) -> SceneTransform? {
        let (c0, c1, c2, c3) = matrix.columns
        guard c0.w == 0, c1.w == 0, c2.w == 0, c3.w == 1 else { return nil }
        func double(_ column: SIMD4<Float>) -> SIMD3<Double> { SIMD3(Double(column.x), Double(column.y), Double(column.z)) }
        let x = double(c0), y = double(c1), z = double(c2)
        let lengths = SIMD3(simd_length(x), simd_length(y), simd_length(z))
        guard lengths.x > 0, lengths.y > 0, lengths.z > 0, lengths.x.isFinite, lengths.y.isFinite, lengths.z.isFinite else {
            return nil
        }
        var axisX = x / lengths.x
        let axisY = y / lengths.y, axisZ = z / lengths.z
        let tolerance = 0.0001
        let determinant = simd_dot(simd_cross(axisX, axisY), axisZ)
        guard abs(simd_dot(axisX, axisY)) <= tolerance, abs(simd_dot(axisX, axisZ)) <= tolerance,
              abs(simd_dot(axisY, axisZ)) <= tolerance, abs(abs(determinant) - 1) <= tolerance else {
            return nil
        }
        var scaleX = lengths.x
        if determinant < 0 {
            axisX = -axisX
            scaleX = -scaleX
        }
        func float(_ value: SIMD3<Double>) -> SIMD3<Float> { SIMD3(Float(value.x), Float(value.y), Float(value.z)) }
        let basis = simd_float3x3(columns: (float(axisX), float(axisY), float(axisZ)))
        return SceneTransform(
            translation: SIMD3(c3.x, c3.y, c3.z), rotation: simd_normalize(simd_quatf(basis)),
            scale: SIMD3(Float(scaleX), Float(lengths.y), Float(lengths.z))
        )
    }

    // KHR_lights_punctual has no area lights; rectangles are reported and omitted.
    private mutating func light(_ light: PunctualLight) throws -> Int? {
        if case .rectangle = light.kind {
            builder.warn(.rectangleLightDropped)
            return nil
        }
        if let existing = lights.first(where: { $0.light == light }) { return existing.index }
        if light.iesProfile != nil || light.cookie != nil { builder.warn(.lightProfileDropped) }
        let index = try builder.addLight(light, name: nil)
        lights.append((light, index))
        return index
    }
}
