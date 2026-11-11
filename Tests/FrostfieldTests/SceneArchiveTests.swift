import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Lossless scene archives: rendered round trips, idempotent capture, files and validation.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneArchiveTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    @Test("Matrix uniform columns survive an archive and the restored draw")
    func matrixUniforms() throws {
        let model = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(0.05, 0.1, 0.2, 1)
        ))
        let point = SIMD4<Float>(0.2, 0.1, 0.05, 1)
        let graph = try SurfaceShaderGraph { s in
            let transformed = s.uniform("Model", simd_float4x4(1)) * s.vector(
                s.constant(SIMD3(point.x, point.y, point.z)), s.constant(point.w)
            )
            return .init(emission: transformed.xyz * s.uniform("Gain", Float(1)))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        try uniforms.set("Model", to: model)
        try uniforms.set("Gain", to: 2)
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph, surfaceUniforms: uniforms
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(material)
        )
        let scene = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: scene, commandQueue: fixture.queue)
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        guard case .physicallyBased(let restoredMaterial) = loaded.roots.first?.mesh?.material else {
            Issue.record("Expected a physically based material.")
            return
        }
        let restored = try #require(restoredMaterial.surfaceUniforms)
        #expect(restored.matrix4(named: "Model") == model)
        #expect(restored.value(named: "Gain") == SIMD4(2, 0, 0, 0))
        let original = try fixture.render(scene)
        let expected = SIMD3((model * point).x, (model * point).y, (model * point).z) * 2
        let center = fixture.pixel(original.pixels, x: fixture.side / 2, y: fixture.side / 2)
        #expect(abs(center.x - expected.x) < 0.004 && abs(center.y - expected.y) < 0.004 && abs(center.z - expected.z) < 0.004,
                "\(center) vs \(expected)")
        #expect(try fixture.render(loaded).pixels == original.pixels)
    }

    @Test("Integer and boolean uniforms survive an archive, including values whose bits are NaN as floats")
    func integerUniforms() throws {
        let graph = try SurfaceShaderGraph { s in
            let scale = s.uniform("K", Int32(4)).toFloat()
            let picked = s.branch(s.uniform("On", false)) { s.constant(Float(1)) } else: { s.constant(Float(0)) }
            return .init(emission: s.vector(scale, picked, s.constant(Float(0))))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        try uniforms.setInteger("K", Int32(-2))
        try uniforms.setBoolean("On", true)
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph, surfaceUniforms: uniforms
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(material)
        )
        let scene = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: scene, commandQueue: fixture.queue)
        let manifest = try #require(String(data: archive.manifest, encoding: .utf8))
        #expect(manifest.contains("\"integerValues\""))
        #expect(manifest.contains("\"integerValue\" : 4"))
        #expect(!manifest.contains("NaN"))
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        guard case .physicallyBased(let restoredMaterial) = loaded.roots.first?.mesh?.material else {
            Issue.record("Expected a physically based material.")
            return
        }
        let restored = try #require(restoredMaterial.surfaceUniforms)
        #expect(restored.integer(named: "K") == -2)
        #expect(restored.boolean(named: "On") == true)
        #expect(try SceneArchive(capturing: loaded, commandQueue: fixture.queue) == archive)
    }

    /// A scene that exercises every archived feature that draws.
    private func richScene() throws -> (SceneGraph, [AnimationClip], LightRig) {
        let device = fixture.device
        let surface = try ProceduralSurface(device: device, style: .wood, resolution: 32, seed: 7)
        let sampler = try TextureSampling(
            magnification: .nearest, mipFilter: .nearest, addressU: .mirrorRepeat, addressV: .clampToEdge, maxAnisotropy: 4
        ).makeSamplerState(device: device)
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(0.1, 0.2), rotation: 0.3, scale: SIMD2(2, 1.5))
        let normal = PhysicallyBasedMaterial.TextureBinding(texture: surface.normalTexture, transform: transform)
        let data = PhysicallyBasedMaterial.TextureBinding(texture: surface.metallicRoughnessTexture, sampler: sampler)

        // A small HDR panorama with a vertical gradient for image-based lighting.
        let width = 32, height = 16
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(device.makeTexture(descriptor: descriptor))
        let texels = (0..<(width * height)).map { index in
            let v = Float(index / width) / Float(height)
            return SIMD4<Float16>(Float16(1.5 - v), Float16(0.6 + 0.3 * v), Float16(0.2 + v), 1)
        }
        texels.withUnsafeBytes { bytes in
            panorama.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: width * 8)
        }
        let lighting = try EnvironmentLighting(
            queue: fixture.queue, equirectangular: panorama, settings: .init(cubeSize: 16, diffuseSize: 8, brdfSize: 16, sampleCount: 32)
        )

        let layered = try PhysicallyBasedMaterial(
            baseColorTexture: surface.baseColorTexture, metallic: 0.3, roughness: 0.45, directionToLight: SIMD3(0.3, 2, 1),
            lightIntensity: 1.7, ambientIntensity: 0.05, baseColorFactor: SIMD4(0.9, 0.85, 0.8, 1), baseColorSampler: sampler,
            metallicRoughnessTexture: data, normalTexture: normal, normalScale: 0.8,
            occlusionTexture: data, occlusionStrength: 0.6, emissiveFactor: SIMD3(0.1, 0.05, 0), alphaMode: .opaque,
            environment: try .init(lighting: lighting, intensity: 0.7, yaw: 0.4),
            clearcoat: try .init(factor: 0.6, roughness: 0.2, normalTexture: normal, normalScale: 0.5),
            baseColorTextureTransform: transform, emissiveStrength: 3
        )
        // The raster shader's sampler budget does not fit environment, clearcoat and layers together.
        let layers = try PhysicallyBasedMaterial(
            metallic: 0.2, roughness: 0.35, normalTexture: normal,
            specular: try .init(factor: 0.8, colorFactor: SIMD3(1, 0.9, 0.8), colorTexture: .init(texture: surface.baseColorTexture)),
            sheen: try .init(colorFactor: SIMD3(0.3, 0.2, 0.4), roughness: 0.5, roughnessTexture: data),
            anisotropy: try .init(strength: 0.4, rotation: 0.6),
            iridescence: try .init(factor: 0.5, indexOfRefraction: 1.4, minimumThickness: 150, maximumThickness: 450)
        )
        let sphere = try MeshGeometry.uvSphere(radius: 0.45, segments: 24, rings: 12)

        let graph = try SurfaceShaderGraph(
            nodes: [
                .input(.uv0), .sampleTexture(slot: 0, coordinates: 0), .swizzle(1, "rgb"), .uniform(0), .multiply(2, 3),
                .input(.worldPosition), .fractalNoise(.gradient, position: 5, octaves: 3, lacunarity: 2, gain: 0.5),
                .clamp01(6), .multiply(4, 6),
            ],
            uniforms: [ShaderUniform("tint", SIMD3(0.9, 0.5, 0.2))],
            textures: [ShaderTextureInput("pattern", content: .color, addressing: .mirroredRepeat)],
            labels: [4: "tinted"], outputs: .init(baseColor: 4, roughness: 7, emission: 8)
        )
        let uniforms = try ShaderUniforms(graph: graph)
        try uniforms.set("tint", to: SIMD3(0.2, 0.8, 0.4))
        let graphMaterial = try PhysicallyBasedMaterial(
            lightIntensity: 1, ambientIntensity: 0.1, surfaceGraph: graph, surfaceUniforms: uniforms,
            surfaceTextures: ["pattern": surface.baseColorTexture]
        )
        let glass = try PhysicallyBasedMaterial(
            roughness: 0.2, baseColorFactor: SIMD4(0.2, 0.6, 1, 0.5), alphaMode: .blend, isDoubleSided: true
        )
        let unlit = try PhysicallyBasedMaterial(baseColorFactor: SIMD4(1, 0.4, 0.1, 1), shadingModel: .unlit)

        func renderer(_ geometry: MeshGeometry, _ material: MeshMaterial, instancing: Bool = false,
                      skin: MeshSkin? = nil, morph: MeshMorphTargets? = nil) throws -> MeshRenderer {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material,
                skin: skin, morphTargets: morph, supportsInstancing: instancing
            )
        }
        let square = try GeometryRenderFixture.square(0.2)
        let instances = try InstanceSet(
            device: device, transforms: [translationMatrix(-0.3), translationMatrix(0.3, 0.1)],
            tints: [SIMD4(1, 0.2, 0.2, 1), SIMD4(0.2, 1, 0.2, 1)], capacity: 8
        )
        let morph = try MeshMorphTargets(vertexCount: 4, targets: [
            .init(positionOffsets: [SIMD3(0, 0.2, 0), .zero, .zero, SIMD3(-0.1, 0, 0)], normalOffsets: Array(repeating: SIMD3(0, 0.1, 0), count: 4)),
        ])
        let skin = try MeshSkin(jointCount: 2, influences: [
            .init(joints: SIMD4(0, 1, 0, 0), weights: SIMD4(0.5, 0.5, 0, 0)), .init(joints: .zero, weights: SIMD4(1, 0, 0, 0)),
            .init(joints: SIMD4(1, 0, 0, 0), weights: SIMD4(1, 0, 0, 0)), .init(joints: .zero, weights: SIMD4(1, 0, 0, 0)),
        ])
        let near = try renderer(square, .unlit)
        let far = try renderer(try GeometryRenderFixture.square(0.1), .lambert)

        let lamp = SceneNode(
            transform: SceneTransform(translation: SIMD3(0.5, 0.8, -1.2)),
            light: try PunctualLight(kind: .point(range: 6), color: SIMD3(1, 0.9, 0.7), intensity: 3), name: "lamp"
        )
        let spot = SceneNode(
            transform: try { var t = SceneTransform(translation: SIMD3(-1, 1, -1)); try t.look(at: SIMD3(0, 0, -2.5)); return t }(),
            light: try PunctualLight(kind: .spot(range: nil, innerConeAngle: 0.2, outerConeAngle: 0.6), intensity: 5)
        )
        let hidden = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1.5)), mesh: near, isVisible: false)
        let offLayer = SceneNode(
            transform: SceneTransform(translation: SIMD3(0.4, -0.4, -1.5)), mesh: near, layers: SceneLayers(layer: 5)
        )
        let roots: [SceneNode] = [
            SceneNode(
                transform: SceneTransform(translation: SIMD3(-0.55, 0.45, -2.2), rotation: simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0))),
                mesh: try renderer(sphere, .physicallyBased(layered)), name: "hero",
                metadata: ["sku": "W-42", "price": 12.5, "tags": ["wood", "coated"]]
            ),
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0.55, 0.45, -2.2)), mesh: try renderer(sphere, .physicallyBased(graphMaterial)),
                materialParameters: SIMD4(0.5, 0.25, 0, 1), renderOrder: 3
            ),
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, -0.5, -2)), mesh: try renderer(square, .unlit, instancing: true),
                instances: instances, name: "crowd"
            ),
            SceneNode(
                transform: SceneTransform(translation: SIMD3(-0.6, -0.55, -1.6)), mesh: try renderer(square, .unlit, morph: morph),
                materialParameters: SIMD4(0.3, 0.6, 1, 1), morphWeights: [0.75]
            ),
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0.6, -0.5, -1.6)), mesh: try renderer(square, .unlit, skin: skin),
                materialParameters: SIMD4(1, 1, 0.2, 1),
                jointPalette: try JointPalette(matrices: [translationMatrix(0, 0.05), translationMatrix(0.1)]),
                cullingBounds: try AxisAlignedBox(minimum: SIMD3(-0.5, -0.5, -0.1), maximum: SIMD3(0.5, 0.5, 0.1))
            ),
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0.6, -2)),
                children: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0.25, 0)), mesh: try renderer(square, .textured(surface.baseColorTexture)))],
                levelsOfDetail: try LevelOfDetail(levels: [.init(mesh: near, distance: 0), .init(mesh: far, distance: 1.5, hysteresis: 0.1)])
            ),
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1.4)), mesh: try renderer(square, .physicallyBased(glass))),
            SceneNode(transform: SceneTransform(translation: SIMD3(0.2, 0.1, -1.8)), mesh: try renderer(GeometryRenderFixture.square(0.1), .physicallyBased(unlit))),
            lamp, spot, hidden, offLayer,
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0.05, -3)), mesh: try renderer(sphere, .physicallyBased(layers)),
                name: "layers"
            ),
        ]
        var sceneGraph = SceneGraph(roots: roots, camera: fixture.camera)
        sceneGraph.environmentIntensity = 0.8
        sceneGraph.environmentRotation = 0.2
        sceneGraph.cameraTransform = SceneTransform(translation: SIMD3(0, 0, 0.1))

        let animation = try SceneAnimation(tracks: [
            SceneAnimationTrack(nodeID: lamp.id, keyframes: [
                TransformKeyframe(time: 0, transform: lamp.transform),
                TransformKeyframe(time: 1.5, transform: SceneTransform(translation: SIMD3(-0.5, 1, -1), scale: SIMD3(repeating: 2))),
            ]),
        ])
        let clip = AnimationClip(animation, name: "swing")
        let additive = try clip.makingAdditive(referenceTime: 0.25)
        let rig = try LightRig(lights: [
            .init(id: "key", kind: .spot(innerAngle: 0.3, outerAngle: 0.5, range: 9), position: SIMD3(2, 3, 1), target: .zero, emission: .power(800)),
            .init(id: "sun", kind: .directional(angularDiameter: 0.0093), position: SIMD3(0, 10, 0), target: .zero,
                  color: SIMD3(1, 0.95, 0.9), emission: .intensity(5), castsShadow: false),
        ])
        return (sceneGraph, [clip, additive], rig)
    }

    @Test("A captured scene reloads into renderers that draw identical pixels")
    func renderedRoundTrip() throws {
        let (graph, clips, rig) = try richScene()
        let original = try fixture.render(graph)
        #expect(original.statistics.drawnMeshCount >= 10)

        let archive = try SceneArchive(capturing: graph, animations: clips, lightRig: rig, commandQueue: fixture.queue)
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue)
        let reloaded = try fixture.render(loaded.graph)
        #expect(reloaded.statistics == original.statistics)
        #expect(reloaded.pixels == original.pixels)
        // The scene covers a tenth of the image, so equality compares real content.
        #expect(original.pixels.filter { $0.w > 0 }.count > fixture.side * fixture.side / 10)

        // Capturing the loaded scene reproduces the same bytes, so nothing was dropped or changed.
        let again = try SceneArchive(capturing: loaded.graph, animations: loaded.animations, lightRig: loaded.lightRig)
        #expect(again.manifest == archive.manifest)
        #expect(again.binary == archive.binary)

        // Node identity, names, metadata, layers and shared objects survive.
        let hero = try #require(loaded.graph.node(named: "hero"))
        #expect(hero.id == graph.roots[0].id)
        #expect(hero.metadata["tags"] == ["wood", "coated"])
        #expect(loaded.graph.roots[1].renderOrder == 3)
        #expect(loaded.graph.roots[10].isVisible == false)
        #expect(loaded.graph.roots[11].layers == SceneLayers(layer: 5))
        #expect(loaded.graph.roots[10].mesh === loaded.graph.roots[11].mesh)
        #expect(loaded.graph.roots[10].mesh === loaded.graph.roots[5].levelsOfDetail?.levels[0].mesh)
        #expect(loaded.graph.cameraLayers == graph.cameraLayers)
        #expect(loaded.lightRig == rig)

        // Animation clips sample to the same poses, including the additive reference.
        for (source, restored) in zip(clips, loaded.animations) {
            #expect(restored.name == source.name && restored.blendMode == source.blendMode && restored.duration == source.duration)
            for time in [Float(0), 0.4, 1.1, 1.5] {
                #expect(try restored.sample(at: time) == source.sample(at: time))
            }
        }
    }

    @Test("The archive preserves graph inputs, alpha sampling, vertex colors, groups and render state")
    func currentMeshRoundTrip() throws {
        let source = try GeometryRenderFixture.square(1)
        let vertices = source.vertices.enumerated().map { index, vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                tangent: vertex.tangent, color: index == 0 ? nil : SIMD4(0.8, 0.6, 0.3, 0.5)
            )
        }
        let groups = [MeshGeometryGroup(start: 0, count: 3, materialIndex: 2)]
        let geometry = try MeshGeometry(vertices: vertices, indices: source.indices).withGroups(groups)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 2, height: 2, mipmapped: false
        )
        descriptor.storageMode = .shared
        let alpha = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels: [UInt8] = [0, 255, 0, 255, 0, 64, 0, 255, 0, 192, 0, 255, 0, 128, 0, 255]
        texels.withUnsafeBytes {
            alpha.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 8)
        }
        let sampling = PhysicallyBasedMaterial.TextureSampling(
            addressU: .mirroredRepeat, addressV: .clampToEdge, magnification: .nearest,
            minification: .nearest, mipFilter: .notMipmapped
        )
        let surface = try SurfaceShaderGraph(
            nodes: [.input(.modelMatrix), .matrixColumn(0, 3), .swizzle(1, "x"), .hash(2),
                    .scalar(0.5), .branch(condition: 4, ifTrue: 3, ifFalse: 4), .rgb(red: 5, green: 5, blue: 5)],
            outputs: .init(emission: 6)
        )
        let pbr = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(0.6, 0.8, 0.3, 1), alphaMode: .blend, surfaceGraph: surface,
            alphaTexture: .init(texture: alpha, sampler: try sampling.makeSamplerState(device: fixture.device), sampling: sampling)
        )
        let state = MeshRenderState(blending: .additive, transparent: true, depthWrite: false, depthCompare: .lessEqual)
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(pbr),
            renderState: state
        )
        var graph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0.3, 0, -2)), mesh: renderer)], camera: fixture.camera
        )
        graph.projectionJitter = SIMD2(0.25, -0.25)
        let archive = try SceneArchive(capturing: graph, commandQueue: fixture.queue)
        let restored = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        let mesh = try #require(restored.roots.first?.mesh)
        #expect(restored.projectionJitter == graph.projectionJitter)
        #expect(mesh.geometry.groups == groups)
        #expect(mesh.geometry.vertices.map(\.color) == vertices.map(\.color))
        #expect(mesh.renderState == state)
        guard case .physicallyBased(let restoredPBR) = mesh.material else {
            Issue.record("Expected a PBR material.")
            return
        }
        #expect(restoredPBR.alphaTexture?.sampling == sampling)
        #expect(restoredPBR.surfaceGraph?.nodes == surface.nodes)
        let original = try fixture.render(graph)
        #expect(original.pixels.contains { $0.w > 0 && $0.w < 1 })
        #expect(try fixture.render(restored).pixels == original.pixels)
        #expect(try SceneArchive(capturing: restored, commandQueue: fixture.queue) == archive)
    }

    @Test("Translucent material layers retain values, map bindings and rendered color through an archive")
    func translucentMaterials() throws {
        let texture = try #require(fixture.device.makeTexture(descriptor: .texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
        )))
        let texels: [UInt8] = [127, 143, 255, 64]
        texels.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 4)
        }
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(0.1, 0.2), rotation: 0.3)
        let binding = PhysicallyBasedMaterial.TextureBinding(texture: texture, transform: transform)
        let volume = try PhysicallyBasedMaterial(
            transmission: .init(factor: 0, thickness: 0.5, attenuationDistance: 0.8),
            diffuseTransmission: .init(factor: 0.6, factorTexture: binding),
            diffuseRoughness: .init(factor: 0.7, texture: binding),
            volumeScatter: .init(multiscatterColor: SIMD3(0.2, 0.3, 0.5), anisotropy: 0.4, multiscatterColorTexture: binding)
        )
        let wax = try PhysicallyBasedMaterial(
            subsurface: .init(weight: 0.75, color: SIMD3(1, 0.4, 0.2), radius: SIMD3(1, 0.5, 0.25),
                              scale: 0.1, anisotropy: -0.3, colorTexture: binding)
        )
        let nodes = try [volume, wax].enumerated().map { index, material in
            let renderer = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float,
                geometry: .cube(), material: .physicallyBased(material)
            )
            return SceneNode(transform: SceneTransform(translation: SIMD3(Float(index) - 0.5, 0, -2)), mesh: renderer)
        }
        let original = SceneGraph(roots: nodes, camera: fixture.camera)
        let archive = try SceneArchive(capturing: original, commandQueue: fixture.queue)
        let graph = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        let materials = graph.roots.compactMap { node -> PhysicallyBasedMaterial? in
            if case .physicallyBased(let material)? = node.mesh?.material { return material }
            return nil
        }
        try #require(materials.count == 2)
        #expect(materials[0].diffuseTransmission?.factor == 0.6)
        #expect(materials[0].diffuseRoughness?.factor == 0.7)
        #expect(materials[0].volumeScatter?.anisotropy == 0.4)
        #expect(materials[1].subsurface?.radius == SIMD3(1, 0.5, 0.25))
        #expect(materials[1].subsurface?.colorTexture?.transform == transform)
        #expect(try fixture.render(graph).pixels == fixture.render(original).pixels)
        #expect(try SceneArchive(capturing: graph, commandQueue: fixture.queue) == archive)
    }

    @Test("Classic materials retain their models, maps and rendered pixels")
    func classicMaterialsRoundTrip() throws {
        let image = try #require(fixture.device.makeTexture(descriptor: .texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
        )))
        let texel: [UInt8] = [192, 128, 64, 255]
        image.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: texel, bytesPerRow: 4)
        let binding = ClassicMaterial.TextureBinding(texture: image)
        let models: [ClassicMaterial.Model] = [
            .basic, .lambert, .phong(specular: SIMD3(0.1, 0.2, 0.3), shininess: 9),
            .toon(gradient: binding), .matcap(binding), .normal, .depth,
            .distance(reference: SIMD3(0, 0, -3), near: 0.25, far: 8), .shadow,
        ]
        let geometry = try GeometryRenderFixture.square(0.25)
        let nodes = try models.enumerated().map { index, model in
            let material = try ClassicMaterial(model: model, color: SIMD3(0.9, 0.6, 0.7), map: binding)
            return SceneNode(
                transform: SceneTransform(translation: SIMD3(Float(index % 3) - 1, Float(index / 3) - 1, -2)),
                mesh: try MeshRenderer(
                    device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .classic(material)
                )
            )
        }
        let light = try PunctualLight(kind: .directional, intensity: 2)
        let graph = SceneGraph(roots: nodes + [SceneNode(light: light)], camera: fixture.camera)
        let captured = try SceneArchive(capturing: graph, commandQueue: fixture.queue)
        let loaded = try captured.load(device: fixture.device, commandQueue: fixture.queue).graph
        let original = try fixture.render(graph)
        #expect(original.statistics.drawnMeshCount == models.count)
        #expect(original.pixels.contains { $0.x > 0.1 })
        #expect(try fixture.render(loaded).pixels == original.pixels)
        #expect(try SceneArchive(capturing: loaded, commandQueue: fixture.queue) == captured)
    }

    @Test("Point clouds, dashed lines and sprites retain their appearance and shared renderers")
    func primitiveRoundTrip() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 2, height: 2, mipmapped: false
        )
        descriptor.storageMode = .shared
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels: [UInt8] = [0, 255, 0, 255, 128, 255, 0, 128, 255, 0, 0, 192, 0, 0, 255, 255]
        texture.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: texels, bytesPerRow: 8)
        let pointCloud = try PointCloud(
            positions: [SIMD3(-0.8, 0.5, -2), SIMD3(-0.25, 0.5, -2)],
            colors: [SIMD4(1, 0.8, 0.6, 1), SIMD4(1, 0.7, 1, 0.9)], sizes: [0.7, 1.2]
        )
        let points = try PointsRenderer(
            device: fixture.device, cloud: pointCloud,
            material: PointsMaterial(size: .pixels(9), shape: .circle, map: texture, alphaTest: 0.2, blending: .alpha)
        )
        let lineGeometry = try LineGeometry(
            positions: [SIMD3(-1, -0.8, -2), SIMD3(-0.4, 0, -2), SIMD3(0, -0.8, -2)],
            colors: [SIMD4(1, 0, 0.5, 1), SIMD4(0, 0.7, 1, 0.8), SIMD4(0, 1, 0.5, 1)], topology: .loop
        )
        let line = try LineRenderer(
            device: fixture.device, geometry: lineGeometry,
            material: LineMaterial(
                width: .world(0.15), join: .miter(limit: 3), cap: .square,
                dash: .init(length: 0.2, gap: 0.1, offset: 0.025)
            )
        )
        let sprite = Sprite(
            position: SIMD3(0.65, 0, -2), size: SIMD2(14, 9), rotation: .pi / 7,
            center: SIMD2(0.2, 0.7), color: SIMD4(0.9, 1, 0.7, 0.85), textureRegion: SIMD4(0.25, 0, 0.75, 1)
        )
        let sprites = try SpriteRenderer(
            device: fixture.device, sprites: [sprite],
            material: SpriteMaterial(map: texture, sizeAttenuation: false, blending: .additive, depthTest: false)
        )
        let graph = SceneGraph(
            roots: [SceneNode(primitive: .points(points)),
                    SceneNode(transform: SceneTransform(translation: SIMD3(0, 0.5, 0)), primitive: .points(points)),
                    SceneNode(primitive: .lines(line)), SceneNode(primitive: .sprites(sprites))],
            camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: graph, commandQueue: fixture.queue)
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        let original = try fixture.render(graph)
        #expect(original.statistics.drawnPrimitiveCount == 4)
        #expect(original.pixels.contains { $0.w > 0.5 })
        let reloaded = try fixture.render(loaded)
        #expect(reloaded.statistics == original.statistics)
        #expect(reloaded.pixels == original.pixels)
        #expect(try SceneArchive(capturing: loaded, commandQueue: fixture.queue) == archive)
        guard case .points(let first)? = loaded.roots[0].primitive,
              case .points(let second)? = loaded.roots[1].primitive,
              case .sprites(let loadedSprites)? = loaded.roots[3].primitive else {
            Issue.record("Expected points and sprites.")
            return
        }
        #expect(first === second)
        #expect(first.material.map === loadedSprites.material.map)
        #expect(loadedSprites.sprites == [sprite])
    }

    @Test("Archives write to and read from a directory and render the same after reloading from disk")
    func directoryRoundTrip() throws {
        let (graph, clips, rig) = try richScene()
        let archive = try SceneArchive(capturing: graph, animations: clips, lightRig: rig)
        let directory = FileManager.default.temporaryDirectory.appending(path: "frostfield-archive-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try archive.write(to: directory)
        let read = try SceneArchive(contentsOf: directory)
        #expect(read == archive)
        let loaded = try read.load(device: fixture.device)
        #expect(try fixture.render(loaded.graph).pixels == fixture.render(graph).pixels)

        let manifest = try #require(String(data: archive.manifest, encoding: .utf8))
        #expect(manifest.contains("\"format\" : \"frostfield.scene\""))
        #expect(manifest.contains("\"version\" : 1"))
        #expect(manifest.contains("\"sku\" : \"W-42\""))
    }

    @Test("Loading validates the format, version, sidecar length and references")
    func validation() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        let graph = SceneGraph(roots: [SceneNode(mesh: mesh)], camera: fixture.camera)
        let archive = try SceneArchive(capturing: graph)
        let text = try #require(String(data: archive.manifest, encoding: .utf8))

        let nextVersion = SceneArchive.formatVersion + 1
        let future = Data(text.replacingOccurrences(of: "\"version\" : 1", with: "\"version\" : \(nextVersion)").utf8)
        #expect(throws: SceneArchive.Failure.unsupportedVersion(nextVersion)) {
            try SceneArchive(manifest: future, binary: archive.binary)
        }
        let foreign = Data(text.replacingOccurrences(of: "frostfield.scene", with: "other").utf8)
        #expect(throws: SceneArchive.Failure.format("other")) { try SceneArchive(manifest: foreign, binary: archive.binary) }
        #expect(throws: SceneArchive.Failure.binaryLength) {
            try SceneArchive(manifest: archive.manifest, binary: archive.binary.dropLast(4))
        }
        let dangling = Data(text.replacingOccurrences(of: "\"mesh\" : 0", with: "\"mesh\" : 7").utf8)
        #expect(throws: SceneArchive.Failure.invalidReference("mesh 7")) {
            try SceneArchive(manifest: dangling, binary: archive.binary).load(device: fixture.device)
        }
    }

    @Test("A 1,000-level hierarchy archives, encodes and loads from an async context")
    func deepHierarchy() async throws {
        let depth = 1_000
        var node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0.001, 0)), name: "leaf")
        var ids = [node.id]
        for level in 1..<depth {
            node = SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0.001, 0)), children: [node], name: "level \(level)",
                isVisible: level != 500
            )
            ids.append(node.id)
        }
        let graph = SceneGraph(roots: [node, SceneNode(name: "sibling")], camera: fixture.camera)
        await Task.yield()
        let archive = try SceneArchive(capturing: graph)
        let reread = try SceneArchive(manifest: archive.manifest, binary: archive.binary)
        let loaded = try reread.load(device: fixture.device)

        #expect(loaded.graph.roots.count == 2 && loaded.graph.roots[1].name == "sibling")
        var current: SceneNode? = loaded.graph.roots[0]
        var walked: [UUID] = []
        while let next = current {
            walked.append(next.id)
            #expect(next.children.count <= 1)
            current = next.children.first
        }
        #expect(walked == ids.reversed())
        #expect(loaded.graph.node(named: "level 500")?.isVisible == false)
        #expect(abs(try loaded.graph.worldPosition(of: ids[0]).y - 1) < 1e-4)
        #expect(try SceneArchive(capturing: loaded.graph).manifest == archive.manifest)
    }

    /// Captures a small scene with a texture, an instance set, geometry and a transform clip,
    /// appends `extra` bytes to the sidecar at a 16-byte boundary, applies an edit to the
    /// manifest JSON (given the extra bytes' offset), and returns the edited archive.
    private func corrupted(
        appending extra: Data = Data(), _ edit: (inout [String: Any], _ extraOffset: Int) throws -> Void
    ) throws -> SceneArchive {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let bytes = [UInt8](repeating: 200, count: 64)
        texture.replace(region: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0, withBytes: bytes, bytesPerRow: 16)
        let square = try GeometryRenderFixture.square(0.2)
        let textured = try fixture.unlit(square, material: .textured(texture))
        let instanced = try fixture.unlit(square, supportsInstancing: true)
        let instances = try InstanceSet(device: fixture.device, transforms: [translationMatrix(0.1)], tints: [SIMD4(1, 1, 1, 1)])
        let node = SceneNode(mesh: textured, children: [SceneNode(mesh: instanced, instances: instances)])
        let clip = AnimationClip(try SceneAnimation(tracks: [
            SceneAnimationTrack(nodeID: node.id, keyframes: [
                TransformKeyframe(time: 0, transform: SceneTransform()),
                TransformKeyframe(time: 1, transform: SceneTransform(translation: SIMD3(1, 0, 0))),
            ]),
        ]))
        let archive = try SceneArchive(capturing: SceneGraph(roots: [node], camera: fixture.camera), animations: [clip])
        var binary = archive.binary
        binary.append(Data(count: (16 - binary.count % 16) % 16))
        let offset = binary.count
        binary.append(extra)
        var json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        json["binaryLength"] = binary.count
        try edit(&json, offset)
        return try SceneArchive(manifest: JSONSerialization.data(withJSONObject: json), binary: binary)
    }

    /// Replaces a value at a key path of dictionaries and array indices.
    private func set(_ value: Any, at path: [Any], in json: inout [String: Any]) {
        func update(_ container: Any, _ path: ArraySlice<Any>) -> Any {
            guard let key = path.first else { return value }
            if let key = key as? String, var dictionary = container as? [String: Any] {
                dictionary[key] = update(dictionary[key] as Any, path.dropFirst())
                return dictionary
            }
            if let index = key as? Int, var array = container as? [Any] {
                array[index] = update(array[index], path.dropFirst())
                return array
            }
            return container
        }
        json = update(json, path[...]) as! [String: Any]
    }

    @Test("Corrupt counts, texture records, animation channels and parents throw instead of trapping")
    func corruptManifests() throws {
        let device = fixture.device
        // The unedited archive loads.
        #expect(try corrupted { _, _ in }.load(device: device).graph.roots.count == 1)

        let huge = 4_611_686_018_427_387_904
        #expect(throws: SceneArchive.Failure.self) {
            try self.corrupted { json, _ in self.set(huge, at: ["meshes", 0, "geometry", "vertexCount"], in: &json) }.load(device: device)
        }
        #expect(throws: SceneArchive.Failure.self) {
            try self.corrupted { json, _ in self.set(huge, at: ["instanceSets", 0, "count"], in: &json) }.load(device: device)
        }
        for (key, value) in [("mipmapLevelCount", 9), ("depth", 2), ("arrayLength", 3), ("width", 0), ("width", 20_000), ("textureType", 4)] {
            #expect(throws: SceneArchive.Failure.invalidTexture("unnamed"), "\(key) \(value)") {
                try self.corrupted { json, _ in self.set(value, at: ["textures", 0, key], in: &json) }.load(device: device)
            }
        }
        #expect(throws: SceneArchive.Failure.invalidTexture("unnamed"), "non-square cube") {
            try self.corrupted { json, _ in
                self.set(5, at: ["textures", 0, "textureType"], in: &json)
                self.set(2, at: ["textures", 0, "height"], in: &json)
            }.load(device: device)
        }
        #expect(throws: SceneArchive.Failure.invalidTexture("unnamed"), "overflowing cube array") {
            try self.corrupted { json, _ in
                self.set(MTLTextureType.typeCubeArray.rawValue, at: ["textures", 0, "textureType"], in: &json)
                self.set(Int.max, at: ["textures", 0, "arrayLength"], in: &json)
            }.load(device: device)
        }
        #expect(throws: SceneArchive.Failure.hierarchyDepth) {
            try self.corrupted { json, _ in
                let scene = try #require(json["scene"] as? [String: Any])
                let nodes = try #require(scene["nodes"] as? [[String: Any]])
                let prototype = try #require(nodes.first)
                let chain = (0...SceneArchive.maximumHierarchyDepth).map { index in
                    var node = prototype
                    node["id"] = UUID().uuidString
                    node["parent"] = index == 0 ? nil : index - 1
                    return node
                }
                self.set(chain, at: ["scene", "nodes"], in: &json)
            }.load(device: device)
        }
        #expect(throws: SceneArchive.Failure.invalidReference("parent 1 of node 1")) {
            try self.corrupted { json, _ in self.set(1, at: ["scene", "nodes", 1, "parent"], in: &json) }.load(device: device)
        }
        #expect(throws: SceneArchive.Failure.invalidAnimation("clip duration -1.0")) {
            try self.corrupted { json, _ in self.set(-1, at: ["animations", 0, "duration"], in: &json) }.load(device: device)
        }

        // glTF-style channels on views of known key data.
        var keyData = Data()
        func append<T>(_ values: [T]) {
            keyData.append(Data(count: (16 - keyData.count % 16) % 16))
            values.withUnsafeBytes { keyData.append(contentsOf: $0) }
        }
        append([Float(0), 1])  // increasing times
        append([Float(1), 0.5])  // decreasing times
        append([SIMD4<Float>(0, 0, 0, 1), SIMD4(1, 0, 0, 1)])  // two values
        append([SIMD4<Float>](repeating: SIMD4(0, 0, 0, 1), count: 4))  // four values
        let named = ["empty": (0, 0), "times": (0, 8), "decreasing": (16, 8), "two values": (32, 32), "four values": (64, 64)]
        var viewIndices: [String: Int] = [:]
        func channels(_ records: [[String: Any]]) throws -> SceneArchive {
            try corrupted(appending: keyData) { json, offset in
                var views = json["views"] as! [[String: Any]]
                for (name, (start, length)) in named.sorted(by: { $0.key < $1.key }) {
                    viewIndices[name] = views.count
                    views.append(["offset": offset + start, "length": length])
                }
                json["views"] = views
                let resolved = records.map { record in
                    var channel: [String: Any] = [
                        "sourceNodeIndex": 0, "target": "translation", "interpolation": "LINEAR", "weightCount": 0,
                    ]
                    for (key, value) in record { channel[key] = (value as? String).flatMap { viewIndices[$0] } ?? value }
                    return channel
                }
                let node = (((json["animations"] as! [[String: Any]])[0]["targets"] as! [[String: Any]])[0])["node"]!
                self.set([["node": node, "channels": resolved]], at: ["animations", 0, "targets"], in: &json)
            }
        }
        func view(_ name: String) -> Int { viewIndices[name] ?? -1 }

        // A valid channel loads and samples halfway between its keys.
        let valid = try channels([["times": "times", "values": "two values"]]).load(device: device)
        let pose = try #require(try valid.animations[0].sample(at: 0.5).values.first)
        #expect(pose.translation == SIMD3(0.5, 0, 0))

        let badTimes = SceneArchive.Failure.invalidAnimation("key times must be finite, nonnegative and strictly increasing")
        #expect(throws: badTimes) { try channels([["times": "empty", "values": "empty"]]).load(device: device) }
        #expect(throws: badTimes) { try channels([["times": "decreasing", "values": "two values"]]).load(device: device) }
        // A cubic spline needs in-tangent, value and out-tangent for each of its two keys: six values.
        let cubic = try channels([["times": "times", "values": "four values", "interpolation": "CUBICSPLINE"]])
        #expect(throws: SceneArchive.Failure.invalidView(view("four values"))) { try cubic.load(device: device) }
        #expect(throws: SceneArchive.Failure.invalidAnimation("morph weight count -1")) {
            try channels([["target": "weights", "times": "times", "weights": "four values", "weightCount": -1]]).load(device: device)
        }
        // Three weights per key need six floats, not sixteen.
        let weights = try channels([["target": "weights", "times": "times", "weights": "four values", "weightCount": 3]])
        #expect(throws: SceneArchive.Failure.invalidView(view("four values"))) { try weights.load(device: device) }
        #expect(throws: SceneArchive.Failure.invalidAnimation("rotation key 1 is not a unit quaternion")) {
            try channels([["target": "rotation", "times": "times", "values": "two values"]]).load(device: device)
        }
        #expect(throws: SceneArchive.Failure.self) {
            try channels([["times": "times", "values": "two values"], ["times": "times", "values": "two values"]]).load(device: device)
        }
    }

    @Test("Block-compressed textures with mip chains round-trip byte for byte, and equal bytes share one view")
    func compressedTextures() throws {
        let device = fixture.device
        var formats: [(MTLPixelFormat, bytesPerBlock: Int)] = []
        if device.supportsBCTextureCompression { formats.append((.bc1_rgba, 8)) }
        if device.supportsFamily(.apple2) { formats.append((.astc_4x4_ldr, 16)) }
        try #require(!formats.isEmpty, "The device supports neither BC nor ASTC")
        for (format, bytesPerBlock) in formats {
            func makeTexture(mipmapped: Bool = true) throws -> (any MTLTexture, [[UInt8]]) {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 16, height: 16, mipmapped: mipmapped)
                descriptor.usage = .shaderRead
                descriptor.storageMode = .shared
                let texture = try #require(device.makeTexture(descriptor: descriptor))
                var levels: [[UInt8]] = []
                for level in 0..<texture.mipmapLevelCount {
                    let side = max(1, 16 >> level), blocks = (side + 3) / 4
                    let bytes: [UInt8]
                    if format == .astc_4x4_ldr {
                        // LDR void-extent blocks with distinct colors per block and mip.
                        bytes = (0..<(blocks * blocks)).flatMap { block -> [UInt8] in
                            let red = UInt8(17 + level * 35 + block)
                            let green = UInt8(128 - block)
                            return [0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    red, red, green, green, 0x40, 0x40, 0xFF, 0xFF]
                        }
                    } else {
                        bytes = (0..<(blocks * blocks * bytesPerBlock)).map {
                            UInt8(truncatingIfNeeded: $0 * 37 + level * 11 + 5)
                        }
                    }
                    texture.replace(
                        region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: level, withBytes: bytes, bytesPerRow: blocks * bytesPerBlock
                    )
                    levels.append(bytes)
                }
                return (texture, levels)
            }
            func contents(_ texture: any MTLTexture) -> [[UInt8]] {
                (0..<texture.mipmapLevelCount).map { level in
                    let side = max(1, 16 >> level), blocks = (side + 3) / 4
                    var bytes = [UInt8](repeating: 0, count: blocks * blocks * bytesPerBlock)
                    texture.getBytes(&bytes, bytesPerRow: blocks * bytesPerBlock, from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: level)
                    return bytes
                }
            }
            let square = try GeometryRenderFixture.square(0.2)
            var mipmapped = true
            #if targetEnvironment(simulator)
            if format == .astc_4x4_ldr {
                let (source, _) = try makeTexture()
                let scene = SceneGraph(
                    roots: [SceneNode(mesh: try fixture.unlit(square, material: .textured(source)))], camera: fixture.camera
                )
                #expect(throws: SceneArchive.Failure.simulatorCompressedMipChain) {
                    try SceneArchive(capturing: scene, commandQueue: fixture.queue)
                }
                mipmapped = false
            }
            #endif
            let (texture, levels) = try makeTexture(mipmapped: mipmapped)
            #expect(texture.mipmapLevelCount == (mipmapped ? 5 : 1))
            #expect(contents(texture) == levels)
            let one = SceneGraph(roots: [SceneNode(mesh: try fixture.unlit(square, material: .textured(texture)))], camera: fixture.camera)
            let archive = try SceneArchive(capturing: one, commandQueue: fixture.queue)
            let loaded = try archive.load(device: device, commandQueue: fixture.queue)
            guard case .textured(let restored) = try #require(loaded.graph.roots.first?.mesh).material else {
                Issue.record("The material is not textured")
                continue
            }
            #expect(restored.pixelFormat == format && restored.mipmapLevelCount == texture.mipmapLevelCount && restored.storageMode == .shared)
            #expect(contents(restored) == levels, "\(format)")
            #expect(try SceneArchive(capturing: loaded.graph).binary == archive.binary)

            // A second texture object with the same bytes is stored once.
            let (copy, _) = try makeTexture(mipmapped: mipmapped)
            let two = SceneGraph(roots: [
                SceneNode(mesh: try fixture.unlit(square, material: .textured(texture))),
                SceneNode(mesh: try fixture.unlit(square, material: .textured(copy))),
            ], camera: fixture.camera)
            #expect(try SceneArchive(capturing: two).binary == archive.binary)
        }
    }

    @Test("Custom libraries and unrecorded samplers are reported instead of being dropped")
    func unsupportedContent() throws {
        let library = try fixture.device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            struct In { float3 normal [[user(frostfield_normal)]]; };
            fragment float4 tint(In input [[stage_in]], constant float4 &p [[buffer(0)]]) { return p; }
            """, options: nil)
        let custom = try fixture.unlit(try GeometryRenderFixture.square(0.25), material: .custom(library: library, fragment: "tint"))
        #expect(throws: SceneArchive.Failure.self) {
            try SceneArchive(capturing: SceneGraph(roots: [SceneNode(mesh: custom)], camera: fixture.camera))
        }

        let descriptor = MTLSamplerDescriptor()
        let unknown = try #require(fixture.device.makeSamplerState(descriptor: descriptor))
        let surface = try ProceduralSurface(device: fixture.device, style: .leatherette, resolution: 32, seed: 1)
        let material = try PhysicallyBasedMaterial(normalTexture: .init(texture: surface.normalTexture, sampler: unknown))
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25), material: .physicallyBased(material))
        #expect(throws: SceneArchive.Failure.unrecordedSampler("the normal map")) {
            try SceneArchive(capturing: SceneGraph(roots: [SceneNode(mesh: mesh)], camera: fixture.camera))
        }
        let recorded = try TextureSampling(addressU: .clampToZero).makeSamplerState(device: fixture.device)
        #expect(TextureSampling.recorded(for: recorded)?.addressU == .clampToZero)
        #expect(TextureSampling.recorded(for: unknown) == nil)
    }
}

/// JSON coding of shader graphs and light rigs, like Three.js node material serialization.
struct SceneCodingTests {
    @Test
    func `Shader graphs round-trip through JSON and generate the same Metal source`() throws {
        let graph = try SurfaceShaderGraph(
            nodes: [
                .input(.uv0), .uv(.u), .scalar(2), .multiply(1, 2), .sine(3), .vector2(SIMD2(0.5, 0.25)),
                .binary(.distance, 0, 5), .unary(.oneMinus, 6), .smoothstep(edge0: 2, edge1: 7, value: 4),
                .color(SIMD3(0.8, 0.3, 0.1)), .combine([8, 8, 7]), .mix(9, 10, factor: 8), .latticeNoise(.value, position: 10),
                .cellularNoise(position: 10), .swizzle(13, "x"), .remap(14, fromMinimum: 2, fromMaximum: 7, toMinimum: 8, toMaximum: 4),
                .function(0, arguments: [15]), .select(condition: 16, ifTrue: 11, ifFalse: 10),
                .clamp(17, minimum: 9, maximum: 10), .parameter(.y), .worldPosition(.z), .normal(.x), .add(19, 20),
                .subtract(22, 21), .divide(23, 2), .clamp01(24), .floor(25), .fract(24), .cosine(26), .noise(x: 25, y: 26, z: 27),
                .rgb(red: 28, green: 29, blue: 25), .uniform(0), .multiply(30, 31), .vector4(SIMD4(1, 2, 3, 4)),
            ],
            uniforms: [ShaderUniform("gain", SIMD3(1, 0.5, 0.25))],
            functions: [ShaderFunction(name: "wave", parameters: [.init("x", .scalar)], returns: .scalar, body: "return sin(x) * 0.5 + 0.5;")],
            labels: [18: "clamped"], outputs: .init(baseColor: 18, metallic: 7, emission: 32)
        )
        let data = try JSONEncoder().encode(graph)
        let decoded = try JSONDecoder().decode(SurfaceShaderGraph.self, from: data)
        #expect(decoded == graph)
        #expect(try decoded.metalFunction(named: "f") == graph.metalFunction(named: "f"))

        let post = try PostProcessGraph(nodes: [.input(.screenUV), .sampleScene(coordinates: 0)], output: 1)
        #expect(try JSONDecoder().decode(PostProcessGraph.self, from: JSONEncoder().encode(post)) == post)
        let buffer = try BufferShaderGraph(
            nodes: [.input(.elementField(0)), .scalar(2), .multiply(0, 1)], fieldCount: 1, outputs: [0: 2]
        )
        #expect(try JSONDecoder().decode(BufferShaderGraph.self, from: JSONEncoder().encode(buffer)) == buffer)

        // Decoding re-validates: a dangling reference is rejected like the initializer rejects it.
        var text = try #require(String(data: data, encoding: .utf8))
        text = text.replacingOccurrences(of: "\"baseColor\":18", with: "\"baseColor\":99")
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(SurfaceShaderGraph.self, from: Data(text.utf8)) }
    }

    @Test
    func `Lights, rigs, cameras and transforms round-trip and validate`() throws {
        let rig = try LightRig(lights: [
            .init(id: "a", kind: .rectangle(width: 2, height: 1), position: SIMD3(0, 2, 0), target: .zero, emission: .power(100)),
            .init(id: "b", kind: .point(range: nil), position: SIMD3(1, 1, 1), target: .zero, emission: .intensity(30), isEnabled: false),
        ])
        #expect(try JSONDecoder().decode(LightRig.self, from: JSONEncoder().encode(rig)) == rig)
        let light = try PunctualLight(
            kind: .spot(range: 4, innerConeAngle: 0.1, outerConeAngle: 0.5), color: SIMD3(1, 0.5, 0.25), intensity: 12,
            shadow: try .init(depthBias: 0.01, normalBias: 2, filterRadius: 1, nearPlane: 0.1)
        )
        #expect(try JSONDecoder().decode(PunctualLight.self, from: JSONEncoder().encode(light)) == light)
        let invalid = Data(#"{"type":"point","color":[2,0,0],"intensity":1}"#.utf8)
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(PunctualLight.self, from: invalid) }
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.5, far: 20)
        let decodedCamera = try JSONDecoder().decode(PerspectiveCamera.self, from: JSONEncoder().encode(camera))
        #expect(decodedCamera.verticalFieldOfView == 1 && decodedCamera.far == 20)
        let transform = SceneTransform(translation: SIMD3(1, 2, 3), rotation: simd_quatf(angle: 0.3, axis: SIMD3(0, 0, 1)), scale: SIMD3(1, 2, 1))
        let decodedTransform = try JSONDecoder().decode(SceneTransform.self, from: JSONEncoder().encode(transform))
        #expect(decodedTransform.matrix == transform.matrix)
    }
}


/// Writes linear RGBA16Float pixels as an sRGB PNG for visual inspection.
