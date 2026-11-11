import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LiveMaterialRenderingTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture(side: 32)
    }

    private func graph(_ material: PhysicallyBasedMaterial) throws -> SceneGraph {
        let renderer = try fixture.unlit(GeometryRenderFixture.square(1.5), material: .physicallyBased(material))
        return SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -1)), mesh: renderer)], camera: fixture.camera
        )
    }

    @Test
    func `Animated emission renders the analytic radiance at the sampled time`() throws {
        let material = try PhysicallyBasedMaterial(
            metallic: 0, roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
            emissiveFactor: .zero, alphaMode: .opaque
        ).withLiveFactors()
        let live = try #require(material.liveFactors)
        let scene = try graph(material)
        let clip = try AnimationClip(tracks: [
            .color(.material(live, .emissiveFactor), times: [0, 1], values: [.zero, SIMD3(0.5, 1, 0.25)]),
            .number(.material(live, .emissiveStrength), times: [0, 1], values: [1, 3]),
        ])
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: clip))

        try mixer.applyLiveProperties()
        let dark = try fixture.render(scene).pixels
        #expect(isNearly(fixture.pixel(dark, x: 16, y: 16), SIMD4(0, 0, 0, 1)))

        try mixer.update(deltaTime: 0.5)
        try mixer.applyLiveProperties()
        // Factor (0.25, 0.5, 0.125) times strength 2, with no light and a black base.
        let lit = try fixture.render(scene).pixels
        #expect(isNearly(fixture.pixel(lit, x: 16, y: 16), SIMD4(0.5, 1, 0.25, 1), tolerance: 0.002))
    }

    @Test
    func `Live factors render exactly like a material made with the same factors`() throws {
        let texture = try checker()
        func material(
            roughness: Float, metallic: Float, base: SIMD4<Float>, offset: SIMD2<Float>, rotation: Float
        ) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                baseColorTexture: texture, metallic: metallic, roughness: roughness, directionToLight: SIMD3(0.3, 0.4, 1),
                lightIntensity: 3, ambientIntensity: 0.1, baseColorFactor: base, alphaMode: .opaque,
                baseColorTextureTransform: try .init(offset: offset, rotation: rotation)
            )
        }
        let authored = try material(roughness: 0.2, metallic: 0, base: SIMD4(0.8, 0.3, 0.2, 1), offset: .zero, rotation: 0)
        let animatable = authored.withLiveFactors()
        let live = try #require(animatable.liveFactors)
        let scene = try graph(animatable)
        let rest = try fixture.render(scene).pixels
        #expect(rest == (try fixture.render(graph(authored)).pixels))

        let clip = try AnimationClip(tracks: [
            .number(.material(live, .roughness), times: [0, 2], values: [0.2, 0.9]),
            .number(.material(live, .metallic), times: [0, 2], values: [0, 1]),
            AnimationTrack(property: .material(live, .baseColorFactor), times: [0, 2], values: [
                .vector4(SIMD4(0.8, 0.3, 0.2, 1)), .vector4(SIMD4(0.2, 0.6, 1, 1)),
            ]),
            AnimationTrack(property: .material(live, .textureOffset(.baseColor)), times: [0, 2], values: [
                .vector2(.zero), .vector2(SIMD2(0.5, 0.25)),
            ]),
            .number(.material(live, .textureRotation(.baseColor)), times: [0, 2], values: [0, 0.8]),
        ])
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: clip))
        try mixer.update(deltaTime: 1)
        try mixer.applyLiveProperties()
        let current = live.current
        #expect(abs(current.roughness - 0.55) < 1e-6 && abs(current.metallic - 0.5) < 1e-6)
        #expect(simd_distance(current.baseColorFactor, SIMD4(0.5, 0.45, 0.6, 1)) < 1e-6)
        let transform = try #require(current.textureTransforms[.baseColor])
        #expect(simd_distance(transform.offset, SIMD2(0.25, 0.125)) < 1e-6 && abs(transform.rotation - 0.4) < 1e-6)
        let animated = try fixture.render(scene).pixels
        let reference = try fixture.render(graph(material(
            roughness: current.roughness, metallic: current.metallic, base: current.baseColorFactor,
            offset: transform.offset, rotation: transform.rotation
        ))).pixels
        #expect(animated == reference)
        let change = zip(animated, rest).map { simd_reduce_max(abs($0 - $1)) }.max() ?? 0
        #expect(change > 0.05)
    }

    @Test
    func `Path tracing scenes repack live factors and match a fresh snapshot`() throws {
        let path = try PathTestFixture()
        let material = try PhysicallyBasedMaterial(
            metallic: 0, roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
            emissiveFactor: SIMD3(1, 0.5, 0.25), alphaMode: .opaque
        ).withLiveFactors()
        let live = try #require(material.liveFactors)
        func scene(_ material: PhysicallyBasedMaterial) throws -> SceneGraph {
            let renderer = try MeshRenderer(
                device: path.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 10, height: 8),
                material: .physicallyBased(material)
            )
            return SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: renderer)], camera: path.camera)
        }
        let graph = try scene(material)
        let traced = try PathTracingScene(queue: path.queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: path.queue, scene: traced)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 3)
        let before = try path.trace(tracer, width: 16, height: 16, samples: 2, settings: settings)
        #expect(isNearly(before[8 * 16 + 8], SIMD4(1, 0.5, 0.25, 1), tolerance: 0.002))
        #expect(try traced.updateMaterials() == false)

        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: AnimationClip(tracks: [
            .color(.material(live, .emissiveFactor), times: [0, 1], values: [SIMD3(1, 0.5, 0.25), SIMD3(0, 1, 0.5)]),
            .number(.material(live, .emissiveStrength), times: [0, 1], values: [1, 4]),
        ])))
        try mixer.update(deltaTime: 0.5)
        try mixer.applyLiveProperties()
        #expect(try traced.updateMaterials())
        #expect(traced.revision == 1)
        let after = try path.trace(tracer, width: 16, height: 16, samples: 2, settings: settings)
        // Factor (0.5, 0.75, 0.375) at strength 2.5.
        #expect(isNearly(after[8 * 16 + 8], SIMD4(1.25, 1.875, 0.9375, 1), tolerance: 0.004))
        let fresh = try PathTracingScene(
            queue: path.queue, graph: scene(PhysicallyBasedMaterial(
                metallic: 0, roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveFactor: SIMD3(0.5, 0.75, 0.375), alphaMode: .opaque, emissiveStrength: 2.5
            )), textureSize: 16
        )
        let reference = try path.trace(
            try PathTracer(queue: path.queue, scene: fresh), width: 16, height: 16, samples: 2, settings: settings
        )
        #expect(PathTestFixture.maximumDifference(after, reference) < 1e-3)
        #expect(try traced.updateMaterials() == false)
        #expect(traced.revision == 1)
    }

    @Test("Repacking keeps every material's textures when a live layer turns off or back on",
          arguments: [PathTracingScene.Options.Textures.baked(size: 16), .fullResolution(fallbackSize: 16)])
    func stableTextureIndices(textures: PathTracingScene.Options.Textures) throws {
        let path = try PathTestFixture()
        func solid(_ rgba: [UInt8]) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            let texture = try #require(path.device.makeTexture(descriptor: descriptor))
            let texels = Array(repeating: rgba, count: 16).flatMap { $0 }
            texels.withUnsafeBytes { texture.replace(region: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 16) }
            return texture
        }
        // A packs first: a clearcoat factor map, behind the camera. B, in view, glows with an emissive map.
        let coated = try PhysicallyBasedMaterial(
            metallic: 0, roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
            alphaMode: .opaque,
            clearcoat: .init(factor: 1, roughness: 0.2, factorTexture: PhysicallyBasedMaterial.TextureBinding(texture: try solid([255, 0, 0, 255])))
        ).withLiveFactors()
        let live = try #require(coated.liveFactors)
        let greenTexture = try solid([0, 255, 0, 255])
        let sampling = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, magnification: .nearest, minification: .nearest, mipFilter: .notMipmapped
        )
        let sampler = try sampling.makeSamplerState(device: path.device)
        let glowing = try PhysicallyBasedMaterial(
            baseColorTexture: greenTexture, metallic: 0, roughness: 1, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(0, 0, 0, 1), baseColorSampler: sampler,
            emissiveTexture: .init(texture: greenTexture, sampler: sampler, sampling: sampling),
            emissiveFactor: SIMD3(1, 1, 1), alphaMode: .opaque, baseColorSampling: sampling
        ).withLiveFactors()
        let liveGlow = try #require(glowing.liveFactors)
        func renderer(_ material: PhysicallyBasedMaterial) throws -> MeshRenderer {
            try MeshRenderer(
                device: path.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 10, height: 8),
                material: .physicallyBased(material)
            )
        }
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, 50)), mesh: try renderer(coated)),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: try renderer(glowing)),
        ], camera: path.camera)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 3)
        func center(_ scene: PathTracingScene) throws -> SIMD4<Float> {
            try path.trace(PathTracer(queue: path.queue, scene: scene), width: 16, height: 16, samples: 2, settings: settings)[8 * 16 + 8]
        }
        let green = SIMD4<Float>(0, 1, 0, 1)

        let traced = try PathTracingScene(queue: path.queue, graph: graph, options: .init(textures: textures))
        #expect(traced.textureCount == 2)
        #expect(isNearly(try center(traced), green, tolerance: 0.01))
        try live.update { $0.clearcoatFactor = 0 }
        #expect(try traced.updateMaterials())
        #expect(isNearly(try center(traced), green, tolerance: 0.01))
        try live.update { $0.clearcoatFactor = 0.5 }
        try liveGlow.update { $0.roughness = 0.5 }
        #expect(try traced.updateMaterials())
        #expect(isNearly(try center(traced), green, tolerance: 0.01))

        // Prepared while the layer is off, the scene still reserves the layer's map for when it turns on.
        try live.update { $0.clearcoatFactor = 0 }
        let preparedOff = try PathTracingScene(queue: path.queue, graph: graph, options: .init(textures: textures))
        #expect(preparedOff.textureCount == 2)
        live.reset()
        #expect(try preparedOff.updateMaterials())
        #expect(isNearly(try center(preparedOff), green, tolerance: 0.01))
    }

    @Test("Blended opacity animates on both acceleration paths, including with vertex alpha",
          arguments: [PathTracingScene.Options.Acceleration.software, .automatic], [false, true])
    func animateOpacity(acceleration: PathTracingScene.Options.Acceleration, vertexColors: Bool) throws {
        let path = try PathTestFixture()
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(repeating: 1), alphaMode: .blend
        ).withLiveFactors()
        let plane = try MeshGeometry.plane(width: 10, height: 8)
        let geometry = try vertexColors
            ? MeshGeometry(vertices: plane.vertices.map {
                MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate,
                           color: SIMD4(1, 1, 1, 0.5))
            }, indices: plane.indices)
            : plane
        let renderer = try MeshRenderer(
            device: path.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(material)
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: renderer)], camera: path.camera
        )
        let scene = try PathTracingScene(queue: path.queue, graph: graph, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: path.queue, scene: scene)
        let lighting = try PathTracer.Lighting(backgroundVisible: false)
        func coverage(samples: Int) throws -> Float {
            let pixels = try path.trace(
                tracer, width: 4, height: 4, samples: samples, settings: .init(maxBounces: 1, seed: 3), lighting: lighting
            )
            return pixels.reduce(0) { $0 + $1.w } / Float(pixels.count)
        }
        let vertexAlpha: Float = vertexColors ? 0.5 : 1
        #expect(abs(try coverage(samples: vertexColors ? 128 : 4) - vertexAlpha) < 0.04)
        let live = try #require(material.liveFactors)
        try live.update { $0.baseColorFactor.w = 0.5 }
        #expect(try scene.updateMaterials())
        #expect(abs(try coverage(samples: 128) - 0.5 * vertexAlpha) < 0.04)
    }

    @Test("A live edit during asynchronous preparation remains pending for the packed scene")
    func editDuringPreparation() async throws {
        let path = try PathTestFixture()
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque
        ).withLiveFactors()
        let live = try #require(material.liveFactors)
        try live.update { $0.emissiveFactor = SIMD3(repeating: 0.25) }
        let renderer = try MeshRenderer(
            device: path.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 10, height: 8),
            material: .physicallyBased(material)
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: renderer)], camera: path.camera
        )
        let scene = try await PathTracingScene.prepare(queue: path.queue, graph: graph) { _ in live.reset() }
        let tracer = try PathTracer(queue: path.queue, scene: scene)
        func color() throws -> SIMD4<Float> {
            let pixels = try path.trace(tracer, width: 4, height: 4, samples: 1, settings: .init(maxBounces: 1, seed: 3))
            return pixels[10]
        }
        #expect(isNearly(try color(), SIMD4(0.25, 0.25, 0.25, 1), tolerance: 0.002))
        #expect(try scene.updateMaterials())
        #expect(isNearly(try color(), SIMD4(repeating: 1), tolerance: 0.002))
    }

    /// A 4×4 RGBA8 texture whose texels all differ.
    private func checker() throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        var texels: [UInt8] = []
        for y in 0..<4 {
            for x in 0..<4 { texels += [UInt8(40 + 60 * x), UInt8(40 + 60 * y), UInt8(255 - 50 * (x + y) / 2), 255] }
        }
        texels.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: 16)
        }
        return texture
    }
}
