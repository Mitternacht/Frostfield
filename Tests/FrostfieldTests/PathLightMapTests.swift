import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathLightMapTests {
    private let fixture: PathTestFixture

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Traced light maps select both UVs and clamp transformed HDR irradiance",
          arguments: PathAccelerationTests.modes, [false, true])
    func authoredAtlas(acceleration: PathTracingScene.Options.Acceleration, bakeTextures: Bool) throws {
        let warm = SIMD3<Float>(3, 1, 0.125)
        let cool = SIMD3<Float>(0.125, 0.5, 4)
        let atlas = try texture([warm, cool])
        let uv0 = try PhysicallyBasedMaterial.LightMap(texture: atlas, textureCoordinateSet: .primary)
        let uv1 = try PhysicallyBasedMaterial.LightMap(texture: atlas)
        let shifted = try PhysicallyBasedMaterial.LightMap(texture: atlas, transform: .init(offset: SIMD2(-2, 0)))
        let geometry = try plane()
        let color = SIMD3<Float>(0.25, 0.5, 0.75)
        let specular = try PhysicallyBasedMaterial.Specular(factor: 0)
        let options = PathTracingScene.Options(
            textures: bakeTextures ? .baked(size: 16) : .fullResolution(fallbackSize: 16),
            acceleration: acceleration, sampleEmissiveMeshes: true
        )

        func trace(_ map: PhysicallyBasedMaterial.LightMap?, strength: Float = 2, metal: Float = 0) throws -> SIMD4<Float> {
            let material = try PhysicallyBasedMaterial(
                metallic: metal, baseColorFactor: SIMD4(color, 1), alphaMode: .opaque, specular: specular,
                lightMap: map, lightMapIntensity: strength
            )
            let scene = try prepare([node(geometry, material: material)], options: options)
            #expect(scene.sampledEmissiveTriangleCount == 0)
            #expect(scene.statistics.acceleration == acceleration)
            if map != nil && strength > 0 {
                let shouldBake = bakeTextures || fixture.device.argumentBuffersSupport != .tier2
                #expect(scene.statistics.bakedTextureCount == (shouldBake ? 1 : 0))
                #expect(scene.statistics.boundTextureCount == (shouldBake ? 0 : 1))
            }
            let tracer = try PathTracer(queue: fixture.queue, scene: scene)
            return try fixture.trace(tracer, width: 1, height: 1, samples: 4, settings: .init(maxBounces: 1))[0]
        }

        let diffuse = color * (2 / Float.pi)
        #expect(close(try trace(uv0), SIMD4(warm * diffuse, 1)))
        #expect(close(try trace(uv1), SIMD4(cool * diffuse, 1)))
        #expect(close(try trace(shifted), SIMD4(warm * diffuse, 1)))
        #expect(try trace(uv0, strength: 0) == trace(nil))
        #expect(try trace(uv0, metal: 1) == SIMD4(0, 0, 0, 1))
    }

    @Test("Baked diffuse adds independently of live direct light, emission, MIS and albedo",
          arguments: PathAccelerationTests.modes)
    func lightTransportSeparation(acceleration: PathTracingScene.Options.Acceleration) throws {
        let geometry = try MeshGeometry.plane(width: 32, height: 32)
        let color = SIMD3<Float>(0.25, 0.5, 0.75)
        let emission = SIMD3<Float>(0.125, 0.25, 0.5)
        let irradiance = SIMD3<Float>(2, 1, 0.25)
        let baked = try MeshLightmapBaker(
            device: fixture.device, geometry: geometry, width: 8, height: 8, textureCoordinateSet: .primary,
            lights: [.init(directionToLight: SIMD3(0, 0, 1), irradiance: irradiance)]
        )
        let map = try baked.pbrLightMapBinding()
        let sun = try PathTracer.AnalyticLight(.init(kind: .directional, color: SIMD3(0.25, 0.5, 1), intensity: 1))
        let lighting = try PathTracer.Lighting(analyticLights: [sun])
        let specular = try PhysicallyBasedMaterial.Specular(factor: 0)

        func trace(_ map: PhysicallyBasedMaterial.LightMap?, mis: Bool) throws -> SIMD4<Float> {
            let material = try PhysicallyBasedMaterial(
                baseColorFactor: SIMD4(color, 1), emissiveFactor: emission, alphaMode: .opaque,
                specular: specular, lightMap: map
            )
            let scene = try prepare([node(geometry, material: material)], options: .init(
                acceleration: acceleration, sampleEmissiveMeshes: true
            ))
            #expect(scene.sampledEmissiveTriangleCount == 2)
            let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: lighting)
            let accumulation = try tracer.makeAccumulation(width: 1, height: 1)
            let output = try fixture.output(width: 1, height: 1)
            let albedo = try fixture.output(width: 1, height: 1)
            let emitted = try fixture.output(width: 1, height: 1)
            let command = try #require(fixture.queue.makeCommandBuffer())
            _ = try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: output, accumulation: accumulation,
                commandBuffer: command, samplesPerPixel: 4, settings: .init(maxBounces: 1, multipleImportanceSampling: mis),
                aovs: .init(albedo: albedo, emission: emitted)
            )
            try fixture.finish(command)
            #expect(try fixture.readback(albedo)[0] == SIMD4(color, 1))
            #expect(try fixture.readback(emitted)[0] == SIMD4(emission, 1))
            return try fixture.readback(output)[0]
        }

        let unbaked = try trace(nil, mis: true)
        let withBake = try trace(map, mis: true)
        let expected = unbaked + SIMD4(irradiance * color / Float.pi, 0)
        #expect(unbaked.x > emission.x && unbaked.z > emission.z)
        #expect(close(withBake, expected))
        #expect(try trace(map, mis: false) == withBake)
    }

    @Test("A reflected hit adds baked light once, independently of the emissive surface's MIS weight",
          arguments: PathAccelerationTests.modes)
    func bouncedAtlas(acceleration: PathTracingScene.Options.Acceleration) throws {
        let geometry = try plane()
        let mirror = try node(geometry, material: .init(metallic: 1, roughness: 0, alphaMode: .opaque))
        let irradiance = SIMD3<Float>(2, 0.5, 0.25)
        let map = try PhysicallyBasedMaterial.LightMap(texture: texture([irradiance]))
        let emission = SIMD3<Float>(0.125, 0.25, 0.5)
        let facingMirror = SceneTransform(
            translation: SIMD3(0, 0, 2), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        )

        func tracer(_ map: PhysicallyBasedMaterial.LightMap?) throws -> PathTracer {
            let material = try PhysicallyBasedMaterial(
                emissiveFactor: emission, alphaMode: .opaque, specular: .init(factor: 0), lightMap: map
            )
            let receiver = try node(geometry, material: material, transform: facingMirror)
            let scene = try prepare(
                [mirror, receiver], options: .init(acceleration: acceleration, sampleEmissiveMeshes: true)
            )
            #expect(scene.sampledEmissiveTriangleCount == 2)
            return try PathTracer(queue: fixture.queue, scene: scene)
        }
        let plain = try tracer(nil)
        let mapped = try tracer(map)
        func sample(_ tracer: PathTracer, bounces: Int, mis: Bool) throws -> SIMD4<Float> {
            let settings = PathTracer.Settings(maxBounces: bounces, multipleImportanceSampling: mis)
            return try fixture.trace(tracer, width: 1, height: 1, samples: 16, settings: settings)[0]
        }

        #expect(try sample(mapped, bounces: 1, mis: true) == sample(plain, bounces: 1, mis: true))
        for mis in [false, true] {
            let reference = try sample(plain, bounces: 2, mis: mis)
            let lit = try sample(mapped, bounces: 2, mis: mis)
            #expect(reference.x > 0 && reference.w == 1)
            #expect(close(lit - reference, SIMD4(irradiance / Float.pi, 0), tolerance: 0.003))
        }
    }

    private func plane() throws -> MeshGeometry {
        let plane = try MeshGeometry.plane(width: 32, height: 32)
        return try MeshGeometry(vertices: plane.vertices.map {
            MeshVertex(
                position: $0.position, normal: $0.normal,
                textureCoordinate: SIMD2(0.125, 0.5), secondaryTextureCoordinate: SIMD2(0.875, 0.5)
            )
        }, indices: plane.indices)
    }

    private func node(
        _ geometry: MeshGeometry, material: PhysicallyBasedMaterial,
        transform: SceneTransform = .init(translation: SIMD3(0, 0, -3))
    ) throws -> SceneNode {
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        return SceneNode(transform: transform, mesh: renderer)
    }

    private func prepare(_ nodes: [SceneNode], options: PathTracingScene.Options) throws -> PathTracingScene {
        try PathTracingScene(queue: fixture.queue, graph: SceneGraph(roots: nodes, camera: fixture.camera), options: options)
    }

    private func texture(_ colors: [SIMD3<Float>]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: colors.count, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let pixels = colors.map { SIMD4<Float16>(SIMD4($0, 0)) }
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, colors.count, 1), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: bytes.count
            )
        }
        return texture
    }

    private func close(_ actual: SIMD4<Float>, _ expected: SIMD4<Float>, tolerance: Float = 0.001) -> Bool {
        simd_reduce_max(abs(actual - expected)) < tolerance
    }
}
