import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathRetroreflectionTests {
    private let fixture: PathCameraFixture
    private let tilt = simd_quatf(angle: 0.62, axis: SIMD3(0, 1, 0))
    private let narrow = PathTracer.CameraModel(projection: .orthographic(height: 0.0001), near: 0.1, far: 20)

    init() throws {
        fixture = try PathCameraFixture()
    }

    @Test("Omitted and explicit zero retroreflectivity give identical paths",
          arguments: PathAccelerationTests.modes)
    func zeroDefault(acceleration: PathTracingScene.Options.Acceleration) throws {
        let defaultMaterial = try PhysicallyBasedMaterial(metallic: 0.35, roughness: 0.4, alphaMode: .opaque)
        let zero = try PhysicallyBasedMaterial(metallic: 0.35, roughness: 0.4, alphaMode: .opaque, retroreflectivity: 0)
        let panel = try PathTracer.AreaLight(
            center: SIMD3(0.3, 0, -1), halfU: SIMD3(0.6, 0, 0), halfV: SIMD3(0, -0.6, 0),
            radiance: SIMD3(2, 1, 0.5), cameraVisible: false
        )
        let lighting = try PathTracer.Lighting(lights: [panel], environment: SIMD3(0.05, 0.2, 0.4))
        let a = try tracer(defaultMaterial, acceleration: acceleration)
        let b = try tracer(zero, acceleration: acceleration)
        for sampler: PathTracer.Sampler in [.independent, .sobol] {
            let settings = PathTracer.Settings(maxBounces: 2, seed: 19, sampler: sampler)
            let pixels = try fixture.base.trace(a, width: 8, height: 5, samples: 16, settings: settings, lighting: lighting)
            let actual = try fixture.base.trace(b, width: 8, height: 5, samples: 16, settings: settings, lighting: lighting)
            #expect(pixels == actual)
        }
    }

    @Test("Path-traced punctual GGX peaks match ordinary and retroreflective directions",
          arguments: PathAccelerationTests.modes)
    func punctualDirections(acceleration: PathTracingScene.Options.Acceleration) throws {
        let base = SIMD3<Float>(0.7, 0.45, 0.25)
        let roughness: Float = 0.4
        let normal = tilt.act(SIMD3(0, 0, 1))
        let view = SIMD3<Float>(0, 0, 1)
        let reflected = 2 * simd_dot(normal, view) * normal - view
        for strength: Float in [0, 0.35, 1] {
            let material = try PhysicallyBasedMaterial(
                metallic: 1, roughness: roughness, baseColorFactor: SIMD4(base, 1),
                alphaMode: .opaque, retroreflectivity: strength
            )
            let tracer = try self.tracer(material, acceleration: acceleration)
            for direction in [view, reflected] {
                let light = try PathTracer.AnalyticLight(.init(kind: .directional), direction: -direction)
                let pixel = try sample(tracer, lighting: .init(analyticLights: [light]), samples: 8)
                let expected = reference(light: direction, base: base, roughness: roughness, strength: strength)
                #expect(simd_reduce_max(abs(pixel - SIMD4(expected, 1))) < 0.005, "\(pixel) vs \(expected)")
            }
        }
    }

    @Test("Retroreflective VNDF alone and light/BSDF MIS match independent area integration",
          arguments: PathAccelerationTests.modes)
    func areaSampling(acceleration: PathTracingScene.Options.Acceleration) throws {
        let base = SIMD3<Float>(0.8, 0.5, 0.25)
        let radiance = SIMD3<Float>(3, 1, 0.5)
        let roughness: Float = 0.31
        let strength: Float = 0.65
        let material = try PhysicallyBasedMaterial(
            metallic: 1, roughness: roughness, baseColorFactor: SIMD4(base, 1),
            alphaMode: .opaque, retroreflectivity: strength
        )
        let tracer = try self.tracer(material, acceleration: acceleration)
        let panel = try PathTracer.AreaLight(
            center: SIMD3(0.1, -0.05, -1), halfU: SIMD3(0.45, 0, 0), halfV: SIMD3(0, -0.4, 0),
            radiance: radiance, cameraVisible: false
        )
        let lighting = try PathTracer.Lighting(lights: [panel])
        let resolution = 100
        let cellArea = 4 * simd_length(simd_cross(panel.halfU, panel.halfV)) / Float(resolution * resolution)
        var integral = SIMD3<Float>.zero
        for y in 0..<resolution {
            for x in 0..<resolution {
                let uv = (SIMD2(Float(x), Float(y)) + 0.5) / Float(resolution) * 2 - 1
                let delta = panel.center + uv.x * panel.halfU + uv.y * panel.halfV - SIMD3(0, 0, -3)
                let squaredDistance = simd_length_squared(delta)
                let direction = delta / sqrt(squaredDistance)
                integral += reference(light: direction, base: base, roughness: roughness, strength: strength)
                    * (direction.z * cellArea / squaredDistance)
            }
        }
        let expected = SIMD4(integral * radiance, 1)
        #expect(expected.x > 0.5)
        for mis in [false, true] {
            let pixel = try sample(tracer, lighting: lighting, samples: 4_096, mis: mis)
            #expect(simd_reduce_max(abs(pixel - expected)) < 0.012, "MIS \(mis) \(pixel) vs \(expected)")
        }
    }

    @Test("A retroreflective material keeps the transmission proposal for thin and index-matched glass",
          arguments: PathAccelerationTests.modes)
    func keepsTransmission(acceleration: PathTracingScene.Options.Acceleration) throws {
        let sky = SIMD3<Float>(0.5, 0.25, 0.125)
        let lighting = try PathTracer.Lighting(environment: sky)
        for thick in [false, true] {
            let transmission = try PhysicallyBasedMaterial.Transmission(
                factor: 1, indexOfRefraction: thick ? 1 : 1.5, thickness: thick ? 1 : 0
            )
            let geometry = try thick ? MeshGeometry.box(width: 8, height: 8, depth: 1) : .plane(width: 8, height: 8)
            func trace(_ strength: Float) throws -> SIMD4<Float> {
                let material = try PhysicallyBasedMaterial(
                    roughness: thick ? 0 : 0.12, alphaMode: .opaque,
                    transmission: transmission, specular: .init(factor: 0), retroreflectivity: strength
                )
                let tracer = try self.tracer(
                    material, acceleration: acceleration, geometry: geometry,
                    rotation: simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
                )
                return try sample(tracer, lighting: lighting, samples: 1_024, bounces: 3)
            }
            let ordinary = try trace(0)
            let retro = try trace(1)
            #expect(ordinary.x > 0.48 && ordinary.w == 1, "\(ordinary)")
            #expect(simd_reduce_max(abs(ordinary - retro)) < 0.012, "\(ordinary) vs \(retro)")
        }
    }

    private func sample(
        _ tracer: PathTracer, lighting: PathTracer.Lighting, samples: Int, mis: Bool = true, bounces: Int = 1
    ) throws -> SIMD4<Float> {
        try fixture.trace(
            tracer, model: narrow, width: 1, height: 1, samples: samples,
            settings: .init(maxBounces: bounces, seed: 7, multipleImportanceSampling: mis), lighting: lighting
        )[0]
    }

    private func reference(
        light: SIMD3<Float>, base: SIMD3<Float>, roughness: Float, strength: Float
    ) -> SIMD3<Float> {
        let normal = tilt.act(SIMD3(0, 0, 1))
        let view = SIMD3<Float>(0, 0, 1)
        let retroView = 2 * simd_dot(normal, view) * normal - view
        let noL = max(0, simd_dot(normal, light))
        let alpha2 = roughness * roughness * roughness * roughness
        func ggx(_ outgoing: SIMD3<Float>) -> SIMD3<Float> {
            let halfway = simd_normalize(outgoing + light)
            let noH = simd_clamp(simd_dot(normal, halfway), 0, 1)
            let noV = simd_dot(normal, outgoing)
            let voH = simd_clamp(simd_dot(outgoing, halfway), 0, 1)
            let denominator = 1 - noH * noH + noH * noH * alpha2
            let distribution = alpha2 / (.pi * denominator * denominator)
            let visibility = 0.5 / (noL * sqrt(noV * noV * (1 - alpha2) + alpha2)
                                  + noV * sqrt(noL * noL * (1 - alpha2) + alpha2))
            let falloff = 1 - voH
            let fresnel = base + (1 - base) * (falloff * falloff * falloff * falloff * falloff)
            return fresnel * (distribution * visibility * noL)
        }
        return (1 - strength) * ggx(view) + strength * ggx(retroView)
    }

    private func tracer(
        _ material: PhysicallyBasedMaterial, acceleration: PathTracingScene.Options.Acceleration,
        geometry: MeshGeometry? = nil, rotation: simd_quatf? = nil
    ) throws -> PathTracer {
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: geometry ?? .plane(width: 8, height: 8), material: .physicallyBased(material)
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3), rotation: rotation ?? tilt), mesh: mesh)
        ], camera: fixture.base.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        return try PathTracer(queue: fixture.queue, scene: scene)
    }
}
