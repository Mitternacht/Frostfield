@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathFiniteSpotLightTests {
    private let fixture: PathCameraFixture
    private let model = PathTracer.CameraModel(projection: .orthographic(height: 0.0001), near: 0.05, far: 10)
    private let receiver = SIMD3<Float>(0, 0, -4)
    private let view = SceneTransform(translation: SIMD3(0, 0, -3.4))

    init() throws { fixture = try PathCameraFixture() }

    @Test("A finite spot radius is validated and packed with its aperture offset")
    func validationAndPacking() throws {
        let light = try PunctualLight(kind: .spot(innerConeAngle: 0.3, outerConeAngle: 0.6))
        let analytic = try PathTracer.AnalyticLight(light, direction: SIMD3(2, -1, -3), spotRadius: 0.5)
        let packed = PathTracer.PackedAnalyticLight(analytic, profileLayer: 2, cookieLayer: 3)
        #expect(analytic.spotRadius == 0.5)
        #expect(abs(simd_length(analytic.direction) - 1) < 1e-6)
        #expect(MemoryLayout<PathTracer.PackedAnalyticLight>.stride == 96)
        #expect(packed.tables.x == 3 && packed.tables.z == 0.5)
        #expect(abs(packed.tables.y - tan(0.6)) < 1e-6)
        #expect(abs(packed.tables.w - 0.5 / tan(0.6)) < 1e-6)
        let delta = try PathTracer.AnalyticLight(light)
        #expect(delta.spotRadius == 0)
        #expect(PathTracer.PackedAnalyticLight(delta).tables.z == 0 && PathTracer.PackedAnalyticLight(delta).tables.w == 0)
        let narrow = try PunctualLight(kind: .spot(outerConeAngle: 1e-7))
        let capped = try PathTracer.AnalyticLight(narrow, spotRadius: 0.001)
        #expect(abs(PathTracer.PackedAnalyticLight(capped).tables.w - 100) < 1e-4)

        for radius: Float in [-1, .nan, .infinity, 1e12] {
            #expect(throws: PathTracer.Failure.invalidAnalyticLight) {
                try PathTracer.AnalyticLight(light, spotRadius: radius)
            }
        }
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) {
            try PathTracer.AnalyticLight(PunctualLight(kind: .point()), spotRadius: 0.1)
        }
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) {
            try PathTracer.AnalyticLight(light, direction: SIMD3(.nan, 0, -1), spotRadius: 0.1)
        }
    }

    @Test("A shifted aperture softens a blocker and ends shadow rays at the sampled disk",
          arguments: PathAccelerationTests.modes)
    func apertureAndShadows(acceleration: PathTracingScene.Options.Acceleration) throws {
        let outer: Float = .pi / 4
        let radius: Float = 0.6
        let intensity: Float = 10
        let albedo: Float = 0.8
        let spot = try PunctualLight(kind: .spot(innerConeAngle: 0.6, outerConeAngle: outer), intensity: intensity)
        let apex = SIMD3<Float>(0, 0, -1)
        let finite = try PathTracer.AnalyticLight(spot, position: apex, spotRadius: radius)
        let delta = try PathTracer.AnalyticLight(spot, position: apex)
        let wall = try fixture.lambert(.plane(width: 3, height: 3), albedo: albedo)
        let screen = SceneNode(transform: .init(translation: receiver), mesh: wall)
        let blocker = try fixture.lambert(.plane(width: 0.26, height: 2), albedo: 0)
        let facing = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        let partial = SceneNode(transform: .init(translation: SIMD3(0, 0, -2.6), rotation: facing), mesh: blocker)
        let backstop = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.25), rotation: facing), mesh: blocker)

        func render(_ obstacles: [SceneNode], _ light: PathTracer.AnalyticLight, samples: Int = 2_048) throws -> SIMD4<Float> {
            let tracer = try makeTracer(roots: [screen] + obstacles, acceleration: acceleration)
            return try trace(tracer, light, samples: samples)
        }
        let open = try render([], finite)
        let distance = 3 - radius / tan(outer)
        // A uniform aperture average of cos(theta) / distance² over the disk.
        let factor = 2 * (1 - distance / sqrt(distance * distance + radius * radius)) / (radius * radius)
        let expected = albedo * intensity * factor / .pi
        #expect(abs(open.x - expected) < 0.002, "\(open.x) vs \(expected)")
        let softened = try render([partial], finite)
        #expect(softened.x > open.x * 0.3 && softened.x < open.x * 0.8)
        #expect(try render([partial], delta, samples: 16) == SIMD4(0, 0, 0, 1))
        #expect(try render([backstop], finite) == open)
        #expect(try render([backstop], delta, samples: 16) == SIMD4(0, 0, 0, 1))
        let reverse = try PathTracer.AnalyticLight(spot, position: apex, direction: SIMD3(0, 0, 1), spotRadius: radius)
        #expect(try render([], reverse, samples: 16) == SIMD4(0, 0, 0, 1))
    }

    @Test("A rotated finite spotlight filters every aperture sample through its cone, IES and cookie")
    func angularTables() throws {
        let albedo: Float = 0.8
        let color = SIMD3<Float>(1, 0.8, 0.6)
        let inner: Float = 0.08, outer: Float = 0.62, range: Float = 2.8, radius: Float = 0.55
        let intensity: Float = 8
        let profile = try IESProfile(string: IESFixture.asymmetricDownlight())
        let texels = (0..<64).map { index -> SIMD3<Float> in
            let x = Float(index % 8) / 7, y = Float(index / 8) / 7
            return SIMD3(0.25 + 0.65 * x, 0.3 + 0.6 * y, 0.95 - 0.6 * x * y)
        }
        let cookie = try LightCookie(width: 8, height: 8, transmittance: texels)
        let light = try PunctualLight(
            kind: .spot(range: range, innerConeAngle: inner, outerConeAngle: outer),
            color: color, intensity: intensity, iesProfile: profile, cookie: cookie
        )
        let apex = SIMD3<Float>(0.7, 0.25, -1.1)
        let forward = simd_normalize(SIMD3<Float>(0.4, 0, -4) - apex)
        let orientation = simd_quatf(from: SIMD3(0, 0, -1), to: forward)
            * simd_quatf(angle: 0.8, axis: SIMD3(0, 0, 1))
        let source = SceneNode(transform: .init(translation: apex, rotation: orientation), light: light)
        let wall = try fixture.lambert(.plane(width: 3, height: 3), albedo: albedo)
        let screen = SceneNode(transform: .init(translation: receiver), mesh: wall)
        let graph = SceneGraph(roots: [screen, source], camera: fixture.base.camera)
        let resolved = try #require(try graph.resolve().first { $0.id == source.id }?.light)
        let finite = try PathTracer.AnalyticLight(resolved, spotRadius: radius)
        let right = orientation.act(SIMD3<Float>(1, 0, 0))
        let up = orientation.act(SIMD3<Float>(0, 1, 0))
        #expect(simd_distance(finite.right, right) < 1e-6)
        #expect(simd_distance(finite.direction, forward) < 1e-6)
        let tracer = try makeTracer(roots: [screen, source])
        let result = try trace(tracer, finite, samples: 4_096)

        let coneTangent = tan(outer)
        let center = apex + forward * (radius / coneTangent)
        let nearCosine = cos(inner), farCosine = cos(outer)
        var integrated = SIMD3<Float>.zero
        for radial in 0..<64 {
            let r = radius * sqrt((Float(radial) + 0.5) / 64)
            for angle in 0..<128 {
                let theta = 2 * Float.pi * (Float(angle) + 0.5) / 128
                let sample = center + right * (r * cos(theta)) + up * (r * sin(theta))
                let delta = receiver - sample
                let distance = simd_length(delta)
                let ray = delta / distance
                let x = simd_dot(ray, right), y = simd_dot(ray, up), z = simd_dot(ray, forward)
                let cone = simd_clamp((z - farCosine) / (nearCosine - farCosine), 0, 1)
                let rangeRatio = distance / range
                let window = max(0, 1 - rangeRatio * rangeRatio * rangeRatio * rangeRatio)
                let ies = profile.intensity(toward: SIMD3(x, y, -z)) / profile.maximumIntensity
                let gobo = cookie.transmittance(u: 0.5 + 0.5 * x / (z * coneTangent),
                                               v: 0.5 - 0.5 * y / (z * coneTangent))
                let weight = (-ray.z) * window * cone * cone * ies / (distance * distance)
                integrated += gobo * weight
            }
        }
        let expected = integrated * color * (albedo * intensity / (.pi * 64 * 128))
        let measured = SIMD3(result.x, result.y, result.z)
        #expect(simd_reduce_min(expected) > 0.025)
        #expect(simd_reduce_max(abs(measured / expected - 1)) < 0.012, "\(measured) vs \(expected)")
    }

    @Test("A finite spot stays invisible to camera rays and a radius change invalidates history")
    func invisibleAndHistory() throws {
        let dummy = try fixture.lambert(.plane(width: 1, height: 1), albedo: 0)
        let tracer = try makeTracer(roots: [
            SceneNode(transform: .init(translation: SIMD3(10, 0, -4)), mesh: dummy)
        ])
        let spot = try PunctualLight(kind: .spot(), intensity: 8)
        let light = try PathTracer.AnalyticLight(spot, position: SIMD3(0, 0, -2), direction: SIMD3(0, 0, 1), spotRadius: 0.5)
        let changed = try PathTracer.AnalyticLight(spot, position: light.position, direction: light.direction, spotRadius: 0.6)
        let target = try fixture.base.output(width: 1, height: 1)
        let accumulation = try tracer.makeAccumulation(width: 1, height: 1)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let history = try tracer.encode(
            cameraModel: model, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: command, lighting: .init(analyticLights: [light], backgroundVisible: false)
        )
        try fixture.base.finish(command)
        #expect(try fixture.base.readback(target)[0] == .zero)
        let next = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.lightingChanged) {
            try tracer.encode(cameraModel: model, cameraTransform: .init(), destination: target, accumulation: accumulation,
                              previous: history, commandBuffer: next,
                              lighting: .init(analyticLights: [changed], backgroundVisible: false))
        }
    }

    private func makeTracer(
        roots: [SceneNode], acceleration: PathTracingScene.Options.Acceleration = .software
    ) throws -> PathTracer {
        let graph = SceneGraph(roots: roots, camera: fixture.base.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        return try PathTracer(queue: fixture.queue, scene: scene)
    }

    private func trace(_ tracer: PathTracer, _ light: PathTracer.AnalyticLight, samples: Int) throws -> SIMD4<Float> {
        try fixture.trace(
            tracer, model: model, transform: view, width: 1, height: 1, samples: samples,
            lighting: .init(analyticLights: [light])
        )[0]
    }
}
