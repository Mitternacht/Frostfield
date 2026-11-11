import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathShapedAreaLightTests {
    private let fixture: PathCameraFixture
    private let narrow = PathTracer.CameraModel(projection: .orthographic(height: 0.0001), near: 0.1, far: 10)

    init() throws { fixture = try PathCameraFixture() }

    @Test("Aimed disk lights interpret dimensions as diameters and normalize front hemispheric power")
    func pathDiskLightPower() throws {
        let power = SIMD3<Float>(2 * .pi * .pi, 4 * .pi * .pi, 0)
        let disk = try PathTracer.AreaLight(
            center: SIMD3(0, 2, 0), aim: .zero, size: SIMD2(2, 4),
            radiantPower: power, shape: .disk, cameraVisible: false
        )
        let rectangle = try PathTracer.AreaLight(
            center: SIMD3(0, 2, 0), aim: .zero, size: SIMD2(2, 4), radiantPower: power, cameraVisible: false
        )
        #expect(disk.shape == .disk && rectangle.shape == .rectangle)
        #expect(!disk.cameraVisible)
        #expect(disk.halfU == rectangle.halfU && disk.halfV == rectangle.halfV)
        #expect(simd_length(disk.halfU) == 1 && simd_length(disk.halfV) == 2)
        #expect(simd_normalize(simd_cross(disk.halfU, disk.halfV)) == SIMD3(0, -1, 0))
        #expect(simd_distance(disk.radiance, SIMD3(1, 2, 0)) < 1e-6)
        #expect(abs(disk.radiance.x / rectangle.radiance.x - 4 / .pi) < 1e-6)
        #expect(throws: PathTracer.Failure.invalidAreaLight) {
            try PathTracer.AreaLight(center: .zero, halfU: SIMD3(1, 0, 0), halfV: SIMD3(1, 1, 0),
                                     radiance: SIMD3(repeating: 1), shape: .disk)
        }
    }

    @Test("A visible ellipse has matching beauty, guide and emission silhouettes in both traversers",
          arguments: PathAccelerationTests.modes)
    func ellipseVisibility(acceleration: PathTracingScene.Options.Acceleration) throws {
        let dummy = try fixture.lambert(.plane(width: 2, height: 2), albedo: 0)
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(20, 0, -4)), mesh: dummy)
        ], camera: fixture.base.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let side = 65
        let center = 32 * side + 32
        let corner = 22 * side + 48
        let model = PathTracer.CameraModel(projection: .orthographic(height: 4), near: 0.1, far: 10)
        let radiance = SIMD3<Float>(4, 0.5, 0.25)

        func light(_ shape: PathTracer.AreaLight.Shape, visible: Bool = true, front: Bool = true) throws -> PathTracer.AreaLight {
            try .init(
                center: SIMD3(0, 0, -3), halfU: SIMD3(1.25, 0, 0), halfV: SIMD3(0, front ? 0.75 : -0.75, 0),
                radiance: radiance, shape: shape, cameraVisible: visible
            )
        }
        let target = try fixture.base.output(width: side, height: side)
        let guides = PathTracer.Guides(
            normalDepth: try fixture.base.output(width: side, height: side),
            albedoRoughness: try fixture.base.output(width: side, height: side)
        )
        let emission = try fixture.base.output(width: side, height: side)
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let disk = try PathTracer.Lighting(lights: [light(.disk)], backgroundVisible: false)
        let history = try tracer.encode(
            cameraModel: model, cameraTransform: .init(), destination: target, accumulation: accumulation,
            guides: guides, commandBuffer: command, samplesPerPixel: 8, lighting: disk, aovs: .init(emission: emission)
        )
        try fixture.base.finish(command)
        let pixels = try fixture.base.readback(target)
        let normals = try fixture.base.readback(guides.normalDepth)
        let emitted = try fixture.base.readback(emission)
        #expect(pixels[center] == SIMD4(radiance, 1))
        #expect(normals[center] == SIMD4(0, 0, 1, 3))
        #expect(emitted[center] == SIMD4(radiance, 1))
        #expect(pixels[corner] == .zero && normals[corner] == .zero && emitted[corner] == .zero)
        #expect(pixels[32 * side + 48] == SIMD4(radiance, 1))
        #expect(pixels[16 * side + 32] == .zero)

        let rectangle = try PathTracer.Lighting(lights: [light(.rectangle)], backgroundVisible: false)
        let next = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.lightingChanged) {
            try tracer.encode(cameraModel: model, cameraTransform: .init(), destination: target, accumulation: accumulation,
                              previous: history, commandBuffer: next, lighting: rectangle)
        }
        func render(_ lighting: PathTracer.Lighting) throws -> [SIMD4<Float>] {
            try fixture.trace(tracer, model: model, width: side, height: side, samples: 8, lighting: lighting)
        }
        #expect(try render(rectangle)[corner] == SIMD4(radiance, 1))
        let hidden = try PathTracer.Lighting(lights: [light(.disk, visible: false)], backgroundVisible: false)
        #expect(try render(hidden)[center] == .zero)
        let back = try PathTracer.Lighting(lights: [light(.disk, front: false)], backgroundVisible: false)
        #expect(try render(back)[center] == SIMD4(0, 0, 0, 1))
    }

    @Test("Disk light sampling and BSDF-hit MIS agree with analytic on-axis diffuse irradiance",
          arguments: [SIMD2<Float>(1, 1), SIMD2<Float>(1.4, 0.6)])
    func diskIrradiance(radii: SIMD2<Float>) throws {
        let tracer = try receiverTracer(albedo: 0.8)
        let radiance = SIMD3<Float>(2, 0.5, 1)
        let disk = try PathTracer.AreaLight(
            center: SIMD3(0, 0, -2), halfU: SIMD3(radii.x, 0, 0), halfV: SIMD3(0, -radii.y, 0),
            radiance: radiance, shape: .disk, cameraVisible: false
        )
        let lighting = try PathTracer.Lighting(lights: [disk])
        let pixels = try fixture.trace(tracer, model: narrow, width: 1, height: 1, samples: 4_096, lighting: lighting)
        // At unit separation, integrate the cosine-weighted ellipse over the plane.
        let fraction = radii.x * radii.y / sqrt((1 + radii.x * radii.x) * (1 + radii.y * radii.y))
        let expected = SIMD4(radiance * (0.8 * fraction), 1)
        #expect(simd_reduce_max(abs(pixels[0] - expected)) < 0.006, "\(pixels[0]) vs \(expected)")
    }

    @Test("A disk occludes direct light inside its rim and passes light through the rectangle-only corners")
    func diskShadows() throws {
        let tracer = try receiverTracer(albedo: 0.7)
        let point = try PathTracer.AnalyticLight(PunctualLight(kind: .point(), intensity: 12), position: SIMD3(0, 0, -1))
        func blocker(_ shape: PathTracer.AreaLight.Shape, shift: Float) throws -> PathTracer.AreaLight {
            try .init(
                center: SIMD3(-shift, -shift, -2), halfU: SIMD3(1, 0, 0), halfV: SIMD3(0, 1, 0),
                radiance: .zero, shape: shape, cameraVisible: false
            )
        }
        func render(_ lights: [PathTracer.AreaLight]) throws -> SIMD4<Float> {
            try fixture.trace(
                tracer, model: narrow, width: 1, height: 1, samples: 16,
                lighting: .init(lights: lights, analyticLights: [point])
            )[0]
        }
        let open = try render([])
        #expect(open.x > 0.1)
        #expect(try render([blocker(.disk, shift: 0.8)]) == open)
        #expect(try render([blocker(.rectangle, shift: 0.8)]) == SIMD4(0, 0, 0, 1))
        #expect(try render([blocker(.disk, shift: 0.5)]) == SIMD4(0, 0, 0, 1))
    }

    private func receiverTracer(albedo: Float) throws -> PathTracer {
        let receiver = try fixture.lambert(.plane(width: 4, height: 4), albedo: albedo)
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: receiver)],
                               camera: fixture.base.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        return try PathTracer(queue: fixture.queue, scene: scene)
    }
}
