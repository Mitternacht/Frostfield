import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
struct PathSceneEnvironmentTests {
    @Test("Scene sky preserves orientation and importance weights against a directly authored panorama")
    func cubeToPanorama() throws {
        let gpu = try PathTestFixture()
        let (source, direct) = try environments(gpu)
        var graph = SceneGraph(roots: [], camera: gpu.camera)
        graph.environment = source
        let lighting = try PathTracer.Lighting(queue: gpu.queue, graph: graph)
        let converted = try #require(sceneMap(lighting))
        #expect(converted.width == 128 && converted.height == 64 && converted.isSampled)
        let expected = try gpu.readback(direct.texture)
        let actual = try gpu.readback(converted.texture)
        #expect(PathTestFixture.maximumDifference(actual, expected) < 0.006)
        #expect(actual.allSatisfy { $0.w == 1 })

        // Compare the weighted densities at texel centers around the sphere.
        for row in stride(from: 1, to: converted.height, by: 8) {
            for column in stride(from: 1, to: converted.width, by: 8) {
                let phi = ((Float(column) + 0.5) / Float(converted.width) - 0.5) * 2 * .pi
                let theta = (Float(row) + 0.5) / Float(converted.height) * .pi
                let direction = SIMD3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi))
                let density = direct.pdf(direction: direction)
                #expect(abs(converted.pdf(direction: direction) - density) < 0.01 * density)
            }
        }
        let sample = try #require(converted.sample(SIMD2(0.21, 0.74), yaw: 0.7))
        #expect(abs(converted.pdf(direction: sample.direction, yaw: 0.7) - sample.pdf) < 0.00001)
    }

    @Test("XYZ importance samples and reciprocal PDFs agree with pinned r187 rotation")
    func rotatedImportance() throws {
        let gpu = try PathTestFixture()
        let (_, map) = try environments(gpu)
        let reference = Self.referenceLocalFromWorld
        for random: SIMD2<Float> in [SIMD2(0.21, 0.74), SIMD2(0.58, 0.3), SIMD2(0.89, 0.63), SIMD2(0.07, 0.88)] {
            let original = try #require(map.sample(random))
            let rotated = try #require(map.sample(random, eulerRotation: Self.fullRotation))
            #expect(simd_length(reference.transpose * original.direction - rotated.direction) < 1e-6)
            #expect(rotated.pdf == original.pdf)
            let density = map.pdf(direction: rotated.direction, eulerRotation: Self.fullRotation)
            #expect(abs(density - original.pdf) < 0.00001)
        }
        #expect(map.pdf(direction: SIMD3(0, 0, 1), eulerRotation: SIMD3(.nan, 0, 0)) == 0)
        #expect(map.sample(SIMD2(0.5, 0.5), eulerRotation: SIMD3(0, 0, .infinity)) == nil)
    }

    @Test("Scene snapshots preserve explicit skies and reuse map identity and accumulation")
    func snapshots() throws {
        let gpu = try PathTestFixture()
        let (source, direct) = try environments(gpu)
        var graph = try sphereScene(gpu)
        graph.environment = source
        graph.roots.append(try SceneNode(
            transform: .init(translation: SIMD3(0, 4, -2)), light: PunctualLight(kind: .rectangle(width: 2, height: 1))
        ))
        graph.roots.append(try SceneNode(light: PunctualLight(kind: .point(), intensity: 2)))
        let first = try PathTracer.Lighting(queue: gpu.queue, graph: graph)
        let repeated = try PathTracer.Lighting(queue: gpu.queue, graph: graph)
        #expect(first == repeated)
        #expect(first.lights.count == 1 && first.analyticLights.count == 1)

        graph.environmentIntensity = 0.5
        graph.environmentRotation = 0.31
        let adjusted = try PathTracer.Lighting(queue: gpu.queue, graph: graph, backgroundVisible: false)
        let cached = try #require(sceneMap(first))
        #expect(adjusted.sky == .map(cached, intensity: 0.5, yaw: 0.31))
        #expect(!adjusted.backgroundVisible && adjusted != first)
        #expect(adjusted.skyEulerRotation == SIMD3(0, 0.31, 0))
        graph.environmentEulerRotation = Self.additionalRotation
        let rotated = try PathTracer.Lighting(queue: gpu.queue, graph: graph, backgroundVisible: false)
        #expect(rotated.sky == adjusted.sky && rotated != adjusted)
        let effective = Self.additionalRotation + SIMD3<Float>(0, graph.environmentRotation, 0)
        #expect(rotated.skyEulerRotation == effective)
        graph.environmentEulerRotation.x = .nan
        #expect(throws: PathTracer.Failure.invalidLighting) { try PathTracer.Lighting(queue: gpu.queue, graph: graph) }
        graph.environmentIntensity = -.infinity
        #expect(throws: PathTracer.Failure.invalidLighting) { try PathTracer.Lighting(queue: gpu.queue, graph: graph) }
        let explicit = try PathTracer.Lighting(
            queue: gpu.queue, graph: graph, sky: .map(direct, intensity: 2, yaw: -0.7), backgroundVisible: false
        )
        #expect(explicit.sky == .map(direct, intensity: 2, yaw: -0.7) && !explicit.backgroundVisible)
        #expect(explicit.skyEulerRotation == SIMD3(0, -0.7, 0))
        #expect(try PathTracer.Lighting(queue: gpu.queue, graph: graph, sky: .constant(.zero)).sky == .constant(.zero))
        graph.environment = nil
        #expect(try PathTracer.Lighting(queue: gpu.queue, graph: graph).sky == .constant(.zero))

        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: first)
        let target = try gpu.output(width: 9, height: 9)
        let accumulation = try tracer.makeAccumulation(width: 9, height: 9)
        let settings = PathTracer.Settings(maxBounces: 2)
        func step(
            previous: PathTracer.History?, lighting: PathTracer.Lighting
        ) throws -> PathTracer.History {
            let command = try #require(gpu.queue.makeCommandBuffer())
            let history = try tracer.encode(
                camera: gpu.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: previous, commandBuffer: command, settings: settings, lighting: lighting
            )
            try gpu.finish(command)
            return history
        }
        let initial = try step(previous: nil, lighting: first)
        let continued = try step(previous: initial, lighting: repeated)
        #expect(continued.sampleCount == 2)
        #expect(throws: PathTracer.Failure.lightingChanged) { try step(previous: continued, lighting: adjusted) }
        let reset = try step(previous: nil, lighting: adjusted)
        #expect(throws: PathTracer.Failure.lightingChanged) { try step(previous: reset, lighting: rotated) }
    }

    @Test("Software BVH uses scene sky for camera radiance and importance-sampled surface energy", arguments: [false, true])
    func software(euler: Bool) throws { try compareTrace(acceleration: .software, euler: euler) }

    @Test("Metal intersector uses scene sky for camera radiance and importance-sampled surface energy",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true, "Requires Metal ray tracing"),
          arguments: [false, true])
    func metal(euler: Bool) throws { try compareTrace(acceleration: .metal, euler: euler) }

    private func compareTrace(acceleration: PathTracingScene.Options.Acceleration, euler: Bool) throws {
        let gpu = try PathTestFixture()
        let (source, direct) = try environments(gpu)
        var graph = try sphereScene(gpu)
        graph.environment = source
        graph.environmentIntensity = 1.2
        graph.environmentRotation = euler ? 0.22 : .pi / 2
        graph.environmentEulerRotation = euler ? Self.additionalRotation : .zero
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: acceleration))
        #expect(scene.accelerationKind == acceleration)
        let inherited = try PathTracer.Lighting(queue: gpu.queue, graph: graph)
        let sky: PathTracer.Environment
        if euler {
            let baked = try PathTracer.EnvironmentMap(device: gpu.device, width: 128, height: 64) { texel in
                Self.directionalColor(Self.referenceLocalFromWorld * texel.direction)
            }
            sky = .map(baked, intensity: 1.2)
        } else {
            sky = .map(direct, intensity: 1.2, yaw: .pi / 2)
        }
        let authored = try PathTracer.Lighting(queue: gpu.queue, graph: graph, sky: sky)
        let hidden = try PathTracer.Lighting(queue: gpu.queue, graph: graph, backgroundVisible: false)
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: inherited)
        let width = 25
        let height = 19
        let settings = PathTracer.Settings(maxBounces: 3, seed: 103)
        func trace(_ lighting: PathTracer.Lighting) throws -> [SIMD4<Float>] {
            try gpu.trace(tracer, width: width, height: height, samples: euler ? 384 : 96, settings: settings, lighting: lighting)
        }

        let converted = try trace(inherited)
        let reference = try trace(authored)
        let transparent = try trace(hidden)
        for index in [0, width - 1, width * (height - 1), width * height - 1] {
            #expect(isNearly(converted[index], reference[index], tolerance: euler ? 0.012 : 0.006))
            #expect(converted[index].w == 1)
            if !euler { #expect(converted[index].x > converted[index].z) }
            #expect(transparent[index] == .zero)
        }

        var sumConverted = SIMD3<Float>.zero
        var sumReference = SIMD3<Float>.zero
        for y in 7...11 {
            for x in 10...14 {
                let index = y * width + x
                let a = converted[index]
                let b = reference[index]
                #expect(a == transparent[index] && a.w == 1 && b.w == 1)
                sumConverted += SIMD3(a.x, a.y, a.z)
                sumReference += SIMD3(b.x, b.y, b.z)
            }
        }
        #expect(simd_reduce_min(sumReference / 25) > 0.2)
        #expect(simd_reduce_max(abs(sumConverted - sumReference) / sumReference) < (euler ? 0.04 : 0.02))
    }

    private func sphereScene(_ gpu: PathTestFixture) throws -> SceneGraph {
        let sphere = try gpu.pbr(.uvSphere(radius: 1, segments: 24, rings: 16))
        return SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: sphere),
        ], camera: gpu.camera)
    }

    private func sceneMap(_ lighting: PathTracer.Lighting) -> PathTracer.EnvironmentMap? {
        if case .map(let map, _, _) = lighting.sky { return map }
        return nil
    }

    private func environments(
        _ gpu: PathTestFixture
    ) throws -> (EnvironmentLighting, PathTracer.EnvironmentMap) {
        let width = 128
        let height = 64
        let direct = try PathTracer.EnvironmentMap(device: gpu.device, width: width, height: height) { texel in
            Self.directionalColor(texel.direction)
        }
        let source = try EnvironmentLighting(
            queue: gpu.queue, equirectangular: direct.texture,
            settings: .init(cubeSize: 32, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        )
        return (source, direct)
    }

    private static let additionalRotation = SIMD3<Float>(0.68, 0.60, -0.71)
    private static let fullRotation = SIMD3<Float>(0.68, 0.82, -0.71)

    // r187 Matrix4.makeRotationFromEuler(new Euler(0.68, 0.82, -0.71)).transpose().
    private static let referenceLocalFromWorld = simd_float3x3(
        SIMD3(0.51737055, 0.44469482, 0.73114583), SIMD3(-0.15819933, 0.88935517, -0.42897594),
        SIMD3(-0.84101170, 0.10627274, 0.53047660)
    )

    private static func directionalColor(_ local: SIMD3<Float>) -> SIMD3<Float> {
        let x = (1 + local.x) * 0.5
        return SIMD3(0.2 + 1.4 * x * x, 0.6 + 0.4 * local.y, 0.8 + 0.55 * local.z)
    }

}
