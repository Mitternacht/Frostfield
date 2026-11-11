import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathSamplingTests {
    private let fixture: PathTestFixture
    private let width = 32
    private let height = 24

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Scrambled Sobol sampling lowers area-light error against a converged reference at equal sample counts")
    func sobolConvergesFaster() throws {
        let (tracer, lighting) = try areaLitPlane()
        let reference = try fixture.trace(
            tracer, width: width, height: height, samples: 4_096,
            settings: .init(maxBounces: 2, seed: 1_000, sampler: .sobol), lighting: lighting
        )
        let seeds: [UInt32] = [1, 2, 3, 4]
        for samples in [16, 64] {
            var errors: [PathTracer.Sampler: Double] = [:]
            for sampler in [PathTracer.Sampler.independent, .sobol] {
                let total = try seeds.reduce(0.0) { sum, seed in
                    let estimate = try fixture.trace(
                        tracer, width: width, height: height, samples: samples,
                        settings: .init(maxBounces: 2, seed: seed, sampler: sampler), lighting: lighting
                    )
                    return sum + PathTestFixture.rmse(estimate, reference)
                }
                errors[sampler] = total / Double(seeds.count)
            }
            let independent = try #require(errors[.independent])
            let sobol = try #require(errors[.sobol])
            print("Path sampling RMSE at \(samples) spp: independent \(independent), sobol \(sobol), ratio \(sobol / independent)")
            #expect(sobol < 0.75 * independent, "\(samples) spp: independent \(independent), sobol \(sobol)")
        }
    }

    @Test("All samplers are deterministic, split batches match, and sampling settings own their history")
    func determinismAndHistory() throws {
        let (tracer, lighting) = try areaLitPlane()
        for sampler in [PathTracer.Sampler.independent, .sobol, .blueDither] {
            let settings = PathTracer.Settings(maxBounces: 2, seed: 21, sampler: sampler)
            let once = try fixture.trace(tracer, width: width, height: height, samples: 8, settings: settings, lighting: lighting)
            let again = try fixture.trace(tracer, width: width, height: height, samples: 8, settings: settings, lighting: lighting)
            #expect(once == again)
            var reseeded = settings
            reseeded.seed = 22
            let other = try fixture.trace(tracer, width: width, height: height, samples: 8, settings: reseeded, lighting: lighting)
            #expect(other != once)

            // One sample, then seven more, equals eight at once.
            let accumulation = try tracer.makeAccumulation(width: width, height: height)
            let target = try fixture.output(width: width, height: height)
            let first = try #require(fixture.queue.makeCommandBuffer())
            let history = try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                commandBuffer: first, samplesPerPixel: 1, settings: settings, lighting: lighting
            )
            try fixture.finish(first)
            let second = try #require(fixture.queue.makeCommandBuffer())
            for changed in [
                PathTracer.Settings(maxBounces: 2, seed: 21, sampler: sampler == .sobol ? .independent : .sobol),
                PathTracer.Settings(maxBounces: 2, seed: 21, sampler: sampler, maximumSampleRadiance: 4),
                PathTracer.Settings(maxBounces: 2, seed: 21, sampler: sampler, clampDirect: 4),
                PathTracer.Settings(maxBounces: 2, seed: 21, sampler: sampler, clampIndirect: 4),
                PathTracer.Settings(maxBounces: 2, seed: 21, sampler: sampler, multipleImportanceSampling: false),
            ] {
                #expect(throws: PathTracer.Failure.history) {
                    try tracer.encode(
                        camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                        previous: history, commandBuffer: second, settings: changed, lighting: lighting
                    )
                }
            }
            _ = try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: history, commandBuffer: second, samplesPerPixel: 7, settings: settings, lighting: lighting
            )
            try fixture.finish(second)
            #expect(PathTestFixture.maximumDifference(try fixture.readback(target), once) < 0.006)
        }
    }

    @Test("A radiance limit caps scattered contributions, keeps camera-visible emission, and validates")
    func fireflyClamp() throws {
        // A diffuse wall sees a small, bright panel only through an out-of-view mirror,
        // which next-event estimation cannot sample: a caustic that arrives as fireflies.
        let wall = try fixture.pbr(.plane(width: 6, height: 6), color: SIMD3(repeating: 0.8))
        let mirror = try fixture.pbr(.plane(width: 2, height: 2), color: SIMD3(repeating: 1), roughness: 0.02, metallic: 1)
        let patch = try fixture.pbr(.plane(width: 0.4, height: 0.4), color: .zero, emission: SIMD3(1, 0.8, 0.6))
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: wall),
            SceneNode(
                transform: .init(translation: SIMD3(0, 1, -1), rotation: simd_quatf(angle: 0.75 * .pi, axis: SIMD3(1, 0, 0))),
                mesh: mirror
            ),
            SceneNode(transform: .init(translation: SIMD3(-1, -0.6, -2.5)), mesh: patch),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        let panel = try PathTracer.AreaLight(
            center: SIMD3(0, 0.2, -1.4), aim: SIMD3(0, 1, -1), size: SIMD2(0.3, 0.3), radiance: SIMD3(repeating: 200),
            cameraVisible: false
        )
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lights: [panel])
        func render(_ limit: Float?) throws -> [SIMD4<Float>] {
            try fixture.trace(
                tracer, width: width, height: height, samples: 16,
                settings: .init(maxBounces: 2, seed: 17, maximumSampleRadiance: limit)
            )
        }
        let unclamped = try render(nil)
        let limit: Float = 1
        let clamped = try render(limit)
        let peak = { (image: [SIMD4<Float>]) in image.reduce(Float(0)) { max($0, simd_reduce_max(SIMD3($1.x, $1.y, $1.z))) } }
        let mean = { (image: [SIMD4<Float>]) in image.reduce(Float(0)) { $0 + $1.x } / Float(image.count) }
        print("Path clamp: unclamped peak \(peak(unclamped)) mean \(mean(unclamped)); clamped peak \(peak(clamped)) mean \(mean(clamped))")
        #expect(peak(unclamped) > 4 * limit)
        // Per sample, the wall's scattered panel hit and the mirror's next-event term are each capped.
        let patchPixel = pixel(SIMD3(-1, -0.6, -2.5))
        for (index, value) in clamped.enumerated() where index != patchPixel {
            #expect(simd_reduce_max(SIMD3(value.x, value.y, value.z)) <= 2 * limit + 0.002)
        }
        #expect(mean(clamped) > 0 && mean(clamped) < mean(unclamped))
        // The emissive patch seen by camera rays is exact even with a lower limit.
        let strict = try render(0.05)
        #expect(simd_reduce_max(abs(strict[patchPixel] - SIMD4(1, 0.8, 0.6, 1))) < 0.002)
        // A limit above every contribution changes nothing.
        #expect(try render(1e6) == unclamped)

        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try fixture.output(width: width, height: height)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        for invalid: Float in [0, -1, .nan, .infinity] {
            #expect(throws: PathTracer.Failure.settings) {
                try tracer.encode(
                    camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                    commandBuffer: buffer, settings: .init(maximumSampleRadiance: invalid)
                )
            }
        }
    }

    @Test("The direct clamp scales mean RGB for camera emission and one-scatter light")
    func directClamp() throws {
        let emission = SIMD3<Float>(1, 0.5, 0.25)
        let emitting = try fixture.pbr(.plane(width: 6, height: 6), color: .zero, emission: emission)
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: emitting)
        ], camera: fixture.camera)
        let tracer = try PathTracer(queue: fixture.queue, scene: PathTracingScene(queue: fixture.queue, graph: graph))
        let center = pixel(SIMD3(0, 0, -3))
        func sample(_ settings: PathTracer.Settings) throws -> SIMD4<Float> {
            try fixture.trace(tracer, width: width, height: height, samples: 4, settings: settings)[center]
        }
        let raw = try sample(.init(maxBounces: 1))
        let explicit = try sample(.init(maxBounces: 1, clampDirect: 0.125))
        let expected = SIMD4(emission * (0.375 / (1 + 0.5 + 0.25)), 1)
        #expect(raw == SIMD4(emission, 1))
        #expect(simd_distance(explicit, expected) < 0.001)
        #expect(try sample(.init(maxBounces: 1, clampIndirect: 0.125)) == raw)
        #expect(try sample(.init(maxBounces: 1, maximumSampleRadiance: 0.125)) == raw)

        let wall = try fixture.pbr(.plane(width: 6, height: 6))
        var litGraph = graph
        litGraph.roots[0].mesh = wall
        let light = try PathTracer.AnalyticLight(PunctualLight(kind: .point(), intensity: 50), position: SIMD3(0, 0, -1))
        let lit = try PathTracer(
            queue: fixture.queue, scene: PathTracingScene(queue: fixture.queue, graph: litGraph),
            lighting: .init(analyticLights: [light])
        )
        func litSample(_ settings: PathTracer.Settings) throws -> SIMD4<Float> {
            try fixture.trace(lit, width: width, height: height, samples: 4, settings: settings)[center]
        }
        let unbounded = try litSample(.init(maxBounces: 1))
        #expect(unbounded.x > 1)
        #expect(try litSample(.init(maxBounces: 1, clampIndirect: 0.125)) == unbounded)
        #expect(try litSample(.init(maxBounces: 1, multipleImportanceSampling: false)) == unbounded)
        #expect(try litSample(.init(maxBounces: 1, clampDirect: 0.125)) == SIMD4(0.125, 0.125, 0.125, 1))
        #expect(try litSample(.init(maxBounces: 1, maximumSampleRadiance: 0.125, clampDirect: 0)) == unbounded)
    }

    @Test("Two mirror reflections reach the indirect clamp while a direct-only limit leaves the path intact")
    func indirectClamp() throws {
        let mirror = try fixture.pbr(
            .plane(width: 1.5, height: 1.5), color: SIMD3(repeating: 1), roughness: 0.015, metallic: 1
        )
        let patch = try fixture.pbr(.plane(width: 1.5, height: 1.5), color: .zero, emission: SIMD3(1, 0.5, 0.25))
        let graph = SceneGraph(roots: [
            SceneNode(
                transform: .init(translation: SIMD3(0, 0, -2), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 1, 0))),
                mesh: mirror
            ),
            SceneNode(
                transform: .init(translation: SIMD3(2, 0, -2), rotation: simd_quatf(angle: -3 * .pi / 4, axis: SIMD3(0, 1, 0))),
                mesh: mirror
            ),
            SceneNode(transform: .init(translation: SIMD3(2, 0, -4)), mesh: patch),
        ], camera: fixture.camera)
        let tracer = try PathTracer(queue: fixture.queue, scene: PathTracingScene(queue: fixture.queue, graph: graph))
        let center = pixel(SIMD3(0, 0, -2))
        func sample(_ settings: PathTracer.Settings) throws -> SIMD4<Float> {
            try fixture.trace(tracer, width: width, height: height, samples: 8, settings: settings)[center]
        }
        let raw = try sample(.init(maxBounces: 2))
        #expect(raw.x > 0.75 && raw.w == 1, "\(raw)")
        #expect(try sample(.init(maxBounces: 1)) == SIMD4(0, 0, 0, 1))
        #expect(try sample(.init(maxBounces: 2, clampDirect: 0.125)) == raw)
        let limited = try sample(.init(maxBounces: 2, clampIndirect: 0.125))
        let expected = SIMD4(SIMD3<Float>(1, 0.5, 0.25) * (0.375 / (1 + 0.5 + 0.25)), 1)
        #expect(simd_distance(limited, expected) < 0.002, "\(limited)")
    }

    @Test("Disabling MIS keeps the converged energy of area and sampled environment light")
    func optionalMIS() throws {
        let (tracer, panel) = try areaLitPlane()
        let sky = try PathTracer.EnvironmentMap(
            device: fixture.device, pixels: Array(repeating: SIMD3<Float>(0.15, 0.25, 0.45), count: 128), width: 16, height: 8
        )
        let lighting = try PathTracer.Lighting(lights: panel.lights, sky: .map(sky))
        func render(_ enabled: Bool) throws -> [SIMD4<Float>] {
            try fixture.trace(
                tracer, width: width, height: height, samples: 256,
                settings: .init(maxBounces: 2, multipleImportanceSampling: enabled), lighting: lighting
            )
        }
        let combined = try render(true), scattered = try render(false)
        let sum = { (image: [SIMD4<Float>]) -> SIMD3<Float> in
            image.reduce(.zero) { $0 + SIMD3($1.x, $1.y, $1.z) }
        }
        let reference = sum(combined), estimate = sum(scattered)
        #expect(reference.x > 20 && scattered != combined)
        #expect(simd_reduce_max(abs(estimate / reference - 1)) < 0.06, "\(estimate) vs \(reference)")
    }

    @Test("GPU timestamps measure one tracing pass and guard their pending command buffer")
    func timing() throws {
        guard fixture.device.supportsCounterSampling(.atStageBoundary),
              fixture.device.counterSets?.contains(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) == true else {
            #expect(throws: PathTracer.Failure.timingUnsupported) { try PathTracer.Timing(device: fixture.device) }
            return
        }
        let (tracer, lighting) = try areaLitPlane()
        let timing = try PathTracer.Timing(device: fixture.device)
        #expect(timing.duration == nil)
        let accumulation = try tracer.makeAccumulation(width: 256, height: 256)
        let other = try tracer.makeAccumulation(width: 256, height: 256)
        let target = try fixture.output(width: 256, height: 256)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: buffer, samplesPerPixel: 16, settings: .init(maxBounces: 3), lighting: lighting, timing: timing
        )
        let pending = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.timingBusy) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: other,
                commandBuffer: pending, settings: .init(), lighting: lighting, timing: timing
            )
        }
        try fixture.finish(buffer)
        let duration = try #require(timing.duration)
        let commandBufferTime = buffer.gpuEndTime - buffer.gpuStartTime
        print("Path timing: pass \(duration * 1e3) ms, command buffer \(commandBufferTime * 1e3) ms")
        #expect(duration > 0)
        #expect(duration <= commandBufferTime * 1.05 + 1e-4)

        // After completion the timing measures the next pass.
        _ = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: other,
            commandBuffer: pending, settings: .init(maxBounces: 1), lighting: lighting, timing: timing
        )
        #expect(timing.duration == nil)
        try fixture.finish(pending)
        #expect(try #require(timing.duration) > 0)
    }

    // A diffuse plane lit by one area light, with a box casting a soft shadow.
    private func areaLitPlane() throws -> (PathTracer, PathTracer.Lighting) {
        let plane = try fixture.pbr(.plane(width: 4, height: 4), color: SIMD3(repeating: 0.8))
        let box = try fixture.pbr(.cube(), color: SIMD3(0.6, 0.5, 0.4))
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: plane),
            SceneNode(transform: .init(translation: SIMD3(0.3, 0.35, -2.3), scale: SIMD3(repeating: 0.4)), mesh: box),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        let lighting = try PathTracer.Lighting(lights: [try PathTracer.AreaLight(
            center: SIMD3(0, 1.4, -1.4), aim: SIMD3(0, 0, -3), size: SIMD2(1.2, 1.2), radiance: SIMD3(repeating: 8),
            cameraVisible: false
        )])
        return (try PathTracer(queue: fixture.queue, scene: scene, lighting: lighting), lighting)
    }

    private func pixel(_ point: SIMD3<Float>) -> Int {
        let tanY = Float(tan(Double.pi / 6))
        let tanX = tanY * Float(width) / Float(height)
        let column = Int(((point.x / -point.z) / tanX + 1) / 2 * Float(width))
        let row = Int((1 - (point.y / -point.z) / tanY) / 2 * Float(height))
        return row * width + column
    }
}
