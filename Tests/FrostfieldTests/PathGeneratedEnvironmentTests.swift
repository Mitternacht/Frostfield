import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
struct PathGeneratedEnvironmentTests {
    private let gpu: PathTestFixture

    init() throws { gpu = try PathTestFixture() }

    @Test("Procedural HDR uses top-left texel centers, exact pixels, and the existing importance distribution")
    func procedural() throws {
        let width = 8, height = 4
        var visited: [PathTracer.EnvironmentMap.Texel] = []
        let generated = try PathTracer.EnvironmentMap(device: gpu.device, width: width, height: height) { texel in
            visited.append(texel)
            return SIMD3(Float(texel.coordinates.x) * 4 + 1, Float(texel.coordinates.y) + 0.25, 2)
        }
        let authored = try PathTracer.EnvironmentMap(
            device: gpu.device,
            pixels: (0..<(width * height)).map { SIMD3(Float($0 % width) * 4 + 1, Float($0 / width) + 0.25, 2) },
            width: width, height: height
        )
        #expect(visited.count == width * height)
        for (index, texel) in visited.enumerated() {
            #expect(texel.coordinates == SIMD2(index % width, index / width))
            #expect(texel.dimensions == SIMD2(width, height))
            #expect(texel.uv == (SIMD2(Float(index % width), Float(index / width)) + 0.5) / SIMD2(Float(width), Float(height)))
            #expect(abs(simd_length(texel.direction) - 1) < 1e-6)
            #expect(generated.pdf(direction: texel.direction) == authored.pdf(direction: texel.direction))
        }
        #expect(visited[3].direction.y > 0.9 && visited[3].direction.x > 0 && visited[3].direction.z < 0)
        #expect(visited[4].direction.z > 0 && visited[31].direction.y < -0.9)
        let a = try #require(generated.sample(SIMD2(0.37, 0.69)))
        let b = try #require(authored.sample(SIMD2(0.37, 0.69)))
        #expect(a.direction == b.direction && a.pdf == b.pdf)

        let tracer = try makeTracer()
        let texel = visited[12]
        let expected = SIMD4<Float>(17, 1.25, 2, 1)
        let pixels = try trace(
            tracer, pose: look(toward: texel.direction), lighting: .init(sky: .map(generated))
        )
        #expect(pixels.allSatisfy { simd_distance($0, expected) < 0.025 })
    }

    @Test("Gradient generation matches the latitude formula and rejects invalid inputs before callbacks")
    func gradient() throws {
        let gradient = PathTracer.EnvironmentMap.VerticalGradient(top: SIMD3(12, 8, 4), bottom: SIMD3(1, 2, 3))
        let map = try PathTracer.EnvironmentMap(device: gpu.device, width: 8, height: 4, gradient: gradient)
        var pixels = [SIMD4<Float16>](repeating: .zero, count: 32)
        pixels.withUnsafeMutableBytes {
            map.texture.getBytes($0.baseAddress!, bytesPerRow: 8 * 8, from: MTLRegionMake2D(0, 0, 8, 4), mipmapLevel: 0)
        }
        for row in 0..<4 {
            let y = cos((Double(row) + 0.5) * Double.pi / 4)
            let t = pow(0.5 + 0.5 * y, 2)
            let expected = SIMD4(SIMD3<Float16>(SIMD3<Double>(1, 2, 3) + SIMD3<Double>(11, 6, 1) * t), 1)
            #expect(pixels[(row * 8)..<(row * 8 + 8)].allSatisfy { $0 == expected })
        }
        let y: Float = 0.5
        let top = map.pdf(direction: SIMD3(sqrt(1 - y * y), y, 0))
        let bottom = map.pdf(direction: SIMD3(sqrt(1 - y * y), -y, 0))
        #expect(top > bottom)
        #expect(throws: PathTracer.Failure.invalidEnvironmentMap) {
            try PathTracer.EnvironmentMap(device: gpu.device, width: 8, height: 4, gradient: .init(exponent: -.infinity))
        }
        #expect(throws: PathTracer.Failure.invalidEnvironmentMap) {
            try PathTracer.EnvironmentMap(device: gpu.device, width: 8, height: 4, gradient: .init(bottom: SIMD3(0, -1, 0), exponent: 0))
        }
        var invoked = false
        #expect(throws: PathTracer.Failure.invalidEnvironmentMap) {
            try PathTracer.EnvironmentMap(device: gpu.device, width: 8, height: 8) { _ in invoked = true; return .zero }
        }
        #expect(!invoked)
        #expect(throws: PathTracer.Failure.invalidEnvironmentMap) {
            try PathTracer.EnvironmentMap(device: gpu.device, width: 8, height: 4) { _ in SIMD3(.nan, 0, 0) }
        }
        enum GenerationFailure: Error, Equatable { case interrupted }
        #expect(throws: GenerationFailure.interrupted) {
            try PathTracer.EnvironmentMap(device: gpu.device, width: 8, height: 4) { _ in throw GenerationFailure.interrupted }
        }
        let black = try PathTracer.EnvironmentMap(device: gpu.device, width: 8, height: 4) { _ in .zero }
        #expect(!black.isSampled && black.sample(SIMD2(repeating: 0.5)) == nil)
    }

    @Test("The spectral beam appears as one finite solar disk with the expected HDR radiance")
    func solarDisk() throws {
        let model = try SkyModel(sunDirection: SIMD3(0, 0.6, -0.8), luminanceScale: 3e-6)
        let rig = try LightRig(lights: [model.sunLight(castsShadow: false)])
        let lighting = try rig.pathLighting()
        #expect(lighting.analyticLights.count == 1 && lighting.analyticLights.first?.diskVisible == true)
        let tracer = try makeTracer()
        let axis = model.sunDirection
        let expected = model.sunIlluminance / Float(.pi * pow(sin(model.sunAngularRadius), 2))
        let centered = try trace(tracer, pose: look(toward: axis), lighting: lighting)
        #expect(centered.allSatisfy {
            simd_reduce_max(abs(SIMD3($0.x, $0.y, $0.z) - expected) / expected) < 0.002 && $0.w == 1
        })
        let outside = try trace(tracer, pose: look(toward: simd_normalize(axis + SIMD3(0.02, 0, 0))), lighting: lighting)
        #expect(outside.allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
        let hidden = try PathTracer.Lighting(analyticLights: lighting.analyticLights, backgroundVisible: false)
        #expect(try trace(tracer, pose: look(toward: axis), lighting: hidden).allSatisfy { $0 == .zero })

        let blocker = try makeTracer(pose: .init(
            translation: 2 * axis, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: -axis)
        ))
        #expect(try trace(blocker, pose: look(toward: axis), lighting: lighting).allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
    }

    @Test("A visible directional disk reflects from a mirror without doubling diffuse irradiance")
    func scatteredDisk() throws {
        let source = try PunctualLight(kind: .directional, color: SIMD3(1, 0.5, 0.25), intensity: .pi)
        func lighting(visible: Bool) throws -> PathTracer.Lighting {
            try .init(analyticLights: [
                .init(source, direction: SIMD3(0, 0, -1), angularDiameter: .pi / 2, diskVisible: visible)
            ])
        }
        let mirror = try makeTracer(
            material: .init(metallic: 1, roughness: 0, baseColorFactor: SIMD4(0.75, 0.75, 0.75, 1), alphaMode: .opaque),
            pose: .init(translation: SIMD3(0, 0, -2))
        )
        let visible = try lighting(visible: true), invisible = try lighting(visible: false)
        let reflection = try trace(mirror, samples: 16, lighting: visible)
        let hiddenReflection = try trace(mirror, samples: 16, lighting: invisible)
        // A matching constant background gives the same radiance through the
        // near-delta mirror. Direct-only uniform disk samples are very noisy
        // for this lobe and typically miss it, so compare their median.
        let ideal = try trace(mirror, samples: 16, lighting: .init(environment: SIMD3(2, 1, 0.5)))
        let mirrorError = PathTestFixture.rmse(reflection, ideal)
        let directMedian = hiddenReflection.map(\.x).sorted()[hiddenReflection.count / 2]
        #expect(mirrorError < 0.05, "\(mirrorError)")
        #expect(directMedian < 0.01, "\(directMedian)")
        let reflectionMean = reflection.map(\.x).reduce(0, +) / Float(reflection.count)
        #expect(abs(reflectionMean - 1.5) < 0.025)

        let diffuse = try makeTracer(
            material: .init(roughness: 1, alphaMode: .opaque, specular: .init(factor: 0)),
            pose: .init(translation: SIMD3(0, 0, -2))
        )
        let sampled = try trace(diffuse, size: 16, samples: 256, lighting: visible)
        let reference = try trace(diffuse, size: 16, samples: 256, lighting: invisible)
        let a = sampled.map(\.x).reduce(0, +) / Float(sampled.count)
        let b = reference.map(\.x).reduce(0, +) / Float(reference.count)
        #expect(abs(a - 1) < 0.012 && abs(b - 1) < 0.012, "\(a) visible versus \(b) direct only")
        #expect(abs(a - b) < 0.012)

        // The incident directional sample exceeds half range, but the
        // reflected value fits. Both MIS techniques must use the same beam.
        let brightSource = try PunctualLight(kind: .directional, intensity: 100_000)
        let bright = try PathTracer.Lighting(analyticLights: [
            .init(brightSource, direction: SIMD3(0, 0, -1), angularDiameter: .pi / 2, diskVisible: true)
        ])
        let brightPixels = try trace(diffuse, size: 12, samples: 128, lighting: bright)
        let measured = brightPixels.map(\.x).reduce(0, +) / Float(brightPixels.count)
        #expect(abs(measured / (100_000 / .pi) - 1) < 0.012, "\(measured)")
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) { try PathTracer.AnalyticLight(source, diskVisible: true) }
    }

    private func look(toward direction: SIMD3<Float>) -> SceneTransform {
        .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: direction))
    }

    private func makeTracer(
        material: PhysicallyBasedMaterial? = nil, pose: SceneTransform = .init(translation: SIMD3(10, 0, -3))
    ) throws -> PathTracer {
        let surface = try material ?? .init(baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque, shadingModel: .unlit)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(surface)
        )
        let graph = SceneGraph(roots: [SceneNode(transform: pose, mesh: mesh)], camera: gpu.camera)
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        return try PathTracer(queue: gpu.queue, scene: scene)
    }

    private func trace(
        _ tracer: PathTracer, pose: SceneTransform = .init(), size: Int = 3, samples: Int = 1, lighting: PathTracer.Lighting
    ) throws -> [SIMD4<Float>] {
        let accumulation = try tracer.makeAccumulation(width: size, height: size)
        let target = try gpu.output(width: size, height: size)
        let camera = PathTracer.CameraModel(projection: .orthographic(height: 1))
        var history: PathTracer.History?
        for offset in stride(from: 0, to: samples, by: 16) {
            let command = try #require(gpu.queue.makeCommandBuffer())
            history = try tracer.encode(
                cameraModel: camera, cameraTransform: pose, destination: target, accumulation: accumulation,
                previous: history, commandBuffer: command, samplesPerPixel: min(16, samples - offset),
                settings: .init(maxBounces: 1, seed: 19), lighting: lighting
            )
            try gpu.finish(command)
        }
        return try gpu.readback(target)
    }
}
