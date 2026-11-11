import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GraphCoverageTests {
    private let gpu: PathTestFixture
    private let side = 33
    private var center: Int { index(x: side / 2) }

    init() throws { gpu = try PathTestFixture() }

    @Test("Texture opacity, explicit cutoff equality and a mask select lit raster and first-hit path pixels",
          arguments: [false, true])
    func cutouts(overrideMaterialCutoff: Bool) throws {
        let shader = try SurfaceShaderGraph { graph in
            let opacity = graph.texture("Coverage", content: .data, addressing: .clampToEdge, filtering: .nearest)
            return .init(
                opacity: opacity.sample(graph.uv0).x,
                alphaTest: graph.uniform("Cutoff", Float(0.5)),
                mask: graph.uv0.x.lessThan(graph.constant(Float(0.75)))
            )
        }
        let material = try PhysicallyBasedMaterial(
            roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 1, 0, 1),
            alphaMode: overrideMaterialCutoff ? .mask(cutoff: 0.95) : .opaque,
            surfaceGraph: shader, surfaceTextures: ["Coverage": try texture(SIMD4(0.5, 0, 0, 1), SIMD4(0.875, 0, 0, 1))],
            specular: .init(factor: 0)
        )
        let foreground = try plane(material, z: -2, width: 2.4)
        let background = try plane(unlit(SIMD4(1, 0, 0, 1)), z: -3, width: 4)
        let graph = SceneGraph(roots: [foreground, background, try light()], camera: gpu.camera)
        let equality = index(x: 4), visible = index(x: 20), masked = index(x: 29)
        let raster = try gpu.rasterize(graph, width: side, height: side)
        #expect(raster[equality] == SIMD4(1, 0, 0, 1) && raster[masked] == raster[equality])
        #expect(raster[visible].y > 0.2 && raster[visible].x < 0.01 && raster[visible].w == 1)

        let traced = try trace(graph, samples: 8, acceleration: overrideMaterialCutoff ? .software : .automatic)
        #expect(traced.albedo[equality].x > 0.99 && traced.albedo[masked].x > 0.99)
        #expect(traced.albedo[visible].y > 0.99 && traced.albedo[visible].x < 0.01)
        #expect(traced.color[visible].y > 0.2 && traced.color[visible].x < 0.01)
        #expect(traced.color[equality].x > 0.99 && traced.color[masked].x > 0.99)

        if overrideMaterialCutoff {
            var archivedGraph = graph
            let probe = try LightProbe(coefficients: Array(repeating: .zero, count: 9))
            archivedGraph.lightProbes = [probe]
            let archive = try SceneArchive(capturing: archivedGraph, commandQueue: gpu.queue)
            var manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
            #expect(manifest["version"] as? Int == 7)
            let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
            #expect(restored.lightProbes == [probe])
            #expect(try gpu.rasterize(restored, width: side, height: side) == raster)
            manifest["version"] = 6
            #expect(throws: SceneArchive.Failure.decoding("surface graph coverage requires archive version 7")) {
                try SceneArchive(manifest: JSONSerialization.data(withJSONObject: manifest), binary: archive.binary)
            }
            let uniforms = try #require(material.surfaceUniforms)
            try uniforms.set("Cutoff", to: Float(0.9))
            #expect(try gpu.rasterize(graph, width: side, height: side)[visible] == raster[equality])
        }
    }

    @Test("Graph opacity multiplies material alpha in raster blends, stochastic paths and unblended output",
          arguments: [false, true])
    func opacity(blends: Bool) throws {
        let shader = try SurfaceShaderGraph { graph in .init(opacity: graph.constant(Float(0.25))) }
        let front = try plane(unlit(SIMD4(0, 1, 0, 0.5), alpha: blends ? .blend : .materialAlpha, shader: shader), z: -2, width: 2.4)
        let back = try plane(unlit(SIMD4(1, 0, 0, 1)), z: -3, width: 4)
        let scene = SceneGraph(roots: [front, back], camera: gpu.camera)
        let raster = try gpu.rasterize(scene, width: side, height: side)[center]
        let expected: SIMD4<Float> = blends ? SIMD4(0.875, 0.125, 0, 1) : SIMD4(0, 0.125, 0, 0.125)
        #expect(simd_distance(raster, expected) < 0.002)
        let traced = try trace(scene, samples: blends ? 64 : 8)
        let middle = (12...20).flatMap { y in (12...20).map { x in traced.color[y * side + x] } }
        let mean = middle.reduce(SIMD4<Float>.zero, +) / Float(middle.count)
        #expect(simd_distance(mean, expected) < (blends ? 0.035 : 0.002), "mean \(mean), expected \(expected)")
        #expect(traced.albedo[center][blends ? 0 : 1] > 0.99)
    }

    @Test("Path shadow traversal honors graph mask with a live uniform on the same prepared scene")
    func shadowRays() throws {
        let receiver = try plane(PhysicallyBasedMaterial(
            roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, specular: .init(factor: 0)
        ), z: -4, width: 4)
        let shader = try SurfaceShaderGraph { graph in .init(mask: graph.uniform("Visible", Float(0))) }
        let blocker = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque,
            isDoubleSided: true, surfaceGraph: shader, shadingModel: .unlit
        )
        let card = try plane(blocker, x: 0.7, z: -3, width: 0.65)
        let graph = SceneGraph(roots: [receiver, card, try light(toward: SIMD3(0.7, 0, 1))], camera: gpu.camera)
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .automatic))
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph)))
        let clear = try render(tracer, samples: 8).color[center].x
        let uniforms = try #require(blocker.surfaceUniforms)
        try uniforms.set("Visible", to: Float(1))
        let occluded = try render(tracer, samples: 8).color[center].x
        #expect(clear > 0.2 && occluded < clear * 0.05, "clear \(clear), occluded \(occluded)")
    }

    private func index(x: Int) -> Int { side / 2 * side + x }

    private func unlit(
        _ color: SIMD4<Float>, alpha: PhysicallyBasedMaterial.AlphaMode = .opaque, shader: SurfaceShaderGraph? = nil
    ) throws -> PhysicallyBasedMaterial {
        try .init(lightIntensity: 0, ambientIntensity: 0, baseColorFactor: color, alphaMode: alpha,
                  surfaceGraph: shader, shadingModel: .unlit)
    }

    private func light(toward direction: SIMD3<Float> = SIMD3(0, 0, 1)) throws -> SceneNode {
        SceneNode(transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -simd_normalize(direction))),
                  light: try PunctualLight(kind: .directional, intensity: 1))
    }

    private func plane(_ material: PhysicallyBasedMaterial, x: Float = 0, z: Float, width: Float) throws -> SceneNode {
        let renderer = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: width, height: width), material: .physicallyBased(material))
        return SceneNode(transform: .init(translation: SIMD3(x, 0, z)), mesh: renderer)
    }

    private func texture(_ left: SIMD4<Float16>, _ right: SIMD4<Float16>) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 2, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        [left, right].withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: bytes.count)
        }
        return texture
    }

    private func trace(
        _ graph: SceneGraph, samples: Int, acceleration: PathTracingScene.Options.Acceleration = .software
    ) throws -> (color: [SIMD4<Float>], albedo: [SIMD4<Float>]) {
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(textures: .fullResolution(fallbackSize: 16), acceleration: acceleration))
        #expect(scene.accelerationKind == (acceleration == .automatic && gpu.device.supportsRaytracing ? .metal : .software))
        let lights = try PathTracer.AnalyticLight.lights(in: graph)
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: .init(analyticLights: lights))
        return try render(tracer, samples: samples)
    }

    private func render(_ tracer: PathTracer, samples: Int) throws -> (color: [SIMD4<Float>], albedo: [SIMD4<Float>]) {
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        let color = try gpu.output(width: side, height: side)
        let albedo = try gpu.output(width: side, height: side)
        var history: PathTracer.History?
        while (history?.sampleCount ?? 0) < samples {
            let command = try #require(gpu.queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: gpu.camera, cameraTransform: .init(), destination: color, accumulation: accumulation,
                previous: history, commandBuffer: command, samplesPerPixel: min(16, samples - (history?.sampleCount ?? 0)),
                settings: .init(maxBounces: 1, seed: 7), aovs: .init(albedo: albedo)
            )
            try gpu.finish(command)
        }
        return (try gpu.readback(color), try gpu.readback(albedo))
    }
}
