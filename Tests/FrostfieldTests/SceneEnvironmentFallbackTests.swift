import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneEnvironmentFallbackTests {
    @Test("Scene environment replacement keeps one warmed pipeline and leases the previous source")
    func liveReplacement() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let mesh = try metal(gpu)
        var graph = scene(gpu, meshes: [mesh])
        let baseline = graph.resourceStatistics()
        #expect(try center(gpu, graph) == SIMD4(0, 0, 0, 1))

        var red: EnvironmentLighting? = try lighting(gpu, repeating: SIMD3(1, 0, 0))
        weak var retainedRed = red
        graph.environment = red
        let target = try gpu.makeTarget()
        let old = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        try graph.encode(pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: old)
        let warmed = graph.resourceStatistics()
        #expect(warmed.renderPipelineCount == baseline.renderPipelineCount + 1)
        #expect(warmed.textureCount == baseline.textureCount + 4)

        graph.environment = try lighting(gpu, repeating: SIMD3(0, 1, 0))
        red = nil
        #expect(retainedRed != nil)
        let green = try center(gpu, graph)
        #expect(green.y > 0.85 && green.x == 0 && green.z == 0 && green.w == 1)
        #expect(graph.resourceStatistics().renderPipelineCount == warmed.renderPipelineCount)
        try gpu.finish(old)
        let previous = try gpu.pixel(gpu.pixels(of: target.color), x: 32, y: 32)
        #expect(previous.x > 0.85 && previous.y == 0 && previous.z == 0 && previous.w == 1)

        graph.environment = nil
        #expect(try center(gpu, graph) == SIMD4(0, 0, 0, 1))
        #expect(graph.resourceStatistics().renderPipelineCount == warmed.renderPipelineCount)
    }

    @Test("A generated PBR material inherits scene controls and an authored environment keeps its own controls")
    func precedence() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        var faces = [SIMD3<Float>](repeating: SIMD3(0, 0, 1), count: 6)
        faces[1] = SIMD3(0, 1, 0) // −X, seen after a +90° environment rotation
        let cube = try lighting(gpu, faces: faces)
        let surface = try SurfaceShaderGraph { p in .init(roughness: p.sceneColor(at: p.screenUV).r) }
        let inherited = try metal(gpu, surface: surface)
        let authored = try metal(gpu, environment: .init(lighting: cube, intensity: 0.8))
        var graph = scene(gpu, meshes: [inherited, authored])
        let capture = PostProcessScenePass(queue: gpu.queue)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let frame = try capture.encode(
            SceneGraph(roots: [], camera: graph.camera), width: 1, height: 1, commandBuffer: command
        )
        try gpu.finish(command)
        graph.viewportSnapshot = try SceneViewportSnapshot(color: frame.color)
        let right = gpu.column(x: 0.8, depth: 3)
        let left = gpu.column(x: -0.8, depth: 3)

        let legacy = try gpu.render(graph).pixels
        let authoredBlue = gpu.pixel(legacy, x: right, y: 32)
        #expect(authoredBlue.z > 0.65 && authoredBlue.y < 0.03)
        graph.environmentIntensity = 0.25
        graph.environmentRotation = .pi / 2
        let adjusted = try gpu.render(graph).pixels
        let legacyGreen = gpu.pixel(adjusted, x: right, y: 32)
        #expect(abs(legacyGreen.y - authoredBlue.z * 0.25) < 0.01 && legacyGreen.z < 0.03)
        #expect(gpu.pixel(adjusted, x: left, y: 32) == SIMD4(0, 0, 0, 1))

        graph.environment = cube
        let inheritedAndAuthored = try gpu.render(graph).pixels
        let inheritedGreen = gpu.pixel(inheritedAndAuthored, x: left, y: 32)
        #expect(inheritedGreen.y > 0.2 && inheritedGreen.z < 0.03 && inheritedGreen.w == 1)
        #expect(gpu.pixel(inheritedAndAuthored, x: right, y: 32) == authoredBlue)
        graph.environmentIntensity = 0
        let muted = try gpu.render(graph).pixels
        #expect(gpu.pixel(muted, x: left, y: 32) == SIMD4(0, 0, 0, 1))
        #expect(gpu.pixel(muted, x: right, y: 32) == authoredBlue)
    }

    @Test("XYZ rotation selects the r187 cube face and keeps material-owned yaw and archive behavior")
    func fullRotation() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let faces: [SIMD3<Float>] = [
            SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0),
            SIMD3(1, 0, 1), SIMD3(0, 0, 1), SIMD3(0, 1, 1),
        ]
        let map = try lighting(gpu, faces: faces)
        let inherited = try metal(gpu)
        let authored = try metal(gpu, environment: .init(lighting: map, intensity: 0.8))
        var graph = scene(gpu, meshes: [inherited, authored])
        graph.environment = map
        graph.environmentIntensity = 0.5
        graph.environmentRotation = 0.22
        let x = gpu.column(x: -0.8, depth: 3)
        let right = gpu.column(x: 0.8, depth: 3)
        let baseline = try gpu.render(graph).pixels
        #expect(gpu.pixel(baseline, x: x, y: 32).z > 0.4)
        let pipelines = graph.resourceStatistics().renderPipelineCount

        // Pinned r187 inverse XYZ maps +Z to (-0.84101, 0.10627, 0.53048).
        graph.environmentEulerRotation = SIMD3(0.68, 0.60, -0.71)
        let rotated = try gpu.render(graph).pixels
        let green = gpu.pixel(rotated, x: x, y: 32)
        #expect(green.y > 0.4 && green.x < 0.02 && green.z < 0.02 && green.w == 1)
        #expect(gpu.pixel(rotated, x: right, y: 32) == gpu.pixel(baseline, x: right, y: 32))
        #expect(graph.resourceStatistics().renderPipelineCount == pipelines)

        let archive = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        var manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 18)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        #expect(restored.environmentEulerRotation == graph.environmentEulerRotation)
        #expect(restored.environmentRotation == graph.environmentRotation)
        #expect(try gpu.render(restored).pixels == rotated)
        manifest["version"] = 17
        #expect(throws: SceneArchive.Failure.decoding("scene environment Euler rotation requires archive version 18")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: manifest), binary: archive.binary)
        }

        graph.environment = nil
        let noSource = try gpu.render(graph).pixels
        graph.environmentEulerRotation = .zero
        #expect(try gpu.render(graph).pixels == noSource)
        graph.environmentEulerRotation.y = .greatestFiniteMagnitude
        graph.environmentRotation = .greatestFiniteMagnitude
        #expect(throws: MeshRenderer.Failure.environment) { try gpu.render(graph) }
        #expect(throws: SceneArchive.Failure.encoding("scene environment rotation angles must be finite")) {
            try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        }
    }

    @Test("The surface prepass uses the scene's BRDF LUT while classic and unlit colors ignore it")
    func attributesAndUnlit() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let mesh = try metal(gpu, roughness: 0.6)
        var graph = scene(gpu, meshes: [mesh])
        let map = try lighting(gpu, repeating: SIMD3(1, 0, 0))

        func attributes(_ graph: SceneGraph) throws -> SIMD2<UInt32> {
            let target = try gpu.makeTarget()
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: gpu.side, height: gpu.side, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .renderTarget
            let output = try #require(gpu.device.makeTexture(descriptor: descriptor))
            let command = try #require(gpu.queue.makeCommandBuffer())
            try graph.encodeSurfaceAttributes(pass: gpu.pass(color: output, depth: target.depth), commandBuffer: command)
            try gpu.finish(command)
            var pixel = SIMD2<UInt32>.zero
            output.getBytes(&pixel, bytesPerRow: 8, from: MTLRegionMake2D(32, 32, 1, 1), mipmapLevel: 0)
            return pixel
        }

        let approximate = try attributes(graph)
        let count = graph.resourceStatistics().renderPipelineCount
        graph.environment = map
        let fitted = try attributes(graph)
        #expect(fitted != .zero && fitted != approximate)
        #expect(graph.resourceStatistics().renderPipelineCount == count + 1)
        graph.environment = try lighting(gpu, repeating: SIMD3(0, 1, 0))
        #expect(try attributes(graph) == fitted)
        #expect(graph.resourceStatistics().renderPipelineCount == count + 1)

        let geometry = try GeometryRenderFixture.square(0.6)
        let classic = try gpu.unlit(geometry, material: .classic(ClassicMaterial(model: .basic)))
        let unlit = try gpu.unlit(geometry, material: .physicallyBased(PhysicallyBasedMaterial(shadingModel: .unlit)))
        var unlitGraph = scene(gpu, meshes: [classic, unlit])
        let colors = try gpu.render(unlitGraph).pixels
        #expect(colors.contains { $0.w == 1 })
        let unlitCount = unlitGraph.resourceStatistics().renderPipelineCount
        unlitGraph.environment = map
        #expect(try gpu.render(unlitGraph).pixels == colors)
        #expect(unlitGraph.resourceStatistics().renderPipelineCount == unlitCount)
    }

    @Test("Archive 16 preserves one environment shared by a scene and a material")
    func archive() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let map = try lighting(gpu, repeating: SIMD3(0.2, 0.5, 0.8))
        let meshes = try [metal(gpu), metal(gpu, environment: .init(lighting: map, intensity: 0.8))]
        var graph = scene(gpu, meshes: meshes)
        graph.environment = map
        graph.environmentIntensity = 0.5
        graph.environmentRotation = 0.23
        let before = try gpu.render(graph).pixels
        let archive = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        let manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 16)
        #expect((manifest["environments"] as? [[String: Any]])?.count == 1)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        let restoredEnvironment = try #require(restored.environment)
        let material = try #require(restored.roots.last?.mesh?.material)
        guard case .physicallyBased(let pbr) = material else {
            Issue.record("Expected a restored PBR material")
            return
        }
        #expect(pbr.environment?.lighting === restoredEnvironment)
        #expect(restored.environmentIntensity == 0.5 && restored.environmentRotation == 0.23)
        #expect(restored.environmentEulerRotation == .zero)
        #expect(try gpu.render(restored).pixels == before)
    }

    private func scene(_ gpu: GeometryRenderFixture, meshes: [MeshRenderer]) -> SceneGraph {
        let nodes = meshes.enumerated().map { index, mesh in
            let x: Float = meshes.count == 1 ? 0 : (index == 0 ? -0.8 : 0.8)
            return SceneNode(transform: .init(translation: SIMD3(x, 0, -3)), mesh: mesh)
        }
        return SceneGraph(roots: nodes, camera: gpu.camera)
    }

    private func metal(
        _ gpu: GeometryRenderFixture, roughness: Float = 0, environment: PhysicallyBasedMaterial.Environment? = nil,
        surface: SurfaceShaderGraph? = nil
    ) throws -> MeshRenderer {
        let material = try PhysicallyBasedMaterial(
            metallic: 1, roughness: roughness, lightIntensity: 0, ambientIntensity: 0, environment: environment,
            surfaceGraph: surface
        )
        return try gpu.unlit(GeometryRenderFixture.square(0.6), material: .physicallyBased(material))
    }

    private func center(_ gpu: GeometryRenderFixture, _ graph: SceneGraph) throws -> SIMD4<Float> {
        try gpu.pixel(gpu.render(graph).pixels, x: 32, y: 32)
    }

    private func lighting(_ gpu: GeometryRenderFixture, repeating color: SIMD3<Float>) throws -> EnvironmentLighting {
        try lighting(gpu, faces: Array(repeating: color, count: 6))
    }

    private func lighting(_ gpu: GeometryRenderFixture, faces: [SIMD3<Float>]) throws -> EnvironmentLighting {
        let size = 8
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float, size: size, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let source = try #require(gpu.device.makeTexture(descriptor: descriptor))
        for (slice, color) in faces.enumerated() {
            let pixels = [SIMD4<Float16>](repeating: SIMD4(SIMD3<Float16>(color), 1), count: size * size)
            pixels.withUnsafeBytes {
                source.replace(
                    region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0, slice: slice,
                    withBytes: $0.baseAddress!, bytesPerRow: size * 8, bytesPerImage: size * size * 8
                )
            }
        }
        return try EnvironmentLighting(
            queue: gpu.queue, cube: source, settings: .init(cubeSize: size, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        )
    }
}
