import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SurfaceRenderTargetsTests {
    private let gpu: PathTestFixture
    private let side = 17

    init() throws { gpu = try PathTestFixture() }

    @Test("Named standalone attachments contain the regular PBR color and signed mapped world normal")
    func stockSurface() async throws {
        let layout = try targets()
        let normal = try texture(.rgba16Float, side: 1, usage: .shaderRead, storage: .shared)
        var encoded = SIMD4<Float16>(0.125, 0.5, 0.75, 1)
        normal.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &encoded, bytesPerRow: 8)
        let material = try PhysicallyBasedMaterial(
            roughness: 0.625, directionToLight: SIMD3(-0.4, 0, 1), lightIntensity: 1, ambientIntensity: 0,
            normalTexture: .init(texture: normal, normalSpace: .objectSpace),
            emissiveFactor: SIMD3(0.125, 0.25, 0.375), alphaMode: .opaque
        )
        let geometry = try MeshGeometry.plane(width: 2, height: 2)
        let renderer = try MeshRenderer(device: gpu.device, surfaceRenderTargets: layout, geometry: geometry, material: material)
        let ordinary = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        let transform = SceneTransform(
            translation: SIMD3(0, 0, -3),
            rotation: simd_quatf(angle: 0.2, axis: SIMD3(0, 1, 0)), scale: SIMD3(1.4, 1, 0.75)
        )
        let output = try attachments(for: layout)
        let command = try #require(gpu.queue.makeCommandBuffer())
        try renderer.encode(MeshScene(camera: gpu.camera, transform: transform), pass: output.pass, commandBuffer: command)
        try gpu.finish(command)

        let middle = side / 2
        let beauty = try await read(output.colors[SurfaceRenderTargets.colorName], x: middle, as: SIMD4<Float16>.self)
        let reference = try gpu.rasterize(
            SceneGraph(roots: [SceneNode(transform: transform, mesh: ordinary)], camera: gpu.camera), width: side, height: side
        )[middle * side + middle]
        #expect(SIMD4<Float>(beauty) == reference)
        #expect(beauty.x > 0.2 && beauty.y > 0.3 && beauty.z > 0.4 && beauty.w == 1)
        let transformed = simd_transpose(simd_inverse(transform.matrix)) * SIMD4<Float>(-0.75, 0, 0.5, 0)
        let expected = simd_normalize(SIMD3(transformed.x, transformed.y, transformed.z))
        let world = try await read(output.colors["world"], x: middle, as: SIMD4<Float16>.self)
        #expect(world.x < -0.3 && simd_distance(SIMD4<Float>(world), SIMD4(expected, 1)) < 0.002)
        #expect(try await read(output.colors["material-roughness"], x: middle, as: Float.self) == 0.625)
    }

    @Test("An owned PBR graph writes live named values and matching cutouts in both depth modes",
          arguments: [MeshDepthMode.forward, .reversed])
    func graphSurface(_ depthMode: MeshDepthMode) async throws {
        let layout = try targets()
        let shader = try SurfaceShaderGraph { graph in
            .init(
                roughness: graph.uniform("Roughness", Float(0.75)),
                emission: graph.constant(SIMD3<Float>(0.25, 0.5, 0.125)),
                normal: graph.constant(SIMD3<Float>(-0.6, 0, 0.8)),
                mask: graph.uv0.x.lessThan(graph.uniform("Reveal", Float(0.5)))
            )
        }
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, receivesShadows: true, alphaMode: .opaque, surfaceGraph: shader
        )
        let renderer = try MeshRenderer(
            device: gpu.device, surfaceRenderTargets: layout, geometry: .plane(width: 3, height: 3),
            material: material, depthMode: depthMode
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: renderer)
        ], camera: gpu.camera)
        let shadowMap = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 32, directionToLight: SIMD3(0.3, 0, 1),
            center: SIMD3(0, 0, -2), halfExtent: 3, intensity: 0
        )
        let output = try attachments(for: layout, depthMode: depthMode)
        func draw() throws {
            let command = try #require(gpu.queue.makeCommandBuffer())
            let stats = try graph.encode(pass: output.pass, commandBuffer: command, shadowMap: shadowMap, depthMode: depthMode)
            #expect(stats.drawnMeshCount == 1)
            try gpu.finish(command)
        }
        try draw()

        let left = 4, right = side - 5
        let projection = try gpu.camera.projection(aspectRatio: 1, depthMode: depthMode)
        let clip = projection * SIMD4<Float>(0, 0, -2, 1)
        let expectedDepth = clip.z / clip.w
        let clearDepth: Float = depthMode == .reversed ? 0 : 1
        #expect(try await read(output.colors[SurfaceRenderTargets.colorName], x: left, as: SIMD4<Float16>.self) == SIMD4(0.25, 0.5, 0.125, 1))
        let world = try await read(output.colors["world"], x: left, as: SIMD4<Float16>.self)
        #expect(simd_distance(SIMD4<Float>(world), SIMD4(-0.6, 0, 0.8, 1)) < 0.002)
        #expect(try await read(output.colors["material-roughness"], x: left, as: Float.self) == 0.75)
        #expect(abs(try await read(output.depth, x: left, as: Float.self) - expectedDepth) < 0.000001)
        #expect(try await read(output.colors[SurfaceRenderTargets.colorName], x: right, as: SIMD4<Float16>.self) == .zero)
        #expect(try await read(output.colors["world"], x: right, as: SIMD4<Float16>.self) == .zero)
        #expect(try await read(output.colors["material-roughness"], x: right, as: Float.self) == 0.125)
        #expect(try await read(output.depth, x: right, as: Float.self) == clearDepth)

        let uniforms = try #require(material.surfaceUniforms)
        try uniforms.set("Reveal", to: Float(1))
        try uniforms.set("Roughness", to: Float(0.25))
        try draw()
        #expect(try await read(output.colors["material-roughness"], x: right, as: Float.self) == 0.25)
        #expect(try await read(output.colors["world"], x: right, as: SIMD4<Float16>.self) == world)
        #expect(abs(try await read(output.depth, x: right, as: Float.self) - expectedDepth) < 0.000001)
    }

    @Test("Names and value types are validated independently of the ordered Metal formats")
    func invalidLayout() throws {
        #expect(throws: SurfaceRenderTargets.Failure.names) {
            try SurfaceRenderTargets(outputs: [.init("output", value: .roughness, pixelFormat: .r32Float)])
        }
        #expect(throws: SurfaceRenderTargets.Failure.format("normal")) {
            try SurfaceRenderTargets(outputs: [.init("normal", value: .worldNormal, pixelFormat: .r32Float)])
        }
        let layout = try targets()
        let material = try PhysicallyBasedMaterial()
        let renderer = try MeshRenderer(device: gpu.device, surfaceRenderTargets: layout, geometry: .cube(), material: material)
        let transform = SceneTransform(translation: SIMD3(0, 0, -3))
        let scene = MeshScene(camera: gpu.camera, transform: transform)
        let graph = SceneGraph(roots: [SceneNode(transform: transform, mesh: renderer)], camera: gpu.camera)
        let wrongNames = try attachments(for: targets(normalName: "other"))
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: SurfaceRenderTargets.Failure.rendererLayout) {
            try renderer.encode(scene, pass: wrongNames.pass, commandBuffer: command)
        }
        #expect(throws: SurfaceRenderTargets.Failure.rendererLayout) {
            try graph.encode(pass: wrongNames.pass, commandBuffer: command)
        }
        #expect(throws: SurfaceRenderTargets.Failure.attachments) {
            try layout.makePass(colorAttachments: [:], depthAttachment: MTLRenderPassDepthAttachmentDescriptor())
        }
        #expect(throws: SurfaceRenderTargets.Failure.material) {
            try MeshRenderer(device: gpu.device, surfaceRenderTargets: layout, geometry: .cube(),
                             material: PhysicallyBasedMaterial(alphaMode: .blend))
        }
        #expect(throws: SceneArchive.Failure.unsupportedMaterial("named surface render targets require a runtime pass layout")) {
            try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        }
        try renderer.encode(scene, pass: attachments(for: layout).pass, commandBuffer: command)
        try gpu.finish(command)
    }

    @Test("A material override supplies the named PBR pipeline for an ordinary mesh")
    func overrideSurface() async throws {
        let geometry = try MeshGeometry.plane(width: 2, height: 2)
        let renderer = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)
        var graph = SceneGraph(roots: [node], camera: gpu.camera)
        graph.overrideMaterial = try SceneMaterialOverride(material: .physicallyBased(PhysicallyBasedMaterial(
            roughness: 0.625, lightIntensity: 0, ambientIntensity: 0,
            emissiveFactor: SIMD3(0.125, 0.25, 0.375)
        )))
        let output = try attachments(for: targets())
        let command = try #require(gpu.queue.makeCommandBuffer())
        try graph.encode(pass: output.pass, commandBuffer: command)
        try gpu.finish(command)
        let center = side / 2
        let color = try await read(output.colors[SurfaceRenderTargets.colorName], x: center, as: SIMD4<Float16>.self)
        let normal = try await read(output.colors["world"], x: center, as: SIMD4<Float16>.self)
        #expect(color == SIMD4(0.125, 0.25, 0.375, 1))
        #expect(normal == SIMD4(0, 0, 1, 1))
        #expect(try await read(output.colors["material-roughness"], x: center, as: Float.self) == 0.625)
    }

    private func targets(normalName: String = "world") throws -> SurfaceRenderTargets {
        try SurfaceRenderTargets(outputs: [
            .init("material-roughness", value: .roughness, pixelFormat: .r32Float),
            .init(normalName, value: .worldNormal, pixelFormat: .rgba16Float)
        ])
    }

    private func attachments(
        for layout: SurfaceRenderTargets, depthMode: MeshDepthMode = .forward
    ) throws -> (pass: SurfaceRenderTargets.Pass, colors: [String: any MTLTexture], depth: any MTLTexture) {
        let depth = try texture(.depth32Float, side: side)
        let depthAttachment = MTLRenderPassDepthAttachmentDescriptor()
        depthAttachment.texture = depth
        depthAttachment.loadAction = .clear
        depthAttachment.storeAction = .store
        depthAttachment.clearDepth = depthMode == .reversed ? 0 : 1

        var colors: [String: any MTLTexture] = [:]
        var descriptors: [String: MTLRenderPassColorAttachmentDescriptor] = [:]
        let formats = [(SurfaceRenderTargets.colorName, layout.colorPixelFormat)] + layout.outputs.map { ($0.name, $0.pixelFormat) }
        for (name, format) in formats.reversed() {
            let image = try texture(format, side: side)
            let attachment = MTLRenderPassColorAttachmentDescriptor()
            attachment.texture = image
            attachment.loadAction = .clear
            attachment.storeAction = .store
            attachment.clearColor = MTLClearColor(red: format == .r32Float ? 0.125 : 0, green: 0, blue: 0, alpha: 0)
            colors[name] = image
            descriptors[name] = attachment
        }
        return (try layout.makePass(colorAttachments: descriptors, depthAttachment: depthAttachment), colors, depth)
    }

    private func texture(
        _ format: MTLPixelFormat, side: Int, usage: MTLTextureUsage = .renderTarget, storage: MTLStorageMode = .private
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: side, height: side, mipmapped: false)
        descriptor.storageMode = storage
        descriptor.usage = usage
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func read<T>(_ texture: (any MTLTexture)?, x: Int, as type: T.Type) async throws -> T {
        let pixel = try await TextureTransfer.read(
            from: #require(texture), region: MTLRegionMake2D(x, side / 2, 1, 1), commandQueue: gpu.queue
        )
        try #require(pixel.bytes.count == MemoryLayout<T>.size)
        return pixel.bytes.withUnsafeBytes { $0.loadUnaligned(as: type) }
    }
}
