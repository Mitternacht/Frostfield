import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshMultipleRenderTargetsTests {
    private static let formats: [MTLPixelFormat] = [.rgba16Float, .r32Uint, .r32Float]
    private static let side = 16
    private let fixture: PathTestFixture

    init() throws { fixture = try PathTestFixture() }

    @Test("A prepared mesh writes independent HDR, integer and scalar pixels with shared depth")
    func standaloneOutputs() async throws {
        let renderer = try makeRenderer()
        #expect(renderer.colorPixelFormat == .rgba16Float)
        #expect(renderer.colorPixelFormats == Self.formats)
        #expect(renderer.sampleCount == 1)
        let target = try makePass()
        let command = try #require(fixture.queue.makeCommandBuffer())
        try renderer.encode(scene, pass: target.pass, commandBuffer: command)
        try fixture.finish(command)

        let center = Self.side / 2
        let hdr = try await pixel(target.colors[0], at: center, center, as: SIMD4<Float16>.self)
        let id = try await pixel(target.colors[1], at: center, center, as: UInt32.self)
        let intensity = try await pixel(target.colors[2], at: center, center, as: Float.self)
        let depth = try await pixel(target.depth, at: center, center, as: Float.self)
        #expect(hdr == SIMD4(1.5, 0.25, 0.75, 1))
        #expect(id == 1029)
        #expect(intensity == 0.5)
        #expect(depth > 0 && depth < 1)

        #expect(try await pixel(target.colors[0], at: 0, 0, as: SIMD4<Float16>.self) == .zero)
        #expect(try await pixel(target.colors[1], at: 0, 0, as: UInt32.self) == 0)
        #expect(try await pixel(target.colors[2], at: 0, 0, as: Float.self) == 0.125)
        #expect(try await pixel(target.depth, at: 0, 0, as: Float.self) == 1)
    }

    @Test("One mesh draw writes a chosen array layer and volume plane alongside an ordinary target")
    func layeredOutputs() async throws {
        let renderer = try makeRenderer()
        let targets = try makePass()
        let side = Self.side
        let arrayDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false
        )
        arrayDescriptor.textureType = .type2DArray
        arrayDescriptor.arrayLength = 2
        arrayDescriptor.storageMode = .private
        arrayDescriptor.usage = .renderTarget
        let array = try #require(fixture.device.makeTexture(descriptor: arrayDescriptor))
        let volumeDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: side, height: side, mipmapped: false
        )
        volumeDescriptor.textureType = .type3D
        volumeDescriptor.depth = 2
        volumeDescriptor.storageMode = .private
        volumeDescriptor.usage = .renderTarget
        let volume = try #require(fixture.device.makeTexture(descriptor: volumeDescriptor))
        targets.pass.colorAttachments[0].texture = array
        targets.pass.colorAttachments[0].slice = 1
        targets.pass.colorAttachments[2].texture = volume
        targets.pass.colorAttachments[2].depthPlane = 1

        let command = try #require(fixture.queue.makeCommandBuffer())
        try renderer.encode(scene, pass: targets.pass, commandBuffer: command)
        try fixture.finish(command)
        let center = side / 2
        let arrayPixel = try await TextureTransfer.read(
            from: array, slice: 1, region: MTLRegionMake2D(center, center, 1, 1), commandQueue: fixture.queue
        )
        let volumePixel = try await TextureTransfer.read(
            from: volume, region: MTLRegionMake3D(center, center, 1, 1, 1, 1), commandQueue: fixture.queue
        )
        #expect(arrayPixel.bytes.withUnsafeBytes { $0.loadUnaligned(as: SIMD4<Float16>.self) } == SIMD4(1.5, 0.25, 0.75, 1))
        #expect(volumePixel.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) } == 0.5)
        #expect(try await pixel(targets.colors[1], at: center, center, as: UInt32.self) == 1029)
    }

    @Test("The graph uses one depth test and respects visibility across every color target")
    func sceneOutputs() async throws {
        let renderer = try makeRenderer()
        let single = try MeshRenderer(device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .cube())
        let target = try makePass()
        let graph = SceneGraph(roots: [
            SceneNode(transform: scene.transform, mesh: renderer, materialParameters: scene.materialParameters),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: renderer,
                      materialParameters: SIMD4(0.25, 1, 0, 71)),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: single, isVisible: false)
        ], camera: fixture.camera)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let statistics = try graph.encode(pass: target.pass, commandBuffer: command)
        try fixture.finish(command)
        #expect(statistics.drawnMeshCount == 2)
        let center = Self.side / 2
        #expect(try await pixel(target.colors[0], at: center, center, as: SIMD4<Float16>.self) == SIMD4(1.5, 0.25, 0.75, 1))
        #expect(try await pixel(target.colors[1], at: center, center, as: UInt32.self) == 1029)
        #expect(try await pixel(target.colors[2], at: center, center, as: Float.self) == 0.5)
    }

    @Test("MRT layout mismatches fail before encoding, leaving the command buffer usable")
    func incompatiblePasses() throws {
        let renderer = try makeRenderer()
        let target = try makePass()
        let pass = target.pass
        let command = try #require(fixture.queue.makeCommandBuffer())
        func expectRejection() {
            #expect(throws: MeshRenderer.Failure.attachments) {
                try renderer.encode(scene, pass: pass, commandBuffer: command)
            }
        }

        pass.colorAttachments[1].texture = nil // A hole between active attachments.
        expectRejection()
        pass.colorAttachments[2].texture = nil // Only attachment zero.
        expectRejection()
        pass.colorAttachments[1].texture = target.colors[2]
        pass.colorAttachments[2].texture = target.colors[1] // Valid pass, wrong order for this pipeline.
        expectRejection()
        pass.colorAttachments[1].texture = target.colors[1]
        pass.colorAttachments[2].texture = try texture(.r32Float, width: Self.side - 1)
        expectRejection()
        pass.colorAttachments[2].texture = try texture(.r32Float, usage: .shaderRead)
        expectRejection()
        pass.colorAttachments[2].texture = target.colors[2]
        pass.colorAttachments[2].level = 1
        expectRejection()
        pass.colorAttachments[2].level = 0
        pass.colorAttachments[2].resolveTexture = target.colors[2]
        expectRejection()
        pass.colorAttachments[2].resolveTexture = nil
        pass.colorAttachments[1].texture = target.colors[0]
        let empty = SceneGraph(roots: [], camera: fixture.camera)
        #expect(throws: MeshRenderer.Failure.attachments) {
            try empty.encode(pass: pass, commandBuffer: command)
        }
        pass.colorAttachments[1].texture = target.colors[1]
        try renderer.encode(scene, pass: pass, commandBuffer: command)
        try fixture.finish(command)
    }

    @Test("Stock and primitive pipelines cannot be mixed into a custom MRT graph")
    func incompatiblePipelines() throws {
        let renderer = try makeRenderer()
        let target = try makePass()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let single = try MeshRenderer(device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .cube())
        let node = SceneNode(transform: scene.transform, mesh: renderer)
        let mixed = SceneGraph(roots: [node, SceneNode(transform: scene.transform, mesh: single)], camera: fixture.camera)
        #expect(throws: MeshRenderer.Failure.attachments) {
            try mixed.encode(pass: target.pass, commandBuffer: command)
        }
        #expect(throws: MeshRenderer.Failure.attachments) {
            try single.encode(scene, pass: target.pass, commandBuffer: command)
        }
        let points = try PointsRenderer(
            device: fixture.device, cloud: PointCloud(positions: [SIMD3(0, 0, -2)]),
            material: PointsMaterial(size: .pixels(2))
        )
        let primitives = SceneGraph(roots: [node, SceneNode(primitive: .points(points))], camera: fixture.camera)
        #expect(throws: MeshRenderer.Failure.multipleRenderTargets) {
            try primitives.encode(pass: target.pass, commandBuffer: command)
        }

        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let attributes = try makePass(formats: [SceneGraph.surfaceAttributesPixelFormat])
        #expect(try graph.encodeSurfaceAttributes(pass: attributes.pass, commandBuffer: command).drawnMeshCount == 1)
        try fixture.finish(command)
    }

    @Test("Preparation rejects stock materials, blending and incomplete attachment lists")
    func invalidPreparation() throws {
        let material = try customMaterial()
        for formats: [MTLPixelFormat] in [[], [.rgba16Float], [.rgba16Float, .invalid], Array(repeating: .rgba8Unorm, count: 9)] {
            #expect(throws: MeshRenderer.Failure.multipleRenderTargets) {
                try MeshRenderer(device: fixture.device, colorPixelFormats: formats, geometry: .cube(), material: material)
            }
        }
        #expect(throws: MeshRenderer.Failure.multipleRenderTargets) {
            try MeshRenderer(device: fixture.device, colorPixelFormats: Self.formats, geometry: .cube(), material: .unlit)
        }
        #expect(throws: MeshRenderer.Failure.multipleRenderTargets) {
            try MeshRenderer(
                device: fixture.device, colorPixelFormats: Self.formats, geometry: .cube(),
                material: material, renderState: .transparent
            )
        }
    }

    private var scene: MeshScene {
        MeshScene(
            camera: fixture.camera, transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            materialParameters: SIMD4(1.5, 0.25, 0.75, 1029)
        )
    }

    private func makeRenderer() throws -> MeshRenderer {
        try MeshRenderer(device: fixture.device, colorPixelFormats: Self.formats, geometry: .cube(), material: customMaterial())
    }

    private func customMaterial() throws -> MeshMaterial {
        let library = try fixture.device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            struct Outputs {
                float4 hdr [[color(0)]];
                uint objectID [[color(1)]];
                float intensity [[color(2)]];
            };
            fragment Outputs outputs(constant float4& parameters [[buffer(0)]]) {
                return {float4(parameters.xyz, 1.0f), uint(parameters.w), parameters.y * 2.0f};
            }
            """, options: nil)
        return .custom(library: library, fragment: "outputs")
    }

    private func makePass(
        formats: [MTLPixelFormat] = Self.formats
    ) throws -> (pass: MTLRenderPassDescriptor, colors: [any MTLTexture], depth: any MTLTexture) {
        let pass = MTLRenderPassDescriptor()
        let colors = try formats.map { try texture($0) }
        for (index, texture) in colors.enumerated() {
            let attachment = pass.colorAttachments[index]!
            attachment.texture = texture
            attachment.loadAction = .clear
            attachment.storeAction = .store
            attachment.clearColor = MTLClearColor(red: index == 2 ? 0.125 : 0, green: 0, blue: 0, alpha: 0)
        }
        let depth = try texture(.depth32Float)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1
        return (pass, colors, depth)
    }

    private func texture(
        _ format: MTLPixelFormat, width: Int = Self.side, usage: MTLTextureUsage = .renderTarget
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: Self.side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = usage
        return try #require(fixture.device.makeTexture(descriptor: descriptor))
    }

    private func pixel<T>(
        _ texture: any MTLTexture, at x: Int, _ y: Int, as type: T.Type
    ) async throws -> T {
        let result = try await TextureTransfer.read(
            from: texture, region: MTLRegionMake2D(x, y, 1, 1), commandQueue: fixture.queue
        )
        try #require(result.bytes.count == MemoryLayout<T>.size)
        return result.bytes.withUnsafeBytes { $0.loadUnaligned(as: type) }
    }
}
