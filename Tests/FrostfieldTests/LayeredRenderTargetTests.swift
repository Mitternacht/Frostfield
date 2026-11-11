import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LayeredRenderTargetTests {
    enum Layout {
        case array, volume, cube, cubeArray

        var type: MTLTextureType {
            switch self {
            case .array: .type2DArray
            case .volume: .type3D
            case .cube: .typeCube
            case .cubeArray: .typeCubeArray
            }
        }

        var selected: Int { self == .cubeArray ? 7 : 1 }
        var neighbor: Int { selected - 1 }
        var size: SIMD2<Int> { self == .cube || self == .cubeArray ? SIMD2(19, 19) : SIMD2(23, 19) }

        func select(_ layer: Int, in attachment: MTLRenderPassColorAttachmentDescriptor) {
            if self == .volume { attachment.depthPlane = layer } else { attachment.slice = layer }
        }
    }

    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture() }

    private static var supportsCubeArrays: Bool {
        #if targetEnvironment(simulator)
        false
        #else
        let device = MTLCreateSystemDefaultDevice()
        return device?.supportsFamily(.apple4) == true || device?.supportsFamily(.mac2) == true
        #endif
    }

    @Test("Scene color draws into the selected native mip and layer with the mip's pixel dimensions",
          arguments: [Layout.array, .volume, .cube])
    func sceneDraw(_ layout: Layout) async throws {
        try await verifyLayerDraw(layout)
    }

    @Test("Cube-array faces select one cube and one face",
          .enabled(if: LayeredRenderTargetTests.supportsCubeArrays, "Requires native Metal cube-array textures"))
    func cubeArrayDraw() async throws {
        try await verifyLayerDraw(.cubeArray)
    }

    private func verifyLayerDraw(_ layout: Layout) async throws {
        let base = layout.size
        let size = base / 2
        let color = try makeColor(layout, size: base, format: .rgba16Float)
        let depth = try makeDepth(size)
        let surface = try SurfaceShaderGraph { shader in
            .init(emission: shader.vector(shader.screenSize / 32, shader.viewportSize.y / 32))
        }
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 5, height: 5),
            material: .physicallyBased(try .init(lightIntensity: 0, ambientIntensity: 0, surfaceGraph: surface))
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let graph = SceneGraph(roots: [node], camera: gpu.camera)
        let empty = SceneGraph(roots: [], camera: gpu.camera)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        let pass = gpu.pass(color: color, depth: depth)
        let attachment = pass.colorAttachments[0]!
        attachment.level = 1
        layout.select(layout.neighbor, in: attachment)
        attachment.clearColor = MTLClearColor(red: 0, green: 1, blue: 0, alpha: 1)
        try empty.encode(pass: pass, commandBuffer: buffer)

        layout.select(layout.selected, in: attachment)
        attachment.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        let center = size / 2
        let scissor = MTLScissorRect(x: center.x, y: center.y, width: 1, height: 1)
        let statistics = try graph.encode(pass: pass, commandBuffer: buffer, scissor: scissor)
        #expect(statistics.submittedTriangleCount == 2)
        try gpu.finish(buffer)

        let colored = try await pixel(color, layout: layout, layer: layout.selected, at: center)
        let expected = SIMD4<Float16>(Float16(Float(size.x) / 32), Float16(Float(size.y) / 32),
                                      Float16(Float(size.y) / 32), 1)
        #expect(colored == expected)
        #expect(try await pixel(color, layout: layout, layer: layout.selected, at: center &+ SIMD2(1, 0)) == .zero)
        #expect(try await pixel(color, layout: layout, layer: layout.neighbor, at: center) == SIMD4(0, 1, 0, 1))
    }

    @Test("Transmission copies the selected array or volume subresource into its backdrop",
          arguments: [Layout.array, .volume, .cube])
    func layeredTransmission(_ layout: Layout) async throws {
        let size = layout.size / 2
        let color = try makeColor(layout, size: layout.size, format: .rgba16Float)
        let depth = try makeDepth(size)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: size.x, height: size.y, mipmapped: true
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let backdrop = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let glass = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(try .init(
                roughness: 0, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(0.8, 0.6, 1, 1), transmission: .init(factor: 1, indexOfRefraction: 1)
            ))
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: glass)
        let graph = SceneGraph(roots: [node], camera: gpu.camera)
        let empty = SceneGraph(roots: [], camera: gpu.camera)
        let pass = gpu.pass(color: color, depth: depth)
        let attachment = pass.colorAttachments[0]!
        attachment.level = 1
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        layout.select(layout.neighbor, in: attachment)
        attachment.clearColor = MTLClearColor(red: 0, green: 1, blue: 0, alpha: 1)
        try empty.encode(pass: pass, commandBuffer: buffer)
        layout.select(layout.selected, in: attachment)
        attachment.clearColor = MTLClearColor(red: 2, green: 1, blue: 0.5, alpha: 1)
        try graph.encode(pass: pass, commandBuffer: buffer, transmissionBackdrop: backdrop)
        try gpu.finish(buffer)

        let center = size / 2
        let tinted = try await pixel(color, layout: layout, layer: layout.selected, at: center)
        #expect(simd_distance(SIMD4<Float>(tinted), SIMD4(1.6, 0.6, 0.5, 1)) < 0.012)
        #expect(try await pixel(color, layout: layout, layer: layout.neighbor, at: center) == SIMD4(0, 1, 0, 1))
        let snapshot = try await TextureTransfer.read(
            from: backdrop, region: MTLRegionMake2D(center.x, center.y, 1, 1), commandQueue: gpu.queue
        )
        #expect(snapshot.bytes.withUnsafeBytes { $0.loadUnaligned(as: SIMD4<Float16>.self) } == SIMD4(2, 1, 0.5, 1))
    }

    @Test("Cube faces accept ordinary single-mesh rendering in an 8-bit format with matching array depth")
    func cubeColorFormat() async throws {
        let layout = Layout.cube
        let size = layout.size / 2
        let color = try makeColor(layout, size: layout.size, format: .rgba8Unorm)
        let depth = try makeDepth(size, layers: 2)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba8Unorm, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(try .init(lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(1, 0, 0)))
        )
        let scene = MeshScene(camera: gpu.camera, transform: .init(translation: SIMD3(0, 0, -3)))
        let pass = gpu.pass(color: color, depth: depth)
        pass.colorAttachments[0].level = 1
        pass.colorAttachments[0].slice = 5
        pass.depthAttachment.slice = 1
        let command = try #require(gpu.queue.makeCommandBuffer())
        try mesh.encode(scene, pass: pass, commandBuffer: command)
        try gpu.finish(command)
        let center = size / 2
        let sample = try await TextureTransfer.read(
            from: color, mipLevel: 1, slice: 5, region: MTLRegionMake2D(center.x, center.y, 1, 1), commandQueue: gpu.queue
        )
        #expect(Array(sample.bytes) == [255, 0, 0, 255])
    }

    @Test("An invalid layer, mip or depth size fails before creating an encoder")
    func attachmentValidation() throws {
        let layout = Layout.volume
        let color = try makeColor(layout, size: layout.size, format: .rgba16Float)
        let depth = try makeDepth(layout.size / 2)
        let empty = SceneGraph(roots: [], camera: gpu.camera)
        let pass = gpu.pass(color: color, depth: depth)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let attachment = pass.colorAttachments[0]!
        #expect(throws: MeshRenderer.Failure.attachments) { try empty.encode(pass: pass, commandBuffer: command) }
        attachment.level = 1
        attachment.depthPlane = 2 // Mip one has two planes, indexed zero and one.
        #expect(throws: MeshRenderer.Failure.attachments) { try empty.encode(pass: pass, commandBuffer: command) }
        attachment.depthPlane = 1
        attachment.level = 2
        #expect(throws: MeshRenderer.Failure.attachments) { try empty.encode(pass: pass, commandBuffer: command) }
        attachment.level = 1
        attachment.slice = 1
        #expect(throws: MeshRenderer.Failure.attachments) { try empty.encode(pass: pass, commandBuffer: command) }
        attachment.slice = 0
        try empty.encode(pass: pass, commandBuffer: command)
        try gpu.finish(command)
    }

    private func makeColor(_ layout: Layout, size: SIMD2<Int>, format: MTLPixelFormat) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: size.x, height: size.y, mipmapped: false
        )
        descriptor.textureType = layout.type
        descriptor.mipmapLevelCount = 2
        descriptor.depth = layout == .volume ? 4 : 1
        descriptor.arrayLength = layout == .array || layout == .cubeArray ? 2 : 1
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func makeDepth(_ size: SIMD2<Int>, layers: Int = 1) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: size.x, height: size.y, mipmapped: false
        )
        if layers > 1 {
            descriptor.textureType = .type2DArray
            descriptor.arrayLength = layers
        }
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func pixel(
        _ texture: any MTLTexture, layout: Layout, layer: Int, at point: SIMD2<Int>
    ) async throws -> SIMD4<Float16> {
        let isVolume = layout == .volume
        let region = MTLRegionMake3D(point.x, point.y, isVolume ? layer : 0, 1, 1, 1)
        let sample = try await TextureTransfer.read(
            from: texture, mipLevel: 1, slice: isVolume ? 0 : layer, region: region, commandQueue: gpu.queue
        )
        return sample.bytes.withUnsafeBytes { $0.loadUnaligned(as: SIMD4<Float16>.self) }
    }
}
