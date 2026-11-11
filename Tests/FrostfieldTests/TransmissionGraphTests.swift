import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TransmissionGraphTests {
    @Test("A transmission graph snapshots opaque color and mips before preserving depth for blended draws")
    func opaqueBackdrop() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let size = 32
        let geometry = MeshGeometry.cube()
        let opaque = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(1, 0, 0), alphaMode: .opaque
            ))
        )
        let glass = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try PhysicallyBasedMaterial(
                alphaMode: .opaque, transmission: .init(factor: 1)
            ))
        )
        func blend(emission: SIMD3<Float>) throws -> MeshRenderer {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(try PhysicallyBasedMaterial(
                    lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(1, 1, 1, 0.5),
                    emissiveFactor: emission, alphaMode: .blend
                ))
            )
        }
        let blue = try blend(emission: SIMD3(0, 0, 1))
        let green = try blend(emission: SIMD3(0, 1, 0))
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let graph = SceneGraph(roots: [
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: blue),
            SceneNode(transform: SceneTransform(translation: SIMD3(10, 0, -3)), mesh: glass),
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -4)), mesh: opaque),
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -6)), mesh: green),
        ], camera: camera)

        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: size, height: size, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let backdropDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: size, height: size, mipmapped: true
        )
        backdropDescriptor.storageMode = .shared
        backdropDescriptor.usage = .shaderRead
        let backdrop = try #require(device.makeTexture(descriptor: backdropDescriptor))
        backdropDescriptor.mipmapLevelCount = 1
        let incompleteBackdrop = try #require(device.makeTexture(descriptor: backdropDescriptor))

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0.25, blue: 0, alpha: 1)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        #expect(glass.requiresTransmissionBackdrop)
        #expect(!opaque.requiresTransmissionBackdrop)
        #expect(throws: MeshRenderer.Failure.transmissionBackdrop) {
            try graph.encode(pass: pass, commandBuffer: buffer)
        }
        #expect(throws: MeshRenderer.Failure.invalidTransmissionBackdrop) {
            try graph.encode(pass: pass, commandBuffer: buffer, transmissionBackdrop: incompleteBackdrop)
        }
        #expect(throws: MeshRenderer.Failure.transmissionBackdrop) {
            try glass.encode(MeshScene(camera: camera), pass: pass, commandBuffer: buffer)
        }
        try graph.encode(pass: pass, commandBuffer: buffer, transmissionBackdrop: backdrop)
        #expect(pass.colorAttachments[0].loadAction == .clear)
        #expect(pass.depthAttachment.loadAction == .clear)
        #expect(pass.depthAttachment.storeAction == .dontCare)
        buffer.commit()
        // Blit the selected mip explicitly because Simulator getBytes can read level zero.
        let finalLevel = try await TextureTransfer.read(
            from: backdrop, mipLevel: backdrop.mipmapLevelCount - 1, commandQueue: queue
        )
        try #require(buffer.status == .completed)
        #expect(readPixel(in: backdrop, x: size / 2, y: size / 2) == SIMD4(1, 0, 0, 1))
        #expect(readPixel(in: color, x: size / 2, y: size / 2) == SIMD4(0.5, 0, 0.5, 1))
        let mip = finalLevel.bytes.withUnsafeBytes { $0.loadUnaligned(as: SIMD4<Float16>.self) }
        #expect(mip.x > 0.02 && mip.y > 0.05 && mip.z == 0 && mip.w == 1)
    }

    private func readPixel(in texture: any MTLTexture, x: Int, y: Int) -> SIMD4<Float16> {
        var value = SIMD4<Float16>.zero
        withUnsafeMutableBytes(of: &value) { bytes in
            texture.getBytes(bytes.baseAddress!, bytesPerRow: 8, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        }
        return value
    }
}
