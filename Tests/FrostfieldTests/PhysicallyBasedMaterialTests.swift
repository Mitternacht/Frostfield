import Frostfield
import Metal
import Testing

struct PhysicallyBasedMaterialTests {
    @Test("PBR draws metallic highlights and premultiplied ambient color",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func render() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let size = 64
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))

        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.depthPixelFormat, width: size, height: size, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1

        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let scene = MeshScene(
            camera: camera,
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            materialParameters: SIMD4(0.8, 0.3, 0.1, 0.5)
        )

        func centerPixel(roughness: Float, direct: Float, ambient: Float) throws -> [UInt8] {
            let material = try PhysicallyBasedMaterial(
                metallic: 1, roughness: roughness, directionToLight: SIMD3(0, 0, 1),
                lightIntensity: direct, ambientIntensity: ambient
            )
            let renderer = try MeshRenderer(
                device: device, colorPixelFormat: color.pixelFormat,
                geometry: .cube(), material: .physicallyBased(material)
            )
            let command = try #require(queue.makeCommandBuffer())
            try renderer.encode(scene, pass: pass, commandBuffer: command)
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed)

            var result = [UInt8](repeating: 0, count: 4)
            try result.withUnsafeMutableBytes { bytes in
                color.getBytes(
                    try #require(bytes.baseAddress), bytesPerRow: 4,
                    from: MTLRegionMake2D(size / 2, size / 2, 1, 1), mipmapLevel: 0
                )
            }
            return result
        }

        let polished = try centerPixel(roughness: 0.35, direct: 0.2, ambient: 0)
        let rough = try centerPixel(roughness: 0.8, direct: 0.2, ambient: 0)
        #expect(Int(polished[0]) > Int(rough[0]) + 30)
        #expect(polished[0] > polished[1])
        #expect(polished[1] > polished[2])
        #expect(abs(Int(polished[3]) - 128) <= 1)
        #expect(abs(Int(rough[3]) - 128) <= 1)

        // Pure-metal constant fill is base RGB × 0.5 ambient × 0.5 alpha.
        let ambient = try centerPixel(roughness: 0.8, direct: 0, ambient: 0.5)
        for (actual, expected) in zip(ambient, [51, 19, 6, 128]) {
            #expect(abs(Int(actual) - expected) <= 2)
        }
    }
    @Test("Material packs a finite unit light direction and the requested parameters")
    func parameters() throws {
        let material = try PhysicallyBasedMaterial(
            metallic: 0.3, roughness: 0.7,
            directionToLight: SIMD3(.greatestFiniteMagnitude, -.greatestFiniteMagnitude, 0),
            lightIntensity: 1.5, ambientIntensity: 0.25
        )
        #expect(material.baseColorTexture == nil)
        #expect(abs(material.lightParameters.x - 0.70710678) < 0.00001)
        #expect(abs(material.lightParameters.y + 0.70710678) < 0.00001)
        #expect(material.lightParameters.z == 0)
        #expect(material.lightParameters.w == 1.5)
        #expect(material.settings == SIMD4(0.7, 0.3, 0.25, 0))
    }

    @Test("Metallic and roughness reject values outside the unit interval",
          arguments: [-0.1, 1.1, .infinity, .nan] as [Float])
    func invalidFactors(_ value: Float) {
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidMetallic) {
            try PhysicallyBasedMaterial(metallic: value)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidRoughness) {
            try PhysicallyBasedMaterial(roughness: value)
        }
    }

    @Test("Direction and intensity reject invalid lighting values")
    func invalidLight() {
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidDirection) {
            try PhysicallyBasedMaterial(directionToLight: .zero)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidDirection) {
            try PhysicallyBasedMaterial(directionToLight: SIMD3(1, .nan, 0))
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidLightIntensity) {
            try PhysicallyBasedMaterial(lightIntensity: -.infinity)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidAmbientIntensity) {
            try PhysicallyBasedMaterial(ambientIntensity: -1)
        }
    }
}
