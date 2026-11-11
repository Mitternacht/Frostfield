import Foundation
import Frostfield
import Metal
import simd
import Testing

struct ClearcoatTests {
    @Test("Clearcoat validates factors and normal scale")
    func validation() {
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidClearcoat) {
            try PhysicallyBasedMaterial.Clearcoat(factor: 1.1)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidClearcoat) {
            try PhysicallyBasedMaterial.Clearcoat(roughness: -0.1)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidClearcoat) {
            try PhysicallyBasedMaterial.Clearcoat(normalScale: .nan)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ClearcoatRenderingTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let plane: MeshGeometry
    private let color: any MTLTexture
    private let depth: any MTLTexture
    private static let side = 33

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        plane = try MeshGeometry(vertices: [
            SIMD3<Float>(-1, -1, -2), SIMD3(1, -1, -2), SIMD3(1, 1, -2), SIMD3(-1, 1, -2),
        ].map { point in
            MeshVertex(
                position: point, normal: SIMD3(0, 0, 1),
                textureCoordinate: SIMD2(0.25 + 0.15 * point.x, 0.5 - 0.15 * point.y),
                secondaryTextureCoordinate: SIMD2(0.75 + 0.15 * point.x, 0.5 - 0.15 * point.y),
                tangent: SIMD4(1, 0, 0, -1)
            )
        }, indices: [0, 1, 2, 0, 2, 3])
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.side, height: Self.side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: Self.side, height: Self.side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        depth = try #require(device.makeTexture(descriptor: depthDescriptor))
    }

    @Test("Clearcoat maps use red, green, and independent normal UVs and scale")
    func directLayer() throws {
        let mask = try binding([0, 255, 0, 255, 128, 255, 0, 255], coordinates: .secondary)
        let roughness = try binding([255, 128, 0, 255, 255, 255, 0, 255], coordinates: .primary)
        let coatNormal = try binding([128, 128, 255, 255, 255, 128, 128, 255], coordinates: .secondary)
        let baseNormal = try binding([255, 128, 128, 255, 255, 128, 128, 255], coordinates: .primary)
        let texel = Float(128) / 255
        let baseline = try render()
        let disabled = try render(coat: .init(factor: 0, roughness: 0.3, factorTexture: mask))
        let reference = try render(coat: .init(factor: 0.5 * texel, roughness: 0.5 * texel))
        let mapped = try render(
            coat: .init(factor: 0.5, roughness: 0.5, factorTexture: mask, roughnessTexture: roughness),
            normal: baseNormal
        )
        let tilted = try render(coat: .init(
            factor: 0.5, roughness: 0.5, factorTexture: mask, roughnessTexture: roughness, normalTexture: coatNormal
        ))
        let zeroScale = try render(coat: .init(
            factor: 0.5, roughness: 0.5, factorTexture: mask, roughnessTexture: roughness,
            normalTexture: coatNormal, normalScale: 0
        ))

        #expect(baseline == SIMD4(0, 0, 0, 1))
        #expect(disabled == baseline)
        #expect(reference.x > 0.15 && reference.w == 1)
        #expect(simd_distance(mapped, reference) < 0.01)
        #expect(tilted.x < reference.x * 0.1)
        #expect(simd_distance(zeroScale, reference) < 0.01)
    }

    @Test("Environment clearcoat adds reflection and attenuates emission")
    func indirectLayer() throws {
        var hdr = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".utf8)
        for _ in 0..<8 { hdr.append(contentsOf: [160, 160, 160, 130]) }
        let lighting = try EnvironmentLighting(
            queue: queue, radianceHDR: hdr,
            settings: .init(cubeSize: 8, diffuseSize: 4, brdfSize: 16, sampleCount: 32)
        )
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting)
        let emission = SIMD3<Float>(0.5, 0.25, 0.1)
        let base = try render(intensity: 0, emission: emission, environment: environment)
        let coated = try render(coat: .init(factor: 1), intensity: 0, emission: emission, environment: environment)
        #expect(simd_distance(base, SIMD4(emission, 1)) < 0.005)
        #expect(simd_distance(coated, SIMD4(emission * 0.96 + SIMD3(repeating: 0.1), 1)) < 0.025)
    }

    private func binding(
        _ pixels: [UInt8], coordinates: MeshTextureCoordinateSet
    ) throws -> PhysicallyBasedMaterial.TextureBinding {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 2, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        try pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: 8
            )
        }
        return PhysicallyBasedMaterial.TextureBinding(texture: texture, textureCoordinateSet: coordinates)
    }

    private func render(
        coat: PhysicallyBasedMaterial.Clearcoat? = nil,
        normal: PhysicallyBasedMaterial.TextureBinding? = nil,
        intensity: Float = 1, emission: SIMD3<Float> = .zero,
        environment: PhysicallyBasedMaterial.Environment? = nil
    ) throws -> SIMD4<Float> {
        let material = try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0.75, directionToLight: SIMD3(0, 0, 1),
            lightIntensity: intensity, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
            normalTexture: normal, emissiveFactor: emission, environment: environment, clearcoat: coat
        )
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(material)
        )
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let command = try #require(queue.makeCommandBuffer())
        try renderer.encode(MeshScene(camera: camera), pass: pass, commandBuffer: command)
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(command.error?.localizedDescription ?? "Clearcoat draw failed.")")
        var pixel = SIMD4<Float16>.zero
        withUnsafeMutablePointer(to: &pixel) {
            color.getBytes(
                $0, bytesPerRow: MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(Self.side / 2, Self.side / 2, 1, 1), mipmapLevel: 0
            )
        }
        return SIMD4<Float>(pixel)
    }
}
