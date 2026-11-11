import Foundation
import Frostfield
import Metal
import simd
import Testing

struct TransmissionTests {
    @Test("Transmission validates volume values and supports factor-zero IOR")
    func validatesInputs() throws {
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidTransmission) {
            try PhysicallyBasedMaterial.Transmission(factor: 1.1)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidTransmission) {
            try PhysicallyBasedMaterial.Transmission(factor: 1, attenuationDistance: 0)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidIndexOfRefraction) {
            try PhysicallyBasedMaterial.Transmission(factor: 0, indexOfRefraction: 0.9)
        }
        let mirror = try PhysicallyBasedMaterial.Transmission(factor: 0, indexOfRefraction: 0)
        #expect(mirror.indexOfRefraction == 0)
        let volume = try PhysicallyBasedMaterial.Transmission(factor: 1, thickness: 2, attenuationDistance: .infinity)
        #expect(volume.attenuationDistance == nil)
        #expect(try !PhysicallyBasedMaterial(isDoubleSided: true, transmission: volume).isDoubleSided)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TransmissionRenderingTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private static let side = 65

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
    }

    @Test("Transmission tints HDR, samples independent map channels and scales Beer travel in world space")
    func tintedVolume() throws {
        let plane = try quad(x: -1...1, y: -1...1)
        let clear = MTLClearColorMake(2, 1, 0.5, 1)
        let transform = SceneTransform(translation: SIMD3(0, 0, -2), scale: SIMD3(1, 1, 2))
        let thin = try glass(plane, transform: transform, tint: SIMD4(0.8, 0.6, 1, 1),
                             optical: .init(factor: 1, indexOfRefraction: 1))
        let tinted = try render([thin], clear: clear)
        #expect(simd_distance(tinted, SIMD4(1.6, 0.6, 0.5, 1)) < 0.012)

        let factorMap = try binding([0, 255, 255, 255, 128, 0, 255, 255], coordinates: .secondary)
        let thicknessMap = try binding([255, 128, 0, 255, 255, 0, 0, 255], coordinates: .primary)
        let thick = try glass(plane, transform: transform, optical: .init(
            factor: 1, indexOfRefraction: 1, thickness: 0.5, attenuationDistance: 1,
            attenuationColor: SIMD3(0.5, 0.25, 1), factorTexture: factorMap, thicknessTexture: thicknessMap
        ))
        let absorbed = try render([thick], clear: clear)
        let channel = Float(128) / 255
        let expected = SIMD4(
            2 * pow(Float(0.5), channel) * channel, pow(Float(0.25), channel) * channel,
            0.5 * channel, 1
        )
        #expect(simd_distance(absorbed, expected) < 0.012)
    }

    @Test("A modeled glass exit projects across an opaque color boundary and roughness blurs it")
    func screenRefraction() throws {
        let left = try background(x: -6 ... -0.4, color: SIMD4(1, 0, 0, 1))
        let right = try background(x: -0.4...6, color: SIMD4(0, 1, 0, 1))
        let tiltedNormal = simd_normalize(SIMD3<Float>(1, 0, 1))
        let plane = try quad(x: -1...1, y: -1...1, normal: tiltedNormal)
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let thin = try glass(plane, transform: transform, optical: .init(factor: 1))
        let thick = try glass(plane, transform: transform, optical: .init(factor: 1, thickness: 2))
        let rough = try glass(plane, transform: transform, roughness: 1, optical: .init(factor: 1))

        let unrefracted = try render([thin, left, right])
        let refracted = try render([thick, left, right])
        let blurred = try render([rough, left, right])
        #expect(unrefracted.y > 0.18 && unrefracted.x < 0.01)
        #expect(refracted.x > 0.18 && refracted.y < 0.01)
        #expect(blurred.x > 0.05 && blurred.y > 0.05)
    }

    @Test("Factor-zero custom IOR changes opaque dielectric reflection without a backdrop")
    func opaqueIOR() throws {
        let plane = try quad(x: -1...1, y: -1...1)
        func renderIOR(_ ior: Float, factor: Float = 0) throws -> SIMD4<Float> {
            let material = try PhysicallyBasedMaterial(
                roughness: 0.5, directionToLight: SIMD3(0, 0, 1), ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), transmission: .init(factor: factor, indexOfRefraction: ior)
            )
            let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: plane,
                                        material: .physicallyBased(material))
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
            return try render([node], supplyBackdrop: false)
        }
        let matched = try renderIOR(1)
        let glass = try renderIOR(2)
        let mirror = try renderIOR(0)
        #expect(matched.x < 0.001 && glass.x > 0.1 && mirror.x > glass.x * 5)
        #expect(try renderIOR(0, factor: 1) == mirror)
    }

    private func quad(
        x: ClosedRange<Float>, y: ClosedRange<Float>, normal: SIMD3<Float> = SIMD3(0, 0, 1)
    ) throws -> MeshGeometry {
        let points = [
            SIMD3(x.lowerBound, y.lowerBound, 0), SIMD3(x.upperBound, y.lowerBound, 0),
            SIMD3(x.upperBound, y.upperBound, 0), SIMD3(x.lowerBound, y.upperBound, 0),
        ]
        return try MeshGeometry(vertices: points.map {
            MeshVertex(
                position: $0, normal: normal, textureCoordinate: SIMD2(0.25, 0.5),
                secondaryTextureCoordinate: SIMD2(0.75, 0.5)
            )
        }, indices: [0, 1, 2, 0, 2, 3])
    }

    private func background(x: ClosedRange<Float>, color: SIMD4<Float>) throws -> SceneNode {
        let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: quad(x: x, y: -6...6))
        return SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: mesh, materialParameters: color)
    }

    private func glass(
        _ geometry: MeshGeometry, transform: SceneTransform, tint: SIMD4<Float> = SIMD4(repeating: 1),
        roughness: Float = 0, optical: PhysicallyBasedMaterial.Transmission
    ) throws -> SceneNode {
        let material = try PhysicallyBasedMaterial(
            roughness: roughness, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: tint,
            alphaMode: .opaque, transmission: optical
        )
        let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
                                    material: .physicallyBased(material))
        return SceneNode(transform: transform, mesh: mesh)
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
        _ nodes: [SceneNode], clear: MTLClearColor = MTLClearColorMake(0, 0, 0, 1),
        supplyBackdrop: Bool = true
    ) throws -> SIMD4<Float> {
        let side = Self.side
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        var backdrop: (any MTLTexture)?
        if supplyBackdrop {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: side, height: side, mipmapped: true
            )
            descriptor.storageMode = .private
            descriptor.usage = .shaderRead
            let texture: any MTLTexture = try #require(device.makeTexture(descriptor: descriptor))
            backdrop = texture
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = clear
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try SceneGraph(roots: nodes, camera: camera).encode(
            pass: pass, commandBuffer: buffer, transmissionBackdrop: backdrop
        )
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "Transmission draw failed.")")
        var pixel = SIMD4<Float16>.zero
        withUnsafeMutablePointer(to: &pixel) {
            color.getBytes(
                $0, bytesPerRow: MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(side / 2, side / 2, 1, 1), mipmapLevel: 0
            )
        }
        return SIMD4<Float>(pixel)
    }
}
