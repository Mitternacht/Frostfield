import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderMaterialMapsTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    @Test("A linear height ramp yields a constant tangent normal, wrapping at the tile edge")
    func rampNormals() throws {
        let maps = try ShaderMaterialMaps(queue: queue, width: 32, height: 16, heightScale: 0.5) { s in
            .init(baseColor: s.constant(SIMD3<Float>(repeating: 0.5)), height: s.gridUV.x)
        }
        let normalTexture = try #require(maps.normalTexture)
        #expect(normalTexture.pixelFormat == .rgba8Unorm)
        #expect(normalTexture.mipmapLevelCount == 6)
        let normals = bytes(normalTexture)

        // Slope = -2/32 height per pair of texels × heightScale × width / 2 = -0.5 everywhere inside.
        let inside = encoded(SIMD3(-0.5, 0, 1))
        #expect(inside == [70, 128, 242, 255])
        for y in 0..<16 {
            for x in 1..<31 {
                let index = (y * 32 + x) * 4
                #expect(close(Array(normals[index..<(index + 4)]), inside))
            }
        }
        // Wrapped neighbors at the seams see the full ramp jump.
        let heights = (0..<32).map { (Float($0) + 0.5) / 32 }
        let reference = referenceNormals(width: 32, height: 16, heightScale: 0.5) { x, _ in heights[x] }
        #expect(close(normals, reference))
        #expect(normals[0] > 250 && normals[31 * 4] > 250)

        let flat = try ShaderMaterialMaps(queue: queue, width: 32, height: 32, heightScale: 1) { s in
            .init(baseColor: s.constant(SIMD3<Float>(repeating: 0.5)), height: s.constant(Float(0.3)))
        }
        let flatNormals = bytes(try #require(flat.normalTexture))
        #expect(stride(from: 0, to: flatNormals.count, by: 4).allSatisfy {
            Array(flatNormals[$0..<($0 + 4)]) == [128, 128, 255, 255]
        })
        let none = try ShaderMaterialMaps(queue: queue, width: 4, height: 4) { s in
            .init(baseColor: s.constant(SIMD3<Float>(repeating: 0.5)))
        }
        #expect(none.normalTexture == nil)
    }

    @Test("Graph maps reproduce ProceduralSurface's color, packed data and normal conventions")
    func proceduralSurfaceParity() throws {
        let reference = try ProceduralSurface(device: device, resolution: 32, heightScale: 0.2) { uv in
            .init(
                baseColor: SIMD3(0.25, 0.5, 0.75), height: 0.5 + 0.25 * sin(2 * .pi * uv.x),
                roughness: uv.y, metallic: 0.5, occlusion: 0.25
            )
        }
        let maps = try ShaderMaterialMaps(queue: queue, width: 32, height: 32, heightScale: 0.2) { s in
            let uv = s.gridUV
            return .init(
                baseColor: s.constant(SIMD3<Float>(0.25, 0.5, 0.75)),
                roughness: uv.y, metallic: s.constant(Float(0.5)), occlusion: s.constant(Float(0.25)),
                height: 0.5 + 0.25 * (uv.x * (2 * Float.pi)).sin()
            )
        }
        let pairs: [(any MTLTexture, any MTLTexture)] = [
            (maps.baseColorTexture, reference.baseColorTexture),
            (maps.metallicRoughnessTexture, reference.metallicRoughnessTexture),
            (try #require(maps.normalTexture), reference.normalTexture),
        ]
        for (generated, expected) in pairs {
            #expect(generated.pixelFormat == expected.pixelFormat)
            #expect(generated.mipmapLevelCount == expected.mipmapLevelCount)
            #expect(close(bytes(generated), bytes(expected)))
            #expect(close(bytes(generated, level: 1), bytes(expected, level: 1), tolerance: 2))
        }
        #expect(Array(bytes(maps.metallicRoughnessTexture).prefix(4)) == [64, 4, 128, 255])
    }

    @Test("Generated maps plug into a physically based material that a mesh renderer draws")
    func physicallyBasedMaterial() throws {
        let maps = try ShaderMaterialMaps(queue: queue, width: 64, height: 64, heightScale: 0.05) { s in
            let grain = s.fractalNoise(s.vector(s.gridUV * 8, s.constant(Float(0))), octaves: 3)
            return .init(
                baseColor: s.uniform("Tint", SIMD3<Float>(0.8, 0.1, 0.05)) * grain.remap(0, 1, 0.6, 1),
                roughness: grain, metallic: s.constant(Float(0)), height: grain
            )
        }
        let material = try maps.material(lightIntensity: 0, ambientIntensity: 1)
        #expect(material.baseColorTexture === maps.baseColorTexture)
        #expect(material.alphaMode == .opaque)
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(), material: .physicallyBased(material)
        )

        // A generated half-float normal map is also accepted directly.
        let flat = try ShaderTextureGenerator(
            graph: TextureShaderGraph { s in s.constant(SIMD3<Float>(0.5, 0.5, 1)) }, device: device
        ).makeTexture(width: 4, height: 4, format: .rgba16Float, mipmapped: true, queue: queue)
        _ = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(),
            material: .physicallyBased(PhysicallyBasedMaterial(normalTexture: .init(texture: flat)))
        )

        let size = 33
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
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .dontCare
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let scene = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -1.2)), mesh: renderer)], camera: camera
        )
        let commandBuffer = try #require(queue.makeCommandBuffer())
        try scene.encode(pass: pass, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        var pixel = SIMD4<Float16>.zero
        withUnsafeMutableBytes(of: &pixel) { raw in
            color.getBytes(
                raw.baseAddress!, bytesPerRow: 8, from: MTLRegionMake2D(size / 2, size / 2, 1, 1), mipmapLevel: 0
            )
        }
        let rendered = SIMD4<Float>(pixel)
        #expect(rendered.w == 1)
        #expect(rendered.x > 0.05 && rendered.x > 3 * rendered.y && rendered.x > 3 * rendered.z)
    }

    @Test("Material maps validate the height scale, dimensions and graph inputs")
    func validation() throws {
        let color: (ShaderGraphBuilder) -> ShaderMaterialMaps.Channels = { s in
            .init(baseColor: s.constant(SIMD3<Float>(repeating: 0.5)))
        }
        #expect(throws: ShaderMaterialMaps.Failure.heightScale) {
            try ShaderMaterialMaps(queue: queue, width: 4, height: 4, heightScale: 2, building: color)
        }
        #expect(throws: ShaderMaterialMaps.Failure.heightScale) {
            try ShaderMaterialMaps(queue: queue, width: 4, height: 4, heightScale: .nan, building: color)
        }
        #expect(throws: ShaderComputeFailure.dimensions(width: 0, height: 4)) {
            try ShaderMaterialMaps(queue: queue, width: 0, height: 4, building: color)
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try ShaderMaterialMaps(queue: queue, width: 4, height: 4) { s in
                .init(baseColor: s.viewDirection)
            }
        }
        #expect(throws: ShaderComputeFailure.missingTexture("Albedo")) {
            try ShaderMaterialMaps(queue: queue, width: 4, height: 4) { s in
                .init(baseColor: s.texture("Albedo").sample(s.gridUV).rgb)
            }
        }
    }

    // MARK: - Helpers

    /// ProceduralSurface's CPU normal derivation.
    private func referenceNormals(
        width: Int, height: Int, heightScale: Float, heightAt: (Int, Int) -> Float
    ) -> [UInt8] {
        let slopeScale = Float(width) * heightScale * 0.5
        var result: [UInt8] = []
        for y in 0..<height {
            for x in 0..<width {
                let slopeX = (heightAt((x + width - 1) % width, y) - heightAt((x + 1) % width, y)) * slopeScale
                let slopeY = (heightAt(x, (y + height - 1) % height) - heightAt(x, (y + 1) % height)) * slopeScale
                result += encoded(SIMD3(slopeX, slopeY, 1))
            }
        }
        return result
    }

    private func encoded(_ normal: SIMD3<Float>) -> [UInt8] {
        let value = simd_normalize(normal) * 0.5 + 0.5
        return [value.x, value.y, value.z].map { UInt8((min(max($0, 0), 1) * 255).rounded()) } + [255]
    }

    private func close(_ a: [UInt8], _ b: [UInt8], tolerance: Int = 1) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { abs(Int($0) - Int($1)) <= tolerance }
    }

    private func bytes(_ texture: any MTLTexture, level: Int = 0) -> [UInt8] {
        let width = max(1, texture.width >> level)
        let height = max(1, texture.height >> level)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes { raw in
            if let address = raw.baseAddress {
                texture.getBytes(
                    address, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: level
                )
            }
        }
        return bytes
    }
}
