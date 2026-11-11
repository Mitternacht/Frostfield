import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct FlakesTextureTests {
    let device: any MTLDevice

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
    }

    @Test("Default flakes reproduce by seed and retain the r187 distribution")
    func seededDistribution() throws {
        let first = try FlakesTexture(device: device, seed: 82).normalTexture
        let second = try FlakesTexture(device: device, seed: 82).normalTexture
        let different = try FlakesTexture(device: device, seed: 83).normalTexture
        #expect(first.pixelFormat == .rgba8Unorm && first.usage.contains(.shaderRead))
        #expect(first.width == 512 && first.height == 512 && first.mipmapLevelCount == 10)
        let texels = Self.read(first)
        #expect(texels == Self.read(second))
        #expect(texels != Self.read(different))
        #expect(Self.read(first, level: 9) == Self.read(second, level: 9))

        let neutral = SIMD4<UInt8>(127, 127, 255, 255)
        var active = 0
        var total = SIMD3<Double>.zero
        var minBlue: UInt8 = 255
        for texel in texels {
            if texel != neutral { active += 1 }
            total += SIMD3(Double(texel.x), Double(texel.y), Double(texel.z))
            minBlue = min(minBlue, texel.z)
        }
        let coverage = Double(active) / Double(texels.count)
        let mean = total / Double(texels.count)
        // Pinned r187 Chromium canvas with the same seed stream:
        // 70.33% active, minimum blue 187, mean [126.44, 126.72, 236.38].
        #expect(abs(coverage - 0.7033) < 0.004)
        #expect(abs(Int(minBlue) - 187) <= 1)
        #expect(simd_reduce_max(abs(mean - SIMD3(126.44, 126.72, 236.38))) < 0.15)
        #expect(texels.allSatisfy { $0.w == 255 })
        #expect(Self.read(first, level: 9).first?.w == 255)
    }

    @Test("Legacy, tangent-space, and wrapped texels have their expected codes")
    func normalCodesAndEdges() throws {
        var settings = FlakesTexture.Settings(width: 64, height: 64, flakeCount: 1, mipmapped: false)
        let clipped = try FlakesTexture(device: device, seed: 82, settings: settings).normalTexture
        let neutral = SIMD4<UInt8>(127, 127, 255, 255)
        // r187 at (3, 49) for the first draw of the seeded sequence.
        let r187 = SIMD4<UInt8>(77, 175, 213, 255)
        let center = 49 * 64 + 3, across = 49 * 64 + 63
        let clippedCodes = Self.read(clipped)
        #expect(clippedCodes[center] == r187 && clippedCodes[across] == neutral)
        #expect(clippedCodes.contains { $0 != neutral && $0 != r187 })
        #expect(clipped.mipmapLevelCount == 1)

        settings.edges = .wrapped
        let wrapped = try FlakesTexture(device: device, seed: 82, settings: settings).normalTexture
        let wrappedCodes = Self.read(wrapped)
        #expect(wrappedCodes[center] == r187 && wrappedCodes[across] == r187)

        settings.encoding = .tangentSpaceRGB
        let tangent = try FlakesTexture(device: device, seed: 82, settings: settings).normalTexture
        let normalized = SIMD4<UInt8>(77, 176, 234, 255)
        let tangentCodes = Self.read(tangent)
        #expect(tangentCodes[center] == normalized && tangentCodes[across] == normalized)
        let point = SIMD3<Float>(3.5 / 64, 49.5 / 64, 0)
        let probe = try DDSProbe(device: device)
        let sample = try probe.sample(tangent, at: point)
        let expected = SIMD4<Float>(77, 176, 234, 255) / 255
        #expect(simd_reduce_max(abs(sample - expected)) < 0.0001)
        let normal = 2 * SIMD3(sample.x, sample.y, sample.z) - 1
        #expect(abs(simd_length(normal) - 1) < 0.006)
    }

    @Test("Zero flakes gives flat normals and invalid dimensions and work are rejected")
    func bounds() throws {
        let neutral = try FlakesTexture(device: device, seed: 0, settings: .init(
            width: 7, height: 3, flakeCount: 0, encoding: .tangentSpaceRGB
        )).normalTexture
        #expect(neutral.width == 7 && neutral.height == 3 && neutral.mipmapLevelCount == 3)
        #expect(Self.read(neutral).allSatisfy { $0 == SIMD4(128, 128, 255, 255) })
        let small = try FlakesTexture(device: device, seed: 82, settings: .init(
            width: 1, height: 1, flakeCount: 1, edges: .wrapped
        )).normalTexture
        #expect(Self.read(small) == [SIMD4(77, 175, 213, 255)])

        #expect(throws: FlakesTexture.Failure.dimensions) {
            try FlakesTexture(device: device, seed: 0, settings: .init(width: Int.max))
        }
        #expect(throws: FlakesTexture.Failure.radius) {
            try FlakesTexture(device: device, seed: 0, settings: .init(radius: 0...3))
        }
        #expect(throws: FlakesTexture.Failure.stampBudget) {
            try FlakesTexture(device: device, seed: 0, settings: .init(flakeCount: -1))
        }
        #expect(throws: FlakesTexture.Failure.stampBudget) {
            try FlakesTexture(device: device, seed: 0, settings: .init(flakeCount: 1_000, radius: 64...64))
        }
    }

    @Test("The flake map redistributes real raster glints through base and coat normals", arguments: [false, true])
    func rendersGlints(clearcoat: Bool) throws {
        let gpu = try PostProcessFixture()
        let flake = try FlakesTexture(device: gpu.device, seed: 82, settings: .init(
            width: 64, height: 64, flakeCount: 56, radius: 2...4, edges: .wrapped
        ))
        let binding = PhysicallyBasedMaterial.TextureBinding(texture: flake.normalTexture)
        let flat = try render(gpu: gpu, normal: nil, clearcoat: clearcoat)
        let textured = try render(gpu: gpu, normal: binding, clearcoat: clearcoat)
        var brighter = 0, dimmer = 0
        var peak: Float = 0
        for y in 8..<56 {
            for x in 8..<56 {
                let i = y * 64 + x
                let a = flat[i], b = textured[i]
                #expect(a.w == 1 && b.w == 1)
                let difference = max(b.x, b.y, b.z) - max(a.x, a.y, a.z)
                if difference > 0.015 { brighter += 1 }
                if difference < -0.015 { dimmer += 1 }
                peak = max(peak, difference)
            }
        }
        #expect(brighter >= 40, "\(brighter)")
        #expect(dimmer >= 100, "\(dimmer)")
        #expect(peak > 2, "\(peak)")
    }

    @Test("A seeded flake changes the path tracer's centered first-hit normal",
          arguments: PathAccelerationTests.modes)
    func pathNormal(acceleration: PathTracingScene.Options.Acceleration) throws {
        let gpu = try PathTestFixture()
        let flakes = try FlakesTexture(device: gpu.device, seed: 82, settings: .init(
            width: 64, height: 64, flakeCount: 1, encoding: .tangentSpaceRGB, edges: .wrapped
        ))
        let binding = PhysicallyBasedMaterial.TextureBinding(texture: flakes.normalTexture)
        let centerUV = SIMD2<Float>(3.5 / 64, 49.5 / 64)
        let geometry = try Self.plane(centerUV: centerUV, uvScale: 0.01)
        let side = 17
        func firstHit(_ binding: PhysicallyBasedMaterial.TextureBinding?) throws -> SIMD4<Float> {
            let material = try PhysicallyBasedMaterial(normalTexture: binding, alphaMode: .opaque)
            let mesh = try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
            )
            let graph = SceneGraph(roots: [SceneNode(mesh: mesh)], camera: gpu.camera)
            let scene = try PathTracingScene(
                queue: gpu.queue, graph: graph, options: .init(textures: .fullResolution(fallbackSize: 64), acceleration: acceleration)
            )
            let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: .init(environment: .zero))
            let accumulation = try tracer.makeAccumulation(width: side, height: side)
            let normal = try gpu.output(width: side, height: side)
            let beauty = try gpu.output(width: side, height: side)
            let command = try #require(gpu.queue.makeCommandBuffer())
            try tracer.encode(
                camera: gpu.camera, cameraTransform: .init(), destination: beauty, accumulation: accumulation,
                commandBuffer: command, settings: .init(maxBounces: 1), aovs: .init(normal: normal)
            )
            try gpu.finish(command)
            return try gpu.readback(normal)[(side / 2) * side + side / 2]
        }
        let flat = try firstHit(nil)
        let flake = try firstHit(binding)
        // The selected conventional RGB texel is (77, 176, 234). Texture V
        // points downward on this plane, so its world bitangent is negative Y.
        let decoded = SIMD3<Float>(77, 176, 234) * (2 / 255) - 1
        let expected = simd_normalize(SIMD3(decoded.x, -decoded.y, decoded.z))
        #expect(flat == SIMD4(0, 0, 1, 1))
        #expect(simd_distance(flake, SIMD4(expected, 1)) < 0.003, "\(flake), \(expected)")
    }

    private static func plane(
        centerUV: SIMD2<Float> = SIMD2(repeating: 0.5), uvScale: Float = 1 / 3.6
    ) throws -> MeshGeometry {
        try MeshGeometry(vertices: [
            SIMD3<Float>(-1.8, -1.8, -2), SIMD3(1.8, -1.8, -2), SIMD3(1.8, 1.8, -2), SIMD3(-1.8, 1.8, -2),
        ].map { point in
            MeshVertex(position: point, normal: SIMD3(0, 0, 1),
                       textureCoordinate: centerUV + SIMD2(point.x, -point.y) * uvScale, tangent: SIMD4(1, 0, 0, -1))
        }, indices: [0, 1, 2, 0, 2, 3])
    }

    private func render(
        gpu: PostProcessFixture, normal: PhysicallyBasedMaterial.TextureBinding?, clearcoat: Bool
    ) throws -> [SIMD4<Float>] {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        let plane = try Self.plane()
        let coat: PhysicallyBasedMaterial.Clearcoat? = clearcoat
            ? try .init(factor: 1, roughness: 0.13, normalTexture: normal, normalScale: 0.65)
            : nil
        let material = try PhysicallyBasedMaterial(
            metallic: clearcoat ? 1 : 0.85, roughness: clearcoat ? 0.75 : 0.18,
            directionToLight: SIMD3(0.5, 0.24, 1), lightIntensity: 3, ambientIntensity: 0,
            baseColorFactor: clearcoat ? SIMD4(0, 0, 0, 1) : SIMD4(0.014, 0.052, 0.24, 1),
            normalTexture: clearcoat ? nil : normal, normalScale: 0.65, clearcoat: coat
        )
        let renderer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(material)
        )
        let color = try gpu.color(width: 64, height: 64)
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: 64, height: 64, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(gpu.device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        try gpu.run { try renderer.encode(MeshScene(camera: camera), pass: pass, commandBuffer: $0) }
        return try gpu.read(color)
    }

    private static func read(_ texture: any MTLTexture, level: Int = 0) -> [SIMD4<UInt8>] {
        let width = max(1, texture.width >> level), height = max(1, texture.height >> level)
        var pixels = [SIMD4<UInt8>](repeating: .zero, count: width * height)
        pixels.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!, bytesPerRow: width * MemoryLayout<SIMD4<UInt8>>.stride,
                             from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: level)
        }
        return pixels
    }
}
