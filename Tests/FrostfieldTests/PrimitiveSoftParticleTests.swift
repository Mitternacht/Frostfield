import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PrimitiveSoftParticleTests {
    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture(side: 128) }

    @Test("Textured points and sprites fade their premultiplied color against captured opaque depth",
          arguments: [MeshDepthMode.forward, .reversed], [false, true])
    func capturedFade(mode: MeshDepthMode, sprite: Bool) throws {
        let fade = try SoftParticleDepth.Fade()
        let position = SIMD3<Float>(0, 0, -2)
        let color = SIMD4<Float>(1, 0.5, 0.25, 0.8)
        let map = try colorMap()
        let cutoff: Float = 0.02
        let primitive: ScenePrimitive
        if sprite {
            primitive = .sprites(try SpriteRenderer(
                device: gpu.device, sprites: [.init(position: position, size: SIMD2(repeating: 80))],
                material: .init(
                    map: map, color: color, sizeAttenuation: false, alphaTest: cutoff, blending: .additive,
                    softParticleFade: fade
                )
            ))
        } else {
            primitive = .points(try PointsRenderer(
                device: gpu.device, cloud: PointCloud(positions: [position]),
                material: .init(size: .pixels(80), color: color, map: map, alphaTest: cutoff, blending: .alpha,
                                softParticleFade: fade)
            ))
        }
        let back = SIMD3<Float>(0.125, 0.25, 0.375)
        let corners = [SIMD2<Float>(-1.25, -0.25), SIMD2(1.25, -0.25), SIMD2(1.25, 0.25), SIMD2(-1.25, 0.25)]
        let plane = try MeshGeometry(vertices: corners.map {
            MeshVertex(position: SIMD3($0.x, $0.y, -2.5 - $0.x), normal: SIMD3(0, 0, 1))
        }, indices: [0, 1, 2, 0, 2, 3])
        let background = try gpu.unlit(plane, material: .classic(ClassicMaterial(model: .basic, color: back)), depthMode: mode)
        var scene = SceneGraph(roots: [SceneNode(primitive: primitive), SceneNode(mesh: background)], camera: gpu.camera)
        let snapshot = try captureDepth()
        scene.viewportSnapshot = snapshot
        let pixels = try gpu.render(scene, depthMode: mode).pixels
        #expect(gpu.pixel(pixels, x: 42, y: 64) == SIMD4(back, 1))
        for x in [49, 52, 59, 64, 71, 76] {
            let u = (Float(x) + 0.5) / Float(gpu.side) * 2 - 1
            let gap = min(max(2.5 / (1 - u) - 2, 0), 1)
            let folded = 2 * min(gap, 1 - gap)
            let curve = gap > 0.5 ? 1 - 0.5 * folded * folded : 0.5 * folded * folded
            let alpha = 0.4 * curve
            let mixed = back * (sprite ? 1 : 1 - alpha) + SIMD3(0.5, 0.5, 0.25) * alpha
            let expected = alpha < cutoff ? back : mixed
            #expect(simd_distance(gpu.pixel(pixels, x: x, y: 64), SIMD4(expected, 1)) < 0.002)
        }
        #expect(simd_distance(gpu.pixel(pixels, x: 64, y: 46), SIMD4(0.2, 0.2, 0.1, 0.4)) < 0.002)

        if mode == .forward {
            let archive = try SceneArchive(capturing: scene, commandQueue: gpu.queue)
            let manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
            #expect(manifest["version"] as? Int == 20)
            var restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
            #expect(restored.viewportSnapshot == nil)
            restored.viewportSnapshot = snapshot
            #expect(try gpu.render(restored).pixels == pixels)
        }
    }

    @Test("A multisampled point pass can fade against a separate single-sample capture",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsTextureSampleCount(4) == true, "Requires 4× raster sampling"))
    func multisampled() throws {
        let floor = try gpu.unlit(.plane(width: 4, height: 4))
        let producer = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2.25)), mesh: floor)
        ], camera: gpu.camera)
        let points = try PointsRenderer(
            device: gpu.device, sampleCount: 4, cloud: PointCloud(positions: [SIMD3(0, 0, -2)]),
            material: .init(size: .pixels(40), blending: .alpha, softParticleFade: SoftParticleDepth.Fade())
        )
        var scene = SceneGraph(roots: [SceneNode(primitive: .points(points))], camera: gpu.camera)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let frame = try PostProcessScenePass(queue: gpu.queue).encode(
            producer, width: gpu.side, height: gpu.side, commandBuffer: command
        )
        scene.viewportSnapshot = try SceneViewportSnapshot(depth: frame.depth)
        let resolved = try gpu.makeTarget().color
        func attachment(_ format: MTLPixelFormat) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: gpu.side, height: gpu.side, mipmapped: false
            )
            descriptor.textureType = .type2DMultisample
            descriptor.sampleCount = 4
            descriptor.storageMode = .private
            descriptor.usage = .renderTarget
            return try #require(gpu.device.makeTexture(descriptor: descriptor))
        }
        let pass = try gpu.pass(color: attachment(.rgba16Float), depth: attachment(.depth32Float))
        pass.colorAttachments[0].resolveTexture = resolved
        pass.colorAttachments[0].storeAction = .multisampleResolve
        try scene.encode(pass: pass, commandBuffer: command)
        try gpu.finish(command)
        #expect(simd_distance(gpu.pixel(try gpu.pixels(of: resolved), x: 64, y: 64), SIMD4(repeating: 0.125)) < 0.002)
    }

    @Test("Soft primitives require blended materials and a stored depth input")
    func validation() throws {
        let cloud = try PointCloud(positions: [SIMD3(0, 0, -2)])
        let fade = try SoftParticleDepth.Fade(distance: 0.5, contrast: 1)
        #expect(throws: PrimitiveRendererError.softParticleBlending) {
            try PointsRenderer(device: gpu.device, cloud: cloud, material: .init(softParticleFade: fade))
        }
        let points = try PointsRenderer(
            device: gpu.device, cloud: cloud, material: .init(blending: .alpha, softParticleFade: fade)
        )
        let scene = SceneGraph(roots: [SceneNode(primitive: .points(points))], camera: gpu.camera)
        #expect(throws: SceneViewportSnapshot.Failure.missingDepth) { try gpu.render(scene) }
    }

    private func captureDepth() throws -> SceneViewportSnapshot {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let depth = try #require(gpu.device.makeTexture(descriptor: descriptor))
        return try SceneViewportSnapshot(depth: depth, source: .opaquePass)
    }

    private func colorMap() throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        var pixel = SIMD4<Float16>(0.5, 1, 1, 0.5)
        withUnsafeBytes(of: &pixel) { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: 8
            )
        }
        return texture
    }
}
