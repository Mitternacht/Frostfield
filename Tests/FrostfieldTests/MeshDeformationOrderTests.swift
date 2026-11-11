import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshDeformationOrderTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Ordinary color draws keep their typed displacement across reverse command submission")
    func reverseSubmission() throws {
        let (mesh, uniforms) = try makeMesh()
        let scene = MeshScene(
            camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10),
            transform: .init(translation: SIMD3(0, 0, -1))
        )
        let colors = try (0..<3).map { _ in try gpu.color(width: 8, height: 8) }
        let passes = try colors.map { try pass(color: $0) }
        let first = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        let next = try #require(gpu.queue.makeCommandBuffer())
        let duplicate = try #require(gpu.queue.makeCommandBuffer())
        try mesh.encode(scene, pass: passes[0], commandBuffer: first)
        try uniforms.set("Offset", to: Float(0.625))
        try mesh.encode(scene, pass: passes[1], commandBuffer: next)
        try mesh.encode(scene, pass: passes[2], commandBuffer: duplicate)
        try withExtendedLifetime(passes) {
            try complete([duplicate, next, first])
        }

        try expectColumns(colors[0], [1])
        try expectColumns(colors[1], [6])
        try expectColumns(colors[2], [6])
        try gpu.run { try mesh.encode(scene, pass: passes[2], commandBuffer: $0) }
        #expect(mesh.deformationEncodeCount == 3, "Completing an older pose out of order keeps the newer cached pose.")
    }

    @Test("A command can reuse its pending pose and later commands can lease the completed result")
    func sameCommandAndAbandonment() throws {
        let (mesh, uniforms) = try makeMesh()
        let scene = MeshScene(
            camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10),
            transform: .init(translation: SIMD3(0, 0, -1))
        )
        let images = try (0..<3).map { _ in try gpu.color(width: 8, height: 8) }
        let passes = try images.map { try pass(color: $0) }
        try autoreleasepool {
            let abandoned = try #require(gpu.queue.makeCommandBuffer())
            try mesh.encode(scene, pass: passes[0], commandBuffer: abandoned)
        }
        try gpu.run { buffer in
            try mesh.encode(scene, pass: passes[0], commandBuffer: buffer)
            try mesh.encode(scene, pass: passes[1], commandBuffer: buffer)
            #expect(mesh.deformationEncodeCount == 2)
            try uniforms.set("Offset", to: Float(0.625))
            try mesh.encode(scene, pass: passes[2], commandBuffer: buffer)
        }
        try expectColumns(images[0], [1])
        try expectColumns(images[1], [1])
        try expectColumns(images[2], [6])
        #expect(mesh.deformationEncodeCount == 3)

        let retainedPose = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        try mesh.encode(scene, pass: passes[1], commandBuffer: retainedPose)
        #expect(mesh.deformationEncodeCount == 3)
        try uniforms.set("Offset", to: Float(-0.625))
        try gpu.run { try mesh.encode(scene, pass: passes[0], commandBuffer: $0) }
        try withExtendedLifetime(passes) { try complete([retainedPose]) }
        try expectColumns(images[0], [1])
        try expectColumns(images[1], [6])
        #expect(mesh.deformationEncodeCount == 4)
    }

    @Test("Scene color, attributes, shadow, and motion agree on a pose first submitted from another command")
    func auxiliaryDrawsUseLease() throws {
        let (mesh, uniforms) = try makeMesh(receivesShadows: true)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let scene = SceneGraph(roots: [
            .init(transform: .init(translation: SIMD3(0, 0, -1)), mesh: mesh),
        ], camera: camera)
        let shadows = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 16, directionToLight: SIMD3(0, 0, 1),
            center: SIMD3(0, 0, -1), halfExtent: 1
        )
        let viewer = try TextureDebugPass(device: gpu.device)
        let shadowPixels = try gpu.color(width: 16, height: 16)
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let previousMotion = try gpu.color(width: 8, height: 8)
        let previous = motion.recording(to: previousMotion)
        let previousPass = try pass(color: gpu.color(width: 8, height: 8))
        try gpu.run { buffer in
            _ = try scene.encode(pass: previousPass, commandBuffer: buffer, shadowMap: shadows, motionVectors: previous)
        }
        let history = try #require(previous.history)

        try uniforms.set("Offset", to: Float(0.625))
        let neverSubmitted = try #require(gpu.queue.makeCommandBuffer())
        let unusedPass = try pass(color: gpu.color(width: 8, height: 8))
        _ = try scene.encode(pass: unusedPass, commandBuffer: neverSubmitted, shadowMap: shadows)

        let image = try gpu.color(width: 8, height: 8)
        let attributes = try gpu.color(width: 8, height: 8, pixelFormat: SceneGraph.surfaceAttributesPixelFormat)
        let colorPass = try pass(color: image)
        let attributePass = try pass(color: attributes)
        let vectors = try gpu.color(width: 8, height: 8)
        let recording = motion.recording(to: vectors, previous: history)
        try withExtendedLifetime(neverSubmitted) {
            try gpu.run { buffer in
                let statistics = try scene.encode(
                    pass: colorPass, commandBuffer: buffer, shadowMap: shadows, motionVectors: recording
                )
                #expect(statistics.shadowDrawCount == 1)
                _ = try scene.encodeSurfaceAttributes(pass: attributePass, commandBuffer: buffer)
                try viewer.encode(source: .directionalShadow(shadows, cascade: 0), destination: shadowPixels, commandBuffer: buffer)
            }
        }
        #expect(mesh.deformationEncodeCount == 3, "Attributes reuse the pose encoded by the color and shadow passes.")
        try expectColumns(image, [6])

        var packed = Array(repeating: SIMD2<UInt32>.zero, count: 64)
        try packed.withUnsafeMutableBytes { bytes in
            attributes.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: 8 * MemoryLayout<SIMD2<UInt32>>.stride,
                from: MTLRegionMake2D(0, 0, 8, 8), mipmapLevel: 0
            )
        }
        #expect(packed[4 * 8 + 1] == .zero && packed[4 * 8 + 6] != .zero)

        let velocity = try gpu.read(vectors)
        let oldClip = try camera.projection(aspectRatio: 1) * SIMD4(-0.625, 0, -1, 1)
        #expect(velocity[4 * 8 + 1] == .zero)
        let moved = velocity[4 * 8 + 6]
        #expect(abs(moved.x + 5) < 0.002 && abs(moved.y) < 0.002 && moved.w == 1)
        #expect(abs(moved.z - oldClip.z / oldClip.w) < 0.001)

        let shadow = try gpu.read(shadowPixels)
        func shadowDepth(at x: Float) -> Float {
            let clip = shadows.clipFromWorld * SIMD4(x, 0, -1, 1)
            let column = Int((clip.x / clip.w * 0.5 + 0.5) * 16)
            let row = Int((0.5 - clip.y / clip.w * 0.5) * 16)
            // The viewer displays forward shadow depth as one minus the stored value.
            return 1 - shadow[row * 16 + column].x
        }
        #expect(shadowDepth(at: -0.625) == 1)
        #expect(abs(shadowDepth(at: 0.625) - 0.5) < 0.001)
    }

    private func makeMesh(receivesShadows: Bool = false) throws -> (MeshRenderer, ShaderUniforms) {
        let graph = try SurfaceShaderGraph { graph in
            let offset = graph.uniform("Offset", Float(-0.625))
            let zero = graph.constant(Float(0))
            return .init(positionOffset: graph.vector(offset, zero, zero))
        }
        let material = try PhysicallyBasedMaterial(
            receivesShadows: receivesShadows, baseColorFactor: SIMD4(1, 0.5, 0.25, 1),
            surfaceGraph: graph, shadingModel: .unlit
        )
        let renderer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float,
            geometry: .plane(width: 0.25, height: 2), material: .physicallyBased(material)
        )
        return (renderer, try #require(material.surfaceUniforms))
    }

    private func pass(color: any MTLTexture) throws -> MTLRenderPassDescriptor {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: color.width, height: color.height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        let depth = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor()
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .store
        return pass
    }

    private func expectColumns(_ texture: any MTLTexture, _ columns: Set<Int>) throws {
        let output = SIMD4<Float>(1, 0.5, 0.25, 1)
        let expected = (0..<(texture.width * texture.height)).map { columns.contains($0 % texture.width) ? output : .zero }
        let pixels = try gpu.read(texture)
        #expect(pixels == expected, "first row: \(pixels.prefix(texture.width))")
    }

    private func complete(_ commands: [any MTLCommandBuffer]) throws {
        for command in commands { command.commit() }
        for command in commands {
            command.waitUntilCompleted()
            try #require(command.status == .completed, "\(String(describing: command.error))")
        }
    }
}
