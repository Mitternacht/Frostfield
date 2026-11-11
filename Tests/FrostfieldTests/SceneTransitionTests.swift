import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneTransitionTests {
    private let gpu: PostProcessFixture
    private let fromColor = SIMD4<Float>(2, 0.125, 0, 0.5)
    private let toColor = SIMD4<Float>(0, 0.25, 1.5, 0.75)

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Scene transitions preserve HDR and alpha at crossfade endpoints and in reverse")
    func crossfade() throws {
        let from = try gpu.color(width: 4, height: 2) { _, _ in fromColor }
        let to = try gpu.color(width: 4, height: 2) { _, _ in toColor }
        let effect = try SceneTransitionPass(device: gpu.device, toTexture: to)
        for (progress, reversed, weight): (Float, Bool, Float) in [
            (0, false, 0), (0.25, false, 0.25), (1, false, 1), (0, true, 1), (0.75, true, 0.25), (1, true, 0)
        ] {
            effect.settings = .init(progress: progress, reversed: reversed)
            let result = try render(effect, source: from)
            let expected = mixed(weight)
            #expect(result.allSatisfy { $0 == expected }, "\(result[0]), \(expected)")
        }
    }

    @Test("The native transition and graph agree on r187 mask feathering, hard cuts and mask reversal")
    func maskReadback() throws {
        let from = try gpu.color(width: 4, height: 2) { _, _ in fromColor }
        let to = try gpu.color(width: 4, height: 2) { _, _ in toColor }
        let mask = try heightMask([0, 0.375, 0.5, 1])
        let effect = try SceneTransitionPass(device: gpu.device, toTexture: to, mask: mask)
        let settings: [(progress: Float, threshold: Float, reversed: Bool, inverted: Bool, expected: [Float])] = [
            (0.375, 0.125, false, false, [1, 0.75, 0, 0]),
            (0.625, 0.125, true, false, [1, 0.75, 0, 0]),
            (0.5, 0, false, false, [1, 1, 1, 0]),
            (0.5, 0, false, true, [0, 0, 1, 1]),
            (0, 0, false, false, [0, 0, 0, 0]),
            (1, 0.125, false, true, [1, 1, 1, 1]),
            (0, 0.125, true, true, [1, 1, 1, 1]),
        ]
        for config in settings {
            effect.settings = .init(
                progress: config.progress, threshold: config.threshold, reversed: config.reversed, invertMask: config.inverted
            )
            let native = try render(effect, source: from)
            let graph = try PostProcessGraph { p in
                let toColor = p.texture("second", addressing: .clampToEdge).sample(p.screenUV)
                let red = p.texture("mask", content: .data, addressing: .clampToEdge).sample(p.screenUV).r
                return p.transition(
                    from: p.sceneColor(at: p.screenUV), to: toColor,
                    progress: p.uniform("Progress", config.progress), maskRed: red,
                    threshold: p.uniform("Threshold", config.threshold), reversed: config.reversed, invertMask: config.inverted
                )
            }
            let pass = try PostProcessPass(graph: graph, device: gpu.device, textures: ["second": to, "mask": mask])
            let destination = try gpu.color(width: 4, height: 2)
            try gpu.run { try pass.encode(source: from, destination: destination, commandBuffer: $0) }
            let composed = try gpu.read(destination)
            #expect(composed == native)
            #expect(native == (config.expected + config.expected).map(mixed))
        }
    }

    @Test("A transition chain composites two scene graphs rendered on the same command buffer")
    func sceneChain() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let first = try scene(camera: camera, cameraX: 0.5, color: SIMD3(1, 0, 0))
        let second = try scene(camera: camera, cameraX: -0.5, color: SIMD3(0, 0, 1))
        let from = try gpu.color(width: 16, height: 16)
        let to = try gpu.color(width: 16, height: 16)
        let output = try gpu.color(width: 16, height: 16)
        let fromPass = try scenePass(for: from)
        let toPass = try scenePass(for: to)
        let transition = try SceneTransitionPass(device: gpu.device, toTexture: to, settings: .init(progress: 0.5))
        let swap = try PostProcessGraph { p in p.sceneColor(at: p.screenUV).swizzle("bgra", as: SIMD4<Float>.self) }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(transition), .graph(swap)])
        let alternate = try gpu.color(width: 16, height: 16) { _, _ in SIMD4(0, 1, 0, 1) }
        let replacement = try gpu.color(width: 16, height: 16)
        try gpu.run { buffer in
            try first.encode(pass: fromPass, commandBuffer: buffer)
            try second.encode(pass: toPass, commandBuffer: buffer)
            try chain.encode(source: from, destination: output, commandBuffer: buffer)

            // The prior encode retains the old second texture and copied settings.
            transition.toTexture = alternate
            transition.settings.progress = 1
            try chain.encode(source: from, destination: replacement, commandBuffer: buffer)
        }
        let a = try gpu.read(from), b = try gpu.read(to), result = try gpu.read(output)
        #expect(a.contains { $0.x == 1 } && b.contains { $0.z == 1 } && a != b)
        for i in result.indices {
            let expected = (a[i] + b[i]) * 0.5
            #expect(result[i] == SIMD4(expected.z, expected.y, expected.x, expected.w))
        }
        #expect(try gpu.read(replacement).allSatisfy { $0 == SIMD4(0, 1, 0, 1) })
    }

    @Test("Transitions reject writable feedback, malformed settings, and incompatible textures")
    func validation() throws {
        let from = try gpu.color(width: 4, height: 2) { _, _ in fromColor }
        let output = try gpu.color(width: 4, height: 2)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 4, height: 2, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .renderTarget, .pixelFormatView]
        let to = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let view = try #require(to.makeTextureView(
            pixelFormat: .rgba16Float, textureType: .type2D, levels: 0..<1, slices: 0..<1
        ))
        let effect = try SceneTransitionPass(device: gpu.device, toTexture: to, settings: .init(progress: 0.5))
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(view !== to)
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: from, destination: from, commandBuffer: command)
        }
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: from, destination: view, commandBuffer: command)
        }
        effect.mask = output
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: from, destination: output, commandBuffer: command)
        }
        effect.mask = try gpu.color(width: 2, height: 1, pixelFormat: .rgba8Unorm_srgb)
        #expect(throws: PostProcessEffectError.unsupportedInput(
            "The transition mask must be a shader-readable, tracked, linear 2D texture."
        )) {
            try effect.encode(source: from, destination: output, commandBuffer: command)
        }
        effect.mask = nil
        effect.toTexture = try gpu.color(width: 2, height: 1)
        #expect(throws: PostProcessEffectError.sizeMismatch) {
            try effect.encode(source: from, destination: output, commandBuffer: command)
        }
        effect.toTexture = from
        effect.settings.threshold = -.infinity
        #expect(throws: PostProcessEffectError.invalidSettings("Transition progress and threshold must be from 0 to 1.")) {
            try effect.encode(source: from, destination: output, commandBuffer: command)
        }
        effect.settings = .init(progress: 0.5)
        effect.mask = from
        // Read-only aliases are permitted, including a mask taken from a source.
        #expect(try render(effect, source: from).allSatisfy { $0 == fromColor })
    }

    private func render(_ effect: SceneTransitionPass, source: any MTLTexture) throws -> [SIMD4<Float>] {
        let destination = try gpu.color(width: source.width, height: source.height)
        try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
        return try gpu.read(destination)
    }

    private func mixed(_ weight: Float) -> SIMD4<Float> {
        fromColor * (1 - weight) + toColor * weight
    }

    private func heightMask(_ samples: [Float16]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: samples.count, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let mask = try #require(gpu.device.makeTexture(descriptor: descriptor))
        samples.withUnsafeBytes { bytes in
            mask.replace(region: MTLRegionMake2D(0, 0, samples.count, 1), mipmapLevel: 0,
                         withBytes: bytes.baseAddress!, bytesPerRow: bytes.count)
        }
        return mask
    }

    private func scene(camera: PerspectiveCamera, cameraX: Float, color: SIMD3<Float>) throws -> SceneGraph {
        let geometry = try MeshGeometry.plane(width: 0.75, height: 1)
        let material = try ClassicMaterial(model: .basic, color: color)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .classic(material)
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        return SceneGraph(roots: [node], camera: camera, cameraTransform: .init(translation: SIMD3(cameraX, 0, 0)))
    }

    private func scenePass(for color: any MTLTexture) throws -> MTLRenderPassDescriptor {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: color.width, height: color.height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        pass.depthAttachment.texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        return pass
    }
}
