import Frostfield
import Metal
import Testing

struct MaterialGraphTests {
    @Test("Graph validation rejects missing references, disconnected cycles, and scalar outputs")
    func graphStructure() {
        #expect(throws: MaterialGraph.Failure.invalidOutput(0)) {
            try MaterialGraph(nodes: [], output: 0)
        }
        #expect(throws: MaterialGraph.Failure.invalidReference(node: 1, referenced: 2)) {
            try MaterialGraph(nodes: [.color(.zero), .add(0, 2)], output: 1)
        }
        #expect(throws: MaterialGraph.Failure.cycle(1)) {
            try MaterialGraph(nodes: [.color(.zero), .sine(2), .sine(1)], output: 0)
        }
        #expect(throws: MaterialGraph.Failure.outputMustBeColor(0)) {
            try MaterialGraph(nodes: [.parameter(.x)], output: 0)
        }
    }

    @Test("Graph validation checks input types, finite literals, and the node limit")
    func valueValidation() {
        #expect(throws: MaterialGraph.Failure.typeMismatch(node: 2, input: 1, expected: .color, actual: .scalar)) {
            try MaterialGraph(nodes: [.color(.zero), .scalar(0.5), .add(0, 1)], output: 2)
        }
        #expect(throws: MaterialGraph.Failure.typeMismatch(node: 1, input: 0, expected: .scalar, actual: .color)) {
            try MaterialGraph(nodes: [.color(.zero), .mix(0, 0, factor: 0)], output: 1)
        }
        #expect(throws: MaterialGraph.Failure.nonfiniteConstant(1)) {
            try MaterialGraph(nodes: [.color(.zero), .scalar(.infinity)], output: 0)
        }
        let limit = MaterialGraph.maximumNodeCount
        #expect(throws: MaterialGraph.Failure.tooManyNodes(limit: limit)) {
            try MaterialGraph(nodes: Array(repeating: .color(.zero), count: limit + 1), output: 0)
        }
    }

    @Test("Compiled material uses mesh inputs, time, and premultiplied opacity",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func gpuRendering() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let graph = try MaterialGraph(nodes: [
            .mix(1, 2, factor: 13),
            .rgb(red: 7, green: 8, blue: 9),
            .color(SIMD3(0.8, 0.2, 0.4)),
            .parameter(.x),
            .sine(3),
            .scalar(0.5),
            .add(5, 10),
            .uv(.u),
            .normal(.z),
            .multiply(11, 12),
            .multiply(4, 5),
            .worldPosition(.z),
            .scalar(-0.2),
            .clamp01(6),
        ], output: 0)
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba8Unorm, geometry: .cube(),
            material: graph.compile(device: device)
        )

        // A constant graph still supplies the per-draw alpha required by the
        // custom material's reflected buffer contract.
        let constant = try MaterialGraph(nodes: [.color(.zero)], output: 0)
        _ = try MeshRenderer(
            device: device, colorPixelFormat: .rgba8Unorm, geometry: .cube(),
            material: constant.compile(device: device)
        )

        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let queue = try #require(device.makeCommandQueue())
        let size = 65
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

        func centerPixel(time: Float) throws -> [UInt8] {
            let command = try #require(queue.makeCommandBuffer())
            let scene = MeshScene(
                camera: camera, transform: SceneTransform(translation: SIMD3(0, 0, -3)),
                materialParameters: SIMD4(time, 0, 0, 0.5)
            )
            try renderer.encode(scene, pass: pass, commandBuffer: command)
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed)
            var pixel = Array(repeating: UInt8.zero, count: 4)
            try pixel.withUnsafeMutableBytes { bytes in
                color.getBytes(
                    try #require(bytes.baseAddress), bytesPerRow: 4,
                    from: MTLRegionMake2D(size / 2, size / 2, 1, 1), mipmapLevel: 0
                )
            }
            return pixel
        }

        // At the center of the +Z face, U is 0.5, the normal's Z is 1,
        // and world Z is -2.5. The graph premultiplies both colors by 0.5.
        let meshInputs = try centerPixel(time: -.pi / 2)
        let solidColor = try centerPixel(time: .pi / 2)
        for (actual, expected) in zip(meshInputs, [64, 128, 64, 128]) {
            #expect(abs(Int(actual) - expected) <= 2)
        }
        for (actual, expected) in zip(solidColor, [102, 26, 51, 128]) {
            #expect(abs(Int(actual) - expected) <= 2)
        }
    }
}
