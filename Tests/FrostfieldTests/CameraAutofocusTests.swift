import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CameraAutofocusTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Autofocus reads submitted raster depth and changes only view-axis focus",
          arguments: [MeshDepthMode.forward, .reversed])
    @MainActor
    func renderedDepth(mode: MeshDepthMode) async throws {
        let width = 33, height = 17
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 50)
        let near = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float,
            geometry: .plane(width: 1.1, height: 1.1), material: .unlit, depthMode: mode
        )
        let far = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float,
            geometry: .plane(width: 1.7, height: 1.7), material: .unlit, depthMode: mode
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(-1, 0.5, -2)), mesh: near),
            SceneNode(transform: .init(translation: SIMD3(2, -0.65, -5)), mesh: far)
        ], camera: camera)
        let depth = try gpu.depthTarget(width: width, height: height)
        let color = try gpu.color(width: width, height: height)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = mode == .reversed ? 0 : 1
        let projection = try camera.projection(aspectRatio: Float(width) / Float(height), depthMode: mode)
        let command = try #require(gpu.queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: command, clipFromWorld: projection, cameraPosition: .zero, depthMode: mode)
        command.commit()

        let lens = try CameraLens(focalLength: 70, fNumber: 1.8, focusDistance: 12, sensorWidth: 32, metersPerUnit: 0.5)
        for (x, y, distance) in [(12, 6, Float(2)), (20, 9, Float(5))] {
            let uv = (SIMD2(Float(x), Float(y)) + 0.5) / SIMD2(Float(width), Float(height))
            let focused = try #require(await lens.autofocused(from: depth, at: uv, camera: camera, depthMode: mode, commandQueue: gpu.queue))
            #expect(abs(focused.focusDistance - distance) < 5e-5)
            #expect(abs(focused.circleOfConfusionRadius(atDistance: distance, imageWidth: width)) < 5e-5)
            var expected = focused
            expected.focusDistance = lens.focusDistance
            #expect(expected == lens)
        }
        #expect(try await lens.autofocused(from: depth, at: SIMD2(1, 0), camera: camera, depthMode: mode, commandQueue: gpu.queue) == nil)
        #expect(command.status == .completed)
    }

    @Test("Autofocus rejects invalid sampling and keeps thin-lens validation")
    func invalidInputs() async throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100)
        let lens = try CameraLens(focalLength: 400, focusDistance: 4)
        let nearDepth = try gpu.depth(width: 3, height: 3) { _, _ in 0 }
        await #expect(throws: CameraLens.AutofocusFailure.coordinates) {
            try await lens.autofocused(from: nearDepth, at: SIMD2(.nan, 0), camera: camera, commandQueue: gpu.queue)
        }
        let color = try gpu.color(width: 3, height: 3)
        await #expect(throws: CameraLens.AutofocusFailure.depthTexture) {
            try await lens.autofocused(from: color, camera: camera, commandQueue: gpu.queue)
        }
        await #expect(throws: CameraLens.Failure.focusInsideFocalLength) {
            try await lens.autofocused(from: nearDepth, camera: camera, commandQueue: gpu.queue)
        }
        await #expect(throws: MeshRenderer.Failure.logarithmicDepthConsumer) {
            try await lens.autofocused(from: nearDepth, camera: camera, depthMode: .logarithmic, commandQueue: gpu.queue)
        }
    }
}
