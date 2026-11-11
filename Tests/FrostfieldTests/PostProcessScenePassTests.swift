import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessScenePassTests {
    @Test("A scaled capture applies its layers and materials, then feeds matching depth to the chain",
          arguments: [MeshDepthMode.forward, .reversed])
    func scopedCapture(depthMode: MeshDepthMode) throws {
        let gpu = try PostProcessFixture()
        let layer = SceneLayers(layer: 2)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.75, height: 0.75),
            depthMode: depthMode
        )
        let red = SIMD4<Float>(1, 0, 0, 1)
        let blue = SIMD4<Float>(0, 0, 1, 1)
        let green = SIMD4<Float>(0, 1, 0, 1)
        let points = try PointsRenderer(
            device: gpu.device, cloud: PointCloud(positions: [SIMD3(0.8, 0, -2)]),
            material: PointsMaterial(size: .pixels(8), color: red)
        )
        var pointNode = SceneNode(primitive: .points(points))
        pointNode.layers = layer
        let scene = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh, materialParameters: red),
            SceneNode(transform: .init(translation: SIMD3(-0.8, 0, -2)), mesh: mesh, materialParameters: red, layers: layer),
            pointNode,
        ], camera: camera)

        let pass = PostProcessScenePass(queue: gpu.queue)
        try pass.setResolutionScale(0.5)
        pass.layers = layer
        pass.overrideMaterial = try SceneMaterialOverride(material: .unlit, parameters: blue)
        pass.primitiveMaterialOverride = try ScenePrimitiveMaterialOverride(
            points: PointsMaterial(size: .pixels(8), color: green)
        )
        let sampleDepth = try PostProcessGraph { graph in
            let color = graph.sceneColor(at: graph.screenUV)
            return graph.vector(graph.sceneDepth(at: graph.screenUV), color.g, color.b, color.a)
        }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(sampleDepth)])
        let plain = try PostProcessChain(queue: gpu.queue, effects: [])
        let destination = try gpu.color(width: 65, height: 41)
        let original = try gpu.color(width: 65, height: 41)
        let projection = try camera.projection(aspectRatio: 65.0 / 41.0, depthMode: depthMode)
        try gpu.run { command in
            let frame = try chain.encode(scene: scene, using: pass, destination: destination, commandBuffer: command, depthMode: depthMode)
            #expect(frame.color.width == 33 && frame.depth.height == 21)
            #expect(frame.color.pixelFormat == .rgba16Float && frame.depth.pixelFormat == .depth32Float)
            #expect(frame.statistics.submittedTriangleCount == 2)
            #expect(frame.camera.projectionMatrix == projection)
            #expect(frame.depthMode == depthMode)

            pass.layers = nil
            pass.overrideMaterial = nil
            pass.primitiveMaterialOverride = nil
            let unmodified = try plain.encode(scene: scene, using: pass, destination: original, commandBuffer: command, depthMode: depthMode)
            #expect(unmodified.color === frame.color && unmodified.depth === frame.depth)
        }
        let result = try gpu.read(destination)
        let originalPixels = try gpu.read(original)
        let left = result[20 * 65 + 17]
        let middle = result[20 * 65 + 32]
        let right = result[20 * 65 + 47]
        let clip = projection * SIMD4(0, 0, -2, 1)
        #expect(PostProcessFixture.close(left, SIMD4(clip.z / clip.w, 0, 1, 1)))
        #expect(middle == SIMD4(depthMode == .forward ? 1 : 0, 0, 0, 0))
        #expect(right.y > 0.99 && right.z == 0 && right.w > 0.99)
        #expect(originalPixels[20 * 65 + 17] == .zero && originalPixels[20 * 65 + 47] == .zero)
        #expect(PostProcessFixture.close(originalPixels[20 * 65 + 32], SIMD4(0.22, 0, 0, 1)))
        #expect(scene.cameraLayers == .default && scene.overrideMaterial == nil && scene.primitiveMaterialOverride == nil)
    }

    @Test("Standalone captures preserve the custom projection and reject invalid settings before allocating")
    func customCapture() throws {
        let gpu = try PostProcessFixture()
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        var graph = SceneGraph(roots: [], camera: camera)
        let view = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: matrix_identity_float4x4, near: 0, far: 1
        )
        let pass = PostProcessScenePass(queue: gpu.queue)
        #expect(throws: PostProcessScenePass.Failure.resolutionScale) { try pass.setResolutionScale(.nan) }
        #expect(pass.resolutionScale == 1)
        try pass.setResolutionScale(0.125)
        let display = try gpu.color(width: 5, height: 3, pixelFormat: .bgra8Unorm_srgb)
        let chain = try PostProcessChain(queue: gpu.queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [])
        let otherQueue = try #require(gpu.device.makeCommandQueue())
        let wrongCommand = try #require(otherQueue.makeCommandBuffer())
        #expect(throws: PostProcessScenePass.Failure.commandBuffer) {
            try pass.encode(graph, width: 5, height: 3, commandBuffer: wrongCommand)
        }
        try gpu.run { command in
            #expect(throws: PostProcessScenePass.Failure.size) {
                try pass.encode(graph, width: 0, height: 1, commandBuffer: command)
            }
            let frame = try pass.encode(graph, width: 5, height: 3, commandBuffer: command, camera: view)
            #expect(frame.color.width == 1 && frame.depth.height == 1 && frame.camera == view)
            graph.projectionJitter = SIMD2(0.25, -0.125)
            let clear = MTLClearColor(red: 0.25, green: 0, blue: 1, alpha: 1)
            let jittered = try chain.encode(
                scene: graph, using: pass, destination: display, commandBuffer: command, camera: view, clearColor: clear
            )
            #expect(jittered.camera.projectionMatrix.columns.3 == SIMD4(0.5, 0.25, 0, 1))
        }
        var bytes = SIMD4<UInt8>.zero
        withUnsafeMutableBytes(of: &bytes) { buffer in
            display.getBytes(buffer.baseAddress!, bytesPerRow: 4, from: MTLRegionMake2D(2, 1, 1, 1), mipmapLevel: 0)
        }
        #expect(bytes.x == 255 && bytes.y == 0 && (136...138).contains(bytes.z) && bytes.w == 255)
    }
}
