import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct FogDepthCaptureTests {
    @Test("A blended surface fogs at its own depth instead of the wall behind it")
    func blendedSurfaceKeepsItsDepth() throws {
        let scene = try GeometryRenderFixture(side: 32)
        let wall = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -6)),
            mesh: try MeshRenderer(
                device: scene.device, colorPixelFormat: .rgba16Float,
                geometry: .plane(width: 4, height: 4),
                material: .classic(try ClassicMaterial(model: .basic, color: SIMD3(0, 1, 0)))
            )
        )
        let glass = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)),
            mesh: try MeshRenderer(
                device: scene.device, colorPixelFormat: .rgba16Float,
                geometry: .plane(width: 0.8, height: 0.8),
                material: .classic(try ClassicMaterial(model: .basic, color: SIMD3(1, 0, 0))),
                renderState: MeshRenderState(blending: .normal, transparent: true, depthWrite: false)
            )
        )
        let graph = SceneGraph(roots: [wall, glass], camera: scene.camera)
        func texture(_ format: MTLPixelFormat, usage: MTLTextureUsage, storage: MTLStorageMode) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: scene.side, height: scene.side, mipmapped: false
            )
            descriptor.usage = usage
            descriptor.storageMode = storage
            return try #require(scene.device.makeTexture(descriptor: descriptor))
        }
        let color = try texture(.rgba16Float, usage: [.renderTarget, .shaderRead], storage: .shared)
        let sceneDepth = try texture(.depth32Float, usage: [.renderTarget, .shaderRead], storage: .private)
        let fogDepth = try texture(.depth32Float, usage: [.renderTarget, .shaderRead], storage: .private)
        let behind = try texture(.rgba16Float, usage: [.renderTarget], storage: .shared)
        let front = try texture(.rgba16Float, usage: [.renderTarget], storage: .shared)
        let fog = try PostProcessGraph.depthFog(
            color: SIMD3(0, 0, 1), near: 0, far: 3, camera: scene.camera
        )
        let pass = try PostProcessPass(graph: fog, device: scene.device)
        let capture = try FogDepthCapture(device: scene.device)
        let buffer = try #require(scene.queue.makeCommandBuffer())
        let colorPass = scene.pass(color: color, depth: sceneDepth)
        colorPass.depthAttachment.storeAction = MTLStoreAction.store
        _ = try graph.encode(pass: colorPass, commandBuffer: buffer)
        try pass.encode(source: color, depth: sceneDepth, destination: behind, commandBuffer: buffer)
        let draws = try capture.encode(graph: graph, into: fogDepth, commandBuffer: buffer)
        try pass.encode(source: color, depth: fogDepth, destination: front, commandBuffer: buffer)
        try scene.finish(buffer)
        let hidden = try scene.pixels(of: behind)
        let shown = try scene.pixels(of: front)
        let far = hidden[16 * scene.side + 16]
        let near = shown[16 * scene.side + 16]
        #expect(draws >= 2)
        #expect(far.z > 0.9 && far.x < 0.1, "\(far)")
        #expect(near.x > 0.2 && near.x > far.x + 0.15, "\(near)")
    }
}
