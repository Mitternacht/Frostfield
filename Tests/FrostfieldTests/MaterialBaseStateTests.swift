import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MaterialBaseStateTests {
    private let scene: GeometryRenderFixture

    init() throws {
        scene = try GeometryRenderFixture(side: 32)
    }

    private func quad(x: Float, material: MeshMaterial) throws -> SceneNode {
        let mesh = try MeshRenderer(
            device: scene.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.5, height: 0.5),
            material: material
        )
        return SceneNode(transform: .init(translation: SIMD3(x, 0, -3)), mesh: mesh)
    }

    @Test("A hidden material is not drawn and can still be picked")
    func hiddenMaterialStaysPickable() throws {
        let hidden = try quad(
            x: 0,
            material: .physicallyBased(try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(1, 0, 0, 1),
                alphaMode: .opaque, shadingModel: .unlit, isVisible: false
            ))
        )
        let classic = try quad(
            x: 0.8,
            material: .classic(try ClassicMaterial(model: .basic, color: SIMD3(0, 1, 0), isVisible: false))
        )
        let graph = SceneGraph(roots: [hidden, classic], camera: scene.camera)
        let image = try scene.render(graph)
        #expect(image.statistics.drawnMeshCount == 0)
        #expect(image.pixels.allSatisfy { $0 == .zero })
        #expect(try graph.pick(at: SIMD2(0.5, 0.5), aspectRatio: 1)?.nodeID == hidden.id)
        let archive = try SceneArchive(capturing: graph, commandQueue: scene.queue)
        let loaded = try archive.load(device: scene.device, commandQueue: scene.queue).graph
        #expect(try scene.render(loaded).statistics.drawnMeshCount == 0)
        #expect(try loaded.pick(at: SIMD2(0.5, 0.5), aspectRatio: 1) != nil)
    }

    @Test("A material can stay unfogged while the rest of the image fogs")
    func materialFog() throws {
        let graph = SceneGraph(
            roots: [
                try quad(x: -0.5, material: .physicallyBased(try PhysicallyBasedMaterial(
                    lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(1, 0, 0, 1),
                    alphaMode: .opaque, shadingModel: .unlit, fog: true
                ))),
                try quad(x: 0.5, material: .classic(try ClassicMaterial(
                    model: .basic, color: SIMD3(1, 0, 0), fog: false
                ))),
            ],
            camera: scene.camera
        )
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: scene.side, height: scene.side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: scene.side, height: scene.side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        let maskDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm, width: scene.side, height: scene.side, mipmapped: false
        )
        maskDescriptor.storageMode = .private
        maskDescriptor.usage = [.renderTarget, .shaderRead]
        let color = try #require(scene.device.makeTexture(descriptor: colorDescriptor))
        let depth = try #require(scene.device.makeTexture(descriptor: depthDescriptor))
        let mask = try #require(scene.device.makeTexture(descriptor: maskDescriptor))
        let displayed = try #require(scene.device.makeTexture(descriptor: colorDescriptor))
        let fog = try PostProcessGraph.depthFog(
            color: SIMD3(0, 0, 1), near: 0, far: 1, camera: scene.camera, respectsMaterialFog: true
        )
        let pass = try PostProcessPass(graph: fog, device: scene.device, textures: ["FogExclusion": mask])
        let buffer = try #require(scene.queue.makeCommandBuffer())
        let colorPass = scene.pass(color: color, depth: depth)
        colorPass.depthAttachment.storeAction = .store
        _ = try graph.encode(pass: colorPass, commandBuffer: buffer)
        try graph.encodeFogExclusionMask(depth: depth, into: mask, commandBuffer: buffer)
        try pass.encode(source: color, depth: depth, destination: displayed, commandBuffer: buffer)
        try scene.finish(buffer)
        let pixels = try scene.pixels(of: displayed)
        let fogged = pixels[16 * scene.side + 13]
        let kept = pixels[16 * scene.side + 19]
        #expect(fogged.z > 0.8 && fogged.x < 0.2, "\(fogged)")
        #expect(kept.x > 0.8 && kept.z < 0.2, "\(kept)")
    }
}
