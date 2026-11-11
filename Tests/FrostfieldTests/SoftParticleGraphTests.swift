import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SoftParticleGraphTests {
    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture(side: 128) }

    @Test("A graph's live opacity and contrast fade a mesh billboard against opaque viewport depth",
          arguments: [MeshDepthMode.forward, .reversed, .logarithmic], [false, true])
    func perspectiveFade(mode: MeshDepthMode, rebased: Bool) throws {
        let vertices = [SIMD2<Float>(-1.25, -0.25), SIMD2(1.25, -0.25), SIMD2(1.25, 0.25), SIMD2(-1.25, 0.25)].map {
            MeshVertex(position: SIMD3($0.x, $0.y, -2.5 - $0.x), normal: SIMD3(0, 0, 1))
        }
        let slope = try gpu.unlit(MeshGeometry(vertices: vertices, indices: [0, 1, 2, 0, 2, 3]), depthMode: mode)
        let graph = try SurfaceShaderGraph { b in
            .init(opacity: b.softParticleOpacity(
                opacity: b.uniform("Opacity", Float(0.8)), distance: b.uniform("Distance", Float(1)),
                contrast: b.uniform("Contrast", Float(2))
            ))
        }
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(1, 0.5, 0.25, 1), alphaMode: .blend, surfaceGraph: graph, shadingModel: .unlit
        )
        let particle = try gpu.unlit(.plane(width: 2.4, height: 2.4), material: .physicallyBased(material), depthMode: mode)
        var scene = try capture([
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: particle),
            SceneNode(mesh: slope, materialParameters: SIMD4(0, 0, 0, 1)),
        ])
        if rebased {
            let origin = SceneTransform(translation: SIMD3(1e9, -1e9, 1e9))
            scene.roots = [SceneNode(transform: origin, children: scene.roots)]
            scene.cameraTransform = origin
            scene.highPrecision = true
        }

        let eased = try gpu.render(scene, depthMode: mode).pixels
        let uniforms = try #require(material.surfaceUniforms)
        try uniforms.set("Contrast", to: Float(1))
        try uniforms.set("Distance", to: Float(0.5))
        try uniforms.set("Opacity", to: Float(0.6))
        let linear = try gpu.render(scene, depthMode: mode).pixels

        let row = gpu.side / 2
        #expect(gpu.pixel(eased, x: 42, y: row) == SIMD4(0, 0, 0, 1))
        for x in [49, 52, 59, 64, 71, 76] {
            let u = (Float(x) + 0.5) / Float(gpu.side) * 2 - 1
            let viewGap = 2.5 / (1 - u) - 2
            let gap = min(max(viewGap, 0), 1)
            let folded = min(gap, 1 - gap) * 2
            let curved = gap > 0.5 ? 1 - 0.5 * folded * folded : 0.5 * folded * folded
            let pixel = gpu.pixel(eased, x: x, y: row)
            #expect(abs(pixel.x - 0.8 * curved) < 0.002)
            #expect(abs(pixel.y - 0.4 * curved) < 0.002)
            #expect(abs(gpu.pixel(linear, x: x, y: row).x - 0.6 * min(max(2 * viewGap, 0), 1)) < 0.002)
        }
        #expect(simd_distance(gpu.pixel(eased, x: row, y: 46), SIMD4(0.8, 0.4, 0.2, 0.8)) < 0.002)
        #expect(simd_distance(gpu.pixel(linear, x: row, y: 46), SIMD4(0.6, 0.3, 0.15, 0.6)) < 0.002)
    }

    @Test("A cropped orthographic viewport uses attachment coordinates and its authored window depth",
          arguments: [MeshDepthMode.forward, .reversed])
    func croppedOrthographic(mode: MeshDepthMode) throws {
        let projection = simd_float4x4(columns: (
            SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0),
            SIMD4(0, 0, mode == .reversed ? 0.1 : -0.1, 0),
            SIMD4(0, 0, mode == .reversed ? 1 : 0, 1)
        ))
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: 0, far: 10
        )
        let viewport = MTLViewport(originX: 16, originY: 32, width: 64, height: 64, znear: 0.2, zfar: 0.8)
        let producer = try gpu.unlit(.plane(width: 1, height: 1), depthMode: mode)
        let shader = try SurfaceShaderGraph { b in .init(opacity: b.softParticleOpacity(opacity: b.constant(0.6))) }
        let material = try PhysicallyBasedMaterial(alphaMode: .blend, surfaceGraph: shader, shadingModel: .unlit)
        let billboard = try gpu.unlit(.plane(width: 4, height: 4), material: .physicallyBased(material), depthMode: mode)
        let scene = try capture([
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2.25)), mesh: producer, materialParameters: SIMD4(0, 0, 0, 1)),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: billboard),
        ])
        let target = try gpu.makeTarget()
        let pass = gpu.pass(color: target.color, depth: target.depth, depthMode: mode)
        pass.depthAttachment.clearDepth = mode == .reversed ? viewport.znear : viewport.zfar
        let command = try #require(gpu.queue.makeCommandBuffer())
        try scene.encode(
            pass: pass, commandBuffer: command, clipFromWorld: projection, cameraPosition: .zero,
            viewport: viewport, depthMode: mode, graphCamera: camera
        )
        try gpu.finish(command)
        let pixels = try gpu.pixels(of: target.color)
        #expect(abs(gpu.pixel(pixels, x: 48, y: 64).x - 0.075) < 0.002)
        #expect(abs(gpu.pixel(pixels, x: 30, y: 64).x - 0.6) < 0.002)
        #expect(gpu.pixel(pixels, x: 12, y: 64) == .zero)
    }

    @Test("A soft-particle graph with an explicit depth expression archives and rejects path tracing")
    func graphContract() throws {
        let clip = try gpu.camera.projection(aspectRatio: 1) * SIMD4(0, 0, -2.25, 1)
        let shader = try SurfaceShaderGraph { b in
            .init(opacity: b.softParticleOpacity(viewportDepth: b.constant(clip.z / clip.w)))
        }
        let material = try PhysicallyBasedMaterial(alphaMode: .blend, surfaceGraph: shader, shadingModel: .unlit)
        let mesh = try gpu.unlit(.plane(width: 2, height: 2), material: .physicallyBased(material))
        var scene = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        ], camera: gpu.camera)
        scene.environmentEulerRotation = SIMD3(0.1, -0.2, 0.3)
        let archive = try SceneArchive(capturing: scene, commandQueue: gpu.queue)
        let metadata = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(metadata["version"] as? Int == 19)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        #expect(restored.environmentEulerRotation == scene.environmentEulerRotation)
        #expect(abs(gpu.pixel(try gpu.render(restored).pixels, x: 64, y: 64).x - 0.125) < 0.002)
        var projection = try gpu.camera.projection(aspectRatio: 1)
        projection[0].z = 0.1
        let oblique = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: gpu.camera.near, far: gpu.camera.far
        )
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: SoftParticleDepth.Failure.projection) {
            try restored.encode(
                pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
                clipFromWorld: projection, cameraPosition: .zero, graphCamera: oblique
            )
        }
        let error = #expect(throws: PathTracingScene.Failure.self) {
            try PathTracingScene(queue: gpu.queue, graph: restored)
        }
        if case .unsupported(_, let feature) = error {
            #expect(feature.contains("soft-particle fade"))
        } else {
            Issue.record("Unexpected path tracing result \(String(describing: error))")
        }
    }

    private func capture(_ roots: [SceneNode]) throws -> SceneGraph {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let depth = try #require(gpu.device.makeTexture(descriptor: descriptor))
        var scene = SceneGraph(roots: roots, camera: gpu.camera)
        scene.viewportSnapshot = try SceneViewportSnapshot(depth: depth, source: .opaquePass)
        return scene
    }
}
