import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SurfaceViewportSnapshotTests {
    private let gpu: PostProcessFixture
    private let camera: PerspectiveCamera

    init() throws {
        gpu = try PostProcessFixture()
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.5, far: 8)
    }

    @Test("Raster surface graphs sample the prior scene's color and depth on the same command buffer",
          arguments: [MeshDepthMode.forward, .reversed])
    func storedScene(depthMode: MeshDepthMode) throws {
        let source = try plane(base: SIMD4(1, 0, 0, 1), graph: nil, depthMode: depthMode)
        let producer = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(-2, 0, -2)), mesh: source)
        ], camera: camera)
        let shader = try SurfaceShaderGraph { b in
            let color = b.sceneColor(at: b.screenUV)
            return .init(baseColor: b.vector(color.r, b.sceneDepth(at: b.screenUV), color.b))
        }
        let consumer = try plane(graph: shader, depthMode: depthMode)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -1)), mesh: consumer)
        var scene = SceneGraph(roots: [node], camera: camera)
        let capture = PostProcessScenePass(queue: gpu.queue)
        try capture.setResolutionScale(0.5)
        let destination = try gpu.color(width: 32, height: 32)
        let pass = try renderPass(destination, depthMode: depthMode)

        try gpu.run { command in
            let frame = try capture.encode(
                producer, width: 32, height: 32, commandBuffer: command, depthMode: depthMode,
                clearColor: MTLClearColor(red: 0, green: 0, blue: 0.25, alpha: 1)
            )
            scene.viewportSnapshot = try SceneViewportSnapshot(color: frame.color, depth: frame.depth)
            try scene.encode(pass: pass, commandBuffer: command, depthMode: depthMode)
        }
        let clip = try camera.projection(aspectRatio: 1, depthMode: depthMode) * SIMD4(0, 0, -2, 1)
        let pixels = try gpu.read(destination)
        #expect(PostProcessFixture.close(pixels[16 * 32 + 4], SIMD4(1, clip.z / clip.w, 0, 1)))
        #expect(pixels[16 * 32 + 27] == SIMD4(0, depthMode == .reversed ? 0 : 1, 0.25, 1))
    }

    @Test("Opaque viewport copies precede overlay shading and generate trilinear color mips",
          arguments: [MeshDepthMode.forward, .reversed], [false, true])
    func opaqueCapture(depthMode: MeshDepthMode, sharesTransmission: Bool) throws {
        let producer = try plane(base: SIMD4(1, 0, 0, 1), graph: nil, depthMode: depthMode)
        let shader = try SurfaceShaderGraph { b in
            let blurred = b.sceneColor(at: b.screenUV, mipLevel: b.constant(99))
            return .init(baseColor: b.vector(blurred.r, b.sceneDepth(at: b.screenUV), blurred.b))
        }
        let consumer = try plane(graph: shader, depthMode: depthMode, alphaMode: .blend)
        var nodes = [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -1)), mesh: consumer),
            SceneNode(transform: .init(translation: SIMD3(-2, 0, -2)), mesh: producer),
        ]
        if sharesTransmission {
            let glass = try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.5, height: 0.5),
                material: .physicallyBased(.init(transmission: .init(factor: 1))), depthMode: depthMode
            )
            nodes.append(SceneNode(transform: .init(translation: SIMD3(0, 0, -1.2)), mesh: glass))
        }
        var scene = SceneGraph(roots: nodes, camera: camera)
        let target = try gpu.color(width: 32, height: 32)
        let pass = try renderPass(target, depthMode: depthMode)
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let color = try captureTarget(format: .rgba16Float, side: 32, mipmapped: true)
        let depth = try captureTarget(format: .depth32Float, side: 32)
        scene.viewportSnapshot = try SceneViewportSnapshot(color: color, depth: depth, source: .opaquePass)
        let filtered = try gpu.color(width: 32, height: 32)
        let fractional = try PostProcessGraph { b in
            b.sceneColor(at: b.constant(SIMD2<Float>(0.25, 0.25)), mipLevel: b.constant(4.5))
        }
        let post = try PostProcessPass(graph: fractional, device: gpu.device)

        try gpu.run { command in
            try scene.encode(
                pass: pass, commandBuffer: command, transmissionBackdrop: sharesTransmission ? color : nil,
                depthMode: depthMode
            )
            try post.encode(source: color, destination: filtered, commandBuffer: command)
        }
        let clip = try camera.projection(aspectRatio: 1, depthMode: depthMode) * SIMD4(0, 0, -2, 1)
        let pixels = try gpu.read(target)
        #expect(PostProcessFixture.close(pixels[16 * 32 + 4], SIMD4(0.5, clip.z / clip.w, 0.5, 1)))
        #expect(pixels[16 * 32 + 27] == SIMD4(0.5, depthMode == .reversed ? 0 : 1, 0.5, 1))
        #expect(try gpu.read(filtered)[16 * 32 + 4] == SIMD4(0.75, 0, 0.25, 1))

        if !sharesTransmission, depthMode == .forward {
            let archive = try SceneArchive(capturing: scene, commandQueue: gpu.queue)
            var manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
            #expect(manifest["version"] as? Int == 17)
            let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
            #expect(restored.viewportSnapshot == nil)
            guard case .physicallyBased(let restoredMaterial) = restored.roots[0].mesh?.material else {
                Issue.record("Expected the archived viewport material")
                return
            }
            #expect(restoredMaterial.surfaceGraph == shader)
            manifest["version"] = 16
            #expect(throws: SceneArchive.Failure.decoding("surface viewport mip levels require archive version 17")) {
                try SceneArchive(manifest: JSONSerialization.data(withJSONObject: manifest), binary: archive.binary)
            }
        }
    }

    @Test("Automatic capture rejects producer feedback and mismatched capture dimensions before clearing")
    func opaqueCaptureValidation() throws {
        let shader = try SurfaceShaderGraph { b in .init(baseColor: b.sceneColor(at: b.screenUV).rgb) }
        let transform = SceneTransform(translation: SIMD3(0, 0, -1))
        let mesh = try plane(graph: shader)
        var graph = SceneGraph(roots: [SceneNode(transform: transform, mesh: mesh)], camera: camera)
        let sentinel = SIMD4<Float>(0.125, 0.25, 0.5, 1)
        let target = try gpu.color(width: 16, height: 16) { _, _ in sentinel }
        let pass = try renderPass(target)
        graph.viewportSnapshot = try SceneViewportSnapshot(
            color: captureTarget(format: .rgba16Float, side: 16), source: .opaquePass
        )
        try gpu.run { command in
            #expect(throws: SceneViewportSnapshot.Failure.captureReader) {
                try graph.encode(pass: pass, commandBuffer: command)
            }
            var standalone = MeshScene(camera: camera, transform: transform)
            standalone.viewportSnapshot = graph.viewportSnapshot
            #expect(throws: SceneViewportSnapshot.Failure.capturePass) {
                try mesh.encode(standalone, pass: pass, commandBuffer: command)
            }
            graph.roots = []
            graph.viewportSnapshot = try SceneViewportSnapshot(
                depth: captureTarget(format: .depth32Float, side: 8), source: .opaquePass
            )
            #expect(throws: SceneViewportSnapshot.Failure.size) {
                try graph.encode(pass: pass, commandBuffer: command)
            }
            graph.viewportSnapshot = try SceneViewportSnapshot(
                color: captureTarget(format: .rgba16Float, side: 16), source: .opaquePass
            )
            pass.renderTargetWidth = 8
            #expect(throws: SceneViewportSnapshot.Failure.capturePass) {
                try graph.encode(pass: pass, commandBuffer: command)
            }
        }
        #expect(try gpu.read(target).allSatisfy { $0 == sentinel })
    }

    @Test("A standalone material samples color bilinearly and depth by the nearest texel")
    func filters() throws {
        let color = try gpu.color(width: 2, height: 2) { x, y in
            SIMD4(Float(x), 0, Float(y), 1)
        }
        let depth = try gpu.depth(width: 2, height: 2) { x, y in Float(2 * y + x + 1) / 4 }
        let shader = try SurfaceShaderGraph { b in
            let uv = b.constant(SIMD2<Float>(0.625, 0.375))
            let sample = b.sceneColor(at: uv)
            return .init(baseColor: b.vector(sample.r, b.sceneDepth(at: uv), sample.b))
        }
        let mesh = try plane(graph: shader)
        let destination = try gpu.color(width: 16, height: 16)
        let pass = try renderPass(destination)
        var scene = MeshScene(camera: camera, transform: .init(translation: SIMD3(0, 0, -1)))
        scene.viewportSnapshot = try SceneViewportSnapshot(color: color, depth: depth)
        try gpu.run { try mesh.encode(scene, pass: pass, commandBuffer: $0) }
        #expect(try gpu.read(destination)[8 * 16 + 8] == SIMD4(0.75, 0.5, 0.25, 1))
    }

    @Test("Missing inputs and feedback are rejected before a scene changes the destination")
    func validation() throws {
        let shader = try SurfaceShaderGraph { b in
            .init(baseColor: b.sceneColor(at: b.uv0).rgb, roughness: b.sceneDepth(at: b.uv0))
        }
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -1)), mesh: try plane(graph: shader))
        var scene = SceneGraph(roots: [node], camera: camera)
        let sentinel = SIMD4<Float>(0.25, 0.5, 0.75, 1)
        let target = try gpu.color(width: 8, height: 8) { _, _ in sentinel }
        let color = try gpu.color(width: 8, height: 8)
        let pass = try renderPass(target)
        try gpu.run { command in
            #expect(throws: SceneViewportSnapshot.Failure.missingColor) {
                try scene.encode(pass: pass, commandBuffer: command)
            }
            scene.viewportSnapshot = try SceneViewportSnapshot(color: color)
            #expect(throws: SceneViewportSnapshot.Failure.missingDepth) {
                try scene.encode(pass: pass, commandBuffer: command)
            }
            scene.viewportSnapshot = try SceneViewportSnapshot(color: target)
            #expect(throws: SceneViewportSnapshot.Failure.feedback) {
                try scene.encode(pass: pass, commandBuffer: command)
            }
        }
        #expect(try gpu.read(target).allSatisfy { $0 == sentinel })
        let rejection = #expect(throws: PathTracingScene.Failure.self) {
            try PathTracingScene(queue: gpu.queue, graph: scene)
        }
        if case .unsupported(_, let feature) = rejection {
            #expect(feature.contains("stored viewport texture"))
        } else {
            Issue.record("Unexpected path tracing result \(String(describing: rejection))")
        }
    }

    @Test("Viewport-independent coverage still writes shadows and viewport-dependent coverage reports an error")
    func shadowCoverage() throws {
        let source = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(0.5, 0, 0, 1) }
        let destination = try gpu.color(width: 16, height: 16)
        let pass = try renderPass(destination)
        let shadow = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 16, directionToLight: SIMD3(0, 0, 1),
            center: SIMD3(0, 0, -2), halfExtent: 3
        )
        func scene(usingViewportForCoverage: Bool) throws -> SceneGraph {
            let shader = try SurfaceShaderGraph { b in
                let color = b.sceneColor(at: b.uv0)
                return .init(baseColor: color.rgb, mask: usingViewportForCoverage ? color.a : b.uv0.x.lessThan(b.constant(0.75)))
            }
            let node = SceneNode(
                transform: .init(translation: SIMD3(0, 0, -2)),
                mesh: try plane(graph: shader, receivesShadows: true)
            )
            var graph = SceneGraph(roots: [node], camera: camera)
            graph.viewportSnapshot = try SceneViewportSnapshot(color: source)
            return graph
        }
        let independent = try scene(usingViewportForCoverage: false)
        let dependent = try scene(usingViewportForCoverage: true)
        try gpu.run { command in
            let stats = try independent.encode(pass: pass, commandBuffer: command, shadowMap: shadow)
            #expect(stats.shadowDrawCount == 1)
        }
        let command = try #require(gpu.queue.makeCommandBuffer())
        let error = #expect(throws: MeshRenderer.Failure.self) {
            try dependent.encode(pass: pass, commandBuffer: command, shadowMap: shadow)
        }
        if case .graphShadowCameraInput(let node, _) = error {
            #expect(node == dependent.roots[0].id)
        } else {
            Issue.record("Unexpected shadow result \(String(describing: error))")
        }
    }

    private func plane(
        base: SIMD4<Float> = SIMD4(repeating: 1), graph: SurfaceShaderGraph?, depthMode: MeshDepthMode = .forward,
        receivesShadows: Bool = false, alphaMode: PhysicallyBasedMaterial.AlphaMode = .opaque
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(.init(
                receivesShadows: receivesShadows, baseColorFactor: base, alphaMode: alphaMode,
                surfaceGraph: graph, shadingModel: .unlit
            )),
            depthMode: depthMode
        )
    }

    private func captureTarget(format: MTLPixelFormat, side: Int, mipmapped: Bool = false) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: side, height: side, mipmapped: mipmapped
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func renderPass(_ target: any MTLTexture, depthMode: MeshDepthMode = .forward) throws -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        let depth = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: target.width, height: target.height, mipmapped: false
        )
        depth.storageMode = .private
        depth.usage = .renderTarget
        pass.depthAttachment.texture = try #require(gpu.device.makeTexture(descriptor: depth))
        pass.depthAttachment.clearDepth = depthMode == .reversed ? 0 : 1
        pass.depthAttachment.loadAction = .clear
        return pass
    }
}
