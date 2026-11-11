import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LogarithmicDepthTests {
    @Test("Log depth resolves far overlaps independently of draw order at a 10¹⁰ clip ratio",
          arguments: [false, true])
    func overlappingMeshes(nearFirst: Bool) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.01, far: 1e8)
        let renderer = try gpu.unlit(GeometryRenderFixture.square(1), depthMode: .logarithmic)
        func panel(distance: Float, x: Float = 0, color: SIMD4<Float>) -> SceneNode {
            SceneNode(
                transform: .init(translation: SIMD3(x * distance, 0, -distance), scale: SIMD3(distance * 0.2, distance * 0.2, 1)),
                mesh: renderer, materialParameters: color
            )
        }
        let near = panel(distance: 1e6, color: SIMD4(0, 0, 1, 1))
        let far = panel(distance: 1_001_000, color: SIMD4(1, 0, 0, 1))
        var beyond = panel(distance: 1.1e8, x: 0.7, color: SIMD4(0, 1, 0, 1))
        beyond.frustumCulled = nearFirst
        var graph = SceneGraph(roots: (nearFirst ? [near, far] : [far, near]) + [beyond], camera: camera)
        graph.highPrecision = true
        let target = try gpu.makeTarget()
        let pass = storedPass(gpu, target: target)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: pass, commandBuffer: command, depthMode: .logarithmic, ordering: .init(sortObjects: false)
        )
        try gpu.finish(command)
        #expect(statistics.culledMeshCount == (nearFirst ? 1 : 0))

        let pixels = try gpu.pixels(of: target.color)
        #expect(gpu.pixel(pixels, x: 32, y: 32) == SIMD4(0, 0, 1, 1))
        #expect(gpu.pixel(pixels, x: 55, y: 32) == .zero)
        let depths = try await readDepth(gpu, target.depth)
        #expect(abs(depths[32 * 65 + 32] - 0.8) < 0.000002)
        #expect(depths[32 * 65 + 55] == 1)
    }

    @Test("Standalone Metal writes forward, reversed, and r187 logarithmic depth",
          arguments: [MeshDepthMode.forward, .reversed, .logarithmic])
    func standalone(mode: MeshDepthMode) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 1000)
        let renderer = try gpu.unlit(GeometryRenderFixture.square(2), depthMode: mode)
        let cameraTransform = SceneTransform(
            translation: SIMD3(3, -2, 1), rotation: simd_quatf(angle: 0.23, axis: SIMD3(0, 1, 0))
        )
        let transform = SceneTransform(
            translation: cameraTransform.translation + cameraTransform.rotation.act(SIMD3(0, 0, -10)),
            rotation: cameraTransform.rotation
        )
        let target = try gpu.makeTarget()
        let pass = storedPass(gpu, target: target, depthMode: mode)
        let command = try #require(gpu.queue.makeCommandBuffer())
        try renderer.encode(
            MeshScene(camera: camera, transform: transform, cameraTransform: cameraTransform),
            pass: pass, commandBuffer: command
        )
        try gpu.finish(command)
        let pixels = try gpu.pixels(of: target.color)
        #expect(gpu.pixel(pixels, x: 32, y: 32).w == 1)
        let depths = try await readDepth(gpu, target.depth)
        let projected = try camera.projection(aspectRatio: 1, depthMode: mode) * SIMD4<Float>(0, 0, -10, 1)
        let expected: Float = mode == .logarithmic ? 0.5 : projected.z / projected.w
        #expect(abs(depths[32 * 65 + 32] - expected) < 0.000002)
    }

    @Test("PBR cutouts write identical logarithmic depth in color and surface attributes")
    func generatedPBRAndAttributes() async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let graphMaterial = try SurfaceShaderGraph { builder in
            .init(mask: builder.uv0.x.greaterThan(builder.constant(Float(0))))
        }
        let material = try PhysicallyBasedMaterial(
            alphaMode: .opaque, surfaceGraph: graphMaterial, shadingModel: .unlit
        )
        let mesh = try gpu.unlit(GeometryRenderFixture.square(4), material: .physicallyBased(material), depthMode: .logarithmic)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -8)), mesh: mesh)
        var graph = SceneGraph(roots: [node], camera: gpu.camera)
        graph.projectionJitter = SIMD2(0.25, -0.25)

        let target = try gpu.makeTarget()
        let colorCommand = try #require(gpu.queue.makeCommandBuffer())
        try graph.encode(pass: storedPass(gpu, target: target), commandBuffer: colorCommand, depthMode: .logarithmic)
        try gpu.finish(colorCommand)
        let color = try gpu.pixels(of: target.color)
        #expect(gpu.pixel(color, x: 25, y: 32) == .zero)
        #expect(gpu.pixel(color, x: 38, y: 32) == .one)
        let colorDepth = try await readDepth(gpu, target.depth)

        let attributesTarget = try gpu.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let attributes = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let attrPass = storedPass(gpu, target: (attributes, attributesTarget.depth))
        let attrCommand = try #require(gpu.queue.makeCommandBuffer())
        try graph.encodeSurfaceAttributes(pass: attrPass, commandBuffer: attrCommand, depthMode: .logarithmic)
        try gpu.finish(attrCommand)
        #expect(try await readDepth(gpu, attributesTarget.depth) == colorDepth)
        #expect(colorDepth[32 * 65 + 25] == 1)
        let expected = Float(log2(8.0 / Double(gpu.camera.near)) / log2(Double(gpu.camera.far) / Double(gpu.camera.near)))
        #expect(abs(colorDepth[32 * 65 + 38] - expected) < 0.000002)
    }

    @Test("The shadowed stock PBR pipeline changes camera depth independently of light depth")
    func shadowDepth() async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let geometry = try GeometryRenderFixture.square(1)
        let material = try PhysicallyBasedMaterial(ambientIntensity: 0.2, receivesShadows: true)
        func render(_ mode: MeshDepthMode) async throws -> (SIMD4<Float>, [Float], [Float]) {
            let mesh = try gpu.unlit(geometry, material: .physicallyBased(material), depthMode: mode)
            let graph = SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: mesh),
            ], camera: gpu.camera)
            let map = try DirectionalShadowMap(
                queue: gpu.queue, resolution: gpu.side, directionToLight: SIMD3(0, 0, 1), halfExtent: 8
            )
            let target = try gpu.makeTarget()
            let command = try #require(gpu.queue.makeCommandBuffer())
            let stats = try graph.encode(
                pass: storedPass(gpu, target: target, depthMode: mode), commandBuffer: command, shadowMap: map, depthMode: mode
            )
            #expect(stats.shadowDrawCount == 1)
            try gpu.finish(command)
            return (
                try gpu.pixel(gpu.pixels(of: target.color), x: 32, y: 32),
                try await readDepth(gpu, target.depth),
                try await readDepth(gpu, map.texture)
            )
        }

        let forward = try await render(.forward)
        let logarithmic = try await render(.logarithmic)
        #expect(logarithmic.0 == forward.0 && logarithmic.0.x > 0.1)
        #expect(logarithmic.1[32 * 65 + 32] < forward.1[32 * 65 + 32])
        #expect(logarithmic.2 == forward.2)
        #expect(logarithmic.2[32 * 65 + 32] < 1)
    }

    @Test("An explicit orthographic view keeps linear depth and rejects missing or reversed projections")
    func explicitCamera() async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let mesh = try gpu.unlit(GeometryRenderFixture.square(1), depthMode: .logarithmic)
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: mesh),
        ], camera: gpu.camera)
        let projection = simd_float4x4(columns: (
            SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0), SIMD4(0, 0, -0.1, 0), SIMD4(0, 0, 0, 1)
        ))
        let camera = try ShaderGraphCamera(viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: 0, far: 10)
        let target = try gpu.makeTarget()
        let pass = storedPass(gpu, target: target)
        let command = try #require(gpu.queue.makeCommandBuffer())
        pass.depthAttachment.clearDepth = 0
        #expect(throws: MeshRenderer.Failure.logarithmicDepthPass) {
            try graph.encode(pass: pass, commandBuffer: command, depthMode: .logarithmic)
        }
        pass.depthAttachment.clearDepth = 1
        #expect(throws: MeshRenderer.Failure.logarithmicDepthCamera) {
            try graph.encode(pass: pass, commandBuffer: command, clipFromWorld: projection, cameraPosition: .zero,
                             depthMode: .logarithmic)
        }
        let reversedProjection = try gpu.camera.projection(aspectRatio: 1, depthMode: .reversed)
        let reversed = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: reversedProjection, near: gpu.camera.near, far: gpu.camera.far
        )
        #expect(throws: MeshRenderer.Failure.logarithmicDepthCamera) {
            try graph.encode(pass: pass, commandBuffer: command, clipFromWorld: reversedProjection, cameraPosition: .zero,
                             depthMode: .logarithmic, graphCamera: reversed)
        }
        try graph.encode(pass: pass, commandBuffer: command, clipFromWorld: projection, cameraPosition: .zero,
                         depthMode: .logarithmic, graphCamera: camera)
        try gpu.finish(command)
        let depths = try await readDepth(gpu, target.depth)
        #expect(abs(depths[32 * 65 + 32] - 0.4) < 0.000002)
    }

    @Test("Projection-depth consumers and incompatible material pipelines reject logarithmic depth")
    func incompatiblePaths() throws {
        let gpu = try GeometryRenderFixture()
        let geometry = try GeometryRenderFixture.square(1)
        let classic = try ClassicMaterial(model: .lambert)
        #expect(throws: MeshRenderer.Failure.logarithmicDepthPass) {
            try gpu.unlit(geometry, material: .classic(classic), depthMode: .logarithmic)
        }
        #expect(throws: MeshRenderer.Failure.logarithmicDepthConsumer) {
            try PostProcessGraph.depthFog(
                color: .one, near: 1, far: 10, camera: gpu.camera, depthMode: .logarithmic
            )
        }
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: gpu.camera.projection(aspectRatio: 1),
            near: gpu.camera.near, far: gpu.camera.far
        )
        #expect(throws: MaterialGraph.Failure.invalidDeclaration(
            "Depth normal reconstruction requires forward or reversed projection depth."
        )) {
            try PostProcessGraph { p in
                p.getNormalFromDepth(at: p.screenUV, inverseProjection: p.inverseProjection(of: camera), depthMode: .logarithmic)
            }
        }
    }

    private func storedPass(
        _ gpu: GeometryRenderFixture, target: (color: any MTLTexture, depth: any MTLTexture),
        depthMode: MeshDepthMode = .logarithmic
    ) -> MTLRenderPassDescriptor {
        let pass = gpu.pass(color: target.color, depth: target.depth, depthMode: depthMode)
        pass.depthAttachment.storeAction = .store
        return pass
    }

    private func readDepth(_ gpu: GeometryRenderFixture, _ depth: any MTLTexture) async throws -> [Float] {
        let data = try await TextureTransfer.read(from: depth, commandQueue: gpu.queue).bytes
        return data.withUnsafeBytes { raw in
            (0..<(gpu.side * gpu.side)).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
        }
    }
}
