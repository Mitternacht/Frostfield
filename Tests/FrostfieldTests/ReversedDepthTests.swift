import Frostfield
import Metal
import Testing
import simd

struct ReversedDepthProjectionTests {
    @Test("Perspective and cropped projections reverse near and far without changing image coordinates")
    func endpoints() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.25, far: 16)
        let window = try CameraViewOffset(fullWidth: 800, fullHeight: 400, x: 125, y: 75, width: 300, height: 200)
        let forward = try camera.projection(aspectRatio: 2)
        let reversed = try camera.projection(aspectRatio: 2, depthMode: .reversed)
        let croppedForward = try camera.projection(viewOffset: window)
        let croppedReversed = try camera.projection(viewOffset: window, depthMode: .reversed)
        #expect(try camera.projection(aspectRatio: 2, depthMode: .forward) == forward)
        #expect(abs(depth(reversed, at: camera.near) - 1) < 0.000001)
        #expect(abs(depth(reversed, at: camera.far)) < 0.000001)
        #expect(abs(reversed.columns.2.z - 1.0 / 63) < 0.000001)
        #expect(abs(reversed.columns.3.z - 16.0 / 63) < 0.000001)
        for distance in [camera.near, 0.75, 3.0, camera.far] {
            let point = SIMD4<Float>(0.14, -0.21, -distance, 1)
            let a = croppedForward * point
            let b = croppedReversed * point
            #expect(SIMD3(a.x, a.y, a.w) == SIMD3(b.x, b.y, b.w))
            #expect(abs(b.z / b.w - depth(reversed, at: distance)) < 0.000001)
            #expect(abs((a.z + b.z) / b.w - 1) < 0.000001)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ReversedDepthTests {
    @Test("Standalone renderer depth compares across draws and its surface graph reads the same projection",
          arguments: [MeshDepthMode.forward, .reversed])
    func standalone(_ mode: MeshDepthMode) async throws {
        let gpu = try GeometryRenderFixture()
        let graph = try SurfaceShaderGraph { s in
            let zero = s.constant(Float(0))
            return .init(baseColor: s.vector(s.projectionMatrix.column(2).z * 0.5 + 0.7, zero, zero))
        }
        let material = try PhysicallyBasedMaterial(surfaceGraph: graph, shadingModel: .unlit)
        let near = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.8),
            material: .physicallyBased(material), depthMode: mode
        )
        let far = try mesh(gpu, mode)
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let cameraTransform = SceneTransform(translation: SIMD3(1, 2, 3))
        let nearPass = pass(gpu, target, mode)
        try near.encode(
            MeshScene(camera: gpu.camera, transform: .init(translation: SIMD3(1, 2, 1)), cameraTransform: cameraTransform),
            pass: nearPass, commandBuffer: command
        )
        let farPass = pass(gpu, target, mode)
        farPass.depthAttachment.loadAction = .load
        farPass.colorAttachments[0].loadAction = .load
        try far.encode(
            MeshScene(
                camera: gpu.camera, transform: .init(translation: SIMD3(1, 2, -2)), cameraTransform: cameraTransform,
                materialParameters: SIMD4(0, 0, 1, 1)
            ),
            pass: farPass, commandBuffer: command
        )
        try gpu.finish(command)

        let projection = try gpu.camera.projection(aspectRatio: 1, depthMode: mode)
        let pixels = try gpu.pixels(of: target.color)
        let center = gpu.side / 2
        let actual = gpu.pixel(pixels, x: center, y: center)
        let expected = SIMD4(projection.columns.2.z * 0.5 + 0.7, 0, 0, 1)
        #expect(isNearly(actual, expected), "\(mode): \(actual), expected \(expected)")
        #expect(gpu.pixel(pixels, x: 1, y: center) == .zero)
        let row = try await readRow(gpu, target.depth, y: center)
        #expect(abs(row[center] - depth(projection, at: 2)) < 0.000001)
        #expect(row[1] == (mode == .reversed ? 0 : 1))
    }

    @Test("Owned reversed-depth color pass preserves primitive occlusion, culling and jitter at a 2⁶⁰ origin")
    func scene() async throws {
        let gpu = try GeometryRenderFixture()
        let renderer = try mesh(gpu, .reversed)
        let point = ScenePrimitive.points(try PointsRenderer(
            device: gpu.device, cloud: PointCloud(positions: [.zero]),
            material: .init(size: .pixels(7), color: SIMD4(0, 1, 0, 1))
        ))
        func pointNode(_ x: Float, _ z: Float) -> SceneNode {
            var node = SceneNode(transform: .init(translation: SIMD3(x, 0, z)), primitive: point)
            node.renderOrder = 1
            return node
        }
        let nodes = [
            SceneNode(
                transform: .init(translation: SIMD3(0, 0, -2)), mesh: renderer,
                materialParameters: SIMD4(1, 0, 0, 1)
            ),
            pointNode(0, -3),
            pointNode(0.95, -1.5),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -0.05)), mesh: renderer),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -70)), mesh: renderer),
            SceneNode(transform: .init(translation: SIMD3(5, 0, -2)), primitive: point),
        ]
        func render(origin: SIMD3<Float>, precise: Bool, jitter: SIMD2<Float>) throws
            -> (pixels: [SIMD4<Float>], depth: any MTLTexture, stats: SceneDrawStatistics) {
            var graph = SceneGraph(
                roots: [SceneNode(transform: .init(translation: origin), children: nodes)],
                camera: gpu.camera, cameraTransform: .init(translation: origin)
            )
            graph.highPrecision = precise
            graph.projectionJitter = jitter
            let target = try gpu.makeTarget()
            let command = try #require(gpu.queue.makeCommandBuffer())
            let stats = try graph.encode(pass: pass(gpu, target, .reversed), commandBuffer: command, depthMode: .reversed)
            try gpu.finish(command)
            return (try gpu.pixels(of: target.color), target.depth, stats)
        }
        let origin = SIMD3<Float>(0x1p60, 0, 0x1p60)
        let jitter = SIMD2<Float>(0.75, -0.625)
        let baseline = try render(origin: .zero, precise: false, jitter: jitter)
        let moved = try render(origin: origin, precise: true, jitter: jitter)
        let steady = try render(origin: origin, precise: true, jitter: .zero)
        #expect(moved.pixels == baseline.pixels)
        #expect(moved.pixels != steady.pixels)
        #expect(moved.stats.drawnMeshCount == 1 && moved.stats.culledMeshCount == 2)
        #expect(moved.stats.drawnPrimitiveCount == 2 && moved.stats.culledPrimitiveCount == 1)
        let center = gpu.side / 2
        let pointColumn = gpu.column(x: 0.95, depth: 1.5)
        #expect(isNearly(gpu.pixel(moved.pixels, x: center, y: center), SIMD4(0.22, 0, 0, 1)))
        #expect(gpu.pixel(moved.pixels, x: pointColumn, y: center) == SIMD4(0, 1, 0, 1))
        let projection = try gpu.camera.projection(aspectRatio: 1, depthMode: .reversed)
        let baselineDepth = try await readRow(gpu, baseline.depth, y: center)
        let movedDepth = try await readRow(gpu, moved.depth, y: center)
        let steadyDepth = try await readRow(gpu, steady.depth, y: center)
        #expect(zip(movedDepth, baselineDepth).allSatisfy { abs($0 - $1) < 0.000001 })
        #expect(abs(movedDepth[center] - depth(projection, at: 2)) < 0.000001)
        #expect(abs(movedDepth[pointColumn] - depth(projection, at: 1.5)) < 0.000001)
        #expect(abs(movedDepth[center] - steadyDepth[center]) < 0.000001)
        #expect(movedDepth[0] == 0)
    }

    @Test("Owned reversed-depth attributes write matching depth and reject a forward pass")
    func attributes() async throws {
        let gpu = try GeometryRenderFixture()
        let renderer = try mesh(gpu, .reversed)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)
        let graph = SceneGraph(roots: [node], camera: gpu.camera)
        let target = try gpu.makeTarget()
        let format = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: gpu.side, height: gpu.side, mipmapped: false
        )
        format.usage = .renderTarget
        let attributes = try #require(gpu.device.makeTexture(descriptor: format))
        let pass = pass(gpu, (attributes, target.depth), .reversed)
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: MeshRenderer.Failure.depthMode) {
            try graph.encodeSurfaceAttributes(pass: pass, commandBuffer: command)
        }
        let stats = try graph.encodeSurfaceAttributes(pass: pass, commandBuffer: command, depthMode: .reversed)
        #expect(stats.drawnMeshCount == 1)
        try gpu.finish(command)
        let projection = try gpu.camera.projection(aspectRatio: 1, depthMode: .reversed)
        let row = try await readRow(gpu, target.depth, y: gpu.side / 2)
        #expect(abs(row[gpu.side / 2] - depth(projection, at: 3)) < 0.000001)
        #expect(row[1] == 0)
    }

    private func mesh(_ gpu: GeometryRenderFixture, _ mode: MeshDepthMode) throws -> MeshRenderer {
        try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.6),
            depthMode: mode
        )
    }

    private func pass(
        _ gpu: GeometryRenderFixture, _ target: (color: any MTLTexture, depth: any MTLTexture), _ mode: MeshDepthMode
    ) -> MTLRenderPassDescriptor {
        let pass = gpu.pass(color: target.color, depth: target.depth)
        pass.depthAttachment.clearDepth = mode == .reversed ? 0 : 1
        pass.depthAttachment.storeAction = .store
        return pass
    }

    private func readRow(_ gpu: GeometryRenderFixture, _ texture: any MTLTexture, y: Int) async throws -> [Float] {
        let row = try await TextureTransfer.read(
            from: texture, region: MTLRegionMake2D(0, y, gpu.side, 1), commandQueue: gpu.queue
        )
        return row.bytes.withUnsafeBytes { bytes in
            (0..<gpu.side).map { bytes.loadUnaligned(fromByteOffset: $0 * MemoryLayout<Float>.stride, as: Float.self) }
        }
    }
}

private func depth(_ projection: simd_float4x4, at distance: Float) -> Float {
    let clip = projection * SIMD4<Float>(0, 0, -distance, 1)
    return clip.z / clip.w
}
