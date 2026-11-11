import Frostfield
import Metal
import Testing
import simd

struct GroundedSkyboxGeometryTests {
    @Test func `Projected vertices match r187 across the upper, transition and flat rings`() throws {
        let mesh = try GroundedSkybox.geometry(height: 3, radius: 10, resolution: 8)
        #expect(mesh.indices.count == 672)
        let pinned: [(Float, SIMD3<Float>)] = [
            (0.25, SIMD3(7.07106781, 7.07106781, 0)), (0.5, SIMD3(10, 0, 0)),
            (0.625, SIMD3(7.01165056, -2.90432095, 0)), (0.75, SIMD3(3, -3, 0)),
            (0.875, SIMD3(1.24264073, -3, 0))
        ]
        for (v, expected) in pinned {
            let vertex = try #require(mesh.vertices.first { $0.textureCoordinate == SIMD2(0.5, v) })
            #expect(simd_distance(vertex.position, expected) < 0.000002)
            if v >= 0.75 { #expect(vertex.normal == SIMD3(0, 1, 0)) }
        }
        for start in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = mesh.vertices[Int(mesh.indices[start])]
            let b = mesh.vertices[Int(mesh.indices[start + 1])]
            let c = mesh.vertices[Int(mesh.indices[start + 2])]
            let face = simd_cross(b.position - a.position, c.position - a.position)
            #expect(simd_dot(face, a.normal + b.normal + c.normal) > 0)
        }
        let bounds = try AxisAlignedBox(enclosing: mesh.vertices.map(\.position))
        #expect(bounds.minimum.y == -3 && bounds.maximum.y == 10)
    }

    @Test func `Skybox dimensions and tessellation bounds are validated`() throws {
        #expect(throws: GroundedSkybox.Failure.dimensions) { try GroundedSkybox.geometry(height: 0, radius: 10) }
        #expect(throws: GroundedSkybox.Failure.dimensions) { try GroundedSkybox.geometry(height: 3, radius: .infinity) }
        #expect(throws: GroundedSkybox.Failure.resolution) { try GroundedSkybox.geometry(height: 3, radius: 10, resolution: Int.max) }
        let coarse = try GroundedSkybox.geometry(height: 3, radius: 10, resolution: 1)
        #expect(coarse.indices.count == 18)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GroundedSkyboxRasterTests {
    @Test func `The mesh samples HDR panorama directions and follows authored yaw`() throws {
        let gpu = try RasterLightingFixture(width: 17, height: 17)
        let panorama = try makePanorama(device: gpu.device)
        let sky = try GroundedSkybox.makeNode(panorama: panorama, height: 3, radius: 15, resolution: 32)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 40)
        let background = try EnvironmentBackground(device: gpu.device, source: .equirectangular(panorama))
        for direction: SIMD3<Float> in [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0),
                                        SIMD3(0, 0, 1), SIMD3(0, 0, -1), simd_normalize(SIMD3(3, -2, 1))] {
            let pose = cameraPose(at: sky.transform.translation, toward: direction)
            let graph = SceneGraph(roots: [sky], camera: camera, cameraTransform: pose)
            let actual = try center(gpu, gpu.render(graph).color)
            let direct = try gpu.colorTarget()
            try gpu.run { try background.encode(destination: direct, camera: camera, cameraTransform: pose, commandBuffer: $0) }
            #expect(try simd_distance(actual, center(gpu, direct)) < 0.036, "Direction \(direction): \(actual)")
            #expect(simd_distance(actual, SIMD4(color(direction), 1)) < 0.045)
        }
        var rotated = sky
        rotated.transform.rotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
        let direction = SIMD3<Float>(0, 0, 1)
        let pose = cameraPose(at: sky.transform.translation, toward: direction)
        let rotatedColor = try center(gpu, gpu.render(SceneGraph(roots: [rotated], camera: camera, cameraTransform: pose)).color)
        #expect(simd_distance(rotatedColor, SIMD4(color(rotated.transform.rotation.inverse.act(direction)), 1)) < 0.035)
    }

    @Test(arguments: [MeshDepthMode.forward, .reversed])
    func `The floor aligns with world contact and leaves the depth attachment unchanged`(mode: MeshDepthMode) async throws {
        let gpu = try RasterLightingFixture(width: 17, height: 17)
        let panorama = try makePanorama(device: gpu.device)
        let sky = try GroundedSkybox.makeNode(panorama: panorama, height: 3, radius: 15, resolution: 32, depthMode: mode)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 40)
        let pose = cameraPose(at: SIMD3(2, 3, 0), toward: SIMD3(0, -1, 0))
        let markerColor = SIMD3<Float>(8, 0, 0)
        let marker = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.7, height: 0.7),
            material: .classic(try ClassicMaterial(model: .basic, color: markerColor)), depthMode: mode
        )
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        func render(markerY: Float?) throws -> (SIMD4<Float>, any MTLTexture) {
            var graph = SceneGraph(roots: [sky], camera: camera, cameraTransform: pose)
            if let markerY {
                graph.roots.append(SceneNode(transform: .init(translation: SIMD3(2, markerY, 0), rotation: rotation), mesh: marker))
            }
            let target = try gpu.colorTarget(), depth = try gpu.depthTarget()
            try gpu.run { command in
                try graph.encode(pass: gpu.pass(color: target, depth: depth, depthMode: mode), commandBuffer: command, depthMode: mode)
            }
            return (try center(gpu, target), depth)
        }
        let (floor, depth) = try render(markerY: nil)
        let expected = SIMD4(color(simd_normalize(SIMD3(2, -3, 0))), 1)
        #expect(simd_distance(floor, expected) < 0.035)
        let readback = try await TextureTransfer.read(from: depth, region: MTLRegionMake2D(8, 8, 1, 1), commandQueue: gpu.queue)
        let windowDepth = readback.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
        #expect(windowDepth == (mode == .forward ? 1 : 0))
        let (above, _) = try render(markerY: 0.25)
        let (below, _) = try render(markerY: -0.25)
        #expect(above == SIMD4(markerColor, 1))
        #expect(below == floor)
    }

    @Test func `A non equirectangular image is rejected before scene preparation`() throws {
        let gpu = try RasterLightingFixture(width: 4, height: 4)
        let square = try gpu.colorTarget()
        #expect(throws: GroundedSkybox.Failure.panorama) {
            try GroundedSkybox.makeNode(panorama: square, height: 3, radius: 15)
        }
    }

    private func color(_ direction: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(2.5, 3, 2.5) + 1.5 * direction
    }

    private func cameraPose(at position: SIMD3<Float>, toward direction: SIMD3<Float>) -> SceneTransform {
        .init(translation: position, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: direction))
    }

    private func center(_ gpu: RasterLightingFixture, _ texture: any MTLTexture) throws -> SIMD4<Float> {
        try gpu.pixels(of: texture)[gpu.width * gpu.height / 2]
    }

    private func makePanorama(device: any MTLDevice) throws -> any MTLTexture {
        let width = 256, height = 128
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let pixels = (0..<(width * height)).map { i -> SIMD4<Float16> in
            let phi = ((Float(i % width) + 0.5) / Float(width) - 0.5) * 2 * .pi
            let theta = (Float(i / width) + 0.5) / Float(height) * .pi
            let direction = SIMD3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi))
            return SIMD4<Float16>(SIMD4(color(direction), 0))
        }
        pixels.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: width * MemoryLayout<SIMD4<Float16>>.stride)
        }
        return texture
    }
}
