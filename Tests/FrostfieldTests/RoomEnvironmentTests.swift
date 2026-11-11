import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RoomEnvironmentTests {
    @Test func `The room exposes the pinned layout and inward shell geometry`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let graph = try RoomEnvironment.makeScene(device: device)
        let root = try #require(graph.roots.first)
        #expect(graph.roots.count == 1 && root.name == "RoomEnvironment")
        #expect(root.transform.translation == SIMD3(0, -3.5, 0))
        #expect(root.children.count == 14)
        let key = try #require(root.children[0].light)
        #expect(key.kind == .point(range: 28) && key.intensity == 900 && key.shadow == nil)
        #expect(root.children[0].transform.translation == SIMD3(0.418, 16.199, 0.300))
        let shell = root.children[1]
        #expect(shell.transform.scale == SIMD3(31.713, 28.305, 28.591))
        let geometry = try #require(shell.mesh?.geometry)
        for index in stride(from: 0, to: geometry.indices.count, by: 3) {
            let a = geometry.vertices[Int(geometry.indices[index])]
            let b = geometry.vertices[Int(geometry.indices[index + 1])]
            let c = geometry.vertices[Int(geometry.indices[index + 2])]
            #expect(simd_dot(simd_cross(b.position - a.position, c.position - a.position), a.normal) > 0)
            #expect(simd_dot(a.normal, a.position) < 0)
        }
        let box = try #require(root.children[2].mesh)
        #expect(root.children[2..<8].allSatisfy { $0.mesh === box })
        #expect(root.children[7].transform.translation == SIMD3(-2.193, -0.369, -5.547))
        #expect(root.children[7].transform.scale == SIMD3(3.875, 3.487, 2.986))
        #expect(graph.camera.near == 0.1 && graph.camera.far == 100)
    }

    @Test func `Room capture preserves all six authored panel radiances and reflected floor light`() throws {
        let gpu = try RasterLightingFixture(width: 16, height: 16)
        let lighting = try RoomEnvironment.prepare(queue: gpu.queue)
        let directions = [
            SIMD3<Float>(-16.116, 10.870, 8.208), SIMD3(-16.109, 14.521, -8.207), SIMD3(14.904, 8.698, -1.832),
            SIMD3(-0.462, 5.390, 14.520), SIMD3(3.235, 7.986, -12.541), SIMD3(0, 1, 0), SIMD3(0, -1, 0)
        ]
        let sky = try gpu.sampleCube(lighting.skyCube, directions: directions)
        for (index, radiance) in [Float(50), 50, 17, 43, 20, 100].enumerated() {
            let actual = sky[index]
            #expect(simd_distance(actual, SIMD4(radiance, radiance, radiance, 1)) < 0.08, "Emitter \(index + 1): \(actual)")
        }
        #expect(sky[6].x > 0.2 && sky[6].x < 3 && sky[6].w == 1)
        let diffuse = try gpu.sampleCube(lighting.diffuseCube, directions: [SIMD3(0, 1, 0), SIMD3(0, 0, 1)])
        #expect(diffuse.allSatisfy { $0.x > 0.1 && $0.x < 20 })
        #expect(lighting.skyCube.width == 256 && lighting.specularCube.mipmapLevelCount == 9)
        #expect(throws: EnvironmentLighting.Failure.settings) {
            try RoomEnvironment.prepare(queue: gpu.queue, settings: .init(cubeSize: 12))
        }
        #expect(throws: CubeCamera.Failure.position) {
            try RoomEnvironment.prepare(queue: gpu.queue, at: SIMD3(.nan, 0, 0), settings: settings)
        }
    }

    @Test func `Raster and path-traced mirrors reflect the captured positive Z panel`() throws {
        let gpu = try PathTestFixture()
        let lighting = try RoomEnvironment.prepare(queue: gpu.queue, settings: settings)
        let mirror = try gpu.pbr(.plane(width: 2, height: 2), color: SIMD3(repeating: 1), roughness: 0, metallic: 1)
        let forward = SIMD3<Float>(0, 0, 1)
        let panelDirection = simd_normalize(SIMD3<Float>(-0.462, 5.390, 14.520))
        let normal = simd_normalize(forward + panelDirection)
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3), rotation: simd_quatf(from: forward, to: normal)), mesh: mirror
        )
        var graph = SceneGraph(roots: [node], camera: gpu.camera)
        graph.environment = lighting
        let size = 33
        let center = size * size / 2
        let raster = try gpu.rasterize(graph, width: size, height: size)[center]
        #expect(raster.x > 20 && raster.w == 1)
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let inherited = try PathTracer.Lighting(queue: gpu.queue, graph: graph, backgroundVisible: false)
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: inherited)
        let traceSettings = PathTracer.Settings(maxBounces: 2, seed: 17)
        let path = try gpu.trace(tracer, width: size, height: size, samples: 4, settings: traceSettings)[center]
        #expect(path.x > 20 && path.w == 1)

        graph.environmentIntensity = 0
        let rasterOff = try gpu.rasterize(graph, width: size, height: size)[center]
        let off = try PathTracer.Lighting(queue: gpu.queue, graph: graph, backgroundVisible: false)
        let pathOff = try gpu.trace(tracer, width: size, height: size, samples: 4, settings: traceSettings, lighting: off)[center]
        #expect(rasterOff == SIMD4(0, 0, 0, 1))
        #expect(pathOff == SIMD4(0, 0, 0, 1))
    }

    private let settings = EnvironmentLighting.Settings(cubeSize: 128, diffuseSize: 8, brdfSize: 32, sampleCount: 64)
}
