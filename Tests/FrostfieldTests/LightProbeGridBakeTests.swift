import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightProbeGridBakeTests {
    @Test func `A closed diffuse chamber adds one transport order on each pass`() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let layers = SceneLayers(layer: 7)
        let bounds = try AxisAlignedBox(minimum: SIMD3(repeating: -1), maximum: SIMD3(repeating: 1))
        let resolution = SIMD3(2, 1, 1)
        let blackProbe = try LightProbe.hemisphere(skyColor: .zero, groundColor: .zero)
        let black = try LightProbeGrid(bounds: bounds, resolution: resolution, probes: [blackProbe, blackProbe])
        var scene = try chamber(device: device, layers: layers)
        scene.lightProbeGridLighting = try LightProbeGridLighting(device: device, volumes: [
            .init(black, priority: 5, layers: layers)
        ])

        let direct = try await LightProbeGrid.capture(
            scene, bounds: bounds, resolution: resolution, commandQueue: queue, layers: layers
        )
        let once = try await LightProbeGrid.bake(
            scene, bounds: bounds, resolution: resolution, commandQueue: queue, bounces: 1, bouncePriority: 5, layers: layers
        )
        let twice = try await LightProbeGrid.bake(
            scene, bounds: bounds, resolution: resolution, commandQueue: queue, bounces: 2, bouncePriority: 5, layers: layers
        )
        // Ambient irradiance is 2π, albedo 0.5, and the cube sees the same radiance
        // in all directions. A bounce adds πL to the ambient, hence L' = 1 + 0.5L.
        for (grid, radiance) in [(direct, Float(1)), (once, 1.5), (twice, 1.75)] {
            for probe in grid.probes {
                let actual = try probe.radiance(in: SIMD3(1, 2, -3))
                #expect(simd_distance(actual, SIMD3(repeating: radiance)) < 0.005)
            }
        }

        let source = LightProbeGridLighting.Volume(direct, priority: 5, layers: layers)
        let first = try await direct.updating(probes: 0..<1, from: scene, commandQueue: queue, bounceSource: source, layers: layers)
        let combined = try await first.updating(probes: 1..<2, from: scene, commandQueue: queue, bounceSource: source, layers: layers)
        #expect(first.probes[0] == once.probes[0] && first.probes[0] != direct.probes[0])
        #expect(first.probes[1] == direct.probes[1])
        #expect(combined.probes == once.probes)
        #expect(source.grid.probes == direct.probes)

        let lowerPriority = LightProbeGridLighting.Volume(direct, priority: 4, layers: layers)
        let masked = try await direct.updating(probes: 0..<1, from: scene, commandQueue: queue,
                                               bounceSource: lowerPriority, layers: layers)
        #expect(masked.probes == direct.probes)
        let hiddenLayer = LightProbeGridLighting.Volume(direct, priority: 6, layers: .default)
        let hidden = try await direct.updating(probes: 1..<2, from: scene, commandQueue: queue,
                                               bounceSource: hiddenLayer, layers: layers)
        #expect(hidden.probes == direct.probes)
        #expect(scene.lightProbeGridLighting?.volumes.count == 1)
        #expect(scene.lightProbeGridLighting?.volumes[0].grid.probes == black.probes)
    }

    @Test func `Ranged replacement preserves all untouched cells and the grid intensity`() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let bounds = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 1))
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 8)
        let scene = SceneGraph(roots: [], camera: camera)
        let probes = try (0..<8).map { index in
            try LightProbe.hemisphere(skyColor: SIMD3(repeating: Float(index)), groundColor: .zero)
        }
        let grid = try LightProbeGrid(bounds: bounds, resolution: SIMD3(repeating: 2), probes: probes, intensity: 2)
        let updated = try await grid.updating(
            probes: 3..<6, from: scene, commandQueue: queue,
            clearColor: MTLClearColor(red: 2, green: 0.5, blue: 1, alpha: 1)
        )
        #expect(updated.bounds == bounds && updated.resolution == grid.resolution && updated.intensity == 2)
        #expect(updated.probes[0..<3] == grid.probes[0..<3])
        #expect(updated.probes[6..<8] == grid.probes[6..<8])
        for index in 3..<6 {
            #expect(try simd_distance(updated.probes[index].radiance(in: SIMD3(0, 1, 0)), SIMD3(2, 0.5, 1)) < 0.0001)
        }
        #expect(grid.probes == probes)
        let empty = try await grid.updating(probes: 8..<8, from: scene, commandQueue: queue)
        #expect(empty.probes == probes && empty.intensity == 2)
    }

    @Test func `Bakes validate the combined budget and range source layout`() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let bounds = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 1))
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 8)
        let scene = SceneGraph(roots: [], camera: camera)
        for count in [-1, Int.max] {
            await #expect(throws: LightProbeGrid.Failure.bounceCount) {
                try await LightProbeGrid.bake(scene, bounds: bounds, resolution: SIMD3(repeating: 1), commandQueue: queue, bounces: count)
            }
        }
        await #expect(throws: LightProbeGrid.Failure.captureBudget) {
            try await LightProbeGrid.bake(scene, bounds: bounds, resolution: SIMD3(3_000, 1, 1), commandQueue: queue, bounces: 1)
        }
        await #expect(throws: LightProbeGrid.Failure.captureBudget) {
            try await LightProbeGrid.bake(scene, bounds: bounds, resolution: SIMD3(2, 1, 1), commandQueue: queue, bounces: 1, faceSize: 1_024)
        }
        let black = try LightProbe.hemisphere(skyColor: .zero, groundColor: .zero)
        let grid = try LightProbeGrid(bounds: bounds, resolution: SIMD3(repeating: 1), probes: [black])
        let wrong = try LightProbeGrid(bounds: bounds, resolution: SIMD3(2, 1, 1), probes: [black, black])
        await #expect(throws: LightProbeGrid.Failure.bounceLayout) {
            try await grid.updating(probes: 0..<1, from: scene, commandQueue: queue, bounceSource: .init(wrong))
        }
        for range in [-1..<0, 1..<2] {
            await #expect(throws: LightProbeGrid.Failure.probeRange) {
                try await grid.updating(probes: range, from: scene, commandQueue: queue)
            }
        }
    }

    private func chamber(device: any MTLDevice, layers: SceneLayers) throws -> SceneGraph {
        let material = try ClassicMaterial(model: .lambert, color: SIMD3(repeating: 0.5), ambient: SIMD3(repeating: 2 * .pi))
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2.2, height: 2.2), material: .classic(material)
        )
        let x = SIMD3<Float>(1, 0, 0), y = SIMD3<Float>(0, 1, 0), z = SIMD3<Float>(0, 0, 1)
        let walls = [x, -x, y, -y, z, -z].map { outward in
            SceneNode(transform: .init(translation: outward, rotation: simd_quatf(from: z, to: -outward)), mesh: mesh, layers: layers)
        }
        return SceneGraph(roots: walls, camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 8))
    }
}
