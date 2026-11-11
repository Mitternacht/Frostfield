import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightProbeGridRasterTests {
    @Test func `Volume lookup uses layers and priority after applying the normal offset`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let bounds = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 2))
        let red = try uniformGrid(SIMD3(3, 0, 0), bounds: bounds)
        let green = try uniformGrid(SIMD3(0, 3, 0), bounds: bounds)
        let blue = try uniformGrid(SIMD3(0, 0, 3), bounds: bounds)
        let upperLayer = SceneLayers(layer: 31)
        let lighting = try LightProbeGridLighting(device: device, volumes: [
            .init(red, priority: 3), .init(green, priority: 3), .init(blue, priority: 4, layers: upperLayer),
            .init(red, priority: -1)
        ])
        let point = SIMD3<Float>(repeating: 1)
        let upward = SIMD3<Float>(0, 1, 0)
        #expect(try simd_distance(lighting.irradiance(at: point, normal: upward), SIMD3(0, 3, 0)) < 0.000001)
        #expect(try simd_distance(lighting.irradiance(at: point, normal: upward, cameraLayers: .all), SIMD3(0, 0, 3)) < 0.000001)
        #expect(try lighting.irradiance(at: point, normal: upward, cameraLayers: []) == .zero)
        #expect(try lighting.irradiance(at: SIMD3(1, 1.021, 1), normal: upward) == .zero)
        #expect(try lighting.irradiance(at: SIMD3(1, 1.019, 1), normal: upward).y > 2.999)
        #expect(try lighting.irradiance(at: SIMD3(1, 2.7, 1), normal: -upward).y > 2.999)
    }

    @Test func `Raster PBR and lit classic materials sample the local grid and add distant probes`() throws {
        let fixture = try RasterLightingFixture(width: 64, height: 64)
        let bounds = try AxisAlignedBox(minimum: SIMD3(-1.75, -1.75, -4.75), maximum: SIMD3(1.75, 1.75, -0.75))
        let overlap = try AxisAlignedBox(minimum: SIMD3(0.2, -1.5, -4.75), maximum: SIMD3(1.8, 1.5, -0.75))
        let probes = try (0..<8).map { index in
            let i = Float(index)
            var coefficients = [SIMD3(1 + i * 0.2, 0.6 + i * 0.1, 0.3)]
            for coefficient in 1..<LightProbe.coefficientCount {
                let term = Float(coefficient) * 0.03125
                coefficients.append(SIMD3(term, i * term, -term))
            }
            return try LightProbe(coefficients: coefficients, intensity: 1.25)
        }
        let grid = try LightProbeGrid(bounds: bounds, resolution: SIMD3(repeating: 2), probes: probes, intensity: 1.3)
        let lighting = try LightProbeGridLighting(device: fixture.device, volumes: [
            .init(grid), .init(try uniformGrid(SIMD3(0, 4, 0), bounds: overlap), priority: 1),
            .init(try uniformGrid(SIMD3(0, 0, 4), bounds: overlap), priority: 1, layers: SceneLayers(layer: 31))
        ])
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 12)
        let distant = try LightProbe.hemisphere(skyColor: SIMD3(0.25, 0.5, 0.75), groundColor: SIMD3(0.25, 0.5, 0.75))
        let surface = try SurfaceShaderGraph(nodes: [.color(SIMD3(repeating: 1))], outputs: .init(baseColor: 0))
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let atlas = try #require(fixture.device.makeTexture(descriptor: descriptor))
        var bakedTexel = SIMD4<Float16>(1.5, 0.25, 0.5, 1)
        withUnsafeBytes(of: &bakedTexel) { bytes in
            atlas.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: bytes.count)
        }
        let bakedMap = try PhysicallyBasedMaterial.LightMap(texture: atlas, textureCoordinateSet: .primary)
        let materials: [(MeshMaterial, SIMD3<Float>)] = [
            (.classic(try ClassicMaterial(model: .lambert)), .zero),
            (.physicallyBased(try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0,
                transmission: .init(factor: 0, indexOfRefraction: 1)
            )), .zero),
            (.physicallyBased(try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, receivesShadows: true,
                transmission: .init(factor: 0, indexOfRefraction: 1), surfaceGraph: surface, lightMap: bakedMap
            )), SIMD3(1.5, 0.25, 0.5))
        ]
        let shadow = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 16, directionToLight: SIMD3(0, 0, 1), halfExtent: 8, intensity: 0
        )
        let normal = SIMD3<Float>(0, 0, 1)
        let distantIrradiance = try distant.irradiance(at: normal)
        for (material, bakedIrradiance) in materials {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 5, height: 4), material: material
            )
            let receiver = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh, castsShadow: false)
            var graph = SceneGraph(roots: [receiver], camera: camera)
            graph.lightProbeGridLighting = lighting
            for layers: SceneLayers in [.default, .all] {
                graph.cameraLayers = layers
                graph.lightProbes = layers == .all ? [distant] : []
                let pixels = try fixture.pixels(of: fixture.render(graph, shadowMap: shadow).color)
                for (x, y) in [(18, 24), (29, 35), (44, 40), (54, 32)] {
                    let position = SIMD3((Float(x) + 0.5) * 6 / 64 - 3, 3 - (Float(y) + 0.5) * 6 / 64, -3)
                    let local = try lighting.irradiance(at: position, normal: normal, cameraLayers: layers)
                    let global = layers == .all ? distantIrradiance : .zero
                    let expected = SIMD4((local + global + bakedIrradiance) / Float.pi, 1)
                    let actual = pixels[y * 64 + x]
                    #expect(simd_distance(actual, expected) < 0.003, "Pixel \(x), \(y), layer mask \(layers.rawValue): \(actual) vs \(expected)")
                }
            }
        }
    }

    @Test func `Prepared volume limits and archive preflight reject unsupported inputs`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let bounds = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 1))
        let grid = try uniformGrid(SIMD3(1, 0, 0), bounds: bounds)
        let volume = LightProbeGridLighting.Volume(grid)
        #expect(throws: LightProbeGridLighting.Failure.volumeCount) {
            try LightProbeGridLighting(device: device, volumes: [])
        }
        #expect(throws: LightProbeGridLighting.Failure.volumeCount) {
            try LightProbeGridLighting(device: device, volumes: Array(repeating: volume, count: 33))
        }
        let big = try LightProbeGrid(bounds: bounds, resolution: SIMD3(256, 256, 1),
                                     probes: Array(repeating: grid.probes[0], count: 65_536))
        #expect(throws: LightProbeGridLighting.Failure.probeCount) {
            try LightProbeGridLighting(device: device, volumes: [volume, .init(big)])
        }
        let huge = try AxisAlignedBox(minimum: SIMD3(repeating: -.greatestFiniteMagnitude),
                                      maximum: SIMD3(repeating: .greatestFiniteMagnitude))
        let hugeGrid = try uniformGrid(SIMD3(1, 0, 0), bounds: huge)
        #expect(throws: LightProbeGridLighting.Failure.bounds) {
            try LightProbeGridLighting(device: device, volumes: [.init(hugeGrid)])
        }
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 12)
        var graph = SceneGraph(roots: [], camera: camera)
        graph.lightProbeGridLighting = try LightProbeGridLighting(device: device, volumes: [volume])
        #expect(throws: SceneArchive.Failure.encoding("scene light probe volumes need to be restored separately")) {
            try SceneArchive(capturing: graph)
        }
    }

    private func uniformGrid(_ value: SIMD3<Float>, bounds: AxisAlignedBox) throws -> LightProbeGrid {
        let probe = try LightProbe.hemisphere(skyColor: value, groundColor: value)
        return try LightProbeGrid(bounds: bounds, resolution: SIMD3(repeating: 1), probes: [probe])
    }
}
