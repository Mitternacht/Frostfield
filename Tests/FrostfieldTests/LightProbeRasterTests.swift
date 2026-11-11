import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightProbeRasterTests {
    @Test("Hemisphere lighting in classic and PBR pixels follows world normals and the authored sky axis")
    func hemisphereLighting() throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 16)
        let sky = SIMD3<Float>(2, 0.5, 0.25)
        let ground = SIMD3<Float>(0.25, 0.125, 1.5)
        let pbr = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0,
            transmission: .init(factor: 0, indexOfRefraction: 1)
        )
        let rotation = simd_quatf(angle: .pi / 4, axis: SIMD3<Float>(0, 1, 0))
        let normal = rotation.act(SIMD3(0, 0, 1))
        for material: MeshMaterial in [.physicallyBased(pbr), .classic(try ClassicMaterial(model: .lambert))] {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2), material: material
            )
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3), rotation: rotation), mesh: mesh)
            var graph = SceneGraph(roots: [node], camera: camera)
            for direction: SIMD3<Float> in [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(-1, 0, 0)] {
                graph.lightProbes = [try LightProbe.hemisphere(skyColor: sky, groundColor: ground, direction: direction)]
                let weight = (1 + simd_dot(normal, direction)) / 2
                let irradiance = ground * (1 - weight) + sky * weight
                let pixel = try center(fixture, graph: graph)
                let expected = SIMD4(irradiance / Float.pi, 1)
                #expect(simd_distance(pixel, expected) < 0.001, "Axis \(direction): \(pixel) vs \(expected)")
            }
        }
    }

    @Test("Projected HDR lighting responds to transformed normals in PBR and classic materials")
    func transformedNormals() async throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        let probe = try await projectedProbe(device: fixture.device, queue: fixture.queue)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 16)
        let pbr = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0,
            transmission: .init(factor: 0, indexOfRefraction: 1)
        )
        for material: MeshMaterial in [.physicallyBased(pbr), .classic(try ClassicMaterial(model: .lambert)),
                                      .classic(try ClassicMaterial(model: .phong()))] {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float,
                geometry: .plane(width: 2, height: 2), material: material
            )
            for angle: Float in [-.pi / 4, .pi / 4] {
                let rotation = simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))
                let object = SceneNode(
                    transform: .init(translation: SIMD3(0, 0, -3), rotation: rotation), mesh: mesh
                )
                var graph = SceneGraph(roots: [object], camera: camera)
                #expect(try center(fixture, graph: graph) == SIMD4(0, 0, 0, 1))
                graph.lightProbes = [probe]
                let normal = rotation.act(SIMD3(0, 0, 1))
                let expected = try simd_max(probe.irradiance(at: normal), .zero) / Float.pi
                let lit = try center(fixture, graph: graph)
                #expect(simd_distance(lit, SIMD4(expected, 1)) < 0.0015)
                #expect(angle > 0 ? lit.x > lit.z : lit.z > lit.x)

                var partial = probe
                try partial.setIntensity(0.5)
                graph.lightProbes = [partial, partial]
                #expect(try center(fixture, graph: graph) == lit)
            }
        }
    }

    @Test("Diffuse probes add to cubemap IBL in the shadowed PBR pipeline with independent scale")
    func independentIBL() async throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 16)
        let probe = try await projectedProbe(device: fixture.device, queue: fixture.queue)
        let environment = try fixture.environment { _, _ in SIMD3(0.25, 0.125, 0.5) }
        let material = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, receivesShadows: true, environment: environment,
            transmission: .init(factor: 0, indexOfRefraction: 1)
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(material)
        )
        let shadow = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 32, directionToLight: SIMD3(0, 0, 1), halfExtent: 8, intensity: 0
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        var graph = SceneGraph(roots: [node], camera: camera)
        let baseline = try center(fixture, graph: graph, shadow: shadow)
        #expect(baseline.x > 0.2 && baseline.z > 0.4)
        graph.lightProbes = [probe]
        let combined = try center(fixture, graph: graph, shadow: shadow)
        graph.environmentIntensity = 0
        graph.environmentRotation = 1
        let probeOnly = try center(fixture, graph: graph, shadow: shadow)
        let expected = try simd_max(probe.irradiance(at: SIMD3(0, 0, 1)), .zero) / Float.pi
        #expect(simd_distance(probeOnly, SIMD4(expected, 1)) < 0.0015)
        #expect(simd_distance(combined, baseline + probeOnly - SIMD4(0, 0, 0, 1)) < 0.002)
        graph.environmentIntensity = 1
        graph.environmentRotation = 0
        var off = probe
        try off.setIntensity(0)
        graph.lightProbes = [off]
        #expect(try center(fixture, graph: graph, shadow: shadow) == baseline)
    }

    @Test("Probes leave unlit color and purely metallic surfaces unchanged")
    func diffuseOnly() throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        var coefficients = [SIMD3<Float>](repeating: .zero, count: 9)
        coefficients[0] = SIMD3(repeating: 2)
        let probe = try LightProbe(coefficients: coefficients)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 16)
        let metallic = try PhysicallyBasedMaterial(metallic: 1, roughness: 1, lightIntensity: 0, ambientIntensity: 0)
        let basic = try ClassicMaterial(model: .basic, color: SIMD3(0.25, 0.5, 0.75))
        for (material, expected): (MeshMaterial, SIMD4<Float>) in [
            (.physicallyBased(metallic), SIMD4(0, 0, 0, 1)), (.classic(basic), SIMD4(0.25, 0.5, 0.75, 1)),
        ] {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
                material: material
            )
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
            var graph = SceneGraph(roots: [node], camera: camera)
            graph.lightProbes = [probe]
            #expect(try center(fixture, graph: graph) == expected)
        }
    }

    private func projectedProbe(device: any MTLDevice, queue: any MTLCommandQueue) async throws -> LightProbe {
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float, size: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let cube = try #require(device.makeTexture(descriptor: descriptor))
        for face in 0..<6 {
            var color: SIMD4<Float16> = face == 0 ? SIMD4(3, 0, 0, 1)
                : (face == 1 ? SIMD4(0, 0, 2, 1) : SIMD4(0, 0, 0, 1))
            withUnsafeBytes(of: &color) {
                cube.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, slice: face,
                             withBytes: $0.baseAddress!, bytesPerRow: $0.count, bytesPerImage: $0.count)
            }
        }
        return try await LightProbe(cube: cube, commandQueue: queue)
    }

    private func center(
        _ fixture: RasterLightingFixture, graph: SceneGraph, shadow: DirectionalShadowMap? = nil
    ) throws -> SIMD4<Float> {
        let color = try fixture.render(graph, shadowMap: shadow).color
        return try fixture.pixels(of: color)[(color.height / 2) * color.width + color.width / 2]
    }
}
