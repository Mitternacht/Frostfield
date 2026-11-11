import Foundation
import Frostfield
import Metal
import Testing
import simd

struct AmbientLightTests {
    @Test("Ambient lights default to linear white and reject invalid authored values")
    func validation() throws {
        let light = try AmbientLight()
        #expect(light.color == SIMD3(repeating: 1))
        #expect(light.intensity == 1)
        #expect(throws: AmbientLight.Failure.color) { try AmbientLight(color: SIMD3(1, .nan, 0)) }
        #expect(throws: AmbientLight.Failure.color) { try AmbientLight(color: SIMD3(1.01, 0, 0)) }
        #expect(throws: AmbientLight.Failure.intensity) { try AmbientLight(intensity: -1) }
        #expect(throws: AmbientLight.Failure.intensity) { try AmbientLight(intensity: .infinity) }
    }

    @Test("Ambient scene nodes sum across lit materials independently of normals, visibility and layers",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func rasterMaterials() throws {
        let fixture = try GeometryRenderFixture(side: 32)
        let first = try AmbientLight(color: SIMD3(0.5, 0.25, 0.125), intensity: 2)
        let second = try AmbientLight(color: SIMD3(0.25, 0.5, 1), intensity: 0.5)
        let excluded = try AmbientLight(intensity: 8)
        let hidden = SceneNode(children: [SceneNode(ambientLight: excluded)], isVisible: false)
        let layer = SceneNode(ambientLight: excluded, layers: .init(layer: 1))
        let moved = SceneNode(
            transform: .init(translation: SIMD3(400, -80, 200), rotation: simd_quatf(angle: 1, axis: SIMD3(0, 1, 0))),
            ambientLight: first
        )
        let lights = [moved, SceneNode(ambientLight: second), hidden, layer]
        let expected = SIMD4((first.color * first.intensity + second.color * second.intensity) / Float.pi, 0)
        let pbr = try PhysicallyBasedMaterial(roughness: 1, lightIntensity: 0, ambientIntensity: 0)
        let metal = try PhysicallyBasedMaterial(metallic: 1, lightIntensity: 0, ambientIntensity: 0)
        let shadow = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 16, directionToLight: SIMD3(0, 0, 1), halfExtent: 4, intensity: 0
        )
        let materials: [(MeshMaterial, SIMD4<Float>, DirectionalShadowMap?, Float)] = [
            (.lambert, .zero, nil, 1),
            (.shadowedLambert, SIMD4(repeating: 1), shadow, 1),
            (.classic(try ClassicMaterial(model: .lambert)), SIMD4(repeating: 1), nil, 1),
            (.classic(try ClassicMaterial(model: .phong())), SIMD4(repeating: 1), nil, 1),
            (.classic(try ClassicMaterial(model: .toon())), SIMD4(repeating: 1), nil, 1),
            // The default dielectric retains 4% of incoming light for Fresnel reflection.
            (.physicallyBased(pbr), SIMD4(repeating: 1), nil, 0.96),
            (.physicallyBased(metal), SIMD4(repeating: 1), nil, 0),
            (.classic(try ClassicMaterial(model: .basic)), SIMD4(repeating: 1), nil, 0),
            (.unlit, SIMD4(repeating: 1), nil, 0),
        ]
        for (material, parameters, shadow, diffuseResponse) in materials {
            let mesh = try fixture.unlit(GeometryRenderFixture.square(1), material: material)
            for angle: Float in [0, .pi / 4] {
                let receiver = SceneNode(
                    transform: .init(
                        translation: SIMD3(0, 0, -2), rotation: simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))
                    ),
                    mesh: mesh, materialParameters: parameters, castsShadow: false
                )
                var graph = SceneGraph(roots: [receiver], camera: fixture.camera)
                let baseline = try center(fixture, graph: graph, shadow: shadow)
                #expect(baseline.w == 1)
                graph.roots.append(contentsOf: lights)
                let lit = try center(fixture, graph: graph, shadow: shadow)
                #expect(isNearly(lit - baseline, expected * diffuseResponse, tolerance: 0.0015))
            }
        }
    }

    @Test("Ambient adds to environment and hemisphere light without cancellation by signed probes",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func independentIrradiance() throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        let environment = try fixture.environment { _, _ in SIMD3(0.25, 0.125, 0.5) }
        let material = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, environment: environment,
            specular: .init(factor: 0)
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: GeometryRenderFixture.square(1), material: .physicallyBased(material)
        )
        let receiver = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 12)
        var graph = SceneGraph(roots: [receiver], camera: camera)
        let index = 16 * 32 + 16
        let environmentOnly = try fixture.pixels(of: fixture.render(graph).color)[index]
        #expect(environmentOnly.z > 0.4)
        let sky = SIMD3<Float>(0.5, 0.25, 0.125)
        graph.lightProbes = [try LightProbe.hemisphere(skyColor: sky, groundColor: .zero, direction: SIMD3(0, 0, 1))]
        let environmentAndProbe = try fixture.pixels(of: fixture.render(graph).color)[index]
        #expect(isNearly(environmentAndProbe - environmentOnly, SIMD4(sky / Float.pi, 0), tolerance: 0.001))

        let ambient = try AmbientLight(color: SIMD3(0.25, 0.5, 1), intensity: 0.5)
        graph.roots.append(SceneNode(ambientLight: ambient))
        let expected = SIMD4(ambient.color * ambient.intensity / Float.pi, 0)
        let combined = try fixture.pixels(of: fixture.render(graph).color)[index]
        #expect(isNearly(combined - environmentAndProbe, expected, tolerance: 0.001))

        var coefficients = [SIMD3<Float>](repeating: .zero, count: LightProbe.coefficientCount)
        coefficients[0] = SIMD3(repeating: -1)
        graph.lightProbes = [try LightProbe(coefficients: coefficients)]
        let signed = try fixture.pixels(of: fixture.render(graph).color)[index]
        #expect(isNearly(signed - environmentOnly, expected, tolerance: 0.001))
    }

    @Test("Version 12 archives preserve ambient node metadata and rendered pixels",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func archive() throws {
        let fixture = try GeometryRenderFixture(side: 32)
        let material = try ClassicMaterial(model: .lambert)
        let mesh = try fixture.unlit(GeometryRenderFixture.square(1), material: .classic(material))
        let receiver = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        var graph = SceneGraph(roots: [receiver], camera: fixture.camera)
        let legacy = try SceneArchive(capturing: graph)
        #expect(try manifest(legacy)["version"] as? Int == 1)
        #expect(try legacy.load(device: fixture.device).graph.roots.allSatisfy { $0.ambientLight == nil })

        let warm = try AmbientLight(color: SIMD3(1, 0.5, 0.25), intensity: 2)
        let cool = try AmbientLight(color: SIMD3(0.25, 0.5, 1), intensity: 3)
        let group = SceneNode(children: [
            SceneNode(ambientLight: warm),
            SceneNode(ambientLight: cool, isVisible: false, layers: .init(layer: 2)),
        ])
        graph.roots.append(group)
        let expected = try center(fixture, graph: graph)
        #expect(isNearly(expected, SIMD4(warm.color * warm.intensity / Float.pi, 1), tolerance: 0.001))
        let archive = try SceneArchive(capturing: graph)
        var document = try manifest(archive)
        #expect(document["version"] as? Int == 12)
        let restored = try archive.load(device: fixture.device).graph
        let children = try #require(restored.roots.last?.children)
        #expect(children.map(\.ambientLight) == [warm, cool])
        #expect(children[1].layers == SceneLayers(layer: 2))
        #expect(!children[1].isVisible)
        #expect(try center(fixture, graph: restored) == expected)
        #expect(try SceneArchive(capturing: restored) == archive)

        document["version"] = 11
        #expect(throws: SceneArchive.Failure.decoding("ambient lights require archive version 12")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: document), binary: archive.binary)
        }
    }

    @Test("Unrepresentable ambient sums fail before a draw is encoded",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func aggregateValidation() throws {
        let fixture = try GeometryRenderFixture()
        let bright = try AmbientLight(intensity: .greatestFiniteMagnitude / 32)
        let graph = SceneGraph(roots: (0..<4).map { _ in SceneNode(ambientLight: bright) }, camera: fixture.camera)
        #expect(throws: AmbientLight.Failure.irradianceRange) { try fixture.render(graph) }
    }

    private func center(
        _ fixture: GeometryRenderFixture, graph: SceneGraph, shadow: DirectionalShadowMap? = nil
    ) throws -> SIMD4<Float> {
        let rendered = try fixture.render(graph, shadowMap: shadow)
        return fixture.pixel(rendered.pixels, x: fixture.side / 2, y: fixture.side / 2)
    }

    private func manifest(_ archive: SceneArchive) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
    }
}
