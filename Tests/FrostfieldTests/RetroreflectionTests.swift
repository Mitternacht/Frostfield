import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RetroreflectionTests {
    private let fixture: PathTestFixture
    private let tilt = simd_quatf(angle: 0.62, axis: SIMD3(0, 1, 0))

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Retroreflectivity mixes the ordinary and source-facing GGX responses",
          arguments: [false, true])
    func directLight(sceneLight: Bool) throws {
        let albedo = SIMD3<Float>(0.31, 0.5, 0.76)
        let normal = tilt.act(SIMD3(0, 0, 1))
        let view = SIMD3<Float>(0, 0, 1)
        let reflected = 2 * simd_dot(normal, view) * normal - view
        let roughness: Float = 0.4

        func pixel(_ direction: SIMD3<Float>, strength: Float) throws -> SIMD4<Float> {
            let material = try PhysicallyBasedMaterial(
                roughness: roughness, directionToLight: direction, lightIntensity: sceneLight ? 0 : 1,
                ambientIntensity: 0, baseColorFactor: SIMD4(albedo, 1), alphaMode: .opaque,
                retroreflectivity: strength
            )
            let graph = try makeScene(material, direction: sceneLight ? direction : nil)
            return try center(graph)
        }

        for light in [view, reflected] {
            let ordinary = reference(albedo, light: light, view: view, normal: normal, roughness: roughness)
            let retro = reference(albedo, light: light, view: reflected, normal: normal, roughness: roughness)
            #expect(close(try pixel(light, strength: 0), SIMD4(ordinary, 1)))
            #expect(close(try pixel(light, strength: 0.25), SIMD4(ordinary * 0.75 + retro * 0.25, 1)))
            #expect(close(try pixel(light, strength: 1), SIMD4(retro, 1)))
        }
        let returnToSource = try pixel(view, strength: 1)
        let returnToMirror = try pixel(reflected, strength: 1)
        #expect(returnToSource.x > returnToMirror.x * 2)
    }

    @Test("Filtered environment specular blends from the mirror ray toward the viewer")
    func imageBasedLight() throws {
        let gpu = try PostProcessFixture()
        let panorama = try gpu.color(width: 128, height: 64) { x, _ in
            x < 86 ? SIMD4(2, 0, 0, 1) : SIMD4(0, 0, 1, 1)
        }
        let lighting = try EnvironmentLighting(
            queue: fixture.queue, equirectangular: panorama,
            settings: .init(cubeSize: 32, diffuseSize: 8, brdfSize: 16, sampleCount: 32)
        )
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting)
        func pixel(_ strength: Float) throws -> SIMD4<Float> {
            try center(makeScene(.init(
                metallic: 1, roughness: 0.15, lightIntensity: 0, ambientIntensity: 0, environment: environment,
                retroreflectivity: strength
            )))
        }
        let ordinary = try pixel(0)
        let retro = try pixel(1)
        let mixed = try pixel(0.4)
        #expect(ordinary.x > 1.5 && ordinary.z < 0.05)
        #expect(retro.z > 0.75 && retro.x < 0.05)
        #expect(close(mixed, ordinary * 0.6 + retro * 0.4))
    }

    @Test("Archive v23 and live factor copies preserve retroreflectivity")
    func archive() throws {
        for invalid: Float in [-0.01, 1.01, .nan, .infinity] {
            #expect(throws: PhysicallyBasedMaterial.Failure.invalidRetroreflectivity) {
                try PhysicallyBasedMaterial(retroreflectivity: invalid)
            }
        }
        let source = try PhysicallyBasedMaterial(roughness: 0.4, retroreflectivity: 0.75).withLiveFactors()
        let factors = try #require(source.liveFactors)
        try factors.update { $0.baseColorFactor = SIMD4(0.25, 0.5, 0.75, 1) }
        let graph = try makeScene(source)
        let archived = try SceneArchive(capturing: graph)
        let manifest = try #require(try JSONSerialization.jsonObject(with: archived.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 23)
        #expect(try center(archived.load(device: fixture.device).graph) == center(graph))
        var older = manifest
        older["version"] = 22
        #expect(throws: SceneArchive.Failure.decoding("PBR retroreflectivity requires archive version 23")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: older), binary: archived.binary)
        }
        let zero = try SceneArchive(capturing: makeScene(.init()))
        let zeroManifest = try #require(try JSONSerialization.jsonObject(with: zero.manifest) as? [String: Any])
        #expect(zeroManifest["version"] as? Int == 1)

        let exported = try GLTFExporter().export(graph, includesCamera: false)
        #expect(exported.warnings.contains(.materialLayersDropped(material: 0, layers: ["retroreflection"])))
    }

    private func makeScene(_ material: PhysicallyBasedMaterial, direction: SIMD3<Float>? = nil) throws -> SceneGraph {
        let plane = try MeshGeometry.plane(width: 4, height: 4)
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(material)
        )
        var roots = [SceneNode(transform: .init(translation: SIMD3(0, 0, -2), rotation: tilt), mesh: renderer)]
        if let direction {
            let rotation = simd_quatf(from: SIMD3(0, 0, -1), to: -direction)
            roots.append(SceneNode(transform: .init(rotation: rotation), light: try .init(kind: .directional)))
        }
        return SceneGraph(roots: roots, camera: fixture.camera)
    }

    private func center(_ graph: SceneGraph) throws -> SIMD4<Float> {
        try fixture.rasterize(graph, width: 9, height: 9)[40]
    }

    private func close(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Bool {
        simd_reduce_max(abs(a - b)) < 0.008
    }

    private func reference(
        _ albedo: SIMD3<Float>, light: SIMD3<Float>, view: SIMD3<Float>, normal: SIMD3<Float>, roughness: Float
    ) -> SIMD3<Float> {
        let halfway = simd_normalize(view + light)
        let noH = simd_dot(normal, halfway)
        let noV = simd_dot(normal, view)
        let noL = simd_dot(normal, light)
        let voH = simd_dot(view, halfway)
        let alpha2 = pow(roughness, 4)
        let denominator = 1 - noH * noH + noH * noH * alpha2
        let distribution = alpha2 / (.pi * denominator * denominator)
        let visibility = 0.5 / (noL * sqrt(noV * noV * (1 - alpha2) + alpha2)
                              + noV * sqrt(noL * noL * (1 - alpha2) + alpha2))
        let fresnel = 0.04 + 0.96 * pow(1 - voH, 5)
        return (albedo * ((1 - fresnel) / .pi) + fresnel * distribution * visibility) * noL
    }
}
