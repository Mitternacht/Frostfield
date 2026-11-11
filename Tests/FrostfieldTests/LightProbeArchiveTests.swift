import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightProbeArchiveTests {
    @Test("Archive version 6 retains probe order, signed coefficients, intensity and raster pixels")
    func roundTrip() throws {
        let fixture = try RasterLightingFixture(width: 16, height: 16)
        let material = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0,
            transmission: .init(factor: 0, indexOfRefraction: 1)
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(material)
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 12)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        var graph = SceneGraph(roots: [node], camera: camera)
        let legacy = try SceneArchive(capturing: graph)
        let legacyDocument = try manifest(legacy)
        #expect(legacyDocument["version"] as? Int == 1)
        #expect(try legacy.load(device: fixture.device).graph.lightProbes.isEmpty)

        var coefficients = [SIMD3<Float>](repeating: .zero, count: 9)
        coefficients[0] = SIMD3(0.5, 1, 0.25)
        coefficients[2] = SIMD3(0.125, -0.25, 0.25)
        let first = try LightProbe(coefficients: coefficients, intensity: 0.5)
        coefficients[2] = SIMD3(-0.125, 0.5, 0)
        let second = try LightProbe(coefficients: coefficients, intensity: 2)
        graph.lightProbes = [first, second]
        let expected = try fixture.pixels(of: fixture.render(graph).color)
        #expect(expected.contains { $0.x > 0.2 && $0.y > 0.5 })
        let archive = try SceneArchive(capturing: graph)
        #expect(try manifest(archive)["version"] as? Int == 6)
        let restored = try archive.load(device: fixture.device).graph
        #expect(restored.lightProbes == [first, second])
        #expect(try fixture.pixels(of: fixture.render(restored).color) == expected)
        #expect(try SceneArchive(capturing: restored) == archive)
    }

    @Test("Probe archives reject a legacy header, malformed coefficients and out-of-range sums")
    func validation() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 12)
        var graph = SceneGraph(roots: [], camera: camera)
        var coefficients = [SIMD3<Float>](repeating: .zero, count: 9)
        coefficients[0].x = 1
        graph.lightProbes = [try LightProbe(coefficients: coefficients)]
        let archive = try SceneArchive(capturing: graph)
        var document = try manifest(archive)
        document["version"] = 5
        #expect(throws: SceneArchive.Failure.invalidLightProbe("light probes require archive version 6")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: document), binary: archive.binary)
        }
        document["version"] = 6
        var scene = try #require(document["scene"] as? [String: Any])
        var probes = try #require(scene["lightProbes"] as? [[String: Any]])
        probes[0]["coefficients"] = [[1, 0, 0]]
        scene["lightProbes"] = probes
        document["scene"] = scene
        #expect(throws: SceneArchive.Failure.invalidLightProbe(LightProbe.Failure.coefficients.localizedDescription)) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: document), binary: archive.binary)
        }

        coefficients[0].x = Float.greatestFiniteMagnitude / 32
        let bright = try LightProbe(coefficients: coefficients)
        graph.lightProbes = [bright, bright, bright]
        #expect(throws: SceneArchive.Failure.invalidLightProbe(LightProbe.Failure.coefficientRange.localizedDescription)) {
            try SceneArchive(capturing: graph)
        }
    }

    private func manifest(_ archive: SceneArchive) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
    }
}
