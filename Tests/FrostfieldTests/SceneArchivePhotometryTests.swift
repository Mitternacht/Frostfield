import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneArchivePhotometryTests {
    private func lightResources() throws -> (IESProfile, LightCookie) {
        let profile = try IESProfile(string: IESFixture.quadrantDownlight())
        let cookie = try LightCookie(width: 2, height: 2, transmittance: [
            SIMD3(1, 0.125, 0.25), SIMD3(0.25, 1, 0.125), SIMD3(0.125, 0.25, 1), SIMD3(0.5, 0.75, 0.25),
        ])
        return (profile, cookie)
    }

    @Test("Shared IES profiles and cookies keep the same raster and path-traced lighting after load")
    func renderingAndAnimation() throws {
        let fixture = try PathCameraFixture()
        let (profile, cookie) = try lightResources()
        let lamp = LightRig.Light(
            id: "projector", kind: .spot(innerAngle: 0.4, outerAngle: 0.6), position: SIMD3(0.2, 2, 0.1),
            target: SIMD3(0.5, 0, -0.2), emission: .power(5), castsShadow: false, iesProfile: profile, cookie: cookie
        )
        let rig = try LightRig(lights: [lamp])
        let lightNode = try #require(rig.sceneNodes().first)
        let floor = try fixture.lambert(.plane(width: 10, height: 10), albedo: 0.8)
        let floorNode = SceneNode(transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: floor)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
        let cameraTransform = SceneTransform(
            translation: SIMD3(0, 4, 0), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        )
        let graph = SceneGraph(roots: [floorNode, lightNode], camera: camera, cameraTransform: cameraTransform)
        let clip = try AnimationClip(tracks: [
            .number(.node(lightNode.id, .lightIntensity), times: [0, 1], values: [1, 3]),
            .number(.rigLight(id: lamp.id, .power), times: [0, 1], values: [5, 9]),
        ])
        let archive = try SceneArchive(capturing: graph, animations: [clip], lightRig: rig)
        let manifest = try JSONDecoder().decode(ArchiveManifest.self, from: archive.manifest)
        #expect(manifest.version == 3)
        #expect(manifest.photometry?.profiles.count == 1)
        #expect(manifest.photometry?.cookies.count == 1)

        let decoded = try archive.load(device: fixture.device)
        let decodedRig = try #require(decoded.lightRig)
        let decodedNodeLight = try #require(decoded.graph.roots[1].light)
        let decodedRigLight = try #require(decodedRig.lights.first)
        #expect(decodedNodeLight == lightNode.light)
        #expect(decodedRigLight == lamp)
        #expect(decodedNodeLight.iesProfile?.storage === decodedRigLight.iesProfile?.storage)
        #expect(decodedNodeLight.cookie?.storage === decodedRigLight.cookie?.storage)
        #expect(try decodedRigLight.intensity() == lamp.intensity())
        #expect(try SceneArchive(capturing: decoded.graph, animations: decoded.animations, lightRig: decodedRig) == archive)

        let size = 32
        func raster(_ scene: SceneGraph) throws -> [SIMD4<Float>] {
            try fixture.base.readback(fixture.rasterize(scene, width: size, height: size).color)
        }
        func traced(_ graph: SceneGraph, rig: LightRig) throws -> [SIMD4<Float>] {
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
            let tracer = try PathTracer(queue: fixture.queue, scene: scene)
            return try fixture.trace(
                tracer, model: .init(projection: .perspective(verticalFieldOfView: camera.verticalFieldOfView),
                                     near: camera.near, far: camera.far),
                transform: graph.cameraTransform, width: size, height: size, samples: 1, lighting: rig.pathLighting()
            )
        }
        let originalRaster = try raster(graph)
        let originalPath = try traced(graph, rig: rig)
        #expect(originalRaster.contains { $0.x > 0.02 && abs($0.x - $0.y) > 0.01 })
        #expect(originalPath.contains { $0.x > 0.02 && abs($0.x - $0.y) > 0.01 })
        #expect(try raster(decoded.graph) == originalRaster)
        #expect(try traced(decoded.graph, rig: decodedRig) == originalPath)

        let restored = try #require(decoded.animations.first)
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: restored)
        try mixer.play(action)
        _ = try mixer.update(deltaTime: 0.5)
        let posedNodeLight = try #require(mixer.apply(to: decoded.graph).roots[1].light)
        let posedRigLight = try #require(mixer.apply(to: decodedRig).lights.first)
        #expect(posedNodeLight.intensity == 2)
        #expect(posedRigLight.emission == .power(7))
        #expect(posedNodeLight.iesProfile?.storage === decodedRigLight.iesProfile?.storage)
        #expect(posedNodeLight.cookie?.storage === decodedRigLight.cookie?.storage)
        #expect(posedRigLight.iesProfile?.storage === decodedRigLight.iesProfile?.storage)
        #expect(posedRigLight.cookie?.storage === decodedRigLight.cookie?.storage)
    }

    @Test("Photometric archive rejects missing references, malformed sidecar data and schema downgrades")
    func malformed() throws {
        let fixture = try PathCameraFixture()
        let (profile, cookie) = try lightResources()
        let light = try PunctualLight(kind: .spot(), iesProfile: profile, cookie: cookie)
        let node = SceneNode(light: light)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let graph = SceneGraph(roots: [node], camera: camera)
        let rig = try LightRig(lights: [
            .init(id: "spot", kind: .spot(innerAngle: 0, outerAngle: 0.5), position: SIMD3(0, 1, 0),
                  target: .zero, emission: .intensity(1), cookie: cookie),
        ])
        let archive = try SceneArchive(capturing: graph, lightRig: rig)
        let manifest = try JSONDecoder().decode(ArchiveManifest.self, from: archive.manifest)
        let resources = try #require(manifest.photometry)
        func modified(_ change: (inout ArchiveManifest) -> Void, binary: Data? = nil) throws -> SceneArchive {
            var altered = manifest
            change(&altered)
            return try SceneArchive(manifest: JSONEncoder().encode(altered), binary: binary ?? archive.binary)
        }
        #expect(throws: SceneArchive.Failure.invalidReference("IES profile 99")) {
            try modified { $0.scene.nodes[0].photometry?.profile = 99 }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidReference("node light \(node.id)")) {
            try modified { $0.scene.nodes[0].light = nil }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidReference("rig light missing")) {
            try modified { $0.photometry?.rigLights[0].id = "missing" }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidPhotometry("duplicate resources for rig light spot")) {
            try modified { $0.photometry?.rigLights.append(resources.rigLights[0]) }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidPhotometry("light cookie 0 dimensions")) {
            try modified { $0.photometry?.cookies[0].width = .max }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidView(99)) {
            try modified { $0.photometry?.cookies[0].texels = 99 }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidPhotometry("photometric resources require archive version 3")) {
            try modified { $0.version = 2 }
        }
        var corruptProfile = archive.binary
        corruptProfile[manifest.views[resources.profiles[0]].offset] = 255
        #expect(throws: SceneArchive.Failure.invalidPhotometry("IES profile 0 is not UTF-8")) {
            try modified({ _ in }, binary: corruptProfile).load(device: fixture.device)
        }
        var corruptCookie = archive.binary
        let badTexel = withUnsafeBytes(of: Float(2).bitPattern.littleEndian) { Data($0) }
        let offset = manifest.views[resources.cookies[0].texels].offset
        corruptCookie.replaceSubrange(offset..<(offset + badTexel.count), with: badTexel)
        #expect(throws: SceneArchive.Failure.invalidPhotometry("light cookie 0: \(LightCookie.Failure.values.localizedDescription)")) {
            try modified({ _ in }, binary: corruptCookie).load(device: fixture.device)
        }
    }
}
