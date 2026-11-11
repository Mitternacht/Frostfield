import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneArchiveAnimationTests {
    @Test("Restored typed bindings drive the graph, shared live resources and a light rig")
    func applyRestoredClip() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try GeometryRenderFixture.square(0.25)
        let material = try PhysicallyBasedMaterial(
            roughness: 0.25, clearcoat: .init(factor: 0.5, roughness: 0.25)
        ).withLiveFactors()
        let live = try #require(material.liveFactors)
        try live.update {
            $0.roughness = 0.75
            $0.clearcoatFactor = 0
        }

        let surface = try SurfaceShaderGraph(
            nodes: [.uniform(0)], uniforms: [ShaderUniform("tint", SIMD3<Float>(0.25, 0.25, 0.25))],
            outputs: .init(baseColor: 0)
        )
        let uniforms = try ShaderUniforms(graph: surface)
        try uniforms.set("tint", to: SIMD3<Float>(0.5, 0.25, 0.75))
        let graphMaterial = try PhysicallyBasedMaterial(surfaceGraph: surface, surfaceUniforms: uniforms)
        func node(_ pbr: PhysicallyBasedMaterial, morph: MeshMorphTargets? = nil) throws -> SceneNode {
            let renderer = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(pbr), morphTargets: morph
            )
            return SceneNode(mesh: renderer)
        }
        let morph = try MeshMorphTargets(vertexCount: 4, targets: [
            .init(positionOffsets: Array(repeating: SIMD3<Float>(0, 0.125, 0), count: 4)),
            .init(positionOffsets: Array(repeating: SIMD3<Float>(0.125, 0, 0), count: 4)),
        ])
        let hero = try node(material, morph: morph)
        let lamp = SceneNode(light: try PunctualLight(kind: .spot(range: 6, innerConeAngle: 0.125), intensity: 2))
        let graph = SceneGraph(
            roots: [hero, try node(material), try node(graphMaterial), try node(graphMaterial), lamp], camera: fixture.camera
        )
        let rig = try LightRig(lights: [
            .init(id: "key", kind: .spot(innerAngle: 0.125, outerAngle: 0.5, range: 4),
                  position: SIMD3(1, 2, 3), target: .zero, emission: .intensity(2)),
        ])
        let times: [Float] = [0, 1]
        let gltfCamera = AnimationProperty.gltfCamera(3, .verticalFieldOfView)
        let clip = try AnimationClip(name: "Bindings", tracks: [
            .vector(.node(hero.id, .translation), times: times, values: [.zero, SIMD3(2, 4, -6)]),
            .boolean(.node(hero.id, .castsShadow), times: [0, 0.25], values: [true, false]),
            .init(property: .node(hero.id, .morphWeights), times: times, values: [.numbers([0, 0]), .numbers([0.5, 1])]),
            .number(.node(hero.id, .morphWeight(0)), times: times, values: [0.5, 1]),
            .init(property: .node(hero.id, .materialParameters), times: times, values: [.vector4(.zero), .vector4(SIMD4(2, 4, 6, 8))]),
            .number(.node(lamp.id, .lightIntensity), times: times, values: [2, 6]),
            .number(.node(lamp.id, .lightRange), times: times, values: [6, 8]),
            .number(.node(lamp.id, .spotInnerConeAngle), times: times, values: [0.125, 0.375]),
            .number(.camera(.verticalFieldOfView), times: times, values: [0.5, 1]),
            .vector(.camera(.translation), times: times, values: [.zero, SIMD3(0, 2, 4)]),
            .number(.environment(.intensity), times: times, values: [1, 3]),
            .number(.material(live, .roughness), times: times, values: [0.25, 0.75]),
            .color(.uniform(uniforms, name: "tint"), times: times, values: [.zero, SIMD3(0.5, 0.75, 1)]),
            .number(.rigLight(id: "key", .intensity), times: times, values: [2, 10]),
            .number(gltfCamera, times: times, values: [0.5, 1]),
            .string(.custom("shot"), times: [0, 0.25], values: ["wide", "détail"]),
        ])
        let archive = try SceneArchive(capturing: graph, animations: [clip], lightRig: rig)
        let manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 2)
        let decoded = try archive.load(device: fixture.device)
        let restored = try #require(decoded.animations.first)
        let restoredRig = try #require(decoded.lightRig)
        #expect(restored.name == "Bindings")
        #expect(try SceneArchive(capturing: decoded.graph, animations: decoded.animations, lightRig: restoredRig) == archive)

        let factors = try #require(decoded.graph.roots[0].mesh?.pathTracingMaterial?.liveFactors)
        let tint = try #require(decoded.graph.roots[2].mesh?.pathTracingMaterial?.surfaceUniforms)
        #expect(factors !== live)
        #expect(tint !== uniforms)
        #expect(decoded.graph.roots[1].mesh?.pathTracingMaterial?.liveFactors === factors)
        #expect(decoded.graph.roots[3].mesh?.pathTracingMaterial?.surfaceUniforms === tint)
        #expect(restored.properties.contains(.material(factors, .roughness)))
        #expect(restored.properties.contains(.uniform(tint, name: "tint")))
        #expect(factors.rest.roughness == 0.25)
        #expect(factors.current.roughness == 0.75)
        #expect(factors.rest.clearcoatFactor == 0.5)
        #expect(factors.current.clearcoatFactor == 0)
        #expect(tint.value(named: "tint") == SIMD4(0.5, 0.25, 0.75, 0))

        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: restored)
        try mixer.play(action)
        _ = try mixer.update(deltaTime: 0.5)
        let posed = try mixer.apply(to: decoded.graph)
        try mixer.applyLiveProperties()
        let posedRig = try mixer.apply(to: restoredRig)
        #expect(posed.roots[0].transform.translation == SIMD3(1, 2, -3))
        #expect(!posed.roots[0].castsShadow)
        #expect(posed.roots[0].morphWeights == [0.75, 0.5])
        #expect(posed.roots[0].materialParameters == SIMD4(1, 2, 3, 4))
        #expect(posed.roots[4].light?.intensity == 4)
        #expect(posed.roots[4].light?.kind == .spot(range: 7, innerConeAngle: 0.25))
        #expect(posed.camera.verticalFieldOfView == 0.75)
        #expect(posed.cameraTransform.translation == SIMD3(0, 1, 2))
        #expect(posed.environmentIntensity == 2)
        #expect(try posedRig.lights[0].intensity() == 6)
        #expect(factors.current.roughness == 0.5)
        #expect(factors.current.clearcoatFactor == 0)
        #expect(tint.value(named: "tint") == SIMD4(0.25, 0.375, 0.5, 0))
        #expect(try mixer.value(of: gltfCamera, rest: .number(0.25)) == .number(0.75))
        #expect(try mixer.value(of: .custom("shot"), rest: .string("")) == .string("détail"))

        try mixer.stop(action)
        try mixer.applyLiveProperties()
        #expect(factors.current.roughness == 0.25)
        #expect(tint.value(named: "tint") == SIMD4(0.25, 0.25, 0.25, 0))
        factors.reset()
        #expect(factors.current.clearcoatFactor == 0.5)
    }

    @Test("Additive references and non-unit split spline keys retain their exact samples")
    func splitAdditiveClip() throws {
        let fixture = try GeometryRenderFixture()
        let node = SceneNode()
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let zeroTangent = AnimationValue.quaternion(simd_quatf(vector: .zero))
        let rotation = AnimationProperty.node(node.id, .rotation)
        let exposure = AnimationProperty.custom("exposure")
        let caption = AnimationProperty.custom("caption")
        let source = try AnimationClip(tracks: [
            .init(property: rotation, times: [0, 1], values: [
                zeroTangent, .quaternion(simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))), zeroTangent,
                zeroTangent, .quaternion(simd_quatf(angle: 2, axis: SIMD3(0, 1, 0))), zeroTangent,
            ], interpolation: .cubicSpline),
            .number(exposure, times: [0, 1], values: [2, 4], interpolation: .smooth),
            .string(caption, times: [0, 0.5], values: ["first", "second"]),
        ])
        let clipped = try source.makingAdditive(referenceTime: 0.125)
            .subclip(startFrame: 0.25, endFrame: 0.875, framesPerSecond: 1)
        let splitKey = try #require(clipped.tracks[0].values[1].quaternionValue)
        #expect(abs(simd_length(splitKey.vector) - 1) > 0.001)
        let archive = try SceneArchive(capturing: graph, animations: [clipped])
        let restored = try #require(archive.load(device: fixture.device).animations.first)
        #expect(restored.blendMode == .additive)
        #expect(restored.tracks.map(\.values) == clipped.tracks.map(\.values))
        #expect(restored.trackReference == clipped.trackReference)
        for time: Float in [0, 0.125, 0.25, 0.5, 0.625] {
            #expect(try restored.sampleProperties(at: time) == clipped.sampleProperties(at: time))
        }
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: restored)
        try mixer.play(action)
        _ = try mixer.update(deltaTime: 0.25)
        #expect(try mixer.value(of: exposure, rest: .number(10)) == .number(10.75))
        #expect(try mixer.value(of: caption, rest: .string("rest")) == .string("second"))
    }

    @Test("Detached live targets are shared across restored clips")
    func detachedBindings() throws {
        let fixture = try GeometryRenderFixture()
        let graph = SceneGraph(roots: [], camera: fixture.camera)
        let original = try ShaderUniforms([ShaderUniform("offset", SIMD2<Float>(0.5, 1))])
        let material = try PhysicallyBasedMaterial(roughness: 0.25).withLiveFactors()
        let live = try #require(material.liveFactors)
        let clips = try (1...2).map { index in
            try AnimationClip(tracks: [
                .init(property: .uniform(original, name: "offset"), times: [0],
                      values: [.vector2(SIMD2(Float(index), 3))]),
                .number(.material(live, .roughness), times: [0], values: [Float(index) * 0.25]),
            ])
        }
        let archive = try SceneArchive(capturing: graph, animations: clips)
        let restored = try archive.load(device: fixture.device).animations
        #expect(restored[0].properties == restored[1].properties)
        guard case .uniform(let uniforms, _) = restored[1].properties[0],
              case .material(let factors, _) = restored[1].properties[1] else {
            Issue.record("Typed bindings did not restore their live resources")
            return
        }
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: restored[1])
        try mixer.play(action)
        try mixer.applyLiveProperties()
        #expect(uniforms.value(named: "offset") == SIMD4(2, 3, 0, 0))
        #expect(factors.current.roughness == 0.5)
    }

    @Test("Malformed references, keys and schema versions report staged archive errors")
    func invalidBindings() throws {
        let fixture = try GeometryRenderFixture()
        let node = SceneNode()
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let uniformSet = try ShaderUniforms([ShaderUniform("gain", 1.0 as Float)])
        let clip = try AnimationClip(tracks: [
            .boolean(.node(node.id, .castsShadow), times: [0, 1], values: [true, false]),
            .number(.uniform(uniformSet, name: "gain"), times: [0, 1], values: [1, 2]),
        ])
        let archive = try SceneArchive(capturing: graph, animations: [clip])
        let manifest = try JSONDecoder().decode(ArchiveManifest.self, from: archive.manifest)
        func modified(_ change: (inout ArchiveManifest) -> Void, binary: Data? = nil) throws -> SceneArchive {
            var altered = manifest
            change(&altered)
            return try SceneArchive(manifest: JSONEncoder().encode(altered), binary: binary ?? archive.binary)
        }

        let missingNode = UUID()
        let property = AnimationProperty.node(missingNode, .castsShadow)
        let dangling = try AnimationClip(tracks: [.boolean(property, times: [0], values: [false])])
        #expect(throws: SceneArchive.Failure.invalidReference(property.description)) {
            try SceneArchive(capturing: graph, animations: [dangling])
        }
        let light = LightRig.Light(id: "duplicate", kind: .point(), position: SIMD3(0, 1, 0),
                                   target: .zero, emission: .intensity(2))
        var rig = try LightRig(lights: [light])
        rig.lights.append(light)
        #expect(throws: LightRig.Failure.duplicateID(light.id)) {
            try SceneArchive(capturing: graph, animations: [clip], lightRig: rig)
        }
        #expect(throws: SceneArchive.Failure.invalidReference(property.description)) {
            try modified { $0.animations[0].propertyTracks?[0].binding = .node(id: missingNode, field: .castsShadow) }
                .load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidReference("uniform set 99")) {
            try modified { $0.animations[0].propertyTracks?[1].binding = .uniform(index: 99, name: "gain") }
                .load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidAnimation("duplicate property track for \(clip.properties[0])")) {
            try modified { altered in
                if let first = altered.animations[0].propertyTracks?.first {
                    altered.animations[0].propertyTracks?.append(first)
                }
            }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidView(99)) {
            try modified { $0.animations[0].propertyTracks?[0].values = .booleans(view: 99) }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidAnimation("unexpected or missing additive reference for \(clip.properties[1])")) {
            try modified { $0.animations[0].additive = true }.load(device: fixture.device)
        }
        #expect(throws: SceneArchive.Failure.invalidAnimation("typed tracks and live materials require archive version 2")) {
            try modified { $0.version = 1 }
        }
        let keys = try #require(manifest.animations[0].propertyTracks?.first?.values)
        guard case .booleans(let view) = keys else {
            Issue.record("Boolean keys must be in the sidecar")
            return
        }
        var binary = archive.binary
        binary[manifest.views[view].offset] = 2
        #expect(throws: SceneArchive.Failure.invalidAnimation("invalid boolean key")) {
            try modified({ _ in }, binary: binary).load(device: fixture.device)
        }
    }
}
