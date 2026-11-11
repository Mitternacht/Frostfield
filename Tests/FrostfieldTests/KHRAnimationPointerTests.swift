import Foundation
import Frostfield
import Metal
import Testing
import simd

struct KHRAnimationPointerTests {
    /// A quad with one material, a point light on node 1 and a camera on node 2.
    private func asset(
        roughness: Float = 0.5, lightIntensity: Float = 10, lightY: Float = 0.5, emissive: SIMD3<Float> = .zero,
        pointers: [(String, [Float], [Float], String, String)] = [], required: [String] = []
    ) throws -> Data {
        var fixture = GLTFMotionFixture()
        let h: Float = 1.5
        let positions = fixture.floats([-h, -h, 0, h, -h, 0, h, h, 0, -h, h, 0], type: "VEC3", count: 4)
        let normals = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 4)
        let indices = fixture.bytes(
            [0, 1, 2, 0, 2, 3].flatMap { (value: UInt16) in [UInt8(value & 0xff), UInt8(value >> 8)] },
            type: "SCALAR", count: 6, component: 5123
        )
        var channels: [[String: Any]] = []
        var samplers: [[String: Any]] = []
        for (pointer, times, values, type, interpolation) in pointers {
            let input = fixture.floats(times, type: "SCALAR", count: times.count, bounds: [times.first!, times.last!])
            let components = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4][type]!
            let output = fixture.floats(values, type: type, count: values.count / components)
            samplers.append(["input": input, "output": output, "interpolation": interpolation])
            channels.append([
                "sampler": samplers.count - 1,
                "target": ["path": "pointer", "extensions": ["KHR_animation_pointer": ["pointer": pointer]]],
            ])
        }
        var sections: [String: Any] = [
            "scenes": [["nodes": [0, 1, 2]]],
            "nodes": [
                ["mesh": 0, "translation": [0, 0, -1]],
                ["extensions": ["KHR_lights_punctual": ["light": 0]], "translation": [0, lightY, 0]],
                ["camera": 0, "translation": [0, 0, 3]],
            ],
            "meshes": [["primitives": [["attributes": ["POSITION": positions, "NORMAL": normals], "indices": indices, "material": 0]]]],
            "materials": [[
                "pbrMetallicRoughness": ["baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0, "roughnessFactor": roughness],
                "emissiveFactor": [emissive.x, emissive.y, emissive.z],
            ]],
            "extensions": ["KHR_lights_punctual": ["lights": [["type": "point", "intensity": lightIntensity]]]],
            "cameras": [["type": "perspective", "perspective": ["yfov": 0.5, "znear": 0.1, "zfar": 100]]],
            "extensionsUsed": ["KHR_lights_punctual", "KHR_animation_pointer"],
        ]
        if !channels.isEmpty { sections["animations"] = [["name": "Pointers", "channels": channels, "samplers": samplers]] }
        if !required.isEmpty { sections["extensionsRequired"] = required }
        return try fixture.document(sections)
    }

    private var animatedPointers: [(String, [Float], [Float], String, String)] {
        [
            ("/materials/0/pbrMetallicRoughness/roughnessFactor", [0, 2], [0.2, 0.8], "SCALAR", "LINEAR"),
            ("/extensions/KHR_lights_punctual/lights/0/intensity", [0, 2], [10, 30], "SCALAR", "LINEAR"),
            ("/materials/0/emissiveFactor", [0, 0.4], [0, 0, 0, 0.5, 0.25, 0], "VEC3", "STEP"),
            ("/nodes/1/translation", [0, 2], [0, 0.5, 0, 0, 1.5, 0], "VEC3", "LINEAR"),
            ("/cameras/0/perspective/yfov", [0, 2], [0.5, 1], "SCALAR", "LINEAR"),
        ]
    }

    private func load(_ data: Data, materials: Bool = true) throws -> GLTFScene {
        try GLTFMeshLoader.loadScene(
            data: data, includeMaterials: materials, includeAnimations: true, includeLights: true, includeCameras: true
        )
    }

    @Test
    func `Imports material, light, camera and node pointers and skips data that was not loaded`() throws {
        let scene = try load(asset(pointers: animatedPointers))
        let clip = try #require(scene.animations.first)
        #expect(clip.duration == 2)
        #expect(clip.channels.map(\.sourceNodeIndex) == [1])
        #expect(clip.channels.map(\.target) == [.translation])
        #expect(clip.pointerChannels.map(\.target) == [
            .material(0, .roughness), .light(0, .lightIntensity), .material(0, .emissiveFactor), .camera(0, .verticalFieldOfView),
        ])
        let withoutMaterials = try load(asset(pointers: animatedPointers), materials: false)
        #expect(withoutMaterials.animations.first?.pointerChannels.map(\.target) == [.light(0, .lightIntensity), .camera(0, .verticalFieldOfView)])
    }

    @Test
    func `Skips pointers Frostfield cannot bind unless required, and rejects duplicates and unsupported required use`() throws {
        let unknown = "/materials/0/extensions/KHR_materials_volume/thicknessFactor"
        let visibility = "/nodes/0/extensions/KHR_node_visibility/visible"
        let unbindable: [(String, [Float], [Float], String, String)] = [
            (unknown, [0, 1], [0, 1], "SCALAR", "LINEAR"), (visibility, [0, 1], [0, 1], "SCALAR", "STEP"),
        ]
        // Without the extension required, the other channels still load, as clients without it would.
        let skipped = try #require(load(asset(pointers: animatedPointers + unbindable)).animations.first)
        #expect(skipped.unsupportedPointers == [unknown, visibility])
        #expect(skipped.channels.map(\.target) == [.translation])
        #expect(skipped.pointerChannels.count == 4)
        let onlyUnbindable = try #require(load(asset(pointers: unbindable)).animations.first)
        #expect(onlyUnbindable.channels.isEmpty && onlyUnbindable.pointerChannels.isEmpty)
        #expect(onlyUnbindable.unsupportedPointers == [unknown, visibility])
        #expect(try load(asset(pointers: animatedPointers)).animations.first?.unsupportedPointers == [])
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_animation_pointer target \(unknown)")) {
            try load(asset(pointers: animatedPointers + unbindable, required: ["KHR_animation_pointer"]))
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_animation_pointer target \(visibility)")) {
            try load(asset(pointers: [unbindable[1]], required: ["KHR_animation_pointer"]))
        }
        let roughness = "/materials/0/pbrMetallicRoughness/roughnessFactor"
        #expect(throws: GLTFMeshLoader.Failure.invalidAnimation(0, "pointer \(roughness) has duplicate channels")) {
            try load(asset(pointers: [(roughness, [0, 1], [0, 1], "SCALAR", "LINEAR"), (roughness, [0, 1], [1, 0], "SCALAR", "LINEAR")]))
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidAccessor(4, "animation pointer values requires VEC3 with float components")) {
            try load(asset(pointers: [("/materials/0/emissiveFactor", [0, 1], [0, 1], "SCALAR", "LINEAR")]))
        }
        let required = try asset(pointers: animatedPointers, required: ["KHR_animation_pointer"])
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_animation_pointer")) {
            try GLTFMeshLoader.loadScene(data: required)
        }
        #expect(try load(required).animations.count == 1)
        #expect(GLTFMeshLoader.supportedExtensions.contains("KHR_animation_pointer"))
    }

    @Test
    func `Export reports the pointer channels it drops and keeps node pointers as node channels`() throws {
        let scene = try load(asset(pointers: animatedPointers))
        let exported = try GLTFExporter(format: .gltf).export(scene)
        #expect(exported.warnings.contains(.animationPointersDropped(sourceAnimation: 0, pointers: [
            "/materials/0/pbrMetallicRoughness/roughnessFactor", "/extensions/KHR_lights_punctual/lights/0/intensity",
            "/materials/0/emissiveFactor", "/cameras/0/perspective/yfov",
        ])))
        let reloaded = try #require(load(exported.data).animations.first)
        #expect(reloaded.pointerChannels.isEmpty)
        #expect(reloaded.channels.map(\.target) == [.translation])
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Pointers to a material only another variant draws are skipped when binding`() throws {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0], type: "VEC3", count: 3)
        let normals = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 3)
        let input = fixture.floats([0, 2], type: "SCALAR", count: 2, bounds: [0, 2])
        let roughness = fixture.floats([0.2, 0.8], type: "SCALAR", count: 2)
        let move = fixture.floats([0, 0, 0, 2, 0, 0], type: "VEC3", count: 2)
        func pointer(_ path: String, _ sampler: Int) -> [String: Any] {
            ["sampler": sampler, "target": ["path": "pointer", "extensions": ["KHR_animation_pointer": ["pointer": path]]]]
        }
        let data = try fixture.document([
            "nodes": [["mesh": 0]],
            "meshes": [["primitives": [[
                "attributes": ["POSITION": positions, "NORMAL": normals], "material": 0,
                "extensions": ["KHR_materials_variants": ["mappings": [["variants": [0], "material": 1]]]],
            ]]]],
            "materials": [
                ["pbrMetallicRoughness": ["roughnessFactor": 0.5]], ["pbrMetallicRoughness": ["roughnessFactor": 0.5]],
            ],
            "extensions": ["KHR_materials_variants": ["variants": [["name": "Alternate"]]]],
            "extensionsUsed": ["KHR_materials_variants", "KHR_animation_pointer"],
            "animations": [[
                "channels": [
                    pointer("/materials/1/pbrMetallicRoughness/roughnessFactor", 0),
                    ["sampler": 1, "target": ["node": 0, "path": "translation"]],
                ],
                "samplers": [["input": input, "output": roughness], ["input": input, "output": move]],
            ]],
        ])
        let device = try #require(MTLCreateSystemDefaultDevice())
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true, includeAnimations: true)
        #expect(scene.animations.first?.pointerChannels.map(\.target) == [.material(1, .roughness)])

        // The default look draws material 0, so the pointer has nothing to animate; the node track still binds.
        let standard = try PreparedGLTFScene(device: device, colorPixelFormat: .rgba16Float, scene: scene)
        #expect(standard.materialFactors.isEmpty)
        let clip = try standard.animationClip(0)
        #expect(clip.properties.isEmpty)
        #expect(clip.nodeIDs == [try #require(standard.nodeIDs[0])])

        // The variant draws material 1, which the pointer animates.
        let alternate = try PreparedGLTFScene(
            device: device, colorPixelFormat: .rgba16Float, scene: scene.selectingMaterialVariant(named: "Alternate")
        )
        let live = try #require(alternate.materialFactors[1])
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: alternate.animationClip(0)))
        try mixer.update(deltaTime: 0.5)
        try mixer.applyLiveProperties()
        #expect(abs(live.current.roughness - 0.35) < 1e-6)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Pointer animation sets roughness, light intensity and camera at a time and renders like a static asset`() throws {
        let render = try GeometryRenderFixture(side: 32)
        func prepare(_ scene: GLTFScene) throws -> PreparedGLTFScene {
            try PreparedGLTFScene(
                device: render.device, colorPixelFormat: .rgba16Float, scene: scene, ambientIntensity: 0
            )
        }
        let prepared = try prepare(load(asset(pointers: animatedPointers)))
        let live = try #require(prepared.materialFactors[0])
        let clip = try prepared.animationClip(0)
        var mixer = AnimationMixer()
        try mixer.play(mixer.addAction(for: clip))
        let rest = try render.render(mixer.apply(to: prepared.graph(camera: render.camera))).pixels

        try mixer.update(deltaTime: 0.5)
        try mixer.applyLiveProperties()
        let graph = try mixer.apply(to: prepared.graph(camera: render.camera))
        // Linear roughness 0.2 → 0.8 and intensity 10 → 30 over two seconds, sampled at 0.5 s.
        #expect(abs(live.current.roughness - 0.35) < 1e-6)
        #expect(live.current.emissiveFactor == SIMD3(0.5, 0.25, 0))
        let lightID = try #require(prepared.nodeIDs[1])
        let lightNode = try #require(graph.resolve().first { $0.id == lightID })
        #expect(abs(try #require(lightNode.light).light.intensity - 15) < 1e-5)
        #expect(simd_distance(lightNode.light!.position, SIMD3(0, 0.75, 0)) < 1e-6)
        let view = try prepared.cameraView(nodeIndex: 2, in: graph, viewportAspectRatio: 1, animatedBy: mixer)
        #expect(abs(view.clipFromWorld.columns.1.y - 1 / tan(0.3125)) < 1e-5)
        let still = try prepared.cameraView(nodeIndex: 2, in: graph, viewportAspectRatio: 1)
        #expect(abs(still.clipFromWorld.columns.1.y - 1 / tan(0.25)) < 1e-5)

        let animated = try render.render(graph).pixels
        let reference = try render.render(prepare(load(asset(
            roughness: live.current.roughness, lightIntensity: lightNode.light!.light.intensity, lightY: 0.75,
            emissive: SIMD3(0.5, 0.25, 0)
        ))).graph(camera: render.camera)).pixels
        #expect(animated == reference)
        let change = zip(animated, rest).map { simd_reduce_max(abs($0 - $1)) }.max() ?? 0
        #expect(change > 0.05)
    }
}
