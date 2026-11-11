import Foundation
import Frostfield
import Testing

struct GLTFLightsTests {
    @Test
    func importsPhotometricKindsDefaultsAndSelectedReferences() throws {
        let data = try asset(
            lights: [
                ["type": "directional", "name": "Sun"],
                ["type": "point", "intensity": 75, "color": [1, 0.5, 0.25], "range": 6],
                ["type": "spot", "spot": [:]],
                ["type": "point", "name": "Unselected"],
            ],
            nodes: [
                ["children": [1, 2, 3, 4], "translation": [1, 2, 3]],
                lightNode(2), lightNode(1), lightNode(0), lightNode(1), lightNode(3),
            ],
            scenes: [[0], [5]]
        )

        let plain = try GLTFMeshLoader.loadScene(data: data)
        #expect(plain.lights.isEmpty)
        #expect(plain.nodes.allSatisfy { $0.lightIndex == nil })

        let scene = try GLTFMeshLoader.loadScene(data: data, includeLights: true)
        #expect(scene.lights.map(\.sourceIndex) == [0, 1, 2])
        #expect(scene.nodes.map(\.lightIndex) == [nil, 2, 1, 0, 1])
        #expect(scene.nodes[1].parentIndex == 0)
        #expect(scene.nodes[0].localTransform.translation == SIMD3(1, 2, 3))
        try #require(scene.lights.count == 3)
        #expect(scene.lights[0].name == "Sun")
        #expect(scene.lights[0].light == (try PunctualLight(kind: .directional)))
        #expect(scene.lights[1].light == (try PunctualLight(kind: .point(range: 6), color: SIMD3(1, 0.5, 0.25), intensity: 75)))
        #expect(scene.lights[2].light == (try PunctualLight(kind: .spot())))
    }

    @Test
    func requiredLightsNeedExplicitSupportAndDefinitions() throws {
        let data = try asset(lights: [["type": "point"]], nodes: [lightNode(0)], required: ["KHR_lights_punctual"])
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("KHR_lights_punctual")) {
            try GLTFMeshLoader.loadScene(data: data)
        }
        #expect(try GLTFMeshLoader.loadScene(data: data, includeLights: true).lights.count == 1)
        #expect(throws: GLTFMeshLoader.Failure.invalidLightExtension) {
            try GLTFMeshLoader.loadScene(
                data: asset(lights: nil, nodes: [[:]], required: ["KHR_lights_punctual"]), includeLights: true
            )
        }
        #expect(throws: GLTFMeshLoader.Failure.unsupportedFeature("VENDOR_unsupported")) {
            try GLTFMeshLoader.loadScene(
                data: asset(
                    lights: [["type": "point"]], nodes: [lightNode(0)],
                    required: ["KHR_lights_punctual", "VENDOR_unsupported"]
                ), includeLights: true
            )
        }
    }

    @Test
    func rejectsInvalidReferencesPropertiesAndTooManyInstances() throws {
        #expect(throws: GLTFMeshLoader.Failure.missingLight(1)) {
            try GLTFMeshLoader.loadScene(
                data: asset(lights: [["type": "point"]], nodes: [lightNode(1)]), includeLights: true
            )
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidLight(0, "spot lights require a spot object")) {
            try GLTFMeshLoader.loadScene(
                data: asset(lights: [["type": "spot"]], nodes: [lightNode(0)]), includeLights: true
            )
        }
        let invalidCone = #expect(throws: GLTFMeshLoader.Failure.self) {
            try GLTFMeshLoader.loadScene(
                data: asset(
                    lights: [["type": "spot", "spot": ["innerConeAngle": 1, "outerConeAngle": 0.5]]],
                    nodes: [lightNode(0)]
                ), includeLights: true
            )
        }
        if case .invalidLight(0, _) = invalidCone {
        } else {
            Issue.record("Expected an invalid cone error")
        }

        let limit = SceneGraph.maximumPunctualLightCount
        #expect(throws: GLTFMeshLoader.Failure.lightInstanceLimit(limit)) {
            try GLTFMeshLoader.loadScene(
                data: asset(
                    lights: [["type": "point"]], nodes: Array(repeating: lightNode(0), count: limit + 1),
                    scenes: [Array(0...limit)]
                ), includeLights: true
            )
        }
    }

    private func lightNode(_ index: Int) -> [String: Any] {
        ["extensions": ["KHR_lights_punctual": ["light": index]]]
    }

    private func asset(
        lights: [[String: Any]]?, nodes: [[String: Any]], scenes: [[Int]] = [[0]], required: [String] = []
    ) throws -> Data {
        var document: [String: Any] = [
            "asset": ["version": "2.0"], "nodes": nodes,
            "scenes": scenes.map { ["nodes": $0] },
            "extensionsUsed": ["KHR_lights_punctual"],
        ]
        if let lights { document["extensions"] = ["KHR_lights_punctual": ["lights": lights]] }
        if !required.isEmpty { document["extensionsRequired"] = required }
        return try JSONSerialization.data(withJSONObject: document)
    }
}
