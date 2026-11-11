import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Visibility, layers, metadata, lookup, coordinate conversion, look-at and attach.
struct SceneNodeConveniencesTests {
    private let camera: PerspectiveCamera

    init() throws {
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 50)
    }

    // MARK: Lookup and editing (CPU)

    private func sampleGraph() -> (SceneGraph, ids: [String: UUID]) {
        let leaf = SceneNode(transform: SceneTransform(translation: SIMD3(0, 1, 0)), name: "leaf", metadata: ["kind": "leaf"])
        let twin = SceneNode(name: "twin", isVisible: false)
        let branch = SceneNode(
            transform: SceneTransform(translation: SIMD3(2, 0, 0), scale: SIMD3(repeating: 2)),
            children: [leaf, twin], name: "branch"
        )
        let other = SceneNode(name: "twin", layers: SceneLayers(layer: 3))
        let graph = SceneGraph(roots: [branch, other], camera: camera)
        return (graph, ["leaf": leaf.id, "twin": twin.id, "branch": branch.id, "other": other.id])
    }

    @Test
    func `Lookup by ID, name and predicate follows depth-first order`() throws {
        let (graph, ids) = sampleGraph()
        #expect(graph.node(withID: try #require(ids["leaf"]))?.name == "leaf")
        #expect(graph.node(withID: UUID()) == nil)
        #expect(graph.node(named: "twin")?.id == ids["twin"])
        #expect(graph.nodes(named: "twin").map(\.id) == [ids["twin"], ids["other"]].compactMap { $0 })
        #expect(graph.firstNode { $0.metadata["kind"] == "leaf" }?.id == ids["leaf"])
        #expect(graph.nodes { $0.layers.contains(layer: 3) }.map(\.id) == [ids["other"]].compactMap { $0 })

        var visited: [String?] = []
        var parents: [UUID?] = []
        graph.forEachNode { node, parent in
            visited.append(node.name)
            parents.append(parent)
        }
        #expect(visited == ["branch", "leaf", "twin", "twin"])
        #expect(parents == [nil, ids["branch"], ids["branch"], nil])
        var visible: [UUID] = []
        graph.forEachVisibleNode { node, _ in visible.append(node.id) }
        #expect(visible == [ids["branch"], ids["leaf"], ids["other"]].compactMap { $0 })
        #expect(graph.ancestors(of: try #require(ids["leaf"]))?.map(\.id) == [ids["branch"]].compactMap { $0 })
        #expect(graph.ancestors(of: try #require(ids["branch"]))?.isEmpty == true)
        #expect(graph.ancestors(of: UUID()) == nil)
        #expect(graph.parentID(of: try #require(ids["twin"])) == ids["branch"])
    }

    @Test
    func `Add, remove and clear edit the tree and reject duplicate IDs`() throws {
        var (graph, ids) = sampleGraph()
        let branch = try #require(ids["branch"])
        let added = SceneNode(children: [SceneNode(name: "grandchild")], name: "added")
        try graph.add(added, toParent: branch)
        #expect(graph.node(withID: branch)?.children.map(\.name) == ["leaf", "twin", "added"])
        #expect(throws: SceneGraph.EditingError.duplicateID(added.id)) { try graph.add(added) }
        #expect(throws: SceneGraph.EditingError.missingNode(ids["other"]!)) {
            var copy = graph
            copy.removeNode(withID: ids["other"]!)
            try copy.add(SceneNode(), toParent: ids["other"]!)
        }

        let removed = graph.removeNode(withID: try #require(ids["leaf"]))
        #expect(removed?.name == "leaf")
        #expect(graph.node(withID: try #require(ids["leaf"])) == nil)
        #expect(graph.removeNode(withID: UUID()) == nil)

        let cleared = try graph.removeChildren(of: branch)
        #expect(cleared.map(\.name) == ["twin", "added"])
        #expect(graph.node(withID: branch)?.children.isEmpty == true)
        try graph.updateNode(withID: branch) { $0.metadata["note"] = ["a": 1, "b": [true, nil, "x"]] }
        #expect(graph.node(withID: branch)?.metadata["note"]?["b"]?.arrayValue?.count == 3)
        _ = try graph.resolve()
    }

    @Test
    func `World and local conversions invert each other`() throws {
        let (graph, ids) = sampleGraph()
        let leaf = try #require(ids["leaf"])
        // branch: translate (2, 0, 0), scale 2; leaf: translate (0, 1, 0).
        #expect(simd_distance(try graph.worldPosition(of: leaf), SIMD3(2, 2, 0)) < 1e-6)
        let world = try graph.localToWorld(SIMD3(1, 0, 0), in: leaf)
        #expect(simd_distance(world, SIMD3(4, 2, 0)) < 1e-6)
        #expect(simd_distance(try graph.worldToLocal(world, in: leaf), SIMD3(1, 0, 0)) < 1e-6)
        let resolved = try #require(try graph.resolve().first { $0.id == leaf })
        #expect(try graph.worldTransform(of: leaf) == resolved.worldTransform)
        let missing = UUID()
        #expect(throws: SceneGraph.EditingError.missingNode(missing)) { try graph.worldTransform(of: missing) }
    }

    @Test
    func `Attach keeps the world matrix under a new parent and back at the root`() throws {
        let parent = SceneNode(transform: SceneTransform(
            translation: SIMD3(3, -1, 2),
            rotation: simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3(1, 2, 0.5))),
            scale: SIMD3(repeating: 1.5)
        ))
        let child = SceneNode(transform: SceneTransform(
            translation: SIMD3(-2, 4, 1),
            rotation: simd_quatf(angle: -1.1, axis: simd_normalize(SIMD3(0.2, 1, -0.3))),
            scale: SIMD3(0.5, 2, 1)
        ))
        var graph = SceneGraph(roots: [parent, child], camera: camera)
        let before = try graph.worldTransform(of: child.id)
        try graph.attach(child.id, to: parent.id)
        #expect(graph.parentID(of: child.id) == parent.id)
        #expect(graph.roots.count == 1)
        let attached = try graph.worldTransform(of: child.id)
        #expect(sceneMatrixDifference(before, attached) < 2e-5)
        // The local transform changed to compensate for the parent.
        #expect(graph.node(withID: child.id)?.transform.translation != child.transform.translation)

        try graph.attach(child.id, to: nil)
        #expect(graph.parentID(of: child.id) == nil)
        #expect(sceneMatrixDifference(before, try graph.worldTransform(of: child.id)) < 2e-5)

        // A mirrored parent flips the child's local handedness but keeps its world matrix.
        try graph.updateNode(withID: parent.id) { $0.transform.scale = SIMD3(-1.5, 1.5, 1.5) }
        try graph.attach(child.id, to: parent.id)
        #expect(sceneMatrixDifference(before, try graph.worldTransform(of: child.id)) < 2e-5)
        #expect(throws: SceneGraph.EditingError.cycle(parent.id)) { try graph.attach(parent.id, to: child.id) }
        #expect(throws: SceneGraph.EditingError.cycle(parent.id)) { try graph.attach(parent.id, to: parent.id) }
    }

    @Test
    func `Attach refuses a sheared result and leaves the graph unchanged`() throws {
        let stretched = SceneNode(transform: SceneTransform(scale: SIMD3(1, 3, 1)))
        let tilted = SceneNode(transform: SceneTransform(rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1))))
        var graph = SceneGraph(roots: [stretched, tilted], camera: camera)
        let before = graph.roots.map(\.id)
        // A 45° child of a parent stretched along Y would need shear in its local matrix.
        #expect(throws: SceneGraph.EditingError.unrepresentableTransform) {
            try graph.attach(tilted.id, to: stretched.id)
        }
        #expect(graph.roots.map(\.id) == before)
        // Decomposition reports shear directly.
        let shear = simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0.5, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(0, 0, 0, 1)))
        #expect(throws: SceneGraph.EditingError.unrepresentableTransform) { try SceneTransform(decomposing: shear) }
    }

    @Test
    func `Look-at points the chosen axis at a world target under a stretched parent`() throws {
        let parent = SceneNode(transform: SceneTransform(
            translation: SIMD3(1, 2, 3), rotation: simd_quatf(angle: 0.9, axis: SIMD3(0, 1, 0)), scale: SIMD3(2, 0.5, 1)
        ))
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0.5, -1, 0.25)))
        var graph = SceneGraph(roots: [parent], camera: camera)
        try graph.add(node, toParent: parent.id)
        let target = SIMD3<Float>(-4, 5, 2)
        for axis in [SceneForwardAxis.negativeZ, .positiveZ] {
            try graph.lookAt(node.id, worldTarget: target, forward: axis)
            let world = try graph.worldTransform(of: node.id)
            let local = SIMD4<Float>(0, 0, axis == .negativeZ ? -1 : 1, 0)
            let facing = simd_normalize(SIMD3((world * local).x, (world * local).y, (world * local).z))
            let toTarget = simd_normalize(target - (try graph.worldPosition(of: node.id)))
            #expect(simd_dot(facing, toTarget) > 0.99999, "\(axis)")
            #expect(abs(simd_length(graph.node(withID: node.id)!.transform.rotation.vector) - 1) < 1e-6)
        }
        // The camera follows the same convention as lights: local -Z faces the target.
        try graph.pointCamera(at: SIMD3(0, 0, -10))
        #expect(simd_distance(graph.cameraTransform.rotation.act(SIMD3(0, 0, -1)), SIMD3(0, 0, -1)) < 1e-6)
        try graph.pointCamera(at: SIMD3(5, 0, 0))
        #expect(simd_distance(graph.cameraTransform.rotation.act(SIMD3(0, 0, -1)), SIMD3(1, 0, 0)) < 1e-6)
        #expect(simd_distance(graph.cameraTransform.rotation.act(SIMD3(0, 1, 0)), SIMD3(0, 1, 0)) < 1e-6)
        #expect(throws: SceneGraph.EditingError.invalidLookAt) { try graph.pointCamera(at: graph.cameraTransform.translation) }
    }

    @Test
    func `Layers follow Three.js Layers semantics and metadata round-trips as JSON`() throws {
        var layers = SceneLayers.default
        #expect(layers.contains(layer: 0) && !layers.contains(layer: 1))
        layers.enable(5)
        layers.toggle(0)
        #expect(layers.layerNumbers == [5])
        #expect(!layers.intersects(.default) && layers.intersects(SceneLayers(layers: [1, 5])))
        layers.set(only: 31)
        #expect(layers.rawValue == 1 << 31)
        #expect(SceneLayers.all.layerNumbers.count == 32)

        let metadata: [String: SceneMetadataValue] = [
            "count": 3, "ratio": 0.25, "tags": ["hero", "metal"], "nested": ["on": true, "none": nil],
        ]
        let data = try JSONEncoder().encode(metadata)
        #expect(try JSONDecoder().decode([String: SceneMetadataValue].self, from: data) == metadata)
        #expect(metadata["nested"]?["on"]?.boolValue == true)
        #expect(SceneMetadataValue.number(.nan).isFinite == false)
    }

    @Test
    func `glTF export names nodes from SceneNode.name unless a name is supplied`() throws {
        let graph = SceneGraph(roots: [SceneNode(name: "Lamp"), SceneNode(name: "Table")], camera: camera)
        let output = try GLTFExporter(format: .gltf).export(
            graph, nodeNames: [graph.roots[1].id: "Desk"], includesCamera: false
        )
        let document = try #require(try JSONSerialization.jsonObject(with: output.data) as? [String: Any])
        let nodes = try #require(document["nodes"] as? [[String: Any]])
        #expect(nodes.compactMap { $0["name"] as? String } == ["Lamp", "Desk"])
    }

    @Test
    func `Adding many subtrees at once checks duplicates in one pass and keeps order`() throws {
        var graph = SceneGraph(roots: [SceneNode(name: "existing")], camera: camera)
        let nodes = (0..<8_000).map { SceneNode(children: [SceneNode()], name: "n\($0)") }
        let clock = ContinuousClock()
        let elapsed = try clock.measure { try graph.add(contentsOf: nodes) }
        #expect(graph.roots.count == 8_001)
        #expect(graph.roots.dropFirst().map(\.id) == nodes.map(\.id))
        // One walk of the graph; adding these one at a time walked it 8,000 times.
        #expect(elapsed < .milliseconds(500), "\(elapsed)")

        let fresh = SceneNode(name: "fresh")
        #expect(throws: SceneGraph.EditingError.duplicateID(fresh.id)) { try graph.add(contentsOf: [fresh, fresh]) }
        #expect(throws: SceneGraph.EditingError.duplicateID(nodes[4_321].children[0].id)) {
            try graph.add(contentsOf: [fresh, nodes[4_321].children[0]])
        }
        #expect(graph.roots.count == 8_001 && graph.node(withID: fresh.id) == nil)

        let a = SceneNode(name: "a"), b = SceneNode(name: "b")
        try graph.add(contentsOf: [a, b], toParent: nodes[7].id)
        #expect(graph.node(withID: nodes[7].id)?.children.map(\.name) == [nil, "a", "b"])
        #expect(throws: SceneGraph.EditingError.duplicateID(b.id)) { try graph.add(b) }
        try graph.add(fresh)
        #expect(graph.roots.last?.id == fresh.id)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `glTF import sets SceneNode.name, so names survive export and import`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let graph = SceneGraph(
            roots: [SceneNode(children: [SceneNode(name: "Bulb")], name: "Lamp"), SceneNode(), SceneNode(name: "Table")],
            camera: camera
        )
        let output = try GLTFExporter(format: .glb).export(graph, includesCamera: false)
        let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .bgra8Unorm, scene: GLTFMeshLoader.loadScene(data: output.data))
        let imported = try prepared.graph(camera: camera)
        #expect(imported.node(named: "Bulb") != nil)
        var names: [String?] = []
        imported.forEachNode { node, _ in names.append(node.name) }
        #expect(names == ["Lamp", "Bulb", nil, "Table"])
    }

    @Test
    func `glTF export skips hidden subtrees and their animation tracks unless asked to keep them`() throws {
        let inner = SceneNode(name: "inner")
        let hidden = SceneNode(children: [inner], name: "hidden", isVisible: false)
        let shown = SceneNode(name: "shown")
        let offLayer = SceneNode(name: "offLayer", layers: SceneLayers(layer: 4))
        let graph = SceneGraph(roots: [shown, hidden, offLayer], camera: camera)
        func track(_ id: UUID) throws -> SceneAnimationTrack {
            try SceneAnimationTrack(nodeID: id, keyframes: [
                TransformKeyframe(time: 0, transform: SceneTransform()),
                TransformKeyframe(time: 1, transform: SceneTransform(translation: SIMD3(1, 0, 0))),
            ])
        }
        let clip = GLTFExporter.Animation(animation: try SceneAnimation(tracks: [track(inner.id), track(shown.id)]))
        func document(_ output: GLTFExporter.Output) throws -> (names: [String], channels: Int) {
            let json = try #require(try JSONSerialization.jsonObject(with: output.data) as? [String: Any])
            let nodes = try #require(json["nodes"] as? [[String: Any]])
            let animations = json["animations"] as? [[String: Any]] ?? []
            return (nodes.compactMap { $0["name"] as? String }, animations.reduce(0) { $0 + (($1["channels"] as? [Any])?.count ?? 0) })
        }
        let visible = try GLTFExporter(format: .gltf).export(graph, animations: [clip], includesCamera: false)
        #expect(try document(visible).names == ["shown", "offLayer"])
        #expect(try document(visible).channels == 3)
        #expect(visible.warnings.contains(.hiddenAnimationTargetDropped(inner.id)))
        #expect(visible.nodeIndices[inner.id] == nil)

        let everything = try GLTFExporter(format: .gltf).export(graph, animations: [clip], includesCamera: false, onlyVisible: false)
        #expect(try document(everything).names == ["shown", "hidden", "inner", "offLayer"])
        #expect(try document(everything).channels == 6)
        #expect(everything.warnings.isEmpty)
        #expect(throws: GLTFExporter.Failure.missingAnimationTarget(offLayer.id)) {
            try GLTFExporter(format: .gltf).export(
                SceneGraph(roots: [shown], camera: camera), animations: [.init(animation: try SceneAnimation(tracks: [track(offLayer.id)]))],
                includesCamera: false
            )
        }
    }

    @Test
    func `Only visible lights in the camera layers count toward the light limit`() throws {
        let light = try PunctualLight(kind: .point())
        var roots = (0..<SceneGraph.maximumPunctualLightCount).map { _ in SceneNode(light: light) }
        roots.append(SceneNode(light: light, isVisible: false))
        roots.append(SceneNode(light: light, layers: SceneLayers(layer: 2)))
        var graph = SceneGraph(roots: roots, camera: camera)
        #expect(try graph.resolve().count == roots.count)
        graph.cameraLayers.enable(2)
        #expect(throws: SceneGraph.ValidationError.tooManyLights(maximum: SceneGraph.maximumPunctualLightCount)) {
            try graph.resolve()
        }
    }
}

/// Rendered checks that hidden and off-layer nodes skip draws, shadows, lighting and picking.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneVisibilityRenderingTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    @Test("Hidden subtrees and off-layer nodes render exactly like a graph without them")
    func draws() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        let kept = SceneNode(transform: SceneTransform(translation: SIMD3(-0.6, 0, -2)), mesh: mesh)
        let child = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0.7, 0)), mesh: mesh)
        let parent = SceneNode(transform: SceneTransform(translation: SIMD3(0.6, 0, -2)), mesh: mesh, children: [child])
        let reference = try fixture.render(SceneGraph(roots: [kept], camera: fixture.camera))
        let everything = try fixture.render(SceneGraph(roots: [kept, parent], camera: fixture.camera))
        #expect(everything.statistics.drawnMeshCount == 3)
        #expect(everything.pixels != reference.pixels)

        var hidden = parent
        hidden.isVisible = false
        let hiddenResult = try fixture.render(SceneGraph(roots: [kept, hidden], camera: fixture.camera))
        #expect(hiddenResult.statistics.drawnMeshCount == 1)
        #expect(hiddenResult.pixels == reference.pixels)

        // Layers are per node: the child stays visible when only its parent leaves the camera's layers.
        var offLayer = parent
        offLayer.layers = SceneLayers(layer: 4)
        var graph = SceneGraph(roots: [kept, offLayer], camera: fixture.camera)
        let layered = try fixture.render(graph)
        #expect(layered.statistics.drawnMeshCount == 2)
        let childPixel = fixture.pixel(layered.pixels, x: fixture.column(x: 0.6, depth: 2), y: fixture.row(y: 0.7, depth: 2))
        let parentPixel = fixture.pixel(layered.pixels, x: fixture.column(x: 0.6, depth: 2), y: 32)
        #expect(childPixel.w == 1 && parentPixel.w == 0)
        graph.cameraLayers = SceneLayers(layers: [0, 4])
        #expect(try fixture.render(graph).pixels == everything.pixels)
    }

    @Test("A hidden or off-layer blocker casts no directional shadow")
    func shadows() throws {
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3(2.625, 0, 0.5)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(0.25))
        )
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 512, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 5
        )
        let shadowed = try fixture.render(SceneGraph(roots: [receiver, blocker], camera: fixture.camera), shadowMap: shadowMap)
        #expect(shadowed.statistics.shadowDrawCount == 1)
        #expect(abs(fixture.pixel(shadowed.pixels, x: 32, y: 32).x - 0.16) < 0.01)
        let unshadowed = try fixture.render(SceneGraph(roots: [receiver], camera: fixture.camera), shadowMap: shadowMap)

        var hidden = blocker
        hidden.isVisible = false
        let hiddenResult = try fixture.render(SceneGraph(roots: [receiver, hidden], camera: fixture.camera), shadowMap: shadowMap)
        #expect(hiddenResult.statistics.shadowDrawCount == 0)
        #expect(hiddenResult.pixels == unshadowed.pixels)
        #expect(abs(fixture.pixel(hiddenResult.pixels, x: 32, y: 32).x - 0.832) < 0.01)

        var offLayer = blocker
        offLayer.layers = SceneLayers(layer: 1)
        let layered = try fixture.render(SceneGraph(roots: [receiver, offLayer], camera: fixture.camera), shadowMap: shadowMap)
        #expect(layered.statistics.shadowDrawCount == 0)
        #expect(layered.pixels == unshadowed.pixels)
    }

    @Test("A hidden light node adds no light to PBR surfaces")
    func lights() throws {
        let material = try PhysicallyBasedMaterial(roughness: 0.6, lightIntensity: 0, ambientIntensity: 0.05)
        let surface = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -2)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.5), material: .physicallyBased(material))
        )
        let lamp = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -1)), light: try PunctualLight(kind: .point(), intensity: 2)
        )
        let dark = try fixture.render(SceneGraph(roots: [surface], camera: fixture.camera))
        let lit = try fixture.render(SceneGraph(roots: [surface, lamp], camera: fixture.camera))
        // 2 cd at 1 m gives 2 lux; a white Lambertian surface reflects about 2 / π of it.
        let added = fixture.pixel(lit.pixels, x: 32, y: 32).x - fixture.pixel(dark.pixels, x: 32, y: 32).x
        #expect(added > 0.4)

        var hidden = lamp
        hidden.isVisible = false
        #expect(try fixture.render(SceneGraph(roots: [surface, hidden], camera: fixture.camera)).pixels == dark.pixels)
        var offLayer = lamp
        offLayer.layers = SceneLayers(layer: 7)
        var graph = SceneGraph(roots: [surface, offLayer], camera: fixture.camera)
        #expect(try fixture.render(graph).pixels == dark.pixels)
        graph.cameraLayers.enable(7)
        #expect(try fixture.render(graph).pixels == lit.pixels)
    }

    @Test("Picking skips hidden subtrees and tests the requested layers")
    func picking() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.5))
        let back = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -4)), mesh: mesh)
        let frontChild = SceneNode(mesh: mesh)
        let front = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), children: [frontChild])
        var graph = SceneGraph(roots: [back, front], camera: fixture.camera)
        let center = SIMD2<Float>(0.5, 0.5)
        #expect(try graph.pick(at: center, aspectRatio: 1)?.nodeID == frontChild.id)

        try graph.updateNode(withID: front.id) { $0.isVisible = false }
        let hit = try #require(try graph.pick(at: center, aspectRatio: 1))
        #expect(hit.nodeID == back.id)
        #expect(abs(hit.intersection.distance - 3.9) < 1e-4)

        try graph.updateNode(withID: front.id) { $0.isVisible = true }
        try graph.updateNode(withID: frontChild.id) { $0.layers = SceneLayers(layer: 2) }
        #expect(try graph.pick(at: center, aspectRatio: 1)?.nodeID == back.id)
        #expect(try graph.pick(at: center, aspectRatio: 1, layers: SceneLayers(layer: 2))?.nodeID == frontChild.id)
        try graph.updateNode(withID: back.id) { $0.layers = SceneLayers(layer: 9) }
        #expect(try graph.pick(at: center, aspectRatio: 1) == nil)
    }
}

private func sceneMatrixDifference(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
    let columns = [(a.columns.0, b.columns.0), (a.columns.1, b.columns.1), (a.columns.2, b.columns.2), (a.columns.3, b.columns.3)]
    return columns.map { simd_reduce_max(abs($0.0 - $0.1)) }.max() ?? 0
}
