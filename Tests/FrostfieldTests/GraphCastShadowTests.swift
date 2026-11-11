import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

/// A floor lit by a directional light from +X at 45 degrees, seen from above. A caster plane one
/// unit over the origin throws its shadow onto the floor point at x = -1, which the camera sees
/// beside the plane. The shadow map lets a test read the shadowed and an open floor pixel.
private struct CastShadowScene {
    struct Rendered {
        var shadowed: SIMD3<Float>
        var open: SIMD3<Float>
        var passes: Int
        var map: DirectionalShadowMap
    }

    static let light = SIMD3<Float>(1, 1, 0)

    let gpu: RasterLightingFixture
    let camera: PerspectiveCamera
    let pose: SceneTransform
    let projection: simd_float4x4
    let view: ShaderGraphCamera
    let backdrop: any MTLTexture
    let color: any MTLTexture
    let depth: any MTLTexture
    let floor: SceneNode
    private let shadowedIndex: Int
    private let openIndex: Int

    init() throws {
        gpu = try RasterLightingFixture(width: 132, height: 100)
        color = try gpu.colorTarget()
        depth = try gpu.depthTarget()
        let backdropDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.width, height: gpu.height, mipmapped: true
        )
        backdropDescriptor.storageMode = .private
        backdropDescriptor.usage = .shaderRead
        backdrop = try #require(gpu.device.makeTexture(descriptor: backdropDescriptor))
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 15)
        pose = RasterLightingFixture.cameraTransform(position: SIMD3(0, 6, 0), pitch: -.pi / 2)
        projection = simd_float4x4(
            SIMD4(0.25, 0, 0, 0), SIMD4(0, 1.0 / 3, 0, 0), SIMD4(0, 0, -1.0 / 15, 0), SIMD4(0, 0, 0, 1)
        )
        view = try ShaderGraphCamera(
            viewMatrix: simd_inverse(pose.matrix), projectionMatrix: projection, near: 0, far: 15
        )
        let diffuse = MeshMaterial.physicallyBased(try PhysicallyBasedMaterial(
            roughness: 1, ambientIntensity: 0.1, receivesShadows: true,
            transmission: .init(factor: 0, indexOfRefraction: 1)
        ))
        let floorMesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 7, height: 5.5), material: diffuse
        )
        floor = SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: floorMesh, castsShadow: false)
        let fixture = gpu, clip = projection, worldFromView = pose.matrix
        func index(_ point: SIMD3<Float>) -> Int {
            let pixel = fixture.project(point, clipFromView: clip, worldFromView: worldFromView)
            return Int(pixel.y) * fixture.width + Int(pixel.x)
        }
        shadowedIndex = index(SIMD3(-1, 0, 0))
        openIndex = index(SIMD3(1.5, 0, 0))
    }

    /// A one-unit-wide square at `center` facing up or down.
    func node(
        _ material: MeshMaterial, at center: SIMD3<Float> = SIMD3(0, 1, 0), facingUp: Bool = true, castsShadow: Bool = true,
        instances: InstanceSet? = nil
    ) throws -> SceneNode {
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.5, height: 0.5), material: material,
            supportsInstancing: instances != nil
        )
        let rotation = simd_quatf(angle: facingUp ? -.pi / 2 : .pi / 2, axis: SIMD3(1, 0, 0))
        return SceneNode(
            transform: SceneTransform(translation: center, rotation: rotation), mesh: mesh, castsShadow: castsShadow,
            instances: instances
        )
    }

    /// A graph material with a known base color. The caller's alpha mode and base alpha drive coverage.
    func graphMaterial(
        alphaMode: PhysicallyBasedMaterial.AlphaMode = .opaque, baseAlpha: Float = 1,
        _ build: (ShaderGraphBuilder) throws -> SurfaceShaderGraph.Result
    ) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(Self.baseColor, baseAlpha), alphaMode: alphaMode,
            isDoubleSided: false, surfaceGraph: SurfaceShaderGraph(building: build)
        )
    }

    static let baseColor = SIMD3<Float>(0.9, 0.4, 0.6)

    func shadowMap(layers: Int, cascaded: Bool) throws -> DirectionalShadowMap {
        if cascaded {
            let map = try DirectionalShadowMap(
                queue: gpu.queue, resolution: 192, directionToLight: Self.light,
                cascades: .init(count: 2, maximumDistance: 8, blendFraction: 0.2, casterMargin: 2),
                intensity: 2, sampling: .basic, updateMode: .onDemand, transmissionLayers: layers
            )
            try map.fitCascades(view: view, splitFractions: [0.65])
            return map
        }
        return try DirectionalShadowMap(
            queue: gpu.queue, resolution: 192, directionToLight: Self.light, halfExtent: 6,
            intensity: 2, sampling: .basic, updateMode: .onDemand, transmissionLayers: layers
        )
    }

    func render(_ casters: [SceneNode], layers: Int, cascaded: Bool = false) throws -> Rendered {
        try render(SceneGraph(roots: [floor] + casters, camera: camera), layers: layers, cascaded: cascaded)
    }

    func render(_ graph: SceneGraph, layers: Int, cascaded: Bool = false) throws -> Rendered {
        let map = try shadowMap(layers: layers, cascaded: cascaded)
        var passes = -1
        try gpu.run { command in
            let statistics = try graph.encode(
                pass: gpu.pass(color: color, depth: depth), commandBuffer: command,
                clipFromWorld: projection * view.viewMatrix, cameraPosition: pose.translation,
                directionToCamera: SIMD3(0, 1, 0), shadowMap: map, transmissionBackdrop: backdrop, graphCamera: view
            )
            passes = statistics.shadowPassCount
        }
        let pixels = try gpu.pixels(of: color)
        func rgb(_ index: Int) -> SIMD3<Float> {
            let value = pixels[index]
            return SIMD3(value.x, value.y, value.z)
        }
        return Rendered(shadowed: rgb(shadowedIndex), open: rgb(openIndex), passes: passes, map: map)
    }

    /// The floor's ambient color, from an opaque blocker, and its fully lit color.
    func references(cascaded: Bool = false) throws -> (ambient: SIMD3<Float>, open: SIMD3<Float>) {
        let blocker = try node(.unlit)
        let blocked = try render([blocker], layers: 0, cascaded: cascaded)
        let empty = try render([], layers: 0, cascaded: cascaded)
        #expect(empty.open.x > blocked.shadowed.x + 0.3, "blocked \(blocked.shadowed), open \(empty.open)")
        #expect(simd_distance(blocked.open, empty.open) < 0.006)
        return (blocked.shadowed, empty.open)
    }

    func depthBytes(_ map: DirectionalShadowMap, slices: Int = 1) async throws -> [Data] {
        var depths: [Data] = []
        for slice in 0..<slices {
            depths.append(try await TextureTransfer.read(from: map.texture, slice: slice, commandQueue: gpu.queue).bytes)
        }
        return depths
    }
}

private func tinted(
    _ references: (ambient: SIMD3<Float>, open: SIMD3<Float>), by filter: SIMD3<Float>
) -> SIMD3<Float> {
    references.ambient + (references.open - references.ambient) * filter
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GraphCastShadowTests {
    private let scene: CastShadowScene

    init() throws { scene = try CastShadowScene() }

    @Test("A graph caster tints its receiver with the authored RGB, stacks with a second caster and casts opaque shadows without layers",
          arguments: [false, true])
    func tintsReceiver(cascaded: Bool) throws {
        let references = try scene.references(cascaded: cascaded)
        let first = SIMD3<Float>(0.8, 0.3, 0.55)
        let second = SIMD3<Float>(0.5, 0.9, 0.7)
        let near = try scene.node(.physicallyBased(try scene.graphMaterial { s in
            .init(castShadowColor: s.constant(SIMD4<Float>(first, 1)))
        }))
        // The second pane sits on the same light ray, one unit higher and one unit toward the light.
        let far = try scene.node(.physicallyBased(try scene.graphMaterial { s in
            .init(castShadowColor: s.constant(SIMD4<Float>(second, 1)))
        }), at: SIMD3(1, 2, 0))
        let slices = cascaded ? 2 : 1

        // With no layers the ordinary depth path casts an opaque shadow.
        let opaque = try scene.render([near], layers: 0, cascaded: cascaded)
        #expect(simd_distance(opaque.shadowed, references.ambient) < 0.006, "\(opaque.shadowed)")
        #expect(simd_distance(opaque.open, references.open) < 0.006)
        #expect(opaque.passes == slices)

        let one = try scene.render([near], layers: 2, cascaded: cascaded)
        let expectedOne = tinted(references, by: first)
        #expect(simd_distance(one.shadowed, expectedOne) < 0.006, "actual \(one.shadowed) expected \(expectedOne)")
        #expect(simd_distance(one.open, references.open) < 0.006)
        #expect(one.passes == slices * 3)

        let both = try scene.render([far, near], layers: 2, cascaded: cascaded)
        let expectedBoth = tinted(references, by: first * second)
        #expect(simd_distance(both.shadowed, expectedBoth) < 0.006, "actual \(both.shadowed) expected \(expectedBoth)")
        #expect(simd_distance(both.open, references.open) < 0.006)
    }

    @Test("A shadow mask replaces the mask in the ordinary and the colored directional passes", arguments: [0, 1])
    func maskPrecedence(layers: Int) throws {
        let references = try scene.references()
        let filter = SIMD3<Float>(0.8, 0.3, 0.55)
        let shadowedColor = layers == 0 ? references.ambient : tinted(references, by: filter)
        let withShadowMask = try scene.graphMaterial { s in
            .init(
                mask: s.uniform("Visible", Float(1)), castShadowColor: s.constant(SIMD4<Float>(filter, 1)),
                castShadowMask: s.uniform("Casts", Float(1))
            )
        }
        let maskOnly = try scene.graphMaterial { s in
            .init(mask: s.uniform("Visible", Float(1)), castShadowColor: s.constant(SIMD4<Float>(filter, 1)))
        }
        func shadow(_ material: PhysicallyBasedMaterial, visible: Float? = nil, casts: Float? = nil) throws -> SIMD3<Float> {
            let uniforms = try #require(material.surfaceUniforms)
            if let visible { try uniforms.set("Visible", to: visible) }
            if let casts { try uniforms.set("Casts", to: casts) }
            return try scene.render([try scene.node(.physicallyBased(material))], layers: layers).shadowed
        }
        // `castShadowMask` wins either way: a visible mesh can opt out and a hidden mesh can still cast.
        for (visible, casts, casting) in [(Float(1), Float(1), true), (1, 0, false), (0, 1, true), (0, 0, false)] {
            let actual = try shadow(withShadowMask, visible: visible, casts: casts)
            let expected = casting ? shadowedColor : references.open
            #expect(simd_distance(actual, expected) < 0.006, "visible \(visible) casts \(casts): \(actual) expected \(expected)")
        }
        // Without one, the mask drives shadows as it always did.
        for (visible, casting) in [(Float(1), true), (0, false)] {
            let actual = try shadow(maskOnly, visible: visible)
            let expected = casting ? shadowedColor : references.open
            #expect(simd_distance(actual, expected) < 0.006, "mask only, visible \(visible): \(actual) expected \(expected)")
        }
        // A shadow mask alone gates every shadow pass, though the color pass has no coverage output at all.
        let shadowMaskOnly = try scene.graphMaterial { s in
            .init(castShadowColor: s.constant(SIMD4<Float>(filter, 1)), castShadowMask: s.uniform("Casts", Float(1)))
        }
        for (casts, casting) in [(Float(1), true), (0, false)] {
            let actual = try shadow(shadowMaskOnly, casts: casts)
            let expected = casting ? shadowedColor : references.open
            #expect(simd_distance(actual, expected) < 0.006, "shadow mask only, casts \(casts): \(actual) expected \(expected)")
        }
    }

    struct CoverageCase: Sendable, CustomTestStringConvertible {
        let mode: PhysicallyBasedMaterial.AlphaMode
        let baseAlpha: Float
        let authoredAlpha: Float
        /// The expected combined coverage. Zero means the mask cutoff rejects the fragment.
        let coverage: Float
        var testDescription: String { "\(mode) base alpha \(baseAlpha), authored alpha \(authoredAlpha)" }
    }

    static let coverageCases = [
        // Opaque ignores the base alpha, as stock glass does.
        CoverageCase(mode: .opaque, baseAlpha: 0.4, authoredAlpha: 1, coverage: 1),
        CoverageCase(mode: .opaque, baseAlpha: 0.4, authoredAlpha: 0.5, coverage: 0.5),
        // The cutoff tests the base alpha alone, and a passing fragment covers fully before the authored alpha.
        CoverageCase(mode: .mask(cutoff: 0.5), baseAlpha: 0.6, authoredAlpha: 1, coverage: 1),
        CoverageCase(mode: .mask(cutoff: 0.5), baseAlpha: 0.6, authoredAlpha: 0.5, coverage: 0.5),
        CoverageCase(mode: .mask(cutoff: 0.5), baseAlpha: 0.4, authoredAlpha: 1, coverage: 0),
        // Material alpha and blend scale coverage by the base alpha.
        CoverageCase(mode: .materialAlpha, baseAlpha: 0.4, authoredAlpha: 1, coverage: 0.4),
        CoverageCase(mode: .materialAlpha, baseAlpha: 0.4, authoredAlpha: 0.5, coverage: 0.2),
        CoverageCase(mode: .blend, baseAlpha: 0.4, authoredAlpha: 1, coverage: 0.4),
    ]

    @Test("Colored graph casters decide alpha and the mask cutoff like stock glass", arguments: coverageCases)
    func alphaMatchesStockGlass(_ item: CoverageCase) throws {
        let references = try scene.references()
        let graph = try scene.graphMaterial(alphaMode: item.mode, baseAlpha: item.baseAlpha) { s in
            .init(castShadowColor: s.vector(s.baseColor, s.constant(item.authoredAlpha)))
        }
        let actual = try scene.render([try scene.node(.physicallyBased(graph))], layers: 1).shadowed
        let filter = 1 - item.coverage * (1 - CastShadowScene.baseColor)
        let expected = item.coverage == 0 ? references.open : tinted(references, by: filter)
        #expect(simd_distance(actual, expected) < 0.006, "actual \(actual) expected \(expected)")

        if item.authoredAlpha == 1 {
            let glass = try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(CastShadowScene.baseColor, item.baseAlpha),
                alphaMode: item.mode, transmission: .init(factor: 1, indexOfRefraction: 1)
            )
            let stock = try scene.render([try scene.node(.physicallyBased(glass))], layers: 1).shadowed
            #expect(simd_distance(actual, stock) < 0.004, "graph \(actual) stock glass \(stock)")
        }
    }

    @Test("A single-sided graph caster casts through either face with or without colored layers", arguments: [true, false])
    func bothFacesCast(facingUp: Bool) throws {
        let references = try scene.references()
        let filter = SIMD3<Float>(0.8, 0.3, 0.55)
        let material = try scene.graphMaterial { s in .init(castShadowColor: s.constant(SIMD4<Float>(filter, 1))) }
        let caster = try scene.node(.physicallyBased(material), facingUp: facingUp)
        let opaque = try scene.render([caster], layers: 0).shadowed
        #expect(simd_distance(opaque, references.ambient) < 0.006, "opaque \(opaque)")
        let colored = try scene.render([caster], layers: 1).shadowed
        let expected = tinted(references, by: filter)
        #expect(simd_distance(colored, expected) < 0.006, "colored \(colored) expected \(expected)")
    }

    @Test("Authored cast outputs leave the ordinary opaque shadow depth byte for byte unchanged when layers are off")
    func ordinaryDepthUnchanged() async throws {
        let cutout = try scene.graphMaterial { s in
            .init(mask: s.uv0.x.lessThan(s.constant(Float(0.6))))
        }
        let authored = try scene.graphMaterial { s in
            .init(
                mask: s.uv0.x.lessThan(s.constant(Float(0.6))),
                castShadowColor: s.vector(s.baseColor, s.constant(Float(0.5)))
            )
        }
        let stock = try scene.node(.unlit, at: SIMD3(0, 1, 0))
        let beside = SIMD3<Float>(0, 1.5, 0)
        let reference = try scene.render([stock, try scene.node(.physicallyBased(cutout), at: beside)], layers: 0)
        let withOutputs = try scene.render([stock, try scene.node(.physicallyBased(authored), at: beside)], layers: 0)
        let referenceDepth = try await scene.depthBytes(reference.map)
        let emptyDepth = try await scene.depthBytes(try scene.render([], layers: 0).map)
        #expect(referenceDepth != emptyDepth, "The reference scene must write shadow depth.")
        #expect(try await scene.depthBytes(withOutputs.map) == referenceDepth)
        #expect(withOutputs.shadowed == reference.shadowed && withOutputs.open == reference.open)

        // Enabling layers adds color passes and leaves the opaque depth of stock casters alone.
        let layered = try scene.render([stock], layers: 3)
        let stockOnly = try scene.render([stock], layers: 0)
        #expect(layered.passes == 4 && stockOnly.passes == 1)
        let layeredDepth = try await scene.depthBytes(layered.map)
        let stockDepth = try await scene.depthBytes(stockOnly.map)
        #expect(layeredDepth == stockDepth)
    }

    @Test("An instanced graph caster tints through its instance transform")
    func instancedCaster() throws {
        let references = try scene.references()
        let filter = SIMD3<Float>(0.8, 0.3, 0.55)
        let material = try scene.graphMaterial { s in .init(castShadowColor: s.constant(SIMD4<Float>(filter, 1))) }
        // The node alone would shadow x = 2. Its only instance moves it back over the origin.
        let instances = try InstanceSet(
            device: scene.gpu.device, transforms: [SceneTransform(translation: SIMD3(-3, 0, 0)).matrix]
        )
        let caster = try scene.node(.physicallyBased(material), at: SIMD3(3, 1, 0), instances: instances)
        let opaque = try scene.render([caster], layers: 0).shadowed
        #expect(simd_distance(opaque, references.ambient) < 0.006, "opaque \(opaque)")
        let colored = try scene.render([caster], layers: 1).shadowed
        let expected = tinted(references, by: filter)
        #expect(simd_distance(colored, expected) < 0.006, "colored \(colored) expected \(expected)")
    }

    @Test("Camera-dependent coverage on a colored caster is rejected as a shadow caster, like other graph coverage")
    func cameraCoverageRejected() throws {
        let material = try scene.graphMaterial { s in
            .init(opacity: s.cameraPosition.x, castShadowColor: s.constant(SIMD4<Float>(0.8, 0.3, 0.55, 1)))
        }
        let caster = try scene.node(.physicallyBased(material))
        let graph = try #require(material.surfaceGraph)
        let input = try #require(graph.nodes.firstIndex { if case .input(.cameraPosition) = $0 { true } else { false } })
        #expect(throws: MeshRenderer.Failure.graphShadowCameraInput(caster.id, input)) {
            try scene.render([caster], layers: 1)
        }
    }

    @Test("A batched colored caster throws a typed error when layers are on and casts through the ordinary pass otherwise")
    func batchedCaster() throws {
        let references = try scene.references()
        let material = try scene.graphMaterial { s in .init(castShadowColor: s.constant(SIMD4<Float>(0.8, 0.3, 0.55, 1))) }
        let batch = try BatchedMesh(
            device: scene.gpu.device, colorPixelFormat: .rgba16Float, geometries: [.plane(width: 0.5, height: 0.5)],
            capacity: 2, material: .physicallyBased(material)
        )
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        try batch.addInstance(geometryIndex: 0, transform: SceneTransform(translation: SIMD3(0, 1, 0), rotation: rotation).matrix)
        let node = SceneNode(batch: batch)
        let ordinary = try scene.render([node], layers: 0).shadowed
        #expect(simd_distance(ordinary, references.ambient) < 0.006, "\(ordinary)")
        #expect(throws: MeshRenderer.Failure.graphCastShadowGeometry(node.id)) {
            try scene.render([node], layers: 1)
        }
        #expect(MeshRenderer.Failure.graphCastShadowGeometry(node.id).localizedDescription.contains("batched"))
    }

    // MARK: Archives

    @Test("Archive version 29 round-trips authored shadow outputs and renders the same tint")
    func archiveRoundTrip() throws {
        let material = try scene.graphMaterial { s in
            .init(
                baseColor: s.baseColor, castShadowColor: s.vector(s.baseColor, s.uniform("Alpha", Float(1))),
                castShadowMask: s.uniform("Casts", Float(1))
            )
        }
        let graph = SceneGraph(roots: [scene.floor, try scene.node(.physicallyBased(material))], camera: scene.camera)
        let archive = try SceneArchive(capturing: graph, commandQueue: scene.gpu.queue)
        let json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(json["version"] as? Int == 29)
        #expect(String(decoding: archive.manifest, as: UTF8.self).contains("castShadowMask"))

        let restored = try archive.load(device: scene.gpu.device, commandQueue: scene.gpu.queue).graph
        let caster = try #require(restored.roots.last?.mesh)
        guard case .physicallyBased(let restoredMaterial) = caster.material else {
            Issue.record("Expected the archived PBR material.")
            return
        }
        #expect(restoredMaterial.surfaceGraph == material.surfaceGraph)
        #expect(restoredMaterial.surfaceGraph?.outputs.castShadowColor == material.surfaceGraph?.outputs.castShadowColor)
        #expect(restoredMaterial.surfaceGraph?.outputs.castShadowMask == material.surfaceGraph?.outputs.castShadowMask)
        #expect(try SceneArchive(capturing: restored, commandQueue: scene.gpu.queue) == archive)

        let references = try scene.references()
        let original = try scene.render(graph, layers: 1)
        let reloaded = try scene.render(restored, layers: 1)
        let expected = tinted(references, by: CastShadowScene.baseColor)
        #expect(simd_distance(original.shadowed, expected) < 0.006, "original \(original.shadowed) expected \(expected)")
        #expect(original.shadowed == reloaded.shadowed && original.open == reloaded.open)

        // The version gate rejects the outputs below 29 and reads relabeled older manifests as before.
        var older = json
        older["version"] = 28
        #expect(throws: SceneArchive.Failure.decoding("surface graph colored shadow outputs require archive version 29")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: older), binary: archive.binary)
        }
    }

    @Test("Archives without authored shadow outputs keep their lowest version and an unchanged manifest")
    func lowestVersions() throws {
        func capture(_ build: (ShaderGraphBuilder) throws -> SurfaceShaderGraph.Result) throws -> SceneArchive {
            let material = try scene.graphMaterial(build)
            let graph = SceneGraph(roots: [try scene.node(.physicallyBased(material))], camera: scene.camera)
            return try SceneArchive(capturing: graph, commandQueue: scene.gpu.queue)
        }
        func version(_ archive: SceneArchive) throws -> Int? {
            try (JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])?["version"] as? Int
        }
        let plain = try capture { s in .init(baseColor: s.baseColor) }
        let coverage = try capture { s in .init(mask: s.constant(Float(1))) }
        let authored = try capture { s in .init(mask: s.constant(Float(1)), castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1))) }
        #expect(try version(plain) == 1)
        #expect(try version(coverage) == 7)
        #expect(try version(authored) == 29)
        for archive in [plain, coverage] {
            #expect(!String(decoding: archive.manifest, as: UTF8.self).contains("castShadow"))
            // Older manifests, even relabeled with a later version, still load unchanged.
            var json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
            json["version"] = 28
            let relabeled = try SceneArchive(manifest: JSONSerialization.data(withJSONObject: json), binary: archive.binary)
            _ = try relabeled.load(device: scene.gpu.device, commandQueue: scene.gpu.queue)
        }
        #expect(throws: SceneArchive.Failure.unsupportedVersion(30)) {
            var json = try #require(try JSONSerialization.jsonObject(with: authored.manifest) as? [String: Any])
            json["version"] = 30
            return try SceneArchive(manifest: JSONSerialization.data(withJSONObject: json), binary: authored.binary)
        }
    }

    @Test("Decoding rejects a colored shadow output that reads a camera input")
    func archiveRejectsCameraInput() throws {
        let material = try scene.graphMaterial { s in
            .init(castShadowColor: s.vector(s.baseColor, s.constant(Float(1))))
        }
        let graph = SceneGraph(roots: [try scene.node(.physicallyBased(material))], camera: scene.camera)
        let archive = try SceneArchive(capturing: graph, commandQueue: scene.gpu.queue)
        var json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        var graphs = try #require(json["surfaceGraphs"] as? [[String: Any]])
        var nodes = try #require(graphs[0]["nodes"] as? [[String: Any]])
        let index = try #require(nodes.firstIndex { $0["node"] as? String == "input" && $0["value"] as? String == "baseColor" })
        nodes[index]["value"] = "cameraPosition"
        graphs[0]["nodes"] = nodes
        json["surfaceGraphs"] = graphs
        do {
            _ = try SceneArchive(manifest: JSONSerialization.data(withJSONObject: json), binary: archive.binary)
            Issue.record("A camera read in a colored shadow output was accepted.")
        } catch SceneArchive.Failure.decoding(let message) {
            #expect(message.contains("castShadowColor") && message.contains("camera"), "\(message)")
        }
    }

    // MARK: Other consumers of surface outputs

    @Test("Path tracing and raster color ignore authored shadow outputs, including a mask of zero")
    func pathTracingIgnoresCastOutputs() throws {
        let fixture = try PathTestFixture()
        let graph = try SurfaceShaderGraph { s in
            .init(
                emission: s.constant(SIMD3<Float>(0.75, 0.25, 0)),
                castShadowColor: s.constant(SIMD4<Float>(0, 0, 0, 1)), castShadowMask: s.constant(Float(0))
            )
        }
        let material = try PhysicallyBasedMaterial(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph)
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(material)
        )
        let sceneGraph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: fixture.camera
        )
        let side = 33
        let center = side / 2 * side + side / 2
        let raster = try fixture.rasterize(sceneGraph, width: side, height: side)[center]
        let tracer = try PathTracer(queue: fixture.queue, scene: try PathTracingScene(queue: fixture.queue, graph: sceneGraph))
        let traced = try fixture.trace(
            tracer, width: side, height: side, samples: 4, settings: PathTracer.Settings(maxBounces: 1, seed: 3)
        )[center]
        let expected = SIMD4<Float>(0.75, 0.25, 0, 1)
        #expect(simd_distance(raster, expected) < 0.002, "raster \(raster)")
        #expect(simd_distance(traced, expected) < 0.002, "path \(traced)")
    }

    @Test("Exporters report authored shadow outputs as dropped graph channels instead of failing")
    func exportersReportDroppedOutputs() throws {
        let material = try scene.graphMaterial { s in
            .init(
                baseColor: s.baseColor, castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1)),
                castShadowMask: s.constant(Float(1))
            )
        }
        let graph = SceneGraph(roots: [try scene.node(.physicallyBased(material))], camera: scene.camera)

        let gltf = try GLTFExporter(commandQueue: scene.gpu.queue).export(graph, includesCamera: false)
        let dropped = GLTFExporter.Warning.surfaceGraphChannelsDropped(
            material: 0, channels: [.baseColor, .castShadowColor, .castShadowMask]
        )
        #expect(gltf.warnings.contains(dropped), "\(gltf.warnings)")
        #expect(dropped.description.contains("castShadowColor, castShadowMask"))

        let usdz = try USDZExporter(commandQueue: scene.gpu.queue).export(graph, includesCamera: false)
        #expect(usdz.warnings.contains { warning in
            if case .materialFeatureDropped(_, .surfaceGraph) = warning { true } else { false }
        }, "\(usdz.warnings)")

        // A graph with only shadow outputs still exports its base material and names what it dropped.
        let shadowOnly = try scene.graphMaterial { s in .init(castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1))) }
        let onlyGraph = SceneGraph(roots: [try scene.node(.physicallyBased(shadowOnly))], camera: scene.camera)
        let only = try GLTFExporter(commandQueue: scene.gpu.queue).export(onlyGraph, includesCamera: false)
        #expect(only.warnings.contains(.surfaceGraphChannelsDropped(material: 0, channels: [.castShadowColor])))
    }
}

@Suite
struct GraphCastShadowDefinitionTests {
    private func failure(_ build: (ShaderGraphBuilder) throws -> SurfaceShaderGraph.Result) -> SurfaceShaderGraph.Failure? {
        do {
            _ = try SurfaceShaderGraph(building: build)
            return nil
        } catch let failure as SurfaceShaderGraph.Failure {
            return failure
        } catch {
            Issue.record("Unexpected error \(error)")
            return nil
        }
    }

    @Test("The typed builder authors both outputs, and a shadow-only graph has no per-pixel surface outputs")
    func typedBuilder() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(
                baseColor: s.baseColor, castShadowColor: s.vector(s.baseColor, s.constant(Float(0.5))),
                castShadowMask: s.constant(Float(1))
            )
        }
        #expect(graph.outputs.castShadowColor != nil && graph.outputs.castShadowMask != nil)
        #expect(graph.hasSurfaceOutputs)
        let shadowOnly = try SurfaceShaderGraph { s in .init(castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1))) }
        #expect(!shadowOnly.hasSurfaceOutputs && !shadowOnly.hasVertexOutputs)
        #expect(shadowOnly.outputs.castShadowColor != nil)
    }

    @Test("Shadow coverage source is the color-pass coverage source unless a shadow mask replaces the mask")
    func coverageSource() throws {
        func coverage(_ graph: SurfaceShaderGraph) throws -> (shadow: String, color: String) {
            (try graph.metalShadowCoverageFunction(named: "coverage"), try graph.metalCoverageFunction(named: "coverage"))
        }
        let plain = try SurfaceShaderGraph { s in
            .init(
                opacity: s.constant(Float(0.5)), alphaTest: s.constant(Float(0.2)),
                mask: s.uv0.x.lessThan(s.constant(Float(0.5)))
            )
        }
        let colored = try SurfaceShaderGraph { s in
            .init(
                opacity: s.constant(Float(0.5)), alphaTest: s.constant(Float(0.2)),
                mask: s.uv0.x.lessThan(s.constant(Float(0.5))), castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1))
            )
        }
        for graph in [plain, colored] {
            let sources = try coverage(graph)
            #expect(sources.shadow == sources.color)
            #expect(graph.hasShadowCoverageOutputs == graph.hasCoverageOutputs)
            #expect(graph.hasShadowCutoutOutputs == graph.hasCutoutOutputs)
        }
        let replaced = try SurfaceShaderGraph { s in
            .init(
                mask: s.constant(Float(1)), castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1)),
                castShadowMask: s.uv0.x.lessThan(s.constant(Float(0.5)))
            )
        }
        let sources = try coverage(replaced)
        #expect(sources.shadow != sources.color)
        let shadowMaskOnly = try SurfaceShaderGraph { s in
            .init(castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1)), castShadowMask: s.constant(Float(1)))
        }
        #expect(!shadowMaskOnly.hasCoverageOutputs && shadowMaskOnly.hasShadowCoverageOutputs)
        #expect(shadowMaskOnly.hasShadowCutoutOutputs && !shadowMaskOnly.hasCutoutOutputs)
    }

    @Test("A shadow mask needs a colored shadow output")
    func maskWithoutColor() {
        let result = failure { s in .init(baseColor: s.baseColor, castShadowMask: s.constant(Float(1))) }
        #expect(result == SurfaceShaderGraph.Failure.shadowMaskWithoutColor)
    }

    @Test("Camera, viewport and shadow-visibility inputs in colored shadow outputs fail at construction with a named channel")
    func unsupportedInputs() {
        let builders: [(String, (ShaderGraphBuilder) throws -> SurfaceShaderGraph.Result)] = [
            ("camera position", { s in .init(castShadowColor: s.vector(s.cameraPosition, s.constant(Float(1)))) }),
            ("viewport color", { s in .init(castShadowColor: s.sceneColor(at: s.screenUV)) }),
            ("shadow visibility", { s in .init(castShadowColor: s.vector(s.baseColor, s.directionalShadowVisibility)) }),
        ]
        for (name, build) in builders {
            guard case .castShadowInput(let channel, _)? = failure(build) else {
                Issue.record("\(name) in castShadowColor was not rejected at construction.")
                continue
            }
            #expect(channel == .castShadowColor, "\(name)")
        }
        let mask = failure { s in
            .init(
                castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1)),
                castShadowMask: s.cameraPosition.x.greaterThan(s.constant(Float(0)))
            )
        }
        guard case .castShadowInput(let channel, _)? = mask else {
            Issue.record("A camera read in castShadowMask was not rejected at construction.")
            return
        }
        #expect(channel == .castShadowMask)
        #expect(mask?.errorDescription?.contains("castShadowMask") == true)
        let visibility = failure { s in
            .init(
                castShadowColor: s.constant(SIMD4<Float>(1, 0, 0, 1)),
                castShadowMask: s.directionalShadowVisibility
            )
        }
        #expect(visibility == SurfaceShaderGraph.Failure.directionalShadowCoverage)
        // Ordinary per-pixel inputs stay available to the authored output.
        let ordinary = failure { s in
            .init(castShadowColor: s.vector(s.worldPosition, s.uv0.x))
        }
        #expect(ordinary == nil)
    }
}
