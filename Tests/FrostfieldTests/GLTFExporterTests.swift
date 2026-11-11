import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GLTFExporterTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 50)
    }

    @Test
    func `Round-trips a procedural scene graph through GLB bytes`() throws {
        let fixture = try ProceduralFixture(device: device, camera: camera)
        let output = try GLTFExporter(commandQueue: queue).export(
            fixture.graph, sceneName: "Procedural", nodeNames: fixture.names,
            animations: [.init(name: "Bob", animation: fixture.animation)]
        )
        #expect(output.format == .glb)
        #expect(output.warnings.isEmpty)

        // Container and document structure.
        let (document, _) = try inspectGLB(output.data)
        #expect(document["extensionsUsed"] as? [String] == [
            "KHR_lights_punctual", "KHR_materials_clearcoat", "KHR_materials_ior",
            "KHR_materials_transmission", "KHR_materials_volume",
        ])
        #expect((document["images"] as? [Any])?.count == 3)
        #expect((document["textures"] as? [Any])?.count == 3)
        #expect((document["samplers"] as? [Any])?.count == 1)
        let accessors = try #require(document["accessors"] as? [[String: Any]])
        let meshRecords = try #require(document["meshes"] as? [[String: Any]])
        try #require(meshRecords.count == 2)
        #expect(fixture.boxGeometry.vertices.allSatisfy { $0.tangent != nil })
        for (record, geometry) in zip(meshRecords, [fixture.boxGeometry, fixture.sphereGeometry]) {
            let primitive = try #require((record["primitives"] as? [[String: Any]])?.first)
            let attributes = try #require(primitive["attributes"] as? [String: Int])
            // The bevel box authors tangents; the UV sphere leaves them to the renderer.
            let tangents: Set<String> = geometry.vertices.allSatisfy { $0.tangent != nil } ? ["TANGENT"] : []
            #expect(Set(attributes.keys) == Set(["POSITION", "NORMAL", "TEXCOORD_0"]).union(tangents))
            let position = accessors[try #require(attributes["POSITION"])]
            let points = geometry.vertices.map(\.position)
            let minimum = points.reduce(points[0], simd_min), maximum = points.reduce(points[0], simd_max)
            #expect((position["min"] as? [Double])?.map(Float.init) == [minimum.x, minimum.y, minimum.z])
            #expect((position["max"] as? [Double])?.map(Float.init) == [maximum.x, maximum.y, maximum.z])
            #expect(accessors[try #require(primitive["indices"] as? Int)]["componentType"] as? Int == 5_123)
        }
        let animation = try #require((document["animations"] as? [[String: Any]])?.first)
        let samplers = try #require(animation["samplers"] as? [[String: Any]])
        #expect(samplers.count == 3)
        #expect(Set(samplers.compactMap { $0["input"] as? Int }).count == 1)
        #expect(samplers.allSatisfy { $0["interpolation"] as? String == "LINEAR" })

        let scene = try load(output.data)

        // Hierarchy, names, and transforms.
        let group = try #require(output.nodeIndices[fixture.group.id])
        let box = try #require(output.nodeIndices[fixture.box.id])
        let sphere = try #require(output.nodeIndices[fixture.sphere.id])
        let spot = try #require(output.nodeIndices[fixture.spot.id])
        let cameraNode = try #require(output.cameraNodeIndex)
        #expect([group, box, sphere, spot, cameraNode] == Array(0..<5))
        #expect(scene.name == "Procedural")
        #expect(scene.rootNodeIndices == [group, spot, cameraNode])
        #expect(scene.nodes.map(\.name) == ["Group", "Box", "Sphere", "Spot", "Camera"])
        #expect(scene.nodes.map(\.parentIndex) == [nil, group, box, nil, nil])
        #expect(scene.nodes.map(\.childIndices) == [[box], [sphere], [], [], []])
        for (index, source) in [fixture.group, fixture.box, fixture.sphere, fixture.spot].enumerated() {
            #expect(close(scene.nodes[index].localTransform, source.transform))
        }
        #expect(close(scene.nodes[cameraNode].localTransform, fixture.graph.cameraTransform))

        // Vertex data and indices.
        let boxMesh = try #require(scene.meshes.first { $0.sourceIndex == scene.nodes[box].meshIndex })
        let sphereMesh = try #require(scene.meshes.first { $0.sourceIndex == scene.nodes[sphere].meshIndex })
        expectSameGeometry(try #require(boxMesh.primitives.first), fixture.boxGeometry)
        expectSameGeometry(try #require(sphereMesh.primitives.first), fixture.sphereGeometry)

        // Generated maps keep their pixels, color encoding, and shared occlusion image.
        let textured = try material(boxMesh, in: scene)
        #expect(textured.baseColorFactor == SIMD4(repeating: 1))
        #expect(textured.metallicFactor == 1)
        #expect(textured.roughnessFactor == 1)
        #expect(textured.normalScale == 0.75)
        #expect(textured.occlusionStrength == 1)
        #expect(textured.alphaMode == .opaque)
        #expect(!textured.isDoubleSided)
        #expect(textured.clearcoat == nil && textured.transmission == nil && textured.volume == nil)
        #expect(textured.indexOfRefraction == 1.5)
        let base = try #require(textured.baseColorTexture)
        let surfaceData = try #require(textured.metallicRoughnessTexture)
        let normal = try #require(textured.normalTexture)
        #expect(textured.occlusionTexture?.image.sourceIndex == surfaceData.image.sourceIndex)
        #expect(textured.emissiveTexture == nil)
        for texture in [base, surfaceData, normal] {
            #expect(texture.textureCoordinateSet == .primary)
            #expect(texture.image.mimeType == "image/png")
            #expect(texture.sampling.magnification == .linear && texture.sampling.minification == .linearMipmapLinear)
            #expect(texture.sampling.wrapS == .repeat && texture.sampling.wrapT == .repeat)
        }
        #expect(maximumDifference(
            try decode(base.image.data, .encodedSRGB), pixels(fixture.surface.baseColorTexture)
        ) <= 1)
        #expect(maximumDifference(
            try decode(surfaceData.image.data, .linearData), pixels(fixture.surface.metallicRoughnessTexture)
        ) <= 1)
        #expect(maximumDifference(try decode(normal.image.data, .linearData), pixels(fixture.surface.normalTexture)) <= 1)

        // Factors and optical extensions.
        let glass = try material(sphereMesh, in: scene)
        #expect(glass.baseColorFactor == SIMD4(0.8, 0.3, 0.2, 0.9))
        #expect(glass.metallicFactor == 0.25)
        #expect(glass.roughnessFactor == 0.35)
        #expect(glass.emissiveFactor == SIMD3(0.05, 0.1, 0.15))
        #expect(glass.alphaMode == .blend)
        #expect(glass.baseColorTexture == nil)
        #expect(glass.clearcoat?.factor == 0.6)
        #expect(glass.clearcoat?.roughness == 0.15)
        #expect(glass.transmission?.factor == 0.4)
        #expect(glass.volume?.thickness == 0.2)
        #expect(glass.volume?.attenuationDistance == 2)
        #expect(glass.volume?.attenuationColor == SIMD3(0.9, 0.8, 0.7))
        #expect(glass.indexOfRefraction == 1.45)

        // Light and camera parameters.
        try #require(scene.lights.count == 1)
        #expect(scene.lights[0].light == fixture.spotLight)
        #expect(scene.nodes[spot].lightIndex == scene.lights[0].sourceIndex)
        try #require(scene.cameras.count == 1)
        #expect(scene.cameras[0].projection == .perspective(verticalFieldOfView: .pi / 3, aspectRatio: nil, near: 0.1, far: 50))
        #expect(scene.nodes[cameraNode].cameraIndex == scene.cameras[0].sourceIndex)

        // Animation samples, including before, between, at, and after keys.
        let clip = try #require(scene.animations.first)
        #expect(clip.name == "Bob")
        #expect(clip.duration == fixture.animation.duration)
        #expect(clip.channels.map(\.target) == [.translation, .rotation, .scale])
        #expect(clip.channels.allSatisfy { $0.interpolation == .linear && $0.sourceNodeIndex == sphere })
        let (poseGraph, ids) = try poses(scene)
        let sphereID = try #require(ids[sphere])
        for time: Float in [-1, 0, 0.2, 0.5, 0.8, 1.25, 3] {
            let expected = try transform(of: fixture.sphere.id, in: fixture.animation.applying(to: fixture.graph, at: time))
            let actual = try transform(of: sphereID, in: clip.applying(to: poseGraph, nodeIDs: ids, at: time))
            #expect(close(actual, expected, tolerance: 0.000_01), "time \(time)")
        }
    }

    @Test
    func `Re-exports an imported scene as embedded glTF without changing it`() throws {
        let fixture = try ProceduralFixture(device: device, camera: camera)
        let binary = try GLTFExporter(commandQueue: queue).export(
            fixture.graph, sceneName: "Procedural", nodeNames: fixture.names,
            animations: [.init(name: "Bob", animation: fixture.animation)]
        )
        let first = try load(binary.data)
        let output = try GLTFExporter(format: .gltf).export(first)
        #expect(output.format == .gltf)
        #expect(output.warnings.isEmpty)
        #expect(output.nodeIndices.isEmpty && output.cameraNodeIndex == nil)
        #expect(output.sourceNodeIndices == Dictionary(uniqueKeysWithValues: first.nodes.map { ($0.sourceIndex, $0.sourceIndex) }))

        let document = try #require(JSONSerialization.jsonObject(with: output.data) as? [String: Any])
        #expect((document["asset"] as? [String: String])?["generator"] == "Frostfield")
        let buffer = try #require((document["buffers"] as? [[String: Any]])?.first)
        #expect((buffer["uri"] as? String)?.hasPrefix("data:application/octet-stream;base64,") == true)
        let images = try #require(document["images"] as? [[String: Any]])
        #expect(images.count == 3)
        #expect(images.allSatisfy { ($0["uri"] as? String)?.hasPrefix("data:image/png;base64,") == true && $0["bufferView"] == nil })

        let second = try load(output.data)
        expectSameScene(first, second)

        // A geometry-only import keeps material references it did not load, and export reports them.
        let geometryOnly = try GLTFExporter().export(GLTFMeshLoader.loadScene(data: binary.data))
        #expect(geometryOnly.warnings == [
            .materialNotLoaded(mesh: 0, primitive: 0, sourceMaterial: 0),
            .materialNotLoaded(mesh: 1, primitive: 0, sourceMaterial: 1),
        ])
        #expect(try GLTFMeshLoader.loadScene(data: geometryOnly.data, includeMaterials: true).materials.isEmpty)

        let (firstGraph, firstIDs) = try poses(first)
        let (secondGraph, secondIDs) = try poses(second)
        let firstClip = try #require(first.animations.first), secondClip = try #require(second.animations.first)
        for time: Float in [0, 0.3, 0.5, 1, 1.25] {
            let a = try firstClip.applying(to: firstGraph, nodeIDs: firstIDs, at: time)
            let b = try secondClip.applying(to: secondGraph, nodeIDs: secondIDs, at: time)
            for node in first.nodes {
                let id = try #require(firstIDs[node.sourceIndex]), other = try #require(secondIDs[node.sourceIndex])
                #expect(close(try transform(of: id, in: a), try transform(of: other, in: b), tolerance: 0))
            }
        }
    }

    @Test
    func `Keeps imported skins and spline and step animation through a GLTFScene export`() throws {
        let data = try skinnedAsset()
        let first = try GLTFMeshLoader.loadScene(data: data, includeAnimations: true, includeSkins: true)
        let output = try GLTFExporter().export(first)
        #expect(output.warnings.isEmpty)
        let (document, _) = try inspectGLB(output.data)
        let accessors = try #require(document["accessors"] as? [[String: Any]])
        let primitive = try #require(((document["meshes"] as? [[String: Any]])?.first?["primitives"] as? [[String: Any]])?.first)
        let attributes = try #require(primitive["attributes"] as? [String: Int])
        #expect(accessors[try #require(attributes["JOINTS_0"])]["componentType"] as? Int == 5_121)
        #expect(accessors[try #require(attributes["WEIGHTS_0"])]["componentType"] as? Int == 5_126)
        let nodes = try #require(document["nodes"] as? [[String: Any]])
        #expect(nodes[1]["matrix"] == nil && nodes[1]["rotation"] != nil)

        let second = try GLTFMeshLoader.loadScene(data: output.data, includeAnimations: true, includeSkins: true)
        expectSameScene(first, second)
        let skin = try #require(second.skins.first)
        #expect(skin.name == "Rig")
        #expect(skin.jointNodeIndices == [2, 1])
        #expect(skin.skeletonRootNodeIndex == 1)
        #expect(skin.inverseBindMatrices == first.skins.first?.inverseBindMatrices)
        #expect(second.meshes.first?.primitiveSkinInfluences.first ?? nil == first.meshes.first?.primitiveSkinInfluences.first ?? nil)
        let clip = try #require(second.animations.first)
        #expect(clip.channels.map(\.interpolation) == [.cubicSpline, .step])

        let (firstGraph, firstIDs) = try poses(first)
        let (secondGraph, secondIDs) = try poses(second)
        let firstClip = try #require(first.animations.first)
        for time: Float in [0, 0.2, 0.25, 0.5, 0.75, 1] {
            let a = try transform(of: try #require(firstIDs[2]), in: firstClip.applying(to: firstGraph, nodeIDs: firstIDs, at: time))
            let b = try transform(of: try #require(secondIDs[2]), in: clip.applying(to: secondGraph, nodeIDs: secondIDs, at: time))
            #expect(close(a, b, tolerance: 0), "time \(time)")
        }
    }

    @Test
    func `Surface graph materials export their base material and report every lossy conversion`() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(
                baseColor: s.constant(SIMD3<Float>(0.9, 0.1, 0.1)), roughness: s.constant(Float(0.2)),
                positionOffset: s.constant(SIMD3<Float>(0, 0.5, 0))
            )
        }
        let shaded = try PhysicallyBasedMaterial(
            metallic: 0.3, roughness: 0.6, baseColorFactor: SIMD4(0.2, 0.4, 0.6, 1), alphaMode: .opaque, surfaceGraph: graph
        )
        let plane = try MeshGeometry.plane()
        let graphRenderer = try renderer(plane, material: .physicallyBased(shaded))
        let tinted = try renderer(.cube(), material: .physicallyBased(PhysicallyBasedMaterial(
            roughness: 0.5, baseColorFactor: SIMD4(0.8, 0.6, 1, 1), alphaMode: .opaque
        )))
        let lambert = try renderer(.cube(), material: .lambert)
        let sampler = try #require(device.makeSamplerState(descriptor: MTLSamplerDescriptor()))
        let sampled = try renderer(.cube(), material: .physicallyBased(PhysicallyBasedMaterial(
            baseColorTexture: texture(.rgba8Unorm, width: 1, height: 1, bytes: [10, 20, 30, 255]),
            baseColorSampler: sampler, alphaMode: .opaque
        )))
        let roots = [
            SceneNode(mesh: graphRenderer, materialParameters: SIMD4(0.5, 0.5, 0.5, 1)),
            SceneNode(transform: .init(translation: SIMD3(2, 0, 0)), mesh: tinted, materialParameters: SIMD4(0.5, 1, 0.25, 1)),
            SceneNode(transform: .init(translation: SIMD3(4, 0, 0)), mesh: tinted),
            SceneNode(transform: .init(translation: SIMD3(-2, 0, 0)), mesh: lambert),
            SceneNode(transform: .init(translation: SIMD3(-4, 0, 0)), mesh: sampled),
        ]
        let output = try GLTFExporter(commandQueue: queue).export(SceneGraph(roots: roots, camera: camera), includesCamera: false)
        #expect(output.warnings == [
            .surfaceGraphChannelsDropped(material: 0, channels: [.baseColor, .roughness]),
            .deformationBaked(mesh: 0),
            .unsupportedMaterial(mesh: 3),
            .samplerUnavailable(material: 3),
        ])
        #expect(output.warnings.allSatisfy { !$0.description.isEmpty })
        #expect(output.warnings.first?.description.contains("baseColor, roughness") == true)

        let scene = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true)
        try #require(scene.materials.count == 4 && scene.meshes.count == 5)
        // The authored factors survive; the graph's color, roughness, and per-draw inputs do not.
        #expect(scene.materials[0].baseColorFactor == SIMD4(0.2, 0.4, 0.6, 1))
        #expect(scene.materials[0].metallicFactor == 0.3)
        #expect(scene.materials[0].roughnessFactor == 0.6)
        // Without a graph, each draw's tint multiplies the base color factor.
        #expect(scene.materials[1].baseColorFactor == SIMD4(0.4, 0.6, 0.25, 1))
        #expect(scene.materials[2].baseColorFactor == SIMD4(0.8, 0.6, 1, 1))
        #expect(scene.meshes.map(\.primitiveMaterialIndices) == [[0], [1], [2], [nil], [3]])
        // The exported plane matches the drawn, deformed vertices.
        let deformed = try #require(scene.meshes[0].primitives.first)
        #expect(deformed.indices == plane.indices)
        #expect(zip(deformed.vertices, plane.vertices).allSatisfy { simd_distance($0.position, $1.position + SIMD3(0, 0.5, 0)) < 0.000_01 })
        // Tinted copies of one renderer share vertex accessors.
        let (document, _) = try inspectGLB(output.data)
        let records = try #require(document["meshes"] as? [[String: Any]])
        let first = try #require((records[1]["primitives"] as? [[String: Any]])?.first?["attributes"] as? [String: Int])
        let second = try #require((records[2]["primitives"] as? [[String: Any]])?.first?["attributes"] as? [String: Int])
        #expect(first == second)

        let missing = UUID()
        let stray = try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: missing, keyframes: [.init(time: 0, transform: .init())])])
        #expect(throws: GLTFExporter.Failure.missingAnimationTarget(missing)) {
            try GLTFExporter().export(SceneGraph(roots: roots, camera: camera), animations: [.init(animation: stray)])
        }
    }

    @Test
    func `Encodes texture formats with importer color conventions and rejects others`() throws {
        let exporter = GLTFExporter(commandQueue: queue)
        // A linear color map is sRGB-encoded. Alpha stays linear and straight, even at zero.
        let linear: [UInt8] = [0, 10, 128, 255, 200, 50, 3, 128, 255, 255, 255, 0, 64, 32, 16, 8]
        let linearMap = try texture(.rgba8Unorm, width: 2, height: 2, bytes: linear)
        let encoded = try decode(exporter.pngData(from: linearMap), .encodedSRGB)
        let expectedSRGB = linear.enumerated().map { index, code in
            index % 4 == 3 ? code : UInt8((Self.encodeSRGB(Float(code) / 255) * 255).rounded())
        }
        #expect(maximumDifference(encoded, expectedSRGB) <= 1)
        #expect(maximumDifference(try decode(exporter.pngData(from: linearMap, interpretation: .linearData), .linearData), linear) == 0)

        // sRGB storage keeps its codes for color and decodes for data. BGRA is swizzled.
        let stored: [UInt8] = [30, 60, 90, 255, 250, 128, 5, 200]
        let bgra = try texture(.bgra8Unorm_srgb, width: 2, height: 1, bytes: stored)
        let swizzled: [UInt8] = [90, 60, 30, 255, 5, 128, 250, 200]
        #expect(try decode(exporter.pngData(from: bgra, interpretation: .color), .encodedSRGB) == swizzled)
        let decodedLinear = swizzled.enumerated().map { index, code in
            index % 4 == 3 ? code : UInt8((Self.decodeSRGB(Float(code) / 255) * 255).rounded())
        }
        #expect(maximumDifference(try decode(exporter.pngData(from: bgra, interpretation: .linearData), .linearData), decodedLinear) <= 1)

        // A private half-float map is copied on the queue, clamped, and quantized linearly.
        let halves: [Float16] = [0.25, 0.5, 2, -1, 1, 0, 0.75, 0.5]
        let half = try privateTexture(.rgba16Float, width: 2, height: 1, bytes: halves.flatMap {
            withUnsafeBytes(of: $0.bitPattern.littleEndian) { Array($0) }
        })
        #expect(try decode(exporter.pngData(from: half, interpretation: .linearData), .linearData) == [64, 128, 255, 0, 255, 0, 191, 128])

        // Two-channel data expands as Metal samples it.
        let pair = try texture(.rg8Unorm, width: 1, height: 1, bytes: [10, 20])
        #expect(try decode(exporter.pngData(from: pair, interpretation: .linearData), .linearData) == [10, 20, 0, 255])

        let packed = try texture(.rgb10a2Unorm, width: 1, height: 1, bytes: [0, 0, 0, 0])
        #expect(throws: GLTFExporter.Failure.unsupportedTextureFormat(.rgb10a2Unorm)) {
            try exporter.pngData(from: packed)
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        descriptor.textureType = .type2DArray
        descriptor.arrayLength = 2
        let array = try #require(device.makeTexture(descriptor: descriptor))
        #expect(throws: GLTFExporter.Failure.unsupportedTexture("Only single-sample 2D textures can be exported.")) {
            try exporter.pngData(from: array)
        }
    }

    @Test
    func `GLB chunks are aligned and padded, and large meshes use 32-bit indices`() throws {
        let triangle = try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(0, 0, 0), normal: SIMD3(0, 0, 1), secondaryTextureCoordinate: SIMD2(0.25, 0.5)),
            MeshVertex(
                position: SIMD3(1, 0, 0), normal: SIMD3(0, 0, 2), textureCoordinate: SIMD2(1, 0),
                secondaryTextureCoordinate: SIMD2(0.75, 0.5)
            ),
            MeshVertex(
                position: SIMD3(0, 1, 0), normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(0, 1),
                secondaryTextureCoordinate: SIMD2(0.25, 1)
            ),
        ], indices: [0, 1, 2])
        let small = SceneGraph(roots: [SceneNode(mesh: try renderer(triangle, material: .physicallyBased(.init(alphaMode: .opaque))))], camera: camera)
        let output = try GLTFExporter().export(small, includesCamera: false)
        #expect(output.warnings.isEmpty)
        let (document, binary) = try inspectGLB(output.data)
        // Four float attributes and three UInt16 indices need 126 bytes and two bytes of padding.
        #expect((document["buffers"] as? [[String: Any]])?.first?["byteLength"] as? Int == 126)
        #expect(binary.count == 128)
        let imported = try #require(GLTFMeshLoader.loadScene(data: output.data).meshes.first?.primitives.first)
        // Non-unit normals are normalized; already-unit values are written unchanged.
        #expect(imported.vertices.map(\.normal) == Array(repeating: SIMD3(0, 0, 1), count: 3))
        #expect(imported.vertices.map(\.secondaryTextureCoordinate) == triangle.vertices.map(\.secondaryTextureCoordinate))
        #expect(imported.indices == [0, 1, 2])

        // An attribute only some vertices carry is omitted and reported.
        let partial = try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(0, 0, 0), normal: SIMD3(0, 0, 1), tangent: SIMD4(1, 0, 0, 1)),
            MeshVertex(position: SIMD3(1, 0, 0), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0, 1, 0), normal: SIMD3(0, 0, 1)),
        ], indices: [0, 1, 2])
        let partialGraph = SceneGraph(roots: [SceneNode(mesh: try renderer(partial, material: .physicallyBased(.init())))], camera: camera)
        let dropped = try GLTFExporter().export(partialGraph, includesCamera: false)
        #expect(dropped.warnings == [.attributeDropped(mesh: 0, primitive: 0, attribute: "TANGENT")])
        #expect(try GLTFMeshLoader.loadScene(data: dropped.data).meshes.first?.primitives.first?.vertices.allSatisfy { $0.tangent == nil } == true)

        let embedded = try GLTFExporter(format: .gltf).export(small, includesCamera: false)
        #expect(embedded.data.first == UInt8(ascii: "{"))
        #expect(try GLTFMeshLoader.loadScene(data: embedded.data).meshes.first?.primitives.first?.vertices.map(\.position)
            == triangle.vertices.map(\.position))

        let grid = try MeshGeometry.plane(width: 4, height: 4, columns: 256, rows: 256)
        try #require(grid.vertices.count > Int(UInt16.max))
        let large = SceneGraph(roots: [SceneNode(mesh: try renderer(grid, material: .unlit))], camera: camera)
        let wide = try GLTFExporter().export(large, includesCamera: false)
        #expect(wide.warnings == [.unsupportedMaterial(mesh: 0)])
        let (wideDocument, _) = try inspectGLB(wide.data)
        let accessors = try #require(wideDocument["accessors"] as? [[String: Any]])
        let primitive = try #require(((wideDocument["meshes"] as? [[String: Any]])?.first?["primitives"] as? [[String: Any]])?.first)
        #expect(accessors[try #require(primitive["indices"] as? Int)]["componentType"] as? Int == 5_125)
        #expect(try GLTFMeshLoader.loadScene(data: wide.data).meshes.first?.primitives.first?.indices == grid.indices)
    }

    // MARK: Fixtures

    private struct ProceduralFixture {
        let surface: ProceduralSurface
        let boxGeometry: MeshGeometry
        let sphereGeometry: MeshGeometry
        let group: SceneNode
        let box: SceneNode
        let sphere: SceneNode
        let spot: SceneNode
        let spotLight: PunctualLight
        let graph: SceneGraph
        let animation: SceneAnimation
        let names: [UUID: String]

        init(device: any MTLDevice, camera: PerspectiveCamera) throws {
            surface = try ProceduralSurface(device: device, style: .wood, resolution: 64, seed: 11)
            let textured = try surface.material(normalScale: 0.75)
            let glass = try PhysicallyBasedMaterial(
                metallic: 0.25, roughness: 0.35, baseColorFactor: SIMD4(0.8, 0.3, 0.2, 0.9),
                emissiveFactor: SIMD3(0.05, 0.1, 0.15), alphaMode: .blend,
                clearcoat: .init(factor: 0.6, roughness: 0.15),
                transmission: .init(
                    factor: 0.4, indexOfRefraction: 1.45, thickness: 0.2, attenuationDistance: 2,
                    attenuationColor: SIMD3(0.9, 0.8, 0.7)
                )
            )
            boxGeometry = try .bevelBox(size: SIMD3(1, 0.5, 0.75), bevel: 0.08, segments: 3)
            sphereGeometry = try .uvSphere(radius: 0.4, segments: 24, rings: 12)
            let boxRenderer = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: boxGeometry, material: .physicallyBased(textured)
            )
            let sphereRenderer = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: sphereGeometry, material: .physicallyBased(glass)
            )
            let rest = SceneTransform(
                translation: SIMD3(0, 0.9, 0), rotation: simd_quatf(angle: 0.3, axis: simd_normalize(SIMD3(1, 1, 0))),
                scale: SIMD3(1, 1.2, 1)
            )
            sphere = SceneNode(transform: rest, mesh: sphereRenderer)
            box = SceneNode(
                transform: .init(translation: SIMD3(0.5, 0, -1), rotation: simd_quatf(angle: 0.7, axis: SIMD3(0, 1, 0))),
                mesh: boxRenderer, children: [sphere]
            )
            group = SceneNode(transform: .init(translation: SIMD3(0, 0.25, 0), scale: SIMD3(repeating: 2)), children: [box])
            spotLight = try PunctualLight(
                kind: .spot(range: 12, innerConeAngle: 0.2, outerConeAngle: 0.6), color: SIMD3(1, 0.9, 0.8), intensity: 40
            )
            spot = SceneNode(
                transform: .init(translation: SIMD3(2, 3, 1), rotation: simd_quatf(angle: -0.9, axis: SIMD3(1, 0, 0))),
                light: spotLight
            )
            graph = SceneGraph(
                roots: [group, spot], camera: camera, cameraTransform: .init(translation: SIMD3(0, 1, 5))
            )
            animation = try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: sphere.id, keyframes: [
                TransformKeyframe(time: 0, transform: rest),
                TransformKeyframe(time: 0.5, transform: .init(
                    translation: SIMD3(0, 1.2, 0.3), rotation: simd_quatf(angle: 1.2, axis: SIMD3(0, 0, 1)),
                    scale: SIMD3(1.1, 1, 0.9)
                )),
                TransformKeyframe(time: 1.25, transform: .init(
                    translation: SIMD3(0, 0.9, -0.2), rotation: simd_quatf(angle: 2.4, axis: SIMD3(0, 1, 0))
                )),
            ])])
            names = [group.id: "Group", box.id: "Box", sphere.id: "Sphere", spot.id: "Spot"]
        }
    }

    /// A skinned quad whose joint carries a matrix transform, with spline rotation and step translation.
    private func skinnedAsset() throws -> Data {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], type: "VEC3", count: 4)
        let faces = fixture.bytes([0, 1, 2, 0, 2, 3], type: "SCALAR", count: 6, component: 5121)
        let lanes: [UInt8] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        let joints = fixture.bytes(lanes.flatMap { [$0, 0] }, type: "VEC4", count: 4, component: 5123)
        let weights = fixture.bytes(
            [255, 0, 0, 0, 255, 0, 0, 0, 128, 127, 0, 0, 255, 0, 0, 0], type: "VEC4", count: 4, component: 5121, normalized: true
        )
        let binds = fixture.floats([
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1,
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1,
        ], type: "MAT4", count: 2)
        let splineTimes = fixture.floats([0, 1], type: "SCALAR", count: 2, bounds: [0, 1])
        let half = Float(0.5).squareRoot()
        let splineValues = fixture.floats([
            0, 0, 0.1, 0, 0, 0, 0, 1, 0, 0, 0.2, 0,
            0, 0, -0.1, 0, 0, 0, half, half, 0.05, 0, 0, 0,
        ], type: "VEC4", count: 6)
        let stepTimes = fixture.floats([0, 0.25, 0.75], type: "SCALAR", count: 3, bounds: [0, 0.75])
        let stepValues = fixture.floats([0, 1, 0, 0.5, 1, 0, 0, 1.5, 0.25], type: "VEC3", count: 3)
        let primitive: [String: Any] = [
            "attributes": ["POSITION": positions, "JOINTS_0": joints, "WEIGHTS_0": weights], "indices": faces,
        ]
        let nodes: [[String: Any]] = [
            ["name": "Skinned", "mesh": 0, "skin": 0],
            ["name": "Root", "children": [2], "matrix": [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0.5, 0, 0, 1]],
            ["name": "Tip", "translation": [0, 1, 0]],
        ]
        let animation: [String: Any] = [
            "name": "Wave",
            "channels": [
                ["sampler": 0, "target": ["node": 2, "path": "rotation"]],
                ["sampler": 1, "target": ["node": 2, "path": "translation"]],
            ],
            "samplers": [
                ["input": splineTimes, "output": splineValues, "interpolation": "CUBICSPLINE"],
                ["input": stepTimes, "output": stepValues, "interpolation": "STEP"],
            ],
        ]
        return try fixture.document([
            "scenes": [["name": "Rigged", "nodes": [0, 1]]], "nodes": nodes, "meshes": [["name": "Quad", "primitives": [primitive]]],
            "skins": [["name": "Rig", "joints": [2, 1], "skeleton": 1, "inverseBindMatrices": binds]],
            "animations": [animation],
        ])
    }

    // MARK: Helpers

    private func load(_ data: Data) throws -> GLTFScene {
        try GLTFMeshLoader.loadScene(
            data: data, includeMaterials: true, includeAnimations: true, includeSkins: true,
            includeLights: true, includeCameras: true
        )
    }

    private func renderer(_ geometry: MeshGeometry, material: MeshMaterial) throws -> MeshRenderer {
        try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material)
    }

    private func material(_ mesh: GLTFScene.Mesh, in scene: GLTFScene) throws -> GLTFScene.Material {
        let index = try #require(mesh.primitiveMaterialIndices.first ?? nil)
        return try #require(scene.materials.first { $0.sourceIndex == index })
    }

    private func texture(_ format: MTLPixelFormat, width: Int, height: Int, bytes: [UInt8]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        bytes.withUnsafeBytes { raw in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: raw.baseAddress!, bytesPerRow: bytes.count / height
            )
        }
        return texture
    }

    private func privateTexture(_ format: MTLPixelFormat, width: Int, height: Int, bytes: [UInt8]) throws -> any MTLTexture {
        let staging = try texture(format, width: width, height: height, bytes: bytes)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let command = try #require(queue.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(from: staging, to: texture)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        return texture
    }

    /// Decodes a PNG the way the importer prepares glTF maps and returns its RGBA bytes.
    private func decode(_ data: Data, _ interpretation: TextureLoader.Interpretation) throws -> [UInt8] {
        try pixels(TextureLoader(device: device).load(data: data, interpretation: interpretation))
    }

    /// Level 0 of a shared four-byte-per-pixel texture.
    private func pixels(_ texture: any MTLTexture) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: texture.width * texture.height * 4)
        bytes.withUnsafeMutableBytes { raw in
            texture.getBytes(
                raw.baseAddress!, bytesPerRow: texture.width * 4,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return bytes
    }

    private func maximumDifference(_ a: [UInt8], _ b: [UInt8]) -> Int {
        guard a.count == b.count else { return .max }
        return zip(a, b).map { abs(Int($0) - Int($1)) }.max() ?? 0
    }

    private static func encodeSRGB(_ value: Float) -> Float {
        value <= 0.003_130_8 ? value * 12.92 : 1.055 * pow(value, 1 / 2.4) - 0.055
    }

    private static func decodeSRGB(_ value: Float) -> Float {
        value <= 0.040_45 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
    }

    /// Checks the GLB header, chunk order, lengths, and padding, then the buffer view and accessor bounds.
    private func inspectGLB(_ data: Data) throws -> (document: [String: Any], binary: Data) {
        let bytes = [UInt8](data)
        func word(_ offset: Int) -> Int {
            (0..<4).reduce(0) { $0 | Int(bytes[offset + $1]) << (8 * $1) }
        }
        try #require(bytes.count >= 20)
        #expect(word(0) == 0x4654_6C67)
        #expect(word(4) == 2)
        #expect(word(8) == bytes.count)
        let jsonLength = word(12)
        #expect(word(16) == 0x4E4F_534A)
        #expect(jsonLength.isMultiple(of: 4))
        try #require(20 + jsonLength <= bytes.count)
        let json = Array(bytes[20..<(20 + jsonLength)])
        let closing = try #require(json.lastIndex(of: UInt8(ascii: "}")))
        #expect(json[(closing + 1)...].allSatisfy { $0 == 0x20 })
        #expect(json.count - closing - 1 < 4)
        let document = try #require(JSONSerialization.jsonObject(with: Data(json)) as? [String: Any])
        let asset = try #require(document["asset"] as? [String: String])
        #expect(asset == ["version": "2.0", "generator": "Frostfield"])

        var binary = Data()
        let start = 20 + jsonLength
        if start < bytes.count {
            try #require(bytes.count - start >= 8)
            let length = word(start)
            #expect(word(start + 4) == 0x004E_4942)
            #expect(length.isMultiple(of: 4))
            #expect(start + 8 + length == bytes.count)
            binary = Data(bytes[(start + 8)...])
        }
        guard let buffers = document["buffers"] as? [[String: Any]] else {
            #expect(binary.isEmpty)
            return (document, binary)
        }
        try #require(buffers.count == 1)
        #expect(buffers[0]["uri"] == nil)
        let byteLength = try #require(buffers[0]["byteLength"] as? Int)
        #expect(binary.count >= byteLength && binary.count - byteLength < 4)
        #expect(binary.suffix(binary.count - byteLength).allSatisfy { $0 == 0 })

        let views = try #require(document["bufferViews"] as? [[String: Any]])
        for view in views {
            let offset = view["byteOffset"] as? Int ?? 0, length = try #require(view["byteLength"] as? Int)
            #expect(offset.isMultiple(of: 4) && length > 0 && offset + length <= byteLength)
        }
        let componentSizes = [5_121: 1, 5_123: 2, 5_125: 4, 5_126: 4]
        let componentCounts = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16]
        for accessor in try #require(document["accessors"] as? [[String: Any]]) {
            let view = views[try #require(accessor["bufferView"] as? Int)]
            let componentType = try #require(accessor["componentType"] as? Int)
            let type = try #require(accessor["type"] as? String)
            let size = try #require(componentSizes[componentType])
            let components = try #require(componentCounts[type])
            let count = try #require(accessor["count"] as? Int)
            #expect(count > 0 && accessor["byteOffset"] == nil)
            #expect(count * size * components == view["byteLength"] as? Int)
        }
        return (document, binary)
    }

    private func poses(_ scene: GLTFScene) throws -> (graph: SceneGraph, ids: [Int: UUID]) {
        let ids = Dictionary(uniqueKeysWithValues: scene.nodes.map { ($0.sourceIndex, UUID()) })
        let nodes = Dictionary(uniqueKeysWithValues: scene.nodes.map { ($0.sourceIndex, $0) })
        func build(_ index: Int) throws -> SceneNode {
            let node = try #require(nodes[index])
            return SceneNode(id: try #require(ids[index]), transform: node.localTransform, children: try node.childIndices.map(build))
        }
        return (SceneGraph(roots: try scene.rootNodeIndices.map(build), camera: camera), ids)
    }

    private func transform(of id: UUID, in graph: SceneGraph) throws -> SceneTransform {
        var pending = graph.roots
        while let node = pending.popLast() {
            if node.id == id { return node.transform }
            pending += node.children
        }
        throw GLTFExporter.Failure.missingAnimationTarget(id)
    }

    private func close(_ a: SceneTransform, _ b: SceneTransform, tolerance: Float = 0.000_001) -> Bool {
        simd_distance(a.translation, b.translation) <= tolerance && simd_distance(a.scale, b.scale) <= tolerance
            && 1 - abs(simd_dot(a.rotation.vector, b.rotation.vector)) <= max(tolerance, 0.000_000_1)
    }

    private func expectSameGeometry(_ actual: MeshGeometry, _ expected: MeshGeometry) {
        #expect(actual.indices == expected.indices)
        #expect(actual.vertices.count == expected.vertices.count)
        let mismatches = zip(actual.vertices, expected.vertices).filter { a, e in
            a.position != e.position || a.normal != e.normal || a.textureCoordinate != e.textureCoordinate
                || a.secondaryTextureCoordinate != e.secondaryTextureCoordinate || a.tangent != e.tangent
        }
        #expect(mismatches.isEmpty)
    }

    /// Compares every public field the importer produces, including encoded image bytes.
    private func expectSameScene(_ a: GLTFScene, _ b: GLTFScene) {
        #expect(a.name == b.name)
        #expect(a.rootNodeIndices == b.rootNodeIndices)
        #expect(a.nodes.count == b.nodes.count)
        for (x, y) in zip(a.nodes, b.nodes) {
            #expect(x.sourceIndex == y.sourceIndex && x.name == y.name && x.parentIndex == y.parentIndex)
            #expect(x.childIndices == y.childIndices && x.meshIndex == y.meshIndex && x.skinIndex == y.skinIndex)
            #expect(x.lightIndex == y.lightIndex && x.cameraIndex == y.cameraIndex)
            #expect(close(x.localTransform, y.localTransform, tolerance: 0))
        }
        #expect(a.meshes.count == b.meshes.count)
        for (x, y) in zip(a.meshes, b.meshes) {
            #expect(x.name == y.name && x.primitiveMaterialIndices == y.primitiveMaterialIndices)
            #expect(x.primitiveSkinInfluences == y.primitiveSkinInfluences)
            for (p, q) in zip(x.primitives, y.primitives) { expectSameGeometry(q, p) }
        }
        #expect(a.materials.count == b.materials.count)
        for (x, y) in zip(a.materials, b.materials) {
            #expect(x.name == y.name && x.baseColorFactor == y.baseColorFactor)
            #expect(x.metallicFactor == y.metallicFactor && x.roughnessFactor == y.roughnessFactor)
            #expect(x.normalScale == y.normalScale && x.occlusionStrength == y.occlusionStrength)
            #expect(x.emissiveFactor == y.emissiveFactor && x.alphaMode == y.alphaMode && x.isDoubleSided == y.isDoubleSided)
            #expect(x.indexOfRefraction == y.indexOfRefraction)
            #expect(x.clearcoat?.factor == y.clearcoat?.factor && x.clearcoat?.roughness == y.clearcoat?.roughness)
            #expect(x.transmission?.factor == y.transmission?.factor)
            #expect(x.volume?.thickness == y.volume?.thickness && x.volume?.attenuationDistance == y.volume?.attenuationDistance)
            #expect(x.volume?.attenuationColor == y.volume?.attenuationColor)
            let pairs = [
                (x.baseColorTexture, y.baseColorTexture), (x.metallicRoughnessTexture, y.metallicRoughnessTexture),
                (x.normalTexture, y.normalTexture), (x.occlusionTexture, y.occlusionTexture),
                (x.emissiveTexture, y.emissiveTexture),
            ]
            for (p, q) in pairs {
                #expect(p?.image.data == q?.image.data && p?.image.name == q?.image.name && p?.image.mimeType == q?.image.mimeType)
                #expect(p?.sampling == q?.sampling && p?.textureCoordinateSet == q?.textureCoordinateSet)
            }
        }
        #expect(a.lights.map(\.light) == b.lights.map(\.light) && a.lights.map(\.name) == b.lights.map(\.name))
        #expect(a.cameras.map(\.projection) == b.cameras.map(\.projection) && a.cameras.map(\.name) == b.cameras.map(\.name))
        #expect(a.skins.map(\.name) == b.skins.map(\.name) && a.skins.map(\.jointNodeIndices) == b.skins.map(\.jointNodeIndices))
        #expect(a.skins.map(\.inverseBindMatrices) == b.skins.map(\.inverseBindMatrices))
        #expect(a.skins.map(\.skeletonRootNodeIndex) == b.skins.map(\.skeletonRootNodeIndex))
        #expect(a.animations.map(\.name) == b.animations.map(\.name) && a.animations.map(\.duration) == b.animations.map(\.duration))
        for (x, y) in zip(a.animations, b.animations) {
            #expect(x.channels.map(\.sourceNodeIndex) == y.channels.map(\.sourceNodeIndex))
            #expect(x.channels.map(\.target) == y.channels.map(\.target))
            #expect(x.channels.map(\.interpolation) == y.channels.map(\.interpolation))
            #expect(x.channels.map(\.times) == y.channels.map(\.times))
        }
    }
}
