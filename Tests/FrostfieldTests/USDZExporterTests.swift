import Foundation
import Frostfield
import Metal
import ModelIO
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct USDZExporterTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
    }

    @Test
    func `Writes a stored zip with 64-byte aligned files and the USDA root layer first`() throws {
        let fixture = try USDZFixture(device: device)
        let output = try USDZExporter(commandQueue: queue).export(fixture.graph, nodeNames: fixture.names)
        let records = try USDZArchive.records(in: output.data)

        #expect(records.first?.path == "scene.usda")
        #expect(records.count == 1 + 2 + 3) // root layer, two geometry layers, three textures
        for record in records {
            #expect(record.method == 0, "\(record.path) must be stored")
            #expect(record.dataOffset % 64 == 0, "\(record.path) data starts at \(record.dataOffset)")
            let start = output.data.startIndex + record.dataOffset
            let data = output.data.subdata(in: start..<(start + record.size))
            #expect(USDZArchive.crc32(data) == record.crc32, "\(record.path) CRC")
            if record.path.hasSuffix(".png") {
                #expect(data.starts(with: [0x89, 0x50, 0x4E, 0x47]))
            }
        }
        let stored = try #require(try USDZArchive.file("scene.usda", in: output.data))
        #expect(String(decoding: stored, as: UTF8.self) == output.rootLayer)
        for line in ["defaultPrim = \"Root\"", "metersPerUnit = 1", "upAxis = \"Y\"", "kind = \"component\""] {
            #expect(output.rootLayer.contains(line), "missing \(line)")
        }

        #if os(macOS)
        // An independent zip reader agrees with the directory.
        let url = try temporaryFile("archive.usdz", output.data)
        let listing = try run("/usr/bin/unzip", ["-v", url.path])
        #expect(listing.status == 0)
        #expect(listing.output.contains("Stored") && !listing.output.contains("Defl"))
        #endif
    }

    @Test
    func `Round-trips hierarchy, transforms, meshes, camera, and materials through ModelIO`() throws {
        let fixture = try USDZFixture(device: device)
        let output = try USDZExporter(commandQueue: queue).export(fixture.graph, nodeNames: fixture.names)
        #expect(output.nodePaths[fixture.group.id] == "/Root/Group")
        #expect(output.nodePaths[fixture.box.id] == "/Root/Group/Box")
        #expect(output.nodePaths[fixture.sphere.id] == "/Root/Group/Box/Sphere")
        #expect(output.nodePaths[fixture.crowd.id] == "/Root/Crowd")
        #expect(output.cameraPath == "/Root/Camera")

        let url = try temporaryFile("roundtrip.usdz", output.data)
        let asset = MDLAsset(url: url)
        let objects = flatten(asset)
        #expect(asset.upAxis == SIMD3(0, 1, 0))

        // Local transforms compose exactly like the scene graph.
        for (node, path) in [(fixture.group, "/Root/Group"), (fixture.box, "/Root/Group/Box"), (fixture.sphere, "/Root/Group/Box/Sphere")] {
            let object = try #require(objects[path], "missing \(path)")
            let matrix = try #require(object.transform?.localTransform?(atTime: 0))
            #expect(maximumDifference(matrix, node.transform.matrix) < 1e-5, "\(path)")
        }
        let instanceMatrices = try (0..<3).map { index in
            try #require(objects["/Root/Crowd/Instance_\(index)"]?.transform?.localTransform?(atTime: 0))
        }
        for (matrix, expected) in zip(instanceMatrices, fixture.instanceTransforms) {
            #expect(maximumDifference(matrix, expected) < 1e-6)
        }

        // Geometry: counts, positions, normals, and top-left UVs after undoing USD's flip.
        let box = try #require(objects["/Root/Group/Box/Mesh"] as? MDLMesh)
        let sphere = try #require(objects["/Root/Group/Box/Sphere/Mesh"] as? MDLMesh)
        for (mesh, geometry) in [(box, fixture.boxGeometry), (sphere, fixture.sphereGeometry)] {
            #expect(mesh.vertexCount == geometry.vertices.count)
            let submesh = try #require(mesh.submeshes?.firstObject as? MDLSubmesh)
            #expect(submesh.indexCount == geometry.indices.count)
            let positions = attribute(mesh, MDLVertexAttributePosition, SIMD3<Float>.self)
            let normals = attribute(mesh, MDLVertexAttributeNormal, SIMD3<Float>.self)
            let uvs = attribute(mesh, MDLVertexAttributeTextureCoordinate, SIMD2<Float>.self)
            for (index, vertex) in geometry.vertices.enumerated() {
                #expect(positions[index] == vertex.position)
                #expect(normals[index] == vertex.normal)
                #expect(simd_distance(SIMD2(uvs[index].x, 1 - uvs[index].y), vertex.textureCoordinate) <= 1e-6)
            }
            let indexBuffer = submesh.indexBuffer(asIndexType: .uInt32)
            let indices = Array(UnsafeBufferPointer(
                start: indexBuffer.map().bytes.assumingMemoryBound(to: UInt32.self), count: submesh.indexCount
            ))
            #expect(indices == geometry.indices)
        }
        // Instances share one geometry layer.
        for index in 0..<3 {
            let mesh = try #require(objects["/Root/Crowd/Instance_\(index)/Mesh"] as? MDLMesh)
            #expect(mesh.vertexCount == fixture.sphereGeometry.vertices.count)
        }
        #expect(output.rootLayer.components(separatedBy: "geometries/Geometry_1.usda").count - 1 == 4)

        let camera = try #require(objects["/Root/Camera"] as? MDLCamera)
        let cameraMatrix = try #require(camera.transform?.localTransform?(atTime: 0))
        #expect(abs(camera.fieldOfView - fixture.graph.camera.verticalFieldOfView * 180 / .pi) < 1e-3)
        #expect(camera.nearVisibilityDistance == fixture.graph.camera.near)
        #expect(camera.farVisibilityDistance == fixture.graph.camera.far)
        #expect(maximumDifference(cameraMatrix, fixture.graph.cameraTransform.matrix) < 1e-6)

        // UsdPreviewSurface inputs as ModelIO reads them.
        let textured = try #require((box.submeshes?.firstObject as? MDLSubmesh)?.material)
        let base = try #require(textured.propertyNamed("diffuseColor")?.stringValue)
        let surfaceData = try #require(textured.propertyNamed("metallic")?.stringValue)
        #expect(textured.propertyNamed("roughness")?.stringValue == surfaceData)
        #expect(textured.propertyNamed("occlusion")?.stringValue == surfaceData)
        #expect(textured.propertyNamed("normal")?.stringValue?.hasSuffix("[textures/Texture_2.png]") == true)
        #expect(base.hasSuffix("[textures/Texture_0.png]"))
        #expect(surfaceData.hasSuffix("[textures/Texture_1.png]"))
        let glass = try #require((sphere.submeshes?.firstObject as? MDLSubmesh)?.material)
        #expect(glass.propertyNamed("diffuseColor")?.float3Value == SIMD3(0.8, 0.3, 0.2))
        #expect(glass.propertyNamed("opacity")?.floatValue == 0.9)
        #expect(glass.propertyNamed("metallic")?.floatValue == 0.25)
        #expect(glass.propertyNamed("roughness")?.floatValue == 0.35)
        #expect(glass.propertyNamed("clearcoat")?.floatValue == 0.6)
        #expect(glass.propertyNamed("clearcoatRoughness")?.floatValue == 0.15)
        #expect(glass.propertyNamed("ior")?.floatValue == 1.45)
        #expect(glass.propertyNamed("emissiveColor")?.float3Value == SIMD3(0.05, 0.1, 0.15))

        // Stored textures decode to the source maps' bytes.
        let color = try #require(try USDZArchive.file("textures/Texture_0.png", in: output.data))
        let decoded = try TextureLoader(device: device).load(data: color, interpretation: .encodedSRGB)
        #expect(try bytes(decoded) == bytes(fixture.surface.baseColorTexture))

        #expect(output.warnings == [
            .materialFeatureDropped(material: "/Root/Materials/Material_0", feature: .normalScale),
            .materialFeatureDropped(material: "/Root/Materials/Material_1", feature: .transmission),
            .materialFeatureDropped(material: "/Root/Materials/Material_1", feature: .volume),
            .instanceTintsDropped(node: "/Root/Crowd"),
            .lightOmitted(node: "/Root/Lamp"),
        ])
    }

    @Test(.enabled(if: hasUSDChecker, "Requires macOS usdchecker"))
    func `Passes usdchecker's ARKit compliance rules`() throws {
        let fixture = try USDZFixture(device: device)
        let output = try USDZExporter(commandQueue: queue).export(fixture.graph, nodeNames: fixture.names)
        let url = try temporaryFile("arkit.usdz", output.data)
        let result = try run("/usr/bin/usdchecker", ["--arkit", "--strict", url.path])
        #expect(result.status == 0, "\(result.output)")
        #expect(result.output.contains("Success"), "\(result.output)")
    }

    @Test
    func `UsdTransform2d parameters reproduce Frostfield texture transforms in st space`() throws {
        let transforms = [
            try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(0.25, -0.5), rotation: 0.7, scale: SIMD2(2, 3)),
            try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(1, 2), rotation: -2.2, scale: SIMD2(-1, 0.5)),
            try PhysicallyBasedMaterial.TextureTransform(scale: SIMD2(4, 4)),
        ]
        for transform in transforms {
            let parameters = USDTextureTransform.parameters(transform)
            for uv in [SIMD2<Float>(0, 0), SIMD2(1, 0), SIMD2(0.3, 0.8), SIMD2(-2, 5)] {
                let expected = transform.apply(to: uv)
                let st = USDTextureTransform.evaluate(
                    SIMD2(uv.x, 1 - uv.y), scale: parameters.scale, rotationDegrees: parameters.rotationDegrees,
                    translation: parameters.translation
                )
                #expect(simd_distance(SIMD2(st.x, 1 - st.y), expected) < 1e-5)
            }
        }
    }

    @Test
    func `Exports an imported glTF scene with sampler wrap modes and shared primitives`() throws {
        let fixture = try USDZFixture(device: device)
        let glb = try GLTFExporter(commandQueue: queue).export(fixture.graph, nodeNames: fixture.names)
        let scene = try GLTFMeshLoader.loadScene(
            data: glb.data, includeMaterials: true, includeLights: true, includeCameras: true, includeGPUInstancing: true
        )
        let output = try USDZExporter().export(scene)
        let records = try USDZArchive.records(in: output.data)
        #expect(records.allSatisfy { $0.dataOffset % 64 == 0 && $0.method == 0 })

        let objects = flatten(MDLAsset(url: try temporaryFile("gltf.usdz", output.data)))
        for node in scene.nodes {
            let path = try #require(output.sourceNodePaths[node.sourceIndex])
            let object = try #require(objects[path], "missing \(path)")
            let matrix = try #require(object.transform?.localTransform?(atTime: 0))
            #expect(maximumDifference(matrix, node.localTransform.matrix) < 1e-5)
        }
        let boxIndex = try #require(glb.nodeIndices[fixture.box.id])
        let lampIndex = try #require(glb.nodeIndices[fixture.lamp.id])
        let boxPath = try #require(output.sourceNodePaths[boxIndex])
        let lampPath = try #require(output.sourceNodePaths[lampIndex])
        let box = try #require(objects["\(boxPath)/Mesh"] as? MDLMesh)
        #expect(box.vertexCount == fixture.boxGeometry.vertices.count)
        #expect(output.rootLayer.contains("token inputs:wrapS = \"repeat\""))
        #expect(output.rootLayer.contains("def Camera \"Camera\""))
        #expect(output.warnings.contains(.lightOmitted(node: lampPath)))

        // A LINEAR clip becomes time samples at its key times.
        let source = SceneTransform(translation: SIMD3(0, 0.9, 0))
        let bob = try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: fixture.sphere.id, keyframes: [
            TransformKeyframe(time: 0, transform: source),
            TransformKeyframe(time: 0.5, transform: .init(translation: SIMD3(0, 1.2, 0.3), rotation: simd_quatf(angle: 1.2, axis: SIMD3(0, 0, 1)))),
            TransformKeyframe(time: 1.25, transform: .init(translation: SIMD3(0, 0.9, -0.2), scale: SIMD3(1.1, 1, 0.9))),
        ])])
        let animated = try GLTFExporter().export(fixture.graph, nodeNames: fixture.names, animations: [.init(name: "Bob", animation: bob)])
        let animatedScene = try GLTFMeshLoader.loadScene(
            data: animated.data, includeMaterials: true, includeAnimations: true, includeLights: true,
            includeCameras: true, includeGPUInstancing: true
        )
        let usd = try USDZExporter().export(animatedScene)
        let sphereIndex = try #require(animated.nodeIndices[fixture.sphere.id])
        let spherePath = try #require(usd.sourceNodePaths[sphereIndex])
        let sphere = try #require(flatten(MDLAsset(url: try temporaryFile("animated.usdz", usd.data)))[spherePath])
        #expect(sphere.transform?.keyTimes.map(\.doubleValue) == [0, 0.5, 1.25])
        let middle = try #require(sphere.transform?.localTransform?(atTime: 0.5))
        #expect(maximumDifference(middle, bob.tracks[0].keyframes[1].transform.matrix) < 1e-5)
    }

    @Test
    func `Samples a 400-node glTF clip by evaluating each node's own channels`() throws {
        let quad = try GeometryRenderFixture.square(0.5)
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: quad, material: .unlit)
        let nodes = (0..<400).map { index in
            SceneNode(transform: .init(translation: SIMD3(Float(index), 0, 0)), mesh: index == 0 ? renderer : nil)
        }
        func pose(node: Int, key: Int) -> SceneTransform {
            SceneTransform(
                translation: SIMD3(Float(node), Float(key) * 0.1, Float(node % 7) * 0.2),
                rotation: simd_quatf(angle: Float(key) * 0.05 + Float(node) * 0.01, axis: simd_normalize(SIMD3(0.2, 1, 0.1))),
                scale: SIMD3(repeating: 1 + Float(key) * 0.01)
            )
        }
        let animation = try SceneAnimation(tracks: nodes.enumerated().map { index, node in
            try SceneAnimationTrack(nodeID: node.id, keyframes: (0..<60).map {
                TransformKeyframe(time: Float($0) / 10, transform: pose(node: index, key: $0))
            })
        })
        let graph = SceneGraph(roots: nodes, camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let glb = try GLTFExporter().export(graph, animations: [.init(name: "Wave", animation: animation)], includesCamera: false)
        let scene = try GLTFMeshLoader.loadScene(data: glb.data, includeAnimations: true)
        #expect(scene.animations.first?.channels.count == 1_200)

        var exported: USDZExporter.Output?
        let elapsed = try threadCPUTime { exported = try USDZExporter().export(scene) }
        let output = try #require(exported)
        // About 0.2 s in a Debug build. Sampling every channel of the clip for each node at each
        // key, nodes² × keys channel evaluations, used 14 s of CPU time.
        #expect(elapsed < .seconds(3), "export used \(elapsed) of CPU time")

        let objects = flatten(MDLAsset(url: try temporaryFile("wave.usdz", output.data)))
        for node in [0, 151, 399] {
            let index = try #require(glb.nodeIndices[nodes[node].id])
            let path = try #require(output.sourceNodePaths[index])
            let object = try #require(objects[path])
            #expect(object.transform?.keyTimes.count == 60)
            for key in [0, 23, 59] {
                let matrix = try #require(object.transform?.localTransform?(atTime: Double(key) / 10))
                #expect(maximumDifference(matrix, pose(node: node, key: key).matrix) < 1e-5, "node \(node) key \(key)")
            }
        }
    }

    @Test
    func `Writes unlit, masked, and transformed maps as UsdPreviewSurface networks`() throws {
        let texture = try solidTexture(SIMD4(255, 128, 0, 200))
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(0.5, 0), rotation: .pi / 2, scale: SIMD2(2, 2))
        let masked = try PhysicallyBasedMaterial(
            baseColorTexture: texture, baseColorFactor: SIMD4(1, 0.5, 0.25, 1), alphaMode: .mask(cutoff: 0.4),
            isDoubleSided: true, baseColorTextureTransform: transform
        )
        let unlit = try PhysicallyBasedMaterial(baseColorFactor: SIMD4(0.2, 0.4, 0.6, 1), shadingModel: .unlit)
        let quad = try GeometryRenderFixture.square(0.5)
        let graph = SceneGraph(roots: [
            SceneNode(mesh: try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: quad, material: .physicallyBased(masked))),
            SceneNode(mesh: try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: quad, material: .physicallyBased(unlit))),
        ], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let output = try USDZExporter().export(graph, includesCamera: false)
        let layer = output.rootLayer
        let geometry = try geometryLayers(output.data)
        for line in [
            "float inputs:opacity.connect = </Root/Materials/Material_0/BaseColorTexture.outputs:a>",
            "float inputs:opacityThreshold = 0.4",
            "float4 inputs:scale = (1, 0.5, 0.25, 1)",
            "uniform token info:id = \"UsdTransform2d\"",
            "float inputs:rotation = 90",
            "float2 inputs:scale = (2, 2)",
            "uniform bool doubleSided = 1",
            "color3f inputs:emissiveColor = (0.2, 0.4, 0.6)",
            "color3f inputs:diffuseColor = (0, 0, 0)",
            "texCoord2f[] primvars:st",
            "float4[] primvars:tangents = [(1, 0, 0, -1)",
        ] {
            #expect(layer.contains(line) || geometry.contains(line), "missing \(line)")
        }
        #expect(output.warnings == [.unlitApproximated(material: "/Root/Materials/Material_1")])
        if hasUSDChecker {
            let result = try run("/usr/bin/usdchecker", ["--arkit", "--strict", try temporaryFile("networks.usdz", output.data).path])
            #expect(result.status == 0, "\(result.output)")
        }
    }

    @Test
    func `Rejects invalid options and package paths`() throws {
        let graph = SceneGraph(roots: [], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        #expect(throws: USDZExporter.Failure.self) {
            try USDZExporter(options: .init(metersPerUnit: 0)).export(graph)
        }
        #expect(throws: USDZArchive.Failure.invalidPath("../escape.png")) {
            try USDZArchive.write([.init(path: "../escape.png", data: Data())])
        }
        #expect(throws: USDZArchive.Failure.duplicatePath("a.usda")) {
            try USDZArchive.write([.init(path: "a.usda", data: Data()), .init(path: "a.usda", data: Data())])
        }
        // Padding that would be shorter than the four-byte extra field header grows by a full block.
        for nameLength in 1...80 {
            let archive = try USDZArchive.write([.init(path: String(repeating: "a", count: nameLength), data: Data([1, 2, 3]))])
            let record = try #require(try USDZArchive.records(in: archive).first)
            #expect(record.dataOffset % 64 == 0)
            #expect(try USDZArchive.file(record.path, in: archive) == Data([1, 2, 3]))
        }
    }

    // MARK: Helpers

    private func solidTexture(_ rgba: SIMD4<UInt8>) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb, width: 2, height: 2, mipmapped: false)
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let pixels = Array(repeating: rgba, count: 4)
        pixels.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: 8)
        }
        return texture
    }

    private func bytes(_ texture: any MTLTexture) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: texture.width * texture.height * 4)
        let source: any MTLTexture
        if texture.storageMode == .private {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: texture.pixelFormat, width: texture.width, height: texture.height, mipmapped: false
            )
            descriptor.storageMode = .shared
            let copy = try #require(device.makeTexture(descriptor: descriptor))
            let command = try #require(queue.makeCommandBuffer())
            let blit = try #require(command.makeBlitCommandEncoder())
            blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, to: copy, destinationSlice: 0, destinationLevel: 0, sliceCount: 1, levelCount: 1)
            blit.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            source = copy
        } else {
            source = texture
        }
        bytes.withUnsafeMutableBytes { raw in
            source.getBytes(raw.baseAddress!, bytesPerRow: texture.width * 4, from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        }
        return bytes
    }

    private func geometryLayers(_ archive: Data) throws -> String {
        try USDZArchive.records(in: archive).filter { $0.path.hasPrefix("geometries/") }.map {
            String(decoding: try USDZArchive.file($0.path, in: archive) ?? Data(), as: UTF8.self)
        }.joined()
    }
}

/// A textured box with a glass sphere child, three sphere instances, a lamp, and a camera.
struct USDZFixture {
    let surface: ProceduralSurface
    let boxGeometry: MeshGeometry
    let sphereGeometry: MeshGeometry
    let group: SceneNode
    let box: SceneNode
    let sphere: SceneNode
    let crowd: SceneNode
    let lamp: SceneNode
    let instanceTransforms: [simd_float4x4]
    let graph: SceneGraph
    let names: [UUID: String]

    init(device: any MTLDevice) throws {
        surface = try ProceduralSurface(device: device, style: .wood, resolution: 32, seed: 7)
        let textured = try surface.material(normalScale: 0.75)
        let glass = try PhysicallyBasedMaterial(
            metallic: 0.25, roughness: 0.35, baseColorFactor: SIMD4(0.8, 0.3, 0.2, 0.9),
            emissiveFactor: SIMD3(0.05, 0.1, 0.15), alphaMode: .blend,
            clearcoat: .init(factor: 0.6, roughness: 0.15),
            transmission: .init(factor: 0.4, indexOfRefraction: 1.45, thickness: 0.2)
        )
        boxGeometry = try .bevelBox(size: SIMD3(1, 0.5, 0.75), bevel: 0.08, segments: 3)
        sphereGeometry = try .uvSphere(radius: 0.4, segments: 16, rings: 8)
        let boxRenderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: boxGeometry, material: .physicallyBased(textured)
        )
        let sphereRenderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: sphereGeometry, material: .physicallyBased(glass),
            supportsInstancing: true
        )
        sphere = SceneNode(
            transform: .init(
                translation: SIMD3(0, 0.9, 0), rotation: simd_quatf(angle: 0.3, axis: simd_normalize(SIMD3(1, 1, 0))),
                scale: SIMD3(1, 1.2, 1)
            ),
            mesh: sphereRenderer
        )
        box = SceneNode(
            transform: .init(translation: SIMD3(0.5, 0, -1), rotation: simd_quatf(angle: 0.7, axis: SIMD3(0, 1, 0))),
            mesh: boxRenderer, children: [sphere]
        )
        group = SceneNode(transform: .init(translation: SIMD3(0, 0.25, 0), scale: SIMD3(repeating: 2)), children: [box])
        instanceTransforms = [
            SceneTransform(translation: SIMD3(-2, 0, 0)).matrix,
            SceneTransform(translation: SIMD3(2, 0, 0), rotation: simd_quatf(angle: 1, axis: SIMD3(0, 0, 1))).matrix,
            SceneTransform(translation: SIMD3(0, 0, 2), scale: SIMD3(0.5, 2, 1)).matrix,
        ]
        crowd = SceneNode(
            transform: .init(translation: SIMD3(0, -1, 0)), mesh: sphereRenderer,
            instances: try InstanceSet(device: device, transforms: instanceTransforms, tints: [
                SIMD4(repeating: 1), SIMD4(1, 0, 0, 1), SIMD4(repeating: 1),
            ])
        )
        lamp = SceneNode(
            transform: .init(translation: SIMD3(2, 3, 1)),
            light: try PunctualLight(kind: .point(range: 10), intensity: 20)
        )
        graph = SceneGraph(
            roots: [group, crowd, lamp], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 50),
            cameraTransform: .init(translation: SIMD3(0, 1, 5), rotation: simd_quatf(angle: -0.1, axis: SIMD3(1, 0, 0)))
        )
        names = [group.id: "Group", box.id: "Box", sphere.id: "Sphere", crowd.id: "Crowd", lamp.id: "Lamp"]
    }
}

/// Whether this macOS host has Apple's USD compliance checker.
let hasUSDChecker: Bool = {
    #if os(macOS)
    FileManager.default.isExecutableFile(atPath: "/usr/bin/usdchecker")
    #else
    false
    #endif
}()

/// Every object in a ModelIO asset, keyed by its USD-style path.
func flatten(_ asset: MDLAsset) -> [String: MDLObject] {
    var result: [String: MDLObject] = [:]
    var pending = (0..<asset.count).map { asset.object(at: $0) }
    while let object = pending.popLast() {
        result[object.path] = object
        pending += object.children.objects
    }
    return result
}

func attribute<T>(_ mesh: MDLMesh, _ name: String, _ type: T.Type) -> [T] {
    let format: MDLVertexFormat = MemoryLayout<T>.size == 8 ? .float2 : .float3
    guard let data = mesh.vertexAttributeData(forAttributeNamed: name, as: format) else { return [] }
    return (0..<mesh.vertexCount).map { index in
        if MemoryLayout<T>.size == 8 {
            return data.dataStart.advanced(by: index * data.stride).loadUnaligned(as: T.self)
        }
        let value = data.dataStart.advanced(by: index * data.stride).loadUnaligned(as: (Float, Float, Float).self)
        return SIMD3<Float>(value.0, value.1, value.2) as! T
    }
}

func maximumDifference(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
    let columns = [a.columns.0 - b.columns.0, a.columns.1 - b.columns.1, a.columns.2 - b.columns.2, a.columns.3 - b.columns.3]
    return columns.map { simd_reduce_max(abs($0)) }.max() ?? 0
}

func temporaryFile(_ name: String, _ data: Data) throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("FrostfieldFormats-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent(name)
    try data.write(to: url)
    return url
}

private let processLock = NSLock()

/// Runs a system tool and returns its exit status with combined output. Launches are
/// serialized because parallel tests spawning processes at once have failed with EFAULT.
func run(_ executable: String, _ arguments: [String]) throws -> (status: Int32, output: String) {
    #if os(macOS)
    processLock.lock()
    defer { processLock.unlock() }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    let output = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: output, as: UTF8.self))
    #else
    return (-1, "Processes are unavailable on this platform.")
    #endif
}
