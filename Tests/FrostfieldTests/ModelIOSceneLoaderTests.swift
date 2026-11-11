import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import ModelIO
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ModelIOSceneLoaderTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture(side: 96)
    }

    @Test
    func `Imports an OBJ with MTL factors, maps, packed metallic-roughness, and a concave face`() throws {
        let folder = try temporaryFolder()
        let base: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]
        try writePNG(base, width: 2, height: 2, to: folder.appendingPathComponent("textures/base.png"))
        try writePNG([128, 128, 255, 255], width: 1, height: 1, to: folder.appendingPathComponent("textures/normal.png"))
        try writePNG([64, 64, 64, 255], width: 1, height: 1, to: folder.appendingPathComponent("rough.png"))
        try writePNG([200, 200, 200, 255], width: 1, height: 1, to: folder.appendingPathComponent("metal.png"))
        try """
        # Two materials with PBR statements
        newmtl Painted
        Kd 0.8 0.1 0.05
        Ke 0.1 0.2 0.3
        d 0.75
        Ni 1.45
        Pr 0.4
        Pm 0.9
        Pc 0.5
        Pcr 0.25
        map_Kd -s 2 2 1 textures/base.png
        norm textures/normal.png
        newmtl Rough
        Kd 0 0 1
        Ns 98
        map_Pr rough.png
        map_Pm metal.png
        map_Ka ambient.png
        """.write(to: folder.appendingPathComponent("scene.mtl"), atomically: true, encoding: .utf8)
        // The second object is a concave L-shaped hexagon, which OBJMeshLoader rejects. It starts
        // next to the reflex corner, where a triangle fan would cover the notch.
        let obj = folder.appendingPathComponent("scene.obj")
        try """
        mtllib scene.mtl
        o Panel
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        vt 0 0
        vt 1 0
        vt 1 1
        vt 0 1
        vn 0 0 1
        usemtl Painted
        f 1/1/1 2/2/1 3/3/1 4/4/1
        o Ell
        v 0 0 1
        v 2 0 1
        v 2 1 1
        v 1 1 1
        v 1 2 1
        v 0 2 1
        usemtl Rough
        f 7 8 9 10 5 6
        """.write(to: obj, atomically: true, encoding: .utf8)
        #expect(throws: OBJMeshLoader.Failure.self) { try OBJMeshLoader.load(contentsOf: obj) }

        let asset = try ModelIOSceneLoader.load(contentsOf: obj)
        let scene = asset.scene
        let mesh = try #require(scene.meshes.first)
        #expect(mesh.primitives.count == 2)
        let panel = mesh.primitives[0], ell = mesh.primitives[1]
        #expect(panel.indices.count == 6)
        #expect(ell.indices.count == 12) // four triangles
        #expect(Set(panel.vertices.map(\.position)) == [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(0, 1, 0)])
        for vertex in panel.vertices {
            // OBJ's bottom-left vt maps to top-left UVs: the vertex at the origin samples v = 1.
            #expect(vertex.textureCoordinate == SIMD2(vertex.position.x, 1 - vertex.position.y))
            #expect(vertex.normal == SIMD3(0, 0, 1))
        }
        // The concave outline's triangles cover exactly its area of three square units.
        let area = stride(from: 0, to: ell.indices.count, by: 3).reduce(Float(0)) { total, start in
            let a = ell.vertices[Int(ell.indices[start])].position, b = ell.vertices[Int(ell.indices[start + 1])].position
            let c = ell.vertices[Int(ell.indices[start + 2])].position
            #expect(simd_cross(b - a, c - a).z >= 0)
            return total + simd_length(simd_cross(b - a, c - a)) / 2
        }
        #expect(abs(area - 3) < 1e-5)

        let painted = try #require(scene.materials.first { $0.name == "Painted" })
        #expect(painted.baseColorFactor == SIMD4(0.8, 0.1, 0.05, 0.75))
        #expect(painted.alphaMode == .blend)
        #expect(painted.metallicFactor == 0.9 && painted.roughnessFactor == 0.4)
        #expect(painted.emissiveFactor == SIMD3(0.1, 0.2, 0.3) && painted.emissiveStrength == 1)
        #expect(painted.indexOfRefraction == 1.45)
        #expect(painted.clearcoat?.factor == 0.5 && painted.clearcoat?.roughness == 0.25)
        let baseTexture = try #require(painted.baseColorTexture)
        #expect(baseTexture.image.data == (try Data(contentsOf: folder.appendingPathComponent("textures/base.png"))))
        // -s 2 2 doubles the repeat around MTL's bottom-left origin.
        #expect(baseTexture.transform.scale == SIMD2(2, 2))
        #expect(baseTexture.transform.apply(to: SIMD2(0, 1)) == SIMD2(0, 1))
        #expect(painted.normalTexture?.image.mimeType == "image/png")

        let rough = try #require(scene.materials.first { $0.name == "Rough" })
        #expect(rough.baseColorFactor == SIMD4(0, 0, 1, 1) && rough.alphaMode == .opaque)
        #expect(rough.metallicFactor == 1 && rough.roughnessFactor == 1)
        let packed = try #require(rough.metallicRoughnessTexture)
        let channels = try decodeRGBA(packed.image.data)
        #expect(Array(channels.prefix(4)) == [255, 64, 200, 255])
        #expect(asset.warnings.contains(.materialInputIgnored(material: "Rough", input: "map_Ka")))

        // Phong exponent 98 without Pr uses the Beckmann conversion before packing replaces it.
        let phong = try ModelIOSceneLoader.load(contentsOf: try objWithMaterial("Ns 98", in: temporaryFolder()))
        #expect(abs((phong.scene.materials.first?.roughnessFactor ?? 0) - pow(2 / Float(100), 0.25)) < 1e-6)
    }

    @Test
    func `Imports MTL transparency, emission, bump, and texture options`() throws {
        let folder = try temporaryFolder()
        try writePNG([200, 100, 50, 255, 10, 20, 30, 255], width: 2, height: 1, to: folder.appendingPathComponent("base.png"))
        try writePNG([64, 64, 64, 255, 192, 192, 192, 255], width: 2, height: 1, to: folder.appendingPathComponent("alpha.png"))
        try writePNG([255, 255, 0, 255], width: 1, height: 1, to: folder.appendingPathComponent("glow.png"))
        try writePNG([128, 128, 255, 255], width: 1, height: 1, to: folder.appendingPathComponent("bumpy.png"))
        let obj = try objWithMaterial("""
        Kd 1 1 1
        Tr 0.25
        map_Kd -o 0.25 0 -clamp on base.png
        map_d alpha.png
        map_Ke glow.png
        bump -bm 0.5 bumpy.png
        """, in: folder)
        let asset = try ModelIOSceneLoader.load(contentsOf: obj)
        let material = try #require(asset.scene.materials.first)
        #expect(material.alphaMode == .blend && material.baseColorFactor.w == 0.75)
        let base = try #require(material.baseColorTexture)
        // The separate dissolve map becomes the base color texture's alpha.
        #expect(try decodeRGBA(base.image.data) == [200, 100, 50, 64, 10, 20, 30, 192])
        #expect(base.sampling.wrapS == .clampToEdge && base.sampling.wrapT == .clampToEdge)
        // -o shifts u by 0.25; v keeps MTL's bottom-left origin in top-left space.
        #expect(base.transform.offset == SIMD2(0.25, 0) && base.transform.scale == SIMD2(1, 1))
        #expect(material.emissiveTexture?.image.data == (try Data(contentsOf: folder.appendingPathComponent("glow.png"))))
        #expect(material.emissiveFactor == SIMD3(repeating: 1))
        #expect(material.normalTexture == nil && material.normalScale == 1)
        let bump = try #require(material.bumpMap)
        #expect(bump.scale == 0.5)
        #expect(try decodeRGBA(bump.texture.image.data) == [128, 128, 255, 255])
        #expect(!asset.warnings.contains(.bumpMapUsedAsNormalMap(material: "Only")))
    }

    @Test("OBJ map_bump survives prepared rendering and norm takes precedence")
    func preparedHeightMap() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let texels = (0..<256).flatMap { value -> [UInt8] in [UInt8(value), 0, 0, 255] }
        try writePNG(texels + texels, width: 256, height: 2, to: folder.appendingPathComponent("height.png"))
        try writePNG([128, 128, 255, 255], width: 1, height: 1, to: folder.appendingPathComponent("normal.png"))
        let height = "map_bump -bm -0.75 -s 0.75 0.5 1 -o 0.125 0.125 0 -clamp on height.png"
        let obj = try objWithMaterial(height, in: folder)
        try """
        mtllib only.mtl
        v 0 0 0
        v 1 0 0
        v 0 1 0
        vt 0 0
        vt 1 0
        vt 0 1
        usemtl Only
        f 1/1 2/2 3/3
        """.write(to: obj, atomically: true, encoding: .utf8)

        let gpu = try PathTestFixture()
        let asset = try ModelIOSceneLoader.load(contentsOf: obj)
        let material = try #require(asset.scene.materials.first)
        let bump = try #require(material.bumpMap)
        #expect(material.normalTexture == nil && bump.scale == -0.75)
        #expect(bump.texture.transform.scale == SIMD2(0.75, 0.5))
        #expect(bump.texture.transform.offset == SIMD2(0.125, 0.375))
        #expect(bump.texture.sampling.wrapS == .clampToEdge && bump.texture.sampling.wrapT == .clampToEdge)

        func renderNormal(_ source: GLTFScene) throws -> SIMD3<Float> {
            let prepared = try PreparedGLTFScene(device: gpu.device, colorPixelFormat: .rgba16Float, scene: source)
            let graph = try prepared.graph(camera: gpu.camera, cameraTransform: .init(translation: SIMD3(0.25, 0.25, 1)))
            return try gpu.rasterSurfaceNormal(graph, side: 17)
        }
        let bumped = try renderNormal(asset.scene)
        // The red ramp rises one unit per U, and one screen pixel spans
        // 2*tan(fov/2)/17 units on this plane. Negative strength tilts toward +X.
        let slope = 0.75 * 0.75 * 2 * tan(gpu.camera.verticalFieldOfView / 2) / 17
        #expect(simd_distance(bumped, simd_normalize(SIMD3(slope, 0, 1))) < 0.006, "\(bumped)")

        #expect(throws: GLTFExporter.Failure.unsupportedTexture("Convert bump maps to tangent-space normal maps before glTF export.")) {
            try GLTFExporter().export(asset.scene)
        }
        #expect(throws: USDZExporter.Failure.invalidData("Convert bump maps to tangent-space normal maps before USDZ export.")) {
            try USDZExporter().export(asset.scene)
        }

        let library = folder.appendingPathComponent("only.mtl")
        try "newmtl Only\n\(height)\nnorm normal.png\n".write(to: library, atomically: true, encoding: .utf8)
        let both = try ModelIOSceneLoader.load(contentsOf: obj)
        let bothMaterial = try #require(both.scene.materials.first)
        #expect(bothMaterial.bumpMap != nil && bothMaterial.normalTexture != nil)
        #expect(try bothMaterial.prepare(device: gpu.device).bumpMap == nil)
        let normalOnly = try renderNormal(both.scene)
        #expect(abs(normalOnly.x) < 0.006 && abs(normalOnly.y) < 0.006 && normalOnly.z > 0.999)
        _ = try GLTFExporter().export(both.scene)
        _ = try USDZExporter().export(both.scene)
    }

    @Test
    func `Reads MTL libraries named by an OBJ that is not valid UTF-8`() throws {
        let folder = try temporaryFolder()
        try writePNG([10, 20, 30, 255], width: 1, height: 1, to: folder.appendingPathComponent("base.png"))
        // The MTL also carries a Latin-1 byte (0xB0, a degree sign) in a comment.
        var mtl = Data("newmtl Only\n# Rotated 90".utf8)
        mtl.append(0xB0)
        mtl.append(contentsOf: Data("\nKd 0.5 0.5 0.5\nPr 0.4\nPm 0.9\nmap_Kd -s 2 2 1 base.png\n".utf8))
        try mtl.write(to: folder.appendingPathComponent("only.mtl"))
        // One ISO Latin-1 byte (0xE9, é) in a comment makes the OBJ invalid UTF-8.
        var obj = Data("# Exported by Caf".utf8)
        obj.append(0xE9)
        obj.append(contentsOf: Data("\nmtllib only.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl Only\nf 1/1 2/2 3/3\n".utf8))
        #expect(String(data: obj, encoding: .utf8) == nil)
        let url = folder.appendingPathComponent("latin1.obj")
        try obj.write(to: url)

        let asset = try ModelIOSceneLoader.load(contentsOf: url)
        #expect(!asset.warnings.contains { if case .materialInputIgnored = $0 { true } else { false } }, "\(asset.warnings)")
        let material = try #require(asset.scene.materials.first)
        #expect(material.metallicFactor == 0.9 && material.roughnessFactor == 0.4)
        #expect(material.baseColorTexture?.transform.scale == SIMD2(2, 2))
    }

    @Test
    func `Imports ASCII and binary PLY files written by the test`() throws {
        let folder = try temporaryFolder()
        let positions: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(2, 0, 0), SIMD3(2, 1, 0), SIMD3(0, 1, 0)]
        let ascii = folder.appendingPathComponent("quad.ply")
        var text = """
        ply
        format ascii 1.0
        element vertex 4
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property uchar red
        property uchar green
        property uchar blue
        element face 1
        property list uchar int vertex_indices
        end_header

        """
        for position in positions { text += "\(position.x) \(position.y) \(position.z) 0 0 1 255 128 0\n" }
        text += "4 0 1 2 3\n"
        try text.write(to: ascii, atomically: true, encoding: .ascii)

        var binary = Data("""
        ply
        format binary_little_endian 1.0
        element vertex 4
        property float x
        property float y
        property float z
        element face 1
        property list uchar int vertex_indices
        end_header

        """.utf8)
        for position in positions {
            for value in [position.x, position.y, position.z] { withUnsafeBytes(of: value.bitPattern.littleEndian) { binary.append(contentsOf: $0) } }
        }
        binary.append(4)
        for index: Int32 in [0, 1, 2, 3] { withUnsafeBytes(of: index.littleEndian) { binary.append(contentsOf: $0) } }
        let binaryURL = folder.appendingPathComponent("binary.ply")
        try binary.write(to: binaryURL)

        for (url, colors) in [(ascii, true), (binaryURL, false)] {
            let asset = try ModelIOSceneLoader.load(contentsOf: url)
            let geometry = try #require(asset.scene.meshes.first?.primitives.first)
            #expect(geometry.indices.count == 6, "\(url.lastPathComponent)")
            #expect(Set(geometry.vertices.map(\.position)) == Set(positions))
            #expect(geometry.vertices.allSatisfy { simd_distance($0.normal, SIMD3(0, 0, 1)) < 1e-6 })
            // Both triangles face +Z, matching the counterclockwise source winding.
            for start in stride(from: 0, to: geometry.indices.count, by: 3) {
                let a = geometry.vertices[Int(geometry.indices[start])].position
                let b = geometry.vertices[Int(geometry.indices[start + 1])].position
                let c = geometry.vertices[Int(geometry.indices[start + 2])].position
                #expect(simd_cross(b - a, c - a).z > 0)
            }
            #expect(!asset.warnings.contains { if case .vertexColorsDropped = $0 { true } else { false } })
            if colors {
                for vertex in geometry.vertices {
                    let color = try #require(vertex.color)
                    #expect(simd_distance(color, SIMD4(1, 0.2158605, 0, 1)) < 1e-6)
                }
            } else {
                #expect(!geometry.hasVertexColors)
            }
            #expect(asset.warnings.contains { if case .normalsGenerated = $0 { true } else { false } } == !colors)
        }
    }

    @Test
    func `Imports STL and ModelIO-written Alembic geometry`() throws {
        let folder = try temporaryFolder()
        let stl = folder.appendingPathComponent("wedge.stl")
        try """
        solid wedge
        facet normal 0 0 1
        outer loop
        vertex 0 0 0
        vertex 1 0 0
        vertex 0 1 0
        endloop
        endfacet
        facet normal 0 0 1
        outer loop
        vertex 1 0 0
        vertex 1 1 0
        vertex 0 1 0
        endloop
        endfacet
        endsolid wedge
        """.write(to: stl, atomically: true, encoding: .ascii)
        let square = try #require(try ModelIOSceneLoader.load(contentsOf: stl).scene.meshes.first?.primitives.first)
        #expect(square.indices.count == 6)
        #expect(Array(square.indices.prefix(3).map { square.vertices[Int($0)].position }) == [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0)])
        #expect(square.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })

        // ModelIO writes Alembic; the importer reads it back.
        let box = MDLMesh(boxWithExtent: SIMD3(1, 2, 3), segments: SIMD3(1, 1, 1), inwardNormals: false,
                          geometryType: .triangles, allocator: nil)
        let source = MDLAsset()
        source.add(box)
        let abc = folder.appendingPathComponent("box.abc")
        try source.export(to: abc)
        let asset = try ModelIOSceneLoader.load(contentsOf: abc)
        let geometry = try #require(asset.scene.meshes.first?.primitives.first)
        let points = geometry.vertices.map(\.position)
        let lower = points.reduce(points[0], simd_min), upper = points.reduce(points[0], simd_max)
        #expect(lower == SIMD3(-0.5, -1, -1.5) && upper == SIMD3(0.5, 1, 1.5))
        #expect(geometry.indices.count == 36)
    }

    @Test
    func `A Frostfield USDZ imports back and renders the same pixels as the source graph`() throws {
        let device = fixture.device
        let colors: [(SIMD4<Float>, Float, Float)] = [
            (SIMD4(0.9, 0.2, 0.1, 1), 0, 0.3), (SIMD4(0.2, 0.7, 0.3, 1), 1, 0.5), (SIMD4(0.3, 0.3, 0.9, 1), 0.4, 0.8),
        ]
        let sphere = try MeshGeometry.uvSphere(radius: 0.6, segments: 32, rings: 16)
        var nodes: [SceneNode] = []
        for (index, (color, metallic, roughness)) in colors.enumerated() {
            let material = try PhysicallyBasedMaterial(
                metallic: metallic, roughness: roughness, directionToLight: SIMD3(0.3, 1, 0.8), lightIntensity: 2,
                baseColorFactor: color, emissiveFactor: index == 2 ? SIMD3(0.1, 0, 0.2) : .zero, alphaMode: .opaque,
                clearcoat: index == 1 ? try .init(factor: 0.7, roughness: 0.1) : nil
            )
            let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: sphere, material: .physicallyBased(material))
            nodes.append(SceneNode(
                transform: .init(translation: SIMD3(Float(index - 1) * 1.4, 0, -2.5), rotation: simd_quatf(angle: Float(index), axis: SIMD3(0, 1, 0))),
                mesh: renderer
            ))
        }
        let group = SceneNode(transform: .init(translation: SIMD3(0, 0.1, 0), scale: SIMD3(repeating: 1.1)), children: nodes)
        let graph = SceneGraph(roots: [group], camera: fixture.camera)
        let output = try USDZExporter().export(graph)
        #expect(output.warnings.isEmpty)
        let url = try temporaryFolder().appendingPathComponent("spheres.usdz")
        try output.data.write(to: url)

        let asset = try ModelIOSceneLoader.load(contentsOf: url)
        #expect(asset.warnings.isEmpty, "\(asset.warnings)")
        #expect(asset.sourceMetersPerUnit == 1)
        #expect(asset.scene.materials.count == 3)
        let prepared = try PreparedGLTFScene(
            device: device, colorPixelFormat: .rgba16Float, scene: asset.scene,
            directionToLight: SIMD3(0.3, 1, 0.8), lightIntensity: 2
        )
        let expected = try fixture.render(graph).pixels
        let imported = try fixture.render(try prepared.graph(camera: fixture.camera)).pixels
        let covered = expected.filter { $0.w > 0 }.count
        #expect(covered > fixture.side * fixture.side / 10)
        let difference = zip(expected, imported).map { simd_reduce_max(abs($0 - $1)) }.max() ?? 0
        #expect(difference < 1e-3, "maximum channel difference \(difference)")
        try writeDebugPNG(imported, side: fixture.side, name: "modelio-usdz-roundtrip.png")
    }

    @Test
    func `Textured USDZ maps import with their stored bytes and render like the source`() throws {
        let device = fixture.device
        let surface = try ProceduralSurface(device: device, style: .wood, resolution: 64, seed: 3)
        let material = try surface.material(directionToLight: SIMD3(0, 0.4, 1), lightIntensity: 1.5)
        let box = try MeshGeometry.bevelBox(size: SIMD3(1.6, 1.6, 1), bevel: 0.1, segments: 2)
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2.6), rotation: simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3(1, 1, 0)))),
            mesh: try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: box, material: .physicallyBased(material))
        )
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let output = try USDZExporter().export(graph)
        // ModelIO names packaged files `package.usdz[entry]`; brackets in the folder name must not split that.
        let folder = try temporaryFolder().appendingPathComponent("a[1]")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("wood.usdz")
        try output.data.write(to: url)

        let asset = try ModelIOSceneLoader.load(contentsOf: url)
        #expect(asset.warnings.isEmpty, "\(asset.warnings)")
        let imported = try #require(asset.scene.materials.first)
        // The metallic, roughness, and occlusion inputs share one stored image, as exported.
        #expect(imported.metallicRoughnessTexture?.image.sourceIndex == imported.occlusionTexture?.image.sourceIndex)
        #expect(imported.baseColorTexture?.image.data == (try USDZArchive.file("textures/Texture_0.png", in: output.data)))
        #expect(imported.normalTexture?.image.data == (try USDZArchive.file("textures/Texture_2.png", in: output.data)))

        let prepared = try PreparedGLTFScene(
            device: device, colorPixelFormat: .rgba16Float, scene: asset.scene, directionToLight: SIMD3(0, 0.4, 1), lightIntensity: 1.5
        )
        let expected = try fixture.render(graph).pixels
        let rendered = try fixture.render(try prepared.graph(camera: fixture.camera)).pixels
        let differences = zip(expected, rendered).map { simd_reduce_max(abs($0 - $1)) }
        let mean = differences.reduce(0, +) / Float(differences.count)
        // Measured: mean 2.5 × 10⁻⁸, maximum 1.2 × 10⁻⁴. Halving the normal scale gives mean 1.2 × 10⁻⁴, maximum 4.4 × 10⁻³.
        #expect(mean < 1e-5, "mean channel difference \(mean)")
        #expect((differences.max() ?? 0) < 1e-3, "maximum channel difference \(differences.max() ?? 0)")
        try writeDebugPNG(rendered, side: fixture.side, name: "modelio-textured-roundtrip.png")
    }

    @Test
    func `Animated USDZ exports play back through ModelIO like the source animation`() throws {
        let device = fixture.device
        let box = try MeshGeometry.bevelBox(size: SIMD3(1.6, 1, 0.6), bevel: 0.05, segments: 2)
        let material = try PhysicallyBasedMaterial(metallic: 0.2, roughness: 0.4, baseColorFactor: SIMD4(0.9, 0.6, 0.2, 1), alphaMode: .opaque)
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: box, material: .physicallyBased(material))
        let spinner = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)
        let graph = SceneGraph(roots: [spinner], camera: fixture.camera)
        let keys = [
            TransformKeyframe(time: 0, transform: .init(translation: SIMD3(-0.8, 0, -3))),
            TransformKeyframe(time: 0.5, transform: .init(
                translation: SIMD3(0, 0.5, -2.5), rotation: simd_quatf(angle: 2.4, axis: simd_normalize(SIMD3(0.3, 1, 0.2))),
                scale: SIMD3(1.3, 0.8, 1)
            )),
            TransformKeyframe(time: 1.25, transform: .init(
                translation: SIMD3(0.7, -0.3, -3.5), rotation: simd_quatf(angle: -1.9, axis: SIMD3(1, 0, 0)), scale: SIMD3(repeating: 0.9)
            )),
        ]
        let animation = try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: spinner.id, keyframes: keys)])
        let output = try USDZExporter().export(graph, nodeNames: [spinner.id: "Spinner"], animation: animation, includesCamera: false)
        #expect(output.rootLayer.contains("timeCodesPerSecond = 60") && output.rootLayer.contains("endTimeCode = 75"))
        let url = try temporaryFolder().appendingPathComponent("animated.usdz")
        try output.data.write(to: url)
        if hasUSDChecker {
            let result = try run("/usr/bin/usdchecker", ["--arkit", "--strict", url.path])
            #expect(result.status == 0, "\(result.output)")
        }

        // ModelIO reads the samples at their times.
        let object = try #require(flatten(MDLAsset(url: url))["/Root/Spinner"])
        let component = try #require(object.transform)
        #expect(component.keyTimes.map(\.doubleValue) == [0, 0.5, 1.25])
        for key in keys {
            let matrix = try #require(component.localTransform?(atTime: Double(key.time)))
            #expect(maximumDifference(matrix, key.transform.matrix) < 1e-5, "key \(key.time)")
        }

        // The importer turns them into a LINEAR clip that renders like the source between keys.
        let asset = try ModelIOSceneLoader.load(contentsOf: url)
        let clip = try #require(asset.scene.animations.first)
        #expect(clip.duration == 1.25)
        let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .rgba16Float, scene: asset.scene)
        for time: Float in [0, 0.3, 0.5, 0.9, 1.25] {
            let expected = try fixture.render(try animation.applying(to: graph, at: time)).pixels
            let imported = try fixture.render(try prepared.graph(camera: fixture.camera, animationIndex: 0, time: time)).pixels
            let differences = zip(expected, imported).map { simd_reduce_max(abs($0 - $1)) }
            // Measured maximum 6.1 × 10⁻⁵ at every time.
            #expect((differences.max() ?? 0) < 2e-3, "time \(time): maximum channel difference \(differences.max() ?? 0)")
            #expect(expected.filter { $0.w > 0 }.count > 200)
            if time == 0.9 { try writeDebugPNG(imported, side: fixture.side, name: "modelio-animated-0.9.png") }
        }
    }

    @Test
    func `Double-sided USD meshes import double-sided and render like the source`() throws {
        let device = fixture.device
        // Both squares face away from the camera; only the double-sided one is visible.
        let quad = try GeometryRenderFixture.square(0.6)
        func node(_ x: Float, doubleSided: Bool, color: SIMD4<Float>) throws -> SceneNode {
            let material = try PhysicallyBasedMaterial(
                metallic: 0, roughness: 0.6, directionToLight: SIMD3(0, 0, -1), lightIntensity: 2,
                baseColorFactor: color, alphaMode: .opaque, isDoubleSided: doubleSided
            )
            return SceneNode(
                transform: .init(translation: SIMD3(x, 0, -2), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))),
                mesh: try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: quad, material: .physicallyBased(material))
            )
        }
        let front = try node(-0.7, doubleSided: true, color: SIMD4(0.9, 0.5, 0.2, 1))
        let culled = try node(0.7, doubleSided: false, color: SIMD4(0.2, 0.5, 0.9, 1))
        let graph = SceneGraph(roots: [front, culled], camera: fixture.camera)
        let output = try USDZExporter().export(graph, nodeNames: [front.id: "Front", culled.id: "Culled"], includesCamera: false)
        let url = try temporaryFolder().appendingPathComponent("sides.usdz")
        try output.data.write(to: url)

        let asset = try ModelIOSceneLoader.load(contentsOf: url)
        #expect(asset.warnings.isEmpty, "\(asset.warnings)")
        let scene = asset.scene
        func material(_ name: String) throws -> GLTFScene.Material {
            let node = try #require(scene.nodes.first { $0.name == name })
            let mesh = try #require(node.childIndices.compactMap { scene.nodes[$0].meshIndex }.first)
            return try #require(scene.meshes[mesh].primitiveMaterialIndices.first.flatMap { $0 }.map { scene.materials[$0] })
        }
        #expect(try material("Front").isDoubleSided)
        #expect(try !material("Culled").isDoubleSided)

        let prepared = try PreparedGLTFScene(
            device: device, colorPixelFormat: .rgba16Float, scene: scene, directionToLight: SIMD3(0, 0, -1), lightIntensity: 2
        )
        let expected = try fixture.render(graph).pixels
        let imported = try fixture.render(try prepared.graph(camera: fixture.camera)).pixels
        let side = fixture.side
        let left = (0..<side).filter { expected[(side / 2) * side + $0].w > 0 }
        #expect(!left.isEmpty && left.allSatisfy { $0 < side / 2 }, "covered columns \(left)")
        let difference = zip(expected, imported).map { simd_reduce_max(abs($0 - $1)) }.max() ?? 0
        #expect(difference < 1e-3, "maximum channel difference \(difference)")
        try writeDebugPNG(imported, side: side, name: "modelio-double-sided.png")
    }

    @Test
    func `Finds doubleSided opinions in USDA text without misreading strings, metadata, or comments`() throws {
        let layer = #"""
        #usda 1.0
        (
            customLayerData = { string note = "def Mesh "Fake" { uniform bool doubleSided = 1 }" }
        )
        def Xform "World" (
            customData = { dictionary inner = { bool doubleSided = 1 } }
        )
        {
            def Mesh "A" { uniform bool doubleSided = 1 }
            def Mesh "B"
            {
                uniform bool doubleSided = 0
                # uniform bool doubleSided = 1
                // uniform bool doubleSided = 1
                /* uniform bool doubleSided = 1 */
                string label = "doubleSided = 1 } {"
                string block = """ { doubleSided = 1 """
                asset file = @a{b}.png@
                rel material:binding = </World/Looks/M>
                double3 xformOp:translate.timeSamples = { 0: (0, 0, 0), 24: (1, 0, 0) }
            }
            over "Group" (
                prepend references = @@@other{1}.usda@@@
            )
            {
                def Mesh "C" (kind = "x")
                {
                    int[] faceVertexCounts = [3]
                    uniform bool doubleSided = true
                }
            }
            def "Untyped" { uniform bool doubleSided = 1 }
        }
        """#
        #expect(USDLayerMetadata.doubleSidedPrims(inUSDA: Data(layer.utf8)) == ["/World/A", "/World/Group/C", "/World/Untyped"])
        #expect(USDLayerMetadata.metersPerUnit(inUSDA: Data(layer.utf8)) == nil)
        #expect(USDLayerMetadata.metersPerUnit(inUSDA: Data("#usda 1.0\n(\n    metersPerUnit = 0.025\n)\n".utf8)) == 0.025)
    }

    @Test
    func `ModelIO exposes no tangent primvar, so imported primitives carry none`() throws {
        let url = try temporaryFolder().appendingPathComponent("tangents.usda")
        try """
        #usda 1.0
        def Mesh "M"
        {
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = "vertex")
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (interpolation = "vertex")
            float4[] primvars:tangents = [(1, 0, 0, -1), (1, 0, 0, -1), (1, 0, 0, -1)] (interpolation = "vertex")
            float3[] primvars:tangent = [(1, 0, 0), (1, 0, 0), (1, 0, 0)] (interpolation = "vertex")
            uniform token subdivisionScheme = "none"
        }
        """.write(to: url, atomically: true, encoding: .utf8)
        // If a future ModelIO reports tangents, this fails and the importer should read them.
        let mesh = try #require(flatten(MDLAsset(url: url))["/M"] as? MDLMesh)
        #expect(mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeTangent)?.format ?? .invalid == .invalid)
        let geometry = try #require(try ModelIOSceneLoader.load(contentsOf: url).scene.meshes.first?.primitives.first)
        #expect(geometry.vertices.allSatisfy { $0.tangent == nil })
    }

    @Test
    func `Ear clipping triangulates keyholes, collinear runs, stars, and spirals inside their outline`() throws {
        // Every triangle must wind with the polygon; together they then cover it exactly once.
        func check(_ outline: [SIMD2<Double>], _ label: String) {
            let positions = outline.map { SIMD3<Float>(Float($0.x), Float($0.y), 0) }
            let triangles = PolygonTriangulation.triangulate(Array(0..<UInt32(outline.count)), positions: positions)
            #expect(triangles.count == (outline.count - 2) * 3, "\(label): \(triangles.count / 3) triangles")
            func flat(_ index: Int) -> SIMD2<Double> { SIMD2(Double(positions[index].x), Double(positions[index].y)) }
            var area = 0.0
            for start in stride(from: 0, to: triangles.count, by: 3) {
                let a = flat(Int(triangles[start])), b = flat(Int(triangles[start + 1])), c = flat(Int(triangles[start + 2]))
                let signed = ((b - a).x * (c - a).y - (b - a).y * (c - a).x) / 2
                #expect(signed >= -1e-9, "\(label): triangle \(start / 3) is inverted")
                area += abs(signed)
            }
            let expected = (0..<positions.count).reduce(0.0) { total, index in
                let a = flat(index), b = flat((index + 1) % positions.count)
                return total + (a.x * b.y - b.x * a.y) / 2
            }
            #expect(abs(area - expected) <= 1e-9 * max(1, expected), "\(label): area \(area), expected \(expected)")
        }
        // A square with a square hole, joined by a two-way bridge.
        check([SIMD2(0, 0), SIMD2(3, 3), SIMD2(3, 7), SIMD2(7, 7), SIMD2(7, 3), SIMD2(3, 3), SIMD2(0, 0),
               SIMD2(10, 0), SIMD2(10, 10), SIMD2(0, 10)], "keyhole")
        check([SIMD2(0, 0), SIMD2(1, 0), SIMD2(2, 0), SIMD2(3, 0), SIMD2(3, 1), SIMD2(2, 1), SIMD2(2, 2), SIMD2(1, 2),
               SIMD2(1, 1), SIMD2(0, 1)], "collinear")
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func random() -> Double {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(state >> 11) / Double(1 << 53)
        }
        for star in 0..<200 {
            let count = 5 + star % 60
            check((0..<count).map { index in
                let angle = 2 * Double.pi * (Double(index) + 0.8 * random()) / Double(count)
                let radius = 0.2 + random()
                return SIMD2(radius * cos(angle), radius * sin(angle))
            }, "star \(star)")
        }
        check(Self.doubleSpiral(corners: 1_020), "spiral")
    }

    @Test
    func `A 1,020-corner spiral face triangulates without restarting the ear search`() throws {
        // Restarting the search at the first corner after every ear used 3.2 s of CPU time on this
        // face in the Debug test build; resuming beside the ear uses about 0.06 s.
        let outline = Self.doubleSpiral(corners: 1_020)
        let positions = outline.map { SIMD3<Float>(Float($0.x), Float($0.y), 0) }
        var triangles: [UInt32] = []
        let elapsed = threadCPUTime {
            triangles = PolygonTriangulation.triangulate(Array(0..<UInt32(outline.count)), positions: positions)
        }
        #expect(triangles.count == (outline.count - 2) * 3)
        #expect(elapsed < .seconds(0.75), "triangulation used \(elapsed) of CPU time")
    }

    /// A strip 0.04 wide along a double Archimedean spiral, an S wound six turns each way:
    /// a simple polygon whose ears lie along the strip, with about half of its corners reflex.
    private static func doubleSpiral(corners: Int) -> [SIMD2<Double>] {
        let samples = corners / 2
        // The arms are point reflections of each other, so the curve passes smoothly through the
        // center with a curvature radius of 1/(4π) ≈ 0.08, more than the strip's half-width.
        let center = (0..<samples).map { index -> SIMD2<Double> in
            let t = -6 + 12 * Double(index) / Double(samples - 1)
            let radius = abs(t), arm = SIMD2(radius * cos(2 * .pi * radius), radius * sin(2 * .pi * radius))
            return t < 0 ? -arm : arm
        }
        var left: [SIMD2<Double>] = [], right: [SIMD2<Double>] = []
        for index in center.indices {
            let direction = simd_normalize(center[min(index + 1, center.count - 1)] - center[max(index - 1, 0)])
            let normal = SIMD2(-direction.y, direction.x)
            left.append(center[index] + 0.02 * normal)
            right.append(center[index] - 0.02 * normal)
        }
        let outline = right + left.reversed()
        let signed = (0..<outline.count).reduce(0.0) { total, index in
            let a = outline[index], b = outline[(index + 1) % outline.count]
            return total + a.x * b.y - b.x * a.y
        }
        return signed > 0 ? outline : outline.reversed()
    }

    @Test
    func `Converts Z-up centimeter stages to meters and imports time samples as animation`() throws {
        let folder = try temporaryFolder()
        let url = folder.appendingPathComponent("stage.usda")
        try """
        #usda 1.0
        (
            defaultPrim = "World"
            metersPerUnit = 0.01
            upAxis = "Z"
            timeCodesPerSecond = 24
        )
        def Xform "World"
        {
            def Mesh "Panel"
            {
                int[] faceVertexCounts = [4]
                int[] faceVertexIndices = [0, 1, 2, 3]
                point3f[] points = [(-50, 0, -50), (50, 0, -50), (50, 0, 50), (-50, 0, 50)]
                normal3f[] normals = [(0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0)] (interpolation = "vertex")
                uniform token subdivisionScheme = "none"
                double3 xformOp:translate.timeSamples = { 0: (0, 0, 0), 24: (100, 0, 0) }
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Camera "Eye"
            {
                float focalLength = 12
                float verticalAperture = 24
                float horizontalAperture = 24
                float2 clippingRange = (10, 1000)
            }
        }
        """.write(to: url, atomically: true, encoding: .utf8)
        let asset = try ModelIOSceneLoader.load(contentsOf: url)
        #expect(asset.sourceMetersPerUnit == 0.01)
        #expect(asset.sourceUpAxis == SIMD3(0, 0, 1))
        let scene = asset.scene
        let root = try #require(scene.nodes.first)
        // Z-up becomes Y-up and centimeters become meters at the root.
        let world = root.localTransform.matrix
        let corner = world * SIMD4(-50, 0, 50, 1)
        #expect(simd_distance(SIMD3(corner.x, corner.y, corner.z), SIMD3(-0.5, 0.5, 0)) < 1e-6)
        let normal = world * SIMD4(0, -1, 0, 0)
        #expect(simd_distance(simd_normalize(SIMD3(normal.x, normal.y, normal.z)), SIMD3(0, 0, 1)) < 1e-6)
        let camera = try #require(scene.cameras.first)
        guard case .perspective(let fieldOfView, let aspect, let near, let far) = camera.projection else {
            Issue.record("Expected a perspective camera")
            return
        }
        #expect(abs(fieldOfView - 2 * atan(Float(12) / 12)) < 1e-4)
        #expect(aspect == 1 && abs(near - 0.1) < 1e-6 && abs((far ?? 0) - 10) < 1e-5)

        // The time-sampled translation plays back as a LINEAR clip and renders.
        let clip = try #require(scene.animations.first)
        #expect(clip.duration == 1)
        let panelIndex = try #require(scene.nodes.firstIndex { $0.name == "Panel" })
        let translation = try #require(clip.channels.first { $0.sourceNodeIndex == panelIndex && $0.target == .translation })
        #expect(translation.times == [0, 1] && translation.interpolation == .linear)

        let prepared = try PreparedGLTFScene(device: fixture.device, colorPixelFormat: .rgba16Float, scene: scene, defaultMaterial: .unlit)
        let eye = SceneTransform(translation: SIMD3(0, 0, 2))
        for (time, center) in [(Float(0), Float(0)), (0.5, 0.5), (1, 1)] {
            let pixels = try fixture.render(try prepared.graph(camera: fixture.camera, cameraTransform: eye, animationIndex: 0, time: time)).pixels
            let row = fixture.side / 2
            let covered = (0..<fixture.side).filter { pixels[row * fixture.side + $0].w > 0 }
            // At depth 2 the 90° view spans ±2 m, so a 1 m panel covers a quarter of the row.
            let middle = Float(covered.reduce(0, +)) / Float(covered.count) + 0.5
            let expected = (center / 2 * 0.5 + 0.5) * Float(fixture.side)
            #expect(abs(middle - expected) <= 1, "time \(time): center column \(middle), expected \(expected)")
            #expect(abs(covered.count - fixture.side / 4) <= 1)
        }
    }

    @Test
    func `Rejects unsupported inputs and textures outside the asset folder`() throws {
        let folder = try temporaryFolder()
        #expect(throws: ModelIOSceneLoader.Failure.localFileRequired) {
            try ModelIOSceneLoader.load(contentsOf: URL(string: "https://example.com/a.usdz")!)
        }
        let text = folder.appendingPathComponent("notes.txt")
        try "x".write(to: text, atomically: true, encoding: .utf8)
        #expect(throws: ModelIOSceneLoader.Failure.unsupportedFormat("txt")) { try ModelIOSceneLoader.load(contentsOf: text) }
        #expect(throws: ModelIOSceneLoader.Failure.invalidOptions) {
            try ModelIOSceneLoader.load(contentsOf: text, options: .init(normalCreaseThreshold: 2))
        }

        let inner = folder.appendingPathComponent("asset")
        try FileManager.default.createDirectory(at: inner, withIntermediateDirectories: true)
        try writePNG([1, 2, 3, 255], width: 1, height: 1, to: folder.appendingPathComponent("outside.png"))
        let obj = try objWithMaterial("map_Kd ../outside.png", in: inner)
        let asset = try ModelIOSceneLoader.load(contentsOf: obj)
        #expect(asset.scene.materials.first?.baseColorTexture == nil)
        #expect(asset.warnings.contains(.textureNotLoaded(material: "Only", path: "../outside.png")))
        #expect(ModelIOSceneLoader.supportedFileExtensions.contains("usdz"))
    }

    // MARK: Helpers

    private func temporaryFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("FrostfieldModelIO-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    /// A one-triangle OBJ with a material named "Only" made from the given MTL statements.
    private func objWithMaterial(_ statements: String, in folder: URL) throws -> URL {
        try "newmtl Only\n\(statements)\n".write(to: folder.appendingPathComponent("only.mtl"), atomically: true, encoding: .utf8)
        let url = folder.appendingPathComponent("only.obj")
        try "mtllib only.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Only\nf 1 2 3\n".write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private func writePNG(_ rgba: [UInt8], width: Int, height: Int, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(rgba) as CFData))
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ))
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        try #require(CGImageDestinationFinalize(destination))
    }

    private func decodeRGBA(_ data: Data) throws -> [UInt8] {
        let texture = try TextureLoader(device: fixture.device).load(data: data, interpretation: .linearData)
        var bytes = [UInt8](repeating: 0, count: texture.width * texture.height * 4)
        bytes.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!, bytesPerRow: texture.width * 4, from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        }
        return bytes
    }
}

/// The CPU time the calling thread spends in `body`. Unlike wall-clock time, other processes
/// sharing the machine do not inflate it, so scaling tests can use fixed limits.
func threadCPUTime(_ body: () throws -> Void) rethrows -> Duration {
    let start = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)
    try body()
    return .nanoseconds(Int64(clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) - start))
}

/// Writes rendered linear pixels as an sRGB PNG in the temporary folder for inspection.
func writeDebugPNG(_ pixels: [SIMD4<Float>], side: Int, name: String) throws {
    let bytes = pixels.flatMap { pixel -> [UInt8] in
        func encode(_ value: Float) -> UInt8 {
            let clamped = min(max(value, 0), 1)
            let srgb = clamped <= 0.003_130_8 ? clamped * 12.92 : 1.055 * pow(clamped, 1 / 2.4) - 0.055
            return UInt8((srgb * 255).rounded())
        }
        return [encode(pixel.x), encode(pixel.y), encode(pixel.z), 255]
    }
    guard let space = CGColorSpace(name: CGColorSpace.sRGB), let provider = CGDataProvider(data: Data(bytes) as CFData),
          let image = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: side * 4, space: space,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                              decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(destination, image, nil)
    CGImageDestinationFinalize(destination)
}
