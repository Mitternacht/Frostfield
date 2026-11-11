import Foundation
import Frostfield
import Metal
import Testing
import simd

struct MeshFileExporterTests {
    @Test("OBJ restores UVs and linear colors and keeps material group labels")
    func objRoundTrip() throws {
        let mesh = try triangle().withGroups([MeshGeometryGroup(start: 0, count: 3, materialIndex: 7)])
        let text = String(decoding: OBJExporter.data(from: mesh, name: "panel\n# injected"), as: UTF8.self)
        #expect(text.hasPrefix("o panel__ injected\n"))
        #expect(text.contains("\ng panel__ injected_material_7\n"))
        #expect(text.hasSuffix("\ng default\n"))
        let restored = try OBJMeshLoader.load(text)
        #expect(restored.indices == mesh.indices)
        for (actual, original) in zip(restored.vertices, mesh.vertices) {
            #expect(actual.position == original.position)
            #expect(actual.normal == original.normal)
            #expect(actual.textureCoordinate == original.textureCoordinate)
            if let color = original.color {
                let exported = try #require(actual.color)
                #expect(simd_distance(exported, SIMD4(color.x, color.y, color.z, 1)) < 1e-6)
            } else {
                #expect(actual.color == nil)
            }
        }
    }

    @Test("OBJ loads the normalized RGBA extension and rejects invalid colors")
    func objColorImport() throws {
        let source = """
        v 0 0 0 1 0.5 0 0.25
        v 1 0 0 1
        v 0 1 0
        f 1 2 3
        """
        let geometry = try OBJMeshLoader.load(source)
        let color = try #require(geometry.vertices[0].color)
        #expect(simd_distance(color, SIMD4(1, 0.21404114, 0, 0.25)) < 1e-6)
        #expect(geometry.vertices[1].color == nil)
        #expect(throws: OBJMeshLoader.Failure.invalidLine(1, "Vertex color components must be in 0...1.")) {
            try OBJMeshLoader.load("v 0 0 0 2 0.5 0\n")
        }
    }

    @Test("OBJ material names resolve through an independently supplied MTL file")
    func objMaterialBinding() throws {
        let mesh = try triangle()
        let obj = try OBJExporter.data(from: mesh, name: "wedge", materialName: "Painted Alloy", materialLibrary: "paints.mtl")
        #expect(String(decoding: obj, as: UTF8.self).hasPrefix("mtllib paints.mtl\no wedge\nusemtl Painted Alloy\n"))
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appending(path: "wedge.obj")
        let libraryURL = folder.appending(path: "paints.mtl")
        try "newmtl Painted Alloy\nKd 0.25 0.5 0.75\n".write(to: libraryURL, atomically: true, encoding: .utf8)
        let library = try MTLLoader.load(contentsOf: libraryURL)
        #expect(library.materials["Painted Alloy"]?.baseColorFactor == SIMD4(0.25, 0.5, 0.75, 1))

        // Model I/O reads a single token for usemtl. The full record above
        // remains available to consumers that support multiword names.
        try OBJExporter.data(from: mesh, name: "wedge", materialName: "Painted", materialLibrary: "paints.mtl").write(to: url)
        try "newmtl Painted\nKd 0.25 0.5 0.75\n".write(to: libraryURL, atomically: true, encoding: .utf8)
        let imported = try ModelIOSceneLoader.load(contentsOf: url)
        let material = try #require(imported.scene.materials.first(where: { $0.name == "Painted" }))
        #expect(material.baseColorFactor == SIMD4(0.25, 0.5, 0.75, 1))
        #expect(throws: OBJExporter.MaterialReferenceFailure.invalidName) {
            try OBJExporter.data(from: mesh, materialName: "Painted\nf 1 2 3")
        }
        #expect(throws: OBJExporter.MaterialReferenceFailure.invalidLibraryName) {
            try OBJExporter.data(from: mesh, materialName: "Painted", materialLibrary: "a.mtl b.mtl")
        }
    }

    @Test("OBJ and STL flatten mirrored posed instances and keep independent OBJ attribute indices",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Needs Metal for the scene's renderers"))
    func sceneExport() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try triangle()
        let morph = try MeshMorphTargets(
            vertexCount: geometry.vertices.count,
            targets: [.init(positionOffsets: Array(repeating: SIMD3(0, 0, 2), count: 3))]
        )
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: geometry, morphTargets: morph)
        let instances = try InstanceSet(device: device, transforms: [
            SceneTransform(translation: SIMD3(1, 0, 0)).matrix,
            SceneTransform(translation: SIMD3(4, 0, 0)).matrix,
        ])
        let mesh = SceneNode(mesh: renderer, instances: instances, morphWeights: [0.5], name: "panel", isVisible: false)
        let cloud = try PointCloud(positions: [SIMD3(0, 1, 0)])
        var points = SceneNode(name: "dots")
        points.primitive = .points(try PointsRenderer(device: device, cloud: cloud, material: PointsMaterial()))
        let ring = try LineGeometry(positions: [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0)], topology: .loop)
        var lines = SceneNode(name: "ring")
        lines.primitive = .lines(try LineRenderer(device: device, geometry: ring))
        let parent = SceneNode(
            transform: SceneTransform(translation: SIMD3(10, 0, 0), scale: SIMD3(-1, 1, 1)),
            children: [points, mesh, lines]
        )
        let graph = SceneGraph(roots: [parent], camera: try camera())

        let obj = String(decoding: try OBJExporter.data(from: graph), as: UTF8.self)
        #expect(obj.contains("o dots\nv 10.0 1.0 0.0\np 1\n"))
        #expect(obj.contains("o panel_0\n") && obj.contains("o panel_1\n"))
        #expect(obj.contains("f 2/1/1 4/3/3 3/2/2\n"))
        #expect(obj.contains("f 5/4/4 7/6/6 6/5/5\n"))
        #expect(obj.contains("l 8 9 10 8\n"))
        let bound = String(decoding: try OBJExporter.data(
            from: graph, materialNames: [mesh.id: "Panels"], defaultMaterialName: "Default"
        ), as: UTF8.self)
        #expect(bound.contains("o panel_0\nusemtl Panels\n") && bound.contains("o panel_1\nusemtl Panels\n"))
        #expect(bound.contains("o ring\nusemtl Default\n"))
        let exported = try OBJMeshLoader.load(obj)
        let expected: [SIMD3<Float>] = [
            SIMD3(9, 0, 1), SIMD3(9, 1, 1), SIMD3(7, 0, 1),
            SIMD3(6, 0, 1), SIMD3(6, 1, 1), SIMD3(4, 0, 1),
        ]
        #expect(exported.vertices.map(\.position) == expected)
        #expect(exported.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) })

        let stl = try STLExporter.data(from: graph, format: .binary)
        #expect(stl.count == 184)
        #expect(read(UInt32.self, from: stl, at: 80) == 2)
        for i in 0..<2 {
            let record = 84 + 50 * i
            #expect(vector(from: stl, at: record) == SIMD3(0, 0, 1))
            for vertex in 0..<3 {
                #expect(vector(from: stl, at: record + (vertex + 1) * 12) == expected[i * 3 + vertex])
            }
            #expect(read(UInt16.self, from: stl, at: record + 48) == 0)
        }
    }

    @Test("STL uses geometric facet normals and round-trips through ModelIO",
          arguments: [STLExporter.Format.ascii, .binary])
    func stlRoundTrip(format: STLExporter.Format) throws {
        let geometry = try triangle(normal: SIMD3(1, 0, 0))
        let data = try STLExporter.data(from: geometry, format: format, name: "wedge\nextra")
        if format == .ascii {
            let text = String(decoding: data, as: UTF8.self)
            #expect(text.hasPrefix("solid wedge_extra\n  facet normal 0.0 0.0 1.0\n"))
            #expect(text.hasSuffix("endsolid wedge_extra\n"))
        } else {
            #expect(data.count == 134)
            #expect(read(UInt32.self, from: data, at: 80) == 1)
            #expect(vector(from: data, at: 84) == SIMD3(0, 0, 1))
        }
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appending(path: "wedge.stl")
        try data.write(to: url, options: .atomic)
        let asset = try ModelIOSceneLoader.load(contentsOf: url)
        let restored = try #require(asset.scene.meshes.first?.primitives.first)
        #expect(restored.indices.count == 3)
        #expect(Set(restored.vertices.map(\.position)) == Set(geometry.vertices.map(\.position)))
        #expect(restored.vertices.allSatisfy { simd_distance($0.normal, SIMD3(0, 0, 1)) < 1e-6 })
    }

    @Test("Scene exports reject empty input and degenerate STL facets have zero normals")
    func emptySceneAndDegenerateFacet() throws {
        let graph = SceneGraph(roots: [], camera: try camera())
        #expect(throws: OBJExporter.Failure.noGeometry) { try OBJExporter.data(from: graph) }
        #expect(throws: STLExporter.Failure.noTriangles) { try STLExporter.data(from: graph) }
        let mesh = try MeshGeometry(vertices: triangle().vertices, indices: [0, 1, 1])
        let stl = try STLExporter.data(from: mesh, format: .binary)
        #expect(vector(from: stl, at: 84) == .zero)
    }

    private func triangle(normal: SIMD3<Float> = SIMD3(0, 0, 1)) throws -> MeshGeometry {
        try MeshGeometry(vertices: [
            .init(position: .zero, normal: normal, textureCoordinate: SIMD2(0.25, 0.75), color: SIMD4(0.25, 0.5, 0.75, 0.5)),
            .init(position: SIMD3(2, 0, 0), normal: normal, textureCoordinate: SIMD2(1, 1)),
            .init(position: SIMD3(0, 1, 0), normal: normal, textureCoordinate: SIMD2(0, 0)),
        ], indices: [0, 1, 2])
    }

    private func camera() throws -> PerspectiveCamera {
        try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 50)
    }

    private func read<T: FixedWidthInteger>(_ type: T.Type, from data: Data, at offset: Int) -> T {
        T(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) })
    }

    private func vector(from data: Data, at offset: Int) -> SIMD3<Float> {
        SIMD3(
            Float(bitPattern: read(UInt32.self, from: data, at: offset)),
            Float(bitPattern: read(UInt32.self, from: data, at: offset + 4)),
            Float(bitPattern: read(UInt32.self, from: data, at: offset + 8))
        )
    }
}
