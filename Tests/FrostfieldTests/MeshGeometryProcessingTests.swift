import Frostfield
import Metal
import Testing
import simd

struct MeshGeometryProcessingTests {
    @Test("Normals weight larger faces, preserve vertex attributes, and keep a fallback on unused vertices")
    func areaWeightedNormals() throws {
        let positions: [SIMD3<Float>] = [.zero, SIMD3(2, 0, 0), SIMD3(0, 2, 0), SIMD3(0, 0, 1), SIMD3(2, 2, 2)]
        let original = positions.enumerated().map { index, point in
            MeshVertex(
                position: point, normal: SIMD3(1, 0, 0), textureCoordinate: SIMD2(Float(index), 0),
                secondaryTextureCoordinate: SIMD2(0.5, 0.5), tangent: SIMD4(0, 1, 0, 1),
                color: SIMD4(0.25, 0.5, 0.75, 1)
            )
        }
        let groups = [MeshGeometryGroup(start: 0, count: 3, materialIndex: 1), .init(start: 3, count: 3, materialIndex: 2)]
        let mesh = try MeshGeometry(vertices: original, indices: [0, 1, 2, 0, 3, 1]).withGroups(groups)
        let generated = mesh.generatingVertexNormals()
        let expected = simd_normalize(SIMD3<Float>(0, 2, 4))
        #expect(simd_distance(generated.vertices[0].normal, expected) < 0.000001)
        #expect(simd_distance(generated.vertices[1].normal, expected) < 0.000001)
        #expect(generated.vertices[2].normal == SIMD3(0, 0, 1))
        #expect(generated.vertices[3].normal == SIMD3(0, 1, 0))
        #expect(generated.vertices[4].normal == SIMD3(1, 0, 0))
        #expect(generated.indices == mesh.indices && generated.groups == groups)
        for (a, b) in zip(original, generated.vertices) {
            #expect(a.position == b.position && a.textureCoordinate == b.textureCoordinate)
            #expect(a.secondaryTextureCoordinate == b.secondaryTextureCoordinate && a.color == b.color)
            #expect(b.tangent == nil)
        }
    }

    @Test("Affine baking keeps material groups and corrects mirrored winding, normals and tangent signs")
    func affineAndCentering() throws {
        let normal = simd_normalize(SIMD3<Float>(0, 1, 1))
        let vertices = [
            MeshVertex(position: .zero, normal: normal, tangent: SIMD4(1, 0, 0, 1)),
            MeshVertex(position: SIMD3(1, 0, 0), normal: normal, tangent: SIMD4(1, 0, 0, 1)),
            MeshVertex(position: SIMD3(0, 1, -1), normal: normal, tangent: SIMD4(1, 0, 0, 1)),
        ]
        let group = MeshGeometryGroup(start: 0, count: 3, materialIndex: 2)
        let mesh = try MeshGeometry(vertices: vertices, indices: [0, 1, 2]).withGroups([group])
        let matrix = SceneTransform(translation: SIMD3(7, -2, 3), scale: SIMD3(-2, 3, 0.5)).matrix
        let baked = try mesh.transformed(by: matrix)
        #expect(baked.indices == [0, 2, 1] && baked.groups == [group])
        #expect(baked.vertices[1].position == SIMD3(5, -2, 3))
        #expect(baked.vertices[0].tangent == SIMD4(-1, 0, 0, -1))
        #expect(simd_distance(baked.vertices[0].normal, simd_normalize(SIMD3(0, 1.0 / 3, 2))) < 0.000001)
        let centered = try baked.centered()
        #expect(centered.bounds.center == .zero)
        #expect(centered.indices == baked.indices && centered.groups == [group])
        var projective = matrix
        projective.columns.0.w = 1
        #expect(throws: MeshGeometry.MergeFailure.invalidTransform(part: 0)) { try mesh.transformed(by: projective) }
    }

    @Test("A baked mirror produces the same raster pixels as a mirrored node",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func bakedMirrorRendering() throws {
        let fixture = try GeometryRenderFixture()
        let source = try GeometryRenderFixture.square(0.75)
        let transform = SceneTransform(translation: SIMD3(0.25, 0, -3), scale: SIMD3(-1, 0.5, 1))
        let live = SceneNode(transform: transform, mesh: try fixture.unlit(source))
        let baked = SceneNode(mesh: try fixture.unlit(source.transformed(by: transform.matrix)))
        let livePixels = try fixture.render(SceneGraph(roots: [live], camera: fixture.camera)).pixels
        let bakedPixels = try fixture.render(SceneGraph(roots: [baked], camera: fixture.camera)).pixels
        #expect(livePixels.contains { $0.w == 1 })
        #expect(bakedPixels == livePixels)
    }
}
