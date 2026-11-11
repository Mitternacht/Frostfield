import Frostfield
import Metal
import Testing
import simd

struct MeshUVAtlasTests {
    @Test("A connected tessellated plane stays in one chart")
    func planarChart() throws {
        let mesh = try MeshGeometry.plane(width: 4, height: 2, columns: 12, rows: 6)
        let atlas = try mesh.unwrappedUVAtlas(resolution: 128, padding: 4)
        #expect(atlas.chartCount == 1)
        #expect(atlas.geometry.vertices.count == mesh.vertices.count)
        let coordinates = try atlas.geometry.vertices.map { try #require($0.secondaryTextureCoordinate) }
        let a = coordinates.map(\.x)
        let b = coordinates.map(\.y)
        let width = try #require(a.max()) - #require(a.min())
        let height = try #require(b.max()) - #require(b.min())
        let aspect = max(width, height) / min(width, height)
        #expect(abs(aspect - 2) < 0.08)
        #expect(width * height > 0.35)
    }

    @Test("A closed surface is cut deterministically and every original corner and channel survives")
    func seamsAndRemap() throws {
        let torus = try MeshGeometry.torus(radialSegments: 12, tubularSegments: 24)
        let vertices = torus.vertices.enumerated().map { i, vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: .zero, tangent: vertex.tangent, color: SIMD4(Float(i) / 400, 0.2, 0.4, 1)
            )
        } + [MeshVertex(position: SIMD3(repeating: .greatestFiniteMagnitude), normal: SIMD3(0, 0, 1))]
        let groups = [
            MeshGeometryGroup(start: 0, count: 72, materialIndex: 4),
            MeshGeometryGroup(start: 144, count: torus.indices.count - 144, materialIndex: 2),
        ]
        let mesh = try MeshGeometry(vertices: vertices, indices: torus.indices).withGroups(groups)
        let atlas = try mesh.unwrappedUVAtlas(resolution: 256)
        let repeatAtlas = try mesh.unwrappedUVAtlas(resolution: 256)
        #expect(atlas.chartCount > 1)
        #expect(atlas.chartCount < mesh.indices.count / 6)
        #expect(atlas.geometry.vertices.count > torus.vertices.count)
        #expect(atlas.sourceVertexIndices == repeatAtlas.sourceVertexIndices)
        #expect(atlas.geometry.indices == repeatAtlas.geometry.indices)
        #expect(atlas.geometry.vertices.map(\.secondaryTextureCoordinate) == repeatAtlas.geometry.vertices.map(\.secondaryTextureCoordinate))
        #expect(atlas.geometry.groups == groups)
        #expect(atlas.geometry.indices.map { UInt32(atlas.sourceVertexIndices[Int($0)]) } == mesh.indices)
        #expect(!atlas.sourceVertexIndices.contains(vertices.count - 1))

        for (vertex, sourceIndex) in zip(atlas.geometry.vertices, atlas.sourceVertexIndices) {
            let source = mesh.vertices[sourceIndex]
            #expect(vertex.position == source.position)
            #expect(vertex.normal == source.normal)
            #expect(vertex.textureCoordinate == source.textureCoordinate)
            #expect(vertex.tangent == source.tangent)
            #expect(vertex.color == source.color)
        }
        let sourceDeltas = mesh.vertices.indices.map { SIMD3(0, Float($0) / 1_000, 0) }
        let deltas = atlas.sourceVertexIndices.map { sourceDeltas[$0] }
        let targets = try MeshMorphTargets(vertexCount: deltas.count, targets: [.init(positionOffsets: deltas)])
        let morphed = try targets.apply(weights: [1], to: atlas.geometry)
        for (index, vertex) in morphed.vertices.enumerated() {
            let sourceIndex = atlas.sourceVertexIndices[index]
            #expect(vertex.position == mesh.vertices[sourceIndex].position + sourceDeltas[sourceIndex])
        }
    }

    @Test("Packed UV1 feeds both bakers and separates islands' padding",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func bakeWorkflow() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let plane = try MeshGeometry.plane()
        let red = SIMD4<Float>(1, 0, 0, 1), green = SIMD4<Float>(0, 1, 0, 1)
        let vertices = [red, green].enumerated().flatMap { island, color in
            plane.vertices.map { vertex in
                MeshVertex(
                    position: vertex.position + SIMD3(Float(island) * 2, 0, 0), normal: vertex.normal,
                    textureCoordinate: vertex.textureCoordinate, color: color
                )
            }
        }
        let mesh = try MeshGeometry(vertices: vertices, indices: plane.indices + plane.indices.map { $0 + 4 })
        let size = 64, padding = 4
        let atlas = try mesh.unwrappedUVAtlas(resolution: size, padding: padding)
        #expect(atlas.chartCount == 2)
        let baked = try MeshTextureBaker(
            device: device, geometry: atlas.geometry, width: size, height: size,
            textureCoordinateSet: .secondary, padding: padding
        )
        let irradiance = SIMD3<Float>(2, 0.5, 0.25)
        let lightmap = try MeshLightmapBaker(
            device: device, geometry: atlas.geometry, width: size, height: size, textureCoordinateSet: .secondary,
            lights: [.init(directionToLight: SIMD3(0, 0, 1), irradiance: irradiance)], padding: padding
        )
        let region = MTLRegionMake2D(0, 0, size, size)
        var mask = [UInt8](repeating: 0, count: size * size)
        var colors = [SIMD4<Float16>](repeating: .zero, count: size * size)
        var light = colors
        mask.withUnsafeMutableBytes { baked.coverageTexture.getBytes($0.baseAddress!, bytesPerRow: size, from: region, mipmapLevel: 0) }
        colors.withUnsafeMutableBytes {
            baked.vertexColorTexture.getBytes($0.baseAddress!, bytesPerRow: size * 8, from: region, mipmapLevel: 0)
        }
        light.withUnsafeMutableBytes {
            lightmap.irradianceTexture.getBytes($0.baseAddress!, bytesPerRow: size * 8, from: region, mipmapLevel: 0)
        }
        #expect(mask.contains(128))
        #expect(mask.contains(0))
        #expect(Set(mask.indices.filter { mask[$0] == 255 }.map { colors[$0] }) == [SIMD4<Float16>(red), SIMD4<Float16>(green)])
        #expect(mask.indices.filter { mask[$0] != 0 }.allSatisfy { light[$0] == SIMD4<Float16>(SIMD4(irradiance, 1)) })
        let safeDistance = 2 * padding
        for pixel in mask.indices where mask[pixel] == 255 && colors[pixel] == SIMD4<Float16>(red) {
            let x = pixel % size, y = pixel / size
            for otherY in max(0, y - safeDistance)...min(size - 1, y + safeDistance) {
                let span = safeDistance - abs(otherY - y)
                for otherX in max(0, x - span)...min(size - 1, x + span) {
                    let other = otherY * size + otherX
                    #expect(mask[other] != 255 || colors[other] == SIMD4<Float16>(red))
                }
            }
        }
    }

    @Test("Invalid atlas options, degenerate faces and unfit charts fail explicitly")
    func validation() throws {
        let plane = try MeshGeometry.plane()
        #expect(throws: MeshGeometry.UVAtlasError.resolution) { try plane.unwrappedUVAtlas(resolution: 8192) }
        #expect(throws: MeshGeometry.UVAtlasError.padding) { try plane.unwrappedUVAtlas(resolution: 16, padding: 7) }
        let degenerate = try MeshGeometry(vertices: plane.vertices, indices: plane.indices + [0, 0, 1])
        #expect(throws: MeshGeometry.UVAtlasError.degenerateTriangle(2)) { try degenerate.unwrappedUVAtlas() }
        let cube = MeshGeometry.cube()
        #expect(throws: MeshGeometry.UVAtlasError.cannotPack) { try cube.unwrappedUVAtlas(resolution: 16, padding: 6) }
    }
}
