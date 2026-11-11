import Frostfield
import Metal
import Testing
import simd

struct MeshTangentGenerationTests {
    private static func mesh(_ rows: [[Float]], _ indices: [UInt32]) throws -> MeshGeometry {
        try MeshGeometry(vertices: rows.map { row in
            MeshVertex(
                position: SIMD3(row[0], row[1], row[2]), normal: SIMD3(row[3], row[4], row[5]),
                textureCoordinate: SIMD2(row[6], row[7])
            )
        }, indices: indices)
    }

    /// Compares each corner's generated frame with Blender's. Corners Blender leaves without
    /// a frame must instead get a unit tangent perpendicular to the normal with sign +1.
    private static func expectAgreement(
        _ result: MeshGeometry, source: MeshGeometry, reference: [SIMD4<Float>], frameless: Set<UInt32> = [],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(result.indices.count == source.indices.count, sourceLocation: sourceLocation)
        for corner in source.indices.indices {
            let vertex = result.vertices[Int(result.indices[corner])]
            let original = source.vertices[Int(source.indices[corner])]
            #expect(vertex.position == original.position && vertex.textureCoordinate == original.textureCoordinate,
                    sourceLocation: sourceLocation)
            guard let tangent = vertex.tangent else {
                Issue.record("corner \(corner) has no tangent", sourceLocation: sourceLocation)
                continue
            }
            let expected = reference[corner]
            if frameless.contains(source.indices[corner]) {
                #expect(expected == SIMD4(1, 0, 0, -1), sourceLocation: sourceLocation)
                let direction = SIMD3(tangent.x, tangent.y, tangent.z)
                #expect(abs(simd_dot(direction, simd_normalize(original.normal))) < 1e-6, "corner \(corner)",
                        sourceLocation: sourceLocation)
                #expect(tangent.w == 1, "corner \(corner)", sourceLocation: sourceLocation)
                continue
            }
            #expect(simd_distance(tangent, expected) < 2e-6, "corner \(corner): \(tangent) vs \(expected)",
                    sourceLocation: sourceLocation)
        }
    }

    @Test
    func `Tangents match Blender's MikkTSpace across mirrored UVs, zero UV area and degenerate triangles`() throws {
        let saddle = try Self.mesh(MikkTSpaceReference.saddleVertices, MikkTSpaceReference.saddleIndices)
        Self.expectAgreement(
            try saddle.generatingTangents(convention: .openGL), source: saddle, reference: MikkTSpaceReference.saddleOpenGL,
            frameless: [15, 16]
        )
        Self.expectAgreement(
            try saddle.generatingTangents(convention: .directX), source: saddle, reference: MikkTSpaceReference.saddleDirectX,
            frameless: [15, 16]
        )
        let sphere = try Self.mesh(MikkTSpaceReference.sphereVertices, MikkTSpaceReference.sphereIndices)
        Self.expectAgreement(
            try sphere.generatingTangents(convention: .openGL), source: sphere, reference: MikkTSpaceReference.sphereOpenGL
        )
        Self.expectAgreement(
            try sphere.generatingTangents(convention: .directX), source: sphere, reference: MikkTSpaceReference.sphereDirectX
        )
    }

    @Test
    func `Vertices split once per distinct frame and keep their source index for the first`() throws {
        let saddle = try Self.mesh(MikkTSpaceReference.saddleVertices, MikkTSpaceReference.saddleIndices)
        let result = try saddle.generatingTangents()
        var frames: [UInt32: Set<SIMD4<Float>>] = [:]
        for (corner, index) in saddle.indices.enumerated() {
            let reference = MikkTSpaceReference.saddleOpenGL[corner]
            frames[index, default: []].insert(index == 15 || index == 16 ? SIMD4(0, 0, 0, 0) : reference)
        }
        // The x = 0 column (vertices 2, 7, 12) and its copy 17 carry two orientations.
        let expectedCount = saddle.vertices.count + frames.values.reduce(0) { $0 + $1.count - 1 }
        #expect(expectedCount == saddle.vertices.count + 3)
        #expect(result.vertices.count == expectedCount)
        for (corner, index) in result.indices.enumerated() where index < UInt32(saddle.vertices.count) {
            #expect(index == saddle.indices[corner])
        }
        // Vertex 17 copies vertex 7's attributes, so MikkTSpace welds it into vertex 7's
        // left-hand group: it matches 7's left-hand corner (35), not 7's first frame.
        #expect(result.vertices[17].tangent == result.vertices[Int(result.indices[35])].tangent)
        #expect(result.vertices[17].tangent != result.vertices[7].tangent)
    }

    @Test("Tangent splits preserve vertex colors and material groups and report each source vertex")
    func vertexChannels() throws {
        let saddle = try Self.mesh(MikkTSpaceReference.saddleVertices, MikkTSpaceReference.saddleIndices)
        let vertices = saddle.vertices.enumerated().map { index, vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: SIMD2(vertex.textureCoordinate.y, vertex.textureCoordinate.x),
                color: SIMD4(Float(index) / Float(saddle.vertices.count), 0.25, 0.5, 1)
            )
        }
        let groups = [
            MeshGeometryGroup(start: 0, count: 6),
            MeshGeometryGroup(start: 6, count: saddle.indices.count - 6, materialIndex: 1),
        ]
        let source = try MeshGeometry(vertices: vertices, indices: saddle.indices).withGroups(groups)
        let result = try source.generatingTangentsWithRemap()
        let geometry = result.geometry
        let remap = result.sourceVertexIndices
        #expect(geometry.groups == source.groups)
        #expect(geometry.vertices.count > vertices.count)
        try #require(remap.count == geometry.vertices.count)
        try #require(remap.allSatisfy { vertices.indices.contains($0) })
        #expect(Array(remap.prefix(vertices.count)) == Array(vertices.indices))
        #expect(geometry.indices.map { remap[Int($0)] } == source.indices.map(Int.init))
        for (vertex, index) in zip(geometry.vertices, remap) {
            #expect(vertex.color == vertices[index].color)
            #expect(vertex.secondaryTextureCoordinate == vertices[index].secondaryTextureCoordinate)
        }
    }

    @Test("Finite normals with extreme magnitudes still produce a unit tangent",
          arguments: [Float.leastNormalMagnitude, Float.greatestFiniteMagnitude])
    func normalMagnitude(_ magnitude: Float) throws {
        let plane = try MeshGeometry.plane(width: 1, height: 1)
        let vertices = plane.vertices.map {
            MeshVertex(position: $0.position, normal: SIMD3(0, 0, magnitude), textureCoordinate: $0.textureCoordinate)
        }
        let geometry = try MeshGeometry(vertices: vertices, indices: plane.indices).generatingTangents()
        for vertex in geometry.vertices {
            let tangent = try #require(vertex.tangent)
            #expect(simd_distance(tangent, SIMD4(1, 0, 0, 1)) < 1e-6)
        }
    }

    @Test
    func `A front-facing plane gets plus X with the sign of each normal-map convention`() throws {
        let plane = try MeshGeometry.plane(width: 2, height: 1, columns: 3, rows: 2)
        let openGL = try plane.generatingTangents(convention: .openGL)
        let directX = try plane.generatingTangents(convention: .directX)
        #expect(openGL.vertices.count == plane.vertices.count)
        #expect(openGL.vertices.allSatisfy { simd_distance($0.tangent ?? .zero, SIMD4(1, 0, 0, 1)) < 1e-6 })
        #expect(directX.vertices.allSatisfy { simd_distance($0.tangent ?? .zero, SIMD4(1, 0, 0, -1)) < 1e-6 })
        // Secondary coordinates are opt-in and must exist on every referenced vertex.
        #expect(throws: MeshGeometry.TangentGenerationError.missingSecondaryTextureCoordinates(vertex: 0)) {
            try plane.generatingTangents(textureCoordinateSet: .secondary)
        }
        let rotated = try MeshGeometry(vertices: plane.vertices.map {
            MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate,
                       secondaryTextureCoordinate: SIMD2($0.textureCoordinate.y, $0.textureCoordinate.x))
        }, indices: plane.indices)
        // Swapping U and V makes U run down the plane (−Y) and mirrors the map.
        let secondary = try rotated.generatingTangents(textureCoordinateSet: .secondary)
        #expect(secondary.vertices.allSatisfy { simd_distance($0.tangent ?? .zero, SIMD4(0, -1, 0, -1)) < 1e-6 })
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshTangentRenderingTests {
    @Test
    func `Generated tangents decode each convention's green channel in the renderer`() throws {
        let fixture = try MaterialRenderFixture()
        let side = MaterialRenderFixture.side
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        // A uniform normal map leaning along green: tangent space (0, 0.6, 0.8).
        let normalMap = try fixture.dataTexture(width: 1, height: 1, pixels: [128, 204, 230, 255])
        let placement = SceneTransform(translation: SIMD3(0, 0, -3))
        let plane = try MeshGeometry.plane(width: 4, height: 4)
        let local = simd_normalize(SIMD3<Float>(128, 204, 230) / 255 * 2 - 1)
        let fromAbove = simd_normalize(SIMD3<Float>(0, 1, 1))
        let fromBelow = simd_normalize(SIMD3<Float>(0, -1, 1))

        func brightness(_ geometry: MeshGeometry, light: SIMD3<Float>) throws -> Float {
            let material = try PhysicallyBasedMaterial(
                roughness: 1, directionToLight: light, ambientIntensity: 0,
                normalTexture: .init(texture: normalMap), alphaMode: .opaque, specular: .init(factor: 0)
            )
            let image = try fixture.raster(material, geometry: geometry, transform: placement, camera: camera)
            return image[(side / 2) * side + side / 2].x
        }

        // Lambertian white reflects cos(normal, light) / π. Green tilts toward +Y (up the
        // image) for OpenGL and toward −Y (increasing top-left V) for DirectX.
        let cases: [(MeshGeometry, SIMD3<Float>)] = [
            (try plane.generatingTangents(convention: .openGL), SIMD3(local.x, local.y, local.z)),
            (try plane.generatingTangents(convention: .directX), SIMD3(local.x, -local.y, local.z)),
            // The derivative fallback for meshes without tangents decodes DirectX green.
            (plane, SIMD3(local.x, -local.y, local.z)),
        ]
        for (index, (geometry, normal)) in cases.enumerated() {
            for light in [fromAbove, fromBelow] {
                let expected = simd_dot(normal, light) / .pi
                let measured = try brightness(geometry, light: light)
                #expect(abs(measured - expected) < 0.01 * expected, "case \(index), light \(light): \(measured) vs \(expected)")
            }
        }
        #expect(simd_dot(SIMD3(local.x, local.y, local.z), fromAbove) > simd_dot(SIMD3(local.x, local.y, local.z), fromBelow) + 0.5)
    }
}
