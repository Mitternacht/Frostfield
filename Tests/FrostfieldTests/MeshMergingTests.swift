import Frostfield
import Testing
import simd

private func triangleCorners(_ mesh: MeshGeometry, _ triangle: Int) -> (MeshVertex, MeshVertex, MeshVertex) {
    (mesh.vertices[Int(mesh.indices[3 * triangle])], mesh.vertices[Int(mesh.indices[3 * triangle + 1])],
     mesh.vertices[Int(mesh.indices[3 * triangle + 2])])
}

struct MeshMergingTests {
    @Test
    func `Merging bakes transforms, keeps fronts under mirroring and reports index ranges`() throws {
        let plane = try MeshGeometry.plane(width: 2, height: 2).generatingTangents()
        let stretch = SceneTransform(translation: SIMD3(5, 0, 0), scale: SIMD3(2, 1, 0.5)).matrix
        let mirror = SceneTransform(translation: SIMD3(0, 0, -3), rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0)),
                                    scale: SIMD3(-1, 1, 1)).matrix
        let merged = try MeshGeometry.merged([.init(plane), .init(plane, transform: stretch), .init(MeshGeometry.cube(), transform: mirror)])
        #expect(merged.indexRanges == [0..<6, 6..<12, 12..<48])
        #expect(merged.geometry.vertices.count == 4 + 4 + 24)
        let stretched = merged.geometry.vertices[4]
        #expect(stretched.position == SIMD3<Float>(3, 1, 0))
        #expect(simd_distance(stretched.normal, SIMD3(0, 0, 1)) < 1e-6)
        #expect(simd_distance(stretched.tangent ?? .zero, SIMD4(1, 0, 0, 1)) < 1e-6)
        // Every triangle still faces along its vertex normals, including the mirrored cube.
        for triangle in 0..<(merged.geometry.indices.count / 3) {
            let (a, b, c) = triangleCorners(merged.geometry, triangle)
            let face = simd_cross(b.position - a.position, c.position - a.position)
            #expect(simd_dot(face, a.normal + b.normal + c.normal) > 0, "triangle \(triangle)")
        }
        // A non-uniform scale tilts normals by the inverse transpose.
        let shear = SceneTransform(scale: SIMD3(1, 4, 1)).matrix
        let slope = try MeshGeometry(vertices: [
            MeshVertex(position: .zero, normal: simd_normalize(SIMD3(0, 1, 1))),
            MeshVertex(position: SIMD3(1, 0, 0), normal: simd_normalize(SIMD3(0, 1, 1))),
            MeshVertex(position: SIMD3(0, 1, -1), normal: simd_normalize(SIMD3(0, 1, 1))),
        ], indices: [0, 1, 2])
        let sheared = try MeshGeometry.merged([.init(slope, transform: shear)]).geometry
        #expect(simd_distance(sheared.vertices[0].normal, simd_normalize(SIMD3(0, 0.25, 1))) < 1e-6)
        // Mirroring negates tangent handedness.
        let mirrored = try MeshGeometry.merged([.init(plane, transform: SceneTransform(scale: SIMD3(-1, 1, 1)).matrix)]).geometry
        #expect(mirrored.vertices.allSatisfy { simd_distance($0.tangent ?? .zero, SIMD4(-1, 0, 0, -1)) < 1e-6 })
        var singular = matrix_identity_float4x4
        singular.columns.2 = .zero
        #expect(throws: MeshGeometry.MergeFailure.invalidTransform(part: 1)) {
            try MeshGeometry.merged([.init(plane), .init(plane, transform: singular)])
        }
        #expect(throws: MeshGeometry.MergeFailure.noParts) { try MeshGeometry.merged([]) }
    }

    @Test("Merging keeps colors and UVs and can assign one material group per part")
    func materialGroups() throws {
        let source = try MeshGeometry.plane(width: 2, height: 2)
        let vertices = source.vertices.enumerated().map { index, vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.textureCoordinate * 0.5,
                color: SIMD4(Float(index) / Float(source.vertices.count), 0.25, 0.5, 1)
            )
        }
        let colored = try MeshGeometry(vertices: vertices, indices: source.indices).withGroups([
            MeshGeometryGroup(start: 0, count: 3, materialIndex: 3),
        ])
        let parts: [MeshGeometry.MergePart] = [.init(source), .init(colored)]
        let merged = try MeshGeometry.merged(parts, useGroups: true)
        #expect(merged.geometry.groups == [
            MeshGeometryGroup(start: 0, count: 6, materialIndex: 0),
            MeshGeometryGroup(start: 6, count: 6, materialIndex: 1),
        ])
        let sourceVertices = source.vertices + colored.vertices
        #expect(merged.geometry.vertices.map(\.color) == sourceVertices.map(\.color))
        #expect(merged.geometry.vertices.map(\.secondaryTextureCoordinate) == sourceVertices.map(\.secondaryTextureCoordinate))
        #expect(try MeshGeometry.merged(parts).geometry.groups.isEmpty)
    }
}
