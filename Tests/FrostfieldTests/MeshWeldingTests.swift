import Frostfield
import Testing
import simd

/// Exact-position edge counts: a closed, crack-free surface has every edge in two triangles.
private func positionEdgeCounts(_ mesh: MeshGeometry) -> [SIMD2<UInt32>: Int] {
    var ids: [SIMD3<Float>: UInt32] = [:]
    func id(_ index: UInt32) -> UInt32 {
        let p = mesh.vertices[Int(index)].position
        let key = SIMD3(p.x == 0 ? 0 : p.x, p.y == 0 ? 0 : p.y, p.z == 0 ? 0 : p.z)
        if let found = ids[key] { return found }
        ids[key] = UInt32(ids.count)
        return UInt32(ids.count - 1)
    }
    var counts: [SIMD2<UInt32>: Int] = [:]
    for start in stride(from: 0, to: mesh.indices.count, by: 3) {
        for k in 0..<3 {
            let a = id(mesh.indices[start + k]), b = id(mesh.indices[start + (k + 1) % 3])
            counts[SIMD2(min(a, b), max(a, b)), default: 0] += 1
        }
    }
    return counts
}

private func triangleCorners(_ mesh: MeshGeometry, _ triangle: Int) -> (MeshVertex, MeshVertex, MeshVertex) {
    (mesh.vertices[Int(mesh.indices[3 * triangle])], mesh.vertices[Int(mesh.indices[3 * triangle + 1])],
     mesh.vertices[Int(mesh.indices[3 * triangle + 2])])
}

struct MeshWeldingTests {
    @Test
    func `Welding keeps attribute seams unless told to ignore them`() throws {
        let cube = MeshGeometry.cube()
        // Every cube corner has three normals and UVs, so attribute-aware welding keeps all 24.
        #expect(try cube.welded().vertices.count == 24)
        let topology = try cube.welded(tolerance: .position(1e-4))
        #expect(topology.vertices.count == 8)
        #expect(topology.indices.count == 36)
        #expect(positionEdgeCounts(topology).values.allSatisfy { $0 == 2 })
        // Ignoring normals but not UVs still separates the cube's UV charts.
        let byUV = try cube.welded(tolerance: .init(position: 1e-4, normalAngle: nil, textureCoordinate: 1e-4, tangentAngle: nil))
        #expect(byUV.vertices.count > 8)
    }

    @Test
    func `Vertices within tolerance merge even when they straddle a rounding boundary`() throws {
        // Three.js rounds x × 10⁴: 1.000049 → 10000 and 1.000051 → 10001, so it keeps both.
        let a = MeshVertex(position: SIMD3(1.000049, 0, 0), normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(0.5, 0.5))
        let b = MeshVertex(position: SIMD3(1.000051, 0, 0), normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(0.5, 0.5))
        let c = MeshVertex(position: SIMD3(0, 1, 0), normal: SIMD3(0, 0, 1))
        let d = MeshVertex(position: SIMD3(2, 1, 0), normal: SIMD3(0, 0, 1))
        let mesh = try MeshGeometry(vertices: [a, c, b, d, c], indices: [0, 1, 3, 2, 3, 4])
        let welded = try mesh.welded()
        #expect(welded.vertices.count == 3)
        // Representatives keep their own attributes: the lower index wins.
        #expect(welded.vertices[0].position == a.position)
        #expect(welded.indices == [0, 1, 2, 0, 2, 1])

        // The search grid has 2 × 10⁻⁴ cells: these two straddle the boundary at 1.0002.
        let left = MeshVertex(position: SIMD3(1.0001999, 0, 0), normal: SIMD3(0, 0, 1))
        let right = MeshVertex(position: SIMD3(1.0002001, 0, 0), normal: SIMD3(0, 0, 1))
        let straddling = try MeshGeometry(vertices: [left, c, d, right], indices: [0, 1, 2, 3, 2, 1])
        #expect(try straddling.welded().vertices.count == 3)

        // Two vertices of one triangle merging leaves a degenerate triangle.
        let sliver = try MeshGeometry(vertices: [a, b, c, d], indices: [0, 1, 2, 0, 3, 2])
        #expect(try sliver.welded().indices.count == 3)
        #expect(try sliver.welded(removingDegenerateTriangles: false).indices.count == 6)
        #expect(throws: MeshGeometry.GeometryProcessingError.noTrianglesRemain) {
            try MeshGeometry(vertices: [a, b, c], indices: [0, 1, 2]).welded()
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidTolerance) {
            try mesh.welded(tolerance: .init(position: -1))
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidTolerance) {
            try mesh.welded(tolerance: .init(normalAngle: 4))
        }
    }

    @Test
    func `Exact welding and normal-angle tolerance behave at their edges`() throws {
        let tilted = simd_normalize(SIMD3<Float>(0, sin(0.01), cos(0.01)))
        let vertices = [
            MeshVertex(position: .zero, normal: SIMD3(0, 0, 1)), MeshVertex(position: SIMD3(1, 0, 0), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0, 1, 0), normal: SIMD3(0, 0, 1)), MeshVertex(position: SIMD3(-0.0, 0, 0), normal: tilted),
        ]
        let mesh = try MeshGeometry(vertices: vertices, indices: [0, 1, 2, 3, 1, 2])
        // −0 equals +0. A 0.01 radian normal difference merges at 0.02 and splits at 0.005.
        #expect(try mesh.welded(tolerance: .init(position: 0, normalAngle: 0.02)).vertices.count == 3)
        #expect(try mesh.welded(tolerance: .init(position: 0, normalAngle: 0.005), removingDegenerateTriangles: false).vertices.count == 4)
    }

    @Test
    func `Creased normals split at sharp edges and smooth across UV seams`() throws {
        let corners = try MeshGeometry.cube().welded(tolerance: .position(1e-4))
        let flat = try corners.creasedNormals()
        #expect(flat.vertices.count == 24)
        for triangle in 0..<12 {
            let (a, b, c) = triangleCorners(flat, triangle)
            let face = simd_normalize(simd_cross(b.position - a.position, c.position - a.position))
            for vertex in [a, b, c] { #expect(simd_distance(vertex.normal, face) < 1e-6) }
        }
        // With no crease, each corner's three faces weigh 90° each, whatever the triangulation.
        let smooth = try corners.creasedNormals(creaseAngle: .pi)
        #expect(smooth.vertices.count == 8)
        for vertex in smooth.vertices {
            #expect(simd_distance(vertex.normal, simd_normalize(vertex.position)) < 1e-6)
        }

        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 24, rings: 12)
        let resmoothed = try sphere.creasedNormals(creaseAngle: .pi / 3)
        // UV seam vertices stay separate but share one smooth normal. Corner-angle weighting on
        // this 24 × 12 sphere stays within 0.0105 (0.6°) of the true normal.
        #expect(resmoothed.vertices.count == sphere.vertices.count)
        let worst = resmoothed.vertices.map { simd_distance($0.normal, simd_normalize($0.position)) }.max() ?? 0
        #expect(worst < 0.011, "\(worst)")
        var seamNormals: [SIMD3<Float>: SIMD3<Float>] = [:]
        for vertex in resmoothed.vertices {
            if let shared = seamNormals[vertex.position] { #expect(shared == vertex.normal) }
            seamNormals[vertex.position] = vertex.normal
        }
        #expect(seamNormals.count < resmoothed.vertices.count)
        #expect(resmoothed.vertices.allSatisfy { $0.tangent == nil })
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidCreaseAngle) { try sphere.creasedNormals(creaseAngle: -1) }
    }

    @Test
    func `Edge split keeps unsplit normals and splits only along the fold`() throws {
        // A strip folded 90° along x = 0, with deliberately tilted normals.
        let marker = simd_normalize(SIMD3<Float>(0.1, 0.2, 1))
        let positions: [SIMD3<Float>] = [SIMD3(-1, 0, 0), SIMD3(0, 0, 0), SIMD3(0, 0, 1), SIMD3(-1, 0, 1),
                                         SIMD3(0, -1, 0), SIMD3(0, -1, 1)]
        let strip = try MeshGeometry(vertices: positions.map { MeshVertex(position: $0, normal: marker) },
                                     indices: [0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2])
        let split = try strip.splittingEdges(creaseAngle: .pi / 4)
        #expect(split.vertices.count == 8)
        let fold = split.vertices.filter { $0.position.x == 0 && $0.position.y == 0 }
        #expect(fold.count == 4)
        #expect(fold.allSatisfy { simd_distance($0.normal, SIMD3(0, -1, 0)) < 1e-6 || simd_distance($0.normal, SIMD3(-1, 0, 0)) < 1e-6 })
        let away = split.vertices.filter { !($0.position.x == 0 && $0.position.y == 0) }
        #expect(away.allSatisfy { $0.normal == marker })
        #expect(try strip.splittingEdges(creaseAngle: .pi / 4, keepingNormals: false).vertices.allSatisfy { $0.normal != marker })
        #expect(try strip.splittingEdges(creaseAngle: .pi * 0.75).vertices.count == 6)
    }

    @Test
    func `Tessellation bounds edge length without cracks across seams`() throws {
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 12, rings: 6)
        let limit: Float = 0.12
        let fine = try sphere.tessellated(maximumEdgeLength: limit)
        for triangle in 0..<(fine.indices.count / 3) {
            let (a, b, c) = triangleCorners(fine, triangle)
            for (p, q) in [(a, b), (b, c), (c, a)] { #expect(simd_distance(p.position, q.position) <= limit) }
            #expect(simd_dot(simd_cross(b.position - a.position, c.position - a.position), a.position) > 0)
        }
        // Closed and crack free: every edge, by position, is shared by exactly two triangles.
        #expect(positionEdgeCounts(fine).values.allSatisfy { $0 == 2 })
        #expect(positionEdgeCounts(sphere).values.allSatisfy { $0 == 2 })
        // Every pass is crack free on its own, not only after edges converge.
        #expect(positionEdgeCounts(try sphere.tessellated(maximumEdgeLength: limit, maximumPasses: 1)).values.allSatisfy { $0 == 2 })
        #expect(fine.indices.count > 8 * sphere.indices.count)
        // Midpoints interpolate: on a plane, UV stays the affine function of position.
        let plane = try MeshGeometry.plane(width: 2, height: 1).tessellated(maximumEdgeLength: 0.2)
        for vertex in plane.vertices {
            #expect(simd_distance(vertex.textureCoordinate, SIMD2(vertex.position.x / 2 + 0.5, 0.5 - vertex.position.y)) < 1e-6)
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.tooManyTriangles) {
            try sphere.tessellated(maximumEdgeLength: 0.001, maximumTriangleCount: 10_000)
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidTessellation) { try sphere.tessellated(maximumEdgeLength: 0) }
    }

    @Test func `Welding remaps material slots and respects external channels`() throws {
        let normal = SIMD3<Float>(0, 0, 1)
        let a = MeshVertex(position: .zero, normal: normal, color: SIMD4(1, 0, 0, 1))
        let b = MeshVertex(position: SIMD3(1, 0, 0), normal: normal)
        let c = MeshVertex(position: SIMD3(0, 1, 0), normal: normal)
        let alternate = MeshVertex(position: .zero, normal: normal, color: SIMD4(0, 0, 1, 1))
        let mesh = try MeshGeometry(vertices: [a, b, c, a, alternate, a], indices: [0, 3, 1, 3, 1, 2, 4, 1, 2])
            .withGroups([.init(start: 0, count: 6, materialIndex: 3), .init(start: 6, count: 3, materialIndex: 8)])
        let result = try mesh.weldedWithRemap()
        #expect(result.sourceVertexIndices == [0, 1, 2, 4])
        #expect(result.geometry.vertices.last?.color == alternate.color)
        #expect(result.geometry.groups == [.init(start: 0, count: 3, materialIndex: 3), .init(start: 3, count: 3, materialIndex: 8)])
        let weights = [0, 1, 1, 2, 0, 0]
        let attached = try mesh.weldedWithRemap { weights[$0] == weights[$1] }
        #expect(attached.sourceVertexIndices == [0, 1, 2, 3, 4])
        #expect(attached.geometry.groups == mesh.groups)
        let onlyCollapsed = try mesh.withGroups([.init(start: 0, count: 3)])
        #expect(throws: MeshGeometry.GeometryProcessingError.noMaterialGroupsRemain) {
            try onlyCollapsed.welded()
        }
    }

    @Test func `Edge splitting keeps original slots and copies UVs and colors to every split`() throws {
        let marker = SIMD3<Float>(0, 0, 1)
        let points: [SIMD3<Float>] = [.zero, SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        let vertices = points.enumerated().map { index, p in
            MeshVertex(
                position: p, normal: marker, textureCoordinate: SIMD2(Float(index), 0),
                secondaryTextureCoordinate: SIMD2(0, Float(index)), tangent: SIMD4(1, 0, 0, 1),
                color: SIMD4(Float(index), 0, 0, 1)
            )
        }
        let mesh = try MeshGeometry(vertices: vertices, indices: [2, 0, 1, 3, 1, 0])
            .withGroups([.init(start: 0, count: 3, materialIndex: 3), .init(start: 3, count: 3, materialIndex: 4)])
        let split = try mesh.splittingEdgesWithRemap(creaseAngle: .pi / 4)
        #expect(split.sourceVertexIndices == [0, 1, 2, 3, 1, 0])
        #expect(split.geometry.indices == [2, 0, 1, 3, 4, 5])
        #expect(split.geometry.groups == mesh.groups)
        for (vertex, source) in zip(split.geometry.vertices, split.sourceVertexIndices) {
            let original = mesh.vertices[source]
            #expect(vertex.position == original.position)
            #expect(vertex.color == original.color)
            #expect(vertex.textureCoordinate == original.textureCoordinate)
            #expect(vertex.secondaryTextureCoordinate == original.secondaryTextureCoordinate)
            #expect(vertex.tangent == (source >= 2 ? original.tangent : nil))
        }
        let smooth = try mesh.splittingEdgesWithRemap(creaseAngle: .pi)
        #expect(smooth.sourceVertexIndices == Array(mesh.vertices.indices))
        #expect(smooth.geometry.indices == mesh.indices)
    }

    @Test func `Tessellation carries attachments and groups across multiple passes`() throws {
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let vertices = plane.vertices.map {
            MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate,
                       color: SIMD4($0.textureCoordinate.x, $0.textureCoordinate.y, 0.5, 0.25))
        }
        let mesh = try MeshGeometry(vertices: vertices, indices: plane.indices)
            .withGroups([.init(start: 0, count: 3, materialIndex: 7), .init(start: 3, count: 3, materialIndex: 2)])
        let tessellation = try mesh.tessellatedWithRemap(maximumEdgeLength: 0.8, maximumPasses: 4)
        let fine = tessellation.geometry
        #expect(tessellation.midpointSources.count == fine.vertices.count - mesh.vertices.count)
        var customChannel = mesh.vertices.map { $0.position.x * 0.5 + 0.5 }
        for edge in tessellation.midpointSources {
            let midpoint = (customChannel[Int(edge.x)] + customChannel[Int(edge.y)]) * 0.5
            customChannel.append(midpoint)
        }
        for (index, vertex) in fine.vertices.enumerated() {
            #expect(vertex.color?.x == customChannel[index])
            #expect(vertex.color?.w == 0.25)
        }
        #expect(fine.groups.map(\.materialIndex) == [7, 2])
        #expect(fine.groups[0].range.upperBound == fine.groups[1].start)
        #expect(fine.groups[1].range.upperBound == fine.indices.count)
        #expect(fine.estimatedGPUByteCount == fine.vertices.count * 80 + fine.indices.count * 4)
        #expect(plane.estimatedGPUByteCount == plane.vertices.count * 64 + plane.indices.count * 4)
        #expect(throws: MeshGeometry.GeometryProcessingError.tooManyTriangles) {
            try mesh.tessellated(maximumEdgeLength: 100, maximumTriangleCount: 1)
        }
    }

    @Test func `Tessellation handles Float scale and rejects an unrepresentable midpoint`() throws {
        let normal = SIMD3<Float>(0, 0, 1)
        func triangle(_ a: Float, _ b: Float, height: Float) throws -> MeshGeometry {
            try MeshGeometry(vertices: [
                MeshVertex(position: SIMD3(a, 0, 0), normal: normal),
                MeshVertex(position: SIMD3(b, 0, 0), normal: normal),
                MeshVertex(position: SIMD3(a, height, 0), normal: normal),
            ], indices: [0, 1, 2])
        }
        let huge = try triangle(2e38, 3e38, height: 1e38)
        let divided = try huge.tessellated(maximumEdgeLength: 9e37, maximumPasses: 1)
        #expect(divided.indices.count == 12)
        #expect(divided.vertices.allSatisfy { $0.position.x.isFinite })
        let adjacent = try triangle(1, Float(1).nextUp, height: 0.01)
        #expect(throws: MeshGeometry.GeometryProcessingError.unrepresentableMidpoint) {
            try adjacent.tessellated(maximumEdgeLength: 1e-8, maximumPasses: 1)
        }
    }

    @Test("Separate vertex channels interleave in order and read back")
    func interleaveAttributes() throws {
        let mesh = try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(1, 2, 3), normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(0.25, 0.5)),
            MeshVertex(position: SIMD3(4, 5, 6), normal: SIMD3(0, 1, 0), textureCoordinate: SIMD2(1, 0)),
        ], indices: [0, 1, 0])
        let packed = try mesh.interleavedAttributes()
        #expect(packed.vertexCount == 2)
        #expect(packed.stride == 8)
        #expect(packed.values == [1, 2, 3, 0, 0, 1, 0.25, 0.5, 4, 5, 6, 0, 1, 0, 1, 0])
        #expect(try packed.values(named: "uv") == [0.25, 0.5, 1, 0])
        #expect(throws: InterleavedVertexAttributes.Failure.count("uv")) {
            try InterleavedVertexAttributes([
                .init(name: "position", components: 3, values: [1, 2, 3]),
                .init(name: "uv", components: 2, values: [0, 0, 1, 0]),
            ])
        }
        #expect(throws: InterleavedVertexAttributes.Failure.nonfinite("position")) {
            try InterleavedVertexAttributes([
                .init(name: "position", components: 3, values: [1, .nan, 3]),
            ])
        }
    }

}
