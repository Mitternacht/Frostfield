import Frostfield
import Metal
import Testing
import simd

struct CurveDeformerTests {
    @Test("Straight flow preserves topology and attributes and maps normals by inverse transpose")
    func straight() throws {
        let path = Curve3D(position: { SIMD3(10 + 10 * $0, 3, -2) })
        let flow = try CurveDeformer(curve: path, subdivisions: 1)
        #expect(flow.length == 10)
        let mesh = try sourceTriangle()
        let result = try flow.deform(mesh, offset: 2, sourceXOrigin: 2, distanceScale: 2)
        #expect(result.indices == mesh.indices)
        #expect(result.groups == mesh.groups)
        #expect(result.vertices.map(\.position) == [SIMD3(12, 3, -2), SIMD3(14, 4, -2), SIMD3(12, 4, -1)])
        #expect(result.vertices.map(\.textureCoordinate) == mesh.vertices.map(\.textureCoordinate))
        #expect(result.vertices.map(\.secondaryTextureCoordinate) == mesh.vertices.map(\.secondaryTextureCoordinate))
        #expect(result.vertices.map(\.color) == mesh.vertices.map(\.color))
        let expectedNormal = simd_normalize(SIMD3<Float>(0.5, -1, 1))
        let expectedTangent = simd_normalize(SIMD3<Float>(2, 1, 0))
        let face = simd_cross(
            result.vertices[1].position - result.vertices[0].position,
            result.vertices[2].position - result.vertices[0].position
        )
        #expect(simd_dot(simd_normalize(face), expectedNormal) > 0.999_999)
        for vertex in result.vertices {
            let tangent = try #require(vertex.tangent)
            #expect(simd_distance(vertex.normal, expectedNormal) < 1e-6)
            #expect(simd_distance(xyz(tangent), expectedTangent) < 1e-6)
            #expect(tangent.w == -1)
        }
        #expect(try flow.transform(atDistance: -3).columns.3 == SIMD4(7, 3, -2, 1))
        #expect(try flow.transform(atDistance: 13).columns.3 == SIMD4(23, 3, -2, 1))
        let moved = try flow.deform(mesh, offset: -3, sourceXOrigin: 2)
        #expect(moved.vertices[0].position == SIMD3(7, 3, -2))
    }

    @Test("Quarter-circle flow bends the cross-section and tilts oblique normals by its local stretch")
    func curved() throws {
        let flow = try CurveDeformer(curve: quarterCircle(radius: 10), subdivisions: 512)
        #expect(abs(flow.length - 5 * .pi) < 0.000_1)
        let offset = flow.length * 0.5
        let frame = try flow.transform(atDistance: offset)
        let cosine = Float(0.5).squareRoot()
        let along = SIMD3(cosine, cosine, 0)
        let across = SIMD3(-cosine, cosine, 0)
        let center = SIMD3(10 * cosine, 10 * (1 - cosine), 0)
        #expect(simd_distance(xyz(frame.columns.3), center) < 0.000_1)
        #expect(simd_distance(xyz(frame.columns.0), along) < 0.000_1)
        #expect(simd_distance(xyz(frame.columns.1), across) < 0.000_1)

        let normal = simd_normalize(SIMD3<Float>(1, 1, 0))
        let tangent = simd_normalize(SIMD3<Float>(1, -1, 0))
        let mesh = try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(0, 2, 0), normal: normal, tangent: SIMD4(tangent, 1)),
            MeshVertex(position: SIMD3(0.25, 1.75, 0), normal: normal, tangent: SIMD4(tangent, 1)),
            MeshVertex(position: SIMD3(0.25, 1.75, 0.5), normal: normal, tangent: SIMD4(tangent, 1)),
        ], indices: [0, 2, 1])
        let result = try flow.deform(mesh, offset: offset)
        let vertex = result.vertices[0]
        #expect(simd_distance(vertex.position, center + 2 * across) < 0.000_2)
        // At an inside distance of 2, longitudinal stretch is 1 - 2/10.
        let expectedNormal = simd_normalize(along / 0.8 + across)
        let expectedTangent = simd_normalize(0.8 * along - across)
        #expect(simd_distance(vertex.normal, expectedNormal) < 0.001)
        #expect(simd_distance(xyz(try #require(vertex.tangent)), expectedTangent) < 0.001)
    }

    @Test("Closed flow wraps signed offsets and welds the orientation of a spatial loop")
    func closed() throws {
        let path = Curve3D(position: { parameter in
            let angle = 2 * Double.pi * parameter
            let radius = 2 + 0.35 * cos(3 * angle)
            return SIMD3(radius * cos(angle), radius * sin(angle), 0.35 * sin(3 * angle))
        })
        let flow = try CurveDeformer(curve: path, subdivisions: 512, boundary: .closed)
        let start = try flow.transform(atDistance: 0)
        let end = try flow.transform(atDistance: flow.length)
        #expect(start == end)
        let previous = try flow.transform(atDistance: -0.25)
        #expect(try previous == flow.transform(atDistance: flow.length - 0.25))
        let before = try flow.transform(atDistance: -0.000_1)
        let after = try flow.transform(atDistance: 0.000_1)
        for column in 0..<4 {
            #expect(simd_distance(before[column], after[column]) < 0.001)
        }

        let mesh = try MeshGeometry.plane(width: 0.2, height: 0.08, columns: 4)
        let crossing = try flow.deform(mesh, offset: flow.length)
        let atZero = try flow.deform(mesh)
        #expect(crossing.indices == mesh.indices)
        for (a, b) in zip(crossing.vertices, atZero.vertices) {
            #expect(simd_distance(a.position, b.position) < 1e-6)
            #expect(simd_distance(a.normal, b.normal) < 1e-6)
        }
    }

    @Test("A placement batch retains per-copy indices and material groups")
    func batching() throws {
        let first = try CurveDeformer(curve: Curve3D(position: { SIMD3($0 * 10, 0, 0) }), subdivisions: 2)
        let second = try CurveDeformer(curve: Curve3D(position: { SIMD3(4, $0 * 10, 0) }), subdivisions: 2)
        let mesh = try sourceTriangle()
        var placement = CurveDeformer.Placement(curve: first, offset: 1)
        placement.offset += 2
        let copies = try CurveDeformer.deform(
            mesh, placements: [placement, .init(curve: second, offset: 1)], sourceXOrigin: 2
        )
        #expect(copies.indexRanges == [0..<3, 3..<6])
        #expect(copies.geometry.indices == [0, 1, 2, 3, 4, 5])
        #expect(copies.geometry.groups == [
            MeshGeometryGroup(start: 0, count: 3, materialIndex: 2),
            MeshGeometryGroup(start: 3, count: 3, materialIndex: 2),
        ])
        #expect(copies.geometry.vertices[0].position == SIMD3(3, 0, 0))
        #expect(simd_distance(copies.geometry.vertices[3].position, SIMD3(4, 1, 0)) < 1e-6)
        let placementTransform = try second.transform(atDistance: 1)
        #expect(simd_distance(xyz(placementTransform.columns.0), SIMD3(0, 1, 0)) < 1e-6)
        #expect(simd_determinant(placementTransform) > 0.999_999)
    }

    @Test("Invalid curves, folded cross-sections and collapsed triangles report specific errors")
    func validation() throws {
        let straight = Curve3D(position: { SIMD3($0, 0, 0) })
        #expect(throws: CurveDeformer.Failure.openLoop) { try CurveDeformer(curve: straight, boundary: .closed) }
        #expect(throws: CurveDeformer.Failure.upDirection) { try CurveDeformer(curve: straight, up: .zero) }
        #expect(throws: CurveDeformer.Failure.resolution) { try CurveDeformer(curve: straight, subdivisions: 4097) }
        let stationary = Curve3D(position: { _ in .zero })
        #expect(throws: CurveDeformer.Failure.coincidentSample(1)) { try CurveDeformer(curve: stationary) }
        let reversal = Curve3D(position: { SIMD3(abs($0 - 0.5), 0, 0) })
        #expect(throws: CurveDeformer.Failure.reversingSample(2)) {
            try CurveDeformer(curve: reversal, subdivisions: 4)
        }
        let flow = try CurveDeformer(curve: straight)
        let plane = try MeshGeometry.plane()
        #expect(throws: CurveDeformer.Failure.placement) { try flow.deform(plane, distanceScale: 0) }
        #expect(throws: CurveDeformer.Failure.collapsedTriangle(0)) { try flow.deform(plane, offset: 1e20) }
        #expect(throws: CurveDeformer.Failure.meshBudget) { try CurveDeformer.deform(plane, placements: []) }

        let tight = try CurveDeformer(curve: quarterCircle(radius: 1))
        let wide = try MeshGeometry.plane(width: 0.1, height: 4)
        #expect(throws: CurveDeformer.Failure.foldedVertex(0)) {
            try tight.deform(wide, offset: tight.length * 0.5)
        }
    }

    private func sourceTriangle() throws -> MeshGeometry {
        let normal = simd_normalize(SIMD3<Float>(1, -1, 1))
        let tangent = SIMD4(simd_normalize(SIMD3<Float>(1, 1, 0)), -1)
        let positions: [SIMD3<Float>] = [SIMD3(2, 0, 0), SIMD3(3, 1, 0), SIMD3(2, 1, 1)]
        let vertices = positions.enumerated().map { index, position in
            MeshVertex(
                position: position, normal: normal, textureCoordinate: SIMD2(Float(index) / 2, 0),
                secondaryTextureCoordinate: SIMD2(0, Float(index) / 2), tangent: tangent,
                color: SIMD4(0.25, Float(index) / 2, 0.5, 1)
            )
        }
        return try MeshGeometry(vertices: vertices, indices: [0, 1, 2]).withGroups([
            MeshGeometryGroup(start: 0, count: 3, materialIndex: 2),
        ])
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CurveDeformerRenderTests {
    @Test("A batch of curved copies draws with the original front-face winding")
    func renders() throws {
        let fixture = try GeometryRenderFixture()
        let lower = try CurveDeformer(curve: quarterCircle(radius: 1.1))
        let upperPath = Curve3D(position: { t in
            let angle = t * Double.pi * 0.5
            return SIMD3(1.1 * sin(angle), 0.7 + 1.1 * (1 - cos(angle)), 0)
        })
        let upper = try CurveDeformer(curve: upperPath)
        let plane = try MeshGeometry.plane(width: 1.5, height: 0.24, columns: 16, rows: 2)
        let batch = try CurveDeformer.deform(
            plane, placements: [.init(curve: lower, offset: 0), .init(curve: upper, offset: 0)],
            sourceXOrigin: -0.75
        )
        let material = try ClassicMaterial(model: .basic, color: SIMD3(0.75, 0.25, 0.125))
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: batch.geometry, material: .classic(material)
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let output = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera))
        #expect(output.statistics.drawnMeshCount == 1)
        for path in [lower, upper] {
            let center = try path.transform(atDistance: 0.75).columns.3
            let x = fixture.column(x: center.x, depth: 2)
            let y = fixture.row(y: center.y, depth: 2)
            #expect(fixture.pixel(output.pixels, x: x, y: y) == SIMD4(0.75, 0.25, 0.125, 1))
        }
        let x = fixture.column(x: 0.7, depth: 2)
        #expect(fixture.pixel(output.pixels, x: x, y: 32) == .zero)
    }
}

private func quarterCircle(radius: Double) -> Curve3D {
    Curve3D(position: { parameter in
        let angle = 0.5 * Double.pi * parameter
        return SIMD3(radius * sin(angle), radius * (1 - cos(angle)), 0)
    })
}

private func xyz(_ value: SIMD4<Float>) -> SIMD3<Float> {
    SIMD3(value.x, value.y, value.z)
}
