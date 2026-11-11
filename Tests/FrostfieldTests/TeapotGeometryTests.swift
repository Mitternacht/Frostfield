import Frostfield
import Metal
import Testing
import simd

struct TeapotGeometryTests {
    @Test("The 32 teapot patches match r187 positions, normals, winding and texture coordinates")
    func referenceSamples() throws {
        let mesh = try MeshGeometry.teapot(segments: 4)
        try #require(mesh.vertices.count == 800)
        #expect(mesh.indices.count == 992 * 3)
        #expect(Array(mesh.indices.prefix(12)) == [0, 1, 6, 0, 6, 5, 1, 2, 7, 1, 7, 6])
        // Pinned r187 samples, with V converted to Frostfield's top-left convention.
        let reference: [(Int, SIMD3<Float>, SIMD3<Float>, SIMD2<Float>)] = [
            (0, SIMD3(0.888889, 0.523810, 0), SIMD3(-0.902861, -0.429934, 0), SIMD2(1, 0)),
            (6, SIMD3(0.808560, 0.570685, 0.344022), SIMD3(-0.918898, 0.095044, -0.382874), SIMD2(0.75, 0.25)),
            (19, SIMD3(0, 0.570685, 0.920139), SIMD3(0, 0.723356, 0.690476), SIMD2(0, 0.75)),
            (105, SIMD3(1.068948, 0.274554, 0), SIMD3(0.911768, 0.410706, 0), SIMD2(1, 0.25)),
            (307, SIMD3(-1.340774, 0.353795, 0.142857), SIMD3(0, 0, 1), SIMD2(0.5, 0.25)),
            (417, SIMD3(1.704365, 0.305804, 0.149554), SIMD3(0.062719, 0.177893, 0.982049), SIMD2(0.5, 0.75)),
            (500, SIMD3(0, 1, 0), SIMD3(0, 1, 0), SIMD2(1, 0)),
            (506, SIMD3(0.214952, 0.968750, 0.091666), SIMD3(0.526763, 0.821469, 0.218424), SIMD2(0.75, 0.25)),
            (622, SIMD3(0.631156, 0.523810, 0.631156), SIMD3(0.707107, 0, 0.707107), SIMD2(0.5, 1)),
            (700, SIMD3(0, -1, 0), SIMD3(0, -1, 0), SIMD2(1, 0)),
            (718, SIMD3(0.365340, -0.939732, -0.858666), SIMD3(0.197520, -0.858059, -0.474047), SIMD2(0.25, 0.75)),
            (797, SIMD3(0.676190, -0.904762, 0.676190), SIMD3(0.707107, 0, 0.707107), SIMD2(0.5, 1)),
        ]
        for (index, position, normal, uv) in reference {
            let vertex = mesh.vertices[index]
            #expect(simd_distance(vertex.position, position) < 0.000002)
            #expect(simd_distance(vertex.normal, normal) < 0.000002)
            #expect(vertex.textureCoordinate == uv)
        }
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = mesh.vertices[Int(mesh.indices[offset])]
            let b = mesh.vertices[Int(mesh.indices[offset + 1])]
            let c = mesh.vertices[Int(mesh.indices[offset + 2])]
            let face = simd_cross(b.position - a.position, c.position - a.position)
            #expect(simd_dot(face, a.normal + b.normal + c.normal) > 0)
        }
        let bounds = ShapeCheck.bounds(mesh.vertices.map(\.position))
        #expect(simd_distance(bounds.lower, SIMD3(-1.904762, -1, -1.269841)) < 0.000002)
        #expect(simd_distance(bounds.upper, SIMD3(2.176587, 1, 1.269841)) < 0.000002)
    }

    @Test("Lid fit, Blinn proportions and independently selected parts match r187")
    func partOptions() throws {
        let body = try MeshGeometry.teapot(segments: 4, bottom: false, lid: false)
        let bottom = try MeshGeometry.teapot(segments: 4, lid: false, body: false)
        let fitted = try MeshGeometry.teapot(segments: 4, bottom: false, body: false)
        let raw = try MeshGeometry.teapot(segments: 4, bottom: false, body: false, fitLid: false)
        #expect(body.vertices.count == 500 && body.indices.count == 640 * 3)
        #expect(bottom.vertices.count == 100 && bottom.indices.count == 112 * 3)
        #expect(fitted.vertices.count == 200 && fitted.indices.count == 240 * 3)
        #expect(raw.indices == fitted.indices)
        #expect(zip(raw.vertices, fitted.vertices).allSatisfy { original, scaled in
            simd_distance(original.position * SIMD3(1.077, 1, 1.077), scaled.position) < 1e-6
        })
        #expect(simd_distance(raw.vertices[6].normal, SIMD3(0.553123, 0.800907, 0.229354)) < 2e-6)

        let full = try MeshGeometry.teapot(segments: 4)
        let tall = try MeshGeometry.teapot(segments: 4, blinn: false)
        #expect(tall.indices == full.indices)
        #expect(zip(full.vertices, tall.vertices).allSatisfy { original, scaled in
            simd_distance(original.position * SIMD3(1 / 1.3, 1, 1 / 1.3), scaled.position) < 1e-6
        })
        #expect(simd_distance(tall.vertices[6].normal, SIMD3(-0.920597, 0.073246, -0.383582)) < 2e-6)
        let mixed = try MeshGeometry.teapot(size: 2, segments: 10, lid: false)
        #expect(mixed.vertices.count == 2_904 && mixed.indices.count == 4_760 * 3)
        #expect(mixed.vertices.map(\.position.y).min() == -2)
    }

    @Test("Pole rows keep independent UVs and tangent frames without emitting collapsed faces")
    func poles() throws {
        let steps = 7, rowLength = steps + 1
        for isLid in [true, false] {
            let mesh = try MeshGeometry.teapot(segments: steps, bottom: !isLid, lid: isLid, body: false)
            let direction: Float = isLid ? 1 : -1
            let patchCount = isLid ? 8 : 4
            #expect(mesh.vertices.count == patchCount * rowLength * rowLength)
            #expect(mesh.indices.count == (patchCount * 2 * steps * steps - 4 * steps) * 3)
            #expect(Array(mesh.indices.prefix(6)) == [0, UInt32(rowLength + 1), UInt32(rowLength), 1,
                                                      UInt32(rowLength + 2), UInt32(rowLength + 1)])
            for patch in 0..<4 {
                let start = patch * rowLength * rowLength
                let pole = mesh.vertices[start..<start + rowLength]
                #expect(pole.allSatisfy { $0.position == SIMD3(0, direction, 0) && $0.normal == SIMD3(0, direction, 0) })
                #expect(Set(pole.map(\.textureCoordinate)).count == rowLength)
            }
            for vertex in mesh.vertices {
                let tangent = try #require(vertex.tangent)
                let alongU = SIMD3(tangent.x, tangent.y, tangent.z)
                #expect(abs(simd_length(alongU) - 1) < 1e-6)
                #expect(abs(simd_dot(alongU, vertex.normal)) < 1e-6)
                #expect(tangent.w == -1)
            }
        }
    }

    @Test("The minimum segment count is two and invalid sizes, parts and budgets fail")
    func validation() throws {
        let coarse = try MeshGeometry.teapot(segments: 1)
        #expect(coarse.vertices.count == 288 && coarse.indices.count == 240 * 3)
        #expect(throws: MeshGeometry.TeapotError.size) { try MeshGeometry.teapot(size: .nan) }
        #expect(throws: MeshGeometry.TeapotError.noPatches) {
            try MeshGeometry.teapot(bottom: false, lid: false, body: false)
        }
        #expect(throws: MeshGeometry.TeapotError.resolution) { try MeshGeometry.teapot(segments: .max) }
        #expect(throws: MeshGeometry.TeapotError.resolution) { try MeshGeometry.teapot(segments: 499) }
        #expect(throws: MeshGeometry.TeapotError.unrepresentableSurface) {
            try MeshGeometry.teapot(size: .greatestFiniteMagnitude)
        }
    }

    @Test("Metal renders the teapot's body, handle, spout and removable lid",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func silhouette() throws {
        let gpu = try GeometryRenderFixture(side: 160)
        let depth: Float = 4
        let color = SIMD3<Float>(0.25, 0.75, 0.5)
        let fill = SIMD4(color, 1)
        func draw(lid: Bool) throws -> [SIMD4<Float>] {
            let mesh = try MeshGeometry.teapot(size: 0.75, segments: 12, lid: lid)
            let renderer = try gpu.unlit(mesh, material: .classic(ClassicMaterial(model: .basic, color: color)))
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -depth)), mesh: renderer)
            let rendered = try gpu.render(SceneGraph(roots: [node], camera: gpu.camera))
            #expect(rendered.statistics.drawnMeshCount == 1)
            return rendered.pixels
        }
        let full = try draw(lid: true)
        let open = try draw(lid: false)
        let center = gpu.side / 2
        let lidRow = gpu.row(y: 0.69, depth: depth)
        #expect(gpu.pixel(full, x: center, y: center) == fill)
        #expect(gpu.pixel(full, x: center, y: lidRow) == fill)
        #expect(gpu.pixel(open, x: center, y: lidRow) == .zero)
        let pixels = full.indices.filter { full[$0] == fill }
        #expect(pixels.count > 400)
        #expect(pixels.contains { $0 % gpu.side < center - 25 })
        #expect(pixels.contains { $0 % gpu.side > center + 29 })
    }
}
