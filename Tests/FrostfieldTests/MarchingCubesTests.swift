import Frostfield
import Metal
import Testing
import simd

struct MarchingCubesTests {
    @Test("A sampled sphere is closed and indexed, with outward winding and gradients")
    func sampledSphere() throws {
        let grid = try sphereField()
        let mesh = try #require(try grid.mesh(isovalue: 0, boundary: .requireClosed))
        let edges = edgeCounts(mesh)
        #expect(edges.values.allSatisfy { $0 == 2 })
        #expect(mesh.vertices.count - edges.count + mesh.indices.count / 3 == 2)
        #expect(mesh.vertices.count < mesh.indices.count / 2)
        #expect(mesh.vertices.allSatisfy {
            (0.60...0.64).contains(simd_length($0.position))
                && simd_dot(simd_normalize($0.position), $0.normal) > 0.997
                && (0...1).contains($0.textureCoordinate.x) && (0...1).contains($0.textureCoordinate.y)
        })
        #expect(stride(from: 0, to: mesh.indices.count, by: 3).allSatisfy { start in
            let a = mesh.vertices[Int(mesh.indices[start])]
            let b = mesh.vertices[Int(mesh.indices[start + 1])]
            let c = mesh.vertices[Int(mesh.indices[start + 2])]
            return simd_dot(simd_cross(b.position - a.position, c.position - a.position),
                            a.normal + b.normal + c.normal) > 0
        })
        #expect(throws: MarchingCubes.Failure.triangleLimit) {
            try grid.mesh(isovalue: 0, maximumTriangles: 3)
        }
    }

    @Test("All nonuniform cube masks produce indexed triangles with valid edge incidence")
    func cubeMasks() throws {
        let bounds = try unitBounds()
        for mask in 1..<255 {
            let values = (0..<8).map { i in (mask & (1 << i) == 0 ? -1 : 1) * (1 + Float(i) / 16) }
            let grid = try MarchingCubes(resolution: SIMD3(repeating: 2), bounds: bounds, values: values)
            let mesh = try #require(try grid.mesh(isovalue: 0), "mask \(mask)")
            #expect(!mesh.indices.isEmpty)
            #expect(edgeCounts(mesh).values.allSatisfy { (1...2).contains($0) })
        }
    }

    @Test("Neighboring cubes share their checkerboard contour and agree on both saddle choices")
    func saddle() throws {
        for (positive, negative, connectedHigh): (Float, Float, Bool) in [(2, -1, true), (1, -2, false), (1, -1, true)] {
            let grid = try MarchingCubes(resolution: SIMD3(3, 2, 2), bounds: unitBounds()) { p in
                if p.x != 0 { return -0.75 }
                return p.y * p.z > 0 ? positive : negative
            }
            let mesh = try #require(try grid.mesh(isovalue: 0))
            let seam = edgeCounts(mesh).filter { edge, _ in
                mesh.vertices[Int(edge.low)].position.x == 0 && mesh.vertices[Int(edge.high)].position.x == 0
            }
            #expect(seam.count == 2)
            #expect(seam.values.allSatisfy { $0 == 2 })
            for edge in seam.keys {
                let midpoint = (mesh.vertices[Int(edge.low)].position + mesh.vertices[Int(edge.high)].position) * 0.5
                #expect(connectedHigh ? midpoint.y * midpoint.z < 0 : midpoint.y * midpoint.z > 0)
            }
        }
    }

    @Test("An exact-level plane welds grid intersections and exposes only outer boundary edges")
    func alignedPlane() throws {
        let grid = try MarchingCubes(resolution: SIMD3(3, 5, 7), bounds: unitBounds()) { $0.x }
        let mesh = try #require(try grid.mesh(isovalue: 0))
        #expect(mesh.vertices.allSatisfy { $0.position.x == 0 && $0.normal == SIMD3(-1, 0, 0) })
        let open = edgeCounts(mesh).filter { $0.value == 1 }
        #expect(open.count == 20)
        #expect(open.keys.allSatisfy {
            let a = mesh.vertices[Int($0.low)].position, b = mesh.vertices[Int($0.high)].position
            return (abs(a.y) == 1 && a.y == b.y) || (abs(a.z) == 1 && a.z == b.z)
        })
        #expect(throws: MarchingCubes.Failure.openSurface) { try grid.mesh(isovalue: 0, boundary: .requireClosed) }
        let singular = try MarchingCubes(resolution: SIMD3(3, 3, 3), bounds: unitBounds()) { abs($0.x) }
        #expect(throws: MarchingCubes.Failure.singularSurface) { try singular.mesh(isovalue: 0) }
    }

    @Test("Metaball potentials merge, support subtraction, and boundary planes generate open surfaces")
    func fieldOperations() throws {
        var grid = try MarchingCubes(resolution: SIMD3(repeating: 25), bounds: unitBounds())
        #expect(try grid.mesh(isovalue: 1) == nil)
        try grid.addMetaball(center: SIMD3(-0.18, 0, 0), strength: 0.04, subtract: 0.1)
        try grid.addMetaball(center: SIMD3(0.18, 0, 0), strength: 0.04, subtract: 0.1)
        let mesh = try #require(try grid.mesh(isovalue: 1, boundary: .requireClosed))
        let edges = edgeCounts(mesh)
        #expect(edges.values.allSatisfy { $0 == 2 })
        #expect(mesh.vertices.count - edges.count + mesh.indices.count / 3 == 2)
        #expect(mesh.bounds.minimum.x < -0.35 && mesh.bounds.maximum.x > 0.35)
        let before = grid.values
        try grid.addMetaball(center: .zero, strength: -0.02, subtract: 0.1)
        #expect(grid.values[grid.values.count / 2] < before[before.count / 2])

        grid.reset()
        try grid.addPlane(perpendicularTo: .x, at: -1, strength: 0.04, subtract: 0.1)
        let plane = try #require(try grid.mesh(isovalue: 1))
        #expect(plane.vertices.allSatisfy { (-0.82 ... -0.79).contains($0.position.x) && $0.normal == SIMD3(1, 0, 0) })
        #expect(throws: MarchingCubes.Failure.openSurface) { try grid.mesh(isovalue: 1, boundary: .requireClosed) }
    }

    @Test("Grid and potential validation preserves values and bounds work")
    func validation() throws {
        let bounds = try unitBounds()
        for invalid: SIMD3<Int> in [SIMD3(1, 2, 2), SIMD3(repeating: 1_024), SIMD3(Int.max, 2, 2)] {
            #expect(throws: MarchingCubes.Failure.resolution) {
                try MarchingCubes(resolution: invalid, bounds: bounds)
            }
        }
        let narrow = try AxisAlignedBox(minimum: SIMD3(1e20, 0, 0), maximum: SIMD3(Float(1e20).nextUp, 1, 1))
        #expect(throws: MarchingCubes.Failure.bounds) {
            try MarchingCubes(resolution: SIMD3(repeating: 3), bounds: narrow)
        }
        #expect(throws: MarchingCubes.Failure.sampleCount) {
            try MarchingCubes(resolution: SIMD3(repeating: 3), bounds: bounds, values: [0])
        }
        var grid = try MarchingCubes(resolution: SIMD3(repeating: 3), bounds: bounds)
        #expect(throws: MarchingCubes.Failure.sampleValue) {
            try grid.replaceValues(Array(repeating: .nan, count: 27))
        }
        #expect(throws: MarchingCubes.Failure.sampleValue) {
            try grid.addMetaball(center: .zero, strength: .greatestFiniteMagnitude, subtract: 1)
        }
        #expect(grid.values.allSatisfy { $0 == 0 })
        #expect(throws: MarchingCubes.Failure.isovalue) { try grid.mesh(isovalue: .infinity) }
        #expect(throws: MarchingCubes.Failure.triangleLimit) { try grid.mesh(isovalue: 0, maximumTriangles: 0) }
    }

    @Test("Metaball colors sum their r187 palette weights and interpolate between grid samples")
    func palette() throws {
        var grid = try mixedField()
        let isovalue = grid.values[63] // World (+0.5, 0, 0), half the support radius.
        let expected = SIMD4<Float>(0.5, 0, 0.5, 1)
        let exact = try #require(try grid.mesh(isovalue: isovalue, boundary: .requireClosed))
        #expect(exact.vertices.count == 6)
        #expect(exact.vertices.allSatisfy { $0.color == expected })

        let interpolated = try #require(try grid.mesh(isovalue: isovalue / 2, boundary: .requireClosed))
        let midpoint = try #require(interpolated.vertices.first { $0.position == SIMD3(0.75, 0, 0) })
        #expect(midpoint.color == SIMD4(0.25, 0, 0.25, 1))
        let previous = grid.values
        #expect(throws: MarchingCubes.Failure.color) {
            try grid.addMetaball(center: .zero, strength: 1, subtract: 1, color: SIMD3(2, 0, 0))
        }
        #expect(grid.values == previous)
        #expect(try grid.mesh(isovalue: isovalue)?.vertices.allSatisfy { $0.color == expected } == true)

        try grid.replaceValues(previous)
        try grid.addMetaball(center: SIMD3(repeating: 10), strength: 0.5, subtract: 0.5, color: SIMD3(0, 1, 0))
        #expect(try grid.mesh(isovalue: isovalue)?.vertices.allSatisfy { $0.color == nil } == true)
    }

    @Test("A mixed metaball palette reaches the raster vertex-color path",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func rasterPalette() throws {
        let fixture = try RasterLightingFixture(width: 48, height: 48)
        let grid = try mixedField()
        let mesh = try #require(try grid.mesh(isovalue: grid.values[63], boundary: .requireClosed))
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: mesh,
            material: .physicallyBased(try PhysicallyBasedMaterial(shadingModel: .unlit))
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)],
            camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
        let pixels = try fixture.pixels(of: fixture.render(graph).color)
        #expect(pixels[24 * 48 + 24] == SIMD4(0.5, 0, 0.5, 1))
        #expect(pixels[0] == .zero)
    }

    @Test("The indexed sphere rasterizes with its outward normals and an open background",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func rasterSphere() throws {
        let fixture = try RasterLightingFixture(width: 64, height: 64)
        let mesh = try #require(try sphereField().mesh(isovalue: 0, boundary: .requireClosed))
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: mesh,
            material: .classic(try ClassicMaterial(model: .normal))
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)],
            camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
        let pixels = try fixture.pixels(of: fixture.render(graph).color)
        let center = pixels[32 * 64 + 32]
        #expect(center.w == 1 && center.z > 0.98 && abs(center.x - 0.5) < 0.04 && abs(center.y - 0.5) < 0.04)
        #expect(pixels[32 * 64 + 26].x < 0.4 && pixels[32 * 64 + 38].x > 0.6)
        #expect(pixels[0] == .zero)
    }

    private func sphereField() throws -> MarchingCubes {
        let bounds = try AxisAlignedBox(minimum: SIMD3(-1, -1.3, -0.9), maximum: SIMD3(1, 1.3, 0.9))
        return try MarchingCubes(resolution: SIMD3(21, 19, 17), bounds: bounds) {
            0.64 * 0.64 - simd_length_squared($0)
        }
    }

    private func unitBounds() throws -> AxisAlignedBox {
        try AxisAlignedBox(minimum: SIMD3(repeating: -1), maximum: SIMD3(repeating: 1))
    }

    private func mixedField() throws -> MarchingCubes {
        var grid = try MarchingCubes(resolution: SIMD3(repeating: 5), bounds: unitBounds())
        try grid.addMetaball(center: .zero, strength: 0.5, subtract: 0.5, color: SIMD3(1, 0, 0))
        try grid.addMetaball(center: .zero, strength: 0.5, subtract: 0.5, color: SIMD3(0, 0, 1))
        return grid
    }

    private struct Edge: Hashable {
        let low: UInt32
        let high: UInt32
        init(_ a: UInt32, _ b: UInt32) {
            low = min(a, b)
            high = max(a, b)
        }
    }

    private func edgeCounts(_ mesh: MeshGeometry) -> [Edge: Int] {
        var edges: [Edge: Int] = [:]
        for start in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = mesh.indices[start], b = mesh.indices[start + 1], c = mesh.indices[start + 2]
            for edge in [Edge(a, b), Edge(b, c), Edge(c, a)] { edges[edge, default: 0] += 1 }
        }
        return edges
    }
}
