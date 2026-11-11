import Frostfield
import Metal
import Testing
import simd

/// A torus around Z with major radius 1 and minor radius 0.35. U runs around Z and V
/// around the tube; both wrap, so the mesh has two UV seams.
private func testTorus(_ u: Int, _ v: Int) throws -> MeshGeometry {
    try MeshGeometry.parametricSurface(uSegments: u, vSegments: v, wrapU: true, wrapV: true) { uv in
        let a = 2 * Float.pi * uv.x, b = 2 * Float.pi * uv.y
        let ring = 1 + 0.35 * cos(b)
        return SIMD3(ring * cos(a), ring * sin(a), 0.35 * sin(b))
    }
}

/// Distance from a point to the ideal torus of `testTorus`.
private func torusDistance(_ p: SIMD3<Float>) -> Float {
    abs((pow(simd_length(SIMD2(p.x, p.y)) - 1, 2) + p.z * p.z).squareRoot() - 0.35)
}

/// The closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5).
private func closestPointOnTriangle(_ p: SIMD3<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> SIMD3<Float> {
    let ab = b - a, ac = c - a, ap = p - a
    let d1 = simd_dot(ab, ap), d2 = simd_dot(ac, ap)
    if d1 <= 0 && d2 <= 0 { return a }
    let bp = p - b
    let d3 = simd_dot(ab, bp), d4 = simd_dot(ac, bp)
    if d3 >= 0 && d4 <= d3 { return b }
    let vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0 { return a + ab * (d1 / (d1 - d3)) }
    let cp = p - c
    let d5 = simd_dot(ab, cp), d6 = simd_dot(ac, cp)
    if d6 >= 0 && d5 <= d6 { return c }
    let vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0 { return a + ac * (d2 / (d2 - d6)) }
    let va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 { return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))) }
    let denominator = 1 / (va + vb + vc)
    return a + ab * (vb * denominator) + ac * (vc * denominator)
}

/// The largest distance from the vertices and triangle centroids of `from` to the surface of `to`.
private func oneSidedHausdorff(from: MeshGeometry, to: MeshGeometry) -> Float {
    let triangles = (0..<(to.indices.count / 3)).map { t in
        (to.vertices[Int(to.indices[3 * t])].position, to.vertices[Int(to.indices[3 * t + 1])].position,
         to.vertices[Int(to.indices[3 * t + 2])].position)
    }
    let spheres = triangles.map { t -> (SIMD3<Float>, Float) in
        let center = (t.0 + t.1 + t.2) / 3
        return (center, max(simd_distance(center, t.0), simd_distance(center, t.1), simd_distance(center, t.2)))
    }
    var points = from.vertices.map(\.position)
    for t in 0..<(from.indices.count / 3) {
        points.append((from.vertices[Int(from.indices[3 * t])].position + from.vertices[Int(from.indices[3 * t + 1])].position
                       + from.vertices[Int(from.indices[3 * t + 2])].position) / 3)
    }
    var worst: Float = 0
    for p in points {
        var best = Float.infinity
        for (index, sphere) in spheres.enumerated() where simd_distance(p, sphere.0) - sphere.1 < best {
            let t = triangles[index]
            best = min(best, simd_distance(p, closestPointOnTriangle(p, t.0, t.1, t.2)))
        }
        worst = max(worst, best)
    }
    return worst
}

/// No triangle spans a torus UV seam, flips its UV orientation or faces against its normals,
/// and both seams stay closed with wedges on each side.
private func expectSeamsIntact(_ mesh: MeshGeometry, checkingNormals: Bool = true, sourceLocation: SourceLocation = #_sourceLocation) {
    for t in 0..<(mesh.indices.count / 3) {
        let (a, b, c) = triangleCorners(mesh, t)
        let e1 = b.textureCoordinate - a.textureCoordinate, e2 = c.textureCoordinate - a.textureCoordinate
        #expect(e1.x * e2.y - e1.y * e2.x > 0, "triangle \(t)", sourceLocation: sourceLocation)
        #expect(max(simd_length(e1), simd_length(e2)) < 0.5, "triangle \(t)", sourceLocation: sourceLocation)
        if checkingNormals {
            let face = simd_cross(b.position - a.position, c.position - a.position)
            #expect(simd_dot(face, a.normal + b.normal + c.normal) > 0, "triangle \(t)", sourceLocation: sourceLocation)
        }
    }
    var wedges: [SIMD3<Float>: [SIMD2<Float>]] = [:]
    for vertex in mesh.vertices { wedges[vertex.position, default: []].append(vertex.textureCoordinate) }
    for (_, coordinates) in wedges {
        for uv in coordinates {
            if uv.x == 0 { #expect(coordinates.contains { $0.x == 1 && $0.y == uv.y }, sourceLocation: sourceLocation) }
            if uv.y == 0 { #expect(coordinates.contains { $0.y == 1 && $0.x == uv.x }, sourceLocation: sourceLocation) }
        }
    }
    #expect(wedges.values.contains { $0.count == 4 }, sourceLocation: sourceLocation)
    #expect(positionEdgeCounts(mesh).values.allSatisfy { $0 == 2 }, sourceLocation: sourceLocation)
}

private func triangleCorners(_ mesh: MeshGeometry, _ triangle: Int) -> (MeshVertex, MeshVertex, MeshVertex) {
    let start = triangle * 3
    return (
        mesh.vertices[Int(mesh.indices[start])],
        mesh.vertices[Int(mesh.indices[start + 1])],
        mesh.vertices[Int(mesh.indices[start + 2])]
    )
}

private func positionEdgeCounts(_ mesh: MeshGeometry) -> [Set<SIMD3<Float>>: Int] {
    var edges: [Set<SIMD3<Float>>: Int] = [:]
    for triangle in 0..<(mesh.indices.count / 3) {
        let (a, b, c) = triangleCorners(mesh, triangle)
        for edge in [Set([a.position, b.position]), Set([b.position, c.position]), Set([c.position, a.position])] {
            edges[edge, default: 0] += 1
        }
    }
    return edges
}

struct MeshSimplificationTests {
    @Test
    func `A flat grid with affine UVs collapses to two triangles at zero error`() throws {
        let grid = try MeshGeometry.plane(width: 2, height: 2, columns: 16, rows: 16)
        let result = try grid.simplified(targetTriangleCount: 2)
        #expect(result.geometry.indices.count == 6)
        #expect(result.error < 1e-6)
        #expect(Set(result.geometry.vertices.map(\.position)) == [SIMD3(-1, 1, 0), SIMD3(1, 1, 0), SIMD3(-1, -1, 0), SIMD3(1, -1, 0)])
        for (index, vertex) in result.geometry.vertices.enumerated() {
            let source = grid.vertices[Int(result.sourceVertexIndices[index])]
            #expect(vertex.position == source.position && vertex.textureCoordinate == source.textureCoordinate)
        }
        // Every seam on a cube is a hard edge with three charts at each corner, so nothing collapses.
        let cube = try MeshGeometry.cube().simplified(targetTriangleCount: 0)
        #expect(cube.geometry.indices.count == 36)
        #expect(cube.error == 0)
    }

    @Test(arguments: [MeshGeometry.SimplificationOptions.VertexPlacement.optimal, .endpoints])
    func `A torus at 10 percent stays within a Hausdorff bound and keeps both UV seams`(
        placement: MeshGeometry.SimplificationOptions.VertexPlacement
    ) throws {
        let torus = try testTorus(96, 48)
        let target = 922
        let result = try torus.simplified(targetTriangleCount: target, options: .init(vertexPlacement: placement))
        let simplified = result.geometry
        let triangles = simplified.indices.count / 3
        #expect(triangles <= target && triangles >= target - 4, "\(triangles)")

        // Two-sided Hausdorff distance to the input mesh, sampled at vertices and centroids.
        // Measured: 0.0127 with optimal placement and 0.0224 with endpoints (0.47% and 0.83% of
        // the 2.7 extent). A uniform 30 × 15 torus grid with 900 triangles measures 0.0143.
        let forward = oneSidedHausdorff(from: simplified, to: torus)
        let backward = oneSidedHausdorff(from: torus, to: simplified)
        let bound: Float = placement == .optimal ? 0.0143 : 0.025
        #expect(max(forward, backward) < bound, "forward \(forward), backward \(backward), error \(result.error)")
        #expect(result.error > 0 && result.error < bound)
        if placement == .endpoints {
            #expect(simplified.vertices.allSatisfy { torusDistance($0.position) < 1e-5 })
            for (index, vertex) in simplified.vertices.enumerated() {
                let source = torus.vertices[Int(result.sourceVertexIndices[index])]
                #expect(vertex.position == source.position && vertex.textureCoordinate == source.textureCoordinate)
            }
        }

        expectSeamsIntact(simplified)
        // Seams hold by topology, not only by cost: without texture or normal error at all,
        // a fifth of the triangles still keeps every seam closed and every chart unfolded.
        let blind = try torus.simplified(
            targetTriangleCount: 2 * target,
            options: .init(textureCoordinateWeight: 0, normalWeight: 0, vertexPlacement: placement)
        )
        expectSeamsIntact(blind.geometry, checkingNormals: false)

        let again = try torus.simplified(targetTriangleCount: target, options: .init(vertexPlacement: placement))
        #expect(again.geometry.indices == simplified.indices && again.sourceVertexIndices == result.sourceVertexIndices)
        #expect(again.geometry.vertices.map(\.position) == simplified.vertices.map(\.position))
    }

    @Test
    func `Open boundaries only slide along themselves or stay locked`() throws {
        let grid = try MeshGeometry.plane(width: 2, height: 2, columns: 32, rows: 32)
        let bumpy = try MeshGeometry(vertices: grid.vertices.map { vertex in
            let p = vertex.position
            let z = 0.1 * sin(3 * p.x) * cos(2 * p.y)
            let normal = simd_normalize(SIMD3(-0.3 * cos(3 * p.x) * cos(2 * p.y), 0.2 * sin(3 * p.x) * sin(2 * p.y), 1))
            return MeshVertex(position: SIMD3(p.x, p.y, z), normal: normal, textureCoordinate: vertex.textureCoordinate)
        }, indices: grid.indices)
        func boundary(_ mesh: MeshGeometry) -> [(SIMD3<Float>, SIMD3<Float>)] {
            var counts: [SIMD2<UInt32>: Int] = [:]
            for start in stride(from: 0, to: mesh.indices.count, by: 3) {
                for k in 0..<3 {
                    let a = mesh.indices[start + k], b = mesh.indices[start + (k + 1) % 3]
                    counts[SIMD2(min(a, b), max(a, b)), default: 0] += 1
                }
            }
            return counts.filter { $0.value == 1 }.map { (mesh.vertices[Int($0.key.x)].position, mesh.vertices[Int($0.key.y)].position) }
        }
        let free = try bumpy.simplified(ratio: 0.2).geometry
        // Down to a handful of triangles, where cost alone no longer protects the outline.
        let tiny = try bumpy.simplified(targetTriangleCount: 8).geometry
        #expect(tiny.indices.count <= 8 * 3)
        for mesh in [free, tiny] {
            let edges = boundary(mesh)
            #expect(!edges.isEmpty)
            for (p, q) in edges {
                let onSide = (p.x == q.x && abs(p.x) == 1) || (p.y == q.y && abs(p.y) == 1)
                #expect(onSide, "\(p) – \(q)")
            }
        }
        let corners = Set(free.vertices.map { SIMD2($0.position.x, $0.position.y) })
        for corner in [SIMD2<Float>(-1, -1), SIMD2(-1, 1), SIMD2(1, -1), SIMD2(1, 1)] { #expect(corners.contains(corner)) }

        let locked = try bumpy.simplified(ratio: 0.2, options: .init(locksBoundary: true)).geometry
        #expect(Set(boundary(locked).flatMap { [$0.0, $0.1] }).count == 128)
        #expect(Set(boundary(free).flatMap { [$0.0, $0.1] }).count < 128)
    }

    @Test
    func `Collapses keep a torus a manifold torus and never fold a flat sheet`() throws {
        // Without seams or attribute error, only the link condition stops collapses that pinch
        // the surface. The result must stay a closed manifold of genus one: V − E + F = 0.
        let torus = try testTorus(12, 8).welded(tolerance: .position(1e-5))
        for placement in [MeshGeometry.SimplificationOptions.VertexPlacement.optimal, .endpoints] {
            let minimal = try torus.simplified(
                targetTriangleCount: 0, options: .init(textureCoordinateWeight: 0, normalWeight: 0, vertexPlacement: placement)
            ).geometry
            let edges = positionEdgeCounts(minimal)
            #expect(edges.values.allSatisfy { $0 == 2 }, "\(placement)")
            let vertexCount = Set(minimal.indices).count
            #expect(vertexCount - edges.count + minimal.indices.count / 3 == 0, "\(placement)")
            #expect(minimal.indices.count / 3 < 48)
        }

        // A flat L-shaped sheet with constant UVs. Moving its reflex corner, or a vertex next to
        // it, can fold triangles over without any geometric, texture or normal error, so only
        // the fold check keeps every triangle facing +Z as the sheet shrinks to its outline.
        let grid = try MeshGeometry.plane(width: 2, height: 2, columns: 24, rows: 24)
        var kept: [UInt32] = []
        for start in stride(from: 0, to: grid.indices.count, by: 3) {
            let corners = (0..<3).map { grid.vertices[Int(grid.indices[start + $0])].position }
            let centroid = (corners[0] + corners[1] + corners[2]) / 3
            if !(centroid.x > 0 && centroid.y > 0) { kept += grid.indices[start..<(start + 3)] }
        }
        let sheet = try MeshGeometry(vertices: grid.vertices.map { MeshVertex(position: $0.position, normal: SIMD3(0, 0, 1)) }, indices: kept)
        for placement in [MeshGeometry.SimplificationOptions.VertexPlacement.optimal, .endpoints] {
            let reduced = try sheet.simplified(targetTriangleCount: 2, options: .init(vertexPlacement: placement)).geometry
            #expect(reduced.indices.count / 3 < 20)
            for t in 0..<(reduced.indices.count / 3) {
                let (a, b, c) = triangleCorners(reduced, t)
                #expect(simd_cross(b.position - a.position, c.position - a.position).z > 0, "\(placement) triangle \(t)")
            }
        }
    }

    @Test
    func `Texture coordinate error keeps a stretched UV map from collapsing`() throws {
        // A flat grid whose U grows quadratically: geometry alone could reduce it to two triangles.
        let grid = try MeshGeometry.plane(width: 2, height: 2, columns: 16, rows: 16)
        let stretched = try MeshGeometry(vertices: grid.vertices.map { vertex in
            let u = (vertex.position.x + 1) / 2
            return MeshVertex(position: vertex.position, normal: vertex.normal, textureCoordinate: SIMD2(u * u, vertex.textureCoordinate.y))
        }, indices: grid.indices)
        func uvError(_ mesh: MeshGeometry) -> Float {
            var worst: Float = 0
            for t in 0..<(mesh.indices.count / 3) {
                let (a, b, c) = triangleCorners(mesh, t)
                for weights in [SIMD3<Float>(1, 1, 1) / 3, SIMD3(0.5, 0.5, 0), SIMD3(0, 0.5, 0.5), SIMD3(0.5, 0, 0.5)] {
                    let p = a.position * weights.x + b.position * weights.y + c.position * weights.z
                    let uv = a.textureCoordinate * weights.x + b.textureCoordinate * weights.y + c.textureCoordinate * weights.z
                    let u = (p.x + 1) / 2
                    worst = max(worst, abs(uv.x - u * u))
                }
            }
            return worst
        }
        let ignoring = try stretched.simplified(targetTriangleCount: 2, options: .init(maximumError: 0.002, textureCoordinateWeight: 0))
        let aware = try stretched.simplified(targetTriangleCount: 2, options: .init(maximumError: 0.002))
        #expect(ignoring.geometry.indices.count == 6)
        #expect(aware.geometry.indices.count > 6)
        #expect(aware.error <= 0.002)
        #expect(uvError(aware.geometry) < 0.01, "\(uvError(aware.geometry)) vs \(uvError(ignoring.geometry))")
        #expect(uvError(ignoring.geometry) > 0.1)
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidSimplification) {
            try stretched.simplified(targetTriangleCount: -1)
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidSimplification) {
            try stretched.simplified(ratio: 1.5)
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidSimplification) {
            try stretched.simplified(ratio: 0.5, options: .init(normalWeight: -1))
        }
    }

    @Test
    func `A colored mesh keeps its material boundaries and group assignments`() throws {
        let grid = try MeshGeometry.plane(width: 2, height: 2, columns: 16, rows: 16)
        let colors = grid.vertices.map { vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.textureCoordinate * 0.5, tangent: vertex.tangent,
                color: SIMD4(vertex.textureCoordinate.x, vertex.textureCoordinate.y, 0.5, 0.75)
            )
        }
        let indicesPerRow = 16 * 6
        let indexCount = grid.indices.count
        let mesh = try MeshGeometry(vertices: colors, indices: grid.indices + [0, 0, 0]).withGroups([
            .init(start: 0, count: 8 * indicesPerRow, materialIndex: 4),
            .init(start: 9 * indicesPerRow, count: 7 * indicesPerRow, materialIndex: 2),
            .init(start: indexCount, count: 3, materialIndex: 7),
        ])
        let result = try mesh.simplified(ratio: 0.22, options: .init(vertexPlacement: .endpoints))
        let reduced = result.geometry
        #expect(reduced.indices.count < indexCount / 2)
        #expect(reduced.groups.map(\.materialIndex) == [4, 2, 7])
        #expect(reduced.groups[0].range.upperBound < reduced.groups[1].start)
        #expect(reduced.groups[2].count == 3)
        #expect(reduced.groups[2].range.upperBound == reduced.indices.count)
        let first = try reduced.subset(reduced.groups[0].range)
        let second = try reduced.subset(reduced.groups[1].range)
        #expect(first.vertices.allSatisfy { $0.position.y >= 0 })
        #expect(second.vertices.allSatisfy { $0.position.y <= -0.125 })
        #expect(Set(first.vertices.filter { $0.position.y == 0 }.map(\.position)).count == 17)
        #expect(Set(second.vertices.filter { $0.position.y == -0.125 }.map(\.position)).count == 17)
        #expect(reduced.vertices.count == result.sourceVertexIndices.count)
        for (index, vertex) in reduced.vertices.enumerated() {
            let source = mesh.vertices[result.sourceVertexIndices[index]]
            #expect(vertex.position == source.position && vertex.normal == source.normal)
            #expect(vertex.textureCoordinate == source.textureCoordinate && vertex.secondaryTextureCoordinate == source.secondaryTextureCoordinate)
            #expect(vertex.tangent == source.tangent && vertex.color == source.color)
        }
    }

    @Test
    func `Optimal placement carries linear colors and UV1 to moved vertices`() throws {
        let torus = try testTorus(32, 20)
        // These channels are affine functions of UV0 on every face, so the same least-squares
        // placement preserves the relationship even when the new point moves off the source surface.
        func color(_ uv: SIMD2<Float>) -> SIMD4<Float> {
            SIMD4(0.4 + 0.2 * uv.x, 0.5 + 0.15 * uv.y, 0.25 + 0.4 * uv.x + 0.2 * uv.y, 0.8)
        }
        let mesh = try MeshGeometry(vertices: torus.vertices.map { vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.textureCoordinate * 0.5, tangent: vertex.tangent, color: color(vertex.textureCoordinate)
            )
        }, indices: torus.indices)
        let result = try mesh.simplified(ratio: 0.4)
        #expect(result.geometry.indices.count <= torus.indices.count / 2)
        var moved = 0
        for (index, vertex) in result.geometry.vertices.enumerated() {
            if vertex.position != mesh.vertices[result.sourceVertexIndices[index]].position { moved += 1 }
            let tint = try #require(vertex.color)
            let secondary = try #require(vertex.secondaryTextureCoordinate)
            #expect(simd_length(tint - color(vertex.textureCoordinate)) < 1e-5)
            #expect(simd_length(secondary - vertex.textureCoordinate * 0.5) < 1e-5)
        }
        #expect(moved > 10)
    }

    @Test
    func `A no-op retains source order and a tetrahedron cannot collapse into coincident faces`() throws {
        let tetrahedron = try MeshGeometry.tetrahedron().welded(tolerance: .position(1e-5))
        let noOp = try tetrahedron.simplified(ratio: 1)
        #expect(noOp.geometry.indices == tetrahedron.indices)
        #expect(noOp.sourceVertexIndices == Array(tetrahedron.vertices.indices))
        let minimum = try tetrahedron.simplified(
            ratio: 0, options: .init(textureCoordinateWeight: 0, normalWeight: 0)
        )
        #expect(minimum.geometry.indices == tetrahedron.indices)
        #expect(minimum.sourceVertexIndices == Array(tetrahedron.vertices.indices))
    }

    @Test
    func `Simplification cannot delete separate triangle components`() throws {
        let vertices = [SIMD3<Float>(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0)]
        let positions = vertices + vertices.map { $0 + SIMD3(2, 0, 0) }
        let mesh = try MeshGeometry(
            vertices: positions.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) },
            indices: [0, 1, 2, 3, 4, 5]
        )
        let simplified = try mesh.simplified(ratio: 0)
        #expect(simplified.geometry.indices == mesh.indices)
        #expect(simplified.sourceVertexIndices == Array(mesh.vertices.indices))
    }

}


struct MeshLevelOfDetailChainTests {
    @Test
    func `Generated levels use the transformed near-surface depth and increasing distances`() throws {
        let centered = try testTorus(64, 32)
        let offset = SIMD3<Float>(3.5, -2, 1)
        let mesh = try MeshGeometry(vertices: centered.vertices.map { vertex in
            MeshVertex(position: vertex.position + offset, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate)
        }, indices: centered.indices)
        let budget = MeshGeometry.ScreenSpaceErrorBudget(pixels: 2, viewportHeight: 1_000, verticalFieldOfView: .pi / 2)
        let chain = try mesh.levelOfDetailChain(levelCount: 5, budget: budget)
        #expect(chain.count == 5)
        #expect(chain[0].distance == 0 && chain[0].geometry.indices == mesh.indices)
        #expect(chain[0].sourceVertexIndices == Array(mesh.vertices.indices))

        let transform = SceneTransform(
            translation: SIMD3(50, -8, 25),
            rotation: simd_quatf(angle: 0.73, axis: simd_normalize(SIMD3<Float>(2, 3, 1))),
            scale: SIMD3(repeating: 2.5)
        )
        let worldChain = try mesh.levelOfDetailChain(levelCount: 5, budget: budget, worldTransform: transform.matrix)
        #expect(worldChain.count == chain.count)
        let originalRadius = mesh.vertices.map { simd_length(SIMD3<Double>($0.position)) }.max() ?? 0
        for index in chain.indices.dropFirst() {
            let level = chain[index]
            let goal = 4096 >> index
            #expect(level.geometry.indices.count / 3 <= goal && level.geometry.indices.count / 3 >= goal - 4)
            #expect(level.error >= chain[index - 1].error)
            #expect(level.distance > chain[index - 1].distance)
            let levelRadius = level.geometry.vertices.map { simd_length(SIMD3<Double>($0.position)) }.max() ?? 0
            let nearestDepth = Double(level.distance) - max(originalRadius, levelRadius)
            #expect(nearestDepth > 0)
            let projectedError = Double(level.error) * 500 / nearestDepth
            #expect(abs(projectedError - 2) < 1e-4)
            #expect(abs(worldChain[index].distance - level.distance * 2.5) < 2e-5)
            #expect(worldChain[index].geometry.indices == level.geometry.indices)
        }
    }

    @Test
    func `Sheared geometry uses a stretch bound and rejects distances outside Float range`() throws {
        let mesh = try testTorus(40, 20)
        let budget = MeshGeometry.ScreenSpaceErrorBudget(pixels: 2, viewportHeight: 1_000, verticalFieldOfView: .pi / 2)
        let shear = simd_float4x4(columns: (
            SIMD4(2, 0, 0, 0), SIMD4(1, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(0, 0, 0, 1)
        ))
        let chain = try mesh.levelOfDetailChain(levelCount: 2, budget: budget, worldTransform: shear)
        #expect(chain.count == 2)
        let coarse = try #require(chain.last)
        let radius = (mesh.vertices + coarse.geometry.vertices).map { vertex -> Double in
            let p = SIMD3<Double>(vertex.position)
            return simd_length(SIMD3(2 * p.x + p.y, p.y, p.z))
        }.max() ?? 0
        // AᵀA has rows (4, 2, 0), (2, 2, 0), and (0, 0, 1).
        let estimatedDistance = radius + Double(coarse.error) * 250 * sqrt(6)
        #expect(abs(Double(coarse.distance) - estimatedDistance) < 1e-5)

        #expect(throws: MeshGeometry.GeometryProcessingError.unrepresentableLevelOfDetailDistance) {
            try mesh.levelOfDetailChain(levelCount: 2, budget: .init(pixels: .leastNonzeroMagnitude))
        }
    }

    @Test
    func `The full mesh survives equal-cost levels and rounding cannot hide a terminal reduction`() throws {
        let grid = try MeshGeometry.plane(columns: 16, rows: 16)
        let chain = try grid.levelOfDetailChain(levelCount: 16, options: .init(maximumError: 1e-6))
        #expect(chain.count == 2)
        #expect(chain.first?.distance == 0)
        #expect(chain.first?.geometry.indices == grid.indices)
        #expect(chain.last?.geometry.indices.count == 6)
        #expect(chain.last?.error == 0)
        let coarse = try #require(chain.last)
        #expect(coarse.distance > 0.7071)

        let small = try MeshGeometry.plane(columns: 2, rows: 2)
        let almostOne = try small.levelOfDetailChain(levelCount: 2, triangleRatio: 0.99)
        #expect(almostOne.count == 2)
        #expect(almostOne.last?.geometry.indices.count ?? .max < small.indices.count)
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidLevelOfDetail) {
            try grid.levelOfDetailChain(levelCount: 17)
        }
        #expect(throws: MeshGeometry.GeometryProcessingError.invalidLevelOfDetail) {
            try grid.levelOfDetailChain(levelCount: 3, worldTransform: SceneTransform(scale: SIMD3(1, 0, 1)).matrix)
        }
    }

    @Test("Generated renderers follow scene selection under a scaled parent",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func sceneSelection() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try MeshGeometry.plane(columns: 12, rows: 12)
        let parent = SceneTransform(
            translation: SIMD3(12, 5, -4),
            rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0)),
            scale: SIMD3(3, 1, 0.5)
        )
        let local = SceneTransform(
            translation: SIMD3(0, 0.5, 2), rotation: simd_quatf(angle: 0.35, axis: SIMD3(0, 0, 1))
        )
        let world = parent.matrix * local.matrix
        let chain = try geometry.levelOfDetailChain(
            levelCount: 8, worldTransform: world, options: .init(maximumError: 1e-6)
        )
        #expect(chain.count == 2)
        let coarse = try #require(chain.last)
        let maxRadius = geometry.vertices.map { vertex -> Float in
            let v = world * SIMD4(vertex.position, 0)
            return simd_length(SIMD3(v.x, v.y, v.z))
        }.max() ?? 0
        #expect(abs(coarse.distance - maxRadius) < 1e-5)
        let prepared = try LevelOfDetail(chain: chain) { mesh in
            try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: mesh)
        }
        let graph = try SceneGraph(roots: [
            SceneNode(transform: parent, children: [SceneNode(transform: local, levelsOfDetail: prepared)]),
        ], camera: PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100))
        let origin = SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
        let near = try #require(graph.resolve(cameraPosition: origin + SIMD3(0, 0, maxRadius * 0.9)).last)
        let far = try #require(graph.resolve(cameraPosition: origin + SIMD3(0, 0, coarse.distance + 0.01)).last)
        #expect(near.levelOfDetail == 0 && near.mesh === prepared.levels[0].mesh)
        #expect(far.levelOfDetail == 1 && far.mesh === prepared.levels[1].mesh)
    }
}
