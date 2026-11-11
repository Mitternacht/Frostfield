import Foundation
import Frostfield
import Metal
import simd
import Testing

struct PathExtrusionTests {
    @Test("An arc-length Bézier extrusion matches pinned r187 positions and cap and wall groups")
    func bezierReference() throws {
        let path = try Curve3D.cubicBezier(
            from: SIMD3(0.2, -0.4, -0.3), control1: SIMD3(0.9, 0.8, 1.1),
            control2: SIMD3(1.7, -0.4, 2), to: SIMD3(2.3, 0.9, 3.5)
        )
        let shape = try PlanarShape(
            outer: [SIMD2(-0.26, -0.18), SIMD2(0.31, -0.18), SIMD2(0.31, 0.25), SIMD2(-0.26, 0.25)],
            holes: [[SIMD2(-0.09, -0.07), SIMD2(-0.09, 0.09), SIMD2(0.12, 0.09), SIMD2(0.12, -0.07)]]
        )
        let mesh = try MeshGeometry.extrude([shape], along: path, steps: 5)
        let points = Set(mesh.vertices.map(\.position))
        #expect(points.count == 48)
        #expect(mesh.groups == [
            .init(start: 0, count: 48, materialIndex: 0),
            .init(start: 48, count: 240, materialIndex: 1),
        ])
        // CubicBezierCurve3.getSpacedPoints, computeFrenetFrames, and ExtrudeGeometry
        // from r187. Its endpoint tangents use a finite difference.
        let reference: [SIMD3<Float>] = [
            SIMD3(0.44307006, -0.32337840, -0.48720598), SIMD3(0.93759737, 0.15444577, 1.26967293),
            SIMD3(1.32902019, 0.50128335, 2.08498632), SIMD3(1.99717384, 0.77714131, 3.72760066),
        ]
        for expected in reference {
            #expect(points.contains { simd_distance($0, expected) < 0.00012 })
        }
        #expect(ShapeCheck.isClosed(mesh))
        #expect(mesh.signedVolume > 0)
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
        let cap = try mesh.subset(mesh.groups[0].range)
        let start = try path.tangent(at: 0)
        let end = try path.tangent(at: 1)
        #expect(cap.vertices.allSatisfy { max(simd_dot($0.normal, -start), simd_dot($0.normal, end)) > 0.99999 })
        #expect(cap.vertices.allSatisfy { (0...1).contains($0.textureCoordinate.x) && (0...1).contains($0.textureCoordinate.y) })

        let world = try MeshGeometry.extrude([shape], along: path, steps: 5, textureMapping: .world(unitsPerTile: 0.5))
        let wall = try world.subset(world.groups[1].range)
        #expect(abs((wall.vertices.map(\.textureCoordinate.y).max() ?? 0) - 9.279221) < 0.00005)
        #expect(abs((wall.vertices.map(\.textureCoordinate.x).max() ?? 0) - 4) < 0.00005)
    }

    @Test("Straight axial intervals preserve cap bounds and the existing bevel profile")
    func straightSteps() throws {
        let shape = try PlanarShape(outer: square(0.5))
        let standard = try MeshGeometry.extrude([shape], ShapeExtrusion(depth: 2))
        let split = try MeshGeometry.extrude([shape], ShapeExtrusion(depth: 2, steps: 4))
        #expect(standard.indices.count == 36 && split.indices.count == 108)
        #expect(standard.groups.isEmpty && split.groups.isEmpty)
        #expect(Set(split.vertices.map(\.position.z)) == Set([-1, -0.5, 0, 0.5, 1]))
        #expect(abs(standard.signedVolume - split.signedVolume) < 1e-9)
        #expect(ShapeCheck.isClosed(split))

        let bevel = ShapeExtrusion(depth: 2, bevelSize: 0.125, bevelDepth: 0.25, bevelSegments: 3)
        let ordinary = try MeshGeometry.extrude([shape], bevel)
        var more = bevel
        more.steps = 4
        let refined = try MeshGeometry.extrude([shape], more)
        #expect(refined.indices.count == ordinary.indices.count + 3 * 4 * 6)
        #expect(Set(refined.vertices.map(\.position.z)).subtracting(Set(ordinary.vertices.map(\.position.z))) == Set([-0.375, 0, 0.375]))
        #expect(abs(refined.signedVolume - ordinary.signedVolume) < 1e-8)
        #expect(ShapeCheck.isClosed(refined))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(refined))
    }

    @Test("A closed spatial sweep shares exact endpoint positions and normals across its UV seam")
    func closedLoop() throws {
        let path = Curve3D { fraction in
            let angle = 2 * Double.pi * fraction
            return SIMD3(2 * cos(angle), 2 * sin(angle), 0.22 * sin(2 * angle))
        }
        let steps = 32
        let shape = try PlanarShape(outer: square(0.16))
        let mesh = try MeshGeometry.extrude([shape], along: path, steps: steps, closed: true)
        #expect(mesh.groups == [.init(start: 0, count: 4 * 6 * steps, materialIndex: 1)])
        #expect(ShapeCheck.isClosed(mesh))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
        let atStart = mesh.vertices.filter { $0.textureCoordinate.y == 0 }
        let atEnd = mesh.vertices.filter { $0.textureCoordinate.y == 1 }
        #expect(!atStart.isEmpty && atStart.count == atEnd.count)
        #expect(zip(atStart, atEnd).allSatisfy { a, b in
            a.position == b.position && a.normal == b.normal && a.tangent == b.tangent
                && a.textureCoordinate.x == b.textureCoordinate.x
        })
    }

    @Test("Path extrusion rejects invalid steps, stationary samples, open loops and folded surfaces")
    func invalidInput() throws {
        let shape = try PlanarShape(outer: square(0.5))
        let line = Curve3D { SIMD3(0, 0, $0 * 2) }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidSteps) { try MeshGeometry.extrude([shape], along: line, steps: 0) }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidSteps) { try MeshGeometry.extrude([shape], .init(depth: 1, steps: 4_097)) }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidClosedPath) {
            try MeshGeometry.extrude([shape], along: line, steps: 6, closed: true)
        }
        #expect(throws: Curve3D.Failure.resolution) {
            try MeshGeometry.extrude([shape], along: line, arcLengthSubdivisions: 0)
        }
        let stopped = Curve3D { _ in .zero }
        #expect(throws: Curve3D.Failure.undefinedTangent) { try MeshGeometry.extrude([shape], along: stopped) }
        let reversed = Curve3D { t in
            let displacement = 2 * t - 1
            return SIMD3(displacement * displacement, 0, 0)
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidPathFrame(1)) {
            try MeshGeometry.extrude([shape], along: reversed)
        }
        let tight = Curve3D { t in
            let angle = .pi / 2 * t
            return SIMD3(0.2 * cos(angle), 0.2 * sin(angle), 0)
        }
        #expect(throws: MeshGeometry.ShapeMeshError.foldedPathSurface(0)) {
            try MeshGeometry.extrude([shape], along: tight, steps: 8)
        }
    }

    @Test("Open path extrusion renders cap and wall materials around a visible through-hole",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func groupedRender() throws {
        let gpu = try GeometryRenderFixture()
        let shape = try PlanarShape(outer: square(0.65), holes: [square(0.31)])
        let path = Curve3D { t in SIMD3(0.4, 0, -2 - t) }
        let mesh = try MeshGeometry.extrude([shape], along: path, steps: 4)
        let cap = SIMD3<Float>(1, 0, 0), wall = SIMD3<Float>(0, 1, 0)
        let node = try SceneNode.grouped(
            mesh, materials: [
                .classic(ClassicMaterial(model: .basic, color: cap)),
                .classic(ClassicMaterial(model: .basic, color: wall)),
            ],
            device: gpu.device, colorPixelFormat: .rgba16Float
        )
        let graph = SceneGraph(roots: [node], camera: gpu.camera)
        let rendered = try gpu.render(graph)
        #expect(rendered.statistics.drawnMeshCount == 2)
        #expect(rendered.pixels.contains { $0 == SIMD4(cap, 1) })
        #expect(rendered.pixels.contains { $0 == SIMD4(wall, 1) })
        let center = gpu.column(x: 0.4, depth: 2)
        #expect(gpu.pixel(rendered.pixels, x: center, y: 32) == .zero)
    }

    private func square(_ halfExtent: Float) -> [SIMD2<Float>] {
        let h = halfExtent
        return [SIMD2(-h, -h), SIMD2(h, -h), SIMD2(h, h), SIMD2(-h, h)]
    }
}
