import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PlanarPathTests {
    @Test("A mixed open path matches r187 arc-length samples and endpoint")
    func mixedReference() throws {
        let path = PlanarPath([
            .move(to: .zero),
            .line(to: SIMD2(1, 0)),
            .quadratic(to: SIMD2(3, 0), control: SIMD2(2, 1)),
            .cubic(to: SIMD2(6, 0), control1: SIMD2(4, -1), control2: SIMD2(5, 1)),
            .spline(through: [SIMD2(7, 0.5), SIMD2(8, -0.3), SIMD2(9, 0)]),
            .ellipticalArc(center: SIMD2(10, 1), radii: SIMD2(0.75, 0.35),
                           startAngle: 0.4, sweepAngle: -1.6, rotation: 0.31),
        ])
        let sampled = try path.sampledSubpaths(tolerance: 0.0001)
        try #require(sampled.count == 1)
        let subpath = sampled[0]
        #expect(!subpath.isClosed)
        #expect(abs(subpath.length - 12.942911499829762) < 0.001)
        #expect(subpath.points.contains(SIMD2(7, 0.5)))
        #expect(subpath.points.contains(SIMD2(9, 0)))
        let expected: [SIMD2<Float>] = [
            .zero, SIMD2(1.491240, 0.370581), SIMD2(2.957218, 0.041867), SIMD2(4.428970, -0.035436),
            SIMD2(5.927532, 0.067301), SIMD2(7.318200, 0.303617), SIMD2(8.669451, -0.152387),
            SIMD2(9.964950, 0.800319), SIMD2(10.358328, 0.772241),
        ]
        let samples = try subpath.spacedPoints(segments: 8)
        #expect(samples.count == expected.count)
        #expect(zip(samples, expected).allSatisfy { simd_distance($0, $1) < 0.001 })
        #expect(try subpath.point(atLengthFraction: 1) == subpath.points.last)
    }

    @Test("Open, closed and restarted subpaths keep independent length and value semantics")
    func closure() throws {
        let open = PlanarPath([
            .move(to: .zero), .line(to: .zero), .line(to: SIMD2(3, 0)), .line(to: SIMD2(3, 4)),
        ])
        var closed = open
        closed.commands.append(contentsOf: [.close, .close, .line(to: SIMD2(-1, 0))])
        let openSamples = try open.sampledSubpaths()
        let closedSamples = try closed.sampledSubpaths()
        try #require(openSamples.count == 1 && closedSamples.count == 2)
        #expect(open != closed)
        #expect(openSamples[0].length == 7)
        #expect(openSamples[0].points == [.zero, SIMD2(3, 0), SIMD2(3, 4)])
        #expect(closedSamples[0].isClosed && closedSamples[0].length == 12)
        #expect(closedSamples[0].points == openSamples[0].points)
        #expect(try closedSamples[0].spacedPoints(segments: 4) == [.zero, SIMD2(3, 0), SIMD2(3, 3), SIMD2(1.8, 2.4), .zero])
        #expect(closedSamples[1].points == [.zero, SIMD2(-1, 0)])
        #expect(!closedSamples[1].isClosed && closedSamples[1].length == 1)
        #expect(throws: PlanarPath.Failure.invalidFraction) { try openSamples[0].point(atLengthFraction: .nan) }
        #expect(throws: PlanarPath.Failure.invalidResolution) { try openSamples[0].spacedPoints(segments: 0) }
    }

    @Test("A full elliptic arc honors the chord tolerance and keeps its exact closure")
    func ellipseTolerance() throws {
        let path = PlanarPath([.ellipticalArc(center: .zero, radii: SIMD2(2, 1), startAngle: 0, sweepAngle: 2 * .pi), .close])
        let coarse = try #require(path.sampledSubpaths(tolerance: 0.05).first)
        let fine = try #require(path.sampledSubpaths(tolerance: 0.001).first)
        #expect(coarse.isClosed && fine.isClosed)
        #expect(fine.points.count > coarse.points.count)
        #expect(fine.points.first == SIMD2(2, 0))
        #expect(fine.points.last != fine.points.first)
        #expect(try fine.point(atLengthFraction: 1) == SIMD2(2, 0))
        for index in fine.points.indices {
            let a = SIMD2<Double>(fine.points[index]), b = SIMD2<Double>(fine.points[(index + 1) % fine.points.count])
            let t0 = atan2(a.y, a.x / 2), t1 = atan2(b.y, b.x / 2)
            let middle = t0 + (t1 - t0).remainder(dividingBy: 2 * .pi) / 2
            let ellipseMiddle = SIMD2(2 * cos(middle), sin(middle))
            #expect(simd_distance(ellipseMiddle, (a + b) / 2) <= 0.001001)
        }
    }

    @Test("Compound paths support both fill rules and explicit holes")
    func shapeComposition() throws {
        let outer = PlanarPath([
            .move(to: SIMD2(-1, -1)), .line(to: SIMD2(1, -1)), .line(to: SIMD2(1, 1)), .line(to: SIMD2(-1, 1)),
        ])
        let hole = PlanarPath([
            .ellipticalArc(center: .zero, radii: SIMD2(0.3, 0.25), startAngle: 0, sweepAngle: 2 * .pi,
                           rotation: 0.2, connecting: false),
            .close,
        ])
        let compound = PlanarPath(outer.commands + hole.commands)
        let nonzero = try compound.shapes()
        let evenOdd = try compound.shapes(fillRule: .evenOdd, tolerance: 0.0001)
        let explicit = try PlanarShape(outerPath: outer, holePaths: [hole], tolerance: 0.0001)
        try #require(nonzero.count == 1 && evenOdd.count == 1)
        #expect(nonzero[0].holes.isEmpty && nonzero[0].area == 4)
        #expect(evenOdd[0].holes.count == 1)
        #expect(abs(evenOdd[0].area - 3.7644042061800427) < 0.00025)
        #expect(abs(explicit.area - evenOdd[0].area) < 1e-12)
        let native = try compound.makeCGPath(tolerance: 0.0001)
        #expect(try PlanarShape.shapes(from: native, fillRule: .evenOdd, tolerance: 0.0001) == evenOdd)
        #expect(throws: PlanarPath.Failure.expectedSingleContour(0)) {
            try PlanarShape(outerPath: compound)
        }
    }

    @Test("A curved authored path renders an extruded rim and a clear hole",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func renderExtrusion() throws {
        let path = PlanarPath([
            .move(to: SIMD2(-0.9, -0.65)), .line(to: SIMD2(0.9, -0.65)),
            .quadratic(to: SIMD2(0.9, 0.65), control: SIMD2(1.15, 0)), .line(to: SIMD2(-0.9, 0.65)),
            .cubic(to: SIMD2(-0.9, -0.65), control1: SIMD2(-1.04, 0.45), control2: SIMD2(-1.04, -0.45)),
            .close,
            .ellipticalArc(center: .zero, radii: SIMD2(0.3, 0.25), startAngle: 0, sweepAngle: -2 * .pi,
                           rotation: 0.2, connecting: false),
            .close,
        ])
        let shapes = try path.shapes(tolerance: 0.002)
        try #require(shapes.count == 1 && shapes[0].holes.count == 1)
        let flat = try MeshGeometry.shape(shapes)
        #expect(abs(ShapeCheck.surfaceArea(flat) - shapes[0].area) < 1e-8)
        let depth: Float = 0.3
        let mesh = try MeshGeometry.extrude(shapes, .init(depth: depth))
        #expect(ShapeCheck.isClosed(mesh))
        #expect(abs(mesh.signedVolume - shapes[0].area * Double(depth)) < 1e-6)
        let gpu = try GeometryRenderFixture(side: 96)
        let color = SIMD3<Float>(0.25, 0.75, 0.5)
        let renderer = try gpu.unlit(mesh, material: .classic(ClassicMaterial(model: .basic, color: color)))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2 - depth / 2)), mesh: renderer)
        let rendered = try gpu.render(SceneGraph(roots: [node], camera: gpu.camera))
        #expect(rendered.statistics.drawnMeshCount == 1)
        let middle = gpu.side / 2
        #expect(gpu.pixel(rendered.pixels, x: middle, y: middle) == .zero)
        #expect(gpu.pixel(rendered.pixels, x: gpu.column(x: 0.6, depth: 2), y: middle) == SIMD4(color, 1))
        #expect(gpu.pixel(rendered.pixels, x: gpu.column(x: 1.3, depth: 2), y: middle) == .zero)
    }

    @Test("Typed paths reject malformed commands and oversized tessellation")
    func validation() {
        #expect(throws: PlanarPath.Failure.missingMove(0)) {
            try PlanarPath([.line(to: SIMD2(1, 0))]).sampledSubpaths()
        }
        #expect(throws: PlanarPath.Failure.invalidPoint(1)) {
            try PlanarPath([.move(to: .zero), .quadratic(to: SIMD2(1, 0), control: SIMD2(.nan, 1))]).sampledSubpaths()
        }
        #expect(throws: PlanarPath.Failure.invalidEllipse(0)) {
            try PlanarPath([.ellipticalArc(center: .zero, radii: SIMD2(0, 1), startAngle: 0, sweepAngle: .pi)]).shapes()
        }
        #expect(throws: PlanarPath.Failure.invalidSpline(1)) {
            try PlanarPath([.move(to: .zero), .spline(through: [])]).sampledSubpaths()
        }
        #expect(throws: PlanarPath.Failure.invalidTolerance) { try PlanarPath().sampledSubpaths(tolerance: 0) }
        #expect(throws: PlanarPath.Failure.complexity) {
            try PlanarPath([
                .ellipticalArc(center: .zero, radii: SIMD2(2, 1), startAngle: 0, sweepAngle: 2 * .pi),
            ]).sampledSubpaths(tolerance: 1e-15)
        }
    }
}
