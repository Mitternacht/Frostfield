import Foundation
import Frostfield
import Testing
import simd

struct Curve3DTests {
    @Test("Arc-length sampling inverts a nonuniform parameterization")
    func arcLength() throws {
        let curve = Curve3D(position: { t in SIMD3(4 * t * t, 0, 0) })
        let samples = try curve.points(segments: 4, spacing: .arcLength(subdivisions: 2048))
        #expect(samples.count == 5)
        for (index, point) in samples.enumerated() {
            #expect(abs(point.x - Float(index)) < 1e-6)
        }
        #expect(try curve.length() == 4)
        #expect(try curve.point(at: 0.5) == SIMD3(1, 0, 0))
        #expect(try abs(curve.point(atLengthFraction: 0.5, subdivisions: 2048).x - 2) < 1e-6)
        #expect(try curve.tangent(at: 0) == SIMD3(1, 0, 0))
    }

    @Test("Bezier positions and analytic tangents feed the existing tube path")
    func bezierSweep() throws {
        let a = SIMD3<Float>(0, 0, 0), b = SIMD3<Float>(1, 0, 0)
        let c = SIMD3<Float>(2, 1, 0), d = SIMD3<Float>(2, 2, 0)
        let curve = try Curve3D.cubicBezier(from: a, control1: b, control2: c, to: d)
        #expect(try curve.point(at: 0.5) == SIMD3(1.375, 0.625, 0))
        #expect(try curve.tangent(at: 0) == SIMD3(1, 0, 0))
        #expect(try curve.tangent(at: 1) == SIMD3(0, 1, 0))
        let old = try ProceduralPath.cubicBezier(from: a, control1: b, control2: c, to: d, segments: 16)
        #expect(try curve.points(segments: 16) == old)
        let spaced = try curve.points(segments: 16, spacing: .arcLength())
        let mesh = try MeshGeometry.tube(along: spaced, radius: 0.1, radialSegments: 8)
        #expect(mesh.indices.count == 17 * 8 * 6)
    }

    @Test("All Catmull–Rom parameterizations match Three.js r187 for open and closed curves")
    func catmullRomReference() throws {
        let points: [SIMD3<Float>] = [
            SIMD3(0, 0, 0), SIMD3(1, 2, 0.25), SIMD3(4, 3, -0.5), SIMD3(5, 1, 1),
        ]
        // Generated with the three@0.187.0 npm module and tension 0.3.
        let fixture: [(Curve3D.CatmullRomParameterization, Bool, Double, SIMD3<Float>)] = [
            (.centripetal, false, 0.37, SIMD3(1.2552004, 2.194784, 0.20394049)),
            (.chordal, false, 0.37, SIMD3(1.2604698, 2.2273595, 0.20962681)),
            (.uniform(tension: 0.3), false, 0.37, SIMD3(1.1925484, 2.1152866, 0.20927883)),
            (.centripetal, true, 0.93, SIMD3(1.0389404, -0.13199005, 0.26040918)),
            (.chordal, true, 0.93, SIMD3(0.917278, -0.39546034, 0.26503748)),
            (.uniform(tension: 0.3), true, 0.93, SIMD3(1.0629248, 0.0969472, 0.2324224)),
        ]
        for (mode, closed, t, expected) in fixture {
            let curve = try Curve3D.catmullRom(points, closed: closed, parameterization: mode)
            #expect(try simd_distance(curve.point(at: t), expected) < 1e-6)
            #expect(try curve.point(at: 0) == points[0])
            #expect(try curve.point(at: 1) == (closed ? points[0] : points[3]))
            if closed {
                #expect(try simd_distance(curve.tangent(at: 0), curve.tangent(at: 1)) < 1e-6)
            }
        }
        let repeated = try Curve3D.catmullRom([points[0], points[1], points[1], points[2]])
        #expect(try simd_distance(repeated.point(at: 0.55), SIMD3(0.9334351, 2.0132006, 0.2772579)) < 1e-6)
    }

    @Test("A rational quadratic follows a unit quarter circle and its analytic tangent")
    func nurbsCircle() throws {
        let points: [NURBSControlPoint] = [
            .init(SIMD3(1, 0, 0)), .init(SIMD3(1, 1, 0), weight: sqrt(0.5)), .init(SIMD3(0, 1, 0)),
        ]
        let curve = try Curve3D.nurbs(degree: 2, knots: [0, 0, 0, 1, 1, 1], controlPoints: points)
        #expect(try curve.point(at: 0) == SIMD3(1, 0, 0))
        #expect(try curve.point(at: 1) == SIMD3(0, 1, 0))
        #expect(try curve.tangent(at: 0) == SIMD3(0, 1, 0))
        #expect(try curve.tangent(at: 1) == SIMD3(-1, 0, 0))
        let diagonal: Float = sqrt(0.5)
        #expect(try simd_distance(curve.point(at: 0.5), SIMD3(diagonal, diagonal, 0)) < 1e-6)
        #expect(try simd_distance(curve.tangent(at: 0.5), SIMD3(-diagonal, diagonal, 0)) < 1e-6)
        #expect(try abs(curve.length(subdivisions: 2048) - .pi / 2) < 1e-6)
    }

    @Test("Knot windows and repeated active endpoints evaluate the correct span")
    func nurbsKnotDomains() throws {
        let points: [NURBSControlPoint] = (0..<5).map { .init(SIMD3(Float($0), 0, 0)) }
        let curve = try Curve3D.nurbs(degree: 2, knots: [-1, 0, 0, 1, 1, 1, 2, 3], controlPoints: points)
        #expect(try curve.point(at: 1) == SIMD3(2, 0, 0))
        #expect(try curve.tangent(at: 1) == SIMD3(1, 0, 0))
        let firstHalf = try Curve3D.nurbs(
            degree: 1, knots: [0, 0, 0.5, 1, 1], controlPoints: Array(points.prefix(3)), knotRange: 1...2
        )
        #expect(try firstHalf.point(at: 1) == SIMD3(1, 0, 0))
        let tiny = try Curve3D.nurbs(degree: 1, knots: [0, 0, 1e-320, 1e-320], controlPoints: Array(points.prefix(2)))
        #expect(try tiny.tangent(at: 0.5) == SIMD3(1, 0, 0))
    }

    @Test("A weighted control net evaluates and tessellates in UV order")
    func nurbsSurface() throws {
        let surface = try NURBSSurface(
            uDegree: 1, vDegree: 1, uKnots: [0, 0, 1, 1], vKnots: [0, 0, 1, 1],
            controlPoints: [
                [.init(.zero), .init(SIMD3(2, 0, 0))],
                [.init(SIMD3(0, 2, 0)), .init(SIMD3(2, 2, 2), weight: 3)],
            ]
        )
        #expect(try surface.point(u: 1, v: 0) == SIMD3(2, 0, 0))
        #expect(try surface.point(u: 0, v: 1) == SIMD3(0, 2, 0))
        let center = try surface.point(u: 0.5, v: 0.5)
        #expect(simd_distance(center, SIMD3(4.0 / 3, 4.0 / 3, 1)) < 1e-6)
        let mesh = try surface.mesh(uSegments: 2, vSegments: 2)
        #expect(mesh.vertices.count == 9 && mesh.indices.count == 24)
        #expect(mesh.vertices[4].position == center && mesh.vertices[4].textureCoordinate == SIMD2(0.5, 0.5))
        #expect(mesh.vertices.allSatisfy { $0.normal.z > 0 && abs(simd_length($0.normal) - 1) < 1e-6 })
    }

    @Test("Invalid curve settings, knots, weights, and singular tangents are reported")
    func validation() throws {
        let constant = Curve3D(position: { _ in .zero })
        #expect(throws: Curve3D.Failure.undefinedTangent) { try constant.tangent(at: 0.5) }
        #expect(throws: Curve3D.Failure.parameter) { try constant.point(at: .nan) }
        #expect(throws: Curve3D.Failure.resolution) { try constant.points(segments: 0) }
        #expect(throws: Curve3D.Failure.controlPoints) { try Curve3D.catmullRom([.zero]) }
        #expect(throws: NURBSFailure.knots) {
            try Curve3D.nurbs(degree: 1, knots: [0, 1, 0, 1], controlPoints: [.init(.zero), .init(SIMD3(1, 0, 0))])
        }
        #expect(throws: NURBSFailure.controlPoint) {
            try Curve3D.nurbs(degree: 1, knots: [0, 0, 1, 1], controlPoints: [.init(.zero, weight: 0), .init(.zero)])
        }
        #expect(throws: NURBSFailure.controlNet) {
            try NURBSSurface(uDegree: 1, vDegree: 1, uKnots: [], vKnots: [], controlPoints: [[.init(.zero)], []])
        }
    }
}
