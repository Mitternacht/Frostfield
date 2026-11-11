import Frostfield
import Testing
import simd

struct CurveExtrasTests {
    private struct Fixture: Sendable {
        let name: String
        let curve: @Sendable () throws -> Curve3D
        let points: [SIMD3<Float>]
        var scaled: (curve: @Sendable (Double) throws -> Curve3D, point: SIMD3<Float>)? = nil
    }

    @Test("Every catalog curve matches the tagged JavaScript samples", arguments: Array(CurveExtrasTests.fixtures.indices))
    func matchesCatalog(index: Int) throws {
        let fixture = Self.fixtures[index]
        let curve = try fixture.curve()
        for (parameter, reference) in zip(Self.parameters, fixture.points) {
            let point = try curve.point(at: parameter)
            expect(point, equals: reference, label: "\(fixture.name) at \(parameter)")
        }
        if let scaled = fixture.scaled {
            let curve = try scaled.curve(Self.overrideScale)
            let point = try curve.point(at: Self.scaledParameter)
            expect(point, equals: scaled.point, label: "\(fixture.name) at scale \(Self.overrideScale)")
        }
    }

    @Test("A built-in helix can feed a length-spaced tube")
    func helixTube() throws {
        let points = try Curve3D.helix.points(segments: 64, spacing: .arcLength(subdivisions: 512))
        #expect(points.count == 65)
        expect(try #require(points.first), equals: SIMD3(30, 0, 0), label: "helix start")
        expect(try #require(points.last), equals: SIMD3(30, 0, 150), label: "helix end")
        let tube = try MeshGeometry.tube(along: points, radius: 1, radialSegments: 8, capped: false)
        #expect(tube.vertices.count == 65 * 9)
        #expect(tube.indices.count == 64 * 8 * 6)
    }

    @Test("Scale permits zero and validates non-finite and unrepresentable results")
    func scaleValidation() throws {
        let flat = try Curve3D.heart(scale: 0)
        #expect(try flat.points(segments: 4) == Array(repeating: .zero, count: 5))
        #expect(throws: Curve3D.Failure.controlPoints) { try Curve3D.viviani(scale: .nan) }
        #expect(throws: Curve3D.Failure.controlPoints) { try Curve3D.torusKnot(scale: .infinity) }
        #expect(throws: Curve3D.Failure.unrepresentablePoint) {
            try Curve3D.heart(scale: .greatestFiniteMagnitude).point(at: 0)
        }
        #expect(throws: Curve3D.Failure.parameter) { try Curve3D.helix.point(at: 1.001) }
    }

    private func expect(_ point: SIMD3<Float>, equals reference: SIMD3<Float>, label: String) {
        let tolerance = simd_max(SIMD3<Float>(repeating: 2e-6), simd_abs(reference) * 2e-7)
        #expect(all(simd_abs(point - reference) .<= tolerance), "\(label) expected \(reference), got \(point)")
    }

    // Float-rounded output from Node executing the unmodified Three.js r187
    // CurveExtras.js module. The tagged file was byte-identical to three@0.187.0.
    private static let parameters: [Double] = [0, 0.137, 0.382, 0.731, 1]
    private static let scaledParameter = 0.2718281828459045
    private static let overrideScale = -2.75

    private static let fixtures: [Fixture] = [
        Fixture(name: "GrannyKnot", curve: { Curve3D.grannyKnot }, points: [
            SIMD3(-13.1999998, 5.59999990, 14.0000000),
            SIMD3(-23.0985413, -15.0280209, -16.1115799),
            SIMD3(-31.7794991, -3.88948202, 2.15910101),
            SIMD3(8.24694157, 3.18787122, -2.58491850),
            SIMD3(-13.1999998, 5.59999990, 14.0000000),
        ]),
        Fixture(name: "HeartCurve", curve: { try Curve3D.heart() }, points: [
            SIMD3(0.00000000, 25.0000000, 0.00000000),
            SIMD3(34.8914909, 59.3759346, 0.00000000),
            SIMD3(24.6401596, -51.2911148, 0.00000000),
            SIMD3(-78.3039551, 8.60345268, 0.00000000),
            SIMD3(-.leastNonzeroMagnitude, 25.0000000, 0.00000000),
        ], scaled: ({ try Curve3D.heart(scale: $0) }, SIMD3(-42.7720718, -3.80205870, 0.00000000))),
        Fixture(name: "VivianiCurve", curve: { try Curve3D.viviani() }, points: [
            SIMD3(70.0000000, 0.00000000, 0.00000000),
            SIMD3(29.7421036, 34.6028099, 53.0853348),
            SIMD3(38.0747910, -34.8646774, 47.2732964),
            SIMD3(0.992889345, 8.27746487, -69.5017853),
            SIMD3(70.0000000, -1.71450548e-14, -1.71450548e-14),
        ], scaled: ({ try Curve3D.viviani(scale: $0) }, SIMD3(-0.0514046997, 0.372451991, -2.72417641))),
        Fixture(name: "KnotCurve", curve: { Curve3D.knot }, points: [
            SIMD3(0.00000000, 60.0000000, 0.00000000),
            SIMD3(37.9180946, 27.7626972, 32.2999115),
            SIMD3(33.7666397, 19.8211479, -18.1500130),
            SIMD3(-49.6441307, -0.481764913, -4.01635122),
            SIMD3(-1.22464676e-14, 60.0000000, -1.46957607e-14),
        ]),
        Fixture(name: "HelixCurve", curve: { Curve3D.helix }, points: [
            SIMD3(30.0000000, 0.00000000, 0.00000000),
            SIMD3(-11.9144363, -27.5326385, 20.5499992),
            SIMD3(25.3298378, -16.0748043, 57.2999992),
            SIMD3(-16.8625011, -24.8124180, 109.650002),
            SIMD3(30.0000000, -3.67394036e-14, 150.000000),
        ]),
        Fixture(name: "TrefoilKnot", curve: { try Curve3D.trefoilKnot() }, points: [
            SIMD3(30.0000000, 0.00000000, 0.00000000),
            SIMD3(-1.73108256, 11.3924522, 5.30511189),
            SIMD3(2.29109788, -25.9784698, 7.93990421),
            SIMD3(-22.8385372, 5.55899000, 9.36549854),
            SIMD3(30.0000000, -1.46957607e-14, -7.34788037e-15),
        ], scaled: ({ try Curve3D.trefoilKnot(scale: $0) }, SIMD3(6.35309887, 1.78772497, 2.52048850))),
        Fixture(name: "TorusKnot", curve: { try Curve3D.torusKnot() }, points: [
            SIMD3(30.0000000, 0.00000000, 0.00000000),
            SIMD3(-8.85938263, 5.54456043, -2.97041583),
            SIMD3(6.17314100, 8.06246185, -1.75023055),
            SIMD3(10.1239090, 27.0488396, -4.59579849),
            SIMD3(30.0000000, -2.20436428e-14, -9.79717440e-15),
        ], scaled: ({ try Curve3D.torusKnot(scale: $0) }, SIMD3(-3.13811016, 7.19159460, -1.43411112))),
        Fixture(name: "CinquefoilKnot", curve: { try Curve3D.cinquefoilKnot() }, points: [
            SIMD3(30.0000000, 0.00000000, 0.00000000),
            SIMD3(-2.40789413, 15.8466253, -9.17754650),
            SIMD3(2.49877620, -28.3333073, -5.35826778),
            SIMD3(-13.9712543, 3.40065837, -8.27080536),
            SIMD3(30.0000000, -1.46957607e-14, -1.22464676e-14),
        ], scaled: ({ try Curve3D.cinquefoilKnot(scale: $0) }, SIMD3(3.61803269, 1.01809323, -2.12834239))),
        Fixture(name: "TrefoilPolynomialKnot", curve: { try Curve3D.trefoilPolynomialKnot() }, points: [
            SIMD3(-20.0000000, 0.00000000, -24.0000000),
            SIMD3(12.9474258, -39.8827019, 16.1318779),
            SIMD3(13.1084595, -8.41503334, 9.39314651),
            SIMD3(-19.8311100, -26.8617058, -17.1329308),
            SIMD3(20.0000000, 0.00000000, 24.0000000),
        ], scaled: ({ try Curve3D.trefoilPolynomialKnot(scale: $0) }, SIMD3(-5.43893671, 7.25479269, -4.67146444))),
        Fixture(name: "FigureEightPolynomialKnot", curve: { try Curve3D.figureEightPolynomialKnot() }, points: [
            SIMD3(-86.4000015, 48.0000000, -134.399994),
            SIMD3(2.60841918, -38.5126762, -2.60261488),
            SIMD3(-21.0114803, -10.7906446, 26.4364643),
            SIMD3(17.4496784, -32.7334175, -5.18239784),
            SIMD3(86.4000015, 48.0000000, 134.399994),
        ], scaled: ({ try Curve3D.figureEightPolynomialKnot(scale: $0) }, SIMD3(49.1101227, 88.5877457, -16.4746399))),
        Fixture(name: "DecoratedTorusKnot4a", curve: { try Curve3D.decoratedTorusKnot4a() }, points: [
            SIMD3(82.0000000, 0.00000000, 0.00000000),
            SIMD3(-2.72608399, 17.9406700, -12.8485651),
            SIMD3(5.96754694, -67.6652603, -7.50157499),
            SIMD3(-19.3196888, 4.70248842, -11.5791283),
            SIMD3(82.0000000, -4.01684166e-14, -1.71450548e-14),
        ], scaled: ({ try Curve3D.decoratedTorusKnot4a(scale: $0) }, SIMD3(1.40554714, 0.395512760, -0.744919837))),
        Fixture(name: "DecoratedTorusKnot4b", curve: { try Curve3D.decoratedTorusKnot4b() }, points: [
            SIMD3(74.0000000, 0.00000000, 0.00000000),
            SIMD3(-3.97310257, 26.1474419, 7.95440626),
            SIMD3(3.17507482, -36.0017738, 3.03823280),
            SIMD3(-31.3259487, 7.62485933, -3.80990553),
            SIMD3(74.0000000, -3.62495441e-14, -1.76349143e-14),
        ], scaled: ({ try Curve3D.decoratedTorusKnot4b(scale: $0) }, SIMD3(2.12410283, 0.597710133, -0.181571692))),
        Fixture(name: "DecoratedTorusKnot5a", curve: { try Curve3D.decoratedTorusKnot5a() }, points: [
            SIMD3(72.0000000, 0.00000000, 0.00000000),
            SIMD3(-18.2617664, 11.4289522, -7.98421383),
            SIMD3(35.6536064, 46.5655670, -6.16410589),
            SIMD3(9.07621288, 24.2496262, -5.47637701),
            SIMD3(72.0000000, -5.29047428e-14, -3.91886976e-14),
        ], scaled: ({ try Curve3D.decoratedTorusKnot5a(scale: $0) }, SIMD3(-0.782022595, 1.79215813, -0.213461682))),
        Fixture(name: "DecoratedTorusKnot5c", curve: { try Curve3D.decoratedTorusKnot5c() }, points: [
            SIMD3(68.0000000, 0.00000000, 0.00000000),
            SIMD3(-30.1304817, -9.37306309, 4.74233103),
            SIMD3(-50.9877968, -9.06394863, -13.8896055),
            SIMD3(20.3619347, -10.5365963, -3.05400538),
            SIMD3(68.0000000, -6.66207879e-14, -1.50911144e-13),
        ], scaled: ({ try Curve3D.decoratedTorusKnot5c(scale: $0) }, SIMD3(-1.17099500, -0.715692759, -0.449969471))),
    ]
}
