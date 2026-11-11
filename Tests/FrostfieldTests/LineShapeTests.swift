import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Caps, joins, loops and dashes of wide lines against closed-form coverage.
struct LineShapeTests {
    /// The distance in pixels from a point to the segment from `a` to `b`.
    private func distance(_ point: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        let delta = b - a
        let lengthSquared = simd_length_squared(delta)
        let t = lengthSquared > 0 ? min(max(simd_dot(point - a, delta) / lengthSquared, 0), 1) : 0
        return simd_distance(point, a + delta * t)
    }

    /// The distance from a pixel center to the nearest segment body of a
    /// polyline, measured across the segment; infinite where the center projects
    /// past every segment's ends. Every join and cap style covers these bands.
    private func bandDistance(column: Int, row: Int, polyline: [SIMD2<Float>], closed: Bool = false) -> Float {
        let center = SIMD2(Float(column) + 0.5, Float(row) + 0.5)
        let count = closed ? polyline.count : polyline.count - 1
        return (0..<count).map { index -> Float in
            let a = polyline[index]
            let delta = polyline[(index + 1) % polyline.count] - a
            let t = simd_dot(center - a, delta) / simd_length_squared(delta)
            return (0...1).contains(t) ? simd_distance(center, a + delta * t) : .infinity
        }.min() ?? .infinity
    }

    /// The distance from a pixel center to the nearest segment of a polyline.
    private func distance(column: Int, row: Int, polyline: [SIMD2<Float>], closed: Bool = false) -> Float {
        let center = SIMD2(Float(column) + 0.5, Float(row) + 0.5)
        let count = closed ? polyline.count : polyline.count - 1
        return (0..<count).map { distance(center, polyline[$0], polyline[($0 + 1) % polyline.count]) }.min() ?? .infinity
    }

    /// The antialiased alpha of a round-joined stroke: a one-pixel ramp across
    /// the edge of the set of points within `halfWidth` of the polyline, which is
    /// what the stroke's exact distance gives each pixel center.
    private func strokeAlpha(
        column: Int, row: Int, polyline: [SIMD2<Float>], halfWidth: Float, opacity: Float, closed: Bool = false
    ) -> Float {
        let d = distance(column: column, row: row, polyline: polyline, closed: closed)
        return opacity * min(max(halfWidth + 0.5 - d, 0), 1)
    }

    @Test
    func `Butt, square and round caps cover their analytic areas`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        // An 8-pixel stroke whose centerline runs 48 pixels between off-grid ends.
        let start = fixture.point(pixel: SIMD2(40.3, 64.4), depth: 4)
        let end = fixture.point(pixel: SIMD2(88.3, 64.4), depth: 4)
        let line = try LineGeometry(positions: [start, end])
        func area(_ cap: LineMaterial.Cap) throws -> Float {
            let material = LineMaterial(width: .pixels(8), cap: cap)
            let pixels = try fixture.render(
                fixture.graph([SceneNode(primitive: try fixture.lines(line, material))]), name: "cap-\(cap)"
            ).pixels
            return pixels.reduce(0) { $0 + $1.w }
        }
        // Butt: 48 × 8. Square: each end extends by the half width, 56 × 8. A
        // square corner pixel takes the smaller of its two edge ramps instead of
        // their product: at these offsets the four corners add 0.18 + 0.12 + 0.12
        // + 0.18. Round: two half discs of radius 4, plus π/12 that a one-pixel
        // ramp adds to a disc.
        let butt = try area(.butt)
        let square = try area(.square)
        let round = try area(.round)
        #expect(abs(butt - 384.6) < 0.1, "\(butt)")
        #expect(abs(square - 448.6) < 0.1, "\(square)")
        #expect(abs(round - (384 + .pi * 16 + .pi / 12)) < 0.3, "\(round)")
    }

    @Test
    func `Caps of a receding segment stay at their endpoint depths`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let material = LineMaterial(width: .pixels(20), cap: .round)
        // The far end projects `length` pixels from the near end. A 20-pixel
        // round-capped stroke covers a stadium of 20 × length + π × 10² whatever
        // the depths, as long as both ends lie between the clip planes (far is 50).
        for (length, farDepth) in [(Float(2), Float(40)), (0.05, 40), (30, 10), (12, 49)] {
            let near = fixture.point(pixel: SIMD2(50.3, 64.4), depth: 2)
            let far = fixture.point(pixel: SIMD2(50.3 + length, 64.4), depth: farDepth)
            let expected = 20 * length + .pi * 100 + .pi / 12
            for positions in [[near, far], [far, near]] {
                let graph = fixture.graph([
                    SceneNode(primitive: try fixture.lines(try LineGeometry(positions: positions), material)),
                ])
                for depthMode in [MeshDepthMode.forward, .reversed] {
                    let pixels = try fixture.render(graph, depthMode: depthMode, name: "receding-\(length)-\(depthMode)")
                    let area = pixels.reduce(0) { $0 + $1.w }
                    #expect(abs(area - expected) < 0.5, "\(length) px to depth \(farDepth), \(depthMode): \(area) vs \(expected)")
                }
            }
        }
    }

    @Test
    func `A join at a receding vertex keeps its far half`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        // An L whose corner lies at depth 40: in along the image row from depth 3,
        // out upward at the corner's depth. The first segment spans 4 pixels, less
        // than the 6-pixel overlap of the right-angle join.
        let polyline = [SIMD2<Float>(66.3, 70.4), SIMD2(70.3, 70.4), SIMD2(70.3, 30.4)]
        let positions = zip(polyline, [Float(3), 40, 40]).map { fixture.point(pixel: $0.0, depth: $0.1) }
        let material = LineMaterial(width: .pixels(12), join: .round, color: SIMD4(1, 1, 1, 0.5))
        let pixels = try fixture.render(fixture.graph([
            SceneNode(primitive: try fixture.lines(try LineGeometry(positions: positions), material)),
        ]), name: "receding-join").pixels
        var worst: Float = 0
        for row in 0..<128 {
            for column in 0..<128 {
                let expected = strokeAlpha(column: column, row: row, polyline: polyline, halfWidth: 6, opacity: 0.5)
                worst = max(worst, abs(fixture.alpha(pixels, x: column, y: row) - expected))
            }
        }
        #expect(worst < 0.003, "largest alpha error \(worst)")
    }

    @Test
    func `A short segment after a sharp turn leaves no gap and blends once`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        // An 8-pixel stroke runs 60 pixels, then turns 170° and runs back 15 pixels.
        // At that turn the strokes overlap for 4 × tan 85° ≈ 46 pixels, beyond the
        // short segment's end.
        let turn: Float = 170 * .pi / 180
        let corner = SIMD2<Float>(80, 64)
        let polyline = [SIMD2<Float>(20, 64), corner, corner + 15 * SIMD2(cos(turn), sin(turn))]
        let line = try LineGeometry(positions: polyline.map { fixture.point(pixel: $0, depth: 4) })
        func render(_ material: LineMaterial, name: String) throws -> [SIMD4<Float>] {
            try fixture.render(fixture.graph([SceneNode(primitive: try fixture.lines(line, material))]), name: name).pixels
        }
        // Round joins and caps: the stroke is every point within 4 pixels of the
        // polyline, so each pixel's alpha follows from its distance alone.
        let color = SIMD4<Float>(1, 1, 1, 0.5)
        let alpha = try render(LineMaterial(width: .pixels(8), join: .round, color: color), name: "sharp-turn")
        let additive = try render(
            LineMaterial(width: .pixels(8), join: .round, color: color, blending: .additive), name: "sharp-turn-additive"
        )
        let opaque = try render(LineMaterial(width: .pixels(8), join: .round, blending: .opaque), name: "sharp-turn-opaque")
        var worst: Float = 0
        for row in 0..<128 {
            for column in 0..<128 {
                let index = row * 128 + column
                let expected = strokeAlpha(column: column, row: row, polyline: polyline, halfWidth: 4, opacity: 0.5)
                worst = max(worst, abs(alpha[index].w - expected), abs(additive[index].x - expected))
                let d = distance(column: column, row: row, polyline: polyline)
                if abs(d - 4) > 0.01 { #expect(opaque[index].w == (d < 4 ? 1 : 0), "opaque (\(column), \(row))") }
            }
        }
        #expect(worst < 0.003, "largest alpha error \(worst)")
        // The overlap beyond the short segment is covered once.
        #expect(abs(fixture.alpha(alpha, x: 63, y: 65) - 0.5) < 0.002)
        #expect(abs(fixture.alpha(alpha, x: 45, y: 66) - 0.5) < 0.002)

        // Miter (beyond its limit here) and bevel joins: every pixel well inside
        // either segment's band is covered exactly once.
        for join in [LineMaterial.Join.miter(), .bevel] {
            let pixels = try render(LineMaterial(width: .pixels(8), join: join, color: color), name: "sharp-turn-\(join)")
            #expect((pixels.map(\.w).max() ?? 0) <= 0.5 + 0.001)
            for row in 0..<128 {
                for column in 0..<128 where bandDistance(column: column, row: row, polyline: polyline) < 3.5 {
                    #expect(abs(fixture.alpha(pixels, x: column, y: row) - 0.5) < 0.002, "\(join) (\(column), \(row))")
                }
            }
        }
    }

    @Test
    func `A translucent loop closes its last join without overlap`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let polyline = [SIMD2<Float>(24.3, 96.2), SIMD2(104.3, 96.2), SIMD2(64.3, 30.2)]
        let line = try LineGeometry(positions: polyline.map { fixture.point(pixel: $0, depth: 4) }, topology: .loop)
        let color = SIMD4<Float>(1, 1, 1, 0.5)
        func render(_ join: LineMaterial.Join) throws -> [SIMD4<Float>] {
            // Butt caps would leave a notch at the first point if the loop did not close.
            let material = LineMaterial(width: .pixels(8), join: join, cap: .butt, color: color)
            return try fixture.render(fixture.graph([SceneNode(primitive: try fixture.lines(line, material))]), name: "loop-\(join)").pixels
        }
        let round = try render(.round)
        var worst: Float = 0
        for row in 0..<128 {
            for column in 0..<128 {
                let expected = strokeAlpha(column: column, row: row, polyline: polyline, halfWidth: 4, opacity: 0.5, closed: true)
                worst = max(worst, abs(fixture.alpha(round, x: column, y: row) - expected))
            }
        }
        #expect(worst < 0.003, "largest alpha error \(worst)")
        for join in [LineMaterial.Join.miter(), .bevel] {
            let pixels = try render(join)
            #expect((pixels.map(\.w).max() ?? 0) <= 0.5 + 0.001)
            for row in 0..<128 {
                for column in 0..<128 where bandDistance(column: column, row: row, polyline: polyline, closed: true) < 3.5 {
                    #expect(abs(fixture.alpha(pixels, x: column, y: row) - 0.5) < 0.002, "\(join) (\(column), \(row))")
                }
            }
        }
        // Past each bevel a miter adds a triangle of area w² cos³(θ/2) / sin(θ/2) for
        // half width w and interior angle θ, at half opacity.
        let miter = try render(.miter()).reduce(0) { $0 + $1.w }
        let bevel = try render(.bevel).reduce(0) { $0 + $1.w }
        let tips = polyline.indices.reduce(Float(0)) { sum, index in
            let point = polyline[index]
            let a = simd_normalize(polyline[(index + 1) % 3] - point)
            let b = simd_normalize(polyline[(index + 2) % 3] - point)
            let half = acos(simd_dot(a, b)) / 2
            return sum + 16 * pow(cos(half), 3) / sin(half)
        }
        #expect(abs((miter - bevel) - 0.5 * tips) < 1, "\(miter - bevel) vs \(0.5 * tips)")
    }

    @Test
    func `Hairline and wide dashes follow the pattern and its offset`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // At depth 4 one world unit spans 8 pixels: the line runs from x = 8 to 56.
        let start = fixture.point(pixel: SIMD2(8, 20.4), depth: 4)
        let end = fixture.point(pixel: SIMD2(56, 20.4), depth: 4)
        let wideStart = fixture.point(pixel: SIMD2(8, 40.4), depth: 4)
        let wideEnd = fixture.point(pixel: SIMD2(56, 40.4), depth: 4)
        func lit(offset: Float) throws -> (hairline: [Int], wide: [Int]) {
            let dash = try LineMaterial.Dash(length: 1, gap: 1, offset: offset)
            let hairline = LineMaterial(width: .hairline, dash: dash, blending: .opaque)
            let wide = LineMaterial(width: .pixels(4), cap: .butt, dash: dash)
            let pixels = try fixture.render(fixture.graph([
                SceneNode(primitive: try fixture.lines(try LineGeometry(positions: [start, end]), hairline)),
                SceneNode(primitive: try fixture.lines(try LineGeometry(positions: [wideStart, wideEnd]), wide)),
            ]), name: "dashes-\(offset)").pixels
            return (
                (9..<55).filter { fixture.alpha(pixels, x: $0, y: 20) > 0 },
                (9..<55).filter { fixture.alpha(pixels, x: $0, y: 40) > 0.5 }
            )
        }
        // Distance at a pixel center is (x + 0.5 − 8) / 8; dashes cover distances [0, 1), [2, 3), [4, 5).
        let plain = try lit(offset: 0)
        let dashes = Array(9...15) + Array(24...31) + Array(40...47)
        #expect(plain.hairline == dashes)
        #expect(plain.wide == dashes)
        // An offset of 0.5 shifts the pattern back along the line: dashes start at
        // distances 1.5, 3.5 and 5.5.
        let shifted = try lit(offset: 0.5)
        let shiftedDashes = Array(9...11) + Array(20...27) + Array(36...43) + Array(52...54)
        #expect(shifted.hairline == shiftedDashes)
        #expect(shifted.wide == shiftedDashes)
    }
}
