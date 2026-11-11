import Foundation
import Frostfield
import Metal
import Testing
import simd

struct LineRenderingTests {
    /// Coverage-weighted row centroid and total coverage of one pixel column.
    private func columnCoverage(_ pixels: [SIMD4<Float>], column: Int, side: Int) -> (sum: Float, centroid: Float) {
        var sum: Float = 0
        var moment: Float = 0
        for row in 0..<side {
            let alpha = pixels[row * side + column].w
            sum += alpha
            moment += alpha * (Float(row) + 0.5)
        }
        return (sum, sum > 0 ? moment / sum : .nan)
    }

    @Test
    func `A four-pixel line covers four pixels per column centered on its projection`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let depth: Float = 5
        // Center the line between pixel rows so its edges cut pixels fractionally.
        let start = fixture.point(pixel: SIMD2(44.8, 40.3), depth: depth)
        let end = fixture.point(pixel: SIMD2(83.2, 40.3), depth: depth)
        let line = try LineGeometry(positions: [start, end])
        let material = LineMaterial(width: .pixels(4), cap: .butt, color: SIMD4(1, 1, 1, 1))
        let pixels = try fixture.render(
            fixture.graph([SceneNode(primitive: try fixture.lines(line, material))]), name: "line-4px"
        ).pixels

        for column in 47...80 {
            let coverage = columnCoverage(pixels, column: column, side: 128)
            #expect(abs(coverage.sum - 4) < 0.02, "column \(column) covers \(coverage.sum) pixels")
            #expect(abs(coverage.centroid - 40.3) < 0.02, "column \(column) is centered at \(coverage.centroid)")
        }
        // Butt ends stop at the projected endpoints: total coverage is length × width.
        let total = pixels.reduce(0) { $0 + $1.w }
        #expect(abs(total - (83.2 - 44.8) * 4) < 0.5)
        #expect(fixture.alpha(pixels, x: 43, y: 40) == 0)
        #expect(fixture.alpha(pixels, x: 84, y: 40) == 0)
    }

    @Test
    func `A slanted line keeps its width perpendicular to its direction`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let depth: Float = 4
        // Rise 0.5 pixel per pixel: a column crosses the stroke over width / cos(θ).
        let start = fixture.point(pixel: SIMD2(20, 90.25), depth: depth)
        let end = fixture.point(pixel: SIMD2(108, 46.25), depth: depth)
        let material = LineMaterial(width: .pixels(3), cap: .butt)
        let pixels = try fixture.render(fixture.graph([
            SceneNode(primitive: try fixture.lines(try LineGeometry(positions: [start, end]), material)),
        ])).pixels
        let expected = 3 / cos(atan(Float(0.5)))
        for column in stride(from: 30, through: 98, by: 4) {
            let coverage = columnCoverage(pixels, column: column, side: 128)
            let center = 90.25 - 0.5 * (Float(column) + 0.5 - 20)
            #expect(abs(coverage.sum - expected) < 0.05, "column \(column) covers \(coverage.sum)")
            #expect(abs(coverage.centroid - center) < 0.05)
        }
    }

    @Test
    func `A world-unit width projects to width times focal scale over depth`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        // Two vertical lines 0.25 units wide at depths 2 and 8: 8 and 2 pixels on screen.
        func node(depth: Float, x: Float) throws -> SceneNode {
            let top = fixture.point(pixel: SIMD2(x, 20), depth: depth)
            let bottom = fixture.point(pixel: SIMD2(x, 100), depth: depth)
            let material = LineMaterial(width: .world(0.25), cap: .butt)
            return SceneNode(primitive: try fixture.lines(try LineGeometry(positions: [top, bottom]), material))
        }
        let pixels = try fixture.render(fixture.graph([try node(depth: 2, x: 40), try node(depth: 8, x: 90)])).pixels
        for row in 30...90 {
            let near = (20..<60).reduce(Float(0)) { $0 + fixture.alpha(pixels, x: $1, y: row) }
            let far = (70..<110).reduce(Float(0)) { $0 + fixture.alpha(pixels, x: $1, y: row) }
            #expect(abs(near - 0.25 * 64 / 2) < 0.02, "row \(row): \(near)")
            #expect(abs(far - 0.25 * 64 / 8) < 0.02, "row \(row): \(far)")
        }
    }

    @Test
    func `Dashes break at perspective-correct world distances along a receding line`() throws {
        let fixture = try PrimitiveRenderFixture(side: 256)
        // From depth 2 to 10 at x = −1 and y = 0: the line distance equals depth − 2.
        let line = try LineGeometry(positions: [SIMD3(-1, 0, -2), SIMD3(-1, 0, -10)])
        let material = LineMaterial(width: .pixels(4), cap: .butt, dash: try LineMaterial.Dash(length: 1, gap: 1))
        let pixels = try fixture.render(
            fixture.graph([SceneNode(primitive: try fixture.lines(line, material))]), name: "line-dashes"
        ).pixels
        func coverage(atDepth depth: Float) -> Float {
            let column = Int(fixture.pixel(SIMD3(-1, 0, -depth)).x)
            return (125..<131).reduce(Float(0)) { $0 + fixture.alpha(pixels, x: column, y: $1) }
        }
        // Dashes span distances [0, 1], [2, 3], [4, 5]; the middle of each dash and gap.
        for depth: Float in [2.5, 4.5, 6.5] { #expect(abs(coverage(atDepth: depth) - 4) < 0.05, "dash at \(depth)") }
        for depth: Float in [3.5, 5.5] { #expect(coverage(atDepth: depth) < 0.01, "gap at \(depth)") }
        // Each on–off transition crosses half coverage within a third of a pixel of the projected boundary.
        for boundary: Float in [3, 4, 5, 6] {
            let expected = fixture.pixel(SIMD3(-1, 0, -boundary)).x
            var crossing: Float?
            for column in Int(expected) - 3...Int(expected) + 2 {
                let a = (125..<131).reduce(Float(0)) { $0 + fixture.alpha(pixels, x: column, y: $1) } / 4
                let b = (125..<131).reduce(Float(0)) { $0 + fixture.alpha(pixels, x: column + 1, y: $1) } / 4
                if (a - 0.5) * (b - 0.5) <= 0, a != b {
                    crossing = Float(column) + 0.5 + (0.5 - a) / (b - a)
                }
            }
            let found = try #require(crossing, "no transition near \(expected)")
            #expect(abs(found - expected) < 0.35, "boundary at depth \(boundary): \(found) vs \(expected)")
        }
    }

    @Test
    func `Hairlines rasterize one pixel wide on the projected row`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        let start = fixture.point(pixel: SIMD2(10.5, 20.4), depth: 3)
        let end = fixture.point(pixel: SIMD2(50.5, 20.4), depth: 3)
        let material = LineMaterial(width: .hairline, color: SIMD4(0.2, 0.6, 1, 1), blending: .opaque)
        let pixels = try fixture.render(fixture.graph([
            SceneNode(primitive: try fixture.lines(try LineGeometry(positions: [start, end]), material)),
        ])).pixels
        for column in 12...48 {
            let rows = (0..<64).filter { fixture.alpha(pixels, x: column, y: $0) > 0 }
            #expect(rows == [20], "column \(column) lit rows \(rows)")
            #expect(isNearly(pixels[20 * 64 + column], SIMD4(0.2, 0.6, 1, 1)))
        }
    }

    @Test
    func `Joins follow miter, bevel and round geometry without shading any pixel twice`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let depth: Float = 4
        // A right-angle corner at pixel (64, 64): in from the left, out upward.
        let corner = SIMD2<Float>(64, 64)
        let points = [SIMD2<Float>(24, 64), corner, SIMD2(64, 24)].map { fixture.point(pixel: $0, depth: depth) }
        let line = try LineGeometry(positions: points)
        func render(_ join: LineMaterial.Join) throws -> [SIMD4<Float>] {
            let material = LineMaterial(width: .pixels(8), join: join, cap: .butt, color: SIMD4(1, 1, 1, 0.5))
            return try fixture.render(
                fixture.graph([SceneNode(primitive: try fixture.lines(line, material))]), name: "join-\(join)"
            ).pixels
        }
        let miter = try render(.miter())
        let bevel = try render(.bevel)
        let round = try render(.round)
        // The outer corner lies down and to the right of the vertex. Its miter tip
        // is at (+4, +4) pixels; a bevel cuts at 4·cos 45° along the diagonal, and
        // a round join at radius 4.
        func alpha(_ pixels: [SIMD4<Float>], _ dx: Int, _ dy: Int) -> Float {
            fixture.alpha(pixels, x: Int(corner.x) + dx, y: Int(corner.y) + dy)
        }
        #expect(abs(alpha(miter, 3, 3) - 0.5) < 0.01)
        #expect(alpha(bevel, 3, 3) == 0)
        #expect(alpha(round, 3, 3) == 0)
        // At (+2.5, +2.5) from the vertex the round join's edge is 0.46 pixels out: 96% coverage.
        #expect(abs(alpha(round, 2, 2) - 0.48) < 0.01)
        #expect(alpha(bevel, 2, 2) == 0)
        // Half-transparent strokes stay at exactly half alpha through the join:
        // the two segments split it along the bisector instead of overlapping.
        for pixels in [miter, bevel, round] {
            for dy in -3...1 {
                for dx in -3...1 {
                    #expect(abs(alpha(pixels, dx, dy) - 0.5) < 0.01, "(\(dx), \(dy)) = \(alpha(pixels, dx, dy))")
                }
            }
        }
        let peak = [miter, bevel, round].flatMap { $0.map(\.w) }.max() ?? 0
        #expect(peak <= 0.5 + 0.001)
    }

    @Test
    func `A miter beyond its limit becomes a bevel`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        // Turning back to a 20° interior angle: the miter would reach 1 / sin(10°) ≈ 5.8 times
        // the half width past the vertex, beyond the default limit of 4.
        let corner = SIMD2<Float>(80, 64)
        let back = SIMD2<Float>(corner.x - 40 * cos(.pi / 9), corner.y - 40 * sin(.pi / 9))
        let points = [SIMD2<Float>(30, 64), corner, back].map { fixture.point(pixel: $0, depth: 4) }
        let line = try LineGeometry(positions: points)
        func coverageBeyondCorner(_ join: LineMaterial.Join) throws -> Float {
            let material = LineMaterial(width: .pixels(6), join: join, cap: .butt, blending: .opaque)
            let pixels = try fixture.render(fixture.graph([SceneNode(primitive: try fixture.lines(line, material))])).pixels
            return (Int(corner.x) + 4..<128).reduce(Float(0)) { sum, x in
                sum + (50..<78).reduce(Float(0)) { $0 + fixture.alpha(pixels, x: x, y: $1) }
            }
        }
        let limited = try coverageBeyondCorner(.miter(limit: 4))
        let bevel = try coverageBeyondCorner(.bevel)
        let unlimited = try coverageBeyondCorner(.miter(limit: 10))
        // The unlimited tip reaches about 17 pixels past the vertex; a bevel stays within one.
        #expect(limited == 0)
        #expect(bevel == 0)
        #expect(unlimited > 20)
    }

    @Test
    func `Segments crossing the near plane are trimmed in forward and reversed depth`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // From 5 units ahead, past the camera, to 5 units behind it, in both directions.
        let ahead = SIMD3<Float>(0.5, -1, -5)
        let behind = SIMD3<Float>(0.5, -1, 5)
        let material = LineMaterial(width: .pixels(2), cap: .butt)
        var images: [[SIMD4<Float>]] = []
        for (index, positions) in [[ahead, behind], [behind, ahead]].enumerated() {
            let graph = fixture.graph([SceneNode(primitive: try fixture.lines(try LineGeometry(positions: positions), material))])
            images.append(try fixture.render(graph, depthMode: .forward, name: "near-forward-\(index)"))
            images.append(try fixture.render(graph, depthMode: .reversed, name: "near-reversed-\(index)"))
        }
        let forward = images[0]
        let reversed = images[1]
        // The visible half runs from the far end down toward the bottom-right edge.
        let start = fixture.pixel(ahead)
        for pixels in images {
            let lit = pixels.enumerated().filter { $0.element.w > 0.01 }.map { SIMD2(Float($0.offset % 64), Float($0.offset / 64)) }
            #expect(!lit.isEmpty)
            #expect(lit.allSatisfy { $0.y + 2 >= start.y && $0.x + 2 >= start.x }, "pixels outside the visible half")
            #expect(lit.contains { $0.y >= 62 })
            #expect(pixels.allSatisfy { $0.w.isFinite && $0.x.isFinite })
        }
        #expect(zip(forward, reversed).allSatisfy { isNearly($0, $1, tolerance: 0.02) })
        #expect(zip(forward, images[2]).allSatisfy { isNearly($0, $1, tolerance: 0.02) })
    }

    @Test
    func `Wide lines are depth tested against meshes in forward and reversed depth`() throws {
        let fixture = try PrimitiveRenderFixture(side: 64)
        // A square at depth 4 hides the middle of a line at depth 6.
        func graph(_ depthMode: MeshDepthMode) throws -> SceneGraph {
            let square = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.5),
                material: .unlit, depthMode: depthMode
            )
            let line = try LineGeometry(positions: [SIMD3(-4, 0, -6), SIMD3(4, 0, -6)])
            let material = LineMaterial(width: .pixels(4), color: SIMD4(1, 0, 0, 1), blending: .opaque)
            return fixture.graph([
                SceneNode(primitive: try fixture.lines(line, material)),
                SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -4)), mesh: square, materialParameters: SIMD4(0, 0, 1, 1)),
            ])
        }
        let forward = try fixture.render(try graph(.forward), depthMode: .forward, name: "occlusion-forward")
        let reversed = try fixture.render(try graph(.reversed), depthMode: .reversed, name: "occlusion-reversed")
        for pixels in [forward, reversed] {
            // Inside the square (±4 pixels) the blue mesh wins; outside it the red line shows.
            #expect(pixels[32 * 64 + 32].z > 0.9 && pixels[32 * 64 + 32].x == 0)
            #expect(pixels[32 * 64 + 14].x > 0.99 && pixels[32 * 64 + 14].z == 0)
        }
        #expect(zip(forward, reversed).allSatisfy { isNearly($0, $1) })
    }

    @Test
    func `Vertex colors interpolate perspective-correctly along a segment`() throws {
        let fixture = try PrimitiveRenderFixture(side: 256)
        let line = try LineGeometry(
            positions: [SIMD3(-1, 0, -2), SIMD3(-1, 0, -10)],
            colors: [SIMD4(1, 0, 0, 1), SIMD4(0, 0, 1, 1)]
        )
        let pixels = try fixture.render(fixture.graph([
            SceneNode(primitive: try fixture.lines(line, LineMaterial(width: .pixels(4), cap: .butt, blending: .opaque))),
        ])).pixels
        // Near depth 6 the pixel is about halfway along the line in world space, but
        // 83% of the way along it on screen.
        let column = Int(fixture.pixel(SIMD3(-1, 0, -6)).x)
        let depth = 128 / (128 - (Float(column) + 0.5))
        let t = (depth - 2) / 8
        let sample = pixels[127 * 256 + column]
        #expect(abs(sample.x - (1 - t)) < 0.01 && abs(sample.z - t) < 0.01, "\(sample) vs \(t)")
    }

    @Test
    func `Line materials and geometry reject invalid input`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let line = try LineGeometry(positions: [.zero, SIMD3(1, 0, 0)])
        #expect(throws: PrimitiveRendererError.size) {
            try LineRenderer(device: device, geometry: line, material: LineMaterial(width: .pixels(0)))
        }
        #expect(throws: PrimitiveRendererError.miterLimit) {
            try LineRenderer(device: device, geometry: line, material: LineMaterial(join: .miter(limit: 0.5)))
        }
        #expect(throws: PrimitiveRendererError.color) {
            try LineRenderer(device: device, geometry: line, material: LineMaterial(color: SIMD4(1, 1, 1, 2)))
        }
        #expect(throws: PrimitiveRendererError.dash) { try LineMaterial.Dash(length: 0, gap: 1) }
        #expect(throws: PrimitiveGeometryError.lineTopology) { try LineGeometry(positions: [.zero]) }
        #expect(throws: PrimitiveGeometryError.lineTopology) {
            try LineGeometry(positions: [.zero, SIMD3(1, 0, 0), SIMD3(2, 0, 0)], topology: .segments)
        }
        #expect(throws: PrimitiveGeometryError.lineTopology) {
            try LineGeometry(positions: [.zero, SIMD3(1, 0, 0)], topology: .loop)
        }
        #expect(throws: PrimitiveGeometryError.invalidPosition(1)) {
            try LineGeometry(positions: [.zero, SIMD3(.nan, 0, 0)])
        }
        #expect(throws: PrimitiveGeometryError.attributeCount) {
            try LineGeometry(positions: [.zero, SIMD3(1, 0, 0)], colors: [SIMD4(1, 1, 1, 1)])
        }
    }

    @Test
    func `Line distances accumulate like computeLineDistances and joins link neighbors`() throws {
        let points: [SIMD3<Float>] = [.zero, SIMD3(3, 0, 0), SIMD3(3, 4, 0), SIMD3(3, 4, 0), SIMD3(0, 4, 0)]
        let strip = try LineGeometry(positions: points)
        // The repeated point is skipped, so its neighbors join directly.
        #expect(strip.segments.map(\.start) == [0, 1, 3])
        #expect(strip.segments.map(\.end) == [1, 2, 4])
        #expect(strip.segments.map(\.previous) == [nil, 0, 1])
        #expect(strip.segments.map(\.next) == [2, 4, nil])
        #expect(strip.segments.map(\.startDistance) == [0, 3, 7])
        #expect(strip.segments.map(\.endDistance) == [3, 7, 10])

        let loop = try LineGeometry(positions: [.zero, SIMD3(3, 0, 0), SIMD3(3, 4, 0)], topology: .loop)
        #expect(loop.segments.count == 3)
        #expect(loop.segments.last?.endDistance == 12)
        #expect(loop.segments.first?.previous == 2)
        #expect(loop.segments.last?.next == 1)

        let pairs = try LineGeometry(positions: [.zero, SIMD3(2, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 2, 0)], topology: .segments)
        #expect(pairs.segments.map(\.startDistance) == [0, 2])
        #expect(pairs.segments.map(\.endDistance) == [2, 3])
        #expect(pairs.segments.allSatisfy { $0.previous == nil && $0.next == nil })
    }
}
