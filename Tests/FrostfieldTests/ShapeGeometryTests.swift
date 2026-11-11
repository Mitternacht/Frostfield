import CoreGraphics
import Foundation
import Frostfield
import Testing
import simd

struct ShapeGeometryTests {
    private func square(_ lower: SIMD2<Float>, _ side: Float) -> [SIMD2<Float>] {
        [lower, lower + SIMD2(side, 0), lower + SIMD2(side, side), lower + SIMD2(0, side)]
    }

    @Test("A hole touching its outer ring fills correctly but the extrusion is a non-manifold Boolean operand")
    func touchingRings() throws {
        let outer = square(.zero, 4)
        let hole: [SIMD2<Float>] = [.zero, SIMD2(1, 0.5), SIMD2(0.5, 1)]
        let shapes = try PlanarShape.shapes(rings: [outer, hole], fillRule: .evenOdd)
        #expect(shapes.map(\.area) == [15.625])
        let extrusion = try MeshGeometry.extrude(shapes, ShapeExtrusion(depth: 0.5))
        #expect(abs(extrusion.signedVolume - 7.8125) < 1e-8)
        #expect(!ShapeCheck.isClosed(extrusion))
        #expect(throws: MeshBoolean.Failure.nonManifoldEdge(operand: 1)) {
            try MeshGeometry.cube().subtracting(extrusion)
        }
    }

    @Test("Overlapping squares resolve to their union under nonzero and their symmetric difference under even-odd")
    func fillRules() throws {
        let rings = [square(SIMD2(0, 0), 2), square(SIMD2(1, 1), 2)]
        let union = try PlanarShape.shapes(rings: rings, fillRule: .winding)
        #expect(union.count == 1)
        #expect(union[0].holes.isEmpty)
        #expect(union[0].area == 7)
        let difference = try PlanarShape.shapes(rings: rings, fillRule: .evenOdd)
        // Two L shapes that touch only at (1, 2) and (2, 1).
        #expect(difference.count == 2)
        #expect(difference.allSatisfy { $0.area == 3 && $0.holes.isEmpty })

        // A square drawn inside another with the same winding is no hole under nonzero, but is under even-odd.
        let nested = [square(SIMD2(0, 0), 4), square(SIMD2(1, 1), 2)]
        #expect(try PlanarShape.shapes(rings: nested, fillRule: .winding).map(\.area) == [16])
        let ring = try PlanarShape.shapes(rings: nested, fillRule: .evenOdd)
        #expect(ring.count == 1 && ring[0].holes.count == 1 && ring[0].area == 12)
        // Reversing the inner square makes it a hole under both rules.
        let reversed = [square(SIMD2(0, 0), 4), square(SIMD2(1, 1), 2).reversed()]
        #expect(try PlanarShape.shapes(rings: reversed, fillRule: .winding).map(\.area) == [12])
    }

    @Test("A self-intersecting pentagram fills its center under nonzero and leaves it empty under even-odd")
    func pentagram() throws {
        let radius = 2.0
        let tips = (0..<5).map { index -> SIMD2<Double> in
            let angle = Double.pi / 2 + 2 * Double.pi * Double(index) / 5
            return SIMD2(cos(angle), sin(angle)) * radius
        }
        // Drawn tip to tip, every second point, so the five edges cross.
        let star = (0..<5).map { SIMD2<Float>(tips[($0 * 2) % 5]) }
        // Independent reference: the inner pentagon's circumradius is R cos 72° / cos 36°.
        let inner = radius * cos(2 * .pi / 5) / cos(.pi / 5)
        let pentagonArea = 5.0 / 2 * inner * inner * sin(2 * .pi / 5)
        let outline: [SIMD2<Double>] = (0..<10).map { index in
            let angle = Double.pi / 2 + Double.pi * Double(index) / 5
            return SIMD2(cos(angle), sin(angle)) * (index.isMultiple(of: 2) ? radius : inner)
        }
        var starArea = 0.0
        for index in outline.indices {
            let a = outline[index], b = outline[(index + 1) % outline.count]
            starArea += (a.x * b.y - b.x * a.y) / 2
        }

        let filled = try PlanarShape.shapes(rings: [star], fillRule: .winding)
        #expect(filled.count == 1 && filled[0].holes.isEmpty && filled[0].outer.count == 10)
        #expect(abs(filled[0].area - starArea) < 1e-6)
        let points = try PlanarShape.shapes(rings: [star], fillRule: .evenOdd)
        #expect(points.count == 5)
        #expect(abs(points.reduce(0) { $0 + $1.area } - (starArea - pentagonArea)) < 1e-6)
    }

    /// The exact area inside a path's Bézier curves: Green's theorem with
    /// 3-point Gauss–Legendre quadrature, exact for the degree-5 cubic integrand.
    private func exactArea(_ path: CGPath) -> Double {
        var area = 0.0
        var current = SIMD2<Double>.zero, start = SIMD2<Double>.zero
        func cubic(_ p: [SIMD2<Double>]) {
            let nodes = [(-(0.6).squareRoot(), 5.0 / 9), (0, 8.0 / 9), ((0.6).squareRoot(), 5.0 / 9)]
            for (node, weight) in nodes {
                let t = (node + 1) / 2, s = 1 - t
                let point = s * s * s * p[0] + 3 * s * s * t * p[1] + 3 * s * t * t * p[2] + t * t * t * p[3]
                let derivative = 3 * s * s * (p[1] - p[0]) + 6 * s * t * (p[2] - p[1]) + 3 * t * t * (p[3] - p[2])
                area += weight / 2 * (point.x * derivative.y - point.y * derivative.x) / 2
            }
        }
        path.applyWithBlock { element in
            let points = element.pointee.points
            func at(_ index: Int) -> SIMD2<Double> { SIMD2(Double(points[index].x), Double(points[index].y)) }
            switch element.pointee.type {
            case .moveToPoint: current = at(0); start = current
            case .addLineToPoint:
                area += (current.x * at(0).y - at(0).x * current.y) / 2
                current = at(0)
            case .addCurveToPoint:
                cubic([current, at(0), at(1), at(2)])
                current = at(2)
            case .closeSubpath:
                area += (current.x * start.y - start.x * current.y) / 2
                current = start
            default: break
            }
        }
        return area
    }

    @Test("Curves flatten within the tolerance: chords lose at most perimeter × tolerance of the exact Bézier area")
    func flattenedCurves() throws {
        let path = CGMutablePath()
        path.addRect(CGRect(x: -2, y: -2, width: 4, height: 4))
        let circle = CGPath(ellipseIn: CGRect(x: -1, y: -1, width: 2, height: 2), transform: nil)
        path.addPath(circle)
        let exactHole = abs(exactArea(circle))
        // Core Graphics approximates the circle with four cubics that bulge by under 0.03%.
        #expect(abs(exactHole - .pi) < 2e-3)
        for tolerance: Float in [0.01, 0.0001] {
            let shapes = try PlanarShape.shapes(from: path, fillRule: .evenOdd, tolerance: tolerance)
            try #require(shapes.count == 1)
            #expect(shapes[0].holes.count == 1)
            let hole = 16 - shapes[0].area
            #expect(hole <= exactHole + 1e-6)
            #expect(hole >= exactHole - 2 * .pi * Double(tolerance))
            // Every flattened point lies on the circle, which Core Graphics draws with four cubic arcs.
            for point in shapes[0].holes[0] {
                #expect(abs(simd_length(SIMD2<Double>(point)) - 1) < 3e-4)
            }
        }
        #expect(throws: PlanarShape.Failure.invalidTolerance) {
            try PlanarShape.shapes(from: path, tolerance: 0)
        }
    }

    @Test("Explicit shapes validate orientation, crossings and hole placement")
    func explicitValidation() throws {
        let shape = try PlanarShape(outer: square(SIMD2(0, 0), 4).reversed(), holes: [square(SIMD2(1, 1), 1)])
        // The outer ring turns counterclockwise and the hole clockwise, whatever was supplied.
        func signedArea(_ ring: [SIMD2<Float>]) -> Float {
            ring.indices.reduce(0) { sum, index in
                let a = ring[index], b = ring[(index + 1) % ring.count]
                return sum + (a.x * b.y - b.x * a.y) / 2
            }
        }
        #expect(signedArea(shape.outer) == 16)
        #expect(signedArea(shape.holes[0]) == -1)
        #expect(shape.area == 15)
        #expect(throws: PlanarShape.Failure.holeOutsideOuterRing(0)) {
            try PlanarShape(outer: square(SIMD2(0, 0), 4), holes: [square(SIMD2(5, 5), 1)])
        }
        #expect(throws: PlanarShape.Failure.crossingRings) {
            try PlanarShape(outer: square(SIMD2(0, 0), 4), holes: [square(SIMD2(3, 3), 2)])
        }
        #expect(throws: PlanarShape.Failure.nestedHole(1)) {
            try PlanarShape(outer: square(SIMD2(0, 0), 4), holes: [square(SIMD2(0.5, 0.5), 3), square(SIMD2(1, 1), 1)])
        }
        #expect(throws: PlanarShape.Failure.degenerateRing(0)) {
            try PlanarShape(outer: [SIMD2(0, 0), SIMD2(1, 1), SIMD2(2, 2)])
        }
    }

    @Test("A flat shape mesh faces +Z, covers the shape's area and maps its bounds to UV 0...1")
    func flatShape() throws {
        let shape = try PlanarShape(outer: square(SIMD2(-1, -2), 4), holes: [square(SIMD2(0, -1), 1)])
        let mesh = try MeshGeometry.shape([shape])
        #expect(abs(ShapeCheck.surfaceArea(mesh) - 15) < 1e-12)
        #expect(mesh.vertices.allSatisfy { $0.normal == SIMD3(0, 0, 1) && $0.position.z == 0 })
        for vertex in mesh.vertices {
            let expected = SIMD2((vertex.position.x + 1) / 4, (2 - vertex.position.y) / 4)
            #expect(simd_distance(vertex.textureCoordinate, expected) < 1e-6)
        }
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
        let world = try MeshGeometry.shape([shape], textureMapping: .world(unitsPerTile: 0.5))
        #expect(world.vertices.allSatisfy { $0.textureCoordinate == SIMD2($0.position.x / 0.5, -$0.position.y / 0.5) })
    }

    @Test("A plain extrusion is a closed prism with exactly area × depth volume")
    func plainExtrusion() throws {
        let path = CGMutablePath()
        path.addRoundedRect(in: CGRect(x: -1, y: -0.5, width: 2, height: 1), cornerWidth: 0.3, cornerHeight: 0.3)
        path.addEllipse(in: CGRect(x: -0.6, y: -0.2, width: 0.4, height: 0.4))
        let shapes = try PlanarShape.shapes(from: path, fillRule: .evenOdd, tolerance: 0.001)
        let mesh = try MeshGeometry.extrude(shapes, ShapeExtrusion(depth: 0.25))
        #expect(ShapeCheck.isClosed(mesh))
        let area = shapes.reduce(0) { $0 + $1.area }
        #expect(abs(mesh.signedVolume - area * 0.25) < 1e-6 * area)
        let bounds = ShapeCheck.bounds(mesh.vertices.map(\.position))
        #expect(bounds.lower == SIMD3(-1, -0.5, -0.125) && bounds.upper == SIMD3(1, 0.5, 0.125))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
        // Wall normals are horizontal; caps face ±Z.
        #expect(mesh.vertices.allSatisfy { abs($0.normal.z) == 1 || abs($0.normal.z) < 1e-6 })
    }

    @Test("A beveled square prism matches the prismatoid volume of its bevel profile")
    func beveledVolume() throws {
        let side = 2.0, depth = 1.0, size = 0.2, bevelDepth = 0.15
        let shape = try PlanarShape(outer: square(SIMD2(-1, -1), Float(side)))
        for segments in [1, 3, 8] {
            let extrusion = ShapeExtrusion(
                depth: Float(depth), bevelSize: Float(size), bevelDepth: Float(bevelDepth), bevelSegments: segments
            )
            let mesh = try MeshGeometry.extrude([shape], extrusion)
            #expect(ShapeCheck.isClosed(mesh))
            #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
            // Independent reference: each bevel slab is a frustum between squares of side L − 2d.
            var bevelVolume = 0.0
            for step in 0..<segments {
                let phi0 = Double.pi / 2 * Double(step) / Double(segments)
                let phi1 = Double.pi / 2 * Double(step + 1) / Double(segments)
                let a = side - 2 * size * (1 - sin(phi0)), b = side - 2 * size * (1 - sin(phi1))
                let height = bevelDepth * (cos(phi0) - cos(phi1))
                bevelVolume += height * (a * a + a * b + b * b) / 3
            }
            let expected = side * side * (depth - 2 * bevelDepth) + 2 * bevelVolume
            #expect(abs(mesh.signedVolume - expected) < 1e-5)
            // The walls stay on the outline; the caps are inset by the bevel size.
            let bounds = ShapeCheck.bounds(mesh.vertices.map(\.position))
            #expect(bounds.lower == SIMD3(-1, -1, -0.5) && bounds.upper == SIMD3(1, 1, 0.5))
            let front = mesh.vertices.filter { $0.normal == SIMD3(0, 0, 1) && $0.position.z == 0.5 }
            #expect(front.allSatisfy { abs(abs($0.position.x) - 0.8) < 1e-6 && abs(abs($0.position.y) - 0.8) < 1e-6 })
        }
    }

    @Test("Bevel normals follow the rounded profile and walls of curved outlines are smooth")
    func bevelNormals() throws {
        let path = CGPath(ellipseIn: CGRect(x: -1, y: -1, width: 2, height: 2), transform: nil)
        let shapes = try PlanarShape.shapes(from: path, tolerance: 0.0005)
        let size: Float = 0.1, bevelDepth: Float = 0.05
        let mesh = try MeshGeometry.extrude(
            shapes, ShapeExtrusion(depth: 0.4, bevelSize: size, bevelDepth: bevelDepth, bevelSegments: 6)
        )
        #expect(ShapeCheck.isClosed(mesh))
        for vertex in mesh.vertices where abs(vertex.normal.z) < 0.999 {
            let p = vertex.position
            let radial = simd_normalize(SIMD2(p.x, p.y))
            // The normal's planar part is radial, so no crease splits the smooth outline.
            let planar = SIMD2(vertex.normal.x, vertex.normal.y)
            #expect(simd_length(planar - radial * simd_length(planar)) < 2e-3)
            // On the quarter ellipse (r, z) = (1 − s(1 − sin φ), zWall + t cos φ) the normal is ∝ (t sin φ, s cos φ).
            let z = abs(p.z) - (0.2 - bevelDepth)
            if z > 1e-6 {
                let cosine = min(Double(z / bevelDepth), 1), sine = (1 - cosine * cosine).squareRoot()
                let expected = simd_normalize(SIMD2(Double(bevelDepth) * sine, Double(size) * cosine))
                #expect(abs(Double(simd_length(planar)) - expected.x) < 2e-3)
                #expect(abs(Double(abs(vertex.normal.z)) - expected.y) < 2e-3)
            }
        }
    }

    @Test("Bevel caps are exact offsets at right angles, and sharp notches use a limited miter instead of a spike")
    func bevelOffsets() throws {
        let extrusion = ShapeExtrusion(depth: 1, bevelSize: 0.1, bevelSegments: 3)
        func frontCap(_ mesh: MeshGeometry) -> Set<SIMD2<Float>> {
            Set(mesh.vertices.filter { $0.normal == SIMD3(0, 0, 1) }.map { SIMD2($0.position.x, $0.position.y) })
        }
        // An L shape inset by 0.1, including its reflex corner at (1, 1).
        let ell = try PlanarShape(outer: [SIMD2(0, 0), SIMD2(2, 0), SIMD2(2, 1), SIMD2(1, 1), SIMD2(1, 2), SIMD2(0, 2)])
        let ellMesh = try MeshGeometry.extrude([ell], extrusion)
        #expect(ShapeCheck.isClosed(ellMesh))
        let expected: [SIMD2<Float>] = [SIMD2(0.1, 0.1), SIMD2(1.9, 0.1), SIMD2(1.9, 0.9), SIMD2(0.9, 0.9), SIMD2(0.9, 1.9), SIMD2(0.1, 1.9)]
        let cap = frontCap(ellMesh)
        #expect(cap.count == 6)
        for point in expected { #expect(cap.contains { simd_distance($0, point) < 1e-6 }) }

        // A 20° notch from the top of a 4 × 1 bar down to y = 0.4. Its exact miter is
        // 1 / sin 10° ≈ 5.76 times the inset and would cut through the 0.4 of material below
        // the notch; the true offset only rounds the notch. Limited to twice the inset,
        // the apex moves from y = 0.4 to y = 0.2 and the bar stays one solid.
        let halfWidth = 0.6 * tan(Float.pi / 18)
        let notched = try PlanarShape(outer: [
            SIMD2(0, 0), SIMD2(4, 0), SIMD2(4, 1), SIMD2(2 + halfWidth, 1), SIMD2(2, 0.4), SIMD2(2 - halfWidth, 1), SIMD2(0, 1),
        ])
        let notchMesh = try MeshGeometry.extrude([notched], extrusion)
        #expect(ShapeCheck.isClosed(notchMesh))
        #expect(frontCap(notchMesh).contains { simd_distance($0, SIMD2(2, 0.2)) < 1e-6 })
        #expect(notchMesh.signedVolume < notched.area * 1 && notchMesh.signedVolume > notched.area * 0.9)
    }

    @Test("World texture mapping keeps one texel density on walls and caps")
    func worldWallMapping() throws {
        let shape = try PlanarShape(outer: square(SIMD2(0, 0), 3))
        let mesh = try MeshGeometry.extrude([shape], ShapeExtrusion(depth: 1, textureMapping: .world(unitsPerTile: 0.5)))
        // Each wall's U spans its length over 0.5 units; V spans the depth.
        let wallUs = mesh.vertices.filter { $0.normal.z == 0 }.map(\.textureCoordinate.x)
        #expect(wallUs.min() == 0 && wallUs.max() == 24)
        #expect(mesh.vertices.filter { $0.normal.z == 0 }.allSatisfy { [0, 2].contains($0.textureCoordinate.y) })
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
    }

    @Test("Bevels that would collapse a feature, and invalid extrusion settings, fail with specific errors")
    func extrusionValidation() throws {
        let thin = try PlanarShape(outer: [SIMD2(0, 0), SIMD2(4, 0), SIMD2(4, 0.2), SIMD2(0, 0.2)])
        #expect(throws: MeshGeometry.ShapeMeshError.bevelTooLarge(0)) {
            try MeshGeometry.extrude([thin], ShapeExtrusion(depth: 1, bevelSize: 0.15))
        }
        #expect(throws: Never.self) {
            try MeshGeometry.extrude([thin], ShapeExtrusion(depth: 1, bevelSize: 0.05))
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidBevel) {
            try MeshGeometry.extrude([thin], ShapeExtrusion(depth: 0.2, bevelSize: 0.05, bevelDepth: 0.2))
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidDepth) {
            try MeshGeometry.extrude([thin], ShapeExtrusion(depth: 0))
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidBevelSegments) {
            try MeshGeometry.extrude([thin], ShapeExtrusion(depth: 1, bevelSize: 0.01, bevelSegments: 0))
        }
        #expect(throws: MeshGeometry.ShapeMeshError.noShapes) {
            try MeshGeometry.extrude([], ShapeExtrusion(depth: 1))
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidTextureScale) {
            try MeshGeometry.shape([thin], textureMapping: .world(unitsPerTile: -1))
        }
    }
}
