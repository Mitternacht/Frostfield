import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Three.js primitive generators against closed-form counts, areas, volumes and frames.
struct PrimitiveGeometryTests {
    // MARK: Measures

    static func area(_ geometry: MeshGeometry) -> Double {
        triangles(geometry).reduce(0) { $0 + simd_length(simd_cross($1.1 - $1.0, $1.2 - $1.0)) / 2 }
    }

    /// The divergence-theorem volume; positive when every face winds counterclockwise from outside.
    static func volume(_ geometry: MeshGeometry) -> Double {
        triangles(geometry).reduce(0) { $0 + simd_dot($1.0, simd_cross($1.1, $1.2)) / 6 }
    }

    static func triangles(_ geometry: MeshGeometry) -> [(SIMD3<Double>, SIMD3<Double>, SIMD3<Double>)] {
        stride(from: 0, to: geometry.indices.count, by: 3).map { start in
            let p = (0..<3).map { SIMD3<Double>(geometry.vertices[Int(geometry.indices[start + $0])].position) }
            return (p[0], p[1], p[2])
        }
    }

    /// After welding equal positions, every undirected edge belongs to exactly two
    /// triangles that traverse it in opposite directions.
    static func isClosedAndConsistent(_ geometry: MeshGeometry, weld: Double = 1e-5) -> Bool {
        var ids: [SIMD3<Int64>: Int] = [:]
        func id(_ index: UInt32) -> Int {
            let p = SIMD3<Double>(geometry.vertices[Int(index)].position) / weld
            let key = SIMD3(Int64(p.x.rounded()), Int64(p.y.rounded()), Int64(p.z.rounded()))
            if let existing = ids[key] { return existing }
            ids[key] = ids.count
            return ids.count - 1
        }
        var directed: [SIMD2<Int>: Int] = [:]
        for start in stride(from: 0, to: geometry.indices.count, by: 3) {
            let v = (0..<3).map { id(geometry.indices[start + $0]) }
            guard Set(v).count == 3 else { return false }
            for k in 0..<3 { directed[SIMD2(v[k], v[(k + 1) % 3]), default: 0] += 1 }
        }
        return directed.allSatisfy { edge, count in count == 1 && directed[SIMD2(edge.y, edge.x)] == 1 }
    }

    /// Every vertex tangent is a unit vector perpendicular to its normal, points along
    /// increasing U of its triangles, and its sign makes cross(n, t) × w follow increasing V.
    static func tangentFailures(_ geometry: MeshGeometry) -> Int {
        var failures = 0
        for start in stride(from: 0, to: geometry.indices.count, by: 3) {
            let index = (0..<3).map { Int(geometry.indices[start + $0]) }
            let v = index.map { geometry.vertices[$0] }
            let e1 = SIMD3<Double>(v[1].position - v[0].position)
            let e2 = SIMD3<Double>(v[2].position - v[0].position)
            let d1 = SIMD2<Double>(v[1].textureCoordinate - v[0].textureCoordinate)
            let d2 = SIMD2<Double>(v[2].textureCoordinate - v[0].textureCoordinate)
            let determinant = d1.x * d2.y - d1.y * d2.x
            guard abs(determinant) > 1e-10 else { continue }
            let dPdu = (e1 * d2.y - e2 * d1.y) / determinant
            let dPdv = (e2 * d1.x - e1 * d2.x) / determinant
            for vertex in v {
                guard let tangent = vertex.tangent else { failures += 1; continue }
                let n = SIMD3<Double>(vertex.normal)
                let t = SIMD3<Double>(Double(tangent.x), Double(tangent.y), Double(tangent.z))
                if abs(simd_length(t) - 1) > 1e-4 || abs(simd_dot(t, n)) > 1e-3 { failures += 1; continue }
                let u = dPdu - n * simd_dot(n, dPdu)
                if simd_length(u) > 0.3 * simd_length(dPdu), simd_dot(t, u) <= 0 { failures += 1 }
                let side = simd_dot(simd_cross(n, t), dPdv)
                if abs(side) > 0.3 * simd_length(dPdv), (side > 0 ? 1 : -1) != Double(tangent.w) { failures += 1 }
            }
        }
        return failures
    }

    static func relative(_ value: Double, _ expected: Double) -> Double { abs(value - expected) / abs(expected) }

    /// The first difference between a Frostfield primitive and its Three.js r187 port, or nil.
    /// Vertex counts, positions, normals and U must agree (U after `uOffset`), V must be
    /// flipped, groups must match, and the indices must equal the reference's once its
    /// zero-area triangles are removed.
    static func mismatch(
        _ geometry: MeshGeometry, _ reference: ThreeGeometryReference.Mesh, uOffset: (Int) -> Double = { _ in 0 }
    ) -> String? {
        guard geometry.vertices.count == reference.positions.count else {
            return "vertex count \(geometry.vertices.count) vs \(reference.positions.count)"
        }
        for (index, vertex) in geometry.vertices.enumerated() {
            if simd_distance(SIMD3<Double>(vertex.position), reference.positions[index]) > 1e-5 {
                return "position \(index): \(vertex.position) vs \(reference.positions[index])"
            }
            if simd_distance(SIMD3<Double>(vertex.normal), reference.normals[index]) > 1e-5 {
                return "normal \(index): \(vertex.normal) vs \(reference.normals[index])"
            }
            let expected = SIMD2(reference.uvs[index].x + uOffset(index), 1 - reference.uvs[index].y)
            if simd_distance(SIMD2<Double>(vertex.textureCoordinate), expected) > 1e-6 {
                return "uv \(index): \(vertex.textureCoordinate) vs \(expected)"
            }
        }
        var kept: [UInt32] = []
        for start in stride(from: 0, to: reference.indices.count, by: 3) {
            let corners = reference.indices[start..<start + 3]
            let p = corners.map { reference.positions[$0] }
            if simd_length(simd_cross(p[1] - p[0], p[2] - p[0])) > 1e-9 { kept += corners.map { UInt32($0) } }
        }
        if kept != geometry.indices {
            let first = zip(kept, geometry.indices).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            return "indices differ at \(first.map(String.init) ?? "the end"): \(kept.count) vs \(geometry.indices.count)"
        }
        // Group ranges shift when zero-area triangles before them are removed, so groups
        // are compared only for references without such triangles.
        if kept.count == reference.indices.count, !reference.groups.isEmpty {
            let groups = geometry.groups.map { [$0.start, $0.count, $0.materialIndex] }
            let expected = reference.groups.map { [$0.start, $0.count, $0.materialIndex] }
            if groups != expected { return "groups \(groups) vs \(expected)" }
        }
        return nil
    }

    @Test("Box, cylinder, cone, capsule, torus and ring match the Three.js r187 generators vertex by vertex and index by index")
    func threeParity() throws {
        #expect(Self.mismatch(
            try .box(width: 2, height: 3, depth: 4, widthSegments: 2, heightSegments: 3, depthSegments: 4),
            ThreeGeometryReference.box(width: 2, height: 3, depth: 4, widthSegments: 2, heightSegments: 3, depthSegments: 4)
        ) == nil)
        for (top, bottom, open, start, length) in [
            (1.0, 2.0, false, 0.0, 2 * Double.pi), (0.0, 1.5, false, 0.3, 4.0), (1.0, 0.0, true, 0.0, 2 * Double.pi),
        ] {
            let geometry = try MeshGeometry.cylinder(
                radiusTop: Float(top), radiusBottom: Float(bottom), height: 3, radialSegments: 7, heightSegments: 3,
                openEnded: open, thetaStart: Float(start), thetaLength: Float(length)
            )
            let reference = ThreeGeometryReference.cylinder(
                radiusTop: top, radiusBottom: bottom, height: 3, radialSegments: 7, heightSegments: 3,
                openEnded: open, thetaStart: Double(Float(start)), thetaLength: Double(Float(length))
            )
            let failure = Self.mismatch(geometry, reference)
            #expect(failure == nil, "cylinder \(top), \(bottom): \(failure ?? "")")
        }
        #expect(Self.mismatch(
            try .cone(radius: 1, height: 2, radialSegments: 5, heightSegments: 2),
            ThreeGeometryReference.cylinder(
                radiusTop: 0, radiusBottom: 1, height: 2, radialSegments: 5, heightSegments: 2, openEnded: false,
                thetaStart: 0, thetaLength: Double(Float(2 * Double.pi))
            )
        ) == nil)

        for (height, caps, radial, rows) in [(1.0, 4, 8, 1), (0.6, 3, 5, 3), (0.0, 2, 6, 2)] {
            let geometry = try MeshGeometry.capsule(
                radius: 0.5, height: Float(height), capSegments: caps, radialSegments: radial, heightSegments: rows
            )
            let reference = ThreeGeometryReference.capsule(
                radius: 0.5, height: height, capSegments: caps, radialSegments: radial, heightSegments: rows
            )
            // Pole vertices sit at the center of the wedge they close; Three.js puts them one wedge off.
            let rowLength = radial + 1
            let lastRow = (2 * caps + rows) * rowLength
            let failure = Self.mismatch(geometry, reference) { index in
                index < rowLength ? -1 / Double(radial) : index >= lastRow ? 1 / Double(radial) : 0
            }
            #expect(failure == nil, "capsule height \(height): \(failure ?? "")")
            // Three.js counts, less 2 × radialSegments pole triangles and the flat band at zero height.
            #expect(geometry.vertices.count == (radial + 1) * (2 * caps + rows + 1))
            let dropped = 2 * radial + (height == 0 ? 2 * radial * rows : 0)
            #expect(geometry.indices.count / 3 == 2 * radial * (2 * caps + rows) - dropped)
        }

        let torusArguments = (radius: 1.0, tube: 0.3, radial: 6, tubular: 9, arc: 5.0, start: -1.0, length: 4.0)
        #expect(Self.mismatch(
            try .torus(
                radius: 1, tube: 0.3, radialSegments: 6, tubularSegments: 9, arc: 5, thetaStart: -1, thetaLength: 4
            ),
            ThreeGeometryReference.torus(
                radius: torusArguments.radius, tube: torusArguments.tube, radialSegments: torusArguments.radial,
                tubularSegments: torusArguments.tubular, arc: torusArguments.arc, thetaStart: torusArguments.start,
                thetaLength: torusArguments.length
            )
        ) == nil)

        for inner in [0.5, 0.0] {
            let failure = Self.mismatch(
                try .ring(innerRadius: Float(inner), outerRadius: 2, thetaSegments: 6, phiSegments: 3, thetaStart: 0.5, thetaLength: 3),
                ThreeGeometryReference.ring(
                    innerRadius: inner, outerRadius: 2, thetaSegments: 6, phiSegments: 3, thetaStart: 0.5, thetaLength: 3
                )
            )
            #expect(failure == nil, "ring \(inner): \(failure ?? "")")
        }
    }

    // MARK: Box

    @Test("A segmented box has Three.js counts, six face groups, exact area and volume, and outward faces")
    func box() throws {
        let box = try MeshGeometry.box(width: 2, height: 3, depth: 4, widthSegments: 2, heightSegments: 3, depthSegments: 4)
        #expect(box.vertices.count == 2 * ((2 + 1) * (3 + 1) + (4 + 1) * (3 + 1) + (2 + 1) * (4 + 1)))
        #expect(box.indices.count / 3 == 4 * (2 * 3 + 4 * 3 + 2 * 4))
        #expect(abs(Self.area(box) - 52) < 1e-4)
        #expect(abs(Self.volume(box) - 24) < 1e-4)
        #expect(Self.isClosedAndConsistent(box))
        #expect(Self.tangentFailures(box) == 0)
        #expect(box.groups.map(\.materialIndex) == Array(0..<6))
        // ±X faces are depth × height cells, ±Y width × depth, ±Z width × height; six indices per cell.
        #expect(box.groups.map(\.count) == [4 * 3, 4 * 3, 2 * 4, 2 * 4, 2 * 3, 2 * 3].map { $0 * 6 })
        #expect(box.groups.reduce(0) { $0 + $1.count } == box.indices.count)
        // Each face's normal is its axis, and its positions lie on that face's plane.
        let axes: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)]
        let half = SIMD3<Float>(1, 1.5, 2)
        for (group, axis) in zip(box.groups, axes) {
            for index in box.indices[group.range] {
                let vertex = box.vertices[Int(index)]
                #expect(vertex.normal == axis)
                #expect(abs(simd_dot(vertex.position, axis) - simd_dot(abs(axis), half)) < 1e-6)
            }
        }
    }

    // MARK: Sphere

    @Test("A UV sphere has Three.js counts, radial normals, and area within 0.5% of 4πr²")
    func sphere() throws {
        let radius: Float = 1.5
        let sphere = try MeshGeometry.sphere(radius: radius, widthSegments: 64, heightSegments: 32)
        #expect(sphere.vertices.count == 65 * 33)
        #expect(sphere.indices.count / 3 == 2 * 64 * 32 - 2 * 64)
        let r = Double(radius)
        #expect(Self.relative(Self.area(sphere), 4 * .pi * r * r) < 0.005)
        #expect(Self.relative(Self.volume(sphere), 4 / 3 * .pi * r * r * r) < 0.01)
        #expect(Self.isClosedAndConsistent(sphere))
        #expect(Self.tangentFailures(sphere) == 0)
        for vertex in sphere.vertices {
            #expect(simd_length(vertex.normal - vertex.position / radius) < 1e-5)
        }
        // Three.js orientation: phi = 0 starts at −X, and the first ring is the +Y pole.
        #expect(simd_length(sphere.vertices[65 * 16].position - SIMD3(-radius, 0, 0)) < 1e-5)
        #expect(sphere.vertices[0].position.y == radius)
        // Pole vertices sit at their wedge centers; V runs from 0 at the north pole to 1 at the south.
        #expect(abs(sphere.vertices[0].textureCoordinate.x - 0.5 / 64) < 1e-6)
        #expect(sphere.vertices[0].textureCoordinate.y == 0)
        #expect(sphere.vertices[65 * 33 - 1].textureCoordinate.y == 1)

        // A hemisphere patch covers half the area and stays open.
        let hemisphere = try MeshGeometry.sphere(radius: 1, widthSegments: 128, heightSegments: 32, thetaLength: .pi / 2)
        #expect(Self.relative(Self.area(hemisphere), 2 * .pi) < 0.005)
        #expect(hemisphere.vertices.allSatisfy { $0.position.y >= -1e-6 })
        #expect(!Self.isClosedAndConsistent(hemisphere))
        // A quarter turn of longitude starting at phi = π/2 spans +Z toward +X.
        let wedge = try MeshGeometry.sphere(widthSegments: 32, heightSegments: 16, phiStart: .pi / 2, phiLength: .pi / 2)
        #expect(wedge.vertices.allSatisfy { $0.position.x >= -1e-6 && $0.position.z >= -1e-6 })
        #expect(Self.relative(Self.area(wedge), .pi) < 0.01)
    }

    // MARK: Cylinder and cone

    @Test("Cylinders and cones match frustum areas and volumes, with caps as groups 1 and 2")
    func cylinder() throws {
        let cylinder = try MeshGeometry.cylinder(
            radiusTop: 1, radiusBottom: 2, height: 3, radialSegments: 32, heightSegments: 2
        )
        #expect(cylinder.vertices.count == 33 * 3 + 2 * (32 + 33))
        #expect(cylinder.indices.count / 3 == 2 * 32 * 2 + 2 * 32)
        #expect(cylinder.groups.map(\.materialIndex) == [0, 1, 2])
        #expect(Self.isClosedAndConsistent(cylinder))
        #expect(Self.tangentFailures(cylinder) == 0)

        let fine = try MeshGeometry.cylinder(radiusTop: 1, radiusBottom: 2, height: 3, radialSegments: 512)
        let slant = (1.0 + 9.0).squareRoot()
        let lateral = Double.pi * (1 + 2) * slant
        #expect(Self.relative(Self.area(fine), lateral + .pi * (1 + 4)) < 0.001)
        #expect(Self.relative(Self.volume(fine), .pi * 3 / 3 * (1 + 2 + 4)) < 0.001)
        // Side normals are perpendicular to the slanted generator and point outward.
        let side = try fine.subset(fine.groups[0].range)
        for vertex in side.vertices {
            let radial = simd_normalize(SIMD2(vertex.position.x, vertex.position.z))
            let generator = simd_normalize(SIMD3(radial.x, -3, radial.y))
            #expect(abs(simd_dot(vertex.normal, generator)) < 1e-5)
            #expect(simd_dot(SIMD2(vertex.normal.x, vertex.normal.z), radial) > 0)
        }

        let cone = try MeshGeometry.cone(radius: 1, height: 2, radialSegments: 512, heightSegments: 3)
        #expect(cone.groups.map(\.materialIndex) == [0, 2])
        #expect(Self.isClosedAndConsistent(cone))
        #expect(Self.relative(Self.volume(cone), .pi * 2 / 3) < 0.001)
        #expect(Self.relative(Self.area(cone), .pi * 1 * 5.0.squareRoot() + .pi) < 0.001)

        let open = try MeshGeometry.cylinder(radialSegments: 16, openEnded: true)
        #expect(open.vertices.count == 17 * 2)
        #expect(open.groups.count == 1)
        #expect(!Self.isClosedAndConsistent(open))
        // Theta from 0 to π sweeps from +Z through +X, so every vertex has x ≥ 0.
        let half = try MeshGeometry.cylinder(radialSegments: 64, thetaLength: .pi)
        #expect(half.vertices.allSatisfy { $0.position.x >= -1e-6 })
        #expect(Self.relative(Self.area(try half.subset(half.groups[0].range)), .pi) < 0.001)
        #expect(throws: MeshGeometry.PrimitiveError.size) { try MeshGeometry.cylinder(radiusTop: 0, radiusBottom: 0) }
        #expect(throws: MeshGeometry.PrimitiveError.angle) { try MeshGeometry.cylinder(thetaLength: 7) }
    }

    // MARK: Capsule

    @Test("A capsule matches 4πr² + 2πrh and its volume, closed with radial normals on the caps")
    func capsule() throws {
        let capsule = try MeshGeometry.capsule(radius: 0.5, height: 1, capSegments: 32, radialSegments: 256, heightSegments: 2)
        let r = 0.5
        #expect(Self.relative(Self.area(capsule), 4 * .pi * r * r + 2 * .pi * r * 1) < 0.002)
        #expect(Self.relative(Self.volume(capsule), 4 / 3 * .pi * r * r * r + .pi * r * r * 1) < 0.002)
        #expect(Self.isClosedAndConsistent(capsule))
        #expect(Self.tangentFailures(capsule) == 0)
        let ys = capsule.vertices.map(\.position.y)
        #expect(abs(ys.max()! - 1) < 1e-6 && abs(ys.min()! + 1) < 1e-6)
        for vertex in capsule.vertices {
            let center = SIMD3<Float>(0, min(max(vertex.position.y, -0.5), 0.5), 0)
            #expect(simd_length(vertex.normal - (vertex.position - center) / 0.5) < 1e-4)
        }
        let sphere = try MeshGeometry.capsule(radius: 1, height: 0, capSegments: 16, radialSegments: 64)
        #expect(Self.isClosedAndConsistent(sphere))
    }

    // MARK: Torus and torus knot

    @Test("A torus matches 4π²Rr and 2π²Rr² with normals from the tube center")
    func torus() throws {
        let torus = try MeshGeometry.torus(radius: 1, tube: 0.25, radialSegments: 64, tubularSegments: 256)
        #expect(torus.vertices.count == 65 * 257)
        #expect(torus.indices.count / 3 == 2 * 64 * 256)
        #expect(Self.relative(Self.area(torus), 4 * .pi * .pi * 0.25) < 0.002)
        #expect(Self.relative(Self.volume(torus), 2 * .pi * .pi * 0.0625) < 0.002)
        #expect(Self.isClosedAndConsistent(torus))
        #expect(Self.tangentFailures(torus) == 0)
        for vertex in torus.vertices {
            let ring = simd_normalize(SIMD3(vertex.position.x, vertex.position.y, 0))
            #expect(simd_length(vertex.normal - (vertex.position - ring) / 0.25) < 1e-4)
        }
        let arc = try MeshGeometry.torus(radius: 1, tube: 0.25, radialSegments: 64, tubularSegments: 256, arc: .pi)
        #expect(Self.relative(Self.area(arc), 2 * .pi * .pi * 0.25) < 0.002)
        // The outer half of the tube, from −π/2 to π/2: area = 2π r (R π + 2 r).
        let outer = try MeshGeometry.torus(
            radius: 1, tube: 0.25, radialSegments: 64, tubularSegments: 256, thetaStart: -.pi / 2, thetaLength: .pi
        )
        #expect(Self.relative(Self.area(outer), 2 * .pi * 0.25 * (.pi + 0.5)) < 0.002)
        #expect(outer.vertices.allSatisfy { simd_length(SIMD2($0.position.x, $0.position.y)) >= 1 - 1e-5 })
        #expect(Self.tangentFailures(outer) == 0)
        #expect(throws: MeshGeometry.PrimitiveError.angle) { try MeshGeometry.torus(thetaLength: 0) }
    }

    @Test("A (2, 3) torus knot closes exactly and encloses π r² times its curve length")
    func torusKnot() throws {
        let tubular = 512
        let radial = 32
        let knot = try MeshGeometry.torusKnot(radius: 1, tube: 0.1, tubularSegments: tubular, radialSegments: radial, p: 2, q: 3)
        #expect(knot.vertices.count == (tubular + 1) * (radial + 1))
        #expect(knot.indices.count / 3 == 2 * tubular * radial)
        // The last ring repeats the first, so the seam has no gap.
        for j in 0...radial {
            #expect(knot.vertices[j].position == knot.vertices[tubular * (radial + 1) + j].position)
            #expect(knot.vertices[j].normal == knot.vertices[tubular * (radial + 1) + j].normal)
        }
        #expect(Self.isClosedAndConsistent(knot))
        #expect(Self.tangentFailures(knot) == 0)
        // Curve length by fine numerical integration of the Three.js curve.
        func curve(_ u: Double) -> SIMD3<Double> {
            let c = cos(1.5 * u)
            return SIMD3((2 + c) * 0.5 * cos(u), (2 + c) * 0.5 * sin(u), sin(1.5 * u) * 0.5)
        }
        let steps = 200_000
        var length = 0.0
        for i in 0..<steps {
            length += simd_distance(curve(4 * .pi * Double(i) / Double(steps)), curve(4 * .pi * Double(i + 1) / Double(steps)))
        }
        // A 32-gon tube encloses (n/2π) sin(2π/n) of the circular cross-section.
        let polygon = Double(radial) / (2 * .pi) * sin(2 * .pi / Double(radial))
        #expect(Self.relative(Self.volume(knot), .pi * 0.01 * length * polygon) < 0.005)
        #expect(throws: MeshGeometry.PrimitiveError.knotWinding) { try MeshGeometry.torusKnot(p: 2, q: 4) }
    }

    // MARK: Flat shapes

    @Test("Rings and circles match polygon areas exactly and face +Z")
    func flatShapes() throws {
        let ring = try MeshGeometry.ring(innerRadius: 0.5, outerRadius: 1, thetaSegments: 64, phiSegments: 3)
        #expect(ring.vertices.count == 65 * 4)
        #expect(ring.indices.count / 3 == 2 * 64 * 3)
        let polygon = 32 * sin(2 * Double.pi / 64)
        #expect(abs(Self.area(ring) - polygon * (1 - 0.25)) < 1e-5)
        #expect(Self.tangentFailures(ring) == 0)
        #expect(Self.triangles(ring).allSatisfy { simd_cross($0.1 - $0.0, $0.2 - $0.0).z > 0 })
        // UVs map the outer square: +X is U = 1 and +Y is the top, V = 0.
        let east = try #require(ring.vertices.first { $0.position == SIMD3(1, 0, 0) })
        #expect(east.textureCoordinate == SIMD2(1, 0.5))

        let sector = try MeshGeometry.ring(innerRadius: 0, outerRadius: 2, thetaSegments: 16, phiSegments: 2, thetaLength: .pi / 2)
        #expect(abs(Self.area(sector) - 8 * sin(Double.pi / 32) * 4) < 1e-5)
        #expect(Self.triangles(sector).allSatisfy { simd_length(simd_cross($0.1 - $0.0, $0.2 - $0.0)) > 0 })

        let circle = try MeshGeometry.circle(radius: 2, segments: 48)
        #expect(circle.vertices.count == 50)
        #expect(circle.indices.count / 3 == 48)
        #expect(abs(Self.area(circle) - 24 * 4 * sin(2 * .pi / 48)) < 1e-5)
        #expect(Self.triangles(circle).allSatisfy { simd_cross($0.1 - $0.0, $0.2 - $0.0).z > 0 })
        let pie = try MeshGeometry.circle(radius: 1, segments: 30, thetaStart: .pi / 2, thetaLength: .pi)
        #expect(pie.vertices.allSatisfy { $0.position.x <= 1e-6 })
    }

    // MARK: Polyhedra

    @Test("Platonic polyhedra match closed-form areas, flat normals at detail 0, and smooth normals above")
    func polyhedra() throws {
        let root5 = 5.0.squareRoot()
        let cases: [(name: String, make: (Int) throws -> MeshGeometry, faces: Int, area: Double)] = [
            ("tetrahedron", { try .tetrahedron(radius: 1, detail: $0) }, 4, 3.0.squareRoot() * 8 / 3),
            ("octahedron", { try .octahedron(radius: 1, detail: $0) }, 8, 2 * 3.0.squareRoot() * 2),
            ("icosahedron", { try .icosahedron(radius: 1, detail: $0) }, 20,
             5 * 3.0.squareRoot() * pow(4 / (10 + 2 * root5).squareRoot(), 2)),
            ("dodecahedron", { try .dodecahedron(radius: 1, detail: $0) }, 36,
             3 * (25 + 10 * root5).squareRoot() * pow(4 / (3.0.squareRoot() * (1 + root5)), 2)),
        ]
        for item in cases {
            let flat = try item.make(0)
            #expect(flat.indices.count == item.faces * 3, "\(item.name)")
            #expect(flat.vertices.count == item.faces * 3, "\(item.name)")
            #expect(Self.relative(Self.area(flat), item.area) < 1e-5, "\(item.name) area \(Self.area(flat)) vs \(item.area)")
            #expect(Self.volume(flat) > 0, "\(item.name)")
            #expect(Self.isClosedAndConsistent(flat), "\(item.name)")
            #expect(Self.tangentFailures(flat) == 0, "\(item.name)")
            for (a, b, c) in Self.triangles(flat) {
                let normal = simd_normalize(simd_cross(b - a, c - a))
                #expect(simd_dot(normal, (a + b + c) / 3) > 0, "\(item.name)")
            }
            for start in stride(from: 0, to: flat.vertices.count, by: 3) {
                let (a, b, c) = (flat.vertices[start].position, flat.vertices[start + 1].position, flat.vertices[start + 2].position)
                let face = simd_normalize(simd_cross(b - a, c - a))
                #expect(simd_length(flat.vertices[start].normal - face) < 1e-5, "\(item.name)")
            }
            let smooth = try item.make(3)
            #expect(smooth.indices.count == item.faces * 16 * 3, "\(item.name)")
            #expect(smooth.vertices.allSatisfy { simd_length($0.normal - $0.position) < 1e-5 }, "\(item.name)")
            #expect(Self.isClosedAndConsistent(smooth), "\(item.name)")
            #expect(Self.tangentFailures(smooth) == 0, "\(item.name)")
        }
        // A detailed icosahedron is a geodesic sphere.
        let geodesic = try MeshGeometry.icosahedron(radius: 2, detail: 12)
        #expect(Self.relative(Self.area(geodesic), 16 * .pi) < 0.005)
        #expect(Self.relative(Self.volume(geodesic), 4 / 3 * .pi * 8) < 0.01)
        // Seam correction keeps every face's UV winding the same as its surface winding.
        var flipped = 0
        for start in stride(from: 0, to: geodesic.vertices.count, by: 3) {
            let t = (0..<3).map { SIMD2<Double>(geodesic.vertices[start + $0].textureCoordinate) }
            let signed = (t[1] - t[0]).x * (t[2] - t[0]).y - (t[1] - t[0]).y * (t[2] - t[0]).x
            if signed > 0 { flipped += 1 }
        }
        #expect(flipped == 0)
        #expect(Self.tangentFailures(geodesic) == 0)
    }

    // MARK: Groups

    @Test("Groups validate their ranges and subsets keep only the vertices they use")
    func groups() throws {
        let plane = try MeshGeometry.plane(width: 2, height: 1, columns: 2, rows: 1)
        let grouped = try plane.withGroups([
            MeshGeometryGroup(start: 6, count: 6, materialIndex: 1), MeshGeometryGroup(start: 0, count: 6, materialIndex: 0),
        ])
        #expect(grouped.groups.map(\.start) == [0, 6])
        let right = try grouped.subset(grouped.groups[1].range)
        #expect(right.vertices.count == 4)
        #expect(right.indices.count == 6)
        #expect(right.vertices.allSatisfy { $0.position.x >= 0 })
        #expect(throws: MeshGeometry.ValidationError.invalidGroup(1)) {
            try plane.withGroups([MeshGeometryGroup(start: 0, count: 6), MeshGeometryGroup(start: 3, count: 6)])
        }
        #expect(throws: MeshGeometry.ValidationError.invalidGroup(0)) {
            try plane.withGroups([MeshGeometryGroup(start: 0, count: 4)])
        }
        #expect(throws: MeshGeometry.ValidationError.invalidGroup(0)) {
            try plane.withGroups([MeshGeometryGroup(start: 6, count: 12)])
        }
        // Unvalidated groups still give ranges that subset rejects instead of trapping.
        #expect(MeshGeometryGroup(start: 0, count: -3).range.isEmpty)
        #expect(throws: MeshGeometry.ValidationError.invalidTriangleIndexCount) {
            try plane.subset(MeshGeometryGroup(start: 3, count: -3).range)
        }
        #expect(MeshGeometryGroup(start: Int.max - 1, count: 6).range.upperBound == Int.max)
        #expect(throws: MeshGeometry.ValidationError.invalidGroup(0)) {
            try plane.withGroups([MeshGeometryGroup(start: 0, count: -6)])
        }
    }

    @Test("A grouped node draws each group with its own material, like a Three.js material array")
    func groupedRendering() throws {
        let fixture = try GeometryRenderFixture()
        let plane = try MeshGeometry.plane(width: 2, height: 2, columns: 2, rows: 1).withGroups([
            MeshGeometryGroup(start: 0, count: 6, materialIndex: 1), MeshGeometryGroup(start: 6, count: 6, materialIndex: 0),
        ])
        let node = try SceneNode.grouped(
            plane, materials: [
                .classic(try ClassicMaterial(model: .basic, color: SIMD3(0.1, 0.8, 0.2))),
                .classic(try ClassicMaterial(model: .basic, color: SIMD3(0.9, 0.3, 0.1))),
            ],
            device: fixture.device, colorPixelFormat: .rgba16Float, transform: SceneTransform(translation: SIMD3(0, 0, -2))
        )
        #expect(node.children.count == 2)
        let pixels = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera)).pixels
        #expect(isNearly(fixture.pixel(pixels, x: 20, y: 32), SIMD4(0.9, 0.3, 0.1, 1)))
        #expect(isNearly(fixture.pixel(pixels, x: 44, y: 32), SIMD4(0.1, 0.8, 0.2, 1)))
        #expect(throws: GroupedMeshError.missingMaterial(1)) {
            try SceneNode.grouped(
                plane, materials: [.unlit], device: fixture.device, colorPixelFormat: .rgba16Float
            )
        }

        // A box draws its six faces with six materials.
        let box = try MeshGeometry.box()
        let colors: [SIMD3<Float>] = (0..<6).map { SIMD3(Float($0) / 5, 0.5, 1 - Float($0) / 5) }
        let cube = try SceneNode.grouped(
            box, materials: try colors.map { .classic(try ClassicMaterial(model: .basic, color: $0)) },
            device: fixture.device, colorPixelFormat: .rgba16Float, transform: SceneTransform(translation: SIMD3(0, 0, -3))
        )
        let front = try fixture.render(SceneGraph(roots: [cube], camera: fixture.camera)).pixels
        // The camera sees the +Z face, group 4.
        #expect(isNearly(fixture.pixel(front, x: 32, y: 32), SIMD4(colors[4], 1)))
    }
}
