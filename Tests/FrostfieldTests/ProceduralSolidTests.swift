import Frostfield
import simd
import Testing

struct ProceduralSolidTests {
    @Test("Capped lathe has separate side and cap UVs and a closed outward surface")
    func cappedLathe() throws {
        let segments = 13
        let profile: [LatheProfilePoint] = [.init(radius: 1, y: -1), .init(radius: 1, y: 1)]
        let solid = try MeshGeometry.lathe(profile: profile, segments: segments)
        #expect(solid.indices.count == 4 * segments * 3)
        #expect(solid.vertices[0].position == solid.vertices[segments].position)
        #expect(solid.vertices[0].normal == solid.vertices[segments].normal)
        #expect(solid.vertices[0].textureCoordinate == SIMD2(0, 1))
        #expect(solid.vertices[segments].textureCoordinate == SIMD2(1, 1))
        let bottomCenter = solid.vertices[2 * (segments + 1)]
        #expect(bottomCenter.position == SIMD3(0, -1, 0))
        #expect(bottomCenter.normal == SIMD3(0, -1, 0))
        #expect(bottomCenter.textureCoordinate == SIMD2(repeating: 0.5))
        #expect(isClosedAndOriented(solid))
        #expect(normalsFaceOut(solid))

        let open = try MeshGeometry.lathe(profile: profile, segments: segments, capBottom: false, capTop: false)
        #expect(open.indices.count == 2 * segments * 3)
        #expect(!isClosedAndOriented(open))
        #expect(normalsFaceOut(open))
    }

    @Test("Lathe shoulders can crease and authored normals affect only their profile ring")
    func profileNormalControl() throws {
        let profile: [LatheProfilePoint] = [
            .init(radius: 0.75, y: -1),
            .init(radius: 0.75, y: 0, hard: true),
            .init(radius: 1, y: 0, hard: true),
            .init(radius: 1, y: 1, normal: SIMD2(3, 4)),
        ]
        let solid = try MeshGeometry.lathe(profile: profile, segments: 12)
        let crease = solid.vertices.filter { $0.position == SIMD3(0.75, 0, 0) }
        #expect(crease.contains { $0.normal == SIMD3(1, 0, 0) })
        #expect(crease.contains { $0.normal == SIMD3(0, -1, 0) })
        #expect(crease.allSatisfy { abs($0.textureCoordinate.y - 5.0 / 9.0) < 0.00001 })
        #expect(solid.vertices.contains {
            $0.position == SIMD3(1, 1, 0) && simd_distance($0.normal, SIMD3(0.6, 0.8, 0)) < 0.00001
        })
        #expect(isClosedAndOriented(solid))
        #expect(normalsFaceOut(solid))
    }

    @Test("Axis endpoints close with one nondegenerate triangle per wedge")
    func poles() throws {
        let segments = 11
        let solid = try MeshGeometry.lathe(
            profile: [.init(radius: 0, y: -1), .init(radius: 1, y: 0), .init(radius: 0, y: 1)],
            segments: segments
        )
        #expect(solid.indices.count == 2 * segments * 3)
        let lowerPole = solid.vertices.filter { $0.position == SIMD3(0, -1, 0) }
        #expect(lowerPole.count == segments)
        #expect(lowerPole.first?.textureCoordinate.x == 0.5 / Float(segments))
        #expect(isClosedAndOriented(solid))
        #expect(normalsFaceOut(solid))
    }

    @Test("Rounded box preserves size and joins smooth circular patches across chart seams")
    func roundedBox() throws {
        let size = SIMD3<Float>(2, 1, 3)
        let bevel: Float = 0.2
        let geometry = try MeshGeometry.bevelBox(size: size, bevel: bevel, segments: 3)
        let half = size * 0.5
        let core = half - bevel
        for axis in 0..<3 {
            #expect(geometry.vertices.map { $0.position[axis] }.max() == half[axis])
            #expect(geometry.vertices.map { $0.position[axis] }.min() == -half[axis])
        }
        #expect(geometry.vertices.allSatisfy { vertex in
            let offset = vertex.position - simd_clamp(vertex.position, -core, core)
            return abs(simd_length(offset) - bevel) < 0.000002
                && simd_dot(simd_normalize(offset), vertex.normal) > 0.99999
                && (0...1).contains(vertex.textureCoordinate.x)
                && (0...1).contains(vertex.textureCoordinate.y)
        })
        let seams = Dictionary(grouping: geometry.vertices, by: \.position).values.filter { $0.count > 1 }
        #expect(!seams.isEmpty)
        #expect(seams.allSatisfy { seam in seam.allSatisfy { $0.normal == seam[0].normal } })
        #expect(isClosedAndOriented(geometry))
        #expect(normalsFaceOut(geometry))

        let sharp = try MeshGeometry.bevelBox(size: size, bevel: 0)
        #expect(sharp.vertices.count == 24)
        #expect(sharp.indices.count == 36)
        #expect(isClosedAndOriented(sharp))
        #expect(normalsFaceOut(sharp))
    }

    @Test("Procedural solids reject reversed shoulders, overlapping caps and collapsed bevels")
    func invalidProfilesAndEdges() {
        #expect(throws: MeshGeometry.SolidValidationError.profileEdge(2)) {
            try MeshGeometry.lathe(profile: [
                .init(radius: 1, y: -1), .init(radius: 1, y: 0),
                .init(radius: 0.5, y: 0), .init(radius: 0.75, y: 0), .init(radius: 1, y: 1),
            ])
        }
        #expect(throws: MeshGeometry.SolidValidationError.overlappingCap(0)) {
            try MeshGeometry.lathe(profile: [
                .init(radius: 1, y: 0), .init(radius: 0.5, y: 0), .init(radius: 0.5, y: 1),
            ])
        }
        #expect(throws: MeshGeometry.SolidValidationError.profilePoint(1)) {
            try MeshGeometry.lathe(profile: [
                .init(radius: 1, y: -1), .init(radius: 0, y: 0), .init(radius: 1, y: 1),
            ])
        }
        #expect(throws: MeshGeometry.SolidValidationError.profileNormal(0)) {
            try MeshGeometry.lathe(profile: [.init(radius: 1, y: 0, normal: .zero), .init(radius: 1, y: 1)])
        }
        #expect(throws: MeshGeometry.SolidValidationError.bevel) { try MeshGeometry.bevelBox(bevel: 0.5) }
        #expect(throws: MeshGeometry.SolidValidationError.collapsedTriangle) {
            try MeshGeometry.bevelBox(bevel: .leastNonzeroMagnitude)
        }
    }

    private struct Edge: Hashable {
        let a: Int
        let b: Int
    }

    private func isClosedAndOriented(_ geometry: MeshGeometry) -> Bool {
        var points: [SIMD3<Float>: Int] = [:]
        let welded = geometry.vertices.map { vertex in
            if let id = points[vertex.position] { return id }
            let id = points.count
            points[vertex.position] = id
            return id
        }
        var edges: [Edge: (count: Int, orientation: Int)] = [:]
        for offset in stride(from: 0, to: geometry.indices.count, by: 3) {
            let a = welded[Int(geometry.indices[offset])]
            let b = welded[Int(geometry.indices[offset + 1])]
            let c = welded[Int(geometry.indices[offset + 2])]
            for (from, to) in [(a, b), (b, c), (c, a)] {
                guard from != to else { return false }
                let edge = Edge(a: min(from, to), b: max(from, to))
                let prior = edges[edge, default: (0, 0)]
                edges[edge] = (prior.count + 1, prior.orientation + (from < to ? 1 : -1))
            }
        }
        return edges.values.allSatisfy { $0.count == 2 && $0.orientation == 0 }
    }

    private func normalsFaceOut(_ geometry: MeshGeometry) -> Bool {
        stride(from: 0, to: geometry.indices.count, by: 3).allSatisfy { offset in
            let a = geometry.vertices[Int(geometry.indices[offset])]
            let b = geometry.vertices[Int(geometry.indices[offset + 1])]
            let c = geometry.vertices[Int(geometry.indices[offset + 2])]
            let geometric = simd_cross(b.position - a.position, c.position - a.position)
            return simd_dot(geometric, a.normal + b.normal + c.normal) > 0
        }
    }
}
