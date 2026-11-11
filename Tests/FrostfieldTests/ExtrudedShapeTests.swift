import Frostfield
import Testing
import simd

struct ExtrudedShapeTests {
    @Test("Concave extrusion keeps both caps inside the outline with outward winding", arguments: [false, true])
    func concaveExtrusion(clockwise: Bool) throws {
        let outline: [SIMD2<Float>] = [
            SIMD2(0, 0), SIMD2(1, 0), SIMD2(3, 0), SIMD2(3, 1),
            SIMD2(1, 1), SIMD2(1, 3), SIMD2(0, 3),
        ]
        let contour = clockwise ? [outline[0]] + outline.dropFirst().reversed() : outline
        let depth: Float = 0.4
        let mesh = try MeshGeometry.extrude(contour: contour, depth: depth)
        try #require(mesh.vertices.count == 6 * contour.count)
        #expect(mesh.indices.count == 12 * contour.count - 12)
        #expect(mesh.vertices.allSatisfy { abs($0.position.z) == depth / 2 })
        #expect(mesh.vertices.allSatisfy { abs(simd_length($0.normal) - 1) < 1e-6 })
        #expect(mesh.vertices.allSatisfy { $0.tangent != nil })

        var frontArea: Float = 0
        var backArea: Float = 0
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = mesh.vertices[Int(mesh.indices[offset])]
            let b = mesh.vertices[Int(mesh.indices[offset + 1])]
            let c = mesh.vertices[Int(mesh.indices[offset + 2])]
            let cross = simd_cross(b.position - a.position, c.position - a.position)
            #expect(simd_dot(cross, a.normal) > 0)
            #expect(a.normal == b.normal && a.normal == c.normal)
            if a.normal.z != 0 {
                let middle = (a.position + b.position + c.position) / 3
                #expect(middle.x <= 1 || middle.y <= 1)
                #expect(!triangleContains(SIMD2(2, 2), a.position, b.position, c.position))
                if a.normal.z > 0 { frontArea += cross.z / 2 }
                else { backArea -= cross.z / 2 }
            }
        }
        #expect(frontArea == 5)
        #expect(backArea == 5)
    }

    @Test("Cap and wall UVs use top-left coordinates and a continuous perimeter seam")
    func textureCoordinates() throws {
        let contour: [SIMD2<Float>] = [SIMD2(1, 1), SIMD2(5, 1), SIMD2(5, 3), SIMD2(1, 3)]
        let mesh = try MeshGeometry.extrude(contour: contour, depth: 0.5)
        #expect(mesh.vertices[0].position == SIMD3<Float>(1, 1, 0.25))
        #expect(mesh.vertices[0].textureCoordinate == SIMD2<Float>(0, 1))
        #expect(mesh.vertices[3].textureCoordinate == SIMD2<Float>(0, 0))
        #expect(mesh.vertices[0].tangent == SIMD4<Float>(1, 0, 0, -1))
        #expect(mesh.vertices[4].tangent == SIMD4<Float>(1, 0, 0, 1))
        let wall = mesh.vertices[8...11]
        #expect(wall.allSatisfy { $0.normal == SIMD3<Float>(0, -1, 0) })
        #expect(wall.map(\.textureCoordinate) == [SIMD2(0, 0), SIMD2(1 / 3, 0), SIMD2(0, 1), SIMD2(1 / 3, 1)])
        #expect(mesh.vertices[8].position == mesh.vertices[21].position)
        #expect(mesh.vertices[21].textureCoordinate == SIMD2<Float>(1, 0))
    }

    @Test("Extrusion rejects non-simple contours, duplicate points and unsafe sizes")
    func validatesContour() {
        let start = SIMD2<Float>(0, 0)
        let valid = [start, SIMD2(2, 0), SIMD2(2, 2)]
        #expect(throws: MeshGeometry.ExtrusionError.duplicatePoint(3)) {
            try MeshGeometry.extrude(contour: valid + [start], depth: 1)
        }
        #expect(throws: MeshGeometry.ExtrusionError.invalidPoint(2)) {
            try MeshGeometry.extrude(contour: [start, SIMD2(2, 0), SIMD2(2, .nan)], depth: 1)
        }
        #expect(throws: MeshGeometry.ExtrusionError.invalidVertexCount) {
            try MeshGeometry.extrude(contour: Array(repeating: start, count: 257), depth: 1)
        }
        #expect(throws: MeshGeometry.ExtrusionError.invalidDepth) {
            try MeshGeometry.extrude(contour: valid, depth: .infinity)
        }
        let invalid: [[SIMD2<Float>]] = [
            [start, SIMD2(2, 2), SIMD2(0, 2), SIMD2(2, 0)],
            [start, SIMD2(3, 0), SIMD2(3, 3), SIMD2(1, 0), SIMD2(0, 3)],
            [start, SIMD2(2, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)],
        ]
        for contour in invalid {
            #expect(throws: MeshGeometry.ExtrusionError.selfIntersection) {
                try MeshGeometry.extrude(contour: contour, depth: 1)
            }
        }
    }

    private func triangleContains(_ p: SIMD2<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> Bool {
        func side(_ u: SIMD3<Float>, _ v: SIMD3<Float>) -> Float {
            (v.x - u.x) * (p.y - u.y) - (v.y - u.y) * (p.x - u.x)
        }
        let sides = [side(a, b), side(b, c), side(c, a)]
        return sides.allSatisfy { $0 >= 0 } || sides.allSatisfy { $0 <= 0 }
    }
}
