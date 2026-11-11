import Frostfield
import Testing
import simd

struct ParametricSurfaceTests {
    @Test
    func periodicCylinderKeepsItsNormalsAndWindingAcrossTheUVSeam() throws {
        let mesh = try MeshGeometry.parametricSurface(uSegments: 32, vSegments: 3, wrapU: true) { uv in
            let angle = -2 * Float.pi * uv.x
            return SIMD3(cos(angle), uv.y * 2, sin(angle))
        }
        #expect(mesh.vertices.count == 132)
        #expect(mesh.indices.count == 576)
        for row in 0...3 {
            let first = mesh.vertices[row * 33]
            let last = mesh.vertices[row * 33 + 32]
            #expect(first.position == last.position)
            #expect(first.normal == last.normal)
            #expect(first.tangent == last.tangent)
            #expect(first.textureCoordinate.x == 0 && last.textureCoordinate.x == 1)
        }
        for offset in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = mesh.vertices[Int(mesh.indices[offset])]
            let b = mesh.vertices[Int(mesh.indices[offset + 1])]
            let c = mesh.vertices[Int(mesh.indices[offset + 2])]
            #expect(simd_dot(simd_cross(b.position - a.position, c.position - a.position), a.normal) > 0)
        }
    }

    @Test
    func rejectsSingularAndNonfiniteInput() {
        #expect(throws: MeshGeometry.ParametricSurfaceError.singularSample(u: 0, v: 0)) {
            try MeshGeometry.parametricSurface(uSegments: 2, vSegments: 2) { SIMD3($0.x, 0, 0) }
        }
        #expect(throws: MeshGeometry.ParametricSurfaceError.nonfiniteSample(u: 0, v: 0)) {
            try MeshGeometry.parametricSurface(uSegments: 2, vSegments: 2) { _ in SIMD3(.nan, 0, 0) }
        }
    }
}
