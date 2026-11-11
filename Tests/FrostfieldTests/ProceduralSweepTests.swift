import Frostfield
import Testing
import simd

struct ProceduralSweepTests {
    @Test("Tube transports orthonormal frames with an exact seam and arc-length UVs")
    func followsNonplanarPath() throws {
        let path: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(0, 0, 1), SIMD3(1, 0, 1), SIMD3(1, 2, 2)]
        let segments = 12
        let stride = segments + 1
        let radius: Float = 0.12
        let mesh = try MeshGeometry.tube(along: path, radius: radius, radialSegments: segments, capped: false)
        try #require(mesh.vertices.count == path.count * stride)
        #expect(mesh.indices.count == (path.count - 1) * segments * 6)
        let directions = [SIMD3<Float>(0, 0, 1), SIMD3<Float>(1, 0, 0), simd_normalize(SIMD3<Float>(0, 2, 1))]
        let tangents = [directions[0], simd_normalize(directions[0] + directions[1]),
                        simd_normalize(directions[1] + directions[2]), directions[2]]
        let expectedV: [Float] = [0, 1 / (2 + sqrt(5)), 2 / (2 + sqrt(5)), 1]

        for index in path.indices {
            let ring = index * stride
            let start = mesh.vertices[ring]
            let end = mesh.vertices[ring + segments]
            #expect(start.position == end.position)
            #expect(start.normal == end.normal)
            #expect(start.tangent == end.tangent)
            #expect(start.textureCoordinate.x == 0)
            #expect(end.textureCoordinate.x == 1)
            #expect(abs(start.textureCoordinate.y - expectedV[index]) < 1e-6)
            if index > 0 {
                let transport = simd_quatf(from: tangents[index - 1], to: tangents[index])
                let expectedNormal = transport.act(mesh.vertices[ring - stride].normal)
                #expect(simd_length(start.normal - expectedNormal) < 1e-5)
            }
            for vertex in mesh.vertices[ring...ring + segments] {
                let offset = vertex.position - path[index]
                let tangent = try #require(vertex.tangent)
                let circumferential = SIMD3(tangent.x, tangent.y, tangent.z)
                #expect(abs(simd_length(offset) - radius) < 1e-6)
                #expect(abs(simd_dot(vertex.normal, tangents[index])) < 1e-6)
                #expect(abs(simd_dot(vertex.normal, circumferential)) < 1e-6)
                #expect(tangent.w == 1)
            }
        }
        checkWinding(mesh)
    }

    @Test("Tube caps preserve the side rim positions and face outward")
    func capsUseIndependentNormalsAndUVs() throws {
        let path: [SIMD3<Float>] = [.zero, SIMD3(0, 0, 2)]
        let segments = 8
        let stride = segments + 1
        let mesh = try MeshGeometry.tube(along: path, radius: 0.25, radialSegments: segments)
        try #require(mesh.vertices.count == 4 * stride)
        #expect(mesh.indices.count == 12 * segments)
        for endpoint in path.indices {
            let capStart = (2 + endpoint) * stride
            let normal = SIMD3<Float>(0, 0, endpoint == 0 ? -1 : 1)
            #expect(mesh.vertices[capStart].position == path[endpoint])
            #expect(mesh.vertices[capStart].textureCoordinate == SIMD2<Float>(0.5, 0.5))
            #expect(mesh.vertices[capStart + 1].textureCoordinate == SIMD2<Float>(1, 0.5))
            for column in 0..<segments {
                let rim = mesh.vertices[capStart + 1 + column]
                #expect(rim.position == mesh.vertices[endpoint * stride + column].position)
                #expect(rim.normal == normal)
                #expect(rim.tangent?.w == (endpoint == 0 ? 1 : -1))
            }
        }
        checkWinding(mesh)
    }

    @Test("Cubic Bézier samples both endpoints and produces a reusable tube path")
    func samplesBezier() throws {
        let end = SIMD3<Float>(2, 0, 0)
        let points = try ProceduralPath.cubicBezier(
            from: .zero, control1: SIMD3(0, 2, 0), control2: SIMD3(2, 2, 0), to: end, segments: 12
        )
        try #require(points.count == 13)
        #expect(points[0] == .zero)
        #expect(points[6] == SIMD3<Float>(1, 1.5, 0))
        #expect(points[12] == end)
        let mesh = try MeshGeometry.tube(along: points, radius: 0.08)
        #expect(mesh.vertices.count == 15 * 13)
    }

    @Test("Sweep validation rejects coincident points, cusps, and collapsed rings")
    func rejectsInvalidPath() {
        #expect(throws: MeshGeometry.SweepError.coincidentPoints(1)) {
            try MeshGeometry.tube(along: [.zero, .zero], radius: 1)
        }
        #expect(throws: MeshGeometry.SweepError.reversingPath(1)) {
            try MeshGeometry.tube(along: [.zero, SIMD3(1, 0, 0), .zero], radius: 0.2)
        }
        #expect(throws: MeshGeometry.SweepError.unrepresentableSurface) {
            try MeshGeometry.tube(along: [SIMD3(1e20, 1e20, 0), SIMD3(1e20, 1e20, 1)], radius: 0.1)
        }
        #expect(throws: MeshGeometry.SweepError.invalidRadialSegments) {
            try MeshGeometry.tube(along: [.zero, SIMD3(1, 0, 0)], radius: 0.2, radialSegments: .max)
        }
        #expect(throws: ProceduralPath.ValidationError.invalidSegments) {
            try ProceduralPath.cubicBezier(from: .zero, control1: .zero, control2: .zero, to: .zero, segments: .max)
        }
    }

    private func checkWinding(_ mesh: MeshGeometry) {
        for index in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = mesh.vertices[Int(mesh.indices[index])]
            let b = mesh.vertices[Int(mesh.indices[index + 1])]
            let c = mesh.vertices[Int(mesh.indices[index + 2])]
            let outward = a.normal + b.normal + c.normal
            #expect(simd_dot(simd_cross(b.position - a.position, c.position - a.position), outward) > 0)
        }
    }
}
