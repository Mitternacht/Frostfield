import Frostfield
import Metal
import Testing
import simd

struct CurveTubeGeometryTests {
    @Test("Curve tube matches the r187 cubic Bézier oracle and top-left UV layout")
    func matchesReference() throws {
        let path = try Curve3D.cubicBezier(
            from: SIMD3(0.2, -0.4, -0.3), control1: SIMD3(0.9, 0.8, 1.1),
            control2: SIMD3(1.7, -0.4, 2), to: SIMD3(2.3, 0.9, 3.5)
        )
        let mesh = try MeshGeometry.tube(along: path, radius: 0.2, tubularSegments: 5)
        try #require(mesh.vertices.count == 54)
        #expect(mesh.indices.count == 240)
        #expect(Array(mesh.indices.prefix(18)) == [0, 9, 1, 9, 10, 1, 1, 10, 2, 10, 11, 2, 2, 11, 3, 11, 12, 3])
        #expect(Array(mesh.indices.suffix(12)) == [42, 51, 43, 51, 52, 43, 43, 52, 44, 52, 53, 44])
        // From r187 TubeGeometry with radius 0.2 and five axial by eight radial segments.
        let reference: [(Int, SIMD3<Float>, SIMD3<Float>, SIMD2<Float>)] = [
            (0, SIMD3(0.386977, -0.446195, -0.353904), SIMD3(0.934885, -0.230977, -0.269518), SIMD2(0, 1)),
            (2, SIMD3(0.2, -0.551862, -0.169854), SIMD3(0, -0.759309, 0.650730), SIMD2(0, 0.75)),
            (12, SIMD3(0.444831, -0.041240, 0.527543), SIMD3(-0.703272, -0.421033, 0.572836), SIMD2(0.2, 0.625)),
            (29, SIMD3(1.493539, 0.042810, 2.028104), SIMD3(-0.242131, -0.943692, 0.225427), SIMD2(0.6, 0.75)),
            (50, SIMD3(2.168176, 1.035704, 3.435141), SIMD3(-0.659120, 0.678523, -0.324296), SIMD2(1, 0.375)),
        ]
        for (index, position, normal, uv) in reference {
            let vertex = mesh.vertices[index]
            #expect(simd_distance(vertex.position, position) < 0.00012)
            #expect(simd_distance(vertex.normal, normal) < 0.0004)
            #expect(vertex.textureCoordinate == uv)
        }
        let centers = try path.points(segments: 5, spacing: .arcLength(subdivisions: 200))
        for row in 0...5 {
            let ring = mesh.vertices[(row * 9)...(row * 9 + 8)]
            for vertex in ring {
                #expect(simd_distance(vertex.position, centers[row] + 0.2 * vertex.normal) < 1e-6)
                #expect(abs(simd_length(vertex.normal) - 1) < 1e-6)
            }
            let start = mesh.vertices[row * 9], end = mesh.vertices[row * 9 + 8]
            #expect(start.position == end.position && start.normal == end.normal && start.tangent == end.tangent)
            #expect(start.textureCoordinate.y == 1 && end.textureCoordinate.y == 0)
        }
        #expect(!ShapeCheck.isClosed(mesh))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
    }

    @Test("A spatial closed tube welds its positions and frames across both UV seams")
    func closedLoop() throws {
        let path = Curve3D { fraction in
            let angle = 2 * Double.pi * fraction
            return SIMD3(2 * cos(angle), 1.4 * sin(angle) + 0.18 * cos(2 * angle),
                         0.22 * sin(2 * angle) + 0.16 * cos(3 * angle))
        }
        let steps = 36, radial = 10
        let mesh = try MeshGeometry.tube(along: path, radius: 0.16, tubularSegments: steps, radialSegments: radial, closed: true)
        try #require(mesh.vertices.count == (steps + 1) * (radial + 1))
        #expect(mesh.indices.count == steps * radial * 6)
        #expect(ShapeCheck.isClosed(mesh))
        #expect(mesh.signedVolume > 0)
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
        for column in 0...radial {
            let start = mesh.vertices[column]
            let end = mesh.vertices[steps * (radial + 1) + column]
            #expect(start.position == end.position && start.normal == end.normal && start.tangent == end.tangent)
            #expect(start.textureCoordinate.x == 0 && end.textureCoordinate.x == 1)
            #expect(start.textureCoordinate.y == end.textureCoordinate.y)
        }
    }

    @Test("Curve tube caps retain exact rim positions and use endpoint normals and disk UVs")
    func cappedTube() throws {
        let path = Curve3D { t in SIMD3(0, 0, 2 * t) }
        let mesh = try MeshGeometry.tube(along: path, radius: 0.25, tubularSegments: 3, capped: true)
        try #require(mesh.vertices.count == 54)
        #expect(mesh.indices.count == (3 + 1) * 8 * 6)
        #expect(ShapeCheck.isClosed(mesh))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh))
        let octagonVolume = 2 * 8 * 0.25 * 0.25 * sqrt(2.0) / 4
        #expect(abs(mesh.signedVolume - octagonVolume) < 1e-6)
        for (endpoint, row) in [0, 3].enumerated() {
            let cap = 36 + endpoint * 9
            let center = mesh.vertices[cap]
            #expect(center.position == SIMD3<Float>(0, 0, Float(endpoint * 2)))
            #expect(center.textureCoordinate == SIMD2<Float>(0.5, 0.5))
            #expect(center.normal == SIMD3<Float>(0, 0, endpoint == 0 ? -1 : 1))
            for column in 0..<8 {
                let rim = mesh.vertices[cap + 1 + column]
                #expect(rim.position == mesh.vertices[row * 9 + column].position)
                #expect(rim.normal == center.normal && rim.tangent == center.tangent)
            }
        }
    }

    @Test("Curve tube rejects invalid limits, mismatched loops, cusps and collapsed positions")
    func validation() {
        let line = Curve3D { SIMD3(0, 0, 2 * $0) }
        #expect(throws: MeshGeometry.CurveTubeError.invalidTubularSegments) {
            try MeshGeometry.tube(along: line, radius: 1, tubularSegments: .max)
        }
        #expect(throws: MeshGeometry.CurveTubeError.invalidTubularSegments) {
            try MeshGeometry.tube(along: line, radius: 1, tubularSegments: 2, closed: true)
        }
        #expect(throws: MeshGeometry.SweepError.invalidRadialSegments) {
            try MeshGeometry.tube(along: line, radius: 1, radialSegments: 2)
        }
        #expect(throws: MeshGeometry.SweepError.vertexBudget) {
            try MeshGeometry.tube(along: line, radius: 1, tubularSegments: 4_096, radialSegments: 128)
        }
        #expect(throws: MeshGeometry.SweepError.invalidRadius) {
            try MeshGeometry.tube(along: line, radius: .infinity)
        }
        #expect(throws: MeshGeometry.CurveTubeError.closedWithCaps) {
            try MeshGeometry.tube(along: line, radius: 1, closed: true, capped: true)
        }
        #expect(throws: MeshGeometry.CurveTubeError.invalidClosedPath) {
            try MeshGeometry.tube(along: line, radius: 1, closed: true)
        }
        let returnAlongLine = Curve3D { t in SIMD3(t * (1 - t), 0, 0) }
        #expect(throws: MeshGeometry.CurveTubeError.invalidClosedPath) {
            try MeshGeometry.tube(along: returnAlongLine, radius: 0.01, tubularSegments: 3, closed: true)
        }
        #expect(throws: MeshGeometry.CurveTubeError.invalidPathFrame(1)) {
            try MeshGeometry.tube(along: returnAlongLine, radius: 0.01, tubularSegments: 1)
        }
        #expect(throws: Curve3D.Failure.resolution) {
            try MeshGeometry.tube(along: line, radius: 1, arcLengthSubdivisions: 0)
        }
        let collapsed = Curve3D { t in SIMD3(1e20, 1e20, t) }
        #expect(throws: MeshGeometry.SweepError.unrepresentableSurface) {
            try MeshGeometry.tube(along: collapsed, radius: 1)
        }
        let tight = Curve3D { t in
            let angle = .pi / 2 * t
            return SIMD3(0.2 * cos(angle), 0.2 * sin(angle), 0)
        }
        #expect(throws: MeshGeometry.SweepError.unrepresentableSurface) {
            try MeshGeometry.tube(along: tight, radius: 0.3)
        }
    }

    @Test("Metal renders the open tube's side and fills the bore only with a cap",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func renders() throws {
        let gpu = try GeometryRenderFixture()
        let color = SIMD3<Float>(0.25, 0.75, 0.5)
        let longitudinal = Curve3D { t in SIMD3(0, 0, -2 - t) }
        let lateral = Curve3D { t in SIMD3(-1 + 2 * t, 0, -2) }
        func centerPixel(path: Curve3D, capped: Bool) throws -> SIMD4<Float> {
            let geometry = try MeshGeometry.tube(along: path, radius: 0.5, tubularSegments: 5, capped: capped)
            let renderer = try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .classic(ClassicMaterial(model: .basic, color: color))
            )
            let rendered = try gpu.render(SceneGraph(roots: [SceneNode(mesh: renderer)], camera: gpu.camera))
            #expect(rendered.statistics.drawnMeshCount == 1)
            return gpu.pixel(rendered.pixels, x: gpu.side / 2, y: gpu.side / 2)
        }
        #expect(try centerPixel(path: lateral, capped: false) == SIMD4(color, 1))
        #expect(try centerPixel(path: longitudinal, capped: false) == .zero)
        #expect(try centerPixel(path: longitudinal, capped: true) == SIMD4(color, 1))
    }
}
