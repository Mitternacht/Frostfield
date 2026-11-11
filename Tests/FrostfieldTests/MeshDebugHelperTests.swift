import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshDebugHelperTests {
    private let fixture: GeometryRenderFixture
    private let normal = simd_normalize(SIMD3<Float>(1, 1, 0))
    private let tangent = simd_normalize(SIMD3<Float>(1, -1, 0))

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    @Test("Vertex helpers transform every authored point and preserve world length under reflection and shear")
    func vertexDirections() throws {
        let geometry = try mesh()
        let parent = SceneTransform(
            translation: SIMD3(0.3, -0.15, -2), rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 0, 1)),
            scale: SIMD3(-2, 3, 0.5)
        )
        let child = SceneTransform(
            translation: SIMD3(0.1, 0.2, 0), rotation: simd_quatf(angle: -0.25, axis: SIMD3(0, 0, 1))
        )
        let node = SceneNode(transform: child, mesh: try fixture.unlit(geometry))
        let graph = SceneGraph(roots: [SceneNode(transform: parent, children: [node])], camera: fixture.camera)
        let size: Float = 0.6
        let normals = try #require(try LineGeometry.vertexNormals(of: node.id, in: graph, size: size))
        let tangents = try #require(try LineGeometry.vertexTangents(of: node.id, in: graph, size: size))
        #expect(normals.segments.count == 4 && tangents.segments.count == 4)
        #expect(normals.colors?.first == SIMD4(1, 0, 0, 1))
        #expect(tangents.colors?.first == SIMD4(0, 1, 1, 1))
        let expectedNormal = simd_normalize(parent.rotation.act(child.rotation.act(normal) / parent.scale))
        let expectedTangent = simd_normalize(parent.rotation.act(child.rotation.act(tangent) * parent.scale))
        #expect(abs(simd_dot(expectedNormal, expectedTangent)) < 1e-6)
        for (index, vertex) in geometry.vertices.enumerated() {
            let local = child.rotation.act(vertex.position) + child.translation
            let origin = parent.rotation.act(local * parent.scale) + parent.translation
            #expect(simd_distance(normals.positions[2 * index], origin) < 1e-6)
            #expect(simd_distance(tangents.positions[2 * index], origin) < 1e-6)
            #expect(simd_distance(normals.positions[2 * index + 1], origin + expectedNormal * size) < 1e-6)
            #expect(simd_distance(tangents.positions[2 * index + 1], origin + expectedTangent * size) < 1e-6)
        }
        // Sequential indices represent a native nonindexed triangle stream.
        let expanded = try MeshGeometry(vertices: geometry.indices.map { geometry.vertices[Int($0)] }, indices: [0, 1, 2])
        let sequential = try LineGeometry.vertexTangents(of: expanded, worldFromObject: parent.matrix * child.matrix, size: size)
        #expect(sequential.positions == geometry.indices.flatMap {
            let index = 2 * Int($0)
            return [tangents.positions[index], tangents.positions[index + 1]]
        })
        #expect(try LineGeometry.vertexNormals(of: UUID(), in: graph) == nil)
    }

    @Test("World normal and tangent segments draw in their distinct reflected directions")
    func rasterDirections() throws {
        let geometry = try mesh()
        let transform = SceneTransform(translation: SIMD3(-0.4, 0, -2), scale: SIMD3(-2, 3, 0.5))
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: LineMaterial(width: .pixels(3)))
        let size: Float = 0.7
        let normals = try helpers.node(LineGeometry.vertexNormals(of: geometry, worldFromObject: transform.matrix, size: size))
        let tangents = try helpers.node(LineGeometry.vertexTangents(of: geometry, worldFromObject: transform.matrix, size: size))
        let scene = SceneGraph(roots: [normals, tangents], camera: fixture.camera)
        let pixels = try fixture.render(scene).pixels
        let reflectedNormal = simd_normalize(SIMD3<Float>(-0.5, 1.0 / 3.0, 0))
        let reflectedTangent = simd_normalize(SIMD3<Float>(-2, -3, 0))
        for (direction, color) in [(reflectedNormal, SIMD4<Float>(1, 0, 0, 1)), (reflectedTangent, SIMD4<Float>(0, 1, 1, 1))] {
            let middle = transform.translation + direction * (size * 0.5)
            let x = fixture.column(x: middle.x, depth: 2)
            let y = fixture.row(y: middle.y, depth: 2)
            #expect(isNearly(fixture.pixel(pixels, x: x, y: y), color))
        }
    }

    @Test("UV layout follows indexed face order, sequential triples and authored UV1")
    func uvTopology() throws {
        let geometry = try mesh()
        let layout = try UVLayoutDebug(geometry: geometry)
        let corner = try #require(layout.labels.dropFirst().first)
        #expect(layout.lines.segments.count == 3 && layout.labels.count == 4)
        #expect(layout.labels.map(\.text) == ["0", "a2", "b0", "c1"])
        #expect(corner.triangle == 0 && corner.kind == .corner(order: 0, vertex: 2))
        #expect(simd_distance(layout.labels[0].position, SIMD2(0.4, 0.4)) < 1e-6)
        #expect(simd_distance(corner.position, SIMD2(0.3, 0.6)) < 1e-6)
        let sequential = try UVLayoutDebug(coordinates: geometry.indices.map { geometry.vertices[Int($0)].textureCoordinate })
        #expect(sequential.lines.positions == layout.lines.positions)
        #expect(sequential.labels.map(\.text) == ["0", "a0", "b1", "c2"])
        let secondary = try UVLayoutDebug(geometry: geometry, coordinateSet: .secondary)
        #expect(simd_distance(secondary.lines.positions[0], SIMD3(1.2, 0.2, 0)) < 1e-6)
        #expect(simd_distance(secondary.labels[0].position, SIMD2(1.4, 0.4)) < 1e-6)
    }

    @Test("A UV layout renders its top edge and a native Core Text face label")
    func rasterUVLayout() throws {
        let layout = try UVLayoutDebug(geometry: mesh(), color: SIMD4(0, 1, 0, 1))
        let helpers = DebugHelpers(device: fixture.device, lineMaterial: LineMaterial(width: .pixels(2.5)))
        let node = try helpers.node(layout.lines, transform: .init(translation: SIMD3(-0.5, -0.5, -2)))
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let command = try #require(fixture.queue.makeCommandBuffer())
        try graph.encode(
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command,
            clipFromWorld: simd_float4x4(diagonal: SIMD4(2, 2, -0.1, 1)),
            cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1)
        )

        let face = try #require(layout.labels.first)
        let label = try TextureDebugPass.Label(face.text, device: fixture.device)
        let anchor = SIMD2<Int>(Int(face.position.x * Float(fixture.side)), Int(face.position.y * Float(fixture.side)))
        let origin = SIMD2(anchor.x - label.width / 2, anchor.y - label.height / 2)
        let debug = try TextureDebugPass(device: fixture.device)
        try debug.encode(label: label, destination: target.color, origin: origin, commandBuffer: command)
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: target.color)
        #expect(isNearly(fixture.pixel(pixels, x: 32, y: 13), SIMD4(0, 1, 0, 1)))
        #expect(fixture.pixel(pixels, x: 32, y: 51) == .zero)
        let glyph = (origin.y..<(origin.y + label.height)).flatMap { y in
            (origin.x..<(origin.x + label.width)).map { x in fixture.pixel(pixels, x: x, y: y) }
        }
        #expect(glyph.contains { $0.x > 0.2 && $0.y == 0 && $0.z == 0 })
        #expect(glyph.contains(.zero))
    }

    @Test("Vertex and UV helpers reject invalid attributes and transforms")
    func validation() throws {
        let plane = try MeshGeometry.plane(width: 1, height: 1)
        let bare = try MeshGeometry(vertices: plane.vertices.map { MeshVertex(position: $0.position, normal: $0.normal) },
                                    indices: plane.indices)
        #expect(throws: MeshDebugFailure.missingTangent(0)) { try LineGeometry.vertexTangents(of: bare) }
        #expect(throws: MeshDebugFailure.missingSecondaryUV(0)) { try UVLayoutDebug(geometry: bare, coordinateSet: .secondary) }
        #expect(throws: MeshDebugFailure.transform) {
            try LineGeometry.vertexNormals(of: plane, worldFromObject: simd_float4x4(diagonal: SIMD4(1, 0, 1, 1)))
        }
        #expect(throws: MeshDebugFailure.invalidUV(1)) {
            try UVLayoutDebug(coordinates: [.zero, SIMD2(.nan, 0), SIMD2(1, 0)])
        }
        #expect(throws: MeshGeometry.ValidationError.invalidTriangleIndexCount) {
            try UVLayoutDebug(coordinates: [.zero, SIMD2(1, 0)])
        }
        #expect(throws: MeshGeometry.ValidationError.indexOutOfBounds(1)) {
            try UVLayoutDebug(coordinates: [.zero], indices: [0, 1, 0])
        }
    }

    private func mesh() throws -> MeshGeometry {
        let positions: [SIMD3<Float>] = [.zero, SIMD3(-0.5, 0.3, 0), SIMD3(-0.5, -0.3, 0), SIMD3(0, 0.8, 0)]
        let uv: [SIMD2<Float>] = [SIMD2(0.2, 0.2), SIMD2(0.8, 0.2), SIMD2(0.2, 0.8), SIMD2(0.8, 0.8)]
        let vertices = zip(positions, uv).map { point, uv in
            MeshVertex(position: point, normal: normal, textureCoordinate: uv,
                       secondaryTextureCoordinate: uv + SIMD2(1, 0), tangent: SIMD4(tangent, -1))
        }
        return try MeshGeometry(vertices: vertices, indices: [2, 0, 1])
    }
}
