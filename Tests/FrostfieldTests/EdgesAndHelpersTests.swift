import Foundation
import Frostfield
import Metal
import Testing
import simd

struct EdgesAndHelpersTests {
    private func segments(_ line: LineGeometry) -> [(SIMD3<Float>, SIMD3<Float>)] {
        line.segments.map { (line.positions[$0.start], line.positions[$0.end]) }
    }

    @Test
    func `EdgesGeometry of a cube keeps its twelve edges at a one degree threshold`() throws {
        let edges = try #require(try LineGeometry.edges(of: .cube()))
        #expect(edges.topology == .segments)
        #expect(edges.segments.count == 12)
        let corners = Set(edges.positions.map { SIMD3<Int>(($0 * 2).rounded(.toNearestOrAwayFromZero)) })
        #expect(corners.count == 8)
        for (a, b) in segments(edges) {
            #expect(abs(simd_distance(a, b) - 1) < 1e-6)
            // Each edge runs along one axis.
            #expect((0..<3).filter { abs(a[$0] - b[$0]) > 1e-6 }.count == 1)
        }
        // Past 90° no face pair qualifies, so a closed cube has no edges.
        #expect(try LineGeometry.edges(of: .cube(), thresholdAngle: .pi / 2 + 0.01) == nil)
    }

    @Test
    func `Edges keep boundaries and creases and skip coplanar diagonals`() throws {
        // Two coplanar triangles: the four outline edges remain.
        let square = try GeometryRenderFixture.square(1)
        #expect(try LineGeometry.edges(of: square)?.segments.count == 4)
        // A fold of 30° between two triangles keeps the shared edge only below 30°.
        let angle: Float = .pi / 6
        let folded = try MeshGeometry(
            vertices: [
                MeshVertex(position: SIMD3(0, 0, 0), normal: SIMD3(0, 0, 1)),
                MeshVertex(position: SIMD3(0, 1, 0), normal: SIMD3(0, 0, 1)),
                MeshVertex(position: SIMD3(-1, 0, 0), normal: SIMD3(0, 0, 1)),
                MeshVertex(position: SIMD3(cos(angle), 0, -sin(angle)), normal: SIMD3(0, 0, 1)),
            ],
            indices: [0, 1, 2, 0, 3, 1]
        )
        #expect(try LineGeometry.edges(of: folded, thresholdAngle: .pi / 6 - 0.01)?.segments.count == 5)
        #expect(try LineGeometry.edges(of: folded, thresholdAngle: .pi / 6 + 0.01)?.segments.count == 4)
        #expect(throws: PrimitiveGeometryError.invalidThresholdAngle) {
            try LineGeometry.edges(of: square, thresholdAngle: -1)
        }
    }

    @Test
    func `WireframeGeometry lists each distinct triangle edge once by exact position`() throws {
        // 12 cube edges and one diagonal per face; faces share edges by equal positions.
        let wireframe = try #require(try LineGeometry.wireframe(of: .cube()))
        #expect(wireframe.segments.count == 18)
        func mesh(_ positions: [SIMD3<Float>]) throws -> MeshGeometry {
            try MeshGeometry(
                vertices: positions.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) },
                indices: (0..<UInt32(positions.count)).map { $0 }
            )
        }
        // Three.js does not weld: a triangle smaller than 10⁻⁴ keeps its three edges,
        // and an edge 10⁻⁵ away from another stays separate.
        let tiny = try #require(try LineGeometry.wireframe(of: mesh([.zero, SIMD3(4e-5, 0, 0), SIMD3(0, 4e-5, 0)])))
        #expect(tiny.segments.count == 3)
        let near = try #require(try LineGeometry.wireframe(of: mesh([
            .zero, SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(1.00001, 0, 0), .zero, SIMD3(0, -1, 0),
        ])))
        #expect(near.segments.count == 6)
        // A shared edge in either direction appears once; zero-length edges are skipped.
        let shared = try #require(try LineGeometry.wireframe(of: mesh([
            .zero, SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 0, 0), .zero, SIMD3(0, -1, 0),
        ])))
        #expect(shared.segments.count == 5)
        let collapsed = try #require(try LineGeometry.wireframe(of: mesh([.zero, .zero, SIMD3(1, 0, 0)])))
        #expect(collapsed.segments.count == 1)
        #expect(try LineGeometry.wireframe(of: mesh([.zero, .zero, .zero])) == nil)
    }

    @Test
    func `EdgesGeometry welds with JavaScript rounding, ties toward positive infinity`() throws {
        // Two coplanar triangles whose shared corner differs in the two copies:
        // −0.03125 × 10⁴ is the tie −312.5, which Math.round sends to −312, the same
        // grid point as −0.0312495. Welded, only the outline remains.
        let tie = SIMD3<Float>(-0.03125, 0, 0)
        let nearby = SIMD3<Float>(-0.0312495, 0, 0)
        let mesh = try MeshGeometry(
            vertices: [tie, SIMD3(1, 0, 0), SIMD3(1, 1, 0), nearby, SIMD3(1, 1, 0), SIMD3(0, 1, 0)]
                .map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) },
            indices: [0, 1, 2, 3, 4, 5]
        )
        #expect(try LineGeometry.edges(of: mesh)?.segments.count == 4)
    }

    @Test
    func `Axes, grid and polar grid match the Three.js helper layouts`() throws {
        let axes = try LineGeometry.axes(size: 2)
        #expect(segments(axes).map(\.1) == [SIMD3(2, 0, 0), SIMD3(0, 2, 0), SIMD3(0, 0, 2)])
        #expect(axes.colors?[0] == SIMD4(1, 0, 0, 1) && axes.colors?[3] == SIMD4(0.6, 1, 0, 1))

        let grid = try LineGeometry.grid(size: 10, divisions: 10)
        #expect(grid.segments.count == 22)
        #expect(grid.positions.allSatisfy { $0.y == 0 && abs($0.x) <= 5 && abs($0.z) <= 5 })
        let center = LineGeometry.helperColor(0x444444)
        // The two center lines cross the origin.
        let centerLines = grid.segments.filter { grid.colors?[$0.start] == center }
        #expect(centerLines.count == 2)
        #expect(centerLines.allSatisfy { grid.positions[$0.start].x == 0 && grid.positions[$0.end].x == 0
            || grid.positions[$0.start].z == 0 && grid.positions[$0.end].z == 0 })
        #expect(abs(center.x - 0.0578054) < 1e-5)
        #expect(try LineGeometry.grid(size: 10, divisions: 5).colors?.contains(center) == false)

        let polar = try LineGeometry.polarGrid(radius: 10, sectors: 16, rings: 8, divisions: 64)
        #expect(polar.segments.count == 16 + 8 * 64)
        let radii = Set(polar.positions.map { (simd_length($0) * 100).rounded() / 100 })
        #expect(radii == Set([0] + (1...8).map { Float($0) * 1.25 }))
    }

    @Test
    func `Box and camera helpers trace the box and the unprojected frustum`() throws {
        let box = try AxisAlignedBox(minimum: SIMD3(-1, -2, -3), maximum: SIMD3(1, 2, 3))
        let outline = try LineGeometry.box(box)
        #expect(outline.segments.count == 12)
        #expect(Set(outline.positions.map { SIMD3<Int>($0) }).count == 8)
        let lengths = segments(outline).map { simd_distance($0.0, $0.1) }.sorted()
        #expect(lengths == [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6])

        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.5, far: 10)
        let frustum = try LineGeometry.cameraFrustum(camera: camera, aspectRatio: 2)
        let halfHeight = tan(Float.pi / 6)
        for depth: Float in [0.5, 10] {
            for sx: Float in [-1, 1] {
                for sy: Float in [-1, 1] {
                    let corner = SIMD3(sx * 2 * halfHeight * depth, sy * halfHeight * depth, -depth)
                    let nearest = frustum.positions.map { simd_distance($0, corner) }.min() ?? .infinity
                    #expect(nearest < 1e-5 * depth, "corner \(corner) is \(nearest) away")
                }
            }
        }
        // The cone starts at the camera center, the local origin.
        let cone = LineGeometry.helperColor(0xff0000)
        let coneStarts = frustum.segments.filter { frustum.colors?[$0.start] == cone }.map { frustum.positions[$0.start] }
        #expect(coneStarts.count == 4)
        #expect(coneStarts.allSatisfy { simd_length($0) < 1e-5 })
        // A world-space frustum from a clip-from-world matrix finds the moved center.
        let transform = SceneTransform(translation: SIMD3(3, 1, -2))
        let world = try LineGeometry.cameraFrustum(projection: camera.projection(aspectRatio: 2) * transform.matrix.inverse)
        let worldCone = world.segments.filter { world.colors?[$0.start] == cone }.map { world.positions[$0.start] }
        #expect(worldCone.allSatisfy { simd_distance($0, SIMD3(3, 1, -2)) < 1e-4 })

        // Three.js order: near, far and side frustum edges, cone, up triangle, the
        // center line from c to t, the line from the camera to c, then the crosses.
        #expect(frustum.segments.count == 25)
        let colors = frustum.segments.map { frustum.colors?[$0.start] }
        let expectedColors = Array(repeating: 0xffaa00, count: 12) + Array(repeating: 0xff0000, count: 4)
            + Array(repeating: 0x00aaff, count: 3) + [0xffffff] + Array(repeating: 0x333333, count: 5)
        #expect(colors == expectedColors.map { LineGeometry.helperColor(UInt32($0)) })
        let target = frustum.segments[19]
        let toCenter = frustum.segments[20]
        #expect(simd_distance(frustum.positions[target.start], SIMD3(0, 0, -0.5)) < 1e-5)
        #expect(simd_distance(frustum.positions[target.end], SIMD3(0, 0, -10)) < 1e-4)
        #expect(simd_length(frustum.positions[toCenter.start]) < 1e-5)
        #expect(simd_distance(frustum.positions[toCenter.end], SIMD3(0, 0, -0.5)) < 1e-5)
        // u1 is on the right, as in Three.js.
        #expect(frustum.positions[frustum.segments[16].start].x > 0)

        // An orthographic projection draws its cone from the local origin, as Three.js does.
        let orthographic = simd_float4x4(columns: (
            SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0), SIMD4(0, 0, -0.1, 0), SIMD4(0, 0, -0.1, 1)
        ))
        let parallel = try LineGeometry.cameraFrustum(projection: orthographic)
        #expect(parallel.segments.count == 25)
        let origins = [12, 13, 14, 15, 20].map { parallel.positions[parallel.segments[$0].start] }
        #expect(origins.allSatisfy { simd_length($0) < 1e-6 })
        #expect(simd_distance(parallel.positions[parallel.segments[20].end], SIMD3(0, 0, -1)) < 1e-5)
    }

    @Test
    func `Light, plane and skeleton helpers place their geometry`() throws {
        let spot = try PunctualLight(kind: .spot(range: 4, innerConeAngle: 0.2, outerConeAngle: 0.5), color: SIMD3(1, 0.5, 0))
        let cone = try LineGeometry.light(spot)
        // Axis, four generators and two 32-segment rims.
        #expect(cone.segments.count == 5 + 64)
        let rim = cone.positions.filter { abs($0.z + 4) < 1e-5 }.map { simd_length(SIMD2($0.x, $0.y)) }
        #expect(rim.contains { abs($0 - 4 * tan(0.5)) < 1e-4 })
        #expect(rim.contains { abs($0 - 4 * tan(0.2)) < 1e-4 })
        #expect(cone.colors?.allSatisfy { $0 == SIMD4(1, 0.5, 0, 1) } == true)
        let point = try LineGeometry.light(try PunctualLight(kind: .point()), size: 0.5)
        #expect(point.positions.allSatisfy { abs(simd_length($0) - 0.5) < 1e-5 })
        let rectangle = try LineGeometry.light(try PunctualLight(kind: .rectangle(width: 2, height: 1)))
        #expect(rectangle.positions.contains(SIMD3(1, 0.5, 0)) && rectangle.positions.contains(SIMD3(0, 0, -0.5)))

        // The plane y = 2 (normal +Y, constant −2): the Three.js strip's seven segments,
        // the four sides (one twice) and both diagonals, all on the plane.
        let plane = try LineGeometry.plane(normal: SIMD3(0, 1, 0), constant: -2, size: 4)
        #expect(plane.segments.count == 7)
        for position in plane.positions {
            let corner = SIMD3<Float>(abs(position.x), position.y, abs(position.z))
            #expect(simd_distance(corner, SIMD3(2, 2, 2)) < 1e-5)
        }
        let lengths = segments(plane).map { simd_distance($0.0, $0.1) }.sorted()
        #expect(lengths.prefix(5).allSatisfy { abs($0 - 4) < 1e-5 })
        let diagonal = Float(4 * 2.0.squareRoot())
        #expect(lengths.suffix(2).allSatisfy { abs($0 - diagonal) < 1e-5 })

        // Joints two levels apart still connect; a non-joint parent is skipped.
        let hand = SceneNode(transform: SceneTransform(translation: SIMD3(0, 1, 0)))
        let wrist = SceneNode(transform: SceneTransform(translation: SIMD3(0, 1, 0)), children: [hand])
        let shoulder = SceneNode(transform: SceneTransform(translation: SIMD3(2, 0, 0)), children: [wrist])
        let graph = SceneGraph(roots: [shoulder], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let bones = try #require(try LineGeometry.skeleton(in: graph, joints: [shoulder.id, hand.id]))
        // As in Three.js: the joint first and blue, then its ancestor in green.
        #expect(bones.positions == [SIMD3(2, 2, 0), SIMD3(2, 0, 0)])
        #expect(bones.colors == [SIMD4(0, 0, 1, 1), SIMD4(0, 1, 0, 1)])
        #expect(try LineGeometry.skeleton(in: graph, joints: [wrist.id]) == nil)
    }

    @Test
    func `BoxHelper bounds a subtree's meshes, instances and primitives in world space`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: .cube())
        let points = try PointsRenderer(
            device: device, cloud: try PointCloud(positions: [SIMD3(0, 5, 0)]), material: PointsMaterial(size: .world(1))
        )
        let child = SceneNode(transform: SceneTransform(translation: SIMD3(4, 0, 0)), primitive: .points(points))
        let parent = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -1), scale: SIMD3(repeating: 2)), mesh: mesh, children: [child]
        )
        let graph = SceneGraph(roots: [parent], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let helper = try #require(try LineGeometry.boundingBox(of: parent.id, in: graph))
        // The cube spans ±1 around (0, 0, −1); the point sits at (8, 10, −1) with radius 0.5.
        let lower = helper.positions.reduce(SIMD3<Float>(repeating: .infinity)) { simd_min($0, $1) }
        let upper = helper.positions.reduce(SIMD3<Float>(repeating: -.infinity)) { simd_max($0, $1) }
        #expect(lower == SIMD3(-1, -1, -2))
        #expect(upper == SIMD3(8.5, 10.5, 0))
        let rolled = SceneGraph(
            roots: [parent], camera: graph.camera,
            cameraTransform: SceneTransform(rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1)))
        )
        let rotated = try #require(try LineGeometry.boundingBox(of: parent.id, in: rolled))
        let rotatedUpper = rotated.positions.reduce(SIMD3<Float>(repeating: -.infinity)) { simd_max($0, $1) }
        let halfDiagonal = Float(0.5).squareRoot()
        #expect(simd_distance(rotatedUpper, SIMD3(8 + halfDiagonal, 10 + halfDiagonal, 0)) < 1e-5)
        #expect(try LineGeometry.boundingBox(of: UUID(), in: graph) == nil)
    }

    @Test
    func `Arrow and plane helpers render with reversed depth`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        func render(_ depthMode: MeshDepthMode) throws -> [SIMD4<Float>] {
            let helpers = DebugHelpers(device: fixture.device, depthMode: depthMode)
            let arrow = try helpers.arrow(
                direction: SIMD3(1, 1, 0), origin: SIMD3(-1.5, 0, -5), length: 1.5, color: SIMD3(0, 1, 1),
                headLength: 0.4, headWidth: 0.4
            )
            let plane = try helpers.plane(normal: SIMD3(0, 0, 1), constant: 6, size: 3, color: SIMD3(1, 0, 1))
            return try fixture.render(fixture.graph([arrow, plane]), depthMode: depthMode, name: "helpers-\(depthMode)")
        }
        let forward = try render(.forward)
        let reversed = try render(.reversed)
        let tip = fixture.pixel(SIMD3(-1.5, 0, -5) + simd_normalize(SIMD3<Float>(1, 1, 0)) * 1.2)
        let head = reversed[Int(tip.y) * 128 + Int(tip.x)]
        #expect(head.x < 0.01 && head.y > 0.9 && head.z > 0.9, "\(head)")
        // The plane's 20% magenta fill shows between its diagonals, 10 pixels below the center.
        #expect(isNearly(reversed[74 * 128 + 64], SIMD4(0.2, 0, 0.2, 0.2)), "\(reversed[74 * 128 + 64])")
        #expect(zip(forward, reversed).allSatisfy { isNearly($0, $1) })
    }

    @Test
    func `Helper nodes render through the scene graph`() throws {
        let fixture = try PrimitiveRenderFixture(side: 128)
        let helpers = DebugHelpers(device: fixture.device)
        let grid = try helpers.node(try LineGeometry.grid(size: 8, divisions: 8), transform: SceneTransform(translation: SIMD3(0, -1, -5)))
        let axes = try helpers.node(try LineGeometry.axes(size: 1), transform: SceneTransform(translation: SIMD3(0, -1, -5)))
        let arrow = try helpers.arrow(
            direction: SIMD3(1, 1, 0), origin: SIMD3(-1.5, 0, -5), length: 1.5, color: SIMD3(0, 1, 1),
            headLength: 0.4, headWidth: 0.4
        )
        let plane = try helpers.plane(normal: SIMD3(0, 0, 1), constant: 6, size: 3, color: SIMD3(1, 0, 1))
        let result = try fixture.render(fixture.graph([grid, axes, arrow, plane]), name: "helpers")
        #expect(result.statistics.drawnPrimitiveCount == 4)
        #expect(result.statistics.drawnMeshCount == 2)
        // The arrow head's cyan cone sits at the arrow's tip.
        // 1.2 units along, the head is 0.3 units wide: about 4 pixels.
        let tip = fixture.pixel(SIMD3(-1.5, 0, -5) + simd_normalize(SIMD3<Float>(1, 1, 0)) * 1.2)
        let head = result.pixels[Int(tip.y) * 128 + Int(tip.x)]
        #expect(head.x < 0.01 && head.y > 0.9 && head.z > 0.9, "\(head)")
        // The X axis line starts red at the grid's center.
        let origin = fixture.pixel(SIMD3(0.1, -1, -5))
        let red = result.pixels[Int(origin.y) * 128 + Int(origin.x)]
        #expect(red.x > 0.9 && red.y < 0.2, "\(red)")
    }
}
