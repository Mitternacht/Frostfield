import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PrimitivePickingTests {
    private let fixture: PrimitiveRenderFixture
    private let size = SIMD2<Float>(64, 64)

    init() throws {
        fixture = try PrimitiveRenderFixture(side: 64)
    }

    private func normalized(_ point: SIMD3<Float>, offset: SIMD2<Float> = .zero) -> SIMD2<Float> {
        (fixture.pixel(point) + offset) / size
    }

    private func pick(
        _ roots: [SceneNode], at point: SIMD3<Float>, offset: SIMD2<Float> = .zero,
        tolerance: PrimitivePickingTolerance = .pixels(0)
    ) throws -> PrimitiveHit? {
        try fixture.graph(roots).pickPrimitive(at: normalized(point, offset: offset), drawableSize: size, tolerance: tolerance)
    }

    @Test
    func `Picking returns the point under the cursor within its drawn size plus tolerance`() throws {
        // A 5 × 5 grid at depth 6, 1.5 units (8 pixels) apart.
        let positions: [SIMD3<Float>] = (0..<25).map { index in
            let column = Float(index % 5 - 2)
            let row = Float(index / 5 - 2)
            return SIMD3(column * 1.5, row * 1.5, -6)
        }
        let renderer = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: positions), material: PointsMaterial(size: .pixels(4))
        )
        let node = SceneNode(primitive: .points(renderer))
        let hit = try #require(try pick([node], at: positions[7]))
        #expect(hit.element == PrimitiveHit.Element.point(7))
        #expect(hit.nodeID == node.id)
        #expect(simd_distance(hit.position, positions[7]) < 1e-5)
        // The 4-pixel square reaches 2 pixels from the center; one more pixel of tolerance reaches 3.
        #expect(try pick([node], at: positions[7], offset: SIMD2(2.5, 0), tolerance: .pixels(1))?.element == .point(7))
        #expect(try pick([node], at: positions[7], offset: SIMD2(3.5, 0), tolerance: .pixels(1)) == nil)
        // Squares reach their corners; discs do not.
        #expect(try pick([node], at: positions[7], offset: SIMD2(1.9, 1.9))?.element == .point(7))
        let discs = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: positions),
            material: PointsMaterial(size: .pixels(4), shape: .circle)
        )
        #expect(try pick([SceneNode(primitive: .points(discs))], at: positions[7], offset: SIMD2(1.9, 1.9)) == nil)
    }

    @Test
    func `World thresholds and distances follow the Raycaster convention`() throws {
        let points = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0, 0, -8), SIMD3(0, 0, -4)]),
            material: PointsMaterial(size: .world(0.2))
        )
        let node = SceneNode(primitive: .points(points))
        // The nearer point wins; distance runs from the ray's origin on the near plane.
        let hit = try #require(try pick([node], at: SIMD3(0, 0, -4)))
        #expect(hit.element == .point(1))
        #expect(abs(hit.distance - 3.9) < 1e-5)
        // Radius 0.1 plus a 0.1 threshold reaches 0.2 units at depth 4 (1.6 pixels).
        let pixel = fixture.pixelsPerUnit / 4
        #expect(try pick([node], at: SIMD3(0, 0, -4), offset: SIMD2(0.15 * pixel, 0), tolerance: .world(0.1)) != nil)
        #expect(try pick([node], at: SIMD3(0, 0, -4), offset: SIMD2(0.25 * pixel, 0), tolerance: .world(0.1)) == nil)
        // World units shrink on screen with depth: 2 pixels at depth 8 is 0.5 units, so the
        // far point (radius 0.1, drawn at the one-pixel minimum) needs a threshold near 0.4.
        let farOnly = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0, 0, -8)]), material: PointsMaterial(size: .world(0.2))
        )
        let farNode = SceneNode(primitive: .points(farOnly))
        #expect(try pick([farNode], at: SIMD3(0, 0, -8), offset: SIMD2(2, 0), tolerance: .world(0.2)) == nil)
        #expect(try pick([farNode], at: SIMD3(0, 0, -8), offset: SIMD2(2, 0), tolerance: .world(0.4)) != nil)
        // A unit-wide point at depth 8 is drawn 4 pixels wide, so its edge is 2 pixels out.
        let wide = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0, 0, -8)]), material: PointsMaterial(size: .world(1))
        )
        let wideNode = SceneNode(primitive: .points(wide))
        #expect(try pick([wideNode], at: SIMD3(0, 0, -8), offset: SIMD2(1.9, 0)) != nil)
        #expect(try pick([wideNode], at: SIMD3(0, 0, -8), offset: SIMD2(2.1, 0)) == nil)
    }

    @Test
    func `Picking a polyline returns the segment index and its parameter`() throws {
        let positions: [SIMD3<Float>] = [SIMD3(-3, -1, -5), SIMD3(-1, 1, -5), SIMD3(1, -1, -5), SIMD3(3, 1, -5), SIMD3(4, -1, -5)]
        let line = try LineRenderer(
            device: fixture.device, geometry: try LineGeometry(positions: positions), material: LineMaterial(width: .pixels(2))
        )
        let node = SceneNode(primitive: .lines(line))
        let midpoint = (positions[2] + positions[3]) / 2
        let hit = try #require(try pick([node], at: midpoint))
        #expect(hit.element == .segment(2))
        #expect(abs((hit.segmentParameter ?? 0) - 0.5) < 1e-4)
        #expect(hit.distanceToRay < 1e-4)
        #expect(simd_distance(hit.position, midpoint) < 1e-4)
        // Perpendicular to the segment, the 1-pixel half width plus 1 pixel of tolerance reaches 2 pixels.
        let normal = simd_normalize(SIMD2<Float>(1, 1))
        #expect(try pick([node], at: midpoint, offset: normal * 1.9, tolerance: .pixels(1))?.element == .segment(2))
        #expect(try pick([node], at: midpoint, offset: normal * 2.1, tolerance: .pixels(1)) == nil)
        // A world width is measured in world units at any depth.
        let wide = try LineRenderer(
            device: fixture.device, geometry: try LineGeometry(positions: positions), material: LineMaterial(width: .world(0.4))
        )
        let wideNode = SceneNode(primitive: .lines(wide))
        let unit = fixture.pixelsPerUnit / 5
        #expect(try pick([wideNode], at: midpoint, offset: normal * 0.19 * unit)?.element == .segment(2))
        #expect(try pick([wideNode], at: midpoint, offset: normal * 0.21 * unit) == nil)
    }

    @Test
    func `Sprite picking follows rotation, anchor and pixel sizing`() throws {
        // Rotated a quarter turn, a 2 × 0.5 sprite stands upright.
        let sprite = Sprite(position: SIMD3(0, 0, -5), size: SIMD2(2, 0.5), rotation: .pi / 2)
        let renderer = try SpriteRenderer(device: fixture.device, sprites: [sprite, Sprite(position: SIMD3(3, 0, -5), size: SIMD2(1, 1))])
        let node = SceneNode(primitive: .sprites(renderer))
        let unit = fixture.pixelsPerUnit / 5
        #expect(try pick([node], at: SIMD3(0, 0, -5), offset: SIMD2(0, 0.8 * unit))?.element == .sprite(0))
        #expect(try pick([node], at: SIMD3(0, 0, -5), offset: SIMD2(0.8 * unit, 0)) == nil)
        #expect(try pick([node], at: SIMD3(3, 0, -5))?.element == .sprite(1))

        let pixels = try SpriteRenderer(
            device: fixture.device, sprites: [Sprite(position: SIMD3(0, 0, -9), size: SIMD2(20, 10), center: .zero)],
            material: SpriteMaterial(sizeAttenuation: false)
        )
        let pixelNode = SceneNode(primitive: .sprites(pixels))
        // A top-left anchor: the sprite covers 0…20 pixels right and 0…10 down from its position.
        #expect(try pick([pixelNode], at: SIMD3(0, 0, -9), offset: SIMD2(19, 9)) != nil)
        #expect(try pick([pixelNode], at: SIMD3(0, 0, -9), offset: SIMD2(-1, 5)) == nil)
        #expect(try pick([pixelNode], at: SIMD3(0, 0, -9), offset: SIMD2(5, -1)) == nil)
        #expect(try pick([pixelNode], at: SIMD3(0, 0, -9), offset: SIMD2(21, 5)) == nil)
    }

    @Test
    func `Primitive picking validates its inputs`() throws {
        let graph = fixture.graph([])
        #expect(throws: PrimitivePickingError.drawableSize) {
            try graph.pickPrimitive(at: SIMD2(0.5, 0.5), drawableSize: SIMD2(0, 10))
        }
        #expect(throws: PrimitivePickingError.tolerance) {
            try graph.pickPrimitive(at: SIMD2(0.5, 0.5), drawableSize: size, tolerance: .pixels(-1))
        }
        #expect(try graph.pickPrimitive(at: SIMD2(0.5, 0.5), drawableSize: size) == nil)
    }

    @Test
    func `Picking a receding segment reports its perspective-correct parameter`() throws {
        // From depth 2 to 10 at x = −1: halfway in world space is 83% of the way on screen.
        let line = try LineRenderer(
            device: fixture.device, geometry: try LineGeometry(positions: [SIMD3(-1, 0, -2), SIMD3(-1, 0, -10)]),
            material: LineMaterial(width: .pixels(4))
        )
        let node = SceneNode(primitive: .lines(line))
        for depth: Float in [3, 6, 9] {
            let target = SIMD3<Float>(-1, 0, -depth)
            let hit = try #require(try pick([node], at: target))
            #expect(hit.element == .segment(0))
            #expect(abs((hit.segmentParameter ?? -1) - (depth - 2) / 8) < 1e-3, "depth \(depth): \(hit.segmentParameter ?? -1)")
            #expect(simd_distance(hit.position, target) < 1e-3)
            #expect(abs(hit.distance - (depth - 0.1) * simd_length(target / depth)) < 1e-3)
        }
    }

    @Test
    func `A unified ray returns mixed drawables in stable distance order`() throws {
        let pointRenderer = try PointsRenderer(
            device: fixture.device, cloud: PointCloud(positions: [SIMD3(0, 0, -8), SIMD3(0, 0, -2), SIMD3(0, 0, -5)]),
            material: PointsMaterial(size: .pixels(2))
        )
        let points = SceneNode(primitive: .points(pointRenderer))
        let lineRenderer = try LineRenderer(
            device: fixture.device, geometry: LineGeometry(positions: [SIMD3(-1, 0, -3), SIMD3(1, 0, -3)]),
            material: LineMaterial(width: .pixels(2))
        )
        let line = SceneNode(primitive: .lines(lineRenderer))
        let spriteRenderer = try SpriteRenderer(device: fixture.device, sprites: [
            Sprite(position: SIMD3(0, 0, -4), size: SIMD2(2, 2), textureRegion: SIMD4(0.25, 0.5, 0.5, 0.25)),
        ])
        let sprite = SceneNode(primitive: .sprites(spriteRenderer))
        let vertices = [SIMD3<Float>(-1, -1, 0), SIMD3(1, -1, 0), SIMD3(0, 1, 0)].map {
            MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
        }
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: MeshGeometry(vertices: vertices, indices: [0, 1, 2])
        )
        let meshNode = SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: mesh)
        let hidden = SceneNode(children: [SceneNode(primitive: .points(pointRenderer))], isVisible: false)
        let graph = fixture.graph([meshNode, sprite, points, line, hidden])
        let center = SIMD2<Float>(repeating: 0.5)
        let hits = try graph.raycast(at: center, drawableSize: size, tolerance: .pixels(0))
        #expect(hits.map(\.nodeID) == [points.id, line.id, sprite.id, meshNode.id, points.id, points.id])
        #expect(zip(hits.map(\.distance), [1.9, 2.9, 3.9, 4.9, 4.9, 7.9]).allSatisfy { abs($0.0 - $0.1) < 1e-5 })
        #expect(hits[2].textureCoordinate == SIMD2(0.5, 0.625))
        #expect(hits[1].textureCoordinate == nil)
        #expect(try graph.pick(at: center, drawableSize: size, tolerance: .pixels(0)) == hits.first)
    }

    @Test(arguments: [MeshDepthMode.forward, .reversed])
    func `An orthographic ray uses constant world scale and clips receding lines`(mode: MeshDepthMode) throws {
        let pointRenderer = try PointsRenderer(
            device: fixture.device, cloud: PointCloud(positions: [SIMD3(0, 0, -3), SIMD3(0, 0, -6), SIMD3(0, 0, -7)]),
            material: PointsMaterial(size: .world(0.5))
        )
        let points = SceneNode(primitive: .points(pointRenderer))
        let lineRenderer = try LineRenderer(
            device: fixture.device, geometry: LineGeometry(positions: [SIMD3(-1, 0, -2), SIMD3(1, 0, -10)]),
            material: LineMaterial(width: .pixels(2))
        )
        let line = SceneNode(primitive: .lines(lineRenderer))
        let graph = fixture.graph([points, line])
        let depthScale: Float = mode == .forward ? -0.2 : 0.2
        let depthOffset: Float = mode == .forward ? -0.2 : 1.2
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: simd_float4x4(columns: (
                SIMD4(0.5, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, depthScale, 0), SIMD4(0, 0, depthOffset, 1)
            )), near: 1, far: 6
        )
        let viewport = SIMD2<Float>(160, 80)
        let hits = try graph.raycast(at: SIMD2(0.5, 0.5), drawableSize: viewport, tolerance: .pixels(0),
                                     camera: camera, depthMode: mode)
        #expect(hits.map(\.nodeID) == [points.id, points.id, line.id])
        guard case .primitive(let segment)? = hits.last else {
            Issue.record("Expected the line hit at the far clip plane")
            return
        }
        #expect(abs((segment.segmentParameter ?? 0) - 0.5) < 1e-6)
        #expect(abs(segment.distance - 5) < 1e-6)
        let inside = try graph.raycast(at: SIMD2(0.55, 0.55), drawableSize: viewport, tolerance: .pixels(0),
                                       camera: camera, depthMode: mode)
        #expect(inside.map(\.nodeID) == [points.id, points.id])
        #expect(try graph.raycast(at: SIMD2(0.57, 0.5), drawableSize: viewport, tolerance: .pixels(0),
                                  camera: camera, depthMode: mode).isEmpty)
    }

    @Test
    func `A grouped sprite handler receives atlas UV through an off-axis view`() throws {
        let renderer = try SpriteRenderer(
            device: fixture.device,
            sprites: [Sprite(position: SIMD3(0, 0, -2), size: SIMD2(16, 16), textureRegion: SIMD4(0.25, 0.5, 0.5, 0.25))],
            material: SpriteMaterial(sizeAttenuation: false)
        )
        let sprite = SceneNode(primitive: .sprites(renderer))
        let groupNode = SceneNode(children: [sprite])
        let graph = fixture.graph([groupNode])
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: simd_float4x4(columns: (
                SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0.25, 0, -1, -1), SIMD4(0, 0, -0.5, 0)
            )), near: 0.5, far: .infinity
        )
        var group = SceneInteractionGroup(nodeID: groupNode.id)
        var events: [SceneInteractionGroup.DrawableEvent] = []
        group.setDrawableHandler(for: sprite.id) { events.append($0) }
        let hit = try group.dispatch(.click, at: SIMD2(0.4375, 0.5), drawableSize: size, camera: camera, in: graph)
        #expect(hit?.nodeID == sprite.id)
        #expect(hit?.textureCoordinate == SIMD2(0.625, 0.625))
        #expect(events.map(\.phase) == [.click])
        #expect(events.first?.hit == hit)
        group.setDrawableHandler(for: sprite.id, nil)
        #expect(try group.dispatch(.move, at: SIMD2(0.4375, 0.5), drawableSize: size, camera: camera, in: graph) == hit)
        #expect(events.count == 1)
    }
}
