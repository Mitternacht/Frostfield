import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PrimitiveSceneGraphTests {
    private let fixture: PrimitiveRenderFixture

    init() throws {
        fixture = try PrimitiveRenderFixture(side: 64)
    }

    private func sprite(
        at position: SIMD3<Float>, color: SIMD4<Float>, blending: PrimitiveBlending = .alpha, depthTest: Bool = true
    ) throws -> SceneNode {
        let renderer = try SpriteRenderer(
            device: fixture.device, sprites: [Sprite(position: position, size: SIMD2(4, 4) * -position.z / 8)],
            material: SpriteMaterial(color: color, blending: blending, depthTest: depthTest)
        )
        return SceneNode(primitive: .sprites(renderer))
    }

    @Test
    func `Frustum culling skips primitives by their world bounds and counts them`() throws {
        let points = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0, 0, 5)]), material: PointsMaterial()
        )
        let behind = SceneNode(primitive: .points(points))
        let visible = try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 1, 1, 1))
        var statistics = try fixture.render(fixture.graph([behind, visible])).statistics
        #expect(statistics.culledPrimitiveCount == 1)
        #expect(statistics.drawnPrimitiveCount == 1)
        #expect(statistics.submittedSpriteCount == 1 && statistics.submittedTriangleCount == 2)
        #expect(statistics.submittedPointCount == 0)
        let line = try LineRenderer(device: fixture.device, geometry: LineGeometry(positions: [
            SIMD3(-1, -0.5, -4), SIMD3(0, 0, -4), SIMD3(1, -0.5, -4),
        ]))
        statistics = try fixture.render(fixture.graph([
            SceneNode(primitive: .points(points), frustumCulled: false), visible, SceneNode(primitive: .lines(line)),
        ])).statistics
        #expect(statistics.culledPrimitiveCount == 0 && statistics.drawnPrimitiveCount == 3)
        #expect(statistics.submittedPointCount == 1 && statistics.submittedLineSegmentCount == 2)
        #expect(statistics.submittedSpriteCount == 1 && statistics.submittedTriangleCount == 2)

        // A world-sized point centered just outside the left plane still reaches into view.
        let edge = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(-4.2, 0, -4)]), material: PointsMaterial(size: .world(1))
        )
        let result = try fixture.render(fixture.graph([SceneNode(primitive: .points(edge))]))
        #expect(result.statistics.drawnPrimitiveCount == 1)
        #expect(fixture.alpha(result.pixels, x: 0, y: 32) == 1)
    }

    @Test
    func `Blended primitives composite back to front regardless of node order`() throws {
        let near = try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 0, 0, 0.5))
        let far = try sprite(at: SIMD3(0, 0, -8), color: SIMD4(0, 0, 1, 0.5))
        // Red over blue: 0.5 red + 0.5 × 0.5 blue, with alpha 0.75.
        let expected = SIMD4<Float>(0.5, 0, 0.25, 0.75)
        for roots in [[near, far], [far, near]] {
            let pixels = try fixture.render(fixture.graph(roots)).pixels
            #expect(isNearly(pixels[32 * 64 + 32], expected), "\(pixels[32 * 64 + 32])")
        }
    }

    @Test
    func `Blended primitives interleave with blended meshes by depth`() throws {
        func glass(depth: Float) throws -> SceneNode {
            let material = try PhysicallyBasedMaterial(
                baseColorFactor: SIMD4(0, 1, 0, 0.5), alphaMode: .blend, shadingModel: .unlit
            )
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(2),
                material: .physicallyBased(material)
            )
            return SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -depth)), mesh: mesh)
        }
        // The sprite comes first in node order both times.
        let behind = try fixture.render(fixture.graph([try sprite(at: SIMD3(0, 0, -8), color: SIMD4(1, 0, 0, 0.5)), try glass(depth: 4)]), name: "interleave-behind").pixels
        #expect(isNearly(behind[32 * 64 + 32], SIMD4(0.25, 0.5, 0, 0.75)), "\(behind[32 * 64 + 32])")
        let front = try fixture.render(fixture.graph([try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 0, 0, 0.5)), try glass(depth: 8)])).pixels
        #expect(isNearly(front[32 * 64 + 32], SIMD4(0.5, 0.25, 0, 0.75)), "\(front[32 * 64 + 32])")
    }

    @Test
    func `Primitives share mesh render order and group order`() throws {
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(0, 1, 0, 0.5), alphaMode: .blend, shadingModel: .unlit
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: GeometryRenderFixture.square(2), material: .physicallyBased(material)
        )
        let far = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -8)), mesh: mesh)
        var near = try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 0, 0, 0.5))
        let pixel = 32 * 64 + 32
        let natural = try fixture.render(fixture.graph([near, far])).pixels
        #expect(isNearly(natural[pixel], SIMD4(0.5, 0.25, 0, 0.75)))

        near.renderOrder = -1
        let explicit = try fixture.render(fixture.graph([far, near])).pixels
        #expect(isNearly(explicit[pixel], SIMD4(0.25, 0.5, 0, 0.75)))

        near.renderOrder = 100
        var group = SceneNode(children: [near])
        group.renderOrder = -2
        let inherited = try fixture.render(fixture.graph([far, group])).pixels
        #expect(isNearly(inherited[pixel], SIMD4(0.25, 0.5, 0, 0.75)))

        let opaqueMesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: GeometryRenderFixture.square(2), material: .unlit,
            renderState: MeshRenderState(depthCompare: .lessEqual)
        )
        let green = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -4)), mesh: opaqueMesh,
            materialParameters: SIMD4(0, 1, 0, 1)
        )
        let red = try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 0, 0, 1), blending: .opaque)
        let meshLast = try fixture.render(fixture.graph([red, green])).pixels
        let primitiveLast = try fixture.render(fixture.graph([green, red])).pixels
        #expect(isNearly(meshLast[pixel], GeometryRenderFixture.unlitFacingCamera(SIMD4(0, 1, 0, 1))))
        #expect(isNearly(primitiveLast[pixel], SIMD4(1, 0, 0, 1)))
    }

    @Test
    func `Transparent comparisons and traversal sorting reorder both meshes and sprites`() throws {
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(2),
            material: .physicallyBased(try .init(
                baseColorFactor: SIMD4(0, 1, 0, 0.5), alphaMode: .blend, shadingModel: .unlit
            ))
        )
        let near = try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 0, 0, 0.5))
        let far = SceneNode(transform: .init(translation: SIMD3(0, 0, -8)), mesh: mesh)
        let graph = fixture.graph([near, far])
        let pixel = 32 * 64 + 32
        let natural = try fixture.render(graph).pixels
        #expect(isNearly(natural[pixel], SIMD4(0.5, 0.25, 0, 0.75)))

        var seen = Set<UUID>()
        let byNearDepth = SceneRenderOrdering(transparent: { a, b in
            seen.formUnion([a.nodeID, b.nodeID])
            guard let first = a.projectedDepth, let second = b.projectedDepth else {
                Issue.record("Expected finite projected depths")
                return .orderedSame
            }
            if first < second { return .orderedAscending }
            if first > second { return .orderedDescending }
            return .orderedSame
        })
        let nearFirst = try fixture.render(graph, depthMode: .forward, ordering: byNearDepth)
        #expect(seen == [near.id, far.id])
        let expected = SIMD4<Float>(0.25, 0.5, 0, 0.75)
        #expect(isNearly(nearFirst[pixel], expected))

        let disabled = SceneRenderOrdering(sortObjects: false, transparent: { _, _ in
            Issue.record("Disabled sorting invoked a comparator")
            return .orderedDescending
        })
        let inTraversal = try fixture.render(graph, depthMode: .forward, ordering: disabled)
        #expect(isNearly(inTraversal[pixel], expected))
        let tied = try fixture.render(graph, depthMode: .forward, ordering: .init(transparent: { _, _ in .orderedSame }))
        #expect(isNearly(tied[pixel], expected))
    }

    @Test
    func `Opaque comparisons receive draw metadata and can move a sprite after a mesh`() throws {
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: GeometryRenderFixture.square(2), material: .unlit,
            renderState: MeshRenderState(depthCompare: .lessEqual)
        )
        let green = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4)), mesh: mesh, materialParameters: SIMD4(0, 1, 0, 1)
        )
        var red = try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 0, 0, 1), blending: .opaque)
        red.renderOrder = -2
        let graph = fixture.graph([red, green])
        let natural = try fixture.render(graph).pixels
        let pixel = 32 * 64 + 32
        #expect(isNearly(natural[pixel], GeometryRenderFixture.unlitFacingCamera(SIMD4(0, 1, 0, 1))))

        let base = fixture.base
        let target = try base.makeTarget()
        let buffer = try #require(base.queue.makeCommandBuffer())
        let byKind = SceneRenderOrdering(opaque: { a, b in
            #expect(a.projectedDepth == b.projectedDepth && a.projectedDepth != nil)
            if a.nodeID == red.id { #expect(a.kind == .sprites && a.renderOrder == -2) }
            if b.nodeID == red.id { #expect(b.kind == .sprites && b.renderOrder == -2) }
            if a.kind == b.kind { return .orderedSame }
            return a.kind == .mesh ? .orderedAscending : .orderedDescending
        })
        try graph.encode(pass: base.pass(color: target.color, depth: target.depth), commandBuffer: buffer, ordering: byKind)
        try base.finish(buffer)
        #expect(try isNearly(base.pixels(of: target.color)[pixel], SIMD4(1, 0, 0, 1)))
    }

    @Test
    func `Opaque primitives write depth and overlays draw over everything`() throws {
        let near = try sprite(at: SIMD3(0, 0, -4), color: SIMD4(1, 0, 0, 1), blending: .opaque)
        let far = try sprite(at: SIMD3(0, 0, -8), color: SIMD4(0, 0, 1, 1), blending: .opaque)
        // The nearer sprite wins even though it draws first.
        let pixels = try fixture.render(fixture.graph([near, far])).pixels
        #expect(isNearly(pixels[32 * 64 + 32], SIMD4(1, 0, 0, 1)))

        // Without depth testing, a sprite behind a mesh still shows on top.
        let wall = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(4), material: .unlit
        )
        let wallNode = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -3)), mesh: wall, materialParameters: SIMD4(0, 1, 0, 1))
        let hidden = try fixture.render(fixture.graph([far, wallNode])).pixels
        #expect(hidden[32 * 64 + 32].z == 0)
        let overlay = try sprite(at: SIMD3(0, 0, -8), color: SIMD4(0, 0, 1, 1), blending: .opaque, depthTest: false)
        let shown = try fixture.render(fixture.graph([overlay, wallNode])).pixels
        #expect(isNearly(shown[32 * 64 + 32], SIMD4(0, 0, 1, 1)))
    }

    @Test
    func `Node material parameters tint primitives and reversed depth matches forward`() throws {
        let line = try LineRenderer(
            device: fixture.device, geometry: try LineGeometry(positions: [SIMD3(-2, 0, -5), SIMD3(2, 0, -5)]),
            material: LineMaterial(width: .pixels(4), blending: .opaque)
        )
        let points = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0, 1, -5)]),
            material: PointsMaterial(size: .world(0.5), shape: .circle, blending: .alpha)
        )
        let graph = fixture.graph([
            SceneNode(primitive: .lines(line), materialParameters: SIMD4(0.5, 0.25, 1, 1)),
            SceneNode(primitive: .points(points), materialParameters: SIMD4(1, 1, 0, 1)),
            try sprite(at: SIMD3(1, -1, -6), color: SIMD4(0, 1, 1, 0.5)),
        ])
        let forward = try fixture.render(graph, depthMode: .forward)
        #expect(isNearly(forward[31 * 64 + 20], SIMD4(0.5, 0.25, 1, 1)))
        let reversed = try fixture.render(graph, depthMode: .reversed)
        #expect(zip(forward, reversed).allSatisfy { isNearly($0, $1) })
        #expect(forward.contains { isNearly($0, SIMD4(1, 1, 0, 1)) })
    }

    @Test
    func `World sizes stay constant with depth in an orthographic view`() throws {
        // A view 8 units tall on a 64-pixel target: 8 pixels per world unit at any depth.
        let near: Float = 0.1
        let far: Float = 50
        let orthographic = simd_float4x4(columns: (
            SIMD4(0.25, 0, 0, 0), SIMD4(0, 0.25, 0, 0), SIMD4(0, 0, -1 / (far - near), 0), SIMD4(0, 0, -near / (far - near), 1)
        ))
        let points = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(-2, 0, -4), SIMD3(2, 0, -40)]),
            material: PointsMaterial(size: .world(1))
        )
        let line = try LineRenderer(
            device: fixture.device, geometry: try LineGeometry(positions: [SIMD3(-3, 2, -2), SIMD3(3, 2, -30)]),
            material: LineMaterial(width: .world(0.5), cap: .butt, blending: .opaque)
        )
        let graph = fixture.graph([SceneNode(primitive: .points(points)), SceneNode(primitive: .lines(line))])
        let target = try fixture.base.makeTarget()
        let buffer = try #require(fixture.base.queue.makeCommandBuffer())
        try graph.encode(
            pass: fixture.base.pass(color: target.color, depth: target.depth), commandBuffer: buffer,
            clipFromWorld: orthographic, cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1)
        )
        try fixture.base.finish(buffer)
        let pixels = try fixture.base.pixels(of: target.color)
        // Each point is an 8 × 8 square centered at x = 32 ∓ 16, y = 32.
        let left = (28..<36).flatMap { y in (12..<20).map { x in pixels[y * 64 + x].w } }
        let right = (28..<36).flatMap { y in (44..<52).map { x in pixels[y * 64 + x].w } }
        #expect(left.allSatisfy { $0 == 1 } && right.allSatisfy { $0 == 1 })
        #expect(pixels[27 * 64 + 16].w == 0 && pixels[32 * 64 + 20].w == 0)
        // The receding line is 4 pixels tall at both ends, centered on y = 32 − 16.
        for x in [10, 54] {
            let rows = (0..<64).filter { pixels[$0 * 64 + x].w > 0 }
            #expect(rows == Array(14..<18), "column \(x) lit rows \(rows)")
        }
    }

    @Test
    func `Primitives follow a viewport's origin and size`() throws {
        // A 32-pixel view whose top-left corner sits at (16, 8) in the 64-pixel target.
        let viewport = MTLViewport(originX: 16, originY: 8, width: 32, height: 32, znear: 0, zfar: 1)
        // At depth 4 the view spans ±4 units: 4 pixels per unit about its center (32, 24).
        let points = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0.5, 0.25, -4)]),
            material: PointsMaterial(size: .pixels(2))
        )
        let line = try LineRenderer(
            device: fixture.device, geometry: try LineGeometry(positions: [SIMD3(-1, -0.5, -4), SIMD3(1, -0.5, -4)]),
            material: LineMaterial(width: .pixels(2), cap: .butt, blending: .opaque)
        )
        let graph = fixture.graph([SceneNode(primitive: .points(points)), SceneNode(primitive: .lines(line))])
        for depthMode in [MeshDepthMode.forward, .reversed] {
            let pixels = try fixture.render(graph, depthMode: depthMode, viewport: viewport, name: "viewport-\(depthMode)")
            let lit = Set(pixels.indices.filter { pixels[$0].w > 0 }.map { SIMD2($0 % 64, $0 / 64) })
            // The point lands at (34, 23): a 2-pixel square over columns 33…34 and rows 22…23.
            let point = Set((22...23).flatMap { y in (33...34).map { SIMD2($0, y) } })
            // The line runs from x = 28 to 36 at y = 26, two pixels tall.
            let stroke = Set((25...26).flatMap { y in (28...35).map { SIMD2($0, y) } })
            #expect(lit == point.union(stroke), "\(depthMode): \(lit.sorted { ($0.y, $0.x) < ($1.y, $1.x) })")
        }
    }

    @Test
    func `Opaque primitives are part of the backdrop seen through transmission`() throws {
        // A glass cube between the camera and an opaque red line.
        let glass = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .cube(),
            material: .physicallyBased(try PhysicallyBasedMaterial(alphaMode: .opaque, transmission: .init(factor: 1)))
        )
        let line = try LineRenderer(
            device: fixture.device, geometry: try LineGeometry(positions: [SIMD3(-3, 0, -6), SIMD3(3, 0, -6)]),
            material: LineMaterial(width: .pixels(8), color: SIMD4(1, 0, 0, 1), blending: .opaque)
        )
        let graph = fixture.graph([
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -3)), mesh: glass),
            SceneNode(primitive: .lines(line)),
        ])
        let target = try fixture.base.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 64, height: 64, mipmapped: true)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let backdrop = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let buffer = try #require(fixture.base.queue.makeCommandBuffer())
        try graph.encode(
            pass: fixture.base.pass(color: target.color, depth: target.depth), commandBuffer: buffer,
            transmissionBackdrop: backdrop
        )
        try fixture.base.finish(buffer)
        // The opaque pass drew the line before the backdrop snapshot.
        let snapshot = try fixture.base.pixels(of: backdrop)
        #expect(isNearly(snapshot[32 * 64 + 32], SIMD4(1, 0, 0, 1)), "\(snapshot[32 * 64 + 32])")
        // Through the glass the line still reads red.
        let pixels = try fixture.base.pixels(of: target.color)
        #expect(pixels[32 * 64 + 32].x > pixels[32 * 64 + 32].y + 0.4, "\(pixels[32 * 64 + 32])")
    }

    @Test
    func `Encoding rejects primitives prepared for another format or device state`() throws {
        let line = try LineRenderer(
            device: fixture.device, colorPixelFormat: .bgra8Unorm,
            geometry: try LineGeometry(positions: [SIMD3(-1, 0, -5), SIMD3(1, 0, -5)])
        )
        #expect(throws: MeshRenderer.Failure.attachments) {
            try fixture.render(fixture.graph([SceneNode(primitive: .lines(line))]))
        }
        let points = try PointsRenderer(
            device: fixture.device, cloud: try PointCloud(positions: [SIMD3(0, 0, -5)]), material: PointsMaterial()
        )
        #expect(throws: MeshRenderer.Failure.transform) {
            try fixture.render(fixture.graph([SceneNode(primitive: .points(points), materialParameters: SIMD4(.nan, 1, 1, 1))]))
        }
    }
}
