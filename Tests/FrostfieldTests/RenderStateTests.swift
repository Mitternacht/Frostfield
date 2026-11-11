import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Blending, depth, color writes, polygon offset, sides, sorting, alpha hashing and
/// alpha to coverage, each against the analytic result for known colors.
struct RenderStateTests {
    let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    func basic(
        _ color: SIMD3<Float>, opacity: Float = 1, state: MeshRenderState? = nil, halfExtent: Float = 0.5,
        depthMode: MeshDepthMode = .forward, sampleCount: Int = 1
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(halfExtent),
            material: .classic(try ClassicMaterial(model: .basic, color: color, opacity: opacity)),
            depthMode: depthMode, renderState: state, sampleCount: sampleCount
        )
    }

    func node(_ mesh: MeshRenderer, z: Float, renderOrder: Int = 0, rotation: simd_quatf? = nil) -> SceneNode {
        var node = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, z), rotation: rotation ?? simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))),
            mesh: mesh
        )
        node.renderOrder = renderOrder
        return node
    }

    @Test("Raster scissor clips meshes and points to the intersection with the viewport")
    func rasterScissor() throws {
        let red = try basic(SIMD3(1, 0, 0), halfExtent: 2)
        let cloud = try PointCloud(positions: [SIMD3<Float>(0, 0, -0.9)], colors: [SIMD4(0, 1, 0, 1)])
        let points = try PointsRenderer(
            device: fixture.device, cloud: cloud, material: PointsMaterial(size: .world(1))
        )
        let graph = SceneGraph(roots: [node(red, z: -1), SceneNode(primitive: .points(points))], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let clip = try fixture.camera.projection(aspectRatio: 1)
        let viewport = MTLViewport(originX: 8, originY: 10, width: 40, height: 42, znear: 0, zfar: 1)
        let scissor = MTLScissorRect(x: 19, y: 5, width: 24, height: 30)
        let draws = try graph.encode(
            pass: pass, commandBuffer: command, clipFromWorld: clip, cameraPosition: .zero,
            viewport: viewport, scissor: scissor
        )
        try fixture.finish(command)

        #expect(draws.drawnMeshCount == 1 && draws.drawnPrimitiveCount == 1)
        let pixels = try fixture.pixels(of: target.color)
        for y in 0..<fixture.side {
            for x in 0..<fixture.side {
                let inside = (19..<43).contains(x) && (10..<35).contains(y)
                #expect((pixels[y * fixture.side + x].w > 0) == inside, "pixel (\(x), \(y))")
            }
        }
        #expect(pixels.contains { $0.y > 0.5 })
        #expect(pixels.contains { $0.x > 0.5 })
    }

    @Test("Out-of-bounds scissors fail before encoding and an empty scissor preserves pass clears")
    func scissorValidation() throws {
        let graph = SceneGraph(roots: [node(try basic(SIMD3(1, 0, 0)), z: -1)], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.25, green: 0, blue: 0.5, alpha: 1)
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: MeshRenderer.Failure.scissor) {
            try graph.encode(
                pass: pass, commandBuffer: command,
                scissor: MTLScissorRect(x: fixture.side - 1, y: 0, width: Int.max, height: 1)
            )
        }
        let draws = try graph.encode(
            pass: pass, commandBuffer: command, scissor: MTLScissorRect(x: fixture.side, y: 0, width: 0, height: 5)
        )
        try fixture.finish(command)
        #expect(draws.drawnMeshCount == 0 && draws.drawnPrimitiveCount == 0)
        #expect(try fixture.pixels(of: target.color).allSatisfy { $0 == SIMD4(0.25, 0, 0.5, 1) })
    }

    /// The center pixel of a graph rendered over a clear color.
    func center(_ roots: [SceneNode], clear: SIMD4<Float> = .zero) throws -> SIMD4<Float> {
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.colorAttachments[0].clearColor = MTLClearColor(
            red: Double(clear.x), green: Double(clear.y), blue: Double(clear.z), alpha: Double(clear.w)
        )
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        try SceneGraph(roots: roots, camera: fixture.camera).encode(pass: pass, commandBuffer: buffer)
        try fixture.finish(buffer)
        return fixture.pixel(try fixture.pixels(of: target.color), x: 32, y: 32)
    }

    // MARK: Blending

    @Test("Each blend mode matches its premultiplied equation for known source and destination colors")
    func blendModes() throws {
        let destination = SIMD4<Float>(0.2, 0.5, 0.8, 0.6)
        let color = SIMD3<Float>(0.9, 0.4, 0.1)
        let alpha: Float = 0.7
        let source = color * alpha
        let d = SIMD3(destination.x, destination.y, destination.z)
        let constant = SIMD4<Float>(0.25, 0.5, 0.75, 0.4)
        let k = SIMD3(constant.x, constant.y, constant.z)
        let cases: [(MeshRenderState.Blending, SIMD4<Float>)] = [
            (.none, SIMD4(color, 1)),
            (.normal, SIMD4(source + d * (1 - alpha), alpha + destination.w * (1 - alpha))),
            (.additive, SIMD4(source + d, alpha + destination.w)),
            (.subtractive, SIMD4(d * (1 - source), destination.w)),
            (.multiply, SIMD4(source * d + d * (1 - alpha), destination.w)),
            (.custom(.init(sourceRGB: .one, destinationRGB: .one, rgbOperation: .reverseSubtract)),
             SIMD4(d - source, destination.w - alpha)),
            (.custom(.init(
                sourceRGB: .blendColor, destinationRGB: .oneMinusBlendColor,
                sourceAlpha: .blendAlpha, destinationAlpha: .oneMinusBlendAlpha, constantColor: constant
            )), SIMD4(source * k + d * (1 - k), alpha * constant.w + destination.w * (1 - constant.w))),
            (.custom(.init(sourceRGB: .one, destinationRGB: .one, rgbOperation: .max)), SIMD4(simd_max(source, d), max(alpha, destination.w))),
        ]
        for (blending, expected) in cases {
            let state = MeshRenderState(blending: blending, transparent: blending != .none, depthWrite: blending == .none)
            let mesh = try basic(color, opacity: alpha, state: state)
            let pixel = try center([node(mesh, z: -2)], clear: destination)
            #expect(isNearly(pixel, expected), "\(blending): \(pixel) vs \(expected)")
        }

        // Physically based materials take the same state; unlit PBR writes premultiplied base color.
        let pbr = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(0.5),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                baseColorFactor: SIMD4(color, alpha), alphaMode: .blend, shadingModel: .unlit
            )),
            renderState: MeshRenderState(blending: .additive, transparent: true, depthWrite: false)
        )
        let added = try center([node(pbr, z: -2)], clear: destination)
        #expect(isNearly(added, SIMD4(source + d, alpha + destination.w)), "\(added)")
    }

    // MARK: Depth and color writes

    @Test("Depth compare, depth test, depth writes and color write masks change which quad survives")
    func depthAndColorWrites() throws {
        let red = SIMD3<Float>(0.9, 0.1, 0.1)
        let green = SIMD3<Float>(0.1, 0.9, 0.1)
        let clear = SIMD4<Float>(0.05, 0.05, 0.3, 0.5)
        func scene(near: MeshRenderState? = nil, far: MeshRenderState? = nil) throws -> SIMD4<Float> {
            try center([node(try basic(red, state: near), z: -2), node(try basic(green, state: far), z: -3)], clear: clear)
        }
        #expect(isNearly(try scene(), SIMD4(red, 1)))
        #expect(isNearly(try scene(far: MeshRenderState(depthCompare: .greater)), SIMD4(green, 1)))
        #expect(isNearly(try scene(far: MeshRenderState(depthCompare: .never)), SIMD4(red, 1)))
        #expect(isNearly(try scene(far: MeshRenderState(depthTest: false)), SIMD4(green, 1)))
        #expect(isNearly(try scene(near: MeshRenderState(depthWrite: false)), SIMD4(green, 1)))
        // A depth-only occluder hides the far quad and leaves the clear color.
        #expect(isNearly(try scene(near: MeshRenderState(colorWriteMask: [])), clear))
        #expect(isNearly(try scene(near: MeshRenderState(colorWriteMask: [.red, .alpha])), SIMD4(red.x, clear.y, clear.z, 1)))

        // As in WebGL, a disabled depth test writes no depth: the near quad drawn first
        // without a depth test does not hide the far quad drawn after it. An `.always`
        // compare with the test enabled passes every fragment and still writes depth.
        func ordered(near: MeshRenderState) throws -> SIMD4<Float> {
            try center([
                node(try basic(red, state: near), z: -1, renderOrder: 0), node(try basic(green), z: -3, renderOrder: 1),
            ], clear: clear)
        }
        #expect(isNearly(try ordered(near: MeshRenderState(depthTest: false)), SIMD4(green, 1)))
        #expect(isNearly(try ordered(near: MeshRenderState(depthCompare: .always)), SIMD4(red, 1)))
    }

    /// Renders one quad and returns the window depth of every pixel, read back from the depth attachment.
    func depths(_ mesh: MeshRenderer, rotation: simd_quatf? = nil, depthMode: MeshDepthMode = .forward) throws -> [Float] {
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.depthAttachment.storeAction = .store
        let graph = SceneGraph(roots: [node(mesh, z: -3, rotation: rotation)], camera: fixture.camera)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        if depthMode == .reversed {
            var reverse = matrix_identity_float4x4
            reverse.columns.2.z = -1
            reverse.columns.3.z = 1
            pass.depthAttachment.clearDepth = 0
            try graph.encode(
                pass: pass, commandBuffer: buffer, clipFromWorld: reverse * (try fixture.camera.projection(aspectRatio: 1)),
                cameraPosition: .zero, depthMode: .reversed
            )
        } else {
            try graph.encode(pass: pass, commandBuffer: buffer)
        }
        let side = fixture.side
        let readback = try #require(fixture.device.makeBuffer(length: side * side * 4, options: .storageModeShared))
        let blit = try #require(buffer.makeBlitCommandEncoder())
        blit.copy(
            from: target.depth, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: side, height: side, depth: 1), to: readback, destinationOffset: 0,
            destinationBytesPerRow: side * 4, destinationBytesPerImage: side * side * 4
        )
        blit.endEncoding()
        try fixture.finish(buffer)
        return Array(UnsafeBufferPointer(start: readback.contents().assumingMemoryBound(to: Float.self), count: side * side))
    }

    @Test("Polygon offset adds factor × depth slope plus units × the depth format's resolution, clamped, in both depth modes")
    func polygonOffsetMagnitudes() throws {
        let side = fixture.side
        // Turning the quad 40° about Y gives a depth slope along X only; facing the camera gives none.
        let tilted = simd_quatf(angle: .pi * 40 / 180, axis: SIMD3(0, 1, 0))
        for depthMode in [MeshDepthMode.forward, .reversed] {
            func offsetDepths(_ offset: MeshRenderState.DepthOffset?, rotation: simd_quatf?) throws -> [Float] {
                let mesh = try basic(
                    SIMD3(1, 1, 1), state: MeshRenderState(depthOffset: offset), halfExtent: 1.5, depthMode: depthMode
                )
                return try depths(mesh, rotation: rotation, depthMode: depthMode)
            }
            // Positive offsets move away from the camera: larger forward depth, smaller reversed depth.
            let away: Float = depthMode == .forward ? 1 : -1
            let (x, y) = (30, 32)

            // Slope term: factor × max(|∂z/∂x|, |∂z/∂y|) in window pixels.
            let plainTilted = try offsetDepths(nil, rotation: tilted)
            let slope = abs(plainTilted[y * side + x + 1] - plainTilted[y * side + x - 1]) / 2
            let rowSlope = abs(plainTilted[(y + 1) * side + x] - plainTilted[(y - 1) * side + x]) / 2
            #expect(slope > 1e-4 && rowSlope < slope * 1e-3, "\(depthMode): slopes \(slope), \(rowSlope)")
            let factor: Float = 4
            let sloped = try offsetDepths(.init(factor: factor, units: 0), rotation: tilted)
            let slopeShift = (sloped[y * side + x] - plainTilted[y * side + x]) * away
            #expect(abs(slopeShift - factor * slope) < 0.02 * factor * slope, "\(depthMode): \(slopeShift) vs \(factor * slope)")

            // Units term: units × 2^(exponent(z) − 23) for a 32-bit float depth buffer, with no slope.
            let plainFacing = try offsetDepths(nil, rotation: nil)
            let z = plainFacing[y * side + x]
            let resolution = Float(sign: .plus, exponent: z.exponent - 23, significand: 1)
            let units: Float = 1_000
            let unitShift = (try offsetDepths(.init(factor: 0, units: units), rotation: nil)[y * side + x] - z) * away
            #expect(abs(unitShift - units * resolution) < 0.01 * units * resolution, "\(depthMode): \(unitShift) vs \(units * resolution)")
            // The factor alone does nothing without a slope, and units alone barely move a tilted quad.
            #expect(try offsetDepths(.init(factor: factor, units: 0), rotation: nil)[y * side + x] == z)
            let tiltedUnits = (try offsetDepths(.init(factor: 0, units: units), rotation: tilted)[y * side + x]
                - plainTilted[y * side + x]) * away
            #expect(tiltedUnits > 0 && tiltedUnits < 0.1 * factor * slope, "\(depthMode): \(tiltedUnits)")

            // Clamp: a positive clamp bounds the total bias at that magnitude.
            let limit = factor * slope / 4
            let clamped = try offsetDepths(.init(factor: factor, units: 0, clamp: limit), rotation: tilted)
            let clampedShift = (clamped[y * side + x] - plainTilted[y * side + x]) * away
            #expect(abs(clampedShift - limit) < 0.01 * limit, "\(depthMode): \(clampedShift) vs \(limit)")
        }
    }

    @Test("Polygon offset resolves coplanar quads toward or away from the camera in both depth modes")
    func polygonOffset() throws {
        let red = SIMD3<Float>(0.9, 0.1, 0.1)
        let green = SIMD3<Float>(0.1, 0.9, 0.1)
        func forward(_ state: MeshRenderState) throws -> SIMD4<Float> {
            try center([node(try basic(red), z: -2), node(try basic(green, state: state), z: -2)])
        }
        // Equal depth fails `.less`, so the first quad wins until the second is pulled forward.
        #expect(isNearly(try forward(MeshRenderState()), SIMD4(red, 1)))
        #expect(isNearly(try forward(MeshRenderState(depthOffset: .init(factor: -1, units: -1))), SIMD4(green, 1)))
        // Equal depth passes `.lessEqual` until the second quad is pushed back.
        #expect(isNearly(try forward(MeshRenderState(depthCompare: .lessEqual)), SIMD4(green, 1)))
        #expect(isNearly(
            try forward(MeshRenderState(depthCompare: .lessEqual, depthOffset: .init(factor: 1, units: 1))), SIMD4(red, 1)
        ))

        // Reversed depth: the same signs keep the same meaning.
        func reversed(_ state: MeshRenderState) throws -> SIMD4<Float> {
            let graph = SceneGraph(roots: [
                node(try basic(red, depthMode: .reversed), z: -2), node(try basic(green, state: state, depthMode: .reversed), z: -2),
            ], camera: fixture.camera)
            var reverse = matrix_identity_float4x4
            reverse.columns.2.z = -1
            reverse.columns.3.z = 1
            let target = try fixture.makeTarget()
            let pass = fixture.pass(color: target.color, depth: target.depth)
            pass.depthAttachment.clearDepth = 0
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            try graph.encode(
                pass: pass, commandBuffer: buffer, clipFromWorld: reverse * (try fixture.camera.projection(aspectRatio: 1)),
                cameraPosition: .zero, depthMode: .reversed
            )
            try fixture.finish(buffer)
            return fixture.pixel(try fixture.pixels(of: target.color), x: 32, y: 32)
        }
        #expect(isNearly(try reversed(MeshRenderState()), SIMD4(red, 1)))
        #expect(isNearly(try reversed(MeshRenderState(depthOffset: .init(factor: -1, units: -1))), SIMD4(green, 1)))
        #expect(isNearly(try reversed(MeshRenderState(depthCompare: .lessEqual)), SIMD4(green, 1)))
        #expect(isNearly(
            try reversed(MeshRenderState(depthCompare: .lessEqual, depthOffset: .init(factor: 1, units: 1))), SIMD4(red, 1)
        ))
    }

    @Test("Front, back and double sides rasterize the matching faces")
    func sides() throws {
        let color = SIMD3<Float>(0.3, 0.6, 0.9)
        let facingAway = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        func pixel(_ side: MeshRenderState.Side, rotation: simd_quatf? = nil) throws -> SIMD4<Float> {
            try center([node(try basic(color, state: MeshRenderState(side: side)), z: -2, rotation: rotation)])
        }
        #expect(isNearly(try pixel(.front), SIMD4(color, 1)))
        #expect(isNearly(try pixel(.front, rotation: facingAway), .zero))
        #expect(isNearly(try pixel(.back), .zero))
        #expect(isNearly(try pixel(.back, rotation: facingAway), SIMD4(color, 1)))
        #expect(isNearly(try pixel(.double), SIMD4(color, 1)))
        #expect(isNearly(try pixel(.double, rotation: facingAway), SIMD4(color, 1)))
    }

    // MARK: Sorting

    @Test("Transparent quads composite back to front regardless of graph order, after opaque draws")
    func transparentSorting() throws {
        let colors: [SIMD3<Float>] = [SIMD3(0.9, 0.2, 0.1), SIMD3(0.1, 0.8, 0.3), SIMD3(0.2, 0.3, 0.9)]
        let depths: [Float] = [-2, -3, -4]
        let alpha: Float = 0.5
        func composite(_ order: [Int], over background: SIMD4<Float> = .zero) -> SIMD4<Float> {
            order.reduce(background) { result, index in
                SIMD4(colors[index] * alpha, alpha) + result * (1 - alpha)
            }
        }
        let quads = try (0..<3).map { index in
            node(try basic(colors[index], opacity: alpha, state: .transparent), z: depths[index])
        }
        // Graph order is near to far; sorting draws far first.
        let sorted = try center(quads)
        #expect(isNearly(sorted, composite([2, 1, 0])), "\(sorted)")
        #expect(isNearly(try center(quads.reversed()), composite([2, 1, 0])))

        // Render order wins over depth within the transparent group.
        var ordered = quads
        ordered[2].renderOrder = 1
        let reordered = try center(ordered)
        #expect(isNearly(reordered, composite([1, 0, 2])), "\(reordered)")

        // An opaque quad behind them draws first even when listed last.
        let wall = node(try basic(SIMD3(1, 1, 0), halfExtent: 3), z: -6)
        let withWall = try center(quads + [wall])
        #expect(isNearly(withWall, composite([2, 1, 0], over: SIMD4(1, 1, 0, 1))), "\(withWall)")
    }

    @Test("Opaque render order draws lower values first and keeps graph order for ties")
    func opaqueRenderOrder() throws {
        let unordered = MeshRenderState(depthTest: false)
        let red = try basic(SIMD3(0.9, 0.1, 0.1), state: unordered)
        let green = try basic(SIMD3(0.1, 0.9, 0.1), state: unordered)
        #expect(isNearly(try center([node(red, z: -2), node(green, z: -3)]), SIMD4(0.1, 0.9, 0.1, 1)))
        #expect(isNearly(try center([node(red, z: -2, renderOrder: 1), node(green, z: -3)]), SIMD4(0.9, 0.1, 0.1, 1)))
        #expect(isNearly(try center([node(red, z: -2, renderOrder: -1), node(green, z: -3, renderOrder: -1)]), SIMD4(0.1, 0.9, 0.1, 1)))
    }

    @Test("A mesh-less node's render order sorts its whole subtree first, like a Three.js Group")
    func groupRenderOrder() throws {
        let unordered = MeshRenderState(depthTest: false)
        let red = SIMD3<Float>(0.9, 0.1, 0.1)
        let green = SIMD3<Float>(0.1, 0.9, 0.1)
        func group(_ order: Int, _ children: [SceneNode]) -> SceneNode {
            var node = SceneNode(children: children)
            node.renderOrder = order
            return node
        }
        // The red child's own render order 5 loses to the green group's order.
        let redGroup = group(2, [node(try basic(red, state: unordered), z: -2, renderOrder: 5)])
        let greenGroup = group(1, [node(try basic(green, state: unordered), z: -3)])
        #expect(isNearly(try center([redGroup, greenGroup]), SIMD4(red, 1)))
        #expect(isNearly(try center([group(0, redGroup.children), greenGroup]), SIMD4(green, 1)))
        // The nearest ordered ancestor wins: green sorts at 2, after red at 0, though its
        // outermost ancestor says −3 and graph order puts it first.
        let nested = group(-3, [group(0, [group(2, greenGroup.children)])])
        #expect(isNearly(try center([nested, node(try basic(red, state: unordered), z: -2)]), SIMD4(green, 1)))
        let resolved = try SceneGraph(roots: [group(4, [group(0, [node(try basic(red), z: -2)])])], camera: fixture.camera).resolve()
        #expect(resolved.map(\.groupOrder) == [0, 4, 4])

        // Transparent draws sort by group order before depth: the near quad's group draws first.
        let alpha: Float = 0.5
        let nearQuad = node(try basic(red, opacity: alpha, state: .transparent), z: -2)
        let farQuad = node(try basic(green, opacity: alpha, state: .transparent), z: -3)
        let composite = SIMD4(green * alpha, alpha) + SIMD4(red * alpha, alpha) * (1 - alpha)
        #expect(isNearly(try center([group(-1, [nearQuad]), farQuad]), composite))

        // A grouped multi-material node takes a group order on the returned parent.
        let plane = try MeshGeometry.plane(width: 1, height: 1).withGroups([MeshGeometryGroup(start: 0, count: 6)])
        var grouped = try SceneNode.grouped(
            plane, materials: [.classic(try ClassicMaterial(model: .basic, color: red))], device: fixture.device,
            colorPixelFormat: .rgba16Float, renderStates: [unordered], transform: SceneTransform(translation: SIMD3(0, 0, -2))
        )
        grouped.renderOrder = 1
        #expect(isNearly(try center([grouped, node(try basic(green, state: unordered), z: -3)]), SIMD4(red, 1)))
    }

    // MARK: Path tracing

    @Test("The path tracer rejects raster-only render state and traces the state's sides")
    func pathTracerRenderState() throws {
        let paths = try PathTestFixture()
        let geometry = try GeometryRenderFixture.square(1)
        func quad(_ state: MeshRenderState?, facingAway: Bool = false) throws -> SceneGraph {
            let mesh = try MeshRenderer(
                device: paths.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(try PhysicallyBasedMaterial(
                    baseColorFactor: SIMD4(0.2, 0.6, 0.9, 1), alphaMode: .opaque, shadingModel: .unlit
                )),
                renderState: state
            )
            let rotation = simd_quatf(angle: facingAway ? .pi : 0, axis: SIMD3(0, 1, 0))
            return SceneGraph(
                roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -3), rotation: rotation), mesh: mesh)],
                camera: paths.camera
            )
        }
        let rejected: [(MeshRenderState, String)] = [
            (MeshRenderState(blending: .additive, transparent: true, depthWrite: false), "render state blending"),
            (MeshRenderState(colorWriteMask: []), "color write masks"),
            (MeshRenderState(stencil: .init(reference: 1, compare: .equal)), "raster stencil state"),
            (MeshRenderState(alphaHash: true), "alpha hashing"),
            (MeshRenderState(side: .back), "back-face-only sides"),
            (MeshRenderState(depthTest: false), "depth tests other than nearest-surface visibility"),
            (MeshRenderState(depthCompare: .greater), "depth tests other than nearest-surface visibility"),
        ]
        for (state, feature) in rejected {
            let graph = try quad(state)
            let id = try #require(graph.roots.first?.id)
            #expect(throws: PathTracingScene.Failure.unsupported(id, feature)) {
                try PathTracingScene(queue: paths.queue, graph: graph, textureSize: 16)
            }
        }
        // Ordering-only state is accepted.
        _ = try PathTracingScene(
            queue: paths.queue, graph: try quad(MeshRenderState(depthWrite: false, depthOffset: .init(factor: 1, units: 1))),
            textureSize: 16
        )

        // A quad facing away shows in both renderers with `.double` sides and in neither with `.front`.
        let (width, height) = (33, 25)
        let center = height / 2 * width + width / 2
        for (side, visible) in [(MeshRenderState.Side.double, true), (.front, false)] {
            let graph = try quad(MeshRenderState(side: side), facingAway: true)
            let scene = try PathTracingScene(queue: paths.queue, graph: graph, textureSize: 16)
            let traced = try paths.trace(
                try PathTracer(queue: paths.queue, scene: scene), width: width, height: height, samples: 4,
                settings: PathTracer.Settings(maxBounces: 1, seed: 5)
            )[center]
            let raster = try paths.rasterize(graph, width: width, height: height)[center]
            let expected: SIMD3<Float> = visible ? SIMD3(0.2, 0.6, 0.9) : .zero
            #expect(simd_reduce_max(abs(SIMD3(traced.x, traced.y, traced.z) - expected)) < 0.002, "\(side) traced \(traced)")
            #expect(simd_reduce_max(abs(SIMD3(raster.x, raster.y, raster.z) - expected)) < 0.002, "\(side) raster \(raster)")
        }
    }

    // MARK: Alpha hashing

    @Test("Alpha hashing keeps a stable fraction of opaque pixels equal to alpha")
    func alphaHash() throws {
        let large = try GeometryRenderFixture(side: 128)
        func coverage(_ material: MeshMaterial) throws -> (fraction: Double, pixels: [SIMD4<Float>]) {
            let mesh = try MeshRenderer(
                device: large.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(3),
                material: material, renderState: MeshRenderState(alphaHash: true)
            )
            let graph = SceneGraph(roots: [node(mesh, z: -2)], camera: large.camera)
            let pixels = try large.render(graph).pixels
            let kept = pixels.filter { $0.w > 0.5 }
            #expect(kept.allSatisfy { $0.w == 1 }, "Hashed survivors are opaque")
            return (Double(kept.count) / Double(pixels.count), pixels)
        }
        for alpha: Float in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let classic = try coverage(.classic(try ClassicMaterial(model: .basic, color: SIMD3(1, 0.5, 0.25), opacity: alpha)))
            #expect(abs(classic.fraction - Double(alpha)) < 0.03, "classic \(alpha): \(classic.fraction)")
            #expect(try coverage(.classic(try ClassicMaterial(model: .basic, color: SIMD3(1, 0.5, 0.25), opacity: alpha))).pixels
                == classic.pixels, "The pattern is stable between frames")
            let pbr = try coverage(.physicallyBased(try PhysicallyBasedMaterial(
                baseColorFactor: SIMD4(1, 0.5, 0.25, alpha), shadingModel: .unlit
            )))
            #expect(abs(pbr.fraction - Double(alpha)) < 0.03, "PBR \(alpha): \(pbr.fraction)")
        }
        #expect(throws: MeshRenderer.Failure.self) {
            try MeshRenderer(
                device: large.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(1),
                material: .unlit, renderState: MeshRenderState(alphaHash: true)
            )
        }
    }

    // MARK: Multisampling

    func multisampled(_ roots: [SceneNode], samples: Int = 4) throws -> [SIMD4<Float>] {
        let side = fixture.side
        let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false)
        color.textureType = .type2DMultisample
        color.sampleCount = samples
        color.storageMode = .private
        color.usage = .renderTarget
        let depth = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: side, height: side, mipmapped: false)
        depth.textureType = .type2DMultisample
        depth.sampleCount = samples
        depth.storageMode = .private
        depth.usage = .renderTarget
        let resolved = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false)
        resolved.storageMode = .shared
        resolved.usage = .renderTarget
        let resolveTexture = try #require(fixture.device.makeTexture(descriptor: resolved))
        let pass = fixture.pass(
            color: try #require(fixture.device.makeTexture(descriptor: color)),
            depth: try #require(fixture.device.makeTexture(descriptor: depth))
        )
        pass.colorAttachments[0].storeAction = .multisampleResolve
        pass.colorAttachments[0].resolveTexture = resolveTexture
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        try SceneGraph(roots: roots, camera: fixture.camera).encode(pass: pass, commandBuffer: buffer)
        try fixture.finish(buffer)
        return try fixture.pixels(of: resolveTexture)
    }

    @Test("Multisampled renderers resolve partial edge coverage, and sample counts must match")
    func multisampling() throws {
        // The quad's right edge crosses column 40 at 48% of its width.
        let mesh = try basic(SIMD3(1, 1, 1), halfExtent: 0.53, sampleCount: 4)
        let pixels = try multisampled([node(mesh, z: -2)])
        // Rows 25 to 38 lie inside the quad's vertical extent (rows 23.5 to 40.5).
        let edge = (25..<39).map { fixture.pixel(pixels, x: 40, y: $0).x }
        #expect(edge.allSatisfy { $0 > 0.2 && $0 < 0.8 }, "\(edge)")
        #expect(fixture.pixel(pixels, x: 39, y: 32).x == 1 && fixture.pixel(pixels, x: 41, y: 32).x == 0)
        #expect(fixture.pixel(pixels, x: 32, y: 32) == SIMD4(1, 1, 1, 1))
        #expect(throws: MeshRenderer.Failure.sampleCount) {
            try multisampled([node(try basic(SIMD3(1, 1, 1)), z: -2)])
        }
        #expect(throws: MeshRenderer.Failure.sampleCount) {
            try basic(SIMD3(1, 1, 1), sampleCount: 3)
        }
    }

    @Test("Point, line and sprite pipelines resolve MSAA with a matching sample count")
    func primitiveMultisampling() throws {
        let cloud = try PointCloud(positions: [SIMD3(-1.25, 0, -4)])
        let pointMaterial = PointsMaterial(size: .pixels(3), color: SIMD4(1, 0, 0, 1))
        let point = try PointsRenderer(device: fixture.device, sampleCount: 4, cloud: cloud, material: pointMaterial)
        let geometry = try LineGeometry(positions: [SIMD3(0.5, 0, -4), SIMD3(1.5, 0, -4)])
        let line = try LineRenderer(
            device: fixture.device, sampleCount: 4, geometry: geometry,
            material: LineMaterial(width: .pixels(3), color: SIMD4(0, 1, 0, 1), blending: .opaque)
        )
        let sprite = try SpriteRenderer(
            device: fixture.device, sampleCount: 4,
            sprites: [Sprite(position: SIMD3(0, 1.25, -4), size: SIMD2(repeating: 0.375))],
            material: SpriteMaterial(color: SIMD4(0, 0, 1, 1), blending: .opaque)
        )
        let pixels = try multisampled([
            SceneNode(primitive: .points(point)), SceneNode(primitive: .lines(line)), SceneNode(primitive: .sprites(sprite)),
        ])
        #expect(fixture.pixel(pixels, x: 22, y: 32) == SIMD4(1, 0, 0, 1))
        #expect(fixture.pixel(pixels, x: 40, y: 32) == SIMD4(0, 1, 0, 1))
        #expect(fixture.pixel(pixels, x: 32, y: 22) == SIMD4(0, 0, 1, 1))
        let edge = fixture.pixel(pixels, x: 20, y: 32)
        #expect(edge.x > 0 && edge.x < 1)

        let single = try PointsRenderer(device: fixture.device, cloud: cloud, material: pointMaterial)
        #expect(throws: MeshRenderer.Failure.sampleCount) {
            try multisampled([SceneNode(primitive: .points(single))])
        }
        #expect(throws: MeshRenderer.Failure.sampleCount) {
            try LineRenderer(device: fixture.device, sampleCount: 3, geometry: geometry)
        }
    }

    @Test("Alpha to coverage turns opacity into MSAA coverage of straight color")
    func alphaToCoverage() throws {
        let coverage = MeshRenderState(alphaToCoverage: true)
        for alpha: Float in [0.25, 0.5, 0.75] {
            let classic = try basic(SIMD3(1, 1, 1), opacity: alpha, state: coverage, halfExtent: 3, sampleCount: 4)
            let pbr = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(3),
                material: .physicallyBased(try PhysicallyBasedMaterial(
                    baseColorFactor: SIMD4(1, 1, 1, alpha), shadingModel: .unlit
                )),
                renderState: coverage, sampleCount: 4
            )
            for mesh in [classic, pbr] {
                let pixels = try multisampled([node(mesh, z: -2)])
                // Every pixel resolves exactly alpha × 4 of its four samples.
                let wrong = pixels.filter { abs($0.x - alpha) > 0.001 || abs($0.w - alpha) > 0.001 }
                #expect(wrong.isEmpty, "alpha \(alpha): \(wrong.count) pixels differ, first \(wrong.first.map { "\($0)" } ?? "")")
            }
        }
        #expect(throws: MeshRenderer.Failure.self) {
            try basic(SIMD3(1, 1, 1), state: coverage, sampleCount: 1)
        }
        // Blending would apply alpha a second time on top of coverage.
        #expect(throws: MeshRenderer.Failure.self) {
            try basic(SIMD3(1, 1, 1), state: MeshRenderState(blending: .normal, transparent: true, alphaToCoverage: true), sampleCount: 4)
        }
    }

    @Test("Alpha to coverage ramps an alpha cutoff over one pixel, as Three.js alphaTest does with alphaToCoverage")
    func alphaToCoverageCutoff() throws {
        let coverage = MeshRenderState(alphaToCoverage: true)
        // Constant alpha above the cutoff covers every sample, instead of alpha's share of them.
        let solid = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(3),
            material: .classic(try ClassicMaterial(model: .basic, opacity: 0.75, alphaTest: 0.5)),
            renderState: coverage, sampleCount: 4
        )
        #expect(try multisampled([node(solid, z: -2)]).allSatisfy { $0 == SIMD4(1, 1, 1, 1) })

        // Alpha rises linearly from 0 at the left edge to 1 at the right edge of a quad
        // that fills the 64-pixel view, so column c samples alpha (c + 0.5) / 64 and
        // fwidth(alpha) is 1/64. With cutoff 0.5, column 32 sits halfway up the ramp
        // (partial coverage), columns from 33 are covered and columns to 31 empty.
        let ramp = try VertexColorTests.gradientSquare(
            2, left: SIMD4(1, 1, 1, 0), right: SIMD4(1, 1, 1, 1)
        )
        let materials: [MeshMaterial] = [
            .physicallyBased(try PhysicallyBasedMaterial(alphaMode: .mask(cutoff: 0.5), shadingModel: .unlit)),
            .classic(try ClassicMaterial(model: .basic, alphaTest: 0.5)),
        ]
        for (index, material) in materials.enumerated() {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: ramp, material: material,
                renderState: coverage, sampleCount: 4
            )
            let pixels = try multisampled([node(mesh, z: -2)])
            for row in [8, 32, 56] {
                let columns = (28...36).map { fixture.pixel(pixels, x: $0, y: row).x }
                #expect(columns[0..<4].allSatisfy { $0 == 0 } && columns[5...].allSatisfy { $0 == 1 },
                        "material \(index) row \(row): \(columns)")
                #expect(columns[4] >= 0.25 && columns[4] <= 0.75, "material \(index) row \(row): edge \(columns[4])")
            }
            #expect((40..<64).allSatisfy { fixture.pixel(pixels, x: $0, y: 20) == SIMD4(1, 1, 1, 1) })
        }
    }
}
