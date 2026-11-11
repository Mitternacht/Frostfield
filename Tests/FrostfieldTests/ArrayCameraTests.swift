import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ArrayCameraTests {
    @Test("Independent cameras fill offset rectangles with per-view layers and scissor",
          arguments: [MeshDepthMode.forward, .reversed])
    func distinctPerspectiveViews(_ mode: MeshDepthMode) throws {
        let fixture = try GeometryRenderFixture(side: 80)
        let mesh = try unlitPlane(fixture, size: 2.4, mode: mode)
        let greenLayer = SceneLayers(layer: 1)
        var graph = SceneGraph(
            roots: [
                node(mesh, at: SIMD3(-3, 0, -3), tint: SIMD4(1, 0, 0, 1)),
                node(mesh, at: SIMD3(3, 0, -3), tint: SIMD4(0, 1, 0, 1), layers: greenLayer),
                node(mesh, at: SIMD3(3, 0, -2), tint: SIMD4(0, 0, 1, 1))
            ],
            camera: fixture.camera
        )
        // The graph's mask lets both layers through; the right view then narrows to the green one.
        let sharedLayers: SceneLayers = [.default, greenLayer]
        graph.cameraLayers = sharedLayers
        let left = try ArrayCamera.View(
            camera: fixture.camera, transform: .init(translation: SIMD3(-3, 0, 0)),
            viewport: viewport(x: 3, y: 5, width: 28, height: 32), depthMode: mode
        )
        let right = try ArrayCamera.View(
            camera: fixture.camera, transform: .init(translation: SIMD3(3, 0, 0)),
            viewport: viewport(x: 44, y: 26, width: 30, height: 24),
            scissor: MTLScissorRect(x: 56, y: 34, width: 14, height: 12), layers: greenLayer, depthMode: mode
        )
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth, depthMode: mode)
        let clear = SIMD4<Float>(0.03125, 0.0625, 0.125, 1)
        try FramebufferClear(color: SIMD4<Double>(clear)).apply(to: pass)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let statistics = try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [left, right]))
        #expect(statistics.map(\.drawnMeshCount) == [1, 1])
        #expect(graph.cameraLayers == sharedLayers)
        #expect(pass.colorAttachments[0].loadAction == .clear && pass.depthAttachment.loadAction == .clear)
        #expect(pass.colorAttachments[0].storeAction == .store && pass.depthAttachment.storeAction == .dontCare)
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: target.color)
        #expect(fixture.pixel(pixels, x: 17, y: 21) == SIMD4(1, 0, 0, 1))
        #expect(fixture.pixel(pixels, x: 59, y: 38) == SIMD4(0, 1, 0, 1))
        #expect(fixture.pixel(pixels, x: 55, y: 38) == clear)
        #expect(fixture.pixel(pixels, x: 39, y: 39) == clear)
        #expect(fixture.pixel(pixels, x: 59, y: 20) == clear)
    }

    @Test("Overlapping orthographic views load prior color and depth and preserve the prefilled exterior")
    func overlapsWithLoad() throws {
        let fixture = try GeometryRenderFixture(side: 64)
        let mesh = try unlitPlane(fixture, size: 4)
        let graph = SceneGraph(
            roots: [
                node(mesh, at: SIMD3(0, 0, -2), tint: SIMD4(1, 0, 0, 1)),
                node(mesh, at: SIMD3(6, 0, -4), tint: SIMD4(0, 1, 0, 1))
            ],
            camera: fixture.camera
        )
        let near = try orthographic(x: 0, viewport: viewport(x: 8, y: 8, width: 36, height: 36))
        let far = try orthographic(x: 6, viewport: viewport(x: 20, y: 20, width: 36, height: 36))
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.depthAttachment.storeAction = .store
        let clear = SIMD4<Float>(0.125, 0, 0.25, 1)
        let prefill = try #require(fixture.queue.makeCommandBuffer())
        try FramebufferClear(color: SIMD4<Double>(clear), depth: 1).encode(in: pass, commandBuffer: prefill)
        try fixture.finish(prefill)

        pass.colorAttachments[0].loadAction = .load
        pass.depthAttachment.loadAction = .load
        let command = try #require(fixture.queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [near, far]))
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: target.color)
        #expect(fixture.pixel(pixels, x: 12, y: 12) == SIMD4(1, 0, 0, 1))
        #expect(fixture.pixel(pixels, x: 30, y: 30) == SIMD4(1, 0, 0, 1))
        #expect(fixture.pixel(pixels, x: 50, y: 50) == SIMD4(0, 1, 0, 1))
        #expect(fixture.pixel(pixels, x: 1, y: 1) == clear)
        #expect(pass.colorAttachments[0].loadAction == .load && pass.depthAttachment.loadAction == .load)
    }

    @Test("Invalid later views are rejected before clearing or encoding the first view")
    func preflight() throws {
        let fixture = try GeometryRenderFixture(side: 32)
        let graph = SceneGraph(roots: [], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        let prefill = try #require(fixture.queue.makeCommandBuffer())
        let blue = SIMD4<Float>(0, 0, 1, 1)
        try FramebufferClear(color: SIMD4<Double>(blue)).encode(in: pass, commandBuffer: prefill)
        try fixture.finish(prefill)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let rect = viewport(x: 0, y: 0, width: 16, height: 16)
        let valid = try orthographic(x: 0, viewport: rect)
        let outside = try orthographic(x: 0, viewport: viewport(x: 28, y: 28, width: 16, height: 16))
        #expect(throws: MeshRenderer.Failure.viewport) {
            try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [valid, outside]))
        }
        let reversed = ArrayCamera.View(camera: valid.camera, position: .zero, viewport: rect, depthMode: .reversed)
        #expect(throws: ArrayCamera.Failure.mixedDepthModes) {
            try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [valid, reversed]))
        }
        let badDirection = ArrayCamera.View(camera: valid.camera, position: .zero, viewport: rect, directionToCamera: .zero)
        #expect(throws: MeshRenderer.Failure.transform) {
            try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [valid, badDirection]))
        }
        let badScissor = ArrayCamera.View(
            camera: valid.camera, position: .zero, viewport: rect, scissor: MTLScissorRect(x: 20, y: 0, width: 14, height: 8)
        )
        #expect(throws: MeshRenderer.Failure.scissor) {
            try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [valid, badScissor]))
        }
        let empty = try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera())
        #expect(empty.isEmpty)
        try fixture.finish(command)
        #expect(try fixture.pixels(of: target.color).allSatisfy { $0 == blue })
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsTextureSampleCount(4) == true, "Requires 4x MSAA"))
    func `Multisample color resolves after both views and retains the first view`() throws {
        let fixture = try GeometryRenderFixture(side: 48)
        let mesh = try unlitPlane(fixture, size: 4, sampleCount: 4)
        let graph = SceneGraph(
            roots: [
                node(mesh, at: SIMD3(0, 0, -2), tint: SIMD4(1, 0, 0, 1)),
                node(mesh, at: SIMD3(6, 0, -2), tint: SIMD4(0, 1, 0, 1))
            ],
            camera: fixture.camera
        )
        let left = try orthographic(x: 0, viewport: viewport(x: 2, y: 3, width: 18, height: 20))
        let right = try orthographic(x: 6, viewport: viewport(x: 27, y: 18, width: 18, height: 20))
        let resolved = try fixture.makeTarget().color
        func multisample(_ format: MTLPixelFormat) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor()
            descriptor.textureType = .type2DMultisample
            descriptor.pixelFormat = format
            descriptor.width = fixture.side
            descriptor.height = fixture.side
            descriptor.sampleCount = 4
            descriptor.storageMode = .private
            descriptor.usage = .renderTarget
            return try #require(fixture.device.makeTexture(descriptor: descriptor))
        }
        let pass = fixture.pass(color: try multisample(.rgba16Float), depth: try multisample(.depth32Float))
        pass.colorAttachments[0].resolveTexture = resolved
        pass.colorAttachments[0].storeAction = .multisampleResolve
        let command = try #require(fixture.queue.makeCommandBuffer())
        let log = PassLog()
        try graph.encodeArrayCamera(
            pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [left, right]), shadowMap: nil,
            transmissionBackdrop: nil, punctualShadowMaps: nil, ordering: .init(),
            observingViewPasses: { log.record($0, $1, original: pass) }
        )
        try fixture.finish(command)
        // Only the last view resolves; the first stores its multisample color unresolved.
        #expect(log.snapshots.map(\.colorStore) == [.store, .multisampleResolve])
        #expect(log.snapshots.map(\.hasColorResolve) == [false, true])
        let pixels = try fixture.pixels(of: resolved)
        #expect(fixture.pixel(pixels, x: 10, y: 10) == SIMD4(1, 0, 0, 1))
        #expect(fixture.pixel(pixels, x: 32, y: 25) == SIMD4(0, 1, 0, 1))
        #expect(fixture.pixel(pixels, x: 25, y: 12) == .zero)
        #expect(pass.colorAttachments[0].storeAction == .multisampleResolve)
        #expect(pass.colorAttachments[0].resolveTexture === resolved)
    }

    @Test("Each view's pass loads and stores as documented, even when the caller stores nothing", arguments: [1, 2, 3, 4])
    func passDescriptorsPerView(_ count: Int) throws {
        let fixture = try GeometryRenderFixture(side: 32)
        let graph = SceneGraph(roots: [], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.colorAttachments[0].storeAction = .dontCare
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1)
        pass.depthAttachment.clearDepth = 0.5
        #expect(pass.depthAttachment.storeAction == .dontCare)
        let views = try (0..<count).map {
            try orthographic(x: Float($0), viewport: viewport(x: Double($0 * 4), y: 0, width: 16, height: 16))
        }
        let log = PassLog()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let statistics = try graph.encodeArrayCamera(
            pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: views), shadowMap: nil,
            transmissionBackdrop: nil, punctualShadowMaps: nil, ordering: .init(),
            observingViewPasses: { log.record($0, $1, original: pass) }
        )
        try fixture.finish(command)
        #expect(statistics.count == count)
        #expect(log.snapshots.map(\.index) == Array(0..<count))
        for snapshot in log.snapshots {
            let isFirst = snapshot.index == 0, isLast = snapshot.index == count - 1
            #expect(snapshot.colorLoad == (isFirst ? .clear : .load), "view \(snapshot.index)")
            #expect(snapshot.depthLoad == (isFirst ? .clear : .load), "view \(snapshot.index)")
            // Only the last view applies the caller's dontCare; the others must keep depth for the next view.
            #expect(snapshot.colorStore == (isLast ? .dontCare : .store), "view \(snapshot.index)")
            #expect(snapshot.depthStore == (isLast ? .dontCare : .store), "view \(snapshot.index)")
            #expect(snapshot.isCallerPass == (count == 1), "view \(snapshot.index)")
            if isFirst {
                #expect(snapshot.clearBlue == 0.75 && snapshot.clearDepth == 0.5)
            }
        }
        #expect(pass.colorAttachments[0].loadAction == .clear && pass.colorAttachments[0].storeAction == .dontCare)
        #expect(pass.depthAttachment.loadAction == .clear && pass.depthAttachment.storeAction == .dontCare)
    }

    @Test("A middle view stores the depth that a later view tests against, although the caller discards depth")
    func middleViewKeepsDepth() throws {
        let fixture = try GeometryRenderFixture(side: 64)
        let mesh = try unlitPlane(fixture, size: 2)
        let yellow = SIMD4<Float>(1, 1, 0, 1), red = SIMD4<Float>(1, 0, 0, 1), blue = SIMD4<Float>(0, 0, 1, 1)
        var graph = SceneGraph(
            roots: [
                node(mesh, at: SIMD3(-3, 0, -2), tint: yellow, layers: SceneLayers(layer: 1)),
                node(mesh, at: SIMD3(0, 0, -2), tint: red, layers: SceneLayers(layer: 2)),
                node(mesh, at: SIMD3(0, 0, -6), tint: blue, layers: SceneLayers(layer: 3))
            ],
            camera: fixture.camera
        )
        graph.cameraLayers = SceneLayers(layers: [1, 2, 3])
        // The first view draws elsewhere. Only the middle view writes depth where the last one draws.
        let full = viewport(x: 0, y: 0, width: 64, height: 64)
        let views = try (1...3).map { try wideOrthographic(viewport: full, layers: SceneLayers(layer: $0)) }
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        #expect(pass.depthAttachment.storeAction == .dontCare)
        let log = PassLog()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let statistics = try graph.encodeArrayCamera(
            pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: views), shadowMap: nil,
            transmissionBackdrop: nil, punctualShadowMaps: nil, ordering: .init(),
            observingViewPasses: { log.record($0, $1, original: pass) }
        )
        try fixture.finish(command)
        #expect(statistics.map(\.drawnMeshCount) == [1, 1, 1])
        #expect(log.snapshots.map(\.depthStore) == [.store, .store, .dontCare])
        #expect(log.snapshots.map(\.depthLoad) == [.clear, .load, .load])
        let pixels = try fixture.pixels(of: target.color)
        #expect(fixture.pixel(pixels, x: 8, y: 32) == yellow)
        #expect(fixture.pixel(pixels, x: 32, y: 32) == red)
        #expect(!pixels.contains(blue))
    }

    @Test("Multiple views reject untracked attachments, which one view encodes like an explicit view")
    func untrackedDepthGate() throws {
        let fixture = try GeometryRenderFixture(side: 32)
        let graph = SceneGraph(roots: [], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: fixture.side, height: fixture.side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        descriptor.hazardTrackingMode = .untracked
        let untracked = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let pass = fixture.pass(color: target.color, depth: untracked)
        let rect = viewport(x: 0, y: 0, width: 16, height: 16)
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: ArrayCamera.Failure.transientAttachment) {
            try graph.encode(
                pass: pass, commandBuffer: command,
                arrayCamera: ArrayCamera(views: [try orthographic(x: 0, viewport: rect), try orthographic(x: 1, viewport: rect)])
            )
        }
        let single = try graph.encode(
            pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [try orthographic(x: 0, viewport: rect)])
        )
        #expect(single.count == 1)
        try fixture.finish(command)
    }

    @Test("A view's layers narrow the graph's camera layers instead of replacing them")
    func viewLayersIntersectGraphLayers() throws {
        let fixture = try GeometryRenderFixture(side: 64)
        let mesh = try unlitPlane(fixture, size: 1.8)
        let red = SIMD4<Float>(1, 0, 0, 1), green = SIMD4<Float>(0, 1, 0, 1)
        let blue = SIMD4<Float>(0, 0, 1, 1), yellow = SIMD4<Float>(1, 1, 0, 1)
        var graph = SceneGraph(
            roots: [
                node(mesh, at: SIMD3(-3, 0, -2), tint: red, layers: SceneLayers(layers: [0])),
                node(mesh, at: SIMD3(-1, 0, -2), tint: green, layers: SceneLayers(layers: [1])),
                node(mesh, at: SIMD3(1, 0, -2), tint: blue, layers: SceneLayers(layers: [2])),
                node(mesh, at: SIMD3(3, 0, -2), tint: yellow, layers: SceneLayers(layers: [0, 2]))
            ],
            camera: fixture.camera
        )
        graph.cameraLayers = SceneLayers(layers: [0, 1])
        let full = viewport(x: 0, y: 0, width: 64, height: 64)
        func render(layers: SceneLayers?) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
            let target = try fixture.makeTarget()
            let pass = fixture.pass(color: target.color, depth: target.depth)
            let command = try #require(fixture.queue.makeCommandBuffer())
            let view = try wideOrthographic(viewport: full, layers: layers)
            let statistics = try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [view]))
            try fixture.finish(command)
            return (try fixture.pixels(of: target.color), statistics[0])
        }
        // Layers 1 and 2: the blue node is on layer 2 only, which the graph's mask lacks, so it stays hidden.
        // The yellow node is on layers 0 and 2 and passes both tests separately, as in Three.js.
        let narrowed = try render(layers: SceneLayers(layers: [1, 2]))
        #expect(narrowed.statistics.drawnMeshCount == 2)
        #expect(fixture.pixel(narrowed.pixels, x: 8, y: 32) == .zero)
        #expect(fixture.pixel(narrowed.pixels, x: 24, y: 32) == green)
        #expect(fixture.pixel(narrowed.pixels, x: 40, y: 32) == .zero)
        #expect(fixture.pixel(narrowed.pixels, x: 56, y: 32) == yellow)
        let inherited = try render(layers: nil)
        #expect(inherited.statistics.drawnMeshCount == 3)
        #expect(fixture.pixel(inherited.pixels, x: 8, y: 32) == red)
        #expect(fixture.pixel(inherited.pixels, x: 24, y: 32) == green)
        #expect(fixture.pixel(inherited.pixels, x: 40, y: 32) == .zero)
        #expect(fixture.pixel(inherited.pixels, x: 56, y: 32) == yellow)
        #expect(graph.cameraLayers == SceneLayers(layers: [0, 1]))
    }

    @Test("One view encodes like the explicit-view overload, and an unretained command buffer is accepted")
    func oneViewMatchesExplicitEncode() throws {
        let fixture = try GeometryRenderFixture(side: 64)
        let mesh = try unlitPlane(fixture, size: 4)
        let graph = SceneGraph(
            roots: [node(mesh, at: SIMD3(0, 0, -2), tint: SIMD4(1, 0, 0, 1))], camera: fixture.camera
        )
        let rect = viewport(x: 10, y: 6, width: 40, height: 44)
        let scissor = MTLScissorRect(x: 14, y: 10, width: 30, height: 30)
        let view = try orthographic(x: 0, viewport: rect, scissor: scissor)

        func explicit() throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
            let target = try fixture.makeTarget()
            let pass = fixture.pass(color: target.color, depth: target.depth)
            let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
            let statistics = try graph.encode(
                pass: pass, commandBuffer: command, clipFromWorld: view.clipFromWorld, cameraPosition: view.position,
                directionToCamera: view.directionToCamera, viewport: view.viewport, depthMode: view.depthMode,
                graphCamera: view.camera, scissor: view.scissor
            )
            try fixture.finish(command)
            return (try fixture.pixels(of: target.color), statistics)
        }
        func array(_ views: [ArrayCamera.View]) throws -> (pixels: [SIMD4<Float>], statistics: [SceneDrawStatistics]) {
            let target = try fixture.makeTarget()
            let pass = fixture.pass(color: target.color, depth: target.depth)
            let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
            let statistics = try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: views))
            try fixture.finish(command)
            return (try fixture.pixels(of: target.color), statistics)
        }
        let reference = try explicit()
        let single = try array([view])
        #expect(single.pixels == reference.pixels)
        #expect(single.statistics == [reference.statistics])
        #expect(fixture.pixel(single.pixels, x: 20, y: 20) == SIMD4(1, 0, 0, 1))
        #expect(fixture.pixel(single.pixels, x: 12, y: 20) == .zero)

        // Several views take the same command buffers; the attachments just stay alive here.
        let second = try orthographic(x: 0, viewport: viewport(x: 0, y: 0, width: 8, height: 8))
        let pair = try array([view, second])
        #expect(pair.statistics.map(\.drawnMeshCount) == [1, 1])
        #expect(fixture.pixel(pair.pixels, x: 20, y: 20) == SIMD4(1, 0, 0, 1))
        #expect(fixture.pixel(pair.pixels, x: 4, y: 4) == SIMD4(1, 0, 0, 1))

        // A committed command buffer cannot take another encoder, for any number of views.
        let committed = try #require(fixture.queue.makeCommandBuffer())
        committed.commit()
        committed.waitUntilCompleted()
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        for views in [[view], [view, second]] {
            #expect(throws: ArrayCamera.Failure.commandBuffer) {
                try graph.encode(pass: pass, commandBuffer: committed, arrayCamera: ArrayCamera(views: views))
            }
        }
    }

    @Test("A failure that only a later view hits throws before the first view encodes")
    func laterLevelOfDetailFailureLeavesTheBufferUntouched() throws {
        let fixture = try GeometryRenderFixture(side: 32)
        let near = try unlitPlane(fixture, size: 2)
        // This level draws into a different color format, which only a distant view selects.
        let far = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba8Unorm, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(.init(shadingModel: .unlit))
        )
        let levels = try LevelOfDetail(levels: [.init(mesh: near, distance: 0), .init(mesh: far, distance: 3)])
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), levelsOfDetail: levels)],
            camera: fixture.camera
        )
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        let blue = SIMD4<Float>(0, 0, 1, 1)
        let prefill = try #require(fixture.queue.makeCommandBuffer())
        try FramebufferClear(color: SIMD4<Double>(blue)).encode(in: pass, commandBuffer: prefill)
        try fixture.finish(prefill)

        let rect = viewport(x: 0, y: 0, width: 16, height: 16)
        let nearView = try orthographic(x: 0, viewport: rect)
        let farView = ArrayCamera.View(
            camera: nearView.camera, position: SIMD3(0, 0, 8), viewport: rect, directionToCamera: SIMD3(0, 0, 1)
        )
        // The first view alone encodes fine, which a separate target shows.
        let spare = try fixture.makeTarget()
        let alonePass = fixture.pass(color: spare.color, depth: spare.depth)
        let aloneCommand = try #require(fixture.queue.makeCommandBuffer())
        let alone = try graph.encode(pass: alonePass, commandBuffer: aloneCommand, arrayCamera: ArrayCamera(views: [nearView]))
        try fixture.finish(aloneCommand)
        #expect(alone[0].drawnMeshCount == 1)
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: MeshRenderer.Failure.attachments) {
            try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [nearView, farView]))
        }
        try fixture.finish(command)
        #expect(try fixture.pixels(of: target.color).allSatisfy { $0 == blue })
    }

    @Test("A transmissive scene needs its backdrop in any view with a visible scissor, before any view encodes")
    func laterTransmissionFailureLeavesTheBufferUntouched() throws {
        let fixture = try GeometryRenderFixture(side: 32)
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: try glassPlane(fixture, size: 6))],
            camera: fixture.camera
        )
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        let blue = SIMD4<Float>(0, 0, 1, 1)
        let prefill = try #require(fixture.queue.makeCommandBuffer())
        try FramebufferClear(color: SIMD4<Double>(blue)).encode(in: pass, commandBuffer: prefill)
        try fixture.finish(prefill)

        let rect = viewport(x: 0, y: 0, width: 16, height: 16)
        // An empty scissor draws nothing, so the first view alone never asks for the backdrop.
        let hidden = try orthographic(x: 0, viewport: rect, scissor: MTLScissorRect(x: 0, y: 0, width: 0, height: 0))
        let shown = try orthographic(x: 0, viewport: viewport(x: 16, y: 16, width: 16, height: 16))
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: MeshRenderer.Failure.transmissionBackdrop) {
            try graph.encode(pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [hidden, shown]))
        }
        try fixture.finish(command)
        #expect(try fixture.pixels(of: target.color).allSatisfy { $0 == blue })
    }

    @Test("Transmission samples each view's own backdrop copy across the views of one target")
    func transmissionAcrossViews() throws {
        let fixture = try GeometryRenderFixture(side: 64)
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: try glassPlane(fixture, size: 8))],
            camera: fixture.camera
        )
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        pass.colorAttachments[0].clearColor = MTLClearColorMake(2, 1, 0.5, 1)
        let backdrop = try makeBackdrop(fixture)
        let left = try orthographic(x: 0, viewport: viewport(x: 0, y: 0, width: 32, height: 64))
        let right = try orthographic(x: 0, viewport: viewport(x: 32, y: 0, width: 32, height: 64))
        let command = try #require(fixture.queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [left, right]),
            transmissionBackdrop: backdrop
        )
        try fixture.finish(command)
        #expect(statistics.map(\.drawnMeshCount) == [1, 1])
        let pixels = try fixture.pixels(of: target.color)
        // Tint (0.8, 0.6, 1) over the clear color (2, 1, 0.5), as in the single-view transmission test.
        for x in [16, 48] {
            let pixel = fixture.pixel(pixels, x: x, y: 32)
            #expect(simd_distance(pixel, SIMD4(1.6, 0.6, 0.5, 1)) < 0.02, "x \(x): \(pixel)")
        }
    }

    @Test("Every view redraws the shared shadow depth for its own casters")
    func shadowDepthIsRedrawnPerView() throws {
        let fixture = try GeometryRenderFixture(side: 64)
        let receiver = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert), castsShadow: false
        )
        let blocker = SceneNode(
            transform: .init(translation: SIMD3(0.75, 0, -2)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(0.25))
        )
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        let graph = SceneGraph(roots: [receiver, blocker], camera: fixture.camera)
        let left = try orthographic(x: 0, viewport: viewport(x: 0, y: 0, width: 32, height: 32))
        let right = try orthographic(x: 0, viewport: viewport(x: 32, y: 32, width: 32, height: 32))
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: pass, commandBuffer: command, arrayCamera: ArrayCamera(views: [left, right]), shadowMap: shadowMap
        )
        try fixture.finish(command)
        // One shared map, cleared and redrawn by each view in order, not computed once for the array.
        #expect(statistics.map(\.shadowPassCount) == [1, 1])
        #expect(statistics.map(\.shadowDrawCount) == [1, 1])
    }

    @Test("Picking goes through one chosen view, with its viewport, scissor, layers and graph layers")
    func pickingThroughViews() throws {
        let fixture = try GeometryRenderFixture(side: 64)
        let mesh = try unlitPlane(fixture, size: 1.8)
        let a = node(mesh, at: SIMD3(-3, 0, -2), tint: SIMD4(repeating: 1), layers: SceneLayers(layers: [0]))
        let b = node(mesh, at: SIMD3(-1, 0, -2), tint: SIMD4(repeating: 1), layers: SceneLayers(layers: [1]))
        let c = node(mesh, at: SIMD3(1, 0, -2), tint: SIMD4(repeating: 1), layers: SceneLayers(layers: [2]))
        let d = node(mesh, at: SIMD3(3, 0, -2), tint: SIMD4(repeating: 1), layers: SceneLayers(layers: [0, 2]))
        var graph = SceneGraph(roots: [a, b, c, d], camera: fixture.camera)
        graph.cameraLayers = SceneLayers(layers: [0, 1])
        let full = viewport(x: 0, y: 0, width: 64, height: 64)
        let inheriting = try wideOrthographic(viewport: full, layers: nil)
        let narrowed = try wideOrthographic(viewport: full, layers: SceneLayers(layers: [1, 2]))
        let leftHalf = try wideOrthographic(viewport: viewport(x: 0, y: 0, width: 32, height: 64), layers: nil)
        let clipped = try wideOrthographic(viewport: full, layers: nil, scissor: MTLScissorRect(x: 0, y: 0, width: 16, height: 64))
        let array = ArrayCamera(views: [inheriting, narrowed, leftHalf, clipped])

        func hit(_ x: Float, view: Int) throws -> UUID? {
            try graph.pick(atPixel: SIMD2(x, 32), in: array, viewIndex: view)?.nodeID
        }
        // Layer 2 alone is not in the graph's mask, so blue never picks, whichever view asks.
        #expect(try [8, 24, 40, 56].map { try hit(Float($0), view: 0) } == [a.id, b.id, nil, d.id])
        #expect(try [8, 24, 40, 56].map { try hit(Float($0), view: 1) } == [nil, b.id, nil, d.id])
        // A narrower viewport maps the same camera onto fewer pixels; points outside it hit nothing.
        #expect(try hit(4, view: 2) == a.id)
        #expect(try hit(40, view: 2) == nil)
        #expect(try hit(8, view: 3) == a.id)
        #expect(try hit(24, view: 3) == nil)
        // The nil-layer view agrees with the explicit camera pick.
        let explicit = try graph.pick(at: SIMD2(0.125, 0.5), camera: inheriting.camera)
        #expect(explicit?.nodeID == a.id)
        #expect(throws: ArrayCamera.Failure.viewIndex) {
            try graph.pick(atPixel: SIMD2(8, 32), in: array, viewIndex: 4)
        }
        #expect(throws: ScenePickingError.nonfiniteQueryPoint) {
            try graph.pick(atPixel: SIMD2(.nan, 32), in: array, viewIndex: 0)
        }
    }

    private final class PassLog {
        struct Snapshot {
            let index: Int
            let isCallerPass: Bool
            let colorLoad: MTLLoadAction
            let colorStore: MTLStoreAction
            let hasColorResolve: Bool
            let depthLoad: MTLLoadAction
            let depthStore: MTLStoreAction
            let clearBlue: Double
            let clearDepth: Double
        }

        private(set) var snapshots: [Snapshot] = []

        func record(_ index: Int, _ pass: MTLRenderPassDescriptor, original: MTLRenderPassDescriptor) {
            let color = pass.colorAttachments[0]!
            snapshots.append(Snapshot(
                index: index, isCallerPass: pass === original,
                colorLoad: color.loadAction, colorStore: color.storeAction, hasColorResolve: color.resolveTexture != nil,
                depthLoad: pass.depthAttachment.loadAction, depthStore: pass.depthAttachment.storeAction,
                clearBlue: color.clearColor.blue, clearDepth: pass.depthAttachment.clearDepth
            ))
        }
    }

    private func wideOrthographic(
        viewport: MTLViewport, layers: SceneLayers?, scissor: MTLScissorRect? = nil
    ) throws -> ArrayCamera.View {
        // Covers x and y in -4...4 and view depth 0...10.
        let projection = simd_float4x4(diagonal: SIMD4(0.25, 0.25, -0.1, 1))
        let camera = try ShaderGraphCamera(viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: 0, far: 10)
        return ArrayCamera.View(
            camera: camera, position: .zero, viewport: viewport, directionToCamera: SIMD3(0, 0, 1),
            scissor: scissor, layers: layers
        )
    }

    private func glassPlane(_ fixture: GeometryRenderFixture, size: Float) throws -> MeshRenderer {
        let material = try PhysicallyBasedMaterial(
            roughness: 0, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.8, 0.6, 1, 1),
            alphaMode: .opaque, transmission: try PhysicallyBasedMaterial.Transmission(factor: 1, indexOfRefraction: 1)
        )
        return try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: size, height: size),
            material: .physicallyBased(material)
        )
    }

    private func makeBackdrop(_ fixture: GeometryRenderFixture) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: fixture.side, height: fixture.side, mipmapped: true
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        return try #require(fixture.device.makeTexture(descriptor: descriptor))
    }

    private func viewport(x: Double, y: Double, width: Double, height: Double) -> MTLViewport {
        MTLViewport(originX: x, originY: y, width: width, height: height, znear: 0, zfar: 1)
    }

    private func orthographic(
        x: Float, viewport: MTLViewport, scissor: MTLScissorRect? = nil, layers: SceneLayers? = nil
    ) throws -> ArrayCamera.View {
        let projection = simd_float4x4(diagonal: SIMD4(0.5, 0.5, -0.1, 1))
        let camera = try ShaderGraphCamera(viewMatrix: translationMatrix(-x), projectionMatrix: projection, near: 0, far: 10)
        return ArrayCamera.View(
            camera: camera, position: SIMD3(x, 0, 0), viewport: viewport, directionToCamera: SIMD3(0, 0, 1),
            scissor: scissor, layers: layers
        )
    }

    private func unlitPlane(
        _ fixture: GeometryRenderFixture, size: Float, mode: MeshDepthMode = .forward, sampleCount: Int = 1
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: size, height: size),
            material: .physicallyBased(.init(shadingModel: .unlit)), depthMode: mode, sampleCount: sampleCount
        )
    }

    private func node(_ mesh: MeshRenderer, at position: SIMD3<Float>, tint: SIMD4<Float>, layers: SceneLayers = .default) -> SceneNode {
        SceneNode(transform: .init(translation: position), mesh: mesh, materialParameters: tint, layers: layers)
    }
}
