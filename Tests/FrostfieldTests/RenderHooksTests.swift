import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Records hook calls in order. The hooks and the encode run on the test's thread.
private final class HookLog {
    var before: [SceneRenderObjectContext] = []
    var after: [SceneRenderObjectContext] = []
    var events: [String] = []
    var labels: [UUID: String] = [:]

    /// Installs both hooks on a node and remembers its label.
    func install(on node: inout SceneNode, label: String) {
        labels[node.id] = label
        node.onBeforeEncode = { [self] context in
            before.append(context)
            events.append("before \(label)")
        }
        node.onAfterEncode = { [self] context in
            after.append(context)
            events.append("after \(label)")
        }
    }

    func contexts(for label: String) -> [SceneRenderObjectContext] {
        before.filter { labels[$0.nodeID] == label }
    }
}

private final class RecordingInspector: SceneEncodeInspector {
    enum Event {
        case begin(SceneEncodeInfo)
        case draw(SceneEncodeInfo, SceneRenderObjectContext, SceneDrawStatistics)
        case finish(SceneEncodeInfo, SceneDrawStatistics)
        case fail(SceneEncodeInfo, any Error)
    }

    var events: [Event] = []
    var graphNodeCounts: [Int] = []

    func sceneEncodeDidBegin(_ info: SceneEncodeInfo, graph: SceneGraph) {
        events.append(.begin(info))
        graphNodeCounts.append(graph.roots.count)
    }

    func sceneEncode(_ info: SceneEncodeInfo, didEncode draw: SceneRenderObjectContext, statistics: SceneDrawStatistics) {
        events.append(.draw(info, draw, statistics))
    }

    func sceneEncodeDidFinish(_ info: SceneEncodeInfo, statistics: SceneDrawStatistics) {
        events.append(.finish(info, statistics))
    }

    func sceneEncodeDidFail(_ info: SceneEncodeInfo, error: any Error) {
        events.append(.fail(info, error))
    }

    var begins: [SceneEncodeInfo] { events.compactMap { if case .begin(let info) = $0 { info } else { nil } } }
    var finishes: [(SceneEncodeInfo, SceneDrawStatistics)] {
        events.compactMap { if case .finish(let info, let statistics) = $0 { (info, statistics) } else { nil } }
    }
    var failures: [(SceneEncodeInfo, any Error)] {
        events.compactMap { if case .fail(let info, let error) = $0 { (info, error) } else { nil } }
    }
    var draws: [(SceneEncodeInfo, SceneRenderObjectContext, SceneDrawStatistics)] {
        events.compactMap { if case .draw(let info, let context, let statistics) = $0 { (info, context, statistics) } else { nil } }
    }
}

/// Forwards every Metal message to a real render command encoder and logs each draw, so a test can
/// see whether a hook ran before or after the draw's commands reached the encoder.
private final class ProbedEncoder: NSObject {
    private let real: any MTLRenderCommandEncoder
    private let log: HookLog

    init(real: any MTLRenderCommandEncoder, log: HookLog) {
        self.real = real
        self.log = log
    }

    override func forwardingTarget(for selector: Selector!) -> Any? {
        let name = NSStringFromSelector(selector)
        if name.hasPrefix("draw") || name.hasPrefix("executeCommandsInBuffer") { log.events.append("draw") }
        return real
    }

    override func responds(to selector: Selector!) -> Bool {
        super.responds(to: selector) || real.responds(to: selector)
    }
}

/// Forwards every Metal message to a real command buffer, except that it wraps the render command
/// encoders it hands out in a ``ProbedEncoder``.
private final class ProbedCommandBuffer: NSObject {
    private let real: any MTLCommandBuffer
    private let log: HookLog

    init(real: any MTLCommandBuffer, log: HookLog) {
        self.real = real
        self.log = log
    }

    @objc(renderCommandEncoderWithDescriptor:)
    func makeEncoder(_ descriptor: MTLRenderPassDescriptor) -> NSObject? {
        real.makeRenderCommandEncoder(descriptor: descriptor).map { ProbedEncoder(real: $0, log: log) }
    }

    override func forwardingTarget(for selector: Selector!) -> Any? { real }

    override func responds(to selector: Selector!) -> Bool {
        super.responds(to: selector) || real.responds(to: selector)
    }

    var command: any MTLCommandBuffer { unsafeBitCast(self, to: (any MTLCommandBuffer).self) }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RenderHooksTests {
    private let scenes: RenderHooksScenes
    private var gpu: GeometryRenderFixture { scenes.fixture.base }

    init() throws {
        scenes = try RenderHooksScenes()
    }

    private func hooked(_ labeled: [RenderHooksScenes.Labeled], log: HookLog, only: Set<String>? = nil) -> [SceneNode] {
        labeled.map { item in
            var node = item.node
            if only?.contains(item.label) != false { log.install(on: &node, label: item.label) }
            return node
        }
    }

    @Test("Every drawable node kind calls its hooks once, with its final transform and material state")
    func kindsCallHooks() throws {
        let log = HookLog()
        let labeled = try scenes.kinds()
        let nodes = hooked(labeled, log: log)
        let rendered = try scenes.fixture.render(scenes.graph(nodes))

        #expect(log.before.count == 7 && log.after.count == 7)
        #expect(rendered.statistics.drawnPrimitiveCount == 3)
        for item in labeled {
            let before = log.contexts(for: item.label)
            let after = log.after.filter { log.labels[$0.nodeID] == item.label }
            #expect(before.count == 1 && after == before, "\(item.label) must see the same context before and after its draw")
            guard let context = before.first else { continue }
            #expect(context.kind == item.kind, "\(item.label)")
            #expect(context.pass == .color && context.viewIndex == nil && context.viewCount == nil)
            #expect(context.depthMode == .forward && !context.usesMaterialOverride)
            #expect(context.instanceCount == (item.label == "instanced" ? 2 : nil))
            #expect(context.worldTransform == item.node.transform.matrix, "\(item.label) world transform")
            #expect(context.materialParameters == item.node.materialParameters)
        }
        #expect(Set(log.before.map(\.drawIndex)) == Set(0..<7), "indices name seven distinct draws")

        // The transform and tint the hook saw are the ones the draw used: the square's center
        // pixel carries the node's tint at the position that the context's matrix projects to.
        for (label, tint) in [("plain", SIMD4<Float>(1, 0, 0, 1)), ("skinned", SIMD4(0, 0, 1, 1))] {
            let context = try #require(log.contexts(for: label).first)
            let center = SIMD3(context.worldTransform.columns.3.x, context.worldTransform.columns.3.y, context.worldTransform.columns.3.z)
            let (x, y) = scenes.pixel(of: center)
            let pixel = gpu.pixel(rendered.pixels, x: x, y: y)
            #expect(isNearly(pixel, GeometryRenderFixture.unlitFacingCamera(tint)), "\(label) at (\(x), \(y)) is \(pixel)")
        }
    }

    @Test("A node with a mesh and a primitive reports one context per draw")
    func meshAndPrimitiveOnOneNode() throws {
        let log = HookLog()
        var node = try #require(scenes.kinds().first { $0.label == "points" }).node
        node.mesh = try gpu.unlit(try GeometryRenderFixture.square(0.5))
        log.install(on: &node, label: "both")
        let statistics = try scenes.fixture.render(scenes.graph([node])).statistics
        #expect(statistics.drawnMeshCount == 1 && statistics.drawnPrimitiveCount == 1)
        #expect(Set(log.before.map(\.kind)) == [.mesh, .points] && log.before.count == 2)
        #expect(Set(log.before.map(\.drawIndex)) == [0, 1])
        #expect(log.events == ["before both", "after both", "before both", "after both"], "each draw is bracketed")
    }

    @Test("Hooks run before the draw's commands reach the encoder and after, in draw order")
    func orderAgainstEncoder() throws {
        let log = HookLog()
        let nodes = hooked(try scenes.kinds(), log: log)
        let graph = scenes.graph(nodes)
        let target = try gpu.makeTarget()
        let real = try #require(gpu.queue.makeCommandBuffer())
        let probe = ProbedCommandBuffer(real: real, log: log)
        try graph.encode(pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: probe.command)
        try gpu.finish(real)

        // Each draw is bracketed: before, one or more draw calls, after. Nothing draws outside a bracket.
        var index = 0
        var order: [Int] = []
        while index < log.events.count {
            let event = log.events[index]
            try #require(event.hasPrefix("before "), "event \(index) is \(event); expected a before hook")
            let label = event.dropFirst("before ".count)
            index += 1
            var draws = 0
            while index < log.events.count, log.events[index] == "draw" { draws += 1; index += 1 }
            #expect(draws >= 1, "\(label) encoded no draw between its hooks")
            try #require(index < log.events.count && log.events[index] == "after \(label)",
                         "after \(label) must follow its draws, found \(index < log.events.count ? log.events[index] : "end")")
            index += 1
            order.append(log.before[order.count].drawIndex)
        }
        #expect(order == Array(0..<7) && log.before.count == 7)
        #expect(log.events.filter { $0 == "draw" }.count >= 7)
    }

    @Test("A hidden, filtered, culled, empty or group node never calls its hooks, and an uncullable one still does")
    func suppressedNodes() throws {
        let log = HookLog()
        let square = try GeometryRenderFixture.square(0.5)
        let mesh = try gpu.unlit(square)
        let green = SceneLayers(layer: 1)
        func node(_ label: String, x: Float = 0, _ configure: (inout SceneNode) -> Void = { _ in }) -> SceneNode {
            var result = SceneNode(transform: .init(translation: SIMD3(x, 0, -4)), mesh: mesh, name: label)
            configure(&result)
            log.install(on: &result, label: label)
            return result
        }
        var group = SceneNode(transform: .init(translation: SIMD3(0, 1, 0)), children: [
            node("child of hidden parent") { _ in }
        ], name: "hidden group")
        group.isVisible = false
        log.install(on: &group, label: "hidden group")
        var transformOnly = SceneNode(children: [node("child of group", x: -1)], name: "group")
        log.install(on: &transformOnly, label: "group")
        let emptyInstances = try InstanceSet(device: gpu.device, transforms: [], capacity: 1)
        let instanceable = try gpu.unlit(square, supportsInstancing: true)
        let roots = [
            node("visible", x: 1),
            node("hidden") { $0.isVisible = false },
            node("other layer") { $0.layers = green },
            node("culled", x: 90),
            node("uncullable", x: 90) { $0.frustumCulled = false },
            node("no instances") { $0.mesh = instanceable; $0.instances = emptyInstances },
            group, transformOnly,
        ]
        let rendered = try scenes.fixture.render(scenes.graph(roots))
        #expect(Set(log.before.compactMap { log.labels[$0.nodeID] }) == ["visible", "uncullable", "child of group"])
        #expect(log.before.count == 3 && log.after.count == 3)
        #expect(rendered.statistics.drawnMeshCount == 3 && rendered.statistics.culledMeshCount == 1)

        // An empty scissor skips every draw of the camera, so no hook runs.
        log.before.removeAll(); log.after.removeAll()
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let skipped = try scenes.graph(roots).encode(
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
            scissor: MTLScissorRect(x: 0, y: 0, width: 0, height: 0)
        )
        try gpu.finish(command)
        #expect(skipped.drawnMeshCount == 0 && log.before.isEmpty && log.after.isEmpty)
    }

    @Test("Hooks run once per drawn view of an array-camera encode, and the preflight pass calls none")
    func arrayCameraViews() throws {
        let log = HookLog()
        let green = SceneLayers(layer: 1)
        let mesh = try gpu.unlit(try GeometryRenderFixture.square(0.5))
        func node(_ label: String, x: Float, layers: SceneLayers = .default) -> SceneNode {
            var result = SceneNode(transform: .init(translation: SIMD3(x, 0, -2)), mesh: mesh, layers: layers)
            log.install(on: &result, label: label)
            return result
        }
        var graph = scenes.graph([
            node("default only", x: -0.8), node("green", x: 0.8, layers: green), node("culled", x: 100),
        ])
        graph.cameraLayers = [.default, green]
        func view(_ x: Double, layers: SceneLayers?) throws -> ArrayCamera.View {
            let projection = simd_float4x4(diagonal: SIMD4(0.5, 0.5, -0.1, 1))
            let camera = try ShaderGraphCamera(viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: 0, far: 10)
            return ArrayCamera.View(
                camera: camera, position: .zero, viewport: MTLViewport(originX: x, originY: 0, width: 32, height: 64, znear: 0, zfar: 1),
                directionToCamera: SIMD3(0, 0, 1), layers: layers
            )
        }
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
            arrayCamera: ArrayCamera(views: [try view(0, layers: nil), try view(32, layers: green)])
        )
        try gpu.finish(command)
        #expect(statistics.map(\.drawnMeshCount) == [2, 1])
        // View 0 sees both nodes on the graph's layers; view 1 narrows to green. The culled node is in neither.
        let seen = log.before.map { "\(log.labels[$0.nodeID]!)@\($0.viewIndex!)/\($0.viewCount!)#\($0.drawIndex)" }
        #expect(seen == ["default only@0/2#0", "green@0/2#1", "green@1/2#0"], "\(seen)")
        #expect(log.after == log.before)
        #expect(log.events == ["before default only", "after default only", "before green", "after green", "before green", "after green"])

        // One view still reports its index and count.
        log.before.removeAll(); log.after.removeAll()
        let single = try #require(gpu.queue.makeCommandBuffer())
        try graph.encode(
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: single,
            arrayCamera: ArrayCamera(views: [try view(0, layers: nil)])
        )
        try gpu.finish(single)
        #expect(log.before.map(\.viewIndex) == [0, 0] && log.before.map(\.viewCount) == [1, 1])
    }

    @Test("Draw indices count every draw, so a node sees the same index whichever other nodes have hooks")
    func drawIndicesAreStable() throws {
        let all = HookLog()
        _ = try scenes.fixture.render(scenes.graph(hooked(try scenes.kinds(), log: all)))
        let some = HookLog()
        let only: Set<String> = ["batched", "sprites"]
        _ = try scenes.fixture.render(scenes.graph(hooked(try scenes.kinds(), log: some, only: only)))
        #expect(some.before.count == 2)
        for label in only {
            #expect(some.contexts(for: label).first?.drawIndex == all.contexts(for: label).first?.drawIndex, "\(label)")
        }
    }

    @Test("A material override replaces the parameters the hook sees, and the draw uses them")
    func materialOverrideState() throws {
        let log = HookLog()
        var plain = try #require(scenes.kinds().first { $0.label == "plain" }).node
        log.install(on: &plain, label: "plain")
        var graph = scenes.graph([plain])
        let override = try SceneMaterialOverride(material: .unlit, parameters: SIMD4(0, 0, 1, 1))
        try override.prepare(in: graph, colorPixelFormat: .rgba16Float)
        graph.overrideMaterial = override
        let rendered = try scenes.fixture.render(graph)
        let context = try #require(log.before.first)
        #expect(log.before.count == 1 && context.usesMaterialOverride)
        #expect(context.materialParameters == SIMD4(0, 0, 1, 1) && plain.materialParameters == SIMD4(1, 0, 0, 1))
        let (x, y) = scenes.pixel(of: SIMD3(-2.6, 2.4, -4))
        #expect(isNearly(gpu.pixel(rendered.pixels, x: x, y: y), GeometryRenderFixture.unlitFacingCamera(SIMD4(0, 0, 1, 1))))
    }

    @Test("The surface attribute prepass calls hooks for the opaque meshes it draws and reports its pass")
    func surfaceAttributePass() throws {
        let log = HookLog()
        let nodes = hooked(try scenes.kinds(), log: log)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let color = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let depth = try gpu.makeTarget().depth
        let command = try #require(gpu.queue.makeCommandBuffer())
        let statistics = try scenes.graph(nodes).encodeSurfaceAttributes(
            pass: gpu.pass(color: color, depth: depth), commandBuffer: command
        )
        try gpu.finish(command)
        // Plain, instanced, skinned and batched meshes draw; primitives do not write attributes.
        #expect(Set(log.before.compactMap { log.labels[$0.nodeID] }) == ["plain", "instanced", "skinned", "batched"])
        #expect(log.before.allSatisfy { $0.pass == .surfaceAttributes } && log.after == log.before)
        #expect(log.before.count == 4 && statistics.drawnMeshCount >= 4)
        #expect(Set(log.before.map(\.drawIndex)) == Set(0..<4))
    }

    @Test("An opaque and an overlay pass of one transmission encode share one draw count")
    func transmissionSplitsOnePassCount() throws {
        let log = HookLog()
        let material = try PhysicallyBasedMaterial(
            roughness: 0, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.8, 0.6, 1, 1),
            alphaMode: .opaque, transmission: try PhysicallyBasedMaterial.Transmission(factor: 1, indexOfRefraction: 1)
        )
        let glass = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(material)
        )
        var behind = SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: try gpu.unlit(try GeometryRenderFixture.square(1)))
        var front = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: glass)
        log.install(on: &behind, label: "opaque")
        log.install(on: &front, label: "glass")
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: true
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let backdrop = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        // The glass is listed first, so only the transmission split can put the opaque mesh ahead of it.
        let statistics = try scenes.graph([front, behind]).encode(
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command, transmissionBackdrop: backdrop
        )
        try gpu.finish(command)
        #expect(statistics.drawnMeshCount == 2)
        #expect(log.events == ["before opaque", "after opaque", "before glass", "after glass"], "\(log.events)")
        #expect(log.before.map(\.drawIndex) == [0, 1])
    }

    @Test("The inspector sees begin, every draw with its own counts, and finish with the returned statistics")
    func inspectorFollowsEncode() throws {
        let inspector = RecordingInspector()
        var graph = scenes.graph(try scenes.kinds().map(\.node))
        graph.encodeInspector = inspector
        let rendered = try scenes.fixture.render(graph)
        let draws = inspector.draws
        #expect(inspector.begins.count == 1 && inspector.finishes.count == 1 && inspector.failures.isEmpty)
        #expect(draws.count == 7)
        guard case .begin? = inspector.events.first, case .finish? = inspector.events.last else {
            Issue.record("begin must come first and finish last: \(inspector.events.count) events")
            return
        }
        let info = try #require(inspector.begins.first)
        #expect(info.id != 0 && info.pass == .color && info.viewIndex == nil && info.depthMode == .forward)
        #expect(draws.allSatisfy { $0.0 == info } && inspector.finishes.first?.0 == info)
        #expect(inspector.finishes.first?.1 == rendered.statistics && inspector.graphNodeCounts == [7])
        #expect(draws.map(\.1.drawIndex) == Array(0..<7), "per-draw events arrive in draw order")

        // The draws' counts add up to the pass counters the encode returns.
        var sum = SceneDrawStatistics()
        for (_, _, counts) in draws {
            sum.drawnMeshCount += counts.drawnMeshCount
            sum.drawnPrimitiveCount += counts.drawnPrimitiveCount
            sum.drawnInstanceCount += counts.drawnInstanceCount
            sum.submittedTriangleCount += counts.submittedTriangleCount
            sum.submittedPointCount += counts.submittedPointCount
            sum.submittedLineSegmentCount += counts.submittedLineSegmentCount
            sum.submittedSpriteCount += counts.submittedSpriteCount
            sum.indirectBatchExecutionCount += counts.indirectBatchExecutionCount
            sum.indirectArgumentDrawCount += counts.indirectArgumentDrawCount
        }
        let total = rendered.statistics
        #expect(sum.drawnMeshCount == total.drawnMeshCount && sum.drawnPrimitiveCount == total.drawnPrimitiveCount)
        #expect(sum.drawnInstanceCount == total.drawnInstanceCount && sum.submittedTriangleCount == total.submittedTriangleCount)
        #expect(sum.submittedPointCount == 3 && sum.submittedPointCount == total.submittedPointCount)
        #expect(sum.submittedLineSegmentCount == 1 && sum.submittedLineSegmentCount == total.submittedLineSegmentCount)
        #expect(sum.submittedSpriteCount == 2 && sum.submittedSpriteCount == total.submittedSpriteCount)
        #expect(sum.indirectBatchExecutionCount == total.indirectBatchExecutionCount)
        // One draw's own counts: the plain square is one draw of two triangles.
        let plain = try #require(draws.first { $0.1.kind == .mesh && $0.1.instanceCount == nil && $0.1.materialParameters == SIMD4(1, 0, 0, 1) })
        #expect(plain.2.drawnMeshCount == 1 && plain.2.submittedTriangleCount == 2 && plain.2.drawnInstanceCount == 1)
        let instanced = try #require(draws.first { $0.1.instanceCount == 2 })
        #expect(instanced.2.drawnInstanceCount == 2)

        // Every encode gets its own ID.
        _ = try scenes.fixture.render(graph)
        let ids = inspector.begins.map(\.id)
        #expect(ids.count == 2 && ids[0] != ids[1] && ids[1] > ids[0])
    }

    @Test("An encode that throws reports the failure and no finish, and array-camera preflight is not reported")
    func inspectorFailuresAndArrayCamera() throws {
        let inspector = RecordingInspector()
        var inView = try #require(scenes.kinds().first).node
        inView.transform = .init(translation: SIMD3(0, 0, -2))
        var graph = scenes.graph([inView])
        graph.encodeInspector = inspector
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let outside = MTLViewport(originX: 60, originY: 60, width: 32, height: 32, znear: 0, zfar: 1)
        #expect(throws: MeshRenderer.Failure.viewport) {
            try graph.encode(
                pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
                clipFromWorld: simd_float4x4(diagonal: SIMD4(0.5, 0.5, -0.1, 1)), cameraPosition: .zero,
                directionToCamera: SIMD3(0, 0, 1), viewport: outside
            )
        }
        try gpu.finish(command)
        #expect(inspector.begins.count == 1 && inspector.finishes.isEmpty)
        #expect((inspector.failures.first?.1 as? MeshRenderer.Failure) == .viewport)
        #expect(inspector.failures.first?.0 == inspector.begins.first)

        // Two views preflight the second one without encoding, yet are reported exactly twice.
        let multiple = RecordingInspector()
        graph.encodeInspector = multiple
        func view(_ x: Double) throws -> ArrayCamera.View {
            let projection = simd_float4x4(diagonal: SIMD4(0.5, 0.5, -0.1, 1))
            let camera = try ShaderGraphCamera(viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: 0, far: 10)
            return ArrayCamera.View(
                camera: camera, position: .zero, viewport: MTLViewport(originX: x, originY: 0, width: 32, height: 64, znear: 0, zfar: 1),
                directionToCamera: SIMD3(0, 0, 1)
            )
        }
        let second = try #require(gpu.queue.makeCommandBuffer())
        try graph.encode(
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: second,
            arrayCamera: ArrayCamera(views: [try view(0), try view(32)])
        )
        try gpu.finish(second)
        #expect(multiple.begins.map(\.viewIndex) == [0, 1] && multiple.begins.map(\.viewCount) == [2, 2])
        #expect(multiple.finishes.count == 2 && multiple.failures.isEmpty)
        var order: [String] = []
        for event in multiple.events {
            switch event {
            case .begin(let info): order.append("begin \(info.viewIndex!)")
            case .draw(_, let context, _): order.append("draw \(context.viewIndex!)")
            case .finish(let info, _): order.append("finish \(info.viewIndex!)")
            case .fail: order.append("fail")
            }
        }
        #expect(order == ["begin 0", "draw 0", "finish 0", "begin 1", "draw 1", "finish 1"], "\(order)")
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RenderHooksDefaultPathTests {
    /// Digests measured on the unmodified tip b756c35, on the Mac Studio's Apple M4 Max.
    private static let baselineDevice = "Apple M4 Max"
    private static let baseline: [String: UInt64] = [
        "plain": 0xd56bc48b3618ada5, "instanced": 0xc13314b3449ddecc, "skinned": 0x62762605f4fbcf25,
        "batched": 0xe191ca19629a43fd, "points": 0xb7c837bb36d16725, "line": 0xbd56c18f0b897a1c,
        "sprites": 0x08f2f9703e19d625, "all": 0x42276cc6deb254fd,
    ]

    @Test("Hooks and an inspector leave every pixel and every statistic of the default path unchanged")
    func hooksDoNotChangeTheRender() throws {
        let scenes = try RenderHooksScenes()
        for (name, graph) in try scenes.digestScenes() {
            let plain = try scenes.fixture.render(graph)
            #expect(plain.pixels.contains { $0.w > 0 }, "scene \(name) drew nothing")
            #expect(plain.statistics.drawnMeshCount + plain.statistics.drawnPrimitiveCount > 0)

            // The same scene with every hook and an inspector installed.
            var observed = graph
            observed.encodeInspector = RecordingInspector()
            let counter = HookLog()
            for index in observed.roots.indices { counter.install(on: &observed.roots[index], label: "\(index)") }
            let hooked = try scenes.fixture.render(observed)
            #expect(counter.before.count == (name == "all" ? 7 : 1), "scene \(name) must have called its hooks")
            #expect(hooked.pixels == plain.pixels, "scene \(name) pixels changed when hooks were installed")
            #expect(hooked.statistics == plain.statistics, "scene \(name) statistics changed")
            #expect(RenderHooksScenes.digest(hooked.pixels) == RenderHooksScenes.digest(plain.pixels))

            let digest = RenderHooksScenes.digest(plain.pixels)
            print("RENDER-HOOKS-DIGEST \(name) \(String(digest, radix: 16)) device=\(scenes.fixture.device.name)")
            #if os(macOS)
            if scenes.fixture.device.name == Self.baselineDevice, let expected = Self.baseline[name] {
                #expect(digest == expected, "scene \(name) differs from the digest recorded on the unmodified tip")
                print("RENDER-HOOKS-BASELINE-COMPARED \(name)")
            }
            #endif
        }
    }
}
