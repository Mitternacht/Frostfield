import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct BatchedMeshTests {
    let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    @Test("Different geometries share indexed draws and cull each visible slot")
    func draws() throws {
        let square = try GeometryRenderFixture.square(0.24)
        let triangle = try triangleGeometry()
        let batch = try makeBatch([square, triangle])
        let placements = [
            SceneTransform(translation: SIMD3(-1.25, 0, 0)),
            SceneTransform(translation: SIMD3(-0.55, 0, 0)),
            SceneTransform(translation: SIMD3(0.7, 0, 0), scale: SIMD3(-1, 1, 1)),
        ]
        let tints: [SIMD4<Float>] = [SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1)]
        for (index, placement) in placements.enumerated() {
            try batch.addInstance(geometryIndex: index < 2 ? 0 : 1, transform: placement.matrix, tint: tints[index])
        }
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-8))
        try batch.addInstance(geometryIndex: 1, transform: translationMatrix(8))
        try batch.addInstance(geometryIndex: 1, transform: translationMatrix(0, 0.8), isVisible: false)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), batch: batch)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let rendered = try fixture.render(graph)
        #expect(rendered.statistics.drawnMeshCount == 2)
        #expect(rendered.statistics.drawnInstanceCount == 3)
        #expect(rendered.statistics.culledMeshCount == 2)

        let squareRenderer = try fixture.unlit(square)
        let triangleRenderer = try fixture.unlit(triangle)
        let referenceChildren = placements.enumerated().map { index, placement in
            SceneNode(
                transform: placement, mesh: index < 2 ? squareRenderer : triangleRenderer,
                materialParameters: tints[index]
            )
        }
        let reference = try fixture.render(SceneGraph(
            roots: [SceneNode(transform: node.transform, children: referenceChildren)], camera: fixture.camera
        ))
        #expect(reference.statistics.drawnMeshCount == 3)
        #expect(zip(reference.pixels, rendered.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.002) })
        #expect(fixture.pixel(rendered.pixels, x: 43, y: 32).z > 0.95)

        batch.perObjectFrustumCulled = false
        let uncull = try fixture.render(graph)
        #expect(uncull.statistics.drawnInstanceCount == 5 && uncull.statistics.culledMeshCount == 0)
        #expect(uncull.pixels == rendered.pixels)
    }

    @Test("Transparent sorting preserves back-to-front composition while grouping adjacent ranges")
    func transparentOrder() throws {
        let square = try GeometryRenderFixture.square(0.45)
        let triangle = try triangleGeometry()
        let batch = try makeBatch([square, triangle], state: .transparent)
        let red = SIMD4<Float>(1, 0, 0, 0.5)
        let blue = SIMD4<Float>(0, 0, 1, 0.5)
        let green = SIMD4<Float>(0, 1, 0, 0.5)
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -1), tint: red)
        try batch.addInstance(geometryIndex: 1, transform: translationMatrix(0, 0, -3), tint: blue)
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -2), tint: green)
        let graph = SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)
        let sorted = try fixture.render(graph)
        #expect(sorted.statistics.drawnMeshCount == 2 && sorted.statistics.drawnInstanceCount == 3)
        let sortedColor = fixture.pixel(sorted.pixels, x: 32, y: 32)
        #expect(isNearly(sortedColor, SIMD4(0.11, 0.055, 0.125, 0.875)))

        batch.sortsObjects = false
        let bySlot = try fixture.render(graph)
        #expect(bySlot.statistics.drawnMeshCount == 3)
        #expect(isNearly(fixture.pixel(bySlot.pixels, x: 32, y: 32), SIMD4(0.0275, 0.11, 0.25, 0.875)))
    }

    @Test("Stable slot edits do not overwrite an earlier draw's transforms or colors")
    func updates() throws {
        let batch = try makeBatch([try GeometryRenderFixture.square(0.24), try triangleGeometry()], capacity: 2)
        let original = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-0.7), tint: SIMD4(1, 0, 0, 1))
        let second = try batch.addInstance(geometryIndex: 1, transform: translationMatrix(0.7), isVisible: false)
        #expect(throws: BatchedMesh.Failure.capacity) { try batch.addInstance(geometryIndex: 0) }
        #expect(throws: BatchedMesh.Failure.geometry(3)) { try batch.update(second, geometryIndex: 3, isVisible: true) }
        #expect(try batch.instance(for: second).isVisible == false)

        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), batch: batch)], camera: fixture.camera
        )
        let target = try fixture.makeTarget()
        let earlier = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let oldStats = try graph.encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: earlier)
        #expect(oldStats.drawnInstanceCount == 1)
        try batch.remove(original)
        let replacement = try batch.addInstance(
            geometryIndex: 1, transform: translationMatrix(0.7), tint: SIMD4(0, 0, 1, 1)
        )
        #expect(original.index == replacement.index && original != replacement)
        #expect(throws: BatchedMesh.Failure.instance) { try batch.update(original, isVisible: false) }
        try batch.update(second, geometryIndex: 0, transform: translationMatrix(0, 0.7), isVisible: true)
        let current = try fixture.render(graph)
        #expect(current.statistics.drawnInstanceCount == 2)
        try fixture.finish(earlier)
        let old = try fixture.pixels(of: target.color)
        #expect(isNearly(fixture.pixel(old, x: 21, y: 32), SIMD4(0.22, 0, 0, 1)))
        #expect(fixture.pixel(old, x: 43, y: 32) == .zero)
        #expect(fixture.pixel(current.pixels, x: 21, y: 32) == .zero)
        #expect(fixture.pixel(current.pixels, x: 43, y: 32) == SIMD4(0, 0, 1, 1))
        #expect(fixture.pixel(current.pixels, x: 32, y: 21).w == 1)
    }

    @Test("Picking uses each visible slot's own geometry and reports stable handles")
    func picks() throws {
        let batch = try makeBatch([try GeometryRenderFixture.square(0.45), try triangleGeometry()])
        let far = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -3))
        let near = try batch.addInstance(geometryIndex: 1, transform: translationMatrix(0, 0, -2))
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -1), isVisible: false)
        let node = SceneNode(batch: batch)
        let graph = SceneGraph(roots: [SceneNode(children: [node])], camera: fixture.camera)
        let point = SIMD2<Float>(0.52, 0.485)
        let first = try #require(try graph.pick(at: point, aspectRatio: 1))
        #expect(first.nodeID == node.id && first.instanceIndex == nil && first.batchInstanceID == near)
        let hits = try graph.intersections(at: point, aspectRatio: 1)
        #expect(hits.map(\.batchInstanceID) == [near, far])
        try batch.update(near, isVisible: false)
        #expect(try graph.pick(at: point, aspectRatio: 1)?.batchInstanceID == far)
    }

    @Test("A batch selects individual casters in the directional light view")
    func shadowCasters() throws {
        let receiver = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let batch = try makeBatch([try GeometryRenderFixture.square(0.25), try triangleGeometry()])
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0.75))
        try batch.addInstance(geometryIndex: 1, transform: translationMatrix(-0.55, 0.6))
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(12, 0.6))
        let blockers = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), batch: batch)
        let shadow = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        let graph = SceneGraph(roots: [receiver, blockers], camera: fixture.camera)
        var passes: [BatchedMesh.SortPass] = []
        batch.customSort = { items, context in
            passes.append(context.pass)
            #expect(items.count == 2 && !context.isBlended)
            return items.map(\.instance.id)
        }
        let output = try fixture.render(graph, shadowMap: shadow)
        #expect(passes == [.color, .directionalShadow(cascade: 0)])
        #expect(output.statistics.shadowDrawCount == 2 && output.statistics.culledShadowCount == 1)
        #expect(abs(fixture.pixel(output.pixels, x: 32, y: 32).x - 0.16) < 0.01)
    }

    @Test("Unretained encodes keep the batch's geometry and draw records alive after teardown")
    func teardown() throws {
        let target = try fixture.makeTarget()
        let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        try autoreleasepool {
            let batch = try makeBatch([try GeometryRenderFixture.square(0.25)])
            try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -2))
            let graph = SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)
            try graph.encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command)
        }
        try fixture.finish(command)
        #expect(try fixture.pixel(fixture.pixels(of: target.color), x: 32, y: 32).w == 1)
    }

    @Test("The surface attribute prepass writes visible batched geometry")
    func attributes() throws {
        let source = try GeometryRenderFixture.square(0.35)
        let tilted = try MeshGeometry(vertices: source.vertices.map {
            MeshVertex(position: $0.position, normal: simd_normalize(SIMD3(0, 1, 1)))
        }, indices: source.indices)
        let batch = try makeBatch([tilted])
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-0.6, 0, -2))
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0.6, 0, -2), isVisible: false)
        let graph = SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: fixture.side, height: fixture.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let color = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let command = try #require(fixture.queue.makeCommandBuffer())
        let statistics = try graph.encodeSurfaceAttributes(
            pass: fixture.pass(color: color, depth: target.depth), commandBuffer: command
        )
        #expect(statistics.drawnMeshCount == 1 && statistics.drawnInstanceCount == 1)
        try fixture.finish(command)
        func normal(at x: Int) -> SIMD2<UInt32> {
            var value = SIMD2<UInt32>.zero
            color.getBytes(&value, bytesPerRow: MemoryLayout<SIMD2<UInt32>>.stride,
                           from: MTLRegionMake2D(x, 32, 1, 1), mipmapLevel: 0)
            return value
        }
        #expect(normal(at: 22).x != 0)
        #expect(normal(at: 42) == .zero)
    }

    @Test("Mesh-only export flattens visible batch slots and scene archive rejects unrepresented batches")
    func snapshots() throws {
        let batch = try makeBatch([try triangleGeometry()])
        let first = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(2, 0, -2))
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(6, 0, -2), isVisible: false)
        let node = SceneNode(name: "Batch", batch: batch)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let data = try OBJExporter.data(from: graph)
        let obj = String(decoding: data, as: UTF8.self)
        #expect(obj.contains("o Batch_\(first.index)"))
        #expect(obj.split(separator: "\n").filter { $0.hasPrefix("v ") }.count == 3)
        let hiddenRay = try SceneRay(origin: SIMD3(6, 0, 0), direction: SIMD3(0, 0, -1))
        #expect(try graph.collisionBVH()?.intersection(ray: hiddenRay) != nil)
        #expect(try graph.collisionBVH(visibleOnly: true)?.intersection(ray: hiddenRay) == nil)
        #expect(throws: SceneArchive.Failure.unsupportedBatch(node.id)) { try SceneArchive(capturing: graph) }
        #expect(throws: PathTracingScene.Failure.unsupported(node.id, "batched meshes")) {
            try PathTracingScene(queue: fixture.queue, graph: graph)
        }
    }

    @Test("Box selection uses transformed batch origins and ignores vacant slots")
    func selection() throws {
        let displaced = try MeshGeometry.merged([
            .init(try triangleGeometry(), transform: translationMatrix(3))
        ]).geometry
        let batch = try makeBatch([displaced])
        let deleted = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-0.35))
        let hidden = try batch.addInstance(geometryIndex: 0, isVisible: false)
        let visible = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0.35))
        try batch.remove(deleted)
        let node = SceneNode(batch: batch)
        var graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0.35, 0, -2)), children: [node])],
            camera: fixture.camera
        )
        let rectangle = try SceneSelectionRectangle(from: SIMD2(0.48, 0.48), to: SIMD2(0.7, 0.52))
        let selected = try graph.select(in: rectangle, aspectRatio: 1)
        #expect(selected.map(\.nodeID) == [node.id, node.id])
        #expect(selected.allSatisfy { $0.instanceIndex == nil })
        #expect(selected.map(\.batchInstanceID) == [hidden, visible])

        let reused = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-0.35))
        #expect(reused.index == deleted.index && reused != deleted)
        #expect(try graph.select(in: rectangle, aspectRatio: 1).map(\.batchInstanceID) == [reused, hidden, visible])
        graph.roots[0].isVisible = false
        #expect(try graph.select(in: rectangle, aspectRatio: 1).isEmpty)
    }

    @Test("Batch bounds include hidden active slots until visible-only filtering is requested")
    func worldBounds() throws {
        let geometry = try GeometryRenderFixture.square(0.25)
        let batch = try makeBatch([geometry])
        let first = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-1))
        let hidden = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(2), isVisible: false)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 1, -2)), batch: batch)
        var graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(5, 0, 0)), children: [node])],
            camera: fixture.camera
        )
        let all = try #require(try graph.worldBounds(of: node.id))
        #expect(all.minimum == SIMD3(3.75, 0.75, -2))
        #expect(all.maximum == SIMD3(7.25, 1.25, -2))
        let visible = try #require(try graph.worldBounds(precise: true, visibleOnly: true))
        #expect(visible.maximum == SIMD3(4.25, 1.25, -2))
        #expect(try batch.bounds(visibleOnly: true)?.minimum == SIMD3(-1.25, -0.25, 0))
        try batch.remove(first)
        #expect(try graph.worldBounds(visibleOnly: true) == nil)
        #expect(try graph.worldBounds()?.minimum.x == 6.75)
        try batch.update(hidden, isVisible: true)
        graph.roots[0].isVisible = false
        #expect(try graph.worldBounds(visibleOnly: true) == nil)
        #expect(try graph.worldBounds()?.maximum.x == 7.25)
    }

    @Test("Palette replacement moves all referencing draws while pending renders keep their original index ranges")
    func paletteEdits() throws {
        let square = try GeometryRenderFixture.square(0.24)
        let triangle = try triangleGeometry()
        let batch = try BatchedMesh(
            device: fixture.device, colorPixelFormat: .rgba16Float, capacity: 4,
            vertexCapacity: 7, indexCapacity: 9
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), batch: batch)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        #expect(try fixture.render(graph).statistics.drawnMeshCount == 0)
        let a = try batch.addGeometry(square)
        let b = try batch.addGeometry(triangle)
        let left = try batch.addInstance(geometryID: a, transform: translationMatrix(-0.7), tint: SIMD4(1, 0, 0, 1))
        let second = try batch.addInstance(geometryID: a, transform: translationMatrix(-0.7, 0.7))
        let right = try batch.addInstance(geometryID: b, transform: translationMatrix(0.7), tint: SIMD4(0, 0, 1, 1))
        #expect(try graph.pick(at: SIMD2(0.325, 0.5), aspectRatio: 1)?.batchInstanceID == left)
        #expect(throws: BatchedMesh.Failure.geometryBudget) { try batch.addGeometry(triangle) }
        #expect(batch.geometryIDs == [a, b])

        let target = try fixture.makeTarget()
        let old = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let stats = try graph.encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: old)
        #expect(stats.drawnMeshCount == 2 && stats.drawnInstanceCount == 3)

        let shifted = try MeshGeometry.merged([.init(triangle, transform: translationMatrix(0.6))]).geometry
        try batch.replaceGeometry(a, with: shifted)
        #expect(try batch.geometryID(at: a.index) == a)
        #expect(try batch.instance(for: left).geometryID == a)
        #expect(try graph.pick(at: SIMD2(0.325, 0.5), aspectRatio: 1) == nil)
        #expect(try graph.pick(at: SIMD2(0.475, 0.5), aspectRatio: 1)?.batchInstanceID == left)
        let current = try fixture.render(graph)
        #expect(fixture.pixel(current.pixels, x: 21, y: 32) == .zero)
        #expect(isNearly(fixture.pixel(current.pixels, x: 30, y: 32), SIMD4(0.22, 0, 0, 1)))
        #expect(fixture.pixel(current.pixels, x: 43, y: 32) == SIMD4(0, 0, 1, 1))

        try batch.removeGeometry(a)
        #expect(batch.count == 1 && batch.geometryIDs == [b])
        #expect(throws: BatchedMesh.Failure.instance) { try batch.instance(for: left) }
        #expect(throws: BatchedMesh.Failure.instance) { try batch.instance(for: second) }
        #expect(try batch.instance(for: right).geometryID == b)
        #expect(try graph.pick(at: SIMD2(0.675, 0.5), aspectRatio: 1)?.batchInstanceID == right)
        let reused = try batch.addGeometry(square)
        #expect(reused.index == a.index && reused != a)
        #expect(throws: BatchedMesh.Failure.geometryHandle) { try batch.addInstance(geometryID: a) }

        try fixture.finish(old)
        let original = try fixture.pixels(of: target.color)
        #expect(isNearly(fixture.pixel(original, x: 21, y: 32), SIMD4(0.22, 0, 0, 1)))
        #expect(fixture.pixel(original, x: 30, y: 32) == .zero)
        #expect(fixture.pixel(original, x: 43, y: 32) == SIMD4(0, 0, 1, 1))
    }

    @Test("Three in-flight palettes bound memory and allow edits to retry after completion")
    func paletteVersions() throws {
        let square = try GeometryRenderFixture.square(0.24)
        let triangle = try triangleGeometry()
        let batch = try makeBatch([square])
        let id = try batch.geometryID(at: 0)
        try batch.addInstance(geometryID: id, transform: translationMatrix(0, 0, -2))
        let graph = SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)
        var pending: [any MTLCommandBuffer] = []
        for index in 0..<3 {
            if index > 0 { try batch.replaceGeometry(id, with: index == 1 ? triangle : square) }
            let target = try fixture.makeTarget()
            let command = try #require(fixture.queue.makeCommandBuffer())
            try graph.encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command)
            pending.append(command)
        }
        #expect(throws: BatchedMesh.Failure.paletteBusy) { try batch.replaceGeometry(id, with: triangle) }
        #expect(try batch.geometry(for: id).indices.count == 6)
        try fixture.finish(pending.removeFirst())
        try batch.replaceGeometry(id, with: triangle)
        #expect(try batch.geometry(for: id).indices.count == 3)
        for command in pending { try fixture.finish(command) }
        try batch.removeGeometry(id)
        #expect(batch.count == 0 && batch.geometryIDs.isEmpty)
        #expect(try fixture.render(graph).statistics.drawnMeshCount == 0)
    }

    @Test("Custom draw order uses each pass's visible slots and rejects duplicate or foreign handles")
    func customSort() throws {
        let square = try GeometryRenderFixture.square(0.4)
        let triangle = try triangleGeometry()
        let batch = try makeBatch([square, triangle], state: .transparent)
        let near = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -1), tint: SIMD4(1, 0, 0, 0.5))
        let far = try batch.addInstance(geometryIndex: 1, transform: translationMatrix(0, 0, -2), tint: SIMD4(0, 0, 1, 0.5))
        let culled = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(10, 0, -2))
        let graph = SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)
        var calls = 0
        batch.customSort = { items, context in
            calls += 1
            #expect(context.pass == .color && context.isBlended)
            #expect(items.map(\.instance.id) == [near, far])
            #expect(items[0].worldBounds?.center.z == -1)
            #expect(throws: BatchedMesh.Failure.reentrantSort) { try batch.update(near, isVisible: false) }
            return [near, far]
        }
        let ordered = try fixture.render(graph)
        #expect(calls == 1)
        #expect(isNearly(fixture.pixel(ordered.pixels, x: 32, y: 32), SIMD4(0.055, 0, 0.5, 0.75)))
        batch.customSort = { _, _ in [far, far] }
        #expect(throws: BatchedMesh.Failure.customOrder) { try fixture.render(graph) }
        batch.customSort = { _, _ in [near, culled] }
        #expect(throws: BatchedMesh.Failure.customOrder) { try fixture.render(graph) }
        batch.customSort = { _, _ in [] }
        batch.sortsObjects = false
        #expect(try fixture.render(graph).pixels == ordered.pixels)
    }

    @Test("A sort callback may edit another batch without changing that batch's prepared color and shadow draws")
    func sortsWhileAnotherPaletteChanges() throws {
        let receiver = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let square = try GeometryRenderFixture.square(0.25)
        let blocker = try makeBatch([square])
        let geometry = try blocker.geometryID(at: 0)
        try blocker.addInstance(geometryID: geometry, transform: translationMatrix(0.75, 0, -2))
        let editor = try makeBatch([square])
        try editor.addInstance(geometryIndex: 0, transform: translationMatrix(-0.7, 0.7, -2))
        let shifted = try MeshGeometry.merged([.init(try triangleGeometry(), transform: translationMatrix(2))]).geometry
        editor.customSort = { items, _ in
            try blocker.replaceGeometry(geometry, with: shifted)
            return items.map(\.instance.id)
        }
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        let graph = SceneGraph(
            roots: [receiver, SceneNode(batch: blocker), SceneNode(castsShadow: false, batch: editor)],
            camera: fixture.camera
        )
        let before = try fixture.render(graph, shadowMap: shadowMap)
        #expect(abs(fixture.pixel(before.pixels, x: 31, y: 31).x - 0.16) < 0.01)
        #expect(try blocker.geometry(for: geometry).indices.count == 3)
        editor.customSort = nil
        let after = try fixture.render(graph, shadowMap: shadowMap)
        #expect(abs(fixture.pixel(after.pixels, x: 31, y: 31).x - 0.832) < 0.01)
    }

    private func makeBatch(
        _ geometries: [MeshGeometry], capacity: Int = 8, state: MeshRenderState? = nil
    ) throws -> BatchedMesh {
        try BatchedMesh(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometries: geometries,
            capacity: capacity, renderState: state
        )
    }

    private func triangleGeometry() throws -> MeshGeometry {
        try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(-0.27, -0.23, 0), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0.27, -0.23, 0), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0, 0.3, 0), normal: SIMD3(0, 0, 1)),
        ], indices: [0, 1, 2])
    }
}
