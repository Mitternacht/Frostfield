import Foundation
import Frostfield
import Metal
import Synchronization
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshDrawRangeTests {
    @MainActor
    @Test("Background picking prepares the active range and scene queries keep the original triangle IDs")
    func preparesPickingRange() async throws {
        let fixture = try GeometryRenderFixture()
        let plane = try MeshGeometry.plane(width: 0.68, height: 0.68)
        let geometry = try MeshGeometry.merged([
            .init(plane, transform: translationMatrix(-0.55)),
            .init(plane, transform: translationMatrix(0.55)),
        ]).geometry
        let renderer = try fixture.unlit(geometry)
        renderer.drawRange = try .init(start: 6, count: 6)
        let updates = Mutex<[MeshBVH.BuildProgress]>([])
        let result = try await renderer.preparePickingAcceleration { report in updates.withLock { $0.append(report) } }
        let prepared = try #require(result)
        #expect(prepared.geometry.indices == Array(geometry.indices[6..<12]))
        #expect(updates.withLock { $0.last?.fractionCompleted } == 1)

        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let left = try SceneRay(origin: SIMD3(-0.55, 0, 0), direction: SIMD3(0, 0, -1))
        let right = try SceneRay(origin: SIMD3(0.55, 0, 0), direction: SIMD3(0, 0, -1))
        #expect(try graph.pick(ray: left) == nil)
        let hit = try #require(try graph.pick(ray: right))
        #expect(hit.nodeID == node.id && (2...3).contains(hit.intersection.triangleIndex))
        let repeated = Mutex(0)
        let cached = try await renderer.preparePickingAcceleration { _ in repeated.withLock { $0 += 1 } }
        #expect(cached?.geometry.indices == prepared.geometry.indices)
        #expect(repeated.withLock { $0 } == 0)

        renderer.drawRange = try .init(start: 0, count: 0)
        #expect(try await renderer.preparePickingAcceleration() == nil)
        #expect(try graph.pick(ray: right) == nil)
    }

    @MainActor
    @Test("A geometry or range edit supersedes an in-flight picking build", arguments: [false, true])
    func rejectsStalePickingBuild(geometryEdit: Bool) async throws {
        let fixture = try GeometryRenderFixture()
        let plane = try MeshGeometry.plane(width: 0.6, height: 0.6)
        let geometry = try MeshGeometry.merged([
            .init(plane, transform: translationMatrix(-0.5)),
            .init(plane, transform: translationMatrix(0.5)),
        ]).geometry
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, dynamicGeometry: geometry,
            capacity: .init(vertices: 8, triangles: 4)
        )
        let (ready, signal) = AsyncStream<Bool>.makeStream()
        let release = DispatchSemaphore(value: 0)
        let build = Task {
            defer { signal.finish() }
            return try await renderer.preparePickingAcceleration { report in
                if report.fractionCompleted == 1 {
                    signal.yield(true)
                    _ = release.wait(timeout: .now() + 10)
                }
            }
        }
        defer { release.signal() }
        var iterator = ready.makeAsyncIterator()
        try #require(await iterator.next() == true)
        if geometryEdit {
            let translated = try MeshGeometry.merged([.init(plane, transform: translationMatrix(0.5))]).geometry
            try renderer.updateGeometry(translated)
        } else {
            renderer.drawRange = try .init(start: 6)
        }
        release.signal()
        await #expect(throws: MeshRenderer.Failure.pickingGeometryChanged) { try await build.value }

        let cancelled = Task {
            try await renderer.preparePickingAcceleration { _ in withUnsafeCurrentTask { $0?.cancel() } }
        }
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        let retried = try await renderer.preparePickingAcceleration()
        #expect(retried?.triangleCount == 2)
        let oldSide = try SceneRay(origin: SIMD3(-0.5, 0, 1), direction: SIMD3(0, 0, -1))
        let newSide = try SceneRay(origin: SIMD3(0.5, 0, 1), direction: SIMD3(0, 0, -1))
        #expect(try retried?.intersection(ray: oldSide) == nil)
        #expect(try retried?.intersection(ray: newSide) != nil)
    }

    @Test("Range edits keep raster, picking, collision, path rebuilds and archives on the same triangles")
    func crossRendererRange() throws {
        let gpu = try PathTestFixture()
        let side = 48
        let radiance = SIMD3<Float>(0.75, 0.25, 0.5)
        let square = try MeshGeometry.plane(width: 0.68, height: 0.68)
        let geometry = try MeshGeometry.merged([
            .init(square, transform: translationMatrix(-0.55)),
            .init(square, transform: translationMatrix(0.55)),
        ]).geometry
        let renderer = try gpu.pbr(geometry, color: .zero, emission: radiance)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)
        let graph = SceneGraph(roots: [node], camera: gpu.camera)
        let left = try SceneRay(origin: SIMD3(-0.55, 0, 0), direction: SIMD3(0, 0, -1))
        let right = try SceneRay(origin: SIMD3(0.55, 0, 0), direction: SIMD3(0, 0, -1))
        renderer.drawRange = try MeshDrawRange(start: 0, count: 6)
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: gpu.queue, scene: scene)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 11)
        func traced() throws -> [SIMD4<Float>] {
            try gpu.trace(tracer, width: side, height: side, samples: 1, settings: settings)
        }
        let oldRaster = try gpu.rasterize(graph, width: side, height: side)
        let oldPath = try traced()
        #expect(try graph.pick(ray: left)?.nodeID == node.id)
        #expect(try graph.pick(ray: right) == nil)
        #expect(scene.statistics.instancedTriangleCount == 2)

        renderer.drawRange = try MeshDrawRange(start: 6, count: Int.max)
        let hit = try #require(try graph.pick(ray: right))
        #expect((2...3).contains(hit.intersection.triangleIndex))
        #expect(try graph.pick(ray: left) == nil)
        let collision = try #require(try graph.collisionBVH())
        #expect(try collision.intersection(ray: left) == nil)
        #expect(try collision.intersection(ray: right) != nil)
        try scene.updateGeometry(for: [node.id], in: graph)
        #expect(scene.statistics.instancedTriangleCount == 2)
        let newRaster = try gpu.rasterize(graph, width: side, height: side)
        let newPath = try traced()
        for (before, after) in [(oldRaster, newRaster), (oldPath, newPath)] {
            #expect(before[24 * side + 16].x == radiance.x)
            #expect(before[24 * side + 31].x == 0)
            #expect(after[24 * side + 16].x == 0)
            #expect(after[24 * side + 31].x == radiance.x)
        }

        let archive = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        let manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 9)
        let restored = try archive.load(device: gpu.device).graph
        #expect(restored.roots[0].mesh?.drawRange == renderer.drawRange)
        #expect(restored.roots[0].mesh?.geometry.indices == geometry.indices)
        #expect(try restored.pick(ray: right)?.intersection == hit.intersection)

        renderer.drawRange = try MeshDrawRange(start: 0, count: 0)
        #expect(throws: PathTracingScene.Failure.structureChanged(node.id)) {
            try scene.updateGeometry(for: [node.id], in: graph)
        }
        #expect(scene.statistics.instancedTriangleCount == 2)
    }

    @Test("One draw window intersects all material groups before triangle assembly")
    func groupedRange() throws {
        let gpu = try PathTestFixture()
        let side = 48
        let leftColor = SIMD3<Float>(0.75, 0.25, 0.5)
        let rightColor = SIMD3<Float>(0.25, 0.75, 0.5)
        let square = try MeshGeometry.plane(width: 0.68, height: 0.68)
        let merged = try MeshGeometry.merged([
            .init(square, transform: translationMatrix(-0.55)),
            .init(square, transform: translationMatrix(0.55)),
        ]).geometry
        let geometry = try merged.withGroups([
            .init(start: 0, count: 6, materialIndex: 0),
            .init(start: 6, count: 6, materialIndex: 1),
        ])
        let materials = try [leftColor, rightColor].map { emission in
            MeshMaterial.physicallyBased(try .init(
                metallic: 0, roughness: 1, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), emissiveFactor: emission
            ))
        }
        let transform = SceneTransform(translation: SIMD3(0, 0, -3))
        let prepared = try GroupedMesh(
            geometry, materials: materials, device: gpu.device, colorPixelFormat: .rgba16Float, transform: transform
        )
        let graph = SceneGraph(roots: [prepared.node], camera: gpu.camera)
        let both = try gpu.rasterize(graph, width: side, height: side)
        let leftPixel = 24 * side + 16
        let rightPixel = 24 * side + 31
        #expect(both[leftPixel].x == leftColor.x)
        #expect(both[rightPixel].y == rightColor.y)

        prepared.drawRange = try GeometryDrawRange(start: 5, count: 7)
        let left = try SceneRay(origin: SIMD3(-0.55, 0, 0), direction: SIMD3(0, 0, -1))
        let right = try SceneRay(origin: SIMD3(0.55, 0, 0), direction: SIMD3(0, 0, -1))
        #expect(try graph.pick(ray: left) == nil)
        #expect(try graph.pick(ray: right)?.nodeID == prepared.node.children[1].id)
        let raster = try gpu.rasterize(graph, width: side, height: side)
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: gpu.queue, scene: scene)
        let traced = try gpu.trace(tracer, width: side, height: side, samples: 1, settings: .init(maxBounces: 1, seed: 11))
        for pixels in [raster, traced] {
            #expect(pixels[leftPixel].x == 0)
            #expect(pixels[rightPixel].y == rightColor.y)
        }
        let archive = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        let restored = try archive.load(device: gpu.device).graph
        #expect(try restored.pick(ray: left) == nil)
        #expect(try restored.pick(ray: right)?.nodeID == prepared.node.children[1].id)

        let unaligned = try GeometryDrawRange(start: 5, count: 4)
        let initial = try SceneNode.grouped(
            geometry, materials: materials, device: gpu.device, colorPixelFormat: .rgba16Float,
            drawRange: unaligned, transform: transform
        )
        let initialGraph = SceneGraph(roots: [initial], camera: gpu.camera)
        let partial = try PathTracingScene(queue: gpu.queue, graph: initialGraph, options: .init(acceleration: .software))
        #expect(partial.statistics.instancedTriangleCount == 1)
        prepared.drawRange = .all
        #expect(try gpu.rasterize(graph, width: side, height: side) == both)
    }

    @Test("Directional shadow maps receive only the selected raster triangles")
    func shadowDepth() async throws {
        let fixture = try GeometryRenderFixture()
        let square = try GeometryRenderFixture.square(0.31)
        let geometry = try MeshGeometry.merged([
            .init(square, transform: translationMatrix(-0.65)),
            .init(square, transform: translationMatrix(0.65)),
        ]).geometry
        let mesh = try fixture.unlit(geometry, material: .shadowedLambert)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let shadow = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 64, directionToLight: SIMD3(0, 0, 1),
            center: SIMD3(0, 0, -3), halfExtent: 1.6
        )
        func read(_ x: Float) async throws -> Float {
            let clip = shadow.clipFromWorld * SIMD4(x, 0, -3, 1)
            let px = Int((clip.x / clip.w * 0.5 + 0.5) * 64)
            let py = Int((0.5 - clip.y / clip.w * 0.5) * 64)
            let result = try await TextureTransfer.read(
                from: shadow.texture, region: MTLRegionMake2D(px, py, 1, 1), commandQueue: fixture.queue
            )
            return result.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
        }
        mesh.drawRange = try MeshDrawRange(start: 0, count: 6)
        let left = try fixture.render(graph, shadowMap: shadow)
        #expect(left.statistics.shadowTriangleCount == 2)
        #expect(try await read(-0.65) < 1)
        #expect(try await read(0.65) == 1)
        mesh.drawRange = try MeshDrawRange(start: 6)
        let right = try fixture.render(graph, shadowMap: shadow)
        #expect(right.statistics.shadowTriangleCount == 2)
        #expect(try await read(-0.65) == 1)
        #expect(try await read(0.65) < 1)
    }

    @Test("Shifted index starts use complete triangles, and an empty range applies pass clears")
    func shiftedRangeAndEmpty() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try MeshGeometry(vertices: [
            .init(position: SIMD3(-1, -1, 0), normal: SIMD3(0, 0, 1)),
            .init(position: SIMD3(1, -1, 0), normal: SIMD3(0, 0, 1)),
            .init(position: SIMD3(0, 1, 0), normal: SIMD3(0, 0, 1)),
        ], indices: [1, 0, 1, 2, 2, 2])
        let renderer = try fixture.unlit(geometry)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        #expect(throws: MeshDrawRange.Failure.negative) { try MeshDrawRange(start: -1) }
        #expect(throws: MeshDrawRange.Failure.negative) { try MeshDrawRange(start: 0, count: -1) }
        renderer.drawRange = try MeshDrawRange(start: 1, count: 4)
        let first = try fixture.render(graph)
        #expect(first.statistics.submittedTriangleCount == 1)
        #expect(fixture.pixel(first.pixels, x: 32, y: 32).w == 1)
        #expect(try graph.pick(at: SIMD2(0.5, 0.5), aspectRatio: 1)?.intersection.triangleIndex == 0)
        renderer.drawRange = try MeshDrawRange(start: Int.max, count: Int.max)
        let empty = try fixture.render(graph)
        #expect(empty.statistics.drawnMeshCount == 0)
        #expect(empty.statistics.submittedTriangleCount == 0)
        #expect(empty.pixels.allSatisfy { $0 == .zero })
        #expect(try graph.pick(at: SIMD2(0.5, 0.5), aspectRatio: 1) == nil)
    }
}
