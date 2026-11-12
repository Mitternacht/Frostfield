import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct DynamicGeometryTests {
    private let gpu: RasterLightingFixture

    init() throws {
        gpu = try RasterLightingFixture(width: 64, height: 64)
    }

    @Test("An in-flight draw keeps its old vertex and index data as the next draw grows")
    func inFlightRevisions() throws {
        let red = SIMD4<Float>(1, 0, 0, 1)
        let green = SIMD4<Float>(0, 1, 0, 1)
        let left = try patch(x: -0.7, quad: false, color: red)
        let right = try patch(x: 0.7, quad: true, color: green)
        let renderer = try dynamic(left, vertexColors: true)
        let graph = try scene(renderer)
        let initial = graph.resourceStatistics()
        let screenBinding = try #require(renderer.screenInputsFallback?.arguments)
        #expect(initial.preparedGeometryCount == 1)
        #expect(initial.bufferCount == 4)
        #expect(renderer.geometryRevision == 0)

        let oldColor = try gpu.colorTarget(), oldDepth = try gpu.depthTarget()
        let newColor = try gpu.colorTarget(), newDepth = try gpu.depthTarget()
        let gate = try #require(gpu.device.makeSharedEvent())
        defer { gate.signaledValue = 1 }
        let old = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        old.encodeWaitForEvent(gate, value: 1)
        let oldStats = try graph.encode(pass: gpu.pass(color: oldColor, depth: oldDepth), commandBuffer: old)
        old.commit()

        try renderer.updateGeometry(right)
        #expect(renderer.geometryRevision == 1)
        #expect(renderer.bounds == right.bounds)
        let new = try #require(gpu.queue.makeCommandBuffer())
        let newStats = try graph.encode(pass: gpu.pass(color: newColor, depth: newDepth), commandBuffer: new)
        new.commit()
        let pending = graph.resourceStatistics()
        #expect(pending.preparedGeometryCount == 1)
        #expect(pending.renderPipelineCount == initial.renderPipelineCount)
        // Only the vertex, index and color buffers have per-revision copies.
        #expect(pending.bufferCount == initial.bufferCount * 2 - 1)
        #expect(pending.bufferLengthBytes == initial.bufferLengthBytes * 2 - screenBinding.length)
        if let allocated = initial.bufferAllocatedBytes {
            #expect(pending.bufferAllocatedBytes == allocated * 2 - screenBinding.allocatedSize)
        } else {
            #expect(pending.bufferAllocatedBytes == nil)
        }
        #expect(oldStats.submittedTriangleCount == 1)
        #expect(newStats.submittedTriangleCount == 2)

        gate.signaledValue = 1
        for command in [old, new] {
            command.waitUntilCompleted()
            try #require(command.status == .completed)
        }
        let original = try gpu.pixels(of: oldColor)
        let replacement = try gpu.pixels(of: newColor)
        #expect(pixel(original, 18, 32) == red)
        #expect(pixel(original, 46, 32) == .zero)
        #expect(pixel(replacement, 18, 32) == .zero)
        #expect(pixel(replacement, 43, 35) == green)
        #expect(pixel(replacement, 48, 28) == green)
    }

    @Test("The bounded pool shares repeated revisions and releases an abandoned draw")
    func ringPressure() throws {
        let renderer = try dynamic(patch(x: 0, quad: false))
        let graph = try scene(renderer)
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let pass = gpu.pass(color: color, depth: depth)
        var pending: [any MTLCommandBuffer] = []
        try autoreleasepool {
            for revision in 0..<3 {
                if revision > 0 { try renderer.updateGeometry(patch(x: Float(revision) * 0.1, quad: false)) }
                let command = try #require(gpu.queue.makeCommandBuffer())
                try graph.encode(pass: pass, commandBuffer: command)
                pending.append(command)
            }
            // Another draw of revision 2 shares the third slot.
            let repeated = try #require(gpu.queue.makeCommandBuffer())
            try graph.encode(pass: pass, commandBuffer: repeated)
            pending.append(repeated)
        }
        #expect(graph.resourceStatistics().bufferCount == 6)
        try renderer.updateGeometry(patch(x: 0.3, quad: true))
        let retry = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: MeshRenderer.Failure.dynamicGeometryBuffersBusy) {
            try graph.encode(pass: pass, commandBuffer: retry)
        }
        pending.removeFirst()
        let result = try graph.encode(pass: pass, commandBuffer: retry)
        retry.commit()
        retry.waitUntilCompleted()
        #expect(retry.status == .completed)
        #expect(result.submittedTriangleCount == 2)
        #expect(graph.resourceStatistics().bufferCount == 6)
        withExtendedLifetime(pending) {}
    }

    @Test("A color-reading override adds only the needed channel to live revisions and retains pending draws")
    func materialOverrideRevisions() throws {
        let red = SIMD4<Float>(1, 0, 0, 1), green = SIMD4<Float>(0, 1, 0, 1)
        let renderer = try dynamic(patch(x: -0.7, quad: false, color: red))
        var graph = try scene(renderer)
        #expect(graph.resourceStatistics().bufferCount == 2)
        graph.overrideMaterial = try SceneMaterialOverride(material: .classic(ClassicMaterial(model: .basic)))
        let oldColor = try gpu.colorTarget(), oldDepth = try gpu.depthTarget()
        let newColor = try gpu.colorTarget(), newDepth = try gpu.depthTarget()
        let gate = try #require(gpu.device.makeSharedEvent())
        defer { gate.signaledValue = 1 }
        let old = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        old.encodeWaitForEvent(gate, value: 1)
        try graph.encode(pass: gpu.pass(color: oldColor, depth: oldDepth), commandBuffer: old)
        old.commit()
        #expect(graph.resourceStatistics().bufferCount == 3)
        try renderer.updateGeometry(patch(x: 0.7, quad: true, color: green))
        let next = try #require(gpu.queue.makeCommandBuffer())
        try graph.encode(pass: gpu.pass(color: newColor, depth: newDepth), commandBuffer: next)
        next.commit()
        graph.overrideMaterial = nil
        gate.signaledValue = 1
        for command in [old, next] {
            command.waitUntilCompleted()
            try #require(command.status == .completed)
        }
        #expect(try pixel(gpu.pixels(of: oldColor), 18, 32) == red)
        #expect(try pixel(gpu.pixels(of: newColor), 46, 32) == green)
        let authored = try pixel(gpu.pixels(of: gpu.render(graph).color), 46, 32)
        #expect(abs(authored.x - 0.22) < 0.0001 && authored.x == authored.y && authored.z == 1 && authored.w == 1)
    }

    @Test("Updates replace CPU picking and culling bounds, and invalid capacity is transactional")
    func boundsAndValidation() throws {
        let left = try patch(x: -0.7, quad: false)
        let right = try patch(x: 0.7, quad: true)
        let renderer = try dynamic(left, capacity: .init(vertices: 3, triangles: 1))
        let graph = try scene(renderer)
        let near = try SceneRay(origin: SIMD3(-0.7, 0, 0), direction: SIMD3(0, 0, -1))
        let far = try SceneRay(origin: SIMD3(100, 0, 0), direction: SIMD3(0, 0, -1))
        #expect(try graph.pick(ray: near) != nil)
        #expect(throws: MeshRenderer.Failure.dynamicGeometryCapacity) { try renderer.updateGeometry(right) }
        #expect(renderer.bounds == left.bounds && renderer.geometryRevision == 0)
        #expect(try graph.pick(ray: near) != nil)

        let offscreen = try patch(x: 100, quad: false)
        try renderer.updateGeometry(offscreen)
        #expect(renderer.bounds == offscreen.bounds && renderer.geometryRevision == 1)
        #expect(try graph.pick(ray: near) == nil)
        #expect(try graph.pick(ray: far) != nil)
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        try gpu.run { command in
            let stats = try graph.encode(pass: gpu.pass(color: color, depth: depth), commandBuffer: command)
            #expect(stats.drawnMeshCount == 0 && stats.culledMeshCount == 1)
        }
        try renderer.updateGeometry(left)
        try gpu.run { command in
            let stats = try graph.encode(pass: gpu.pass(color: color, depth: depth), commandBuffer: command)
            #expect(stats.drawnMeshCount == 1 && stats.culledMeshCount == 0)
        }
        let arguments = try #require(gpu.device.makeBuffer(length: MeshIndirectDraws.recordByteCount, options: .storageModeShared))
        let indirect = try MeshIndirectDraws(buffer: arguments)
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: MeshIndirectDraws.Failure.incompatibleNode(graph.roots[0].id)) {
            try graph.encode(
                pass: gpu.pass(color: color, depth: depth), commandBuffer: command,
                indirectDraws: [graph.roots[0].id: indirect]
            )
        }
        let staticMesh = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: left)
        #expect(staticMesh.geometryRevision == nil)
        #expect(throws: MeshRenderer.Failure.dynamicGeometryNotPrepared) { try staticMesh.updateGeometry(left) }
        #expect(throws: MeshRenderer.Failure.dynamicGeometryCapacity) {
            try MeshRenderer.DynamicGeometryCapacity(vertices: 1, triangles: Int.max)
        }
    }

    @Test("Selection and directional shadow depth use the updated positions and triangle count")
    func depthPasses() throws {
        let renderer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, dynamicGeometry: patch(x: -0.7, quad: false),
            capacity: .init(vertices: 4, triangles: 2), material: .shadowedLambert
        )
        let graph = try scene(renderer)
        let viewer = try TextureDebugPass(device: gpu.device)
        let selection = try SelectionOutline(device: gpu.device)
        let shadow = try DirectionalShadowMap(
            queue: gpu.queue, resolution: gpu.width, directionToLight: SIMD3(0, 0, 1),
            center: SIMD3(0, 0, -3), halfExtent: 1.6
        )
        let old = try depthImages(graph, selection: selection, shadow: shadow, viewer: viewer, triangles: 1)
        try renderer.updateGeometry(patch(x: 0.7, quad: true))
        let new = try depthImages(graph, selection: selection, shadow: shadow, viewer: viewer, triangles: 2)
        for (before, after) in zip(old, new) {
            let previous = depthCentroid(before)
            let current = depthCentroid(after)
            #expect(previous.count > 0 && current.count > previous.count)
            #expect(abs(current.x - previous.x) > 20)
            #expect((previous.x < 32) != (current.x < 32))
        }
    }

    @Test("A rebuilt metaball field reuses one prepared renderer as its visible triangle range changes")
    func marchingCubes() throws {
        let volume = try AxisAlignedBox(minimum: SIMD3(repeating: -1), maximum: SIMD3(repeating: 1))
        var field = try MarchingCubes(resolution: SIMD3(repeating: 16), bounds: volume)
        try field.addMetaball(center: SIMD3(-0.45, 0, 0), strength: 0.08, subtract: 0.1, color: SIMD3(1, 0, 0))
        let left = try #require(try field.mesh(isovalue: 1, boundary: .requireClosed))
        field.reset()
        try field.addMetaball(center: SIMD3(0.45, 0, 0), strength: 0.18, subtract: 0.1, color: SIMD3(0, 1, 0))
        let right = try #require(try field.mesh(isovalue: 1, boundary: .requireClosed))
        let capacity = try MeshRenderer.DynamicGeometryCapacity(
            vertices: max(left.vertices.count, right.vertices.count),
            triangles: max(left.indices.count, right.indices.count) / 3
        )
        let renderer = try dynamic(left, vertexColors: true, capacity: capacity)
        let graph = try scene(renderer)
        let old = try gpu.pixels(of: gpu.render(graph).color)
        try renderer.updateGeometry(right)
        let new = try gpu.pixels(of: gpu.render(graph).color)
        #expect(left.indices.count < right.indices.count)
        #expect(pixel(old, 23, 32).x > 0 && pixel(old, 41, 32) == .zero)
        #expect(pixel(new, 41, 32).y > 0 && pixel(new, 23, 32) == .zero)
        #expect(graph.resourceStatistics().preparedGeometryCount == 1)
        #expect(graph.resourceStatistics().bufferCount == 4)
    }

    @Test("A vertex span upload moves that corner and leaves the other vertices in the buffer")
    func vertexSpan() throws {
        let mesh = try patch(x: 0, quad: true)
        let renderer = try dynamic(mesh)
        let graph = try scene(renderer)
        let before = try gpu.pixels(of: gpu.render(graph).color)
        let covered = before.filter { $0.w > 0.5 }.count
        #expect(covered > 20)
        #expect(!renderer.lastVertexUploadWasPartial)
        var corner = mesh.vertices[1]
        corner = MeshVertex(position: SIMD3(8, corner.position.y, 0), normal: corner.normal, color: corner.color)
        let kept = mesh.vertices[3].position
        try renderer.updateVertices([corner], startingAt: 1)
        #expect(renderer.geometry.vertices[3].position == kept)
        #expect(renderer.geometry.vertices[1].position.x == 8)
        let after = try gpu.pixels(of: gpu.render(graph).color)
        let remaining = after.filter { $0.w > 0.5 }.count
        #expect(remaining > covered)
        #expect(renderer.lastVertexUploadWasPartial)
        #expect(throws: MeshRenderer.Failure.dynamicGeometryRange) {
            try renderer.updateVertices([corner], startingAt: 4)
        }
        let frozen = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: mesh, material: .unlit
        )
        #expect(throws: MeshRenderer.Failure.dynamicGeometryNotPrepared) {
            try frozen.updateVertices([corner], startingAt: 0)
        }
    }

    private func patch(x: Float, quad: Bool, color: SIMD4<Float>? = nil) throws -> MeshGeometry {
        let offsets: [SIMD2<Float>] = quad
            ? [SIMD2(-0.38, -0.52), SIMD2(0.38, -0.52), SIMD2(0.38, 0.52), SIMD2(-0.38, 0.52)]
            : [SIMD2(-0.38, -0.52), SIMD2(0.38, -0.52), SIMD2(0, 0.52)]
        return try MeshGeometry(vertices: offsets.map {
            MeshVertex(position: SIMD3(x + $0.x, $0.y, 0), normal: SIMD3(0, 0, 1), color: color)
        }, indices: quad ? [0, 1, 2, 0, 2, 3] : [0, 1, 2])
    }

    private func dynamic(
        _ mesh: MeshGeometry, vertexColors: Bool = false, capacity: MeshRenderer.DynamicGeometryCapacity? = nil
    ) throws -> MeshRenderer {
        let material: MeshMaterial = vertexColors
            ? .physicallyBased(try PhysicallyBasedMaterial(shadingModel: .unlit)) : .unlit
        return try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, dynamicGeometry: mesh,
            capacity: capacity ?? .init(vertices: 4, triangles: 2), material: material
        )
    }

    private func scene(_ mesh: MeshRenderer) throws -> SceneGraph {
        SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)],
            camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
    }

    private func depthImages(
        _ graph: SceneGraph, selection: SelectionOutline, shadow: DirectionalShadowMap,
        viewer: TextureDebugPass, triangles: Int
    ) throws -> [[SIMD4<Float>]] {
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let shadowImage = try gpu.colorTarget(), selectionImage = try gpu.colorTarget()
        try gpu.run { command in
            let stats = try graph.encode(pass: gpu.pass(color: color, depth: depth), commandBuffer: command, shadowMap: shadow)
            #expect(stats.shadowTriangleCount == triangles)
            try viewer.encode(source: .directionalShadow(shadow), destination: shadowImage, commandBuffer: command)
            let count = try selection.encodeSelection(
                graph: graph, selection: [graph.roots[0].id], width: gpu.width, height: gpu.height, commandBuffer: command
            )
            #expect(count == 1)
            try viewer.encode(source: .depth(try #require(selection.selectionMask)), destination: selectionImage, commandBuffer: command)
        }
        return [try gpu.pixels(of: shadowImage), try gpu.pixels(of: selectionImage)]
    }

    private func depthCentroid(_ pixels: [SIMD4<Float>]) -> (count: Int, x: Float) {
        var sum = 0, count = 0
        for (index, pixel) in pixels.enumerated() where pixel.x > 0 {
            sum += index % gpu.width
            count += 1
        }
        return (count, Float(sum) / Float(max(1, count)))
    }

    private func pixel(_ pixels: [SIMD4<Float>], _ x: Int, _ y: Int) -> SIMD4<Float> {
        pixels[y * gpu.width + x]
    }
}
