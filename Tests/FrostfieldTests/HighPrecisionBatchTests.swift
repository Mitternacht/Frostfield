import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionBatchTests {
    private let farOrigin = SIMD3<Float>(0x1p60, 0, 0x1p60)

    @Test("Instance groups retain placement, tint, culling and reflected winding at a shared 2⁶⁰ origin",
          arguments: [false, true])
    func instances(mirrored: Bool) throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let mesh = try gpu.unlit(GeometryRenderFixture.square(0.3), supportsInstancing: true)
        let placements = [-0.8 as Float, 0.8].map {
            SceneTransform(translation: SIMD3($0, 0, -3), scale: SIMD3(mirrored ? -1 : 1, 1, 1)).matrix
        }
        let tints: [SIMD4<Float>] = [SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1)]
        let set = try InstanceSet(device: gpu.device, transforms: placements, tints: tints)
        let offscreen = try InstanceSet(device: gpu.device, transforms: [translationMatrix(24, 0, -3)])
        let nodes = [SceneNode(mesh: mesh, instances: set), SceneNode(mesh: mesh, instances: offscreen)]

        let near = try gpu.render(scene(gpu, nodes: nodes, origin: .zero, precise: false))
        let moved = try gpu.render(scene(gpu, nodes: nodes, origin: farOrigin, precise: true))
        #expect(moved.pixels == near.pixels)
        #expect(moved.statistics.drawnInstanceCount == 2 && moved.statistics.culledMeshCount == 1)
        for (x, tint) in zip([-0.8 as Float, 0.8], tints) {
            let pixel = gpu.pixel(moved.pixels, x: gpu.column(x: x, depth: 3), y: 32)
            #expect(isNearly(pixel, GeometryRenderFixture.unlitFacingCamera(tint)))
        }
    }

    @Test("Direct batches retain geometry, tint, culling, and transparent depth order at a shared 2⁶⁰ origin")
    func directBatch() throws {
        try checkBatch(.direct)
    }

    @Test("Indirect batches inherit camera-relative clip and preserve winding spans and bundle reuse",
          .enabled(if: MTLCreateSystemDefaultDevice().map { BatchedMesh.supportsIndirectSubmission(on: $0) } ?? false,
                   "Requires Metal indirect render commands"))
    func indirectBatch() throws {
        try checkBatch(.indirectCommandBuffer)
    }

    private func checkBatch(_ submission: BatchedMesh.Submission) throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let batch = try BatchedMesh(
            device: gpu.device, colorPixelFormat: .rgba16Float,
            geometries: [GeometryRenderFixture.square(0.45), GeometryRenderFixture.square(0.65)],
            capacity: 5, renderState: .transparent, submission: submission
        )
        let red = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -1), tint: SIMD4(1, 0, 0, 0.5))
        let blue = try batch.addInstance(geometryIndex: 1, transform: translationMatrix(0, 0, -3), tint: SIMD4(0, 0, 1, 0.5))
        let mirror = SceneTransform(translation: SIMD3(0, 0, -2), scale: SIMD3(-1, 1, 1))
        let green = try batch.addInstance(geometryIndex: 0, transform: mirror.matrix, tint: SIMD4(0, 1, 0, 0.5))
        try batch.addInstance(geometryIndex: 1, transform: translationMatrix(20, 0, -2))
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0, 0, -0.5), isVisible: false)
        let nodes = [SceneNode(batch: batch)]
        let near = try gpu.render(scene(gpu, nodes: nodes, origin: .zero, precise: false))
        let graph = scene(gpu, nodes: nodes, origin: farOrigin, precise: true)
        let moved = try gpu.render(graph)
        #expect(moved.pixels == near.pixels)
        #expect(moved.statistics.drawnInstanceCount == 3 && moved.statistics.culledMeshCount == 1)
        #expect(moved.statistics.drawnMeshCount == 3 && moved.statistics.submittedTriangleCount == 6)
        #expect(moved.statistics.indirectBatchExecutionCount == (submission == .direct ? 0 : 3))
        #expect(batch.indirectBundleBuildCount == (submission == .direct ? 0 : 1))
        #expect(isNearly(gpu.pixel(moved.pixels, x: 32, y: 32), SIMD4(0.11, 0.055, 0.125, 0.875)))

        let origin = farOrigin

        batch.customSort = { items, _ in
            #expect(items.map(\.instance.id) == [red, blue, green])
            #expect(items.allSatisfy { $0.worldTransform.columns.3.z == origin.z })
            return [blue, green, red]
        }
        #expect(try gpu.render(graph).pixels == moved.pixels)
        #expect(batch.indirectBundleBuildCount == (submission == .direct ? 0 : 1))
    }

    @Test("Batched surface attributes and explicit culling bounds use camera-relative transforms")
    func batchAttributes() async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let batch = try BatchedMesh(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometries: [GeometryRenderFixture.square(0.4)],
            capacity: 1
        )
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-0.6, 0, -2))
        let override = try AxisAlignedBox(minimum: SIMD3(20, -1, -3), maximum: SIMD3(21, 1, -1))
        let nodes = [
            SceneNode(batch: batch),
            SceneNode(cullingBounds: override, batch: batch),
        ]
        func render(_ origin: SIMD3<Float>, precise: Bool) async throws -> (Data, SceneDrawStatistics) {
            let graph = scene(gpu, nodes: nodes, origin: origin, precise: precise)
            let target = try gpu.makeTarget()
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: gpu.side, height: gpu.side, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .renderTarget
            let color = try #require(gpu.device.makeTexture(descriptor: descriptor))
            let command = try #require(gpu.queue.makeCommandBuffer())
            let statistics = try graph.encodeSurfaceAttributes(
                pass: gpu.pass(color: color, depth: target.depth), commandBuffer: command
            )
            try gpu.finish(command)
            let pixels = try await TextureTransfer.read(from: color, commandQueue: gpu.queue)
            return (pixels.bytes, statistics)
        }

        let (near, _) = try await render(.zero, precise: false)
        let (moved, statistics) = try await render(farOrigin, precise: true)
        #expect(moved == near)
        #expect(statistics.drawnInstanceCount == 1 && statistics.culledMeshCount == 1)
        let hit = (32 * gpu.side + gpu.column(x: -0.6, depth: 2)) * 8
        let miss = (32 * gpu.side + gpu.column(x: 0.6, depth: 2)) * 8
        moved.withUnsafeBytes {
            #expect($0.loadUnaligned(fromByteOffset: hit, as: UInt64.self) != 0)
            #expect($0.loadUnaligned(fromByteOffset: miss, as: UInt64.self) == 0)
        }
    }

    @Test("Unskinned instances use light-relative caster depth and receivers for directional and point shadows",
          arguments: [false, true])
    func instanceShadows(punctual: Bool) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let receiverMaterial: MeshMaterial = if punctual {
            .physicallyBased(try PhysicallyBasedMaterial(roughness: 0.8, lightIntensity: 0, ambientIntensity: 0))
        } else {
            .shadowedLambert
        }
        let receiver = try gpu.unlit(
            GeometryRenderFixture.square(2.3), supportsInstancing: true, material: receiverMaterial
        )
        let caster = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.3),
            supportsInstancing: true, renderState: .init(depthWrite: false, colorWriteMask: [], side: .double)
        )
        let receiverSet = try InstanceSet(device: gpu.device, transforms: [translationMatrix(0, 0, -5)])
        let casterSet = try InstanceSet(device: gpu.device, transforms: [translationMatrix(0.125, 0, -3)])
        var nodes = [
            SceneNode(mesh: receiver, castsShadow: false, instances: receiverSet),
            SceneNode(mesh: caster, instances: casterSet),
        ]
        if punctual {
            let shadow = try PunctualLight.Shadow(normalBias: 0.5, nearPlane: 0.2, sampling: .basic)
            nodes.append(SceneNode(
                transform: .init(translation: SIMD3(0, 0, -2)),
                light: try PunctualLight(kind: .point(), intensity: 36, shadow: shadow)
            ))
        }

        func render(_ origin: SIMD3<Float>, precise: Bool) async throws
            -> (pixels: [SIMD4<Float>], depth: Data, statistics: SceneDrawStatistics) {
            let graph = scene(gpu, nodes: nodes, origin: origin, precise: precise)
            if punctual {
                let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 0, pointCapacity: 1)
                let frame = try gpu.render(graph, punctualShadowMaps: maps)
                let depth = try await TextureTransfer.read(from: maps.texture, slice: 5, commandQueue: gpu.queue)
                return (frame.pixels, depth.bytes, frame.statistics)
            }
            let map = try DirectionalShadowMap(
                queue: gpu.queue, resolution: 64, directionToLight: SIMD3(0, 0, 1),
                center: origin, halfExtent: 8, sampling: .basic
            )
            let frame = try gpu.render(graph, shadowMap: map)
            let depth = try await TextureTransfer.read(from: map.texture, commandQueue: gpu.queue)
            return (frame.pixels, depth.bytes, frame.statistics)
        }

        let ordinary = try await render(.zero, precise: false)
        let near = try await render(.zero, precise: true)
        let moved = try await render(farOrigin, precise: true)
        #expect(moved.depth == near.depth && moved.pixels == near.pixels)
        #expect(moved.statistics.shadowDrawCount == 1 && moved.statistics.drawnInstanceCount == 2)
        let middle = moved.depth.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: (32 * 64 + 32) * 4, as: Float.self) }
        #expect(middle > 0 && middle < 1)
        let shadow = gpu.pixel(moved.pixels, x: 32, y: 32)
        let lit = gpu.pixel(moved.pixels, x: 42, y: 32)
        let ambient: Float = punctual ? 0 : 0.16
        #expect(shadow.w == 1 && abs(shadow.y - ambient) < 0.01, "\(shadow)")
        #expect(lit.w == 1 && lit.y > ambient + 0.2, "\(lit)")
        #expect(isNearly(lit, gpu.pixel(ordinary.pixels, x: 42, y: 32), tolerance: 0.003))
    }

    private func scene(
        _ gpu: GeometryRenderFixture, nodes: [SceneNode], origin: SIMD3<Float>, precise: Bool
    ) -> SceneGraph {
        let anchor = SceneTransform(translation: origin)
        var graph = SceneGraph(
            roots: [SceneNode(transform: anchor, children: nodes)], camera: gpu.camera, cameraTransform: anchor
        )
        graph.highPrecision = precise
        return graph
    }
}
