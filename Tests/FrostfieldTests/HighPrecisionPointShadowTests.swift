import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionPointShadowTests {
    private let farOrigin = SIMD3<Float>(0x1p60, 0, 0x1p60)
    private let lightPosition = SIMD3<Float>(0, 0, -2)
    private let nearPlane: Float = 0.2

    @Test("All point-shadow faces and the lit receiver survive a 2⁶⁰ origin",
          arguments: [ShadowMapSampling.basic, .filtered])
    func translatedCube(_ sampling: ShadowMapSampling) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let receiver = try mesh(gpu, halfExtent: 2.3)
        let caster = try blocker(gpu)
        let distances: [Float] = [1, 1.2, 1.4, 1.6, 0.8, 1.1]
        let casters = zip(PunctualShadowMaps.faceDirections, distances).map { direction, distance in
            SceneNode(
                transform: .init(
                    translation: lightPosition + direction * distance,
                    rotation: simd_quatf(from: SIMD3(0, 0, 1), to: -direction)
                ),
                mesh: caster
            )
        }
        let nodes = try [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: receiver, castsShadow: false),
            point(sampling, bias: 1),
        ] + casters

        func render(origin: SIMD3<Float>, precise: Bool) async throws
            -> (pixels: [SIMD4<Float>], slices: [Data], stats: SceneDrawStatistics) {
            let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 0, pointCapacity: 1)
            let graph = scene(gpu, nodes: nodes, origin: origin, precise: precise)
            let frame = try gpu.render(graph, punctualShadowMaps: maps)
            var slices: [Data] = []
            for index in 0..<6 { slices.append(try await depth(gpu, maps, slice: index)) }
            return (frame.pixels, slices, frame.statistics)
        }

        let near = try await render(origin: .zero, precise: true)
        let moved = try await render(origin: farOrigin, precise: true)
        let ordinary = try await render(origin: .zero, precise: false)
        #expect(moved.pixels == near.pixels && moved.slices == near.slices)
        #expect(moved.stats.shadowPassCount == 6)
        #expect(moved.stats.shadowDrawCount == 6 && moved.stats.culledShadowCount == 30)
        for (image, distance) in zip(moved.slices, distances) {
            let center = image.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: (32 * 64 + 32) * 4, as: Float.self) }
            #expect(abs(center - nearPlane / distance) < 0.00001)
        }
        let mid = gpu.side / 2
        let shadow = gpu.pixel(moved.pixels, x: mid, y: mid)
        let lit = gpu.pixel(moved.pixels, x: mid + 10, y: mid)
        #expect(shadow.y < 0.01 && lit.y > 0.2)
        #expect(isNearly(lit, gpu.pixel(ordinary.pixels, x: mid + 10, y: mid), tolerance: 0.003))
    }

    @Test("Point cube ties, biased face selection, and cache placement use the light-relative vector")
    func faceSelectionAndCache() async throws {
        let gpu = try GeometryRenderFixture(side: 64)
        let receiver = try mesh(gpu, halfExtent: 2)
        let target = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2.9375)), mesh: receiver, castsShadow: false
        )
        let source = try point(.basic, bias: 0)
        let maps = try PunctualShadowMaps(
            queue: gpu.queue, resolution: 64, spotCapacity: 0, pointCapacity: 1, updateMode: .onDemand
        )
        var graph = scene(gpu, nodes: [target, source], origin: farOrigin, precise: true)

        // Distinguish the overlapping faces at the seam. +X is occluded and
        // -Z is lit so the GPU receiver makes its face selection observable.
        func seedFaces() throws {
            let command = try #require(gpu.queue.makeCommandBuffer())
            for face in 0..<6 {
                try maps.makeEncoder(slice: face, commandBuffer: command) {
                    $0.depthAttachment.clearDepth = face == 0 ? 1 : 0
                }.endEncoding()
            }
            try gpu.finish(command)
        }
        func draw() throws -> (pixels: [SIMD4<Float>], stats: SceneDrawStatistics) {
            try orthographic(gpu, graph: graph, maps: maps)
        }

        #expect(try draw().stats.shadowPassCount == 6)
        try seedFaces()
        let initial = try draw()
        #expect(initial.stats.shadowPassCount == 0)
        // X at columns 38 and 39 is 0.8125 and 0.9375, respectively.
        // Both positions are 0.9375 from the light on -Z. X wins the tie.
        let beforeSeam = gpu.pixel(initial.pixels, x: 38, y: 32)
        let seam = gpu.pixel(initial.pixels, x: 39, y: 32)
        #expect(beforeSeam.y > 0.2 && seam.y < 0.01, "Before \\(beforeSeam), tie \\(seam)")
        let seeded = try await depth(gpu, maps, slice: 0)
        graph.cameraTransform.translation.y = 0.25
        #expect(try draw().stats.shadowPassCount == 0)
        #expect(try await depth(gpu, maps, slice: 0) == seeded)

        graph.cameraTransform.translation.y = 0
        graph.roots[0].children[1].light = try point(.basic, bias: 12).light
        #expect(try draw().stats.shadowPassCount == 6)
        try seedFaces()
        let biased = try draw()
        #expect(biased.stats.shadowPassCount == 0)
        #expect(gpu.pixel(biased.pixels, x: 38, y: 32).y < 0.01)

        graph.roots[0].children[1].transform.translation.x += 0.25
        #expect(try draw().stats.shadowPassCount == 6)
        #expect(try await depth(gpu, maps, slice: 0).allSatisfy { $0 == 0 })
        #expect(try draw().stats.shadowPassCount == 0)
    }

    @Test("A normal-scale pass can mix relative and batched world-space point receivers")
    func mixedReceivers() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let ordinary = try mesh(gpu, halfExtent: 0.5)
        let batch = try BatchedMesh(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometries: [GeometryRenderFixture.square(0.5)],
            capacity: 1, material: .physicallyBased(receiverMaterial())
        )
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-1.25, 0, -5))
        let nodes = try [
            SceneNode(transform: .init(translation: SIMD3(1.25, 0, -5)), mesh: ordinary, castsShadow: false),
            SceneNode(castsShadow: false, batch: batch),
            SceneNode(transform: .init(translation: SIMD3(-0.42, 0, -3)), mesh: blocker(gpu)),
            point(.basic, bias: 0),
        ]
        func render(precise: Bool) throws -> [SIMD4<Float>] {
            let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 0, pointCapacity: 1)
            return try gpu.render(scene(gpu, nodes: nodes, origin: .zero, precise: precise), punctualShadowMaps: maps).pixels
        }
        let a = try render(precise: false)
        let b = try render(precise: true)
        let x = gpu.column(x: 1.25, depth: 5)
        let hiddenX = gpu.column(x: -1.25, depth: 5)
        #expect(gpu.pixel(b, x: x, y: 32).y > 0.2)
        let shadow = gpu.pixel(b, x: hiddenX, y: 32)
        #expect(shadow.w == 1 && shadow.y < 0.01)
        #expect(zip(a, b).allSatisfy { isNearly($0, $1, tolerance: 0.003) })
    }

    private func receiverMaterial() throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque)
    }

    private func mesh(_ gpu: GeometryRenderFixture, halfExtent: Float) throws -> MeshRenderer {
        try gpu.unlit(GeometryRenderFixture.square(halfExtent), material: .physicallyBased(receiverMaterial()))
    }

    private func blocker(_ gpu: GeometryRenderFixture) throws -> MeshRenderer {
        try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.3),
            renderState: .init(depthWrite: false, colorWriteMask: [], side: .double)
        )
    }

    private func point(_ sampling: ShadowMapSampling, bias: Float) throws -> SceneNode {
        let settings = try PunctualLight.Shadow(normalBias: bias, filterRadius: 1, nearPlane: nearPlane, sampling: sampling)
        return SceneNode(transform: .init(translation: lightPosition), light: try PunctualLight(kind: .point(), intensity: 36, shadow: settings))
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

    private func orthographic(
        _ gpu: GeometryRenderFixture, graph: SceneGraph, maps: PunctualShadowMaps
    ) throws -> (pixels: [SIMD4<Float>], stats: SceneDrawStatistics) {
        let projection = simd_float4x4(columns: (
            SIMD4(0.25, 0, 0, 0), SIMD4(0, 0.25, 0, 0), SIMD4(0, 0, -0.1, 0), SIMD4(0, 0, 0, 1)
        ))
        let view = simd_inverse(graph.cameraTransform.matrix)
        let camera = try ShaderGraphCamera(viewMatrix: view, projectionMatrix: projection, near: 0, far: 10)
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let stats = try graph.encode(
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
            clipFromWorld: projection * view, cameraPosition: graph.cameraTransform.translation,
            directionToCamera: SIMD3(0, 0, 1), punctualShadowMaps: maps, graphCamera: camera
        )
        try gpu.finish(command)
        return (try gpu.pixels(of: target.color), stats)
    }

    private func depth(_ gpu: GeometryRenderFixture, _ maps: PunctualShadowMaps, slice: Int) async throws -> Data {
        try await TextureTransfer.read(from: maps.texture, slice: slice, commandQueue: gpu.queue).bytes
    }
}
