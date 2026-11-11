import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionRectangleShadowTests {
    private let farOrigin = SIMD3<Float>(0x1p60, 0, 0x1p60)
    private let lightPosition = SIMD3<Float>(0.55, 0.3, -2)
    private let receiverPosition = SIMD3<Float>(0, 0, -5)
    private let nearPlane: Float = 0.2

    @Test("A rolled asymmetric panel retains its irradiance, specular shape, and facing at 2⁶⁰")
    func orientedPanel() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let target = try receiver(gpu)
        func render(origin: SIMD3<Float>, roll: Float, back: Bool = false, precise: Bool = true) throws -> [SIMD4<Float>] {
            let source = try panel(roll: roll, back: back)
            return try gpu.render(scene(gpu, nodes: [target, source], origin: origin, precise: precise)).pixels
        }
        let baseline = try render(origin: .zero, roll: 0.35)
        let relocated = try render(origin: farOrigin, roll: 0.35)
        let rolled = try render(origin: farOrigin, roll: 1.4)
        let ordinary = try render(origin: .zero, roll: 0.35, precise: false)
        let backward = try render(origin: farOrigin, roll: 0.35, back: true)
        #expect(relocated == baseline)
        #expect(zip(relocated, rolled).contains { simd_reduce_max(abs($0 - $1)) > 0.025 })
        #expect(zip(ordinary, baseline).allSatisfy { isNearly($0, $1, tolerance: 0.003) })
        let center = gpu.side / 2
        #expect(gpu.pixel(relocated, x: center, y: center).y > 0.1)
        #expect(gpu.pixel(backward, x: center, y: center) == SIMD4(0, 0, 0, 1))
    }

    @Test("The rectangle's six-face center proxy preserves depth and receiver visibility",
          arguments: [ShadowMapSampling.basic, .filtered])
    func relocatedProxy(_ sampling: ShadowMapSampling) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let target = try receiver(gpu)
        let mesh = try blocker(gpu)
        let distances: [Float] = [1.1, 1.25, 1.4, 1.5, 0.8, 1.0]
        let casters = zip(PunctualShadowMaps.faceDirections, distances).map { direction, distance in
            SceneNode(
                transform: .init(
                    translation: lightPosition + direction * distance,
                    rotation: simd_quatf(from: SIMD3(0, 0, 1), to: -direction)
                ),
                mesh: mesh
            )
        }
        let source = try panel(roll: 0.35, sampling: sampling, bias: 1)
        let nodes = [target, source] + casters

        func render(origin: SIMD3<Float>, precise: Bool) async throws
            -> (pixels: [SIMD4<Float>], slices: [Data], stats: SceneDrawStatistics) {
            let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 0, pointCapacity: 1)
            let frame = try gpu.render(scene(gpu, nodes: nodes, origin: origin, precise: precise), punctualShadowMaps: maps)
            var slices: [Data] = []
            for slice in 0..<6 { slices.append(try await depth(gpu, maps, slice: slice)) }
            return (frame.pixels, slices, frame.statistics)
        }

        let near = try await render(origin: .zero, precise: true)
        let far = try await render(origin: farOrigin, precise: true)
        let ordinary = try await render(origin: .zero, precise: false)
        #expect(far.pixels == near.pixels && far.slices == near.slices)
        #expect(far.stats.shadowPassCount == 6 && far.stats.shadowDrawCount == 6)
        for (image, distance) in zip(far.slices, distances) {
            let sample = image.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: (32 * 64 + 32) * 4, as: Float.self) }
            #expect(abs(sample - nearPlane / distance) < 0.00001)
        }
        let center = gpu.side / 2
        let shadow = gpu.pixel(far.pixels, x: center, y: center)
        let lit = gpu.pixel(far.pixels, x: center - 10, y: center)
        #expect(shadow.y < 0.01 && lit.y > 0.08, "Shadow \\(shadow), lit \\(lit)")
        #expect(isNearly(lit, gpu.pixel(ordinary.pixels, x: center - 10, y: center), tolerance: 0.003))
    }

    @Test("Rectangle proxy normal bias and cached depth preserve emitter placement")
    func biasAndCache() async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let target = try receiver(gpu)
        let shallow = SceneNode(
            transform: .init(translation: receiverPosition + (lightPosition - receiverPosition) / 60),
            mesh: try blocker(gpu)
        )
        let source = try panel(roll: 0.35, sampling: .basic, bias: 0)
        var graph = scene(gpu, nodes: [target, shallow, source], origin: farOrigin, precise: true)
        let maps = try PunctualShadowMaps(
            queue: gpu.queue, resolution: 64, spotCapacity: 0, pointCapacity: 1, updateMode: .onDemand
        )
        let center = gpu.side / 2
        let first = try gpu.render(graph, punctualShadowMaps: maps)
        let initial = try await depth(gpu, maps, slice: 5)
        #expect(first.statistics.shadowPassCount == 6)
        #expect(gpu.pixel(first.pixels, x: center, y: center).y < 0.01)
        #expect(initial.contains { $0 != 0 })

        graph.cameraTransform.translation.y = 0.125
        #expect(try gpu.render(graph, punctualShadowMaps: maps).statistics.shadowPassCount == 0)
        #expect(try await depth(gpu, maps, slice: 5) == initial)
        graph.cameraTransform.translation.y = 0
        let biased = try panel(roll: 0.35, sampling: .basic, bias: 16)
        graph.roots[0].children[2].light = biased.light
        let raised = try gpu.render(graph, punctualShadowMaps: maps)
        let raisedDepth = try await depth(gpu, maps, slice: 5)
        #expect(raised.statistics.shadowPassCount == 6)
        let raisedColor = gpu.pixel(raised.pixels, x: center, y: center)
        #expect(raisedColor.y > 0.1)
        let nearMaps = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 0, pointCapacity: 1)
        let near = try gpu.render(scene(gpu, nodes: [target, shallow, biased], origin: .zero, precise: true),
                                  punctualShadowMaps: nearMaps)
        #expect(gpu.pixel(near.pixels, x: center, y: center) == raisedColor)

        graph.roots[0].children[2].transform.rotation = aim * simd_quatf(angle: 1.4, axis: SIMD3(0, 0, 1))
        let rotated = try gpu.render(graph, punctualShadowMaps: maps)
        #expect(rotated.statistics.shadowPassCount == 0 && rotated.pixels != raised.pixels)
        #expect(try await depth(gpu, maps, slice: 5) == raisedDepth)
        graph.roots[0].children[2].transform.translation.x += 0.25
        #expect(try gpu.render(graph, punctualShadowMaps: maps).statistics.shadowPassCount == 6)
        #expect(try await depth(gpu, maps, slice: 5) != raisedDepth)
    }

    private var aim: simd_quatf {
        simd_quatf(from: SIMD3(0, 0, -1), to: simd_normalize(receiverPosition - lightPosition))
    }

    private func panel(
        roll: Float, back: Bool = false, sampling: ShadowMapSampling? = nil, bias: Float = 0
    ) throws -> SceneNode {
        let spin = simd_quatf(angle: roll, axis: SIMD3(0, 0, 1))
        let facing = back ? aim * simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0)) : aim
        let shadow = try sampling.map {
            try PunctualLight.Shadow(normalBias: bias, filterRadius: 1, nearPlane: nearPlane, sampling: $0)
        }
        return SceneNode(
            transform: .init(translation: lightPosition, rotation: facing * spin),
            light: try PunctualLight(kind: .rectangle(width: 2.6, height: 0.55),
                                     color: SIMD3(0.65, 1, 0.8), intensity: 9, shadow: shadow)
        )
    }

    private func receiver(_ gpu: GeometryRenderFixture) throws -> SceneNode {
        let material = try PhysicallyBasedMaterial(
            metallic: 0.4, roughness: 0.35, lightIntensity: 0, ambientIntensity: 0,
            alphaMode: .opaque, clearcoat: .init(factor: 0.6, roughness: 0.3)
        )
        let mesh = try gpu.unlit(GeometryRenderFixture.square(2.25), material: .physicallyBased(material))
        return SceneNode(transform: .init(translation: receiverPosition), mesh: mesh, castsShadow: false)
    }

    private func blocker(_ gpu: GeometryRenderFixture) throws -> MeshRenderer {
        try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.34),
            renderState: .init(depthWrite: false, colorWriteMask: [], side: .double)
        )
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

    private func depth(_ gpu: GeometryRenderFixture, _ maps: PunctualShadowMaps, slice: Int) async throws -> Data {
        try await TextureTransfer.read(from: maps.texture, slice: slice, commandQueue: gpu.queue).bytes
    }
}
