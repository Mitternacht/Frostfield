import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SpotVarianceShadowTests {
    private let gpu: GeometryRenderFixture
    private let outer: Float = 0.7
    private var center: Int { gpu.side / 2 }

    init() throws { gpu = try GeometryRenderFixture(side: 129) }

    @Test("Spot variance filters reversed depth and softens edges without leaking through the blocker",
          arguments: [false, true])
    func filteredReceiver(_ catcher: Bool) async throws {
        let models = try meshes(catcher: catcher)
        var source = try spot(sampling: .basic, radius: 0)
        var graph = SceneGraph(roots: [models.receiver, models.blocker, source], camera: gpu.camera)
        let basic = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 1, pointCapacity: 0)
        let hard = try gpu.render(graph, punctualShadowMaps: basic).pixels
        #expect(basic.shadowMoments == nil)
        source.light = try spot(sampling: .variance, radius: 4).light
        graph.roots[2] = source
        let variance = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 1, pointCapacity: 0)
        let soft = try gpu.render(graph, punctualShadowMaps: variance).pixels
        graph.roots[1].isVisible = false
        let open = try gpu.render(graph).pixels
        func visibility(_ image: [SIMD4<Float>], at x: Int) -> Float {
            let pixel = gpu.pixel(image, x: x, y: center)
            if catcher { return 1 - pixel.w }
            return pixel.x / gpu.pixel(open, x: x, y: center).x
        }
        let outside = gpu.side / 4
        #expect(visibility(soft, at: center) < 0.01 && visibility(hard, at: center) < 0.01)
        #expect(abs(visibility(soft, at: outside) - 1) < 0.01)
        let range = outside...(gpu.side - outside)
        #expect(!range.contains { (0.05..<0.95).contains(visibility(hard, at: $0)) })
        #expect(range.filter { (0.05..<0.95).contains(visibility(soft, at: $0)) }.count >= 2)

        if !catcher {
            let momentsTexture = try #require(variance.shadowMoments)
            let depth = try await TextureTransfer.read(from: variance.texture, commandQueue: gpu.queue)
            let filtered = try await TextureTransfer.read(from: momentsTexture, commandQueue: gpu.queue)
            #expect(filtered.pixelFormat == .rg32Float)
            let depths = depth.bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let moments = filtered.bytes.withUnsafeBytes { Array($0.bindMemory(to: SIMD2<Float>.self)) }
            let size = variance.resolution, row = size / 2
            #expect(abs(depths[row * size + row] - 0.5 / 3) < 0.00001)
            for column in [0, row, row + 7] {
                var sum = SIMD2<Float>.zero
                for y in (row - 4)...(row + 4) {
                    for x in (column - 4)...(column + 4) {
                        let z = depths[y * size + min(max(x, 0), size - 1)]
                        sum += SIMD2(z, z * z)
                    }
                }
                #expect(simd_distance(moments[row * size + column], sum / 81) < 0.00001)
            }
            let edge = moments[row * size + row + 7]
            #expect(edge.y - edge.x * edge.x > 0.0001)
        }
    }

    @Test("Switching a cached spot to variance refreshes its moments and leaves other modes intact")
    func mixedOnDemand() async throws {
        let models = try meshes(catcher: false, blockerWidth: 3.5)
        let red = try spot(sampling: .basic, radius: 0, color: SIMD3(1, 0, 0))
        let green = try spot(sampling: .basic, radius: 0, color: SIMD3(0, 1, 0))
        let point = SceneNode(light: try .init(
            kind: .point(), color: SIMD3(0, 0, 1), intensity: 40,
            shadow: .init(normalBias: 0, filterRadius: 1, nearPlane: 0.5)
        ))
        let rectangle = SceneNode(light: try .init(
            kind: .rectangle(width: 0.5, height: 0.5), color: SIMD3(0, 1, 0), intensity: 40,
            shadow: .init(normalBias: 0, sampling: .basic)
        ))
        var graph = SceneGraph(roots: [models.receiver, models.blocker, red, green, point, rectangle], camera: gpu.camera)
        let maps = try PunctualShadowMaps(
            queue: gpu.queue, resolution: 32, spotCapacity: 2, pointCapacity: 2, updateMode: .onDemand
        )
        let original = try gpu.render(graph, punctualShadowMaps: maps)
        #expect(original.statistics.shadowPassCount == 14 && maps.shadowMoments == nil)
        func depths() async throws -> [Data] {
            var result: [Data] = []
            for slice in [1, 7, 13] {
                result.append(try await TextureTransfer.read(from: maps.texture, slice: slice, commandQueue: gpu.queue).bytes)
            }
            return result
        }
        let untouched = try await depths()

        // Basic and variance with radius zero have the same fitted projection.
        graph.roots[2].light = try spot(sampling: .variance, radius: 0, color: SIMD3(1, 0, 0)).light
        let switched = try gpu.render(graph, punctualShadowMaps: maps)
        #expect(switched.statistics.shadowPassCount == 1)
        #expect(try await depths() == untouched)
        let moments = try #require(maps.shadowMoments)
        let sample = MTLRegionMake2D(16, 16, 1, 1)
        let first = try await TextureTransfer.read(from: moments, region: sample, commandQueue: gpu.queue)
        let centerMoments = first.bytes.withUnsafeBytes { $0.loadUnaligned(as: SIMD2<Float>.self) }
        #expect(abs(centerMoments.x - 0.5 / 3) < 0.00001)

        graph.roots[2].light = try spot(sampling: .variance, radius: 4, color: SIMD3(1, 0, 0)).light
        #expect(try gpu.render(graph, punctualShadowMaps: maps).statistics.shadowPassCount == 1)
        graph.roots[2].light = try spot(sampling: .variance, radius: 5, color: SIMD3(1, 0, 0)).light
        let wider = try gpu.render(graph, punctualShadowMaps: maps)
        #expect(wider.statistics.shadowPassCount == 1) // Both radii hit the same projection guard-band cap.
        let previous = gpu.pixel(wider.pixels, x: center, y: center)
        graph.roots[1].isVisible = false
        let cached = try gpu.render(graph, punctualShadowMaps: maps)
        #expect(cached.statistics.shadowPassCount == 0 && gpu.pixel(cached.pixels, x: center, y: center) == previous)
        #expect(maps.requestUpdate(for: red.id))
        let refreshed = try gpu.render(graph, punctualShadowMaps: maps)
        let after = gpu.pixel(refreshed.pixels, x: center, y: center)
        #expect(refreshed.statistics.shadowPassCount == 1 && after.x > previous.x + 0.3)
        #expect(after.y == previous.y && after.z == previous.z)
        #expect(try await depths() == untouched)
        let clear = try await TextureTransfer.read(from: moments, region: sample, commandQueue: gpu.queue)
        #expect(clear.bytes.withUnsafeBytes { $0.loadUnaligned(as: SIMD2<Float>.self) } == .zero)

        try maps.setUpdateMode(.automatic, for: red.id)
        let blocked = try gpu.makeTarget()
        let unblocked = try gpu.makeTarget()
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        graph.roots[1].isVisible = true
        let firstDraw = try graph.encode(
            pass: gpu.pass(color: blocked.color, depth: blocked.depth), commandBuffer: buffer, punctualShadowMaps: maps
        )
        graph.roots[1].isVisible = false
        let secondDraw = try graph.encode(
            pass: gpu.pass(color: unblocked.color, depth: unblocked.depth), commandBuffer: buffer, punctualShadowMaps: maps
        )
        #expect(firstDraw.shadowPassCount == 1)
        #expect(secondDraw.shadowPassCount == 1)
        try gpu.finish(buffer)
        let firstPixels = try gpu.pixels(of: blocked.color)
        let secondPixels = try gpu.pixels(of: unblocked.color)
        #expect(gpu.pixel(firstPixels, x: center, y: center) == previous)
        #expect(gpu.pixel(secondPixels, x: center, y: center) == after)
        #expect(try await depths() == untouched)
    }

    @Test("Spot variance settings round trip, and point or rectangle modes reject variance")
    func settings() throws {
        let light = try #require(spot(sampling: .variance, radius: 7).light)
        let saved = try JSONEncoder().encode(light)
        #expect(try JSONDecoder().decode(PunctualLight.self, from: saved) == light)
        #expect(throws: PunctualLight.Failure.shadowSampling) { try PunctualLight(kind: .point(), shadow: light.shadow) }
        #expect(throws: PunctualLight.Failure.shadowSampling) {
            try PunctualLight(kind: .rectangle(width: 1, height: 1), shadow: light.shadow)
        }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(sampling: .variance, varianceBlurRadius: 17) }
    }

    private func spot(
        sampling: ShadowMapSampling, radius: Int, color: SIMD3<Float> = SIMD3(repeating: 1)
    ) throws -> SceneNode {
        SceneNode(light: try .init(
            kind: .spot(innerConeAngle: 0.6, outerConeAngle: outer), color: color, intensity: 40,
            shadow: .init(normalBias: 0, nearPlane: 0.5, sampling: sampling, varianceBlurRadius: radius)
        ))
    }

    private func meshes(catcher: Bool, blockerWidth: Float = 1.8) throws -> (receiver: SceneNode, blocker: SceneNode) {
        let material: MeshMaterial = catcher
            ? .classic(try .shadowCatcher())
            : .physicallyBased(try .init(
                roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, specular: .init(factor: 0)
            ))
        let receiver = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 9, height: 9), material: material
        )
        let blocker = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: blockerWidth, height: blockerWidth),
            renderState: MeshRenderState(depthWrite: false, colorWriteMask: [])
        )
        return (
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: receiver, castsShadow: false),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: blocker)
        )
    }
}
