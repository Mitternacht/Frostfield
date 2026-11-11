import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct VarianceShadowTests {
    enum Receiver: CaseIterable, Sendable {
        case lambert, pbr, catcher

        var material: MeshMaterial {
            get throws {
                switch self {
                case .lambert: .shadowedLambert
                case .pbr:
                    .physicallyBased(try .init(
                        roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, receivesShadows: true,
                        alphaMode: .opaque, specular: .init(factor: 0)
                    ))
                case .catcher: .classic(try .shadowCatcher(receivesDirectionalShadows: true))
                }
            }
        }

        func level(_ pixel: SIMD4<Float>) -> Float {
            self == .catcher ? 1 - pixel.w : pixel.x
        }
    }

    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture(side: 129) }

    @Test("Variance receiver edges soften while the deep shadow stays dark", arguments: Receiver.allCases)
    func filteredReceiver(_ receiver: Receiver) async throws {
        let graph = try scene(receiver)
        let basic = try map(.basic)
        let variance = try map(.variance)
        #expect(variance.varianceBlurRadius == 4)
        let hard = try gpu.render(graph, shadowMap: basic).pixels
        let soft = try gpu.render(graph, shadowMap: variance).pixels
        let center = gpu.side / 2
        let outside = gpu.side / 4
        func level(_ image: [SIMD4<Float>], at x: Int) -> Float {
            receiver.level(gpu.pixel(image, x: x, y: center))
        }
        let dark = level(hard, at: center), bright = level(hard, at: outside)
        #expect(bright - dark > 0.2)
        #expect(abs(level(soft, at: center) - dark) < 0.004)
        #expect(abs(level(soft, at: outside) - bright) < 0.004)
        func transitions(_ image: [SIMD4<Float>]) -> Int {
            (outside...(gpu.side - outside)).filter { x in
                let fraction = (level(image, at: x) - dark) / (bright - dark)
                return (0.05..<0.95).contains(fraction)
            }.count
        }
        #expect(transitions(hard) == 0)
        #expect(transitions(soft) >= 2)

        if receiver == .lambert {
            let depth = try await TextureTransfer.read(from: variance.texture, commandQueue: gpu.queue)
            let filtered = try await TextureTransfer.read(from: variance.shadowMoments, commandQueue: gpu.queue)
            #expect(filtered.pixelFormat == .rg32Float)
            let depths = depth.bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let moments = filtered.bytes.withUnsafeBytes { Array($0.bindMemory(to: SIMD2<Float>.self)) }
            let size = variance.resolution
            let radius = 4
            let row = size / 2
            for column in [0, size / 2, size / 2 + 8] {
                var sum = SIMD2<Float>.zero
                for y in (row - radius)...(row + radius) {
                    for x in (column - radius)...(column + radius) {
                        let z = depths[y * size + min(max(x, 0), size - 1)]
                        sum += SIMD2(z, z * z)
                    }
                }
                let reference = sum / Float((2 * radius + 1) * (2 * radius + 1))
                #expect(simd_distance(moments[row * size + column], reference) < 0.00001)
            }
            let edge = moments[row * size + size / 2 + 8]
            #expect(edge.y - edge.x * edge.x > 0.01)
        }
    }

    @Test("On-demand cascades keep and refresh their filtered moments with depth")
    func manualCascades() async throws {
        let map = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 96, directionToLight: SIMD3(0, 0, 1),
            cascades: .init(count: 2, splitLambda: 0, maximumDistance: 8, blendFraction: 0, casterMargin: 2),
            sampling: .variance, updateMode: .onDemand
        )
        try map.fitCascades(camera: gpu.camera, cameraTransform: .init(), aspectRatio: 1)
        var graph = try scene(.pbr)
        let initial = try gpu.render(graph, shadowMap: map)
        let center = gpu.side / 2 * gpu.side + gpu.side / 2
        #expect(initial.statistics.shadowPassCount == 2)
        #expect(initial.pixels[center].x < 0.005)
        let near = try await TextureTransfer.read(from: map.shadowMoments, slice: 0, commandQueue: gpu.queue)
        let far = try await TextureTransfer.read(from: map.shadowMoments, slice: 1, commandQueue: gpu.queue)
        for slice in [near, far] {
            let values = slice.bytes.withUnsafeBytes { Array($0.bindMemory(to: SIMD2<Float>.self)) }
            #expect(values.contains { $0.x < 0.99 })
            #expect(values.contains { $0 == SIMD2<Float>(repeating: 1) })
        }

        graph.roots[1].isVisible = false
        let cached = try gpu.render(graph, shadowMap: map)
        #expect(cached.statistics.shadowPassCount == 0 && cached.pixels[center] == initial.pixels[center])
        let cachedNear = try await TextureTransfer.read(from: map.shadowMoments, slice: 0, commandQueue: gpu.queue)
        let cachedFar = try await TextureTransfer.read(from: map.shadowMoments, slice: 1, commandQueue: gpu.queue)
        #expect(cachedNear.bytes == near.bytes && cachedFar.bytes == far.bytes)

        map.requestUpdate()
        let rebuilt = try gpu.render(graph, shadowMap: map)
        #expect(rebuilt.statistics.shadowPassCount == 2 && rebuilt.statistics.shadowDrawCount == 0)
        #expect(rebuilt.pixels[center].x > 0.25)
        for index in 0..<2 {
            let refreshed = try await TextureTransfer.read(from: map.shadowMoments, slice: index, commandQueue: gpu.queue)
            #expect(refreshed.bytes.withUnsafeBytes { $0.bindMemory(to: SIMD2<Float>.self).allSatisfy { $0 == SIMD2(repeating: 1) } })
        }
    }

    @Test("Variance settings bound filter memory and reject unsupported punctual and angular sampling")
    func settings() throws {
        for (resolution, radius, angle): (Int, Int, Float) in [(2049, 3, 0), (32, 17, 0), (32, 3, 0.1)] {
            #expect(throws: DirectionalShadowMap.Failure.varianceSettings) {
                try DirectionalShadowMap(
                    queue: gpu.queue, resolution: resolution, directionToLight: SIMD3(0, 0, 1),
                    halfExtent: 2, lightAngularRadius: angle, sampling: .variance, varianceBlurRadius: radius
                )
            }
        }
        let variance = try map(.variance)
        #expect(throws: DirectionalShadowMap.Failure.varianceSettings) {
            try variance.update(directionToLight: SIMD3(0, 0, 1), halfExtent: 2, lightAngularRadius: 0.1)
        }
        let spot = try PunctualLight.Shadow(sampling: .variance)
        #expect(throws: PunctualLight.Failure.shadowSampling) { try PunctualLight(kind: .point(), shadow: spot) }
    }

    private func scene(_ receiver: Receiver) throws -> SceneGraph {
        let floor = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 9, height: 9), material: receiver.material
        )
        let blocker = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1.4, height: 1.4),
            renderState: MeshRenderState(depthWrite: false, colorWriteMask: [])
        )
        return SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: floor, castsShadow: false),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: blocker)
        ], camera: gpu.camera)
    }

    private func map(_ sampling: ShadowMapSampling) throws -> DirectionalShadowMap {
        try DirectionalShadowMap(
            queue: gpu.queue, resolution: 48, directionToLight: SIMD3(0, 0, 1), center: SIMD3(0, 0, -3.5),
            halfExtent: 2, sampling: sampling, varianceBlurRadius: 4
        )
    }
}
