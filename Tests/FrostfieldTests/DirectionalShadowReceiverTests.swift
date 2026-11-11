import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct DirectionalShadowReceiverTests {
    enum Region: CaseIterable, Sendable {
        case single, firstCascade, secondCascade

        var cameraHeight: Float { self == .firstCascade ? 1.8 : 3 }
        var mapCount: Int { self == .single ? 1 : 2 }
    }

    @Test("Directional normal bias and strength adjust real receiver shading while keeping cached depth",
          arguments: Region.allCases)
    func receiverControls(_ region: Region) async throws {
        let gpu = try RasterLightingFixture(width: 144, height: 96)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 80)
        let aspect = Float(gpu.width) / Float(gpu.height)
        let projection = try camera.projection(aspectRatio: aspect)
        let localPose = RasterLightingFixture.cameraTransform(
            position: SIMD3(0, region.cameraHeight, 0), pitch: -.pi / 2
        )
        let floor = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 10, height: 8),
            material: .shadowedLambert
        )
        let caster = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .box(width: 0.6, height: 1, depth: 0.6)
        )
        let nodes = [
            SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: floor, castsShadow: false),
            SceneNode(transform: .init(translation: SIMD3(0, 0.6, 0)), mesh: caster),
        ]
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let shadowedPixel = gpu.project(SIMD3(-0.9, 0, 0), clipFromView: projection, worldFromView: localPose.matrix)
        let litPixel = gpu.project(SIMD3(0.9, 0, 0), clipFromView: projection, worldFromView: localPose.matrix)

        func render(origin: SIMD3<Float>, highPrecision: Bool) async throws -> [Float] {
            var pose = localPose
            pose.translation += origin
            var graph = SceneGraph(
                roots: [SceneNode(transform: .init(translation: origin), children: nodes)],
                camera: camera, cameraTransform: pose
            )
            graph.highPrecision = highPrecision
            let direction = SIMD3<Float>(0.9, 1, 0)
            let map: DirectionalShadowMap
            if region == .single {
                map = try DirectionalShadowMap(
                    queue: gpu.queue, resolution: 128, directionToLight: direction,
                    center: origin, halfExtent: 6, sampling: .basic, updateMode: .onDemand
                )
            } else {
                let sun = try SunLight(
                    queue: gpu.queue, directionToLight: direction, resolution: 128,
                    cascades: .init(count: 2, splitLambda: 0.5, maximumDistance: 8),
                    sampling: .basic, updateMode: .onDemand
                )
                map = sun.shadowMap
            }

            func draw(expectedShadowPasses: Int) throws -> (shadowed: Float, lit: Float) {
                try gpu.run { command in
                    let statistics = try graph.encode(
                        pass: gpu.pass(color: color, depth: depth), commandBuffer: command, shadowMap: map
                    )
                    #expect(statistics.shadowPassCount == expectedShadowPasses)
                }
                let pixels = try gpu.pixels(of: color)
                func red(_ position: SIMD2<Float>) -> Float {
                    pixels[Int(position.y) * gpu.width + Int(position.x)].x
                }
                return (red(shadowedPixel), red(litPixel))
            }

            let full = try draw(expectedShadowPasses: region.mapCount)
            #expect(full.shadowed < 0.2 && full.lit > 0.65, "\(full)")
            if region == .firstCascade { #expect(region.cameraHeight < map.cascadeSplits[0]) }
            if region == .secondCascade { #expect(region.cameraHeight > map.cascadeSplits[0]) }
            let slice = region == .secondCascade ? 1 : 0
            let originalDepth = try await TextureTransfer.read(from: map.texture, slice: slice, commandQueue: gpu.queue)
            #expect(originalDepth.bytes.withUnsafeBytes { bytes in
                stride(from: 0, to: bytes.count, by: 4).contains {
                    bytes.loadUnaligned(fromByteOffset: $0, as: Float.self) < 1
                }
            })

            try map.updateReceiver(normalBias: 0, shadowStrength: 0.35)
            let partial = try draw(expectedShadowPasses: 0)
            #expect(abs(partial.shadowed - (0.35 * full.shadowed + 0.65 * full.lit)) < 0.003, "\(partial)")
            #expect(partial.lit == full.lit)

            try map.updateReceiver(normalBias: 0, shadowStrength: 0)
            let disabled = try draw(expectedShadowPasses: 0)
            #expect(disabled.shadowed == full.lit && disabled.lit == full.lit)

            try map.updateReceiver(normalBias: 0.85, shadowStrength: 1)
            let shifted = try draw(expectedShadowPasses: 0)
            #expect(shifted.shadowed == full.lit && shifted.lit == full.lit, "\(shifted)")
            let reusedDepth = try await TextureTransfer.read(from: map.texture, slice: slice, commandQueue: gpu.queue)
            #expect(reusedDepth.bytes == originalDepth.bytes)
            return [full.shadowed, full.lit, partial.shadowed, disabled.shadowed, shifted.shadowed]
        }

        let ordinary = try await render(origin: .zero, highPrecision: false)
        let relocated = try await render(origin: SIMD3(0x1p60, 0, 0x1p60), highPrecision: true)
        #expect(ordinary == relocated)
    }

    @Test("Invalid receiver controls are atomic and sun updates preserve authored values")
    func validation() throws {
        let gpu = try RasterLightingFixture(width: 16, height: 16)
        #expect(throws: DirectionalShadowMap.Failure.invalidShadowStrength) {
            try DirectionalShadowMap(
                queue: gpu.queue, resolution: 32, directionToLight: SIMD3(0, 1, 0),
                halfExtent: 5, shadowStrength: .nan
            )
        }
        let sun = try SunLight(
            queue: gpu.queue, intensity: 2, resolution: 32, normalBias: -0.05, shadowStrength: 0.7
        )
        let map = sun.shadowMap
        let uniforms = map.shaderUniforms
        #expect(throws: DirectionalShadowMap.Failure.invalidNormalBias) {
            try map.updateReceiver(normalBias: .infinity, shadowStrength: 0)
        }
        #expect(throws: DirectionalShadowMap.Failure.invalidShadowStrength) {
            try map.updateReceiver(normalBias: 0, shadowStrength: 1.1)
        }
        #expect(map.shaderUniforms == uniforms)
        try sun.update(directionToLight: SIMD3(1, 1, 0))
        #expect(map.normalBias == -0.05 && map.shadowStrength == 0.7 && sun.intensity == 2)
    }
}
