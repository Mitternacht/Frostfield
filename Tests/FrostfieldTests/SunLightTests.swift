import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SunLightTests {
    @Test("The sun fits the live perspective view, reuses its maps and follows the light direction")
    func movingViewAndSun() async throws {
        let gpu = try RasterLightingFixture(width: 144, height: 96)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 80)
        let direction = SIMD3<Float>(0.9, 1, 0)
        let sun = try SunLight(
            queue: gpu.queue, directionToLight: direction, resolution: 128,
            cascades: .init(count: 2, splitLambda: 0.5, maximumDistance: 8),
            sampling: .basic, updateMode: .onDemand
        )
        #expect(sun.shadowMap.cascadeSplits.isEmpty)
        let texture = sun.shadowMap.texture
        let geometry = try scene(gpu, camera: camera, positions: [0, 8], depthMode: .forward)
        let projection = try camera.projection(aspectRatio: Float(gpu.width) / Float(gpu.height))
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()

        func draw(x: Float) throws -> (SceneDrawStatistics, Float, Float) {
            var graph = geometry
            let pose = RasterLightingFixture.cameraTransform(position: SIMD3(x, 3, 0), pitch: -.pi / 2)
            graph.cameraTransform = pose
            var statistics = SceneDrawStatistics()
            try gpu.run { command in
                statistics = try graph.encode(
                    pass: gpu.pass(color: color, depth: depth), commandBuffer: command, shadowMap: sun.shadowMap
                )
            }
            let pixels = try gpu.pixels(of: color)
            return (statistics, sample(pixels, gpu: gpu, point: SIMD3(x - 0.66, 0, 0), projection: projection, pose: pose),
                    sample(pixels, gpu: gpu, point: SIMD3(x + 0.66, 0, 0), projection: projection, pose: pose))
        }

        let first = try draw(x: 0)
        #expect(first.0.shadowPassCount == 2 && first.0.shadowDrawCount >= 2)
        #expect(first.1 < 0.2 && first.2 > 0.65)
        let splits = sun.shadowMap.cascadeSplits
        let expected = Float((4.05 + (0.8).squareRoot()) * 0.5)
        #expect(splits.count == 2 && abs(splits[0] - expected) < 1e-5 && splits[1] == 8)
        let firstViews = sun.shadowMap.cascadeViews
        let firstDepth = try await TextureTransfer.read(from: texture, slice: 0, commandQueue: gpu.queue).bytes
        #expect(firstDepth.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).contains {
                bytes.loadUnaligned(fromByteOffset: $0, as: Float.self) < 1
            }
        })

        let manual = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 128, directionToLight: direction,
            cascades: .init(count: 2, splitLambda: 0.5, maximumDistance: 8), sampling: .basic
        )
        try manual.fitCascades(
            camera: camera, cameraTransform: RasterLightingFixture.cameraTransform(position: SIMD3(0, 3, 0), pitch: -.pi / 2),
            aspectRatio: Float(gpu.width) / Float(gpu.height)
        )
        #expect(manual.cascadeViews == firstViews)
        let repeated = try draw(x: 0)
        #expect(repeated.0.shadowPassCount == 0)
        #expect(repeated.1 == first.1 && repeated.2 == first.2)

        let moved = try draw(x: 8)
        #expect(moved.0.shadowPassCount == 2 && moved.1 < 0.2 && moved.2 > 0.65)
        #expect(sun.shadowMap.cascadeViews != firstViews && sun.shadowMap.texture === texture)
        #expect(sun.shadowMap.cascadeSplits == splits)

        try sun.update(directionToLight: SIMD3(-0.9, 1, 0))
        let reaimed = try draw(x: 8)
        #expect(reaimed.0.shadowPassCount == 2 && reaimed.1 > 0.65 && reaimed.2 < 0.2)
        #expect(sun.shadowMap.texture === texture && sun.intensity == 1)
        let settled = try draw(x: 8)
        #expect(settled.0.shadowPassCount == 0)
    }

    @Test("A shifted explicit view fits the sun independently of the graph camera")
    func explicitView() throws {
        let gpu = try RasterLightingFixture(width: 144, height: 96)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.2, far: 45)
        let sun = try SunLight(
            queue: gpu.queue, directionToLight: SIMD3(0.9, 1, 0), resolution: 128,
            cascades: .init(count: 2, splitLambda: 0.5, maximumDistance: 7), sampling: .basic
        )
        let mode = MeshDepthMode.reversed
        var graph = try scene(gpu, camera: camera, positions: [4], depthMode: mode)
        graph.cameraTransform.translation = SIMD3(-50, 3, 0)
        let pose = RasterLightingFixture.cameraTransform(position: SIMD3(4, 3, 0), pitch: -.pi / 2)
        let projection = simd_float4x4(
            SIMD4(0.4, 0, 0, 0), SIMD4(0, 0.55, 0, 0), SIMD4(0, 0, 0.05, 0), SIMD4(-0.15, 0.12, 1, 1)
        )
        let view = try ShaderGraphCamera(viewMatrix: simd_inverse(pose.matrix), projectionMatrix: projection, near: 0, far: 20)
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let pass = gpu.pass(color: color, depth: depth, depthMode: mode)
        let clip = projection * view.viewMatrix
        try gpu.run { command in
            let stats = try graph.encode(
                pass: pass, commandBuffer: command, clipFromWorld: clip, cameraPosition: pose.translation,
                directionToCamera: pose.rotation.act(SIMD3(0, 0, 1)), depthMode: mode, shadowMap: sun.shadowMap, graphCamera: view
            )
            #expect(stats.shadowPassCount == 2)
        }
        #expect(sun.shadowMap.cascadeSplits == [3.5, 7])
        let pixels = try gpu.pixels(of: color)
        #expect(sample(pixels, gpu: gpu, point: SIMD3(3.34, 0, 0), projection: projection, pose: pose) < 0.2)
        #expect(sample(pixels, gpu: gpu, point: SIMD3(4.66, 0, 0), projection: projection, pose: pose) > 0.65)
        let priorViews = sun.shadowMap.cascadeViews
        let invalid = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: DirectionalShadowMap.Failure.invalidCamera) {
            try graph.encode(pass: pass, commandBuffer: invalid, clipFromWorld: clip, cameraPosition: pose.translation,
                             depthMode: mode, shadowMap: sun.shadowMap)
        }
        #expect(sun.shadowMap.cascadeViews == priorViews)
    }

    @Test("Sun defaults and solar-position updates keep the native map and authored intensity")
    func solarPosition() throws {
        let gpu = try RasterLightingFixture(width: 16, height: 16)
        let position = try SolarPosition(julianDay: 2_452_930.312847, latitude: 39.742476, longitude: -105.1786)
        let sun = try SunLight(queue: gpu.queue, solarPosition: position, refracted: false, intensity: 2, resolution: 32)
        #expect(simd_distance(sun.directionToLight, position.direction(refracted: false)) < 1e-6)
        #expect(sun.shadowMap.cascades?.count == 2)
        #expect(sun.shadowMap.cascades?.splitLambda == 0.5)
        #expect(sun.shadowMap.cascades?.maximumDistance == 500)
        #expect(sun.shadowMap.cascades?.blendFraction == 0.1)
        let texture = sun.shadowMap.texture
        try sun.update(solarPosition: position)
        #expect(sun.intensity == 2 && sun.shadowMap.texture === texture)
        #expect(simd_distance(sun.directionToLight, position.direction()) < 1e-6)
        #expect(throws: DirectionalShadowMap.Failure.invalidDirection) {
            try sun.update(directionToLight: .zero, intensity: 3)
        }
        #expect(sun.intensity == 2 && simd_distance(sun.directionToLight, position.direction()) < 1e-6)
    }

    enum Receiver: CaseIterable, Sendable {
        case simpleLambert, classicLambert, pbr, graphPBR

        func material() throws -> MeshMaterial {
            switch self {
            case .simpleLambert:
                return .shadowedLambert
            case .classicLambert:
                return .classic(try ClassicMaterial(
                    model: .lambert, ambient: SIMD3(repeating: 0.5), receivesShadows: true
                ))
            case .pbr, .graphPBR:
                let graph = self == .graphPBR ? try SurfaceShaderGraph { s in
                    .init(baseColor: s.constant(SIMD3<Float>(repeating: 0.8)))
                } : nil
                return .physicallyBased(try PhysicallyBasedMaterial(
                    metallic: 0, roughness: 0.8, lightIntensity: 0, ambientIntensity: 0.18,
                    receivesShadows: true, surfaceGraph: graph
                ))
            }
        }
    }

    @Test("Sun RGB scales only direct light and reuses the fitted cascades", arguments: Receiver.allCases)
    func coloredDirectLight(_ receiver: Receiver) throws {
        let gpu = try RasterLightingFixture(width: 144, height: 96)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 80)
        let direction = SIMD3<Float>(0.9, 1, 0)
        let sun = try SunLight(
            queue: gpu.queue, directionToLight: direction, intensity: 1.4, resolution: 128,
            cascades: .init(count: 2, splitLambda: 0.5, maximumDistance: 8),
            sampling: .basic, updateMode: .onDemand
        )
        let floor = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 10, height: 8),
            material: receiver.material()
        )
        let caster = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .box(width: 0.6, height: 1, depth: 0.6)
        )
        let pose = RasterLightingFixture.cameraTransform(position: SIMD3(0, 3, 0), pitch: -.pi / 2)
        let graph = SceneGraph(
            roots: [
                SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: floor, castsShadow: false),
                SceneNode(transform: .init(translation: SIMD3(0, 0.6, 0)), mesh: caster)
            ],
            camera: camera, cameraTransform: pose
        )
        let projection = try camera.projection(aspectRatio: Float(gpu.width) / Float(gpu.height))
        let colorTarget = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let texture = sun.shadowMap.texture

        func draw(expectedPasses: Int) throws -> (lit: SIMD3<Float>, shadowed: SIMD3<Float>) {
            try gpu.run { command in
                let statistics = try graph.encode(
                    pass: gpu.pass(color: colorTarget, depth: depth), commandBuffer: command, shadowMap: sun.shadowMap
                )
                #expect(statistics.shadowPassCount == expectedPasses)
            }
            let pixels = try gpu.pixels(of: colorTarget)
            func rgb(_ x: Float) -> SIMD3<Float> {
                let p = gpu.project(SIMD3(x, 0, 0), clipFromView: projection, worldFromView: pose.matrix)
                let pixel = pixels[Int(p.y) * gpu.width + Int(p.x)]
                #expect(pixel.w == 1)
                return SIMD3(pixel.x, pixel.y, pixel.z)
            }
            return (rgb(0.9), rgb(-0.9))
        }

        let neutral = try draw(expectedPasses: 2)
        let fit = sun.shadowMap.cascadeViews
        #expect(sun.color == SIMD3(repeating: 1))
        try sun.update(color: .zero)
        let ambient = try draw(expectedPasses: 0)
        #expect(simd_reduce_min(ambient.lit) > 0.1)
        #expect(simd_reduce_max(abs(neutral.shadowed - ambient.shadowed)) < 0.002)
        let uncoloredDirect = neutral.lit - ambient.lit
        #expect(simd_reduce_min(uncoloredDirect) > 0.17)

        let tint = SIMD3<Float>(0.15, 0.5, 0.9)
        try sun.update(color: tint)
        let tinted = try draw(expectedPasses: 0)
        #expect(simd_reduce_max(abs(tinted.lit - (ambient.lit + tint * uncoloredDirect))) < 0.003)
        #expect(tinted.shadowed == ambient.shadowed)
        #expect(sun.intensity == 1.4 && sun.color == tint)
        #expect(sun.shadowMap.shaderUniforms.last == SIMD4(tint, 0))
        #expect(sun.shadowMap.texture === texture && sun.shadowMap.cascadeViews == fit)

        let uniforms = sun.shadowMap.shaderUniforms
        #expect(throws: DirectionalShadowMap.Failure.invalidColor) {
            try sun.update(directionToLight: -direction, color: SIMD3(0.5, .nan, 1), intensity: 3)
        }
        #expect(sun.intensity == 1.4 && sun.color == tint && sun.shadowMap.cascadeViews == fit)
        #expect(sun.shadowMap.shaderUniforms == uniforms)
        let afterFailure = try draw(expectedPasses: 0)
        #expect(afterFailure.lit == tinted.lit && afterFailure.shadowed == tinted.shadowed)

        try sun.update(color: SIMD3(repeating: 1))
        let restored = try draw(expectedPasses: 0)
        #expect(restored.lit == neutral.lit && restored.shadowed == neutral.shadowed)
    }

    @Test("Single-map and solar-position color updates validate and preserve other light parameters")
    func colorUpdates() throws {
        let gpu = try RasterLightingFixture(width: 16, height: 16)
        let tint = SIMD3<Float>(0.25, 0.5, 1)
        let overhead = SIMD3<Float>(0, 1, 0)
        #expect(throws: DirectionalShadowMap.Failure.invalidColor) {
            try DirectionalShadowMap(
                queue: gpu.queue, resolution: 32, directionToLight: overhead,
                halfExtent: 5, color: SIMD3(1, 1.1, 1)
            )
        }
        let single = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 32, directionToLight: overhead, halfExtent: 5, color: tint, intensity: 2
        )
        #expect(single.color == tint && single.shaderUniforms.last == SIMD4(tint, 0))
        let uniformState = single.shaderUniforms
        #expect(throws: DirectionalShadowMap.Failure.invalidColor) {
            try single.update(directionToLight: -overhead, halfExtent: 2, color: SIMD3(-0.1, 1, 1))
        }
        #expect(single.shaderUniforms == uniformState)
        try single.update(directionToLight: overhead, halfExtent: 4, intensity: 3)
        #expect(single.color == tint && single.lightParameters.w == 3)

        let position = try SolarPosition(julianDay: 2_452_930.312847, latitude: 39.742476, longitude: -105.1786)
        let sun = try SunLight(queue: gpu.queue, solarPosition: position, color: tint, intensity: 2, resolution: 32)
        #expect(sun.color == tint)
        try sun.update(solarPosition: position, refracted: false, color: SIMD3(1, 0.75, 0.5))
        #expect(sun.color == SIMD3(1, 0.75, 0.5) && sun.intensity == 2)
        #expect(simd_distance(sun.directionToLight, position.direction(refracted: false)) < 1e-6)
    }

    private func scene(
        _ gpu: RasterLightingFixture, camera: PerspectiveCamera, positions: [Float], depthMode: MeshDepthMode
    ) throws -> SceneGraph {
        let floor = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 40, height: 20),
            material: .shadowedLambert, depthMode: depthMode
        )
        let caster = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .box(width: 0.6, height: 1, depth: 0.6),
            depthMode: depthMode
        )
        var roots = [SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: floor, castsShadow: false)]
        for x in positions {
            roots.append(SceneNode(transform: .init(translation: SIMD3(x, 0.6, 0)), mesh: caster))
        }
        return SceneGraph(roots: roots, camera: camera)
    }

    private func sample(
        _ pixels: [SIMD4<Float>], gpu: RasterLightingFixture, point: SIMD3<Float>,
        projection: simd_float4x4, pose: SceneTransform
    ) -> Float {
        let position = gpu.project(point, clipFromView: projection, worldFromView: pose.matrix)
        return pixels[Int(position.y) * gpu.width + Int(position.x)].x
    }
}
