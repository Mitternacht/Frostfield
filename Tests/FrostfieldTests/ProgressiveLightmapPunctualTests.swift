import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProgressiveLightmapPunctualTests {
    @Test("Point irradiance follows inverse square, receiver cosine and the range window")
    func pointFalloff() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let color = SIMD3<Float>(0.5, 0.25, 1)
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        func render(z: Float, range: Float? = nil, bias: Float = 0.0001) throws -> [SIMD4<Float>] {
            let light = try ProgressiveLightmapBaker.PositionedLight(
                PunctualLight(kind: .point(range: range), color: color, intensity: 8),
                position: SIMD3(0, 0, z)
            )
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 3, height: 1, textureCoordinateSet: .primary,
                punctualLights: [light], maximumBounces: 0, shadowBias: bias
            )
            return try pixels(baker.snapshot().irradianceTexture)
        }
        let point = try render(z: 2)
        #expect(point[1] == SIMD4(color * 2, 1))
        let squared: Float = 4 + 4 / 9
        let oblique = color * (8 / squared * (2 / squared.squareRoot()))
        #expect(simd_distance(point[0], SIMD4(oblique, 1)) < 0.001)
        #expect(point[0] == point[2])
        #expect(try render(z: 4)[1] == SIMD4(color * 0.5, 1))
        #expect(try render(z: 2, bias: 0.75) == point)
        #expect(try render(z: 2, range: 2).allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
        let rangeWindow: Float = 1 - (2.0 / 3.0) * (2.0 / 3.0) * (2.0 / 3.0) * (2.0 / 3.0)
        #expect(try simd_distance(render(z: 2, range: 3)[1], SIMD4(color * (2 * rangeWindow), 1)) < 0.001)
        #expect(try render(z: -2).allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
        #expect(try render(z: 0)[1] == SIMD4(0, 0, 0, 1))
    }

    @Test("A resolved spot gives full center intensity, a squared penumbra and a dark back side")
    func spotCone() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let inner: Float = .pi / 12, outer: Float = .pi / 5
        let spot = try PunctualLight(kind: .spot(innerConeAngle: inner, outerConeAngle: outer), intensity: 8)
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 4, height: 2))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, 2)), light: spot)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20)
        let graph = SceneGraph(roots: [node], camera: camera)
        let resolved = try #require(try graph.resolve().first?.light)
        func render(_ light: ProgressiveLightmapBaker.PositionedLight) throws -> [SIMD4<Float>] {
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 5, height: 1, textureCoordinateSet: .primary,
                punctualLights: [light], maximumBounces: 0
            )
            return try pixels(baker.snapshot().irradianceTexture)
        }
        let front = try render(.init(resolved))
        #expect(front[2] == SIMD4(2, 2, 2, 1))
        #expect(front[0] == SIMD4(0, 0, 0, 1) && front[4] == front[0])
        let squared: Float = 4 + 0.8 * 0.8
        let cosine = 2 / squared.squareRoot()
        let ramp = (cosine - cos(outer)) / (cos(inner) - cos(outer))
        let penumbra = 8 / squared * cosine * ramp * ramp
        #expect(abs(front[1].x - penumbra) < 0.001 && front[1] == front[3])
        let away = try render(.init(spot, position: resolved.position, direction: SIMD3(0, 0, 5)))
        #expect(away.allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
    }

    @Test("Point shadows end before the light and keep directional visibility separate")
    func finiteShadows() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: plane)
        let light = try ProgressiveLightmapBaker.PositionedLight(
            PunctualLight(kind: .point(), color: SIMD3(1, 0, 0), intensity: 4), position: SIMD3(0, 0, 2)
        )
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0, 1, 0))
        func render(blocker: Float? = nil, bias: Float = 0.0001) throws -> SIMD4<Float> {
            let surfaces = try blocker.map {
                [try ProgressiveLightmapBaker.Surface(geometry: plane, worldFromObject: SceneTransform(translation: SIMD3(0, 0, $0)).matrix)]
            } ?? []
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 1, height: 1, textureCoordinateSet: .primary,
                lights: [sun], punctualLights: [light], surfaces: surfaces, maximumBounces: 0, shadowBias: bias
            )
            return try #require(try pixels(baker.snapshot().irradianceTexture).first)
        }
        #expect(try render() == SIMD4(1, 1, 0, 1))
        #expect(try render(blocker: 1) == SIMD4(0, 0, 0, 1))
        #expect(try render(blocker: 2) == SIMD4(1, 0, 0, 1))
        #expect(try render(blocker: 3) == SIMD4(1, 0, 0, 1))
        #expect(try render(blocker: 1.5, bias: 3).x == 0)
    }

    @Test("A spotlight aimed away from the receiver reaches it after one diffuse reflection")
    func spotBounce() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let panelTransform = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, -1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 0, 1, 1)
        ))
        let panel = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 4, height: 4), worldFromObject: panelTransform, diffuseAlbedo: SIMD3(0.8, 0, 0)
        )
        let light = try ProgressiveLightmapBaker.PositionedLight(
            PunctualLight(kind: .spot(range: 2, innerConeAngle: 0.25, outerConeAngle: 0.8)),
            position: SIMD3(0, 0, 0.5), direction: SIMD3(0, 0, 1)
        )
        func bake(bounces: Int) throws -> SIMD4<Float> {
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 1, height: 1, textureCoordinateSet: .primary,
                punctualLights: [light], surfaces: [panel], includesReceiverInTransport: false, maximumBounces: bounces, seed: 9
            )
            #expect(try pixels(baker.snapshot().irradianceTexture).first == SIMD4(0, 0, 0, 1))
            for _ in 0..<4 { try baker.accumulate(samples: 256) }
            return try #require(try pixels(baker.snapshot().irradianceTexture).first)
        }
        #expect(try bake(bounces: 0) == SIMD4(0, 0, 0, 1))
        let bounced = try bake(bounces: 1)
        #expect((0.1...0.65).contains(bounced.x))
        #expect(bounced.y == 0 && bounced.z == 0 && bounced.w == 1)
    }

    @Test("Punctual bake validates pose, supported light kinds and combined light count")
    func validation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let point = try PunctualLight(kind: .point())
        #expect(throws: ProgressiveLightmapBaker.Failure.punctualPose) {
            try ProgressiveLightmapBaker.PositionedLight(point, position: SIMD3(.nan, 0, 0))
        }
        #expect(throws: ProgressiveLightmapBaker.Failure.punctualPose) {
            try ProgressiveLightmapBaker.PositionedLight(point, position: .zero, direction: .zero)
        }
        let directional = try PunctualLight(kind: .directional)
        #expect(throws: ProgressiveLightmapBaker.Failure.unsupportedPunctualLight) {
            try ProgressiveLightmapBaker.PositionedLight(directional, position: .zero)
        }
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let positioned = try ProgressiveLightmapBaker.PositionedLight(point, position: SIMD3(0, 0, 2))
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(repeating: 1))
        #expect(throws: ProgressiveLightmapBaker.Failure.lightCount) {
            try ProgressiveLightmapBaker(
                device: device, receivers: [receiver], resolution: 32, lights: [sun],
                punctualLights: Array(repeating: positioned, count: MeshLightmapBaker.maximumLightCount)
            )
        }
        let packed = try ProgressiveLightmapBaker(device: device, receivers: [receiver], resolution: 32, punctualLights: [positioned])
        #expect(try pixels(packed.snapshot().irradianceTexture).contains { $0.x > 0.2 })
    }

    private func pixels(_ texture: any MTLTexture) -> [SIMD4<Float>] {
        var values = [SIMD4<Float16>](repeating: .zero, count: texture.width * texture.height)
        values.withUnsafeMutableBytes {
            texture.getBytes(
                $0.baseAddress!, bytesPerRow: texture.width * MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return values.map { SIMD4<Float>($0) }
    }
}
