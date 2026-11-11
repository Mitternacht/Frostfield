import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProgressiveLightmapRectangleTests {
    private typealias Bake = ProgressiveLightmapBaker

    @Test("Parallel rectangle irradiance follows authored area, distance and luminance")
    func analyticIrradiance() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let color = SIMD3<Float>(0.25, 0.5, 1)
        for (width, distance) in [(1.0, 2.0), (2, 2), (2, 3)] {
            let light = try Bake.PositionedLight(
                PunctualLight(kind: .rectangle(width: Float(width), height: 1), color: color, intensity: 2),
                position: SIMD3(0, 0, Float(distance))
            )
            let actual = try pixel(makeBake(device: device, lights: [light]).snapshot().irradianceTexture)
            let expected = color * Float(2 * parallelIntegral(width: width, height: 1, distance: distance))
            #expect(simd_distance(actual, SIMD4(expected, 1)) < 0.003)
        }
    }

    @Test("A resolved rectangle keeps roll and authored dimensions under node scale and emits on one side")
    func resolvedPose() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let light = try PunctualLight(kind: .rectangle(width: 3, height: 1))
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        func render(roll: Float, depthScale: Float) throws -> Float {
            let node = SceneNode(transform: .init(
                translation: SIMD3(1.2, 0, 2), rotation: simd_quatf(angle: roll, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(2, 0.25, depthScale)
            ), light: light)
            let graph = SceneGraph(roots: [node], camera: camera)
            let resolved = try #require(try graph.resolve().first?.light)
            return try pixel(makeBake(device: device, lights: [.init(resolved)]).snapshot().irradianceTexture).x
        }
        let unrolled = try render(roll: 0, depthScale: 3)
        let rolled = try render(roll: .pi / 2, depthScale: 3)
        #expect(abs(unrolled - Float(parallelIntegral(width: 3, height: 1, distance: 2, offset: 1.2))) < 0.002)
        #expect(abs(rolled - Float(parallelIntegral(width: 1, height: 3, distance: 2, offset: 1.2))) < 0.002)
        #expect(unrolled - rolled > 0.08)
        #expect(try render(roll: .pi / 2, depthScale: -3) == 0)
        let coplanar = try Bake.PositionedLight(light, position: SIMD3(0.4, 0, 0))
        #expect(try pixel(makeBake(device: device, lights: [coplanar]).snapshot().irradianceTexture).x == 0)
    }

    @Test("A symmetric half blocker halves irradiance and geometry beyond the light does not shadow")
    func finiteShadows() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let light = try rectangle(at: 2)
        func render(_ blockers: [Bake.Surface]) throws -> Float {
            try pixel(makeBake(device: device, lights: [light], surfaces: blockers).snapshot().irradianceTexture).x
        }
        func blocker(x: Float, z: Float) throws -> Bake.Surface {
            var transform = matrix_identity_float4x4
            transform.columns.3 = SIMD4(x, 0, z, 1)
            return try Bake.Surface(geometry: .plane(width: 2, height: 2), worldFromObject: transform)
        }
        let visible = try render([])
        #expect(visible > 0.4)
        #expect(try render([blocker(x: 1, z: 1)]) == visible / 2)
        #expect(try render([blocker(x: -1, z: 1)]) == visible / 2)
        #expect(try render([blocker(x: 0, z: 1)]) == 0)
        #expect(try render([blocker(x: 0, z: 3)]) == visible)
    }

    @Test("Rectangle lighting reaches diffuse hits and relights transactionally")
    func bounceAndRelight() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let panel = try Bake.Surface(
            geometry: .plane(width: 4, height: 4), worldFromObject: simd_float4x4(columns: (
                SIMD4(1, 0, 0, 0), SIMD4(0, -1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 0, 1, 1)
            )), diffuseAlbedo: SIMD3(0.8, 0.25, 0)
        )
        let downward = try rectangle(at: 0.5)
        let upward = try rectangle(at: 0.5, direction: SIMD3(0, 0, 1))
        let relit = try makeBake(device: device, lights: [downward], surfaces: [panel], bounces: 1)
        try relit.accumulate(samples: 16)
        let retained = try relit.snapshot()
        let original = pixel(retained.irradianceTexture)
        #expect(original.x > 2 && original.x == original.z)

        try relit.replaceLighting(punctualLights: [upward])
        #expect(relit.samplesPerTexel == 0)
        #expect(try pixel(relit.snapshot().irradianceTexture) == SIMD4(0, 0, 0, 1))
        let fresh = try makeBake(device: device, lights: [upward], surfaces: [panel], bounces: 1)
        let directOnly = try makeBake(device: device, lights: [upward], surfaces: [panel])
        try directOnly.accumulate(samples: 256)
        #expect(try pixel(directOnly.snapshot().irradianceTexture) == SIMD4(0, 0, 0, 1))

        try relit.accumulate(samples: 67)
        let beforeFailure = try pixel(relit.snapshot().irradianceTexture)
        let tooBright = try rectangle(at: 0.5, intensity: 1_000_000)
        #expect(throws: MeshLightmapBaker.Failure.irradianceRange) {
            try relit.replaceLighting(punctualLights: [tooBright])
        }
        #expect(relit.samplesPerTexel == 67)
        #expect(try pixel(relit.snapshot().irradianceTexture) == beforeFailure)
        try relit.accumulate(samples: 189)
        try fresh.accumulate(samples: 256)
        let result = try relit.snapshot()
        let bounced = pixel(result.irradianceTexture)
        #expect((0.2...1.1).contains(bounced.x))
        #expect(abs(bounced.x * (0.25 / 0.8) - bounced.y) < 0.001)
        #expect(bounced.z == 0 && bounced.w == 1)
        #expect(try pixel(fresh.snapshot().irradianceTexture) == bounced)
        #expect(result.coverageTexture === retained.coverageTexture)
        #expect(pixel(retained.irradianceTexture) == original)
    }

    /// Closed form for the irradiance of a unit-radiance rectangle above a parallel differential area.
    private func parallelIntegral(width: Double, height: Double, distance: Double, offset: Double = 0) -> Double {
        func corner(_ x: Double, _ y: Double) -> Double {
            let rx = (x * x + distance * distance).squareRoot()
            let ry = (y * y + distance * distance).squareRoot()
            return x / rx * atan(y / rx) + y / ry * atan(x / ry)
        }
        // Opposite y corners are symmetric and cancel the antiderivative's factor of 1/2.
        return corner(offset + width / 2, height / 2) - corner(offset - width / 2, height / 2)
    }

    private func rectangle(at depth: Float, direction: SIMD3<Float> = SIMD3(0, 0, -1), intensity: Float = 1) throws -> Bake.PositionedLight {
        try Bake.PositionedLight(
            PunctualLight(kind: .rectangle(width: 2, height: 1), intensity: intensity),
            position: SIMD3(0, 0, depth), direction: direction
        )
    }

    private func makeBake(
        device: any MTLDevice, lights: [Bake.PositionedLight], surfaces: [Bake.Surface] = [], bounces: Int = 0
    ) throws -> Bake {
        try Bake(
            device: device, receiver: .init(geometry: .plane(width: 2, height: 2)), width: 1, height: 1,
            textureCoordinateSet: .primary, punctualLights: lights, surfaces: surfaces,
            includesReceiverInTransport: false, maximumBounces: bounces, seed: 9
        )
    }

    private func pixel(_ texture: any MTLTexture) -> SIMD4<Float> {
        var value = SIMD4<Float16>.zero
        withUnsafeMutableBytes(of: &value) {
            texture.getBytes(
                $0.baseAddress!, bytesPerRow: $0.count, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0
            )
        }
        return SIMD4<Float>(value)
    }
}
