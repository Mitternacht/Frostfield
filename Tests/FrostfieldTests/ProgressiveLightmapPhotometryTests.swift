import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProgressiveLightmapPhotometryTests {
    @Test("IES direct light follows the source angles, seam interpolation and resolved node roll")
    func profileAngles() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let profile = try IESProfile(string: IESFixture.text(
            vertical: [0, 45, 90, 180], horizontal: [0, 90, 180, 270, 360]
        ) { vertical, horizontal in
            if vertical == 0 { return 100 }
            guard vertical == 45 else { return 0 }
            switch horizontal {
            case 90: return 60
            case 180: return 20
            case 270: return 40
            default: return 80
            }
        })
        let light = try PunctualLight(kind: .point(), intensity: 8, iesProfile: profile)
        let position = SIMD3<Float>(0, 0, 2)
        let unrolled = try ProgressiveLightmapBaker.PositionedLight(light, position: position)
        #expect(unrolled.right == SIMD3(1, 0, 0))
        let baker = try ProgressiveLightmapBaker(
            device: device, receiver: .init(geometry: .plane(width: 6, height: 2)), width: 3, height: 1,
            textureCoordinateSet: .primary, punctualLights: [unrolled], maximumBounces: 0
        )
        let baseline = try baker.snapshot()
        let original = pixels(baseline.irradianceTexture)
        let factor = Float(1 / Double(2).squareRoot())
        expectColor(original[0], SIMD3(repeating: 0.2 * factor))
        expectColor(original[1], SIMD3(repeating: 2))
        expectColor(original[2], SIMD3(repeating: 0.8 * factor))

        let rolled = try positioned(light, at: position, roll: .pi / 2)
        #expect(simd_distance(rolled.right, SIMD3(0, 1, 0)) < 0.001)
        try baker.replaceLighting(punctualLights: [rolled])
        let rotated = try pixels(baker.snapshot().irradianceTexture)
        expectColor(rotated[0], SIMD3(repeating: 0.6 * factor))
        expectColor(rotated[1], SIMD3(repeating: 2))
        expectColor(rotated[2], SIMD3(repeating: 0.4 * factor))
        #expect(pixels(baseline.irradianceTexture) == original)

        // The right texel is just across the wrap from 360° to 0°.
        let shifted = try ProgressiveLightmapBaker.PositionedLight(light, position: SIMD3(0, 0.05, 2))
        try baker.replaceLighting(punctualLights: [shifted])
        let local = SIMD3<Float>(2, -0.05, -2)
        let distanceSquared = simd_length_squared(local)
        let falloff = 8 / distanceSquared * (2 / distanceSquared.squareRoot())
        let reference = profile.intensity(toward: local) / profile.maximumIntensity
        try expectColor(pixels(baker.snapshot().irradianceTexture)[2], SIMD3(repeating: falloff * reference))
    }

    @Test("Spot cookies keep their image orientation and the renderer's common-size bilinear grid")
    func cookieProjection() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let red = SIMD3<Float>(1, 0, 0), green = SIMD3<Float>(0, 1, 0), blue = SIMD3<Float>(0, 0, 1)
        let white = SIMD3<Float>(repeating: 1)
        let cookie = try LightCookie(width: 2, height: 2, transmittance: [red, green, blue, white])
        let kind = PunctualLight.Kind.spot(innerConeAngle: 0.75, outerConeAngle: .pi / 4)
        let light = try PunctualLight(kind: kind, intensity: 4, cookie: cookie)
        let position = SIMD3<Float>(0, 0, 2)
        let source = try ProgressiveLightmapBaker.PositionedLight(light, position: position)
        let baker = try ProgressiveLightmapBaker(
            device: device, receiver: .init(geometry: .plane(width: 3, height: 3)), width: 3, height: 3,
            textureCoordinateSet: .primary, punctualLights: [source], maximumBounces: 0
        )
        let original = try pixels(baker.snapshot().irradianceTexture)
        let cornerScale = Float(8 / (6 * Double(6).squareRoot()))
        expectColor(original[0], red * cornerScale)
        expectColor(original[2], green * cornerScale)
        expectColor(original[6], blue * cornerScale)
        expectColor(original[8], white * cornerScale)
        expectColor(original[4], SIMD3(repeating: 0.5))

        try baker.replaceLighting(punctualLights: [positioned(light, at: position, roll: .pi / 2)])
        let rotated = try pixels(baker.snapshot().irradianceTexture)
        expectColor(rotated[0], green * cornerScale)
        expectColor(rotated[2], white * cornerScale)
        expectColor(rotated[6], red * cornerScale)
        expectColor(rotated[8], blue * cornerScale)

        // A second 4×4 cookie sets the grid size for the renderer's array.
        // The small cookie's corner sample at (¼, ¼) now blends ⅛ toward
        // each adjacent source texel after resampling to the larger grid.
        let bigger = try LightCookie(width: 4, height: 4, transmittance: Array(repeating: white, count: 16))
        let zero = try ProgressiveLightmapBaker.PositionedLight(
            PunctualLight(kind: kind, intensity: 0, cookie: bigger), position: position
        )
        try baker.replaceLighting(punctualLights: [source, zero])
        let mixed = try pixels(baker.snapshot().irradianceTexture)
        expectColor(mixed[0], SIMD3(0.78125, 0.125, 0.125) * cornerScale)
        expectColor(mixed[4], SIMD3(repeating: 0.5))
    }

    @Test("An IES profile and linear cookie multiply a spot's diffuse bounce after relighting")
    func bouncedPhotometry() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let panel = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 4, height: 4),
            worldFromObject: SceneTransform(translation: SIMD3(0, 0, 1), scale: SIMD3(1, -1, -1)).matrix,
            diffuseAlbedo: SIMD3(repeating: 0.8)
        )
        let kind = PunctualLight.Kind.spot(range: 2, innerConeAngle: 0.25, outerConeAngle: 0.8)
        let position = SIMD3<Float>(0, 0, 0.5), direction = SIMD3<Float>(0, 0, 1)
        let unfiltered = try ProgressiveLightmapBaker.PositionedLight(PunctualLight(kind: kind), position: position, direction: direction)
        let baker = try ProgressiveLightmapBaker(
            device: device, receiver: receiver, width: 1, height: 1, textureCoordinateSet: .primary,
            punctualLights: [unfiltered], surfaces: [panel], includesReceiverInTransport: false, maximumBounces: 1, seed: 9
        )
        #expect(try pixels(baker.snapshot().irradianceTexture)[0] == SIMD4(0, 0, 0, 1))
        for _ in 0..<4 { try baker.accumulate(samples: 256) }
        let baseline = try baker.snapshot()
        let original = pixels(baseline.irradianceTexture)[0]
        #expect(original.x > 0.1)
        #expect(original.x == original.y && original.y == original.z)

        func filteredLight() throws -> ProgressiveLightmapBaker.PositionedLight {
            // The spot emits only within 46°, where the IES table is at ¼ peak.
            let profile = try IESProfile(string: IESFixture.text(vertical: [0, 90, 180], horizontal: [0]) { vertical, _ in
                vertical == 180 ? 100 : 25
            })
            let cookie = try LightCookie(width: 1, height: 1, transmittance: [SIMD3(0.25, 0.5, 0)])
            return try .init(PunctualLight(kind: kind, iesProfile: profile, cookie: cookie), position: position, direction: direction)
        }
        try baker.replaceLighting(punctualLights: [filteredLight()])
        #expect(baker.samplesPerTexel == 0)
        #expect(try pixels(baker.snapshot().irradianceTexture)[0] == SIMD4(0, 0, 0, 1))
        for _ in 0..<8 { try baker.accumulate(samples: 128) }
        let filtered = try pixels(baker.snapshot().irradianceTexture)[0]
        #expect(filtered == SIMD4(original.x / 16, original.y / 8, 0, 1))
        #expect(pixels(baseline.irradianceTexture)[0] == original)
    }

    private func positioned(
        _ light: PunctualLight, at position: SIMD3<Float>, roll: Float
    ) throws -> ProgressiveLightmapBaker.PositionedLight {
        let pose = SceneTransform(translation: position, rotation: simd_quatf(angle: roll, axis: SIMD3(0, 0, 1)))
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20)
        let graph = SceneGraph(roots: [SceneNode(transform: pose, light: light)], camera: camera)
        let resolved = try #require(try graph.resolve().first?.light)
        return try .init(resolved)
    }

    private func expectColor(_ actual: SIMD4<Float>, _ expected: SIMD3<Float>) {
        #expect(simd_distance(actual, SIMD4(expected, 1)) < 0.001)
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
