import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProgressiveLightmapRelightingTests {
    @Test("A packed bake relights like a fresh bake and restarts its deterministic sequence")
    func packedRelighting() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let diffuse = SIMD3<Float>(repeating: 0.7)
        let lower = try ProgressiveLightmapBaker.Surface(geometry: plane, diffuseAlbedo: diffuse)
        let upper = try ProgressiveLightmapBaker.Surface(
            geometry: plane, worldFromObject: simd_float4x4(columns: (
                SIMD4(1, 0, 0, 0), SIMD4(0, -1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0.25, 0, 1, 1)
            )), diffuseAlbedo: diffuse
        )
        let red = SIMD3<Float>(1, 0, 0)
        let green = SIMD3<Float>(0, 1, 0)
        let blue = SIMD3<Float>(0, 0, 1)
        let oldSun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(4, 0, 1), irradiance: red)
        let newSun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(-4, 0, -1), irradiance: green)
        let oldPoint = try ProgressiveLightmapBaker.PositionedLight(
            PunctualLight(kind: .point(), color: red, intensity: 0.25), position: SIMD3(0, 0, 0.5)
        )
        let newPoint = try ProgressiveLightmapBaker.PositionedLight(
            PunctualLight(kind: .point(range: 3), color: blue, intensity: 0.25), position: SIMD3(0.3, 0, 0.45)
        )
        let newSpot = try ProgressiveLightmapBaker.PositionedLight(
            PunctualLight(kind: .spot(innerConeAngle: 0.2, outerConeAngle: 0.7), color: green, intensity: 0.3),
            position: SIMD3(0, 0, 0.65)
        )
        let newSky = SIMD3<Float>(0, 0.2, 0.4)
        func prepare(
            _ light: MeshLightmapBaker.DirectionalLight, _ positioned: [ProgressiveLightmapBaker.PositionedLight],
            sky: SIMD3<Float>
        ) throws -> ProgressiveLightmapBaker {
            try ProgressiveLightmapBaker(
                device: device, receivers: [lower, upper], resolution: 32, lights: [light], punctualLights: positioned,
                skyRadiance: sky, maximumBounces: 1, seed: 37, padding: 2
            )
        }
        let relit = try prepare(oldSun, [oldPoint], sky: red * 0.2)
        try relit.accumulate(samples: 21)
        let retained = try relit.snapshot()
        let retainedPixels = pixels(retained.irradianceTexture)
        let bindings = relit.receiverBindings
        #expect(retainedPixels.contains { $0.x > 0.1 })
        #expect(retainedPixels.allSatisfy { $0.y == 0 && $0.z == 0 })

        try relit.replaceLighting(lights: [newSun], punctualLights: [newPoint, newSpot], skyRadiance: newSky)
        #expect(relit.samplesPerTexel == 0)
        let direct = try relit.snapshot()
        let directPixels = pixels(direct.irradianceTexture)
        let fresh = try prepare(newSun, [newPoint, newSpot], sky: newSky)
        #expect(try pixels(fresh.snapshot().irradianceTexture) == directPixels)
        #expect(directPixels.contains { $0.y > 0.1 } && directPixels.contains { $0.z > 0.1 })
        #expect(direct.coverageTexture === retained.coverageTexture)
        #expect(direct.irradianceTexture !== retained.irradianceTexture)
        #expect(relit.receiverBindings.map(\.sourceVertexIndices) == bindings.map(\.sourceVertexIndices))
        for (before, after) in zip(bindings, relit.receiverBindings) {
            #expect(before.geometry.indices == after.geometry.indices)
            #expect(before.geometry.vertices.map(\.secondaryTextureCoordinate) == after.geometry.vertices.map(\.secondaryTextureCoordinate))
        }

        try relit.accumulate(samples: 7)
        try relit.accumulate(samples: 25)
        try fresh.accumulate(samples: 32)
        let relitPixels = try pixels(relit.snapshot().irradianceTexture)
        #expect(try pixels(fresh.snapshot().irradianceTexture) == relitPixels)
        #expect(relitPixels != directPixels)
        #expect(relitPixels.allSatisfy { $0.x == 0 })
        #expect(pixels(retained.irradianceTexture) == retainedPixels)

        try relit.replaceLighting()
        #expect(relit.samplesPerTexel == 0)
        try relit.accumulate(samples: 12)
        let cleared = try relit.snapshot()
        #expect(cleared.coverageTexture === retained.coverageTexture)
        #expect(pixels(cleared.irradianceTexture).allSatisfy { $0.x == 0 && $0.y == 0 && $0.z == 0 })
        #expect(pixels(retained.irradianceTexture) == retainedPixels)
    }

    @Test("Rejected relighting preserves the direct result, accumulated samples and future sequence")
    func transactionalFailure() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0, 0.3, 0.1))
        let point = try ProgressiveLightmapBaker.PositionedLight(
            PunctualLight(kind: .point(), intensity: 0.2), position: SIMD3(0.3, 0, 1)
        )
        let panel = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 0.5, height: 0.5),
            worldFromObject: SceneTransform(translation: SIMD3(0.55, 0, 0.6), scale: SIMD3(1, -1, -1)).matrix,
            diffuseAlbedo: SIMD3(repeating: 0.8)
        )
        func prepare() throws -> ProgressiveLightmapBaker {
            try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 5, height: 5, textureCoordinateSet: .primary,
                lights: [sun], punctualLights: [point], surfaces: [panel], skyRadiance: SIMD3(0.2, 0.3, 0.1),
                maximumBounces: 1, seed: 16
            )
        }
        let baker = try prepare()
        try baker.accumulate(samples: 12)
        let retained = try baker.snapshot()
        let original = pixels(retained.irradianceTexture)
        let tooBright = try MeshLightmapBaker.DirectionalLight(
            directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(repeating: 70_000)
        )
        #expect(throws: ProgressiveLightmapBaker.Failure.transport) {
            try baker.replaceLighting(skyRadiance: SIMD3(0, .nan, 0))
        }
        #expect(throws: ProgressiveLightmapBaker.Failure.lightCount) {
            try baker.replaceLighting(lights: [sun], punctualLights: Array(repeating: point, count: MeshLightmapBaker.maximumLightCount))
        }
        #expect(throws: MeshLightmapBaker.Failure.irradianceRange) {
            try baker.replaceLighting(lights: [tooBright], skyRadiance: SIMD3(0, 0, 1))
        }
        #expect(baker.samplesPerTexel == 12)
        #expect(try pixels(baker.snapshot().irradianceTexture) == original)
        try baker.accumulate(samples: 20)
        let control = try prepare()
        try control.accumulate(samples: 32)
        #expect(try pixels(baker.snapshot().irradianceTexture) == pixels(control.snapshot().irradianceTexture))
        #expect(pixels(retained.irradianceTexture) == original)
    }

    @Test("Moving and removing a caster reproduces a fresh bake without replacing the receiver atlas",
          arguments: [false, true])
    func updateCasters(packed: Bool) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let panel = try MeshGeometry.plane(width: 0.64, height: 0.8)
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0.75, 0.5, 0.25))
        func caster(_ x: Float) throws -> ProgressiveLightmapBaker.Surface {
            try .init(
                geometry: panel,
                worldFromObject: SceneTransform(translation: SIMD3(x, 0, 0.45), scale: SIMD3(1, -1, -1)).matrix,
                emissiveRadiance: SIMD3(0.5, 0, 0)
            )
        }
        func prepare(_ casters: [ProgressiveLightmapBaker.Surface]) throws -> ProgressiveLightmapBaker {
            if packed {
                return try .init(
                    device: device, receivers: [receiver], resolution: 32, lights: [sun], surfaces: casters,
                    skyRadiance: SIMD3(repeating: 0.05), maximumBounces: 1, seed: 18, padding: 2
                )
            }
            return try .init(
                device: device, receiver: receiver, width: 13, height: 11, textureCoordinateSet: .primary,
                lights: [sun], surfaces: casters, skyRadiance: SIMD3(repeating: 0.05), maximumBounces: 1, seed: 18
            )
        }

        let bake = try prepare([caster(-0.5)])
        let bindings = bake.receiverBindings
        let initial = try bake.snapshot()
        let initialDirect = pixels(initial.irradianceTexture)
        try bake.accumulate(samples: 12)
        let retained = try bake.snapshot()
        let oldPixels = pixels(retained.irradianceTexture)

        let moved = try caster(0.5)
        try bake.replaceSurfaces([moved])
        #expect(bake.samplesPerTexel == 0)
        let direct = try bake.snapshot()
        let fresh = try prepare([moved])
        let movedDirect = pixels(direct.irradianceTexture)
        #expect(movedDirect != initialDirect)
        #expect(movedDirect == pixels(try fresh.snapshot().irradianceTexture))
        #expect(direct.coverageTexture === initial.coverageTexture)
        #expect(bake.receiverBindings.map(\.sourceVertexIndices) == bindings.map(\.sourceVertexIndices))
        #expect(bake.receiverBindings[0].geometry.indices == bindings[0].geometry.indices)
        #expect(bake.receiverBindings[0].geometry.vertices.map(\.secondaryTextureCoordinate)
                == bindings[0].geometry.vertices.map(\.secondaryTextureCoordinate))
        if !packed {
            let shadow = SIMD4<Float16>(0, 0, 0, 1)
            let lit = SIMD4<Float16>(0.75, 0.5, 0.25, 1)
            #expect(initialDirect[5 * 13 + 3] == shadow && initialDirect[5 * 13 + 9] == lit)
            #expect(movedDirect[5 * 13 + 3] == lit && movedDirect[5 * 13 + 9] == shadow)
        }

        try bake.accumulate(samples: 5)
        try bake.accumulate(samples: 7)
        try fresh.accumulate(samples: 12)
        let updated = try bake.snapshot()
        #expect(pixels(updated.irradianceTexture) == pixels(try fresh.snapshot().irradianceTexture))
        #expect(pixels(updated.irradianceTexture) != oldPixels)
        #expect(pixels(retained.irradianceTexture) == oldPixels)

        try bake.replaceSurfaces([])
        let unblocked = try prepare([])
        #expect(bake.samplesPerTexel == 0)
        #expect(pixels(try bake.snapshot().irradianceTexture) == pixels(try unblocked.snapshot().irradianceTexture))
    }

    @Test("Caster edits that expose unrepresentable irradiance or invalid geometry preserve the bake")
    func invalidCasterEdit() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        let cover = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, worldFromObject: SceneTransform(translation: SIMD3(0, 0, 1)).matrix
        )
        let invalid = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, worldFromObject: SceneTransform(scale: SIMD3(0, 1, 1)).matrix
        )
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(repeating: 70_000))
        func prepare() throws -> ProgressiveLightmapBaker {
            try .init(
                device: device, receiver: receiver, width: 9, height: 7, textureCoordinateSet: .primary,
                lights: [sun], surfaces: [cover], skyRadiance: SIMD3(repeating: 0.25),
                includesReceiverInTransport: false, maximumBounces: 1, seed: 18
            )
        }
        let bake = try prepare()
        try bake.accumulate(samples: 8)
        let original = try bake.snapshot()
        #expect(throws: MeshGeometry.MergeFailure.invalidTransform(part: 0)) { try bake.replaceSurfaces([invalid]) }
        #expect(throws: ProgressiveLightmapBaker.Failure.geometryLimit) {
            try bake.replaceSurfaces(Array(repeating: cover, count: ProgressiveLightmapBaker.maximumSurfaceCount))
        }
        #expect(throws: MeshLightmapBaker.Failure.irradianceRange) { try bake.replaceSurfaces([]) }
        #expect(bake.samplesPerTexel == 8)
        #expect(pixels(try bake.snapshot().irradianceTexture) == pixels(original.irradianceTexture))
        let control = try prepare()
        try control.accumulate(samples: 16)
        try bake.accumulate(samples: 8)
        #expect(pixels(try bake.snapshot().irradianceTexture) == pixels(try control.snapshot().irradianceTexture))

        try bake.replaceLighting(skyRadiance: SIMD3(repeating: 0.25))
        try bake.replaceSurfaces([])
        try bake.accumulate(samples: 2)
        let unblocked = try bake.snapshot()
        #expect(unblocked.coverageTexture === original.coverageTexture)
        #expect(pixels(unblocked.irradianceTexture).allSatisfy { $0 == SIMD4(Float16(Float.pi / 4), Float16(Float.pi / 4), Float16(Float.pi / 4), 1) })
    }

    @Test("Rolling preview blends moving shadows and light changes while cumulative bakes restart")
    func rollingPreview() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let red = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(1, 0, 0))
        let green = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0, 1, 0))
        let blocker = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 0.8, height: 0.8),
            worldFromObject: SceneTransform(translation: SIMD3(0, 0, 0.5)).matrix
        )
        let bake = try ProgressiveLightmapBaker(
            device: device, receiver: receiver, width: 7, height: 7, textureCoordinateSet: .primary, lights: [red]
        )
        try bake.accumulate(samples: 12)
        let first = try bake.updatePreview(samples: 4, blendWindow: 4)
        let original = pixels(first.irradianceTexture)
        #expect(original.allSatisfy { $0 == SIMD4(1, 0, 0, 1) })
        #expect(bake.samplesPerTexel == 12)

        try bake.replaceLighting(lights: [green])
        try bake.replaceSurfaces([blocker])
        #expect(bake.samplesPerTexel == 0)
        let blended = try bake.updatePreview(samples: 2, blendWindow: 4)
        let transitioning = pixels(blended.irradianceTexture)
        #expect(transitioning[24] == SIMD4(0.75, 0, 0, 1))
        #expect(transitioning[0] == SIMD4(0.75, 0.25, 0, 1))
        #expect(blended.coverageTexture === first.coverageTexture)
        #expect(pixels(first.irradianceTexture) == original)

        let immediate = try bake.updatePreview(blendWindow: 1)
        let current = try bake.snapshot()
        #expect(pixels(immediate.irradianceTexture) == pixels(current.irradianceTexture))
        #expect(pixels(immediate.irradianceTexture)[24] == SIMD4(0, 0, 0, 1))
        try bake.replaceSurfaces([])
        bake.resetPreview()
        let reset = try bake.updatePreview(blendWindow: 100)
        #expect(pixels(reset.irradianceTexture).allSatisfy { $0 == SIMD4(0, 1, 0, 1) })
    }

    @Test("Rolling preview samples match cumulative references and failures do not advance the sequence")
    func rollingSequence() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 2, height: 2))
        let blocker = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 1.5, height: 1.5),
            worldFromObject: SceneTransform(translation: SIMD3(0, 0, 0.5)).matrix
        )
        let bake = try ProgressiveLightmapBaker(
            device: device, receiver: receiver, width: 7, height: 7, textureCoordinateSet: .primary,
            surfaces: [blocker], skyRadiance: SIMD3(repeating: 0.25), maximumBounces: 0, seed: 27
        )
        let first = try bake.updatePreview()
        let initial = pixels(first.irradianceTexture)
        try bake.accumulate(samples: 1)
        #expect(pixels(try bake.snapshot().irradianceTexture) == initial)
        #expect(throws: ProgressiveLightmapBaker.Failure.previewWindow) { try bake.updatePreview(blendWindow: 0) }
        #expect(throws: ProgressiveLightmapBaker.Failure.sampleBudget) { try bake.updatePreview(samples: 0) }
        let second = try bake.updatePreview(blendWindow: 2)
        try bake.accumulate(samples: 1)
        let expected = try bake.snapshot()
        #expect(pixels(second.irradianceTexture) != initial)
        #expect(pixels(second.irradianceTexture) == pixels(expected.irradianceTexture))
        bake.resetPreview()
        #expect(bake.samplesPerTexel == 2)
        #expect(pixels(try bake.updatePreview().irradianceTexture) == initial)
    }

    @Test("Custom irradiance receives world positions and sample indices and publishes atomically")
    func customIrradiance() throws {
        enum SourceFailure: Error { case unavailable }

        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 3, height: 1),
            worldFromObject: SceneTransform(translation: SIMD3(2, 0, 3)).matrix
        )
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0.5, 0, 0))
        func prepare() throws -> ProgressiveLightmapBaker {
            try .init(device: device, receiver: receiver, width: 3, height: 1, textureCoordinateSet: .primary, lights: [sun])
        }
        func source(_ sample: ProgressiveLightmapBaker.IrradianceSample) -> SIMD3<Float> {
            SIMD3(Float(sample.worldPosition.x) + Float(sample.index), Float(sample.worldPosition.z) * 0.25,
                  Float(sample.worldNormal.z))
        }
        let bake = try prepare()
        try bake.accumulate(samples: 2, indirectIrradiance: source)
        let first = try bake.snapshot()
        let original = [SIMD4<Float16>(2, 0.75, 1, 1), SIMD4(3, 0.75, 1, 1), SIMD4(4, 0.75, 1, 1)]
        #expect(pixels(first.irradianceTexture) == original)
        #expect(throws: SourceFailure.unavailable) {
            try bake.accumulate(samples: 2) { sample in
                if sample.worldPosition.x > 2.5 { throw SourceFailure.unavailable }
                return source(sample)
            }
        }
        #expect(throws: ProgressiveLightmapBaker.Failure.customIrradiance) {
            try bake.accumulate(samples: 1) { _ in SIMD3(-1, 0, 0) }
        }
        #expect(throws: MeshLightmapBaker.Failure.irradianceRange) {
            try bake.accumulate(samples: 1) { _ in SIMD3(repeating: 1_000_000) }
        }
        #expect(bake.samplesPerTexel == 2)
        #expect(pixels(try bake.snapshot().irradianceTexture) == original)

        try bake.accumulate(samples: 2, indirectIrradiance: source)
        let oneBatch = try prepare()
        try oneBatch.accumulate(samples: 4, indirectIrradiance: source)
        let expected = [SIMD4<Float16>(3, 0.75, 1, 1), SIMD4(4, 0.75, 1, 1), SIMD4(5, 0.75, 1, 1)]
        #expect(pixels(try bake.snapshot().irradianceTexture) == expected)
        #expect(pixels(try oneBatch.snapshot().irradianceTexture) == expected)
        #expect(pixels(first.irradianceTexture) == original)
    }

    @Test("Halton irradiance recovers chart-edge texels and normalizes only covered samples", arguments: [0, 1])
    func haltonIrradiance(padding: Int) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let chart = try MeshGeometry(vertices: plane.vertices.map { vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal,
                textureCoordinate: SIMD2(0.1875 + 0.5 * vertex.textureCoordinate.x, vertex.textureCoordinate.y)
            )
        }, indices: plane.indices)
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: chart)
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0.25, 0, 0))
        func prepare() throws -> ProgressiveLightmapBaker {
            try .init(device: device, receiver: receiver, width: 2, height: 1, textureCoordinateSet: .primary,
                      lights: [sun], padding: padding)
        }
        func source(_ sample: ProgressiveLightmapBaker.IrradianceSample) -> SIMD3<Float> {
            SIMD3(Float(sample.worldPosition.x) + 2, Float(sample.worldPosition.y) + 2, Float(sample.index + 1))
        }
        func coverage(_ snapshot: MeshLightmapBaker) -> [UInt8] {
            var values: [UInt8] = [0, 0]
            values.withUnsafeMutableBytes {
                snapshot.coverageTexture.getBytes($0.baseAddress!, bytesPerRow: 2, from: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0)
            }
            return values
        }

        let split = try prepare()
        try split.accumulate(samples: 1, sampling: .halton, indirectIrradiance: source)
        let first = try split.snapshot()
        let old = pixels(first.irradianceTexture)
        #expect(coverage(first) == [255, padding == 0 ? 0 : 128])
        #expect(old[0].x == 1.5 && abs(Float(old[0].y) - 7 / 3) < 0.002 && old[0].z == 1 && old[0].w == 1)
        #expect(old[1] == (padding == 0 ? .zero : old[0]))

        #expect(throws: ProgressiveLightmapBaker.Failure.samplingMode) { try split.accumulate(samples: 1) }
        #expect(throws: ProgressiveLightmapBaker.Failure.customIrradiance) {
            try split.accumulate(samples: 1, sampling: .halton) { _ in SIMD3(-1, 0, 0) }
        }
        #expect(split.samplesPerTexel == 1)
        try split.accumulate(samples: 2, sampling: .halton, indirectIrradiance: source)
        let updated = try split.snapshot()
        let actual = pixels(updated.irradianceTexture)
        #expect(coverage(updated) == [255, 255])
        #expect(actual[0] == SIMD4(1.75, 2, 1.5, 1))
        #expect(actual[1].x == 3 && abs(Float(actual[1].y) - 25 / 9) < 0.002 && actual[1].z == 3 && actual[1].w == 1)
        #expect(pixels(first.irradianceTexture) == old && coverage(first) == [255, padding == 0 ? 0 : 128])

        let oneBatch = try prepare()
        try oneBatch.accumulate(samples: 3, sampling: .halton, indirectIrradiance: source)
        let combined = try oneBatch.snapshot()
        #expect(pixels(combined.irradianceTexture) == actual && coverage(combined) == [255, 255])
        split.reset()
        try split.accumulate(samples: 1, indirectIrradiance: source)
        #expect(throws: ProgressiveLightmapBaker.Failure.samplingMode) {
            try split.accumulate(samples: 1, sampling: .halton, indirectIrradiance: source)
        }
        #expect(try coverage(split.snapshot()) == [255, padding == 0 ? 0 : 128])
    }

    @Test("Moving and mirroring receivers preserves UV charts and samples the new world-space shadow", arguments: [false, true])
    func moveReceivers(packed: Bool) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: plane, diffuseAlbedo: SIMD3(repeating: 0.5))
        let caster = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 0.9, height: 1.4),
            worldFromObject: SceneTransform(translation: SIMD3(0, 0, 0.5)).matrix
        )
        let lamp = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0.5, 0.25, 0.125))
        let sky = SIMD3<Float>(repeating: 0.05)
        let size = packed ? 32 : 9
        let padding = packed ? 2 : 0
        let bake: ProgressiveLightmapBaker
        if packed {
            bake = try .init(device: device, receivers: [receiver], resolution: size, lights: [lamp],
                             skyRadiance: sky, maximumBounces: 1, seed: 10, padding: padding)
        } else {
            bake = try .init(device: device, receiver: receiver, width: size, height: size, textureCoordinateSet: .primary,
                             lights: [lamp], skyRadiance: sky, maximumBounces: 1, seed: 10, padding: padding)
        }
        try bake.replaceSurfaces([caster])
        let oldDirect = try bake.snapshot()
        let originalBinding = bake.receiverBindings[0]
        try bake.accumulate(samples: 8)
        let previous = try bake.snapshot()
        let previousPixels = pixels(previous.irradianceTexture)
        #expect(throws: ProgressiveLightmapBaker.Failure.receiverPlacementCount) { try bake.replaceReceiverPlacements([]) }
        let singular = SceneTransform(scale: SIMD3(0, 1, 1)).matrix
        #expect(throws: MeshGeometry.MergeFailure.invalidTransform(part: 0)) { try bake.replaceReceiverPlacements([singular]) }
        #expect(bake.samplesPerTexel == 8)
        #expect(pixels(try bake.snapshot().irradianceTexture) == previousPixels)

        let placement = SceneTransform(translation: SIMD3(0.65, 0, 0), scale: SIMD3(-1, 1, 1)).matrix
        try bake.replaceReceiverPlacements([placement])
        #expect(bake.samplesPerTexel == 0)
        let binding = bake.receiverBindings[0]
        #expect(binding.worldFromObject.columns.3 == placement.columns.3)
        #expect(binding.worldFromObject.columns.0 == placement.columns.0)
        #expect(binding.sourceVertexIndices == originalBinding.sourceVertexIndices)
        #expect(binding.geometry.indices == originalBinding.geometry.indices)
        #expect(binding.geometry.vertices.map(\.secondaryTextureCoordinate)
                == originalBinding.geometry.vertices.map(\.secondaryTextureCoordinate))

        let referenceReceiver = try ProgressiveLightmapBaker.Surface(
            geometry: originalBinding.geometry, worldFromObject: placement, diffuseAlbedo: receiver.diffuseAlbedo
        )
        let reference = try ProgressiveLightmapBaker(
            device: device, receiver: referenceReceiver, width: size, height: size, textureCoordinateSet: bake.textureCoordinateSet,
            lights: [lamp], surfaces: [caster], skyRadiance: sky, maximumBounces: 1, seed: 10, padding: padding
        )
        let moved = try bake.snapshot()
        let direct = pixels(moved.irradianceTexture)
        #expect(direct != pixels(oldDirect.irradianceTexture))
        #expect(direct == pixels(try reference.snapshot().irradianceTexture))
        #expect(moved.coverageTexture === previous.coverageTexture)
        #expect(direct.contains(SIMD4(0, 0, 0, 1)))
        #expect(direct.contains(SIMD4(0.5, 0.25, 0.125, 1)))
        try bake.accumulate(samples: 8)
        try reference.accumulate(samples: 8)
        #expect(pixels(try bake.snapshot().irradianceTexture) == pixels(try reference.snapshot().irradianceTexture))
        #expect(pixels(previous.irradianceTexture) == previousPixels)
    }

    @Test("Metal irradiance samples world position, normal, sample index and a texture", arguments: [
        ProgressiveLightmapBaker.IrradianceSampling.texelCenter, .halton,
    ])
    func metalIrradiance(sampling: ProgressiveLightmapBaker.IrradianceSampling) throws {
        let gpu = try PostProcessFixture()
        let receiver = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 2, height: 2),
            worldFromObject: SceneTransform(translation: SIMD3(2, 0, 3)).matrix
        )
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(0.5, 0, 0))
        func prepare() throws -> ProgressiveLightmapBaker {
            try .init(device: gpu.device, receiver: receiver, width: 4, height: 3, textureCoordinateSet: .primary, lights: [sun])
        }
        let ink = SIMD4<Float>(0.25, 0.5, 0.125, 1)
        let map = try gpu.color(width: 1, height: 1) { _, _ in ink }
        let graph = try BufferShaderGraph(fieldCount: 3) { builder in
            let position = builder.elementField(0)
            let normal = builder.elementField(1)
            let texture = builder.texture("Ink", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let color = texture.sample(builder.constant(SIMD2<Float>(repeating: 0.5)))
            let gain = builder.uniform("Gain", Float(1))
            return [2: builder.vector((position.x + position.w) * gain + color.x, position.z * 0.25 + color.y,
                                      normal.z + color.z, builder.constant(Float(0)))]
        }
        let kernel = try ShaderBufferKernel(graph: graph, device: gpu.device)
        let wrongLayout = try ShaderBufferKernel(graph: BufferShaderGraph(fieldCount: 1) { builder in
            [0: builder.elementField(0)]
        }, device: gpu.device)
        let uniforms = try ShaderUniforms(graph: graph)
        let baker = try prepare()
        try baker.accumulate(samples: 1, sampling: sampling, irradianceKernel: kernel,
                             uniforms: uniforms, textures: ["Ink": map], commandQueue: gpu.queue)
        let first = try baker.snapshot()
        let initial = pixels(first.irradianceTexture)
        #expect(throws: ProgressiveLightmapBaker.Failure.irradianceKernelLayout) {
            try baker.accumulate(samples: 1, sampling: sampling, irradianceKernel: wrongLayout, commandQueue: gpu.queue)
        }
        #expect(throws: ShaderComputeFailure.missingTexture("Ink")) {
            try baker.accumulate(samples: 1, sampling: sampling, irradianceKernel: kernel, commandQueue: gpu.queue)
        }
        try uniforms.set("Gain", to: Float.greatestFiniteMagnitude)
        #expect(throws: ProgressiveLightmapBaker.Failure.customIrradiance) {
            try baker.accumulate(samples: 1, sampling: sampling, irradianceKernel: kernel,
                                 uniforms: uniforms, textures: ["Ink": map], commandQueue: gpu.queue)
        }
        #expect(baker.samplesPerTexel == 1)
        #expect(pixels(try baker.snapshot().irradianceTexture) == initial)

        try uniforms.set("Gain", to: Float(1))
        try baker.accumulate(samples: 2, sampling: sampling, irradianceKernel: kernel,
                             uniforms: uniforms, textures: ["Ink": map], commandQueue: gpu.queue)
        let reference = try prepare()
        try reference.accumulate(samples: 3, sampling: sampling) { sample in
            SIMD3(Float(sample.worldPosition.x) + Float(sample.index) + ink.x, Float(sample.worldPosition.z) * 0.25 + ink.y,
                  Float(sample.worldNormal.z) + ink.z)
        }
        let actual = pixels(try baker.snapshot().irradianceTexture)
        let expected = pixels(try reference.snapshot().irradianceTexture)
        #expect(actual.contains { $0.x > 3 && $0.y == 1.25 && $0.z == 1.125 })
        #expect(zip(actual, expected).allSatisfy { simd_distance(SIMD4<Float>($0), SIMD4<Float>($1)) < 0.003 })
        #expect(pixels(first.irradianceTexture) == initial)
    }

    private func pixels(_ texture: any MTLTexture) -> [SIMD4<Float16>] {
        var values = [SIMD4<Float16>](repeating: .zero, count: texture.width * texture.height)
        values.withUnsafeMutableBytes {
            texture.getBytes(
                $0.baseAddress!, bytesPerRow: texture.width * MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return values
    }
}
