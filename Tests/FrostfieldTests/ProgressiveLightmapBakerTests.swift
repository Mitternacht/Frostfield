import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProgressiveLightmapBakerTests {
    @Test("Split batches reproduce one batch, and published atlases stay unchanged")
    func deterministicBatches() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try receiver(atlasInset: 0.25)
        let bounce = try redPanel(width: 1.2)
        let lights = [try diagonalLight()]
        func makeBaker() throws -> ProgressiveLightmapBaker {
            try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 8, height: 8,
                lights: lights, surfaces: [bounce], includesReceiverInTransport: false,
                maximumBounces: 1, seed: 413, padding: 1
            )
        }
        let split = try makeBaker()
        let direct = try split.snapshot()
        try split.accumulate(samples: 16)
        let preview = try split.snapshot()
        let previewPixels = pixels(preview.irradianceTexture)
        try split.accumulate(samples: 48)
        let final = try split.snapshot()
        let finalPixels = pixels(final.irradianceTexture)
        #expect(split.samplesPerTexel == 64)
        #expect(final.irradianceTexture !== preview.irradianceTexture)
        #expect(pixels(preview.irradianceTexture) == previewPixels)
        #expect(previewPixels != finalPixels)
        #expect(finalPixels[27].x > 0.1)
        #expect(pixels(direct.irradianceTexture)[27] == SIMD4(0, 0, 0, 1))
        let coverage = bytes(final.coverageTexture)
        #expect(coverage[0] == 0 && finalPixels[0] == .zero)
        #expect(coverage[17] == 128 && coverage[18] == 255 && finalPixels[17] == finalPixels[18])

        let single = try makeBaker()
        try single.accumulate(samples: 64)
        #expect(try pixels(single.snapshot().irradianceTexture) == finalPixels)
        split.reset()
        #expect(split.samplesPerTexel == 0)
        try split.accumulate(samples: 64)
        #expect(try pixels(split.snapshot().irradianceTexture) == finalPixels)
    }

    @Test("White directional light produces a red diffuse bounce, and opaque geometry blocks it")
    func coloredBounceAndOcclusion() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try receiver()
        let panel = try redPanel(width: 12)
        let blocker = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 12, height: 12),
            worldFromObject: SceneTransform(translation: SIMD3(0, 0, 0.2)).matrix
        )
        let light = try diagonalLight()
        func bake(_ surfaces: [ProgressiveLightmapBaker.Surface], bounces: Int) throws -> SIMD4<Float> {
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 1, height: 1, lights: [light],
                surfaces: surfaces, includesReceiverInTransport: false, maximumBounces: bounces, seed: 9
            )
            try baker.accumulate(samples: 256)
            try baker.accumulate(samples: 256)
            return try #require(try pixels(baker.snapshot().irradianceTexture).first)
        }
        let directOnly = try bake([panel], bounces: 0)
        let reflected = try bake([panel], bounces: 1)
        let occluded = try bake([panel, blocker], bounces: 1)
        #expect(directOnly == SIMD4(0, 0, 0, 1))
        // The large panel covers nearly all projected solid angle. Its outgoing
        // radiance approaches 0.8 / (√2 × π) and its received irradiance 0.8 / √2.
        #expect((0.51...0.57).contains(reflected.x))
        #expect(reflected.y == 0 && reflected.z == 0 && reflected.w == 1)
        #expect(occluded.x < 0.03 && occluded.w == 1)
    }

    @Test("A second diffuse reflection transports light around a directional shadow")
    func twoReflections() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try receiver()
        let upper = try redPanel(width: 2)
        let lower = try ProgressiveLightmapBaker.Surface(
            geometry: .plane(width: 16, height: 16),
            worldFromObject: SceneTransform(translation: SIMD3(0, 0, -1)).matrix,
            diffuseAlbedo: SIMD3(repeating: 0.75)
        )
        let light = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(repeating: 1))
        func bake(bounces: Int) throws -> SIMD4<Float> {
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 1, height: 1,
                lights: [light], surfaces: [upper, lower], maximumBounces: bounces, seed: 82
            )
            try baker.accumulate(samples: 256)
            try baker.accumulate(samples: 256)
            return try #require(try pixels(baker.snapshot().irradianceTexture).first)
        }
        #expect(try bake(bounces: 1) == SIMD4(0, 0, 0, 1))
        let bounced = try bake(bounces: 2)
        #expect(bounced.x > 0.12 && bounced.x < 0.6)
        #expect(bounced.y == 0 && bounced.z == 0 && bounced.w == 1)
    }

    @Test("An unoccluded constant sky integrates to π times radiance")
    func constantSky() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let sky = SIMD3<Float>(2, 0.25, 1.5)
        let baker = try ProgressiveLightmapBaker(
            device: device, receiver: receiver(), width: 4, height: 4, skyRadiance: sky, maximumBounces: 0
        )
        try baker.accumulate(samples: 1)
        let map = try baker.snapshot()
        #expect(pixels(map.irradianceTexture).allSatisfy { simd_distance($0, SIMD4(sky * .pi, 1)) < 0.003 })
    }

    @Test("Front-facing emissive geometry contributes without a diffuse reflection")
    func emissionVisibility() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try MeshGeometry.plane(width: 16, height: 16)
        let emission = SIMD3<Float>(0, 0.5, 1)
        let front = try ProgressiveLightmapBaker.Surface(geometry: geometry, worldFromObject: facingDown, emissiveRadiance: emission)
        let back = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, worldFromObject: SceneTransform(translation: SIMD3(0, 0, 1)).matrix, emissiveRadiance: emission
        )
        func bake(_ surface: ProgressiveLightmapBaker.Surface) throws -> SIMD4<Float> {
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver(), width: 1, height: 1, surfaces: [surface], maximumBounces: 0
            )
            try baker.accumulate(samples: 256)
            return try #require(try pixels(baker.snapshot().irradianceTexture).first)
        }
        #expect(try bake(back) == SIMD4(0, 0, 0, 1))
        let radiance = try bake(front)
        #expect(radiance.x == 0 && radiance.w == 1)
        #expect((2.95...3.15).contains(radiance.z) && abs(radiance.y * 2 - radiance.z) < 0.01)
    }

    @Test("Vertex RGB interpolates at diffuse hits and multiplies a transport texture")
    func vertexColoredBounce() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 14, height: 0.5))
        let panel = try MeshGeometry.plane(width: 24, height: 24)
        let left = SIMD4<Float>(0.25, 0.5, 0.75, 0)
        let colored = try MeshGeometry(vertices: panel.vertices.map { vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal,
                textureCoordinate: SIMD2(0.25 + vertex.textureCoordinate.x * 0.5, vertex.textureCoordinate.y),
                color: vertex.textureCoordinate.x == 0 ? left : nil
            )
        }, indices: panel.indices)
        let white = SIMD4<Float16>(repeating: 1)
        let gradient = try colorTexture(device, format: .rgba16Float,
                                        texels: [SIMD4<Float16>(left), white, SIMD4<Float16>(left), white])
        let tint = SIMD4<Float16>(0.5, 0.25, 0.125, 1)
        let constant = try colorTexture(device, format: .rgba16Float, texels: Array(repeating: tint, count: 4))
        let gradientMap = try ProgressiveLightmapBaker.TextureSnapshot(.init(texture: gradient))
        let tintMap = try ProgressiveLightmapBaker.TextureSnapshot(.init(texture: constant))
        let sun = try diagonalLight()
        func bake(_ vertexColors: Bool, _ map: ProgressiveLightmapBaker.TextureSnapshot? = nil) throws -> [SIMD4<Float>] {
            let surface = try ProgressiveLightmapBaker.Surface(
                geometry: colored, worldFromObject: facingDown, diffuseAlbedo: SIMD3(repeating: 0.75),
                diffuseTexture: map, usesVertexColors: vertexColors
            )
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver, width: 7, height: 3, textureCoordinateSet: .primary,
                lights: [sun], surfaces: [surface], includesReceiverInTransport: false, maximumBounces: 1, seed: 43
            )
            try baker.accumulate(samples: 256)
            return try pixels(baker.snapshot().irradianceTexture)
        }
        let untreated = try bake(false)
        let vertices = try bake(true)
        let texture = try bake(false, gradientMap)
        let multiplied = try bake(true, tintMap)
        #expect(untreated.allSatisfy { $0.x == $0.y && $0.y == $0.z && $0.x > 0.4 })
        #expect(vertices[0].x < vertices[6].x - 0.1)
        for index in vertices.indices {
            #expect(simd_distance(vertices[index], texture[index]) < 0.002)
            #expect(simd_distance(multiplied[index], vertices[index] * SIMD4<Float>(tint)) < 0.002)
        }
    }

    @Test("Diffuse bounce samples linear color at each UV1 hit with top-left coordinates and addressing")
    func texturedDiffuseBounce() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let texture = try colorTexture(device, format: .rgba8Unorm_srgb, texels: [
            SIMD4<UInt8>(128, 0, 0, 0), SIMD4<UInt8>(0, 255, 0, 0),
            SIMD4<UInt8>(0, 0, 255, 0), SIMD4<UInt8>(255, 255, 255, 0),
        ])
        let receivingPlane = try MeshGeometry.plane(width: 8, height: 0.5)
        let receivingGeometry = try MeshGeometry(vertices: receivingPlane.vertices.map {
            MeshVertex(position: $0.position, normal: $0.normal, secondaryTextureCoordinate: $0.textureCoordinate)
        }, indices: receivingPlane.indices)
        let receivingSurface = try ProgressiveLightmapBaker.Surface(geometry: receivingGeometry)
        let panel = try MeshGeometry.plane(width: 24, height: 24)
        let panelGeometry = try MeshGeometry(vertices: panel.vertices.map {
            MeshVertex(
                position: $0.position, normal: $0.normal, textureCoordinate: SIMD2(0.25, 0.75),
                secondaryTextureCoordinate: SIMD2($0.textureCoordinate.x, 0.25)
            )
        }, indices: panel.indices)
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(1, 0))
        let light = try diagonalLight()
        func bake(_ mode: PhysicallyBasedMaterial.TextureSampling.Addressing) throws -> [SIMD4<Float>] {
            let map = try ProgressiveLightmapBaker.TextureSnapshot(.init(
                texture: texture, textureCoordinateSet: .secondary, transform: transform,
                sampling: .init(addressU: mode, addressV: .clampToEdge, magnification: .nearest)
            ))
            let surface = try ProgressiveLightmapBaker.Surface(
                geometry: panelGeometry, worldFromObject: facingDown, diffuseAlbedo: SIMD3(0.5, 1, 1),
                diffuseTexture: map
            )
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receivingSurface, width: 2, height: 1, lights: [light], surfaces: [surface],
                includesReceiverInTransport: false, maximumBounces: 1, seed: 9
            )
            for _ in 0..<4 { try baker.accumulate(samples: 256) }
            return try pixels(baker.snapshot().irradianceTexture)
        }
        let repeated = try bake(.repeat)
        // A near-complete reflected hemisphere approaches the light's cosine
        // factor × the textured albedo: 0.216 sRGB-decoded red × 0.5 / √2.
        #expect((0.068...0.077).contains(repeated[0].x))
        #expect(repeated[0].y < 0.07 && repeated[0].z == 0)
        #expect(repeated[1].x < 0.012 && repeated[1].y > 0.62 && repeated[1].z == 0)
        let mirrored = try bake(.mirroredRepeat)
        #expect(mirrored[0].x < 0.012 && mirrored[0].y > 0.62 && mirrored[0].z == 0)
        #expect((0.068...0.077).contains(mirrored[1].x))
        #expect(mirrored[1].y < 0.07 && mirrored[1].z == 0)
    }

    @Test("Mapped HDR emission uses the level-zero bilinear footprint and an immutable texel snapshot")
    func texturedEmission() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let texture = try colorTexture(device, format: .rgba16Float, mipmapped: true, texels: [
            SIMD4<Float16>(2, 0, 0, 0), SIMD4<Float16>(0, 4, 0, 1),
            SIMD4<Float16>(0, 0, 16, 1), SIMD4<Float16>(0, 0, 32, 1),
        ])
        let base = try MeshGeometry.plane(width: 16, height: 16)
        let geometry = try MeshGeometry(vertices: base.vertices.map {
            // At u = 0 a repeat sampler blends the last and first texels.
            MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: SIMD2(0, 0.25))
        }, indices: base.indices)
        let map = try ProgressiveLightmapBaker.TextureSnapshot(.init(texture: texture))
        let mapped = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, worldFromObject: facingDown, emissiveRadiance: SIMD3(0.5, 0.25, 1),
            diffuseTexture: nil, emissiveTexture: map
        )
        let constant = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, worldFromObject: facingDown, emissiveRadiance: SIMD3(0.5, 0.5, 0)
        )
        func baker(_ surface: ProgressiveLightmapBaker.Surface) throws -> ProgressiveLightmapBaker {
            try ProgressiveLightmapBaker(
                device: device, receiver: receiver(), width: 1, height: 1, surfaces: [surface], maximumBounces: 0, seed: 29
            )
        }
        let textured = try baker(mapped)
        let reference = try baker(constant)
        try textured.accumulate(samples: 64)
        let blank = [SIMD4<Float16>](repeating: .zero, count: 4)
        blank.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 16)
        }
        try textured.accumulate(samples: 192)
        try reference.accumulate(samples: 256)
        let result = try pixels(textured.snapshot().irradianceTexture)
        #expect(try pixels(reference.snapshot().irradianceTexture) == result)
        #expect((1.48...1.58).contains(result[0].x) && result[0].x == result[0].y && result[0].z == 0)
        textured.reset()
        try textured.accumulate(samples: 256)
        #expect(try pixels(textured.snapshot().irradianceTexture) == result)
    }

    @Test("Independent UV transforms and factors sample one private color texture after its queued write")
    func separateMapBindings() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let staging = try colorTexture(device, format: .rgba16Float, texels: [
            SIMD4<Float16>(0.5, 0, 0, 1), SIMD4<Float16>(0, 1, 0, 1),
            SIMD4<Float16>(0, 0, 16, 1), SIMD4<Float16>(0, 0, 0.25, 1),
        ])
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 2, height: 2, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        let privateTexture = try #require(device.makeTexture(descriptor: descriptor))
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(from: staging, to: privateTexture)
        blit.endEncoding()
        command.commit()
        let sampling = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, addressV: .clampToEdge, magnification: .nearest
        )
        let diffuse = try ProgressiveLightmapBaker.TextureSnapshot(.init(
            texture: privateTexture, textureCoordinateSet: .secondary,
            transform: .init(offset: SIMD2(-0.5, -0.5)), sampling: sampling
        ), commandQueue: queue)
        let emissive = try ProgressiveLightmapBaker.TextureSnapshot(.init(
            texture: privateTexture, transform: .init(offset: SIMD2(0.5, 0)), sampling: sampling
        ), commandQueue: queue)

        let base = try MeshGeometry.plane(width: 16, height: 16)
        let geometry = try MeshGeometry(vertices: base.vertices.map {
            MeshVertex(
                position: $0.position, normal: $0.normal, textureCoordinate: SIMD2(0.25, 0.75),
                secondaryTextureCoordinate: SIMD2(0.75, 0.75)
            )
        }, indices: base.indices)
        let mapped = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, worldFromObject: facingDown, diffuseAlbedo: SIMD3(repeating: 0.5),
            emissiveRadiance: SIMD3(repeating: 2), diffuseTexture: diffuse, emissiveTexture: emissive
        )
        let constant = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, worldFromObject: facingDown, diffuseAlbedo: SIMD3(0.25, 0, 0), emissiveRadiance: SIMD3(0, 0, 0.5)
        )
        func bake(_ surface: ProgressiveLightmapBaker.Surface) throws -> SIMD4<Float> {
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: receiver(), width: 1, height: 1, lights: [diagonalLight()],
                surfaces: [surface], includesReceiverInTransport: false, maximumBounces: 1, seed: 48
            )
            try baker.accumulate(samples: 256)
            return try #require(try pixels(baker.snapshot().irradianceTexture).first)
        }
        let result = try bake(mapped)
        #expect(try bake(constant) == result)
        #expect((0.16...0.18).contains(result.x) && result.y == 0 && (1.48...1.58).contains(result.z))
    }

    @Test("Texture alpha cutouts transmit directional, point, spot and rectangle shadow rays")
    func maskedDirectLights() throws {
        typealias Bake = ProgressiveLightmapBaker
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try Bake.Surface(geometry: .plane(width: 4, height: 1))
        let alpha = try colorTexture(device, format: .rgba8Unorm_srgb, texels: [
            SIMD4<UInt8>(0, 0, 0, 255), SIMD4<UInt8>(255, 255, 255, 0),
            SIMD4<UInt8>(0, 0, 0, 255), SIMD4<UInt8>(255, 255, 255, 0),
        ])
        let binding = try Bake.TextureSnapshot(.init(texture: alpha, sampling: .init(magnification: .nearest)))
        let caster = try Bake.Surface(
            geometry: .plane(width: 8, height: 8),
            worldFromObject: SceneTransform(translation: SIMD3(0, 0, 1)).matrix,
            alphaMask: .init(texture: binding)
        )
        let sun = try MeshLightmapBaker.DirectionalLight(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(1, 0, 0))
        let point = try Bake.PositionedLight(PunctualLight(kind: .point(range: nil), color: SIMD3(0, 1, 0)), position: SIMD3(0, 0, 2))
        let spot = try Bake.PositionedLight(
            PunctualLight(kind: .spot(range: nil, innerConeAngle: .pi / 4, outerConeAngle: .pi / 3), color: SIMD3(0, 0, 1)),
            position: SIMD3(0, 0, 2)
        )
        let panel = try Bake.PositionedLight(PunctualLight(kind: .rectangle(width: 1, height: 1)), position: SIMD3(0, 0, 2))
        func bake(_ casters: [Bake.Surface]) throws -> [SIMD4<Float>] {
            let baker = try Bake(
                device: device, receiver: receiver, width: 4, height: 1, textureCoordinateSet: .primary,
                lights: [sun], punctualLights: [point, spot, panel], surfaces: casters, includesReceiverInTransport: false
            )
            return try pixels(baker.snapshot().irradianceTexture)
        }
        let baseline = try bake([])
        let masked = try bake([caster])
        #expect(baseline.allSatisfy { $0.x > 1 && $0.y > 0.1 && $0.z > 0.1 })
        #expect(masked[0] == SIMD4(0, 0, 0, 1) && masked[1] == SIMD4(0, 0, 0, 1))
        #expect(masked[2...] == baseline[2...])
    }

    @Test("Bounce traversal skips masked intersections and reaches the next emissive surface")
    func maskedIndirectHits() throws {
        typealias Bake = ProgressiveLightmapBaker
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try MeshGeometry.plane(width: 24, height: 24)
        let emitter = try Bake.Surface(
            geometry: geometry, worldFromObject: facingDown, emissiveRadiance: SIMD3(0, 0, 2)
        )
        let front = SceneTransform(translation: SIMD3(0, 0, 0.5)).matrix
        let alpha = try colorTexture(device, format: .rgba8Unorm, texels: [SIMD4<UInt8>](repeating: .zero, count: 4))
        let texture = try Bake.TextureSnapshot(.init(texture: alpha))
        let hole = try Bake.Surface(geometry: geometry, worldFromObject: front, alphaMask: .init(texture: texture))
        let solid = try Bake.Surface(geometry: geometry, worldFromObject: front)
        func bake(_ surfaces: [Bake.Surface]) throws -> SIMD4<Float> {
            let baker = try Bake(
                device: device, receiver: receiver(), width: 1, height: 1,
                surfaces: surfaces, includesReceiverInTransport: false, maximumBounces: 0, seed: 14
            )
            try baker.accumulate(samples: 256)
            return try #require(try pixels(baker.snapshot().irradianceTexture).first)
        }
        let visible = try bake([emitter])
        #expect(visible.z > 6)
        #expect(try bake([solid, emitter]) == SIMD4(0, 0, 0, 1))
        #expect(try bake([hole, emitter]) == visible)
    }

    @Test("Packed receiver coverage uses authored UV1, transformed alpha and vertex opacity")
    func maskedPackedReceivers() throws {
        typealias Bake = ProgressiveLightmapBaker
        let device = try #require(MTLCreateSystemDefaultDevice())
        let quad = try MeshGeometry.plane(width: 1, height: 1)
        let alpha = try colorTexture(device, format: .rgba16Float, texels: [
            SIMD4<Float16>(1, 1, 1, 0.5), .zero, SIMD4<Float16>(1, 1, 1, 0.5), .zero,
        ])
        let binding = try Bake.TextureSnapshot(.init(
            texture: alpha, textureCoordinateSet: .secondary, transform: .init(offset: SIMD2(0.5, 0)),
            sampling: .init(addressU: .repeat, magnification: .nearest)
        ))
        let mask = try Bake.AlphaMask(cutoff: 0.25, opacity: 0.8, texture: binding)
        func surface(_ x: Float, uv: Float, vertexOpacity: Float) throws -> Bake.Surface {
            let geometry = try MeshGeometry(vertices: quad.vertices.map {
                MeshVertex(
                    position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate,
                    secondaryTextureCoordinate: SIMD2(uv, 0.25), color: SIMD4(1, 1, 1, vertexOpacity)
                )
            }, indices: quad.indices)
            return try Bake.Surface(
                geometry: geometry, worldFromObject: SceneTransform(translation: SIMD3(x, 0, 0)).matrix, alphaMask: mask
            )
        }
        let baker = try Bake(
            device: device, receivers: [
                surface(-2, uv: 0.75, vertexOpacity: 0.75), surface(0, uv: 0.25, vertexOpacity: 1),
                surface(2, uv: 0.75, vertexOpacity: 0.5),
            ], resolution: 32, padding: 0
        )
        let actual = try bytes(baker.snapshot().coverageTexture)
        let originalMasks = try baker.receiverBindings.map { binding in
            try bytes(MeshTextureBaker(
                device: device, geometry: binding.geometry, width: 32, height: 32, textureCoordinateSet: .secondary
            ).coverageTexture)
        }
        #expect(originalMasks.allSatisfy { $0.contains(255) })
        #expect(actual == originalMasks[0])
        #expect(throws: Bake.Failure.textureCoordinates) { try Bake.Surface(geometry: quad, alphaMask: mask) }
        #expect(throws: Bake.Failure.alphaMask) { try Bake.AlphaMask(cutoff: .nan) }
    }

    @Test("Transport texture bindings validate UVs, samplers, pixel formats and emissive range")
    func textureValidation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try MeshGeometry.plane(width: 2, height: 2)
        let texture = try colorTexture(
            device, format: .rgba16Float, texels: [SIMD4<Float16>](repeating: SIMD4(2, 1, 0, 1), count: 4)
        )
        let secondary = try ProgressiveLightmapBaker.TextureSnapshot(.init(texture: texture, textureCoordinateSet: .secondary))
        #expect(throws: ProgressiveLightmapBaker.Failure.textureCoordinates) {
            try ProgressiveLightmapBaker.Surface(geometry: geometry, diffuseTexture: secondary)
        }
        let sampler = try #require(device.makeSamplerState(descriptor: MTLSamplerDescriptor()))
        #expect(throws: ProgressiveLightmapBaker.Failure.textureSampler) {
            try ProgressiveLightmapBaker.TextureSnapshot(.init(texture: texture, sampler: sampler))
        }
        let receiver = try receiver()
        let integerTexture = try colorTexture(device, format: .r8Uint, texels: [UInt8](repeating: 3, count: 4))
        #expect(throws: ProgressiveLightmapBaker.Failure.unsupportedTexture) {
            try ProgressiveLightmapBaker.TextureSnapshot(.init(texture: integerTexture))
        }
        let map = try ProgressiveLightmapBaker.TextureSnapshot(.init(texture: texture))
        let overRange = try ProgressiveLightmapBaker.Surface(
            geometry: geometry, emissiveRadiance: SIMD3(40_000, 0, 0), diffuseTexture: nil, emissiveTexture: map
        )
        #expect(throws: ProgressiveLightmapBaker.Failure.textureValues) {
            try ProgressiveLightmapBaker(device: device, receiver: receiver, width: 1, height: 1, surfaces: [overRange])
        }
    }

    @Test("The raster material samples baked indirect irradiance through UV1")
    func materialBinding() throws {
        let fixture = try GeometryRenderFixture()
        let receiver = try receiver(atlasInset: 0.25)
        let baker = try ProgressiveLightmapBaker(
            device: fixture.device, receiver: receiver, width: 8, height: 8,
            lights: [diagonalLight()], surfaces: [redPanel(width: 12)],
            includesReceiverInTransport: false, maximumBounces: 1, seed: 2, padding: 2
        )
        try baker.accumulate(samples: 64)
        let map = try baker.snapshot()
        let binding = try map.lightMapBinding()
        #expect(binding.textureCoordinateSet == .secondary)
        #expect(binding.sampling?.addressU == .clampToEdge && binding.sampling?.addressV == .clampToEdge)
        let color = SIMD3<Float>(0.5, 0.75, 0.5)
        let material = try ClassicMaterial(model: .basic, color: color, lightMap: binding)
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: receiver.geometry, material: .classic(material)
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let raster = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera))
        let shaded = fixture.pixel(raster.pixels, x: 32, y: 32)
        #expect((0.08...0.095).contains(shaded.x))
        #expect(shaded.y == 0 && shaded.z == 0 && shaded.w == 1)
    }

    @Test("Invalid settings and over-range snapshots preserve accumulation state")
    func validation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try receiver()
        #expect(throws: ProgressiveLightmapBaker.Failure.surface) {
            try ProgressiveLightmapBaker.Surface(geometry: receiver.geometry, diffuseAlbedo: SIMD3(1.1, 0, 0))
        }
        #expect(throws: ProgressiveLightmapBaker.Failure.texelLimit) {
            try ProgressiveLightmapBaker(device: device, receiver: receiver, width: 513, height: 513)
        }
        #expect(throws: ProgressiveLightmapBaker.Failure.transport) {
            try ProgressiveLightmapBaker(device: device, receiver: receiver, width: 1, height: 1, maximumBounces: 9)
        }
        let baker = try ProgressiveLightmapBaker(
            device: device, receiver: receiver, width: 1, height: 1, skyRadiance: SIMD3(repeating: 40_000)
        )
        #expect(throws: ProgressiveLightmapBaker.Failure.sampleBudget) { try baker.accumulate(samples: 0) }
        #expect(throws: ProgressiveLightmapBaker.Failure.sampleBudget) { try baker.accumulate(samples: 257) }
        #expect(baker.samplesPerTexel == 0)
        try baker.accumulate(samples: 1)
        #expect(throws: MeshLightmapBaker.Failure.irradianceRange) { try baker.snapshot() }
        #expect(baker.samplesPerTexel == 1)
    }

    private var facingDown: simd_float4x4 {
        simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, -1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 0, 1, 1)))
    }

    private func receiver(atlasInset: Float = 0) throws -> ProgressiveLightmapBaker.Surface {
        let plane = try MeshGeometry.plane(width: 0.5, height: 0.5)
        let geometry = try MeshGeometry(vertices: plane.vertices.map {
            MeshVertex(
                position: $0.position, normal: $0.normal, textureCoordinate: .zero,
                secondaryTextureCoordinate: SIMD2<Float>(repeating: atlasInset) + (1 - 2 * atlasInset) * $0.textureCoordinate
            )
        }, indices: plane.indices)
        return try .init(geometry: geometry)
    }

    private func redPanel(width: Float) throws -> ProgressiveLightmapBaker.Surface {
        try .init(
            geometry: .plane(width: width, height: width), worldFromObject: facingDown, diffuseAlbedo: SIMD3(0.8, 0, 0)
        )
    }

    private func diagonalLight() throws -> MeshLightmapBaker.DirectionalLight {
        try .init(directionToLight: SIMD3(1, 0, -1), irradiance: SIMD3(repeating: 1))
    }

    private func colorTexture<T>(
        _ device: any MTLDevice, format: MTLPixelFormat, mipmapped: Bool = false, texels: [T]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 2, height: 2, mipmapped: mipmapped)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        texels.withUnsafeBytes {
            texture.replace(
                region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0,
                withBytes: $0.baseAddress!, bytesPerRow: 2 * MemoryLayout<T>.stride
            )
        }
        if mipmapped {
            let black = [UInt8](repeating: 0, count: MemoryLayout<T>.stride)
            black.withUnsafeBytes {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 1,
                    withBytes: $0.baseAddress!, bytesPerRow: MemoryLayout<T>.stride
                )
            }
        }
        return texture
    }

    private func pixels(_ texture: any MTLTexture) -> [SIMD4<Float>] {
        var result = [SIMD4<Float16>](repeating: .zero, count: texture.width * texture.height)
        texture.getBytes(
            &result, bytesPerRow: texture.width * MemoryLayout<SIMD4<Float16>>.stride,
            from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
        )
        return result.map { SIMD4<Float>($0) }
    }

    private func bytes(_ texture: any MTLTexture) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: texture.width * texture.height)
        texture.getBytes(
            &result, bytesPerRow: texture.width, from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
        )
        return result
    }
}
