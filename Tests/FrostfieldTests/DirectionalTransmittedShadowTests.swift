import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct DirectionalTransmittedShadowTests {
    @Test("Overlapping glass layers tint only receivers behind them and preserve opaque depth",
          arguments: [false, true])
    func glassLayers(cascaded: Bool) async throws {
        let gpu = try RasterLightingFixture(width: 132, height: 100)
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let backdropDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.width, height: gpu.height, mipmapped: true
        )
        backdropDescriptor.storageMode = .private
        backdropDescriptor.usage = .shaderRead
        let backdrop = try #require(gpu.device.makeTexture(descriptor: backdropDescriptor))
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 15)
        let pose = RasterLightingFixture.cameraTransform(position: SIMD3(0, 6, 0), pitch: -.pi / 2)
        let projection = simd_float4x4(
            SIMD4(0.25, 0, 0, 0), SIMD4(0, 1.0 / 3, 0, 0), SIMD4(0, 0, -1.0 / 15, 0), SIMD4(0, 0, 0, 1)
        )
        let view = try ShaderGraphCamera(
            viewMatrix: simd_inverse(pose.matrix), projectionMatrix: projection, near: 0, far: 15
        )
        let light = SIMD3<Float>(1, 1, 0)

        func shadowMap(layers: Int) throws -> DirectionalShadowMap {
            if cascaded {
                let map = try DirectionalShadowMap(
                    queue: gpu.queue, resolution: 192, directionToLight: light,
                    cascades: .init(count: 2, maximumDistance: 8, blendFraction: 0.2, casterMargin: 2),
                    intensity: 2, sampling: .basic, updateMode: .onDemand, transmissionLayers: layers
                )
                try map.fitCascades(view: view, splitFractions: [0.65])
                return map
            }
            return try DirectionalShadowMap(
                queue: gpu.queue, resolution: 192, directionToLight: light, halfExtent: 6,
                intensity: 2, sampling: .basic, updateMode: .onDemand, transmissionLayers: layers
            )
        }

        func plane(
            _ center: SIMD3<Float>, size: SIMD2<Float>, material: MeshMaterial, scale: Float = 1, castsShadow: Bool = true
        ) throws -> SceneNode {
            let mesh = try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float,
                geometry: .plane(width: size.x, height: size.y), material: material
            )
            var transform = RasterLightingFixture.groundTransform(y: center.y)
            transform.translation = center
            transform.scale = SIMD3(repeating: scale)
            return SceneNode(transform: transform, mesh: mesh, castsShadow: castsShadow)
        }

        let diffuse = MeshMaterial.physicallyBased(try PhysicallyBasedMaterial(
            roughness: 1, ambientIntensity: 0.1, receivesShadows: true,
            transmission: .init(factor: 0, indexOfRefraction: 1)
        ))
        let frontFilter = SIMD3<Float>(0.9, 0.2, 0.7)
        let backColor = SIMD3<Float>(0.2, 0.8, 0.5)
        let backFactor: Float = 0.75
        let first = MeshMaterial.physicallyBased(try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, transmission: .init(
                factor: 1, indexOfRefraction: 1, thickness: 0.5, attenuationDistance: 1, attenuationColor: frontFilter
            )
        ))
        let second = MeshMaterial.physicallyBased(try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(backColor, 1), alphaMode: .opaque,
            transmission: .init(factor: backFactor, indexOfRefraction: 1)
        ))
        let floor = try plane(.zero, size: SIMD2(7, 5.5), material: diffuse, castsShadow: false)
        let middle = try plane(SIMD3(1.25, 1.25, 0), size: SIMD2(0.48, 0.85), material: diffuse, castsShadow: false)
        let opaque = try plane(SIMD3(0.45, 0.45, 1.4), size: SIMD2(0.5, 0.65), material: .unlit)
        let upper = try plane(SIMD3(2, 2, 0), size: SIMD2(0.55, 2.5), material: first, scale: 2)
        let lower = try plane(SIMD3(1, 1, 0), size: SIMD2(1.1, 5), material: second)
        let points: [SIMD3<Float>] = [SIMD3(0, 0, -1.4), SIMD3(1.25, 1.25, 0), SIMD3(0, 0, 1.4), SIMD3(-1.4, 0, 0)]
        let indices = points.map { point in
            let pixel = gpu.project(point, clipFromView: projection, worldFromView: pose.matrix)
            return Int(pixel.y) * gpu.width + Int(pixel.x)
        }
        let sliceCount = cascaded ? 2 : 1

        func draw(_ map: DirectionalShadowMap, panes: [SceneNode], passes: Int) throws -> [SIMD3<Float>] {
            let graph = SceneGraph(roots: [floor, middle, opaque] + panes, camera: camera)
            try gpu.run { command in
                let stats = try graph.encode(
                    pass: gpu.pass(color: color, depth: depth), commandBuffer: command,
                    clipFromWorld: projection * view.viewMatrix, cameraPosition: pose.translation,
                    directionToCamera: SIMD3(0, 1, 0), shadowMap: map, transmissionBackdrop: backdrop, graphCamera: view
                )
                #expect(stats.shadowPassCount == passes)
            }
            let pixels = try gpu.pixels(of: color)
            return indices.map {
                let value = pixels[$0]
                #expect(value.w == 1)
                return SIMD3(value.x, value.y, value.z)
            }
        }

        func shadowDepths(_ map: DirectionalShadowMap) async throws -> [Data] {
            var depths: [Data] = []
            for slice in 0..<sliceCount {
                depths.append(try await TextureTransfer.read(from: map.texture, slice: slice, commandQueue: gpu.queue).bytes)
            }
            return depths
        }

        let defaultMap = try shadowMap(layers: 0)
        let baseline = try draw(defaultMap, panes: [lower, upper], passes: sliceCount)
        #expect(baseline[0] == baseline[3] && baseline[1] == baseline[3], "\(baseline)")
        #expect(baseline[3].x > baseline[2].x + 0.3, "\(baseline)")
        let depthBefore = try await shadowDepths(defaultMap)

        let map = try shadowMap(layers: 2)
        let tinted = try draw(map, panes: [lower, upper], passes: sliceCount * 3)
        let ambient = baseline[2]
        let direct = baseline[3] - ambient
        let expected = [ambient + direct * frontFilter * (backColor * backFactor),
                        ambient + direct * frontFilter, ambient, baseline[3]]
        for (actual, expected) in zip(tinted, expected) {
            #expect(simd_distance(actual, expected) < 0.006, "actual \(actual) expected \(expected)")
        }
        #expect(try await shadowDepths(map) == depthBefore)

        let lightColor = SIMD3<Float>(0.25, 0.7, 0.9)
        try map.update(color: lightColor)
        let colored = try draw(map, panes: [lower, upper], passes: 0)
        for (actual, white) in zip(colored, tinted) {
            let expected = ambient + (white - ambient) * lightColor
            #expect(simd_distance(actual, expected) < 0.006)
        }
        try map.update(color: SIMD3(repeating: 1))

        try map.updateReceiver(normalBias: 0, shadowStrength: 0.5)
        let partial = try draw(map, panes: [upper, lower], passes: 0)
        for (index, value) in partial.enumerated() {
            #expect(simd_distance(value, (tinted[index] + baseline[3]) * 0.5) < 0.006)
        }

        try map.updateReceiver(normalBias: 0, shadowStrength: 1)
        #expect(try draw(map, panes: [], passes: 0) == tinted)
        map.requestUpdate()
        #expect(try draw(map, panes: [], passes: sliceCount * 3) == baseline)
        #expect(try await shadowDepths(map) == depthBefore)
    }

    @Test("Textured colored shadows honor sRGB, separate UV sets and samplers, volume thickness and alpha cutout")
    func texturedGlass() throws {
        let gpu = try RasterLightingFixture(width: 120, height: 96)
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let backdropDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.width, height: gpu.height, mipmapped: true
        )
        backdropDescriptor.storageMode = .private
        backdropDescriptor.usage = .shaderRead
        let backdrop = try #require(gpu.device.makeTexture(descriptor: backdropDescriptor))

        func texture(_ width: Int, _ height: Int = 1, format: MTLPixelFormat = .rgba8Unorm,
                     pixels: [SIMD4<UInt8>]) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            let result = try #require(gpu.device.makeTexture(descriptor: descriptor))
            pixels.withUnsafeBytes {
                result.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                               withBytes: $0.baseAddress!, bytesPerRow: width * MemoryLayout<SIMD4<UInt8>>.stride)
            }
            return result
        }
        let firstColor = SIMD4<UInt8>(128, 230, 180, 255)
        let secondColor = SIMD4<UInt8>(255, 100, 220, 255)
        let base = try texture(2, format: .rgba8Unorm_srgb, pixels: [firstColor, secondColor])
        let factor = try texture(4, pixels: [SIMD4(0, 20, 255, 255), SIMD4(64, 20, 255, 255),
                                     SIMD4(192, 20, 255, 255), SIMD4(128, 20, 255, 255)])
        let thickness = try texture(1, 4, pixels: [SIMD4(0, 64, 50, 255), SIMD4(0, 0, 50, 255),
                                          SIMD4(0, 255, 50, 255), SIMD4(0, 192, 50, 255)])
        let alpha = try texture(1, 2, pixels: [SIMD4(0, 255, 0, 0), SIMD4(0, 0, 0, 255)])
        let metal = try gpu.constantTexture(SIMD4(0, 255, 128, 255))
        let nearestRepeat = PhysicallyBasedMaterial.TextureSampling(
            magnification: .nearest, minification: .nearest, mipFilter: .notMipmapped
        )
        let nearestClamp = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, addressV: .clampToEdge, magnification: .nearest,
            minification: .nearest, mipFilter: .notMipmapped
        )
        let factorSampler = try nearestClamp.makeSamplerState(device: gpu.device)
        let absorption = SIMD3<Float>(0.35, 0.65, 0.85)
        let glassMaterial = try PhysicallyBasedMaterial(
            baseColorTexture: base, metallic: 0.4, roughness: 1, lightIntensity: 0, ambientIntensity: 0,
            metallicRoughnessTexture: .init(texture: metal), alphaMode: .mask(cutoff: 0.5),
            transmission: .init(
                factor: 0.85, indexOfRefraction: 1, thickness: 0.5, attenuationDistance: 0.5, attenuationColor: absorption,
                factorTexture: .init(texture: factor, sampler: factorSampler, textureCoordinateSet: .secondary,
                                     transform: .init(offset: SIMD2(0.4, 0))),
                thicknessTexture: .init(texture: thickness, transform: .init(offset: SIMD2(0, 1), rotation: .pi / 2),
                                        sampling: nearestClamp)
            ),
            baseColorTextureTransform: .init(offset: SIMD2(0.5, 0)), baseColorSampling: nearestRepeat,
            alphaTexture: .init(texture: alpha, textureCoordinateSet: .secondary,
                                transform: .init(offset: SIMD2(0, 1), scale: SIMD2(1, -1)))
        )

        let corners: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)]
        let geometry = try MeshGeometry(vertices: corners.map { uv in
            MeshVertex(position: SIMD3((uv.x - 0.5) * 4, (uv.y - 0.5) * 0.9, 0), normal: SIMD3(0, 0, 1),
                       textureCoordinate: uv, secondaryTextureCoordinate: SIMD2(1 - uv.x, uv.y))
        }, indices: [0, 1, 2, 0, 2, 3])
        let glassMesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(glassMaterial)
        )
        var glassPose = RasterLightingFixture.groundTransform(y: 1.2)
        glassPose.translation.z = 1.2
        let receiverMesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 7, height: 5),
            material: .physicallyBased(.init(
                roughness: 1, ambientIntensity: 0, receivesShadows: true, transmission: .init(factor: 0, indexOfRefraction: 1)
            ))
        )
        let receiver = SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: receiverMesh, castsShadow: false)
        let pane = SceneNode(transform: glassPose, mesh: glassMesh)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 15)
        let pose = RasterLightingFixture.cameraTransform(position: SIMD3(0, 5, 0), pitch: -.pi / 2)
        let projection = simd_float4x4(
            SIMD4(0.25, 0, 0, 0), SIMD4(0, 1.0 / 3, 0, 0), SIMD4(0, 0, -1.0 / 15, 0), SIMD4(0, 0, 0, 1)
        )
        let view = try ShaderGraphCamera(viewMatrix: simd_inverse(pose.matrix), projectionMatrix: projection, near: 0, far: 15)
        let graph = SceneGraph(roots: [receiver, pane], camera: camera)

        func render(_ map: DirectionalShadowMap, shadowPasses: Int) throws -> [SIMD3<Float>] {
            try gpu.run { command in
                let stats = try graph.encode(
                    pass: gpu.pass(color: color, depth: depth), commandBuffer: command,
                    clipFromWorld: projection * view.viewMatrix, cameraPosition: pose.translation,
                    directionToCamera: SIMD3(0, 1, 0), shadowMap: map, transmissionBackdrop: backdrop, graphCamera: view
                )
                #expect(stats.shadowPassCount == shadowPasses)
            }
            let pixels = try gpu.pixels(of: color)
            let points = [SIMD3<Float>(-1.2, 0, -0.2), SIMD3(1.2, 0, -0.2), SIMD3(-1.2, 0, 0.2)]
            return points.map {
                let position = gpu.project($0, clipFromView: projection, worldFromView: pose.matrix)
                let value = pixels[Int(position.y) * gpu.width + Int(position.x)]
                return SIMD3(value.x, value.y, value.z)
            }
        }

        func draw(layers: Int) throws -> [SIMD3<Float>] {
            let map = try DirectionalShadowMap(
                queue: gpu.queue, resolution: 256, directionToLight: SIMD3(0, 1, 1), halfExtent: 6,
                intensity: 2, sampling: .basic, transmissionLayers: layers
            )
            return try render(map, shadowPasses: 1 + layers)
        }

        func expected(_ color: SIMD4<UInt8>, factor: UInt8, thickness: UInt8) -> SIMD3<Float> {
            let transmission = 0.85 * Float(factor) / 255 * (1 - 0.4 * Float(128) / 255)
            return SIMD3<Float>((0..<3).map { channel in
                let srgb = Float(color[channel]) / 255
                let linear = srgb <= 0.04045 ? srgb / 12.92 : pow((srgb + 0.055) / 1.055, 2.4)
                return linear * transmission * pow(absorption[channel], Float(thickness) / 255)
            })
        }
        let unshadowed = try draw(layers: 0)
        #expect(unshadowed.allSatisfy { $0.x > 0.4 })
        let tinted = try draw(layers: 1)
        let expectedLeft = unshadowed[0] * expected(secondColor, factor: 128, thickness: 192)
        let expectedRight = unshadowed[1] * expected(firstColor, factor: 192, thickness: 64)
        #expect(simd_distance(tinted[0], expectedLeft) < 0.004, "left: \(tinted[0]), expected \(expectedLeft)")
        #expect(simd_distance(tinted[1], expectedRight) < 0.004, "right: \(tinted[1]), expected \(expectedRight)")
        #expect(tinted[2] == unshadowed[2])

        // Sun RGB radiance scales the textured glass result and reuses the cached glass layers.
        let onDemand = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 256, directionToLight: SIMD3(0, 1, 1), halfExtent: 6,
            intensity: 2, sampling: .basic, updateMode: .onDemand, transmissionLayers: 1
        )
        let white = try render(onDemand, shadowPasses: 2)
        for (actual, expected) in zip(white, tinted) {
            #expect(simd_distance(actual, expected) < 0.001, "on-demand \(actual) expected \(expected)")
        }
        let lightColor = SIMD3<Float>(0.25, 0.7, 0.9)
        try onDemand.update(color: lightColor)
        let colored = try render(onDemand, shadowPasses: 0)
        for (actual, reference) in zip(colored, white) {
            let scaled = reference * lightColor
            #expect(simd_distance(actual, scaled) < 0.004, "colored \(actual) expected \(scaled)")
        }
    }

    @Test("Opted-in colored shadows reject graph casters before encoding")
    func unsupportedCaster() throws {
        let gpu = try RasterLightingFixture(width: 16, height: 16)
        let surface = try SurfaceShaderGraph { builder in
            .init(baseColor: builder.constant(SIMD3<Float>(1, 0, 0)))
        }
        let material = try PhysicallyBasedMaterial(transmission: .init(factor: 1), surfaceGraph: surface)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1, height: 1),
            material: .physicallyBased(material)
        )
        let caster = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let graph = SceneGraph(roots: [caster], camera: try .init(verticalFieldOfView: .pi / 3, near: 0.1, far: 20))
        let map = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 32, directionToLight: SIMD3(0, 0, 1), halfExtent: 5, transmissionLayers: 1
        )
        let command = try #require(gpu.queue.makeCommandBuffer())
        let pass = try gpu.pass(color: gpu.colorTarget(), depth: gpu.depthTarget())
        #expect(throws: MeshRenderer.Failure.transmittedShadowCaster(caster.id)) {
            try graph.encode(pass: pass, commandBuffer: command, shadowMap: map)
        }
        #expect(command.status == .notEnqueued)
    }

    @Test("Directional transmission validates layer count and storage budget before allocation")
    func validation() throws {
        let gpu = try RasterLightingFixture(width: 16, height: 16)
        #expect(throws: DirectionalShadowMap.Failure.transmissionLayers) {
            try DirectionalShadowMap(
                queue: gpu.queue, resolution: 64, directionToLight: SIMD3(0, 1, 0), halfExtent: 5, transmissionLayers: 5
            )
        }
        #expect(throws: DirectionalShadowMap.Failure.transmissionLayers) {
            try DirectionalShadowMap(
                queue: gpu.queue, resolution: 4096, directionToLight: SIMD3(0, 1, 0),
                cascades: .init(), transmissionLayers: 4
            )
        }
    }
}
