import Frostfield
import Metal
import simd

/// Static atlas preparation for the authored studio camera and sweep.
enum LabAssetBaking {
    static func grip(
        _ geometry: MeshGeometry, radius: Float, depth: Float, device: any MTLDevice,
        light: SIMD3<Float>, intensity: Float, environment: PhysicallyBasedMaterial.Environment?
    ) throws -> (MeshGeometry, PhysicallyBasedMaterial) {
        let vertices = geometry.vertices.map { vertex in
            let uv = vertex.textureCoordinate
            let distanceFromAxis = simd_length(SIMD2(vertex.position.x, vertex.position.z))
            let relief = min(max((distanceFromAxis - radius) / depth, 0), 1)
            let wear = relief * (0.24 + 0.12 * cos(2 * .pi * uv.x - 0.8))
            let dark = SIMD3<Float>(0.031, 0.034, 0.04)
            let warm = SIMD3<Float>(0.22, 0.13, 0.065)
            let color = dark * (1 - wear) + warm * wear
            return MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: uv,
                tangent: vertex.tangent, color: SIMD4(color, 1)
            )
        }
        let colored = try MeshGeometry(vertices: vertices, indices: geometry.indices)
        let atlas = try colored.unwrappedUVAtlas(resolution: 512, padding: 4)
        let maps = try MeshTextureBaker(
            device: device, geometry: atlas.geometry, width: atlas.resolution, height: atlas.resolution,
            textureCoordinateSet: .secondary, padding: atlas.padding
        )
        let textured = try MeshGeometry(vertices: atlas.geometry.vertices.map { vertex in
            // The baked texture supplies the wear color at draw time.
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.secondaryTextureCoordinate, tangent: vertex.tangent
            )
        }, indices: atlas.geometry.indices)
        let material = try PhysicallyBasedMaterial(
            baseColorTexture: maps.vertexColorTexture, metallic: 0.89, roughness: 0.33,
            directionToLight: light, lightIntensity: intensity, ambientIntensity: 0.18, receivesShadows: true,
            alphaMode: .opaque, environment: environment, baseColorTextureCoordinateSet: maps.textureCoordinateSet,
            clearcoat: .init(factor: 0.20, roughness: 0.26)
        )
        return (textured, material)
    }

    enum SweepBake: Equatable {
        case staticDirect
        case progressiveReference
        case progressiveIndirect
        case sharedReference
        case sharedIndirect
        case sharedShader

        var usesSharedAtlas: Bool { self == .sharedReference || self == .sharedIndirect || self == .sharedShader }
    }

    static func sweep(
        _ geometry: MeshGeometry, beneath asset: SceneGraph, queue: any MTLCommandQueue, light: SIMD3<Float>,
        intensity: Float, bake: SweepBake, floorAtlasFraction: Float
    ) throws -> [SceneNode] {
        let device = queue.device
        let casters = try asset.resolve().compactMap { node -> (mesh: MeshRenderer, transform: simd_float4x4)? in
            guard node.castsShadow, let mesh = node.mesh, !mesh.requiresTransmissionBackdrop else { return nil }
            return (mesh, node.worldTransform)
        }
        let direction = simd_normalize(light)
        let tangent = simd_normalize(simd_cross(direction, SIMD3(0, 0, 1)))
        let bitangent = simd_cross(direction, tangent)
        let spread: Float = 0.065
        let offsets: [SIMD3<Float>] = [.zero, tangent * spread, -tangent * spread, bitangent * spread, -bitangent * spread]
        var lights = try offsets.map { offset in
            try MeshLightmapBaker.DirectionalLight(
                directionToLight: direction + offset,
                irradiance: SIMD3(1, 0.94, 0.86) * (intensity / Float(offsets.count))
            )
        }
        lights.append(try .init(directionToLight: SIMD3(0.4, 0.9, 1.2), irradiance: SIMD3(0.27, 0.38, 0.52)))
        let albedo = SIMD3<Float>(0.092, 0.126, 0.16)
        let map: MeshLightmapBaker
        switch bake {
        case .staticDirect:
            map = try MeshLightmapBaker(
                device: device, geometry: geometry, width: 512, height: 512,
                textureCoordinateSet: .primary, lights: lights,
                casters: casters.map { .init($0.mesh.geometry, transform: $0.transform) }, receiverCastsShadows: false
            )
        case .progressiveReference, .progressiveIndirect:
            let surfaces: [ProgressiveLightmapBaker.Surface]
            if case .progressiveIndirect = bake {
                surfaces = try transportSurfaces(casters)
            } else {
                surfaces = try casters.map { try .init(geometry: $0.mesh.geometry, worldFromObject: $0.transform) }
            }
            let baker = try ProgressiveLightmapBaker(
                device: device, receiver: .init(geometry: geometry), width: 384, height: 256,
                textureCoordinateSet: .primary, lights: lights, surfaces: surfaces,
                includesReceiverInTransport: false, maximumBounces: 1, seed: 35
            )
            let direct = try baker.snapshot()
            if case .progressiveIndirect = bake {
                for _ in 0..<3 { try baker.accumulate(samples: 16) }
                map = try baker.snapshot()
                reportContribution(
                    direct: direct.irradianceTexture, indirect: map.irradianceTexture,
                    label: "floor", rowFraction: floorAtlasFraction, albedo: albedo, samples: baker.samplesPerTexel
                )
            } else {
                map = direct
            }
        case .sharedReference, .sharedIndirect, .sharedShader:
            // Duplicate the seam vertices so the floor and backdrop get independent UV1 charts.
            let floor = try MeshGeometry(vertices: Array(geometry.vertices.prefix(4)), indices: Array(geometry.indices.prefix(6)))
            let backdrop = try MeshGeometry(vertices: Array(geometry.vertices.dropFirst(2)), indices: geometry.indices.dropFirst(6).map { $0 - 2 })
            let receivers = try [floor, backdrop].map {
                try ProgressiveLightmapBaker.Surface(geometry: $0, diffuseAlbedo: albedo)
            }
            let baker = try ProgressiveLightmapBaker(
                device: device, receivers: receivers, resolution: 320, lights: lights,
                surfaces: transportSurfaces(casters), maximumBounces: 1, seed: 35
            )
            let usesShader = bake == .sharedShader
            let shaderSamples = 16
            if usesShader {
                try baker.accumulate(samples: shaderSamples, sampling: .halton) { _ in .zero }
            }
            let direct = try baker.snapshot()
            let shared: MeshLightmapBaker
            if bake != .sharedReference {
                if usesShader {
                    baker.reset()
                    try baker.accumulate(samples: shaderSamples, irradianceKernel: reflectorKernel(device: device), commandQueue: queue)
                } else {
                    for _ in 0..<3 { try baker.accumulate(samples: 16) }
                }
                shared = try baker.snapshot()
                reportContribution(
                    direct: direct.irradianceTexture, indirect: shared.irradianceTexture,
                    label: "shared atlas", rowFraction: 1, albedo: albedo, samples: baker.samplesPerTexel
                )
                for (name, binding) in zip(["floor receiver", "back receiver"], baker.receiverBindings) {
                    let coverage = try MeshTextureBaker(
                        device: device, geometry: binding.geometry, width: shared.irradianceTexture.width,
                        height: shared.irradianceTexture.height, textureCoordinateSet: shared.textureCoordinateSet
                    ).coverageTexture
                    reportContribution(
                        direct: direct.irradianceTexture, indirect: shared.irradianceTexture, label: name,
                        rowFraction: 1, albedo: albedo, samples: baker.samplesPerTexel, coverage: coverage
                    )
                }
            } else {
                shared = direct
            }
            print("Asset Studio packed \(baker.receiverBindings.count) receivers into one \(shared.irradianceTexture.width)×\(shared.irradianceTexture.height) UV1 atlas")
            return try baker.receiverBindings.map { binding in
                try receiverNode(binding.geometry, device: device, map: shared, albedo: albedo, usesPBRLightMap: usesShader)
            }
        }
        return [try receiverNode(geometry, device: device, map: map, albedo: albedo)]
    }

    private static func reflectorKernel(device: any MTLDevice) throws -> ShaderBufferKernel {
        let graph = try BufferShaderGraph(fieldCount: 3) { builder in
            let position = builder.elementField(0)
            let normal = builder.elementField(1).xyz
            // One sample from each cell of a one-square-metre bounce card.
            let x = position.w.mod(4) * 0.25 - 0.375
            let z = (position.w / 4).floor() * 0.25 - 0.375
            let sample = builder.vector(x - 1.25, builder.constant(Float(1.85)), z + 0.6)
            let toCard = sample - position.xyz
            let direction = toCard.normalized()
            let projectedArea = normal.dot(direction).clamped(0, 1) * direction.y.clamped(0, 1)
            let distanceSquared = toCard.dot(toCard).clamped(0.05, 10_000)
            let warm = builder.constant(SIMD3<Float>(1.2, 0.66, 0.36))
            return [2: builder.vector(warm * (projectedArea / distanceSquared), builder.constant(Float(0)))]
        }
        return try ShaderBufferKernel(graph: graph, device: device)
    }

    private static func receiverNode(
        _ geometry: MeshGeometry, device: any MTLDevice, map: MeshLightmapBaker, albedo: SIMD3<Float>, usesPBRLightMap: Bool = false
    ) throws -> SceneNode {
        let material = if usesPBRLightMap {
            try PhysicallyBasedMaterial(
                roughness: 0.88, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(albedo, 1),
                alphaMode: .opaque, specular: .init(factor: 0), lightMap: map.pbrLightMapBinding()
            )
        } else {
            try PhysicallyBasedMaterial(
                roughness: 0.88, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveTexture: .init(texture: map.irradianceTexture, textureCoordinateSet: map.textureCoordinateSet),
                emissiveFactor: albedo / .pi, alphaMode: .opaque, specular: .init(factor: 0)
            )
        }
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        return SceneNode(mesh: mesh, castsShadow: false)
    }

    private struct TransportMapKey: Hashable {
        let texture: ObjectIdentifier
        let set: MeshTextureCoordinateSet
        let transform: PhysicallyBasedMaterial.TextureTransform
        let sampling: PhysicallyBasedMaterial.TextureSampling?

        init(_ binding: PhysicallyBasedMaterial.TextureBinding) {
            texture = ObjectIdentifier(binding.texture)
            set = binding.textureCoordinateSet
            transform = binding.transform
            sampling = binding.sampling
        }
    }

    private static func transportSurfaces(
        _ casters: [(mesh: MeshRenderer, transform: simd_float4x4)]
    ) throws -> [ProgressiveLightmapBaker.Surface] {
        var maps: [TransportMapKey: ProgressiveLightmapBaker.TextureSnapshot] = [:]
        func snapshot(_ binding: PhysicallyBasedMaterial.TextureBinding?) throws -> ProgressiveLightmapBaker.TextureSnapshot? {
            guard let binding else { return nil }
            let key = TransportMapKey(binding)
            if let existing = maps[key] { return existing }
            let texture = try ProgressiveLightmapBaker.TextureSnapshot(binding)
            maps[key] = texture
            return texture
        }
        return try casters.map { mesh, transform in
            guard case .physicallyBased(let material) = mesh.material, material.surfaceGraph == nil else {
                // The Lab's optional shader finish still uses its earlier matte proxy.
                return try .init(geometry: mesh.geometry, worldFromObject: transform, diffuseAlbedo: SIMD3(0.18, 0.12, 0.075))
            }
            let factors = material.liveFactors?.current ?? .init(material)
            let base = factors.baseColorFactor
            let diffuse = try snapshot(material.baseColorTextureBinding)
            let alphaMask: ProgressiveLightmapBaker.AlphaMask?
            if case .mask(let cutoff) = material.alphaMode {
                alphaMask = try .init(cutoff: cutoff, opacity: min(max(base.w, 0), 1), texture: diffuse)
            } else {
                alphaMask = nil
            }
            return try .init(
                geometry: mesh.geometry, worldFromObject: transform, diffuseAlbedo: SIMD3(base.x, base.y, base.z),
                emissiveRadiance: factors.emissiveFactor * factors.emissiveStrength,
                diffuseTexture: diffuse, emissiveTexture: snapshot(material.emissiveTextureBinding), alphaMask: alphaMask
            )
        }
    }

    private static func reportContribution(
        direct: any MTLTexture, indirect: any MTLTexture, label: String,
        rowFraction: Float, albedo: SIMD3<Float>, samples: Int, coverage: (any MTLTexture)? = nil
    ) {
        let rows = Int(Float(direct.height) * rowFraction)
        let region = MTLRegionMake2D(0, 0, direct.width, rows)
        func read(_ texture: any MTLTexture) -> [SIMD4<Float16>] {
            let rowBytes = texture.width * MemoryLayout<SIMD4<Float16>>.stride
            var pixels = [SIMD4<Float16>](repeating: .zero, count: texture.width * rows)
            pixels.withUnsafeMutableBytes {
                texture.getBytes($0.baseAddress!, bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
            }
            return pixels
        }
        var interior: [UInt8]?
        if let coverage {
            var mask = [UInt8](repeating: 0, count: direct.width * rows)
            mask.withUnsafeMutableBytes {
                coverage.getBytes($0.baseAddress!, bytesPerRow: direct.width, from: region, mipmapLevel: 0)
            }
            interior = mask
        }
        var count = 0
        var changed = 0
        var sum = SIMD3<Double>.zero
        var peak = SIMD3<Float>.zero
        for (index, pair) in zip(read(direct), read(indirect)).enumerated() {
            let (before, after) = pair
            guard before.w == 1, interior.map({ $0[index] == 255 }) ?? true else { continue }
            let difference = SIMD3<Float>(SIMD3(after.x, after.y, after.z)) -
                SIMD3<Float>(SIMD3(before.x, before.y, before.z))
            let contribution = difference * (albedo / .pi)
            count += 1
            if contribution != .zero { changed += 1 }
            sum += SIMD3<Double>(contribution)
            peak = simd_max(peak, contribution)
        }
        let mean = sum / Double(max(1, count))
        print("Asset Studio \(label) bake: \(samples) samples, \(changed)/\(count) texels changed")
        print("Added linear RGB radiance: mean \(mean), componentwise peak \(peak)")
    }
}
