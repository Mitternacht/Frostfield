import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PBRLightMapTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("PBR light maps select either UV set, transform the atlas, and add independent linear irradiance")
    func authoredMap() throws {
        let left = SIMD3<Float>(2, 0.25, 0)
        let right = SIMD3<Float>(0, 1, 3)
        let atlas = try gpu.color(width: 2, height: 1) { x, _ in SIMD4(x == 0 ? left : right, 0) }
        let uv0 = try PhysicallyBasedMaterial.LightMap(texture: atlas, textureCoordinateSet: .primary)
        let uv1 = try PhysicallyBasedMaterial.LightMap(texture: atlas)
        let transformed = try PhysicallyBasedMaterial.LightMap(
            texture: atlas, transform: .init(offset: SIMD2(-0.5, 0))
        )
        let geometry = try plane { _ in (SIMD2(0.25, 0.5), SIMD2(0.75, 0.5)) }
        let albedo = SIMD3<Float>(0.5, 0.25, 0.125)
        func sample(
            _ map: PhysicallyBasedMaterial.LightMap?, scale: Float = 1, direct: Float = 0, metallic: Float = 0
        ) throws -> SIMD4<Float> {
            let material = try PhysicallyBasedMaterial(
                metallic: metallic, directionToLight: SIMD3(0, 0, 1), lightIntensity: direct, ambientIntensity: 0,
                baseColorFactor: SIMD4(albedo, 1), lightMap: map, lightMapIntensity: scale
            )
            return try center(graph(geometry, material: material))
        }
        let diffuse = albedo * (0.96 / Float.pi)
        #expect(PostProcessFixture.close(try sample(uv0), SIMD4(left * diffuse, 1), tolerance: 0.0006))
        let secondary = try sample(uv1, scale: 2)
        #expect(PostProcessFixture.close(secondary, SIMD4(right * diffuse * 2, 1), tolerance: 0.0006))
        #expect(PostProcessFixture.close(try sample(transformed), SIMD4(left * diffuse, 1), tolerance: 0.0006))
        let lit = try sample(nil, direct: 1)
        #expect(try sample(uv1, scale: 0, direct: 1) == lit)
        let mixed = try sample(uv1, scale: 2, direct: 1)
        #expect(PostProcessFixture.close(mixed, lit + secondary - SIMD4(0, 0, 0, 1), tolerance: 0.001))
        #expect(try sample(uv1, metallic: 1) == SIMD4(0, 0, 0, 1))
    }

    @Test("Atlas minification blends authored mip levels without a new sampler slot")
    func mipFiltering() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 8, height: 8, mipmapped: true
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        for mip in 0..<texture.mipmapLevelCount {
            let side = 8 >> mip
            let color: SIMD4<Float16> = mip == 0 ? SIMD4(2, 0, 0, 1) : SIMD4(0, 2, 0, 1)
            let pixels = Array(repeating: color, count: side * side)
            pixels.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: mip,
                    withBytes: bytes.baseAddress!, bytesPerRow: side * MemoryLayout<SIMD4<Float16>>.stride
                )
            }
        }
        let map = try PhysicallyBasedMaterial.LightMap(
            texture: texture, transform: .init(scale: SIMD2(repeating: sqrt(Float(2))))
        )
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, lightMap: map
        )
        let geometry = try plane { ($0, $0) }
        let expected = SIMD4<Float>(0.96 / .pi, 0.96 / .pi, 0, 1)
        #expect(PostProcessFixture.close(try center(graph(geometry, material: material)), expected, tolerance: 0.001))
    }

    @Test("Baked UV1 irradiance survives live factors and archive restoration in the shadowed PBR shader")
    func bakedArchive() throws {
        let geometry = try plane { ($0, $0) }
        let irradiance = SIMD3<Float>(2, 1, 0.5)
        let baked = try MeshLightmapBaker(
            device: gpu.device, geometry: geometry, width: 8, height: 8,
            lights: [.init(directionToLight: SIMD3(0, 0, 1), irradiance: irradiance)]
        )
        let map = try baked.pbrLightMapBinding()
        #expect(map.texture === baked.irradianceTexture && map.textureCoordinateSet == .secondary)
        let surface = try SurfaceShaderGraph { graph in .init(roughness: graph.constant(Float(0.5))) }
        let authored = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, receivesShadows: true,
            baseColorFactor: SIMD4(0.25, 0.5, 0.125, 1), surfaceGraph: surface, specular: .init(factor: 0),
            lightMap: map, lightMapIntensity: 0.5
        ).withLiveFactors()
        let live = try #require(authored.liveFactors)
        let changed = SIMD3<Float>(0.5, 0.25, 0.75)
        try live.update { $0.baseColorFactor = SIMD4(changed, 1) }
        let scene = try graph(geometry, material: authored)
        let archive = try SceneArchive(capturing: scene)
        let json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(json["version"] as? Int == 22)
        let restored = try archive.load(device: gpu.device).graph
        let shadow = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 8, directionToLight: SIMD3(0, 0, 1),
            center: SIMD3(0, 0, -1), halfExtent: 2, intensity: 0
        )
        let before = try center(scene, shadow: shadow)
        let after = try center(restored, shadow: shadow)
        #expect(before == after)
        #expect(PostProcessFixture.close(after, SIMD4(irradiance * changed * (0.5 / Float.pi), 1), tolerance: 0.0006))

        let exported = try GLTFExporter().export(scene, includesCamera: false)
        #expect(exported.warnings.contains(.materialLayersDropped(material: 0, layers: ["light map"])))
    }

    @Test("Light maps reject non-linear inputs, invalid intensity, and missing UV1")
    func validation() throws {
        let srgb = try gpu.color(width: 1, height: 1, pixelFormat: .rgba8Unorm_srgb)
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidLightMap) {
            try PhysicallyBasedMaterial.LightMap(texture: srgb)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidLightMapIntensity) {
            try PhysicallyBasedMaterial(lightMapIntensity: .nan)
        }
        let texture = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(repeating: 1) }
        let map = try PhysicallyBasedMaterial.LightMap(texture: texture)
        let geometry = try MeshGeometry.plane(width: 2, height: 2)
        #expect(throws: MeshRenderer.Failure.missingSecondaryTextureCoordinates) {
            try graph(geometry, material: .init(lightMap: map))
        }
        _ = try graph(geometry, material: .init(lightMap: map, lightMapIntensity: 0))
    }

    private func plane(_ coordinates: (SIMD2<Float>) -> (SIMD2<Float>, SIMD2<Float>)) throws -> MeshGeometry {
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        return try MeshGeometry(vertices: plane.vertices.map { vertex in
            let (primary, secondary) = coordinates(vertex.textureCoordinate)
            return MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: primary, secondaryTextureCoordinate: secondary
            )
        }, indices: plane.indices)
    }

    private func graph(_ geometry: MeshGeometry, material: PhysicallyBasedMaterial) throws -> SceneGraph {
        let renderer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        return SceneGraph(
            roots: [.init(transform: .init(translation: SIMD3(0, 0, -1)), mesh: renderer)],
            camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        )
    }

    private func center(_ graph: SceneGraph, shadow: DirectionalShadowMap? = nil) throws -> SIMD4<Float> {
        let color = try gpu.color(width: 8, height: 8)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: 8, height: 8, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        let depth = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        try gpu.run { buffer in
            _ = try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
        }
        return try gpu.read(color)[4 * 8 + 4]
    }
}
