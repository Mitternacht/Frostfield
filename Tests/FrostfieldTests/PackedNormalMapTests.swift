import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PackedNormalMapTests {
    enum Format: UInt32, CaseIterable, Sendable {
        case bc5 = 0x8DBD
        case bc5Signed = 0x8DBE
        case eac = 0x9272
        case eacSigned = 0x9273

        var isSigned: Bool { self == .bc5Signed || self == .eacSigned }
        var isEAC: Bool { self == .eac || self == .eacSigned }
        var encoding: PhysicallyBasedMaterial.NormalEncoding { isSigned ? .xySNorm : .xyUNorm }
    }

    typealias Binding = PhysicallyBasedMaterial.TextureBinding
    private let gpu: PathTestFixture
    private let side = 17

    init() throws {
        gpu = try PathTestFixture()
    }

    @Test("KTX1 BC5 and EAC normal maps match RGB surface normals on transformed UV1",
          arguments: Format.allCases)
    func rasterNormals(_ format: Format) throws {
        let pair = try textures(format)
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(0.23, 0.14), rotation: .pi / 2)
        let packed = Binding(
            texture: pair.packed, textureCoordinateSet: .secondary, transform: transform, normalEncoding: format.encoding
        )
        let reference = Binding(texture: pair.rgb, textureCoordinateSet: .secondary, transform: transform)
        let actual = try rasterNormal(try material(normal: packed))
        let expected = try rasterNormal(try material(normal: reference))
        #expect(simd_distance(actual, expected) < 0.003, "\(format): \(actual), \(expected)")
        #expect(simd_length(SIMD2(actual.x, actual.y)) > 0.15 && actual.z > 0.8)

        let unscaled = try rasterNormal(try material(normal: packed, scale: 1))
        #expect(simd_length(SIMD2(unscaled.x, unscaled.y)) > simd_length(SIMD2(actual.x, actual.y)) + 0.06)
        let unrotated = try rasterNormal(try material(normal: .init(texture: pair.packed, normalEncoding: format.encoding)))
        #expect(simd_distance(unrotated, actual) > 0.15)
    }

    @Test("Path tracing reconstructs base normals with full-resolution and baked texture sampling",
          arguments: [false, true])
    func pathNormals(baked: Bool) throws {
        let format: Format = baked ? .eac : .bc5Signed
        let pair = try textures(format)
        let transform = try PhysicallyBasedMaterial.TextureTransform(rotation: 0.45)
        let packed = Binding(
            texture: pair.packed, textureCoordinateSet: .secondary, transform: transform, normalEncoding: format.encoding
        )
        let rgb = Binding(texture: pair.rgb, textureCoordinateSet: .secondary, transform: transform)
        let light = try PathTracer.AnalyticLight(PunctualLight(kind: .directional, intensity: 1))
        let actual = try trace(try material(normal: packed), baked: baked, lights: [light])
        let expected = try trace(try material(normal: rgb), baked: baked, lights: [light])
        #expect(simd_distance(actual.normal, expected.normal) < 0.003, "\(actual.normal), \(expected.normal)")
        #expect(simd_length(SIMD2(actual.normal.x, actual.normal.y)) > 0.15)
        #expect(actual.beauty.x > 0.05 && abs(actual.beauty.x - expected.beauty.x) < 0.003)
    }

    @Test("A packed clearcoat normal produces the same raster and path highlight as RGB")
    func clearcoat() throws {
        let format = Format.bc5Signed
        let pair = try textures(format, signedXAxisOnly: true)
        let scale: Float = 0.7
        let tilted = simd_normalize(SIMD3(pair.local.x * scale, pair.local.y * scale, pair.local.z))
        let mirror = simd_normalize(2 * tilted.z * tilted - SIMD3(0, 0, 1))
        func coat(_ binding: Binding?) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: 1, directionToLight: mirror, lightIntensity: 1, ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque,
                clearcoat: .init(factor: 1, roughness: 0.3, normalTexture: binding, normalScale: scale),
                specular: .init(factor: 0)
            )
        }
        let packed = try coat(.init(texture: pair.packed, normalEncoding: format.encoding))
        let rgb = try coat(.init(texture: pair.rgb))
        let flat = try coat(nil)
        let light = try PathTracer.AnalyticLight(PunctualLight(kind: .directional, intensity: 1), direction: -mirror)
        let traced = try trace(packed, baked: true, lights: [light])
        let tracedRGB = try trace(rgb, baked: true, lights: [light])
        let tracedFlat = try trace(flat, baked: true, lights: [light])
        #expect(abs(traced.beauty.x - tracedRGB.beauty.x) < 0.006)
        #expect(traced.beauty.x > tracedFlat.beauty.x * 2 && traced.beauty.x > 0.03, "packed \(traced.beauty.x), flat \(tracedFlat.beauty.x)")
        #expect(traced.normal == SIMD4(0, 0, 1, 1))

        let raster = try gpu.rasterize(graph(packed), width: side, height: side)[center].x
        let rasterRGB = try gpu.rasterize(graph(rgb), width: side, height: side)[center].x
        let rasterFlat = try gpu.rasterize(graph(flat), width: side, height: side)[center].x
        #expect(abs(raster - rasterRGB) < 0.006)
        #expect(raster > rasterFlat * 2 && raster > 0.03, "packed \(raster), flat \(rasterFlat)")
    }

    @Test("Packed KTX1 normal textures retain format, interpretation and rendered color in an archive",
          arguments: Format.allCases)
    func compressedArchive(_ format: Format) throws {
        let pair = try textures(format)
        let original = try graph(material(normal: .init(texture: pair.packed, normalEncoding: format.encoding)))
        let archive = try SceneArchive(capturing: original, commandQueue: gpu.queue)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        let mesh = try #require(restored.roots.first?.mesh)
        guard case .physicallyBased(let loaded) = mesh.material else { Issue.record("Missing restored PBR material"); return }
        let binding = try #require(loaded.coreMaps.bindings[2])
        #expect(binding.texture.pixelFormat == pair.packed.pixelFormat && binding.normalEncoding == format.encoding)
        let before = try gpu.rasterize(original, width: side, height: side)
        #expect(try gpu.rasterize(restored, width: side, height: side) == before)
    }

    @Test("Signed fallback normals keep their encoding through live factors and scene archives")
    func archive() throws {
        let signed = try #require(makeTexture(.rgba8Snorm, pixel: SIMD4<Int8>(48, -33, 0, 127)))
        let unsigned = try #require(makeTexture(.rgba8Unorm, pixel: SIMD4<UInt8>(160, 96, 0, 255)))
        let base = Binding(texture: signed, normalEncoding: .xySNorm)
        let coat = Binding(texture: unsigned, normalEncoding: .xyUNorm)
        let live = try PhysicallyBasedMaterial(
            normalTexture: base, normalScale: 0.7,
            clearcoat: .init(factor: 1, roughness: 0.25, normalTexture: coat)
        ).withLiveFactors()
        let factors = try #require(live.liveFactors)
        var current = factors.current
        current.textureTransforms[.normal] = try .init(offset: SIMD2(0.25, 0.75), rotation: -0.6)
        try factors.set(current)
        #expect(factors.material.baseNormalEncoding == .xySNorm)
        let original = try graph(live)
        let archive = try SceneArchive(capturing: original, commandQueue: gpu.queue)
        let manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 4)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        let mesh = try #require(restored.roots.first?.mesh)
        guard case .physicallyBased(let loaded) = mesh.material else { Issue.record("Missing restored PBR material"); return }
        #expect(loaded.baseNormalEncoding == .xySNorm && loaded.coatNormalEncoding == .xyUNorm)
        #expect(loaded.liveFactors?.current.textureTransforms[.normal] == current.textureTransforms[.normal])
        let before = try gpu.rasterize(original, width: side, height: side)
        #expect(try gpu.rasterize(restored, width: side, height: side) == before)

        let invalid = try material(normal: .init(texture: unsigned, normalEncoding: .xySNorm))
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidPackedNormalTexture) { try graph(invalid) }
    }

    private var center: Int { side / 2 * side + side / 2 }

    private func material(normal: Binding, scale: Float = 0.65) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            roughness: 0.65, directionToLight: SIMD3(0.4, 0.2, 1), lightIntensity: 1, ambientIntensity: 0,
            normalTexture: normal, normalScale: scale, alphaMode: .opaque
        )
    }

    private func textures(
        _ format: Format, signedXAxisOnly: Bool = false
    ) throws -> (packed: any MTLTexture, rgb: any MTLTexture, local: SIMD3<Float>) {
        let header: UInt8 = format.isEAC ? 0x10 : 0
        let red: [UInt8] = [format.isSigned ? 42 : 169, header, 0, 0, 0, 0, 0, 0]
        let greenValue: UInt8 = format.isSigned ? (signedXAxisOnly ? 0 : 235) : 151
        let green: [UInt8] = [greenValue, header, 0, 0, 0, 0, 0, 0]
        let data = KTX1Fixture.data(gl: format.rawValue, base: 0x8227, width: 4, height: 4, levels: [[red + green]])
        let packed = try TextureLoader(device: gpu.device).loadKTX1(data: data, interpretation: .linearData)
        if !format.isEAC, !gpu.device.supportsBCTextureCompression {
            #expect(packed.pixelFormat == (format.isSigned ? .rgba8Snorm : .rgba8Unorm))
        }
        let texel = try DDSProbe(device: gpu.device).sample(packed, at: SIMD3(0.5, 0.5, 0))
        let channels = SIMD2(texel.x, texel.y)
        let xy = format.isSigned ? channels : channels * 2 - 1
        #expect(xy.x > 0.15)
        if signedXAxisOnly {
            #expect(xy.y == 0)
        } else {
            #expect(format.isSigned ? xy.y < -0.08 : xy.y > 0.08)
        }
        let local = SIMD3(xy.x, xy.y, max(0, 1 - simd_length_squared(xy)).squareRoot())
        let encoded = local * 0.5 + 0.5
        let rgb = try #require(makeTexture(
            .rgba16Float, pixel: SIMD4(Float16(encoded.x), Float16(encoded.y), Float16(encoded.z), 1)
        ))
        return (packed, rgb, local)
    }

    private func makeTexture<T>(_ format: MTLPixelFormat, pixel: T) -> (any MTLTexture)? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = gpu.device.makeTexture(descriptor: descriptor) else { return nil }
        withUnsafeBytes(of: pixel) { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: bytes.count
            )
        }
        return texture
    }

    private func graph(_ material: PhysicallyBasedMaterial) throws -> SceneGraph {
        let plane = try MeshGeometry.plane(width: 4, height: 4)
        let vertices = plane.vertices.map { vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: vertex.textureCoordinate, tangent: vertex.tangent
            )
        }
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: MeshGeometry(vertices: vertices, indices: plane.indices),
            material: .physicallyBased(material)
        )
        return SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: gpu.camera
        )
    }

    private func rasterNormal(_ material: PhysicallyBasedMaterial) throws -> SIMD3<Float> {
        let pass = MTLRenderPassDescriptor()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: side, height: side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let color = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(gpu.device.makeTexture(descriptor: depthDescriptor))
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let command = try #require(gpu.queue.makeCommandBuffer())
        try graph(material).encodeSurfaceAttributes(pass: pass, commandBuffer: command)
        try gpu.finish(command)
        var words = SIMD2<UInt32>.zero
        color.getBytes(&words, bytesPerRow: 8, from: MTLRegionMake2D(side / 2, side / 2, 1, 1), mipmapLevel: 0)
        let x = Float(Int16(bitPattern: UInt16(truncatingIfNeeded: words.x))) / 32767
        let y = Float(Int16(bitPattern: UInt16(truncatingIfNeeded: words.x >> 16))) / 32767
        return simd_normalize(SIMD3(x, y, 1 - abs(x) - abs(y)))
    }

    private func trace(
        _ material: PhysicallyBasedMaterial, baked: Bool, lights: [PathTracer.AnalyticLight] = []
    ) throws -> (beauty: SIMD4<Float>, normal: SIMD4<Float>) {
        let scene = try PathTracingScene(
            queue: gpu.queue, graph: graph(material),
            options: .init(textures: baked ? .baked(size: 16) : .fullResolution(fallbackSize: 16), acceleration: .software)
        )
        let useBakedTexture = baked || gpu.device.argumentBuffersSupport != .tier2
        #expect(scene.statistics.bakedTextureCount == (useBakedTexture ? scene.textureCount : 0))
        #expect(scene.statistics.boundTextureCount == (useBakedTexture ? 0 : scene.textureCount))
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: .init(analyticLights: lights, environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        let beauty = try gpu.output(width: side, height: side)
        let normal = try gpu.output(width: side, height: side)
        let command = try #require(gpu.queue.makeCommandBuffer())
        try tracer.encode(
            camera: gpu.camera, cameraTransform: .init(), destination: beauty, accumulation: accumulation,
            commandBuffer: command, samplesPerPixel: 8, settings: .init(maxBounces: 1, seed: 9), aovs: .init(normal: normal)
        )
        try gpu.finish(command)
        return (try gpu.readback(beauty)[center], try gpu.readback(normal)[center])
    }
}
