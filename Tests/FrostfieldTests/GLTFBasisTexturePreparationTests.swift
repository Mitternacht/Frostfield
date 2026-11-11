import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GLTFBasisTexturePreparationTests {
    private let fixture: PathTestFixture

    init() throws { fixture = try PathTestFixture() }

    private static var supportsCompressedBasis: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple2) || device.supportsBCTextureCompression
    }

    @Test("Prepared glTF Basis maps preserve color and data semantics in raster and path tracing",
          .enabled(if: supportsCompressedBasis, "Requires ASTC, ETC2 or BC7"))
    func rasterAndPath() throws {
        let color = try BasisTextureFixture.data(.etc1s)
        var data = try BasisTextureFixture.data(.uastcZstd)
        let dfd = data.withUnsafeBytes {
            Int(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 48, as: UInt32.self)))
        }
        // The encoded channels stay the same. This fixture uses those values
        // as packed material data, so declare unspecified primaries and linear transfer.
        data[dfd + 13] = 0
        data[dfd + 14] = 1
        let scene = try Self.asset(color: color, data: data)
        let original = try PreparedGLTFScene(device: fixture.device, colorPixelFormat: .rgba16Float, scene: scene)
        let prepared = try PreparedGLTFScene(
            device: fixture.device, colorPixelFormat: .rgba16Float, scene: scene,
            lightIntensity: 0, ambientIntensity: 0, basisTextureStorage: .preferDeviceCompressed
        )
        let defaultGraph = try original.graph(camera: fixture.camera)
        let graph = try prepared.graph(camera: fixture.camera)
        let originalMaterial = try Self.material(defaultGraph)
        let material = try Self.material(graph)
        let base = try #require(material.baseColorTexture)
        let packed = try #require(material.pathTracingMaps[2]?.texture)
        let usesAppleFormats = fixture.device.supportsFamily(.apple2)
        #expect(originalMaterial.baseColorTexture?.pixelFormat == .rgba8Unorm_srgb)
        #expect(originalMaterial.pathTracingMaps[2]?.texture.pixelFormat == .rgba8Unorm)
        #expect(base.pixelFormat == (usesAppleFormats ? .eac_rgba8_srgb : .bc7_rgbaUnorm_srgb))
        #expect(packed.pixelFormat == (usesAppleFormats ? .astc_4x4_ldr : .bc7_rgbaUnorm))
        #expect(base.mipmapLevelCount == 4 && packed.mipmapLevelCount == 4)
        #expect(material.emissiveTextureBinding?.texture === base)

        let side = 33
        let raster = try fixture.rasterize(graph, width: side, height: side)
        let (baked, bakedScene) = try guide(graph, side: side, textures: .baked(size: 16))
        #expect(bakedScene.statistics.bakedTextureCount == 2)
        let direct: [SIMD4<Float>]?
        if fixture.device.argumentBuffersSupport == .tier2 {
            let (albedo, path) = try guide(graph, side: side, textures: .fullResolution(fallbackSize: 16))
            #expect(path.statistics.boundTextureCount == 2 && path.statistics.bakedTextureCount == 0)
            direct = albedo
        } else {
            direct = nil
        }

        let probe = try DDSProbe(device: fixture.device)
        let projectedHalfExtent: Float = 2 * tan(.pi / 6)
        let texels: [(x: Int, y: Int, color: SIMD4<Float>)] = [
            (8, 8, SIMD4(200, 40, 30, 64)), (24, 8, SIMD4(30, 180, 70, 192)),
            (8, 24, SIMD4(40, 50, 220, 128)), (24, 24, SIMD4(220, 205, 35, 255)),
        ]
        for (x, y, code) in texels {
            let pixel = y * side + x
            let uv = SIMD2((Float(x) + 0.5) / Float(side), (Float(y) + 0.5) / Float(side))
            let mapped = (uv - 0.5) * projectedHalfExtent + 0.5
            let baseSample = try probe.sample(base, at: SIMD3(mapped.x, mapped.y, 0))
            let packedSample = try probe.sample(packed, at: SIMD3(mapped.x, mapped.y, 0))
            let rgba = code / 255
            let expected = SIMD4(Self.linear(rgba.x), Self.linear(rgba.y), Self.linear(rgba.z), rgba.w)
            #expect(simd_reduce_max(abs(baseSample - expected)) < 0.08)
            #expect(simd_reduce_max(abs(packedSample - rgba)) < 0.06)
            let baseRGB = SIMD3(baseSample.x, baseSample.y, baseSample.z)
            #expect(simd_reduce_max(abs(SIMD3(raster[pixel].x, raster[pixel].y, raster[pixel].z) - baseRGB)) < 0.02)
            let guidePixel = SIMD4(baseRGB, packedSample.y)
            #expect(simd_reduce_max(abs(baked[pixel] - guidePixel)) < 0.02)
            if let direct {
                #expect(simd_reduce_max(abs(direct[pixel] - guidePixel)) < 0.02)
            }
        }
    }

    @Test("glTF Basis compression preference falls back for generated mips and preserves source fallback",
          .enabled(if: supportsCompressedBasis, "Requires ASTC or BC7"))
    func fallback() throws {
        // Pinned Khronos UASTC reference with only a base level.
        let source = try #require(Data(base64Encoded: Self.singleLevel))
        let mipmapped = try Self.asset(color: source)
        let nonMipmapped = try Self.asset(color: source, minFilter: 9728)
        let description = try #require(mipmapped.materials.first)
        let defaultTexture = try #require(description.prepare(device: fixture.device).baseColorTexture)
        let preferred = try description.prepare(device: fixture.device, basisTextureStorage: .preferDeviceCompressed)
        let regenerated = try #require(preferred.baseColorTexture)
        #expect(regenerated.pixelFormat == .rgba8Unorm_srgb && regenerated.mipmapLevelCount == 4)
        let probe = try DDSProbe(device: fixture.device)
        #expect(try probe.sample(regenerated, at: SIMD3(0.5, 0.5, 0), level: 3)
            == probe.sample(defaultTexture, at: SIMD3(0.5, 0.5, 0), level: 3))

        let single = try #require(nonMipmapped.materials.first).prepare(
            device: fixture.device, basisTextureStorage: .preferDeviceCompressed
        )
        let expected: MTLPixelFormat = fixture.device.supportsFamily(.apple2) ? .astc_4x4_srgb : .bc7_rgbaUnorm_srgb
        #expect(single.baseColorTexture?.pixelFormat == expected && single.baseColorTexture?.mipmapLevelCount == 1)

        let damaged = try BasisTextureFixture.corruptZstd(BasisTextureFixture.data(.uastcZstd))
        let optional = try #require(Self.asset(color: damaged, fallback: true).materials.first)
        let fallback = try #require(optional.prepare(device: fixture.device, basisTextureStorage: .preferDeviceCompressed).baseColorTexture)
        #expect(fallback.pixelFormat == .rgba8Unorm_srgb && fallback.width == 1 && fallback.height == 1)
        let required = try #require(Self.asset(color: damaged).materials.first)
        #expect(throws: TextureLoader.Failure.invalidKTX2("Basis Universal could not decode level 0")) {
            try required.prepare(device: fixture.device, basisTextureStorage: .preferDeviceCompressed)
        }
    }

    private func guide(
        _ graph: SceneGraph, side: Int, textures: PathTracingScene.Options.Textures
    ) throws -> ([SIMD4<Float>], PathTracingScene) {
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(textures: textures))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: .init(environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        let albedo = try fixture.output(width: side, height: side)
        let guides = PathTracer.Guides(
            normalDepth: try fixture.output(width: side, height: side), albedoRoughness: albedo
        )
        let command = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: graph.camera, cameraTransform: graph.cameraTransform,
            destination: try fixture.output(width: side, height: side), accumulation: accumulation, guides: guides,
            commandBuffer: command, settings: .init(maxBounces: 1, seed: 3)
        )
        try fixture.finish(command)
        return (try fixture.readback(albedo), scene)
    }

    private static func material(_ graph: SceneGraph) throws -> PhysicallyBasedMaterial {
        let renderer = try #require(graph.roots.first?.mesh)
        if case .physicallyBased(let material) = renderer.material { return material }
        throw FixtureFailure.expectedPBR
    }

    private enum FixtureFailure: Error { case expectedPBR }

    private static func asset(
        color: Data, data: Data? = nil, minFilter: Int = 9984, fallback: Bool = false
    ) throws -> GLTFScene {
        var document = GLTFMotionFixture()
        let positions: [Float] = [-1, 1, 0, -1, -1, 0, 1, 1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0]
        let texcoords: [Float] = [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
        let position = document.floats(positions, type: "VEC3", count: 6)
        let normal = document.floats(Array(repeating: [0, 0, 1], count: 6).flatMap { $0 }, type: "VEC3", count: 6)
        let uv = document.floats(texcoords, type: "VEC2", count: 6)
        let name = "KHR_texture_basisu"
        var images = [["uri": "data:image/ktx2;base64,\(color.base64EncodedString())"]]
        var textures: [[String: Any]] = [["extensions": [name: ["source": 0]], "sampler": 0]]
        var maps: [String: Any] = ["baseColorTexture": ["index": 0], "roughnessFactor": 1, "metallicFactor": 0]
        if let data {
            images.append(["uri": "data:image/ktx2;base64,\(data.base64EncodedString())"])
            textures.append(["extensions": [name: ["source": 1]], "sampler": 0])
            maps["metallicRoughnessTexture"] = ["index": 1]
        }
        if fallback {
            textures[0]["source"] = images.count
            images.append(["uri": "data:image/png;base64,\(Self.png)"])
        }
        let gltf = try document.document([
            "nodes": [["mesh": 0, "translation": [0, 0, -2]]],
            "meshes": [["primitives": [[
                "attributes": ["POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv], "material": 0,
            ]]]],
            "materials": [[
                "pbrMetallicRoughness": maps, "emissiveTexture": ["index": 0], "emissiveFactor": [1, 1, 1],
            ]],
            "textures": textures, "images": images,
            "samplers": [["magFilter": 9728, "minFilter": minFilter, "wrapS": 33071, "wrapT": 33071]],
            "extensionsUsed": [name], "extensionsRequired": fallback ? [] : [name],
        ])
        return try GLTFMeshLoader.loadScene(data: gltf, includeMaterials: true)
    }

    private static func linear(_ code: Float) -> Float {
        code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4)
    }

    private static let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="
    private static let singleLevel = "q0tUWCAyMLsNChoKAAAAAAEAAAAIAAAACAAAAAAAAAAAAAAAAQAAAAEAAAAAAAAAaAAAACwAAACUAAAAMAAAAAAAAAAAAAAAAAAAAAAAAADQAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAAsAAAAAAAAAAIAKACmAQIAAwMAABAAAAAAAAAAAAB/AwAAAAAAAAAA/////ywAAABLVFh3cml0ZXIAa3R4IGNyZWF0ZSB2NC40LjAgLyBsaWJrdHggdjQuNC4wAAAAAAAAAAAAAAAAABcZhGImtQIAAAAAAAAAAAAXGYRiJrUCAAAAAAAAAAAAFxmEYia1AgAAAAAAAAAAABcZhGImtQIAAAAAAAAAAAA="
}
