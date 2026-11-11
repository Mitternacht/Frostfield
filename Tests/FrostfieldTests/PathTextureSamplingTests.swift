import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import simd

private typealias Sampling = PhysicallyBasedMaterial.TextureSampling

struct TextureSamplingDescriptionTests {
    @Test("Texture sampling descriptions produce the matching Metal sampler settings")
    func descriptors() {
        let defaults = Sampling.default.descriptor
        #expect(defaults.sAddressMode == .repeat && defaults.tAddressMode == .repeat)
        #expect(defaults.magFilter == .linear && defaults.minFilter == .linear && defaults.mipFilter == .linear)
        let custom = Sampling(
            addressU: .clampToEdge, addressV: .mirroredRepeat, magnification: .nearest, minification: .linear,
            mipFilter: .notMipmapped
        ).descriptor
        #expect(custom.sAddressMode == .clampToEdge && custom.tAddressMode == .mirrorRepeat)
        #expect(custom.magFilter == .nearest && custom.minFilter == .linear && custom.mipFilter == .notMipmapped)
        #expect(Sampling(minification: .nearest, mipFilter: .nearest).descriptor.mipFilter == .nearest)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathTextureSamplingTests {
    private let fixture: MaterialRenderFixture
    private let camera: PerspectiveCamera
    private let side = MaterialRenderFixture.side
    // Red and green on the top row, blue and white below.
    private static let colors: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 1, 1)]

    init() throws {
        fixture = try MaterialRenderFixture()
        // A 2 × 2 plane at distance 1 fills the 90° view; pixel centers sample UV ((x + 0.5) / 33, (y + 0.5) / 33).
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
    }

    /// A 2 × 2 checker of `block` × `block` texel squares in `colors`.
    private static func checkerPixels(block: Int) -> [UInt8] {
        var pixels: [UInt8] = []
        for y in 0..<(2 * block) {
            for x in 0..<(2 * block) {
                let color = colors[(y / block) * 2 + x / block]
                pixels += [UInt8(color.x * 255), UInt8(color.y * 255), UInt8(color.z * 255), 255]
            }
        }
        return pixels
    }

    private static func address(_ coordinate: Float, _ mode: Sampling.Addressing) -> Float {
        switch mode {
        case .repeat: coordinate - coordinate.rounded(.down)
        case .clampToEdge: min(max(coordinate, 0), 1)
        case .mirroredRepeat: 1 - abs((coordinate * 0.5 - (coordinate * 0.5).rounded(.down)) * 2 - 1)
        }
    }

    /// The checker color at a transformed coordinate, or nil when a filter footprint can reach a block edge.
    private static func expectedColor(
        _ mapped: SIMD2<Float>, u: Sampling.Addressing, v: Sampling.Addressing, margin: Float = 0.3
    ) -> SIMD4<Float>? {
        let cell = mapped * 2 - (mapped * 2).rounded(.down)
        guard (0..<2).allSatisfy({ min(cell[$0], 1 - cell[$0]) > margin }) else { return nil }
        let folded = SIMD2(address(mapped.x, u), address(mapped.y, v))
        let block = min(Int(folded.y * 2), 1) * 2 + min(Int(folded.x * 2), 1)
        return SIMD4(colors[block], 1)
    }

    @Test(
        "Clamp, repeat and mirrored addressing match per axis in raster and the path tracer, sampled directly or baked",
        arguments: [false, true]
    )
    func addressModes(baked: Bool) throws {
        let options = PathTracingScene.Options(textures: baked ? .baked(size: 16) : .fullResolution(fallbackSize: 16))
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let placement = SceneTransform(translation: SIMD3(0, 0, -1))
        let texture = try fixture.dataTexture(width: 16, height: 16, pixels: Self.checkerPixels(block: 8))
        // Scale and offset spread the plane over UV -1 to 2 on both axes.
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(-1, -1), scale: SIMD2(3, 3))
        let lighting = try PathTracer.Lighting(environment: .zero)
        let modes: [Sampling.Addressing] = [.repeat, .clampToEdge, .mirroredRepeat]
        var combination = 0
        for u in modes {
            for v in modes {
                let sampling = Sampling(addressU: u, addressV: v)
                let material: PhysicallyBasedMaterial
                if combination.isMultiple(of: 2) {
                    // A description alone: raster creates the sampler from it.
                    material = try PhysicallyBasedMaterial(
                        baseColorTexture: texture, alphaMode: .opaque, baseColorTextureTransform: transform,
                        shadingModel: .unlit, baseColorSampling: sampling
                    )
                } else {
                    // An emissive layer whose binding carries both the sampler and its description.
                    let binding = try PhysicallyBasedMaterial.TextureBinding(
                        texture: texture, sampling: sampling, device: fixture.device, transform: transform
                    )
                    #expect(binding.sampler != nil && binding.sampling == sampling)
                    material = try PhysicallyBasedMaterial(
                        lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                        emissiveTexture: binding, emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque
                    )
                }
                combination += 1
                let raster = try fixture.raster(material, geometry: plane, transform: placement, camera: camera)
                let traced = try fixture.trace(
                    [(plane, material, placement)], camera: camera, lighting: lighting, samples: 8, bounces: 1,
                    textureSize: 16, options: options
                )
                var checked = 0
                var outside = 0
                for y in 0..<side {
                    for x in 0..<side {
                        let mapped = transform.apply(to: SIMD2(Float(x) + 0.5, Float(y) + 0.5) / Float(side))
                        guard let expected = Self.expectedColor(mapped, u: u, v: v) else { continue }
                        let index = y * side + x
                        #expect(simd_distance(raster[index], expected) < 0.01, "raster \(u), \(v) at \(x), \(y)")
                        #expect(simd_distance(traced[index], expected) < 0.01, "path \(u), \(v) at \(x), \(y)")
                        checked += 1
                        if (0..<2).contains(where: { mapped[$0] < 0 || mapped[$0] > 1 }) { outside += 1 }
                    }
                }
                #expect(checked > 120, "\(u), \(v): \(checked)")
                // Most checked pixels sample outside [0, 1], where the address modes differ.
                #expect(outside > checked / 2, "\(u), \(v)")
            }
        }

        // The default description packs to zero bits, which traces exactly like no description.
        func tracedDefault(_ sampling: Sampling?) throws -> [SIMD4<Float>] {
            let material = try PhysicallyBasedMaterial(
                baseColorTexture: texture, alphaMode: .opaque, baseColorTextureTransform: transform,
                shadingModel: .unlit, baseColorSampling: sampling
            )
            return try fixture.trace(
                [(plane, material, placement)], camera: camera, lighting: lighting, samples: 4, bounces: 1,
                textureSize: 16, options: options
            )
        }
        #expect(try tracedDefault(.default) == tracedDefault(nil))
    }

    @Test(
        "Nearest magnification keeps a 2 × 2 checker hard-edged in both renderers, sampled directly or baked; linear blends it",
        arguments: [false, true]
    )
    func nearestAndLinear(baked: Bool) throws {
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let placement = SceneTransform(translation: SIMD3(0, 0, -1))
        let texture = try fixture.dataTexture(width: 2, height: 2, pixels: Self.checkerPixels(block: 1))
        let lighting = try PathTracer.Lighting(environment: .zero)
        let center = side / 2

        for filter in [Sampling.Filtering.nearest, .linear] {
            let sampling = Sampling(addressU: .clampToEdge, addressV: .clampToEdge, magnification: filter)
            let unlit = try PhysicallyBasedMaterial(
                baseColorTexture: texture, baseColorSampler: try sampling.makeSamplerState(device: fixture.device),
                alphaMode: .opaque, shadingModel: .unlit, baseColorSampling: sampling
            )
            let emissive = try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveTexture: .init(texture: texture, sampling: sampling, device: fixture.device),
                emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque
            )
            for material in [unlit, emissive] {
                let raster = try fixture.raster(material, geometry: plane, transform: placement, camera: camera)
                let traced = try fixture.trace(
                    [(plane, material, placement)], camera: camera, lighting: lighting, samples: 8, bounces: 1,
                    options: .init(textures: baked ? .baked(size: 64) : .fullResolution(fallbackSize: 64))
                )
                if filter == .nearest {
                    // Every pixel off the center row and column shows one texel's exact color.
                    for y in 0..<side where y != center {
                        for x in 0..<side where x != center {
                            let expected = SIMD4(Self.colors[(y > center ? 2 : 0) + (x > center ? 1 : 0)], 1)
                            let index = y * side + x
                            #expect(simd_distance(raster[index], expected) < 0.01, "raster \(x), \(y)")
                            #expect(simd_distance(traced[index], expected) < 0.01, "path \(x), \(y)")
                        }
                    }
                } else {
                    // Next to the center, linear filtering mixes all four texels in both renderers.
                    let index = (center - 1) * side + center - 1
                    for image in [raster, traced] {
                        let nearestPure = Self.colors.map { simd_distance(SIMD3(image[index].x, image[index].y, image[index].z), $0) }.min()!
                        #expect(nearestPure > 0.3)
                    }
                    #expect(PathTestFixture.maximumDifference(raster, traced) < 0.05)
                }
            }
        }
    }

    @Test("A baked nearest map larger than the bake keeps hard texel edges where it is magnified and drops no texels where it shrinks")
    func nearestBakeKeepsTexels() throws {
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let placement = SceneTransform(translation: SIMD3(0, 0, -1))
        // 32 × 2 texels: one-texel white and black columns on top, blue below. A 16-texel bake
        // halves the width and magnifies the height eightfold.
        var pixels: [UInt8] = []
        for x in 0..<32 { pixels += x.isMultiple(of: 2) ? [255, 255, 255, 255] : [0, 0, 0, 255] }
        for _ in 0..<32 { pixels += [0, 0, 255, 255] }
        let texture = try fixture.dataTexture(width: 32, height: 2, pixels: pixels)
        let sampling = Sampling(addressU: .clampToEdge, addressV: .clampToEdge, magnification: .nearest)
        let material = try PhysicallyBasedMaterial(
            baseColorTexture: texture, alphaMode: .opaque, shadingModel: .unlit, baseColorSampling: sampling
        )
        let raster = try fixture.raster(material, geometry: plane, transform: placement, camera: camera)
        let traced = try fixture.trace(
            [(plane, material, placement)], camera: camera, lighting: PathTracer.Lighting(environment: .zero),
            samples: 8, bounces: 1, options: .init(textures: .baked(size: 16))
        )
        let center = side / 2
        let blue = SIMD4<Float>(0, 0, 1, 1)
        for y in 0..<side where y != center {
            let row = (0..<side).map { y * side + $0 }
            if y < center {
                // The bake averages each pair of columns instead of keeping one of them.
                for index in row {
                    #expect(simd_distance(traced[index], SIMD4(0.5, 0.5, 0.5, 1)) < 0.01, "path \(index % side), \(y)")
                }
                let mean = row.reduce(Float(0)) { $0 + raster[$1].x } / Float(side)
                #expect(abs(mean - 0.5) < 0.05, "raster row \(y)")
            } else {
                // Rows next to the center stay pure blue: the magnified axis keeps nearest texels.
                for index in row {
                    #expect(simd_distance(traced[index], blue) < 0.01, "path \(index % side), \(y)")
                    #expect(simd_distance(raster[index], blue) < 0.01, "raster \(index % side), \(y)")
                }
            }
        }
    }

    @Test("Each base and layer map packs its own addressing and filter bits")
    func layerSamplingBits() throws {
        let texture = try fixture.dataTexture(width: 1, height: 1, pixels: [255, 255, 255, 255])
        let modes: [Sampling.Addressing] = [.repeat, .clampToEdge, .mirroredRepeat]
        // Maps 0 base color, 1 normal, 2 metallic-roughness, 3 emissive, then 4 + layer slot: clearcoat
        // factor and roughness, transmission, specular and specular color, sheen color and roughness,
        // anisotropy, iridescence and iridescence thickness. Each gets a different description, except
        // that the iridescence maps share one to stay within the raster limit of six layer samplers.
        func sampling(_ map: Int) -> Sampling {
            let combination = (min(map, 12) * 5) % 18
            return Sampling(
                addressU: modes[combination % 3], addressV: modes[combination / 3 % 3],
                magnification: combination >= 9 ? .nearest : .linear
            )
        }
        func binding(_ map: Int) -> PhysicallyBasedMaterial.TextureBinding {
            .init(texture: texture, sampling: sampling(map))
        }
        // Raster draws bind at most 16 samplers, so the maps split over two materials.
        let base = try PhysicallyBasedMaterial(
            baseColorTexture: texture, metallicRoughnessTexture: binding(2), normalTexture: binding(1),
            emissiveTexture: binding(3), emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque,
            clearcoat: .init(factor: 1, factorTexture: binding(4), roughnessTexture: binding(5)),
            transmission: .init(factor: 0.5, factorTexture: binding(6)), baseColorSampling: sampling(0)
        )
        let layered = try PhysicallyBasedMaterial(
            baseColorTexture: texture, alphaMode: .opaque,
            specular: .init(factorTexture: binding(7), colorTexture: binding(8)),
            sheen: .init(colorFactor: SIMD3(repeating: 1), colorTexture: binding(9), roughnessTexture: binding(10)),
            anisotropy: .init(strength: 0.5, texture: binding(11)),
            iridescence: .init(factor: 1, factorTexture: binding(12), thicknessTexture: binding(13)),
            baseColorSampling: sampling(0)
        )
        // Addressing codes: 0 repeat, 1 clamp to edge, 2 mirrored repeat.
        func code(_ addressing: Sampling.Addressing) -> UInt32 { UInt32(modes.firstIndex(of: addressing)!) }
        for (material, maps) in [(base, Set(0...6)), (layered, Set([0] + Array(7...13)))] {
            let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1)), mesh: try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.plane(),
                material: .physicallyBased(material)
            ))
            let scene = try PathTracingScene(
                queue: fixture.queue, graph: SceneGraph(roots: [node], camera: camera), textureSize: 16
            )
            try #require(scene.materialCount == 1)
            let packed = scene.materials.contents().load(as: PathTracingScene.Material.self).extensions.coordinates
            // Maps without a texture, including the unused slots 10 and 11, keep zero bits.
            for map in 0..<16 {
                let expected = maps.contains(map) ? sampling(map) : .default
                let shift = UInt32(2 * map)
                #expect((packed.y >> shift) & 3 == code(expected.addressU), "map \(map) U")
                #expect((packed.z >> shift) & 3 == code(expected.addressV), "map \(map) V")
                #expect((packed.w >> UInt32(map)) & 1 == (expected.magnification == .nearest ? 1 : 0), "map \(map) filter")
            }
        }
    }

    @Test("A layer map samples with its own addressing, independent of the base color map")
    func layerMapAddressing() throws {
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let placement = SceneTransform(translation: SIMD3(0, 0, -1))
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(-1, -1), scale: SIMD2(3, 3))
        // A white environment reflects off the clearcoat, whose factor map reads the checker's red channel.
        let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 10))
        let white = try fixture.dataTexture(width: 1, height: 1, pixels: [255, 255, 255, 255])
        let layerSampling = Sampling(addressU: .clampToEdge, addressV: .mirroredRepeat, magnification: .nearest)
        func uniform(_ red: UInt8) throws -> any MTLTexture {
            try fixture.dataTexture(width: 16, height: 16, pixels: Array(repeating: [red, 0, 0, 255], count: 256).flatMap { $0 })
        }

        func traced(_ factorMap: any MTLTexture) throws -> [SIMD4<Float>] {
            // The base color map repeats with linear filtering; only the layer clamps and mirrors.
            let material = try PhysicallyBasedMaterial(
                baseColorTexture: white, roughness: 1, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque,
                clearcoat: .init(factor: 1, factorTexture: .init(texture: factorMap, transform: transform, sampling: layerSampling)),
                baseColorTextureTransform: transform, baseColorSampling: .default
            )
            return try fixture.trace(
                [(plane, material, placement)], camera: camera, lighting: lighting, samples: 4, bounces: 1,
                textureSize: 16
            )
        }
        let checker = try traced(try fixture.dataTexture(width: 16, height: 16, pixels: Self.checkerPixels(block: 8)))
        // The same seed gives identical paths wherever the sampled factor is the same.
        let references = [try traced(try uniform(0)), try traced(try uniform(255))]
        let contrast = zip(references[0], references[1]).reduce(Float(0)) { $0 + abs($1.1.x - $1.0.x) } / Float(side * side)
        #expect(contrast > 0.1)

        var checked = 0
        var outside = 0
        for y in 0..<side {
            for x in 0..<side {
                let mapped = transform.apply(to: SIMD2(Float(x) + 0.5, Float(y) + 0.5) / Float(side))
                guard let expected = Self.expectedColor(mapped, u: .clampToEdge, v: .mirroredRepeat) else { continue }
                let index = y * side + x
                let reference = references[expected.x > 0.5 ? 1 : 0][index]
                #expect(simd_distance(checker[index], reference) < 1e-3, "\(x), \(y): \(expected)")
                checked += 1
                if (0..<2).contains(where: { mapped[$0] < 0 || mapped[$0] > 1 }) { outside += 1 }
            }
        }
        #expect(checked > 120 && outside > checked / 2)
    }

    @Test("Scene graph export writes each map's sampling description as its glTF sampler")
    func exportsDescribedSamplers() throws {
        let texture = try fixture.dataTexture(width: 1, height: 1, pixels: [10, 20, 30, 255])
        let color = Sampling(
            addressU: .clampToEdge, addressV: .mirroredRepeat, magnification: .nearest, minification: .nearest,
            mipFilter: .notMipmapped
        )
        let emissive = Sampling(addressV: .clampToEdge, minification: .nearest, mipFilter: .nearest)
        let material = try PhysicallyBasedMaterial(
            baseColorTexture: texture,
            emissiveTexture: .init(texture: texture, sampling: emissive, device: fixture.device),
            emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque, baseColorSampling: color
        )
        let node = SceneNode(mesh: try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.plane(),
            material: .physicallyBased(material)
        ))
        let output = try GLTFExporter(commandQueue: fixture.queue).export(
            SceneGraph(roots: [node], camera: camera), includesCamera: false
        )
        // Described samplers export without the unavailable-sampler warning.
        #expect(output.warnings.isEmpty)
        let imported = try #require(GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true).materials.first)
        let base = try #require(imported.baseColorTexture?.sampling)
        #expect(base.wrapS == .clampToEdge && base.wrapT == .mirroredRepeat)
        #expect(base.magnification == .nearest && base.minification == .nearest)
        let glow = try #require(imported.emissiveTexture?.sampling)
        #expect(glow.wrapS == .repeat && glow.wrapT == .clampToEdge)
        #expect(glow.magnification == .linear && glow.minification == .nearestMipmapNearest)
        #expect(try imported.prepare(device: fixture.device).baseColorSampling == color)
    }

    @Test("A custom sampler without a description is still rejected")
    func undescribedSampler() throws {
        let texture = try fixture.dataTexture(width: 1, height: 1, pixels: [255, 255, 255, 255])
        let sampler = try #require(fixture.device.makeSamplerState(descriptor: MTLSamplerDescriptor()))
        let plane = try MeshGeometry.plane()
        let materials = [
            try PhysicallyBasedMaterial(baseColorTexture: texture, baseColorSampler: sampler, alphaMode: .opaque),
            try PhysicallyBasedMaterial(alphaMode: .opaque, specular: .init(factorTexture: .init(texture: texture, sampler: sampler))),
        ]
        for material in materials {
            let id = UUID()
            let node = SceneNode(id: id, transform: SceneTransform(translation: SIMD3(0, 0, -1)), mesh: try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(material)
            ))
            let failure = PathTracingScene.Failure.unsupported(id, "custom texture samplers without a TextureSampling description")
            #expect(throws: failure) {
                try PathTracingScene(queue: fixture.queue, graph: SceneGraph(roots: [node], camera: camera), textureSize: 16)
            }
            #expect(failure.errorDescription?.contains("TextureSampling") == true)
        }
        // The same sampler with its description traces.
        let described = try PhysicallyBasedMaterial(
            baseColorTexture: texture, baseColorSampler: sampler, alphaMode: .opaque, baseColorSampling: .default
        )
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1)), mesh: try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(described)
        ))
        _ = try PathTracingScene(queue: fixture.queue, graph: SceneGraph(roots: [node], camera: camera), textureSize: 16)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathImportedMaterialTests {
    private static let side = 33
    private static let colors: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 1, 1)]
    private let fixture: PathTestFixture
    private let camera: PerspectiveCamera

    init() throws {
        fixture = try PathTestFixture()
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
    }

    /// A 16 × 16 sRGB PNG checker of 8 × 8 squares in `colors`.
    private static func checkerPNG() throws -> Data {
        var pixels: [UInt8] = []
        for y in 0..<16 {
            for x in 0..<16 {
                let color = colors[(y / 8) * 2 + x / 8]
                pixels += [UInt8(color.x * 255), UInt8(color.y * 255), UInt8(color.z * 255), 255]
            }
        }
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let image = try #require(CGImage(
            width: 16, height: 16, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 64, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        ))
        let output = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(output as CFMutableData, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        try #require(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// A 2 × 2 quad at z = -1 facing the camera, with UV (0, 0) at its top-left corner. Every map
    /// shares one texture whose sampler clamps U and mirrors V with nearest filtering, and whose
    /// KHR_texture_transform spreads the quad over UV -1 to 2.
    private func asset(lit: Bool) throws -> Data {
        var motion = GLTFMotionFixture()
        let corners: [(SIMD3<Float>, SIMD2<Float>)] = [
            (SIMD3(-1, 1, 0), SIMD2(0, 0)), (SIMD3(-1, -1, 0), SIMD2(0, 1)), (SIMD3(1, 1, 0), SIMD2(1, 0)),
            (SIMD3(1, 1, 0), SIMD2(1, 0)), (SIMD3(-1, -1, 0), SIMD2(0, 1)), (SIMD3(1, -1, 0), SIMD2(1, 1)),
        ]
        let position = motion.floats(corners.flatMap { [$0.0.x, $0.0.y, $0.0.z] }, type: "VEC3", count: 6)
        let normal = motion.floats(Array(repeating: [0, 0, 1], count: 6).flatMap { $0 }, type: "VEC3", count: 6)
        let uv = motion.floats(corners.flatMap { [$0.1.x, $0.1.y] }, type: "VEC2", count: 6)
        let map: [String: Any] = [
            "index": 0, "extensions": ["KHR_texture_transform": ["offset": [-1, -1], "scale": [3, 3]]],
        ]
        var material: [String: Any]
        var extensions = ["KHR_texture_transform"]
        if lit {
            material = [
                "pbrMetallicRoughness": ["baseColorTexture": map, "metallicRoughnessTexture": map],
                "normalTexture": map, "occlusionTexture": map, "emissiveTexture": map,
                "extensions": [
                    "KHR_materials_clearcoat": ["clearcoatFactor": 1, "clearcoatTexture": map, "clearcoatRoughnessTexture": map],
                    "KHR_materials_specular": ["specularTexture": map, "specularColorTexture": map],
                    "KHR_materials_sheen": ["sheenColorFactor": [1, 1, 1], "sheenColorTexture": map, "sheenRoughnessTexture": map],
                ],
            ]
            extensions += ["KHR_materials_clearcoat", "KHR_materials_specular", "KHR_materials_sheen"]
        } else {
            material = ["pbrMetallicRoughness": ["baseColorTexture": map], "extensions": ["KHR_materials_unlit": [:] as [String: Any]]]
            extensions.append("KHR_materials_unlit")
        }
        let png = try Self.checkerPNG().base64EncodedString()
        return try motion.document([
            "nodes": [["mesh": 0, "translation": [0, 0, -1]]],
            "meshes": [["primitives": [[
                "attributes": ["POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv], "material": 0,
            ]]]],
            "materials": [material],
            "textures": [["source": 0, "sampler": 0]],
            "samplers": [["magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33648]],
            "images": [["uri": "data:image/png;base64,\(png)"]],
            "extensionsUsed": extensions,
        ])
    }

    /// The first-hit albedo guide of a one-sample trace.
    private func albedoGuide(_ graph: SceneGraph) throws -> [SIMD4<Float>] {
        let side = Self.side
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        let guides = PathTracer.Guides(
            normalDepth: try fixture.output(width: side, height: side),
            albedoRoughness: try fixture.output(width: side, height: side)
        )
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: camera, cameraTransform: SceneTransform(), destination: try fixture.output(width: side, height: side),
            accumulation: accumulation, guides: guides, commandBuffer: buffer, settings: .init(maxBounces: 1, seed: 3)
        )
        try fixture.finish(buffer)
        return try fixture.readback(guides.albedoRoughness)
    }

    @Test("Textured glTF materials with samplers path-trace, and their albedo guide matches raster")
    func importedSamplers() throws {
        let side = Self.side
        let expectedSampling = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, addressV: .mirroredRepeat, magnification: .nearest, minification: .nearest,
            mipFilter: .notMipmapped
        )
        let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(-1, -1), scale: SIMD2(3, 3))
        var raster: [SIMD4<Float>] = []
        for lit in [false, true] {
            let scene = try GLTFMeshLoader.loadScene(data: asset(lit: lit), includeMaterials: true)
            // Preparation describes every binding's sampler.
            let material = try #require(scene.materials.first).prepare(device: fixture.device)
            #expect(material.baseColorSampling == expectedSampling)
            if lit {
                #expect(material.clearcoat?.factorTexture?.sampling == expectedSampling)
                #expect(material.clearcoat?.roughnessTexture?.sampling == expectedSampling)
                #expect(material.specular?.factorTexture?.sampling == expectedSampling)
                #expect(material.specular?.colorTexture?.sampling == expectedSampling)
                #expect(material.sheen?.colorTexture?.sampling == expectedSampling)
                #expect(material.sheen?.roughnessTexture?.sampling == expectedSampling)
                #expect(material.clearcoat?.factorTexture?.sampler != nil)
            }

            let prepared = try PreparedGLTFScene(device: fixture.device, colorPixelFormat: .rgba16Float, scene: scene)
            let graph = try prepared.graph(camera: camera)
            if !lit { raster = try fixture.rasterize(graph, width: side, height: side) }
            let albedo = try albedoGuide(graph)

            var checked = 0
            for y in 0..<side {
                for x in 0..<side {
                    let mapped = transform.apply(to: SIMD2(Float(x) + 0.5, Float(y) + 0.5) / Float(side))
                    let cell = mapped * 2 - (mapped * 2).rounded(.down)
                    guard (0..<2).allSatisfy({ min(cell[$0], 1 - cell[$0]) > 0.15 }) else { continue }
                    // Clamped U and mirrored V.
                    let u = min(max(mapped.x, 0), 1)
                    let v = 1 - abs((mapped.y * 0.5 - (mapped.y * 0.5).rounded(.down)) * 2 - 1)
                    let expected = Self.colors[min(Int(v * 2), 1) * 2 + min(Int(u * 2), 1)]
                    let index = y * side + x
                    let guide = SIMD3(albedo[index].x, albedo[index].y, albedo[index].z)
                    let pixel = SIMD3(raster[index].x, raster[index].y, raster[index].z)
                    #expect(simd_distance(guide, pixel) < 0.01, "lit \(lit) at \(x), \(y): \(guide) \(pixel) \(mapped)")
                    #expect(simd_distance(guide, expected) < 0.01, "lit \(lit) at \(x), \(y): \(expected)")
                    checked += 1
                }
            }
            // Pixel centers on the quad's shared diagonal are checked too: both triangle tests are watertight.
            #expect(checked > 300)
        }
    }

    // MARK: - Stochastic coverage

    private func plane(_ material: PhysicallyBasedMaterial) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.plane(width: 8, height: 8),
            material: .physicallyBased(material)
        )
    }

    private func unlit(
        _ color: SIMD4<Float>, alphaMode: PhysicallyBasedMaterial.AlphaMode, texture: (any MTLTexture)? = nil
    ) throws -> MeshRenderer {
        try plane(PhysicallyBasedMaterial(
            baseColorTexture: texture, baseColorFactor: color, alphaMode: alphaMode, shadingModel: .unlit
        ))
    }

    private func meanRGB(_ image: [SIMD4<Float>]) -> SIMD3<Float> {
        image.reduce(SIMD3<Float>.zero) { $0 + SIMD3($1.x, $1.y, $1.z) } / Float(image.count)
    }

    @Test("Blended surfaces trace as stochastic coverage over an emissive backdrop")
    func blendCoverage() throws {
        let backdrop = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -4)), mesh: try unlit(SIMD4(repeating: 1), alphaMode: .opaque)
        )
        let alphaMap = try fixture.dataTexture(pixels: [0, 0, 0, 64])
        let cases: [(Float, (any MTLTexture)?, Float)] = [
            (0, nil, 1), (0.5, nil, 0.5), (1, nil, 0), (0.5, alphaMap, 1 - 0.5 * 64 / 255), (1, alphaMap, 1 - 64 / 255),
        ]
        let settings = PathTracer.Settings(maxBounces: 1, seed: 11)
        for (alpha, texture, transmitted) in cases {
            let cover = SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, -2)),
                mesh: try unlit(SIMD4(0, 0, 0, alpha), alphaMode: .blend, texture: texture)
            )
            let scene = try PathTracingScene(
                queue: fixture.queue, graph: SceneGraph(roots: [backdrop, cover], camera: fixture.camera), textureSize: 16
            )
            let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
            let image = try fixture.trace(tracer, width: 32, height: 32, samples: 64, settings: settings)
            let mean = meanRGB(image)
            // A camera ray sees the black cover with probability alpha and the white backdrop otherwise.
            #expect(simd_reduce_max(abs(mean - SIMD3(repeating: transmitted))) < 0.015, "\(alpha), \(texture != nil)")
            if transmitted == 0 || transmitted == 1 {
                #expect(image.allSatisfy { abs($0.x - transmitted) < 1e-3 }, "\(alpha)")
            } else {
                // Every pixel varies around the mean rather than splitting into opaque and clear regions.
                #expect(image.allSatisfy { abs($0.x - transmitted) < 0.35 }, "\(alpha), \(texture != nil)")
            }
        }
    }

    @Test("Material-alpha surfaces hide what lies behind them and write their alpha, as in raster")
    func materialAlphaMatchesRaster() throws {
        let backdrop = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -4)), mesh: try unlit(SIMD4(repeating: 1), alphaMode: .opaque)
        )
        let alphaMap = try fixture.dataTexture(pixels: [255, 255, 255, 64])
        let cases: [(Float, (any MTLTexture)?, Float)] = [(0.25, nil, 0.25), (1, alphaMap, 64.0 / 255), (1, nil, 1)]
        let side = 32
        for (alpha, texture, expected) in cases {
            let cover = SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, -2)),
                mesh: try unlit(SIMD4(1, 0.5, 0.25, alpha), alphaMode: .materialAlpha, texture: texture)
            )
            let graph = SceneGraph(roots: [backdrop, cover], camera: fixture.camera)
            let raster = try fixture.rasterize(graph, width: side, height: side)
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
            let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
            let traced = try fixture.trace(
                tracer, width: side, height: side, samples: 4, settings: PathTracer.Settings(maxBounces: 1, seed: 7)
            )
            // Premultiplied color and alpha; the white backdrop never shows through.
            let color = SIMD4<Float>(1 * expected, 0.5 * expected, 0.25 * expected, expected)
            for index in traced.indices {
                #expect(simd_distance(traced[index], color) < 0.01, "path \(alpha), \(index): \(traced[index])")
                #expect(simd_distance(raster[index], color) < 0.01, "raster \(alpha), \(index): \(raster[index])")
            }
        }
    }

    @Test("First-hit guides show a blended surface where its alpha is at least one half, so the denoiser filters it")
    func blendGuides() throws {
        let side = 32
        let backdrop = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -4)), mesh: try unlit(SIMD4(repeating: 1), alphaMode: .opaque)
        )
        let coverColor = SIMD3<Float>(0.2, 0.4, 0.6)
        for alpha: Float in [0.25, 0.5, 0.75] {
            let cover = SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, -2)),
                mesh: try unlit(SIMD4(coverColor, alpha), alphaMode: .blend)
            )
            let scene = try PathTracingScene(
                queue: fixture.queue, graph: SceneGraph(roots: [backdrop, cover], camera: fixture.camera), textureSize: 16
            )
            let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
            let noisy = try fixture.output(width: side, height: side)
            let denoised = try fixture.output(width: side, height: side)
            let guides = PathTracer.Guides(
                normalDepth: try fixture.output(width: side, height: side),
                albedoRoughness: try fixture.output(width: side, height: side)
            )
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            _ = try tracer.encode(
                camera: fixture.camera, cameraTransform: SceneTransform(), destination: noisy,
                accumulation: try tracer.makeAccumulation(width: side, height: side), guides: guides,
                commandBuffer: buffer, samplesPerPixel: 4, settings: .init(maxBounces: 1, seed: 9)
            )
            try PathTraceDenoiser(device: fixture.device).encode(
                source: noisy, normalDepth: guides.normalDepth, albedoRoughness: guides.albedoRoughness,
                destination: denoised, passes: 3, commandBuffer: buffer
            )
            try fixture.finish(buffer)

            // Every guide pixel shows the same layer: the cover at alpha one half and above, else the backdrop.
            let front = alpha >= 0.5
            let albedo = try fixture.readback(guides.albedoRoughness)
            let depth = try fixture.readback(guides.normalDepth)
            #expect(albedo.allSatisfy { simd_distance(SIMD3($0.x, $0.y, $0.z), front ? coverColor : SIMD3(repeating: 1)) < 0.01 })
            #expect(depth.allSatisfy { front ? (2...2.6).contains($0.w) : (4...5.2).contains($0.w) }, "\(alpha)")

            // The cover over the white backdrop converges to alpha · cover + (1 - alpha) · white.
            let expected = alpha * coverColor + (1 - alpha) * SIMD3(repeating: 1)
            func error(_ image: [SIMD4<Float>]) -> Float {
                image.reduce(0) { $0 + simd_reduce_max(abs(SIMD3($1.x, $1.y, $1.z) - expected)) } / Float(image.count)
            }
            let before = error(try fixture.readback(noisy))
            let after = error(try fixture.readback(denoised))
            #expect(after < 0.4 * before, "\(alpha): \(before) → \(after)")
        }
    }

    @Test("Shadow rays through a half-transparent blended surface carry half the light")
    func blendShadows() throws {
        let wall = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -4)),
            mesh: try fixture.pbr(try MeshGeometry.plane(width: 8, height: 8), color: SIMD3(repeating: 1))
        )
        let light = try PathTracer.AnalyticLight(PunctualLight(kind: .directional), direction: SIMD3(0, 0, -1))
        let lighting = try PathTracer.Lighting(analyticLights: [light], environment: .zero)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 5)

        func wallRadiance(coverAlpha: Float?, mode: PhysicallyBasedMaterial.AlphaMode = .blend) throws -> SIMD3<Float> {
            var roots = [wall]
            if let coverAlpha {
                // Single-sided and turned away from the camera, the cover culls camera rays but
                // faces the wall, so only the wall's shadow and bounce rays can hit it.
                roots.append(SceneNode(
                    transform: SceneTransform(
                        translation: SIMD3(0, 0, -2), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
                    ),
                    mesh: try unlit(SIMD4(0, 0, 0, coverAlpha), alphaMode: mode)
                ))
            }
            let scene = try PathTracingScene(
                queue: fixture.queue, graph: SceneGraph(roots: roots, camera: fixture.camera), textureSize: 16
            )
            let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: lighting)
            return meanRGB(try fixture.trace(tracer, width: 32, height: 32, samples: 64, settings: settings))
        }

        let open = try wallRadiance(coverAlpha: nil)
        #expect(open.x > 0.25)
        let clear = try wallRadiance(coverAlpha: 0)
        #expect(simd_reduce_max(abs(clear - open)) < 1e-3)
        let half = try wallRadiance(coverAlpha: 0.5)
        #expect(abs(half.x / open.x - 0.5) < 0.03)
        let opaque = try wallRadiance(coverAlpha: 1)
        #expect(opaque.x < 1e-4)
        // Material alpha casts an opaque shadow, as its raster shadow map does.
        #expect(try wallRadiance(coverAlpha: 0.5, mode: .materialAlpha).x < 1e-4)
    }
}

private extension PathTestFixture {
    /// A 1 × 1 RGBA8 data texture.
    func dataTexture(pixels: [UInt8]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        try pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: try #require(bytes.baseAddress), bytesPerRow: 4
            )
        }
        return texture
    }
}
