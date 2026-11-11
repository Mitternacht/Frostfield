import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import simd

/// Full-resolution material textures in the path tracer: direct sampling through the texture
/// table, ray-cone level of detail with anisotropic filtering, limits and memory statistics.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathFullResolutionTextureTests {
    private let fixture: PathTestFixture

    init() throws {
        fixture = try PathTestFixture()
    }

    private static var supportsDirectTextures: Bool {
        MTLCreateSystemDefaultDevice()?.argumentBuffersSupport == .tier2
    }

    // MARK: - Helpers

    /// Traces one pinhole guide sample per pixel center, as raster samples pixel centers, and
    /// returns the first-hit albedo with the prepared scene.
    private func albedoGuide(
        _ graph: SceneGraph, width: Int, height: Int, options: PathTracingScene.Options = .init()
    ) throws -> (albedo: [SIMD4<Float>], scene: PathTracingScene) {
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: options)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let guides = PathTracer.Guides(
            normalDepth: try fixture.output(width: width, height: height),
            albedoRoughness: try fixture.output(width: width, height: height)
        )
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: graph.camera, cameraTransform: graph.cameraTransform,
            destination: try fixture.output(width: width, height: height), accumulation: accumulation, guides: guides,
            commandBuffer: buffer, settings: .init(maxBounces: 1, seed: 3)
        )
        try fixture.finish(buffer)
        return (try fixture.readback(guides.albedoRoughness), scene)
    }

    /// Accumulates `samples` jittered samples per pixel of unlit emission from the graph's camera.
    private func trace(
        _ graph: SceneGraph, width: Int, height: Int, samples: Int, options: PathTracingScene.Options
    ) throws -> [SIMD4<Float>] {
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: options)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try fixture.output(width: width, height: height)
        var history: PathTracer.History?
        while (history?.sampleCount ?? 0) < samples {
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: graph.camera, cameraTransform: graph.cameraTransform, destination: target,
                accumulation: accumulation, previous: history, commandBuffer: buffer,
                samplesPerPixel: min(16, samples - (history?.sampleCount ?? 0)), settings: .init(maxBounces: 1, seed: 11)
            )
            try fixture.finish(buffer)
        }
        return try fixture.readback(target)
    }

    private static func hash(_ value: UInt32) -> UInt32 {
        var value = value
        value ^= value >> 16
        value &*= 0x7feb352d
        value ^= value >> 15
        value &*= 0x846ca68b
        return value ^ (value >> 16)
    }

    /// RGBA8 noise, one hashed value per channel and texel.
    private static func noise(size: Int, seed: UInt32) -> [UInt8] {
        var pixels = [UInt8](repeating: 255, count: size * size * 4)
        for texel in 0..<(size * size) {
            let value = hash(UInt32(texel) ^ seed)
            pixels[4 * texel] = UInt8(truncatingIfNeeded: value)
            pixels[4 * texel + 1] = UInt8(truncatingIfNeeded: value >> 8)
            pixels[4 * texel + 2] = UInt8(truncatingIfNeeded: value >> 16)
        }
        return pixels
    }

    private static func png(size: Int, pixels: [UInt8]) throws -> Data {
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let image = try #require(CGImage(
            width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        ))
        let output = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(output as CFMutableData, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        try #require(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// An unlit 2 × 2 quad with UV (0, 0) at its top-left corner, facing +Z at `translation`,
    /// whose base color is the PNG with linear filtering and repeat wrapping.
    private static func quadAsset(png: Data, translation: SIMD3<Float>) throws -> Data {
        var motion = GLTFMotionFixture()
        let corners: [(SIMD3<Float>, SIMD2<Float>)] = [
            (SIMD3(-1, 1, 0), SIMD2(0, 0)), (SIMD3(-1, -1, 0), SIMD2(0, 1)), (SIMD3(1, 1, 0), SIMD2(1, 0)),
            (SIMD3(1, 1, 0), SIMD2(1, 0)), (SIMD3(-1, -1, 0), SIMD2(0, 1)), (SIMD3(1, -1, 0), SIMD2(1, 1)),
        ]
        let position = motion.floats(corners.flatMap { [$0.0.x, $0.0.y, $0.0.z] }, type: "VEC3", count: 6)
        let normal = motion.floats(Array(repeating: [0, 0, 1], count: 6).flatMap { $0 }, type: "VEC3", count: 6)
        let uv = motion.floats(corners.flatMap { [$0.1.x, $0.1.y] }, type: "VEC2", count: 6)
        return try motion.document([
            "nodes": [["mesh": 0, "translation": [translation.x, translation.y, translation.z]]],
            "meshes": [["primitives": [[
                "attributes": ["POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv], "material": 0,
            ]]]],
            "materials": [[
                "pbrMetallicRoughness": ["baseColorTexture": ["index": 0]],
                "extensions": ["KHR_materials_unlit": [:] as [String: Any]],
            ]],
            "textures": [["source": 0, "sampler": 0]],
            "samplers": [["magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497]],
            "images": [["uri": "data:image/png;base64,\(png.base64EncodedString())"]],
            "extensionsUsed": ["KHR_materials_unlit"],
        ])
    }

    private func texture(
        _ format: MTLPixelFormat, size: Int, mipmapped: Bool = false, bytesPerRow: Int, bytes: [UInt8]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: size, height: size, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        try bytes.withUnsafeBytes { raw in
            texture.replace(
                region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0,
                withBytes: try #require(raw.baseAddress), bytesPerRow: bytesPerRow
            )
        }
        return texture
    }

    private func generateMipmaps(_ texture: any MTLTexture) throws {
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        let blit = try #require(buffer.makeBlitCommandEncoder())
        blit.generateMipmaps(for: texture)
        blit.endEncoding()
        try fixture.finish(buffer)
    }

    private func unlit(_ texture: any MTLTexture, sampling: PhysicallyBasedMaterial.TextureSampling = .default) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            baseColorTexture: texture, alphaMode: .opaque, shadingModel: .unlit, baseColorSampling: sampling
        )
    }

    private func node(
        _ geometry: MeshGeometry, _ material: PhysicallyBasedMaterial, transform: SceneTransform
    ) throws -> SceneNode {
        SceneNode(transform: transform, mesh: try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        ))
    }

    private static func rgb(_ value: SIMD4<Float>) -> SIMD3<Float> { SIMD3(value.x, value.y, value.z) }

    private static func maximumDifference(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> Float {
        zip(a, b).reduce(0) { max($0, simd_reduce_max(abs(rgb($1.0) - rgb($1.1)))) }
    }

    /// Writes linear RGB as an sRGB PNG when FROSTFIELD_TEST_IMAGES names a directory.
    static func save(_ image: [SIMD4<Float>], width: Int, height: Int, name: String) throws {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_TEST_IMAGES"] else { return }
        let pixels = image.flatMap { pixel -> [UInt8] in
            (0..<3).map { channel in
                let linear = min(max(pixel[channel], 0), 1)
                let encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1 / 2.4) - 0.055
                return UInt8((encoded * 255).rounded())
            } + [255]
        }
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let cgImage = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        ))
        let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, cgImage, nil)
        try #require(CGImageDestinationFinalize(destination))
    }

    // MARK: - Raster parity

    // A 0.0208 rad view at distance 1 sees about 21 texels of a 2,048-texel map across 64 pixels,
    // a threefold magnification. The view sits off the quad's diagonal.
    private static let magnifiedSide = 64

    private func magnifiedCamera() throws -> PerspectiveCamera {
        try PerspectiveCamera(verticalFieldOfView: 0.0208, near: 0.01, far: 10)
    }

    @Test(
        "Magnified 2,048² sRGB glTF textures trace within 1/255 of raster, and statistics report the sampled memory",
        .enabled(if: supportsDirectTextures, "Requires Metal tier 2 argument buffers")
    )
    func magnifiedGLTFMatchesRaster() throws {
        let side = Self.magnifiedSide
        let png = try Self.png(size: 2_048, pixels: Self.noise(size: 2_048, seed: 7))
        let asset = try GLTFMeshLoader.loadScene(
            data: Self.quadAsset(png: png, translation: SIMD3(-0.37, -0.41, -1)), includeMaterials: true
        )
        let prepared = try PreparedGLTFScene(device: fixture.device, colorPixelFormat: .rgba16Float, scene: asset)
        let graph = try prepared.graph(camera: magnifiedCamera())
        let raster = try fixture.rasterize(graph, width: side, height: side)
        let (albedo, scene) = try albedoGuide(graph, width: side, height: side)
        try Self.save(raster, width: side, height: side, name: "magnified-raster")
        try Self.save(albedo, width: side, height: side, name: "magnified-path")
        #expect(raster.allSatisfy { $0.w == 1 })
        let difference = Self.maximumDifference(raster, albedo)
        #expect(difference < 1 / 255, "largest linear difference \(difference)")
        // The view is not flat: neighboring pixels differ by more than the tolerance.
        let contrast = (1..<side).map { simd_reduce_max(abs(Self.rgb(albedo[$0]) - Self.rgb(albedo[$0 - 1]))) }.max() ?? 0
        #expect(contrast > 0.05)

        let material = try #require(asset.materials.first).prepare(device: fixture.device)
        let source = try #require(material.baseColorTexture)
        #expect(source.width == 2_048 && source.pixelFormat == .rgba8Unorm_srgb)
        let statistics = scene.statistics
        #expect(statistics.boundTextureCount == 1 && statistics.bakedTextureCount == 0 && scene.textureCount == 1)
        #expect(statistics.referencedTextureByteCount == source.allocatedSize)
        #expect(statistics.referencedTextureByteCount >= 2_048 * 2_048 * 4)
        #expect(statistics.textureByteCount < statistics.referencedTextureByteCount / 100)

        // Baking the same map at its largest fallback size loses the detail.
        let (baked, bakedScene) = try albedoGuide(graph, width: side, height: side, options: .init(textures: .baked(size: 512)))
        #expect(Self.maximumDifference(raster, baked) > 0.1)
        #expect(bakedScene.statistics.boundTextureCount == 0 && bakedScene.statistics.bakedTextureCount == 1)
        #expect(bakedScene.statistics.referencedTextureByteCount == 0)
        #expect(bakedScene.statistics.textureByteCount >= 512 * 512 * 8)

        let explicitSize = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        #expect(explicitSize.statistics.boundTextureCount == 0)
        #expect(explicitSize.statistics.bakedTextureCount == 1)
        #expect(explicitSize.statistics.textureByteCount < bakedScene.statistics.textureByteCount)
    }

    @Test(
        "Block-compressed, half-float and sRGB textures trace within 1/255 of raster when magnified",
        .enabled(if: supportsDirectTextures, "Requires Metal tier 2 argument buffers")
    )
    func magnifiedFormatsMatchRaster() throws {
        let side = Self.magnifiedSide
        let size = 1_024
        var textures: [(String, any MTLTexture)] = []
        textures.append(("sRGB", try texture(
            .rgba8Unorm_srgb, size: size, bytesPerRow: size * 4, bytes: Self.noise(size: size, seed: 3)
        )))
        let halves = Self.noise(size: size, seed: 5).map { Float16(Float($0) / 255) }
        textures.append(("rgba16Float", try texture(
            .rgba16Float, size: size, bytesPerRow: size * 8,
            bytes: halves.withUnsafeBytes { Array($0) }
        )))
        if fixture.device.supportsBCTextureCompression {
            // Hashed BC1 blocks: two 565 endpoints and 2-bit indices per 4 × 4 block.
            let blocks = (0..<(size / 4) * (size / 4) * 2).flatMap { index -> [UInt8] in
                let value = Self.hash(UInt32(index) ^ 0x51)
                return (0..<4).map { UInt8(truncatingIfNeeded: value >> (8 * $0)) }
            }
            textures.append(("BC1 sRGB", try texture(.bc1_rgba_srgb, size: size, bytesPerRow: (size / 4) * 8, bytes: blocks)))
        }
        // Scaled so the 0.0208 rad view spans about 21 texels: 1,024 texels over a 1 × 1 quad.
        for (name, map) in textures {
            let graph = SceneGraph(
                roots: [try node(
                    .plane(width: 1, height: 1), try unlit(map),
                    transform: SceneTransform(translation: SIMD3(-0.18, -0.21, -0.5))
                )],
                camera: try magnifiedCamera()
            )
            let raster = try fixture.rasterize(graph, width: side, height: side)
            let (albedo, scene) = try albedoGuide(graph, width: side, height: side)
            #expect(raster.allSatisfy { $0.w == 1 }, "\(name)")
            let difference = Self.maximumDifference(raster, albedo)
            #expect(difference < 1 / 255, "\(name): largest linear difference \(difference)")
            let contrast = (1..<side).map { simd_reduce_max(abs(Self.rgb(albedo[$0]) - Self.rgb(albedo[$0 - 1]))) }.max() ?? 0
            #expect(contrast > 0.05, "\(name)")
            #expect(scene.statistics.referencedTextureByteCount == map.allocatedSize, "\(name)")
        }
    }

    // MARK: - Minification

    @Test(
        "A 4,096² checker on a grazing ground plane converges toward a supersampled reference without moiré",
        .enabled(if: supportsDirectTextures, "Requires Metal tier 2 argument buffers")
    )
    func grazingChecker() throws {
        let width = 160
        let height = 96
        let size = 4_096
        // 8-texel squares: 512 across the 40 m plane, 7.8 cm each.
        var pixels = [UInt8](repeating: 255, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size where (x / 8 + y / 8).isMultiple(of: 2) {
                let index = 4 * (y * size + x)
                pixels[index] = 0
                pixels[index + 1] = 0
                pixels[index + 2] = 0
            }
        }
        let checker = try texture(.rgba8Unorm, size: size, mipmapped: true, bytesPerRow: size * 4, bytes: pixels)
        try generateMipmaps(checker)
        let ground = try MeshGeometry.plane(width: 40, height: 40)
        let placement = SceneTransform(
            translation: SIMD3(0, 0, -20), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        )
        // Eye 0.6 m above the plane, pitched 7° down: the far rows meet the plane below 1°.
        let camera = try PerspectiveCamera(verticalFieldOfView: 50 * .pi / 180, near: 0.05, far: 100)
        let view = SceneTransform(
            translation: SIMD3(0, 0.6, 0), rotation: simd_quatf(angle: -7 * .pi / 180, axis: SIMD3(1, 0, 0))
        )
        func graph(_ sampling: PhysicallyBasedMaterial.TextureSampling) throws -> SceneGraph {
            SceneGraph(
                roots: [try node(ground, try unlit(checker, sampling: sampling), transform: placement)],
                camera: camera, cameraTransform: view
            )
        }
        let trilinear = try graph(.default)
        // The reference box-filters the base level: 1,024 jittered bilinear samples per pixel,
        // without mip levels or anisotropic taps, so the footprint does not affect it.
        let baseLevel = PathTracingScene.Options(maximumAnisotropy: 1)
        let reference = try trace(
            try graph(.init(mipFilter: .notMipmapped)), width: width, height: height, samples: 1_024, options: baseLevel
        )
        // Candidates take one prefiltered sample at each pixel center, the first-hit albedo guide,
        // so the comparison measures filtering rather than jitter noise.
        let pointSampled = try albedoGuide(
            try graph(.init(mipFilter: .notMipmapped)), width: width, height: height, options: baseLevel
        ).albedo
        let anisotropic = try albedoGuide(trilinear, width: width, height: height).albedo
        let isotropic = try albedoGuide(trilinear, width: width, height: height, options: .init(maximumAnisotropy: 1)).albedo
        let baked = try albedoGuide(trilinear, width: width, height: height, options: .init(textures: .baked(size: 512))).albedo
        for (name, image) in [
            ("grazing-reference", reference), ("grazing-point", pointSampled), ("grazing-anisotropic", anisotropic),
            ("grazing-isotropic", isotropic), ("grazing-baked", baked),
        ] {
            try Self.save(image, width: width, height: height, name: name)
        }

        let groundPixels = reference.indices.filter { reference[$0].w > 0.5 }
        #expect(groundPixels.count > width * height / 2)
        func rmse(_ image: [SIMD4<Float>], _ pixels: [Int]) -> Double {
            let sum = pixels.reduce(0.0) { total, index in
                total + Double(simd_length_squared(Self.rgb(image[index]) - Self.rgb(reference[index])))
            }
            return (sum / Double(pixels.count * 3)).squareRoot()
        }
        let candidates = [anisotropic, isotropic, baked, pointSampled]
        let errors = candidates.map { rmse($0, groundPixels) }
        // RMSE per band of eight image rows, nearest to the horizon first.
        let bands = stride(from: 0, to: height, by: 8).map { band in groundPixels.filter { (band..<(band + 8)).contains($0 / width) } }
            .filter { !$0.isEmpty }
        let bandErrors = candidates.map { image in bands.map { rmse(image, $0) } }
        print("Grazing checker RMSE: anisotropic \(errors[0]), isotropic \(errors[1]), baked 512 \(errors[2]), point \(errors[3])")
        for (name, values) in zip(["anisotropic", "isotropic", "baked 512", "point"], bandErrors) {
            print("Grazing \(name) by band: \(values.map { String(format: "%.4f", $0) })")
        }
        // Ray cones with anisotropic filtering stay within 0.05 of the reference in every band. Point
        // sampling the base level aliases into moiré, isotropic filtering blurs the receding squares,
        // and the 512² bake of the 4,096² map does both.
        #expect(errors[0] < 0.04)
        #expect(bandErrors[0].allSatisfy { $0 < 0.05 })
        #expect(bandErrors[3].max()! > 0.15 && errors[0] < 0.3 * errors[3])
        #expect(errors[0] < 0.4 * errors[1] && errors[0] < 0.3 * errors[2])
    }

    // MARK: - Limits

    @Test(
        "Three hundred distinct textures, beyond the former 64-texture bake, each trace on their own quad",
        .enabled(if: supportsDirectTextures, "Requires Metal tier 2 argument buffers")
    )
    func manyTextures() throws {
        let columns = 20
        let rows = 15
        let cell = 8
        let camera = try PerspectiveCamera(verticalFieldOfView: 2 * atan(Float(rows) / 2 / 10), near: 0.1, far: 30)
        var roots: [SceneNode] = []
        var colors: [SIMD3<Float>] = []
        for index in 0..<(columns * rows) {
            let value = Self.hash(UInt32(index) &+ 99)
            let bytes = (0..<3).map { UInt8(truncatingIfNeeded: value >> (8 * $0)) }
            colors.append(SIMD3(bytes.map { Float($0) / 255 }))
            let map = try texture(.rgba8Unorm, size: 4, bytesPerRow: 16, bytes: Array((0..<16).map { _ in bytes + [255] }.joined()))
            let x = Float(index % columns) - Float(columns - 1) / 2
            let y = Float(rows - 1) / 2 - Float(index / columns)
            roots.append(try node(.plane(width: 1, height: 1), try unlit(map), transform: SceneTransform(translation: SIMD3(x, y, -10))))
        }
        let graph = SceneGraph(roots: roots, camera: camera)
        let (albedo, scene) = try albedoGuide(graph, width: columns * cell, height: rows * cell)
        #expect(scene.statistics.boundTextureCount == columns * rows && scene.textureCount == columns * rows)
        for index in 0..<(columns * rows) {
            let pixel = (index / columns * cell + cell / 2) * columns * cell + index % columns * cell + cell / 2
            #expect(simd_distance(Self.rgb(albedo[pixel]), colors[index]) < 0.01, "quad \(index)")
        }
    }

    @Test(
        "The texture table holds 4,096 entries, traces with all of them resident, and rejects one more",
        .enabled(if: supportsDirectTextures, "Requires Metal tier 2 argument buffers")
    )
    func textureLimit() throws {
        let plane = try MeshGeometry.plane(width: 0.01, height: 0.01)
        func white() throws -> any MTLTexture { try texture(.rgba8Unorm, size: 1, bytesPerRow: 4, bytes: [255, 255, 255, 255]) }
        func binding() throws -> PhysicallyBasedMaterial.TextureBinding { .init(texture: try white()) }
        // Base color, metallic-roughness, normal and emissive maps: four entries per material.
        var roots: [SceneNode] = []
        for index in 0..<1_024 {
            let material = try PhysicallyBasedMaterial(
                baseColorTexture: try white(), metallicRoughnessTexture: try binding(), normalTexture: try binding(),
                emissiveTexture: try binding(), alphaMode: .opaque
            )
            roots.append(try node(plane, material, transform: SceneTransform(translation: SIMD3(Float(index) * 0.02, 0, -5))))
        }
        let full = SceneGraph(roots: roots, camera: fixture.camera)
        let (_, scene) = try albedoGuide(full, width: 8, height: 8)
        #expect(scene.textureCount == PathTracingScene.maximumTextureCount && scene.statistics.boundTextureCount == 4_096)

        let coated = try PhysicallyBasedMaterial(
            baseColorTexture: try white(), metallicRoughnessTexture: try binding(), normalTexture: try binding(),
            emissiveTexture: try binding(), alphaMode: .opaque,
            clearcoat: try .init(factor: 1, factorTexture: try binding())
        )
        roots[roots.count - 1] = try node(plane, coated, transform: SceneTransform(translation: SIMD3(0, 1, -5)))
        #expect(throws: PathTracingScene.Failure.textureLimit) {
            try PathTracingScene(queue: fixture.queue, graph: SceneGraph(roots: roots, camera: fixture.camera))
        }
    }

    @Test("The default texture path renders through a bounded bake when direct sampling is unavailable")
    func defaultTextureFallback() throws {
        let map = try texture(.rgba8Unorm, size: 2, bytesPerRow: 8, bytes: [
            64, 128, 255, 255, 64, 128, 255, 255,
            64, 128, 255, 255, 64, 128, 255, 255,
        ])
        let graph = SceneGraph(
            roots: [try node(.plane(width: 4, height: 4), try unlit(map), transform: SceneTransform(translation: SIMD3(0, 0, -2)))],
            camera: fixture.camera
        )
        let side = 8
        let (automatic, scene) = try albedoGuide(graph, width: side, height: side)
        let (baked, bakedScene) = try albedoGuide(graph, width: side, height: side, options: .init(textures: .baked(size: 256)))
        let center = side / 2 * side + side / 2
        #expect(simd_reduce_max(abs(Self.rgb(automatic[center]) - SIMD3<Float>(64.0 / 255, 128.0 / 255, 1))) < 0.001)
        #expect(automatic == baked)
        #expect(bakedScene.statistics.bakedTextureCount == 1 && bakedScene.statistics.boundTextureCount == 0)
        #expect(bakedScene.statistics.textureByteCount >= 256 * 256 * 8)
        if Self.supportsDirectTextures {
            #expect(scene.statistics.boundTextureCount == 1 && scene.statistics.bakedTextureCount == 0)
        } else {
            #expect(scene.statistics.boundTextureCount == 0 && scene.statistics.bakedTextureCount == 1)
            #expect(scene.statistics.textureByteCount == bakedScene.statistics.textureByteCount)
        }
        let tier = Self.supportsDirectTextures ? "Tier 2" : "Tier 1"
        print("Path texture table \(tier), working-set recommendation \(fixture.device.recommendedMaxWorkingSetSize), baked bytes \(bakedScene.statistics.textureByteCount)")
    }

    @Test(
        "Directly sampled textures obey the explicit memory budget",
        .enabled(if: supportsDirectTextures, "Requires Metal tier 2 argument buffers")
    )
    func directTextureBudget() throws {
        let map = try texture(.rgba8Unorm, size: 64, bytesPerRow: 256, bytes: Self.noise(size: 64, seed: 1))
        let graph = SceneGraph(
            roots: [try node(.plane(width: 1, height: 1), try unlit(map), transform: SceneTransform(translation: SIMD3(0, 0, -2)))],
            camera: fixture.camera
        )
        #expect(throws: PathTracingScene.Failure.textureMemory(map.allocatedSize, 1_000)) {
            try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(textureMemoryBudget: 1_000))
        }
        let fits = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(textureMemoryBudget: map.allocatedSize))
        #expect(fits.statistics.referencedTextureByteCount == map.allocatedSize)
    }

    @Test("Baked textures obey the explicit memory budget and preparation validates its options")
    func budgetAndOptions() throws {
        let map = try texture(.rgba8Unorm, size: 64, bytesPerRow: 256, bytes: Self.noise(size: 64, seed: 1))
        let graph = SceneGraph(
            roots: [try node(.plane(width: 1, height: 1), try unlit(map), transform: SceneTransform(translation: SIMD3(0, 0, -2)))],
            camera: fixture.camera
        )
        let baked = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        let bakedBytes = baked.statistics.textureByteCount
        #expect(bakedBytes >= (16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + 1) * 8)
        let tooSmall = PathTracingScene.Options(textures: .baked(size: 16), textureMemoryBudget: bakedBytes - 1)
        #expect(throws: PathTracingScene.Failure.textureMemory(bakedBytes, bakedBytes - 1)) {
            try PathTracingScene(queue: fixture.queue, graph: graph, options: tooSmall)
        }
        let exact = PathTracingScene.Options(textures: .baked(size: 16), textureMemoryBudget: bakedBytes)
        let bakedFit = try PathTracingScene(queue: fixture.queue, graph: graph, options: exact)
        #expect(bakedFit.statistics.textureByteCount == bakedBytes)

        #expect(throws: PathTracingScene.Failure.options) {
            try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(maximumAnisotropy: 0))
        }
        #expect(throws: PathTracingScene.Failure.options) {
            try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(textureMemoryBudget: 0))
        }
        #expect(throws: PathTracingScene.Failure.textureSize) {
            try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(textures: .fullResolution(fallbackSize: 8)))
        }
        #expect(throws: PathTracingScene.Failure.textureSize) {
            try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(textures: .baked(size: 1_024)))
        }
    }
}
