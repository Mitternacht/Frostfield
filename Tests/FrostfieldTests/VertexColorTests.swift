import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Vertex colors from glTF COLOR_0 through raster PBR, classic materials, the path
/// tracer and export, and the PBR alpha map.
struct VertexColorTests {
    static let left = SIMD4<Float>(0.9, 0.2, 0.1, 1)
    static let right = SIMD4<Float>(0.1, 0.3, 0.8, 0.5)

    /// A square whose left edge has `left` and right edge has `right`, so color is affine in X.
    static func gradientSquare(_ halfExtent: Float, left: SIMD4<Float> = left, right: SIMD4<Float> = right) throws -> MeshGeometry {
        let h = halfExtent
        let corners = [SIMD3(-h, -h, 0), SIMD3(h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, h, 0)]
        return try MeshGeometry(
            vertices: corners.map {
                MeshVertex(
                    position: $0, normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(($0.x / h + 1) / 2, (1 - $0.y / h) / 2),
                    tangent: SIMD4(1, 0, 0, -1), color: $0.x < 0 ? left : right
                )
            },
            indices: [0, 1, 2, 0, 2, 3]
        )
    }

    // MARK: Raster

    @Test("Raster PBR and classic materials interpolate vertex colors into base color and alpha")
    func rasterInterpolation() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try Self.gradientSquare(1)
        let factor = SIMD4<Float>(0.9, 0.8, 0.7, 1)
        let materials: [MeshMaterial] = [
            .physicallyBased(try PhysicallyBasedMaterial(baseColorFactor: factor, alphaMode: .blend, shadingModel: .unlit)),
            .classic(try ClassicMaterial(model: .basic, color: SIMD3(factor.x, factor.y, factor.z))),
        ]
        for (index, material) in materials.enumerated() {
            for instanced in [false, true] {
                let mesh = try MeshRenderer(
                    device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material,
                    supportsInstancing: instanced, renderState: .transparent
                )
                let image = try fixture.render(SceneGraph(
                    roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: fixture.camera
                )).pixels
                for column in [18, 24, 32, 40, 46] {
                    // The fixture camera sees ±2 at depth 2, and the square spans x in [−1, 1].
                    let x = ((Float(column) + 0.5) / 32 - 1) * 2
                    let color = simd_mix(Self.left, Self.right, SIMD4(repeating: (x + 1) / 2)) * factor
                    let expected = SIMD4(SIMD3(color.x, color.y, color.z) * color.w, color.w)
                    #expect(isNearly(fixture.pixel(image, x: column, y: 32), expected), "material \(index) column \(column)")
                }
            }
        }
        // The legacy unlit fragment does not read vertex colors.
        let legacy = try fixture.unlit(geometry)
        let image = try fixture.render(SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: legacy)], camera: fixture.camera
        )).pixels
        #expect(isNearly(fixture.pixel(image, x: 20, y: 32), GeometryRenderFixture.unlitFacingCamera(SIMD4(repeating: 1))))
    }

    // MARK: Path tracing

    @Test("The path tracer multiplies vertex colors like raster PBR, for unlit and lit surfaces", arguments: PathAccelerationTests.modes)
    func pathTracerMatchesRaster(acceleration: PathTracingScene.Options.Acceleration) throws {
        let fixture = try PathTestFixture()
        let width = 33
        let height = 25
        let unlit = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: try Self.gradientSquare(1, right: SIMD4(0.1, 0.3, 0.8, 1)),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                baseColorFactor: SIMD4(0.9, 0.8, 0.7, 1), alphaMode: .opaque, shadingModel: .unlit
            ))
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -3)), mesh: unlit)], camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let traced = try fixture.trace(
            try PathTracer(queue: fixture.queue, scene: scene), width: width, height: height, samples: 16,
            settings: PathTracer.Settings(maxBounces: 1, seed: 7)
        )
        let raster = try fixture.rasterize(graph, width: width, height: height)
        var compared = 0
        // The square covers columns 10 to 23 and rows 6 to 19; compare the interior.
        for row in 8...17 {
            for column in 11...22 {
                let index = row * width + column
                let difference = simd_reduce_max(abs(traced[index] - raster[index]))
                #expect(difference < 0.01, "pixel \(column), \(row): traced \(traced[index]) raster \(raster[index])")
                compared += 1
            }
        }
        #expect(compared == 120)
        // The gradient really varies across the compared region.
        #expect(abs(raster[12 * width + 11].x - raster[12 * width + 22].x) > 0.3)

        // Lit: a uniform vertex color scales the diffuse albedo of a point-lit plane.
        let tint = SIMD4<Float>(0.25, 0.5, 1, 1)
        let albedo: Float = 0.8
        let plane = try MeshGeometry.plane(width: 4, height: 4)
        let colored = try MeshGeometry(
            vertices: plane.vertices.map { MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate, color: tint) },
            indices: plane.indices
        )
        let receiver = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: colored,
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(albedo, albedo, albedo, 1),
                alphaMode: .opaque
            ))
        )
        let lit = SceneGraph(roots: [
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -3)), mesh: receiver),
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1)), light: try PunctualLight(kind: .point(), intensity: 6)),
        ], camera: fixture.camera)
        let litScene = try PathTracingScene(queue: fixture.queue, graph: lit, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: litScene)
        let lighting = try PathTracer.Lighting(analyticLights: PathTracer.AnalyticLight.lights(in: lit))
        let center = height / 2 * width + width / 2
        let litTraced = try fixture.trace(
            tracer, width: width, height: height, samples: 16, settings: PathTracer.Settings(maxBounces: 1, seed: 19),
            lighting: lighting
        )[center]
        let litRaster = try fixture.rasterize(lit, width: width, height: height)[center]
        // At normal incidence the dielectric BSDF is (1 − 0.04) ρ / π plus 0.04 × 0.25 / π; E = 6 / 2².
        let rho = SIMD3(tint.x, tint.y, tint.z) * albedo
        let expected = ((1 - 0.04) * rho + 0.01) / .pi * 1.5
        for (name, value) in [("traced", litTraced), ("raster", litRaster)] {
            let rgb = SIMD3(value.x, value.y, value.z)
            #expect(simd_reduce_max(abs(rgb - expected) / expected) < 0.02, "\(name) \(rgb) vs \(expected)")
        }
    }

    @Test("The path tracer applies vertex color to tint and alpha", arguments: PathAccelerationTests.modes)
    func pathTracerClampsAndAlpha(acceleration: PathTracingScene.Options.Acceleration) throws {
        let fixture = try PathTestFixture()
        let width = 33
        let height = 25
        let center = height / 2 * width + width / 2
        func unlit(
            _ color: SIMD4<Float>, factor: SIMD4<Float>, alphaMode: PhysicallyBasedMaterial.AlphaMode
        ) throws -> SceneGraph {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try Self.gradientSquare(1, left: color, right: color),
                material: .physicallyBased(try PhysicallyBasedMaterial(
                    baseColorFactor: factor, alphaMode: alphaMode, shadingModel: .unlit
                ))
            )
            return SceneGraph(
                roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: fixture.camera
            )
        }
        func traced(_ graph: SceneGraph, samples: Int = 4) throws -> SIMD4<Float> {
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
            let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: .init(environment: .zero, backgroundVisible: false))
            return try fixture.trace(
                tracer, width: width, height: height, samples: samples,
                settings: PathTracer.Settings(maxBounces: 1, seed: 3)
            )[center]
        }
        // Raster saturates parameters × factor × vertex color as one product: 0.5 × 2 = 1,
        // not 0.5 × saturate(2), and parameters of 2 × 0.5 × 0.4 give 0.4, not saturate(2) × 0.2.
        let one = SIMD4<Float>(repeating: 1)
        for (color, factor, parameters) in [(SIMD4<Float>(2, 2, 2, 1), SIMD4<Float>(0.5, 0.5, 0.5, 1), one),
                                            (SIMD4<Float>(1.6, 0.4, 3, 1), SIMD4<Float>(0.5, 0.9, 0.2, 1), one),
                                            (SIMD4<Float>(3, 0.2, 1.5, 1), SIMD4<Float>(0.5, 0.5, 0.9, 1), one),
                                            (SIMD4<Float>(0.4, 0.5, 0.5, 1), SIMD4<Float>(0.5, 0.5, 0.5, 1), SIMD4<Float>(2, 1, 3, 1))] {
            var graph = try unlit(color, factor: factor, alphaMode: .opaque)
            graph.roots[0].materialParameters = parameters
            let expected = simd_min(parameters * color * factor, SIMD4(repeating: 1))
            let raster = try fixture.rasterize(graph, width: width, height: height)[center]
            let trace = try traced(graph)
            #expect(isNearly(raster, expected, tolerance: 0.002), "raster \(raster) vs \(expected)")
            #expect(isNearly(SIMD4(trace.x, trace.y, trace.z, 1), SIMD4(expected.x, expected.y, expected.z, 1), tolerance: 0.002),
                    "traced \(trace) vs \(expected)")
        }
        // With a base color texture, raster saturates the tinted vertex color before the
        // texture multiplies it: saturate(0.5 × 2) × 0.502, not saturate(0.5 × 2 × 0.502) × 1.
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        descriptor.usage = .shaderRead
        let gray = try #require(fixture.device.makeTexture(descriptor: descriptor))
        gray.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: [UInt8](arrayLiteral: 128, 128, 128, 255), bytesPerRow: 4)
        let textured = SceneGraph(roots: [SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float,
                geometry: try Self.gradientSquare(1, left: SIMD4(2, 4, 2, 1), right: SIMD4(2, 4, 2, 1)),
                material: .physicallyBased(try PhysicallyBasedMaterial(
                    baseColorTexture: gray, baseColorFactor: SIMD4(0.5, 0.5, 0.5, 1), alphaMode: .opaque, shadingModel: .unlit
                ))
            )
        )], camera: fixture.camera)
        let level = Float(128) / 255
        let texturedRaster = try fixture.rasterize(textured, width: width, height: height)[center]
        let texturedTrace = try traced(textured)
        #expect(simd_reduce_max(abs(SIMD3(texturedRaster.x, texturedRaster.y, texturedRaster.z) - SIMD3(repeating: level))) < 0.003,
                "raster \(texturedRaster)")
        #expect(simd_reduce_max(abs(SIMD3(texturedTrace.x, texturedTrace.y, texturedTrace.z) - SIMD3(repeating: level))) < 0.003,
                "traced \(texturedTrace)")

        let color = SIMD4<Float>(1, 1, 1, 0.5)
        let factor = SIMD4<Float>(0.9, 0.8, 0.7, 1)
        let translucent = try unlit(color, factor: factor, alphaMode: .materialAlpha)
        let written = SIMD4<Float>(0.45, 0.4, 0.35, 0.5)
        #expect(isNearly(try traced(translucent), written, tolerance: 0.002))
        #expect(isNearly(try fixture.rasterize(translucent, width: width, height: height)[center], written, tolerance: 0.002))
        let masked = try unlit(color, factor: factor, alphaMode: .mask(cutoff: 0.75))
        #expect(try traced(masked) == .zero)
        let blended = try unlit(color, factor: factor, alphaMode: .blend)
        #expect(isNearly(try traced(blended, samples: 256), written, tolerance: 0.12))
        // A tint that lifts every vertex to full alpha is opaque in both renderers.
        let lifted = try unlit(SIMD4(1, 1, 1, 0.5), factor: SIMD4(0.9, 0.8, 0.7, 1), alphaMode: .materialAlpha)
        var liftedGraph = lifted
        liftedGraph.roots[0].materialParameters = SIMD4(1, 1, 1, 2)
        let liftedTrace = try traced(liftedGraph)
        #expect(simd_reduce_max(abs(SIMD3(liftedTrace.x, liftedTrace.y, liftedTrace.z) - SIMD3(0.9, 0.8, 0.7))) < 0.002, "\(liftedTrace)")
        #expect(isNearly(try fixture.rasterize(liftedGraph, width: width, height: height)[center], SIMD4(0.9, 0.8, 0.7, 1), tolerance: 0.002))
        _ = try PathTracingScene(
            queue: fixture.queue, graph: try unlit(SIMD4(1, 1, 1, 1), factor: SIMD4(0.9, 0.8, 0.7, 1), alphaMode: .materialAlpha),
            textureSize: 16
        )
    }

    // MARK: glTF

    func asset(color: (data: [UInt8], type: String, component: Int, normalized: Bool)?, unlit: Bool = false) throws -> Data {
        var fixture = GLTFMotionFixture()
        let positions = fixture.floats([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], type: "VEC3", count: 4)
        let normals = fixture.floats([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], type: "VEC3", count: 4)
        let indices = fixture.bytes([0, 1, 2, 0, 2, 3], type: "SCALAR", count: 6, component: 5121)
        var attributes = ["POSITION": positions, "NORMAL": normals]
        if let color {
            attributes["COLOR_0"] = fixture.bytes(
                color.data, type: color.type, count: 4, component: color.component, normalized: color.normalized
            )
        }
        var sections: [String: Any] = [
            "nodes": [["mesh": 0]], "meshes": [["primitives": [["attributes": attributes, "indices": indices, "material": 0]]]],
            "materials": [unlit ? ["extensions": ["KHR_materials_unlit": [String: Any]()]] : [String: Any]()],
        ]
        if unlit { sections["extensionsUsed"] = ["KHR_materials_unlit"] }
        return try fixture.document(sections)
    }

    @Test("glTF COLOR_0 imports floats and normalized unsigned bytes and shorts as linear RGBA")
    func gltfImport() throws {
        let floats: [Float] = [1, 0, 0, 0.5, 0, 1, 0, 1, 0, 0, 1, 0.25, 0.5, 0.5, 0.5, 1]
        let floatBytes = floats.flatMap { withUnsafeBytes(of: $0.bitPattern.littleEndian) { Array($0) } }
        let floatColors = try GLTFMeshLoader.load(data: asset(color: (floatBytes, "VEC4", 5126, false))).vertices.map(\.color)
        #expect(floatColors == stride(from: 0, to: 16, by: 4).map { SIMD4(floats[$0], floats[$0 + 1], floats[$0 + 2], floats[$0 + 3]) })

        // VEC3 unsigned bytes are opaque.
        let bytes: [UInt8] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 51, 102, 204]
        let byteColors = try GLTFMeshLoader.load(data: asset(color: (bytes, "VEC3", 5121, true))).vertices.map(\.color)
        #expect(byteColors[3] == SIMD4(0.2, 0.4, 0.8, 1))
        #expect(byteColors[0] == SIMD4(1, 0, 0, 1))

        let shorts: [UInt16] = [65_535, 0, 0, 32_768, 0, 65_535, 0, 65_535, 0, 0, 65_535, 0, 13_107, 26_214, 39_321, 65_535]
        let shortBytes = shorts.flatMap { withUnsafeBytes(of: $0.littleEndian) { Array($0) } }
        let shortColors = try GLTFMeshLoader.load(data: asset(color: (shortBytes, "VEC4", 5123, true))).vertices.map(\.color)
        #expect(shortColors[0] == SIMD4(1, 0, 0, Float(32_768) / 65_535))
        #expect(shortColors[3] == SIMD4(0.2, 0.4, 0.6, 1))

        #expect(throws: GLTFMeshLoader.Failure.self) {
            try GLTFMeshLoader.load(data: asset(color: (bytes, "VEC3", 5121, false)))
        }
        #expect(throws: GLTFMeshLoader.Failure.self) {
            try GLTFMeshLoader.load(data: asset(color: (bytes, "VEC3", 5120, true)))
        }
        #expect(throws: GLTFMeshLoader.Failure.self) {
            try GLTFMeshLoader.load(data: asset(color: (bytes, "VEC2", 5121, true)))
        }
    }

    @Test("A glTF scene with COLOR_0 and materials renders the colors and exports them back")
    func gltfSceneRoundTrip() throws {
        let fixture = try GeometryRenderFixture()
        // Corners (−1, −1) red, (1, −1) blue, (1, 1) yellow and (−1, 1) red.
        let bytes: [UInt8] = [255, 0, 0, 0, 0, 255, 255, 255, 0, 255, 0, 0]
        let scene = try GLTFMeshLoader.loadScene(data: asset(color: (bytes, "VEC3", 5121, true), unlit: true), includeMaterials: true)
        let prepared = try PreparedGLTFScene(device: fixture.device, colorPixelFormat: .rgba16Float, scene: scene)
        let graph = try prepared.graph(camera: fixture.camera, cameraTransform: SceneTransform(translation: SIMD3(0, 0, 2)))
        let image = try fixture.render(graph).pixels
        let red = SIMD3<Float>(1, 0, 0), blue = SIMD3<Float>(0, 0, 1), yellow = SIMD3<Float>(1, 1, 0)
        func expected(column: Int, row: Int) -> SIMD4<Float> {
            // Barycentric interpolation over the triangles (0, 1, 2) and (0, 2, 3).
            let x = ((Float(column) + 0.5) / 32 - 1) * 2
            let y = (1 - (Float(row) + 0.5) / 32) * 2
            if y <= x {
                let b = (y + 1) / 2
                let a = (x + 1) / 2 - b
                return SIMD4(red * (1 - a - b) + blue * a + yellow * b, 1)
            }
            let a = (x + 1) / 2
            let b = (y + 1) / 2 - a
            return SIMD4(red * (1 - a - b) + yellow * a + red * b, 1)
        }
        for (column, row) in [(17, 32), (46, 46), (46, 17), (30, 20), (40, 40)] {
            #expect(isNearly(fixture.pixel(image, x: column, y: row), expected(column: column, row: row), tolerance: 0.01), "\(column), \(row)")
        }

        let exported = try GLTFExporter(commandQueue: fixture.queue).export(graph, includesCamera: false)
        let reloaded = try GLTFMeshLoader.loadScene(data: exported.data)
        let colors = try #require(reloaded.meshes.first?.primitives.first).vertices.map(\.color)
        #expect(colors == scene.meshes[0].primitives[0].vertices.map(\.color))
        #expect(exported.warnings.isEmpty)
    }

    // MARK: Alpha map

    @Test("A PBR alpha map's green channel multiplies alpha on its own UV transform")
    func pbrAlphaMap() throws {
        let fixture = try GeometryRenderFixture()
        // Left texel green 64, right texel green 191; the transform mirrors U.
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 1, mipmapped: false)
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        texture.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: [UInt8]([0, 64, 0, 255, 0, 191, 0, 255]), bytesPerRow: 8)
        let sampler = try #require(fixture.device.makeSamplerState(descriptor: {
            let descriptor = MTLSamplerDescriptor()
            descriptor.minFilter = .nearest
            descriptor.magFilter = .nearest
            return descriptor
        }()))
        func render(
            _ transform: PhysicallyBasedMaterial.TextureTransform, animatedAlpha: Float? = nil
        ) throws -> [SIMD4<Float>] {
            let authored = try PhysicallyBasedMaterial(
                baseColorFactor: SIMD4(0.5, 1, 0.25, 0.8), alphaMode: .blend, shadingModel: .unlit,
                alphaTexture: .init(texture: texture, sampler: sampler, transform: transform)
            )
            let material = animatedAlpha == nil ? authored : authored.withLiveFactors()
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.plane(width: 2, height: 2),
                material: .physicallyBased(material)
            )
            if let animatedAlpha {
                let live = try #require(material.liveFactors)
                try live.update { $0.baseColorFactor.w = animatedAlpha }
            }
            return try fixture.render(SceneGraph(
                roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: fixture.camera
            )).pixels
        }
        let low = 0.8 * Float(64) / 255
        let high = 0.8 * Float(191) / 255
        let plain = try render(.identity)
        #expect(isNearly(fixture.pixel(plain, x: 20, y: 32), SIMD4(0.5 * low, low, 0.25 * low, low)))
        #expect(isNearly(fixture.pixel(plain, x: 44, y: 32), SIMD4(0.5 * high, high, 0.25 * high, high)))
        let mirrored = try render(try .init(offset: SIMD2(1, 0), scale: SIMD2(-1, 1)))
        #expect(isNearly(fixture.pixel(mirrored, x: 20, y: 32), SIMD4(0.5 * high, high, 0.25 * high, high)))
        let animated = try render(.identity, animatedAlpha: 0.4)
        #expect(isNearly(fixture.pixel(animated, x: 20, y: 32), SIMD4(0.5 * low, low, 0.25 * low, low) * 0.5))
        #expect(isNearly(fixture.pixel(animated, x: 44, y: 32), SIMD4(0.5 * high, high, 0.25 * high, high) * 0.5))
        #expect(throws: PhysicallyBasedMaterial.Failure.alphaTextureSampler) {
            try PhysicallyBasedMaterial(baseColorTexture: texture, alphaTexture: .init(texture: texture, sampler: sampler))
        }
        // The path tracer does not take alpha maps.
        let path = try PathTestFixture()
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.plane(),
            material: .physicallyBased(try PhysicallyBasedMaterial(alphaMode: .opaque, alphaTexture: .init(texture: texture)))
        )
        #expect(throws: PathTracingScene.Failure.self) {
            try PathTracingScene(queue: path.queue, graph: SceneGraph(roots: [SceneNode(mesh: mesh)], camera: path.camera))
        }
    }
}
