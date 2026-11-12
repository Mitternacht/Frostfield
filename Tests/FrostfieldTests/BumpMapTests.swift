import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct BumpMapTests {
    typealias BumpMap = PhysicallyBasedMaterial.BumpMap
    typealias Binding = PhysicallyBasedMaterial.TextureBinding

    private let gpu: PathTestFixture
    private let side = 17
    private var center: Int { side / 2 * side + side / 2 }
    private let strength: Float = 5

    init() throws {
        gpu = try PathTestFixture()
    }

    @Test("PBR bump maps use transformed UV1 and signed strength in the raster normal and lighting")
    func raster() throws {
        let map = try heightMap(scale: strength)
        let graph = try scene(bump: map)
        let normal = try gpu.rasterSurfaceNormal(graph, side: side)
        let flat = try scene()
        let flatNormal = try gpu.rasterSurfaceNormal(flat, side: side)
        #expect(simd_distance(normal, expectedNormal(scale: strength)) < 0.003, "\(normal)")
        #expect(flatNormal == SIMD3(0, 0, 1))
        let bumped = try gpu.rasterize(graph, width: side, height: side)[center]
        let smooth = try gpu.rasterize(flat, width: side, height: side)[center]
        #expect(bumped.x > smooth.x + 0.015 && bumped.w == smooth.w)

        let mirrored = try scene(bump: heightMap(scale: strength, mirror: true))
        let negative = try scene(bump: heightMap(scale: -strength))
        #expect(simd_distance(try gpu.rasterSurfaceNormal(mirrored, side: side), expectedNormal(scale: -strength)) < 0.003)
        #expect(simd_distance(try gpu.rasterSurfaceNormal(negative, side: side), expectedNormal(scale: -strength)) < 0.003)
        let uniform = try scene(bump: heightMap(scale: strength, flat: true))
        #expect(try gpu.rasterSurfaceNormal(uniform, side: side) == flatNormal)
    }

    @Test("First-hit bump normals match the ray footprint in software and Metal tracing",
          arguments: PathAccelerationTests.modes)
    func path(acceleration: PathTracingScene.Options.Acceleration) throws {
        let graph = try scene(bump: heightMap(scale: strength, mirror: true))
        for textures in [PathTracingScene.Options.Textures.fullResolution(fallbackSize: 64), .baked(size: 64)] {
            let normal = try pathNormal(graph, acceleration: acceleration, textures: textures)
            #expect(simd_distance(normal, SIMD4(expectedNormal(scale: -strength), 1)) < 0.005, "\(normal)")
        }
    }

    @Test("A bound normal map wins over height and zero bump scale leaves normals flat")
    func precedence() throws {
        let normalTexture = try texture(.rgba16Float, width: 1, height: 1)
        var normal = SIMD4<Float16>(0.5, 0.5, 1, 1)
        normalTexture.replace(
            region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &normal, bytesPerRow: 8
        )
        let bump = try heightMap(scale: strength)
        let zero = try scene(bump: heightMap(scale: 0))
        let both = try scene(bump: bump, normal: Binding(texture: normalTexture))
        let flat = SIMD3<Float>(0, 0, 1)
        #expect(try gpu.rasterSurfaceNormal(both, side: side) == flat)
        #expect(try gpu.rasterSurfaceNormal(zero, side: side) == flat)
        #expect(try pathNormal(both) == SIMD4(flat, 1))
        #expect(try pathNormal(zero) == SIMD4(flat, 1))
    }

    @Test("A height map on UV1 leaves the authored tangent frame of a UV0 clearcoat normal intact")
    func clearcoatIndependence() throws {
        let normalTexture = try texture(.rgba16Float, width: 1, height: 1)
        var normal = SIMD4<Float16>(0.7, 0.5, 1, 1)
        normalTexture.replace(
            region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &normal, bytesPerRow: 8
        )
        let tilt = simd_normalize(SIMD3(2 * Float(normal.x) - 1, 0, 1))
        let light = 2 * tilt.z * tilt - SIMD3(0, 0, 1)
        let coat = try PhysicallyBasedMaterial.Clearcoat(
            factor: 1, roughness: 0.3, normalTexture: Binding(texture: normalTexture)
        )
        func sample(_ bump: BumpMap?) throws -> Float {
            let material = try PhysicallyBasedMaterial(
                roughness: 1, directionToLight: light, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                alphaMode: .opaque, clearcoat: coat, specular: .init(factor: 0), bumpMap: bump
            )
            let graph = try scene(material: material)
            return try gpu.rasterize(graph, width: side, height: side)[center].x
        }
        let reference = try sample(nil)
        let bumped = try sample(heightMap(scale: strength))
        #expect(reference > 0.1 && abs(bumped - reference) < 0.003, "\(bumped), \(reference)")
    }

    @Test("Bump maps survive live factors and versioned archives and reject portable height export")
    func persistence() throws {
        let bump = try heightMap(scale: -strength, mirror: true)
        let material = try pbr(bump: bump).withLiveFactors()
        let graph = try scene(material: material)
        let live = try #require(material.liveFactors)
        var factors = live.current
        factors.roughness = 0.17
        try live.set(factors)

        let normal = try gpu.rasterSurfaceNormal(graph, side: side)
        #expect(simd_distance(normal, expectedNormal(scale: strength)) < 0.003)
        let archive = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        var manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 14)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        #expect(try gpu.rasterSurfaceNormal(restored, side: side) == normal)
        #expect(try gpu.rasterize(restored, width: side, height: side) == gpu.rasterize(graph, width: side, height: side))

        manifest["version"] = 13
        let old = try JSONSerialization.data(withJSONObject: manifest)
        #expect(throws: SceneArchive.Failure.decoding("bump maps require archive version 14")) {
            try SceneArchive(manifest: old, binary: archive.binary)
        }
        #expect(throws: GLTFExporter.Failure.unsupportedTexture("Convert bump maps to tangent-space normal maps before glTF export.")) {
            try GLTFExporter(commandQueue: gpu.queue).export(restored)
        }
        #expect(throws: USDZExporter.Failure.invalidData("Convert bump maps to tangent-space normal maps before USDZ export.")) {
            try USDZExporter(commandQueue: gpu.queue).export(restored)
        }
    }

    @Test("Height bindings require a finite scale and a linear sampleable texture")
    func validation() throws {
        let linear = try texture(.r8Unorm)
        let srgb = try texture(.rgba8Unorm_srgb)
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidBumpScale) {
            try BumpMap(texture: Binding(texture: linear), scale: .infinity)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidBumpTexture) {
            try BumpMap(texture: Binding(texture: srgb))
        }
    }

    @Test("Classic materials share signed height gradients with raster PBR")
    func classicHeight() throws {
        let bump = try heightMap(scale: strength)
        let matcap = try texture(.rgba16Float, width: 1, height: 1)
        var white = SIMD4<Float16>(repeating: 1)
        matcap.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &white, bytesPerRow: 8)
        let models: [ClassicMaterial.Model] = [.lambert, .phong(), .toon(), .matcap(Binding(texture: matcap)), .normal]
        let expected = expectedNormal(scale: strength)
        for model in models {
            let graph = try scene(material: .classic(ClassicMaterial(model: model, bumpMap: bump)))
            #expect(simd_distance(try gpu.rasterSurfaceNormal(graph, side: side), expected) < 0.003)
        }
        let mirrored = try scene(material: .classic(ClassicMaterial(model: .normal, bumpMap: heightMap(scale: strength, mirror: true))))
        let reversed = try scene(material: .classic(ClassicMaterial(model: .normal, bumpMap: heightMap(scale: -strength))))
        let zero = try scene(material: .classic(ClassicMaterial(model: .normal, bumpMap: heightMap(scale: 0))))
        let encoded = try gpu.rasterize(reversed, width: side, height: side)[center]
        let expectedColor = SIMD4(expectedNormal(scale: -strength) * 0.5 + 0.5, 1)
        #expect(simd_distance(encoded, expectedColor) < 0.003)
        let mirroredColor = try gpu.rasterize(mirrored, width: side, height: side)[center]
        // Sampling 1 − u traverses the opposite half-float texels and
        // reverses interpolation order. Compare within the output precision.
        #expect(simd_distance(mirroredColor, encoded) <= 0.001, "\(mirroredColor) vs \(encoded)")
        #expect(try gpu.rasterSurfaceNormal(zero, side: side) == SIMD3(0, 0, 1))

        let lightDirection = simd_normalize(SIMD3<Float>(-0.8, 0, 1))
        let light = try SceneNode(
            transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -lightDirection)),
            light: PunctualLight(kind: .directional, intensity: .pi)
        )
        var lit = try scene(material: .classic(ClassicMaterial(model: .phong(specular: .zero), bumpMap: bump)))
        var smooth = try scene(material: .classic(ClassicMaterial(model: .phong(specular: .zero))))
        lit.roots.append(light)
        smooth.roots.append(light)
        let bumped = try gpu.rasterize(lit, width: side, height: side)[center]
        let flat = try gpu.rasterize(smooth, width: side, height: side)[center]
        #expect(bumped.x > flat.x + 0.03 && bumped.w == flat.w)
    }

    @Test("Classic normal maps take precedence and preserve signed axes and archive versions")
    func classicNormalsAndArchive() throws {
        let image = try texture(.rgba16Float, width: 1, height: 1)
        var value = SIMD4<Float16>(0.75, 0.75, 1, 1)
        image.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &value, bytesPerRow: 8)
        let bump = try heightMap(scale: strength)
        let tangent = Binding(texture: image, textureCoordinateSet: .secondary)
        let both = try ClassicMaterial(model: .normal, bumpMap: bump, normalMap: tangent, normalScale: SIMD2(-1, 0.5))
        let graph = try scene(material: .classic(both))
        let normal = try gpu.rasterSurfaceNormal(graph, side: side)
        #expect(simd_distance(normal, simd_normalize(SIMD3(-0.5, -0.25, 1))) < 0.003, "\(normal)")

        let object = Binding(texture: image, normalSpace: .objectSpace)
        var scaled = try scene(material: .classic(ClassicMaterial(model: .normal, normalMap: object, normalScale: .zero)))
        scaled.roots[0].transform.scale = SIMD3(2, 1, 0.5)
        #expect(simd_distance(try gpu.rasterSurfaceNormal(scaled, side: side), simd_normalize(SIMD3(0.25, 0.5, 2))) < 0.003)

        let archive = try SceneArchive(capturing: graph, commandQueue: gpu.queue)
        var manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 15)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        #expect(try gpu.rasterize(restored, width: side, height: side) == gpu.rasterize(graph, width: side, height: side))
        let renderer = try #require(restored.roots.first?.mesh)
        guard case .classic(let material) = renderer.material else {
            Issue.record("Expected the archived classic material")
            return
        }
        #expect(material.bumpMap?.scale == strength && material.normalScale == SIMD2(-1, 0.5))
        manifest["version"] = 14
        let old = try JSONSerialization.data(withJSONObject: manifest)
        #expect(throws: SceneArchive.Failure.decoding("classic surface detail requires archive version 15")) {
            try SceneArchive(manifest: old, binary: archive.binary)
        }
    }

    @Test("Classic detail rejects unused properties and validates only the active texture coordinates")
    func classicDetailValidation() throws {
        let bump = try heightMap(scale: strength)
        #expect(throws: ClassicMaterial.Failure.unusedProperty("bumpMap or normalMap")) {
            try ClassicMaterial(model: .basic, bumpMap: bump)
        }
        let plane = try MeshGeometry.plane()
        func prepare(_ material: ClassicMaterial) throws -> MeshRenderer {
            try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .classic(material))
        }
        #expect(throws: MeshRenderer.Failure.missingSecondaryTextureCoordinates) {
            try prepare(ClassicMaterial(model: .lambert, bumpMap: bump))
        }
        _ = try prepare(ClassicMaterial(model: .lambert, bumpMap: heightMap(scale: 0)))
        let srgb = try texture(.rgba8Unorm_srgb)
        #expect(throws: ClassicMaterial.Failure.texture(6)) {
            try prepare(ClassicMaterial(model: .lambert, normalMap: Binding(texture: srgb)))
        }
    }

    @Test("The typed height node renders an affine UV1 bump with live strength and a native archive")
    func graphRaster() throws {
        let bump = try heightMap(scale: strength, mirror: true)
        let material = try graphMaterial(bump)
        let scene = try scene(material: material, usesGraphFrame: true)
        let expected = expectedNormal(scale: -strength)
        #expect(simd_distance(try gpu.rasterSurfaceNormal(scene, side: side), expected) < 0.003)

        let uniforms = try #require(material.surfaceUniforms)
        try uniforms.set("strength", to: -strength)
        let flipped = try gpu.rasterSurfaceNormal(scene, side: side)
        #expect(simd_distance(flipped, expectedNormal(scale: strength)) < 0.003)
        let archive = try SceneArchive(capturing: scene, commandQueue: gpu.queue)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        #expect(try gpu.rasterSurfaceNormal(restored, side: side) == flipped)
    }

    @Test("A graph height node follows the ray footprint with native and baked textures",
          arguments: PathAccelerationTests.modes)
    func graphPath(acceleration: PathTracingScene.Options.Acceleration) throws {
        let material = try graphMaterial(heightMap(scale: strength, mirror: true))
        let scene = try scene(material: material, usesGraphFrame: true)
        for textures in [PathTracingScene.Options.Textures.fullResolution(fallbackSize: 64), .baked(size: 64)] {
            let normal = try pathNormal(scene, acceleration: acceleration, textures: textures)
            #expect(simd_distance(normal, SIMD4(expectedNormal(scale: -strength), 1)) < 0.005, "\(normal)")
        }
    }

    @Test("A scalar height follows the same footprint as a red-channel bump")
    func scalarHeight() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(normal: s.bump(height: s.uv1.x, scale: s.constant(Float(strength))))
        }
        let material = try PhysicallyBasedMaterial(alphaMode: .opaque, surfaceGraph: graph)
        let bumped = try scene(material: material, usesGraphFrame: true)
        let expected = expectedNormal(scale: strength)
        let raster = try gpu.rasterSurfaceNormal(bumped, side: side)
        #expect(simd_distance(raster, expected) < 0.003, "\(raster)")
        let traced = try pathNormal(bumped)
        #expect(simd_distance(traced, SIMD4(expected, 1)) < 0.005, "\(traced)")

        let flatGraph = try SurfaceShaderGraph { s in
            .init(normal: s.bump(height: s.constant(Float(0.4)), scale: s.constant(Float(strength))))
        }
        let flat = try scene(
            material: try PhysicallyBasedMaterial(alphaMode: .opaque, surfaceGraph: flatGraph), usesGraphFrame: true
        )
        #expect(try gpu.rasterSurfaceNormal(flat, side: side) == SIMD3(0, 0, 1))

        let archive = try SceneArchive(capturing: bumped, commandQueue: gpu.queue)
        let json = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(json["version"] as? Int == 41)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        #expect(try gpu.rasterSurfaceNormal(restored, side: side) == raster)

        #expect(throws: MaterialGraph.Failure.invalidDeclaration(
            "A scalar bump height cannot read derivatives, coverage, or the frame."
        )) {
            try SurfaceShaderGraph { s in
                .init(normal: s.bump(height: s.uv1.x.dFdx()))
            }
        }
    }

    @Test("A graph bump node requires a data texture and authored secondary coordinates")
    func graphValidation() throws {
        #expect(throws: MaterialGraph.Failure.invalidDeclaration("A bump map needs a linear data texture.")) {
            try SurfaceShaderGraph { graph in
                .init(normal: graph.texture("height", content: .color).bumpMap())
            }
        }
        let material = try graphMaterial(heightMap(scale: strength))
        #expect(throws: MeshRenderer.Failure.missingSecondaryTextureCoordinates) {
            try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: MeshGeometry.plane(),
                material: .physicallyBased(material)
            )
        }
    }

    private func graphMaterial(_ bump: BumpMap) throws -> PhysicallyBasedMaterial {
        let graph = try SurfaceShaderGraph { graph in
            let height = graph.texture("height", content: .data, addressing: .clampToEdge)
            let scale = graph.uniform("strength", bump.scale)
            return .init(normal: height.bumpMap(
                coordinateSet: .secondary, transform: bump.texture.transform, scale: scale
            ))
        }
        return try PhysicallyBasedMaterial(
            alphaMode: .opaque, surfaceGraph: graph, surfaceTextures: ["height": bump.texture.texture]
        )
    }

    private func texture(_ format: MTLPixelFormat, width: Int = 2, height: Int = 2) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func heightMap(scale: Float, mirror: Bool = false, flat: Bool = false) throws -> BumpMap {
        let width = 64
        let texture = try texture(.r16Float, width: width, height: 2)
        let row = (0..<width).map { column in Float16(flat ? 0.4 : (Float(column) + 0.5) / Float(width)) }
        (row + row).withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, 2), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: width * 2
            )
        }
        let transform: PhysicallyBasedMaterial.TextureTransform = mirror
            ? try .init(offset: SIMD2(1, 0), scale: SIMD2(-1, 1)) : .identity
        return try BumpMap(texture: Binding(
            texture: texture, textureCoordinateSet: .secondary, transform: transform,
            sampling: .init(addressU: .clampToEdge, addressV: .clampToEdge)
        ), scale: scale)
    }

    private func expectedNormal(scale: Float) -> SIMD3<Float> {
        // The plane is two world units away. Its UV1 advances a quarter per
        // world unit, and the texture's red channel is exactly u in the interior.
        let worldPixel = 4 * tan(gpu.camera.verticalFieldOfView / 2) / Float(side)
        return simd_normalize(SIMD3(-scale * worldPixel / 4, 0, 1))
    }

    private func pbr(bump: BumpMap? = nil, normal: Binding? = nil) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            directionToLight: SIMD3(-0.8, 0, 1), ambientIntensity: 0, normalTexture: normal,
            alphaMode: .opaque, bumpMap: bump
        )
    }

    private func scene(bump: BumpMap? = nil, normal: Binding? = nil) throws -> SceneGraph {
        try scene(material: pbr(bump: bump, normal: normal))
    }

    private func scene(material: PhysicallyBasedMaterial, usesGraphFrame: Bool = false) throws -> SceneGraph {
        try scene(material: .physicallyBased(material), usesGraphFrame: usesGraphFrame)
    }

    private func scene(material: MeshMaterial, usesGraphFrame: Bool = false) throws -> SceneGraph {
        let plane = try MeshGeometry.plane(width: 4, height: 4)
        let geometry = try MeshGeometry(vertices: plane.vertices.map { vertex in
            let p = vertex.position
            // The graph output is in the UV0 frame, which is rotated relative
            // to the height texture's UV1. Give the other tests a fixed UV0.
            let primary = usesGraphFrame ? SIMD2(0.5 + p.y * 0.125, 0.5 + p.x * 0.125) : SIMD2(0.5, 0.5)
            return MeshVertex(
                position: p, normal: vertex.normal, textureCoordinate: primary,
                secondaryTextureCoordinate: SIMD2(0.5 + p.x * 0.25, 0.5 - p.y * 0.25),
                tangent: usesGraphFrame ? nil : SIMD4(1, 0, 0, 1)
            )
        }, indices: plane.indices)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material
        )
        return SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: gpu.camera)
    }

    private func pathNormal(
        _ graph: SceneGraph, acceleration: PathTracingScene.Options.Acceleration = .software,
        textures: PathTracingScene.Options.Textures = .fullResolution(fallbackSize: 64)
    ) throws -> SIMD4<Float> {
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(textures: textures, acceleration: acceleration))
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: .init(environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        let normal = try gpu.output(width: side, height: side)
        let beauty = try gpu.output(width: side, height: side)
        let command = try #require(gpu.queue.makeCommandBuffer())
        try tracer.encode(
            camera: gpu.camera, cameraTransform: .init(), destination: beauty, accumulation: accumulation,
            commandBuffer: command, settings: .init(maxBounces: 1), aovs: .init(normal: normal)
        )
        try gpu.finish(command)
        return try gpu.readback(normal)[center]
    }
}
