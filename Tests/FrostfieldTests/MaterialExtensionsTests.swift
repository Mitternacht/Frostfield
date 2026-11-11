import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Renders one material with the raster renderer and the path tracer into square
/// RGBA16Float images and returns linear pixels, top row first.
struct MaterialRenderFixture {
    static let side = 33
    let device: any MTLDevice
    let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    func raster(
        _ material: PhysicallyBasedMaterial, geometry: MeshGeometry, transform: SceneTransform,
        camera: PerspectiveCamera
    ) throws -> [SIMD4<Float>] {
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        let color = try texture(.rgba16Float, usage: .renderTarget, storage: .shared)
        let depth = try texture(.depth32Float, usage: .renderTarget, storage: .private)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try renderer.encode(MeshScene(camera: camera, transform: transform), pass: pass, commandBuffer: buffer)
        try finish(buffer)
        return try pixels(color)
    }

    /// Traces the nodes with `samples` per pixel, in dispatches of at most 16.
    func trace(
        _ nodes: [(geometry: MeshGeometry, material: PhysicallyBasedMaterial, transform: SceneTransform)],
        camera: PerspectiveCamera, lighting: PathTracer.Lighting, samples: Int = 16, bounces: Int = 2,
        textureSize: Int = 32, options: PathTracingScene.Options? = nil
    ) throws -> [SIMD4<Float>] {
        let roots = try nodes.map { node in
            SceneNode(transform: node.transform, mesh: try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: node.geometry,
                material: .physicallyBased(node.material)
            ))
        }
        let graph = SceneGraph(roots: roots, camera: camera)
        let scene = try options.map { try PathTracingScene(queue: queue, graph: graph, options: $0) }
            ?? PathTracingScene(queue: queue, graph: graph, textureSize: textureSize)
        let tracer = try PathTracer(queue: queue, scene: scene, lighting: lighting)
        let accumulation = try tracer.makeAccumulation(width: Self.side, height: Self.side)
        let target = try texture(.rgba16Float, usage: [.shaderRead, .shaderWrite], storage: .shared)
        let settings = PathTracer.Settings(maxBounces: bounces, seed: 17)
        var history: PathTracer.History?
        var remaining = samples
        while remaining > 0 {
            let batch = min(16, remaining)
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: batch, settings: settings
            )
            try finish(buffer)
            remaining -= batch
        }
        return try pixels(target)
    }

    static func pixel(_ image: [SIMD4<Float>], x: Int, y: Int) -> SIMD4<Float> {
        image[y * side + x]
    }

    func dataTexture(width: Int, height: Int, pixels: [UInt8]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        try pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 4
            )
        }
        return texture
    }

    private func texture(
        _ format: MTLPixelFormat, usage: MTLTextureUsage, storage: MTLStorageMode
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: Self.side, height: Self.side, mipmapped: false
        )
        descriptor.storageMode = storage
        descriptor.usage = usage
        descriptor.hazardTrackingMode = .tracked
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func finish(_ buffer: any MTLCommandBuffer) throws {
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "No Metal error")")
    }

    private func pixels(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var result = Array(repeating: SIMD4<Float16>.zero, count: Self.side * Self.side)
        try result.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: Self.side * 8,
                from: MTLRegionMake2D(0, 0, Self.side, Self.side), mipmapLevel: 0
            )
        }
        return result.map(SIMD4<Float>.init)
    }
}

struct MaterialExtensionValidationTests {
    @Test("Extension layers validate their documented ranges")
    func validation() throws {
        typealias Material = PhysicallyBasedMaterial
        #expect(throws: Material.Failure.invalidEmissiveStrength) { try Material(emissiveStrength: -1) }
        #expect(throws: Material.Failure.invalidEmissiveStrength) { try Material(emissiveStrength: .infinity) }
        #expect(throws: Material.Failure.invalidSpecular) { try Material.Specular(factor: 1.5) }
        #expect(throws: Material.Failure.invalidSpecular) { try Material.Specular(colorFactor: SIMD3(1, -0.1, 1)) }
        #expect(throws: Material.Failure.invalidSpecular) { try Material.Specular(colorFactor: SIMD3(1, .nan, 1)) }
        #expect(throws: Material.Failure.invalidSheen) { try Material.Sheen(colorFactor: SIMD3(1.2, 0, 0)) }
        #expect(throws: Material.Failure.invalidSheen) { try Material.Sheen(roughness: -0.5) }
        #expect(throws: Material.Failure.invalidAnisotropy) { try Material.Anisotropy(strength: 2) }
        #expect(throws: Material.Failure.invalidAnisotropy) { try Material.Anisotropy(rotation: .infinity) }
        #expect(throws: Material.Failure.invalidIridescence) { try Material.Iridescence(factor: -1) }
        #expect(throws: Material.Failure.invalidIridescence) { try Material.Iridescence(indexOfRefraction: 0.9) }
        #expect(throws: Material.Failure.invalidIridescence) { try Material.Iridescence(minimumThickness: -1) }
        #expect(throws: Material.Failure.invalidIridescence) { try Material.Iridescence(maximumThickness: .nan) }
        #expect(throws: Material.Failure.invalidDispersion) { try Material.Transmission(factor: 1, dispersion: -1) }
        #expect(throws: Material.Failure.invalidDispersion) { try Material.Transmission(factor: 1, dispersion: .nan) }

        // Emissive strength lifts the emission above the factor's zero-to-one range.
        let bright = try Material(emissiveFactor: SIMD3(1, 0.5, 0), emissiveStrength: 40)
        #expect(bright.emissiveStrength == 40)
        let specular = try Material.Specular(factor: 0.25, colorFactor: SIMD3(2, 1, 0))
        #expect(specular.colorFactor == SIMD3(2, 1, 0))
        let iridescence = try Material.Iridescence(factor: 1)
        #expect(iridescence.indexOfRefraction == 1.3 && iridescence.minimumThickness == 100 && iridescence.maximumThickness == 400)
    }

    @Test("Raster and path tracing extension records keep their Metal layouts")
    func layouts() {
        typealias Raster = PhysicallyBasedMaterial.PackedExtensions
        #expect(MemoryLayout<Raster>.stride == 864)
        #expect(MemoryLayout<Raster>.alignment == 16)
        #expect(MemoryLayout<Raster>.offset(of: \.options) == 64)
        #expect(MemoryLayout<Raster>.offset(of: \.maps) == 80)
        #expect(MemoryLayout<Raster>.offset(of: \.transforms) == 96)
        #expect(MemoryLayout<Raster>.offset(of: \.translucency) == 800)
        #expect(MemoryLayout<Raster>.offset(of: \.scattering) == 816)
        #expect(MemoryLayout<Raster>.offset(of: \.attenuation) == 832)
        #expect(MemoryLayout<Raster>.offset(of: \.diffusion) == 848)
        typealias Path = PathTracingScene.MaterialExtensions
        #expect(MemoryLayout<Path>.stride == 592)
        #expect(MemoryLayout<Path>.alignment == 16)
        #expect(MemoryLayout<Path>.offset(of: \.options) == 64)
        #expect(MemoryLayout<Path>.offset(of: \.textures) == 80)
        #expect(MemoryLayout<Path>.offset(of: \.coordinates) == 128)
        #expect(MemoryLayout<Path>.offset(of: \.transforms) == 144)
    }

    @Test("Layer maps use at most six distinct samplers")
    func samplerLimit() throws {
        let fixture = try MaterialRenderFixture()
        let texture = try fixture.dataTexture(width: 1, height: 1, pixels: [255, 255, 255, 255])
        func binding(_ index: Int) throws -> PhysicallyBasedMaterial.TextureBinding {
            let descriptor = MTLSamplerDescriptor()
            descriptor.lodMaxClamp = Float(index + 1)
            return .init(texture: texture, sampler: try #require(fixture.device.makeSamplerState(descriptor: descriptor)))
        }
        let material = try PhysicallyBasedMaterial(
            specular: .init(factorTexture: binding(0), colorTexture: binding(1)),
            sheen: .init(colorFactor: SIMD3(repeating: 1), colorTexture: binding(2), roughnessTexture: binding(3)),
            anisotropy: .init(strength: 1, texture: binding(4)),
            iridescence: .init(factor: 1, factorTexture: binding(5), thicknessTexture: binding(6))
        )
        #expect(throws: PhysicallyBasedMaterial.Failure.layerSamplerLimit) {
            try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(), material: .physicallyBased(material)
            )
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MaterialExtensionsTests {
    private let fixture: MaterialRenderFixture
    private let camera: PerspectiveCamera
    private let center = MaterialRenderFixture.side / 2

    init() throws {
        fixture = try MaterialRenderFixture()
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 30)
    }

    private var facingPlane: SceneTransform { SceneTransform(translation: SIMD3(0, 0, -3)) }

    private func headLight(_ intensity: Float = 1) throws -> PathTracer.Lighting {
        let light = try PunctualLight(kind: .directional, intensity: intensity)
        return try PathTracer.Lighting(
            analyticLights: [PathTracer.AnalyticLight(light, direction: SIMD3(0, 0, -1))], environment: .zero
        )
    }

    @Test("Emissive strength scales emission exactly in both renderers")
    func emissiveStrength() throws {
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        func material(_ strength: Float) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveFactor: SIMD3(0.25, 0.5, 0.125), alphaMode: .opaque, emissiveStrength: strength
            )
        }
        for strength: Float in [0, 1, 6] {
            let expected = SIMD4<Float>(SIMD3(0.25, 0.5, 0.125) * strength, 1)
            let raster = try fixture.raster(material(strength), geometry: geometry, transform: facingPlane, camera: camera)
            #expect(MaterialRenderFixture.pixel(raster, x: center, y: center) == expected)
            let traced = try fixture.trace(
                [(geometry, try material(strength), facingPlane)], camera: camera,
                lighting: PathTracer.Lighting(environment: .zero), samples: 4, bounces: 1
            )
            #expect(MaterialRenderFixture.pixel(traced, x: center, y: center) == expected)
        }
    }

    @Test("Specular factor zero removes dielectric highlights and specular color tints F0")
    func specular() throws {
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        func material(_ specular: PhysicallyBasedMaterial.Specular?) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: 0.4, directionToLight: SIMD3(0, 0, 1), lightIntensity: 1, ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque, specular: specular
            )
        }
        // At normal incidence the GGX highlight is F0 D V with D = 1 / (pi alpha^2) and V = 1/4.
        let alpha: Float = 0.4 * 0.4
        let peak = 0.04 / (.pi * alpha * alpha) * 0.25
        let tint = SIMD3<Float>(1, 0.5, 0.25)
        let cases: [(PhysicallyBasedMaterial.Specular?, SIMD3<Float>)] = [
            (nil, SIMD3(repeating: peak)),
            (try .init(factor: 0), .zero),
            (try .init(factor: 0.5), SIMD3(repeating: peak * 0.5)),
            (try .init(colorFactor: tint), peak * tint),
        ]
        for (layer, expected) in cases {
            let raster = MaterialRenderFixture.pixel(
                try fixture.raster(material(layer), geometry: geometry, transform: facingPlane, camera: camera),
                x: center, y: center
            )
            let traced = MaterialRenderFixture.pixel(
                try fixture.trace([(geometry, try material(layer), facingPlane)], camera: camera, lighting: headLight()),
                x: center, y: center
            )
            for channel in 0..<3 {
                #expect(abs(raster[channel] - expected[channel]) <= 0.01 * peak + 0.002 * expected[channel])
                #expect(abs(traced[channel] - expected[channel]) <= 0.03 * peak)
            }
            if expected == .zero {
                #expect(raster == SIMD4(0, 0, 0, 1))
                #expect(traced == SIMD4(0, 0, 0, 1))
            }
        }
    }

    @Test("Sheen brightens grazing views with the Charlie and Neubelt lobe")
    func sheen() throws {
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        let color = SIMD3<Float>(0.9, 0.6, 0.3)
        let roughness: Float = 0.5
        let intensity: Float = 8
        func render(tilt: Float, sheen: PhysicallyBasedMaterial.Sheen?) throws -> (SIMD4<Float>, SIMD4<Float>, Float) {
            // Tilting about X turns the normal toward +Y; the light stays on the normal.
            let rotation = simd_quatf(angle: -tilt, axis: SIMD3(1, 0, 0))
            let normal = rotation.act(SIMD3<Float>(0, 0, 1))
            let material = try PhysicallyBasedMaterial(
                roughness: 0.5, directionToLight: normal, lightIntensity: intensity, ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque, specular: .init(factor: 0), sheen: sheen
            )
            let transform = SceneTransform(translation: SIMD3(0, 0, -3), rotation: rotation)
            let raster = try fixture.raster(material, geometry: geometry, transform: transform, camera: camera)
            let light = try PathTracer.AnalyticLight(PunctualLight(kind: .directional, intensity: intensity), direction: -normal)
            let traced = try fixture.trace(
                [(geometry, material, transform)], camera: camera,
                lighting: PathTracer.Lighting(analyticLights: [light], environment: .zero)
            )
            // The center ray sees the plane along +Z, so NoL = 1 and NoV = cos(tilt).
            let view = SIMD3<Float>(0, 0, 1)
            let halfway = simd_normalize(view + normal)
            let noV = simd_dot(normal, view)
            let noH = simd_dot(normal, halfway)
            let inverseAlpha = 1 / (roughness * roughness)
            let charlie = (2 + inverseAlpha) * pow(max(1 - noH * noH, 0.0078125), 0.5 * inverseAlpha) / (2 * .pi)
            let visibility = 1 / (4 * (1 + noV - noV))
            return (
                MaterialRenderFixture.pixel(raster, x: center, y: center),
                MaterialRenderFixture.pixel(traced, x: center, y: center),
                charlie * visibility * intensity
            )
        }
        let layer = try PhysicallyBasedMaterial.Sheen(colorFactor: color, roughness: roughness)
        let grazing = try render(tilt: 75 * .pi / 180, sheen: layer)
        let facing = try render(tilt: 0, sheen: layer)
        let plain = try render(tilt: 75 * .pi / 180, sheen: nil)
        for channel in 0..<3 {
            let expected = color[channel] * grazing.2
            #expect(abs(grazing.0[channel] - expected) <= 0.01 * expected)
            #expect(abs(grazing.1[channel] - expected) <= 0.04 * expected)
            #expect(grazing.0[channel] > 100 * facing.0[channel])
            #expect(grazing.1[channel] > 100 * facing.1[channel])
        }
        #expect(grazing.0.x > 0.2)
        #expect(plain.0 == SIMD4(0, 0, 0, 1))
        #expect(plain.1 == SIMD4(0, 0, 0, 1))
    }

    @Test("Image-lit raster sheen matches the path-traced sheen albedo under a uniform sky")
    func imageLitSheen() throws {
        var hdr = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".utf8)
        for _ in 0..<8 { hdr.append(contentsOf: [128, 128, 128, 129]) }
        let lighting = try EnvironmentLighting(
            queue: fixture.queue, radianceHDR: hdr,
            settings: .init(cubeSize: 8, diffuseSize: 4, brdfSize: 16, sampleCount: 32)
        )
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting)
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        let transform = SceneTransform(
            translation: SIMD3(0, 0, -3), rotation: simd_quatf(angle: -70 * .pi / 180, axis: SIMD3(1, 0, 0))
        )
        let color = SIMD3<Float>(0.9, 0.6, 0.3)
        func material(_ sheen: PhysicallyBasedMaterial.Sheen?, base: SIMD4<Float>) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: base, alphaMode: .opaque,
                environment: environment, specular: .init(factor: 0), sheen: sheen
            )
        }
        let layer = try PhysicallyBasedMaterial.Sheen(colorFactor: color, roughness: 0.5)
        // A white Lambertian surface measures the sky radiance the raster image lighting sees.
        let sky = try fixture.raster(material(nil, base: SIMD4(repeating: 1)), geometry: geometry, transform: transform, camera: camera)
        let raster = try fixture.raster(material(layer, base: SIMD4(0, 0, 0, 1)), geometry: geometry, transform: transform, camera: camera)
        // Under a unit sky the traced radiance is the lobe's directional albedo, color E(NoV).
        let traced = try fixture.trace(
            [(geometry, try material(layer, base: SIMD4(0, 0, 0, 1)), transform)], camera: camera,
            lighting: PathTracer.Lighting(environment: SIMD3(repeating: 1)), samples: 256, bounces: 1
        )
        var rasterAlbedo = SIMD3<Float>.zero
        var tracedAlbedo = SIMD3<Float>.zero
        for y in (center - 1)...(center + 1) {
            for x in (center - 1)...(center + 1) {
                let lit = MaterialRenderFixture.pixel(raster, x: x, y: y)
                let reference = MaterialRenderFixture.pixel(sky, x: x, y: y)
                let path = MaterialRenderFixture.pixel(traced, x: x, y: y)
                rasterAlbedo += SIMD3(lit.x, lit.y, lit.z) / reference.x / 9
                tracedAlbedo += SIMD3(path.x, path.y, path.z) / 9
            }
        }
        // E is about 0.4 near cos(70°) for roughness 0.5.
        #expect(tracedAlbedo.x > 0.25 && tracedAlbedo.x < 0.55)
        for channel in 0..<3 {
            #expect(abs(rasterAlbedo[channel] - tracedAlbedo[channel]) <= 0.06 * tracedAlbedo[channel])
        }
    }

    @Test("Surface graph variants shade material layers like the bundled shaders")
    func graphLayers() throws {
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        let graph = try SurfaceShaderGraph(nodes: [.scalar(0.3)], outputs: .init(roughness: 0))
        func material(_ graph: SurfaceShaderGraph?) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: graph == nil ? 0.3 : 0.9, directionToLight: SIMD3(0.2, 0.1, 1), ambientIntensity: 0,
                baseColorFactor: SIMD4(0.2, 0.2, 0.2, 1), alphaMode: .opaque, surfaceGraph: graph,
                specular: .init(colorFactor: SIMD3(1, 0.5, 0.25)), sheen: .init(colorFactor: SIMD3(repeating: 0.5), roughness: 0.4),
                anisotropy: .init(strength: 0.5, rotation: 0.3), iridescence: .init(factor: 0.5, maximumThickness: 300)
            )
        }
        let plain = try material(nil)
        let generated = try material(graph)
        let rasterPlain = try fixture.raster(plain, geometry: geometry, transform: facingPlane, camera: camera)
        let rasterGraph = try fixture.raster(generated, geometry: geometry, transform: facingPlane, camera: camera)
        let light = try PathTracer.AnalyticLight(
            PunctualLight(kind: .directional), direction: -simd_normalize(SIMD3<Float>(0.2, 0.1, 1))
        )
        let lighting = try PathTracer.Lighting(analyticLights: [light], environment: .zero)
        let tracedPlain = try fixture.trace([(geometry, plain, facingPlane)], camera: camera, lighting: lighting)
        let tracedGraph = try fixture.trace([(geometry, generated, facingPlane)], camera: camera, lighting: lighting)
        for (x, y) in [(center, center), (center + 5, center - 3)] {
            let a = MaterialRenderFixture.pixel(rasterPlain, x: x, y: y)
            let b = MaterialRenderFixture.pixel(rasterGraph, x: x, y: y)
            #expect(a.x > 0.01 && simd_distance(a, b) <= 1e-3 * simd_length(a))
            let c = MaterialRenderFixture.pixel(tracedPlain, x: x, y: y)
            let d = MaterialRenderFixture.pixel(tracedGraph, x: x, y: y)
            #expect(c.x > 0.01 && simd_distance(c, d) <= 1e-3 * simd_length(c))
        }
    }

    @Test("Anisotropy stretches the highlight along its direction in both renderers")
    func anisotropy() throws {
        let geometry = try MeshGeometry.plane(width: 6, height: 6)
        let offset = 9
        func spread(_ layer: PhysicallyBasedMaterial.Anisotropy?) throws -> [(x: Float, y: Float, peak: Float)] {
            let material = try PhysicallyBasedMaterial(
                metallic: 1, roughness: 0.3, directionToLight: SIMD3(0, 0, 1), ambientIntensity: 0,
                alphaMode: .opaque, anisotropy: layer
            )
            let images = [
                try fixture.raster(material, geometry: geometry, transform: facingPlane, camera: camera),
                try fixture.trace([(geometry, material, facingPlane)], camera: camera, lighting: headLight(), samples: 64),
            ]
            return images.map { image in
                let right = MaterialRenderFixture.pixel(image, x: center + offset, y: center)
                let left = MaterialRenderFixture.pixel(image, x: center - offset, y: center)
                let below = MaterialRenderFixture.pixel(image, x: center, y: center + offset)
                let above = MaterialRenderFixture.pixel(image, x: center, y: center - offset)
                return ((right.x + left.x) / 2, (below.x + above.x) / 2, MaterialRenderFixture.pixel(image, x: center, y: center).x)
            }
        }
        let isotropic = try spread(nil)
        let alongTangent = try spread(.init(strength: 0.9))
        let alongBitangent = try spread(.init(strength: 0.9, rotation: .pi / 2))
        for renderer in 0..<2 {
            // Nine pixels tilt the half vector about 0.107 rad: GGX with alpha 0.09 falls
            // to about 17% of its peak, while alpha_t = mix(0.09, 1, 0.81) stays near 99%.
            let round = isotropic[renderer]
            #expect(abs(round.x - round.y) <= 0.05 * round.x)
            #expect(round.x < 0.3 * round.peak)
            // The tangent is the direction of increasing U, +X on the plane.
            let tangent = alongTangent[renderer]
            #expect(tangent.x > 0.7 * tangent.peak)
            #expect(tangent.x > 3 * tangent.y)
            let bitangent = alongBitangent[renderer]
            #expect(bitangent.y > 0.7 * bitangent.peak)
            #expect(bitangent.y > 3 * bitangent.x)
            #expect(tangent.peak < 0.2 * round.peak)
        }
    }

    @Test("Iridescence shifts the highlight hue with film thickness, matching across renderers")
    func iridescence() throws {
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        func highlight(_ thickness: Float?) throws -> (raster: SIMD3<Float>, traced: SIMD3<Float>) {
            let material = try PhysicallyBasedMaterial(
                roughness: 0.3, directionToLight: SIMD3(0, 0, 1), ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque,
                iridescence: try thickness.map { try .init(factor: 1, minimumThickness: 100, maximumThickness: $0) }
            )
            let raster = MaterialRenderFixture.pixel(
                try fixture.raster(material, geometry: geometry, transform: facingPlane, camera: camera), x: center, y: center
            )
            let traced = MaterialRenderFixture.pixel(
                try fixture.trace([(geometry, material, facingPlane)], camera: camera, lighting: headLight()),
                x: center, y: center
            )
            return (SIMD3(raster.x, raster.y, raster.z), SIMD3(traced.x, traced.y, traced.z))
        }
        func chroma(_ color: SIMD3<Float>) -> SIMD3<Float> { color / simd_reduce_add(color) }
        let plain = try highlight(nil)
        #expect(plain.raster.x == plain.raster.y && plain.raster.y == plain.raster.z)
        #expect(abs(plain.traced.x - plain.traced.z) <= 1e-3 * plain.traced.x)
        let thin = try highlight(170)
        let thick = try highlight(250)
        for film in [thin, thick] {
            // Both renderers evaluate the film at normal incidence for the center highlight.
            #expect(simd_distance(chroma(film.raster), chroma(film.traced)) < 0.02)
            #expect(simd_reduce_max(chroma(film.raster)) - simd_reduce_min(chroma(film.raster)) > 0.1)
        }
        // Between air and the 1.5 base the film (IOR 1.3) flips phase at both interfaces, so
        // the optical path 2 n d reinforces wavelengths 2 n d / m: 442 nm (blue) for 170 nm
        // and 650 nm (red) for 250 nm, which also cancels near 433 nm.
        for film in [thin.raster, thin.traced] { #expect(film.z > film.x) }
        for film in [thick.raster, thick.traced] { #expect(film.x > 3 * film.z) }
        #expect(simd_distance(chroma(thin.raster), chroma(thick.raster)) > 0.1)
        #expect(simd_distance(chroma(thin.traced), chroma(thick.traced)) > 0.1)
    }

    @Test("Unlit materials output their base color in both renderers regardless of light")
    func unlit() throws {
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        let material = try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0.2, directionToLight: SIMD3(0, 0, 1), lightIntensity: 20, ambientIntensity: 3,
            baseColorFactor: SIMD4(0.25, 0.5, 0.75, 1), emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque,
            clearcoat: .init(factor: 1), sheen: .init(colorFactor: SIMD3(repeating: 1)), shadingModel: .unlit
        )
        let expected = SIMD4<Float>(0.25, 0.5, 0.75, 1)
        let raster = try fixture.raster(material, geometry: geometry, transform: facingPlane, camera: camera)
        #expect(MaterialRenderFixture.pixel(raster, x: center, y: center) == expected)
        let traced = try fixture.trace(
            [(geometry, material, facingPlane)], camera: camera, lighting: headLight(20), samples: 4
        )
        #expect(MaterialRenderFixture.pixel(traced, x: center, y: center) == expected)
    }

    @Test("Path-traced dispersion separates red and blue refraction through a glass ball")
    func dispersion() throws {
        let ball = try MeshGeometry.uvSphere(radius: 1, segments: 64, rings: 32)
        let emitter = try MeshGeometry.plane(width: 10, height: 10)
        let glow = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(0, 0, 0, 1), emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque
        )
        func image(_ dispersion: Float) throws -> [SIMD4<Float>] {
            let glass = try PhysicallyBasedMaterial(
                roughness: 0, alphaMode: .opaque,
                transmission: .init(factor: 1, indexOfRefraction: 1.5, thickness: 1, dispersion: dispersion)
            )
            // The emitter covers x < -1.5 behind the ball, an edge off the lens axis.
            return try fixture.trace(
                [
                    (ball, glass, SceneTransform(translation: SIMD3(0, 0, -4))),
                    (emitter, glow, SceneTransform(translation: SIMD3(-6.5, 0, -9))),
                ],
                camera: camera, lighting: PathTracer.Lighting(environment: .zero), samples: 96, bounces: 6
            )
        }
        let plain = try image(0)
        #expect(plain.allSatisfy { $0.x == $0.y && $0.y == $0.z })
        #expect(plain.contains { $0.x > 0.5 })
        // An exaggerated Abbe number of 1 gives IORs 1.35 (red, C line) and 1.85 (blue, F line).
        let dispersed = try image(20)
        // Blue bends more, so rays from a band of the ball's right half reach the emitter
        // in blue but not in red. Each pixel's 96 samples cover each channel 32 times.
        let fringes = dispersed.filter { $0.z - $0.x > 0.5 }
        #expect(fringes.count >= 5)
        #expect(dispersed.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
    }
}
