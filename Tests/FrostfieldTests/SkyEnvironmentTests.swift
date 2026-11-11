import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SkyEnvironmentTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    private static func sun(elevation degrees: Double, azimuth: Double = 0) -> SIMD3<Float> {
        let e = degrees * .pi / 180
        let a = azimuth * .pi / 180
        return SIMD3<Float>(SIMD3(cos(e) * cos(a), sin(e), cos(e) * sin(a)))
    }

    @Test("The transmittance table integrates the density profiles exactly")
    func transmittanceTable() throws {
        let model = try SkyModel(sunDirection: Self.sun(elevation: 45), atmosphere: try .init(turbidity: 2), luminanceScale: 0.001)
        let sky = try SkyEnvironment(queue: queue, model: model, settings: .init(panoramaWidth: 64))
        let table = sky.transmittanceTable
        let air = model.atmosphere
        let top = air.topRadius - air.groundRadius
        // Texel (0, 0) is the vertical ray from the ground.
        let vertical = try Self.texel(table, 0, 0)
        let expected = SIMD3(
            air.rayleighScaleHeight * (1 - exp(-top / air.rayleighScaleHeight)),
            air.aerosolScaleHeight * (1 - exp(-top / air.aerosolScaleHeight)), 15_000
        )
        for axis in 0..<3 {
            #expect(abs(Double(vertical[axis]) / expected[axis] - 1) < 5e-4, "axis \(axis): \(vertical) vs \(expected)")
        }

        // Slant and horizon rays against the independent 3D integration, inverting
        // Bruneton's texel mapping in double precision.
        let reference = AtmosphereReference(model)
        let horizon = sqrt(air.topRadius * air.topRadius - air.groundRadius * air.groundRadius)
        for (x, y) in [(0, 0), (10, 0), (128, 0), (200, 5), (40, 20), (255, 63), (3, 40)] {
            let rho = horizon * Double(y) / Double(table.height - 1)
            let r = sqrt(rho * rho + air.groundRadius * air.groundRadius)
            let dMin = air.topRadius - r
            let d = dMin + Double(x) / Double(table.width - 1) * (rho + horizon - dMin)
            let mu = d <= 0 ? 1 : simd_clamp((horizon * horizon - rho * rho - d * d) / (2 * r * d), -1, 1)
            let point = SIMD3(0, r, 0)
            let lengths = try #require(reference.opticalLengths(
                from: point, toward: SIMD3(sqrt(max(1 - mu * mu, 0)), mu, 0), steps: 20_000
            ))
            let gpu = try Self.texel(table, x, y)
            for axis in 0..<3 where lengths[axis] > 1 {
                #expect(abs(Double(gpu[axis]) / lengths[axis] - 1) < 2e-3, "texel \(x),\(y) axis \(axis): \(gpu) vs \(lengths)")
            }
        }
    }

    @Test("Single-scattered sky radiance matches a double-precision reference within 1%")
    func singleScattering() throws {
        // The Sun at azimuth 35° toward +Z, so a panorama mirrored about either axis fails.
        let model = try SkyModel(
            sunDirection: Self.sun(elevation: 30, azimuth: 35), atmosphere: try .init(turbidity: 2.5), altitude: 300,
            luminanceScale: 0.001
        )
        let sky = try SkyEnvironment(
            queue: queue, model: model, settings: .init(panoramaWidth: 256), quality: .init(), multipleScattering: false
        )
        let reference = AtmosphereReference(model)
        let width = sky.panorama.width
        let height = sky.panorama.height
        // Near zenith, toward the Sun (texel 152, 42), its mirror images across the
        // X axis and the Z axis, away from it, across it and just above the horizon.
        for (x, y) in [(128, 0), (152, 40), (103, 40), (231, 40), (24, 50), (64, 30), (152, 63), (5, 62), (100, 20)] {
            let direction = Self.direction(x, y, width: width, height: height)
            let expected = reference.rgb(reference.march(from: reference.origin, direction: direction, steps: 2_000).radiance)
            let gpu = try Self.texel(sky.panorama, x, y)
            for channel in 0..<3 {
                let relative = abs(Double(gpu[channel]) / expected[channel] - 1)
                #expect(relative < 0.01, "texel \(x),\(y) channel \(channel): \(gpu) vs \(expected)")
            }
        }

        // The brightest sky texel lies next to the Sun's direction.
        let pixels = try Self.pixels(sky.panorama)
        var brightest = (index: 0, luminance: Float(0))
        for y in 0..<(height / 2) {
            for x in 0..<width {
                let texel = pixels[y * width + x]
                let luminance = 0.2126 * texel.x + 0.7152 * texel.y + 0.0722 * texel.z
                if luminance > brightest.luminance { brightest = (y * width + x, luminance) }
            }
        }
        let direction = Self.direction(brightest.index % width, brightest.index / width, width: width, height: height)
        let degrees = acos(simd_dot(direction, SIMD3<Double>(model.sunDirection))) * 180 / .pi
        #expect(degrees < 2, "brightest texel \(brightest.index % width),\(brightest.index / width) is \(degrees)° from the Sun")
    }

    @Test("Sky irradiance and the ground below the horizon match the double-precision reference")
    func irradianceAndGround() throws {
        // A strongly forward aerosol puts much of the sky light in a glow a few degrees
        // wide around the Sun, which the irradiance must resolve.
        var air = try SkyModel.Atmosphere(turbidity: 3, groundAlbedo: 0.3)
        air.aerosolAsymmetry = 0.95
        let model = try SkyModel(
            sunDirection: Self.sun(elevation: 30, azimuth: 35), atmosphere: air, altitude: 300, luminanceScale: 0.001
        )
        let sky = try SkyEnvironment(
            queue: queue, model: model, settings: .init(panoramaWidth: 256), quality: .init(), multipleScattering: false
        )
        let reference = AtmosphereReference(model)
        let irradiance = reference.skyIrradiance()
        let expectedIrradiance = reference.rgb(irradiance)
        let gpu = SIMD3<Double>(sky.skyIlluminance)
        for channel in 0..<3 {
            #expect(abs(gpu[channel] / expectedIrradiance[channel] - 1) < 0.01, "sky irradiance \(gpu) vs \(expectedIrradiance)")
        }

        // Below the horizon: in-scattering to the ground, plus the Lambertian ground lit
        // by the transmitted Sun and the sky irradiance, seen through the atmosphere.
        let width = sky.panorama.width
        let height = sky.panorama.height
        var groundShare = [Double]()
        for (x, y) in [(128, 127), (152, 64), (24, 66), (64, 80), (200, 70), (152, 100)] {
            let direction = Self.direction(x, y, width: width, height: height)
            let march = reference.march(from: reference.origin, direction: direction, steps: 2_000)
            let distance = try #require(reference.groundDistance(reference.origin, direction))
            let ground = reference.origin + distance * direction
            let cosine = max(simd_dot(simd_normalize(ground), reference.sun), 0)
            let sun = reference.spectralTransmittance(reference.opticalLengths(from: ground, toward: reference.sun, steps: 2_000))
            let reflected = (0..<16).map { band in
                march.transmittance[band] * air.groundAlbedo / .pi * (sun[band] * cosine + irradiance[band])
            }
            let skyLit = (0..<16).map { band in march.transmittance[band] * air.groundAlbedo / .pi * irradiance[band] }
            let expected = reference.rgb((0..<16).map { march.radiance[$0] + reflected[$0] })
            let texel = try Self.texel(sky.panorama, x, y)
            for channel in 0..<3 {
                #expect(abs(Double(texel[channel]) / expected[channel] - 1) < 0.01, "texel \(x),\(y): \(texel) vs \(expected)")
            }
            groundShare.append(reference.rgb(skyLit).y / expected.y)
        }
        // The sky-lit ground is a large share of the nadir, so the check is sensitive to it.
        #expect(groundShare[0] > 0.15, "\(groundShare)")
    }

    @Test("Multiple scattering and zenith radiance match the reference model")
    func multipleScattering() throws {
        // Put the Sun on a table column so texels and reference share its angle.
        let column = 28
        let muSun = -1 + 2 * Double(column) / 31
        var air = try SkyModel.Atmosphere(turbidity: 2)
        air.groundAlbedo = 0.3
        let model = try SkyModel(
            sunDirection: SIMD3<Float>(SIMD3(sqrt(1 - muSun * muSun), muSun, 0)), atmosphere: air, luminanceScale: 0.001
        )
        let sky = try SkyEnvironment(queue: queue, model: model, settings: .init(panoramaWidth: 256))
        let reference = AtmosphereReference(model)
        let rows = [0, 1, 4, 10]
        var altitudes = [Double]()
        var psi = [[Double]]()
        for row in rows {
            let altitude = max(1, (air.topRadius - air.groundRadius) * Double(row) / 31)
            let expected = reference.psi(altitude: altitude, nodes: 10, azimuths: 16, steps: 60)
            altitudes.append(altitude)
            psi.append(expected)
            for batch in 0..<4 {
                let gpu = try Self.texel(sky.multipleScatteringTable, column, row, slice: batch)
                for lane in 0..<4 {
                    let band = batch * 4 + lane
                    #expect(abs(Double(gpu[lane]) / expected[band] - 1) < 0.01, "row \(row) band \(band): \(gpu[lane]) vs \(expected[band])")
                }
            }
        }

        // The zenith adds σ_s Ψ along the ray to single scattering; interpolate Ψ in altitude.
        func interpolated(_ altitude: Double) -> [Double] {
            guard altitude < altitudes.last! else { return psi.last! }
            let upper = altitudes.firstIndex { $0 > altitude } ?? altitudes.count - 1
            let lower = max(upper - 1, 0)
            let f = simd_clamp((altitude - altitudes[lower]) / (altitudes[upper] - altitudes[lower]), 0, 1)
            return (0..<16).map { psi[lower][$0] * (1 - f) + psi[upper][$0] * f }
        }
        let width = sky.panorama.width
        let height = sky.panorama.height
        let zenith = Self.direction(width / 2, 0, width: width, height: height)
        let expected = reference.rgb(reference.march(from: reference.origin, direction: zenith, multiple: interpolated).radiance)
        let gpu = try Self.texel(sky.panorama, width / 2, 0)
        for channel in 0..<3 {
            #expect(abs(Double(gpu[channel]) / expected[channel] - 1) < 0.01, "zenith channel \(channel): \(gpu) vs \(expected)")
        }
        // Multiple scattering brightens the zenith and shifts it toward blue.
        let single = reference.rgb(reference.march(from: reference.origin, direction: zenith).radiance)
        #expect(expected.z / single.z > 1.1 && expected.z / single.z > expected.x / single.x)
        let chromaticity = Self.chromaticity(SIMD3<Double>(Double(gpu.x), Double(gpu.y), Double(gpu.z)))
        let expectedChromaticity = Self.chromaticity(expected)
        #expect(simd_distance(chromaticity, expectedChromaticity) < 0.002, "\(chromaticity) vs \(expectedChromaticity)")
        // A clear blue zenith: x and y well below the white point.
        #expect(chromaticity.x < 0.27 && chromaticity.y < 0.29)
    }

    @Test("Sky and Sun irradiance close: panorama integral, GPU hemisphere and both renderers agree")
    func energyAndRendererAgreement() throws {
        let model = try SkyModel(
            sunDirection: Self.sun(elevation: 50, azimuth: 30), atmosphere: try .init(turbidity: 2), luminanceScale: 0.001
        )
        let sky = try SkyEnvironment(queue: queue, model: model, settings: .init(panoramaWidth: 512))
        let integrated = Self.irradiance(
            try Self.pixels(sky.panorama), width: sky.panorama.width, height: sky.panorama.height, normal: SIMD3(0, 1, 0)
        )
        let gpu = SIMD3<Double>(sky.skyIlluminance)
        for channel in 0..<3 {
            #expect(abs(integrated[channel] / gpu[channel] - 1) < 0.005, "sky irradiance \(integrated) vs \(gpu)")
        }
        let sunCosine = Double(model.sunDirection.y)
        let direct = SIMD3<Double>(model.sunIlluminance) * sunCosine
        // A clear sky at 50°: diffuse light is 10–25% of the global horizontal illuminance.
        let luminance = SIMD3<Double>(0.2126, 0.7152, 0.0722)
        let diffuseFraction = simd_dot(gpu, luminance) / simd_dot(gpu + direct, luminance)
        #expect(diffuseFraction > 0.1 && diffuseFraction < 0.25, "\(diffuseFraction)")

        // A white Lambertian plane facing up reflects (E_sky + E_sun cos θ) / π, with
        // E_sky checked against the reference in `irradianceAndGround`.
        let expected = SIMD3<Float>((gpu + direct) / .pi)
        let plane = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 6, height: 6),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(repeating: 1),
                alphaMode: .opaque, environment: sky.materialEnvironment(), specular: try .init(factor: 0)
            ))
        )
        let down = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        let rig = try LightRig(lights: [sky.sunLight(castsShadow: false)])
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let cameraTransform = SceneTransform(translation: SIMD3(0, 2, 0), rotation: down)
        var graph = SceneGraph(
            roots: [SceneNode(transform: .init(rotation: down), mesh: plane)] + (try rig.sceneNodes()), camera: camera
        )
        graph.cameraTransform = cameraTransform
        let raster = try Self.rasterCenter(graph, device: device, queue: queue, size: 33)
        let tracer = try PathTracer(
            queue: queue, scene: PathTracingScene(queue: queue, graph: graph),
            lighting: try rig.pathLighting(sky: sky.pathEnvironment)
        )
        let traced = try Self.tracedCenter(tracer, camera: camera, transform: cameraTransform, device: device, queue: queue, size: 33)
        for channel in 0..<3 {
            #expect(abs(raster[channel] / expected[channel] - 1) < 0.01, "raster \(raster) vs \(expected)")
            #expect(abs(traced[channel] / expected[channel] - 1) < 0.01, "path \(traced) vs \(expected)")
            #expect(abs(traced[channel] / raster[channel] - 1) < 0.01, "path \(traced) vs raster \(raster)")
        }
    }

    @Test("Strongly forward aerosols keep the sky irradiance closed against the panorama")
    func forwardAerosolClosure() throws {
        // A glow a fraction of a degree wide around the Sun carries much of the sky
        // light at g = 0.99. The 2,048-texel panorama resolves it; its integral is
        // stable to 0.05% from 512 to 8,192 texels.
        for asymmetry in [0.9, 0.95, 0.99] {
            let air = SkyModel.Atmosphere(aerosolOpticalDepth: 0.3, aerosolAsymmetry: asymmetry)
            let model = try SkyModel(sunDirection: Self.sun(elevation: 30, azimuth: 20), atmosphere: air, luminanceScale: 0.001)
            let sky = try SkyEnvironment(queue: queue, model: model, settings: .init(panoramaWidth: 2_048))
            let integrated = Self.irradiance(
                try Self.pixels(sky.panorama), width: sky.panorama.width, height: sky.panorama.height, normal: SIMD3(0, 1, 0)
            )
            let gpu = SIMD3<Double>(sky.skyIlluminance)
            for channel in 0..<3 {
                #expect(abs(gpu[channel] / integrated[channel] - 1) < 0.01, "g \(asymmetry): \(gpu) vs panorama \(integrated)")
            }
        }
    }

    @Test("A vertical wall lit by sky, ground and Sun agrees between both renderers and the panorama")
    func verticalWall() throws {
        let model = try SkyModel(
            sunDirection: Self.sun(elevation: 35, azimuth: 35), atmosphere: try .init(turbidity: 2.5, groundAlbedo: 0.3),
            luminanceScale: 0.001
        )
        let sky = try SkyEnvironment(queue: queue, model: model, settings: .init(panoramaWidth: 512))
        // The wall faces 60° around from the Sun's azimuth toward +Z: lit obliquely by
        // the Sun, it sees half sky and half ground.
        let azimuth = 95.0 * .pi / 180
        let normal = SIMD3<Double>(cos(azimuth), 0, sin(azimuth))
        let pixels = try Self.pixels(sky.panorama)
        let width = sky.panorama.width
        let height = sky.panorama.height
        let all = Self.irradiance(pixels, width: width, height: height, normal: normal)
        let groundOnly = Self.irradiance(pixels, width: width, height: height, normal: normal, rows: (height / 2)..<height)
        let sunCosine = max(simd_dot(SIMD3<Double>(model.sunDirection), normal), 0)
        let expected = SIMD3<Float>((all + SIMD3<Double>(model.sunIlluminance) * sunCosine) / .pi)
        // The ground lights a large share of the wall, so both renderers' use of the
        // panorama's lower half is under test.
        #expect(groundOnly.y / (all.y + Double(model.sunIlluminance.y) * sunCosine) > 0.1, "\(groundOnly) of \(all)")
        #expect(sunCosine > 0.3)

        let wall = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 6, height: 6),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(repeating: 1),
                alphaMode: .opaque, environment: sky.materialEnvironment(), specular: try .init(factor: 0)
            ))
        )
        let facing = SIMD3<Float>(normal)
        let rig = try LightRig(lights: [sky.sunLight(castsShadow: false)])
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let cameraTransform = SceneTransform(translation: 2 * facing, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -facing))
        var graph = SceneGraph(
            roots: [SceneNode(transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, 1), to: facing)), mesh: wall)]
                + (try rig.sceneNodes()),
            camera: camera
        )
        graph.cameraTransform = cameraTransform
        let raster = try Self.rasterCenter(graph, device: device, queue: queue, size: 33)
        let tracer = try PathTracer(
            queue: queue, scene: PathTracingScene(queue: queue, graph: graph),
            lighting: try rig.pathLighting(sky: sky.pathEnvironment)
        )
        let traced = try Self.tracedCenter(tracer, camera: camera, transform: cameraTransform, device: device, queue: queue, size: 33)
        for channel in 0..<3 {
            #expect(abs(raster[channel] / expected[channel] - 1) < 0.02, "raster \(raster) vs \(expected)")
            #expect(abs(traced[channel] / expected[channel] - 1) < 0.01, "path \(traced) vs \(expected)")
        }
    }

    @Test("Hazy skies beyond the half-float range and invalid settings throw")
    func rangeAndValidation() throws {
        // In physical units a turbidity-6 sky passes 65,504 cd/m² around the Sun.
        let hazy = try SkyModel(sunDirection: Self.sun(elevation: 60), atmosphere: try .init(turbidity: 6), luminanceScale: 1)
        #expect(throws: SkyEnvironment.Failure.self) {
            try SkyEnvironment(queue: queue, model: hazy, settings: .init(panoramaWidth: 128))
        }
        do {
            _ = try SkyEnvironment(queue: queue, model: hazy, settings: .init(panoramaWidth: 128))
        } catch let failure as SkyEnvironment.Failure {
            guard case .radianceRange(let peak) = failure else { throw failure }
            #expect(peak > 65_504 && failure.errorDescription?.contains("luminanceScale") == true)
        }
        // A tenth of the scale fits, and the result scales linearly.
        let scaled = try SkyModel(sunDirection: hazy.sunDirection, atmosphere: hazy.atmosphere, luminanceScale: 0.1)
        let sky = try SkyEnvironment(queue: queue, model: scaled, settings: .init(panoramaWidth: 128))
        #expect(sky.peakRadiance < 65_504 && sky.peakRadiance > 6_550)
        #expect(sky.sunLight().emission == .intensity(simd_reduce_max(scaled.sunIlluminance)))
        if case .map(let map, let intensity, let yaw) = sky.pathEnvironment {
            #expect(map === sky.pathMap && intensity == 1 && yaw == 0 && map.isSampled)
        } else {
            Issue.record("Expected an environment map")
        }
        for width in [63, 32, 16_384] {
            #expect(throws: SkyEnvironment.Failure.settings) {
                try SkyEnvironment(queue: queue, model: scaled, settings: .init(panoramaWidth: width))
            }
        }
    }

    @Test("Swift uniform layouts match the Metal structs")
    func layouts() {
        // FrostfieldSkyConstants: 3 float4 plus 8 arrays of 4; FrostfieldAerialUniforms: 7 float4 plus 5 arrays of 4.
        #expect(MemoryLayout<SkyEnvironment.Constants>.size == 35 * 16)
        #expect(MemoryLayout<SkyEnvironment.Constants>.stride == 35 * 16)
        #expect(MemoryLayout<AerialPerspective.Uniforms>.size == 27 * 16)
        #expect(MemoryLayout<AerialPerspective.Uniforms>.stride == 27 * 16)
    }

    // MARK: - Helpers

    static func direction(_ x: Int, _ y: Int, width: Int, height: Int) -> SIMD3<Double> {
        let theta = (Double(y) + 0.5) / Double(height) * .pi
        let phi = ((Double(x) + 0.5) / Double(width) - 0.5) * 2 * .pi
        return SIMD3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi))
    }

    /// ∫ L max(ω·n, 0) dΩ over panorama rows, weighting each texel by its solid angle.
    static func irradiance(
        _ pixels: [SIMD4<Float>], width: Int, height: Int, normal: SIMD3<Double>, rows: Range<Int>? = nil
    ) -> SIMD3<Double> {
        var sum = SIMD3<Double>.zero
        for y in rows ?? 0..<height {
            let theta = (Double(y) + 0.5) / Double(height) * .pi
            let solidAngle = (2 * .pi / Double(width)) * (.pi / Double(height)) * sin(theta)
            for x in 0..<width {
                let cosine = simd_dot(direction(x, y, width: width, height: height), normal)
                guard cosine > 0 else { continue }
                let texel = pixels[y * width + x]
                sum += SIMD3<Double>(Double(texel.x), Double(texel.y), Double(texel.z)) * (cosine * solidAngle)
            }
        }
        return sum
    }

    static func chromaticity(_ rgb: SIMD3<Double>) -> SIMD2<Double> {
        let xyz = SkyModel.Spectrum.xyzToLinearSRGB.inverse * rgb
        return SIMD2(xyz.x, xyz.y) / (xyz.x + xyz.y + xyz.z)
    }

    static func texel(_ texture: any MTLTexture, _ x: Int, _ y: Int, slice: Int = 0) throws -> SIMD4<Float> {
        var value = SIMD4<Float>.zero
        try withUnsafeMutableBytes(of: &value) { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: 16, bytesPerImage: 16,
                from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0, slice: slice
            )
        }
        return value
    }

    static func pixels(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var values = [SIMD4<Float>](repeating: .zero, count: texture.width * texture.height)
        try values.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 16,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return values
    }

    static func half(_ texture: any MTLTexture, _ x: Int, _ y: Int) throws -> SIMD4<Float> {
        var value = SIMD4<Float16>.zero
        try withUnsafeMutableBytes(of: &value) { bytes in
            texture.getBytes(try #require(bytes.baseAddress), bytesPerRow: 8, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        }
        return SIMD4<Float>(value)
    }

    private static func rasterCenter(
        _ graph: SceneGraph, device: any MTLDevice, queue: any MTLCommandQueue, size: Int
    ) throws -> SIMD4<Float> {
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: size, height: size, mipmapped: false)
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = device.makeTexture(descriptor: depthDescriptor)
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        var sum = SIMD4<Float>.zero
        for dy in -1...1 { for dx in -1...1 { sum += try half(color, size / 2 + dx, size / 2 + dy) } }
        return sum / 9
    }

    private static func tracedCenter(
        _ tracer: PathTracer, camera: PerspectiveCamera, transform: SceneTransform, device: any MTLDevice,
        queue: any MTLCommandQueue, size: Int
    ) throws -> SIMD4<Float> {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let output = try #require(device.makeTexture(descriptor: descriptor))
        let accumulation = try tracer.makeAccumulation(width: size, height: size)
        var history: PathTracer.History?
        for _ in 0..<64 {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: camera, cameraTransform: transform, destination: output, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: 16, settings: .init(maxBounces: 1)
            )
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
        }
        var sum = SIMD4<Float>.zero
        for dy in -1...1 { for dx in -1...1 { sum += try half(output, size / 2 + dx, size / 2 + dy) } }
        return sum / 9
    }
}
