import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct AerialPerspectiveTests {
    private let fixture: PostProcessFixture

    init() throws {
        fixture = try PostProcessFixture()
    }

    private static func forwardDepth(axial: Double, camera: PerspectiveCamera) -> Float {
        let near = Double(camera.near)
        let far = Double(camera.far)
        return Float(far * (axial - near) / ((far - near) * axial))
    }

    private func render(
        _ effect: AerialPerspective, source: any MTLTexture, depth: any MTLTexture, camera: PerspectiveCamera,
        transform: SceneTransform, depthMode: MeshDepthMode = .forward, settings: AerialPerspective.Settings = .init()
    ) throws -> [SIMD4<Float>] {
        let destination = try fixture.color(width: source.width, height: source.height)
        try fixture.run { buffer in
            try effect.encode(
                source: source, depth: depth, destination: destination, camera: camera, cameraTransform: transform,
                depthMode: depthMode, settings: settings, commandBuffer: buffer
            )
        }
        var values = [SIMD4<Float16>](repeating: .zero, count: source.width * source.height)
        try values.withUnsafeMutableBytes { bytes in
            destination.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: source.width * 8,
                from: MTLRegionMake2D(0, 0, source.width, source.height), mipmapLevel: 0
            )
        }
        return values.map(SIMD4<Float>.init)
    }

    @Test("Surfaces receive the reference extinction and in-scattering along their view rays")
    func inscatteringAndExtinction() throws {
        let model = try SkyModel(
            sunDirection: SIMD3(0.6, 0.35, -0.5), atmosphere: try .init(turbidity: 3), altitude: 200, luminanceScale: 0.001
        )
        let sky = try SkyEnvironment(
            queue: fixture.queue, model: model, settings: .init(panoramaWidth: 64), quality: .init(), multipleScattering: false
        )
        let effect = try AerialPerspective(environment: sky)
        let size = 33
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.5, near: 1, far: 50_000)
        let transform = SceneTransform(translation: SIMD3(0, 1.5, 0))
        // A sunlit surface's radiance in kilonits.
        let albedo = SIMD4<Float>(16, 10, 4, 1)
        let source = try fixture.color(width: size, height: size) { _, _ in albedo }
        let axial = 4_000.0
        let depthValue = Self.forwardDepth(axial: axial, camera: camera)
        let depth = try fixture.depth(width: size, height: size) { _, _ in depthValue }
        let output = try render(effect, source: source, depth: depth, camera: camera, transform: transform)

        let reference = AtmosphereReference(model)
        let solar = SkyModel.Spectrum.solarIrradiance
        let weights = (0..<16).map { simd_max(SkyModel.Spectrum.rgbWeights[$0], .zero) * solar[$0] }
        let total = weights.reduce(SIMD3<Double>.zero, +)
        let tanY = tan(Double(camera.verticalFieldOfView) / 2)
        // The center pixel looks due north; the others look off-axis and farther,
        // one slightly downward but still meeting the wall before the ground.
        for (x, y) in [(16, 16), (2, 18), (30, 5)] {
            let ndc = SIMD2((Double(x) + 0.5) / Double(size) * 2 - 1, 1 - (Double(y) + 0.5) / Double(size) * 2)
            let ray = SIMD3(ndc.x * tanY, ndc.y * tanY, -1)
            let direction = simd_normalize(ray)
            let start = reference.origin + SIMD3(0, 1.5, 0)
            let march = reference.march(from: start, direction: direction, limit: axial * simd_length(ray), steps: 2_000)
            let inscattered = reference.rgb(march.radiance)
            var transmittance = SIMD3<Double>.zero
            for band in 0..<16 { transmittance += weights[band] * march.transmittance[band] }
            transmittance /= total
            let expected = SIMD3<Double>(Double(albedo.x), Double(albedo.y), Double(albedo.z)) * transmittance + inscattered
            let pixel = output[y * size + x]
            for channel in 0..<3 {
                #expect(abs(Double(pixel[channel]) / expected[channel] - 1) < 0.01, "pixel \(x),\(y): \(pixel) vs \(expected)")
            }
            #expect(pixel.w == 1)
            // Haze contributes a large share, so the comparison is sensitive to it.
            #expect(inscattered.z / expected.z > 0.2 && transmittance.z < 0.8 && transmittance.x > transmittance.z,
                    "\(inscattered) \(transmittance)")
        }

        // Reversed depth and kilometer world units describe the same wall.
        let reversed = try fixture.depth(width: size, height: size) { _, _ in 1 - depthValue }
        let flipped = try render(effect, source: source, depth: reversed, camera: camera, transform: transform, depthMode: .reversed)
        let kmCamera = try PerspectiveCamera(verticalFieldOfView: 0.5, near: 0.001, far: 50)
        let kmDepth = try fixture.depth(width: size, height: size) { _, _ in Self.forwardDepth(axial: 4, camera: kmCamera) }
        let kilometers = try render(
            effect, source: source, depth: kmDepth, camera: kmCamera, transform: SceneTransform(translation: SIMD3(0, 0.0015, 0)),
            settings: .init(metersPerUnit: 1_000)
        )
        for index in [16 * size + 16, 18 * size + 2] {
            #expect(simd_distance(flipped[index], output[index]) < 2e-3 * simd_length(output[index]))
            #expect(simd_distance(kilometers[index], output[index]) < 2e-3 * simd_length(output[index]))
        }
    }

    @Test("The background shows the sky and a limb-darkened solar disk carrying the Sun's illuminance")
    func solarDisk() throws {
        // A small scale keeps the disk below the half-float limit so its shape survives.
        let model = try SkyModel(sunDirection: SIMD3(0.3, 0.6, -0.7), altitude: 50, luminanceScale: 1e-5)
        let sky = try SkyEnvironment(queue: fixture.queue, model: model, settings: .init(panoramaWidth: 256))
        let effect = try AerialPerspective(environment: sky)
        let size = 151
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.03, near: 0.1, far: 100)
        let transform = SceneTransform(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: model.sunDirection))
        let marker = SIMD4<Float>(0.25, 0.5, 0.75, 0.5)
        let source = try fixture.color(width: size, height: size) { _, _ in marker }
        let empty = try fixture.depth(width: size, height: size) { _, _ in 1 }
        let withSun = try render(effect, source: source, depth: empty, camera: camera, transform: transform)
        let skyOnly = try render(
            effect, source: source, depth: empty, camera: camera, transform: transform, settings: .init(drawsSun: false)
        )
        let untouched = try render(
            effect, source: source, depth: empty, camera: camera, transform: transform, settings: .init(drawsSky: false)
        )
        #expect(untouched.allSatisfy { $0 == marker })
        #expect(skyOnly.allSatisfy { $0.w == 1 && simd_reduce_min($0) > 0 })

        // Integrate the disk over each pixel's solid angle.
        let tanY = tan(Double(camera.verticalFieldOfView) / 2)
        let pixelSize = 2 * tanY / Double(size)
        var irradiance = SIMD3<Double>.zero
        var diskPixels = 0
        for y in 0..<size {
            for x in 0..<size {
                let disk = withSun[y * size + x] - skyOnly[y * size + x]
                guard simd_reduce_max(disk) > 0 else { continue }
                diskPixels += 1
                let offset = SIMD2((Double(x) + 0.5) * pixelSize - tanY, (Double(y) + 0.5) * pixelSize - tanY)
                let cosine = 1 / sqrt(1 + simd_length_squared(offset))
                irradiance += SIMD3<Double>(Double(disk.x), Double(disk.y), Double(disk.z)) * pixelSize * pixelSize * pow(cosine, 3)
            }
        }
        let expected = SIMD3<Double>(model.sunIlluminance(altitude: 50))
        let radiusPixels = model.sunAngularRadius / pixelSize
        #expect(abs(Double(diskPixels) / (.pi * radiusPixels * radiusPixels) - 1) < 0.03)
        for channel in 0..<3 {
            #expect(abs(irradiance[channel] / expected[channel] - 1) < 0.01, "\(irradiance) vs \(expected)")
        }

        // Limb darkening: the center is brighter than the disk average by Σ(α + 2)/2 weights.
        let center = withSun[(size / 2) * size + size / 2] - skyOnly[(size / 2) * size + size / 2]
        let solidAngle = 2 * Double.pi * (1 - cos(model.sunAngularRadius))
        let average = expected / solidAngle
        let transmittance = model.sunTransmittance(altitude: 50)
        let centerSpectrum = (0..<16).map { band in
            SkyModel.Spectrum.solarIrradiance[band] * transmittance[band] * (SkyModel.Spectrum.limbDarkening[band] + 2) / 2 / solidAngle
        }
        let expectedCenter = SkyModel.Spectrum.linearSRGB(centerSpectrum) * model.luminanceScale
        for channel in 0..<3 {
            #expect(abs(Double(center[channel]) / expectedCenter[channel] - 1) < 0.01, "\(center) vs \(expectedCenter)")
            #expect(expectedCenter[channel] / average[channel] > 1.2)
        }
        let row = (size / 2) * size
        let limb = Int((radiusPixels * 0.9).rounded(.down))
        let edge = withSun[row + size / 2 + limb] - skyOnly[row + size / 2 + limb]
        #expect(edge.y < 0.8 * center.y && edge.z / center.z < edge.x / center.x)
    }

    @Test("Empty pixels show the sky seen from the camera's altitude, on a non-square target")
    func backgroundFromCamera() throws {
        let model = try SkyModel(
            sunDirection: SIMD3(0.55, 0.3, -0.6), atmosphere: try .init(turbidity: 2.5, groundAlbedo: 0.3), altitude: 100,
            luminanceScale: 0.001
        )
        let sky = try SkyEnvironment(
            queue: fixture.queue, model: model, settings: .init(panoramaWidth: 1_024), quality: .init(), multipleScattering: false
        )
        let effect = try AerialPerspective(environment: sky)
        let reference = AtmosphereReference(model)
        let spectral = sky.spectralIrradiance.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let irradiance = (0..<16).map { Double(spectral[$0 / 4][$0 % 4]) }
        let panorama = try SkyEnvironmentTests.pixels(sky.panorama)

        // A 16:9 view facing 20° left of the Sun's azimuth, level with the horizon.
        let size = (width: 48, height: 27)
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.6, near: 1, far: 1_000)
        let sunAzimuth = atan2(Double(model.sunDirection.x), -Double(model.sunDirection.z))
        let rotation = simd_quatf(angle: Float(-(sunAzimuth - 20 * .pi / 180)), axis: SIMD3(0, 1, 0))
        let source = try fixture.color(width: size.width, height: size.height) { _, _ in SIMD4(0, 0, 0, 0) }
        let empty = try fixture.depth(width: size.width, height: size.height) { _, _ in 1 }
        let tanY = tan(Double(camera.verticalFieldOfView) / 2)
        let aspect = Double(size.width) / Double(size.height)
        var samples = [(x: Int, y: Int)]()
        for x in [0, 9, 18, 27, 36, 47] {
            for y in [0, 7, 12, 13, 14, 15, 16, 20, 26] { samples.append((x, y)) }
        }

        // At the model's altitude the background reads the panorama; 300 m and 10 km
        // above it, where the horizon dips 0.6° and 3.2°, it reads the camera's sky table.
        for height in [Float(0), 300, 10_000] {
            let transform = SceneTransform(translation: SIMD3(0, height, 0), rotation: rotation)
            let output = try render(effect, source: source, depth: empty, camera: camera, transform: transform)
            let start = reference.origin + SIMD3(0, Double(height), 0)
            let directions = samples.map { sample in
                let ndc = SIMD2(
                    (Double(sample.x) + 0.5) / Double(size.width) * 2 - 1, 1 - (Double(sample.y) + 0.5) / Double(size.height) * 2
                )
                let local = SIMD3<Float>(Float(ndc.x * tanY * aspect), Float(ndc.y * tanY), -1)
                return simd_normalize(SIMD3<Double>(rotation.act(local)))
            }
            let expected = AtmosphereReference.concurrentMap(samples.count) { index in
                let rgb = Self.referenceSky(reference, from: start, direction: directions[index], skyIrradiance: irradiance)
                return [rgb.x, rgb.y, rgb.z]
            }
            var ground = 0
            for (index, sample) in samples.enumerated() {
                let pixel = output[sample.y * size.width + sample.x]
                if reference.groundDistance(start, directions[index]) != nil { ground += 1 }
                for channel in 0..<3 {
                    #expect(abs(Double(pixel[channel]) / expected[index][channel] - 1) < 0.02,
                            "altitude \(height), pixel \(sample.x),\(sample.y): \(pixel) vs \(expected[index])")
                }
                #expect(pixel.w == 1)
                // At the model's altitude the background also matches the lighting
                // panorama away from the horizon, where both tables are smooth.
                let elevation = asin(directions[index].y) * 180 / .pi
                if height == 0, abs(elevation) > 2 {
                    let texel = Self.bilinear(panorama, width: sky.panorama.width, direction: directions[index])
                    for channel in 0..<3 {
                        #expect(abs(Double(pixel[channel]) / texel[channel] - 1) < 0.02,
                                "panorama at pixel \(sample.x),\(sample.y): \(pixel) vs \(texel)")
                    }
                }
            }
            #expect(ground >= 12, "\(ground) ground samples at \(height) m")
        }
    }

    /// Single-scattered radiance along a ray from `start`, with the Lambertian ground
    /// lit by the Sun and the given sky irradiance where the ray meets it.
    private static func referenceSky(
        _ reference: AtmosphereReference, from start: SIMD3<Double>, direction: SIMD3<Double>, skyIrradiance: [Double]
    ) -> SIMD3<Double> {
        let march = reference.march(from: start, direction: direction, steps: 800, sunSteps: 96)
        var spectrum = march.radiance
        if let distance = reference.groundDistance(start, direction) {
            let point = start + distance * direction
            let cosine = max(simd_dot(simd_normalize(point), reference.sun), 0)
            let sun = reference.spectralTransmittance(reference.opticalLengths(from: point, toward: reference.sun, steps: 256))
            for band in 0..<16 {
                spectrum[band] += march.transmittance[band] * reference.air.groundAlbedo / .pi
                    * (sun[band] * cosine + skyIrradiance[band])
            }
        }
        return reference.rgb(spectrum)
    }

    /// Bilinear panorama lookup with the lighting convention: +Y up, the center
    /// column facing +X and U turning toward +Z.
    private static func bilinear(_ pixels: [SIMD4<Float>], width: Int, direction: SIMD3<Double>) -> SIMD3<Double> {
        let height = width / 2
        let u = atan2(direction.z, direction.x) / (2 * .pi) + 0.5
        let v = acos(simd_clamp(direction.y, -1, 1)) / .pi
        let x = u * Double(width) - 0.5
        let y = simd_clamp(v * Double(height) - 0.5, 0, Double(height - 1))
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let fx = x - floor(x)
        let fy = y - floor(y)
        func texel(_ x: Int, _ y: Int) -> SIMD3<Double> {
            let value = pixels[min(y, height - 1) * width + ((x % width) + width) % width]
            return SIMD3(Double(value.x), Double(value.y), Double(value.z))
        }
        let top = texel(x0, y0) * (1 - fx) + texel(x0 + 1, y0) * fx
        let bottom = texel(x0, y0 + 1) * (1 - fx) + texel(x0 + 1, y0 + 1) * fx
        return top * (1 - fy) + bottom * fy
    }

    @Test("Aerial perspective rejects mismatched inputs and settings")
    func validation() throws {
        let model = try SkyModel(sunDirection: SIMD3(0, 1, 0), luminanceScale: 0.001)
        let sky = try SkyEnvironment(queue: fixture.queue, model: model, settings: .init(panoramaWidth: 64))
        let effect = try AerialPerspective(environment: sky)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100)
        let source = try fixture.color(width: 8, height: 8)
        let destination = try fixture.color(width: 8, height: 8)
        let depth = try fixture.depth(width: 8, height: 8) { _, _ in 1 }
        let small = try fixture.depth(width: 4, height: 4) { _, _ in 1 }
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: AerialPerspective.Failure.incompatibleTextures) {
            try effect.encode(source: source, depth: small, destination: destination, camera: camera, commandBuffer: buffer)
        }
        #expect(throws: AerialPerspective.Failure.incompatibleTextures) {
            try effect.encode(source: source, depth: depth, destination: source, camera: camera, commandBuffer: buffer)
        }
        #expect(throws: AerialPerspective.Failure.settings) {
            try effect.encode(
                source: source, depth: depth, destination: destination, camera: camera,
                settings: .init(metersPerUnit: 0), commandBuffer: buffer
            )
        }
        #expect(throws: AerialPerspective.Failure.cameraTransform) {
            try effect.encode(
                source: source, depth: depth, destination: destination, camera: camera,
                cameraTransform: SceneTransform(scale: SIMD3(repeating: 2)), commandBuffer: buffer
            )
        }
        #expect(throws: AerialPerspective.Failure.cameraTransform) {
            try effect.encode(
                source: source, depth: depth, destination: destination, camera: camera,
                cameraTransform: SceneTransform(translation: SIMD3(0, 80_000, 0)), commandBuffer: buffer
            )
        }
    }
}
