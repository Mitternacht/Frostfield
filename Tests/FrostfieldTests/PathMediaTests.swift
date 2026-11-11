import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Participating media in the path tracer: volume scattering, random-walk subsurface
/// scattering and fog, checked against closed-form, Monte Carlo and energy references.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathMediaTests {
    private let fixture: MediaFixture

    init() throws {
        fixture = try MediaFixture()
    }

    /// An index-matched (IOR 1) transmissive volume whose medium has extinction `extinction`
    /// per unit and the given multiple-scattering color and anisotropy.
    private func scatteringVolume(
        extinction: SIMD3<Float>, color: SIMD3<Float>, anisotropy: Float = 0, indexOfRefraction: Float = 1
    ) throws -> PhysicallyBasedMaterial {
        // KHR_materials_volume: sigma_t = -ln(c) / d, with a distance that keeps every color representable.
        let distance = 0.5 / simd_reduce_max(extinction)
        let attenuation = SIMD3<Float>((0..<3).map { exp(-extinction[$0] * distance) })
        return try PhysicallyBasedMaterial(
            roughness: 0, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque,
            transmission: .init(
                factor: 1, indexOfRefraction: indexOfRefraction, thickness: 1,
                attenuationDistance: distance, attenuationColor: attenuation
            ),
            volumeScatter: .init(multiscatterColor: color, anisotropy: anisotropy)
        )
    }

    @Test("A non-absorbing scattering volume or subsurface sphere in a white furnace stays white")
    func whiteFurnace() throws {
        let width = 33
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 30)
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 96, rings: 48)
        // Optical radius 4: walks of tens of events, none near the 256-event cap.
        // Gray media carry a weight of exactly one on every path, so every pixel is one. Chromatic
        // media reweight the channels by the free-path MIS, which is unbiased but noisy.
        let materials: [(String, PhysicallyBasedMaterial, gray: Bool)] = [
            ("isotropic", try scatteringVolume(extinction: SIMD3(repeating: 4), color: SIMD3(repeating: 1)), true),
            ("forward", try scatteringVolume(extinction: SIMD3(repeating: 4), color: SIMD3(repeating: 1), anisotropy: 0.7), true),
            ("chromatic", try scatteringVolume(extinction: SIMD3(2, 4, 8), color: SIMD3(repeating: 1), anisotropy: -0.3), false),
            ("subsurface", try PhysicallyBasedMaterial(
                roughness: 0.4, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque,
                specular: .init(factor: 0),
                subsurface: .init(weight: 1, color: SIMD3(repeating: 1), radius: SIMD3(1, 0.5, 0.25), scale: 0.25)
            ), false),
        ]
        for (name, material, gray) in materials {
            let graph = SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -3.5)), mesh: try fixture.renderer(sphere, material)),
            ], camera: camera)
            // One bounce: walk events and the volume's own boundary do not consume bounces.
            let image = try fixture.trace(
                graph, camera: camera, width: width, height: width, samples: 64,
                settings: .init(maxBounces: 1, seed: 3), lighting: .init(environment: SIMD3(repeating: 1))
            )
            MediaFixture.save(image, width: width, height: width, name: "furnace-\(name)", exposure: 0.5)
            let pixels = image.map { SIMD3($0.x, $0.y, $0.z) }
            let mean = pixels.reduce(.zero, +) / Float(pixels.count)
            let worst = pixels.map { simd_reduce_max(abs($0 - 1)) }.max() ?? 0
            let rms = (pixels.map { simd_length_squared($0 - 1) / 3 }.reduce(0, +) / Float(pixels.count)).squareRoot()
            #expect(simd_reduce_max(abs(mean - 1)) < 0.006, "\(name): mean \(mean)")
            if gray {
                #expect(worst < 2e-3, "\(name): worst pixel deviation \(worst)")
            } else {
                #expect(rms < 0.1, "\(name): RMS deviation \(rms)")
            }
        }
    }

    @Test("A deep scattering medium reflects Chandrasekhar's plane albedo per channel")
    func semiInfiniteAlbedo() throws {
        let width = 9
        // A narrow view of the sphere's center sees normal incidence within 0.6 degrees.
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.02, near: 0.1, far: 30)
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 96, rings: 48)
        // Mean free paths of 0.6 to 2.5 thousandths of the radius; a different extinction and
        // albedo per channel exercises the chromatic free-path sampling.
        let colors = SIMD3<Float>(0.25, 0.5, 0.7)
        let material = try scatteringVolume(extinction: SIMD3(1_600, 800, 400), color: colors)
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: try fixture.renderer(sphere, material)),
        ], camera: camera)
        let image = try fixture.trace(
            graph, camera: camera, width: width, height: width, samples: 1_024,
            settings: .init(maxBounces: 1, seed: 11), lighting: .init(environment: SIMD3(repeating: 1))
        )
        let measured = MediaFixture.center(image, width: width, height: width, radius: 4)
        for channel in 0..<3 {
            let albedo = MediaReference.singleScatteringAlbedo(Double(colors[channel]))
            let expected = MediaReference.planeAlbedo(albedo: albedo, cosine: 1)
            #expect(abs(Double(measured[channel]) - expected) < 0.01 * expected + 0.002,
                    "channel \(channel): \(measured[channel]) vs \(expected) (single-scattering albedo \(albedo))")
        }
    }

    @Test("Slab reflectance and transmittance through an absorbing, anisotropically scattering medium match a Monte Carlo reference")
    func slabReference() throws {
        let width = 9
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.05, near: 0.1, far: 30)
        let thickness: Float = 0.5
        let extinction = SIMD3<Float>(4, 2, 1)
        let colors = SIMD3<Float>(0.9, 0.6, 0.3)
        let anisotropy: Float = 0.6
        let material = try scatteringVolume(extinction: extinction, color: colors, anisotropy: anisotropy)
        // A 40 x 40 slab, 80 mean free paths wide, facing the camera along Z.
        let graph = SceneGraph(roots: [
            SceneNode(
                transform: .init(translation: SIMD3(0, 0, -3), scale: SIMD3(40, 40, thickness)),
                mesh: try fixture.renderer(.cube(), material)
            ),
        ], camera: camera)
        // Panoramas lit only behind the slab (-Z) or only in front of it (+Z). By reciprocity
        // the camera then sees the slab's total transmittance or reflectance at normal incidence.
        func panorama(behind: Bool) throws -> PathTracer.EnvironmentMap {
            let mapWidth = 512
            let mapHeight = 256
            var pixels: [SIMD3<Float>] = []
            for _ in 0..<mapHeight {
                for column in 0..<mapWidth {
                    // U below one half faces -Z (atan2(z, x) < 0).
                    let towardNegativeZ = (Float(column) + 0.5) / Float(mapWidth) < 0.5
                    pixels.append(towardNegativeZ == behind ? SIMD3(repeating: 1) : .zero)
                }
            }
            return try PathTracer.EnvironmentMap(device: fixture.device, pixels: pixels, width: mapWidth, height: mapHeight)
        }
        func render(behind: Bool) throws -> SIMD3<Float> {
            let image = try fixture.trace(
                graph, camera: camera, width: width, height: width, samples: 1_024,
                settings: .init(maxBounces: 2, seed: behind ? 5 : 7),
                lighting: .init(sky: .map(try panorama(behind: behind)))
            )
            return MediaFixture.center(image, width: width, height: width, radius: 4)
        }
        let transmitted = try render(behind: true)
        let reflected = try render(behind: false)
        for channel in 0..<3 {
            let reference = MediaReference.slab(
                depth: Double(extinction[channel] * thickness),
                albedo: MediaReference.singleScatteringAlbedo(Double(colors[channel])),
                g: Double(anisotropy), photons: 400_000, seed: UInt64(17 + channel)
            )
            #expect(abs(Double(transmitted[channel]) - reference.transmittance) < 0.02 * reference.transmittance + 0.004,
                    "T channel \(channel): \(transmitted[channel]) vs \(reference.transmittance)")
            #expect(abs(Double(reflected[channel]) - reference.reflectance) < 0.02 * reference.reflectance + 0.004,
                    "R channel \(channel): \(reflected[channel]) vs \(reference.reflectance)")
        }
    }

    @Test("Nested index-matched media preserve each layer of analytic shadow attenuation")
    func nestedShadowMedia() throws {
        let width = 9
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.02, near: 0.1, far: 30)
        let receiverPosition = SIMD3<Float>(0, 0, -2)
        let toLight = SIMD3<Float>(0, 0.6, 0.8)
        let boxPosition = receiverPosition + 4 * toLight
        let boxOrientation = simd_quatf(from: SIMD3(0, 0, 1), to: toLight)
        let outerExtinction = SIMD3<Float>(0.2, 0.4, 0.7)
        let innerExtinction = SIMD3<Float>(0.6, 0.3, 0.1)
        func box(thickness: Float, width: Float, extinction: SIMD3<Float>) throws -> SceneNode {
            let attenuation = SIMD3<Float>((0..<3).map { exp(-extinction[$0]) })
            let material = try PhysicallyBasedMaterial(
                roughness: 0, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque,
                transmission: .init(
                    factor: 1, indexOfRefraction: 1, thickness: 1, attenuationDistance: 1, attenuationColor: attenuation
                )
            )
            return SceneNode(
                transform: .init(translation: boxPosition, rotation: boxOrientation, scale: SIMD3(width, width, thickness)),
                mesh: try fixture.renderer(.cube(), material)
            )
        }
        let receiver = try fixture.renderer(
            .plane(width: 4, height: 4),
            .init(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, specular: .init(factor: 0))
        )
        let light = try SceneNode(
            transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -toLight)),
            light: PunctualLight(kind: .directional, intensity: 1)
        )
        let graph = try SceneGraph(roots: [
            SceneNode(transform: .init(translation: receiverPosition), mesh: receiver),
            box(thickness: 2, width: 2, extinction: outerExtinction),
            box(thickness: 1, width: 1, extinction: innerExtinction),
            light,
        ], camera: camera)
        // The orthogonal shadow path traverses one world unit in each medium.
        let expected = SIMD3<Float>((0..<3).map { exp(-outerExtinction[$0] - innerExtinction[$0]) * toLight.z / .pi })
        let image = try fixture.trace(
            graph, camera: camera, width: width, height: width, samples: 16, settings: .init(maxBounces: 1, seed: 5),
            lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph))
        )
        let measured = MediaFixture.center(image, width: width, height: width, radius: 4)
        #expect(simd_reduce_max(abs(measured - expected) / expected) < 0.005, "\(measured) vs \(expected)")
    }

    @Test("Live transmission switches the shadow pipeline for color, guides and AOVs")
    func liveShadowBoundaryVariant() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.02, near: 0.1, far: 30)
        let receiverPosition = SIMD3<Float>(0, 0, -2)
        let toLight = SIMD3<Float>(0, 0.6, 0.8)
        let window = try PhysicallyBasedMaterial(
            roughness: 0, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque,
            transmission: .init(
                factor: 1, indexOfRefraction: 1, thickness: 1, attenuationDistance: 1,
                attenuationColor: SIMD3(repeating: 0.5)
            )
        ).withLiveFactors()
        let factors = try #require(window.liveFactors)
        try factors.update { $0.transmissionFactor = 0.5 }

        let receiver = try fixture.renderer(
            .plane(width: 4, height: 4),
            .init(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, specular: .init(factor: 0))
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: receiverPosition), mesh: receiver),
            SceneNode(transform: .init(
                translation: receiverPosition + 4 * toLight,
                rotation: simd_quatf(from: SIMD3(0, 0, 1), to: toLight)
            ), mesh: try fixture.renderer(.cube(), window)),
        ], camera: camera)
        let sun = try PathTracer.AnalyticLight(PunctualLight(kind: .directional, intensity: 1), direction: -toLight)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(textures: .baked(size: 16)))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: .init(analyticLights: [sun]))

        let width = 9
        let color = try fixture.output(width: width, height: width)
        let guides = try PathTracer.Guides(
            normalDepth: fixture.output(width: width, height: width), albedoRoughness: fixture.output(width: width, height: width)
        )
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: width, mipmapped: false
        )
        descriptor.usage = .shaderWrite
        let depth = try #require(fixture.device.makeTexture(descriptor: descriptor))
        func render(auxiliary: Bool) throws -> Float {
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            let accumulation = try tracer.makeAccumulation(width: width, height: width)
            try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: color, accumulation: accumulation,
                guides: auxiliary ? guides : nil, commandBuffer: buffer, samplesPerPixel: 16,
                settings: .init(maxBounces: 1, seed: 5), aovs: auxiliary ? .init(depth: depth) : nil
            )
            try fixture.finish(buffer)
            return MediaFixture.center(try fixture.readback(color), width: width, height: width, radius: 3).x
        }

        #expect(try render(auxiliary: false) == 0)
        #expect(try render(auxiliary: true) == 0)
        factors.reset()
        #expect(try scene.updateMaterials())
        let expected = 0.5 * toLight.z / Float.pi
        #expect(abs(try render(auxiliary: false) - expected) < 0.005)
        #expect(abs(try render(auxiliary: true) - expected) < 0.005)
        try factors.update { $0.transmissionFactor = 0.5 }
        #expect(try scene.updateMaterials())
        #expect(try render(auxiliary: true) == 0)
    }

    @Test("Henyey-Greenstein sampling follows its density (chi-square) and reports it as the evaluated phase function")
    func phaseSampling() throws {
        let source = try MaterialShaderSource.splicingExtensions(into: """
        \(MaterialShaderSource.includeLine)
        static uint hgHash(uint value) {
            value ^= value >> 16u; value *= 0x7feb352du; value ^= value >> 15u; value *= 0x846ca68bu;
            return value ^ (value >> 16u);
        }
        kernel void frostfield_test_phase_samples(
            device float4* output [[buffer(0)]], constant float4& parameters [[buffer(1)]],
            uint index [[thread_position_in_grid]]
        ) {
            float3 incoming = normalize(parameters.xyz);
            uint first = hgHash(index * 2u + 0x9e3779b9u);
            uint second = hgHash(first ^ (index * 0x85ebca6bu + 7u));
            float2 random = float2(first >> 8u, second >> 8u) * 0x1p-24f;
            float3 scattered = frostfieldSampleHenyeyGreenstein(incoming, parameters.w, random);
            output[index] = float4(scattered, frostfieldHenyeyGreenstein(dot(incoming, scattered), parameters.w));
        }
        """)
        let library = try fixture.device.makeLibrary(source: source, options: nil)
        let pipeline = try fixture.device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "frostfield_test_phase_samples"))
        )
        let count = 1 << 20
        let buffer = try #require(fixture.device.makeBuffer(length: count * 16, options: .storageModeShared))
        let incoming = simd_normalize(SIMD3<Float>(0.3, -0.5, 0.8))
        for g in [-0.7, 0.0, 0.3, 0.9] as [Float] {
            var parameters = SIMD4(incoming, g)
            let commands = try #require(fixture.queue.makeCommandBuffer())
            let encoder = try #require(commands.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&parameters, length: 16, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: count / 256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            try fixture.finish(commands)
            let samples = UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: count)

            // Histogram the scattering cosine (40 bins) and azimuth about the incoming direction (8 bins).
            let axis = SIMD3<Double>(incoming)
            let tangent = simd_normalize(simd_cross(SIMD3<Double>(0, 0, 1), axis))
            let bitangent = simd_cross(axis, tangent)
            let cosineBins = 40
            let azimuthBins = 8
            var counts = Array(repeating: 0.0, count: cosineBins * azimuthBins)
            var meanCosine = 0.0
            var worstDensity = 0.0
            for sample in samples {
                let direction = SIMD3<Double>(SIMD3(sample.x, sample.y, sample.z))
                let mu = min(max(simd_dot(direction, axis), -1), 1)
                let phi = atan2(simd_dot(direction, bitangent), simd_dot(direction, tangent)) + .pi
                let row = min(Int((mu + 1) / 2 * Double(cosineBins)), cosineBins - 1)
                let column = min(Int(phi / (2 * .pi) * Double(azimuthBins)), azimuthBins - 1)
                counts[row * azimuthBins + column] += 1
                meanCosine += mu
                let density = MediaReference.henyeyGreenstein(mu, g: Double(g))
                worstDensity = max(worstDensity, abs(Double(sample.w) - density) / density)
                #expect(abs(simd_length(direction) - 1) < 1e-5)
            }
            meanCosine /= Double(count)
            var chiSquare = 0.0
            for row in 0..<cosineBins {
                let low = -1 + 2 * Double(row) / Double(cosineBins)
                let high = -1 + 2 * Double(row + 1) / Double(cosineBins)
                let expected = Double(count) * (MediaReference.henyeyGreensteinCDF(high, g: Double(g))
                    - MediaReference.henyeyGreensteinCDF(low, g: Double(g))) / Double(azimuthBins)
                for column in 0..<azimuthBins {
                    let observed = counts[row * azimuthBins + column]
                    chiSquare += (observed - expected) * (observed - expected) / expected
                }
            }
            // 319 degrees of freedom: the Wilson-Hilferty 0.999 quantile is about 403.
            #expect(chiSquare < 410, "g \(g): chi-square \(chiSquare)")
            // The mean scattering cosine of Henyey-Greenstein is g.
            #expect(abs(meanCosine - Double(g)) < 0.003, "g \(g): mean cosine \(meanCosine)")
            #expect(worstDensity < 2e-4, "g \(g): density mismatch \(worstDensity)")
        }
    }

    @Test("Subsurface with a zero radius is the diffuse lobe; a tiny radius walks to the same albedo")
    func subsurfaceRadiusLimit() throws {
        let width = 33
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 30)
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 96, rings: 48)
        let color = SIMD3<Float>(0.8, 0.5, 0.2)
        func graph(_ material: PhysicallyBasedMaterial) throws -> SceneGraph {
            SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -3.5)), mesh: try fixture.renderer(sphere, material)),
            ], camera: camera)
        }
        func material(_ subsurface: PhysicallyBasedMaterial.Subsurface?, base: SIMD3<Float>) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: 0.6, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(base, 1), alphaMode: .opaque,
                specular: .init(factor: 0), subsurface: subsurface
            )
        }
        let diffuse = try graph(material(nil, base: color))
        let zero = try graph(material(.init(color: color, radius: .zero, scale: 1), base: SIMD3(repeating: 1)))
        let tiny = try graph(material(.init(color: color, radius: SIMD3(repeating: 1), scale: 0.002), base: SIMD3(repeating: 1)))
        let sun = try PathTracer.AnalyticLight(PunctualLight(kind: .directional, intensity: 2), direction: simd_normalize(SIMD3(-1, -0.5, -0.6)))
        let lighting = try PathTracer.Lighting(analyticLights: [sun], environment: SIMD3(repeating: 0.25))
        let settings = PathTracer.Settings(maxBounces: 2, seed: 23)
        let reference = try fixture.trace(diffuse, camera: camera, width: width, height: width, samples: 256, settings: settings, lighting: lighting)
        let exact = try fixture.trace(zero, camera: camera, width: width, height: width, samples: 256, settings: settings, lighting: lighting)
        let walked = try fixture.trace(tiny, camera: camera, width: width, height: width, samples: 1_024, settings: settings, lighting: lighting)
        MediaFixture.save(reference, width: width, height: width, name: "subsurface-diffuse")
        MediaFixture.save(walked, width: width, height: width, name: "subsurface-tiny")
        // The zero radius takes the diffuse path, so the images agree up to rounding.
        #expect(PathTestFixture.maximumDifference(exact, reference) < 1e-3)
        // The walk's albedo is the Kulla and Conty inversion's target within about 1%; the
        // remaining difference is Monte Carlo noise and diffusion over 0.2% of the radius.
        let sphereReference = reference.enumerated().filter { $0.element.w > 0 && simd_reduce_max(SIMD3($0.element.x, $0.element.y, $0.element.z)) > 0 }
        var sumReference = SIMD3<Float>.zero
        var sumWalked = SIMD3<Float>.zero
        for (index, pixel) in sphereReference {
            sumReference += SIMD3(pixel.x, pixel.y, pixel.z)
            sumWalked += SIMD3(walked[index].x, walked[index].y, walked[index].z)
        }
        let ratio = sumWalked / sumReference
        #expect(simd_reduce_max(abs(ratio - 1)) < 0.025, "walked / diffuse \(ratio)")
    }

    @Test("Single scattering in fog matches the analytic in-scattered radiance, and an occluder carves a light shaft")
    func fogSingleScattering() throws {
        let width = 33
        let height = 33
        let fov = Float.pi / 6
        let camera = try PerspectiveCamera(verticalFieldOfView: fov, near: 0.1, far: 30)
        let extinction: Float = 0.25
        let albedo: Float = 0.8
        let anisotropy: Float = 0.3
        let irradiance: Float = 4
        let minimum = SIMD3<Float>(-50, -1, -8)
        let maximum = SIMD3<Float>(50, 1, -2)
        let fog = try PathTracer.Fog(
            density: extinction, color: SIMD3(repeating: albedo), anisotropy: anisotropy, minimum: minimum, maximum: maximum
        )
        let sun = try PathTracer.AnalyticLight(PunctualLight(kind: .directional, intensity: irradiance), direction: SIMD3(0, -1, 0))
        // A black, double-sided slat above the fog shadows z in [-5, -4] for every x.
        let slat = try fixture.renderer(
            try MeshGeometry.plane(width: 120, height: 1),
            try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                alphaMode: .opaque, isDoubleSided: true, specular: .init(factor: 0)
            )
        )
        let shaded = SceneGraph(roots: [
            SceneNode(transform: .init(
                translation: SIMD3(0, 2, -4.5), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            ), mesh: slat),
        ], camera: camera)
        // A scene needs geometry; the open graph keeps the slat far behind the camera.
        let open = SceneGraph(roots: [
            SceneNode(transform: .init(
                translation: SIMD3(0, 2, 40), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            ), mesh: slat),
        ], camera: camera)
        let lighting = try PathTracer.Lighting(analyticLights: [sun], fog: fog)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 29)
        let lit = try fixture.trace(open, camera: camera, width: width, height: height, samples: 512, settings: settings, lighting: lighting)
        let shafted = try fixture.trace(shaded, camera: camera, width: width, height: height, samples: 512, settings: settings, lighting: lighting)
        MediaFixture.save(lit, width: width, height: height, name: "fog-lit", exposure: 4)
        MediaFixture.save(shafted, width: width, height: height, name: "fog-shaft", exposure: 4)

        // The single-scattering integral along a pixel's rays, averaged over its footprint.
        let tanY = tan(Double(fov) / 2)
        let tanX = tanY * Double(width) / Double(height)
        func expected(column: Int, row: Int, shadowed: Bool) -> Double {
            var total = 0.0
            let grid = 6
            for sub in 0..<(grid * grid) {
                let u = (Double(column) + (Double(sub % grid) + 0.5) / Double(grid)) / Double(width)
                let v = (Double(row) + (Double(sub / grid) + 0.5) / Double(grid)) / Double(height)
                let direction = simd_normalize(SIMD3<Double>((2 * u - 1) * tanX, (1 - 2 * v) * tanY, -1))
                var entry = 0.0
                var exit = Double.infinity
                for axis in 0..<3 where abs(direction[axis]) > 1e-12 {
                    let a = Double(minimum[axis]) / direction[axis]
                    let b = Double(maximum[axis]) / direction[axis]
                    entry = max(entry, min(a, b))
                    exit = min(exit, max(a, b))
                }
                guard exit > entry else { continue }
                let phase = MediaReference.henyeyGreenstein(direction.y, g: Double(anisotropy))
                let steps = 2_000
                let dt = (exit - entry) / Double(steps)
                var integral = 0.0
                for step in 0..<steps {
                    let t = entry + (Double(step) + 0.5) * dt
                    let point = direction * t
                    if shadowed, point.z >= -5, point.z <= -4 { continue }
                    let up = Double(maximum.y) - point.y
                    integral += exp(-Double(extinction) * (up + (t - entry))) * dt
                }
                total += Double(irradiance) * Double(albedo * extinction) * phase * integral
            }
            return total / Double(grid * grid)
        }
        var worst = 0.0
        var shaftContrast: [Double] = []
        for row in [12, 16, 20] {
            for column in stride(from: 4, through: 28, by: 4) {
                let open = Double(lit[row * width + column].x)
                let reference = expected(column: column, row: row, shadowed: false)
                worst = max(worst, abs(open - reference) / reference)
                let carved = Double(shafted[row * width + column].x)
                let carvedReference = expected(column: column, row: row, shadowed: true)
                worst = max(worst, abs(carved - carvedReference) / carvedReference)
                shaftContrast.append(carved / open)
            }
        }
        #expect(worst < 0.03, "largest relative error \(worst)")
        // The slat removes a visible part of the in-scattered light.
        #expect((shaftContrast.min() ?? 1) < 0.85)
        // With a hidden background, coverage is the fraction of camera rays that scatter in the fog.
        let hidden = try fixture.trace(
            open, camera: camera, width: width, height: height, samples: 64, settings: settings,
            lighting: .init(analyticLights: [sun], backgroundVisible: false, fog: fog)
        )
        // The center ray crosses 6 units of fog.
        let alpha = hidden[16 * width + 16].w
        #expect(abs(Double(alpha) - (1 - exp(-Double(extinction) * 6))) < 0.05, "fog coverage \(alpha)")
    }

    @Test("Absorbing fog attenuates the environment by Beer-Lambert, and fog edits restart accumulation")
    func fogAttenuation() throws {
        let width = 9
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.05, near: 0.1, far: 30)
        let far = try fixture.renderer(
            try MeshGeometry.plane(width: 1, height: 1),
            try PhysicallyBasedMaterial(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque)
        )
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, 20)), mesh: far)], camera: camera)
        let extinction = SIMD3<Float>(0.1, 0.3, 0.6)
        let fog = try PathTracer.Fog(
            extinction: extinction, albedo: .zero, minimum: SIMD3(-10, -10, -7), maximum: SIMD3(10, 10, -3)
        )
        let image = try fixture.trace(
            graph, camera: camera, width: width, height: width, samples: 64, settings: .init(maxBounces: 1, seed: 1),
            lighting: .init(environment: SIMD3(repeating: 1), fog: fog)
        )
        let measured = MediaFixture.center(image, width: width, height: width, radius: 4)
        let expected = fog.transmittance(distance: 4)
        let clearGreen = try PathTracer.Fog(
            extinction: SIMD3(0.1, 0, 0.6), minimum: fog.minimum, maximum: fog.maximum
        )
        #expect(clearGreen.transmittance(distance: .infinity) == SIMD3(0, 1, 0))
        #expect(simd_reduce_max(abs(measured - expected) / expected) < 0.01, "\(measured) vs \(expected)")

        #expect(throws: PathTracer.Failure.invalidFog) {
            try PathTracer.Fog(density: 1, minimum: SIMD3(0, 0, 0), maximum: SIMD3(1, 0, 1))
        }
        #expect(throws: PathTracer.Failure.invalidFog) {
            try PathTracer.Fog(density: -1, minimum: .zero, maximum: SIMD3(repeating: 1))
        }
        #expect(throws: PathTracer.Failure.invalidFog) {
            try PathTracer.Fog(density: 1, anisotropy: 1, minimum: .zero, maximum: SIMD3(repeating: 1))
        }

        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let accumulation = try tracer.makeAccumulation(width: width, height: width)
        let target = try fixture.output(width: width, height: width)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        let first = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: buffer, lighting: .init(fog: fog)
        )
        try fixture.finish(buffer)
        let next = try #require(fixture.queue.makeCommandBuffer())
        let denser = try PathTracer.Fog(extinction: extinction * 2, albedo: .zero, minimum: fog.minimum, maximum: fog.maximum)
        #expect(throws: PathTracer.Failure.lightingChanged) {
            try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: first, commandBuffer: next, lighting: .init(fog: denser)
            )
        }
    }
}
