import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathFogDensityTests {
    private let fixture: PathTestFixture

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Chromatic transmittance matches voxel optical depth in both ray directions",
          arguments: PathAccelerationTests.modes)
    func opticalDepth(acceleration: PathTracingScene.Options.Acceleration) throws {
        // Each x/y column has a different sum over the four z layers. Layers run
        // from world minimum to maximum, and x is the fastest axis.
        let columns: [Float] = [0, 0.5, 1.25, 2]
        let densities: [Float] = [
            0, 0.25, 0.5, 1,
            0, 0, 0.25, 0,
            0, 0.25, 0, 1,
            0, 0, 0.5, 0
        ]
        let field = try grid(SIMD3(2, 2, 4), densities)
        let fog = try PathTracer.Fog(
            extinction: SIMD3(0.12, 0, 0.8), albedo: .zero, minimum: SIMD3(-2, -2, -7), maximum: SIMD3(2, 2, -3),
            densityGrid: field
        )
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 1), fog: fog)
        let reversed = SceneTransform(
            translation: SIMD3(0, 0, -10), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        )
        for reverse in [false, true] {
            let image = try trace(
                tracer, lighting: lighting, samples: 1024, size: 4, height: 4, camera: reverse ? reversed : .init()
            )
            for y in 0..<2 {
                for x in 0..<2 {
                    let optical = columns[(1 - y) * 2 + (reverse ? 1 - x : x)]
                    let expected = fog.transmittance(distance: optical)
                    var measured = SIMD3<Float>.zero
                    for row in y * 2..<y * 2 + 2 {
                        for column in x * 2..<x * 2 + 2 {
                            let pixel = image[row * 4 + column]
                            measured += SIMD3(pixel.x, pixel.y, pixel.z) / 4
                        }
                    }
                    #expect(simd_reduce_max(abs(measured - expected)) < 0.012, "\(measured) vs \(expected)")
                    if optical == 0 { #expect(measured == SIMD3(repeating: 1)) }
                }
            }
        }
    }

    @Test("A heterogeneous shadow segment has the expected finite optical depth",
          arguments: PathAccelerationTests.modes)
    func shadowOpticalDepth(acceleration: PathTracingScene.Options.Acceleration) throws {
        let receiver = try fixture.pbr(.plane(width: 2, height: 2))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: receiver)
        let tracer = try makeTracer(acceleration, nodes: [node])
        let sun = try PathTracer.AnalyticLight(
            PunctualLight(kind: .directional, intensity: 4), direction: SIMD3(-1, 0, -1)
        )
        // The camera ray never enters the box. The 45-degree shadow ray crosses
        // half a world unit in X through each of the 0.2 and 0.8 density cells.
        let fog = try PathTracer.Fog(
            extinction: SIMD3(0.3, 0.5, 1), albedo: .zero,
            minimum: SIMD3(0.2, -1, -3), maximum: SIMD3(1.2, 1, -1),
            densityGrid: grid(SIMD3(2, 1, 1), [0.2, 0.8])
        )
        // The point source stops 0.1 units into the second cell. The directional
        // light traverses the whole grid. A narrow camera window fixes the geometry.
        let point = try PathTracer.AnalyticLight(
            PunctualLight(kind: .point(), intensity: 4), position: SIMD3(0.8, 0, -2.2)
        )
        let mesh = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        let meshFog = try PathTracer.Fog(
            extinction: fog.extinction, albedo: .zero, meshBoundary: mesh,
            transform: .init(translation: SIMD3(0.7, 0, -2), scale: SIMD3(1, 2, 2)),
            densityGrid: fog.densityGrid
        )
        let paths: [(PathTracer.AnalyticLight, Float)] = [(sun, 0.5), (point, 0.18)]
        for (light, opticalInX) in paths {
            let clear = try #require(trace(tracer, lighting: .init(analyticLights: [light]), samples: 16, height: 1e-4).first)
            for volume in [fog, meshFog] {
                let attenuated = try #require(trace(
                    tracer, lighting: .init(analyticLights: [light], fog: volume), samples: 16, height: 1e-4
                ).first)
                let measured = SIMD3(attenuated.x / clear.x, attenuated.y / clear.y, attenuated.z / clear.z)
                let expected = fog.transmittance(distance: sqrt(Float(2)) * opticalInX)
                #expect(simd_reduce_max(abs(measured - expected)) < 0.002, "\(measured) vs \(expected)")
            }
        }
    }

    @Test("Free paths through vacuum land in the dense slab and respect its shadows",
          arguments: PathAccelerationTests.modes)
    func freePathsReachDenseSlab(acceleration: PathTracingScene.Options.Acceleration) throws {
        let minimum = SIMD3<Float>(-2, -1, -7)
        let extinction = SIMD3<Float>(0.22, 0.38, 0.55)
        let albedo = SIMD3<Float>(0.8, 0.75, 0.6)
        let reference = try PathTracer.Fog(
            extinction: extinction, albedo: albedo, anisotropy: 0.23, minimum: minimum, maximum: SIMD3(2, 1, -5)
        )
        let heterogeneous = try PathTracer.Fog(
            extinction: extinction, albedo: albedo, anisotropy: 0.23, minimum: minimum, maximum: SIMD3(2, 1, -3),
            densityGrid: grid(SIMD3(1, 1, 4), [1, 1, 0, 0])
        )
        let sun = try PathTracer.AnalyticLight(
            PunctualLight(kind: .directional, intensity: 4), direction: SIMD3(0, -1, 0)
        )
        let slat = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 0.75),
            material: .physicallyBased(try .init(alphaMode: .opaque, isDoubleSided: true))
        )
        let blocker = SceneNode(
            transform: .init(translation: SIMD3(0, 2, -6), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
            mesh: slat
        )
        let open = try makeTracer(acceleration, nodes: [dummy()])
        let blocked = try makeTracer(acceleration, nodes: [blocker])
        var green: [Float] = []
        for tracer in [open, blocked] {
            let dense = try #require(trace(
                tracer, lighting: .init(analyticLights: [sun], backgroundVisible: false, fog: heterogeneous),
                samples: 512
            ).first)
            let expected = try #require(trace(
                tracer, lighting: .init(analyticLights: [sun], backgroundVisible: false, fog: reference),
                samples: 512
            ).first)
            #expect(simd_reduce_max(abs(dense - expected)) < 0.001, "\(dense) vs \(expected)")
            #expect(dense.x > 0.02 && dense.w > 0.3)
            green.append(dense.y)
        }
        #expect(green[1] < green[0] * 0.85)
    }

    @Test("Overlapping heterogeneous fog multiplies camera transmittance across both grids",
          arguments: PathAccelerationTests.modes)
    func overlappingOpticalDepth(acceleration: PathTracingScene.Options.Acceleration) throws {
        let first = try PathTracer.Fog(
            extinction: SIMD3(0.12, 0, 0.45), albedo: .zero,
            minimum: SIMD3(-2, -2, -7), maximum: SIMD3(2, 2, -3),
            densityGrid: grid(SIMD3(1, 1, 2), [0.2, 0.7])
        )
        let second = try PathTracer.Fog(
            extinction: SIMD3(0.25, 0.3, 0), albedo: .zero,
            minimum: SIMD3(-2, -2, -9), maximum: SIMD3(2, 2, -5),
            densityGrid: grid(SIMD3(1, 1, 4), [0.1, 0.3, 0.6, 0.9])
        )
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let expected = first.transmittance(distance: 1.8) * second.transmittance(distance: 1.9)
        let reverse = SceneTransform(
            translation: SIMD3(0, 0, -12), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        )
        for (camera, volumes) in [(SceneTransform(), [first, second]), (reverse, [second, first])] {
            let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 1), fogVolumes: volumes)
            let image = try trace(tracer, lighting: lighting, samples: 1024, size: 4, camera: camera)
            let mean = image.reduce(SIMD3<Float>.zero) { $0 + SIMD3($1.x, $1.y, $1.z) } / Float(image.count)
            #expect(simd_reduce_max(abs(mean - expected)) < 0.015, "\(mean) vs \(expected)")
        }
    }

    @Test("Overlapping media retain individual phase and color and share shadow extinction",
          arguments: PathAccelerationTests.modes, [false, true])
    func overlappingPhaseFunctions(acceleration: PathTracingScene.Options.Acceleration, secondIsMesh: Bool) throws {
        let minimum = SIMD3<Float>(-2, -1, -7)
        let maximum = SIMD3<Float>(2, 1, -3)
        let first = try PathTracer.Fog(
            extinction: SIMD3(0.25, 0.10, 0.12), albedo: SIMD3(0.85, 0.15, 0.25),
            anisotropy: 0.62, minimum: minimum, maximum: maximum
        )
        // The same second region, once as a world box and once as an invisible mesh cube.
        let second: PathTracer.Fog
        if secondIsMesh {
            let mesh = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
            second = try PathTracer.Fog(
                extinction: SIMD3(0.12, 0.36, 0.20), albedo: SIMD3(0.12, 0.9, 0.6),
                anisotropy: -0.18, meshBoundary: mesh,
                transform: .init(translation: (minimum + maximum) / 2, scale: maximum - minimum),
                densityGrid: grid(SIMD3(1, 1, 2), [0.25, 0.8])
            )
        } else {
            second = try PathTracer.Fog(
                extinction: SIMD3(0.12, 0.36, 0.20), albedo: SIMD3(0.12, 0.9, 0.6),
                anisotropy: -0.18, minimum: minimum, maximum: maximum,
                densityGrid: grid(SIMD3(1, 1, 2), [0.25, 0.8])
            )
        }
        let intensity: Float = 5
        let sun = try PathTracer.AnalyticLight(
            PunctualLight(kind: .directional, intensity: intensity), direction: SIMD3(0, -1, 0)
        )
        let lighting = try PathTracer.Lighting(analyticLights: [sun], backgroundVisible: false, fogVolumes: [first, second])
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let image = try trace(tracer, lighting: lighting, samples: 1024, size: 4, height: 1e-4)
        let measured = image.reduce(SIMD4<Float>.zero, +) / Float(image.count)
        func phase(_ g: Float) -> Float { (1 - g * g) / (4 * .pi * pow(1 + g * g, 1.5)) }
        let firstScatter = first.extinction * first.albedo * phase(first.anisotropy)
        let secondScatter = second.extinction * second.albedo * phase(second.anisotropy)
        var expected = SIMD3<Float>.zero
        var cameraTransmittance = SIMD3<Float>(repeating: 1)
        // Two 2-unit camera segments front-to-back. The transverse shadow
        // travels 1 unit in the same layer before it reaches the light.
        for density: Float in [0.8, 0.25] {
            let extinction = first.extinction + density * second.extinction
            let scatter = firstScatter + density * secondScatter
            for channel in 0..<3 {
                let sigma = extinction[channel]
                expected[channel] += intensity * scatter[channel] * cameraTransmittance[channel]
                    * exp(-sigma) * (1 - exp(-2 * sigma)) / sigma
                cameraTransmittance[channel] *= exp(-2 * sigma)
            }
        }
        #expect(simd_reduce_max(abs(SIMD3(measured.x, measured.y, measured.z) - expected)) < 0.006,
                "\(measured) vs \(expected)")
        let coverage = 1 - (cameraTransmittance.x + cameraTransmittance.y + cameraTransmittance.z) / 3
        #expect(abs(measured.w - coverage) < 0.018)
    }

    @Test("A hollow mesh clips separate voxel intervals and classifies rays that start inside",
          arguments: PathAccelerationTests.modes)
    func hollowMesh(acceleration: PathTracingScene.Options.Acceleration) throws {
        let cube = MeshGeometry.cube()
        let outer = try cube.transformed(by: SceneTransform(scale: SIMD3(4, 4, 6)).matrix)
        let cavity = try cube.transformed(by: SceneTransform(scale: SIMD3(2, 2, 2)).matrix)
        let shift = UInt32(outer.vertices.count)
        let innerIndices = stride(from: 0, to: cavity.indices.count, by: 3).flatMap { offset in
            [cavity.indices[offset] + shift, cavity.indices[offset + 2] + shift, cavity.indices[offset + 1] + shift]
        }
        let geometry = try MeshGeometry(vertices: outer.vertices + cavity.vertices, indices: outer.indices + innerIndices)
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: geometry)
        #expect(boundary.shellCount == 2 && boundary.triangleCount == 24)
        let fog = try PathTracer.Fog(
            extinction: SIMD3(0.2, 0.45, 0.7), albedo: .zero, meshBoundary: boundary,
            transform: .init(translation: SIMD3(0, 0, -7), scale: SIMD3(-1, 1, 1.5)),
            densityGrid: grid(SIMD3(1, 1, 3), [0.2, 0.9, 0.75])
        )
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 1), fog: fog)
        let reversed = SceneTransform(
            translation: SIMD3(0, 0, -15), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        )
        let inside = SceneTransform(translation: SIMD3(0, 0, -4))
        let emitter = try PathTracer.AreaLight(
            center: SIMD3(0, 0, -5), halfU: SIMD3(0.5, 0, 0), halfV: SIMD3(0, 0.5, 0),
            radiance: SIMD3(repeating: 1), cameraVisible: true
        )
        let clipped = try PathTracer.Lighting(lights: [emitter], fog: fog)
        let cases: [(SceneTransform, PathTracer.Lighting, Float)] = [
            (.init(), lighting, 3 * (0.2 + 0.75)),
            (reversed, lighting, 3 * (0.2 + 0.75)),
            (inside, lighting, 1.4 * 0.75 + 3 * 0.2),
            (inside, clipped, 0.9 * 0.75)
        ]
        for (camera, light, opticalDepth) in cases {
            let pixels = try trace(tracer, lighting: light, samples: 1024, size: 4, camera: camera)
            let mean = pixels.reduce(SIMD3<Float>.zero) { $0 + SIMD3($1.x, $1.y, $1.z) } / Float(pixels.count)
            let expected = fog.transmittance(distance: opticalDepth)
            #expect(simd_reduce_max(abs(mean - expected)) < 0.017, "\(mean) vs \(expected)")
        }
    }

    @Test("A single mesh volume attenuates camera rays by the analytic optical depth from either side",
          arguments: PathAccelerationTests.modes)
    func singleMeshCameraRays(acceleration: PathTracingScene.Options.Acceleration) throws {
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        let fog = try PathTracer.Fog(
            extinction: SIMD3(0.35, 0.8, 0.12), albedo: .zero, meshBoundary: boundary,
            transform: .init(translation: SIMD3(0, 0, -5), scale: SIMD3(4, 4, 2))
        )
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 1), fog: fog)
        let reversed = SceneTransform(
            translation: SIMD3(0, 0, -10), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        )
        let expected = fog.transmittance(distance: 2)
        for camera in [SceneTransform(), reversed] {
            // A narrow view crosses the face centers, where triangles meet. A wide view
            // crosses the faces away from their diagonals.
            for height: Float in [0.04, 3] {
                let image = try trace(tracer, lighting: lighting, samples: 1024, size: 4, height: height, camera: camera)
                let mean = average(image)
                #expect(simd_reduce_max(abs(mean - expected)) < 0.015, "\(mean) vs \(expected) at height \(height)")
            }
        }
    }

    @Test("A ray scattered by a mirror crosses a mesh volume the camera ray never entered",
          arguments: PathAccelerationTests.modes)
    func scatteredRayThroughMesh(acceleration: PathTracingScene.Options.Acceleration) throws {
        let mirror = try fixture.pbr(.plane(width: 2, height: 2), color: SIMD3(repeating: 1), roughness: 0.02, metallic: 1)
        // The 45-degree mirror turns the -Z camera ray toward +X, beside the camera's own path.
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 1, 0))),
            mesh: mirror
        )
        let tracer = try makeTracer(acceleration, nodes: [node])
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        let fog = try PathTracer.Fog(
            extinction: SIMD3(0.3, 0.6, 0.15), albedo: .zero, meshBoundary: boundary,
            transform: .init(translation: SIMD3(4, 0, -3), scale: SIMD3(2, 4, 4))
        )
        let clear = average(try trace(
            tracer, lighting: .init(environment: SIMD3(repeating: 1)), samples: 512, size: 4, height: 0.04
        ))
        let fogged = average(try trace(
            tracer, lighting: .init(environment: SIMD3(repeating: 1), fog: fog), samples: 512, size: 4, height: 0.04
        ))
        #expect(simd_reduce_min(clear) > 0.3, "The mirror must reflect the environment: \(clear)")
        let measured = fogged / clear
        let expected = fog.transmittance(distance: 2)
        #expect(simd_reduce_max(abs(measured - expected)) < 0.02, "\(measured) vs \(expected)")
    }

    @Test("Multiple scattering in a mesh cube matches the same cube authored as a world box",
          arguments: PathAccelerationTests.modes)
    func multipleScatteringMatchesBox(acceleration: PathTracingScene.Options.Acceleration) throws {
        let extinction = SIMD3<Float>(0.5, 0.8, 1.2)
        let albedo = SIMD3<Float>(0.9, 0.8, 0.7)
        let box = try PathTracer.Fog(
            extinction: extinction, albedo: albedo, anisotropy: 0.3,
            minimum: SIMD3(-1.5, -1.5, -6.5), maximum: SIMD3(1.5, 1.5, -3.5)
        )
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        let mesh = try PathTracer.Fog(
            extinction: extinction, albedo: albedo, anisotropy: 0.3, meshBoundary: boundary,
            transform: .init(translation: SIMD3(0, 0, -5), scale: SIMD3(3, 3, 3))
        )
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        // Scattered rays start inside the region and must classify it and leave it like a box would.
        // The wide view includes rays that miss the cube.
        var means: [SIMD3<Float>] = []
        for fog in [box, mesh] {
            let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 1), fog: fog)
            means.append(average(try trace(tracer, lighting: lighting, samples: 1024, size: 4, height: 2, bounces: 8)))
        }
        #expect(simd_reduce_min(means[0]) > 0.3 && simd_reduce_max(means[0]) < 0.95, "\(means[0])")
        #expect(simd_reduce_max(abs(means[1] - means[0])) < 0.02, "mesh \(means[1]) vs box \(means[0])")
    }

    @Test("A mesh volume with a grid and a box add their extinction where they overlap",
          arguments: PathAccelerationTests.modes)
    func meshWithBox(acceleration: PathTracingScene.Options.Acceleration) throws {
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        // World z from -6 to -4. The grid's first layer is the far one, so the camera
        // crosses 0.7 then 0.2 density across one unit each.
        let mesh = try PathTracer.Fog(
            extinction: SIMD3(0.30, 0.10, 0.50), albedo: .zero, meshBoundary: boundary,
            transform: .init(translation: SIMD3(0, 0, -5), scale: SIMD3(4, 4, 2)),
            densityGrid: grid(SIMD3(1, 1, 2), [0.2, 0.7])
        )
        let box = try PathTracer.Fog(
            extinction: SIMD3(0.10, 0.50, 0.20), albedo: .zero, minimum: SIMD3(-2, -2, -5), maximum: SIMD3(2, 2, -3)
        )
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let reversed = SceneTransform(
            translation: SIMD3(0, 0, -10), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        )
        let expected = mesh.transmittance(distance: 0.9) * box.transmittance(distance: 2)
        for camera in [SceneTransform(), reversed] {
            for volumes in [[mesh, box], [box, mesh]] {
                let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 1), fogVolumes: volumes)
                let mean = average(try trace(tracer, lighting: lighting, samples: 1024, size: 4, camera: camera))
                #expect(simd_reduce_max(abs(mean - expected)) < 0.015, "\(mean) vs \(expected)")
            }
        }
    }

    @Test("Non-uniform, rotated and mirrored placements keep extinction per world distance",
          arguments: PathAccelerationTests.modes)
    func transformedMesh(acceleration: PathTracingScene.Options.Acceleration) throws {
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let axis = simd_normalize(SIMD3<Float>(1, 2, 3))
        let placements = [
            SceneTransform(
                translation: SIMD3(0.3, -0.2, -6), rotation: simd_quatf(angle: 0.7, axis: axis), scale: SIMD3(2.2, 1.4, 3.1)
            ),
            SceneTransform(
                translation: SIMD3(0, 0.1, -7), rotation: simd_quatf(angle: -1.1, axis: SIMD3(1, 0, 0)),
                scale: SIMD3(-2.5, 1.2, 2)
            ),
            SceneTransform(
                translation: SIMD3(0, 0, -6), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(2, 2, 2)
            ),
        ]
        for placement in placements {
            let fog = try PathTracer.Fog(
                extinction: SIMD3(0.25, 0.45, 0.15), albedo: .zero, meshBoundary: boundary, transform: placement
            )
            // The transform's own float inverse is an independent route to the chord.
            let chord = unitCubeChord(placement, origin: .zero, direction: SIMD3(0, 0, -1))
            #expect(chord > 1, "The ray must cross the placed cube: \(chord)")
            let lighting = try PathTracer.Lighting(environment: SIMD3(repeating: 1), fog: fog)
            let mean = average(try trace(tracer, lighting: lighting, samples: 1024, size: 4, height: 1e-3))
            let expected = fog.transmittance(distance: chord)
            #expect(simd_reduce_max(abs(mean - expected)) < 0.015, "\(mean) vs \(expected) over \(chord)")
        }
    }

    @Test("Single scattering in a rotated mesh volume matches the analytic radiance integral",
          arguments: PathAccelerationTests.modes)
    func meshScatteringRadiance(acceleration: PathTracingScene.Options.Acceleration) throws {
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        // A 2 by 2 square rotated 45 degrees about Z is a diamond with its corners at
        // distance sqrt(2) from the axis. The shadow ray to the sun leaves through the
        // upper corner. The camera ray crosses 2 units of depth.
        let fog = try PathTracer.Fog(
            extinction: SIMD3(0.25, 0.45, 0.15), albedo: SIMD3(0.85, 0.5, 0.2), anisotropy: 0.4,
            meshBoundary: boundary,
            transform: .init(
                translation: SIMD3(0, 0, -5), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(2, 2, 2)
            )
        )
        let intensity: Float = 5
        let sun = try PathTracer.AnalyticLight(
            PunctualLight(kind: .directional, intensity: intensity), direction: SIMD3(0, -1, 0)
        )
        let lighting = try PathTracer.Lighting(analyticLights: [sun], backgroundVisible: false, fog: fog)
        let tracer = try makeTracer(acceleration, nodes: [dummy()])
        let measured = try trace(tracer, lighting: lighting, samples: 2048, size: 4, height: 1e-4)
            .reduce(SIMD4<Float>.zero, +) / 16
        func phase(_ g: Float) -> Float { (1 - g * g) / (4 * .pi * pow(1 + g * g, 1.5)) }
        let shadowDistance = Float(2).squareRoot()
        var expected = SIMD3<Float>.zero
        var transmittance = SIMD3<Float>.zero
        for channel in 0..<3 {
            let sigma = fog.extinction[channel]
            // The integral of sigma_s * phase * exp(-sigma s) over the 2-unit camera segment,
            // times the sun's transmittance along the shadow ray.
            expected[channel] = intensity * fog.albedo[channel] * phase(fog.anisotropy)
                * exp(-sigma * shadowDistance) * (1 - exp(-2 * sigma))
            transmittance[channel] = exp(-2 * sigma)
        }
        #expect(simd_reduce_max(abs(SIMD3(measured.x, measured.y, measured.z) - expected)) < 0.008,
                "\(measured) vs \(expected)")
        let coverage = 1 - (transmittance.x + transmittance.y + transmittance.z) / 3
        #expect(abs(measured.w - coverage) < 0.018, "\(measured.w) vs \(coverage)")
    }

    @Test("A shadow ray tracks exactly the crossing limit and treats the rest as outside the region",
          arguments: PathAccelerationTests.modes)
    func crossingLimit(acceleration: PathTracingScene.Options.Acceleration) throws {
        let receiver = try fixture.pbr(.plane(width: 2, height: 2))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: receiver)
        let tracer = try makeTracer(acceleration, nodes: [node])
        let sun = try PathTracer.AnalyticLight(
            PunctualLight(kind: .directional, intensity: 4), direction: SIMD3(-1, 0, -1)
        )
        // 200 thin slabs stand across the 45-degree path from the receiver to the sun. Each
        // slab is 0.04 thick in X, so the path spends 0.04 * sqrt(2) inside it. The camera
        // ray never enters the region, and the shadow ray is deterministic.
        let count = 200
        let slab = MeshGeometry.cube()
        let parts = (0..<count).map { index in
            MeshGeometry.MergePart(slab, transform: SceneTransform(
                translation: SIMD3(0.5 + 0.13 * Float(index), 0, 0), scale: SIMD3(0.04, 4, 60)
            ).matrix)
        }
        let boundary = try PathTracer.Fog.MeshBoundary(
            device: fixture.device, geometry: MeshGeometry.merged(parts).geometry
        )
        #expect(boundary.shellCount == count && boundary.triangleCount == 12 * count)
        let fog = try PathTracer.Fog(extinction: SIMD3(0.05, 0.1, 0.15), albedo: .zero, meshBoundary: boundary)
        let clear = try #require(trace(tracer, lighting: .init(analyticLights: [sun]), samples: 16, height: 1e-4).first)
        let attenuated = try #require(trace(
            tracer, lighting: .init(analyticLights: [sun], fog: fog), samples: 16, height: 1e-4
        ).first)
        let measured = SIMD3(attenuated.x / clear.x, attenuated.y / clear.y, attenuated.z / clear.z)
        // Entering and leaving a slab are two crossings, so the limit tracks half as many slabs.
        let chord = 0.04 * Float(2).squareRoot()
        let expected = fog.transmittance(distance: Float(PathTracer.Fog.MeshBoundary.maximumRayCrossings / 2) * chord)
        let unlimited = fog.transmittance(distance: Float(count) * chord)
        #expect(simd_reduce_max(abs(measured - expected)) < 0.002, "\(measured) vs \(expected), unlimited \(unlimited)")
        #expect(simd_reduce_min(expected - unlimited) > 0.1)
    }

    @Test("Fog boundaries accept cavities and islands and reject nested, inverted and mixed shells")
    func shellValidation() throws {
        let outer = try cube(6)
        let cavity = try inverted(cube(4))
        let island = try cube(2)
        func boundary(_ parts: [MeshGeometry]) throws -> PathTracer.Fog.MeshBoundary {
            try .init(device: fixture.device, geometry: combined(parts))
        }
        #expect(try boundary([outer]).shellCount == 1)
        #expect(try boundary([outer, cavity]).shellCount == 2)
        #expect(try boundary([outer, cavity, island]).shellCount == 3)
        #expect(try boundary([outer, cube(2, at: SIMD3(10, 0, 0))]).shellCount == 2)
        let rejected: [(String, [MeshGeometry])] = [
            ("a shell nested in a shell with the same orientation", [outer, try cube(4)]),
            ("an inverted shell outside every outward shell", [outer, try inverted(cube(2, at: SIMD3(10, 0, 0)))]),
            ("an inverted shell alone", [try inverted(outer)]),
            ("a cavity inside a cavity", [outer, cavity, try inverted(cube(2))]),
            ("two outward shells that overlap", [try cube(4), try cube(4, at: SIMD3(2, 0, 0))]),
        ]
        for (name, parts) in rejected {
            #expect(throws: PathTracer.Failure.invalidFogBoundary, "\(name)") { try boundary(parts) }
        }
        // One flipped face breaks the edge pairing inside a connected shell.
        var flipped = MeshGeometry.cube().indices
        flipped.swapAt(1, 2)
        let mixed = try MeshGeometry(vertices: MeshGeometry.cube().vertices, indices: flipped)
        #expect(throws: PathTracer.Failure.invalidFogBoundary) { try boundary([mixed]) }
    }

    @Test("Duplicated corners that differ by rounding noise weld, and real gaps stay open")
    func weldTolerance() throws {
        let cube = MeshGeometry.cube()
        func perturbed(_ amount: Float) throws -> MeshGeometry {
            let vertices = cube.vertices.enumerated().map { index, vertex in
                let noise = SIMD3(Float(index % 3) - 1, Float(index / 3 % 3) - 1, Float(index % 5) - 2) * amount
                return MeshVertex(position: vertex.position + noise, normal: vertex.normal)
            }
            return try MeshGeometry(vertices: vertices, indices: cube.indices)
        }
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: perturbed(1e-7))
        #expect(boundary.triangleCount == 12 && boundary.shellCount == 1)
        #expect(throws: PathTracer.Failure.invalidFogBoundary) {
            try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: perturbed(1e-4))
        }
    }

    @Test("Placements reject singular, near-singular, non-finite and non-unit transforms")
    func placementValidation() throws {
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: .cube())
        func fog(_ transform: SceneTransform) throws -> PathTracer.Fog {
            try .init(extinction: SIMD3(repeating: 1), meshBoundary: boundary, transform: transform)
        }
        let invalid: [(String, SceneTransform)] = [
            ("a zero scale axis", .init(scale: SIMD3(1, 0, 1))),
            ("a scale below the minimum", .init(scale: SIMD3(1e-7, 1, 1))),
            ("a scale ratio above the limit", .init(scale: SIMD3(1e-3, 1e4, 1))),
            ("a NaN scale", .init(scale: SIMD3(.nan, 1, 1))),
            ("an infinite translation", .init(translation: SIMD3(.infinity, 0, 0))),
            ("a non-unit rotation", .init(rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 2))),
        ]
        for (name, transform) in invalid {
            #expect(throws: PathTracer.Failure.invalidFogTransform, "\(name)") { try fog(transform) }
        }
        let axis = simd_normalize(SIMD3<Float>(1, 2, 3))
        for transform in [
            SceneTransform(scale: SIMD3(1e-3, 1, 100)),
            SceneTransform(
                translation: SIMD3(3, -2, 5), rotation: simd_quatf(angle: 0.9, axis: axis), scale: SIMD3(-2.2, 1.4, 3.1)
            ),
        ] {
            let placed = try fog(transform)
            #expect(simd_almost_equal_elements(
                placed.meshLocalFromWorld * transform.matrix, matrix_identity_float4x4, 2e-5
            ), "The inverse must undo \(transform)")
        }
    }

    @Test("The runtime shader splice fails when the mesh BVH include is missing")
    func spliceRequiresInclude() throws {
        #expect(throws: MeshBVHShaderBuffer.Failure.shaderSource) {
            try BVHRayTriangleSource.spliceMeshBVH(into: "kernel void empty() {}")
        }
        let spliced = try BVHRayTriangleSource.spliceMeshBVH(into: "#include \"FrostfieldMeshBVH.h\"\nvoid after() {}")
        #expect(spliced.contains("frostfieldMeshBVHNearest") && !spliced.contains("#include \"Frostfield"))
    }

    @Test("Mesh fog rejects open and inverted boundaries and singular placements")
    func meshBoundaryValidation() throws {
        let cube = MeshGeometry.cube()
        let open = try MeshGeometry(vertices: cube.vertices, indices: Array(cube.indices.dropLast(3)))
        let inverted = try MeshGeometry(vertices: cube.vertices, indices: Array(cube.indices.reversed()))
        for geometry in [open, inverted] {
            #expect(throws: PathTracer.Failure.invalidFogBoundary) {
                try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: geometry)
            }
        }
        let boundary = try PathTracer.Fog.MeshBoundary(device: fixture.device, geometry: cube)
        #expect(boundary.device === fixture.device && boundary.triangleCount == 12)
        #expect(throws: PathTracer.Failure.invalidFogTransform) {
            try PathTracer.Fog(
                extinction: SIMD3(repeating: 1), meshBoundary: boundary, transform: .init(scale: SIMD3(1, 0, 1))
            )
        }
    }

    @Test("Density validation, immutable lighting identity and the Metal record layout")
    func validationAndHistory() throws {
        #expect(MemoryLayout<PathTracer.MediaUniforms>.stride == 160)
        #expect(MemoryLayout<PathTracer.MediaUniforms>.offset(of: \.fogResolution) == 64)
        #expect(MemoryLayout<PathTracer.MediaUniforms>.offset(of: \.fogDensityAddress) == 80)
        #expect(MemoryLayout<PathTracer.MediaUniforms>.offset(of: \.fogBoundaryAddress) == 88)
        #expect(MemoryLayout<PathTracer.MediaUniforms>.offset(of: \.fogLocalFromWorld) == 96)
        for size in [SIMD3(0, 1, 1), SIMD3(1, 129, 1), SIMD3(Int.max, 1, 1), SIMD3(2, 1, 1)] {
            #expect(throws: PathTracer.Failure.invalidFogDensityGrid) { try grid(size, [0.25]) }
        }
        for invalid: Float in [-0.01, 1.01, .nan, .infinity] {
            #expect(throws: PathTracer.Failure.invalidFogDensityGrid) { try grid(SIMD3(repeating: 1), [invalid]) }
        }
        let grid = try grid(SIMD3(repeating: 1), [0.25])
        #expect(grid.device === fixture.device && grid.resolution == SIMD3(repeating: 1))
        let replacement = try self.grid(grid.resolution, [0.75])
        let first = try PathTracer.Fog(density: 0.2, minimum: SIMD3(-1, -1, -5), maximum: SIMD3(1, 1, -3))
        func lighting(_ field: PathTracer.Fog.DensityGrid) throws -> PathTracer.Lighting {
            let second = try PathTracer.Fog(
                density: 1, minimum: SIMD3(-1, -1, -4), maximum: SIMD3(1, 1, -2), densityGrid: field
            )
            return try .init(fogVolumes: [first, second])
        }
        let initialLighting = try lighting(grid)
        let revisedLighting = try lighting(replacement)
        #expect(initialLighting.fog == revisedLighting.fog)
        #expect(initialLighting != revisedLighting)
        #expect(try PathTracer.Lighting(fog: first) == PathTracer.Lighting(fogVolumes: [first]))
        let packed = PathTracer.MediaUniforms.pack(initialLighting.fogVolumes)
        #expect(packed.count == 2 && packed[0].fogResolution.w == 2 && packed[1].fogDensityAddress != 0)
        #expect(PathTracer.MediaUniforms.pack([])[0].fogResolution == .zero)
        #expect(throws: PathTracer.Failure.fogVolumeLimit) {
            try PathTracer.Lighting(fogVolumes: Array(repeating: first, count: PathTracer.Lighting.maximumFogVolumes + 1))
        }
        let tracer = try makeTracer(.software, nodes: [dummy()])
        let target = try fixture.output(width: 1, height: 1)
        let accumulation = try tracer.makeAccumulation(width: 1, height: 1)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let initial = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: command, lighting: initialLighting
        )
        try fixture.finish(command)
        let next = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.lightingChanged) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: initial, commandBuffer: next, lighting: revisedLighting
            )
        }
    }

    private func average(_ image: [SIMD4<Float>]) -> SIMD3<Float> {
        image.reduce(SIMD3<Float>.zero) { $0 + SIMD3($1.x, $1.y, $1.z) } / Float(image.count)
    }

    private func cube(_ scale: Float, at translation: SIMD3<Float> = .zero) throws -> MeshGeometry {
        try MeshGeometry.cube().transformed(by: SceneTransform(translation: translation, scale: SIMD3(repeating: scale)).matrix)
    }

    private func inverted(_ geometry: MeshGeometry) throws -> MeshGeometry {
        let indices = stride(from: 0, to: geometry.indices.count, by: 3).flatMap { start in
            [geometry.indices[start], geometry.indices[start + 2], geometry.indices[start + 1]]
        }
        return try MeshGeometry(vertices: geometry.vertices, indices: indices)
    }

    private func combined(_ parts: [MeshGeometry]) throws -> MeshGeometry {
        try MeshGeometry.merged(parts.map { MeshGeometry.MergePart($0) }).geometry
    }

    /// The length in world units that a ray spends inside the unit cube placed by `transform`.
    private func unitCubeChord(_ transform: SceneTransform, origin: SIMD3<Float>, direction: SIMD3<Float>) -> Float {
        let inverse = transform.matrix.inverse
        let local = inverse * SIMD4(origin, 1)
        let step = inverse * SIMD4(direction, 0)
        var entry = -Float.infinity, exit = Float.infinity
        for axis in 0..<3 {
            if abs(step[axis]) < 1e-12 {
                if abs(local[axis]) > 0.5 { return 0 }
                continue
            }
            let a = (-0.5 - local[axis]) / step[axis], b = (0.5 - local[axis]) / step[axis]
            entry = max(entry, min(a, b))
            exit = min(exit, max(a, b))
        }
        return max(0, exit - entry)
    }

    private func grid(_ size: SIMD3<Int>, _ densities: [Float]) throws -> PathTracer.Fog.DensityGrid {
        try .init(device: fixture.device, resolution: size, densities: densities)
    }

    private func dummy() throws -> SceneNode {
        SceneNode(transform: .init(translation: SIMD3(20, 0, -5)), mesh: try fixture.pbr(.plane(width: 1, height: 1)))
    }

    private func makeTracer(_ acceleration: PathTracingScene.Options.Acceleration, nodes: [SceneNode]) throws -> PathTracer {
        let graph = SceneGraph(roots: nodes, camera: fixture.camera)
        let scene = try PathTracingScene(
            queue: fixture.queue, graph: graph, options: .init(textures: .baked(size: 16), acceleration: acceleration)
        )
        return try PathTracer(queue: fixture.queue, scene: scene)
    }

    private func trace(
        _ tracer: PathTracer, lighting: PathTracer.Lighting, samples: Int, size: Int = 1, height: Float = 0.04,
        camera: SceneTransform = .init(), bounces: Int = 1
    ) throws -> [SIMD4<Float>] {
        let target = try fixture.output(width: size, height: size)
        let accumulation = try tracer.makeAccumulation(width: size, height: size)
        let model = PathTracer.CameraModel(projection: .orthographic(height: height), near: 0.1, far: 30)
        var previous: PathTracer.History?
        for start in stride(from: 0, to: samples, by: 16) {
            let command = try #require(fixture.queue.makeCommandBuffer())
            previous = try tracer.encode(
                cameraModel: model, cameraTransform: camera, destination: target, accumulation: accumulation,
                previous: previous, commandBuffer: command, samplesPerPixel: min(16, samples - start),
                settings: .init(maxBounces: bounces, seed: 29), lighting: lighting
            )
            try fixture.finish(command)
        }
        return try fixture.readback(target)
    }
}
