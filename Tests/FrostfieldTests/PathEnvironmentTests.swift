import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathEnvironmentTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 40)
    }

    @Test("Environment sampling tables integrate to one, favor a bright texel and follow yaw")
    func samplingDistribution() throws {
        let width = 64
        let height = 32
        var pixels = Array(repeating: SIMD3<Float>(repeating: 0.01), count: width * height)
        let brightRow = 10
        let brightColumn = 40
        pixels[brightRow * width + brightColumn] = SIMD3(1_000, 900, 800)
        let map = try PathTracer.EnvironmentMap(device: device, pixels: pixels, width: width, height: height)
        #expect(map.isSampled && map.width == 64 && map.height == 32)

        // A grid aligned with the texels integrates the piecewise-constant density exactly.
        let cellsU = width * 4
        let cellsV = height * 4
        var integral = 0.0
        for row in 0..<cellsV {
            for column in 0..<cellsU {
                let theta = (Double(row) + 0.5) / Double(cellsV) * .pi
                let phi = ((Double(column) + 0.5) / Double(cellsU) - 0.5) * 2 * .pi
                let direction = SIMD3<Float>(SIMD3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi)))
                let solidAngle = (2 * .pi / Double(cellsU)) * (.pi / Double(cellsV)) * sin(theta)
                integral += Double(map.pdf(direction: direction)) * solidAngle
            }
        }
        #expect(abs(integral - 1) < 1e-3)

        // Luminance × sin θ gives the texel's filtered footprint about 98.3% of the
        // weight. The texel itself keeps the bilinear average's (3/4)² share of that.
        let side = 64
        var inside = 0
        var neighborhood = 0
        var consistent = 0
        for index in 0..<(side * side) {
            let random = (SIMD2<Float>(Float(index % side), Float(index / side)) + 0.5) / Float(side)
            let sample = try #require(map.sample(random))
            let local = sample.direction
            let u = atan2(local.z, local.x) / (2 * .pi) + 0.5
            let v = acos(simd_clamp(local.y, -1, 1)) / .pi
            let offset = SIMD2(Int(u * Float(width)) - brightColumn, Int(v * Float(height)) - brightRow)
            if offset == .zero { inside += 1 }
            if abs(offset.x) <= 1, abs(offset.y) <= 1 { neighborhood += 1 }
            let evaluated = map.pdf(direction: sample.direction)
            if abs(evaluated - sample.pdf) <= 1e-3 * sample.pdf { consistent += 1 }

            let yaw: Float = .pi / 2
            let turned = try #require(map.sample(random, yaw: yaw))
            let expected = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)).act(sample.direction)
            #expect(simd_distance(turned.direction, expected) < 1e-5)
            #expect(abs(turned.pdf - sample.pdf) <= 1e-5 * sample.pdf)
        }
        #expect(Double(neighborhood) / Double(side * side) > 0.97)
        #expect(abs(Double(inside) / Double(side * side) - 0.553) < 0.02)
        #expect(consistent >= side * side - side * side / 100)

        // A uniform map is close to the uniform sphere density 1/(4π) at row centers.
        let uniform = try PathTracer.EnvironmentMap(
            device: device, pixels: Array(repeating: SIMD3(repeating: 0.5), count: width * height),
            width: width, height: height
        )
        let equator = Float(height / 2 - 1) + 0.5
        let theta = equator / Float(height) * .pi
        let pdf = uniform.pdf(direction: SIMD3(sin(theta), cos(theta), 0))
        #expect(abs(pdf * 4 * .pi - 1) < 2e-3)

        let black = try PathTracer.EnvironmentMap(
            device: device, pixels: Array(repeating: .zero, count: 8), width: 4, height: 2
        )
        #expect(!black.isSampled && black.sample(SIMD2(0.5, 0.5)) == nil)
        #expect(black.pdf(direction: SIMD3(1, 0, 0)) == 0)
    }

    @Test("Environment maps validate dimensions and values and decode Radiance panoramas")
    func mapValidation() throws {
        let invalid: [([SIMD3<Float>], Int, Int)] = [
            (Array(repeating: SIMD3(repeating: 1), count: 16), 4, 4),
            (Array(repeating: SIMD3(repeating: 1), count: 7), 4, 2),
            ([SIMD3(1, .nan, 1)] + Array(repeating: SIMD3(repeating: 1), count: 7), 4, 2),
            ([SIMD3(1, -0.5, 1)] + Array(repeating: SIMD3(repeating: 1), count: 7), 4, 2),
            ([SIMD3(1, 70_000, 1)] + Array(repeating: SIMD3(repeating: 1), count: 7), 4, 2),
            ([], 0, 0),
        ]
        for (pixels, width, height) in invalid {
            #expect(throws: PathTracer.Failure.invalidEnvironmentMap) {
                try PathTracer.EnvironmentMap(device: device, pixels: pixels, width: width, height: height)
            }
        }

        var source = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".utf8)
        for _ in 0..<8 { source.append(contentsOf: [160, 64, 32, 130]) }
        let decoded = try PathTracer.EnvironmentMap(device: device, radianceHDR: source)
        #expect(decoded.width == 4 && decoded.height == 2 && decoded.isSampled)
        let map = try PathTracer.EnvironmentMap(
            device: device, pixels: Array(repeating: SIMD3(repeating: 1), count: 8), width: 4, height: 2
        )
        #expect(map == map && map != decoded)
        #expect(throws: PathTracer.Failure.invalidLighting) {
            try PathTracer.Lighting(sky: .map(map, intensity: -1))
        }
        #expect(throws: PathTracer.Failure.invalidLighting) {
            try PathTracer.Lighting(sky: .map(map, intensity: 70_000))
        }
        #expect(throws: PathTracer.Failure.invalidLighting) {
            try PathTracer.Lighting(sky: .map(map, yaw: .infinity))
        }
        let lighting = try PathTracer.Lighting(sky: .map(map, intensity: 2, yaw: 1))
        #expect(lighting.environment == .zero && lighting.backgroundVisible)
        #expect(try PathTracer.Lighting(environment: SIMD3(1, 2, 3)).sky == .constant(SIMD3(1, 2, 3)))
    }

    @Test("A white sphere under a uniform map converges to the constant-environment result")
    func furnace() throws {
        let width = 33
        let height = 25
        let sphere = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .uvSphere(radius: 1, segments: 64, rings: 32),
            material: .physicallyBased(try .init(roughness: 1, baseColorFactor: SIMD4(repeating: 1), alphaMode: .opaque))
        )
        let center = SIMD3<Float>(0, 0, -4)
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: center), mesh: sphere)], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let radiance: Float = 0.5
        let map = try PathTracer.EnvironmentMap(
            device: device, pixels: Array(repeating: SIMD3(repeating: radiance), count: 64 * 32), width: 64, height: 32
        )
        let constant = try PathTracer.Lighting(environment: SIMD3(repeating: radiance))
        let mapped = try PathTracer.Lighting(sky: .map(map, yaw: 0.7))
        let tracer = try PathTracer(queue: queue, scene: scene, lighting: constant)
        let settings = PathTracer.Settings(maxBounces: 3, seed: 2_024)
        let reference = try render(tracer, width: width, height: height, samples: 256, settings: settings, lighting: constant)
        let estimate = try render(tracer, width: width, height: height, samples: 256, settings: settings, lighting: mapped)

        var interior: [(Float, Float)] = []
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let miss = closestApproach(x: x, y: y, width: width, height: height, center: center)
                if miss < 0.85 {
                    interior.append((reference[index].x, estimate[index].x))
                    // The difference's per-pixel standard error is about 0.011 at 256 samples.
                    #expect(abs(reference[index].x - estimate[index].x) < 0.06)
                } else if miss > 1.15 {
                    #expect(reference[index] == SIMD4(radiance, radiance, radiance, 1))
                    #expect(estimate[index] == SIMD4(radiance, radiance, radiance, 1))
                }
            }
        }
        try #require(interior.count > 40)
        let referenceMean = interior.map(\.0).reduce(0, +) / Float(interior.count)
        let estimateMean = interior.map(\.1).reduce(0, +) / Float(interior.count)
        #expect(referenceMean > 0.4 && referenceMean < 0.52)
        #expect(abs(estimateMean / referenceMean - 1) < 0.01)
    }

    @Test("A bright sky patch casts a shadow that turns with the raster yaw convention")
    func sunShadowFollowsYaw() throws {
        let width = 33
        let height = 33
        let ground = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 8, height: 8),
            material: .physicallyBased(try .init(roughness: 1, baseColorFactor: SIMD4(0.8, 0.8, 0.8, 1), alphaMode: .opaque))
        )
        let block = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .cube(),
            material: .physicallyBased(try .init(roughness: 1, baseColorFactor: SIMD4(0.5, 0.5, 0.5, 1), alphaMode: .opaque))
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: ground),
            SceneNode(transform: .init(translation: SIMD3(0, 1, 0)), mesh: block),
        ], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)

        // A 2 × 2 patch centered on the panorama direction (1, 1, 0) / √2: U 0.5, V 0.25.
        let mapWidth = 128
        let mapHeight = 64
        var panorama = Array(repeating: SIMD3<Float>.zero, count: mapWidth * mapHeight)
        var irradiance: Float = 0
        for row in 15...16 {
            for column in 63...64 {
                panorama[row * mapWidth + column] = SIMD3(repeating: 2_000)
                let theta = (Float(row) + 0.5) / Float(mapHeight) * .pi
                let solidAngle = (2 * .pi / Float(mapWidth)) * (.pi / Float(mapHeight)) * sin(theta)
                irradiance += 2_000 * solidAngle * cos(theta)
            }
        }
        // Diffuse (1 − F) ρ / π plus the rough GGX term seen from above; about 9.6 × 0.248.
        let expected = (0.96 * 0.8 / .pi + 0.04 / .pi * 0.5 / (0.707 + 0.98)) * irradiance
        let map = try PathTracer.EnvironmentMap(device: device, pixels: panorama, width: mapWidth, height: mapHeight)
        let view = SceneTransform(translation: SIMD3(0, 5, 0), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)))
        let tracer = try PathTracer(queue: queue, scene: scene, lighting: .init(sky: .map(map)))
        let settings = PathTracer.Settings(maxBounces: 2, seed: 77)

        /// Averages the 3 × 3 pixels around a ground point. At 256 samples per pixel the
        /// average's noise is near 1.5%; a 1,024-sample run converged within 1% of the
        /// expectation, plus up to 5% light bounced from a sunlit block face.
        func radiance(at point: SIMD3<Float>, in image: [SIMD4<Float>]) -> Float {
            // The camera looks down -Y with +X to the right and -Z up the image.
            let tangent = Float(tan(Double.pi / 6))
            let screen = SIMD2(point.x / 5 / tangent, -point.z / 5 / tangent)
            let x = Int(((screen.x + 1) / 2 * Float(width)).rounded(.down))
            let y = Int(((1 - screen.y) / 2 * Float(height)).rounded(.down))
            var sum: Float = 0
            for dy in -1...1 {
                for dx in -1...1 { sum += image[(y + dy) * width + x + dx].x }
            }
            return sum / 9
        }

        // The unit block spans y 0.5–1.5; from above it hides the ground within |x|, |z| < 0.72.
        let east = SIMD3<Float>(1.2, 0, 0)
        let west = SIMD3<Float>(-1.2, 0, 0)
        let north = SIMD3<Float>(0, 0, -1.2)
        let south = SIMD3<Float>(0, 0, 1.2)
        let unturned = try render(
            tracer, width: width, height: height, samples: 256, settings: settings,
            lighting: .init(sky: .map(map)), cameraTransform: view
        )
        // With the sun toward +X at 45° elevation, the block's umbra covers x −2…0, |z| < 0.5.
        #expect(radiance(at: west, in: unturned) < 0.05 * expected)
        for point in [east, north, south] {
            let lit = radiance(at: point, in: unturned)
            #expect(lit > 0.94 * expected && lit < 1.12 * expected, "\(point): \(lit) vs \(expected)")
        }

        // A +90° yaw turns the panorama's +X toward world -Z, so the shadow moves to +Z.
        let turned = try render(
            tracer, width: width, height: height, samples: 256, settings: settings,
            lighting: .init(sky: .map(map, yaw: .pi / 2)), cameraTransform: view
        )
        #expect(radiance(at: south, in: turned) < 0.05 * expected)
        for point in [north, east, west] {
            let lit = radiance(at: point, in: turned)
            #expect(lit > 0.94 * expected && lit < 1.12 * expected, "\(point): \(lit) vs \(expected)")
        }
    }

    @Test("A camera-visible map matches the raster environment background at two yaws")
    func backgroundMatchesRaster() throws {
        let width = 32
        let height = 24
        let mapWidth = 128
        let mapHeight = 64
        var pixels: [SIMD3<Float>] = []
        for row in 0..<mapHeight {
            for column in 0..<mapWidth {
                let theta = (Float(row) + 0.5) / Float(mapHeight) * .pi
                let phi = ((Float(column) + 0.5) / Float(mapWidth) - 0.5) * 2 * .pi
                let local = SIMD3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi))
                pixels.append(0.5 + 0.5 * local)
            }
        }
        let map = try PathTracer.EnvironmentMap(device: device, pixels: pixels, width: mapWidth, height: mapHeight)
        let panorama = try halfTexture(pixels, width: mapWidth, height: mapHeight)
        let environment = try EnvironmentLighting(
            queue: queue, equirectangular: panorama,
            settings: .init(cubeSize: 64, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        )
        let background = try EnvironmentBackground(lighting: environment)

        // Path scenes need geometry, so a small panel sits behind the camera.
        let panel = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.1, height: 0.1),
            material: .physicallyBased(try .init(alphaMode: .opaque))
        )
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, 5)), mesh: panel)], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let view = SceneTransform(rotation: simd_quatf(angle: 0.3, axis: SIMD3(1, 0, 0)))

        var images: [[SIMD4<Float>]] = []
        for yaw in [Float(0), .pi / 2] {
            let traced = try render(
                tracer, width: width, height: height, samples: 16, settings: .init(maxBounces: 1, seed: 5),
                lighting: .init(sky: .map(map, yaw: yaw)), cameraTransform: view
            )
            let target = try output(width: width, height: height, usage: [.renderTarget, .shaderRead])
            let buffer = try #require(queue.makeCommandBuffer())
            try background.encode(
                destination: target, camera: camera, cameraTransform: view, yaw: yaw, commandBuffer: buffer
            )
            try finish(buffer)
            let raster = try readback(target)
            for (path, reference) in zip(traced, raster) {
                #expect(simd_reduce_max(abs(path - reference)) < 0.03)
            }
            images.append(traced)
        }
        // Looking down -Z, yaw 0 samples panorama -Z and +90° samples panorama +X.
        let center = SIMD4<Float>(0.5 * (images[0][11 * width + 15] + images[0][12 * width + 16]))
        let turned = SIMD4<Float>(0.5 * (images[1][11 * width + 15] + images[1][12 * width + 16]))
        let pitch = sin(Float(0.3)) * 0.5
        #expect(simd_reduce_max(abs(center - SIMD4(0.5, 0.5 + pitch, 0.5 - 0.5 * cos(0.3), 1))) < 0.03)
        #expect(simd_reduce_max(abs(turned - SIMD4(0.5 + 0.5 * cos(0.3), 0.5 + pitch, 0.5, 1))) < 0.03)
    }

    @Test("Hidden backgrounds leave transparent misses while surfaces keep environment light")
    func hiddenBackground() throws {
        let width = 33
        let height = 25
        let sphere = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .uvSphere(radius: 1, segments: 48, rings: 24),
            material: .physicallyBased(try .init(roughness: 0.6, baseColorFactor: SIMD4(0.9, 0.6, 0.3, 1), alphaMode: .opaque))
        )
        let center = SIMD3<Float>(0, 0, -4)
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: center), mesh: sphere)], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let map = try PathTracer.EnvironmentMap(
            device: device, pixels: (0..<(64 * 32)).map { SIMD3(Float($0 % 64) / 32, 0.4, 0.8) }, width: 64, height: 32
        )
        let tracer = try PathTracer(queue: queue, scene: scene)
        let settings = PathTracer.Settings(maxBounces: 3, seed: 11)
        for sky in [PathTracer.Environment.map(map, intensity: 1.5), .constant(SIMD3(0.2, 0.3, 0.4))] {
            let shown = try render(
                tracer, width: width, height: height, samples: 8, settings: settings, lighting: .init(sky: sky)
            )
            let hidden = try render(
                tracer, width: width, height: height, samples: 8, settings: settings,
                lighting: .init(sky: sky, backgroundVisible: false)
            )
            var partial = 0
            for y in 0..<height {
                for x in 0..<width {
                    let index = y * width + x
                    let miss = closestApproach(x: x, y: y, width: width, height: height, center: center)
                    #expect(shown[index].w == 1)
                    if miss < 0.85 {
                        // Hits use the same random sequence, so the surface radiance is identical.
                        #expect(hidden[index] == shown[index])
                    } else if miss > 1.15 {
                        #expect(hidden[index] == .zero)
                        #expect(shown[index].x + shown[index].y + shown[index].z > 0.1)
                    } else if hidden[index].w > 0, hidden[index].w < 1 {
                        partial += 1
                    }
                }
            }
            #expect(partial > 0)
        }
    }

    @Test("Environment intensity, yaw, visibility and map identity changes reject old history")
    func historyRejectsEnvironmentChanges() throws {
        let width = 9
        let height = 7
        let plane = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1, height: 1),
            material: .physicallyBased(try .init(roughness: 0.7, alphaMode: .opaque))
        )
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: plane)], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let pixels = Array(repeating: SIMD3<Float>(0.25, 0.5, 1), count: 32)
        let map = try PathTracer.EnvironmentMap(device: device, pixels: pixels, width: 8, height: 4)
        let twin = try PathTracer.EnvironmentMap(device: device, pixels: pixels, width: 8, height: 4)
        let base = try PathTracer.Lighting(sky: .map(map, intensity: 1, yaw: 0.25))
        let tracer = try PathTracer(queue: queue, scene: scene, lighting: base)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try output(width: width, height: height)
        let settings = PathTracer.Settings(maxBounces: 2, seed: 3)

        let firstBuffer = try #require(queue.makeCommandBuffer())
        let first = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: firstBuffer, settings: settings
        )
        try finish(firstBuffer)
        #expect(try readback(target)[0] == SIMD4(0.25, 0.5, 1, 1))

        let secondBuffer = try #require(queue.makeCommandBuffer())
        let changes: [PathTracer.Lighting] = try [
            .init(sky: .map(map, intensity: 2, yaw: 0.25)),
            .init(sky: .map(map, intensity: 1, yaw: 0.5)),
            .init(sky: .map(twin, intensity: 1, yaw: 0.25)),
            .init(sky: .map(map, intensity: 1, yaw: 0.25), backgroundVisible: false),
            .init(environment: SIMD3(0.25, 0.5, 1)),
        ]
        for change in changes {
            #expect(change != base)
            #expect(throws: PathTracer.Failure.lightingChanged) {
                try tracer.encode(
                    camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                    previous: first, commandBuffer: secondBuffer, settings: settings, lighting: change
                )
            }
        }
        let same = try PathTracer.Lighting(sky: .map(map, intensity: 1, yaw: 0.25))
        let second = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            previous: first, commandBuffer: secondBuffer, settings: settings, lighting: same
        )
        try finish(secondBuffer)
        #expect(second.sampleCount == 2)

        let thirdBuffer = try #require(queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: thirdBuffer, settings: settings, lighting: changes[0]
        )
        try finish(thirdBuffer)
        #expect(try readback(target)[0] == SIMD4(0.5, 1, 2, 1))
    }

    /// The distance, in sphere radii, by which the center ray of a pixel misses the sphere center.
    private func closestApproach(x: Int, y: Int, width: Int, height: Int, center: SIMD3<Float>) -> Float {
        let tangentY = Float(tan(Double.pi / 6))
        let tangentX = tangentY * Float(width) / Float(height)
        let screen = SIMD2(
            (Float(x) + 0.5) / Float(width) * 2 - 1,
            1 - (Float(y) + 0.5) / Float(height) * 2
        )
        let direction = simd_normalize(SIMD3(screen.x * tangentX, screen.y * tangentY, -1))
        return simd_length(center - direction * simd_dot(center, direction))
    }

    private func render(
        _ tracer: PathTracer, width: Int, height: Int, samples: Int, settings: PathTracer.Settings,
        lighting: PathTracer.Lighting, cameraTransform: SceneTransform = .init()
    ) throws -> [SIMD4<Float>] {
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try output(width: width, height: height)
        var history: PathTracer.History?
        while (history?.sampleCount ?? 0) < samples {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: camera, cameraTransform: cameraTransform, destination: target, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: min(16, samples - (history?.sampleCount ?? 0)),
                settings: settings, lighting: lighting
            )
            try finish(buffer)
        }
        return try readback(target)
    }

    private func halfTexture(_ pixels: [SIMD3<Float>], width: Int, height: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let half = pixels.map { SIMD4<Float16>(SIMD4($0, 1)) }
        half.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: width * 8
            )
        }
        return texture
    }

    private func output(
        width: Int, height: Int, usage: MTLTextureUsage = [.shaderWrite, .shaderRead]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = usage
        descriptor.hazardTrackingMode = .tracked
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func finish(_ buffer: any MTLCommandBuffer) throws {
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "No Metal error")")
    }

    private func readback(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var result = Array(repeating: SIMD4<Float16>.zero, count: texture.width * texture.height)
        try result.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 8,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return result.map(SIMD4<Float>.init)
    }
}
