import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathAnalyticLightTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    @Test("Analytic light records match Metal, inputs are validated and scene lights convert")
    func layoutValidationAndConversion() throws {
        #expect(MemoryLayout<PathTracer.PackedAnalyticLight>.stride == 96)
        #expect(MemoryLayout<PathTracer.PackedAnalyticLight>.alignment == 16)
        #expect(MemoryLayout<EnvironmentLookup>.stride == 48)
        #expect(MemoryLayout<EnvironmentLookup>.alignment == 16)
        #expect(MemoryLayout<PathTracer.Uniforms>.stride == 224)
        #expect(MemoryLayout<PathTracer.Uniforms>.alignment == 16)
        #expect(MemoryLayout<PathTracer.Uniforms>.offset(of: \.sky) == 128)
        #expect(MemoryLayout<PathTracer.Uniforms>.offset(of: \.skyInfo) == 176)
        #expect(MemoryLayout<PathTracer.Uniforms>.offset(of: \.lens) == 192)
        #expect(MemoryLayout<PathTracer.Uniforms>.offset(of: \.shift) == 208)

        let point = try PunctualLight(kind: .point(range: 4), intensity: 3)
        let sunLight = try PunctualLight(kind: .directional, color: SIMD3(1, 0.9, 0.8), intensity: 5)
        let spot = try PunctualLight(kind: .spot(innerConeAngle: 0.1, outerConeAngle: 0.4))
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) { try PathTracer.AnalyticLight(point, angularDiameter: 0.01) }
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) { try PathTracer.AnalyticLight(spot, direction: .zero) }
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) {
            try PathTracer.AnalyticLight(point, position: SIMD3(.nan, 0, 0))
        }
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) {
            try PathTracer.AnalyticLight(point, position: SIMD3(0, 2e12, 0))
        }
        #expect(throws: PathTracer.Failure.invalidAnalyticLight) { try PathTracer.AnalyticLight(sunLight, angularDiameter: 2) }
        let sun = try PathTracer.AnalyticLight(sunLight, direction: SIMD3(0, -2, 0), angularDiameter: 0.0093)
        #expect(sun.direction == SIMD3(0, -1, 0) && sun.angularDiameter == 0.0093)
        #expect(throws: PathTracer.Failure.invalidLighting) {
            try PathTracer.Lighting(analyticLights: Array(repeating: sun, count: 17))
        }
        #expect(try PathTracer.Lighting(analyticLights: Array(repeating: sun, count: 16)).analyticLights.count == 16)

        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 10)
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(1, 2, 3)), light: point),
            SceneNode(
                transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
                children: [SceneNode(light: sunLight), SceneNode(transform: .init(translation: SIMD3(0, 0, 1)), light: spot)]
            ),
        ], camera: camera)
        let lights = try PathTracer.AnalyticLight.lights(in: graph, directionalAngularDiameter: 0.02)
        try #require(lights.count == 3)
        #expect(lights[0].light == point && lights[0].position == SIMD3(1, 2, 3) && lights[0].angularDiameter == 0)
        #expect(lights[1].light == sunLight && lights[1].angularDiameter == 0.02)
        #expect(simd_distance(lights[1].direction, SIMD3(0, -1, 0)) < 1e-6)
        #expect(lights[2].light == spot && lights[2].angularDiameter == 0)
        #expect(simd_distance(lights[2].position, SIMD3(0, 1, 0)) < 1e-6)
    }

    @Test("A point light gives inverse-square diffuse radiance that matches the raster renderer")
    func pointLightMatchesRaster() throws {
        let width = 33
        let height = 25
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
        let albedo: Float = 0.8
        let receiver = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(try .init(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(albedo, albedo, albedo, 1), alphaMode: .opaque
            ))
        )
        let color = SIMD3<Float>(1, 0.5, 0.25)
        func graph(distance: Float, range: Float? = nil, intensity: Float = 6) throws -> SceneGraph {
            SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: receiver),
                SceneNode(
                    transform: .init(translation: SIMD3(0, 0, -3 + distance)),
                    light: try PunctualLight(kind: .point(range: range), color: color, intensity: intensity)
                ),
            ], camera: camera)
        }
        // Scene snapshots keep mesh nodes and skip light nodes, which reach the tracer as analytic lights.
        let reference = try graph(distance: 2)
        let scene = try PathTracingScene(queue: queue, graph: reference, textureSize: 16)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 19)
        let center = height / 2 * width + width / 2

        func traced(_ graph: SceneGraph) throws -> SIMD4<Float> {
            let lighting = try PathTracer.Lighting(analyticLights: PathTracer.AnalyticLight.lights(in: graph))
            return try trace(tracer, camera: camera, width: width, height: height, samples: 16, settings: settings, lighting: lighting)[center]
        }

        // At normal incidence the dielectric BSDF is (1 − 0.04) ρ / π plus GGX F D V = 0.04 × 0.25 / π.
        let brdf = ((1 - 0.04) * albedo + 0.01) / .pi
        let near = try traced(reference)
        let expected = SIMD4(brdf * color * 6 / 4, 1)
        #expect(simd_reduce_max(abs(near - expected) / expected) < 0.01, "\(near) vs \(expected)")
        let raster = try rasterize(reference, camera: camera, width: width, height: height)[center]
        #expect(simd_reduce_max(abs(near - raster) / raster) < 0.015, "\(near) vs \(raster)")

        let closer = try traced(graph(distance: 1))
        #expect(abs(closer.x / near.x - 4) < 0.06)
        // The smooth range window of pbrPunctual: 1 − (2 / 2.5)⁴ = 0.5904.
        let windowed = try traced(graph(distance: 2, range: 2.5))
        #expect(abs(windowed.x / near.x - 0.5904) < 0.006)
        #expect(try traced(graph(distance: 2, range: 1.9)) == SIMD4(0, 0, 0, 1))

        // Changing an analytic light invalidates accumulated history.
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try output(width: width, height: height)
        let buffer = try #require(queue.makeCommandBuffer())
        let first = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: buffer, settings: settings, lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: reference))
        )
        try finish(buffer)
        let next = try #require(queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.lightingChanged) {
            try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: first, commandBuffer: next, settings: settings,
                lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph(distance: 2, intensity: 7)))
            )
        }
    }

    @Test("Spot lights are dark outside the outer cone and follow the raster penumbra inside it")
    func spotConeMatchesRaster() throws {
        let side = 33
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 30)
        let receiver = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 8, height: 8),
            material: .physicallyBased(try .init(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(0.8, 0.8, 0.8, 1), alphaMode: .opaque
            ))
        )
        let inner = Float.pi / 12
        let outer = Float.pi / 4
        func graph(_ kind: PunctualLight.Kind, rotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))) throws -> SceneGraph {
            SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: receiver),
                SceneNode(
                    transform: .init(translation: SIMD3(0, 0, -1), rotation: rotation),
                    light: try PunctualLight(kind: kind, color: SIMD3(1, 0.8, 0.6), intensity: 6)
                ),
            ], camera: camera)
        }
        let spotGraph = try graph(.spot(innerConeAngle: inner, outerConeAngle: outer))
        let pointGraph = try graph(.point())
        let scene = try PathTracingScene(queue: queue, graph: spotGraph, textureSize: 16)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 4)
        // Delta lights draw no random numbers, so 256 jittered samples only average the pixel footprint.
        func traced(_ graph: SceneGraph) throws -> [SIMD4<Float>] {
            try trace(
                tracer, camera: camera, width: side, height: side, samples: 256, settings: settings,
                lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph))
            )
        }
        let spot = try traced(spotGraph)
        let point = try traced(pointGraph)
        let rasterSpot = try rasterize(spotGraph, camera: camera, width: side, height: side)
        let rasterPoint = try rasterize(pointGraph, camera: camera, width: side, height: side)

        let row = side / 2 * side
        #expect(spot[row + side / 2] == point[row + side / 2])
        var penumbra = 0
        var dark = 0
        for column in 0..<side {
            let index = row + column
            // The plane lies 2 units beyond the light; a pixel spans 6/33 units and tan(45°) = 1.
            let x = abs(((Float(column) + 0.5) / Float(side) * 2 - 1) * 3)
            if x - 3.0 / Float(side) > 2 {
                dark += 1
                #expect(spot[index] == SIMD4(0, 0, 0, 1))
                #expect(rasterSpot[index] == SIMD4(0, 0, 0, 1))
                #expect(point[index].x > 0.03)
                continue
            }
            // Both renderers use pbrPunctual's falloff: the squared, clamped cosine ramp.
            func falloff(_ radius: Float) -> Float {
                let cosine = 2 / sqrt(4 + radius * radius)
                let cone = simd_clamp((cosine - cos(outer)) / (cos(inner) - cos(outer)), 0, 1)
                return cone * cone
            }
            // The raster shades the pixel center. The tracer averages jittered samples over
            // the footprint, which differs near the clamped inner-cone kink.
            var footprint: Float = 0
            for sub in 0..<256 {
                let offset = (SIMD2(Float(sub % 16), Float(sub / 16)) + 0.5) / 16 - 0.5
                footprint += falloff(simd_length(SIMD2(x, 0) + offset * 6 / Float(side)))
            }
            footprint /= 256
            let traceRatio = spot[index].x / point[index].x
            let rasterRatio = rasterSpot[index].x / rasterPoint[index].x
            #expect(abs(rasterRatio - falloff(x)) < 0.01, "raster \(column): \(rasterRatio) vs \(falloff(x))")
            #expect(abs(traceRatio - footprint) < 0.01, "traced \(column): \(traceRatio) vs \(footprint)")
            if falloff(x) > 0.01, falloff(x) < 0.9 { penumbra += 1 }
        }
        #expect(dark >= 4 && penumbra >= 6)

        let away = try traced(graph(
            .spot(innerConeAngle: inner, outerConeAngle: outer), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        ))
        #expect(away.allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
    }

    @Test("A directional light with an angular diameter softens its shadow edge without losing energy")
    func directionalDiskSoftensShadow() throws {
        let side = 33
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 6, near: 0.1, far: 30)
        let ground = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 8, height: 8),
            material: .physicallyBased(try .init(roughness: 1, baseColorFactor: SIMD4(0.8, 0.8, 0.8, 1), alphaMode: .opaque))
        )
        let block = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .cube(),
            material: .physicallyBased(try .init(roughness: 1, alphaMode: .opaque))
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: ground),
            SceneNode(transform: .init(translation: SIMD3(0, 1, 0)), mesh: block),
        ], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let sunlight = try PunctualLight(kind: .directional, intensity: 3)
        let direction = SIMD3<Float>(-1, -1, 0)
        // Looking down at x = -2, where the block's top edge ends its shadow.
        let view = SceneTransform(translation: SIMD3(-2, 3, 0), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)))
        let settings = PathTracer.Settings(maxBounces: 1, seed: 31)

        func profile(_ diameter: Float) throws -> [Float] {
            let light = try PathTracer.AnalyticLight(sunlight, direction: direction, angularDiameter: diameter)
            let image = try trace(
                tracer, camera: camera, width: side, height: side, samples: 128, settings: settings,
                lighting: .init(analyticLights: [light]), cameraTransform: view
            )
            return (0..<side).map { image[side / 2 * side + $0].x }
        }

        func edgeWidth(_ values: [Float]) -> (lit: Float, shadow: Float, transition: Int) {
            // Columns 0–7 lie beyond x = -2.3 in light; columns 25–32 lie inside x > -1.7 in umbra.
            let lit = values[0..<8].reduce(0, +) / 8
            let shadow = values[25..<33].reduce(0, +) / 8
            let transition = values.filter { $0 > shadow + 0.1 * (lit - shadow) && $0 < shadow + 0.9 * (lit - shadow) }.count
            return (lit, shadow, transition)
        }

        let hard = edgeWidth(try profile(0))
        let soft = edgeWidth(try profile(0.2))
        // Irradiance 3 lux × cos 45°, the dielectric diffuse and rough GGX terms seen from above.
        let expected: Float = (0.96 * 0.8 / .pi + 0.04 / .pi * 0.5 / (0.707 + 1)) * 3 * 0.70710677
        #expect(abs(hard.lit / expected - 1) < 0.02, "\(hard.lit) vs \(expected)")
        #expect(abs(soft.lit / hard.lit - 1) < 0.02, "\(soft.lit) vs \(hard.lit)")
        #expect(hard.shadow == 0 && soft.shadow == 0)
        // The 0.2 rad disk spreads the edge over about 0.6 units, or 12 columns of 0.048 units.
        #expect(hard.transition <= 2, "\(hard.transition)")
        #expect(soft.transition >= 6, "\(soft.transition)")
    }

    private func trace(
        _ tracer: PathTracer, camera: PerspectiveCamera, width: Int, height: Int, samples: Int,
        settings: PathTracer.Settings, lighting: PathTracer.Lighting, cameraTransform: SceneTransform = .init()
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

    private func rasterize(_ graph: SceneGraph, camera: PerspectiveCamera, width: Int, height: Int) throws -> [SIMD4<Float>] {
        let color = try output(width: width, height: height, usage: [.renderTarget, .shaderRead])
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        try finish(buffer)
        return try readback(color)
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
