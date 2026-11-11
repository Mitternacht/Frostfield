import Frostfield
import Metal
import Testing
import simd

@Test("Aimed path lights preserve their dimensions, orientation and total emitted power")
func aimedAreaLight() throws {
    let light = try PathTracer.AreaLight(
        center: SIMD3(0, 2, 0), aim: .zero, size: SIMD2(2, 4),
        radiantPower: SIMD3(8 * .pi, 16 * .pi, 0), cameraVisible: false
    )
    #expect(simd_length(light.halfU) == 1 && simd_length(light.halfV) == 2)
    #expect(simd_normalize(simd_cross(light.halfU, light.halfV)) == SIMD3(0, -1, 0))
    #expect(simd_reduce_max(abs(light.radiance - SIMD3(1, 2, 0))) < 1e-6)
    #expect(!light.cameraVisible)
    #expect(throws: PathTracer.Failure.invalidLighting) {
        try PathTracer.Lighting(lights: Array(repeating: light, count: 17))
    }
    #expect(throws: PathTracer.Failure.invalidLighting) {
        try PathTracer.Lighting(environment: SIMD3(1, .nan, 0))
    }
    #expect(throws: PathTracer.Failure.invalidAreaLight) {
        try PathTracer.AreaLight(center: .zero, aim: .zero, size: SIMD2(2, 4), radiance: SIMD3(repeating: 1))
    }
    #expect(throws: PathTracer.Failure.invalidAreaLight) {
        try PathTracer.AreaLight(center: SIMD3(0, 2, 0), aim: .zero, size: SIMD2(2, 4),
                                 radiantPower: SIMD3(1, -.leastNonzeroMagnitude, 0))
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathTracerTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let width = 33
    private let height = 25

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
    }

    @Test("Two mirror bounces reveal an emissive surface outside the camera view")
    func mirrorsRevealOffCameraEmitter() throws {
        let mirror = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(try .init(metallic: 1, roughness: 0.025, alphaMode: .opaque))
        )
        let emitter = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1.5, height: 1.5),
            material: .physicallyBased(try .init(emissiveFactor: SIMD3(1, 0.2, 0.05), alphaMode: .opaque))
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(
                translation: SIMD3(0, 0, -3), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 1, 0))
            ), mesh: mirror),
            SceneNode(transform: .init(
                translation: SIMD3(2, 0, -3), rotation: simd_quatf(angle: -.pi / 4, axis: SIMD3(0, 1, 0))
            ), mesh: mirror),
            SceneNode(transform: .init(
                translation: SIMD3(2, 0, -1), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
            ), mesh: emitter),
        ], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let accumulator = try tracer.makeAccumulation(width: width, height: height)
        let target = try output()
        for bounces in [1, 2] {
            let buffer = try #require(queue.makeCommandBuffer())
            _ = try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulator,
                commandBuffer: buffer, samplesPerPixel: 8, settings: .init(maxBounces: bounces, seed: 8)
            )
            try finish(buffer)
            let center = try pixels(in: target)[height / 2 * width + width / 2]
            if bounces == 1 {
                #expect(center.x == 0 && center.y == 0 && center.z == 0)
            } else {
                #expect(center.x > 0.25)
                #expect(center.y > 0.03 && center.y < center.x)
            }
            #expect(center.w == 1)
        }
    }

    @Test("One material varies from opaque to clear across a path-traced surface")
    func graphMasksTransmission() throws {
        let surface = try SurfaceShaderGraph(
            nodes: [.uv(.u), .scalar(0.45), .scalar(0.55), .smoothstep(edge0: 1, edge1: 2, value: 0)],
            outputs: .init(transmissionMask: 3)
        )
        let ice = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(try .init(
                roughness: 0, alphaMode: .opaque, transmission: .init(factor: 1, indexOfRefraction: 1), surfaceGraph: surface
            ))
        )
        let emitter = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 6, height: 6),
            material: .physicallyBased(try .init(emissiveFactor: SIMD3(1, 0.2, 0.05), alphaMode: .opaque))
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: ice),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: emitter),
        ], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let accumulator = try tracer.makeAccumulation(width: width, height: height)
        let target = try output()
        let buffer = try #require(queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulator,
            commandBuffer: buffer, samplesPerPixel: 8, settings: .init(maxBounces: 1, seed: 8)
        )
        try finish(buffer)
        let image = try pixels(in: target)
        let row = height / 2 * width
        #expect(image[row + width / 3].x == 0)
        #expect(image[row + width * 2 / 3].x > 0.9)
    }

    @Test("Split sample batches match a fresh render and enforce accumulation ownership")
    func accumulationAndGuides() throws {
        let material = try PhysicallyBasedMaterial(
            roughness: 0.63, baseColorFactor: SIMD4(0.65, 0.35, 0.16, 1), alphaMode: .opaque
        )
        let plane = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(material)
        )
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: plane)], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let light = try PathTracer.AreaLight(
            center: SIMD3(0, 0.75, -1.6), halfU: SIMD3(0.65, 0, 0), halfV: SIMD3(0, -0.5, 0),
            radiance: SIMD3(12, 11, 10), cameraVisible: false
        )
        let tracer = try PathTracer(queue: queue, scene: scene, lights: [light], environment: SIMD3(0.04, 0.08, 0.16))
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try output()
        let normalDepth = try output()
        let albedoRoughness = try output()
        let guides = PathTracer.Guides(normalDepth: normalDepth, albedoRoughness: albedoRoughness)
        let settings = PathTracer.Settings(maxBounces: 2, seed: 903)
        let firstBuffer = try #require(queue.makeCommandBuffer())
        let first = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            guides: guides, commandBuffer: firstBuffer, settings: settings
        )
        let secondBuffer = try #require(queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.accumulationBusy) {
            try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: first, commandBuffer: secondBuffer, settings: settings
            )
        }
        try finish(firstBuffer)
        #expect(first.sampleCount == 1)
        let firstPixels = try pixels(in: target)
        let center = height / 2 * width + width / 2
        let normal = try pixels(in: normalDepth)[center]
        let albedo = try pixels(in: albedoRoughness)[center]
        #expect(normal.z > 0.99 && abs(Float(normal.w) - 3) < 0.01)
        #expect(abs(Float(albedo.x) - 0.65) < 0.002 && abs(Float(albedo.w) - 0.63) < 0.002)
        #expect(try pixels(in: normalDepth)[0] == .zero)

        let second = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            previous: first, commandBuffer: secondBuffer, samplesPerPixel: 3, settings: settings
        )
        try finish(secondBuffer)
        #expect(second.sampleCount == 4)
        let split = try pixels(in: target)
        #expect(split != firstPixels)
        #expect(split[center].x > 0.12)
        #expect(abs(Float(split[0].x) - 0.04) < 0.002)

        let thirdBuffer = try #require(queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.history) {
            try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: first, commandBuffer: thirdBuffer, settings: settings
            )
        }
        #expect(throws: PathTracer.Failure.history) {
            try tracer.encode(
                camera: camera, cameraTransform: .init(translation: SIMD3(1, 0, 0)), destination: target,
                accumulation: accumulation, previous: second, commandBuffer: thirdBuffer, settings: settings
            )
        }
        let restarted = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: thirdBuffer, samplesPerPixel: 4, settings: settings
        )
        try finish(thirdBuffer)
        #expect(restarted.sampleCount == 4)
        let single = try pixels(in: target)
        for (a, b) in zip(split, single) {
            let error = abs(SIMD4<Float>(a) - SIMD4<Float>(b))
            #expect(simd_reduce_max(error) < 0.006)
        }
    }

    @Test("A new rig relights the same tracer and requires a fresh history")
    func relightingRestartsHistory() throws {
        let plane = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .physicallyBased(try .init(roughness: 0.65, baseColorFactor: SIMD4(repeating: 1), alphaMode: .opaque))
        )
        let aim = SIMD3<Float>(0, 0, -3)
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: aim), mesh: plane)], camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 16)
        let center = SIMD3<Float>(0, 0.45, -1.5)
        let size = SIMD2<Float>(1.1, 0.8)
        let red = try PathTracer.AreaLight(
            center: center, aim: aim, size: size, radiance: SIMD3(8, 0, 0), cameraVisible: false
        )
        let blue = try PathTracer.AreaLight(
            center: center, aim: aim, size: size, radiance: SIMD3(0, 0, 8), cameraVisible: false
        )
        let redWorld = SIMD3<Float>(0.08, 0, 0)
        let redRig = try PathTracer.Lighting(lights: [red], environment: redWorld)
        let blueRig = try PathTracer.Lighting(lights: [blue], environment: SIMD3(0, 0, 0.12))
        let tracer = try PathTracer(queue: queue, scene: scene, lights: [red], environment: redWorld)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try output()
        let settings = PathTracer.Settings(maxBounces: 1, seed: 82)
        let firstBuffer = try #require(queue.makeCommandBuffer())
        let first = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: firstBuffer, samplesPerPixel: 8, settings: settings
        )
        try finish(firstBuffer)
        let pixelIndex = height / 2 * width + width / 2
        let redPixels = try pixels(in: target)
        #expect(redPixels[pixelIndex].x > 0.1 && redPixels[pixelIndex].z == 0)
        #expect(abs(Float(redPixels[0].x) - redWorld.x) < 0.001)

        let secondBuffer = try #require(queue.makeCommandBuffer())
        let second = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation, previous: first,
            commandBuffer: secondBuffer, settings: settings, lighting: redRig
        )
        try finish(secondBuffer)
        #expect(second.sampleCount == 9)

        let thirdBuffer = try #require(queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.lightingChanged) {
            try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: second, commandBuffer: thirdBuffer, settings: settings, lighting: blueRig
            )
        }
        let third = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: thirdBuffer, samplesPerPixel: 8, settings: settings, lighting: blueRig
        )
        try finish(thirdBuffer)
        #expect(third.sampleCount == 8)
        let bluePixels = try pixels(in: target)
        #expect(bluePixels[pixelIndex].z > 0.1 && bluePixels[pixelIndex].x == 0)
        #expect(abs(Float(bluePixels[0].z) - blueRig.environment.z) < 0.001)
    }

    private func output() throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.hazardTrackingMode = .tracked
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func finish(_ buffer: any MTLCommandBuffer) throws {
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "No Metal error")")
    }

    private func pixels(in texture: any MTLTexture) throws -> [SIMD4<Float16>] {
        var result = Array(repeating: SIMD4<Float16>.zero, count: width * height)
        try result.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: width * 8,
                from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0
            )
        }
        return result
    }
}
