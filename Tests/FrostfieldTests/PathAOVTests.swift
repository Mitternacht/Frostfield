import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathAOVTests {
    private let fixture: PathTestFixture
    private let width = 23
    private let height = 19
    private let nodeID = UUID(uuid: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("First-hit outputs match material values, world normals, depth and IDs", arguments: PathAccelerationTests.modes)
    func firstHitOutputs(acceleration: PathTracingScene.Options.Acceleration) throws {
        let tracer = try makeTracer(acceleration: acceleration)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let destination = try texture(.rgba16Float)
        let guides = try PathTracer.Guides(normalDepth: texture(.rgba16Float), albedoRoughness: texture(.rgba16Float))
        let albedo = try texture(.rgba32Float)
        let normal = try texture(.rgba16Float)
        let depth = try texture(.r32Float)
        let roughness = try texture(.r16Float)
        let emission = try texture(.rgba32Float)
        let objectID = try texture(.r32Uint)
        let materialID = try texture(.r32Uint)
        let specularAlbedo = try texture(.rgba32Float)
        let projectedDepth = try texture(.r32Float)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let history = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: destination,
            accumulation: accumulation, guides: guides, commandBuffer: command, samplesPerPixel: 4,
            settings: .init(apertureRadius: 0.04, focusDistance: 3),
            aovs: .init(albedo: albedo, normal: normal, depth: depth, roughness: roughness,
                        emission: emission, objectID: objectID, materialID: materialID,
                        specularAlbedo: specularAlbedo, projectedDepth: projectedDepth)
        )
        try fixture.finish(command)

        let colors: [SIMD4<Float>] = read(albedo, as: SIMD4<Float>.self, zero: .zero)
        let normals = try fixture.readback(normal)
        let depths = read(depth, as: Float.self, zero: 0)
        let roughnesses = read(roughness, as: Float16.self, zero: 0)
        let emitted: [SIMD4<Float>] = read(emission, as: SIMD4<Float>.self, zero: .zero)
        let objects = read(objectID, as: UInt32.self, zero: 0)
        let materials = read(materialID, as: UInt32.self, zero: 0)
        let specular = read(specularAlbedo, as: SIMD4<Float>.self, zero: .zero)
        let deviceDepth = read(projectedDepth, as: Float.self, zero: 0)
        let normalDepth = try fixture.readback(guides.normalDepth)
        let albedoRoughness = try fixture.readback(guides.albedoRoughness)
        let center = (height / 2) * width + width / 2
        let expectedObject = PathTracer.objectID(for: nodeID)
        #expect(expectedObject == 0xc8ff_f215)
        #expect(objects[center] == expectedObject)
        #expect(objects[0] == 0)
        // IOR 1.5 yields a 0.04 dielectric F0. The metal fraction reflects its base color.
        #expect(simd_reduce_max(abs(specular[center] - SIMD4(0.23, 0.13, 0.08, 1))) < 1e-6)
        let projection = try fixture.camera.projection(aspectRatio: Float(width) / Float(height))
        let clip = projection * SIMD4<Float>(0, 0, -3, 1)
        for index in colors.indices {
            if objects[index] == expectedObject {
                #expect(simd_reduce_max(abs(colors[index] - SIMD4(0.6, 0.3, 0.15, 1))) < 1e-6)
                #expect(normals[index] == SIMD4(0, 0, 1, 1))
                #expect(abs(depths[index] - 3) < 1e-5)
                #expect(abs(deviceDepth[index] - clip.z / clip.w) < 1e-6)
                #expect(specular[index].w == 1)
                #expect(roughnesses[index] == Float16(0.4))
                #expect(emitted[index] == SIMD4(0.75, 0.25, 0.125, 1))
                #expect(materials[index] == 1)
                #expect(normalDepth[index].z == 1 && normalDepth[index].w >= 3)
                #expect(albedoRoughness[index].x == Float(Float16(0.8)))
            } else {
                #expect(objects[index] == 0 && materials[index] == 0)
                #expect(colors[index] == .zero && normals[index] == .zero && emitted[index] == .zero)
                #expect(depths[index] == .infinity && roughnesses[index] == 0)
                #expect(specular[index] == .zero && deviceDepth[index] == 1)
                #expect(normalDepth[index] == .zero && albedoRoughness[index] == .zero)
            }
        }

        // A later encode may request a different set without rewriting the original guides.
        let nextID = try texture(.r32Uint)
        let next = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: destination,
            accumulation: accumulation, previous: history, commandBuffer: next,
            settings: .init(apertureRadius: 0.04, focusDistance: 3), aovs: .init(objectID: nextID)
        )
        try fixture.finish(next)
        #expect(read(nextID, as: UInt32.self, zero: 0) == objects)
    }

    @Test("AOV encodes reject invalid formats, sizes, and aliases before submission")
    func validatesOutputTextures() throws {
        let tracer = try makeTracer(acceleration: .software)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let destination = try texture(.rgba16Float)
        let color = try texture(.rgba16Float)
        let guides = try PathTracer.Guides(normalDepth: texture(.rgba16Float), albedoRoughness: texture(.rgba16Float))
        let invalid: [PathTracer.AOVs] = [
            .init(depth: color),
            .init(albedo: color, emission: color),
            .init(albedo: destination),
            .init(normal: guides.normalDepth),
            .init(projectedDepth: color),
            .init(albedo: color, specularAlbedo: color),
            .init(objectID: try texture(.r32Uint, width: width - 1))
        ]
        let command = try #require(fixture.queue.makeCommandBuffer())
        for aovs in invalid {
            #expect(throws: PathTracer.Failure.aovs) {
                try tracer.encode(
                    camera: fixture.camera, cameraTransform: .init(), destination: destination,
                    accumulation: accumulation, guides: guides, commandBuffer: command, aovs: aovs
                )
            }
        }
        _ = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: destination,
            accumulation: accumulation, guides: guides, commandBuffer: command, aovs: .init(albedo: color)
        )
        try fixture.finish(command)
    }

    @Test("Projected depth is linear for orthographic cameras and rejects panoramas before encoding")
    func projectedDepthCameraModels() throws {
        let tracer = try makeTracer(acceleration: .software)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let destination = try texture(.rgba16Float)
        let depth = try texture(.r32Float)
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.aovProjection) {
            try tracer.encode(
                cameraModel: .init(projection: .equirectangular), cameraTransform: .init(),
                destination: destination, accumulation: accumulation, commandBuffer: command, aovs: .init(projectedDepth: depth)
            )
        }
        _ = try tracer.encode(
            cameraModel: .init(projection: .orthographic(height: 4), near: 1, far: 9), cameraTransform: .init(),
            destination: destination, accumulation: accumulation, commandBuffer: command, aovs: .init(projectedDepth: depth)
        )
        try fixture.finish(command)
        let pixels = read(depth, as: Float.self, zero: 0)
        #expect(pixels[(height / 2) * width + width / 2] == 0.25)
        #expect(pixels[0] == 1)
    }

    @Test("Specular albedo follows the view angle and omits unlit reflection")
    func specularViewDependence() throws {
        let pbr = try PhysicallyBasedMaterial(
            metallic: 0, roughness: 0.2, baseColorFactor: SIMD4(0.8, 0.4, 0.2, 1), alphaMode: .opaque
        )
        let layered = try PhysicallyBasedMaterial(
            alphaMode: .opaque, clearcoat: .init(factor: 0.75),
            specular: .init(factor: 0.5, colorFactor: SIMD3(1, 0.5, 0.25))
        )
        let unlit = try PhysicallyBasedMaterial(baseColorFactor: SIMD4(1, 0, 0, 1), shadingModel: .unlit)
        let plane = try MeshGeometry.plane(width: 15, height: 15)
        func render(_ material: PhysicallyBasedMaterial) throws -> [SIMD4<Float>] {
            let renderer = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(material)
            )
            let graph = SceneGraph(
                roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)], camera: fixture.camera
            )
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: .software))
            let tracer = try PathTracer(queue: fixture.queue, scene: scene)
            let accumulation = try tracer.makeAccumulation(width: width, height: height)
            let color = try texture(.rgba16Float)
            let specular = try texture(.rgba16Float)
            let command = try #require(fixture.queue.makeCommandBuffer())
            _ = try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: color,
                accumulation: accumulation, commandBuffer: command, aovs: .init(specularAlbedo: specular)
            )
            try fixture.finish(command)
            return try fixture.readback(specular)
        }
        let dielectric = try render(pbr)
        let center = (height / 2) * width + width / 2
        #expect(abs(dielectric[center].x - 0.04) < 1e-4)
        #expect(dielectric[0].x > dielectric[center].x + 0.005 && dielectric[0].w == 1)
        let reflection = try render(layered)[center]
        // 0.75 coat reflects 0.03 at the view normal. It transmits 0.97 of the
        // modulated dielectric reflectance underneath.
        let expected = SIMD4<Float>(0.0494, 0.0397, 0.03485, 1)
        #expect(simd_reduce_max(abs(reflection - expected)) < 3e-5)
        let emissionOnly = try render(unlit)
        #expect(emissionOnly.allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
    }

    private func makeTracer(acceleration: PathTracingScene.Options.Acceleration) throws -> PathTracer {
        let mesh = try fixture.pbr(
            .plane(width: 1.5, height: 1.5), color: SIMD3(0.8, 0.4, 0.2),
            roughness: 0.4, metallic: 0.25, emission: SIMD3(0.75, 0.25, 0.125)
        )
        let graph = SceneGraph(
            roots: [SceneNode(id: nodeID, transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)],
            camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        return try PathTracer(queue: fixture.queue, scene: scene)
    }

    private func texture(_ format: MTLPixelFormat, width: Int? = nil) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width ?? self.width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        descriptor.hazardTrackingMode = .tracked
        return try #require(fixture.device.makeTexture(descriptor: descriptor))
    }

    private func read<T>(_ texture: any MTLTexture, as: T.Type, zero: T) -> [T] {
        var pixels = [T](repeating: zero, count: texture.width * texture.height)
        pixels.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                bytes.baseAddress!, bytesPerRow: texture.width * MemoryLayout<T>.stride,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return pixels
    }
}
