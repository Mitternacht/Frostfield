import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathRegionTests {
    private let fixture: PathTestFixture

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Odd-sized regions match full-frame beauty and guides", arguments: [false, true])
    func regionsMatchFullFrame(stereo: Bool) throws {
        let tracer = try makeTracer()
        let width = 37
        let height = 30

        func render(
            width: Int, height: Int, region: PathTracer.Region?
        ) throws -> (color: [SIMD4<Float>], guides: [SIMD4<Float>]) {
            let accumulation = try tracer.makeAccumulation(width: width, height: height)
            let color = try fixture.output(width: width, height: height)
            let guides = try PathTracer.Guides(
                normalDepth: fixture.output(width: width, height: height),
                albedoRoughness: fixture.output(width: width, height: height)
            )
            var history: PathTracer.History?
            for _ in 0..<2 {
                let command = try #require(fixture.queue.makeCommandBuffer())
                if stereo {
                    history = try tracer.encode(
                        cameraModel: .init(projection: .stereoEquirectangular(interpupillaryDistance: 0.065)),
                        cameraTransform: .init(), destination: color, accumulation: accumulation,
                        previous: history, guides: guides, commandBuffer: command, samplesPerPixel: 4, region: region
                    )
                } else {
                    history = try tracer.encode(
                        camera: fixture.camera, cameraTransform: .init(), destination: color, accumulation: accumulation,
                        previous: history, guides: guides, commandBuffer: command, samplesPerPixel: 4,
                        settings: .init(maxBounces: 3, seed: 77, apertureRadius: 0.04, focusDistance: 3), region: region
                    )
                }
                try fixture.finish(command)
            }
            return try (fixture.readback(color), fixture.readback(guides.normalDepth))
        }

        let full = try render(width: width, height: height, region: nil)
        #expect(full.color.contains { $0.x > 0.2 })
        #expect(full.color.contains { $0.x < 0.1 })
        #expect(full.guides.contains { $0.w > 0 })
        #expect(full.guides.contains { $0.w == 0 })

        for (x, tileWidth) in [(0, 19), (19, width - 19)] {
            for (y, tileHeight) in [(0, 13), (13, height - 13)] {
                let region = try PathTracer.Region(imageWidth: width, imageHeight: height, x: x, y: y)
                let tile = try render(width: tileWidth, height: tileHeight, region: region)
                let offsets = (0..<tileHeight).flatMap { row in
                    let start = (y + row) * width + x
                    return start..<(start + tileWidth)
                }
                #expect(tile.color == offsets.map { full.color[$0] })
                #expect(tile.guides == offsets.map { full.guides[$0] })
            }
        }
    }

    @Test("Regions reject invalid bounds and cannot reuse another region's history")
    func regionValidation() throws {
        #expect(throws: PathTracer.Failure.region) {
            try PathTracer.Region(imageWidth: 32_769, imageHeight: 1, x: 0, y: 0)
        }
        #expect(throws: PathTracer.Failure.region) {
            try PathTracer.Region(imageWidth: 2, imageHeight: 2, x: 0, y: -1)
        }
        let tracer = try makeTracer()
        let accumulation = try tracer.makeAccumulation(width: 2, height: 2)
        let target = try fixture.output(width: 2, height: 2)
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.region) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target,
                accumulation: accumulation, commandBuffer: command,
                region: try .init(imageWidth: 32_768, imageHeight: 32_768, x: 32_767, y: 0)
            )
        }
        let history = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: target,
            accumulation: accumulation, commandBuffer: command,
            region: try .init(imageWidth: 32_768, imageHeight: 32_768, x: 32_766, y: 32_766)
        )
        try fixture.finish(command)

        let next = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.history) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target,
                accumulation: accumulation, previous: history, commandBuffer: next,
                region: try .init(imageWidth: 32_768, imageHeight: 32_768, x: 32_765, y: 32_766)
            )
        }
        #expect(throws: PathTracer.Failure.history) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target,
                accumulation: accumulation, previous: history, commandBuffer: next,
                region: try .init(imageWidth: 32_767, imageHeight: 32_767, x: 32_765, y: 32_765)
            )
        }
    }

    @Test("A streamed beauty still matches the full-view EXR across rows, columns and sample batches")
    func streamedStill() throws {
        let tracer = try makeTracer()
        let width = 1_037, height = 19
        let model = PathTracer.CameraModel(
            projection: .perspective(verticalFieldOfView: 0.9, shift: SIMD2(0.02, 0)),
            lens: .init(apertureRadius: 0.02, focusDistance: 3)
        )
        let settings = PathTracer.Settings(maxBounces: 2, seed: 41)
        let color = try fixture.output(width: width, height: height)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        var history: PathTracer.History?
        for samples in [16, 1] {
            let command = try #require(fixture.queue.makeCommandBuffer())
            history = try tracer.encode(
                cameraModel: model, cameraTransform: .init(), destination: color,
                accumulation: accumulation, previous: history, commandBuffer: command,
                samplesPerPixel: samples, settings: settings
            )
            try fixture.finish(command)
        }

        let pixels = try fixture.readback(color)
        #expect(pixels.contains { $0.x > 0.2 })
        #expect(pixels.contains { $0.x < 0.1 })
        let expected = try EXRExporter().data(width: width, height: height, pixels: pixels)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try tracer.writeStillEXR(
            to: url, cameraModel: model, width: width, height: height, samplesPerPixel: 17, settings: settings
        )
        #expect(try Data(contentsOf: url) == expected)
    }

    @Test("Streamed beauty and first-hit layers match the full-view EXR across both tile boundaries",
          arguments: [EXRExporter.Compression.zip, .piz])
    func streamedLayeredStill(compression: EXRExporter.Compression) throws {
        let blockHeight = compression == .piz ? 32 : 16
        let width = 1_037, height = blockHeight + 3
        let id = UUID(uuid: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
        let mesh = try fixture.pbr(
            .plane(width: 1.5, height: 0.8), color: SIMD3(0.8, 0.3, 0.12),
            roughness: 0.43, metallic: 0.25, emission: SIMD3(0.6, 0.13, 0.05)
        )
        // This tilted surface crosses x=1024 and the first block boundary, so all four tiles must
        // contain nonzero IDs, signed normals, and finite depth.
        let center = SIMD3<Float>(Float(1024.5 - Double(width) / 2) * 2 / Float(height), -0.74, -3)
        let node = SceneNode(id: id, transform: .init(
            translation: center, rotation: simd_quatf(angle: -0.12, axis: SIMD3(0, 1, 0))
        ), mesh: mesh)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, environment: SIMD3(repeating: 0.05))
        let model = PathTracer.CameraModel(projection: .orthographic(height: 2), near: 0.1, far: 30)
        let settings = PathTracer.Settings(maxBounces: 2, seed: 57)

        let beauty = try fixture.output(width: width, height: height)
        let albedo = try stillTexture(.rgba16Float, width: width, height: height)
        let normal = try stillTexture(.rgba16Float, width: width, height: height)
        let emission = try stillTexture(.rgba16Float, width: width, height: height)
        let specular = try stillTexture(.rgba16Float, width: width, height: height)
        let depth = try stillTexture(.r32Float, width: width, height: height)
        let projectedDepth = try stillTexture(.r32Float, width: width, height: height)
        let roughness = try stillTexture(.r32Float, width: width, height: height)
        let objectID = try stillTexture(.r32Uint, width: width, height: height)
        let materialID = try stillTexture(.r32Uint, width: width, height: height)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        var history: PathTracer.History?
        for samples in [16, 1] {
            let command = try #require(fixture.queue.makeCommandBuffer())
            history = try tracer.encode(
                cameraModel: model, cameraTransform: .init(), destination: beauty,
                accumulation: accumulation, previous: history, commandBuffer: command,
                samplesPerPixel: samples, settings: settings,
                aovs: .init(
                    albedo: albedo, normal: normal, depth: depth, roughness: roughness, emission: emission,
                    objectID: objectID, materialID: materialID, specularAlbedo: specular, projectedDepth: projectedDepth
                )
            )
            try fixture.finish(command)
        }

        let depths = stillPixels(depth, zero: Float(0))
        let objects = stillPixels(objectID, zero: UInt32(0))
        let normals = stillPixels(normal, zero: SIMD4<Float16>.zero)
        let idBits: UInt32 = 0xc8ff_f215
        for y in [blockHeight - 1, blockHeight] {
            for x in [1_023, 1_024] {
                let index = y * width + x
                #expect(depths[index].isFinite)
                #expect(objects[index] == idBits)
                #expect(normals[index].x < -0.1)
            }
        }
        #expect(depths[0] == .infinity && objects[0] == 0)

        var channels = rgbaChannels(beauty)
        channels += rgbaChannels(albedo, prefix: "albedo.")
        channels += rgbaChannels(normal, prefix: "normal.", axes: ["X", "Y", "Z", "A"])
        channels += rgbaChannels(emission, prefix: "emission.")
        channels += rgbaChannels(specular, prefix: "specularAlbedo.")
        channels += [
            .init(name: "depth.Z", samples: .float(depths)),
            .init(name: "projectedDepth.Z", samples: .float(stillPixels(projectedDepth, zero: Float(0)))),
            .init(name: "roughness.R", samples: .float(stillPixels(roughness, zero: Float(0)))),
            .init(name: "objectID.id", samples: .unsigned(objects)),
            .init(name: "materialID.id", samples: .unsigned(stillPixels(materialID, zero: UInt32(0)))),
        ]

        let expected = try EXRExporter(compression: compression).data(width: width, height: height, channels: channels)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try tracer.writeStillEXR(
            to: url, cameraModel: model, width: width, height: height, samplesPerPixel: 17, settings: settings,
            compression: compression, layers: Set(PathTracer.StillEXRLayer.allCases)
        )
        #expect(try Data(contentsOf: url) == expected)
    }

    @Test("Panoramic stills stream ray depth and preserve the destination on a projected-depth request")
    func panoramicLayerSelection() throws {
        let tracer = try makeTracer()
        let model = PathTracer.CameraModel(projection: .equirectangular)
        let width = 32, height = 16
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let existing = Data("existing file".utf8)
        try existing.write(to: url)
        #expect(throws: PathTracer.Failure.aovProjection) {
            try tracer.writeStillEXR(
                to: url, cameraModel: model, width: width, height: height, samplesPerPixel: 1,
                layers: [.projectedDepth]
            )
        }
        #expect(try Data(contentsOf: url) == existing)

        let beauty = try fixture.output(width: width, height: height)
        let depth = try stillTexture(.r32Float, width: width, height: height)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let command = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            cameraModel: model, cameraTransform: .init(), destination: beauty, accumulation: accumulation,
            commandBuffer: command, aovs: .init(depth: depth)
        )
        try fixture.finish(command)
        let values = stillPixels(depth, zero: Float(0))
        #expect(values.contains { $0.isFinite })
        #expect(values.contains(.infinity))
        let channels = rgbaChannels(beauty) + [.init(name: "depth.Z", samples: .float(values))]
        let expected = try EXRExporter(compression: .zipScanline).data(width: width, height: height, channels: channels)
        try tracer.writeStillEXR(
            to: url, cameraModel: model, width: width, height: height, samplesPerPixel: 1,
            compression: .zipScanline, layers: [.depth]
        )
        #expect(try Data(contentsOf: url) == expected)
    }

    private func stillTexture(_ format: MTLPixelFormat, width: Int, height: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        return try #require(fixture.device.makeTexture(descriptor: descriptor))
    }

    private func stillPixels<T>(_ texture: any MTLTexture, zero: T) -> [T] {
        var values = [T](repeating: zero, count: texture.width * texture.height)
        values.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                bytes.baseAddress!, bytesPerRow: texture.width * MemoryLayout<T>.stride,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return values
    }

    private func rgbaChannels(
        _ texture: any MTLTexture, prefix: String = "", axes: [String] = ["R", "G", "B", "A"]
    ) -> [EXRExporter.Channel] {
        let pixels = stillPixels(texture, zero: SIMD4<Float16>.zero)
        return [
            .init(name: prefix + axes[0], samples: .half(pixels.map(\.x))),
            .init(name: prefix + axes[1], samples: .half(pixels.map(\.y))),
            .init(name: prefix + axes[2], samples: .half(pixels.map(\.z))),
            .init(name: prefix + axes[3], samples: .half(pixels.map(\.w))),
        ]
    }

    private func makeTracer() throws -> PathTracer {
        let object = try fixture.pbr(
            .box(width: 1.7, height: 1.7, depth: 1.7), color: SIMD3(0.8, 0.2, 0.1), emission: SIMD3(0.4, 0.08, 0.02)
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0.24, 0.17, -3)), mesh: object)],
            camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        let light = try PathTracer.AreaLight(
            center: SIMD3(0, 2, -2.5), aim: SIMD3(0, 0, -3), size: SIMD2(1, 1),
            radiance: SIMD3(repeating: 6), cameraVisible: false
        )
        return try PathTracer(queue: fixture.queue, scene: scene, lights: [light], environment: SIMD3(repeating: 0.02))
    }
}
