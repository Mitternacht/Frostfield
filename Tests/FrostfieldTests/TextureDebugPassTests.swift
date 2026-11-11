import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureDebugPassTests {
    @Test("A color overlay preserves the frame and displays the requested texels in top-left order")
    func overlay() throws {
        let fixture = try RasterLightingFixture(width: 4, height: 4)
        let viewer = try TextureDebugPass(device: fixture.device)
        let source = try texture(fixture.device, type: .type2D, width: 2, height: 2)
        let colors: [SIMD4<Float16>] = [
            SIMD4(2, 0.25, 0.5, 0.125), SIMD4(0, 1, 0, 0.25),
            SIMD4(0, 0, 1, 0.5), SIMD4(0.75, 0.25, 0.125, 0.75),
        ]
        upload(colors, to: source, width: 2, height: 2)
        let target = try fixture.colorTarget()
        let border = SIMD4<Float>(0.125, 0.25, 0.5, 1)
        try fixture.run { buffer in
            let clear = MTLRenderPassDescriptor()
            clear.colorAttachments[0].texture = target
            clear.colorAttachments[0].loadAction = .clear
            clear.colorAttachments[0].storeAction = .store
            clear.colorAttachments[0].clearColor = MTLClearColor(red: 0.125, green: 0.25, blue: 0.5, alpha: 1)
            let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: clear))
            encoder.endEncoding()
            try viewer.encode(source: .color(source), destination: target, region: MTLRegionMake2D(1, 1, 2, 2),
                              commandBuffer: buffer)
        }
        var expected = [SIMD4<Float>](repeating: border, count: 16)
        for (index, color) in zip([5, 6, 9, 10], colors) {
            expected[index] = SIMD4(Float(color.x), Float(color.y), Float(color.z), 1)
        }
        #expect(try fixture.pixels(of: target) == expected)

        try fixture.run { buffer in
            try viewer.encode(source: .color(source, channel: .alpha), destination: target, commandBuffer: buffer)
        }
        let alpha = try fixture.pixels(of: target)
        #expect([alpha[0], alpha[3], alpha[12], alpha[15]] == colors.map {
            SIMD4(Float($0.w), Float($0.w), Float($0.w), 1)
        })
    }

    @Test("Metal decodes source sRGB and encodes an sRGB destination")
    func colorTransfer() throws {
        let fixture = try RasterLightingFixture(width: 1, height: 1)
        let source = try fixture.texture(.rgba8Unorm_srgb, usage: .shaderRead, storage: .shared)
        let original: [UInt8] = [128, 64, 32, 0]
        original.withUnsafeBytes {
            source.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0,
                           withBytes: $0.baseAddress!, bytesPerRow: 4)
        }
        let linear = try fixture.colorTarget()
        let display = try fixture.texture(.rgba8Unorm_srgb, usage: .renderTarget, storage: .shared)
        let linearViewer = try TextureDebugPass(device: fixture.device)
        let displayViewer = try TextureDebugPass(device: fixture.device, pixelFormat: .rgba8Unorm_srgb)
        try fixture.run { buffer in
            try linearViewer.encode(source: .color(source), destination: linear, commandBuffer: buffer)
            try displayViewer.encode(source: .color(source), destination: display, commandBuffer: buffer)
        }
        let expected = SIMD4<Float>(0.21586, 0.051269, 0.014444, 1)
        #expect(try simd_distance(fixture.pixels(of: linear)[0], expected) < 0.001)
        var bytes = [UInt8](repeating: 0, count: 4)
        bytes.withUnsafeMutableBytes {
            display.getBytes($0.baseAddress!, bytesPerRow: 4, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0)
        }
        #expect(bytes == [128, 64, 32, 255])
    }

    @Test("Compressed ASTC texels display as decoded linear color",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires ASTC sampling"))
    func compressedColor() throws {
        let fixture = try RasterLightingFixture(width: 4, height: 4)
        let block = KTX2Fixture.astcConstantBlock([0xFFFF, 0x8080, 0, 0])
        let source = try TextureLoader(device: fixture.device).load(
            data: KTX2Fixture.container(vkFormat: 158, width: 4, height: 4, levels: [block])
        )
        let target = try fixture.colorTarget()
        let viewer = try TextureDebugPass(device: fixture.device)
        try fixture.run { buffer in
            try viewer.encode(source: .color(source), destination: target, commandBuffer: buffer)
        }
        #expect(try fixture.pixels(of: target).allSatisfy { simd_distance($0, SIMD4(1, 0.21586, 0, 1)) < 0.002 })
    }

    @Test("Array, volume and cube inspection selects the exact mip and slice")
    func subresources() throws {
        let fixture = try RasterLightingFixture(width: 2, height: 2)
        let viewer = try TextureDebugPass(device: fixture.device)
        let targets = try (0..<3).map { _ in try fixture.colorTarget() }
        let array = try texture(fixture.device, type: .type2DArray, width: 4, height: 4, slices: 2, mips: 2)
        let volume = try texture(fixture.device, type: .type3D, width: 4, height: 4, slices: 4, mips: 2)
        let cube = try texture(fixture.device, type: .typeCube, width: 4, height: 4, mips: 2)
        let markers = (0..<4).map { i in SIMD4<Float16>(Float16(i) * 0.25, 0.125, 0.25, 0.5) }
        let empty = [SIMD4<Float16>](repeating: .zero, count: 4)
        upload(empty, to: array, width: 2, height: 2, mip: 1)
        upload(markers, to: array, width: 2, height: 2, slice: 1, mip: 1)
        upload(empty, to: volume, width: 2, height: 2, mip: 1)
        upload(markers, to: volume, width: 2, height: 2, slice: 1, mip: 1)
        upload(empty, to: cube, width: 2, height: 2, slice: 4, mip: 1)
        upload(markers, to: cube, width: 2, height: 2, slice: 5, mip: 1)
        try fixture.run { buffer in
            for (source, target) in zip([
                TextureDebugPass.Source.color(array, slice: 1, mipLevel: 1),
                .color(volume, slice: 1, mipLevel: 1), .color(cube, slice: 5, mipLevel: 1),
            ], targets) {
                try viewer.encode(source: source, destination: target, commandBuffer: buffer)
            }
        }
        let expected = markers.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), 1) }
        for target in targets {
            #expect(try fixture.pixels(of: target) == expected)
        }
    }

    @Test("Depth display uses near-white grayscale in either depth convention")
    func depth() throws {
        let fixture = try RasterLightingFixture(width: 2, height: 2)
        let source = try fixture.depthTarget()
        let forward = try fixture.colorTarget(), reverse = try fixture.colorTarget()
        let viewer = try TextureDebugPass(device: fixture.device)
        try fixture.run { buffer in
            let clear = MTLRenderPassDescriptor()
            clear.depthAttachment.texture = source
            clear.depthAttachment.clearDepth = 0.25
            clear.depthAttachment.loadAction = .clear
            clear.depthAttachment.storeAction = .store
            let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: clear))
            encoder.endEncoding()
            try viewer.encode(source: .depth(source), destination: forward, commandBuffer: buffer)
            try viewer.encode(source: .depth(source, reversed: true), destination: reverse, commandBuffer: buffer)
        }
        #expect(try fixture.pixels(of: forward).allSatisfy { $0 == SIMD4(0.75, 0.75, 0.75, 1) })
        #expect(try fixture.pixels(of: reverse).allSatisfy { $0 == SIMD4(0.25, 0.25, 0.25, 1) })
    }

    @Test("Native shadow viewers distinguish cascades and punctual slices rendered on the same command buffer",
          arguments: [ShadowMapUpdateMode.automatic, .onDemand])
    func renderedShadows(_ updateMode: ShadowMapUpdateMode) throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        let viewer = try TextureDebugPass(device: fixture.device)
        let directional = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 32, directionToLight: SIMD3(0, 0, 1),
            cascades: .init(count: 2, splitLambda: 0, casterMargin: 2), updateMode: updateMode
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 12)
        try directional.fitCascades(camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1)
        let punctual = try PunctualShadowMaps(
            queue: fixture.queue, resolution: 32, spotCapacity: 2, pointCapacity: 1, updateMode: updateMode
        )
        let caster = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .cube(),
            material: .physicallyBased(.init(receivesShadows: true))
        )
        let shadow = try PunctualLight.Shadow(nearPlane: 0.5)
        let roots = try [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: caster),
            SceneNode(light: PunctualLight(kind: .spot(innerConeAngle: 0.3, outerConeAngle: 0.6), shadow: shadow)),
            SceneNode(light: PunctualLight(kind: .point(), shadow: shadow)),
        ]
        let graph = SceneGraph(roots: roots, camera: camera)
        let color = try fixture.colorTarget(), depth = try fixture.depthTarget()
        let targets = try (0..<5).map { _ in try fixture.colorTarget() }
        try fixture.run { buffer in
            if updateMode == .onDemand {
                #expect(throws: ShadowMapUpdateError.depthUnavailable) {
                    try viewer.encode(source: .directionalShadow(directional), destination: targets[0], commandBuffer: buffer)
                }
            }
            try graph.encode(pass: fixture.pass(color: color, depth: depth), commandBuffer: buffer,
                             shadowMap: directional, punctualShadowMaps: punctual)
            if updateMode == .onDemand {
                #expect(throws: ShadowMapUpdateError.depthUnavailable) {
                    try viewer.encode(source: .punctualShadow(punctual, slice: 1), destination: targets[0], commandBuffer: buffer)
                }
            }
            for (source, target) in zip([
                TextureDebugPass.Source.directionalShadow(directional, cascade: 0),
                .directionalShadow(directional, cascade: 1), .punctualShadow(punctual, slice: 0),
                .punctualShadow(punctual, slice: 2), .punctualShadow(punctual, slice: 7),
            ], targets) {
                try viewer.encode(source: source, destination: target, commandBuffer: buffer)
            }
        }
        let pixels = try targets.map(fixture.pixels)
        #expect(pixels[0] != pixels[1])
        for index in [0, 1, 2, 4] {
            let slice = pixels[index]
            #expect(slice.contains { $0.x > 0.05 })
            #expect(slice[0] == SIMD4(0, 0, 0, 1))
            #expect(slice.allSatisfy { $0.x == $0.y && $0.y == $0.z && $0.w == 1 })
        }
        #expect(pixels[3].allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
    }

    @Test("Validation rejects missing subresources, invalid rectangles, aliases and the wrong shadow queue")
    func validation() throws {
        let fixture = try RasterLightingFixture(width: 4, height: 4)
        let viewer = try TextureDebugPass(device: fixture.device)
        let source = try fixture.colorTarget(), destination = try fixture.colorTarget()
        let volume = try texture(fixture.device, type: .type3D, width: 4, height: 4, slices: 4, mips: 2)
        let map = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 32, directionToLight: SIMD3(0, 0, 1), cascades: .init(count: 2)
        )
        let otherQueue = try #require(fixture.device.makeCommandQueue())
        let wrongBuffer = try #require(otherQueue.makeCommandBuffer())
        try fixture.run { buffer in
            for invalid in [TextureDebugPass.Source.color(source, mipLevel: -1), .color(source, slice: 1),
                            .color(volume, slice: 2, mipLevel: 1)] {
                #expect(throws: TextureDebugPass.Failure.subresource) {
                    try viewer.encode(source: invalid, destination: destination, commandBuffer: buffer)
                }
            }
            #expect(throws: TextureDebugPass.Failure.source) {
                try viewer.encode(source: .depth(source), destination: destination, commandBuffer: buffer)
            }
            #expect(throws: TextureDebugPass.Failure.region) {
                try viewer.encode(source: .color(source), destination: destination, region: MTLRegionMake2D(3, 3, 2, 2),
                                  commandBuffer: buffer)
            }
            let alias = try #require(source.makeTextureView(pixelFormat: .rgba16Float))
            #expect(throws: TextureDebugPass.Failure.destination) {
                try viewer.encode(source: .color(alias), destination: source, commandBuffer: buffer)
            }
            #expect(throws: DirectionalShadowMap.Failure.cascadesNotFitted) {
                try viewer.encode(source: .directionalShadow(map), destination: destination, commandBuffer: buffer)
            }
            let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 8)
            try map.fitCascades(camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1)
            #expect(throws: TextureDebugPass.Failure.subresource) {
                try viewer.encode(source: .directionalShadow(map, cascade: 2), destination: destination, commandBuffer: buffer)
            }
            #expect(throws: DirectionalShadowMap.Failure.wrongQueue) {
                try viewer.encode(source: .directionalShadow(map), destination: destination, commandBuffer: wrongBuffer)
            }
        }
    }

    private func texture(
        _ device: any MTLDevice, type: MTLTextureType, width: Int, height: Int, slices: Int = 1, mips: Int = 1
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = type
        descriptor.pixelFormat = .rgba16Float
        descriptor.width = width
        descriptor.height = height
        descriptor.depth = type == .type3D ? slices : 1
        descriptor.arrayLength = type == .type2DArray ? slices : 1
        descriptor.mipmapLevelCount = mips
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func upload(
        _ pixels: [SIMD4<Float16>], to texture: any MTLTexture, width: Int, height: Int, slice: Int = 0, mip: Int = 0
    ) {
        let rowBytes = width * MemoryLayout<SIMD4<Float16>>.stride
        let isVolume = texture.textureType == .type3D
        let region = MTLRegionMake3D(0, 0, isVolume ? slice : 0, width, height, 1)
        pixels.withUnsafeBytes {
            texture.replace(region: region, mipmapLevel: mip, slice: isVolume ? 0 : slice,
                            withBytes: $0.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: rowBytes * height)
        }
    }
}
