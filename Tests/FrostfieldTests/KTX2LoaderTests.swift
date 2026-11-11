import Compression
import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct KTX2LoaderTests {
    private let device: any MTLDevice
    private let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    @Test
    func `Uploads stored RGBA8 levels with the container or requested encoding`() throws {
        let levels: [[UInt8]] = [
            Array(0..<32), [200, 10, 20, 255, 30, 40, 50, 128], [7, 8, 9, 10],
        ]
        let srgb = KTX2Fixture.container(vkFormat: 43, width: 4, height: 2, levels: levels)
        let color = try loader.load(data: srgb)
        #expect(color.pixelFormat == .rgba8Unorm_srgb)
        #expect(color.width == 4 && color.height == 2 && color.mipmapLevelCount == 3)
        for (index, level) in levels.enumerated() {
            #if targetEnvironment(simulator)
            // Simulator CPU reads may return level-zero bytes for a smaller mip.
            if index == 0 { #expect(try bytes(color, level: index) == level) }
            #else
            #expect(try bytes(color, level: index) == level)
            #endif
            let width = max(1, 4 >> index), height = max(1, 2 >> index)
            let offset = (height / 2 * width + width / 2) * 4
            let uv = SIMD2((Float(width / 2) + 0.5) / Float(width), (Float(height / 2) + 0.5) / Float(height))
            let expected = SIMD4(
                Self.linear(Float(level[offset]) / 255), Self.linear(Float(level[offset + 1]) / 255),
                Self.linear(Float(level[offset + 2]) / 255), Float(level[offset + 3]) / 255
            )
            #expect(simd_reduce_max(abs(try sample(color, level: index, uv: uv) - expected)) < 0.006)
        }
        #expect(try loader.load(data: srgb, interpretation: .linearData).pixelFormat == .rgba8Unorm)

        let unorm = KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: levels)
        #expect(try loader.load(data: unorm).pixelFormat == .rgba8Unorm)
        #expect(try loader.load(data: unorm, interpretation: .encodedSRGB).pixelFormat == .rgba8Unorm_srgb)

        // levelCount 0 stores one level and asks for a generated chain.
        let generated = KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [Array(repeating: 64, count: 32)], levelCount: 0)
        let chain = try loader.load(data: generated)
        #expect(chain.mipmapLevelCount == 3)
        #expect(try bytes(chain, level: 2) == [64, 64, 64, 64])
        #expect(try loader.load(data: KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [levels[0]]), mipmapped: true)
            .mipmapLevelCount == 3)
    }

    @Test
    func `Inflates zlib-supercompressed levels and verifies their checksum`() throws {
        let pixels: [UInt8] = (0..<64).map { UInt8($0 * 3 & 255) }
        let stream = try KTX2Fixture.zlib(pixels)
        let data = KTX2Fixture.container(
            vkFormat: 37, width: 4, height: 4, levels: [stream], supercompression: 3, uncompressedLengths: [64]
        )
        let texture = try loader.load(data: data)
        #expect(try bytes(texture, level: 0) == pixels)

        var corrupt = stream
        corrupt[corrupt.count - 1] ^= 1
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 0 zlib data fails its Adler-32 check")) {
            try loader.load(data: KTX2Fixture.container(
                vkFormat: 37, width: 4, height: 4, levels: [corrupt], supercompression: 3, uncompressedLengths: [64]
            ))
        }
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 0 zlib data does not inflate to 64 bytes")) {
            try loader.load(data: KTX2Fixture.container(
                vkFormat: 37, width: 4, height: 4, levels: [Array(stream.prefix(6)) + Array(stream.suffix(4))],
                supercompression: 3, uncompressedLengths: [64]
            ))
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires ASTC sampling"))
    func `Samples each stored ASTC level with sRGB or linear decoding`() throws {
        // Constant-color (void-extent) blocks: 8 × 8 has four 4 × 4 blocks, then one block per level.
        let colors: [[UInt16]] = [[0xFFFF, 0x8080, 0, 0xFFFF], [0, 0xFFFF, 0, 0xFFFF], [0, 0, 0xFFFF, 0x8080], [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF]]
        let levels = colors.enumerated().map { level, color in
            Array(repeating: KTX2Fixture.astcConstantBlock(color), count: level == 0 ? 4 : 1).flatMap { $0 }
        }
        let data = KTX2Fixture.container(vkFormat: 158, width: 8, height: 8, levels: levels)
        let texture = try loader.load(data: data)
        #expect(texture.pixelFormat == .astc_4x4_srgb)
        #expect(texture.width == 8 && texture.height == 8 && texture.mipmapLevelCount == 4)
        let half = Float(128) / 255
        let expected: [SIMD4<Float>] = [
            SIMD4(1, Self.linear(half), 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, half), SIMD4(repeating: 1),
        ]
        for level in 0..<4 {
            let sample = try sample(texture, level: level)
            #expect(simd_reduce_max(abs(sample - expected[level])) < 0.01, "level \(level): \(sample)")
        }
        let linear = try loader.load(data: data, interpretation: .linearData)
        #expect(linear.pixelFormat == .astc_4x4_ldr)
        #expect(abs(try sample(linear, level: 0).y - half) < 0.01)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsBCTextureCompression == true, "Requires BC sampling"))
    func `Samples a BC1 block`() throws {
        // RGB565 endpoints red and black with every index selecting the first endpoint.
        let block: [UInt8] = [0x00, 0xF8, 0x00, 0x00, 0, 0, 0, 0]
        let texture = try loader.load(data: KTX2Fixture.container(vkFormat: 133, width: 4, height: 4, levels: [block]))
        #expect(texture.pixelFormat == .bc1_rgba)
        #expect(simd_reduce_max(abs(try sample(texture, level: 0) - SIMD4(1, 0, 0, 1))) < 0.002)
    }

    @Test(arguments: BasisTextureFixture.Encoding.allCases)
    func `Loads Basis textures with their alpha and stored mip levels`(encoding: BasisTextureFixture.Encoding) throws {
        let data = try BasisTextureFixture.data(encoding)
        let texture = try loader.load(data: data)
        #expect(texture.pixelFormat == .rgba8Unorm_srgb)
        #expect(texture.width == 8 && texture.height == 8 && texture.mipmapLevelCount == 4)
        let linear = try loader.load(data: data, interpretation: .linearData)
        #expect(linear.pixelFormat == .rgba8Unorm)

        let base = try bytes(texture, level: 0)
        let coordinates = [(1, 1), (6, 1), (1, 6), (6, 6)]
        let colors = [[200, 40, 30, 64], [30, 180, 70, 192], [40, 50, 220, 128], [220, 205, 35, 255]]
        for ((x, y), expected) in zip(coordinates, colors) {
            let offset = (y * 8 + x) * 4
            for channel in 0..<4 {
                #expect(abs(Int(base[offset + channel]) - expected[channel]) <= 25, "\(encoding), texel (\(x), \(y))")
            }
        }
        for (level, texel) in BasisTextureFixture.centerTexels(encoding).enumerated() {
            let width = max(1, 8 >> level)
            let center = (Float(width / 2) + 0.5) / Float(width)
            let uv = SIMD2<Float>(repeating: center)
            let raw = SIMD4<Float>(texel) / 255
            let encoded = SIMD4(Self.linear(raw.x), Self.linear(raw.y), Self.linear(raw.z), raw.w)
            #expect(simd_reduce_max(abs(try sample(texture, level: level, uv: uv) - encoded)) < 0.008)
            #expect(simd_reduce_max(abs(try sample(linear, level: level, uv: uv) - raw)) < 0.005)
        }
    }

    @Test
    func `Rejects invalid Basis headers and unsupported layouts before allocating a texture`() throws {
        let data = try BasisTextureFixture.data(.uastcZstd)
        let invalid = TextureLoader.Failure.invalidKTX2("invalid ETC1S or UASTC Basis Universal data")
        var enormous = data
        enormous.replaceSubrange(20..<24, with: [255, 255, 255, 127])
        #expect(throws: invalid) { try loader.load(data: enormous) }
        var invalidDecodedSize = data
        invalidDecodedSize.replaceSubrange(96..<104, with: Array(repeating: UInt8(255), count: 8))
        #expect(throws: invalid) { try loader.load(data: invalidDecodedSize) }
        #expect(throws: invalid) { try loader.load(data: Data(data.dropLast())) }
        var unaligned = try BasisTextureFixture.data(.uastc)
        unaligned.append(0)
        unaligned[80] |= 1
        #expect(throws: invalid) { try loader.load(data: unaligned) }
        var array = data
        array[32] = 1
        #expect(throws: TextureLoader.Failure.unsupportedKTX2(
            "Basis Universal layout, transfer function, alpha mode, or supercompression"
        )) { try loader.load(data: array) }
        #expect(throws: TextureLoader.Failure.invalidKTX2("Basis Universal could not decode level 0")) {
            try loader.load(data: BasisTextureFixture.corruptZstd(data))
        }
    }

    @Test
    func `Rejects unsupported layouts and malformed KTX2 containers`() throws {
        let level = [UInt8](repeating: 0, count: 32)
        func load(_ data: Data) throws { _ = try loader.load(data: data) }
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("Zstandard supercompression")) {
            try load(KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [level], supercompression: 2))
        }
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("vkFormat 131")) {
            try load(KTX2Fixture.container(vkFormat: 131, width: 4, height: 4, levels: [Array(level.prefix(8))]))
        }
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("cube maps")) {
            try load(KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [level], faceCount: 6))
        }
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("orientation ru; only rd is supported")) {
            try load(KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [level], keyValues: [("KTXorientation", "ru")]))
        }
        #expect(try loader.load(data: KTX2Fixture.container(
            vkFormat: 37, width: 4, height: 2, levels: [level], keyValues: [("KTXorientation", "rd"), ("KTXwriter", "test")]
        )).width == 4)
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 0 needs 32 uncompressed bytes")) {
            try load(KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [Array(level.prefix(28))]))
        }
        #expect(throws: TextureLoader.Failure.invalidKTX2("levelCount 5 exceeds the 3 levels of a full chain")) {
            try load(KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [level], levelCount: 5))
        }
        #expect(throws: TextureLoader.Failure.invalidKTX2("typeSize 2 does not match vkFormat 37")) {
            try load(KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, typeSize: 2, levels: [level]))
        }
        let truncated = KTX2Fixture.container(vkFormat: 37, width: 4, height: 2, levels: [level]).dropLast(4)
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 0 lies outside the file")) {
            try load(Data(truncated))
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires ASTC sampling"))
    func `Raster PBR and path tracing sample ASTC maps like RGBA8 maps`() throws {
        let block = KTX2Fixture.astcConstantBlock([0xFFFF, 0x8080, 0, 0xFFFF])
        let astc = try loader.load(data: KTX2Fixture.container(vkFormat: 158, width: 4, height: 4, levels: [block]))
        let rgba = try loader.load(data: KTX2Fixture.container(
            vkFormat: 43, width: 1, height: 1, levels: [[255, 128, 0, 255]]
        ))
        let queue = try #require(device.makeCommandQueue())

        // Raster: identical shading for the compressed and uncompressed base color.
        func centerPixel(_ texture: any MTLTexture) throws -> [UInt8] {
            let size = 32
            let color = try #require(device.makeTexture(descriptor: {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false)
                descriptor.storageMode = .shared
                descriptor.usage = .renderTarget
                return descriptor
            }()))
            let depth = try #require(device.makeTexture(descriptor: {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: MeshRenderer.depthPixelFormat, width: size, height: size, mipmapped: false
                )
                descriptor.storageMode = .private
                descriptor.usage = .renderTarget
                return descriptor
            }()))
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .dontCare
            let material = try PhysicallyBasedMaterial(
                baseColorTexture: texture, roughness: 0.8, directionToLight: SIMD3(0, 0, 1),
                lightIntensity: 1, ambientIntensity: 0.2, alphaMode: .opaque
            )
            let renderer = try MeshRenderer(
                device: device, colorPixelFormat: .rgba8Unorm, geometry: .cube(), material: .physicallyBased(material)
            )
            let scene = MeshScene(
                camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10),
                transform: SceneTransform(translation: SIMD3(0, 0, -3))
            )
            let command = try #require(queue.makeCommandBuffer())
            try renderer.encode(scene, pass: pass, commandBuffer: command)
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed)
            return try bytes(color, level: 0, at: SIMD2(size / 2, size / 2), count: 1)
        }
        let compressedPixel = try centerPixel(astc)
        let uncompressedPixel = try centerPixel(rgba)
        #expect(compressedPixel[0] > compressedPixel[1] && compressedPixel[1] > compressedPixel[2])
        #expect(zip(compressedPixel, uncompressedPixel).allSatisfy { abs(Int($0) - Int($1)) <= 2 })

        // Data maps accept linear block formats and reject sRGB ones, as with RGBA8.
        func normalMapped(_ texture: any MTLTexture) throws -> MeshRenderer {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba8Unorm, geometry: .cube(),
                material: .physicallyBased(PhysicallyBasedMaterial(normalTexture: .init(texture: texture)))
            )
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidDataTexture) { try normalMapped(astc) }
        let linearASTC = try loader.load(
            data: KTX2Fixture.container(vkFormat: 157, width: 4, height: 4, levels: [block]), interpretation: .linearData
        )
        #expect(linearASTC.pixelFormat == .astc_4x4_ldr)
        _ = try normalMapped(linearASTC)

        // Path tracing bakes the sRGB map into linear radiance on an emissive surface.
        let emitter = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveTexture: .init(texture: astc), emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque
            ))
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: emitter)], camera: camera)
        let tracer = try PathTracer(queue: queue, scene: PathTracingScene(queue: queue, graph: graph, textureSize: 16))
        let accumulation = try tracer.makeAccumulation(width: 9, height: 9)
        let target = try #require(device.makeTexture(descriptor: {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 9, height: 9, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = [.shaderWrite, .shaderRead]
            descriptor.hazardTrackingMode = .tracked
            return descriptor
        }()))
        let command = try #require(queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: command, samplesPerPixel: 2, settings: .init(maxBounces: 1, seed: 3), lighting: PathTracer.Lighting()
        )
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        var pixel = SIMD4<Float16>.zero
        withUnsafeMutableBytes(of: &pixel) { bytes in
            target.getBytes(bytes.baseAddress!, bytesPerRow: 8, from: MTLRegionMake2D(4, 4, 1, 1), mipmapLevel: 0)
        }
        let radiance = SIMD3(Float(pixel.x), Float(pixel.y), Float(pixel.z))
        #expect(simd_distance(radiance, SIMD3(1, Self.linear(Float(128) / 255), 0)) < 0.01, "\(radiance)")
    }

    private static func linear(_ encoded: Float) -> Float {
        encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4)
    }

    private func bytes(
        _ texture: any MTLTexture, level: Int, at origin: SIMD2<Int> = .zero, count: Int? = nil
    ) throws -> [UInt8] {
        let width = count ?? max(1, texture.width >> level), height = count ?? max(1, texture.height >> level)
        var result = [UInt8](repeating: 0, count: width * height * 4)
        try result.withUnsafeMutableBytes { raw in
            texture.getBytes(
                try #require(raw.baseAddress), bytesPerRow: width * 4,
                from: MTLRegionMake2D(origin.x, origin.y, width, height), mipmapLevel: level
            )
        }
        return result
    }

    /// Samples one coordinate at an explicit level with nearest filtering.
    private func sample(
        _ texture: any MTLTexture, level: Int, uv: SIMD2<Float> = SIMD2(repeating: 0.5)
    ) throws -> SIMD4<Float> {
        let library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void inspect(texture2d<float> image [[texture(0)]], device float4* result [[buffer(0)]],
                                constant float4& inputs [[buffer(1)]], uint index [[thread_position_in_grid]]) {
                constexpr sampler nearest(coord::normalized, filter::nearest, mip_filter::nearest);
                result[index] = image.sample(nearest, inputs.yz, metal::level(inputs.x));
            }
            """, options: nil)
        let pipeline = try device.makeComputePipelineState(function: try #require(library.makeFunction(name: "inspect")))
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        let output = try #require(device.makeBuffer(length: 16, options: .storageModeShared))
        var selected = SIMD4(Float(level), uv.x, uv.y, 0)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBytes(&selected, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        return output.contents().load(as: SIMD4<Float>.self)
    }
}

/// Writes KTX 2.0 containers following the specification's layout.
enum KTX2Fixture {
    static func container(
        vkFormat: UInt32, width: Int, height: Int, typeSize: UInt32 = 1, levels: [[UInt8]], levelCount: Int? = nil,
        supercompression: UInt32 = 0, uncompressedLengths: [Int]? = nil, colorModel: UInt8 = 1, faceCount: UInt32 = 1,
        layerCount: UInt32 = 0, keyValues: [(String, String)] = []
    ) -> Data {
        var header = [UInt8](arrayLiteral: 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A)
        func word(_ value: UInt32, to bytes: inout [UInt8]) { bytes += withUnsafeBytes(of: value.littleEndian) { Array($0) } }
        func long(_ value: Int, to bytes: inout [UInt8]) { bytes += withUnsafeBytes(of: UInt64(value).littleEndian) { Array($0) } }
        for value in [vkFormat, typeSize, UInt32(width), UInt32(height), 0, layerCount, faceCount, UInt32(levelCount ?? levels.count), supercompression] {
            word(value, to: &header)
        }
        let indexEnd = 80 + levels.count * 24
        // Basic data format descriptor block: color model, BT.709 primaries, sRGB or linear transfer.
        var descriptor: [UInt8] = []
        word(28, to: &descriptor)
        word(0, to: &descriptor)
        word(2 | 24 << 16, to: &descriptor)
        descriptor += [colorModel, 1, vkFormat == 43 ? 2 : 1, 0]
        descriptor += [UInt8](repeating: 0, count: 12)
        var keyValueData: [UInt8] = []
        for (key, value) in keyValues {
            let entry = Array(key.utf8) + [0] + Array(value.utf8) + [0]
            word(UInt32(entry.count), to: &keyValueData)
            keyValueData += entry
            while !keyValueData.count.isMultiple(of: 4) { keyValueData.append(0) }
        }
        let keyValueOffset = indexEnd + descriptor.count
        for value in [UInt32(indexEnd), UInt32(descriptor.count), UInt32(keyValueOffset), UInt32(keyValueData.count)] {
            word(value, to: &header)
        }
        long(0, to: &header)
        long(0, to: &header)

        // Levels are stored from the smallest to the largest, each aligned to 16 bytes.
        var payload = [UInt8]()
        var offsets = [Int](repeating: 0, count: levels.count)
        let dataStart = keyValueOffset + keyValueData.count
        for index in levels.indices.reversed() {
            while !(dataStart + payload.count).isMultiple(of: 16) { payload.append(0) }
            offsets[index] = dataStart + payload.count
            payload += levels[index]
        }
        for index in levels.indices {
            long(offsets[index], to: &header)
            long(levels[index].count, to: &header)
            long(uncompressedLengths?[index] ?? levels[index].count, to: &header)
        }
        return Data(header + descriptor + keyValueData + payload)
    }

    /// An LDR void-extent block with one UNORM16 RGBA color.
    static func astcConstantBlock(_ color: [UInt16]) -> [UInt8] {
        [0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF] + color.flatMap { [UInt8($0 & 0xFF), UInt8($0 >> 8)] }
    }

    /// An RFC 1950 zlib stream: Apple's ZLIB algorithm writes the raw DEFLATE body.
    static func zlib(_ bytes: [UInt8]) throws -> [UInt8] {
        let body = try (Data(bytes) as NSData).compressed(using: .zlib) as Data
        var a: UInt32 = 1, b: UInt32 = 0
        for byte in bytes {
            a = (a + UInt32(byte)) % 65_521
            b = (b + a) % 65_521
        }
        let checksum = b << 16 | a
        return [0x78, 0x9C] + [UInt8](body) + [UInt8(checksum >> 24), UInt8(checksum >> 16 & 255), UInt8(checksum >> 8 & 255), UInt8(checksum & 255)]
    }
}
