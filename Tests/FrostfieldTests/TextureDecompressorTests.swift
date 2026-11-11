import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureDecompressorTests {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let decompressor: TextureDecompressor

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        decompressor = try TextureDecompressor(device: device)
    }

    @Test("sRGB ASTC expands with source orientation, alpha, and every stored mip",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires ASTC sampling"))
    func srgbMipChain() throws {
        let topLeft = KTX2Fixture.astcConstantBlock([0x8080, 0, 0, 0x4040])
        let topRight = KTX2Fixture.astcConstantBlock([0, 0xFFFF, 0, 0xFFFF])
        let bottomLeft = KTX2Fixture.astcConstantBlock([0, 0, 0xFFFF, 0])
        let bottomRight = KTX2Fixture.astcConstantBlock([0xFFFF, 0xFFFF, 0, 0xFFFF])
        let first = KTX2Fixture.astcConstantBlock([0xFFFF, 0, 0xFFFF, 0xFFFF])
        let second = KTX2Fixture.astcConstantBlock([0, 0xFFFF, 0xFFFF, 0xFFFF])
        let last = KTX2Fixture.astcConstantBlock([0x2020, 0x4040, 0x8080, 0xFFFF])
        let data = KTX2Fixture.container(
            vkFormat: 158, width: 8, height: 8,
            levels: [topLeft + topRight + bottomLeft + bottomRight, first, second, last]
        )
        let loader = TextureLoader(device: device)
        let source = try loader.loadKTX2(data: data)
        source.label = "ASTC fixture"
        let result = try decompressor.decompress(source, commandQueue: queue)
        #expect(result !== source && result.label == source.label)
        #expect(result.pixelFormat == .rgba8Unorm_srgb)
        #expect(result.width == 8 && result.height == 8 && result.mipmapLevelCount == 4)
        #expect(result.storageMode == .private && result.usage.contains(.shaderRead))
        let exported = try KTX2Exporter.data(from: result, commandQueue: queue)
        let roundTrip = try loader.loadKTX2(data: exported)
        #expect(roundTrip.pixelFormat == .rgba8Unorm_srgb && roundTrip.mipmapLevelCount == 4)

        let probe = try DDSProbe(device: device)
        let samples: [(SIMD3<Float>, Int)] = [
            (SIMD3(0.25, 0.25, 0), 0), (SIMD3(0.75, 0.25, 0), 0),
            (SIMD3(0.25, 0.75, 0), 0), (SIMD3(0.75, 0.75, 0), 0),
            (SIMD3(0.5, 0.5, 0), 1), (SIMD3(0.5, 0.5, 0), 2), (SIMD3(0.5, 0.5, 0), 3),
        ]
        for (uv, level) in samples {
            let native = try probe.sample(source, at: uv, level: level)
            let output = try probe.sample(result, at: uv, level: level)
            #expect(simd_reduce_max(abs(native - output)) < 0.005, "level \(level): \(native), \(output)")
            #expect(try probe.sample(roundTrip, at: uv, level: level) == output)
        }
        let unassociated = try probe.sample(result, at: SIMD3(0.25, 0.25, 0))
        #expect(abs(unassociated.x - 0.21586) < 0.005 && abs(unassociated.w - 64 / 255) < 0.003)
        #expect(try probe.sample(result, at: SIMD3(0.25, 0.75, 0)) == SIMD4(0, 0, 1, 0))
    }

    @Test("Caller encoding scales the cap in both dimensions and preserves selected levels",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires ASTC sampling"))
    func resizedEncoding() throws {
        let source = try resizeSource()
        let command = try #require(queue.makeCommandBuffer())
        let capped = try decompressor.encode(source, maximumSize: 3, commandBuffer: command)
        #expect(command.status == .notEnqueued)
        #expect(capped.width == 3 && capped.height == 3 && capped.mipmapLevelCount == 2)
        #expect(capped.pixelFormat == .rgba8Unorm)
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        let probe = try DDSProbe(device: device)
        let middle = try probe.sample(capped, at: SIMD3(0.5, 0.5, 0))
        #expect(simd_reduce_max(abs(middle - SIMD4(128 / 255, 0, 128 / 255, 1))) < 0.005)
        #expect(try probe.sample(capped, at: SIMD3(0.1, 0.5, 0)) == SIMD4(1, 0, 0, 1))
        #expect(try probe.sample(capped, at: SIMD3(0.9, 0.5, 0)) == SIMD4(0, 0, 1, 1))
        #expect(try probe.sample(capped, at: SIMD3(0.5, 0.5, 0), level: 1) == SIMD4(0, 1, 0, 1))

        let base = try decompressor.decompress(source, includeMipmaps: false)
        #expect(base.width == 8 && base.height == 4 && base.mipmapLevelCount == 1)
        #expect(try probe.sample(base, at: SIMD3(0.25, 0.5, 0)) == SIMD4(1, 0, 0, 1))
        #expect(throws: TextureDecompressor.Failure.size) { try decompressor.decompress(source, maximumSize: 0) }
        #expect(throws: TextureDecompressor.Failure.commandBuffer) {
            try decompressor.encode(source, commandBuffer: command)
        }
    }

    @Test("Signed EAC retains negative channels and native precision",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires EAC sampling"))
    func signedEAC() throws {
        let red: [UInt8] = [64, 0x10, 0, 0, 0, 0, 0, 0]
        let green: [UInt8] = [192, 0x10, 0, 0, 0, 0, 0, 0]
        let source = try compressed(.eac_rg11Snorm, side: 4, rowBytes: 16, bytes: red + green)
        let result = try decompressor.decompress(source)
        #expect(result.pixelFormat == .rgba16Float)
        let probe = try DDSProbe(device: device)
        let uv = SIMD3<Float>(0.5, 0.5, 0)
        let output = try probe.sample(result, at: uv)
        #expect(output.x > 0.2 && output.y < -0.4 && output.z == 0 && output.w == 1)
        #expect(simd_reduce_max(abs(output - (try probe.sample(source, at: uv)))) < 0.0006)
    }

    @Test("PVRTC decompression keeps the source swizzle and its alpha",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires PVRTC sampling"))
    func pvrtcSwizzle() throws {
        // Translucent red endpoints. The texture's swizzle maps red to blue
        // and makes the sampled alpha opaque.
        let word: [UInt8] = [0, 0, 0, 0, 0, 0x4F, 0, 0x4F]
        let source = try compressed(
            .pvrtc_rgba_4bpp, side: 8, rowBytes: 0, bytes: Array(repeating: word, count: 4).flatMap { $0 },
            swizzle: MTLTextureSwizzleChannels(red: .blue, green: .green, blue: .red, alpha: .one)
        )
        let result = try decompressor.decompress(source)
        #expect(result.pixelFormat == .rgba8Unorm)
        #expect(try DDSProbe(device: device).sample(result, at: SIMD3(0.5, 0.5, 0)) == SIMD4(0, 0, 1, 1))
    }

    @Test("BC6H decompresses to signed or unsigned HDR half floats",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsBCTextureCompression == true, "Requires BC sampling"),
          arguments: [false, true])
    func bcHDR(signed: Bool) throws {
        let bytes = DDSFixture.bc6H(red: signed ? 768 : 512, green: 256)
        let dds = DDSFixture.data(width: 4, height: 4, fourCC: "DX10", dxgi: signed ? 96 : 95, payload: bytes)
        let source = try TextureLoader(device: device).loadDDS(data: dds)
        #expect(source.pixelFormat == (signed ? .bc6H_rgbFloat : .bc6H_rgbuFloat))
        let output = try decompressor.decompress(source)
        #expect(output.pixelFormat == .rgba16Float)
        let color = try DDSProbe(device: device).sample(output, at: SIMD3(0.5, 0.5, 0))
        #expect(color.x == Float(Float16(bitPattern: signed ? 0xBE1F : 0x3E0F)))
        #expect(color.y == Float(Float16(bitPattern: signed ? 0x3E1F : 0x1F0F)))
        #expect(color.z == 0 && color.w == 1)
    }

    @Test("ASTC HDR preserves values above one",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple6) == true, "Requires HDR ASTC sampling"))
    func astcHDR() throws {
        let expected = SIMD4<Float>(4, 0.375, 1.25, 0.5)
        var block = KTX2Fixture.astcConstantBlock((0..<4).map { Float16(expected[$0]).bitPattern })
        block[1] |= 0x02 // HDR void-extent blocks store FP16 instead of UNORM16.
        let source = try compressed(.astc_4x4_hdr, side: 4, rowBytes: 16, bytes: block)
        let result = try decompressor.decompress(source)
        #expect(result.pixelFormat == .rgba16Float)
        #expect(try DDSProbe(device: device).sample(result, at: SIMD3(0.5, 0.5, 0)) == expected)
    }

    @Test("Decompression rejects uncompressed and array inputs")
    func invalidSources() throws {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
        desc.usage = .shaderRead
        let uncompressed = try #require(device.makeTexture(descriptor: desc))
        #expect(throws: TextureDecompressor.Failure.pixelFormat(.rgba8Unorm)) {
            try decompressor.decompress(uncompressed)
        }
        desc.textureType = .type2DArray
        desc.arrayLength = 2
        let array = try #require(device.makeTexture(descriptor: desc))
        #expect(throws: TextureDecompressor.Failure.source) { try decompressor.decompress(array) }
    }

    private func resizeSource() throws -> any MTLTexture {
        let red = KTX2Fixture.astcConstantBlock([0xFFFF, 0, 0, 0xFFFF])
        let blue = KTX2Fixture.astcConstantBlock([0, 0, 0xFFFF, 0xFFFF])
        let green = KTX2Fixture.astcConstantBlock([0, 0xFFFF, 0, 0xFFFF])
        let black = KTX2Fixture.astcConstantBlock([0, 0, 0, 0xFFFF])
        return try TextureLoader(device: device).loadKTX2(data: KTX2Fixture.container(
            vkFormat: 157, width: 8, height: 4, levels: [red + blue, green, black]
        ))
    }

    private func compressed(
        _ format: MTLPixelFormat, side: Int, rowBytes: Int, bytes: [UInt8],
        swizzle: MTLTextureSwizzleChannels = MTLTextureSwizzleChannels(red: .red, green: .green, blue: .blue, alpha: .alpha)
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: side, height: side, mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        descriptor.swizzle = swizzle
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        bytes.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: rowBytes)
        }
        return texture
    }
}
