import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RasterTextureUploadTests {
    let device: any MTLDevice
    let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    // Tagged sRGB RGBA8 with three unequal rows, mixed opacity and hidden RGB.
    private static let image = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAYAAAC56t6BAAAAAXNSR0IArs4c6QAAACRJREFUeJxjaHA40aBwIoWB4b8Ng4PIFpv/DCLPUv47NPw5AACONQusOcMDkgAAAABJRU5ErkJggg=="
    private static let unflipped: [UInt8] = [
        128, 64, 200, 128, 32, 200, 100, 0,
        255, 60, 0, 64, 20, 180, 60, 255,
        20, 230, 100, 255, 64, 128, 252, 192,
    ]
    private static let flipped: [UInt8] = [
        20, 230, 100, 255, 64, 128, 252, 192,
        255, 60, 0, 64, 20, 180, 60, 255,
        128, 64, 200, 128, 32, 200, 100, 0,
    ]
    // Exact reference bytes from the pinned Three.js WebGL and WebGPU uploads.
    private static let browserAssociated: [UInt8] = [
        20, 230, 100, 255, 48, 96, 190, 192,
        64, 15, 0, 64, 20, 180, 60, 255,
        64, 32, 100, 128, 0, 0, 0, 0,
    ]

    @Test("Encoded association and row reversal match browser upload codes",
          arguments: [TextureLoader.Interpretation.linearData, .encodedSRGB])
    func browserCodes(interpretation: TextureLoader.Interpretation) throws {
        let png = try #require(Data(base64Encoded: Self.image))
        let original = try loader.load(data: png, interpretation: interpretation)
        let explicitDefault = try loader.load(data: png, interpretation: interpretation, uploadOptions: .init())
        let flipped = try loader.load(data: png, interpretation: interpretation, uploadOptions: .init(flipY: true))
        let associated = try loader.load(
            data: png, interpretation: interpretation, uploadOptions: .init(flipY: true, alphaMode: .premultipliedEncoded)
        )
        #expect(Self.codes(original) == Self.unflipped && Self.codes(explicitDefault) == Self.unflipped)
        #expect(Self.codes(flipped) == Self.flipped)
        #expect(Self.codes(associated) == Self.browserAssociated)
        #expect(associated.pixelFormat == (interpretation == .linearData ? .rgba8Unorm : .rgba8Unorm_srgb))
        #expect(associated.mipmapLevelCount == 1)

        let probe = try DDSProbe(device: device)
        for index in 0..<6 {
            let offset = index * 4
            let code = Self.browserAssociated[offset..<(offset + 4)].map { Float($0) / 255 }
            let rgb = interpretation == .linearData ? Array(code.prefix(3)) : code.prefix(3).map(Self.decodeSRGB)
            let expected = SIMD4(rgb[0], rgb[1], rgb[2], code[3])
            let point = SIMD3((Float(index % 2) + 0.5) / 2, (Float(index / 2) + 0.5) / 3, 0)
            let actual = try probe.sample(associated, at: point)
            #expect(simd_reduce_max(abs(actual - expected)) < 0.003, "texel \(index)")
        }
    }

    @Test("Linear association operates in shader sampling space after ordinary decoding",
          arguments: [TextureLoader.Interpretation.color, .encodedSRGB, .linearData])
    func linearAssociation(interpretation: TextureLoader.Interpretation) throws {
        let png = try #require(Data(base64Encoded: Self.image))
        let original = try loader.load(data: png, interpretation: interpretation)
        let associated = try loader.load(data: png, interpretation: interpretation,
                                         uploadOptions: .init(alphaMode: .premultipliedLinear))
        let probe = try DDSProbe(device: device)
        let point = SIMD3<Float>(0.25, 1 / 6, 0)
        let straight = try probe.sample(original, at: point)
        let output = try probe.sample(associated, at: point)
        #expect(output.w == straight.w)
        #expect(simd_reduce_max(abs(output.xyz - straight.xyz * straight.w)) < 0.004)
        #expect(try probe.sample(associated, at: SIMD3(0.75, 1 / 6, 0)) == .zero)
        if interpretation == .encodedSRGB {
            let encoded = try loader.load(data: png, interpretation: interpretation,
                                           uploadOptions: .init(alphaMode: .premultipliedEncoded))
            #expect(output.x > (try probe.sample(encoded, at: point)).x + 0.04)
        }
    }

    @Test("Generated mips sample flipped and associated colors",
          arguments: [TextureLoader.Interpretation.linearData, .encodedSRGB])
    func generatedMipChain(interpretation: TextureLoader.Interpretation) throws {
        // 4 × 4, with 2 × 2 blocks of opaque red, half-alpha green,
        // transparent blue, and opaque white.
        let png = try #require(Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAAXNSR0IArs4c6QAAABtJREFUeJxj+M/A8B+EgbABitEEQGwwHwowBADv2SHhoG6bEwAAAABJRU5ErkJggg=="
        ))
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("png")
        defer { try? FileManager.default.removeItem(at: file) }
        try png.write(to: file)
        let texture = try loader.load(contentsOf: file, interpretation: interpretation, mipmapped: true,
                                      uploadOptions: .init(flipY: true, alphaMode: .premultipliedLinear))
        #expect(texture.mipmapLevelCount == 3)
        let probe = try DDSProbe(device: device)
        #expect(try probe.sample(texture, at: SIMD3(0.25, 0.25, 0), level: 1) == .zero)
        #expect(try probe.sample(texture, at: SIMD3(0.25, 0.75, 0), level: 1) == SIMD4(1, 0, 0, 1))
        #expect(try probe.sample(texture, at: SIMD3(0.75, 0.25, 0), level: 1) == SIMD4(repeating: 1))
        let half: Float = 128 / 255
        let sample = try probe.sample(texture, at: SIMD3(0.5, 0.5, 0), level: 2)
        let expected = SIMD4<Float>(0.5, (1 + half) / 4, 0.25, (2 + half) / 4)
        #expect(simd_reduce_max(abs(sample - expected)) < 0.006, "\(sample), \(expected)")
    }

    @Test("Native TGA rows are normalized before the requested flip and association")
    func nativeTGA() throws {
        // RLE truecolor with bottom-right origin and eight attribute bits.
        // One raw packet in stored order: hidden blue, opaque green, translucent red.
        let file = Data([
            0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 32, 0x18,
            2, 255, 0, 0, 0, 0, 255, 0, 255, 0, 0, 200, 128,
        ])
        let texture = try loader.load(data: file, interpretation: .linearData,
                                      uploadOptions: .init(flipY: true, alphaMode: .premultipliedLinear))
        #expect(Self.codes(texture) == [0, 0, 0, 0, 0, 255, 0, 255, 100, 0, 0, 128])
        let probe = try DDSProbe(device: device)
        #expect(try probe.sample(texture, at: SIMD3(0.5, 1 / 6, 0)) == .zero)
        #expect(try probe.sample(texture, at: SIMD3(0.5, 5 / 6, 0)) == SIMD4(100 / 255, 0, 0, 128 / 255))
    }

    @Test("Upload transforms reject compressed, spatial, and HDR sources")
    func rejectsStoredLayouts() throws {
        let dds = DDSFixture.data(width: 4, height: 4, fourCC: "DXT1", payload: DDSFixture.colorBlock(first: 0xF800))
        let raster = KTX2Fixture.container(vkFormat: 37, width: 1, height: 1, levels: [[12, 34, 56, 255]])
        let cube = KTX2Fixture.container(vkFormat: 37, width: 1, height: 1,
                                         levels: [[UInt8](repeating: 255, count: 24)], faceCount: 6)
        let hdr = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n".utf8) + Data([128, 64, 32, 129])
        #expect(try loader.loadKTX2(data: cube).textureType == .typeCube)
        #expect(try loader.load(data: hdr).pixelFormat == .rgba16Float)
        #expect(Self.codes(try loader.load(data: raster, uploadOptions: .init())) == [12, 34, 56, 255])

        for (source, kind) in [(dds, "DDS"), (raster, "KTX2"), (cube, "KTX2"), (hdr, "HDR")] {
            let options: [TextureLoader.UploadOptions] = [
                .init(flipY: true), .init(alphaMode: .premultipliedEncoded), .init(alphaMode: .premultipliedLinear),
            ]
            for option in options {
                #expect(throws: TextureLoader.Failure.unsupportedUploadTransform(kind)) {
                    try loader.load(data: source, uploadOptions: option)
                }
            }
        }
    }

    private static func codes(_ texture: any MTLTexture) -> [UInt8] {
        let rowBytes = texture.width * 4
        var result = [UInt8](repeating: 0, count: rowBytes * texture.height)
        result.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!, bytesPerRow: rowBytes,
                             from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        }
        return result
    }

    private static func decodeSRGB(_ code: Float) -> Float {
        code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4)
    }
}

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3(x, y, z) }
}
