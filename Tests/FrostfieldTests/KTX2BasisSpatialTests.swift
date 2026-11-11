import CryptoKit
import Foundation
@testable import Frostfield
import FrostfieldBasis
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct KTX2BasisSpatialTests {
    let device: any MTLDevice
    let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    @Test("BasisLZ and Zstandard UASTC arrays sample independent layer and mip texels",
          arguments: [true, false])
    func arrays(etc1s: Bool) throws {
        let data = try Self.data(etc1s ? Self.etc1s : Self.uastc)
        let raw = try loader.loadKTX2(data: data, interpretation: .linearData, mipmapped: true)
        let color = try loader.loadKTX2(data: data)
        #expect(raw.textureType == .type2DArray && raw.arrayLength == 3 && raw.mipmapLevelCount == 4)
        #expect(raw.width == 8 && raw.height == 8 && raw.pixelFormat == .rgba8Unorm)
        #expect(color.pixelFormat == .rgba8Unorm_srgb)
        #expect(throws: Self.unsupported) { try loader.load(data: data) }

        let probe = try DDSProbe(device: device)
        let samples = etc1s ? Self.etc1sSamples : Self.uastcSamples
        for (mip, layers) in samples.enumerated() {
            for (layer, rgba) in layers.enumerated() {
                let texel = rgba / 255
                let expected = SIMD4(Self.linear(texel.x), Self.linear(texel.y), Self.linear(texel.z), texel.w)
                let at = SIMD3<Float>(0.5, 0.5, Float(layer))
                #expect(simd_reduce_max(abs(try probe.sample(raw, at: at, level: mip) - texel)) < 0.005)
                #expect(simd_reduce_max(abs(try probe.sample(color, at: at, level: mip) - expected)) < 0.008)
            }
        }
    }

    @Test("A raw UASTC singleton remains a one-layer array through explicit loading")
    func singleton() throws {
        let data = try Self.data(Self.singleton)
        let texture = try loader.loadKTX2(data: data, interpretation: .linearData, mipmapped: true)
        #expect(texture.textureType == .type2DArray && texture.arrayLength == 1)
        #expect(texture.mipmapLevelCount == 4 && texture.pixelFormat == .rgba8Unorm)
        #expect(throws: Self.unsupported) { try loader.load(data: data) }

        let probe = try DDSProbe(device: device)
        for level in 0..<4 {
            let actual = try probe.sample(texture, at: SIMD3(0.5, 0.5, 0), level: level)
            #expect(simd_reduce_max(abs(actual - SIMD4<Float>(200, 32, 20, 51) / 255)) < 0.005)
        }
    }

    @Test("A raw UASTC cubemap samples distinct faces and stored mips")
    func cube() throws {
        let data = try Self.data(Self.cube)
        let texture = try loader.loadKTX2(data: data, interpretation: .linearData)
        #expect(texture.textureType == .typeCube && texture.arrayLength == 1)
        #expect(texture.mipmapLevelCount == 3 && texture.pixelFormat == .rgba8Unorm)
        #expect(throws: Self.unsupported) { try loader.load(data: data) }

        let faces: [SIMD3<Float>] = [
            SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
            SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1),
        ]
        let samples: [[SIMD4<Float>]] = [
            [SIMD4(200, 32, 20, 51), SIMD4(25, 172, 71, 163), SIMD4(52, 96, 229, 236), SIMD4(185, 222, 35, 101), SIMD4(117, 61, 165, 149), SIMD4(92, 182, 226, 211)],
            [SIMD4(25, 172, 71, 163), SIMD4(52, 96, 229, 236), SIMD4(185, 222, 35, 101), SIMD4(117, 61, 165, 149), SIMD4(92, 182, 226, 211), SIMD4(200, 32, 20, 51)],
            [SIMD4(52, 96, 229, 236), SIMD4(185, 222, 35, 101), SIMD4(117, 61, 165, 149), SIMD4(92, 182, 226, 211), SIMD4(200, 32, 20, 51), SIMD4(25, 172, 71, 163)],
        ]
        let probe = try DDSProbe(device: device)
        for (mip, colors) in samples.enumerated() {
            for (axis, rgba) in zip(faces, colors) {
                let actual = try probe.sample(texture, at: axis, level: mip)
                #expect(simd_reduce_max(abs(actual - rgba / 255)) < 0.005)
            }
        }
    }

    @Test("Basis KTX2 rejects invalid spatial ranges and unsupported layouts")
    func malformed() throws {
        let etc1s = try Self.data(Self.etc1s)
        let uastc = try Self.data(Self.uastc)

        var cubeArray = uastc
        cubeArray[36] = 6
        #expect(throws: Self.unsupported) { try loader.loadKTX2(data: cubeArray) }

        var extraLayer = uastc
        extraLayer[32] = 4
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: extraLayer) }

        var overlap = uastc
        overlap.replaceSubrange(104..<112, with: uastc[80..<88])
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: overlap) }

        let global = etc1s.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 64, as: UInt64.self))) }
        let secondLayer = global + 40
        var invalidSlice = etc1s
        invalidSlice.replaceSubrange((secondLayer + 4)..<(secondLayer + 8), with: [255, 255, 255, 255])
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: invalidSlice) }

        var missingAlpha = etc1s
        missingAlpha.replaceSubrange((secondLayer + 12)..<(secondLayer + 20), with: Array(repeating: 0, count: 8))
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: missingAlpha) }

        var predicted = etc1s
        predicted[secondLayer] = 2
        #expect(throws: Self.unsupported) { try loader.loadKTX2(data: predicted) }
    }

    @Test("Native Basis array upload preserves exact per-layer and per-mip blocks",
          .enabled(if: supportsNativeBasis, "Requires ASTC, ETC2, or BC7"),
          arguments: [true, false])
    func nativeArrays(etc1s: Bool) throws {
        let data = try Self.data(etc1s ? Self.etc1s : Self.uastc)
        let raw = try loader.loadKTX2(data: data, interpretation: .linearData, mipmapped: true,
                                      basisStorage: .deviceCompressed)
        let color = try loader.loadKTX2(data: data, basisStorage: .deviceCompressed)
        let rgba = try loader.loadKTX2(data: data)
        let usesAppleBlocks = device.supportsFamily(.apple2)
        let expected: MTLPixelFormat = usesAppleBlocks
            ? (etc1s ? .eac_rgba8 : .astc_4x4_ldr) : .bc7_rgbaUnorm
        let expectedSRGB: MTLPixelFormat = usesAppleBlocks
            ? (etc1s ? .eac_rgba8_srgb : .astc_4x4_srgb) : .bc7_rgbaUnorm_srgb
        #expect(raw.textureType == .type2DArray && raw.arrayLength == 3 && raw.mipmapLevelCount == 4)
        #expect(raw.pixelFormat == expected && color.pixelFormat == expectedSRGB)
        #expect(rgba.pixelFormat == .rgba8Unorm_srgb)
        let reference: String
        if usesAppleBlocks {
            reference = etc1s
                ? "68f1755bc623cf45a943454633da18c83722999b838144b9f4ab9e8d0b8f895a"
                : "f98265e43dce405ea769efe4208b8290b7efb472dfb012f03fb3bdd04a1745d9"
        } else {
            reference = etc1s
                ? "20412e10e19eef9fd0e2af3e3146771c5d1c009b7cea81c0cf69280a46d98011"
                : "6cb709f2b956d367be1bbf0366ff6b1ef441c4b3cd4f9ecf017175579202fb8c"
        }
        #expect(Self.blockDigest(raw, slices: 3, bytesPerBlock: 16) == reference)
        #expect(Self.blockDigest(color, slices: 3, bytesPerBlock: 16) == reference)

        let probe = try DDSProbe(device: device)
        let samples = etc1s ? Self.etc1sSamples : Self.uastcSamples
        for (mip, layers) in samples.enumerated() {
            for (layer, rgba) in layers.enumerated() {
                let at = SIMD3<Float>(0.5, 0.5, Float(layer))
                let sample = try probe.sample(raw, at: at, level: mip)
                #expect(simd_reduce_max(abs(sample - rgba / 255)) < 0.035)
                #expect(abs(sample.w - rgba.w / 255) < 0.012)
                let expected = SIMD4(Self.linear(sample.x), Self.linear(sample.y), Self.linear(sample.z), sample.w)
                #expect(simd_reduce_max(abs(try probe.sample(color, at: at, level: mip) - expected)) < 0.008)
            }
        }
    }

    @Test("Native compressed Basis cube preserves all six faces and sub-block mips",
          .enabled(if: supportsNativeBasis, "Requires ASTC or BC7"))
    func nativeCube() throws {
        let data = try Self.data(Self.cube)
        let raw = try loader.loadKTX2(data: data, interpretation: .linearData, basisStorage: .deviceCompressed)
        let rgba = try loader.loadKTX2(data: data, interpretation: .linearData)
        let usesAppleBlocks = device.supportsFamily(.apple2)
        #expect(raw.textureType == .typeCube && raw.mipmapLevelCount == 3)
        #expect(raw.pixelFormat == (usesAppleBlocks ? .astc_4x4_ldr : .bc7_rgbaUnorm))
        let reference = usesAppleBlocks
            ? "b7efb4b2c92c39e65f2e52f34e76d18523812e60c8e06542c0f14ff7df67aa20"
            : "28823dd255c56c27cf96ab5a6afd3c7a597514d8ddcede6eed525192c4afb7fe"
        #expect(Self.blockDigest(raw, slices: 6, bytesPerBlock: 16) == reference)
        let axes: [SIMD3<Float>] = [
            SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
            SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1),
        ]
        let probe = try DDSProbe(device: device)
        for mip in 0..<3 {
            for axis in axes {
                let expected = try probe.sample(rgba, at: axis, level: mip)
                #expect(simd_reduce_max(abs(try probe.sample(raw, at: axis, level: mip) - expected)) < 0.035)
            }
        }
    }

    @Test("Opaque ETC1S transcodes without alpha blocks and preserves odd mip footprints",
          .enabled(if: supportsNativeBasis, "Requires ETC2 or BC7"))
    func nativeOpaque() throws {
        let data = try Self.data(Self.opaque)
        let raw = try loader.loadKTX2(data: data, basisStorage: .deviceCompressed)
        let srgb = try loader.loadKTX2(data: data, interpretation: .encodedSRGB, basisStorage: .deviceCompressed)
        let usesAppleBlocks = device.supportsFamily(.apple2)
        #expect(raw.textureType == .type2D && raw.width == 7 && raw.height == 5 && raw.mipmapLevelCount == 3)
        #expect(raw.pixelFormat == (usesAppleBlocks ? .etc2_rgb8 : .bc7_rgbaUnorm))
        #expect(srgb.pixelFormat == (usesAppleBlocks ? .etc2_rgb8_srgb : .bc7_rgbaUnorm_srgb))
        let reference = usesAppleBlocks
            ? "7d7aca4afd4b495568a36caaa221329efdc8f18d9c191faf0d03af96952c024a"
            : "b7edb406733901524851f113ab15c5355287cadd4a6fd197f43832bae2314377"
        #expect(Self.blockDigest(raw, slices: 1, bytesPerBlock: usesAppleBlocks ? 8 : 16) == reference)

        let probe = try DDSProbe(device: device)
        let colors: [SIMD4<Float>] = [SIMD4(179, 39, 72, 255), SIMD4(26, 158, 59, 255), SIMD4(39, 72, 196, 255)]
        for (mip, rgba) in colors.enumerated() {
            let at = SIMD3<Float>(0.5, 0.5, 0)
            let sample = try probe.sample(raw, at: at, level: mip)
            #expect(simd_reduce_max(abs(sample - rgba / 255)) < 0.035)
            #expect(sample.w == 1)
            let expected = SIMD4(Self.linear(sample.x), Self.linear(sample.y), Self.linear(sample.z), 1)
            #expect(simd_reduce_max(abs(try probe.sample(srgb, at: at, level: mip) - expected)) < 0.008)
        }
    }

    @Test("Native Basis storage reports generation and size failures explicitly",
          .enabled(if: supportsNativeBasis, "Requires ASTC, ETC2, or BC7"))
    func nativeFailures() throws {
        let data = try Self.data(Self.singleton)
        #expect(throws: TextureLoader.Failure.mipmapGeneration) {
            try loader.loadKTX2(data: data, mipmapped: true, basisStorage: .deviceCompressed)
        }
        #expect(try loader.loadKTX2(data: data, mipmapped: true).mipmapLevelCount == 4)
        var wrongSize = data
        wrongSize[96] += 16
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: wrongSize, basisStorage: .deviceCompressed) }
    }

    @Test("Basis target selection prefers Apple block formats, then BC7")
    func targetSelection() {
        typealias Target = BasisKTX2.Target
        #expect(Target.deviceCompressed(isUASTC: true, hasAlpha: true, supportsAppleBlockFormats: true, supportsBC: true) == .astc4x4)
        #expect(Target.deviceCompressed(isUASTC: false, hasAlpha: true, supportsAppleBlockFormats: true, supportsBC: true) == .etcRGBA)
        #expect(Target.deviceCompressed(isUASTC: false, hasAlpha: false, supportsAppleBlockFormats: true, supportsBC: false) == .etcRGB)
        #expect(Target.deviceCompressed(isUASTC: true, hasAlpha: true, supportsAppleBlockFormats: false, supportsBC: true) == .bc7)
        #expect(Target.deviceCompressed(isUASTC: false, hasAlpha: true, supportsAppleBlockFormats: false, supportsBC: false) == nil)
    }

    @Test("The BC7 fallback bridge produces native GPU-sampleable blocks",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsBCTextureCompression == true, "Requires BC7"))
    func bc7Bridge() throws {
        let data = try Self.data(Self.etc1s)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bc7_rgbaUnorm, width: 8, height: 8, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let plane = try #require(device.makeTexture(descriptor: descriptor))
        try data.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
            var info = FrostfieldBasisInfo()
            var status: Int32 = 0
            let decoder = try #require(frostfield_basis_open(source.baseAddress, source.count, 1, &info, &status))
            defer { frostfield_basis_close(decoder) }
            var digest = SHA256()
            for mip in 0..<4 {
                let extent = max(1, 8 >> mip)
                let row = ((extent + 3) / 4) * 16
                var blocks = Data(count: ((extent + 3) / 4) * row)
                for layer in 0..<3 {
                    blocks.withUnsafeMutableBytes { (output: UnsafeMutableRawBufferPointer) in
                        #expect(frostfield_basis_transcode(
                            decoder, FrostfieldBasisBC7, UInt32(mip), UInt32(layer), 0,
                            output.baseAddress, output.count
                        ) == 0)
                        if mip == 0, layer == 0 {
                            plane.replace(region: MTLRegionMake2D(0, 0, 8, 8), mipmapLevel: 0,
                                          withBytes: output.baseAddress!, bytesPerRow: row)
                        }
                    }
                    digest.update(data: blocks)
                }
            }
            #expect(Self.hex(digest.finalize()) == "20412e10e19eef9fd0e2af3e3146771c5d1c009b7cea81c0cf69280a46d98011")
        }
        let probe = try DDSProbe(device: device)
        let color = try probe.sample(plane, at: SIMD3(0.5, 0.5, 0))
        #expect(simd_reduce_max(abs(color - Self.etc1sSamples[0][0] / 255)) < 0.035)
    }

    private static var supportsNativeBasis: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple2) || device.supportsBCTextureCompression
    }

    /// References are from the pinned transcoder's direct per-image block output.
    private static func blockDigest(_ texture: any MTLTexture, slices: Int, bytesPerBlock: Int) -> String {
        var digest = SHA256()
        for mip in 0..<texture.mipmapLevelCount {
            let w = max(1, texture.width >> mip), h = max(1, texture.height >> mip)
            let row = ((w + 3) / 4) * bytesPerBlock
            var bytes = Data(count: row * ((h + 3) / 4))
            for slice in 0..<slices {
                bytes.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                    texture.getBytes(raw.baseAddress!, bytesPerRow: row, bytesPerImage: raw.count,
                                     from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: mip, slice: slice)
                }
                digest.update(data: bytes)
            }
        }
        return hex(digest.finalize())
    }

    private static func hex(_ digest: SHA256Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    private static let invalid = TextureLoader.Failure.invalidKTX2("invalid ETC1S or UASTC Basis Universal data")
    private static let unsupported = TextureLoader.Failure.unsupportedKTX2(
        "Basis Universal layout, transfer function, alpha mode, or supercompression"
    )

    private static func data(_ text: String) throws -> Data {
        try #require(Data(base64Encoded: text, options: .ignoreUnknownCharacters))
    }

    private static func linear(_ code: Float) -> Float {
        code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4)
    }

    // Four authored mip levels and three distinct straight-alpha RGBA layers,
    // encoded with Khronos KTX-Software 4.4.0. Sample codes come from its
    // reference "ktx transcode --target rgba8" output.
    private static let etc1sSamples: [[SIMD4<Float>]] = [
        [SIMD4(198, 33, 17, 51), SIMD4(24, 173, 74, 163), SIMD4(53, 95, 235, 236)],
        [SIMD4(24, 173, 74, 163), SIMD4(53, 95, 235, 236), SIMD4(183, 224, 35, 101)],
        [SIMD4(53, 95, 235, 236), SIMD4(183, 224, 35, 101), SIMD4(198, 33, 17, 51)],
        [SIMD4(183, 224, 35, 101), SIMD4(198, 33, 17, 51), SIMD4(24, 173, 74, 163)],
    ]
    private static let uastcSamples: [[SIMD4<Float>]] = [
        [SIMD4(200, 32, 20, 51), SIMD4(25, 172, 71, 163), SIMD4(52, 96, 229, 236)],
        [SIMD4(25, 172, 71, 163), SIMD4(52, 96, 229, 236), SIMD4(185, 222, 35, 101)],
        [SIMD4(52, 96, 229, 236), SIMD4(185, 222, 35, 101), SIMD4(200, 32, 20, 51)],
        [SIMD4(185, 222, 35, 101), SIMD4(200, 32, 20, 51), SIMD4(25, 172, 71, 163)],
    ]
    private static let etc1s = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAIAAAACAAAAAAAAAADAAAAAQAAAAQAAAABAAAAsAAAADwAAADsAAAAMAAAACAB
        AAAAAAAAgAEAAAAAAACyAgAAAAAAAA0AAAAAAAAAAAAAAAAAAACsAgAAAAAAAAYAAAAAAAAAAAAAAAAAAACmAgAA
        AAAAAAYAAAAAAAAAAAAAAAAAAACgAgAAAAAAAAYAAAAAAAAAAAAAAAAAAAA8AAAAAAAAAAIAOACjAQIAAwMAAAgI
        AAAAAAAAAAA/AAAAAAAAAAAA/////0AAPw8AAAAAAAAAAP////8sAAAAS1RYd3JpdGVyAGt0eCBjcmVhdGUgdjQu
        NC4wIC8gbGlia3R4IHY0LjQuMAAAAAAACAADAD4AAAANAAAAMQAAAAAAAAAAAAAAAAAAAAIAAAACAAAAAgAAAAAA
        AAAEAAAAAgAAAAYAAAADAAAAAAAAAAkAAAACAAAACwAAAAIAAAAAAAAAAAAAAAEAAAABAAAAAQAAAAAAAAACAAAA
        AQAAAAMAAAABAAAAAAAAAAQAAAABAAAABQAAAAEAAAAAAAAAAAAAAAEAAAABAAAAAQAAAAAAAAACAAAAAQAAAAMA
        AAABAAAAAAAAAAQAAAABAAAABQAAAAEAAAAAAAAAAAAAAAEAAAABAAAAAQAAAAAAAAACAAAAAQAAAAMAAAABAAAA
        AAAAAAQAAAABAAAABQAAAAEAAAAdwBuAAACAQWzGjnSkJ6oA4gAIAAAAwTxySgYCxAAYAAAAhEGStSRIjgBEARAA
        AACAyID5yh7lIMHiY33M+vz///+vqqqqUlVVVQUAwUQAAAAAAAAS4b8VAvAABAAAAIgAImBOwAAAgDAAC+XknIRJ
        QQCYABAAAAAAQAQCBw8BCbU9swsHDwEJtT2zCwcPoDmoOfQAPM0B8gCqOQ==
        """
    private static let uastc = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAIAAAACAAAAAAAAAADAAAAAQAAAAQAAAACAAAAsAAAACwAAADcAAAAUAAAAAAA
        AAAAAAAAAAAAAAAAAACtAQAAAAAAAC4AAAAAAAAAwAAAAAAAAACBAQAAAAAAACwAAAAAAAAAMAAAAAAAAABWAQAA
        AAAAACsAAAAAAAAAMAAAAAAAAAAsAQAAAAAAACoAAAAAAAAAMAAAAAAAAAAsAAAAAAAAAAIAKACmAQIAAwMAABAA
        AAAAAAAAAAB/AwAAAAAAAAAA/////ywAAABLVFh3cml0ZXIAa3R4IGNyZWF0ZSB2NC40LjAgLyBsaWJrdHggdjQu
        NC4wABsAAABLVFh3cml0ZXJTY1BhcmFtcwAtLXpzdGQgNQAAKLUv/SAwDQEAuDfXe6RMViwAFxmEYia1AjeD9Wh0
        KHcBAwCA8JnFEIALKLUv/SAwFQEAwJcGrJy9HggDADfXe6RMViwAFxmEYia1AgMAM4sBhGfgCCi1L/0gMB0BAMg3
        g/VodCh3AQCXBqycvR4IAzfXe6RMViwAAwCA8JmEZ+AIKLUv/SDALQEAwBcZhGImtQIAN4P1aHQodwGXBqycvR4I
        AwQABScvOLmZMRCACw==
        """
    private static let cube = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAEAAAABAAAAAAAAAAAAAAABgAAAAMAAAAAAAAAmAAAACwAAADEAAAAMAAAAAAA
        AAAAAAAAAAAAAAAAAADAAQAAAAAAAGAAAAAAAAAAYAAAAAAAAABgAQAAAAAAAGAAAAAAAAAAYAAAAAAAAAAAAQAA
        AAAAAGAAAAAAAAAAYAAAAAAAAAAsAAAAAAAAAAIAKACmAQIAAwMAABAAAAAAAAAAAAB/AwAAAAAAAAAA/////ywA
        AABLVFh3cml0ZXIAa3R4IGNyZWF0ZSB2NC40LjAgLyBsaWJrdHggdjQuNC4wAAAAAAAAAAAAAAAAAJcGrJy9HggD
        AAAAAAAAAAA313ukTFYsAAAAAAAAAAAAt66ntDI/oAEAAAAAAAAAAJfLVnw6J48CAAAAAAAAAAAXGYRiJrUCAAAA
        AAAAAAAAN4P1aHQodwEAAAAAAAAAADeD9Wh0KHcBAAAAAAAAAACXBqycvR4IAwAAAAAAAAAAN9d7pExWLAAAAAAA
        AAAAALeup7QyP6ABAAAAAAAAAACXy1Z8OiePAgAAAAAAAAAAFxmEYia1AgAAAAAAAAAAABcZhGImtQIAAAAAAAAA
        AAA3g/VodCh3AQAAAAAAAAAAlwasnL0eCAMAAAAAAAAAADfXe6RMViwAAAAAAAAAAAC3rqe0Mj+gAQAAAAAAAAAA
        l8tWfDonjwIAAAAAAAAAAA==
        """
    private static let opaque = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAHAAAABQAAAAAAAAAAAAAAAQAAAAMAAAABAAAAmAAAACwAAADEAAAAMAAAAPgA
        AAAAAAAAuAAAAAAAAACyAQAAAAAAAAIAAAAAAAAAAAAAAAAAAACxAQAAAAAAAAEAAAAAAAAAAAAAAAAAAACwAQAA
        AAAAAAEAAAAAAAAAAAAAAAAAAAAsAAAAAAAAAAIAKACjAQEAAwMAAAgAAAAAAAAAAAA/AAAAAAAAAAAA/////ywA
        AABLVFh3cml0ZXIAa3R4IGNyZWF0ZSB2NC40LjAgLyBsaWJrdHggdjQuNC4wAAAAAAADAAIAMAAAAAkAAAAvAAAA
        AAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAB9A
        hAABAAAASBmgHKABxA0QAAAAgGw+oDgAEwEAAAAAAEiBAGACAAAAAAAAAY6jAFRVVVWtqqqqAgDBRAAAAAAAABLh
        v9UAMAEAAAAAAIJsBIgaMAAAAAgaRiMnEABMAAAAAAAAIIAAAR8iDw==
        """
    private static let singleton = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAIAAAACAAAAAAAAAABAAAAAQAAAAEAAAAAAAAAaAAAACwAAACUAAAAMAAAAAAA
        AAAAAAAAAAAAAAAAAADQAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAAsAAAAAAAAAAIAKACmAQIAAwMAABAAAAAAAAAA
        AAB/AwAAAAAAAAAA/////ywAAABLVFh3cml0ZXIAa3R4IGNyZWF0ZSB2NC40LjAgLyBsaWJrdHggdjQuNC4wAAAA
        AAAAAAAAAAAAABcZhGImtQIAAAAAAAAAAAAXGYRiJrUCAAAAAAAAAAAAFxmEYia1AgAAAAAAAAAAABcZhGImtQIA
        AAAAAAAAAAA=
        """
}
