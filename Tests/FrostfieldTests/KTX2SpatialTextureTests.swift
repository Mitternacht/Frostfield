import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct KTX2SpatialTextureTests {
    let device: any MTLDevice
    let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    private static let axes: [SIMD3<Float>] = [
        SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
        SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1),
    ]
    private static let rgba: [[UInt8]] = [
        [128, 32, 210, 64], [16, 220, 55, 128], [170, 27, 66, 255],
        [54, 145, 250, 0], [239, 57, 127, 181], [70, 113, 43, 221],
    ]

    @Test("KTX2 ASTC cube samples each face and each stored mip with either transfer",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires ASTC sampling"))
    func astcCube() throws {
        let levels = (0..<4).map { mip in
            (0..<6).flatMap { face in
                Self.astc(Self.rgba[(face + mip) % 6], copies: mip == 0 ? 4 : 1)
            }
        }
        let data = KTX2Fixture.container(vkFormat: 158, width: 8, height: 8, levels: levels, faceCount: 6)
        let cube = try loader.loadKTX2(data: data, mipmapped: true)
        let linear = try loader.loadKTX2(data: data, interpretation: .linearData)
        #expect(cube.textureType == .typeCube && cube.mipmapLevelCount == 4 && cube.arrayLength == 1)
        #expect(cube.pixelFormat == .astc_4x4_srgb && linear.pixelFormat == .astc_4x4_ldr)
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("cube maps")) { try loader.load(data: data) }

        let probe = try DDSProbe(device: device)
        for mip in 0..<4 {
            for (face, axis) in Self.axes.enumerated() {
                let bytes = Self.rgba[(face + mip) % 6]
                let color = Self.sampleColor(bytes, sRGB: true)
                let raw = Self.sampleColor(bytes)
                #expect(simd_reduce_max(abs(try probe.sample(cube, at: axis, level: mip) - color)) < 0.006)
                #expect(simd_reduce_max(abs(try probe.sample(linear, at: axis, level: mip) - raw)) < 0.004)
            }
        }
    }

    @Test("A ZLIB KTX2 block array samples odd-sized block rows, separate layers, and terminal mips",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple2) == true, "Requires ASTC sampling"))
    func astcArray() throws {
        // 7 × 5 at 6 × 6 has two blocks per layer. A sample at the far
        // right of level zero selects the second block in that layer.
        let plain = (0..<3).map { mip in
            (0..<3).flatMap { layer in
                let index = (mip + 2 * layer) % 6
                return Self.astc(Self.rgba[index]) + (mip == 0 ? Self.astc(Self.rgba[(index + 1) % 6]) : [])
            }
        }
        let data = KTX2Fixture.container(
            vkFormat: 165, width: 7, height: 5, levels: try plain.map(KTX2Fixture.zlib),
            supercompression: 3, uncompressedLengths: plain.map(\.count), layerCount: 3
        )
        let array = try loader.loadKTX2(data: data, mipmapped: true)
        #expect(array.textureType == .type2DArray && array.arrayLength == 3 && array.mipmapLevelCount == 3)
        #expect(array.pixelFormat == .astc_6x6_ldr && array.width == 7 && array.height == 5)
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("texture arrays")) { try loader.load(data: data) }

        let probe = try DDSProbe(device: device)
        for mip in 0..<3 {
            for layer in 0..<3 {
                let index = (mip + 2 * layer) % 6
                let left = try probe.sample(array, at: SIMD3(0.25, 0.5, Float(layer)), level: mip)
                #expect(simd_reduce_max(abs(left - Self.sampleColor(Self.rgba[index]))) < 0.004)
                if mip == 0 {
                    let right = try probe.sample(array, at: SIMD3(6.5 / 7, 0.5, Float(layer)), level: mip)
                    #expect(simd_reduce_max(abs(right - Self.sampleColor(Self.rgba[(index + 1) % 6]))) < 0.004)
                }
            }
        }
    }

    @Test("Native BC1 cube keeps each compressed face and sub-block mip",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsBCTextureCompression == true, "Requires BC sampling"))
    func bcCube() throws {
        let codes: [UInt16] = [0xF800, 0x07E0, 0x001F, 0xFFE0, 0xF81F, 0x07FF]
        let colors: [SIMD4<Float>] = [
            SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1),
            SIMD4(1, 1, 0, 1), SIMD4(1, 0, 1, 1), SIMD4(0, 1, 1, 1),
        ]
        let levels = (0..<2).map { level in
            (0..<6).flatMap { face in DDSFixture.colorBlock(first: codes[(level + face) % 6]) }
        }
        let data = KTX2Fixture.container(vkFormat: 133, width: 4, height: 4, levels: levels, faceCount: 6)
        let cube = try loader.loadKTX2(data: data)
        #expect(cube.textureType == .typeCube && cube.mipmapLevelCount == 2 && cube.pixelFormat == .bc1_rgba)
        let probe = try DDSProbe(device: device)
        for mip in 0..<2 {
            for (face, axis) in Self.axes.enumerated() {
                #expect(try probe.sample(cube, at: axis, level: mip) == colors[(face + mip) % 6])
            }
        }
    }

    @Test("An uncompressed KTX2 HDR array generates mips independently for each layer")
    func hdrArray() throws {
        let colors: [SIMD4<Float>] = [SIMD4(-0.5, 4, 1.5, 1), SIMD4(2, -1, 0.5, 0.75), SIMD4(1, 0, -2, 0.5)]
        let base = colors.flatMap { value in
            let pixel = (0..<4).flatMap { withUnsafeBytes(of: Float16(value[$0]).bitPattern.littleEndian, Array.init) }
            return Array(repeating: pixel, count: 8).flatMap { $0 }
        }
        let data = KTX2Fixture.container(vkFormat: 97, width: 4, height: 2, typeSize: 2, levels: [base],
                                         levelCount: 0, layerCount: 3)
        let array = try loader.loadKTX2(data: data)
        #expect(array.textureType == .type2DArray && array.arrayLength == 3 && array.mipmapLevelCount == 3)
        #expect(array.pixelFormat == .rgba16Float)
        let probe = try DDSProbe(device: device)
        for mip in 0..<3 {
            for (layer, expected) in colors.enumerated() {
                #expect(try probe.sample(array, at: SIMD3(0.5, 0.5, Float(layer)), level: mip) == expected)
            }
        }
    }

    @Test("A singleton layer keeps its array type only through the explicit KTX2 entry point")
    func singleton() throws {
        let data = KTX2Fixture.container(vkFormat: 37, width: 1, height: 1, levels: [[40, 100, 250, 170]], layerCount: 1)
        let explicit = try loader.loadKTX2(data: data)
        let ordinary = try loader.load(data: data)
        #expect(explicit.textureType == .type2DArray && explicit.arrayLength == 1)
        #expect(ordinary.textureType == .type2D)
        let probe = try DDSProbe(device: device)
        let center = SIMD3<Float>(0.5, 0.5, 0)
        let result = try probe.sample(explicit, at: center)
        #expect(try probe.sample(ordinary, at: center) == result)
        #expect(simd_reduce_max(abs(result - Self.sampleColor([40, 100, 250, 170]))) < 0.0001)
    }

    @Test("KTX2 spatial validation checks slice counts, texture bounds, and independent mip ranges")
    func invalidSpatialLayouts() throws {
        let array = KTX2Fixture.container(vkFormat: 37, width: 2, height: 2,
                                         levels: [[UInt8](repeating: 1, count: 32), [UInt8](repeating: 2, count: 8)],
                                         layerCount: 2)
        var wrongLayers = array
        wrongLayers[32] = 3
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 0 needs 48 uncompressed bytes")) {
            try loader.loadKTX2(data: wrongLayers)
        }
        var overlap = array
        overlap.replaceSubrange(104..<112, with: array[80..<88])
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 1 is out of storage order or overlaps other data")) {
            try loader.loadKTX2(data: overlap)
        }
        var unaligned = array
        unaligned[80] += 1
        unaligned.append(0)
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 0 has an unaligned byte offset")) {
            try loader.loadKTX2(data: unaligned)
        }
        let cube = [UInt8](repeating: 0, count: 96)
        #expect(throws: TextureLoader.Failure.invalidKTX2("level 0 needs 96 uncompressed bytes")) {
            try loader.loadKTX2(data: KTX2Fixture.container(
                vkFormat: 157, width: 4, height: 4, levels: [Array(cube.dropLast(16))], faceCount: 6
            ))
        }
        #expect(throws: TextureLoader.Failure.invalidKTX2("block-compressed spatial textures require a stored mip level")) {
            try loader.loadKTX2(data: KTX2Fixture.container(
                vkFormat: 157, width: 4, height: 4, levels: [cube], levelCount: 0, faceCount: 6
            ))
        }
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("cube map arrays")) {
            try loader.loadKTX2(data: KTX2Fixture.container(
                vkFormat: 157, width: 4, height: 4, levels: [cube], faceCount: 6, layerCount: 1
            ))
        }
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("incomplete cube maps")) {
            try loader.loadKTX2(data: KTX2Fixture.container(
                vkFormat: 37, width: 1, height: 1, levels: [[0, 0, 0, 0]], layerCount: 1,
                keyValues: [("KTXcubemapIncomplete", String(UnicodeScalar(1)))]
            ))
        }
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("more than 2048 array layers")) {
            try loader.loadKTX2(data: KTX2Fixture.container(
                vkFormat: 37, width: 1, height: 1, levels: [[0, 0, 0, 0]], layerCount: 2_049
            ))
        }
        #expect(throws: TextureLoader.Failure.imageDimensions(width: 8_192, height: 1_025)) {
            try loader.loadKTX2(data: KTX2Fixture.container(
                vkFormat: 37, width: 8_192, height: 1_025, levels: [[0, 0, 0, 0]], layerCount: 2
            ))
        }
    }

    private static func astc(_ rgba: [UInt8], copies: Int = 1) -> [UInt8] {
        let block = KTX2Fixture.astcConstantBlock(rgba.map { UInt16($0) * 257 })
        return Array(repeating: block, count: copies).flatMap { $0 }
    }

    private static func sampleColor(_ bytes: [UInt8], sRGB: Bool = false) -> SIMD4<Float> {
        let normalized = bytes.map { Float($0) / 255 }
        func rgb(_ code: Float) -> Float {
            guard sRGB else { return code }
            return code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4)
        }
        return SIMD4(rgb(normalized[0]), rgb(normalized[1]), rgb(normalized[2]), normalized[3])
    }
}
