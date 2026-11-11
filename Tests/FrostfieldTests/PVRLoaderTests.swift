import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PVRLoaderTests {
    let device: any MTLDevice
    let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    @Test("Legacy PVR reads RGB and RGBA at both PVRTC1 bit rates", arguments: [2, 4])
    func legacyFormats(bits: Int) throws {
        // Constant translucent red, stored in both endpoint words. PVRTC1
        // RGB sampling ignores the alpha endpoint and returns opaque red.
        let color: UInt16 = 0x4F00
        let width = bits == 2 ? 16 : 8
        let base = PVRFixture.blocks(color, count: width * width * bits / 64)
        let terminal = PVRFixture.blocks(0x83E0, count: 4)
        let probe = try DDSProbe(device: device)
        let center = SIMD3<Float>(0.5, 0.5, 0)
        for hasAlpha in [false, true] {
            let data = PVRFixture.v2(width: width, height: width, bits: bits, alpha: hasAlpha,
                                     levels: [[base], [terminal]])
            let texture = try loader.load(data: data)
            #expect(texture.textureType == .type2D && texture.mipmapLevelCount == 2)
            #expect(texture.pixelFormat == .rgba8Unorm)
            let value = try probe.sample(texture, at: center)
            #expect(value.x > 0.98 && value.y == 0 && value.z == 0)
            #expect(hasAlpha ? value.w > 0.5 && value.w < 0.6 : value.w == 1)
            #expect(try probe.sample(texture, at: center, level: 1) == SIMD4(0, 1, 0, 1))

            let nativeBase = try loader.loadPVR(data: data, loadMipmaps: false)
            #expect(nativeBase.mipmapLevelCount == 1)
            #expect((nativeBase.pixelFormat != .rgba8Unorm) == device.supportsFamily(.apple2))
            let nativeValue = try probe.sample(nativeBase, at: center)
            #expect(simd_reduce_max(abs(nativeValue - value)) < 0.008, "RGBA \(hasAlpha), native \(nativeValue), CPU \(value)")
        }
    }

    @Test("Both PVR v3 header orders honor sRGB and unaligned metadata", arguments: [false, true])
    func v3Color(bigEndian: Bool) throws {
        let payload = PVRFixture.blocks(0xC200, count: 4)
        let meta = PVRFixture.metadata(key: 3, bytes: [0, 0, 0], bigEndian: bigEndian)
        let data = PVRFixture.v3(width: 8, height: 8, format: 2, sRGB: true, bigEndian: bigEndian,
                                 metadata: meta, levels: [[payload]])
        let texture = try loader.load(data: data)
        let linear = try loader.loadPVR(data: data, interpretation: .linearData)
        let probe = try DDSProbe(device: device)
        let center = SIMD3<Float>(0.5, 0.5, 0)
        #expect(texture.pixelFormat == (device.supportsFamily(.apple2) ? .pvrtc_rgb_4bpp_srgb : .rgba8Unorm_srgb))
        #expect(abs(try probe.sample(texture, at: center).x - 0.23074) < 0.01)
        #expect(abs(try probe.sample(linear, at: center).x - 0.51765) < 0.01)
    }

    @Test("PVR routes six cube faces and distinct mips into Metal", arguments: [0, 1, 2])
    func cube(layout: Int) throws {
        let colors: [UInt16] = [0xFC00, 0x83E0, 0x801E, 0xFFE0, 0xFC1E, 0x83FE]
        let base = colors.map { PVRFixture.blocks($0, count: 4) }
        let final = colors.reversed().map { PVRFixture.blocks($0, count: 4) }
        let data: Data
        if layout == 2 {
            let fileOrder = [4, 0, 3, 5, 2, 1]
            let labels = Array("XxYyZz".utf8)
            let meta = PVRFixture.metadata(key: 2, bytes: fileOrder.map { labels[$0] }, bigEndian: true)
            data = PVRFixture.v3(width: 8, height: 8, format: 3, bigEndian: true, metadata: meta,
                                 levels: [fileOrder.map { base[$0] }, fileOrder.map { final[$0] }])
        } else {
            data = PVRFixture.v2(width: 8, height: 8, bits: 4, alpha: true, cubeFlag: layout == 1,
                                 levels: [base, final])
        }
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("pvr")
        defer { try? FileManager.default.removeItem(at: path) }
        try data.write(to: path)
        let texture = try loader.loadPVR(contentsOf: path)
        let top = try loader.loadPVR(data: data, loadMipmaps: false)
        #expect(texture.textureType == .typeCube && texture.mipmapLevelCount == 2)
        #expect(top.textureType == .typeCube && top.mipmapLevelCount == 1)
        #expect(throws: TextureLoader.Failure.unsupportedPVR("cube maps through the 2D loader")) {
            try loader.load(data: data)
        }

        let directions = PVRFixture.directions
        let expected: [SIMD4<Float>] = [
            SIMD4(1,0,0,1), SIMD4(0,1,0,1), SIMD4(0,0,1,1),
            SIMD4(1,1,0,1), SIMD4(1,0,1,1), SIMD4(0,1,1,1),
        ]
        let probe = try DDSProbe(device: device)
        for (face, direction) in directions.enumerated() {
            #expect(try probe.sample(texture, at: direction) == expected[face])
            #expect(try probe.sample(top, at: direction) == expected[face])
            #expect(try probe.sample(texture, at: direction, level: 1) == expected[5 - face])
        }
    }

    @Test("PVR rejects invalid headers, counts, metadata, and incomplete payloads before upload")
    func malformed() throws {
        let blocks = PVRFixture.blocks(0xFC00, count: 4)
        let image = PVRFixture.v3(width: 8, height: 8, format: 2, levels: [[blocks]])
        let cube = PVRFixture.v3(width: 8, height: 8, format: 2,
                                 metadata: PVRFixture.metadata(key: 2, bytes: Array("XxYyZZ".utf8)),
                                 levels: [Array(repeating: blocks, count: 6)])
        let v2 = PVRFixture.v2(width: 8, height: 8, bits: 4, alpha: false, levels: [[blocks]])
        let invalid = [
            Data(image.prefix(51)), Data(image.dropLast()), image + Data([0]),
            PVRFixture.patched(image, word: 12, value: .max), // metadata length
            PVRFixture.patched(image, word: 11, value: 0),    // mip count
            PVRFixture.patched(image, word: 11, value: .max),
            PVRFixture.patched(image, word: 7, value: 7),     // width
            PVRFixture.patched(image, word: 10, value: 5),    // face count
            PVRFixture.patched(image, word: 9, value: 2),     // array
            PVRFixture.patched(image, word: 8, value: 2),     // depth
            PVRFixture.patched(image, word: 3, value: 1),     // high format bits
            PVRFixture.patched(image, word: 2, value: 4),     // PVRTC-II
            PVRFixture.patched(image, word: 1, value: 2),     // premultiplied
            PVRFixture.patched(v2, word: 5, value: 31),       // legacy payload length
            PVRFixture.patched(v2, word: 6, value: 2),        // legacy bit rate
            cube,
            PVRFixture.v3(width: 8, height: 8, format: 2, metadata: Data([1, 2]), levels: [[blocks]]),
            PVRFixture.v3(width: 8, height: 8, format: 2,
                           metadata: PVRFixture.metadata(key: 3, bytes: [0, 1, 0]), levels: [[blocks]]),
            PVRFixture.v3(width: 8, height: 8, format: 2,
                           metadata: PVRFixture.patched(PVRFixture.metadata(key: 5, bytes: [0]), word: 2, value: .max),
                           levels: [[blocks]]),
        ]
        for (index, data) in invalid.enumerated() {
            #expect(throws: TextureLoader.Failure.self, "Malformed fixture \(index)") {
                try loader.loadPVR(data: data, loadMipmaps: false)
            }
        }
    }
}

private enum PVRFixture {
    static let directions: [SIMD3<Float>] = [
        SIMD3(1,0,0), SIMD3(-1,0,0), SIMD3(0,1,0), SIMD3(0,-1,0), SIMD3(0,0,1), SIMD3(0,0,-1),
    ]

    static func blocks(_ endpoint: UInt16, count: Int) -> Data {
        let lo = UInt8(truncatingIfNeeded: endpoint), hi = UInt8(endpoint >> 8)
        return Data(Array(repeating: [0, 0, 0, 0, lo, hi, lo, hi], count: count).joined())
    }

    static func metadata(key: UInt32, bytes: [UInt8], bigEndian: Bool = false) -> Data {
        words([0x03525650, key, UInt32(bytes.count)], bigEndian: bigEndian) + Data(bytes)
    }

    static func v2(
        width: Int, height: Int, bits: Int, alpha: Bool, cubeFlag: Bool = false, levels: [[Data]]
    ) -> Data {
        let faces = levels[0].count
        var payload = Data()
        if cubeFlag {
            for face in 0..<faces { for level in levels { payload.append(level[face]) } }
        } else {
            for level in levels { for face in level { payload.append(face) } }
        }
        var flags: UInt32 = bits == 2 ? 24 : 25
        if alpha { flags |= 0x8000 }
        if levels.count > 1 { flags |= 0x0100 }
        if cubeFlag { flags |= 0x1000 }
        return words([
            52, UInt32(height), UInt32(width), UInt32(levels.count - 1), flags,
            UInt32(cubeFlag ? payload.count / faces : payload.count), UInt32(bits),
            0, 0, 0, alpha ? 1 : 0, 0x21525650, UInt32(faces),
        ]) + payload
    }

    static func v3(
        width: Int, height: Int, format: UInt32, sRGB: Bool = false, bigEndian: Bool = false,
        metadata: Data = Data(), levels: [[Data]]
    ) -> Data {
        var data = words([
            0x03525650, 0, bigEndian ? 0 : format, bigEndian ? format : 0,
            sRGB ? 1 : 0, 0, UInt32(height), UInt32(width), 1, 1,
            UInt32(levels[0].count), UInt32(levels.count), UInt32(metadata.count),
        ], bigEndian: bigEndian) + metadata
        for level in levels { for face in level { data.append(face) } }
        return data
    }

    static func patched(_ source: Data, word: Int, value: UInt32) -> Data {
        var data = source
        data.replaceSubrange((word * 4)..<(word * 4 + 4), with: words([value]))
        return data
    }

    private static func words(_ values: [UInt32], bigEndian: Bool = false) -> Data {
        var data = Data()
        for value in values {
            var encoded = bigEndian ? value.bigEndian : value.littleEndian
            withUnsafeBytes(of: &encoded) { data.append(contentsOf: $0) }
        }
        return data
    }
}
