import Foundation
import Frostfield
import Metal
import simd

import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct KTX1LoaderTests {
    private let device: any MTLDevice
    private let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    @Test("KTX1 retains ASTC footprint, stored mips, and the declared transfer")
    func astc() throws {
        let first = KTX2Fixture.astcConstantBlock([0xFFFF, 0x8080, 0, 0xFFFF])
        let last = KTX2Fixture.astcConstantBlock([0, 0, 0xFFFF, 0x8080])
        let data = KTX1Fixture.data(gl: 0x93D7, base: 0x1908, width: 8, height: 8, levels: [[first], [last]])
        let color = try loader.load(data: data)
        let raw = try loader.loadKTX1(data: data, interpretation: .linearData)
        #expect(color.pixelFormat == .astc_8x8_srgb && raw.pixelFormat == .astc_8x8_ldr)
        #expect(color.mipmapLevelCount == 2 && raw.mipmapLevelCount == 2)
        let probe = try DDSProbe(device: device)
        let point = SIMD3<Float>(0.5, 0.5, 0)
        #expect(abs(try probe.sample(color, at: point).y - 0.21586) < 0.01)
        #expect(abs(try probe.sample(raw, at: point).y - 0.50196) < 0.01)
        #expect(abs(try probe.sample(color, at: point, level: 1).w - 0.50196) < 0.01)

        let runtimeHint = KTX1Fixture.data(
            gl: 0x93B7, base: 0x1908, width: 8, height: 8, mipCount: 0, levels: [[first]]
        )
        let defaultLinear = try loader.load(data: runtimeHint, mipmapped: true)
        let forcedSRGB = try loader.loadKTX1(data: runtimeHint, interpretation: .encodedSRGB)
        #expect(defaultLinear.pixelFormat == .astc_8x8_ldr && defaultLinear.mipmapLevelCount == 1)
        #expect(forcedSRGB.pixelFormat == .astc_8x8_srgb)
    }

    @Test("Big-endian KTX1 cubes keep their level-major faces and base-only selection")
    func cube() throws {
        let base: [[UInt8]] = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255]]
        let faces = base.map { $0 + [0, 0, 0, 0, 0] }
        let white: [UInt8] = [255, 255, 255, 0, 0, 0, 0, 0]
        let data = KTX1Fixture.data(
            gl: 0x8D64, base: 0x1907, width: 4, height: 4, bigEndian: true,
            metadata: [("KTXOrientation", "S=r,T=d,R=i")], levels: [faces, Array(repeating: white, count: 6)]
        )
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("ktx")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        let cube = try loader.loadKTX1(contentsOf: url, interpretation: .linearData)
        let top = try loader.loadKTX1(data: data, interpretation: .linearData, loadMipmaps: false)
        #expect(cube.textureType == .typeCube && cube.mipmapLevelCount == 2)
        #expect(top.textureType == .typeCube && top.mipmapLevelCount == 1)
        let probe = try DDSProbe(device: device)
        let directions: [SIMD3<Float>] = [SIMD3(1,0,0), SIMD3(-1,0,0), SIMD3(0,1,0),
                                          SIMD3(0,-1,0), SIMD3(0,0,1), SIMD3(0,0,-1)]
        for (index, direction) in directions.enumerated() {
            let expected = SIMD4<Float>(
                Float(max(base[index][0], 2)), Float(max(base[index][1], 2)), Float(max(base[index][2], 2)), 255
            ) / 255
            let actual = try probe.sample(cube, at: direction)
            #expect(simd_reduce_max(abs(actual - expected)) < 0.002)
            #expect(try probe.sample(top, at: direction) == actual)
            #expect(try probe.sample(cube, at: direction, level: 1) == SIMD4(repeating: 1))
        }
        #expect(throws: TextureLoader.Failure.unsupportedKTX1("cube maps through the 2D loader")) {
            try loader.load(data: data)
        }
    }

    @Test("PVRTC1 textures sample their minimum-sized base and mip payloads",
          arguments: [UInt32(0x8C00), 0x8C01, 0x8C02, 0x8C03])
    func pvrtc(gl: UInt32) throws {
        let block: [UInt8] = [0, 0, 0, 0, 0, 0xFC, 0, 0xFC]
        let stored = Array(repeating: block, count: 4).flatMap { $0 }
        let width: UInt32 = gl.isMultiple(of: 2) ? 8 : 16
        let base: UInt32 = gl < 0x8C02 ? 0x1907 : 0x1908
        let minimumSquare = width == 16 ? stored + stored : stored
        let squareBase = Array(repeating: minimumSquare, count: 4).flatMap { $0 }
        let square = KTX1Fixture.data(
            gl: gl, base: base, width: width * 2, height: width * 2, levels: [[squareBase], [minimumSquare]]
        )
        let undersizedMip = KTX1Fixture.data(
            gl: gl, base: base, width: width, height: width, levels: [[minimumSquare], [stored]]
        )
        let rectangle = KTX1Fixture.data(
            gl: gl, base: base, width: width * 2, height: 8, levels: [[stored + stored], [stored]]
        )
        let probe = try DDSProbe(device: device)
        for (data, usesFallback) in [(square, !device.supportsFamily(.apple2)), (rectangle, true), (undersizedMip, true)] {
            let texture = try loader.load(data: data, interpretation: .linearData)
            #expect(texture.mipmapLevelCount == 2)
            #expect((texture.pixelFormat == .rgba8Unorm) == usesFallback)
            for level in 0..<2 {
                let color = try probe.sample(texture, at: SIMD3(0.5, 0.5, 0), level: level)
                #expect(color.x > 0.98 && color.y < 0.01 && color.z < 0.01 && color.w > 0.98, "\(color)")
            }
        }
        let baseOnly = try loader.loadKTX1(data: undersizedMip, interpretation: .linearData, loadMipmaps: false)
        #expect((baseOnly.pixelFormat == .rgba8Unorm) == !device.supportsFamily(.apple2))
        #expect(try probe.sample(baseOnly, at: SIMD3(0.5, 0.5, 0)) == SIMD4(1, 0, 0, 1))
    }

    @Test("KTX1 distinguishes opaque and punch-through BC1")
    func bc1() throws {
        let block = DDSFixture.colorBlock(first: 0, second: 0xFFFF, selectors: 0xFF)
        let opaque = KTX1Fixture.data(gl: 0x83F0, base: 0x1907, width: 4, height: 4, levels: [[block]])
        let masked = KTX1Fixture.data(gl: 0x83F1, base: 0x1908, width: 4, height: 4, levels: [[block]])
        let probe = try DDSProbe(device: device)
        let point = SIMD3<Float>(0.5, 0.5, 0)
        #expect(try probe.sample(loader.load(data: opaque), at: point) == SIMD4(0, 0, 0, 1))
        #expect(try probe.sample(loader.load(data: masked), at: point) == .zero)
    }

    @Test("RGTC preserves one or two channels and signed normalized values",
          arguments: [UInt32(0x8DBB), 0x8DBC, 0x8DBD, 0x8DBE])
    func rgtc(gl: UInt32) throws {
        let two = gl >= 0x8DBD
        let signed = gl == 0x8DBC || gl == 0x8DBE
        let red: [UInt8] = [64, 0, 0, 0, 0, 0, 0, 0]
        let green: [UInt8] = [192, 0, 0, 0, 0, 0, 0, 0]
        let data = KTX1Fixture.data(gl: gl, base: two ? 0x8227 : 0x1903, width: 4, height: 4,
                                    levels: [[two ? red + green : red]])
        let texture = try loader.load(data: data, interpretation: .encodedSRGB)
        let color = try DDSProbe(device: device).sample(texture, at: SIMD3(0.5, 0.5, 0))
        let expected = SIMD4<Float>(signed ? 64 / 127 : 64 / 255, two ? (signed ? -64 / 127 : 192 / 255) : 0, 0, 1)
        #expect(simd_reduce_max(abs(color - expected)) < 0.009, "\(color), \(expected)")
        if !device.supportsBCTextureCompression {
            #expect(texture.pixelFormat == (signed ? .rgba8Snorm : .rgba8Unorm))
        }
    }

    @Test("KTX1 EAC exposes the native signed or unsigned data channels",
          arguments: [UInt32(0x9270), 0x9271, 0x9272, 0x9273])
    func eac(gl: UInt32) throws {
        let two = gl >= 0x9272
        let signed = !gl.isMultiple(of: 2)
        let red: [UInt8] = [64, 0x10, 0, 0, 0, 0, 0, 0]
        let green: [UInt8] = [192, 0x10, 0, 0, 0, 0, 0, 0]
        let data = KTX1Fixture.data(gl: gl, base: two ? 0x8227 : 0x1903, width: 4, height: 4,
                                    levels: [[two ? red + green : red]])
        let texture = try loader.load(data: data, interpretation: .encodedSRGB)
        let value = try DDSProbe(device: device).sample(texture, at: SIMD3(0.5, 0.5, 0))
        #expect(value.x > 0.2 && value.x < 0.6 && value.z == 0 && value.w == 1)
        if two {
            #expect(signed ? value.y < -0.4 : value.y > 0.7)
        } else {
            #expect(value.y == 0)
        }
    }

    @Test("KTX1 BPTC keeps BC7 alpha and linear BC6H")
    func bptc() throws {
        let block = KTX1Fixture.bits([(64, 7), (127, 7), (127, 7), (0, 7), (0, 7),
                                     (0, 7), (0, 7), (64, 7), (64, 7)])
        let bc7 = KTX1Fixture.data(gl: 0x8E8D, base: 0x1908, width: 4, height: 4, levels: [[block]])
        let bc6 = KTX1Fixture.data(
            gl: 0x8E8E, base: 0x1907, width: 4, height: 4, levels: [[DDSFixture.bc6H(red: 768, green: 256)]]
        )
        let probe = try DDSProbe(device: device)
        let point = SIMD3<Float>(0.5, 0.5, 0)
        let color = try probe.sample(loader.load(data: bc7), at: point)
        #expect(color.x > 0.98 && color.y == 0 && color.z == 0 && abs(color.w - 128 / 255) < 0.005)
        let hdr = try probe.sample(loader.load(data: bc6, interpretation: .encodedSRGB), at: point)
        #expect(abs(hdr.x - Float(Float16(bitPattern: 0xBE1F))) < 0.003 && hdr.w == 1)
    }

    @Test("KTX1 rejects malformed metadata, nonmatching block lengths, and incomplete cubes")
    func invalid() throws {
        let block = DDSFixture.colorBlock(first: 0xF800)
        let good = KTX1Fixture.data(gl: 0x83F0, base: 0x1907, width: 4, height: 4, levels: [[block]])
        var badMarker = good
        badMarker[12] = 0
        var wrongType = good
        wrongType[20] = 4
        var wrongSize = good
        wrongSize[64] = 9
        let metadata = KTX1Fixture.data(gl: 0x83F0, base: 0x1907, width: 4, height: 4,
                                        metadata: [("data", "x")], levels: [[block]])
        var badPadding = metadata
        badPadding[75] = 3
        let partialCube = KTX1Fixture.data(gl: 0x83F0, base: 0x1907, width: 4, height: 4,
                                           levels: [Array(repeating: block, count: 6)]).dropLast()
        for data in [badMarker, wrongType, wrongSize, badPadding, Data(partialCube), good + Data([0])] {
            #expect(throws: TextureLoader.Failure.self) { try loader.loadKTX1(data: data) }
        }
        let orientation = KTX1Fixture.data(gl: 0x83F0, base: 0x1907, width: 4, height: 4,
                                           metadata: [("KTXorientation", "S=r,T=u")], levels: [[block]])
        #expect(throws: TextureLoader.Failure.unsupportedKTX1("orientation S=r,T=u")) {
            try loader.loadKTX1(data: orientation)
        }
    }
}

enum KTX1Fixture {
    static func data(
        gl: UInt32, base: UInt32, width: UInt32, height: UInt32, bigEndian: Bool = false,
        mipCount: UInt32? = nil, metadata: [(String, String)] = [], levels: [[[UInt8]]]
    ) -> Data {
        func word(_ value: UInt32) -> [UInt8] {
            let bytes = (0..<4).map { UInt8(truncatingIfNeeded: value >> (8 * $0)) }
            return bigEndian ? Array(bytes.reversed()) : bytes
        }
        var kv: [UInt8] = []
        for (key, value) in metadata {
            let bytes = Array(key.utf8) + [0] + Array(value.utf8) + [0]
            kv += word(UInt32(bytes.count)) + bytes + Array(repeating: 0, count: (4 - bytes.count % 4) % 4)
        }
        let fields: [UInt32] = [0x04030201, 0, 1, 0, gl, base, width, height, 0, 0,
                                UInt32(levels[0].count), mipCount ?? UInt32(levels.count), UInt32(kv.count)]
        let identifier: [UInt8] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 13, 10, 26, 10]
        var data = Data(identifier + fields.flatMap(word) + kv)
        for level in levels {
            data.append(contentsOf: word(UInt32(level[0].count)))
            for face in level { data.append(contentsOf: face) }
        }
        return data
    }

    static func bits(_ fields: [(Int, Int)]) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: 16)
        var offset = 0
        for (value, count) in fields {
            for i in 0..<count where value & (1 << i) != 0 {
                let bit = offset + i
                result[bit / 8] |= UInt8(1 << (bit % 8))
            }
            offset += count
        }
        return result
    }
}
