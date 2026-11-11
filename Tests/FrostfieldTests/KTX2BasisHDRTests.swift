import CryptoKit
import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct KTX2BasisHDRTests {
    let device: any MTLDevice
    let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    enum Profile: CaseIterable {
        case uastc4x4, astc6x6, intermediate, legacyIntermediate

        var fixture: String {
            switch self {
            case .uastc4x4: KTX2BasisHDRTests.uastc4x4
            case .astc6x6: KTX2BasisHDRTests.astc6x6
            case .intermediate: KTX2BasisHDRTests.intermediate
            case .legacyIntermediate: KTX2BasisHDRTests.legacyIntermediate
            }
        }
    }

    @Test("HDR Basis profiles retain scene-linear half floats and stored mips",
          arguments: Profile.allCases)
    func profiles(_ profile: Profile) throws {
        let data = try Self.data(profile.fixture)
        let texture = try loader.load(data: data, mipmapped: true)
        let explicit = try loader.loadKTX2(data: data, interpretation: .encodedSRGB)
        #expect(texture.pixelFormat == .rgba16Float && explicit.pixelFormat == .rgba16Float)
        #expect(texture.textureType == .type2D && texture.width == 7 && texture.height == 5)
        #expect(texture.mipmapLevelCount == 3 && explicit.mipmapLevelCount == 3)
        let reference = profile == .uastc4x4
            ? "9a058b6c0bed2d222278d6a79eeeba90bade526fb66b3d98e6576bd3b15255a2" : "06547a92b6ab3ea18353805e9af68ba9d477a53b72021df7017111d54387747a"
        #expect(Self.digest(texture, slices: 1) == reference)
        #expect(Self.digest(explicit, slices: 1) == reference)
        #expect(throws: Self.unsupported) { try BasisKTX2.validate(data) }

        let samples: [SIMD4<Float>] = profile == .uastc4x4
            ? [SIMD4(0.5, 2.0, 16.0, 1.0), SIMD4(0.5, 2.0, 16.0, 1.0), SIMD4(0.423583984375, 1.6943359375, 13.5546875, 1.0)]
            : [SIMD4(0.5, 2.0, 16.0, 1.0), SIMD4(0.479736328125, 1.9267578125, 15.4140625, 1.0), SIMD4(0.423583984375, 1.6943359375, 13.5546875, 1.0)]
        let probe = try DDSProbe(device: device)
        for (mip, expected) in samples.enumerated() {
            let w = max(1, texture.width >> mip), h = max(1, texture.height >> mip)
            let at = SIMD3<Float>(1 - 0.5 / Float(w), 1 - 0.5 / Float(h), 0)
            let actual = try probe.sample(texture, at: at, level: mip)
            #expect(actual == expected)
            #expect(actual.z > 5 && actual.w == 1)
        }
    }

    @Test("Raw UASTC HDR cubemap retains distinct face orientation, mips, and radiance")
    func cube() throws {
        let source = try Self.data(Self.cube)
        let texture = try loader.loadKTX2(data: source, interpretation: .linearData)
        #expect(texture.textureType == .typeCube && texture.pixelFormat == .rgba16Float)
        #expect(texture.width == 4 && texture.height == 4 && texture.mipmapLevelCount == 3)
        #expect(throws: Self.unsupported) { try loader.load(data: source) }
        #expect(Self.digest(texture, slices: 6) == "ada1ccddc3741451cea606d761eb68a0de96b9cd051264b0efe1025789fd23f2")
        let samples: [[SIMD4<Float>]] = [
            [
                SIMD4(0.5, 2.0, 16.0, 1.0), SIMD4(0.25, 1.0, 8.0, 1.0),
                SIMD4(32.0, 8.0, 0.25, 1.0), SIMD4(16.0, 4.0, 0.125, 1.0),
                SIMD4(2.0, 16.0, 4.0, 1.0), SIMD4(1.0, 8.0, 2.0, 1.0),
                SIMD4(4.0, 0.5, 2.0, 1.0), SIMD4(2.0, 0.25, 1.0, 1.0),
                SIMD4(1.0, 32.0, 8.0, 1.0), SIMD4(0.5, 16.0, 4.0, 1.0),
                SIMD4(16.0, 4.0, 32.0, 1.0), SIMD4(8.0, 2.0, 16.0, 1.0),
            ],
            [
                SIMD4(0.5, 2.0, 16.0, 1.0), SIMD4(0.374267578125, 1.4970703125, 11.9765625, 1.0),
                SIMD4(32.0, 8.0, 0.25, 1.0), SIMD4(23.953125, 5.98828125, 0.1871337890625, 1.0),
                SIMD4(2.0, 16.0, 4.0, 1.0), SIMD4(1.4970703125, 11.9765625, 2.994140625, 1.0),
                SIMD4(4.0, 0.5, 2.0, 1.0), SIMD4(2.994140625, 0.374267578125, 1.4970703125, 1.0),
                SIMD4(1.0, 32.0, 8.0, 1.0), SIMD4(0.74853515625, 23.953125, 5.98828125, 1.0),
                SIMD4(16.0, 4.0, 32.0, 1.0), SIMD4(11.9765625, 2.994140625, 23.953125, 1.0),
            ],
            [
                SIMD4(0.390625, 1.5625, 12.5, 1.0), SIMD4(0.390625, 1.5625, 12.5, 1.0),
                SIMD4(25.0, 6.25, 0.1953125, 1.0), SIMD4(25.0, 6.25, 0.1953125, 1.0),
                SIMD4(1.5625, 12.5, 3.125, 1.0), SIMD4(1.5625, 12.5, 3.125, 1.0),
                SIMD4(3.125, 0.390625, 1.5625, 1.0), SIMD4(3.125, 0.390625, 1.5625, 1.0),
                SIMD4(0.78125, 25.0, 6.25, 1.0), SIMD4(0.78125, 25.0, 6.25, 1.0),
                SIMD4(12.5, 3.125, 25.0, 1.0), SIMD4(12.5, 3.125, 25.0, 1.0),
            ],
        ]
        let probe = try DDSProbe(device: device)
        for mip in 0..<3 {
            let near = 0.5 / Float(4 >> mip)
            for face in 0..<6 {
                let lower = Self.cubeVector(face: face, u: 1 - near, v: 1 - near)
                let upper = Self.cubeVector(face: face, u: 1 - near, v: near)
                #expect(try probe.sample(texture, at: lower, level: mip) == samples[mip][2 * face])
                #expect(try probe.sample(texture, at: upper, level: mip) == samples[mip][2 * face + 1])
            }
        }
    }

    @Test("HDR intermediate profile ABCE retains independent array layers")
    func array() throws {
        let source = try Self.data(Self.array)
        let texture = try loader.loadKTX2(data: source)
        #expect(texture.textureType == .type2DArray && texture.pixelFormat == .rgba16Float)
        #expect(texture.arrayLength == 3 && texture.mipmapLevelCount == 3)
        #expect(Self.digest(texture, slices: 3) == "ae604775a5b82649acf4de7fc40ad1fe08b23e20efcf953bc38caa3154c86711")
        let samples: [[SIMD4<Float>]] = [
            [SIMD4(0.5, 2.0, 16.0, 1.0), SIMD4(32.0, 8.0, 0.25, 1.0), SIMD4(2.0, 16.0, 4.0, 1.0)],
            [SIMD4(0.5, 2.0, 16.0, 1.0), SIMD4(32.0, 8.0, 0.25, 1.0), SIMD4(1.98046875, 15.84375, 3.8828125, 1.0)],
            [SIMD4(0.390625, 1.5625, 12.5, 1.0), SIMD4(25.0, 6.25, 0.1953125, 1.0), SIMD4(1.5625, 12.5, 3.125, 1.0)],
        ]
        let probe = try DDSProbe(device: device)
        for (mip, layers) in samples.enumerated() {
            let at = 1 - 0.5 / Float(4 >> mip)
            for (layer, expected) in layers.enumerated() {
                #expect(try probe.sample(texture, at: SIMD3(at, at, Float(layer)), level: mip) == expected)
            }
        }
    }

    @Test("HDR Basis rejects incompatible transfer, malformed blocks and slices, and LDR storage")
    func invalidHDR() throws {
        let hdr = try Self.data(Self.uastc4x4)
        let dfd = hdr.withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 48, as: UInt32.self))) }
        var encoded = hdr
        encoded[dfd + 14] = 2
        #expect(throws: Self.unsupported) { try loader.loadKTX2(data: encoded) }
        var wrongFootprint = hdr
        wrongFootprint[dfd + 16] = 5
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: wrongFootprint) }
        var wrongLength = hdr
        wrongLength[96] += 16
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: wrongLength) }

        let intermediate = try Self.data(Self.intermediate)
        let sgd = intermediate.withUnsafeBytes {
            Int(UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 64, as: UInt64.self)))
        }
        var escapedSlice = intermediate
        escapedSlice[sgd] = 255
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: escapedSlice) }
        var wrongProfile = intermediate
        wrongProfile[sgd + 8] = 0xce
        #expect(throws: Self.invalid) { try loader.loadKTX2(data: wrongProfile) }

        let storage = TextureLoader.Failure.unsupportedTextureFormat("requested LDR storage for HDR Basis Universal")
        #expect(throws: storage) { try loader.loadKTX2(data: hdr, basisStorage: .rgba8) }
        #expect(throws: storage) { try loader.loadKTX2(data: hdr, basisStorage: .deviceCompressed) }
    }

    private static func cubeVector(face: Int, u: Float, v: Float) -> SIMD3<Float> {
        let a = 2 * u - 1, b = 2 * v - 1
        return switch face {
        case 0: SIMD3(1, -b, -a)
        case 1: SIMD3(-1, -b, a)
        case 2: SIMD3(a, 1, b)
        case 3: SIMD3(a, -1, -b)
        case 4: SIMD3(a, -b, 1)
        default: SIMD3(-a, -b, -1)
        }
    }

    private static func digest(_ texture: any MTLTexture, slices: Int) -> String {
        var digest = SHA256()
        for mip in 0..<texture.mipmapLevelCount {
            let w = max(1, texture.width >> mip), h = max(1, texture.height >> mip)
            let row = w * 8
            var plane = Data(count: row * h)
            for slice in 0..<slices {
                plane.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                    texture.getBytes(raw.baseAddress!, bytesPerRow: row, bytesPerImage: raw.count,
                                     from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: mip, slice: slice)
                }
                digest.update(data: plane)
            }
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func data(_ fixture: String) throws -> Data {
        try #require(Data(base64Encoded: fixture, options: .ignoreUnknownCharacters))
    }

    private static let unsupported = TextureLoader.Failure.unsupportedKTX2(
        "Basis Universal layout, transfer function, alpha mode, or supercompression"
    )
    private static let invalid = TextureLoader.Failure.invalidKTX2("invalid ETC1S or UASTC Basis Universal data")

    // Authored RGBE texels encoded with the pinned Basis Universal 2.50 CLI.
    // Exact half-float references are from the same version's independent per-image API.
    private static let uastc4x4 = """
        q0tUWCAyMLsNChoK0MubOwEAAAAHAAAABQAAAAAAAAAAAAAAAQAAAAMAAAACAAAAmAAAACwAAADEAAAAPAAAAAAA
        AAAAAAAAAAAAAAAAAAAyAQAAAAAAADgAAAAAAAAAQAAAAAAAAAAZAQAAAAAAABkAAAAAAAAAEAAAAAAAAAAAAQAA
        AAAAABkAAAAAAAAAEAAAAAAAAAAsAAAAAAAAAAIAKACnAQEAAwMAABAAAAAAAAAAAAB/gAAAAAAAAAAAAACAPxQA
        AABLVFhtYXBSYW5nZQAAAIA/AAAAAB8AAABLVFh3cml0ZXIAQmFzaXMgVW5pdmVyc2FsIDIuNTAAACi1L/0gEIEA
        APz/////////xzbHPsdKADwotS/9IBCBAABCYjE2MCiAX/+v/6//r5lJKLUv/SBAfQEAFAJRYmGgYaAAAAESgUgg
        IgAAiUgkIgGQCPz/ADgAQABMADwEALihoQIARpttLgE=
        """
    private static let astc6x6 = """
        q0tUWCAyMLsNChoK1MubOwEAAAAHAAAABQAAAAAAAAAAAAAAAQAAAAMAAAACAAAAmAAAACwAAADEAAAAPAAAAAAA
        AAAAAAAAAAAAAAAAAAAyAQAAAAAAACYAAAAAAAAAIAAAAAAAAAAZAQAAAAAAABkAAAAAAAAAEAAAAAAAAAAAAQAA
        AAAAABkAAAAAAAAAEAAAAAAAAAAsAAAAAAAAAAIAKACiAQEABQUAABAAAAAAAAAAAAB/wAAAAAAAAIC/AACAPxQA
        AABLVFhtYXBSYW5nZQAAAIA/AAAAAB8AAABLVFh3cml0ZXIAQmFzaXMgVW5pdmVyc2FsIDIuNTAAACi1L/0gEIEA
        APz/////////xzbHPsdKADwotS/9IBCBAAByYRlNhgJ8wfVwPVwP15cKKLUv/SAg7QAAuBRhYaBhoAAAQT5+cnZ6
        +AB+fn5+fj4AAQCmmEs=
        """
    private static let intermediate = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAHAAAABQAAAAAAAAAAAAAAAQAAAAMAAAAEAAAAmAAAACwAAADEAAAAPAAAAAAB
        AAAAAAAAJAAAAAAAAABJAQAAAAAAACUAAAAAAAAAAAAAAAAAAAAyAQAAAAAAABcAAAAAAAAAAAAAAAAAAAAkAQAA
        AAAAAA4AAAAAAAAAAAAAAAAAAAAsAAAAAAAAAAIAKACoAQEABQUAABAAAAAAAAAAAAB/gAAAAAAAAAAAAACAPxQA
        AABLVFhtYXBSYW5nZQAAAIA/AAAAAB8AAABLVFh3cml0ZXIAQmFzaXMgVW5pdmVyc2FsIDIuNTAAAAAAAAAlAAAA
        zasAAAAAAAAXAAAAzasAAAAAAAAOAAAAzasAAM2rAQABADy2HfuOlUKnzasDAAIABQxkjVHweb48P8/P8/N8oVPN
        qwcABQABYKBhoAAAAdTBeaHk4Q0AAw0DBQAIAC8vLy8vr9Ap
        """
    private static let legacyIntermediate = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAHAAAABQAAAAAAAAAAAAAAAQAAAAMAAAABAAAAmAAAACwAAADEAAAAPAAAAAAB
        AAAAAAAAGAAAAAAAAABJAQAAAAAAACUAAAAAAAAAAAAAAAAAAAAyAQAAAAAAABcAAAAAAAAAAAAAAAAAAAAkAQAA
        AAAAAA4AAAAAAAAAAAAAAAAAAAAsAAAAAAAAAAIAKACoAQEABQUAABAAAAAAAAAAAAB/gAAAAAAAAAAAAACAPxQA
        AABLVFhtYXBSYW5nZQAAAIA/AAAAAB8AAABLVFh3cml0ZXIAQmFzaXMgVW5pdmVyc2FsIDIuNTAAAAAAAAAlAAAA
        AAAAABcAAAAAAAAADgAAAAAAAAAAAAAAAAAAAM2rAQABADy2HfuOlUKnzasDAAIABQxkjVHweb48P8/P8/N8oVPN
        qwcABQABYKBhoAAAAdTBeaHk4Q0AAw0DBQAIAC8vLy8vr9Ap
        """
    private static let array = """
        q0tUWCAyMLsNChoKAAAAAAEAAAAEAAAABAAAAAAAAAADAAAAAQAAAAMAAAAEAAAAmAAAACwAAADEAAAAPAAAAAAB
        AAAAAAAAbAAAAAAAAADbAQAAAAAAAEUAAAAAAAAAAAAAAAAAAACWAQAAAAAAAEUAAAAAAAAAAAAAAAAAAABsAQAA
        AAAAACoAAAAAAAAAAAAAAAAAAAAsAAAAAAAAAAIAKACoAQEABQUAABAAAAAAAAAAAAB/gAAAAAAAAAAAAACAPxQA
        AABLVFhtYXBSYW5nZQAAAIA/AAAAAB8AAABLVFh3cml0ZXIAQmFzaXMgVW5pdmVyc2FsIDIuNTAAAAAAAAAXAAAA
        zqsAABcAAAAXAAAAzqsAAC4AAAAXAAAAzqsAAAAAAAAXAAAAzqsAABcAAAAXAAAAzqsAAC4AAAAXAAAAzqsAAAAA
        AAAOAAAAzqsAAA4AAAAOAAAAzqsAABwAAAAOAAAAzqsAAM6rAQABAASyAfmAlEKnzqsBAAEABHICGYFkQqfOqwEA
        AQAE8gEpgYRCp86rAgACAAFgmmGgAADx3NOvQ+XjFToFzqsCAAIAAYCaQeAAAPDc069D5eMVOgXOqwIAAgADHpgk
        /GOq6v/+7//+73+hU86rBAAEAAFgoGGgAAAB1MF5oeThFToFzqsEAAQAAYCgQeAAAADUwXmh5OEVOgXOqwQABAAB
        wEDBgYABANTBeaHk4RU6BQ==
        """
    private static let cube = """
        q0tUWCAyMLsNChoK0MubOwEAAAAEAAAABAAAAAAAAAAAAAAABgAAAAMAAAAAAAAAmAAAACwAAADEAAAAPAAAAAAA
        AAAAAAAAAAAAAAAAAADAAQAAAAAAAGAAAAAAAAAAYAAAAAAAAABgAQAAAAAAAGAAAAAAAAAAYAAAAAAAAAAAAQAA
        AAAAAGAAAAAAAAAAYAAAAAAAAAAsAAAAAAAAAAIAKACnAQEAAwMAABAAAAAAAAAAAAB/gAAAAAAAAAAAAACAPxQA
        AABLVFhtYXBSYW5nZQAAAIA/AAAAAB8AAABLVFh3cml0ZXIAQmFzaXMgVW5pdmVyc2FsIDIuNTAAAPz/////////
        QDZAPkBKADz8/////////0BOQEZAMgA8/P////////9APkBKQEIAPPz/////////QEJANkA+ADz8/////////0A6
        QE5ARgA8/P////////9ASkBCQE4APFFiYZZhoAAAARKHSCPibAZRYoGWQeAAAAASh0gj4mwGUWKBWYAAgcEAEodI
        I+JsBlFigViAgIFAABKHSCPibAZRYoGWoUAAAQASh0gj4mwGUWIB2ICAgUABEodII+JsBlFiYaBhoAAAARKBSCAi
        AABRYoGgQeAAAAASgUggIgAAUWLBQMGBgAEAEoFIICIAAFFiQUDBQYAAABKBSCAiAABRYoGgoUAAAQASgUggIgAA
        UWIBQcFBgAABEoFIICIAAA==
        """
}
