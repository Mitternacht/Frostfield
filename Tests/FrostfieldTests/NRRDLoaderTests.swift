import Foundation
import Frostfield
import Metal
import Testing
import simd

struct NRRDLoaderTests {
    @Test("NRRD preserves big-endian signed voxels, oblique directions, physical origin, and units")
    func physicalSpace() throws {
        let values: [Int16] = [-1_024, 320, 2_001, 16, 8, 0, 12, -3, 49, 400, -1, 999]
        let data = Self.fixture(type: "signed short", sizes: "3 2 2", encoding: "raw", extra: [
            "endian: big", "space: left-posterior-superior", "space directions: (0,2,0) (-3,1,0) (0,0,4)",
            "space origin: (10, -20, 30)", "space units: \"mm\" \"mm\" \"mm\"", "kinds: domain domain domain",
        ], payload: Self.binary(values, littleEndian: false), crlf: true)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("nrrd")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let volume = try NRRDLoader.load(contentsOf: url)
        #expect(volume.dimensions == SIMD3(3, 2, 2) && volume.scalars == .int16(values))
        #expect(volume.spacing == SIMD3(2, sqrt(10), 4))
        #expect(volume.spaceUnits == ["mm", "mm", "mm"])
        let point = SIMD4<Double>(1, 2, 1, 1)
        #expect(volume.voxelToSpace * point == SIMD4(4, -16, 34, 1))
        let ras = try #require(volume.voxelToRAS)
        #expect(ras * point == SIMD4(-4, 16, 34, 1))

        let probe = try NRRDTextureProbe()
        let native = try volume.makeTexture(device: probe.device)
        let float = try volume.makeTexture(device: probe.device, storage: .float32)
        #expect(native.textureType == .type3D && native.pixelFormat == .r16Sint)
        #expect(float.pixelFormat == .r32Float && native.mipmapLevelCount == 1)
        #expect(Int32(bitPattern: try probe.read(native, at: SIMD3(1, 0, 1), kind: .signed)) == -3)
        #expect(Float(bitPattern: try probe.read(float, at: SIMD3(2, 0, 0), kind: .float)) == 2_001)
    }

    @Test("NRRD gzip verifies the compressed stream and uploads unsigned source values")
    func gzip() throws {
        let encoded = try #require(Data(base64Encoded: "H4sIAAAAAAAC/2NgYGT4zwAkmBgaXvz9/x8AFfVeZhAAAAA="))
        func volume(_ payload: Data, sizes: String = "2 2 2") throws -> NRRDLoader.Volume {
            try NRRDLoader.load(Self.fixture(type: "ushort", sizes: sizes, encoding: "gz", payload: payload))
        }
        let cloud = try volume(encoded)
        #expect(cloud.scalars == .uint16([0, 1, 255, 256, 512, 32768, 65000, 65535]))
        #expect(cloud.spacing == SIMD3(repeating: 1) && cloud.voxelToRAS == nil)
        let probe = try NRRDTextureProbe()
        let texture = try cloud.makeTexture(device: probe.device)
        #expect(texture.pixelFormat == .r16Uint)
        #expect(try probe.read(texture, at: SIMD3(0, 1, 1), kind: .unsigned) == 65_000)
        var badCRC = encoded
        badCRC[badCRC.count - 8] ^= 1
        #expect(throws: NRRDLoader.Failure.invalidGzip) { try volume(badCRC) }
        #expect(throws: NRRDLoader.Failure.invalidGzip) { try volume(Data(encoded.dropLast())) }
        #expect(throws: NRRDLoader.Failure.invalidGzip) { try volume(encoded + Data([0])) }
        #expect(throws: NRRDLoader.Failure.invalidGzip) { try volume(encoded, sizes: "2 2 1") }
    }

    @Test("NRRD text retains scalar widths, signed values, and special floats")
    func text() throws {
        let cases: [(String, String, NRRDLoader.Scalars)] = [
            ("int8_t", "-128 127", .int8([-128, 127])),
            ("unsigned char", "1 255", .uint8([1, 255])),
            ("short int", "-32768 32767", .int16([-32768, 32767])),
            ("unsigned short int", "256 65535", .uint16([256, 65535])),
            ("signed int", "-2147483648 2147483647", .int32([.min, .max])),
            ("uint32_t", "2147483648 4294967295", .uint32([2_147_483_648, .max])),
            ("float", "-2.5 1e3", .float32([-2.5, 1_000])),
            ("double", "-2.5 1e70", .float64([-2.5, 1e70])),
        ]
        for (type, text, expected) in cases {
            let volume = try NRRDLoader.load(Self.fixture(type: type, sizes: "2 1 1", encoding: "txt", payload: Data(text.utf8)))
            #expect(volume.scalars == expected && volume.scalars.count == 2)
        }
        let data = Self.fixture(type: "float", sizes: "3 1 1", encoding: "text", extra: [
            "spacings: -2 NaN 4", "space: LAS", "space origin: (5, 6, 7)",
        ], payload: Data("NaN \n-inf\t+Inf".utf8))
        let volume = try NRRDLoader.load(data)
        #expect(volume.spacing == SIMD3(2, 1, 4))
        let ras = try #require(volume.voxelToRAS)
        #expect(ras * SIMD4<Double>(1, 1, 1, 1) == SIMD4(-3, 7, 11, 1))
        if case .float32(let samples) = volume.scalars {
            #expect(samples[0].isNaN && samples[1] == -.infinity && samples[2] == .infinity)
        } else {
            Issue.record("Expected Float32")
        }
    }

    @Test("NRRD hex accepts Teem byte strings and integer tokens used by Three.js")
    func hex() throws {
        let bytes = Self.fixture(
            type: "short", sizes: "2 1 1", encoding: "hex", extra: ["endian: big"],
            payload: Data("FF\nFE 00 12".utf8)
        )
        let tokens = Self.fixture(type: "uint16", sizes: "2 1 1", encoding: "hex", payload: Data("0xF 1AB".utf8))
        #expect(try NRRDLoader.load(bytes).scalars == .int16([-2, 18]))
        #expect(try NRRDLoader.load(tokens).scalars == .uint16([15, 427]))
    }

    @Test("NRRD Metal storage retains 32-bit integers and makes Double conversion explicit")
    func textureFormats() throws {
        let probe = try NRRDTextureProbe()
        let integer = try NRRDLoader.load(Self.fixture(
            type: "uint", sizes: "2 1 1", encoding: "ascii", payload: Data("16777217 4294967295".utf8)
        ))
        let texture = try integer.makeTexture(device: probe.device)
        #expect(texture.pixelFormat == .r32Uint)
        #expect(try probe.read(texture, at: .zero, kind: .unsigned) == 16_777_217)
        #expect(try probe.read(texture, at: SIMD3(1, 0, 0), kind: .unsigned) == .max)

        func doubles(_ text: String) throws -> NRRDLoader.Volume {
            try NRRDLoader.load(Self.fixture(type: "double", sizes: "2 1 1", encoding: "ascii", payload: Data(text.utf8)))
        }
        let float = try doubles("-4.5 0.03125")
        #expect(throws: NRRDLoader.Failure.doubleTexture) { try float.makeTexture(device: probe.device) }
        let converted = try float.makeTexture(device: probe.device, storage: .float32)
        #expect(Float(bitPattern: try probe.read(converted, at: .zero, kind: .float)) == -4.5)
        #expect(Float(bitPattern: try probe.read(converted, at: SIMD3(1, 0, 0), kind: .float)) == 0.03125)
        #expect(throws: NRRDLoader.Failure.floatTextureRange) {
            try doubles("0 1e70").makeTexture(device: probe.device, storage: .float32)
        }
    }

    @Test("NRRD rejects invalid dimensions, physical metadata, type and payload")
    func malformed() {
        #expect(throws: NRRDLoader.Failure.invalidHeader("signature")) {
            try NRRDLoader.load(Data("NRRD+001\n\n".utf8))
        }
        #expect(throws: NRRDLoader.Failure.invalidHeader("terminator")) {
            try NRRDLoader.load(Data("NRRD0005\ntype: float\n".utf8))
        }
        #expect(throws: NRRDLoader.Failure.size) {
            try NRRDLoader.load(Self.fixture(type: "float", sizes: "2048 2048 2048", encoding: "raw"))
        }
        #expect(throws: NRRDLoader.Failure.unsupported("dimension 4")) {
            try NRRDLoader.load(Self.fixture(type: "float", sizes: "1 1 1 1", encoding: "raw", dimension: 4))
        }
        #expect(throws: NRRDLoader.Failure.unsupported("detached data")) {
            try NRRDLoader.load(Self.fixture(type: "float", sizes: "1 1 1", encoding: "raw", extra: ["data file: samples.raw"]))
        }
        #expect(throws: NRRDLoader.Failure.unsupported("encoding bzip2")) {
            try NRRDLoader.load(Self.fixture(type: "float", sizes: "1 1 1", encoding: "bzip2"))
        }
        #expect(throws: NRRDLoader.Failure.unsupported("non-spatial volume axes")) {
            try NRRDLoader.load(Self.fixture(type: "float", sizes: "1 1 1", encoding: "raw", extra: ["kinds: list domain domain"]))
        }
        #expect(throws: NRRDLoader.Failure.invalidHeader("space directions")) {
            try NRRDLoader.load(Self.fixture(
                type: "float", sizes: "1 1 1", encoding: "raw", extra: ["space directions: (1,0,0) (2,0,0) (0,0,1)"]
            ))
        }
        #expect(throws: NRRDLoader.Failure.invalidPayload) {
            try NRRDLoader.load(Self.fixture(type: "float", sizes: "1 1 1", encoding: "raw", payload: Data([0, 0, 0])))
        }
        #expect(throws: NRRDLoader.Failure.invalidPayload) {
            try NRRDLoader.load(Self.fixture(type: "uint8", sizes: "1 1 1", encoding: "ascii", payload: Data("1 2".utf8)))
        }
        #expect(throws: NRRDLoader.Failure.invalidSample(0)) {
            try NRRDLoader.load(Self.fixture(type: "uint8", sizes: "1 1 1", encoding: "ascii", payload: Data("256".utf8)))
        }
    }

    private static func fixture(
        type: String, sizes: String, encoding: String, extra: [String] = [], payload: Data = Data(),
        dimension: Int = 3, crlf: Bool = false
    ) -> Data {
        let lines = ["NRRD0005", "# fixture", "type: \(type)", "dimension: \(dimension)", "sizes: \(sizes)", "encoding: \(encoding)"]
        let separator = crlf ? "\r\n" : "\n"
        return Data((lines + extra + ["", ""]).joined(separator: separator).utf8) + payload
    }

    private static func binary<T: FixedWidthInteger>(_ values: [T], littleEndian: Bool) -> Data {
        values.reduce(into: Data()) { bytes, value in
            let stored = littleEndian ? value.littleEndian : value.bigEndian
            withUnsafeBytes(of: stored) { bytes.append(contentsOf: $0) }
        }
    }
}

struct NRRDTextureProbe {
    enum Kind: String { case float, signed, unsigned }
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let library: any MTLLibrary

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void nrrd_float(texture3d<float> t [[texture(0)]],
                                   constant uint4& p [[buffer(0)]], device uint& out [[buffer(1)]]) {
                constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
                float3 uv = (float3(p.xyz) + 0.5) / float3(t.get_width(), t.get_height(), t.get_depth());
                out = as_type<uint>(t.sample(s, uv, level(0)).x);
            }
            kernel void nrrd_signed(texture3d<int, access::read> t [[texture(0)]],
                                    constant uint4& p [[buffer(0)]], device uint& out [[buffer(1)]]) {
                out = as_type<uint>(t.read(p.xyz).x);
            }
            kernel void nrrd_unsigned(texture3d<uint, access::read> t [[texture(0)]],
                                      constant uint4& p [[buffer(0)]], device uint& out [[buffer(1)]]) {
                out = t.read(p.xyz).x;
            }
            """, options: nil)
    }

    func read(_ texture: any MTLTexture, at coordinate: SIMD3<UInt32>, kind: Kind) throws -> UInt32 {
        let function = try #require(library.makeFunction(name: "nrrd_\(kind.rawValue)"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let buffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        var point = SIMD4(coordinate, 0)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBytes(&point, length: MemoryLayout.size(ofValue: point), index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 1)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(command.error?.localizedDescription ?? "Metal command failed")")
        return buffer.contents().load(as: UInt32.self)
    }
}
