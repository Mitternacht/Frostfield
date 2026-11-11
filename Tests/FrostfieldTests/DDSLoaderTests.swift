import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct DDSLoaderTests {
    private let device: any MTLDevice
    private let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        loader = TextureLoader(device: device)
    }

    @Test("DDS padded BGR rows and stored mips upload as straight RGBA")
    func paddedBGR() throws {
        let rows: [UInt8] = [
            0, 0, 255, 0, 255, 0, 255, 0, 0, 31, 32, 33,
            0, 255, 255, 255, 0, 255, 255, 255, 0, 31, 32, 33,
            30, 20, 10, 99,
        ]
        let data = DDSFixture.data(width: 3, height: 2, mips: 2, pitch: 12, payload: rows)
        let texture = try loader.load(data: data, interpretation: .linearData)
        #expect(texture.pixelFormat == .rgba8Unorm)
        #expect(texture.width == 3 && texture.height == 2 && texture.mipmapLevelCount == 2)
        #expect(rgba(texture) == [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255,
            255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 255,
        ])
        #expect(rgba(texture, level: 1) == [10, 20, 30, 255])
        let single = try loader.loadDDS(data: data, interpretation: .encodedSRGB, loadMipmaps: false)
        #expect(single.mipmapLevelCount == 1 && single.pixelFormat == .rgba8Unorm_srgb)
        let generated = try loader.loadDDS(data: data, loadMipmaps: false, mipmapped: true)
        #expect(generated.mipmapLevelCount == 2)
        #expect(rgba(generated, level: 1) != [10, 20, 30, 255])
    }

    @Test("DDS cubemap face and mip order remains correct when loading base levels only")
    func cubeOrder() throws {
        let faces: [[UInt8]] = [[0, 0, 255, 11], [0, 255, 0, 22], [255, 0, 0, 33],
                                [0, 255, 255, 44], [255, 0, 255, 55], [255, 255, 0, 66]]
        let payload = faces.enumerated().flatMap { face, bgra -> [UInt8] in
            Array(repeating: bgra, count: 4).flatMap { $0 } + [0, 0, UInt8(face), 255]
        }
        let data = DDSFixture.data(width: 2, height: 2, mips: 2, cube: true, bgra: true, payload: payload)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("dds")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        let cube = try loader.loadDDS(contentsOf: url, interpretation: .linearData)
        let base = try loader.loadDDS(data: data, interpretation: .linearData, loadMipmaps: false)
        #expect(cube.textureType == .typeCube && cube.mipmapLevelCount == 2 && base.mipmapLevelCount == 1)
        for (face, p) in faces.enumerated() {
            let expected = Array(repeating: [p[2], p[1], p[0], p[3]], count: 4).flatMap { $0 }
            #expect(rgba(cube, face: face) == expected)
            #expect(rgba(base, face: face) == expected)
            #expect(rgba(cube, face: face, level: 1) == [UInt8(face), 0, 0, 255])
        }
        #expect(throws: TextureLoader.Failure.unsupportedDDS("cube maps through the 2D loader")) {
            try loader.load(data: data)
        }
    }

    @Test("DDS DXT1 samples opaque selector 3 and crops an odd block grid")
    func dxt1() throws {
        let red = DDSFixture.colorBlock(first: 0xF800)
        let black = DDSFixture.colorBlock(first: 0, second: 0xFFFF, selectors: 0xFF)
        let data = DDSFixture.data(width: 5, height: 3, fourCC: "DXT1", payload: red + black)
        let texture = try loader.load(data: data, interpretation: .linearData)
        #expect(texture.pixelFormat == (device.supportsBCTextureCompression ? .bc1_rgba : .rgba8Unorm))
        let probe = try DDSProbe(device: device)
        let a = try probe.sample(texture, at: SIMD3(0.1, 0.5, 0))
        let b = try probe.sample(texture, at: SIMD3(0.9, 0.5, 0))
        #expect(a == SIMD4(1, 0, 0, 1))
        #expect(b == SIMD4(0, 0, 0, 1))
    }

    @Test("DDS compressed cube samples each face and its stored mip")
    func compressedCube() throws {
        let directions: [SIMD3<Float>] = [SIMD3(1,0,0), SIMD3(-1,0,0), SIMD3(0,1,0),
                                          SIMD3(0,-1,0), SIMD3(0,0,1), SIMD3(0,0,-1)]
        let endpoints: [UInt16] = [0xF800, 0x07E0, 0x001F, 0xFFE0, 0xF81F, 0x07FF]
        let colors: [SIMD4<Float>] = [SIMD4(1,0,0,1), SIMD4(0,1,0,1), SIMD4(0,0,1,1),
                                      SIMD4(1,1,0,1), SIMD4(1,0,1,1), SIMD4(0,1,1,1)]
        let payload = endpoints.flatMap {
            DDSFixture.colorBlock(first: $0) + DDSFixture.colorBlock(first: 0xFFFF)
        }
        let data = DDSFixture.data(width: 4, height: 4, mips: 2, fourCC: "DXT1", cube: true, payload: payload)
        let texture = try loader.loadDDS(data: data, interpretation: .linearData)
        let probe = try DDSProbe(device: device)
        for (face, direction) in directions.enumerated() {
            #expect(try probe.sample(texture, at: direction) == colors[face])
            #expect(try probe.sample(texture, at: direction, level: 1) == SIMD4(repeating: 1))
        }
    }

    @Test("DDS DXT3 and DXT5 retain their different alpha encodings", arguments: ["DXT3", "DXT5"])
    func alpha(fourCC: String) throws {
        let bc2 = fourCC == "DXT3"
        let alpha: [UInt8] = bc2 ? Array(repeating: 0x88, count: 8) : [64, 200, 0, 0, 0, 0, 0, 0]
        let data = DDSFixture.data(width: 4, height: 4, fourCC: fourCC,
                                   payload: alpha + DDSFixture.colorBlock(first: 0x07E0))
        let texture = try loader.load(data: data, interpretation: .color)
        let native: MTLPixelFormat = bc2 ? .bc2_rgba_srgb : .bc3_rgba_srgb
        #expect(texture.pixelFormat == (device.supportsBCTextureCompression ? native : .rgba8Unorm_srgb))
        let value = try DDSProbe(device: device).sample(texture, at: SIMD3(0.5, 0.5, 0))
        #expect(value.x == 0 && value.y == 1 && value.z == 0)
        #expect(abs(value.w - Float(bc2 ? 136 : 64) / 255) < 0.002)
    }

    @Test("DDS ETC1 is sampled as compatible ETC2 RGB8 blocks")
    func etc1() throws {
        let bytes: [UInt8] = [0xFF, 0, 0, 0, 0, 0, 0, 0]
        let data = DDSFixture.data(width: 4, height: 4, fourCC: "ETC1", payload: bytes)
        if device.supportsFamily(.apple2) {
            let texture = try loader.load(data: data, interpretation: .linearData)
            #expect(texture.pixelFormat == .etc2_rgb8)
            let value = try DDSProbe(device: device).sample(texture, at: SIMD3(0.5, 0.5, 0))
            #expect(value.x == 1 && value.w == 1)
            #expect(abs(value.y - 2 / 255) < 0.002 && abs(value.z - 2 / 255) < 0.002)
        } else {
            #expect(throws: TextureLoader.Failure.unsupportedTextureFormat("ETC1")) {
                try loader.load(data: data)
            }
        }
    }

    @Test("DDS BC6H decodes signed and unsigned linear HDR", arguments: [true, false])
    func bc6H(signed: Bool) throws {
        let block = DDSFixture.bc6H(red: signed ? 768 : 512, green: 256)
        let data = DDSFixture.data(width: 4, height: 4, fourCC: "DX10", dxgi: signed ? 96 : 95, payload: block)
        let texture = try loader.load(data: data, interpretation: .encodedSRGB)
        let native: MTLPixelFormat = signed ? .bc6H_rgbFloat : .bc6H_rgbuFloat
        #expect(texture.pixelFormat == (device.supportsBCTextureCompression ? native : .rgba16Float))
        let expected = SIMD4<Float>(
            Float(Float16(bitPattern: signed ? 0xBE1F : 0x3E0F)),
            Float(Float16(bitPattern: signed ? 0x3E1F : 0x1F0F)), 0, 1
        )
        let value = try DDSProbe(device: device).sample(texture, at: SIMD3(0.5, 0.5, 0))
        #expect(abs(value.x - expected.x) < 0.003 && abs(value.y - expected.y) < 0.003, "\(value), \(expected)")
        #expect(value.z == 0 && value.w == 1)
    }

    @Test("DDS rejects malformed layouts before allocation")
    func invalid() throws {
        let good = DDSFixture.data(width: 4, height: 4, fourCC: "DXT1", payload: DDSFixture.colorBlock(first: 0xF800))
        var partialCube = DDSFixture.data(width: 4, height: 4, fourCC: "DXT1", cube: true, payload: [])
        partialCube[112] = 0
        partialCube[113] = 2
        var badHeader = good
        badHeader[4] = 123
        for data in [Data(good.prefix(80)), Data(good.dropLast()), good + Data([0]), partialCube, badHeader] {
            #expect(throws: TextureLoader.Failure.self) { try loader.loadDDS(data: data) }
        }
        let tooManyMips = DDSFixture.data(width: 4, height: 4, mips: 4, fourCC: "DXT1", payload: [])
        #expect(throws: TextureLoader.Failure.invalidDDS("the mip count exceeds a full chain")) {
            try loader.loadDDS(data: tooManyMips)
        }
        let array = DDSFixture.data(width: 4, height: 4, fourCC: "DX10", dxgi: 95, arraySize: 2, payload: [])
        #expect(throws: TextureLoader.Failure.unsupportedDDS("DX10 texture arrays")) {
            try loader.loadDDS(data: array)
        }
        let unknown = DDSFixture.data(width: 4, height: 4, fourCC: "DX10", dxgi: 90, payload: [])
        #expect(throws: TextureLoader.Failure.unsupportedDDS("DXGI format 90")) {
            try loader.loadDDS(data: unknown)
        }
        let oversized = DDSFixture.data(width: 8_193, height: 1, fourCC: "DXT1", payload: [])
        #expect(throws: TextureLoader.Failure.imageDimensions(width: 8_193, height: 1)) {
            try loader.loadDDS(data: oversized)
        }
    }

    private func rgba(_ texture: any MTLTexture, face: Int = 0, level: Int = 0) -> [UInt8] {
        let width = max(1, texture.width >> level), height = max(1, texture.height >> level)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: width * 4, bytesPerImage: $0.count,
                             from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: level, slice: face)
        }
        return bytes
    }
}

enum DDSFixture {
    static func data(
        width: UInt32, height: UInt32, mips: UInt32 = 1, pitch: UInt32 = 0,
        fourCC: String? = nil, dxgi: UInt32 = 0, arraySize: UInt32 = 1,
        cube: Bool = false, bgra: Bool = false, payload: [UInt8]
    ) -> Data {
        var words = [UInt32](repeating: 0, count: 32)
        words[0] = 0x20534444
        words[1] = 124
        words[2] = 0x1007 | (mips > 1 ? 0x20000 : 0) | (pitch > 0 ? 8 : 0)
        words[3] = height
        words[4] = width
        words[5] = pitch
        words[7] = mips
        words[19] = 32
        words[20] = fourCC != nil ? 4 : (bgra ? 0x41 : 0x40)
        if let fourCC {
            for (i, byte) in fourCC.utf8.enumerated() { words[21] |= UInt32(byte) << (8 * i) }
        } else {
            words[22] = bgra ? 32 : 24
            words[23] = 0xFF0000
            words[24] = 0xFF00
            words[25] = 0xFF
            words[26] = bgra ? 0xFF000000 : 0
        }
        words[27] = 0x1000
        words[28] = cube ? 0xFE00 : 0
        if fourCC == "DX10" { words += [dxgi, 3, cube ? 4 : 0, arraySize, 0] }
        return Data(words.flatMap { word in
            (0..<4).map { UInt8(truncatingIfNeeded: word >> ($0 * 8)) }
        } + payload)
    }

    static func colorBlock(first: UInt16, second: UInt16 = 0, selectors: UInt8 = 0) -> [UInt8] {
        [UInt8(truncatingIfNeeded: first), UInt8(first >> 8), UInt8(truncatingIfNeeded: second), UInt8(second >> 8)]
            + Array(repeating: selectors, count: 4)
    }

    static func bc6H(red: Int, green: Int) -> [UInt8] {
        // Mode 11 with equal 10-bit endpoint pairs and all index bits zero.
        var bytes = [UInt8](repeating: 0, count: 16)
        var offset = 0
        for (value, count) in [(3,5), (red,10), (green,10), (0,10), (red,10), (green,10), (0,10)] {
            for bit in 0..<count where value & (1 << bit) != 0 {
                let address = offset + bit
                bytes[address / 8] |= UInt8(1 << (address % 8))
            }
            offset += count
        }
        return bytes
    }
}

struct DDSProbe {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let library: any MTLLibrary

    init(device: any MTLDevice) throws {
        self.device = device
        queue = try #require(device.makeCommandQueue())
        library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::nearest);
            kernel void plane(texture2d<float> t [[texture(0)]],
                              constant float4& p [[buffer(0)]], device float4& out [[buffer(1)]]) {
                out = t.sample(s, p.xy, level(p.w));
            }
            kernel void cube(texturecube<float> t [[texture(0)]],
                             constant float4& p [[buffer(0)]], device float4& out [[buffer(1)]]) {
                out = t.sample(s, p.xyz, level(p.w));
            }
            kernel void texture_array(texture2d_array<float> t [[texture(0)]],
                                      constant float4& p [[buffer(0)]], device float4& out [[buffer(1)]]) {
                out = t.sample(s, p.xy, uint(p.z), level(p.w));
            }
            """, options: nil)
    }

    func sample(_ texture: any MTLTexture, at point: SIMD3<Float>, level: Int = 0) throws -> SIMD4<Float> {
        let name = switch texture.textureType {
        case .typeCube: "cube"
        case .type2DArray: "texture_array"
        default: "plane"
        }
        let function = try #require(library.makeFunction(name: name))
        let pipeline = try device.makeComputePipelineState(function: function)
        let output = try #require(device.makeBuffer(length: 16, options: .storageModeShared))
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        var args = SIMD4(point.x, point.y, point.z, Float(level))
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBytes(&args, length: 16, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        let one = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(one, threadsPerThreadgroup: one)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        return output.contents().load(as: SIMD4<Float>.self)
    }
}
