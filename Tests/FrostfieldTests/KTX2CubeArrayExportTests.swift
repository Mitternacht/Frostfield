import Foundation
import Metal
import Testing
@testable import Frostfield

/// Native KTX2 cube-array export. The container layout is checked on every device by feeding
/// synthetic readback bytes to the exporter's encoder. Real cube-array textures, which the
/// iOS Simulator does not expose, additionally prove Metal's slice order and public readback.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct KTX2CubeArrayExportTests {
    struct FormatCase: Sendable {
        let name: String
        let metal: MTLPixelFormat
        let vulkan: UInt32
        let channels: Int
        let typeSize: Int
        var pixelBytes: Int { channels * typeSize }
    }

    static let formats: [FormatCase] = [
        .init(name: "r8", metal: .r8Unorm, vulkan: 9, channels: 1, typeSize: 1),
        .init(name: "rgba8-srgb", metal: .rgba8Unorm_srgb, vulkan: 43, channels: 4, typeSize: 1),
        .init(name: "rgba16f", metal: .rgba16Float, vulkan: 97, channels: 4, typeSize: 2),
    ]

    /// Odd-sized faces give mips of 5, 2 and 1 texels.
    private static let side = 5
    private static let mipCount = 3

    private static var supportsCubeArrays: Bool {
        #if targetEnvironment(simulator)
        false
        #else
        let device = MTLCreateSystemDefaultDevice()
        return device?.supportsFamily(.apple4) == true || device?.supportsFamily(.mac2) == true
        #endif
    }

    // MARK: Layout on every device

    @Test("Cube-array files order each level by cube, then face, with complete-cube metadata",
          arguments: [1, 2], formats)
    func layout(cubes: Int, format: FormatCase) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let payloads = Self.payloads(format, cubes: cubes)
        let file = try Self.encode(format, cubes: cubes, payloads: payloads)
        verifyStructure(file, format, cubes: cubes, payloads: payloads)
        try verifyByImportingAsFlattenedArray(file, format, cubes: cubes, payloads: payloads, device: device)
    }

    @Test("A two-cube RGBA8 sRGB header, level index, descriptor and metadata are byte exact")
    func byteExactPrefix() throws {
        let format = Self.formats[1]
        let file = try Self.encode(format, cubes: 2, payloads: Self.payloads(format, cubes: 2))
        let expected = Self.expectedPrefix
        #expect(expected.count == 296)
        #expect(file.count == 1_736)
        #expect(Data(file.prefix(expected.count)) == expected)
    }

    @Test("Cube and array shapes are validated with typed errors before any texel is read")
    func planValidation() throws {
        func plan(
            _ format: KTX2Format, side: Int = 4, height: Int? = nil, faces: Int = 6, layers: Int, levels: Int = 1
        ) throws -> KTX2Exporter.Plan {
            try KTX2Exporter.plan(
                format: format, width: side, height: height ?? side, depth: 0, faces: faces, layerCount: layers,
                levelCount: levels, options: .init()
            )
        }
        let r8 = try #require(KTX2Format(vkFormat: 9))
        let rgba32 = try #require(KTX2Format(vkFormat: 109))

        #expect(throws: KTX2Exporter.Failure.cubeFaceSize(width: 4, height: 2)) { try plan(r8, height: 2, layers: 1) }
        #expect(throws: KTX2Exporter.Failure.arrayLength(2_049)) { try plan(r8, layers: 2_049) }
        #expect(throws: KTX2Exporter.Failure.arrayLength(-1)) { try plan(r8, layers: -1) }
        #expect(throws: KTX2Exporter.Failure.arrayLength(2_049)) { try plan(r8, faces: 1, layers: 2_049) }

        // Six faces per cube count toward the 16,777,216-texel limit: two 1024² cubes fit, three do not.
        #expect(try plan(r8, side: 1_024, layers: 2).sourceLength == 1_024 * 1_024 * 6 * 2)
        #expect(throws: KTX2Exporter.Failure.size) { try plan(r8, side: 1_024, layers: 3) }

        // 66 faces of 1 MiB exceed the 64 MiB file budget; 60 faces plus metadata still fit.
        #expect(try plan(rgba32, side: 256, layers: 10).fileLength <= TextureLoader.maxEncodedBytes)
        #expect(throws: KTX2Exporter.Failure.size) { try plan(rgba32, side: 256, layers: 11) }

        let planned = try plan(rgba32, side: 5, layers: 2, levels: 3)
        #expect(throws: KTX2Exporter.Failure.readback("Unexpected texture byte count.")) {
            try KTX2Exporter.encode(planned, pixels: Data(count: planned.sourceLength - 1))
        }
    }

    // MARK: Native textures on cube-array GPUs

    @Test("Native cube-array textures export every cube, face and mip in file order",
          .enabled(if: KTX2CubeArrayExportTests.supportsCubeArrays, "Requires native Metal cube-array textures"),
          arguments: [1, 2], formats)
    func nativeCubeArray(cubes: Int, format: FormatCase) throws {
        let fixture = try PathTestFixture()
        let source = try Self.cubeArray(
            fixture.device, format: format.metal, side: Self.side, cubes: cubes, mips: Self.mipCount
        )
        let payloads = Self.payloads(format, cubes: cubes)
        let command = try #require(fixture.queue.makeCommandBuffer())
        for level in 0..<Self.mipCount {
            for cube in 0..<cubes {
                for face in 0..<6 {
                    // Metal addresses a cube-array face as 6 × cube + face.
                    try TextureTransfer.encodeUpload(
                        payloads[level][cube][face], to: source, mipLevel: level, slice: cube * 6 + face,
                        commandBuffer: command
                    )
                }
            }
        }
        command.commit()
        let file = try KTX2Exporter.data(from: source, commandQueue: fixture.queue)
        #expect(command.status == .completed)

        let synthetic = try Self.encode(format, cubes: cubes, payloads: payloads)
        #expect(file == synthetic)
        verifyStructure(file, format, cubes: cubes, payloads: payloads)
        try verifyByImportingAsFlattenedArray(file, format, cubes: cubes, payloads: payloads, device: fixture.device)
        if format.metal == .rgba8Unorm_srgb, cubes == 2 { #expect(Data(file.prefix(296)) == Self.expectedPrefix) }
        try exportForExternalValidation(file, name: "cube-array-\(format.name)-\(cubes)-cube\(cubes == 1 ? "" : "s").ktx2")
    }

    @Test("Native cube-array export rejects unsupported formats and oversized shapes before readback",
          .enabled(if: KTX2CubeArrayExportTests.supportsCubeArrays, "Requires native Metal cube-array textures"))
    func nativeRejections() throws {
        let fixture = try PathTestFixture()
        let snorm = try Self.cubeArray(fixture.device, format: .rgba8Snorm, side: 2, cubes: 2, mips: 1)
        #expect(throws: KTX2Exporter.Failure.pixelFormat(.rgba8Snorm)) {
            try KTX2Exporter.data(from: snorm, commandQueue: fixture.queue)
        }
        // Three 1024² cubes hold 18,874,368 texels, above the 16,777,216-texel limit.
        let large = try Self.cubeArray(fixture.device, format: .r8Unorm, side: 1_024, cubes: 3, mips: 1)
        #expect(throws: KTX2Exporter.Failure.size) { try KTX2Exporter.data(from: large, commandQueue: fixture.queue) }
        // Eleven 256² RGBA32F cubes need 66 MiB of texels.
        let heavy = try Self.cubeArray(fixture.device, format: .rgba32Float, side: 256, cubes: 11, mips: 1)
        #expect(throws: KTX2Exporter.Failure.size) { try KTX2Exporter.data(from: heavy, commandQueue: fixture.queue) }
    }

    // MARK: Fixtures

    /// Distinct bytes for every level, cube and face, as `[level][cube][face]`.
    private static func payloads(_ format: FormatCase, cubes: Int) -> [[[Data]]] {
        (0..<mipCount).map { level in
            let side = max(1, Self.side >> level)
            let count = side * side * format.channels
            return (0..<cubes).map { cube in
                (0..<6).map { face in
                    let seed = cube * 6 + face
                    if format.typeSize == 2 {
                        return raw((0..<count).map { Float16(Float($0) * 0.25 + Float(seed) * 3 - Float(level) * 11) })
                    }
                    return Data((0..<count).map { UInt8(($0 * 29 + seed * 53 + level * 71) % 256) })
                }
            }
        }
    }

    /// Encodes the synthetic readback bytes in Metal's level, then slice, order.
    private static func encode(_ format: FormatCase, cubes: Int, payloads: [[[Data]]]) throws -> Data {
        let ktx = try #require(KTX2Format(vkFormat: format.vulkan))
        let plan = try KTX2Exporter.plan(
            format: ktx, width: side, height: side, depth: 0, faces: 6, layerCount: cubes, levelCount: mipCount,
            options: .init()
        )
        let readback = payloads.reduce(into: Data()) { data, level in
            for cube in level { for face in cube { data.append(face) } }
        }
        return try KTX2Exporter.encode(plan, pixels: readback)
    }

    private static func cubeArray(
        _ device: any MTLDevice, format: MTLPixelFormat, side: Int, cubes: Int, mips: Int
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .typeCubeArray
        descriptor.pixelFormat = format
        descriptor.width = side
        descriptor.height = side
        descriptor.arrayLength = cubes
        descriptor.mipmapLevelCount = mips
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    /// Independent description of the first 296 bytes of the two-cube RGBA8 sRGB file: header,
    /// supercompression-free level index, 92-byte descriptor and 52 bytes of key/value data.
    private static let expectedPrefix: Data = {
        var data = Data([0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])
        func little<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        // vkFormat, typeSize, 5 × 5, no depth, two cubes, six faces, three levels, no supercompression,
        // then the descriptor and key/value offsets and lengths.
        for word: UInt32 in [43, 1, 5, 5, 0, 2, 6, 3, 0, 152, 92, 244, 52] { little(word) }
        little(UInt64(0))
        little(UInt64(0))
        // Base level first: 12 images of 5 × 5, 2 × 2 and 1 × 1 RGBA8 texels, smallest stored first.
        for (offset, length) in [(536, 1_200), (344, 192), (296, 48)] as [(UInt64, UInt64)] {
            little(offset)
            little(length)
            little(length)
        }
        little(UInt32(92))
        little(UInt32(0))
        little(UInt16(2))
        little(UInt16(88))
        data.append(contentsOf: [1, 1, 2, 0]) // RGBSDA, sRGB primaries, sRGB transfer, straight alpha.
        little(UInt32(0))
        little(UInt32(4))
        little(UInt32(0))
        let channels: [UInt8] = [0, 1, 2, 0x1F]
        for (index, channel) in channels.enumerated() {
            little(UInt16(index * 8))
            data.append(7)
            data.append(channel) // Alpha stays linear.
            little(UInt32(0))
            little(UInt32(0))
            little(UInt32(255))
        }
        little(UInt32(18))
        data.append(Data("KTXorientation\0rd\0".utf8))
        data.append(contentsOf: [0, 0])
        little(UInt32(21))
        data.append(Data("KTXwriter\0Frostfield\0".utf8))
        data.append(contentsOf: [0, 0, 0])
        return data
    }()

    // MARK: Verification

    private func verifyStructure(_ file: Data, _ format: FormatCase, cubes: Int, payloads: [[[Data]]]) {
        for level in payloads {
            // Every face of every cube differs, so an ordering mistake cannot go unseen.
            #expect(Set(level.flatMap { $0 }).count == cubes * 6)
        }
        #expect(Array(file.prefix(12)) == [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])
        #expect(word(file, 12) == format.vulkan && word(file, 16) == UInt32(format.typeSize))
        #expect(word(file, 20) == 5 && word(file, 24) == 5 && word(file, 28) == 0)
        #expect(word(file, 32) == UInt32(cubes) && word(file, 36) == 6)
        #expect(word(file, 40) == UInt32(Self.mipCount) && word(file, 44) == 0)
        #expect(long(file, 64) == 0 && long(file, 72) == 0)

        let dfd = Int(word(file, 48))
        let dfdLength = Int(word(file, 52))
        #expect(dfd == 80 + 24 * Self.mipCount && dfdLength == 28 + format.channels * 16)
        let kvd = Int(word(file, 56))
        let kvdLength = Int(word(file, 60))
        #expect(kvd == dfd + dfdLength)
        let entries = keyValues(file, offset: kvd, length: kvdLength)
        // Sorted, so neither KTXcubemapIncomplete nor anything else is present.
        #expect(entries.map(\.key) == ["KTXorientation", "KTXwriter"])
        #expect(entries.map(\.value) == ["rd", "Frostfield"])
        #expect(file.range(of: Data("KTXcubemapIncomplete".utf8)) == nil)

        var preceding = file.count
        for level in 0..<Self.mipCount {
            let index = 80 + level * 24
            let offset = Int(long(file, index))
            let length = Int(long(file, index + 8))
            let side = max(1, Self.side >> level)
            let imageBytes = side * side * format.pixelBytes
            #expect(length == imageBytes * cubes * 6 && long(file, index + 16) == UInt64(length))
            #expect(offset.isMultiple(of: max(4, format.pixelBytes)) && offset + length <= preceding)
            #expect(file[(offset + length)..<preceding].allSatisfy { $0 == 0 })
            if level == 0 { #expect(offset + length == file.count) }
            preceding = offset
            for cube in 0..<cubes {
                for face in 0..<6 {
                    // KTX2 3.13.1 concatenates layers, then faces. A layer is a whole cube.
                    let start = offset + (cube * 6 + face) * imageBytes
                    #expect(Data(file[start..<(start + imageBytes)]) == payloads[level][cube][face])
                }
            }
        }
        #expect(file[(kvd + kvdLength)..<preceding].allSatisfy { $0 == 0 })
    }

    /// The repo importer does not read cube arrays, so a cube-array file must be rejected, while the
    /// same bytes declared as a 2D array of `6 × cubes` layers, which KTX2 stores identically,
    /// import through Metal and expose every face at slice `6 × cube + face`.
    private func verifyByImportingAsFlattenedArray(
        _ file: Data, _ format: FormatCase, cubes: Int, payloads: [[[Data]]], device: any MTLDevice
    ) throws {
        let loader = TextureLoader(device: device)
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("cube map arrays")) { try loader.loadKTX2(data: file) }

        var flattened = file
        setWord(UInt32(cubes * 6), at: 32, in: &flattened)
        setWord(1, at: 36, in: &flattened)
        let loaded = try loader.loadKTX2(data: flattened)
        #expect(loaded.textureType == .type2DArray && loaded.arrayLength == cubes * 6)
        #expect(loaded.pixelFormat == format.metal && loaded.mipmapLevelCount == Self.mipCount)
        for level in 0..<Self.mipCount {
            let side = max(1, Self.side >> level)
            for cube in 0..<cubes {
                for face in 0..<6 {
                    #expect(
                        get(loaded, level: level, slice: cube * 6 + face, side: side, pixelBytes: format.pixelBytes)
                            == payloads[level][cube][face]
                    )
                }
            }
        }
    }

    /// Writes the file for an external validator when `FROSTFIELD_KTX2_EXPORT_DIR` names a directory.
    private func exportForExternalValidation(_ file: Data, name: String) throws {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_KTX2_EXPORT_DIR"] else { return }
        try file.write(to: URL(fileURLWithPath: directory, isDirectory: true).appendingPathComponent(name))
    }

    private func keyValues(_ file: Data, offset start: Int, length: Int) -> [(key: String, value: String)] {
        var entries: [(key: String, value: String)] = []
        var offset = start
        while offset < start + length {
            let entryLength = Int(word(file, offset))
            offset += 4
            let entry = file[offset..<(offset + entryLength)]
            guard let separator = entry.firstIndex(of: 0) else { break }
            entries.append((
                String(decoding: entry[..<separator], as: UTF8.self),
                String(decoding: entry[(separator + 1)...].dropLast(), as: UTF8.self)
            ))
            offset += (entryLength + 3) & ~3
        }
        #expect(offset == start + length)
        return entries
    }

    private func get(_ texture: any MTLTexture, level: Int, slice: Int, side: Int, pixelBytes: Int) -> Data {
        var bytes = Data(count: side * side * pixelBytes)
        bytes.withUnsafeMutableBytes {
            texture.getBytes(
                $0.baseAddress!, bytesPerRow: side * pixelBytes, bytesPerImage: side * side * pixelBytes,
                from: MTLRegionMake3D(0, 0, 0, side, side, 1), mipmapLevel: level, slice: slice
            )
        }
        return bytes
    }

    private static func raw<T>(_ values: [T]) -> Data { values.withUnsafeBytes { Data($0) } }

    private func setWord(_ value: UInt32, at offset: Int, in data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.replaceSubrange(offset..<(offset + 4), with: $0) }
    }

    private func word(_ data: Data, _ offset: Int) -> UInt32 {
        data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }

    private func long(_ data: Data, _ offset: Int) -> UInt64 {
        data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) }
    }
}
