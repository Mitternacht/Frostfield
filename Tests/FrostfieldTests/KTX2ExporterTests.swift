import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct KTX2ExporterTests {
    struct FormatCase: Sendable {
        let metal: MTLPixelFormat
        let vulkan: UInt32
        let channels: Int
        let typeSize: Int
    }

    static let formats: [FormatCase] = {
        var formats = [
            FormatCase(metal: .r8Unorm, vulkan: 9, channels: 1, typeSize: 1),
            FormatCase(metal: .rg8Unorm, vulkan: 16, channels: 2, typeSize: 1),
            FormatCase(metal: .rgba8Unorm, vulkan: 37, channels: 4, typeSize: 1),
            FormatCase(metal: .rgba8Unorm_srgb, vulkan: 43, channels: 4, typeSize: 1),
            FormatCase(metal: .bgra8Unorm, vulkan: 44, channels: 4, typeSize: 1),
            FormatCase(metal: .bgra8Unorm_srgb, vulkan: 50, channels: 4, typeSize: 1),
            FormatCase(metal: .r16Float, vulkan: 76, channels: 1, typeSize: 2),
            FormatCase(metal: .rg16Float, vulkan: 83, channels: 2, typeSize: 2),
            FormatCase(metal: .rgba16Float, vulkan: 97, channels: 4, typeSize: 2),
            FormatCase(metal: .r32Float, vulkan: 100, channels: 1, typeSize: 4),
            FormatCase(metal: .rg32Float, vulkan: 103, channels: 2, typeSize: 4),
            FormatCase(metal: .rgba32Float, vulkan: 109, channels: 4, typeSize: 4),
        ]
        #if !targetEnvironment(simulator)
        formats.append(FormatCase(metal: .r8Unorm_srgb, vulkan: 15, channels: 1, typeSize: 1))
        formats.append(FormatCase(metal: .rg8Unorm_srgb, vulkan: 22, channels: 2, typeSize: 1))
        #endif
        return formats
    }()

    @Test("Uncompressed Metal formats preserve every odd-sized mip and KTX2 descriptor", arguments: formats)
    func formatsRoundTrip(_ format: FormatCase) throws {
        let fixture = try PathTestFixture()
        let width = 5
        let height = 3
        let source = try texture(fixture.device, format: format.metal, width: width, height: height, mips: 3)
        var payloads: [Data] = []
        for level in 0..<3 {
            let w = max(1, width >> level)
            let h = max(1, height >> level)
            let count = w * h * format.channels
            let bytes: Data
            switch format.typeSize {
            case 1:
                bytes = Data((0..<count).map { UInt8(($0 * 31 + level * 7 + 3) % 256) })
            case 2:
                bytes = raw((0..<count).map { Float16(Float($0 + level) / 8 - 1) })
            default:
                bytes = raw((0..<count).map { Float($0 + level) / 8 - 1 })
            }
            put(bytes, into: source, level: level, width: w, height: h, rowBytes: w * format.channels * format.typeSize)
            payloads.append(bytes)
        }

        let options = KTX2Exporter.Options(alphaMode: .premultiplied)
        let file = try KTX2Exporter.data(from: source, commandQueue: fixture.queue, options: options)
        #expect(try KTX2Exporter.data(from: source, options: options) == file)
        #expect(word(file, 12) == format.vulkan && word(file, 16) == UInt32(format.typeSize))
        #expect(word(file, 20) == 5 && word(file, 24) == 3 && word(file, 28) == 0)
        #expect(word(file, 32) == 0 && word(file, 36) == 1 && word(file, 40) == 3 && word(file, 44) == 0)
        #expect(long(file, 64) == 0 && long(file, 72) == 0)
        let dfd = Int(word(file, 48))
        #expect(dfd == 80 + 24 * 3)
        #expect(word(file, 52) == UInt32(28 + format.channels * 16))
        #expect(word(file, dfd) == word(file, 52))
        #expect(word(file, dfd + 8) == (UInt32(24 + format.channels * 16) << 16) | 2)
        let sRGB = [UInt32(15), 22, 43, 50].contains(format.vulkan)
        #expect(Array(file[(dfd + 12)..<(dfd + 16)]) == [1, sRGB ? 1 : 0, sRGB ? 2 : 1, format.channels == 4 ? 1 : 0])
        #expect(file[dfd + 20] == UInt8(format.channels * format.typeSize))
        let first = dfd + 28
        let expectedChannel: UInt8 = format.typeSize > 1 ? 0xC0 : (format.vulkan == 44 || format.vulkan == 50 ? 2 : 0)
        #expect(file[first + 2] == UInt8(format.typeSize * 8 - 1) && file[first + 3] == expectedChannel)
        #expect(word(file, first + 8) == (format.typeSize > 1 ? Float(-1).bitPattern : 0))
        #expect(word(file, first + 12) == (format.typeSize > 1 ? Float(1).bitPattern : 255))
        if sRGB && format.channels == 4 { #expect(file[first + 3 * 16 + 3] == 0x1F) }

        let kvd = Int(word(file, 56))
        let kvdLength = Int(word(file, 60))
        #expect(kvd == dfd + Int(word(file, 52)))
        #expect(file[kvd..<(kvd + kvdLength)].range(of: Data("KTXorientation\0rd\0".utf8)) != nil)
        let loaded = try TextureLoader(device: fixture.device).load(data: file)
        #expect(loaded.textureType == .type2D && loaded.pixelFormat == source.pixelFormat && loaded.mipmapLevelCount == 3)
        var previousOffset = file.count
        for level in 0..<3 {
            let offset = Int(long(file, 80 + level * 24))
            let length = Int(long(file, 88 + level * 24))
            #expect(length == Int(long(file, 96 + level * 24)))
            #expect(offset % max(4, format.channels * format.typeSize) == 0 && offset + length <= previousOffset)
            #expect(file[(offset + length)..<previousOffset].allSatisfy { $0 == 0 })
            #expect(Data(file[offset..<(offset + length)]) == payloads[level])
            previousOffset = offset
            let w = max(1, width >> level)
            let h = max(1, height >> level)
            #expect(get(loaded, level: level, width: w, height: h, rowBytes: w * format.channels * format.typeSize) == payloads[level])
        }
        #expect(file[(kvd + kvdLength)..<previousOffset].allSatisfy { $0 == 0 })
    }

    @Test("Private Metal render-target readback preserves signed HDR values")
    func privateHDR() throws {
        let fixture = try PathTestFixture()
        let source = try texture(
            fixture.device, format: .rgba16Float, width: 3, height: 2, mips: 2,
            storageMode: .private, usage: .renderTarget
        )
        let command = try #require(fixture.queue.makeCommandBuffer())
        let colors = [SIMD4<Float16>(2.5, -1.5, 0.125, 0.5), SIMD4<Float16>(8, -4, 1, 0.25)]
        for (level, color) in colors.enumerated() {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = source
            pass.colorAttachments[0].level = level
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(
                red: Double(color.x), green: Double(color.y), blue: Double(color.z), alpha: Double(color.w)
            )
            try #require(command.makeRenderCommandEncoder(descriptor: pass)).endEncoding()
        }
        command.commit()
        let file = try KTX2Exporter.data(
            from: source, commandQueue: fixture.queue, options: .init(colorPrimaries: .sRGB)
        )
        #expect(command.status == .completed)
        let loaded = try TextureLoader(device: fixture.device).load(data: file)
        #expect(loaded.pixelFormat == .rgba16Float && loaded.mipmapLevelCount == 2)
        #expect(file[Int(word(file, 48)) + 13] == 1)
        #expect(get(loaded, level: 0, width: 3, height: 2, rowBytes: 24) == raw(Array(repeating: colors[0], count: 6)))
        #expect(get(loaded, level: 1, width: 1, height: 1, rowBytes: 8) == raw([colors[1]]))
    }

    @Test("Every cube face and mip survives native KTX2 export and upload")
    func cube() throws {
        let fixture = try PathTestFixture()
        let source = try texture(fixture.device, format: .rgba8Unorm_srgb, type: .typeCube, width: 4, height: 4, mips: 3)
        var payloads: [[Data]] = []
        for level in 0..<3 {
            let side = 4 >> level
            var faces: [Data] = []
            for face in 0..<6 {
                let bytes = Data((0..<(side * side)).flatMap { pixel in
                    [UInt8(face * 37 + level), UInt8(pixel * 13), UInt8(219 - pixel * 7), UInt8(127 + face)]
                })
                put(bytes, into: source, level: level, slice: face, width: side, height: side, rowBytes: side * 4)
                faces.append(bytes)
            }
            payloads.append(faces)
        }
        let file = try KTX2Exporter.data(from: source, commandQueue: fixture.queue)
        let loader = TextureLoader(device: fixture.device)
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("cube maps")) { try loader.load(data: file) }
        let loaded = try loader.loadKTX2(data: file)
        #expect(word(file, 28) == 0 && word(file, 32) == 0 && word(file, 36) == 6)
        #expect(loaded.textureType == .typeCube && loaded.pixelFormat == .rgba8Unorm_srgb && loaded.mipmapLevelCount == 3)
        for level in 0..<3 {
            let side = 4 >> level
            let start = Int(long(file, 80 + level * 24))
            let length = Int(long(file, 88 + level * 24))
            #expect(length == side * side * 4 * 6)
            #expect(Data(file[start..<(start + length)]) == payloads[level].reduce(into: Data()) { $0.append($1) })
            for face in 0..<6 {
                #expect(get(loaded, level: level, slice: face, width: side, height: side, rowBytes: side * 4) == payloads[level][face])
            }
        }
        var bad = file
        bad[24] = 3
        #expect(throws: TextureLoader.Failure.invalidKTX2("cube faces must be square and have no depth")) {
            try loader.loadKTX2(data: bad)
        }
    }

    private static let arrayFormats: [FormatCase] = [
        .init(metal: .r8Unorm, vulkan: 9, channels: 1, typeSize: 1),
        .init(metal: .rgba8Unorm_srgb, vulkan: 43, channels: 4, typeSize: 1),
        .init(metal: .rgba16Float, vulkan: 97, channels: 4, typeSize: 2),
    ]

    @Test("2D array layers and odd-sized mips retain their stored data and KTX2 metadata",
          arguments: [1, 3], arrayFormats)
    func arrayRoundTrip(layerCount: Int, format: FormatCase) async throws {
        let fixture = try PathTestFixture()
        let width = 5, height = 3, mips = 3
        let pixelBytes = format.channels * format.typeSize
        let source = try texture(
            fixture.device, format: format.metal, type: .type2DArray, width: width, height: height,
            mips: mips, arrayLength: layerCount, storageMode: .private
        )
        let command = try #require(fixture.queue.makeCommandBuffer())
        var payloads: [[Data]] = []
        for level in 0..<mips {
            let w = max(1, width >> level)
            let h = max(1, height >> level)
            var layerPayloads: [Data] = []
            for layer in 0..<layerCount {
                let channels = 0..<(w * h * format.channels)
                let payload: Data
                if format.typeSize == 2 {
                    payload = raw(channels.map { Float16(Float($0) * 0.25 + Float(layer) * 3 - Float(level) * 11) })
                } else {
                    payload = Data(channels.map { UInt8(($0 * 29 + layer * 53 + level * 71) % 256) })
                }
                try TextureTransfer.encodeUpload(payload, to: source, mipLevel: level, slice: layer, commandBuffer: command)
                layerPayloads.append(payload)
            }
            payloads.append(layerPayloads)
        }
        command.commit()
        let file = try KTX2Exporter.data(
            from: source, commandQueue: fixture.queue, options: .init(colorPrimaries: .sRGB, alphaMode: .premultiplied)
        )
        #expect(command.status == .completed)
        #expect(word(file, 12) == format.vulkan && word(file, 16) == UInt32(format.typeSize))
        #expect(word(file, 20) == 5 && word(file, 24) == 3 && word(file, 28) == 0)
        #expect(word(file, 32) == UInt32(layerCount) && word(file, 36) == 1 && word(file, 40) == UInt32(mips))
        #expect(word(file, 44) == 0 && long(file, 64) == 0 && long(file, 72) == 0)
        let dfd = Int(word(file, 48))
        let transfer: UInt8 = format.metal == .rgba8Unorm_srgb ? 2 : 1
        #expect(Array(file[(dfd + 12)..<(dfd + 16)]) == [1, 1, transfer, format.channels == 4 ? 1 : 0])
        let kvd = Int(word(file, 56))
        let kvdEnd = kvd + Int(word(file, 60))
        #expect(file[kvd..<kvdEnd].range(of: Data("KTXorientation\0rd\0".utf8)) != nil)

        let loader = TextureLoader(device: fixture.device)
        let loaded = try loader.loadKTX2(data: file)
        #expect(loaded.textureType == .type2DArray && loaded.arrayLength == layerCount)
        #expect(loaded.pixelFormat == format.metal && loaded.mipmapLevelCount == mips)
        if layerCount == 1 {
            #expect(try loader.load(data: file).textureType == .type2D)
        } else {
            #expect(throws: TextureLoader.Failure.unsupportedKTX2("texture arrays")) { try loader.load(data: file) }
        }

        var precedingOffset = file.count
        for level in 0..<mips {
            let index = 80 + level * 24
            let offset = Int(long(file, index))
            let length = Int(long(file, index + 8))
            let w = max(1, width >> level)
            let h = max(1, height >> level)
            #expect(length == w * h * pixelBytes * layerCount)
            #expect(long(file, index + 16) == UInt64(length))
            #expect(offset.isMultiple(of: max(4, pixelBytes)) && offset + length <= precedingOffset)
            #expect(file[(offset + length)..<precedingOffset].allSatisfy { $0 == 0 })
            #expect(Data(file[offset..<(offset + length)]) == payloads[level].reduce(into: Data()) { $0.append($1) })
            precedingOffset = offset
            for layer in 0..<layerCount {
                let stored = try await TextureTransfer.read(from: loaded, mipLevel: level, slice: layer, commandQueue: fixture.queue)
                #expect(stored.size.width == w && stored.size.height == h && stored.size.depth == 1)
                #expect(stored.bytes == payloads[level][layer])
            }
        }
        #expect(file[kvdEnd..<precedingOffset].allSatisfy { $0 == 0 })
    }

    @Test("Array file budget includes all layers and metadata before readback")
    func oversizedArray() throws {
        let fixture = try PathTestFixture()
        // Exactly 64 MiB of texels, before the KTX2 header, index and metadata.
        let source = try texture(
            fixture.device, format: .rgba32Float, type: .type2DArray, width: 512, height: 512,
            arrayLength: 16, storageMode: .private
        )
        #expect(throws: KTX2Exporter.Failure.size) {
            try KTX2Exporter.data(from: source, commandQueue: fixture.queue)
        }
    }

    @Test("Volume mips preserve depth planes and their orientation")
    func volume() throws {
        let fixture = try PathTestFixture()
        let source = try texture(
            fixture.device, format: .rg16Float, type: .type3D, width: 5, height: 3, depth: 2, mips: 3
        )
        var payloads: [Data] = []
        for level in 0..<3 {
            let w = max(1, 5 >> level)
            let h = max(1, 3 >> level)
            let d = max(1, 2 >> level)
            let bytes = raw((0..<(w * h * d * 2)).map { Float16(Float($0) / 4 - Float(level) * 8) })
            put(bytes, into: source, level: level, width: w, height: h, depth: d, rowBytes: w * 4)
            payloads.append(bytes)
        }
        let file = try KTX2Exporter.data(from: source, commandQueue: fixture.queue)
        let loader = TextureLoader(device: fixture.device)
        #expect(throws: TextureLoader.Failure.unsupportedKTX2("3D textures")) { try loader.load(data: file) }
        let loaded = try loader.loadKTX2(data: file)
        #expect(word(file, 28) == 2 && word(file, 36) == 1)
        #expect(file.range(of: Data("KTXorientation\0rdi\0".utf8)) != nil)
        #expect(loaded.textureType == .type3D && loaded.depth == 2 && loaded.pixelFormat == .rg16Float)
        #expect(loaded.mipmapLevelCount == 3)
        for level in 0..<3 {
            let w = max(1, 5 >> level)
            let h = max(1, 3 >> level)
            let d = max(1, 2 >> level)
            #expect(Int(long(file, 88 + level * 24)) == w * h * d * 4)
            #expect(get(loaded, level: level, width: w, height: h, depth: d, rowBytes: w * 4) == payloads[level])
        }
    }

    @Test("Unsupported Metal formats and texture layouts are rejected")
    func unsupported() throws {
        let fixture = try PathTestFixture()
        let uint = try texture(fixture.device, format: .r8Uint, width: 2, height: 2)
        #expect(throws: KTX2Exporter.Failure.pixelFormat(.r8Uint)) { try KTX2Exporter.data(from: uint) }
        let oneDimension = try texture(
            fixture.device, format: .rgba8Unorm, type: .type1DArray, width: 2, height: 1,
            arrayLength: 2, storageMode: .private
        )
        #expect(throws: KTX2Exporter.Failure.textureType) { try KTX2Exporter.data(from: oneDimension) }
    }

    #if targetEnvironment(simulator)
    @Test("Narrow sRGB KTX2 files reject unavailable Simulator formats before Metal allocation")
    func simulatorNarrowSRGB() throws {
        let fixture = try PathTestFixture()
        let loader = TextureLoader(device: fixture.device)
        let cases: [(format: MTLPixelFormat, vk: UInt8, bytes: Data, name: String)] = [
            (.r8Unorm, 15, Data([128]), "R8 sRGB"), (.rg8Unorm, 22, Data([128, 64]), "RG8 sRGB"),
        ]
        for example in cases {
            let source = try texture(fixture.device, format: example.format, width: 1, height: 1)
            put(example.bytes, into: source, level: 0, width: 1, height: 1, rowBytes: example.bytes.count)
            var file = try KTX2Exporter.data(from: source)
            file[12] = example.vk
            let dfd = Int(word(file, 48))
            file[dfd + 13] = 1 // sRGB primaries.
            file[dfd + 14] = 2 // sRGB transfer.
            #expect(throws: TextureLoader.Failure.unsupportedTextureFormat(example.name)) {
                try loader.load(data: file)
            }
            let linear = try loader.load(data: file, interpretation: .linearData)
            #expect(linear.pixelFormat == example.format)
            #expect(get(linear, level: 0, width: 1, height: 1, rowBytes: example.bytes.count) == example.bytes)
        }
    }
    #endif

    private func texture(
        _ device: any MTLDevice, format: MTLPixelFormat, type: MTLTextureType = .type2D,
        width: Int, height: Int, depth: Int = 1, mips: Int = 1, arrayLength: Int = 1,
        storageMode: MTLStorageMode = .shared, usage: MTLTextureUsage = .shaderRead
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = type
        descriptor.pixelFormat = format
        descriptor.width = width
        descriptor.height = height
        descriptor.depth = depth
        descriptor.mipmapLevelCount = mips
        descriptor.arrayLength = arrayLength
        descriptor.storageMode = storageMode
        descriptor.usage = usage
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func put(
        _ bytes: Data, into texture: any MTLTexture, level: Int, slice: Int = 0, width: Int, height: Int, depth: Int = 1,
        rowBytes: Int
    ) {
        bytes.withUnsafeBytes {
            texture.replace(
                region: MTLRegionMake3D(0, 0, 0, width, height, depth), mipmapLevel: level, slice: slice,
                withBytes: $0.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: rowBytes * height
            )
        }
    }

    private func get(
        _ texture: any MTLTexture, level: Int, slice: Int = 0, width: Int, height: Int, depth: Int = 1, rowBytes: Int
    ) -> Data {
        var bytes = Data(count: rowBytes * height * depth)
        bytes.withUnsafeMutableBytes {
            texture.getBytes(
                $0.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: rowBytes * height,
                from: MTLRegionMake3D(0, 0, 0, width, height, depth), mipmapLevel: level, slice: slice
            )
        }
        return bytes
    }

    private func raw<T>(_ values: [T]) -> Data { values.withUnsafeBytes { Data($0) } }

    private func word(_ data: Data, _ offset: Int) -> UInt32 {
        data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }

    private func long(_ data: Data, _ offset: Int) -> UInt64 {
        data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) }
    }
}
