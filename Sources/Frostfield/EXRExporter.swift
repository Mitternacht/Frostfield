import Compression
import Darwin
import Foundation
import FrostfieldEXRPIZ
import Metal

/// Writes scanline OpenEXR images with RGBA or named auxiliary channels.
///
/// Values are written bit for bit: `.float` stores each 32-bit float, and `.half` stores
/// `Float16(value)`, where magnitudes above 65,504 round to infinity as IEEE conversion does.
/// Rows go from the top of the image to the bottom with increasing Y, matching Frostfield's
/// top-left texture origin. The header declares linear Rec. 709 primaries with a D65 white
/// point through the standard `chromaticities` attribute.
///
/// ImageIO can also write OpenEXR, but on macOS 27.2 its encoder applies a color conversion
/// that leaks about 2 × 10⁻⁵ of each pixel's largest channel into the others, so a scene-linear
/// pixel such as (1000, 0.1, 7.5) reads back with a green value near 0.117. This exporter
/// avoids that conversion. [OpenEXR file layout](https://openexr.com/en/latest/OpenEXRFileLayout.html)
public struct EXRExporter {
    public enum PixelType: Equatable, Sendable {
        /// IEEE half precision, as `HALF` channels.
        case half
        /// IEEE single precision, as `FLOAT` channels.
        case float
    }

    /// A planar channel. Dotted names such as `normal.X` group channels into layers.
    public struct Channel: Sendable {
        public enum Samples: Sendable {
            case half([Float16])
            case float([Float])
            case unsigned([UInt32])

            fileprivate var code: Int32 {
                switch self {
                case .unsigned: 0
                case .half: 1
                case .float: 2
                }
            }

            fileprivate var count: Int {
                switch self {
                case .half(let values): values.count
                case .float(let values): values.count
                case .unsigned(let values): values.count
                }
            }
        }

        public let name: String
        public let samples: Samples

        public init(name: String, samples: Samples) {
            self.name = name
            self.samples = samples
        }
    }

    public enum Compression: Equatable, Sendable {
        /// `NO_COMPRESSION`: stored scanlines.
        case uncompressed
        /// `RLE_COMPRESSION`: lossless byte runs over one scanline per block.
        case rle
        /// `ZIPS_COMPRESSION`: zlib over one scanline per block.
        case zipScanline
        /// `ZIP_COMPRESSION`: zlib over sixteen scanlines per block.
        case zip
        /// `PIZ_COMPRESSION`: lossless wavelet and Huffman coding over up to thirty-two scanlines.
        /// Blocks above 64 MiB are stored directly to bound compression scratch space.
        case piz

        fileprivate var code: UInt8 {
            switch self {
            case .uncompressed: 0
            case .rle: 1
            case .zipScanline: 2
            case .zip: 3
            case .piz: 4
            }
        }

        fileprivate var linesPerBlock: Int {
            switch self {
            case .zip: 16
            case .piz: 32
            default: 1
            }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        /// Width and height must each be in 1...16384 and match the pixel count.
        case invalidDimensions(width: Int, height: Int, pixelCount: Int)
        case invalidChannelCount(Int)
        case invalidChannelName(String)
        case duplicateChannelName(String)
        case invalidChannelSamples(name: String, expected: Int, actual: Int)
        case layeredSizeLimit
        case invalidStreamDimensions(width: Int, height: Int)
        case invalidFileURL
        case streamChannelLayout(firstRow: Int)
        /// Texture export reads RGBA16Float and RGBA32Float.
        case unsupportedTextureFormat(MTLPixelFormat)
        /// Texture export reads level 0 of a readable, single-sample 2D texture.
        case unsupportedTexture(String)
        /// The command queue belongs to a different device than the texture.
        case wrongDevice
        /// Metal could not create the named resource, or the readback did not complete.
        case resource(String)

        public var errorDescription: String? {
            switch self {
            case .invalidDimensions(let width, let height, let count):
                "An EXR image needs 1...16384 pixels per side and width × height pixels; got \(width) × \(height) with \(count) pixels."
            case .invalidChannelCount(let count): "A layered EXR needs 1...64 channels; got \(count)."
            case .invalidChannelName(let name): "EXR channel names need 1...31 printable ASCII bytes; got \(name.debugDescription)."
            case .duplicateChannelName(let name): "The EXR channel \(name.debugDescription) appears more than once."
            case .invalidChannelSamples(let name, let expected, let actual):
                "EXR channel \(name.debugDescription) needs \(expected) samples; got \(actual)."
            case .layeredSizeLimit: "Layered EXR export supports up to 512 MiB of uncompressed channel data."
            case .invalidStreamDimensions(let width, let height):
                "A streamed EXR needs 1...32768 pixels per side; got \(width) × \(height)."
            case .invalidFileURL: "EXR streaming requires a destination file URL."
            case .streamChannelLayout(let row): "EXR channel names and sample types changed at row \(row)."
            case .unsupportedTextureFormat(let format):
                "Pixel format \(format.rawValue) cannot be exported as EXR. Use RGBA16Float or RGBA32Float."
            case .unsupportedTexture(let reason): "The texture cannot be exported. \(reason)"
            case .wrongDevice: "The export command queue belongs to a different Metal device than the texture."
            case .resource(let name): "Metal could not provide the \(name) for EXR export."
            }
        }
    }

    public var pixelType: PixelType
    public var compression: Compression
    /// Copies private and managed textures. Without a queue, the exporter creates one.
    public var commandQueue: (any MTLCommandQueue)?

    public init(pixelType: PixelType = .half, compression: Compression = .zip, commandQueue: (any MTLCommandQueue)? = nil) {
        self.pixelType = pixelType
        self.compression = compression
        self.commandQueue = commandQueue
    }

    /// Encodes RGBA pixels in rows from the top of the image.
    public func data(width: Int, height: Int, pixels: [SIMD4<Float>]) throws -> Data {
        guard (1...16_384).contains(width), (1...16_384).contains(height), pixels.count == width * height else {
            throw Failure.invalidDimensions(width: width, height: height, pixelCount: pixels.count)
        }
        let type: Int32 = pixelType == .half ? 1 : 2
        return encode(width: width, height: height, channels: ["A", "B", "G", "R"].map { ($0, type) }) { rows in
            scanlines(rows, width: width, pixels: pixels)
        }
    }

    /// Encodes 1 to 64 named channels in a single-part scanline image.
    ///
    /// Each channel supplies width × height samples in top-left row order.
    /// Names must be unique and contain 1 to 31 printable ASCII bytes. Dimensions
    /// may reach 16,384 per axis and the uncompressed payload may reach 512 MiB.
    /// Channel types come from `samples`; the exporter's `pixelType` applies only
    /// to the RGBA convenience methods.
    public func data(width: Int, height: Int, channels: [Channel]) throws -> Data {
        guard (1...16_384).contains(width), (1...16_384).contains(height) else {
            throw Failure.invalidDimensions(width: width, height: height, pixelCount: channels.first?.samples.count ?? 0)
        }
        let ordered = try orderedChannels(channels, pixelCount: width * height)
        let bytesPerPixel = ordered.reduce(0) { $0 + ($1.samples.code == 1 ? 2 : 4) }
        guard width * height * bytesPerPixel <= 512 * 1_024 * 1_024 else { throw Failure.layeredSizeLimit }
        return encode(width: width, height: height, channels: ordered.map { ($0.name, $0.samples.code) }) { rows in
            Self.packScanlines(rows, width: width, channels: ordered)
        }
    }

    /// Writes a scanline EXR using up to 32,768 pixels per side without holding
    /// the whole image in memory. The provider returns one planar channel array
    /// per channel, containing only the requested full-width rows in top-left
    /// order. The first block defines the channel names and types. Later blocks
    /// must supply the same names and types and may reorder channels.
    ///
    /// Requests arrive once per block, top to bottom, in groups of up to thirty-two
    /// rows for PIZ, sixteen for ZIP, or one for other modes. A thrown error leaves
    /// any existing destination file intact.
    public func write(
        to url: URL, width: Int, height: Int,
        scanlines: (Range<Int>) throws -> [Channel]
    ) throws {
        guard (1...32_768).contains(width), (1...32_768).contains(height) else {
            throw Failure.invalidStreamDimensions(width: width, height: height)
        }
        guard url.isFileURL else { throw Failure.invalidFileURL }

        let lines = compression.linesPerBlock
        let blockCount = (height + lines - 1) / lines
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".frostfield-\(UUID().uuidString).exr")
        let descriptor = try temporary.withUnsafeFileSystemRepresentation { path in
            guard let path else { throw Failure.invalidFileURL }
            let descriptor = Darwin.open(path, O_WRONLY | O_CREAT | O_EXCL, 0o666)
            guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            return descriptor
        }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer {
            try? file.close()
            try? FileManager.default.removeItem(at: temporary)
        }

        var layout: [(name: String, type: Int32)] = []
        var sampleBytes: [UInt8] = []
        var tableOffset: UInt64 = 0
        var offset: UInt64 = 0
        var table = Data()
        table.reserveCapacity(blockCount * 8)
        for block in 0..<blockCount {
            try Task.checkCancellation()
            let first = block * lines
            let rows = first..<min(first + lines, height)
            let ordered = try orderedChannels(scanlines(rows), pixelCount: rows.count * width)
            let types = ordered.map { (name: $0.name, type: $0.samples.code) }
            if block == 0 {
                layout = types
                if compression == .piz { sampleBytes = layout.map { $0.type == 1 ? 2 : 4 } }
                let header = header(width: width, height: height, channels: layout)
                try file.write(contentsOf: header)
                try file.write(contentsOf: Data(count: blockCount * 8))
                tableOffset = UInt64(header.count)
                offset = tableOffset + UInt64(blockCount * 8)
            } else if !types.elementsEqual(layout, by: { $0.name == $1.name && $0.type == $1.type }) {
                throw Failure.streamChannelLayout(firstRow: first)
            }
            let raw = Self.packScanlines(0..<rows.count, width: width, channels: ordered)
            let encoded = scanlineBlock(raw, rows: rows, width: width, sampleBytes: sampleBytes)
            table.appendLittleEndian(offset)
            try file.write(contentsOf: encoded)
            offset += UInt64(encoded.count)
        }
        try Task.checkCancellation()
        try file.seek(toOffset: tableOffset)
        try file.write(contentsOf: table)
        try file.synchronize()
        try file.close()
        try temporary.withUnsafeFileSystemRepresentation { source in
            try url.withUnsafeFileSystemRepresentation { destination in
                guard let source, let destination else { throw Failure.invalidFileURL }
                guard Darwin.rename(source, destination) == 0 else {
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
            }
        }
    }

    private func orderedChannels(_ channels: [Channel], pixelCount: Int) throws -> [Channel] {
        guard (1...64).contains(channels.count) else { throw Failure.invalidChannelCount(channels.count) }
        var names: Set<String> = []
        for channel in channels {
            let name = channel.name.utf8
            guard (1...31).contains(name.count), name.allSatisfy({ (32...126).contains($0) }) else {
                throw Failure.invalidChannelName(channel.name)
            }
            guard names.insert(channel.name).inserted else { throw Failure.duplicateChannelName(channel.name) }
            guard channel.samples.count == pixelCount else {
                throw Failure.invalidChannelSamples(name: channel.name, expected: pixelCount, actual: channel.samples.count)
            }
        }
        return channels.sorted { $0.name < $1.name }
    }

    private static func packScanlines(_ rows: Range<Int>, width: Int, channels: [Channel]) -> Data {
        let bytesPerPixel = channels.reduce(0) { $0 + ($1.samples.code == 1 ? 2 : 4) }
        var data = Data()
        data.reserveCapacity(rows.count * width * bytesPerPixel)
        for row in rows {
            let range = (row * width)..<((row + 1) * width)
            for channel in channels {
                switch channel.samples {
                case .half(let values):
                    for value in values[range] { data.appendLittleEndian(value.bitPattern) }
                case .float(let values):
                    for value in values[range] { data.appendLittleEndian(value.bitPattern) }
                case .unsigned(let values):
                    for value in values[range] { data.appendLittleEndian(value) }
                }
            }
        }
        return data
    }

    private func encode(
        width: Int, height: Int, channels: [(name: String, type: Int32)], scanlines: (Range<Int>) -> Data
    ) -> Data {
        var output = header(width: width, height: height, channels: channels)
        let lines = compression.linesPerBlock
        let blockCount = (height + lines - 1) / lines
        let tableOffset = output.count
        output.append(Data(count: blockCount * 8))
        var offsets: [UInt64] = []
        let sampleBytes: [UInt8] = compression == .piz ? channels.map { $0.type == 1 ? 2 : 4 } : []
        for block in 0..<blockCount {
            let first = block * lines
            let rows = first..<min(first + lines, height)
            let raw = scanlines(rows)
            offsets.append(UInt64(output.count))
            output.append(scanlineBlock(raw, rows: rows, width: width, sampleBytes: sampleBytes))
        }
        var table = Data()
        for offset in offsets { table.appendLittleEndian(offset) }
        output.replaceSubrange(tableOffset..<(tableOffset + table.count), with: table)
        return output
    }

    private func header(width: Int, height: Int, channels: [(name: String, type: Int32)]) -> Data {
        var output = Data()
        output.append(contentsOf: [0x76, 0x2F, 0x31, 0x01])
        output.appendLittleEndian(UInt32(2)) // version 2, single-part scanline
        var channelList = Data()
        for (name, type) in channels {
            channelList.append(contentsOf: Array(name.utf8) + [0])
            channelList.appendLittleEndian(type)
            channelList.append(contentsOf: [0, 0, 0, 0]) // pLinear and reserved
            channelList.appendLittleEndian(Int32(1))
            channelList.appendLittleEndian(Int32(1))
        }
        channelList.append(0)
        var window = Data()
        for value in [0, 0, width - 1, height - 1] { window.appendLittleEndian(Int32(value)) }
        var chromaticities = Data()
        for value: Float in [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290] {
            chromaticities.appendLittleEndian(value.bitPattern)
        }
        var aspect = Data(), center = Data(), screenWidth = Data()
        aspect.appendLittleEndian(Float(1).bitPattern)
        center.appendLittleEndian(UInt64(0))
        screenWidth.appendLittleEndian(Float(1).bitPattern)
        let attributes: [(String, String, Data)] = [
            ("channels", "chlist", channelList),
            ("chromaticities", "chromaticities", chromaticities),
            ("compression", "compression", Data([compression.code])),
            ("dataWindow", "box2i", window),
            ("displayWindow", "box2i", window),
            ("lineOrder", "lineOrder", Data([0])),
            ("pixelAspectRatio", "float", aspect),
            ("screenWindowCenter", "v2f", center),
            ("screenWindowWidth", "float", screenWidth),
        ]
        for (name, kind, value) in attributes {
            output.append(contentsOf: Array(name.utf8) + [0] + Array(kind.utf8) + [0])
            output.appendLittleEndian(Int32(value.count))
            output.append(value)
        }
        output.append(0)

        return output
    }

    private func scanlineBlock(_ raw: Data, rows: Range<Int>, width: Int, sampleBytes: [UInt8]) -> Data {
        let stored: Data = switch compression {
        case .uncompressed: raw
        case .rle: Self.rle(raw) ?? raw
        case .zipScanline, .zip: Self.zip(raw) ?? raw
        case .piz: Self.piz(raw, width: width, rowCount: rows.count, sampleBytes: sampleBytes) ?? raw
        }
        var chunk = Data()
        chunk.reserveCapacity(8 + stored.count)
        chunk.appendLittleEndian(Int32(rows.lowerBound))
        chunk.appendLittleEndian(Int32(stored.count))
        chunk.append(stored)
        return chunk
    }

    private static func piz(_ raw: Data, width: Int, rowCount: Int, sampleBytes: [UInt8]) -> Data? {
        // Eight PIZ header bytes plus at least twenty Huffman header bytes.
        guard raw.count > 28, raw.count <= Int(FF_EXR_PIZ_MAX_BLOCK_BYTES) else { return nil }
        var output = Data(count: raw.count - 1)
        let written = raw.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { target in
                ff_exr_compress_piz(
                    source.baseAddress, source.count, UInt32(width), UInt32(rowCount),
                    sampleBytes, sampleBytes.count, target.baseAddress, target.count
                )
            }
        }
        guard written > 0 else { return nil }
        output.count = written
        return output
    }

    /// Reads level 0 of an RGBA16Float or RGBA32Float 2D texture and encodes it.
    /// Shared textures are read directly, so finish GPU writes first. Private and managed
    /// textures are copied with a blit on `commandQueue`, and this call waits for the copy.
    public func data(from texture: any MTLTexture) throws -> Data {
        let bytesPerPixel: Int
        switch texture.pixelFormat {
        case .rgba16Float: bytesPerPixel = 8
        case .rgba32Float: bytesPerPixel = 16
        default: throw Failure.unsupportedTextureFormat(texture.pixelFormat)
        }
        guard texture.textureType == .type2D, texture.sampleCount == 1, !texture.isFramebufferOnly else {
            throw Failure.unsupportedTexture("Only single-sample, readable 2D textures can be exported.")
        }
        let width = texture.width, height = texture.height
        let rowBytes = width * bytesPerPixel
        var bytes = [UInt8](repeating: 0, count: rowBytes * height)
        let region = MTLRegionMake2D(0, 0, width, height)
        switch texture.storageMode {
        case .shared:
            bytes.withUnsafeMutableBytes { raw in
                texture.getBytes(raw.baseAddress!, bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
            }
        case .private, .managed:
            if let commandQueue, commandQueue.device !== texture.device { throw Failure.wrongDevice }
            guard let queue = commandQueue ?? texture.device.makeCommandQueue(),
                  let buffer = texture.device.makeBuffer(length: bytes.count, options: .storageModeShared),
                  let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
                throw Failure.resource("readback buffer")
            }
            command.label = "Frostfield EXR export"
            blit.copy(
                from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffer, destinationOffset: 0,
                destinationBytesPerRow: rowBytes, destinationBytesPerImage: rowBytes * height
            )
            blit.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw Failure.resource("completed readback") }
            bytes.withUnsafeMutableBytes { $0.baseAddress!.copyMemory(from: buffer.contents(), byteCount: $0.count) }
        default:
            throw Failure.unsupportedTexture("Memoryless textures cannot be read.")
        }
        let pixels: [SIMD4<Float>] = bytes.withUnsafeBytes { raw in
            (0..<(width * height)).map { pixel in
                if bytesPerPixel == 8 {
                    let value = raw.loadUnaligned(fromByteOffset: pixel * 8, as: SIMD4<Float16>.self)
                    return SIMD4<Float>(value)
                }
                return raw.loadUnaligned(fromByteOffset: pixel * 16, as: SIMD4<Float>.self)
            }
        }
        return try data(width: width, height: height, pixels: pixels)
    }

    /// One block's scanlines: per line, the A, B, G, then R channel for every pixel.
    private func scanlines(_ rows: Range<Int>, width: Int, pixels: [SIMD4<Float>]) -> Data {
        let size = pixelType == .half ? 2 : 4
        var data = Data(count: rows.count * width * 4 * size)
        data.withUnsafeMutableBytes { raw in
            pixels.withUnsafeBufferPointer { source in
                var offset = 0
                for row in rows {
                    let line = UnsafeBufferPointer(rebasing: source[(row * width)..<((row + 1) * width)])
                    for lane in [3, 2, 1, 0] {
                        switch pixelType {
                        case .half:
                            for pixel in line {
                                raw.storeBytes(of: Float16(pixel[lane]).bitPattern.littleEndian, toByteOffset: offset, as: UInt16.self)
                                offset += 2
                            }
                        case .float:
                            for pixel in line {
                                raw.storeBytes(of: pixel[lane].bitPattern.littleEndian, toByteOffset: offset, as: UInt32.self)
                                offset += 4
                            }
                        }
                    }
                }
            }
        }
        return data
    }

    /// The shared RLE and ZIP predictor separates even and odd bytes and stores
    /// their successive differences offset by 128.
    private static func predictBytes(_ raw: Data) -> [UInt8] {
        let count = raw.count
        guard count > 0 else { return [] }
        var reordered = [UInt8](repeating: 0, count: count)
        let half = (count + 1) / 2
        raw.withUnsafeBytes { source in
            reordered.withUnsafeMutableBufferPointer { output in
                for index in 0..<count {
                    output[(index & 1) == 0 ? index >> 1 : half + (index >> 1)] = source[index]
                }
                var previous = output[0]
                for index in 1..<count {
                    let current = output[index]
                    output[index] = current &- previous &+ 128
                    previous = current
                }
            }
        }
        return reordered
    }

    /// A nonnegative packet header encodes one repeated byte. A negative
    /// header is followed by that many literal bytes.
    private static func rle(_ raw: Data) -> Data? {
        let predicted = predictBytes(raw)
        var stored = Data()
        stored.reserveCapacity(predicted.count)
        var index = 0
        while index < predicted.count {
            let start = index
            let value = predicted[index]
            index += 1
            while index < predicted.count, index - start < 128, predicted[index] == value {
                index += 1
            }
            if index - start >= 3 {
                stored.append(UInt8(index - start - 1))
                stored.append(value)
            } else {
                while index < predicted.count, index - start < 127 {
                    if index + 2 < predicted.count,
                       predicted[index] == predicted[index + 1], predicted[index] == predicted[index + 2] { break }
                    index += 1
                }
                stored.append(UInt8(bitPattern: -Int8(index - start)))
                stored.append(contentsOf: predicted[start..<index])
            }
            if stored.count >= raw.count { return nil }
        }
        return stored.isEmpty ? nil : stored
    }

    /// OpenEXR ZIP uses the byte predictor followed by zlib (RFC 1950).
    /// Returns nil when readers should treat the block as raw bytes.
    static func zip(_ raw: Data) -> Data? {
        let count = raw.count
        guard count > 0 else { return nil }
        let reordered = predictBytes(raw)
        var deflated = [UInt8](repeating: 0, count: count + 1_024)
        let written = reordered.withUnsafeBufferPointer { input in
            deflated.withUnsafeMutableBufferPointer { output in
                compression_encode_buffer(output.baseAddress!, output.count, input.baseAddress!, input.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard written > 0, written + 6 < count else { return nil }
        var stream = Data([0x78, 0x9C])
        stream.append(contentsOf: deflated.prefix(written))
        let checksum = KTX2Container.adler32(Data(reordered))
        stream.append(contentsOf: [UInt8(checksum >> 24), UInt8(checksum >> 16 & 0xFF), UInt8(checksum >> 8 & 0xFF), UInt8(checksum & 0xFF)])
        return stream
    }
}
