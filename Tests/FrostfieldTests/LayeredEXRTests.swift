import Compression
import Foundation
import Frostfield
import ImageIO
import Testing

struct LayeredEXRTests {
    @Test(arguments: [EXRExporter.Compression.uncompressed, .rle, .zipScanline, .zip])
    func `Layered scanlines preserve mixed sample types and bit patterns`(compression: EXRExporter.Compression) throws {
        let width = 37
        let height = 19
        let count = width * height
        let depths = (0..<count).map { $0.isMultiple(of: 7) ? Float.infinity : 0.25 + Float($0 / width) }
        let normals = (0..<count).map { Float16($0.isMultiple(of: 5) ? -1 : 0.5) }
        let ids = (0..<count).map { $0.isMultiple(of: 3) ? UInt32.max : 0x1234_abcd }
        let alpha = [Float](repeating: 1, count: count)
        let red = [Float](repeating: 1.5, count: count)
        let green = [Float](repeating: 0.25, count: count)
        let blue = [Float](repeating: 0.125, count: count)
        let data = try EXRExporter(compression: compression).data(width: width, height: height, channels: [
            .init(name: "objectID.id", samples: .unsigned(ids)),
            .init(name: "normal.X", samples: .half(normals)),
            .init(name: "depth.Z", samples: .float(depths)),
            .init(name: "R", samples: .float(red)),
            .init(name: "G", samples: .float(green)),
            .init(name: "B", samples: .float(blue)),
            .init(name: "A", samples: .float(alpha)),
        ])
        let decoded = try decodeFixture(data, width: width, height: height)
        #expect(decoded.names == ["A", "B", "G", "R", "depth.Z", "normal.X", "objectID.id"])
        #expect(decoded.types == [2, 2, 2, 2, 2, 1, 0])
        #expect(decoded.samples["A"] == alpha.map(\.bitPattern))
        #expect(decoded.samples["B"] == blue.map(\.bitPattern))
        #expect(decoded.samples["G"] == green.map(\.bitPattern))
        #expect(decoded.samples["R"] == red.map(\.bitPattern))
        #expect(decoded.samples["depth.Z"] == depths.map(\.bitPattern))
        #expect(decoded.samples["normal.X"] == normals.map { UInt32($0.bitPattern) })
        #expect(decoded.samples["objectID.id"] == ids)
        #expect((decoded.compressedBlocks > 0) == (compression != .uncompressed))

        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(image.width == width && image.height == height)
    }

    @Test
    func `Layered export rejects invalid names, duplicate channels, and incomplete planes`() throws {
        let exporter = EXRExporter()
        let channel = EXRExporter.Channel(name: "normal.X", samples: .half([0, 1]))
        #expect(throws: EXRExporter.Failure.invalidChannelCount(0)) {
            try exporter.data(width: 2, height: 1, channels: [])
        }
        #expect(throws: EXRExporter.Failure.duplicateChannelName("normal.X")) {
            try exporter.data(width: 2, height: 1, channels: [channel, channel])
        }
        #expect(throws: EXRExporter.Failure.invalidChannelSamples(name: "normal.X", expected: 4, actual: 2)) {
            try exporter.data(width: 2, height: 2, channels: [channel])
        }
        for name in ["", "depth\0Z", "あ", String(repeating: "x", count: 32)] {
            #expect(throws: EXRExporter.Failure.invalidChannelName(name)) {
                try exporter.data(width: 1, height: 1, channels: [.init(name: name, samples: .unsigned([1]))])
            }
        }
    }

    @Test
    func `Layered export bounds the uncompressed payload before encoding`() {
        let samples = [UInt32](repeating: 0, count: 4_097 * 512)
        let channels = (0..<64).map { EXRExporter.Channel(name: "layer\($0).R", samples: .unsigned(samples)) }
        #expect(throws: EXRExporter.Failure.layeredSizeLimit) {
            try EXRExporter().data(width: 4_097, height: 512, channels: channels)
        }
    }

    @Test(arguments: [EXRExporter.Compression.uncompressed, .rle, .zipScanline, .zip, .piz])
    func `Streamed blocks match the in-memory EXR and preserve mixed channel types`(
        compression: EXRExporter.Compression
    ) throws {
        let exporter = EXRExporter(compression: compression)
        let width = 129, height = 65
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: destination) }

        func channels(_ rows: Range<Int>) -> [EXRExporter.Channel] {
            let range = (rows.lowerBound * width)..<(rows.upperBound * width)
            return [
                .init(name: "normal.X", samples: .half(range.map { $0.isMultiple(of: 5) ? -1 : 0.5 })),
                .init(name: "depth.Z", samples: .float(range.map { $0.isMultiple(of: 7) ? .infinity : Float($0 / width) })),
                .init(name: "objectID.id", samples: .unsigned(range.map { $0.isMultiple(of: 3) ? UInt32.max : 0x1234_abcd })),
            ]
        }
        var requested: [Range<Int>] = []
        try exporter.write(to: destination, width: width, height: height) { rows in
            requested.append(rows)
            let values = channels(rows)
            return requested.count.isMultiple(of: 2) ? Array(values.reversed()) : values
        }
        let expected = try exporter.data(width: width, height: height, channels: channels(0..<height))
        #expect(try Data(contentsOf: destination) == expected)
        #expect(requested.flatMap { $0 } == Array(0..<height))
        let blockHeight = switch compression {
        case .piz: 32
        case .zip: 16
        default: 1
        }
        #expect(requested == stride(from: 0, to: height, by: blockHeight).map { $0..<min($0 + blockHeight, height) })
        if compression == .piz {
            let raw = try EXRExporter(compression: .uncompressed).data(width: width, height: height, channels: channels(0..<height))
            #expect(expected.count < raw.count / 2)
        }
    }

    @Test func `RLE splits long repeats and keeps an unprofitable scanline raw`() throws {
        let long = [UInt32](repeating: 0, count: 193)
        let noisy = (0..<193).map { UInt32(truncatingIfNeeded: UInt64($0) &* 0x9E37_79B9) ^ 0x1F03_CA67 }
        let exporter = EXRExporter(compression: .rle)
        let data = try exporter.data(width: 193, height: 2, channels: [.init(name: "id", samples: .unsigned(long + noisy))])
        let decoded = try decodeFixture(data, width: 193, height: 2)
        #expect(decoded.compressedBlocks == 1)
        #expect(decoded.samples["id"] == long + noisy)
    }

    @Test func `Streaming replaces only a complete file and supports a wider still`() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let destination = directory.appendingPathComponent("still.exr")
        let original = Data([1, 2, 3])
        try original.write(to: destination)

        let exporter = EXRExporter(compression: .uncompressed)
        #expect(throws: EXRExporter.Failure.streamChannelLayout(firstRow: 1)) {
            try exporter.write(to: destination, width: 2, height: 2) { rows in
                rows.lowerBound == 0
                    ? [.init(name: "Z", samples: .half([1, 2]))]
                    : [.init(name: "Z", samples: .float([3, 4]))]
            }
        }
        #expect(try Data(contentsOf: destination) == original)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["still.exr"])

        let width = 17_000
        try exporter.write(to: destination, width: width, height: 2) { rows in
            [.init(name: "objectID.id", samples: .unsigned(Array(repeating: UInt32(rows.lowerBound), count: width)))]
        }
        let image = try decodeFixture(Data(contentsOf: destination), width: width, height: 2)
        #expect(image.samples["objectID.id"] == Array(repeating: 0, count: width) + Array(repeating: 1, count: width))
    }

    /// Reads this fixture's standard scanline headers, offsets and mixed samples.
    private func decodeFixture(_ data: Data, width: Int, height: Int) throws -> (
        names: [String], types: [UInt32], samples: [String: [UInt32]], compressedBlocks: Int
    ) {
        let file = Array(data)
        #expect(integer(file, at: 0, as: UInt32.self) == 20_000_630)
        #expect(integer(file, at: 4, as: UInt32.self) == 2)
        var offset = 8
        var names: [String] = []
        var types: [UInt32] = []
        var compression: UInt8 = 0
        while file[offset] != 0 {
            let name = try string(file, offset: &offset)
            _ = try string(file, offset: &offset)
            let size = Int(integer(file, at: offset, as: UInt32.self))
            offset += 4
            let end = offset + size
            if name == "channels" {
                while file[offset] != 0 {
                    names.append(try string(file, offset: &offset))
                    types.append(integer(file, at: offset, as: UInt32.self))
                    offset += 16
                }
                #expect(offset + 1 == end)
            } else if name == "compression" {
                compression = file[offset]
            }
            offset = end
        }
        let table = offset + 1
        let blockHeight = compression == 3 ? 16 : 1
        let bytesPerPixel = types.reduce(0) { $0 + ($1 == 1 ? 2 : 4) }
        var samples: [String: [UInt32]] = [:]
        var compressedBlocks = 0
        for firstRow in stride(from: 0, to: height, by: blockHeight) {
            let start = Int(integer(file, at: table + firstRow / blockHeight * 8, as: UInt64.self))
            #expect(integer(file, at: start, as: UInt32.self) == UInt32(firstRow))
            let size = Int(integer(file, at: start + 4, as: UInt32.self))
            let stored = Array(file[(start + 8)..<(start + 8 + size)])
            let rows = min(blockHeight, height - firstRow)
            let expectedSize = rows * width * bytesPerPixel
            let raw: [UInt8]
            if size < expectedSize {
                compressedBlocks += 1
                raw = try compression == 1 ? inflateRLE(stored, size: expectedSize) : inflateZIP(stored, size: expectedSize)
            } else {
                raw = stored
            }
            var sampleOffset = 0
            for _ in 0..<rows {
                for (name, type) in zip(names, types) {
                    for _ in 0..<width {
                        let bits: UInt32
                        if type == 1 {
                            bits = UInt32(integer(raw, at: sampleOffset, as: UInt16.self))
                            sampleOffset += 2
                        } else {
                            bits = integer(raw, at: sampleOffset, as: UInt32.self)
                            sampleOffset += 4
                        }
                        samples[name, default: []].append(bits)
                    }
                }
            }
            #expect(sampleOffset == raw.count)
        }
        return (names, types, samples, compressedBlocks)
    }

    private func integer<T: FixedWidthInteger>(_ bytes: [UInt8], at offset: Int, as: T.Type) -> T {
        bytes.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: T.self)) }
    }

    private func string(_ bytes: [UInt8], offset: inout Int) throws -> String {
        let end = try #require(bytes[offset...].firstIndex(of: 0))
        let value = String(decoding: bytes[offset..<end], as: UTF8.self)
        offset = end + 1
        return value
    }

    private func inflateZIP(_ bytes: [UInt8], size: Int) throws -> [UInt8] {
        var filtered = [UInt8](repeating: 0, count: size)
        let written = filtered.withUnsafeMutableBufferPointer { destination in
            bytes.withUnsafeBufferPointer { source in
                compression_decode_buffer(
                    destination.baseAddress!, destination.count, source.baseAddress! + 2,
                    source.count - 6, nil, COMPRESSION_ZLIB
                )
            }
        }
        try #require(written == size)
        return restorePredictor(filtered)
    }

    private func inflateRLE(_ bytes: [UInt8], size: Int) throws -> [UInt8] {
        var filtered: [UInt8] = []
        var index = 0
        while index < bytes.count {
            let header = Int(Int8(bitPattern: bytes[index]))
            index += 1
            if header < 0 {
                try #require(index - header <= bytes.count)
                filtered.append(contentsOf: bytes[index..<(index - header)])
                index -= header
            } else {
                try #require(index < bytes.count)
                filtered.append(contentsOf: repeatElement(bytes[index], count: header + 1))
                index += 1
            }
        }
        try #require(filtered.count == size)
        return restorePredictor(filtered)
    }

    private func restorePredictor(_ bytes: [UInt8]) -> [UInt8] {
        var filtered = bytes
        let size = filtered.count
        for index in 1..<size { filtered[index] = filtered[index - 1] &+ filtered[index] &- 128 }
        let middle = (size + 1) / 2
        return (0..<size).map { index in filtered[(index % 2 == 0 ? 0 : middle) + index / 2] }
    }
}
