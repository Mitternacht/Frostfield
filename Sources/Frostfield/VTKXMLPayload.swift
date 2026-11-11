import Foundation
import zlib

/// Decodes one bounded XML DataArray. VTK stores zlib block sizes separately
/// from the concatenated compressed bytes, with optional independent base64 padding.
struct VTKXMLPayload {
    private typealias Failure = VTKLoader.Failure
    private let littleEndian: Bool
    private let headerWidth: Int
    private let compressed: Bool
    private let appended: Data?
    private let ranges: [Int: Range<Int>]

    init(root: [String: String], appended: Data?, offsets: Set<Int>) throws {
        switch root["byte_order"] ?? "LittleEndian" {
        case "LittleEndian": littleEndian = true
        case "BigEndian": littleEndian = false
        default: throw Failure.unsupported("XML byte order")
        }
        switch root["header_type"] ?? "UInt32" {
        case "UInt32": headerWidth = 4
        case "UInt64": headerWidth = 8
        default: throw Failure.unsupported("XML header type")
        }
        switch root["compressor"] {
        case nil: compressed = false
        case "vtkZLibDataCompressor": compressed = true
        default: throw Failure.unsupported("XML compressor")
        }

        self.appended = appended
        let starts = offsets.sorted()
        if let appended {
            guard starts.last.map({ $0 < appended.count }) ?? true else { throw Failure.malformed("appended offset") }
            let ends = starts.dropFirst() + [appended.count]
            ranges = Dictionary(uniqueKeysWithValues: zip(starts, ends).map {
                ($0, (appended.startIndex + $0)..<(appended.startIndex + $1))
            })
        } else {
            guard starts.isEmpty else { throw Failure.malformed("AppendedData") }
            ranges = [:]
        }
    }

    func word<T: FixedWidthInteger>(_ type: T.Type, bytes: UnsafeRawBufferPointer, at offset: Int) -> T {
        let value = bytes.loadUnaligned(fromByteOffset: offset, as: T.self)
        return littleEndian ? T(littleEndian: value) : T(bigEndian: value)
    }

    func decode(_ attributes: [String: String], inline: Data, size: Int) throws -> Data {
        let source: Data
        switch attributes["format"] {
        case "binary":
            source = inline
        case "appended":
            guard let offset = attributes["offset"].flatMap(Int.init), let range = ranges[offset], let appended else {
                throw Failure.malformed("appended offset")
            }
            source = appended[range]
        default:
            throw Failure.unsupported("XML DataArray format")
        }

        let encoded = Data(source.lazy.filter { !Self.isSpace($0) })
        guard encoded.count.isMultiple(of: 4), !encoded.isEmpty else { throw Failure.malformed("XML base64") }
        return compressed ? try inflate(encoded, size: size) : try uncompressed(encoded, size: size)
    }

    private func header(_ bytes: Data, at index: Int) throws -> Int {
        let offset = index * headerWidth
        guard headerWidth <= bytes.count - offset else { throw Failure.truncated }
        return try bytes.withUnsafeBytes { bytes in
            if headerWidth == 4 { return Int(word(UInt32.self, bytes: bytes, at: offset)) }
            let value = word(UInt64.self, bytes: bytes, at: offset)
            guard let integer = Int(exactly: value) else { throw Failure.size }
            return integer
        }
    }

    private func uncompressed(_ encoded: Data, size: Int) throws -> Data {
        let (header, bytes) = try split(encoded, headerBytes: headerWidth)
        guard try self.header(header, at: 0) == size, bytes.count == size else {
            throw Failure.malformed("XML binary length")
        }
        return bytes
    }

    private func inflate(_ encoded: Data, size: Int) throws -> Data {
        let prefixLength = Self.base64Length(headerWidth)
        guard let prefix = Data(base64Encoded: encoded.prefix(prefixLength)) else { throw Failure.malformed("XML base64") }
        let blocks = try header(prefix, at: 0)
        guard blocks <= 65_536, blocks <= size else { throw Failure.malformed("XML zlib block count") }
        let (headerData, bytes) = try split(encoded, headerBytes: (blocks + 3) * headerWidth)
        let blockSize = try header(headerData, at: 1)
        let partialSize = try header(headerData, at: 2)
        if blocks == 0 {
            guard size == 0, partialSize == 0, bytes.isEmpty else { throw Failure.malformed("XML zlib length") }
            return Data()
        }
        let lastSize = partialSize == 0 ? blockSize : partialSize
        guard blockSize > 0, lastSize > 0, lastSize <= blockSize, lastSize <= size,
              blocks - 1 <= (size - lastSize) / blockSize,
              (blocks - 1) * blockSize + lastSize == size else { throw Failure.malformed("XML zlib length") }

        var output = Data(count: size)
        try bytes.withUnsafeBytes { input in
            try output.withUnsafeMutableBytes { result in
                let source = input.bindMemory(to: UInt8.self)
                let destination = result.bindMemory(to: UInt8.self)
                var position = 0
                for index in 0..<blocks {
                    let expected = index == blocks - 1 ? lastSize : blockSize
                    let stored = try header(headerData, at: index + 3)
                    guard stored > 0, stored <= source.count - position,
                          let start = source.baseAddress, let end = destination.baseAddress else { throw Failure.truncated }
                    var compressedBytes = UInt(stored), decodedBytes = UInt(expected)
                    let status = uncompress2(
                        end + index * blockSize, &decodedBytes, start + position, &compressedBytes
                    )
                    guard status == Z_OK, decodedBytes == expected, compressedBytes == stored else {
                        throw Failure.malformed("XML zlib block")
                    }
                    position += stored
                }
                guard position == source.count else { throw Failure.malformed("XML zlib length") }
            }
        }
        return output
    }

    private func split(_ encoded: Data, headerBytes: Int) throws -> (Data, Data) {
        let headerLength = Self.base64Length(headerBytes)
        guard encoded.count >= headerLength,
              let start = Data(base64Encoded: encoded.prefix(headerLength)) else { throw Failure.malformed("XML base64") }
        if start.count == headerBytes {
            guard let payload = Data(base64Encoded: encoded.dropFirst(headerLength)) else { throw Failure.malformed("XML base64") }
            return (start, payload)
        }
        guard let combined = Data(base64Encoded: encoded), combined.count >= headerBytes else {
            throw Failure.malformed("XML base64")
        }
        return (combined.prefix(headerBytes), combined.dropFirst(headerBytes))
    }

    private static func base64Length(_ count: Int) -> Int { (count + 2) / 3 * 4 }

    private static func isSpace(_ value: UInt8) -> Bool { value == 32 || (9...13).contains(value) }
}
