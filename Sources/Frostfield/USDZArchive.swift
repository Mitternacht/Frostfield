import Foundation

/// Reads and writes the zip container of a USDZ package.
///
/// A USDZ package is an uncompressed zip archive. Every file's data starts at a multiple
/// of 64 bytes from the start of the archive, which lets readers map layers and images
/// in place. The first file is the root layer. Padding sits in the local header's extra
/// field under ID 0x1986, as Pixar's `usdzip` writes it. Writing uses no zip64 records,
/// encryption, or data descriptors. [USDZ specification](https://openusd.org/release/spec_usdz.html)
package enum USDZArchive {
    package static let alignment = 64
    package static let paddingFieldID: UInt16 = 0x1986

    package struct Entry: Equatable, Sendable {
        /// A relative path with `/` separators.
        package let path: String
        package let data: Data

        package init(path: String, data: Data) {
            self.path = path
            self.data = data
        }
    }

    /// One stored file as the central directory and its local header describe it.
    package struct Record: Equatable, Sendable {
        package let path: String
        /// The offset of the file's first data byte from the start of the archive.
        package let dataOffset: Int
        package let size: Int
        package let crc32: UInt32
        package let method: UInt16
        package let localHeaderOffset: Int
    }

    package enum Failure: LocalizedError, Equatable {
        case invalidPath(String)
        case duplicatePath(String)
        case sizeLimit
        case malformed(String)
        case compressedEntry(String)
        case checksum(String)

        package var errorDescription: String? {
            switch self {
            case .invalidPath(let path): "The USDZ entry path '\(path)' must be relative, use '/' separators, and contain no '.' or '..' components."
            case .duplicatePath(let path): "The USDZ package contains '\(path)' more than once."
            case .sizeLimit: "A USDZ package without zip64 records must stay below 4 GiB and 65,535 files."
            case .malformed(let reason): "The USDZ package is not a valid zip archive. \(reason)"
            case .compressedEntry(let path): "USDZ entry '\(path)' is compressed; USDZ requires stored files."
            case .checksum(let path): "USDZ entry '\(path)' fails its CRC-32 check."
            }
        }
    }

    /// Writes stored entries in order, aligning each file's data to 64 bytes.
    package static func write(_ entries: [Entry]) throws -> Data {
        guard entries.count < 0xFFFF else { throw Failure.sizeLimit }
        var seen = Set<String>()
        var output = Data()
        var central = Data()
        for entry in entries {
            try validate(entry.path)
            guard seen.insert(entry.path).inserted else { throw Failure.duplicatePath(entry.path) }
            let name = Array(entry.path.utf8)
            let headerOffset = output.count
            let unpadded = headerOffset + 30 + name.count
            var padding = (alignment - unpadded % alignment) % alignment
            // The extra field needs four bytes for its ID and size before any padding bytes.
            if padding > 0, padding < 4 { padding += alignment }
            let crc = crc32(entry.data)
            guard entry.data.count < 0xFFFF_FFFF, unpadded + padding + entry.data.count < 0xFFFF_FFFF else {
                throw Failure.sizeLimit
            }

            output.appendLittleEndian(UInt32(0x0403_4B50))
            output.appendLittleEndian(UInt16(10)) // version needed: stored
            output.appendLittleEndian(UInt16(0)) // flags
            output.appendLittleEndian(UInt16(0)) // method: stored
            output.appendLittleEndian(UInt16(0)) // time 00:00:00
            output.appendLittleEndian(UInt16(0x21)) // date 1980-01-01, for reproducible output
            output.appendLittleEndian(crc)
            output.appendLittleEndian(UInt32(entry.data.count))
            output.appendLittleEndian(UInt32(entry.data.count))
            output.appendLittleEndian(UInt16(name.count))
            output.appendLittleEndian(UInt16(padding))
            output.append(contentsOf: name)
            if padding > 0 {
                output.appendLittleEndian(paddingFieldID)
                output.appendLittleEndian(UInt16(padding - 4))
                output.append(Data(count: padding - 4))
            }
            output.append(entry.data)

            central.appendLittleEndian(UInt32(0x0201_4B50))
            central.appendLittleEndian(UInt16(20)) // version made by
            central.appendLittleEndian(UInt16(10))
            central.appendLittleEndian(UInt16(0))
            central.appendLittleEndian(UInt16(0))
            central.appendLittleEndian(UInt16(0))
            central.appendLittleEndian(UInt16(0x21))
            central.appendLittleEndian(crc)
            central.appendLittleEndian(UInt32(entry.data.count))
            central.appendLittleEndian(UInt32(entry.data.count))
            central.appendLittleEndian(UInt16(name.count))
            central.appendLittleEndian(UInt16(0)) // extra
            central.appendLittleEndian(UInt16(0)) // comment
            central.appendLittleEndian(UInt16(0)) // disk
            central.appendLittleEndian(UInt16(0)) // internal attributes
            central.appendLittleEndian(UInt32(0)) // external attributes
            central.appendLittleEndian(UInt32(headerOffset))
            central.append(contentsOf: name)
        }
        guard output.count + central.count + 22 < 0xFFFF_FFFF else { throw Failure.sizeLimit }
        let centralOffset = output.count
        output.append(central)
        output.appendLittleEndian(UInt32(0x0605_4B50))
        output.appendLittleEndian(UInt16(0))
        output.appendLittleEndian(UInt16(0))
        output.appendLittleEndian(UInt16(entries.count))
        output.appendLittleEndian(UInt16(entries.count))
        output.appendLittleEndian(UInt32(central.count))
        output.appendLittleEndian(UInt32(centralOffset))
        output.appendLittleEndian(UInt16(0))
        return output
    }

    /// Lists the files in central directory order after checking that each local header
    /// agrees with its directory record. It does not require stored files or alignment;
    /// callers check `method` and `dataOffset` for USDZ conformance. The archive is read in
    /// place, so a memory-mapped package is not copied.
    package static func records(in archive: Data) throws -> [Record] {
        try archive.withUnsafeBytes { raw in try records(in: raw.bindMemory(to: UInt8.self)) }
    }

    private static func records(in bytes: UnsafeBufferPointer<UInt8>) throws -> [Record] {
        func u16(_ offset: Int) throws -> UInt16 {
            guard offset >= 0, offset + 2 <= bytes.count else { throw Failure.malformed("A record is truncated.") }
            return UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
        }
        func u32(_ offset: Int) throws -> UInt32 {
            guard offset >= 0, offset + 4 <= bytes.count else { throw Failure.malformed("A record is truncated.") }
            return (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << (8 * UInt32($1)) }
        }

        // The end record is the last 22 bytes when the archive has no comment.
        guard bytes.count >= 22 else { throw Failure.malformed("The end of central directory record is missing.") }
        let minimumEnd = max(0, bytes.count - 22 - 0xFFFF)
        var end = bytes.count - 22
        while try u32(end) != 0x0605_4B50 {
            end -= 1
            guard end >= minimumEnd else { throw Failure.malformed("The end of central directory record is missing.") }
        }
        let count = Int(try u16(end + 10))
        let centralSize = Int(try u32(end + 12))
        var cursor = Int(try u32(end + 16))
        guard cursor + centralSize <= end else { throw Failure.malformed("The central directory lies outside the archive.") }

        var records: [Record] = []
        for _ in 0..<count {
            guard try u32(cursor) == 0x0201_4B50 else { throw Failure.malformed("A central directory record is invalid.") }
            let flags = try u16(cursor + 8)
            let method = try u16(cursor + 10)
            let crc = try u32(cursor + 16)
            let compressed = Int(try u32(cursor + 20))
            let size = Int(try u32(cursor + 24))
            let nameLength = Int(try u16(cursor + 28))
            let extraLength = Int(try u16(cursor + 30))
            let commentLength = Int(try u16(cursor + 32))
            let headerOffset = Int(try u32(cursor + 42))
            guard cursor + 46 + nameLength <= bytes.count else { throw Failure.malformed("A file name is truncated.") }
            let path = String(decoding: bytes[(cursor + 46)..<(cursor + 46 + nameLength)], as: UTF8.self)
            guard flags & 0x1 == 0 else { throw Failure.malformed("'\(path)' is encrypted.") }

            guard try u32(headerOffset) == 0x0403_4B50 else { throw Failure.malformed("The local header of '\(path)' is missing.") }
            let localNameLength = Int(try u16(headerOffset + 26))
            let localExtraLength = Int(try u16(headerOffset + 28))
            guard localNameLength == nameLength, try u16(headerOffset + 8) == method else {
                throw Failure.malformed("The local header of '\(path)' disagrees with the central directory.")
            }
            let dataOffset = headerOffset + 30 + localNameLength + localExtraLength
            guard compressed == size || method != 0, dataOffset + compressed <= bytes.count else {
                throw Failure.malformed("The data of '\(path)' is truncated.")
            }
            records.append(Record(
                path: path, dataOffset: dataOffset, size: compressed, crc32: crc, method: method,
                localHeaderOffset: headerOffset
            ))
            cursor += 46 + nameLength + extraLength + commentLength
        }
        return records
    }

    /// Reads one stored file by path, verifying its CRC-32.
    package static func file(_ path: String, in archive: Data) throws -> Data? {
        guard let record = try records(in: archive).first(where: { $0.path == path }) else { return nil }
        return try file(record, in: archive)
    }

    /// Copies one stored file that `records(in:)` listed for the same archive, verifying its CRC-32.
    package static func file(_ record: Record, in archive: Data) throws -> Data {
        guard record.method == 0 else { throw Failure.compressedEntry(record.path) }
        guard record.dataOffset >= 0, record.size >= 0, record.dataOffset + record.size <= archive.count else {
            throw Failure.malformed("The data of '\(record.path)' is truncated.")
        }
        let start = archive.startIndex + record.dataOffset
        let data = archive.subdata(in: start..<(start + record.size))
        guard crc32(data) == record.crc32 else { throw Failure.checksum(record.path) }
        return data
    }

    private static func validate(_ path: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, path.utf8.count < 0xFFFF, !path.hasPrefix("/"), !path.contains("\\"),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
              path.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value < 0x7F }) else {
            throw Failure.invalidPath(path)
        }
    }

    private static let crcTable: [UInt32] = (0..<256).map { index in
        (0..<8).reduce(UInt32(index)) { value, _ in value & 1 == 1 ? 0xEDB8_8320 ^ (value >> 1) : value >> 1 }
    }

    /// The zip (ISO 3309) CRC-32.
    package static func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        data.withUnsafeBytes { raw in
            for byte in raw.bindMemory(to: UInt8.self) {
                crc = crcTable[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
            }
        }
        return crc ^ 0xFFFF_FFFF
    }
}

extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}
