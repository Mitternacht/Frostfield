import Foundation
import simd

/// Reads Point Cloud Data (PCD) files into renderable point geometry.
public enum PCDLoader {
    /// Byte order for binary point fields. The LZF size prefix uses little-endian words.
    public enum ByteOrder: Sendable {
        case littleEndian
        case bigEndian
    }

    public struct Asset: Sendable {
        public let pointCloud: PointCloud
        /// Dimensions of the original row-major point grid. An unorganized cloud has height one.
        public let width: Int
        public let height: Int
        /// Source attributes in the same order as `pointCloud.positions`.
        public let normals: [SIMD3<Float>]?
        public let intensities: [Float]?
        public let labels: [Int64]?
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case invalidHeader(String)
        case unsupportedEncoding(String)
        case invalidBody
        case invalidPoint(Int)
        case invalidCompression

        public var errorDescription: String? {
            switch self {
            case .fileURL: "PCD import requires a regular local file."
            case .size: "The PCD file or decoded data exceeds the supported size."
            case .invalidHeader(let field): "The PCD header has an invalid or missing \(field) entry."
            case .unsupportedEncoding(let encoding): "PCD data encoding '\(encoding)' is unsupported."
            case .invalidBody: "The PCD payload does not match its declared point layout."
            case .invalidPoint(let index): "PCD point \(index) has invalid attributes."
            case .invalidCompression: "The PCD LZF stream is invalid or does not match the declared byte count."
            }
        }
    }

    fileprivate static let maximumBytes = 256 * 1_024 * 1_024
    fileprivate static let maximumPoints = 1 << 22

    public static func load(contentsOf url: URL, byteOrder: ByteOrder = .littleEndian) throws -> Asset {
        guard url.isFileURL else { throw Failure.fileURL }
        let info = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true else { throw Failure.fileURL }
        if let size = info.fileSize, size > maximumBytes { throw Failure.size }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var data = Data()
        while data.count <= maximumBytes {
            let count = min(1 << 20, maximumBytes + 1 - data.count)
            guard let chunk = try file.read(upToCount: count), !chunk.isEmpty else { break }
            data.append(chunk)
        }
        return try load(data, byteOrder: byteOrder)
    }

    /// Supports ASCII, binary, and LZF binary_compressed with up to 2²² points.
    /// Encoded and expanded data are each limited to 256 MiB. Packed RGB and RGBA
    /// use sRGB color channels. Point positions and returned float attributes must be finite.
    public static func load(_ data: Data, byteOrder: ByteOrder = .littleEndian) throws -> Asset {
        guard data.count <= maximumBytes else { throw Failure.size }
        return try data.withUnsafeBytes { bytes in
            var text = PCDTextReader(bytes: bytes)
            let header = try PCDHeader(reader: &text)
            let layout = try PointLayout(fields: header.fields)
            var points = PointBuffer(layout: layout)
            let payload = UnsafeRawBufferPointer(rebasing: bytes[text.offset...])
            switch header.encoding {
            case .ascii:
                guard payload.count >= 2 * header.points * header.components - 1 else { throw Failure.invalidBody }
                points.reserveCapacity(header.points)
                var rowReader = PCDTextReader(bytes: payload)
                for row in 0..<header.points {
                    guard let tokens = try rowReader.nonemptyLine(limit: 1 << 20), tokens.count == header.components else {
                        throw Failure.invalidPoint(row)
                    }
                    try points.append(row: row) { field in
                        guard let bits = field.scalar.parse(tokens[field.componentOffset]) else {
                            throw Failure.invalidPoint(row)
                        }
                        return bits
                    }
                }
                guard try rowReader.nonemptyLine(limit: 1 << 20) == nil else { throw Failure.invalidBody }
            case .binary:
                // Some binary producers append complete records after the declared grid.
                guard payload.count >= header.byteCount,
                      (payload.count - header.byteCount).isMultiple(of: header.stride) else { throw Failure.invalidBody }
                points.reserveCapacity(header.points)
                try points.appendBinary(header, payload: payload, byteOrder: byteOrder)
            case .binaryCompressed:
                guard payload.count >= 8 else { throw Failure.invalidCompression }
                let compressedSize = Int(UInt32(littleEndian: payload.loadUnaligned(fromByteOffset: 0, as: UInt32.self)))
                let expandedSize = Int(UInt32(littleEndian: payload.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
                guard compressedSize <= payload.count - 8, expandedSize == header.byteCount else {
                    throw Failure.invalidCompression
                }
                let end = 8 + compressedSize
                // Legacy writers can leave unused zero bytes after the stored LZF stream.
                guard payload[end...].allSatisfy({ $0 == 0 }) else { throw Failure.invalidCompression }
                let encoded = UnsafeRawBufferPointer(rebasing: payload[8..<end])
                let expanded = try PCDLZF.expand(encoded, to: expandedSize)
                points.reserveCapacity(header.points)
                try expanded.withUnsafeBytes { bytes in
                    try points.appendBinary(header, payload: bytes, byteOrder: byteOrder)
                }
            }
            return Asset(
                pointCloud: try PointCloud(positions: points.positions, colors: points.colors),
                width: header.width, height: header.height, normals: points.normals,
                intensities: points.intensities, labels: points.labels
            )
        }
    }

    private struct PointLayout {
        let position: (PCDField, PCDField, PCDField)
        let normal: (PCDField, PCDField, PCDField)?
        let color: PCDField?
        let intensity: PCDField?
        let label: PCDField?

        init(fields: [PCDField]) throws {
            let indexed = Dictionary(uniqueKeysWithValues: fields.filter { $0.name != "_" }.map { ($0.name, $0) })
            func scalar(_ name: String) throws -> PCDField? {
                guard let field = indexed[name] else { return nil }
                guard field.count == 1 else { throw Failure.invalidHeader(name) }
                return field
            }
            guard let x = try scalar("x"), let y = try scalar("y"), let z = try scalar("z") else {
                throw Failure.invalidHeader("x, y, z")
            }
            position = (x, y, z)
            let nx = try scalar("normal_x"), ny = try scalar("normal_y"), nz = try scalar("normal_z")
            if let nx, let ny, let nz {
                normal = (nx, ny, nz)
            } else {
                guard nx == nil, ny == nil, nz == nil else { throw Failure.invalidHeader("normals") }
                normal = nil
            }
            color = try scalar("rgba") ?? scalar("rgb")
            if let color, color.scalar.size != 4 { throw Failure.invalidHeader(color.name) }
            intensity = try scalar("intensity")
            label = try scalar("label")
            if label?.scalar.kind == .float { throw Failure.invalidHeader("label") }
        }
    }

    private struct PointBuffer {
        let layout: PointLayout
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>]?
        var normals: [SIMD3<Float>]?
        var intensities: [Float]?
        var labels: [Int64]?

        init(layout: PointLayout) {
            self.layout = layout
            if layout.color != nil { colors = [] }
            if layout.normal != nil { normals = [] }
            if layout.intensity != nil { intensities = [] }
            if layout.label != nil { labels = [] }
        }

        mutating func reserveCapacity(_ capacity: Int) {
            positions.reserveCapacity(capacity)
            colors?.reserveCapacity(capacity)
            normals?.reserveCapacity(capacity)
            intensities?.reserveCapacity(capacity)
            labels?.reserveCapacity(capacity)
        }

        mutating func append(row: Int, bits: (PCDField) throws -> UInt64) throws {
            func number(_ field: PCDField) throws -> Float {
                let value = Float(field.scalar.number(try bits(field)))
                guard value.isFinite else { throw Failure.invalidPoint(row) }
                return value
            }
            func vector(_ fields: (PCDField, PCDField, PCDField)) throws -> SIMD3<Float> {
                try SIMD3(number(fields.0), number(fields.1), number(fields.2))
            }
            positions.append(try vector(layout.position))
            if let field = layout.color {
                let packed = UInt32(truncatingIfNeeded: try bits(field))
                func channel(_ shift: Int) -> Float { Float((packed >> shift) & 255) / 255 }
                colors?.append(SIMD4(
                    GLTFColorTables.sRGBToLinear(channel(16)), GLTFColorTables.sRGBToLinear(channel(8)),
                    GLTFColorTables.sRGBToLinear(channel(0)), field.name == "rgba" ? channel(24) : 1
                ))
            }
            if let normal = layout.normal { normals?.append(try vector(normal)) }
            if let intensity = layout.intensity { intensities?.append(try number(intensity)) }
            if let label = layout.label { labels?.append(Int64(label.scalar.number(try bits(label)))) }
        }

        mutating func appendBinary(
            _ header: PCDHeader, payload: UnsafeRawBufferPointer, byteOrder: ByteOrder
        ) throws {
            let grouped = header.encoding == .binaryCompressed
            for row in 0..<header.points {
                try append(row: row) { field in
                    let offset = grouped
                        ? field.packedOffset * header.points + field.byteWidth * row
                        : header.stride * row + field.byteOffset
                    return field.scalar.bits(in: payload, at: offset, byteOrder: byteOrder)
                }
            }
        }
    }
}

private struct PCDTextReader {
    let bytes: UnsafeRawBufferPointer
    private(set) var offset = 0

    mutating func nonemptyLine(limit: Int, end: Int? = nil) throws -> [Substring]? {
        let end = end ?? bytes.count
        while offset < end {
            let start = offset
            while offset < end, bytes[offset] != 10, bytes[offset] != 13 { offset += 1 }
            guard offset - start <= limit, offset < end || end == bytes.count else { throw PCDLoader.Failure.size }
            guard let line = String(bytes: bytes[start..<offset], encoding: .ascii) else {
                throw PCDLoader.Failure.invalidBody
            }
            if offset < end {
                let carriageReturn = bytes[offset] == 13
                offset += 1
                if carriageReturn, offset < end, bytes[offset] == 10 { offset += 1 }
            }
            let fields = line.prefix { $0 != "#" }.split(whereSeparator: \.isWhitespace)
            if !fields.isEmpty { return fields }
        }
        return nil
    }
}

private struct PCDHeader {
    enum Encoding: String {
        case ascii
        case binary
        case binaryCompressed = "binary_compressed"
    }

    let encoding: Encoding
    let width: Int
    let height: Int
    let points: Int
    let fields: [PCDField]
    let stride: Int
    let components: Int
    let byteCount: Int

    init(reader: inout PCDTextReader) throws {
        typealias Failure = PCDLoader.Failure
        let limit = min(1 << 16, reader.bytes.count)
        var records: [String: [Substring]] = [:]
        while let line = try reader.nonemptyLine(limit: limit, end: limit) {
            let key = line[0].uppercased()
            guard records.updateValue(Array(line.dropFirst()), forKey: key) == nil else {
                throw Failure.invalidHeader(key)
            }
            if key == "DATA" { break }
        }
        guard let data = records["DATA"], data.count == 1 else { throw Failure.invalidHeader("DATA") }
        guard let encoding = Encoding(rawValue: data[0].lowercased()) else {
            throw Failure.unsupportedEncoding(String(data[0]))
        }
        self.encoding = encoding
        if let version = records["VERSION"] {
            guard version.count == 1, let value = Double(version[0]), value == 0.6 || value == 0.7 else {
                throw Failure.invalidHeader("VERSION")
            }
        }
        func dimension(_ key: String) throws -> Int? {
            guard let tokens = records[key] else { return nil }
            guard tokens.count == 1, let value = Int(tokens[0]), value > 0 else { throw Failure.invalidHeader(key) }
            guard value <= PCDLoader.maximumPoints else { throw Failure.size }
            return value
        }
        let declaredPoints = try dimension("POINTS")
        let declaredWidth = try dimension("WIDTH")
        let height = try dimension("HEIGHT") ?? 1
        guard let width = declaredWidth ?? declaredPoints else { throw Failure.invalidHeader("WIDTH or POINTS") }
        let points = width * height
        guard points <= PCDLoader.maximumPoints else { throw Failure.size }
        guard declaredPoints == nil || declaredPoints == points else { throw Failure.invalidHeader("POINTS") }
        self.width = width
        self.height = height
        self.points = points

        guard let names = records["FIELDS"], !names.isEmpty, names.count <= 128 else {
            throw Failure.invalidHeader("FIELDS")
        }
        guard let sizes = records["SIZE"], sizes.count == names.count else { throw Failure.invalidHeader("SIZE") }
        guard let kinds = records["TYPE"], kinds.count == names.count else { throw Failure.invalidHeader("TYPE") }
        let counts = records["COUNT"] ?? [Substring](repeating: "1", count: names.count)
        guard counts.count == names.count else { throw Failure.invalidHeader("COUNT") }
        var seen: Set<String> = []
        var fields: [PCDField] = []
        var stride = 0, packedStride = 0, components = 0
        for index in names.indices {
            let name = names[index].lowercased()
            guard name == "_" || seen.insert(name).inserted else { throw Failure.invalidHeader("FIELDS") }
            guard let count = Int(counts[index]), (1...1_024).contains(count) else { throw Failure.invalidHeader("COUNT") }
            guard let size = Int(sizes[index]), let kind = PCDField.Scalar.Kind(rawValue: kinds[index].uppercased()),
                  let scalar = PCDField.Scalar(kind: kind, size: size) else { throw Failure.invalidHeader("TYPE/SIZE") }
            fields.append(PCDField(
                name: name, scalar: scalar, count: count, componentOffset: components,
                byteOffset: stride, packedOffset: packedStride
            ))
            components += count
            stride += size * count
            if name != "_" { packedStride += size * count }
        }
        let byteCount = (encoding == .binaryCompressed ? packedStride : stride) * points
        guard components <= 4_096, byteCount <= PCDLoader.maximumBytes else { throw Failure.size }
        self.fields = fields
        self.stride = stride
        self.components = components
        self.byteCount = byteCount
    }
}

private struct PCDField {
    let name: String
    let scalar: Scalar
    let count: Int
    let componentOffset: Int
    let byteOffset: Int
    let packedOffset: Int
    var byteWidth: Int { count * scalar.size }

    struct Scalar {
        enum Kind: String { case float = "F", signed = "I", unsigned = "U" }
        let kind: Kind
        let size: Int

        init?(kind: Kind, size: Int) {
            guard kind == .float ? (size == 4 || size == 8) : (size == 1 || size == 2 || size == 4) else { return nil }
            self.kind = kind
            self.size = size
        }

        func bits(in bytes: UnsafeRawBufferPointer, at offset: Int, byteOrder: PCDLoader.ByteOrder) -> UInt64 {
            func word<T: FixedWidthInteger>(_ type: T.Type) -> UInt64 {
                let value = bytes.loadUnaligned(fromByteOffset: offset, as: type)
                return UInt64(byteOrder == .littleEndian ? T(littleEndian: value) : T(bigEndian: value))
            }
            switch size {
            case 1: return UInt64(bytes[offset])
            case 2: return word(UInt16.self)
            case 4: return word(UInt32.self)
            default: return word(UInt64.self)
            }
        }

        func parse(_ token: Substring) -> UInt64? {
            switch kind {
            case .float:
                return size == 4 ? Float(token).map { UInt64($0.bitPattern) } : Double(token)?.bitPattern
            case .signed:
                let limit = Int64(1) << (size * 8 - 1)
                guard let value = Int64(token), (-limit..<limit).contains(value) else { return nil }
                return UInt64(bitPattern: value)
            case .unsigned:
                guard let value = UInt64(token), value < UInt64(1) << (size * 8) else { return nil }
                return value
            }
        }

        func number(_ bits: UInt64) -> Double {
            switch kind {
            case .float:
                return size == 4 ? Double(Float(bitPattern: UInt32(truncatingIfNeeded: bits))) : Double(bitPattern: bits)
            case .signed:
                let shift = (8 - size) * 8
                return Double(Int64(bitPattern: bits << shift) >> shift)
            case .unsigned:
                return Double(bits)
            }
        }
    }
}

private enum PCDLZF {
    static func expand(_ source: UnsafeRawBufferPointer, to count: Int) throws -> Data {
        typealias Failure = PCDLoader.Failure
        // The longest back reference emits 264 bytes from three encoded bytes.
        guard count <= source.count * 88 else { throw Failure.invalidCompression }
        var output = Data(count: count)
        try output.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
            var input = 0, written = 0
            while input < source.count {
                let control = Int(source[input])
                input += 1
                if control < 32 {
                    let length = control + 1
                    guard length <= source.count - input, length <= count - written else { throw Failure.invalidCompression }
                    let from = source.baseAddress!.advanced(by: input)
                    target.baseAddress!.advanced(by: written).copyMemory(from: from, byteCount: length)
                    input += length
                    written += length
                } else {
                    var length = (control >> 5) + 2
                    if length == 9 {
                        guard input < source.count else { throw Failure.invalidCompression }
                        length += Int(source[input])
                        input += 1
                    }
                    guard input < source.count else { throw Failure.invalidCompression }
                    let distance = ((control & 31) << 8) + Int(source[input]) + 1
                    input += 1
                    guard distance <= written, length <= count - written else { throw Failure.invalidCompression }
                    for index in written..<(written + length) {
                        target[index] = target[index - distance]
                    }
                    written += length
                }
            }
            guard written == count else { throw Failure.invalidCompression }
        }
        return output
    }
}
