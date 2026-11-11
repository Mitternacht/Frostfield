import Foundation

/// Reads the PLY 1.0 schema and consumes its records in declared order.
struct PLYFileReader {
    enum Scalar: Equatable {
        case int8, uint8, int16, uint16, int32, uint32, float32, float64

        init?(_ name: Substring) {
            switch name {
            case "char", "int8": self = .int8
            case "uchar", "uint8": self = .uint8
            case "short", "int16": self = .int16
            case "ushort", "uint16": self = .uint16
            case "int", "int32": self = .int32
            case "uint", "uint32": self = .uint32
            case "float", "float32": self = .float32
            case "double", "float64": self = .float64
            default: return nil
            }
        }

        var width: Int {
            switch self {
            case .int8, .uint8: 1
            case .int16, .uint16: 2
            case .int32, .uint32, .float32: 4
            case .float64: 8
            }
        }

        var integerBounds: ClosedRange<Int64>? {
            switch self {
            case .int8: Int64(Int8.min)...Int64(Int8.max)
            case .uint8: 0...Int64(UInt8.max)
            case .int16: Int64(Int16.min)...Int64(Int16.max)
            case .uint16: 0...Int64(UInt16.max)
            case .int32: Int64(Int32.min)...Int64(Int32.max)
            case .uint32: 0...Int64(UInt32.max)
            case .float32, .float64: nil
            }
        }
    }

    struct Property {
        let name: String
        let scalar: Scalar
        let countType: Scalar?
    }

    struct Element {
        let name: String
        let count: Int
        var properties: [Property] = []

        func property(_ aliases: [String]) -> Int? {
            for name in aliases {
                if let index = properties.firstIndex(where: { $0.name == name }) { return index }
            }
            return nil
        }

        func scalar(_ aliases: [String]) -> Int? {
            guard let index = property(aliases), properties[index].countType == nil else { return nil }
            return index
        }
    }

    static let maximumFileBytes = 256 * 1_024 * 1_024
    static let maximumRecords = 1 << 22
    static let maximumIndices = 1 << 24
    private static let maximumHeaderBytes = 1 << 20
    private static let maximumProperties = 256
    private static let maximumListCount = 1 << 16

    let bytes: UnsafeRawBufferPointer
    private(set) var offset = 0
    private var format = PLYExporter.Format.ascii
    private var carriageReturnOnly = false

    init(_ bytes: UnsafeRawBufferPointer) {
        self.bytes = bytes
    }

    mutating func header(renaming: [String: String]) throws -> [Element] {
        var line = 1
        guard try headerLine(line) == "ply" else { throw PLYLoader.Failure.invalidHeader(line) }
        line += 1
        let formatFields = try headerLine(line).split(whereSeparator: \.isWhitespace)
        guard formatFields.count == 3, formatFields[0] == "format",
              let declaredFormat = PLYExporter.Format(rawValue: String(formatFields[1])),
              formatFields[2] == "1.0" else { throw PLYLoader.Failure.unsupportedFormat }
        format = declaredFormat

        var elements: [Element] = []
        var totalRecords = 0
        while true {
            line += 1
            let fields = try headerLine(line).split(whereSeparator: \.isWhitespace)
            guard let first = fields.first else { continue }
            switch first {
            case "comment", "obj_info": continue
            case "element":
                guard fields.count == 3, let count = Int(fields[2]), count >= 0,
                      Self.validName(String(fields[1])),
                      !elements.contains(where: { $0.name == fields[1] }) else {
                    throw PLYLoader.Failure.invalidHeader(line)
                }
                guard elements.count < Self.maximumProperties, count <= Self.maximumRecords - totalRecords else {
                    throw PLYLoader.Failure.size
                }
                totalRecords += count
                elements.append(Element(name: String(fields[1]), count: count))
            case "property":
                guard !elements.isEmpty else { throw PLYLoader.Failure.invalidHeader(line) }
                let index = elements.count - 1
                guard elements[index].properties.count < Self.maximumProperties else { throw PLYLoader.Failure.size }
                let property = try Self.property(fields, renaming: renaming, line: line)
                guard !elements[index].properties.contains(where: { $0.name == property.name }) else {
                    throw PLYLoader.Failure.invalidHeader(line)
                }
                elements[index].properties.append(property)
            case "end_header":
                guard fields.count == 1,
                      elements.contains(where: { $0.name == "vertex" && $0.count > 0 }),
                      elements.allSatisfy({ $0.count == 0 || !$0.properties.isEmpty }) else {
                    throw PLYLoader.Failure.invalidHeader(line)
                }
                try validateMinimumBody(elements)
                return elements
            default:
                throw PLYLoader.Failure.invalidHeader(line)
            }
        }
    }

    private static func property(_ fields: [Substring], renaming: [String: String], line: Int) throws -> Property {
        let scalar: Scalar
        let countType: Scalar?
        if fields.count == 3, let type = Scalar(fields[1]) {
            scalar = type
            countType = nil
        } else if fields.count == 5, fields[1] == "list", let count = Scalar(fields[2]),
                  count.integerBounds != nil, let type = Scalar(fields[3]) {
            scalar = type
            countType = count
        } else {
            throw PLYLoader.Failure.invalidHeader(line)
        }
        let source = String(fields[fields.count - 1])
        let name = renaming[source] ?? source
        guard validName(source), validName(name) else { throw PLYLoader.Failure.invalidHeader(line) }
        return Property(name: name, scalar: scalar, countType: countType)
    }

    private static func validName(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.allSatisfy { (33...126).contains($0) }
    }

    private mutating func headerLine(_ line: Int) throws -> String {
        let start = offset
        while offset < bytes.count, offset < Self.maximumHeaderBytes,
              bytes[offset] != 10, bytes[offset] != 13 {
            offset += 1
        }
        guard offset < Self.maximumHeaderBytes else { throw PLYLoader.Failure.size }
        guard offset < bytes.count else { throw PLYLoader.Failure.invalidHeader(line) }
        let content = String(decoding: bytes[start..<offset], as: UTF8.self)
        let carriageReturn = bytes[offset] == 13
        offset += 1
        if line == 1, carriageReturn {
            carriageReturnOnly = offset == bytes.count || bytes[offset] != 10
        }
        if carriageReturn, !carriageReturnOnly, offset < bytes.count, bytes[offset] == 10 { offset += 1 }
        return content
    }

    private func validateMinimumBody(_ elements: [Element]) throws {
        var minimum = 0
        for element in elements {
            let units = format == .ascii ? element.properties.count : element.properties.reduce(0) { count, property in
                count + (property.countType ?? property.scalar).width
            }
            minimum += units * element.count
        }
        let remaining = bytes.count - offset
        // A one-byte ASCII token still needs a separator before the next token.
        let available = format == .ascii ? (remaining + 1) / 2 : remaining
        guard minimum <= available else { throw PLYLoader.Failure.invalidBody(offset) }
    }

    mutating func record(_ element: Element, values: inout [Double], lists: inout [Int: [Double]]) throws {
        for (index, property) in element.properties.enumerated() {
            if let countType = property.countType {
                let listOffset = offset
                let count = Int(try scalar(countType))
                guard count >= 0, count <= Self.maximumListCount else {
                    throw PLYLoader.Failure.invalidBody(listOffset)
                }
                let remaining = bytes.count - offset
                let available = format == .ascii ? (remaining + 1) / 2 : remaining / property.scalar.width
                guard count <= available else { throw PLYLoader.Failure.invalidBody(listOffset) }
                if lists[index] != nil {
                    // This array is at most a quad or four UV pairs in supported files.
                    guard count <= 8 else { throw PLYLoader.Failure.invalidBody(listOffset) }
                    lists[index] = try (0..<count).map { _ in try scalar(property.scalar) }
                } else {
                    for _ in 0..<count { _ = try scalar(property.scalar) }
                }
            } else {
                values[index] = try scalar(property.scalar)
            }
        }
    }

    mutating func finish() throws {
        if format == .ascii { skipWhitespace() }
        guard offset == bytes.count else { throw PLYLoader.Failure.invalidBody(offset) }
    }

    private mutating func scalar(_ type: Scalar) throws -> Double {
        if format == .ascii {
            skipWhitespace()
            let start = offset
            while offset < bytes.count, !Self.whitespace(bytes[offset]), offset - start < 128 { offset += 1 }
            guard offset > start, offset - start < 128 else { throw PLYLoader.Failure.invalidBody(start) }
            let token = String(decoding: bytes[start..<offset], as: UTF8.self)
            if let bounds = type.integerBounds {
                guard let value = Int64(token), bounds.contains(value) else {
                    throw PLYLoader.Failure.invalidBody(start)
                }
                return Double(value)
            }
            let value = type == .float32 ? Float(token).map(Double.init) : Double(token)
            guard let value, value.isFinite else { throw PLYLoader.Failure.invalidBody(start) }
            return value
        }
        let start = offset
        guard type.width <= bytes.count - offset else { throw PLYLoader.Failure.invalidBody(start) }
        offset += type.width
        let value: Double = switch type {
        case .int8: Double(Int8(bitPattern: bytes[start]))
        case .uint8: Double(bytes[start])
        case .int16: Double(Int16(bitPattern: integer(UInt16.self, at: start)))
        case .uint16: Double(integer(UInt16.self, at: start))
        case .int32: Double(Int32(bitPattern: integer(UInt32.self, at: start)))
        case .uint32: Double(integer(UInt32.self, at: start))
        case .float32: Double(Float(bitPattern: integer(UInt32.self, at: start)))
        case .float64: Double(bitPattern: integer(UInt64.self, at: start))
        }
        guard value.isFinite else { throw PLYLoader.Failure.invalidBody(start) }
        return value
    }

    private func integer<T: FixedWidthInteger>(_: T.Type, at start: Int) -> T {
        let value = bytes.loadUnaligned(fromByteOffset: start, as: T.self)
        return format == .binaryLittleEndian ? T(littleEndian: value) : T(bigEndian: value)
    }

    private mutating func skipWhitespace() {
        while offset < bytes.count, Self.whitespace(bytes[offset]) { offset += 1 }
    }

    private static func whitespace(_ byte: UInt8) -> Bool {
        byte == 32 || (9...13).contains(byte)
    }
}
