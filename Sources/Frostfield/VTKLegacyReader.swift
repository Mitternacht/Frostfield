import Foundation
import simd

struct VTKLegacyReader {
    private typealias Failure = VTKLoader.Failure
    private typealias PolyData = VTKLoader.PolyData
    private typealias Cells = VTKLoader.Cells
    private typealias CellKind = VTKLoader.CellKind

    let bytes: UnsafeRawBufferPointer
    private var cursor = 0
    private var binary = false

    init(bytes: UnsafeRawBufferPointer) { self.bytes = bytes }

    mutating func read() throws -> VTKLoader.PolyData {
        guard let signature = try line(), signature.lowercased().hasPrefix("# vtk datafile version "),
              try line() != nil, let encoding = try line()?.trimmingCharacters(in: .whitespaces).uppercased(),
              encoding == "ASCII" || encoding == "BINARY" else { throw Failure.header }
        binary = encoding == "BINARY"
        guard let dataset = try header(), dataset.count == 2, dataset[0].uppercased() == "DATASET" else {
            throw Failure.header
        }
        guard dataset[1].uppercased() == "POLYDATA" else { throw Failure.unsupported(String(dataset[1])) }

        var result = PolyData()
        var hasPoints = false
        var association: Association?
        var cellEntries = 0
        while let words = try header() {
            let key = words[0].uppercased()
            switch key {
            case "POINTS":
                guard words.count == 3, !hasPoints, association == nil else { throw Failure.malformed(key) }
                let count = try Self.count(words[1], maximum: VTKLoader.maximumVertices)
                result.points = try vectors(count, type: ScalarType(words[2]))
                hasPoints = true
            case "VERTICES", "LINES", "POLYGONS", "TRIANGLE_STRIPS":
                guard words.count == 3, hasPoints, association == nil else { throw Failure.malformed(key) }
                let kind = Self.cellKind(key)
                guard !result.cells.contains(where: { $0.kind == kind }) else { throw Failure.malformed(key) }
                let count = try Self.count(words[1], maximum: VTKLoader.maximumCells - result.cellCount)
                let entries = try Self.count(words[2], maximum: VTKLoader.maximumConnectivity - cellEntries)
                cellEntries += entries
                result.cells.append(try cells(kind: kind, count: count, entries: entries, pointCount: result.points.count))
            case "POINT_DATA", "CELL_DATA":
                guard words.count == 2, hasPoints else { throw Failure.malformed(key) }
                let kind: Association = key == "POINT_DATA" ? .point : .cell
                let expected = kind == .point ? result.points.count : result.cellCount
                guard Int(words[1]) == expected else { throw Failure.malformed(key) }
                association = kind
            case "NORMALS", "COLOR_SCALARS":
                guard words.count == 3, let association else { throw Failure.malformed(key) }
                let count = association == .point ? result.points.count : result.cellCount
                let path = association == .point ? \PolyData.pointData : \PolyData.cellData
                if key == "NORMALS" {
                    let values = try vectors(count, type: ScalarType(words[2]))
                    if result[keyPath: path].normals == nil { result[keyPath: path].normals = values }
                } else {
                    let components = try Self.count(words[2], minimum: 1, maximum: 4)
                    if components >= 3 {
                        let values = try colors(count, components: components)
                        if result[keyPath: path].colors == nil { result[keyPath: path].colors = values }
                    } else {
                        try skip(count * components, type: binary ? .u8 : .f32)
                    }
                }
            case "SCALARS", "VECTORS", "TENSORS", "TEXTURE_COORDINATES":
                guard let association else { throw Failure.malformed(key) }
                let count = association == .point ? result.points.count : result.cellCount
                try skipAttribute(key, words: words, tuples: count)
            case "LOOKUP_TABLE":
                guard words.count == 3 else { throw Failure.malformed(key) }
                let count = try Self.count(words[2], maximum: VTKLoader.maximumVertices)
                try skip(count * 4, type: binary ? .u8 : .f32)
            case "FIELD":
                guard words.count == 3 else { throw Failure.malformed(key) }
                let count = try Self.count(words[2], maximum: 4_096)
                for _ in 0..<count {
                    guard let array = try header(), array.count == 4 else { throw Failure.malformed(key) }
                    let components = try Self.count(array[1], minimum: 1, maximum: 1_024)
                    let tuples = try Self.count(array[2], maximum: VTKLoader.maximumVertices)
                    try skip(components * tuples, type: ScalarType(array[3]))
                }
            default:
                throw Failure.unsupported(key)
            }
        }
        guard hasPoints else { throw Failure.malformed("POINTS") }
        return result
    }

    private enum Association { case point, cell }

    private enum ScalarType {
        case bit, u8, i8, u16, i16, u32, i32, u64, i64, f32, f64

        init(_ token: Substring) throws {
            switch token.lowercased() {
            case "bit": self = .bit
            case "unsigned_char": self = .u8
            case "char", "signed_char": self = .i8
            case "unsigned_short": self = .u16
            case "short": self = .i16
            case "unsigned_int": self = .u32
            case "int": self = .i32
            case "unsigned_long_long", "vtktypeuint64": self = .u64
            case "long_long", "vtktypeint64": self = .i64
            case "float": self = .f32
            case "double": self = .f64
            default: throw Failure.unsupported(String(token))
            }
        }

        var width: Int {
            switch self {
            case .bit, .u8, .i8: 1
            case .u16, .i16: 2
            case .u32, .i32, .f32: 4
            case .u64, .i64, .f64: 8
            }
        }
    }

    private static func cellKind(_ name: String) -> CellKind {
        switch name {
        case "VERTICES": .vertices
        case "LINES": .lines
        case "POLYGONS": .polygons
        default: .strips
        }
    }

    private static func count(_ token: Substring, minimum: Int = 0, maximum: Int) throws -> Int {
        guard let value = Int(token), value >= minimum else { throw Failure.malformed("count") }
        guard value <= maximum else { throw Failure.size }
        return value
    }

    private mutating func cells(kind: CellKind, count: Int, entries: Int, pointCount: Int) throws -> Cells {
        guard entries >= count else { throw Failure.malformed("connectivity") }
        if bytes[cursor...].starts(with: "OFFSETS ".utf8) {
            throw Failure.unsupported("split legacy OFFSETS/CONNECTIVITY")
        }
        try validateArraySize(entries, type: .i32)
        var result = Cells(kind: kind)
        result.offsets.reserveCapacity(count)
        result.indices.reserveCapacity(entries - count)
        var remaining = entries
        for _ in 0..<count {
            guard remaining > 0 else { throw Failure.malformed("connectivity") }
            let length = try integer()
            remaining -= 1
            let minimum = kind == .vertices ? 1 : (kind == .lines ? 2 : 3)
            guard length >= minimum, length <= remaining else { throw Failure.malformed("connectivity") }
            for _ in 0..<length {
                let index = try integer()
                guard index >= 0, index < pointCount else { throw Failure.malformed("point index") }
                result.indices.append(UInt32(index))
            }
            remaining -= length
            result.offsets.append(result.indices.count)
        }
        guard remaining == 0 else { throw Failure.malformed("connectivity") }
        try endArray(nonempty: entries > 0)
        return result
    }

    private mutating func vectors(_ count: Int, type: ScalarType) throws -> [SIMD3<Float>] {
        guard type != .bit else { throw Failure.unsupported("bit vectors") }
        try validateArraySize(count * 3, type: type)
        var result: [SIMD3<Float>] = []
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(try SIMD3(value(type), value(type), value(type)))
        }
        try endArray(nonempty: count > 0)
        return result
    }

    private mutating func colors(_ count: Int, components: Int) throws -> [SIMD4<Float>] {
        let type: ScalarType = binary ? .u8 : .f32
        try validateArraySize(count * components, type: type)
        let scale: Float = binary ? 255 : 1
        var result: [SIMD4<Float>] = []
        result.reserveCapacity(count)
        for _ in 0..<count {
            let rgb = try SIMD3(value(type), value(type), value(type)) / scale
            let alpha = components == 4 ? try value(type) / scale : 1
            let rgba = SIMD4(rgb, alpha)
            guard (0..<4).allSatisfy({ (0...1).contains(rgba[$0]) }) else { throw Failure.malformed("COLOR_SCALARS") }
            result.append(VTKLoader.linearColor(rgba))
        }
        try endArray(nonempty: count > 0)
        return result
    }

    private mutating func skipAttribute(_ name: String, words: [Substring], tuples: Int) throws {
        switch name {
        case "SCALARS":
            guard (3...4).contains(words.count) else { throw Failure.malformed(name) }
            let components = words.count == 4 ? try Self.count(words[3], minimum: 1, maximum: 4) : 1
            guard let table = try header(), table.count == 2, table[0].uppercased() == "LOOKUP_TABLE" else {
                throw Failure.malformed("LOOKUP_TABLE")
            }
            try skip(tuples * components, type: ScalarType(words[2]))
        case "TEXTURE_COORDINATES":
            guard words.count == 4 else { throw Failure.malformed(name) }
            let components = try Self.count(words[2], minimum: 1, maximum: 3)
            try skip(tuples * components, type: ScalarType(words[3]))
        default:
            guard words.count == 3 else { throw Failure.malformed(name) }
            try skip(tuples * (name == "VECTORS" ? 3 : 9), type: ScalarType(words[2]))
        }
    }

    private mutating func skip(_ count: Int, type: ScalarType) throws {
        try validateArraySize(count, type: type)
        if binary {
            cursor += type == .bit ? (count + 7) / 8 : count * type.width
        } else {
            for _ in 0..<count {
                guard Double(try token()) != nil else { throw Failure.malformed("attribute") }
            }
        }
        try endArray(nonempty: count > 0)
    }

    private func validateArraySize(_ count: Int, type: ScalarType) throws {
        let remaining = bytes.count - cursor
        if binary {
            let size = type == .bit ? (count + 7) / 8 : count * type.width
            guard size <= remaining else { throw Failure.truncated }
        } else {
            guard count <= remaining else { throw Failure.truncated }
        }
    }

    private mutating func integer() throws -> Int {
        if binary { return Int(try word(Int32.self)) }
        let value = try token()
        guard let integer = Int(value) else { throw Failure.malformed("integer") }
        return integer
    }

    private mutating func value(_ type: ScalarType) throws -> Float {
        let result: Float
        if binary {
            switch type {
            case .bit: throw Failure.unsupported("bit scalar")
            case .u8: result = try Float(word(UInt8.self))
            case .i8: result = try Float(word(Int8.self))
            case .u16: result = try Float(word(UInt16.self))
            case .i16: result = try Float(word(Int16.self))
            case .u32: result = try Float(word(UInt32.self))
            case .i32: result = try Float(word(Int32.self))
            case .u64: result = try Float(word(UInt64.self))
            case .i64: result = try Float(word(Int64.self))
            case .f32: result = try Float(bitPattern: word(UInt32.self))
            case .f64: result = try Float(Double(bitPattern: word(UInt64.self)))
            }
        } else {
            guard let parsed = Float(try token()) else { throw Failure.malformed("number") }
            result = parsed
        }
        guard result.isFinite else { throw Failure.malformed("number") }
        return result
    }

    private mutating func word<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let count = MemoryLayout<T>.size
        guard count <= bytes.count - cursor else { throw Failure.truncated }
        let value = bytes.loadUnaligned(fromByteOffset: cursor, as: T.self)
        cursor += count
        return T(bigEndian: value)
    }

    private mutating func header() throws -> [Substring]? {
        while let text = try line() {
            let words = text.split(whereSeparator: \.isWhitespace)
            if !words.isEmpty { return words }
        }
        return nil
    }

    private mutating func line() throws -> String? {
        guard cursor < bytes.count else { return nil }
        let start = cursor
        while cursor < bytes.count, bytes[cursor] != 10 {
            cursor += 1
            guard cursor - start <= 4_096 else { throw Failure.malformed("header line") }
        }
        let range = start..<cursor
        if cursor < bytes.count { cursor += 1 }
        guard let result = String(bytes: bytes[range], encoding: .utf8) else { throw Failure.malformed("header line") }
        return result.hasSuffix("\r") ? String(result.dropLast()) : result
    }

    private mutating func token() throws -> String {
        while cursor < bytes.count, Self.isSpace(bytes[cursor]) { cursor += 1 }
        let start = cursor
        while cursor < bytes.count, !Self.isSpace(bytes[cursor]) {
            cursor += 1
            guard cursor - start <= 128 else { throw Failure.malformed("number") }
        }
        guard start < cursor else { throw Failure.truncated }
        guard let value = String(bytes: bytes[start..<cursor], encoding: .utf8) else {
            throw Failure.malformed("number")
        }
        return value
    }

    private mutating func endArray(nonempty: Bool) throws {
        guard nonempty, cursor < bytes.count else { return }
        if !binary {
            while cursor < bytes.count, bytes[cursor] == 32 || bytes[cursor] == 9 { cursor += 1 }
        }
        if cursor < bytes.count, bytes[cursor] == 13 { cursor += 1 }
        guard cursor == bytes.count || bytes[cursor] == 10 else { throw Failure.malformed("array terminator") }
        if cursor < bytes.count { cursor += 1 }
    }

    private static func isSpace(_ byte: UInt8) -> Bool {
        byte == 32 || (9...13).contains(byte)
    }
}
