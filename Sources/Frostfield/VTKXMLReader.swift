import Foundation
import simd

struct VTKXMLReader {
    private typealias Failure = VTKLoader.Failure
    private typealias PolyData = VTKLoader.PolyData

    let data: Data

    static func recognizes(_ data: Data) -> Bool {
        withoutBOM(data).first { !isSpace($0) } == 60
    }

    private static func withoutBOM(_ data: Data) -> Data {
        data.starts(with: [0xef, 0xbb, 0xbf]) ? data.dropFirst(3) : data
    }

    private static func isSpace(_ value: UInt8) -> Bool {
        value == 32 || (9...13).contains(value)
    }

    private func validateXMLHeader() throws {
        let content = Self.withoutBOM(data)
        guard !content.contains(0) else { throw Failure.unsupported("XML encoding") }
        // Limit the decoder to encodings where XML markup has its literal ASCII bytes.
        if content.starts(with: "<?xml".utf8), let next = content.dropFirst(5).first, Self.isSpace(next) {
            let prefix = content.prefix(1_024)
            guard let end = prefix.range(of: Data("?>".utf8)) else { throw Failure.header }
            let declaration = String(decoding: prefix[..<end.lowerBound], as: UTF8.self)
            if let field = declaration.range(of: "encoding") {
                let tokens = declaration[field.upperBound...].split { $0.isWhitespace || $0 == "=" }
                let allowed = ["\"utf-8\"", "'utf-8'", "\"us-ascii\"", "'us-ascii'"]
                guard let encoding = tokens.first, allowed.contains(encoding.lowercased()) else {
                    throw Failure.unsupported("XML encoding")
                }
            }
        }
        guard content.range(of: Data("<!DOCTYPE".utf8)) == nil else { throw Failure.unsupported("XML DTD") }
    }

    func read() throws -> VTKLoader.PolyData {
        try validateXMLHeader()
        let document = Document()
        let parser = XMLParser(data: data)
        parser.delegate = document
        parser.shouldResolveExternalEntities = false
        let parsed = parser.parse()
        if let failure = document.failure { throw failure }
        guard parsed, let root = document.root else { throw Failure.malformed("XML") }
        guard root["type"] == "PolyData" else { throw Failure.unsupported(root["type"] ?? "XML dataset") }
        guard let piece = document.piece else { throw Failure.malformed("XML Piece") }

        let payload = try VTKXMLPayload(
            root: root, appended: readAppended(document), offsets: document.offsets
        )
        var result = PolyData()
        let pointCount = try Self.count(piece["NumberOfPoints"], maximum: VTKLoader.maximumVertices)
        guard let points = document.arrays["Points"], points.count == 1 else { throw Failure.malformed("Points") }
        result.points = try vectors(points[0], count: pointCount, payload: payload)
        if let normalName = document.pointNormal, !normalName.isEmpty {
            let normals = document.arrays["PointData"]?.filter { $0.attributes["Name"] == normalName } ?? []
            guard normals.count == 1 else { throw Failure.malformed("PointData normals") }
            result.pointData.normals = try vectors(normals[0], count: pointCount, payload: payload)
        }

        let sections: [(String, VTKLoader.CellKind)] = [
            ("Verts", .vertices), ("Lines", .lines), ("Strips", .strips), ("Polys", .polygons),
        ]
        var entriesLeft = VTKLoader.maximumConnectivity
        for (name, kind) in sections {
            let count = try Self.count(piece["NumberOf" + name] ?? "0", maximum: VTKLoader.maximumCells - result.cellCount)
            guard let arrays = document.arrays[name], !arrays.isEmpty else {
                if count == 0 { continue }
                throw Failure.malformed(name)
            }
            guard arrays.count == 2,
                  let connectivity = arrays.first(where: { $0.attributes["Name"] == "connectivity" }),
                  let offsets = arrays.first(where: { $0.attributes["Name"] == "offsets" }) else {
                throw Failure.malformed(name)
            }
            let ends = try integers(offsets, count: count, limit: count, payload: payload)
            let indices = try integers(connectivity, count: ends.last ?? 0, limit: entriesLeft, payload: payload)
            entriesLeft -= indices.count
            let minimum = kind == .vertices ? 1 : (kind == .lines ? 2 : 3)
            var start = 0
            for end in ends {
                guard end <= indices.count, end >= start + minimum else { throw Failure.malformed(name + " offsets") }
                start = end
            }
            guard start == indices.count,
                  indices.allSatisfy({ $0 >= 0 && $0 < pointCount }) else { throw Failure.malformed(name + " connectivity") }
            result.cells.append(VTKLoader.Cells(kind: kind, indices: indices.map(UInt32.init), offsets: ends))
        }
        return result
    }

    private struct ArrayRecord {
        let attributes: [String: String]
        var content = Data()
    }

    private final class Document: NSObject, XMLParserDelegate {
        var root: [String: String]?
        var piece: [String: String]?
        var pointNormal: String?
        var arrays: [String: [ArrayRecord]] = [:]
        var offsets: Set<Int> = []
        var appendedEncoding: String?
        var failure: Failure?
        private var path: [String] = []
        private var active: ArrayRecord?
        private var arrayCount = 0

        func parser(
            _ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
            qualifiedName: String?, attributes: [String: String]
        ) {
            path.append(name)
            guard path.count <= 64 else { fail(.size, parser: parser); return }
            if path.count == 1 {
                guard name == "VTKFile" else { fail(.header, parser: parser); return }
                root = attributes
            } else if path == ["VTKFile", "PolyData", "Piece"] {
                guard piece == nil else { fail(.unsupported("multiple XML pieces"), parser: parser); return }
                piece = attributes
            } else if path == ["VTKFile", "PolyData", "Piece", "PointData"] {
                pointNormal = attributes["Normals"]
            } else if path == ["VTKFile", "AppendedData"] {
                guard appendedEncoding == nil, attributes["encoding"] == "base64" else {
                    fail(.unsupported("XML appended encoding"), parser: parser)
                    return
                }
                appendedEncoding = "base64"
            } else if name == "DataArray" {
                arrayCount += 1
                guard arrayCount <= 4_096, active == nil else { fail(.size, parser: parser); return }
                if attributes["format"] == "appended" {
                    guard let text = attributes["offset"], let offset = Int(text), offset >= 0,
                          offsets.insert(offset).inserted else { fail(.malformed("appended offset"), parser: parser); return }
                }
                if path.count == 5, path.starts(with: ["VTKFile", "PolyData", "Piece"]),
                   ["Points", "PointData", "Verts", "Lines", "Strips", "Polys"].contains(path[3]) {
                    active = ArrayRecord(attributes: attributes)
                }
            }
        }

        func parser(_ parser: XMLParser, foundCharacters text: String) {
            if path.last == "DataArray" { active?.content.append(contentsOf: text.utf8) }
        }

        func parser(_ parser: XMLParser, foundCDATA block: Data) {
            if path.last == "DataArray" { active?.content.append(block) }
        }

        func parser(
            _ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?
        ) {
            if name == "DataArray", path.count == 5, let active {
                arrays[path[3], default: []].append(active)
                self.active = nil
            }
            _ = path.popLast()
        }

        private func fail(_ failure: Failure, parser: XMLParser) {
            self.failure = failure
            parser.abortParsing()
        }
    }

    private func readAppended(_ document: Document) throws -> Data? {
        guard document.appendedEncoding != nil else { return nil }
        // Offsets count bytes in the source XML. XMLParser normalizes newlines.
        let startTag = Data("<AppendedData".utf8)
        let endTag = Data("</AppendedData".utf8)
        guard let start = data.range(of: startTag),
              data.range(of: startTag, in: start.upperBound..<data.endIndex) == nil,
              let close = data.range(of: endTag, in: start.upperBound..<data.endIndex),
              let angle = data[start.upperBound..<close.lowerBound].firstIndex(of: 62),
              let sentinel = data[(angle + 1)..<close.lowerBound].firstIndex(where: { !Self.isSpace($0) }),
              data[sentinel] == 95 else { throw Failure.malformed("AppendedData") }
        return data[(sentinel + 1)..<close.lowerBound]
    }

    private static func count(_ text: String?, maximum: Int) throws -> Int {
        guard let text, let value = Int(text), value >= 0 else { throw Failure.malformed("XML count") }
        guard value <= maximum else { throw Failure.size }
        return value
    }

    private func vectors(_ array: ArrayRecord, count: Int, payload: VTKXMLPayload) throws -> [SIMD3<Float>] {
        let type = array.attributes["type"]
        guard type == "Float32" || type == "Float64", array.attributes["NumberOfComponents"] == "3" else {
            throw Failure.unsupported("XML vector type or width")
        }
        let values: [Float]
        if array.attributes["format"] ?? "ascii" == "ascii" {
            values = try Self.ascii(array.content, count: count * 3, parse: Float.init)
        } else {
            let width = type == "Float32" ? 4 : 8
            let data = try payload.decode(array.attributes, inline: array.content, size: count * 3 * width)
            values = data.withUnsafeBytes { bytes in
                (0..<(count * 3)).map { index in
                    if width == 4 {
                        return Float(bitPattern: payload.word(UInt32.self, bytes: bytes, at: index * width))
                    }
                    return Float(Double(bitPattern: payload.word(UInt64.self, bytes: bytes, at: index * width)))
                }
            }
        }
        guard values.allSatisfy(\.isFinite) else { throw Failure.malformed("XML vector") }
        return stride(from: 0, to: values.count, by: 3).map { SIMD3(values[$0], values[$0 + 1], values[$0 + 2]) }
    }

    private func integers(
        _ array: ArrayRecord, count: Int, limit: Int, payload: VTKXMLPayload
    ) throws -> [Int] {
        guard count >= 0 else { throw Failure.malformed("XML connectivity count") }
        guard count <= limit else { throw Failure.size }
        let type = array.attributes["type"]
        let width: Int
        switch type {
        case "Int32", "UInt32": width = 4
        case "Int64", "UInt64": width = 8
        default: throw Failure.unsupported("XML connectivity type")
        }
        guard array.attributes["NumberOfComponents"] ?? "1" == "1" else {
            throw Failure.malformed("XML connectivity width")
        }

        let signed = type == "Int32" || type == "Int64"
        if array.attributes["format"] ?? "ascii" == "ascii" {
            return try Self.ascii(array.content, count: count) { token in
                if signed { return width == 4 ? Int32(token).map(Int.init) : Int(token) }
                if width == 4 { return UInt32(token).map(Int.init) }
                return UInt64(token).flatMap(Int.init(exactly:))
            }
        }

        let data = try payload.decode(array.attributes, inline: array.content, size: count * width)
        return try data.withUnsafeBytes { bytes in
            try (0..<count).map { index in
                let offset = index * width
                if width == 4 {
                    let word = payload.word(UInt32.self, bytes: bytes, at: offset)
                    return signed ? Int(Int32(bitPattern: word)) : Int(word)
                }
                let word = payload.word(UInt64.self, bytes: bytes, at: offset)
                if signed { return Int(Int64(bitPattern: word)) }
                guard let value = Int(exactly: word) else { throw Failure.malformed("XML connectivity value") }
                return value
            }
        }
    }

    private static func ascii<T>(_ data: Data, count: Int, parse: (String) -> T?) throws -> [T] {
        guard count <= data.count else { throw Failure.truncated }
        var values: [T] = []
        values.reserveCapacity(count)
        var cursor = data.startIndex
        while cursor < data.endIndex {
            while cursor < data.endIndex, isSpace(data[cursor]) { cursor += 1 }
            if cursor == data.endIndex { break }
            let start = cursor
            while cursor < data.endIndex, !isSpace(data[cursor]) {
                cursor += 1
                guard cursor - start <= 128 else { throw Failure.malformed("XML number") }
            }
            guard values.count < count, let value = parse(String(decoding: data[start..<cursor], as: UTF8.self)) else {
                throw Failure.malformed("XML number or count")
            }
            values.append(value)
        }
        guard values.count == count else { throw Failure.truncated }
        return values
    }
}
