import Foundation

/// Builds glTF documents with explicit component types, strides, and extensions.
/// Buffer 0 holds every view appended with `view(_:stride:)`.
struct GLTFExtensionFixture {
    private(set) var binary = Data()
    private(set) var views: [[String: Any]] = []
    private(set) var accessors: [[String: Any]] = []

    /// Appends a buffer view on buffer 0, aligned to four bytes.
    mutating func view(_ bytes: [UInt8], stride: Int? = nil) -> Int {
        while !binary.count.isMultiple(of: 4) { binary.append(0) }
        var view: [String: Any] = ["buffer": 0, "byteOffset": binary.count, "byteLength": bytes.count]
        if let stride { view["byteStride"] = stride }
        binary.append(contentsOf: bytes)
        views.append(view)
        return views.count - 1
    }

    /// Appends a raw view record, for example one with a compression extension.
    mutating func rawView(_ view: [String: Any]) -> Int {
        views.append(view)
        return views.count - 1
    }

    mutating func accessor(
        view: Int?, component: Int, type: String, count: Int, normalized: Bool = false, extra: [String: Any] = [:]
    ) -> Int {
        var accessor: [String: Any] = ["componentType": component, "type": type, "count": count]
        if let view { accessor["bufferView"] = view }
        if normalized { accessor["normalized"] = true }
        for (key, value) in extra { accessor[key] = value }
        accessors.append(accessor)
        return accessors.count - 1
    }

    /// Appends a view and an accessor over it.
    mutating func accessor(
        _ bytes: [UInt8], component: Int, type: String, count: Int, normalized: Bool = false, stride: Int? = nil
    ) -> Int {
        accessor(view: view(bytes, stride: stride), component: component, type: type, count: count, normalized: normalized)
    }

    /// A JSON document with buffer 0 embedded as a data URI.
    func document(_ sections: [String: Any], extraBuffers: [[String: Any]] = []) throws -> Data {
        var document: [String: Any] = [
            "asset": ["version": "2.0"],
            "buffers": [["byteLength": binary.count, "uri": "data:application/octet-stream;base64,\(binary.base64EncodedString())"]]
                + extraBuffers,
            "bufferViews": views, "accessors": accessors,
        ]
        for (key, value) in sections { document[key] = value }
        return try JSONSerialization.data(withJSONObject: document)
    }

    static func bytes<T: FixedWidthInteger>(_ values: [T]) -> [UInt8] {
        values.flatMap { value in withUnsafeBytes(of: value.littleEndian) { Array($0) } }
    }

    static func floats(_ values: [Float]) -> [UInt8] {
        bytes(values.map(\.bitPattern))
    }
}
