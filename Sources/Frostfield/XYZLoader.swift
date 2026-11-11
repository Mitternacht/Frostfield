import Foundation
import simd

/// Imports whitespace-separated XYZ and XYZRGB point records.
public enum XYZLoader {
    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case encoding
        case empty
        case invalidRecord(Int)

        public var errorDescription: String? {
            switch self {
            case .fileURL: "XYZ import requires a regular local file."
            case .size: "The XYZ file or point count exceeds the supported size."
            case .encoding: "The XYZ file needs UTF-8 text."
            case .empty: "The XYZ file contains no points."
            case .invalidRecord(let line): "XYZ line \(line) needs three finite coordinates and optionally three RGB values from 0 to 255."
            }
        }
    }

    private static let maximumBytes = 64 * 1_024 * 1_024
    private static let maximumPoints = 1 << 22

    public static func load(contentsOf url: URL) throws -> PointCloud {
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
        return try load(data)
    }

    public static func load(_ data: Data) throws -> PointCloud {
        guard data.count <= maximumBytes else { throw Failure.size }
        guard let text = String(data: data, encoding: .utf8) else { throw Failure.encoding }
        return try load(text)
    }

    /// Reads up to 2²² points from 64 MiB of text. RGB code values are sRGB.
    /// In mixed files the XYZ-only records use white. Blank lines and # comments are skipped.
    public static func load(_ text: String) throws -> PointCloud {
        guard text.utf8.count <= maximumBytes else { throw Failure.size }
        let content = text.first == "\u{feff}" ? text.dropFirst() : text[...]
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>]?
        let white = SIMD4<Float>(repeating: 1)
        var start = content.startIndex
        var line = 0
        while start < content.endIndex {
            line += 1
            let end = content[start...].firstIndex(where: \.isNewline) ?? content.endIndex
            let tokens = content[start..<end].prefix { $0 != "#" }.split(maxSplits: 6, whereSeparator: \.isWhitespace)
            start = end < content.endIndex ? content.index(after: end) : end
            if tokens.isEmpty { continue }
            guard tokens.count == 3 || tokens.count == 6,
                  let x = Float(tokens[0]), let y = Float(tokens[1]), let z = Float(tokens[2]),
                  x.isFinite, y.isFinite, z.isFinite else { throw Failure.invalidRecord(line) }
            guard positions.count < maximumPoints else { throw Failure.size }
            if tokens.count == 6 {
                guard let r = Float(tokens[3]), let g = Float(tokens[4]), let b = Float(tokens[5]),
                      (0...255).contains(r), (0...255).contains(g), (0...255).contains(b) else {
                    throw Failure.invalidRecord(line)
                }
                if colors == nil { colors = Array(repeating: white, count: positions.count) }
                colors?.append(SIMD4(
                    GLTFColorTables.sRGBToLinear(r / 255), GLTFColorTables.sRGBToLinear(g / 255),
                    GLTFColorTables.sRGBToLinear(b / 255), 1
                ))
            } else {
                colors?.append(white)
            }
            positions.append(SIMD3(x, y, z))
        }
        guard !positions.isEmpty else { throw Failure.empty }
        return try PointCloud(positions: positions, colors: colors)
    }
}
