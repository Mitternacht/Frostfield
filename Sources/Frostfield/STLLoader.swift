import Foundation
import simd

/// Imports triangle facets from ASCII or little-endian binary STL.
public enum STLLoader {
    public struct Solid: Equatable, Sendable {
        public let name: String
        /// The index range of this ASCII solid. Empty solids have an empty range.
        public let indexRange: Range<Int>
    }

    public struct Asset: Sendable {
        /// One vertex per facet corner. Authored normals and winding are preserved.
        public let geometry: MeshGeometry
        /// ASCII solids in file order. Nonempty solids map to mesh groups with
        /// their array index as the material slot. Binary files have no solids.
        public let solids: [Solid]
        /// The binary Magics header's linear RGB and straight alpha, when present.
        /// Imported vertex colors already include this alpha.
        public let defaultColor: SIMD4<Float>?
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case invalidBinaryLength
        case invalidASCII(Int)
        case invalidFacet(Int)
        case noFacets

        public var errorDescription: String? {
            switch self {
            case .fileURL: "STL import requires a regular local file."
            case .size: "The STL file or its geometry exceeds the supported size."
            case .invalidBinaryLength: "The STL byte count must match its declared binary facet count."
            case .invalidASCII(let offset): "The ASCII STL syntax is invalid at byte \(offset)."
            case .invalidFacet(let index): "STL facet \(index) needs a finite normal and three finite positions."
            case .noFacets: "The STL file contains no triangle facets."
            }
        }
    }

    private static let maximumFileBytes = 256 * 1_024 * 1_024
    private static let maximumFacets = 1 << 20

    public static func load(contentsOf url: URL) throws -> Asset {
        guard url.isFileURL else { throw Failure.fileURL }
        let info = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true else { throw Failure.fileURL }
        guard let size = info.fileSize, size <= maximumFileBytes else { throw Failure.size }
        return try load(Data(contentsOf: url, options: .mappedIfSafe))
    }

    /// Reads up to 256 MiB and 2²⁰ facets. Binary files with a `COLOR=` header
    /// use the Magics layout for per-facet RGB. Zero normals use the geometric
    /// normal, or up for a degenerate triangle.
    public static func load(_ data: Data) throws -> Asset {
        guard data.count <= maximumFileBytes else { throw Failure.size }
        return try data.withUnsafeBytes { bytes in
            if bytes.count >= 84 {
                let count = Int(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: 80, as: UInt32.self)))
                // A binary header may itself begin with "solid".
                if 84 + 50 * count == bytes.count { return try binary(bytes, count: count) }
            }
            var reader = ASCIIReader(bytes)
            guard reader.beginsWithSolid else { throw Failure.invalidBinaryLength }
            return try ascii(&reader)
        }
    }

    private static func binary(_ bytes: UnsafeRawBufferPointer, count: Int) throws -> Asset {
        guard count > 0 else { throw Failure.noFacets }
        guard count <= maximumFacets else { throw Failure.size }

        func vector(at offset: Int) -> SIMD3<Float> {
            func component(_ offset: Int) -> Float {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
            }
            return SIMD3(component(offset), component(offset + 4), component(offset + 8))
        }

        let marker = Array("COLOR=".utf8)
        let colorOffset = (0...70).reversed().first { bytes[$0..<($0 + marker.count)].elementsEqual(marker) }
        let defaultColor = colorOffset.map { start in
            linearColor(Float(bytes[start + 6]) / 255, Float(bytes[start + 7]) / 255,
                        Float(bytes[start + 8]) / 255, alpha: Float(bytes[start + 9]) / 255)
        }
        var vertices: [MeshVertex] = []
        vertices.reserveCapacity(count * 3)
        for facet in 0..<count {
            let start = 84 + 50 * facet
            var color = defaultColor
            if let base = defaultColor {
                let packed = UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: start + 48, as: UInt16.self))
                if packed & 0x8000 == 0 {
                    color = linearColor(Float(packed & 31) / 31, Float((packed >> 5) & 31) / 31,
                                        Float((packed >> 10) & 31) / 31, alpha: base.w)
                }
            }
            try appendFacet(
                normal: vector(at: start), a: vector(at: start + 12), b: vector(at: start + 24),
                c: vector(at: start + 36), color: color, to: &vertices
            )
        }
        return Asset(geometry: try geometry(vertices), solids: [], defaultColor: defaultColor)
    }

    private static func ascii(_ reader: inout ASCIIReader) throws -> Asset {
        var vertices: [MeshVertex] = []
        var solids: [Solid] = []
        while let token = try reader.token() {
            guard token.lowercased().hasPrefix("solid") else { throw Failure.invalidASCII(reader.offset) }
            guard solids.count < 65_536 else { throw Failure.size }
            let name = try reader.solidName(token, keywordLength: 5)
            let start = vertices.count
            while true {
                let keyword = try reader.requiredToken().lowercased()
                if keyword.hasPrefix("endsolid") {
                    _ = try reader.solidName(keyword, keywordLength: 8)
                    break
                }
                guard keyword == "facet" else { throw Failure.invalidASCII(reader.offset) }
                guard vertices.count / 3 < maximumFacets else { throw Failure.size }
                try reader.expect("normal")
                let normal = try reader.vector()
                try reader.expect("outer")
                try reader.expect("loop")
                let a = try reader.vertex()
                let b = try reader.vertex()
                let c = try reader.vertex()
                try reader.expect("endloop")
                try reader.expect("endfacet")
                try appendFacet(normal: normal, a: a, b: b, c: c, color: nil, to: &vertices)
            }
            solids.append(Solid(name: name, indexRange: start..<vertices.count))
        }
        let groups = solids.enumerated().compactMap { index, solid in
            solid.indexRange.isEmpty ? nil : MeshGeometryGroup(
                start: solid.indexRange.lowerBound, count: solid.indexRange.count, materialIndex: index
            )
        }
        return Asset(geometry: try geometry(vertices).withGroups(groups), solids: solids, defaultColor: nil)
    }

    private static func appendFacet(
        normal: SIMD3<Float>, a: SIMD3<Float>, b: SIMD3<Float>, c: SIMD3<Float>,
        color: SIMD4<Float>?, to vertices: inout [MeshVertex]
    ) throws {
        guard finite(normal), finite(a), finite(b), finite(c) else { throw Failure.invalidFacet(vertices.count / 3) }
        var resolved = normal
        if normal == .zero {
            let origin = SIMD3<Double>(a)
            let cross = simd_cross(SIMD3<Double>(b) - origin, SIMD3<Double>(c) - origin)
            resolved = cross == .zero ? SIMD3(0, 1, 0) : SIMD3<Float>(simd_normalize(cross))
        }
        vertices.append(MeshVertex(position: a, normal: resolved, color: color))
        vertices.append(MeshVertex(position: b, normal: resolved, color: color))
        vertices.append(MeshVertex(position: c, normal: resolved, color: color))
    }

    private static func finite(_ vector: SIMD3<Float>) -> Bool {
        vector.x.isFinite && vector.y.isFinite && vector.z.isFinite
    }

    private static func linearColor(_ red: Float, _ green: Float, _ blue: Float, alpha: Float) -> SIMD4<Float> {
        SIMD4(GLTFColorTables.sRGBToLinear(red), GLTFColorTables.sRGBToLinear(green),
              GLTFColorTables.sRGBToLinear(blue), alpha)
    }

    private static func geometry(_ vertices: [MeshVertex]) throws -> MeshGeometry {
        guard !vertices.isEmpty else { throw Failure.noFacets }
        return try MeshGeometry(vertices: vertices, indices: (0..<vertices.count).map { UInt32($0) })
    }

    private struct ASCIIReader {
        let bytes: UnsafeRawBufferPointer
        private(set) var offset = 0

        init(_ bytes: UnsafeRawBufferPointer) {
            self.bytes = bytes
            if bytes.prefix(3).elementsEqual([0xEF, 0xBB, 0xBF]) { offset = 3 }
            skipWhitespace()
        }

        var beginsWithSolid: Bool {
            let signature: [UInt8] = [115, 111, 108, 105, 100]
            guard bytes.count - offset >= signature.count else { return false }
            return signature.indices.allSatisfy { (bytes[offset + $0] | 32) == signature[$0] }
        }

        mutating func token() throws -> String? {
            skipWhitespace()
            let start = offset
            guard start < bytes.count else { return nil }
            while offset < bytes.count, !Self.whitespace(bytes[offset]), offset - start < 128 {
                offset += 1
            }
            guard offset - start < 128 else { throw Failure.invalidASCII(start) }
            return String(decoding: bytes[start..<offset], as: UTF8.self)
        }

        mutating func requiredToken() throws -> String {
            guard let token = try token() else { throw Failure.invalidASCII(offset) }
            return token
        }

        mutating func expect(_ keyword: String) throws {
            guard try requiredToken().lowercased() == keyword else { throw Failure.invalidASCII(offset) }
        }

        mutating func vector() throws -> SIMD3<Float> {
            let start = offset
            guard let x = Float(try requiredToken()), let y = Float(try requiredToken()),
                  let z = Float(try requiredToken()) else { throw Failure.invalidASCII(start) }
            return SIMD3(x, y, z)
        }

        mutating func vertex() throws -> SIMD3<Float> {
            try expect("vertex")
            return try vector()
        }

        mutating func solidName(_ token: String, keywordLength: Int) throws -> String {
            let start = offset
            while offset < bytes.count, bytes[offset] != 10, bytes[offset] != 13, offset - start < 4_096 {
                offset += 1
            }
            guard offset - start < 4_096 else { throw Failure.invalidASCII(start) }
            let remainder = String(decoding: bytes[start..<offset], as: UTF8.self)
            return (String(token.dropFirst(keywordLength)) + remainder).trimmingCharacters(in: .whitespaces)
        }

        private mutating func skipWhitespace() {
            while offset < bytes.count, Self.whitespace(bytes[offset]) { offset += 1 }
        }

        private static func whitespace(_ byte: UInt8) -> Bool {
            byte == 32 || (9...13).contains(byte)
        }
    }
}
