import Foundation
import Metal
import simd
import zlib

/// Loads an embedded, scalar three-dimensional Nearly Raw Raster Data volume.
public enum NRRDLoader {
    /// Source samples in file order, with the first axis varying fastest.
    public enum Scalars: Equatable, Sendable {
        case int8([Int8])
        case uint8([UInt8])
        case int16([Int16])
        case uint16([UInt16])
        case int32([Int32])
        case uint32([UInt32])
        case float32([Float])
        case float64([Double])

        public var count: Int {
            switch self {
            case .int8(let values): values.count
            case .uint8(let values): values.count
            case .int16(let values): values.count
            case .uint16(let values): values.count
            case .int32(let values): values.count
            case .uint32(let values): values.count
            case .float32(let values): values.count
            case .float64(let values): values.count
            }
        }
    }

    public enum TextureStorage: Equatable, Sendable {
        /// Keeps integer and Float32 sample bits in single-channel Metal storage.
        case native
        /// Converts samples to Float32 for floating-point shader access.
        case float32
    }

    public struct Volume: Sendable {
        public let dimensions: SIMD3<Int>
        public let scalars: Scalars
        /// Physical displacement of one step along each storage axis, in the declared coordinate space.
        public let spaceDirections: [SIMD3<Double>]
        /// Physical position of voxel (0, 0, 0).
        public let spaceOrigin: SIMD3<Double>
        public let space: String?
        /// Units of the physical coordinate axes, in the declared space.
        public let spaceUnits: [String]?

        public var spacing: SIMD3<Double> {
            SIMD3(simd_length(spaceDirections[0]), simd_length(spaceDirections[1]), simd_length(spaceDirections[2]))
        }

        /// Column-vector transform from voxel index to the file's physical coordinate system.
        public var voxelToSpace: simd_double4x4 { affine(signs: SIMD3(repeating: 1)) }

        /// Voxel index to patient RAS coordinates when the source names RAS, LAS, or LPS.
        public var voxelToRAS: simd_double4x4? {
            switch space?.lowercased() {
            case "ras", "right-anterior-superior": affine(signs: SIMD3(1, 1, 1))
            case "las", "left-anterior-superior": affine(signs: SIMD3(-1, 1, 1))
            case "lps", "left-posterior-superior": affine(signs: SIMD3(-1, -1, 1))
            default: nil
            }
        }

        private func affine(signs: SIMD3<Double>) -> simd_double4x4 {
            simd_double4x4(columns: (
                SIMD4(spaceDirections[0] * signs, 0),
                SIMD4(spaceDirections[1] * signs, 0),
                SIMD4(spaceDirections[2] * signs, 0),
                SIMD4(spaceOrigin * signs, 1)
            ))
        }

        /// Uploads the original voxel order as a shader-readable single-channel 3D texture.
        /// Float64 sources require `.float32`. Integer or Float64 values converted to
        /// Float32 can round, and finite values beyond Float32 range throw.
        public func makeTexture(
            device: any MTLDevice, storage: TextureStorage = .native
        ) throws -> any MTLTexture {
            func upload<T>(_ values: [T], format: MTLPixelFormat) throws -> any MTLTexture {
                let descriptor = MTLTextureDescriptor()
                descriptor.textureType = .type3D
                descriptor.pixelFormat = format
                descriptor.width = dimensions.x
                descriptor.height = dimensions.y
                descriptor.depth = dimensions.z
                descriptor.storageMode = .shared
                descriptor.usage = .shaderRead
                guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.textureAllocation }
                texture.label = "Frostfield NRRD volume"
                let row = dimensions.x * MemoryLayout<T>.stride
                values.withUnsafeBytes { bytes in
                    texture.replace(
                        region: MTLRegionMake3D(0, 0, 0, dimensions.x, dimensions.y, dimensions.z),
                        mipmapLevel: 0, slice: 0, withBytes: bytes.baseAddress!,
                        bytesPerRow: row, bytesPerImage: row * dimensions.y
                    )
                }
                return texture
            }
            func uploadInteger<T: BinaryInteger & Sendable>(_ values: [T], format: MTLPixelFormat) throws -> any MTLTexture {
                if storage == .float32 { return try upload(values.map { Float($0) }, format: .r32Float) }
                return try upload(values, format: format)
            }
            switch scalars {
            case .int8(let values): return try uploadInteger(values, format: .r8Sint)
            case .uint8(let values): return try uploadInteger(values, format: .r8Uint)
            case .int16(let values): return try uploadInteger(values, format: .r16Sint)
            case .uint16(let values): return try uploadInteger(values, format: .r16Uint)
            case .int32(let values): return try uploadInteger(values, format: .r32Sint)
            case .uint32(let values): return try uploadInteger(values, format: .r32Uint)
            case .float32(let values): return try upload(values, format: .r32Float)
            case .float64(let values):
                guard storage == .float32 else { throw Failure.doubleTexture }
                let converted = try values.map { value in
                    let sample = Float(value)
                    guard !value.isFinite || sample.isFinite else { throw Failure.floatTextureRange }
                    return sample
                }
                return try upload(converted, format: .r32Float)
            }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case invalidHeader(String)
        case unsupported(String)
        case invalidPayload
        case invalidGzip
        case invalidSample(Int)
        case textureAllocation
        case doubleTexture
        case floatTextureRange

        public var errorDescription: String? {
            switch self {
            case .fileURL: "NRRD import requires a regular local file."
            case .size: "The NRRD file or decoded volume exceeds the supported size."
            case .invalidHeader(let field): "The NRRD header has an invalid or missing \(field) field."
            case .unsupported(let feature): "NRRD import does not support \(feature)."
            case .invalidPayload: "The NRRD payload does not match its dimensions and scalar type."
            case .invalidGzip: "The NRRD gzip stream is invalid or has the wrong decoded size."
            case .invalidSample(let index): "NRRD sample \(index) cannot be read as the declared scalar type."
            case .textureAllocation: "Metal could not allocate the NRRD texture."
            case .doubleTexture: "Select Float32 conversion to upload an NRRD Float64 volume."
            case .floatTextureRange: "An NRRD sample exceeds the finite Float32 range."
            }
        }
    }

    fileprivate static let maximumFileBytes = 256 * 1_024 * 1_024
    fileprivate static let maximumVoxels = 1 << 24

    public static func load(contentsOf url: URL) throws -> Volume {
        guard url.isFileURL else { throw Failure.fileURL }
        let info = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true else { throw Failure.fileURL }
        if let size = info.fileSize, size > maximumFileBytes { throw Failure.size }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var data = Data()
        while data.count <= maximumFileBytes {
            let count = min(1 << 20, maximumFileBytes + 1 - data.count)
            guard let chunk = try file.read(upToCount: count), !chunk.isEmpty else { break }
            data.append(chunk)
        }
        return try load(data)
    }

    /// Reads raw, gzip, ASCII, or hexadecimal embedded data. The volume
    /// can hold up to 2²⁴ voxels with up to 2,048 samples on each axis.
    public static func load(_ data: Data) throws -> Volume {
        guard data.count <= maximumFileBytes else { throw Failure.size }
        return try data.withUnsafeBytes { bytes in
            let header = try NRRDHeader(bytes)
            let payload = UnsafeRawBufferPointer(rebasing: bytes[header.payloadOffset...])
            let scalars: Scalars
            switch header.encoding {
            case .raw:
                scalars = try header.readBinary(payload)
            case .gzip:
                let expanded = try inflateGzip(payload, count: header.byteCount)
                scalars = try expanded.withUnsafeBytes { try header.readBinary($0) }
            case .ascii:
                scalars = try header.readText(payload, radix: 10)
            case .hex:
                if let decoded = header.decodeHexBytes(payload) {
                    scalars = try decoded.withUnsafeBytes { try header.readBinary($0) }
                } else {
                    // Older NRRDLoader callers also use one hexadecimal number per voxel.
                    scalars = try header.readText(payload, radix: 16)
                }
            }
            return Volume(
                dimensions: header.dimensions, scalars: scalars,
                spaceDirections: header.directions, spaceOrigin: header.origin,
                space: header.space, spaceUnits: header.units
            )
        }
    }

    private static func inflateGzip(_ input: UnsafeRawBufferPointer, count: Int) throws -> Data {
        guard input.count >= 18, let start = input.bindMemory(to: UInt8.self).baseAddress else { throw Failure.invalidGzip }
        var stream = z_stream()
        let gzipWindow = Int32(15 + 16)
        guard inflateInit2_(&stream, gzipWindow, zlibVersion(), Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
            throw Failure.invalidGzip
        }
        defer { inflateEnd(&stream) }
        var result = Data(count: count + 1)
        try result.withUnsafeMutableBytes { (output: UnsafeMutableRawBufferPointer) in
            stream.next_in = UnsafeMutablePointer(mutating: start)
            stream.avail_in = UInt32(input.count)
            stream.next_out = output.bindMemory(to: UInt8.self).baseAddress
            stream.avail_out = UInt32(output.count)
            guard inflate(&stream, Z_FINISH) == Z_STREAM_END, stream.total_out == count, stream.avail_in == 0 else {
                throw Failure.invalidGzip
            }
        }
        result.removeLast()
        return result
    }
}

private struct NRRDHeader {
    typealias Failure = NRRDLoader.Failure
    enum Encoding { case raw, gzip, ascii, hex }

    enum Scalar {
        case int8, uint8, int16, uint16, int32, uint32, float32, float64

        init?(_ name: String) {
            switch name.lowercased() {
            case "signed char", "int8", "int8_t": self = .int8
            case "uchar", "unsigned char", "uint8", "uint8_t": self = .uint8
            case "short", "short int", "signed short", "signed short int", "int16", "int16_t": self = .int16
            case "ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t": self = .uint16
            case "int", "signed int", "int32", "int32_t": self = .int32
            case "uint", "unsigned int", "uint32", "uint32_t": self = .uint32
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
    }

    let dimensions: SIMD3<Int>
    let voxelCount: Int
    let encoding: Encoding
    let scalar: Scalar
    let littleEndian: Bool
    let payloadOffset: Int
    let directions: [SIMD3<Double>]
    let origin: SIMD3<Double>
    let space: String?
    let units: [String]?
    var byteCount: Int { voxelCount * scalar.width }

    init(_ bytes: UnsafeRawBufferPointer) throws {
        let limit = min(bytes.count, 1 << 16)
        var offset = 0
        func line() throws -> String {
            let start = offset
            while offset < limit, bytes[offset] != 10 { offset += 1 }
            guard offset < limit else { throw Failure.invalidHeader("terminator") }
            let end = offset > start && bytes[offset - 1] == 13 ? offset - 1 : offset
            offset += 1
            guard let text = String(bytes: bytes[start..<end], encoding: .ascii) else {
                throw Failure.invalidHeader("ASCII")
            }
            return text
        }

        let signature = try line()
        guard signature.count == 8, signature.hasPrefix("NRRD"),
              signature.suffix(4).utf8.allSatisfy({ (48...57).contains($0) }),
              let version = Int(signature.dropFirst(4)), (1...5).contains(version) else {
            throw Failure.invalidHeader("signature")
        }
        var fields: [String: String] = [:]
        while true {
            let entry = try line().trimmingCharacters(in: .whitespaces)
            if entry.isEmpty { break }
            if entry.hasPrefix("#") { continue }
            guard let colon = entry.firstIndex(of: ":") else { throw Failure.invalidHeader("entry") }
            let value = entry[entry.index(after: colon)...]
            if value.hasPrefix("=") { continue } // Free-form key/value metadata.
            let key = entry[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            guard !key.isEmpty, fields.updateValue(value.trimmingCharacters(in: .whitespaces), forKey: key) == nil else {
                throw Failure.invalidHeader(key)
            }
        }
        payloadOffset = offset

        guard let dimension = fields["dimension"].flatMap(Int.init) else { throw Failure.invalidHeader("dimension") }
        guard dimension == 3 else { throw Failure.unsupported("dimension \(dimension)") }
        guard let sizes = fields["sizes"]?.split(whereSeparator: \.isWhitespace), sizes.count == 3,
              let width = Int(sizes[0]), let height = Int(sizes[1]), let depth = Int(sizes[2]),
              min(width, height, depth) > 0 else { throw Failure.invalidHeader("sizes") }
        guard max(width, height, depth) <= 2_048, width * height * depth <= NRRDLoader.maximumVoxels else {
            throw Failure.size
        }
        dimensions = SIMD3(width, height, depth)
        voxelCount = width * height * depth

        guard let type = fields["type"] else { throw Failure.invalidHeader("type") }
        guard let scalar = Scalar(type) else { throw Failure.unsupported("scalar type \(type)") }
        self.scalar = scalar
        guard let encoding = fields["encoding"]?.lowercased() else { throw Failure.invalidHeader("encoding") }
        switch encoding {
        case "raw": self.encoding = .raw
        case "gzip", "gz": self.encoding = .gzip
        case "ascii", "text", "txt": self.encoding = .ascii
        case "hex": self.encoding = .hex
        default: throw Failure.unsupported("encoding \(encoding)")
        }
        switch fields["endian"]?.lowercased() {
        case nil, "little": littleEndian = true
        case "big": littleEndian = false
        default: throw Failure.invalidHeader("endian")
        }
        if fields["data file"] != nil || fields["datafile"] != nil { throw Failure.unsupported("detached data") }
        for key in ["line skip", "lineskip", "byte skip", "byteskip"] {
            if let value = fields[key] {
                guard let count = Int(value) else { throw Failure.invalidHeader(key) }
                guard count == 0 else { throw Failure.unsupported(key) }
            }
        }
        if let dimension = fields["space dimension"], Int(dimension) != 3 {
            throw Failure.unsupported("non-three-dimensional physical space")
        }
        if let kinds = fields["kinds"] {
            let axes = kinds.split(whereSeparator: \.isWhitespace)
            guard axes.count == 3 else { throw Failure.invalidHeader("kinds") }
            guard axes.allSatisfy({ ["domain", "space", "unknown"].contains($0.lowercased()) }) else {
                throw Failure.unsupported("non-spatial volume axes")
            }
        }
        space = fields["space"]
        if let name = space?.lowercased(), name.hasSuffix("-time") || ["rast", "last", "lpst"].contains(name) {
            throw Failure.unsupported("non-three-dimensional physical space")
        }
        units = try fields["space units"].map(Self.readUnits)
        if let text = fields["space origin"] {
            origin = try Self.readVectors(text, count: 1, field: "space origin")[0]
        } else {
            origin = .zero
        }
        if let text = fields["space directions"] {
            directions = try Self.readVectors(text, count: 3, field: "space directions")
        } else {
            let steps: [Double]
            if let text = fields["spacings"] {
                let tokens = text.split(whereSeparator: \.isWhitespace)
                guard tokens.count == 3 else { throw Failure.invalidHeader("spacings") }
                steps = try tokens.map { token in
                    guard let number = Double(token), number.isFinite || number.isNaN else {
                        throw Failure.invalidHeader("spacings")
                    }
                    return number.isNaN ? 1 : number
                }
            } else {
                steps = [1, 1, 1]
            }
            directions = [SIMD3(steps[0], 0, 0), SIMD3(0, steps[1], 0), SIMD3(0, 0, steps[2])]
        }
        let lengths = directions.map(simd_length)
        guard lengths.allSatisfy({ $0.isFinite && $0 > 0 }) else { throw Failure.invalidHeader("space directions") }
        let normalized = simd_double3x3(columns: (
            directions[0] / lengths[0], directions[1] / lengths[1], directions[2] / lengths[2]
        ))
        guard abs(simd_determinant(normalized)) > 1e-12 else { throw Failure.invalidHeader("space directions") }
    }

    private static func readVectors(_ text: String, count: Int, field: String) throws -> [SIMD3<Double>] {
        var remainder = text[...]
        var vectors: [SIMD3<Double>] = []
        while !remainder.isEmpty {
            remainder = remainder.drop(while: \.isWhitespace)
            if remainder.isEmpty { break }
            guard remainder.first == "(", let end = remainder.firstIndex(of: ")") else {
                throw Failure.invalidHeader(field)
            }
            let tokens = remainder.dropFirst().prefix(upTo: end).split(separator: ",", omittingEmptySubsequences: false)
            let values = tokens.compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard values.count == 3, tokens.count == 3, values.allSatisfy(\.isFinite), vectors.count < count else {
                throw Failure.invalidHeader(field)
            }
            vectors.append(SIMD3(values[0], values[1], values[2]))
            remainder = remainder[remainder.index(after: end)...]
        }
        guard vectors.count == count else { throw Failure.invalidHeader(field) }
        return vectors
    }

    private static func readUnits(_ text: String) throws -> [String] {
        var units: [String] = []
        var remainder = text[...].drop(while: \.isWhitespace)
        while !remainder.isEmpty {
            guard remainder.first == "\"", units.count < 3 else { throw Failure.invalidHeader("space units") }
            remainder = remainder.dropFirst()
            var unit = ""
            var closed = false
            while let character = remainder.popFirst() {
                if character == "\"" { closed = true; break }
                if character == "\\" {
                    guard let escaped = remainder.popFirst(), escaped == "\\" || escaped == "\"" else {
                        throw Failure.invalidHeader("space units")
                    }
                    unit.append(escaped)
                } else {
                    unit.append(character)
                }
            }
            guard closed else { throw Failure.invalidHeader("space units") }
            units.append(unit)
            remainder = remainder.drop(while: \.isWhitespace)
        }
        guard units.count == 3 else { throw Failure.invalidHeader("space units") }
        return units
    }

    func readBinary(_ bytes: UnsafeRawBufferPointer) throws -> NRRDLoader.Scalars {
        guard bytes.count == byteCount else { throw Failure.invalidPayload }
        func values<Word: FixedWidthInteger, Value>(_ type: Word.Type, transform: (Word) -> Value) -> [Value] {
            let stride = MemoryLayout<Word>.stride
            return (0..<voxelCount).map { index in
                let stored = bytes.loadUnaligned(fromByteOffset: index * stride, as: type)
                return transform(littleEndian ? Word(littleEndian: stored) : Word(bigEndian: stored))
            }
        }
        switch scalar {
        case .int8: return .int8(values(Int8.self) { $0 })
        case .uint8: return .uint8(values(UInt8.self) { $0 })
        case .int16: return .int16(values(Int16.self) { $0 })
        case .uint16: return .uint16(values(UInt16.self) { $0 })
        case .int32: return .int32(values(Int32.self) { $0 })
        case .uint32: return .uint32(values(UInt32.self) { $0 })
        case .float32: return .float32(values(UInt32.self, transform: Float.init(bitPattern:)))
        case .float64: return .float64(values(UInt64.self, transform: Double.init(bitPattern:)))
        }
    }

    func decodeHexBytes(_ bytes: UnsafeRawBufferPointer) -> Data? {
        guard bytes.count >= 2 * byteCount else { return nil }
        var output = Data()
        output.reserveCapacity(byteCount)
        var high: UInt8?
        for byte in bytes where !Self.whitespace(byte) {
            let nibble: UInt8
            switch byte {
            case 48...57: nibble = byte - 48
            case 65...70: nibble = byte - 55
            case 97...102: nibble = byte - 87
            default: return nil
            }
            if let first = high {
                guard output.count < byteCount else { return nil }
                output.append(first << 4 | nibble)
                high = nil
            } else {
                high = nibble
            }
        }
        return high == nil && output.count == byteCount ? output : nil
    }

    func readText(_ bytes: UnsafeRawBufferPointer, radix: Int) throws -> NRRDLoader.Scalars {
        guard bytes.count >= 2 * voxelCount - 1 else { throw Failure.invalidPayload }
        var offset = 0
        func values<Value>(_ parse: (String) -> Value?) throws -> [Value] {
            var result: [Value] = []
            result.reserveCapacity(voxelCount)
            for index in 0..<voxelCount {
                while offset < bytes.count, Self.whitespace(bytes[offset]) { offset += 1 }
                let start = offset
                while offset < bytes.count, !Self.whitespace(bytes[offset]) { offset += 1 }
                guard offset > start, offset - start <= 128,
                      let token = String(bytes: bytes[start..<offset], encoding: .ascii), let value = parse(token) else {
                    throw Failure.invalidSample(index)
                }
                result.append(value)
            }
            guard bytes[offset...].allSatisfy(Self.whitespace) else { throw Failure.invalidPayload }
            return result
        }
        func integer<T: FixedWidthInteger>(_ type: T.Type) throws -> [T] {
            try values { token in
                if radix == 16, token.lowercased().hasPrefix("0x") { return T(token.dropFirst(2), radix: radix) }
                return T(token, radix: radix)
            }
        }
        switch scalar {
        case .int8: return .int8(try integer(Int8.self))
        case .uint8: return .uint8(try integer(UInt8.self))
        case .int16: return .int16(try integer(Int16.self))
        case .uint16: return .uint16(try integer(UInt16.self))
        case .int32: return .int32(try integer(Int32.self))
        case .uint32: return .uint32(try integer(UInt32.self))
        case .float32: return .float32(try values(Float.init))
        case .float64: return .float64(try values(Double.init))
        }
    }

    private static func whitespace(_ byte: UInt8) -> Bool { byte == 32 || (9...13).contains(byte) }
}
