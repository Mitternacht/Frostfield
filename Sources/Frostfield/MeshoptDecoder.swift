import Foundation

/// Decodes buffer views compressed with `EXT_meshopt_compression` or
/// `KHR_meshopt_compression`. glTF import calls it for compressed buffer views.
///
/// The decoder follows Appendix A and B of the extension specifications. It checks every
/// read against the stream bounds and rejects reserved encodings, references to FIFO
/// entries that were never written, and bytes left between the decoded data and the tail.
/// Output is little-endian, with `count × byteStride` bytes.
/// [EXT_meshopt_compression](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_meshopt_compression)
public enum MeshoptDecoder {
    /// The extension whose bitstream rules apply.
    public enum Specification: String, Sendable, CaseIterable {
        /// Attribute streams use version 0 only, without the color filter.
        case ext = "EXT_meshopt_compression"
        /// Attribute streams use version 0 or 1, and the color filter is available.
        case khr = "KHR_meshopt_compression"
    }

    public enum Mode: String, Sendable, CaseIterable {
        /// Byte-wise delta coding for vertex and animation data.
        case attributes = "ATTRIBUTES"
        /// Triangle-list index coding. The count must be a multiple of three.
        case triangles = "TRIANGLES"
        /// Index sequence coding for other index data.
        case indices = "INDICES"
    }

    /// The transform applied to attribute data after decoding.
    public enum Filter: String, Sendable, CaseIterable {
        case none = "NONE"
        /// Unit vectors as 8-bit or 16-bit signed normalized components, with the fourth passed through.
        case octahedral = "OCTAHEDRAL"
        /// Unit quaternions as four 16-bit signed normalized components.
        case quaternion = "QUATERNION"
        /// 32-bit words with a signed 8-bit exponent and 24-bit mantissa, decoded to `Float`.
        case exponential = "EXPONENTIAL"
        /// YCoCg color as 8-bit or 16-bit unsigned normalized RGBA. `KHR_meshopt_compression` only.
        case color = "COLOR"
    }

    /// Decodes `count` elements of `byteStride` bytes each.
    ///
    /// Allocation is bounded by the declared size, so validate `count × byteStride`
    /// against your memory budget first. Attribute streams need a stride that is a
    /// multiple of 4 from 4 to 256; index streams use 2-byte or 4-byte indices, and
    /// 2-byte output rejects decoded indices above 65,535.
    public static func decode(
        _ source: Data, count: Int, byteStride: Int, mode: Mode, filter: Filter = .none,
        specification: Specification = .khr
    ) throws -> Data {
        try validateLayout(count: count, byteStride: byteStride, mode: mode, filter: filter, specification: specification)
        let (size, overflow) = count.multipliedReportingOverflow(by: byteStride)
        guard !overflow else { throw Failure.invalidLayout("count × byteStride overflows") }
        if mode == .attributes {
            try checkMinimumAttributeSize(source.count, count: count, byteStride: byteStride)
        }
        var output = Data(count: size)
        try source.withUnsafeBytes { input in
            try output.withUnsafeMutableBytes { destination in
                switch mode {
                case .attributes:
                    try decodeAttributes(input, count: count, stride: byteStride, specification: specification, into: destination)
                    try applyFilter(filter, to: destination, count: count, stride: byteStride)
                case .triangles:
                    try decodeTriangles(input, count: count, stride: byteStride, into: destination)
                case .indices:
                    try decodeSequence(input, count: count, stride: byteStride, into: destination)
                }
            }
        }
        return output
    }

    public enum Failure: LocalizedError, Equatable {
        /// The count, stride, mode, and filter combination is invalid for the specification.
        case invalidLayout(String)
        /// The first byte does not identify a bitstream this mode and specification accept.
        case invalidHeader(found: UInt8, expected: String)
        /// The stream ends before every declared element is decoded.
        case truncatedStream
        /// Bytes remain between the decoded elements and the tail block.
        case unusedBytes(Int)
        /// The stream uses a reserved encoding or references state that was never written.
        case malformedStream(String)
        /// An element's filter input is outside the range the filter defines.
        case invalidFilterInput(element: Int, String)

        public var errorDescription: String? {
            switch self {
            case .invalidLayout(let reason): "Invalid meshopt buffer view layout. \(reason)"
            case .invalidHeader(let found, let expected):
                "Meshopt stream header 0x\(String(found, radix: 16)) is invalid. Expected \(expected)."
            case .truncatedStream: "The meshopt stream ends before all declared elements are decoded."
            case .unusedBytes(let count): "The meshopt stream has \(count) unused bytes before its tail block."
            case .malformedStream(let reason): "Malformed meshopt stream. \(reason)"
            case .invalidFilterInput(let element, let reason): "Invalid meshopt filter input at element \(element). \(reason)"
            }
        }
    }

    static func validateLayout(
        count: Int, byteStride stride: Int, mode: Mode, filter: Filter, specification: Specification
    ) throws {
        guard count >= 0 else { throw Failure.invalidLayout("count must be nonnegative") }
        switch mode {
        case .attributes:
            guard (4...256).contains(stride), stride.isMultiple(of: 4) else {
                throw Failure.invalidLayout("ATTRIBUTES needs a byte stride that is a multiple of 4 from 4 to 256")
            }
        case .triangles, .indices:
            guard stride == 2 || stride == 4 else {
                throw Failure.invalidLayout("\(mode.rawValue) needs a byte stride of 2 or 4")
            }
            guard filter == .none else { throw Failure.invalidLayout("\(mode.rawValue) cannot use a filter") }
            if mode == .triangles, !count.isMultiple(of: 3) {
                throw Failure.invalidLayout("TRIANGLES needs a count that is a multiple of 3")
            }
        }
        switch filter {
        case .none: break
        case .octahedral:
            guard stride == 4 || stride == 8 else { throw Failure.invalidLayout("OCTAHEDRAL needs a byte stride of 4 or 8") }
        case .quaternion:
            guard stride == 8 else { throw Failure.invalidLayout("QUATERNION needs a byte stride of 8") }
        case .exponential:
            break
        case .color:
            guard specification == .khr else {
                throw Failure.invalidLayout("COLOR is available only in KHR_meshopt_compression")
            }
            guard stride == 4 || stride == 8 else { throw Failure.invalidLayout("COLOR needs a byte stride of 4 or 8") }
        }
    }

    /// Rejects streams too short to hold every block's headers before allocating the output.
    private static func checkMinimumAttributeSize(_ size: Int, count: Int, byteStride stride: Int) throws {
        guard count > 0 else { return }
        let blockLimit = maximumBlockElements(stride: stride)
        let blocks = (count + blockLimit - 1) / blockLimit
        // Version 1 needs at least one control byte per channel and block; version 0 needs
        // at least one header byte per byte position and block. Use the smaller bound.
        let (minimum, overflow) = blocks.multipliedReportingOverflow(by: stride / 4)
        guard !overflow, minimum <= size else { throw Failure.truncatedStream }
    }

    private static func maximumBlockElements(stride: Int) -> Int {
        min((8_192 / stride) & ~15, 256)
    }

    // MARK: Attributes

    private static func decodeAttributes(
        _ input: UnsafeRawBufferPointer, count: Int, stride: Int, specification: Specification,
        into output: UnsafeMutableRawBufferPointer
    ) throws {
        guard let header = input.first else { throw Failure.truncatedStream }
        let version: Int
        switch header {
        case 0xA0: version = 0
        case 0xA1 where specification == .khr: version = 1
        default:
            throw Failure.invalidHeader(found: header, expected: specification == .khr ? "0xa0 or 0xa1" : "0xa0")
        }
        let channelCount = stride / 4
        let controlSize = version == 0 ? 0 : channelCount
        let tailSize = stride + controlSize
        let tailPadded = max(tailSize, version == 0 ? 32 : 24)
        guard input.count - 1 >= tailPadded else { throw Failure.truncatedStream }
        let limit = input.count - tailPadded
        let tail = input.count - tailSize

        var last = [UInt8](input[tail..<(tail + stride)])
        var channels = [UInt8](repeating: 0, count: channelCount)
        if version == 1 {
            for channel in 0..<channelCount {
                let value = input[tail + stride + channel]
                guard value & 0x0F <= 2, value & 0x0F == 2 || value >> 4 == 0 else {
                    throw Failure.malformedStream("channel \(channel) has reserved mode 0x\(String(value, radix: 16))")
                }
                channels[channel] = value
            }
        }

        let blockLimit = maximumBlockElements(stride: stride)
        var planes = [UInt8](repeating: 0, count: 4 * planeStride)
        var reader = Reader(input: input, position: 1, limit: limit)
        var start = 0
        while start < count {
            let blockCount = min(blockLimit, count - start)
            let aligned = (blockCount + 15) & ~15
            let control = try reader.skip(controlSize)
            for channel in 0..<channelCount {
                let controlByte = version == 0 ? 0 : input[control + channel]
                for lane in 0..<4 {
                    let base = lane * planeStride
                    let controlMode = Int(controlByte >> (lane * 2)) & 3
                    switch (version, controlMode) {
                    case (1, 3):
                        let literal = try reader.skip(blockCount)
                        for index in 0..<blockCount { planes[base + index] = input[literal + index] }
                    case (1, 2):
                        for index in 0..<blockCount { planes[base + index] = 0 }
                    default:
                        let widths = version == 0 ? versionZeroWidths : (controlMode == 0 ? lowWidths : highWidths)
                        try decodeBytes(&reader, into: &planes, base: base, groups: aligned / 16, widths: widths)
                    }
                }
                try applyDeltas(
                    planes, channel: channel, mode: channels[channel], count: blockCount, first: start,
                    stride: stride, last: &last, into: output
                )
            }
            start += blockCount
        }
        guard reader.position == limit else { throw Failure.unusedBytes(limit - reader.position) }
    }

    private static let planeStride = 256
    private static let versionZeroWidths = [0, 2, 4, 8]
    private static let lowWidths = [0, 1, 2, 4]
    private static let highWidths = [1, 2, 4, 8]

    private static func decodeBytes(
        _ reader: inout Reader, into planes: inout [UInt8], base: Int, groups: Int, widths: [Int]
    ) throws {
        let header = try reader.skip((groups + 3) / 4)
        for group in 0..<groups {
            let code = Int(reader.input[header + group / 4] >> ((group % 4) * 2)) & 3
            let bits = widths[code]
            let target = base + group * 16
            switch bits {
            case 0:
                for index in 0..<16 { planes[target + index] = 0 }
            case 8:
                let bytes = try reader.skip(16)
                for index in 0..<16 { planes[target + index] = reader.input[bytes + index] }
            default:
                let packed = try reader.skip(bits * 2)
                let sentinel = (1 << bits) - 1
                let perByte = 8 / bits
                for index in 0..<16 {
                    let byte = Int(reader.input[packed + index / perByte])
                    // One-bit deltas are packed from the least significant bit; wider ones from the most.
                    let shift = bits == 1 ? index % 8 : 8 - bits * (index % perByte + 1)
                    let value = (byte >> shift) & sentinel
                    planes[target + index] = try value == sentinel ? reader.byte() : UInt8(value)
                }
            }
        }
    }

    private static func applyDeltas(
        _ planes: [UInt8], channel: Int, mode: UInt8, count: Int, first: Int, stride: Int,
        last: inout [UInt8], into output: UnsafeMutableRawBufferPointer
    ) throws {
        let offset = channel * 4
        switch mode & 0x0F {
        case 0:
            for lane in 0..<4 {
                var previous = last[offset + lane]
                for index in 0..<count {
                    let encoded = planes[lane * planeStride + index]
                    let value = ((0 &- (encoded & 1)) ^ (encoded >> 1)) &+ previous
                    output[(first + index) * stride + offset + lane] = value
                    previous = value
                }
                last[offset + lane] = previous
            }
        case 1:
            for half in 0..<2 {
                let lane = offset + half * 2
                var previous = UInt16(last[lane]) | UInt16(last[lane + 1]) << 8
                for index in 0..<count {
                    let encoded = UInt16(planes[half * 2 * planeStride + index])
                        | UInt16(planes[(half * 2 + 1) * planeStride + index]) << 8
                    let value = ((0 &- (encoded & 1)) ^ (encoded >> 1)) &+ previous
                    let target = (first + index) * stride + lane
                    output[target] = UInt8(truncatingIfNeeded: value)
                    output[target + 1] = UInt8(truncatingIfNeeded: value >> 8)
                    previous = value
                }
                last[lane] = UInt8(truncatingIfNeeded: previous)
                last[lane + 1] = UInt8(truncatingIfNeeded: previous >> 8)
            }
        default:
            let rotation = UInt32((32 - Int(mode >> 4)) & 31)
            var previous = (0..<4).reduce(UInt32(0)) { $0 | UInt32(last[offset + $1]) << (8 * UInt32($1)) }
            for index in 0..<count {
                let encoded = (0..<4).reduce(UInt32(0)) {
                    $0 | UInt32(planes[$1 * planeStride + index]) << (8 * UInt32($1))
                }
                let rotated = (encoded << rotation) | (encoded >> ((32 - rotation) & 31))
                let value = rotated ^ previous
                let target = (first + index) * stride + offset
                for byte in 0..<4 { output[target + byte] = UInt8(truncatingIfNeeded: value >> (8 * UInt32(byte))) }
                previous = value
            }
            for byte in 0..<4 { last[offset + byte] = UInt8(truncatingIfNeeded: previous >> (8 * UInt32(byte))) }
        }
    }

    // MARK: Indices

    private static func decodeTriangles(
        _ input: UnsafeRawBufferPointer, count: Int, stride: Int, into output: UnsafeMutableRawBufferPointer
    ) throws {
        let triangles = count / 3
        guard input.count >= 1 + triangles + 16 else { throw Failure.truncatedStream }
        guard input[0] == 0xE1 else { throw Failure.invalidHeader(found: input[0], expected: "0xe1") }
        let table = input.count - 16
        guard input[table + 14] == 0, input[table + 15] == 0 else {
            throw Failure.malformedStream("the last two code table bytes must be zero")
        }
        for index in 0..<16 where input[table + index] & 0x0F == 0x0F || input[table + index] >> 4 == 0x0F {
            throw Failure.malformedStream("code table entry \(index) contains the reserved value 0xf")
        }

        var edges = FIFO<(UInt32, UInt32)>(filler: (0, 0))
        var vertices = FIFO<UInt32>(filler: 0)
        var next: UInt32 = 0
        var last: UInt32 = 0
        var data = Reader(input: input, position: 1 + triangles, limit: table)

        func free(_ data: inout Reader) throws -> UInt32 {
            let value = try data.varint()
            last = last &+ ((value >> 1) ^ (0 &- (value & 1)))
            return last
        }

        for triangle in 0..<triangles {
            let code = input[1 + triangle]
            let a: UInt32, b: UInt32, c: UInt32
            if code < 0xF0 {
                (a, b) = try edges.recent(Int(code >> 4), name: "edge")
                let vertex = Int(code & 0x0F)
                switch vertex {
                case 0:
                    c = next
                    next &+= 1
                    vertices.push(c)
                case 1..<13:
                    c = try vertices.recent(vertex, name: "vertex")
                case 13, 14:
                    last = vertex == 13 ? last &- 1 : last &+ 1
                    c = last
                    vertices.push(c)
                default:
                    c = try free(&data)
                    vertices.push(c)
                }
                edges.push((c, b))
                edges.push((a, c))
            } else {
                let auxiliary: UInt8
                let explicitFirst = code == 0xFF
                if code < 0xFE {
                    auxiliary = input[table + Int(code & 0x0F)]
                } else {
                    auxiliary = try data.byte()
                    if auxiliary == 0 { next = 0 }
                }
                let second = Int(auxiliary >> 4)
                let third = Int(auxiliary & 0x0F)
                // Resolve FIFO references before this triangle's pushes, as the encoder does.
                func fetch(_ code: Int) throws -> UInt32? {
                    switch code {
                    case 0:
                        defer { next &+= 1 }
                        return next
                    case 15: return nil
                    default: return try vertices.recent(code - 1, name: "vertex")
                    }
                }
                var first: UInt32 = 0
                if !explicitFirst {
                    first = next
                    next &+= 1
                }
                let fetchedSecond = try fetch(second)
                let fetchedThird = try fetch(third)
                a = try explicitFirst ? free(&data) : first
                if let fetchedSecond { b = fetchedSecond } else { b = try free(&data) }
                if let fetchedThird { c = fetchedThird } else { c = try free(&data) }
                vertices.push(a)
                if second == 0 || second == 15 { vertices.push(b) }
                if third == 0 || third == 15 { vertices.push(c) }
                edges.push((b, a))
                edges.push((c, b))
                edges.push((a, c))
            }
            try write(a, at: triangle * 3, stride: stride, into: output)
            try write(b, at: triangle * 3 + 1, stride: stride, into: output)
            try write(c, at: triangle * 3 + 2, stride: stride, into: output)
        }
        guard data.position == table else { throw Failure.unusedBytes(table - data.position) }
    }

    private static func decodeSequence(
        _ input: UnsafeRawBufferPointer, count: Int, stride: Int, into output: UnsafeMutableRawBufferPointer
    ) throws {
        guard input.count >= 1 + count + 4 else { throw Failure.truncatedStream }
        guard input[0] == 0xD1 else { throw Failure.invalidHeader(found: input[0], expected: "0xd1") }
        let end = input.count - 4
        var data = Reader(input: input, position: 1, limit: end)
        var baselines: (UInt32, UInt32) = (0, 0)
        for index in 0..<count {
            let encoded = try data.varint()
            let delta = encoded >> 1
            let value = (delta >> 1) ^ (0 &- (delta & 1))
            let decoded: UInt32
            if encoded & 1 == 0 {
                baselines.0 = baselines.0 &+ value
                decoded = baselines.0
            } else {
                baselines.1 = baselines.1 &+ value
                decoded = baselines.1
            }
            try write(decoded, at: index, stride: stride, into: output)
        }
        guard data.position == end else { throw Failure.unusedBytes(end - data.position) }
    }

    private static func write(
        _ value: UInt32, at index: Int, stride: Int, into output: UnsafeMutableRawBufferPointer
    ) throws {
        if stride == 2 {
            guard let narrow = UInt16(exactly: value) else {
                throw Failure.malformedStream("index \(index) decodes to \(value), which does not fit 16 bits")
            }
            output.storeBytes(of: narrow.littleEndian, toByteOffset: index * 2, as: UInt16.self)
        } else {
            output.storeBytes(of: value.littleEndian, toByteOffset: index * 4, as: UInt32.self)
        }
    }

    /// A 16-entry ring that tracks how many entries have been written.
    private struct FIFO<Element> {
        private var storage: [Element]
        private var offset = 0
        private var filled = 0

        init(filler: Element) { storage = Array(repeating: filler, count: 16) }

        mutating func push(_ value: Element) {
            storage[offset] = value
            offset = (offset + 1) & 15
            filled = min(filled + 1, 16)
        }

        /// Zero is the most recently pushed entry.
        func recent(_ index: Int, name: String) throws -> Element {
            guard index < filled else {
                throw Failure.malformedStream("a triangle references \(name) FIFO entry \(index) before it is written")
            }
            return storage[(offset - 1 - index) & 15]
        }
    }

    private struct Reader {
        let input: UnsafeRawBufferPointer
        var position: Int
        let limit: Int

        /// Returns the start of the next `count` bytes and advances past them.
        mutating func skip(_ count: Int) throws -> Int {
            guard limit - position >= count else { throw Failure.truncatedStream }
            defer { position += count }
            return position
        }

        mutating func byte() throws -> UInt8 {
            input[try skip(1)]
        }

        /// Reads an unsigned LEB128 value of one to five bytes, keeping its low 32 bits.
        mutating func varint() throws -> UInt32 {
            var result: UInt32 = 0
            for group in 0..<5 {
                let byte = try self.byte()
                result |= UInt32(byte & 0x7F) << UInt32(7 * group)
                if byte < 0x80 { return result }
            }
            throw Failure.malformedStream("a variable-length integer exceeds five bytes")
        }
    }

    // MARK: Filters

    private static func applyFilter(
        _ filter: Filter, to data: UnsafeMutableRawBufferPointer, count: Int, stride: Int
    ) throws {
        switch filter {
        case .none: break
        case .octahedral:
            if stride == 4 {
                try octahedral(data, count: count, width: 1)
            } else {
                try octahedral(data, count: count, width: 2)
            }
        case .quaternion:
            try quaternion(data, count: count)
        case .exponential:
            try exponential(data, words: count * stride / 4, stride: stride)
        case .color:
            try color(data, count: count, width: stride / 4)
        }
    }

    private static func signed(_ data: UnsafeMutableRawBufferPointer, _ offset: Int, width: Int) -> Int32 {
        width == 1
            ? Int32(Int8(bitPattern: data[offset]))
            : Int32(Int16(bitPattern: UInt16(data[offset]) | UInt16(data[offset + 1]) << 8))
    }

    private static func unsigned(_ data: UnsafeMutableRawBufferPointer, _ offset: Int, width: Int) -> Int32 {
        width == 1 ? Int32(data[offset]) : Int32(UInt16(data[offset]) | UInt16(data[offset + 1]) << 8)
    }

    private static func store(_ value: Int32, _ data: UnsafeMutableRawBufferPointer, _ offset: Int, width: Int) {
        data[offset] = UInt8(truncatingIfNeeded: value)
        if width == 2 { data[offset + 1] = UInt8(truncatingIfNeeded: value >> 8) }
    }

    /// Rounds half away from zero after truncating toward zero, as the reference decoder does.
    private static func rounded(_ value: Float) -> Int32 {
        Int32(value + (value >= 0 ? 0.5 : -0.5))
    }

    /// True for `(1 << (k - 1)) - 1` with `k` in the given bit range.
    private static func encodesOne(_ value: Int32, bits: ClosedRange<Int32>) -> Bool {
        bits.contains { value == (1 << ($0 - 1)) - 1 }
    }

    private static func octahedral(_ data: UnsafeMutableRawBufferPointer, count: Int, width: Int) throws {
        let maximum = Float((1 << (8 * width - 1)) - 1)
        for element in 0..<count {
            let base = element * 4 * width
            let x = signed(data, base, width: width)
            let y = signed(data, base + width, width: width)
            let one = signed(data, base + 2 * width, width: width)
            guard encodesOne(one, bits: 2...Int32(8 * width)) else {
                throw Failure.invalidFilterInput(element: element, "the third component must encode 1.0 for 2 to \(8 * width) bits")
            }
            guard abs(x) <= one, abs(y) <= one else {
                throw Failure.invalidFilterInput(element: element, "octahedral coordinates exceed the encoded range")
            }
            var fx = Float(x), fy = Float(y)
            let fz = Float(one) - abs(fx) - abs(fy)
            let fold = fz >= 0 ? 0 : fz
            fx += fx >= 0 ? fold : -fold
            fy += fy >= 0 ? fold : -fold
            let scale = maximum / (fx * fx + fy * fy + fz * fz).squareRoot()
            store(rounded(fx * scale), data, base, width: width)
            store(rounded(fy * scale), data, base + width, width: width)
            store(rounded(fz * scale), data, base + 2 * width, width: width)
        }
    }

    private static func quaternion(_ data: UnsafeMutableRawBufferPointer, count: Int) throws {
        let scale: Float = 32_767 / Float(2).squareRoot()
        for element in 0..<count {
            let base = element * 8
            let last = signed(data, base + 6, width: 2)
            let one = last | 3
            guard encodesOne(one, bits: 4...16) else {
                throw Failure.invalidFilterInput(element: element, "the fourth component must encode 1.0 for 4 to 16 bits")
            }
            let x = signed(data, base, width: 2), y = signed(data, base + 2, width: 2), z = signed(data, base + 4, width: 2)
            guard abs(x) <= one, abs(y) <= one, abs(z) <= one else {
                throw Failure.invalidFilterInput(element: element, "quaternion components exceed the encoded range")
            }
            let s = Float(one)
            let fx = Float(x), fy = Float(y), fz = Float(z)
            let squared = s * s * 2 - fx * fx - fy * fy - fz * fz
            let w = (squared >= 0 ? squared : 0).squareRoot()
            let factor = scale / s
            let largest = Int(last & 3)
            store(rounded(fx * factor), data, base + ((largest + 1) & 3) * 2, width: 2)
            store(rounded(fy * factor), data, base + ((largest + 2) & 3) * 2, width: 2)
            store(rounded(fz * factor), data, base + ((largest + 3) & 3) * 2, width: 2)
            store(Int32(w * factor + 0.5), data, base + largest * 2, width: 2)
        }
    }

    private static func exponential(_ data: UnsafeMutableRawBufferPointer, words: Int, stride: Int) throws {
        for word in 0..<words {
            let encoded = data.loadUnaligned(fromByteOffset: word * 4, as: UInt32.self).littleEndian
            let exponent = Int32(bitPattern: encoded) >> 24
            guard (-100...100).contains(exponent) else {
                throw Failure.invalidFilterInput(element: word * 4 / stride, "exponent \(exponent) is outside -100 to 100")
            }
            let mantissa = Int32(bitPattern: encoded << 8) >> 8
            let value = Float(bitPattern: UInt32(exponent + 127) << 23) * Float(mantissa)
            data.storeBytes(of: value.bitPattern.littleEndian, toByteOffset: word * 4, as: UInt32.self)
        }
    }

    private static func color(_ data: UnsafeMutableRawBufferPointer, count: Int, width: Int) throws {
        let maximum = Float((1 << (8 * width)) - 1)
        for element in 0..<count {
            let base = element * 4 * width
            let alpha = unsigned(data, base + 3 * width, width: width)
            guard alpha >= 2 else {
                throw Failure.invalidFilterInput(element: element, "the alpha component must mark at least 2 bits")
            }
            let bits = 32 - alpha.leadingZeroBitCount
            let range = Int32((1 << bits) - 1)
            let limit = Int32((1 << (bits - 1)) - 1)
            let luma = unsigned(data, base, width: width)
            let orange = signed(data, base + width, width: width)
            let green = signed(data, base + 2 * width, width: width)
            guard luma <= range, abs(orange) <= limit, abs(green) <= limit else {
                throw Failure.invalidFilterInput(element: element, "YCoCg components exceed \(bits) bits")
            }
            let red = luma + orange - green, greenValue = luma + green, blue = luma - orange - green
            guard (0...range).contains(red), (0...range).contains(greenValue), (0...range).contains(blue) else {
                throw Failure.invalidFilterInput(element: element, "the reconstructed RGB values exceed \(bits) bits")
            }
            let expandedAlpha = ((alpha << 1) & range) | (alpha & 1)
            let factor = maximum / Float(range)
            store(Int32(Float(red) * factor + 0.5), data, base, width: width)
            store(Int32(Float(greenValue) * factor + 0.5), data, base + width, width: width)
            store(Int32(Float(blue) * factor + 0.5), data, base + 2 * width, width: width)
            store(Int32(Float(expandedAlpha) * factor + 0.5), data, base + 3 * width, width: width)
        }
    }
}
