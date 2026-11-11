import Foundation
import Metal
import simd
import UniformTypeIdentifiers

/// A 3D color lookup table: a lattice of output RGB values over an input RGB cube.
///
/// Values are stored with red varying fastest, then green, then blue, as in the
/// Adobe/Resolve `.cube` format. Inputs inside `domainMinimum...domainMaximum`
/// map to lattice coordinates; inputs outside clamp to the domain.
public struct ColorLookupTable: Hashable, Sendable {
    /// The largest lattice edge accepted. 65 covers the common 17, 33 and 65 sizes.
    public static let maximumSize = 65

    public enum Interpolation: Sendable, Hashable {
        /// Hardware trilinear filtering of the eight surrounding lattice points.
        case trilinear
        /// Interpolation within one of six tetrahedra of the lattice cell, using four
        /// points. It preserves the neutral axis and is the usual grading-tool choice.
        case tetrahedral
    }

    public let size: Int
    public let title: String?
    public let domainMinimum: SIMD3<Float>
    public let domainMaximum: SIMD3<Float>
    /// `size³` output colors, red index fastest, then green, then blue.
    public let values: [SIMD3<Float>]

    /// Validates a lattice of 2 to 65 points per axis with finite values and a domain whose
    /// maximum exceeds its minimum on every axis.
    public init(
        size: Int, values: [SIMD3<Float>], domainMinimum: SIMD3<Float> = .zero, domainMaximum: SIMD3<Float> = .one,
        title: String? = nil
    ) throws {
        guard (2...Self.maximumSize).contains(size) else { throw Failure.size(size) }
        guard values.count == size * size * size else { throw Failure.valueCount(expected: size * size * size, actual: values.count) }
        guard values.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { throw Failure.nonfiniteValue }
        guard [domainMinimum.x, domainMinimum.y, domainMinimum.z, domainMaximum.x, domainMaximum.y, domainMaximum.z]
                .allSatisfy(\.isFinite),
              domainMaximum.x > domainMinimum.x, domainMaximum.y > domainMinimum.y, domainMaximum.z > domainMinimum.z else {
            throw Failure.domain
        }
        self.size = size
        self.values = values
        self.domainMinimum = domainMinimum
        self.domainMaximum = domainMaximum
        self.title = title
    }

    /// The table that returns its input on the unit domain.
    public static func identity(size: Int = 33) throws -> ColorLookupTable {
        guard (2...maximumSize).contains(size) else { throw Failure.size(size) }
        let scale = 1 / Float(size - 1)
        var values: [SIMD3<Float>] = []
        values.reserveCapacity(size * size * size)
        for blue in 0..<size {
            for green in 0..<size {
                for red in 0..<size { values.append(SIMD3(Float(red), Float(green), Float(blue)) * scale) }
            }
        }
        return try ColorLookupTable(size: size, values: values)
    }

    /// Samples every lattice point of a Swift function of input RGB on the unit domain.
    public init(size: Int, title: String? = nil, transform: (SIMD3<Float>) -> SIMD3<Float>) throws {
        guard (2...Self.maximumSize).contains(size) else { throw Failure.size(size) }
        let scale = 1 / Float(size - 1)
        var values: [SIMD3<Float>] = []
        values.reserveCapacity(size * size * size)
        for blue in 0..<size {
            for green in 0..<size {
                for red in 0..<size { values.append(transform(SIMD3(Float(red), Float(green), Float(blue)) * scale)) }
            }
        }
        try self.init(size: size, values: values, title: title)
    }

    /// Reads a `.cube`, `.3dl`, or 3D lookup-strip image, chosen by its extension.
    public init(contentsOf url: URL) throws {
        let ext = url.pathExtension.lowercased()
        switch ext {
        case "cube", "3dl":
            let data = try Data(contentsOf: url)
            guard data.count <= 64 << 20 else { throw Failure.syntax(line: 0, "The file is larger than 64 MB.") }
            guard let text = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) else {
                throw Failure.syntax(line: 0, "The file is not text.")
            }
            if ext == "cube" {
                try self.init(cube: text)
            } else {
                try self.init(threeDL: text)
            }
        default:
            guard UTType(filenameExtension: ext)?.conforms(to: .image) == true else {
                throw Failure.unsupportedFormat(url.pathExtension)
            }
            try self.init(imageContentsOf: url)
        }
    }

    /// Parses an Adobe Cube LUT Specification 1.0 or DaVinci Resolve 3D `.cube` table.
    ///
    /// Accepts `TITLE`, `LUT_3D_SIZE`, `DOMAIN_MIN`, `DOMAIN_MAX` and Resolve's
    /// `LUT_3D_INPUT_RANGE`, `#` comments, and `size³` data lines of three numbers
    /// with red varying fastest. Tables with a 1D section are rejected.
    public init(cube text: String) throws {
        var size: Int?
        var title: String?
        var minimum = SIMD3<Float>(repeating: 0)
        var maximum = SIMD3<Float>(repeating: 1)
        var values: [SIMD3<Float>] = []
        for (number, rawLine) in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).enumerated() {
            let line = rawLine.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? ""
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            let fields = trimmed.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
            let keyword = fields[0].uppercased()
            if let first = fields[0].first, first.isLetter {
                guard values.isEmpty else { throw Failure.syntax(line: number + 1, "Keywords must precede the table data.") }
                switch keyword {
                case "TITLE":
                    title = Self.quoted(trimmed.dropFirst(5))
                case "LUT_3D_SIZE":
                    guard fields.count == 2, let value = Int(fields[1]) else {
                        throw Failure.syntax(line: number + 1, "LUT_3D_SIZE needs one integer.")
                    }
                    guard (2...Self.maximumSize).contains(value) else { throw Failure.size(value) }
                    size = value
                case "LUT_1D_SIZE", "LUT_1D_INPUT_RANGE":
                    throw Failure.oneDimensional
                case "DOMAIN_MIN", "DOMAIN_MAX":
                    let triple = try Self.triple(fields.dropFirst(), line: number + 1)
                    if keyword == "DOMAIN_MIN" { minimum = triple } else { maximum = triple }
                case "LUT_3D_INPUT_RANGE":
                    guard fields.count == 3, let low = Float(fields[1]), let high = Float(fields[2]) else {
                        throw Failure.syntax(line: number + 1, "LUT_3D_INPUT_RANGE needs two numbers.")
                    }
                    minimum = SIMD3(repeating: low)
                    maximum = SIMD3(repeating: high)
                default:
                    throw Failure.syntax(line: number + 1, "Unknown keyword \(fields[0]).")
                }
                continue
            }
            guard let size else { throw Failure.syntax(line: number + 1, "LUT_3D_SIZE must precede the table data.") }
            guard values.count < size * size * size else {
                throw Failure.valueCount(expected: size * size * size, actual: values.count + 1)
            }
            values.append(try Self.triple(fields[...], line: number + 1))
        }
        guard let size else { throw Failure.syntax(line: 0, "The file has no LUT_3D_SIZE.") }
        try self.init(size: size, values: values, domainMinimum: minimum, domainMaximum: maximum, title: title)
    }

    /// Parses an Autodesk Lustre/Flame `.3dl` table.
    ///
    /// The first numeric line lists the input lattice positions, one per axis point;
    /// the table size is their count. The positions must start at 0 and be evenly
    /// spaced, except that the last may be one code short of the full step, as in the
    /// common `0 64 … 960 1023`; other lattices are rejected because the table is
    /// applied on a uniform grid. The following `size³` lines hold integer RGB with
    /// blue varying fastest. The output bit depth comes from a `Mesh input output`
    /// header when present, else from the smallest of 8, 10, 12, 14 or 16 bits that
    /// holds the largest value. Values are normalized to 0...1 and reordered red-fastest.
    public init(threeDL text: String) throws {
        var lattice: [Float]?
        var outputBits: Int?
        var rows: [SIMD3<Float>] = []
        for (number, rawLine) in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).enumerated() {
            let line = rawLine.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? ""
            let fields = line.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\r" }).map(String.init)
            guard let first = fields.first else { continue }
            if first.uppercased() == "3DMESH" { continue }
            if first.lowercased() == "mesh" {
                guard fields.count == 3, let bits = Int(fields[2]), (1...32).contains(bits) else {
                    throw Failure.syntax(line: number + 1, "A Mesh header needs input and output bit depths.")
                }
                outputBits = bits
                continue
            }
            if first.uppercased() == "LUT8" || first.uppercased() == "GAMMA" { continue }
            let numbers = try fields.map { field -> Float in
                guard let value = Float(field), value.isFinite, value >= 0 else {
                    throw Failure.syntax(line: number + 1, "Expected nonnegative numbers.")
                }
                return value
            }
            if lattice == nil {
                guard (2...Self.maximumSize).contains(numbers.count) else { throw Failure.size(numbers.count) }
                lattice = numbers
                continue
            }
            guard numbers.count == 3 else { throw Failure.syntax(line: number + 1, "Expected three values per line.") }
            rows.append(SIMD3(numbers[0], numbers[1], numbers[2]))
        }
        guard let lattice else { throw Failure.syntax(line: 0, "The file has no input lattice line.") }
        guard Self.isUniform(lattice) else { throw Failure.nonuniformLattice }
        let size = lattice.count
        guard rows.count == size * size * size else {
            throw Failure.valueCount(expected: size * size * size, actual: rows.count)
        }
        let largest = rows.reduce(Float(0)) { max($0, max($1.x, max($1.y, $1.z))) }
        let bits = outputBits ?? [8, 10, 12, 14, 16].first { Float((1 << $0) - 1) >= largest } ?? 16
        let maximum = Float((UInt64(1) << UInt64(bits)) - 1)
        var values = [SIMD3<Float>](repeating: .zero, count: rows.count)
        // Source order: red slowest, blue fastest. Destination: red fastest.
        for (index, row) in rows.enumerated() {
            let blue = index % size
            let green = (index / size) % size
            let red = index / (size * size)
            values[red + size * (green + size * blue)] = row / maximum
        }
        try self.init(size: size, values: values)
    }

    /// Whether `.3dl` lattice positions are `0, s, 2s, …`, allowing the last to be
    /// `(n − 1) · s − 1` for files whose top position is the largest integer code.
    private static func isUniform(_ lattice: [Float]) -> Bool {
        let step = Double(lattice[1]) - Double(lattice[0])
        guard lattice[0] == 0, step > 0 else { return false }
        let tolerance = 1e-4 * step
        for (index, position) in lattice.enumerated() {
            let expected = Double(index) * step
            let value = Double(position)
            if abs(value - expected) <= tolerance { continue }
            guard index == lattice.count - 1, step > 1, abs(value - (expected - 1)) <= tolerance else { return false }
        }
        return true
    }

    /// The table's output for an input color, evaluated on the CPU in single precision
    /// with the same lattice addressing and interpolation as the GPU pass.
    public func lookup(_ color: SIMD3<Float>, interpolation: Interpolation = .tetrahedral) -> SIMD3<Float> {
        let extent = Float(size - 1)
        var position = (color - domainMinimum) / (domainMaximum - domainMinimum)
        position = simd_clamp(position, .zero, .one) * extent
        let base = simd_min(position.rounded(.down), SIMD3(repeating: extent - 1))
        let f = position - base
        let b = SIMD3<Int>(Int(base.x), Int(base.y), Int(base.z))
        func value(_ dx: Int, _ dy: Int, _ dz: Int) -> SIMD3<Float> {
            values[(b.x + dx) + size * ((b.y + dy) + size * (b.z + dz))]
        }
        switch interpolation {
        case .trilinear:
            let x00 = value(0, 0, 0) + (value(1, 0, 0) - value(0, 0, 0)) * f.x
            let x10 = value(0, 1, 0) + (value(1, 1, 0) - value(0, 1, 0)) * f.x
            let x01 = value(0, 0, 1) + (value(1, 0, 1) - value(0, 0, 1)) * f.x
            let x11 = value(0, 1, 1) + (value(1, 1, 1) - value(0, 1, 1)) * f.x
            let y0 = x00 + (x10 - x00) * f.y
            let y1 = x01 + (x11 - x01) * f.y
            return y0 + (y1 - y0) * f.z
        case .tetrahedral:
            let c000 = value(0, 0, 0)
            let c111 = value(1, 1, 1)
            if f.x >= f.y {
                if f.y >= f.z {
                    return (1 - f.x) * c000 + (f.x - f.y) * value(1, 0, 0) + (f.y - f.z) * value(1, 1, 0) + f.z * c111
                } else if f.x >= f.z {
                    return (1 - f.x) * c000 + (f.x - f.z) * value(1, 0, 0) + (f.z - f.y) * value(1, 0, 1) + f.y * c111
                } else {
                    return (1 - f.z) * c000 + (f.z - f.x) * value(0, 0, 1) + (f.x - f.y) * value(1, 0, 1) + f.y * c111
                }
            } else {
                if f.z >= f.y {
                    return (1 - f.z) * c000 + (f.z - f.y) * value(0, 0, 1) + (f.y - f.x) * value(0, 1, 1) + f.x * c111
                } else if f.z >= f.x {
                    return (1 - f.y) * c000 + (f.y - f.z) * value(0, 1, 0) + (f.z - f.x) * value(0, 1, 1) + f.x * c111
                } else {
                    return (1 - f.y) * c000 + (f.y - f.x) * value(0, 1, 0) + (f.x - f.z) * value(1, 1, 0) + f.z * c111
                }
            }
        }
    }

    private static func quoted(_ text: Substring) -> String {
        var value = text.trimmingCharacters(in: .whitespaces)
        if value.hasPrefix("\"") { value.removeFirst() }
        if value.hasSuffix("\"") { value.removeLast() }
        return value
    }

    private static func triple(_ fields: ArraySlice<String>, line: Int) throws -> SIMD3<Float> {
        let numbers = fields.compactMap(Float.init)
        guard fields.count == 3, numbers.count == 3, numbers.allSatisfy(\.isFinite) else {
            throw Failure.syntax(line: line, "Expected three finite numbers.")
        }
        return SIMD3(numbers[0], numbers[1], numbers[2])
    }

    public enum Failure: LocalizedError, Equatable {
        case size(Int)
        case valueCount(expected: Int, actual: Int)
        case nonfiniteValue
        case domain
        case oneDimensional
        case unsupportedFormat(String)
        case syntax(line: Int, String)
        case nonuniformLattice
        case imageInputTooLarge
        case imageDecode
        case imageLayout(width: Int, height: Int)

        public var errorDescription: String? {
            switch self {
            case .size(let size): "A color lookup table needs 2 to \(ColorLookupTable.maximumSize) points per axis, not \(size)."
            case .valueCount(let expected, let actual): "The lookup table needs \(expected) values but has \(actual)."
            case .nonfiniteValue: "Lookup table values must be finite."
            case .domain: "The lookup table domain maximum must exceed its minimum on every axis."
            case .oneDimensional: "1D lookup tables are not supported; convert the curve to a 3D table."
            case .unsupportedFormat(let ext): "Unsupported lookup table format \".\(ext)\"; use .cube, .3dl, or a 3D lookup-strip image."
            case .syntax(let line, let detail): line > 0 ? "Lookup table line \(line): \(detail)" : detail
            case .imageInputTooLarge: "The encoded lookup image exceeds the 64 MiB limit."
            case .imageDecode: "Could not decode the lookup image as numeric 8-bit color."
            case .imageLayout(let width, let height):
                "A lookup image must have N × N² or N² × N pixels. Got \(width) × \(height)."
            case .nonuniformLattice:
                "The .3dl input lattice must start at 0 and be evenly spaced; only the last position may be one code short."
            }
        }
    }
}

/// Applies a `ColorLookupTable` to premultiplied linear color, like Three.js
/// `LUTPass` with a 3D lookup table.
///
/// The pass unpremultiplies, encodes to sRGB when `inputEncoding` is `.sRGB`, maps
/// through the table's domain, interpolates, decodes back to linear and mixes with
/// the original by `intensity`. Most grading tools author tables for display-encoded
/// input, so apply this after tone mapping to keep HDR values inside the domain.
/// Pixels with zero alpha pass through unchanged.
public final class ColorLookupPass: PostProcessEffect {
    public enum InputEncoding: Sendable, Hashable {
        /// The table expects sRGB-encoded input and produces sRGB-encoded output.
        case sRGB
        /// The table maps linear values directly.
        case linear
    }

    public struct Settings: Sendable, Hashable {
        /// Blend from the original (0) to the graded color (1).
        public var intensity: Float
        public var interpolation: ColorLookupTable.Interpolation
        public var inputEncoding: InputEncoding

        public init(
            intensity: Float = 1, interpolation: ColorLookupTable.Interpolation = .tetrahedral,
            inputEncoding: InputEncoding = .sRGB
        ) {
            self.intensity = intensity
            self.interpolation = interpolation
            self.inputEncoding = inputEncoding
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    public var settings: Settings
    /// Assigning a table uploads it to a new 3D texture; frames already encoded keep the old one.
    public var table: ColorLookupTable {
        didSet { lattice = try? Self.upload(table, device: device) }
    }
    /// The RGBA16Float 3D texture holding the current table, red along width.
    public private(set) var lattice: (any MTLTexture)?
    private let shaders: PostProcessShaders

    public init(device: any MTLDevice, table: ColorLookupTable, settings: Settings = Settings()) throws {
        self.device = device
        self.table = table
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
        lattice = try Self.upload(table, device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_lut_fragment", formats: [pixelFormat])
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard settings.intensity.isFinite, (0...1).contains(settings.intensity) else {
            throw PostProcessEffectError.invalidSettings("Lookup intensity must be from 0 to 1.")
        }
        guard lattice != nil else { throw PostProcessEffectError.allocation }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: false
        )
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render("frostfield_lut_fragment", formats: [destination.pixelFormat])
        guard let lattice else { throw PostProcessEffectError.allocation }
        var parameters = LookupParameters(
            domainMinimum: SIMD4(table.domainMinimum, 0),
            domainScale: SIMD4(1 / (table.domainMaximum - table.domainMinimum), 0),
            size: Float(table.size), intensity: settings.intensity,
            interpolation: settings.interpolation == .tetrahedral ? 1 : 0,
            encoding: settings.inputEncoding == .sRGB ? 1 : 0
        )
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield color lookup", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(lattice, index: 1)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<LookupParameters>.stride, index: 0)
        }
    }

    private struct LookupParameters {
        var domainMinimum: SIMD4<Float>
        var domainScale: SIMD4<Float>
        var size: Float
        var intensity: Float
        var interpolation: UInt32
        var encoding: UInt32
    }

    private static func upload(_ table: ColorLookupTable, device: any MTLDevice) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .rgba16Float
        descriptor.width = table.size
        descriptor.height = table.size
        descriptor.depth = table.size
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw PostProcessEffectError.allocation }
        texture.label = "Frostfield color lookup table"
        let halves = table.values.flatMap { [Float16($0.x), Float16($0.y), Float16($0.z), Float16(1)] }
        halves.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake3D(0, 0, 0, table.size, table.size, table.size), mipmapLevel: 0, slice: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: table.size * 8, bytesPerImage: table.size * table.size * 8
            )
        }
        return texture
    }
}

extension ColorLookupPass: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
