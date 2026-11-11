import Foundation
import simd

/// A luminaire's luminous intensity distribution, read from an IES LM-63 photometric file.
///
/// The parser reads LM-63-1986, -1991, -1995, -2002 and -2019 text files with
/// `TILT=NONE`, photometric types C, B and A, and the standard symmetric
/// horizontal-angle ranges. Candela values are multiplied by the candela multiplier
/// and the ballast factor; LM-63-1995 and earlier files also apply their
/// ballast-lamp photometric factor.
///
/// **Orientation.** A light that uses the profile faces along its local −Z, which
/// is the photometric axis: the nadir (vertical angle 0°) for type C, and the beam
/// center (0°, 0°) for types A and B. Type C horizontal angles turn from local +X
/// (0°) toward local +Y (90°). For types A and B, positive horizontal angles lean
/// toward local +X and positive vertical angles toward local +Y; type A turns
/// horizontal angles about the local Y axis and type B turns vertical angles about
/// the local X axis.
///
/// **Interpolation.** Intensity between tabulated angles is bilinear in the vertical
/// and horizontal angles. Directions outside the tabulated vertical range, or
/// outside the expanded horizontal range of types A and B, receive no light.
///
/// **Rendering.** Both renderers sample one table resampled from the file, with
/// 361 rows of vertical angle from 0° to 180° in 0.5° steps and 144 columns of
/// horizontal angle in 2.5° steps, stored as half floats relative to
/// `maximumIntensity`. A light's intensity then sets the candela in the profile's
/// brightest direction. `luminousFlux` integrates that table exactly over the sphere.
public struct IESProfile: Equatable, Sendable {
    /// The goniometer coordinate system of the file.
    public enum PhotometricType: Int, Equatable, Sendable {
        /// Vertical polar axis; the usual system for architectural luminaires.
        case c = 1
        /// Horizontal polar axis through the luminaire's lateral axis; floodlights.
        case b = 2
        /// Vertical polar axis with latitude-style vertical angles; automotive lamps.
        case a = 3
    }

    /// Rows of the resampled table, covering vertical angles 0°–180° in 0.5° steps.
    public static let tableHeight = 361
    /// Columns of the resampled table, covering horizontal angles 0°–360° in 2.5° steps.
    public static let tableWidth = 144
    /// The most angles of either kind a file may list.
    public static let maximumAngleCount = 10_000

    /// The first line, such as `IES:LM-63-2019`, or `IESNA:LM-63-1986` for a file without one.
    public var format: String { storage.format }
    /// Keyword lines such as `[MANUFAC]` and `[LUMCAT]`, without brackets. Repeated
    /// keywords and `[MORE]` lines are joined with newlines.
    public var keywords: [String: String] { storage.keywords }
    /// The goniometer coordinate system specified in the file.
    public var photometricType: PhotometricType { storage.photometricType }
    /// Tabulated vertical angles in degrees, increasing.
    public var verticalAngles: [Float] { storage.verticalAngles }
    /// Tabulated horizontal angles in degrees, increasing.
    public var horizontalAngles: [Float] { storage.horizontalAngles }
    /// Candela for each horizontal angle, then each vertical angle: index
    /// `horizontal × verticalAngles.count + vertical`. Multipliers are applied.
    public var candela: [Float] { storage.candela }
    /// Number of lamps specified in the file.
    public var lampCount: Int { storage.lampCount }
    /// Rated lumens per lamp, or nil for absolute photometry (`-1` in the file).
    public var lumensPerLamp: Float? { storage.lumensPerLamp }
    /// The file's rated lamp lumens, `lampCount × lumensPerLamp`, or nil for absolute photometry.
    public var ratedLumens: Float? { storage.lumensPerLamp.map { $0 * Float(storage.lampCount) } }
    /// Input power in watts as listed in the file.
    public var inputWatts: Float { storage.inputWatts }
    /// The largest tabulated candela value.
    public var maximumIntensity: Float { storage.maximumIntensity }
    /// The luminous flux in lumens of the distribution both renderers draw: the
    /// resampled table integrated over the sphere and scaled by `maximumIntensity`.
    public var luminousFlux: Float { Float(Double(storage.maximumIntensity) * storage.normalizedSolidAngle) }

    let storage: Storage

    /// Parses LM-63 text in UTF-8, or Latin-1 when it is not valid UTF-8.
    public init(data: Data) throws {
        guard let text = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) else {
            throw Failure.encoding
        }
        try self.init(string: text)
    }

    /// Reads and parses an `.ies` file.
    public init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    /// Parses LM-63 text.
    public init(string: String) throws {
        storage = try Storage(parsing: string)
    }

    public static func == (left: IESProfile, right: IESProfile) -> Bool {
        left.storage === right.storage || left.storage.isEqual(to: right.storage)
    }

    /// Candela at photometric angles in degrees, in the file's coordinate system,
    /// with symmetric ranges expanded and bilinear interpolation. Angles outside the
    /// covered range give zero.
    public func intensity(verticalAngle: Float, horizontalAngle: Float) -> Float {
        storage.intensity(vertical: Double(verticalAngle), horizontal: Double(horizontalAngle))
    }

    /// Candela toward a direction in the light's local frame, whose −Z is the
    /// photometric axis and whose +X is horizontal angle 0°. The direction need not be unit length.
    public func intensity(toward direction: SIMD3<Float>) -> Float {
        let length = simd_length(SIMD3<Double>(direction))
        guard length.isNormal else { return 0 }
        let angles = storage.angles(toward: SIMD3<Double>(direction) / length)
        return storage.intensity(vertical: angles.vertical, horizontal: angles.horizontal)
    }

    /// The renderer’s resampled table intensity, in candela, which both renderers
    /// interpolate at vertical angle `θ` from the axis and horizontal angle `φ`, in radians.
    package func tableIntensity(theta: Float, phi: Float) -> Float {
        storage.tableValue(theta: Double(theta), phi: Double(phi)) * storage.maximumIntensity
    }

    /// `∫ p(ω) w(cos θ) dω` over the sphere, where `p` is the resampled table
    /// relative to its maximum and `θ` is the angle from the light's axis.
    func weightedSolidAngle(_ weight: (Double) -> Double) -> Double {
        storage.weightedSolidAngle(weight)
    }

    public enum Failure: LocalizedError, Equatable {
        case encoding
        case missingTilt
        case unsupportedTilt(String)
        case header(String)
        case angles(String)
        case values(String)

        public var errorDescription: String? {
            switch self {
            case .encoding: "The IES file is not UTF-8 or Latin-1 text."
            case .missingTilt: "The IES file has no TILT= line before its photometric data."
            case .unsupportedTilt(let tilt):
                "The IES file uses TILT=\(tilt). Frostfield reads TILT=NONE photometry; remove the tilt block or export without lamp tilt data."
            case .header(let detail): "The IES photometric header is invalid: \(detail)."
            case .angles(let detail): "The IES angle table is invalid: \(detail)."
            case .values(let detail): "The IES candela table is invalid: \(detail)."
            }
        }
    }

    // Immutable after initialization; shared between copies and used as a cache identity.
    final class Storage: Sendable {
        /// The validated source, so scene archives can reconstruct the same photometry.
        let source: String
        let format: String
        let keywords: [String: String]
        let photometricType: PhotometricType
        let verticalAngles: [Float]
        let horizontalAngles: [Float]
        let candela: [Float]
        let lampCount: Int
        let lumensPerLamp: Float?
        let inputWatts: Float
        let maximumIntensity: Float
        /// `tableHeight` rows of `tableWidth` values, relative to the maximum, row-major from θ = 0.
        let table: [Float]
        /// `∫ p dω` of the normalized table.
        let normalizedSolidAngle: Double
        /// Mean of each table row over the horizontal angle, for cone-weighted integrals.
        private let rowMeans: [Double]

        // Every stored value is checked before the table is built.
        // swiftlint:disable:next function_body_length
        init(parsing text: String) throws {
            let lines = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
                .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            var index = 0
            var format = "IESNA:LM-63-1986"
            if let first = lines.first?.trimmingCharacters(in: .whitespaces),
               first.uppercased().hasPrefix("IES") {
                format = first
                index = 1
            }
            var keywords: [String: String] = [:]
            var lastKeyword: String?
            var tiltLine: String?
            while index < lines.count {
                let line = lines[index].trimmingCharacters(in: .whitespaces)
                index += 1
                if line.uppercased().hasPrefix("TILT") {
                    tiltLine = line
                    break
                }
                guard line.hasPrefix("["), let close = line.firstIndex(of: "]") else { continue }
                var name = String(line[line.index(after: line.startIndex)..<close]).uppercased()
                let value = line[line.index(after: close)...].trimmingCharacters(in: .whitespaces)
                if name == "MORE", let previous = lastKeyword { name = previous }
                keywords[name] = keywords[name].map { $0 + "\n" + value } ?? value
                lastKeyword = name
            }
            guard let tiltLine, let equals = tiltLine.firstIndex(of: "=") else { throw Failure.missingTilt }
            let tilt = tiltLine[tiltLine.index(after: equals)...].trimmingCharacters(in: .whitespaces)
            guard tilt.uppercased() == "NONE" else { throw Failure.unsupportedTilt(tilt) }

            let separators = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ","))
            let tokens = lines[index...].joined(separator: "\n").components(separatedBy: separators).filter { !$0.isEmpty }
            var cursor = 0
            func number(_ name: String) throws -> Double {
                guard cursor < tokens.count else { throw Failure.header("missing \(name)") }
                guard let value = Double(tokens[cursor]), value.isFinite else {
                    throw Failure.header("\(name) “\(tokens[cursor])” is not a finite number")
                }
                cursor += 1
                return value
            }
            func count(_ name: String) throws -> Int {
                let value = try number(name)
                guard value == value.rounded(), value >= 1, value <= Double(IESProfile.maximumAngleCount) else {
                    throw Failure.header("\(name) must be a whole number from 1 to \(IESProfile.maximumAngleCount)")
                }
                return Int(value)
            }

            let lamps = try number("number of lamps")
            guard lamps == lamps.rounded(), lamps >= 1, lamps <= 100_000 else {
                throw Failure.header("the number of lamps must be a whole number from 1 to 100,000")
            }
            let lumens = try number("lumens per lamp")
            guard lumens == -1 || lumens > 0 else { throw Failure.header("lumens per lamp must be positive or -1") }
            let multiplier = try number("candela multiplier")
            guard multiplier > 0 else { throw Failure.header("the candela multiplier must be positive") }
            let verticalCount = try count("number of vertical angles")
            let horizontalCount = try count("number of horizontal angles")
            guard verticalCount * horizontalCount <= 4_000_000 else { throw Failure.header("the table exceeds 4,000,000 values") }
            let typeValue = try number("photometric type")
            guard let rawType = Int(exactly: typeValue), let type = PhotometricType(rawValue: rawType) else {
                throw Failure.header("the photometric type must be 1 (C), 2 (B) or 3 (A)")
            }
            let units = try number("units type")
            guard units == 1 || units == 2 else { throw Failure.header("the units type must be 1 (feet) or 2 (meters)") }
            for name in ["width", "length", "height"] { _ = try number("luminous opening \(name)") }
            let ballast = try number("ballast factor")
            guard ballast > 0 else { throw Failure.header("the ballast factor must be positive") }
            // LM-63-1995 and earlier list a ballast-lamp photometric factor here. LM-63-2002
            // reserves the field and LM-63-2019 uses it for the file generation type.
            let secondFactor = try number("ballast-lamp factor or file generation type")
            let watts = try number("input watts")
            let usesLampFactor = !(format.contains("2002") || format.contains("2019"))
            guard !usesLampFactor || secondFactor > 0 else { throw Failure.header("the ballast-lamp factor must be positive") }
            let scale = multiplier * ballast * (usesLampFactor ? secondFactor : 1)

            var vertical: [Float] = []
            vertical.reserveCapacity(verticalCount)
            for _ in 0..<verticalCount { vertical.append(Float(try number("vertical angle"))) }
            var horizontal: [Float] = []
            horizontal.reserveCapacity(horizontalCount)
            for _ in 0..<horizontalCount { horizontal.append(Float(try number("horizontal angle"))) }
            var values: [Float] = []
            values.reserveCapacity(verticalCount * horizontalCount)
            for _ in 0..<(verticalCount * horizontalCount) {
                guard cursor < tokens.count else {
                    throw Failure.values("expected \(verticalCount * horizontalCount) values, found \(values.count)")
                }
                guard let value = Double(tokens[cursor]), value.isFinite, value >= 0 else {
                    throw Failure.values("“\(tokens[cursor])” is not a finite, nonnegative candela value")
                }
                cursor += 1
                let scaled = value * scale
                guard scaled <= Double(Float.greatestFiniteMagnitude) else { throw Failure.values("a candela value overflows") }
                values.append(Float(scaled))
            }

            try Self.validate(vertical: vertical, horizontal: horizontal, type: type)
            let maximum = values.max() ?? 0
            guard maximum > 0 else { throw Failure.values("every candela value is zero") }

            source = text
            self.format = format
            self.keywords = keywords
            photometricType = type
            verticalAngles = vertical
            horizontalAngles = horizontal
            candela = values
            lampCount = Int(lamps)
            lumensPerLamp = lumens > 0 ? Float(lumens) : nil
            inputWatts = Float(watts)
            maximumIntensity = maximum

            // Resample onto the renderers' grid, then integrate the bilinear interpolant exactly.
            let width = IESProfile.tableWidth
            let height = IESProfile.tableHeight
            var table = [Float](repeating: 0, count: width * height)
            var means = [Double](repeating: 0, count: height)
            let inverseMaximum = 1 / Double(maximum)
            for row in 0..<height {
                let theta = Double(row) * .pi / Double(height - 1)
                var sum = 0.0
                for column in 0..<width {
                    let phi = Double(column) * 2 * .pi / Double(width)
                    let direction = SIMD3(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta))
                    let value = Self.lookup(
                        direction: direction, type: type, vertical: vertical, horizontal: horizontal, candela: values
                    ) * inverseMaximum
                    let stored = Float(Float16(Float(min(max(value, 0), 1))))
                    table[row * width + column] = stored
                    sum += Double(stored)
                }
                means[row] = sum / Double(width)
            }
            self.table = table
            rowMeans = means
            // A row interpolates linearly in θ, so ∫ (a + (b − a) t / Δ) sin(θ₀ + t) dt over each step is exact.
            let step = Double.pi / Double(height - 1)
            var solidAngle = 0.0
            for row in 0..<(height - 1) {
                let low = Double(row) * step
                let high = low + step
                let slope = (sin(high) - sin(low)) / step
                solidAngle += means[row] * (cos(low) - slope) + means[row + 1] * (slope - cos(high))
            }
            normalizedSolidAngle = 2 * .pi * solidAngle
        }

        private static func validate(vertical: [Float], horizontal: [Float], type: PhotometricType) throws {
            guard zip(vertical, vertical.dropFirst()).allSatisfy({ $0 < $1 }) else {
                throw Failure.angles("vertical angles must increase")
            }
            guard zip(horizontal, horizontal.dropFirst()).allSatisfy({ $0 < $1 }) else {
                throw Failure.angles("horizontal angles must increase")
            }
            let firstV = vertical[0], lastV = vertical[vertical.count - 1]
            let firstH = horizontal[0], lastH = horizontal[horizontal.count - 1]
            switch type {
            case .c:
                guard firstV >= 0, lastV <= 180 else { throw Failure.angles("type C vertical angles must lie in 0°–180°") }
                guard firstH >= 0, lastH <= 360 else { throw Failure.angles("type C horizontal angles must lie in 0°–360°") }
                guard horizontal.count == 1 || firstH == 0 || (firstH == 90 && lastH == 270) else {
                    throw Failure.angles("type C horizontal angles must start at 0°, or run from 90° to 270°")
                }
            case .a, .b:
                guard firstV >= -90, lastV <= 90 else { throw Failure.angles("type A and B vertical angles must lie in -90°–90°") }
                guard (firstH == 0 || firstH == -90), lastH <= 90 else {
                    throw Failure.angles("type A and B horizontal angles must run from 0° or -90° to at most 90°")
                }
            }
        }

        /// Photometric angles in degrees for a unit direction in the light's local frame.
        func angles(toward direction: SIMD3<Double>) -> (vertical: Double, horizontal: Double) {
            Self.angles(toward: direction, type: photometricType)
        }

        private static func angles(toward d: SIMD3<Double>, type: PhotometricType) -> (vertical: Double, horizontal: Double) {
            let degrees = 180 / Double.pi
            let forward = -d.z
            switch type {
            case .c:
                var h = atan2(d.y, d.x) * degrees
                if h < 0 { h += 360 }
                return (acos(min(max(forward, -1), 1)) * degrees, h)
            case .a:
                return (asin(min(max(d.y, -1), 1)) * degrees, atan2(d.x, forward) * degrees)
            case .b:
                return (atan2(d.y, forward) * degrees, asin(min(max(d.x, -1), 1)) * degrees)
            }
        }

        func intensity(vertical: Double, horizontal: Double) -> Float {
            Float(Self.interpolate(
                vertical: vertical, horizontal: horizontal, type: photometricType,
                verticalAngles: verticalAngles, horizontalAngles: horizontalAngles, candela: candela
            ))
        }

        private static func lookup(
            direction: SIMD3<Double>, type: PhotometricType, vertical: [Float], horizontal: [Float], candela: [Float]
        ) -> Double {
            let angles = angles(toward: direction, type: type)
            return interpolate(
                vertical: angles.vertical, horizontal: angles.horizontal, type: type,
                verticalAngles: vertical, horizontalAngles: horizontal, candela: candela
            )
        }

        /// Folds a horizontal angle into the tabulated range using the file's symmetry.
        private static func fold(horizontal angle: Double, type: PhotometricType, first: Double, last: Double) -> Double? {
            switch type {
            case .c:
                var h = angle.truncatingRemainder(dividingBy: 360)
                if h < 0 { h += 360 }
                if first == 0 && last == 0 { return 0 }
                if first == 0 && last == 90 {
                    // Quadrant symmetry about the 0°–180° and 90°–270° planes.
                    if h > 270 { h = 360 - h } else if h > 180 { h -= 180 } else if h > 90 { h = 180 - h }
                    return h
                }
                if first == 0 && last == 180 {
                    // Bilateral symmetry about the 0°–180° plane.
                    return h > 180 ? 360 - h : h
                }
                if first == 90 && last == 270 {
                    // Bilateral symmetry about the 90°–270° plane: I(H) = I(180° − H).
                    if h < 90 { return 180 - h }
                    if h > 270 { return 540 - h }
                    return h
                }
                return h
            case .a, .b:
                if first == 0 { return abs(angle) }
                return angle
            }
        }

        // swiftlint:disable:next function_parameter_count
        private static func interpolate(
            vertical: Double, horizontal: Double, type: PhotometricType,
            verticalAngles: [Float], horizontalAngles: [Float], candela: [Float]
        ) -> Double {
            let verticalCount = verticalAngles.count
            let firstV = Double(verticalAngles[0])
            let lastV = Double(verticalAngles[verticalCount - 1])
            let tolerance = 1e-9
            guard vertical >= firstV - tolerance, vertical <= lastV + tolerance else { return 0 }
            let firstH = Double(horizontalAngles[0])
            let lastH = Double(horizontalAngles[horizontalAngles.count - 1])
            guard let folded = fold(horizontal: horizontal, type: type, first: firstH, last: lastH) else { return 0 }

            func bracket(_ angles: [Float], _ value: Double) -> (Int, Int, Double) {
                if angles.count == 1 { return (0, 0, 0) }
                var low = 0
                var high = angles.count - 1
                if value <= Double(angles[0]) { return (0, 0, 0) }
                if value >= Double(angles[high]) { return (high, high, 0) }
                while high - low > 1 {
                    let middle = (low + high) / 2
                    if Double(angles[middle]) <= value { low = middle } else { high = middle }
                }
                let a = Double(angles[low])
                let b = Double(angles[high])
                return (low, high, (value - a) / (b - a))
            }

            let (v0, v1, vt) = bracket(verticalAngles, min(max(vertical, firstV), lastV))
            var h0: Int
            var h1: Int
            var ht: Double
            if horizontalAngles.count == 1 {
                (h0, h1, ht) = (0, 0, 0)
            } else if type == .c, firstH == 0, lastH < 360, ![90.0, 180.0].contains(lastH), folded > lastH {
                // A full table without a 360° column wraps from its last angle back to 0°.
                (h0, h1, ht) = (horizontalAngles.count - 1, 0, (folded - lastH) / (360 - lastH))
            } else {
                guard folded >= firstH - tolerance, folded <= lastH + tolerance else { return 0 }
                (h0, h1, ht) = bracket(horizontalAngles, min(max(folded, firstH), lastH))
            }
            func value(_ h: Int, _ v: Int) -> Double { Double(candela[h * verticalCount + v]) }
            let near = value(h0, v0) + (value(h0, v1) - value(h0, v0)) * vt
            let far = value(h1, v0) + (value(h1, v1) - value(h1, v0)) * vt
            return near + (far - near) * ht
        }

        /// Bilinear lookup in the resampled table with the renderers' addressing.
        func tableValue(theta: Double, phi: Double) -> Float {
            let width = IESProfile.tableWidth
            let height = IESProfile.tableHeight
            let y = min(max(theta / .pi, 0), 1) * Double(height - 1)
            var x = phi / (2 * .pi) * Double(width)
            x = x - floor(x / Double(width)) * Double(width)
            let row = min(Int(y), height - 2)
            let ty = y - Double(row)
            let column = Int(x) % width
            let tx = x - floor(x)
            let next = (column + 1) % width
            func at(_ r: Int, _ c: Int) -> Double { Double(table[r * width + c]) }
            let top = at(row, column) + (at(row, next) - at(row, column)) * tx
            let bottom = at(row + 1, column) + (at(row + 1, next) - at(row + 1, column)) * tx
            return Float(top + (bottom - top) * ty)
        }

        /// `∫ p(ω) w(cos θ) dω`, integrating each row interval with 16 midpoint steps.
        func weightedSolidAngle(_ weight: (Double) -> Double) -> Double {
            let height = IESProfile.tableHeight
            let step = Double.pi / Double(height - 1)
            let substeps = 16
            var total = 0.0
            for row in 0..<(height - 1) {
                for sub in 0..<substeps {
                    let t = (Double(sub) + 0.5) / Double(substeps)
                    let theta = (Double(row) + t) * step
                    let mean = rowMeans[row] + (rowMeans[row + 1] - rowMeans[row]) * t
                    total += mean * weight(cos(theta)) * sin(theta) * step / Double(substeps)
                }
            }
            return 2 * .pi * total
        }

        func isEqual(to other: Storage) -> Bool {
            photometricType == other.photometricType && verticalAngles == other.verticalAngles
                && horizontalAngles == other.horizontalAngles && candela == other.candela
                && lampCount == other.lampCount && lumensPerLamp == other.lumensPerLamp
                && inputWatts == other.inputWatts && format == other.format && keywords == other.keywords
        }
    }
}
