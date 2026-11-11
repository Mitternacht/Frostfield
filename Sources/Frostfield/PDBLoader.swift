import Foundation
import simd

/// Imports Protein Data Bank atom positions and explicit CONECT bonds.
public enum PDBLoader {
    public enum RecordKind: Equatable, Sendable {
        case atom
        case heteroatom
    }

    public struct Atom: Sendable {
        public let serial: Int
        public let kind: RecordKind
        public let name: String
        public let element: String
        public let residueName: String?
        public let chain: String?
        public let residueSequence: Int?
        public let insertionCode: String?
        public let alternateLocation: String?
        public let occupancy: Float?
        public let temperatureFactor: Float?
        /// Authored coordinates in angstroms.
        public let position: SIMD3<Float>
    }

    public struct Bond: Hashable, Sendable {
        /// Zero-based atom indices, in the first recorded CONECT direction.
        public let first: Int
        public let second: Int
    }

    public struct Asset: Sendable {
        public let atoms: [Atom]
        /// Linear CPK vertex colors in atom order.
        public let pointCloud: PointCloud
        public let bonds: [Bond]
        /// One independent segment per resolved bond. Nil when the file has no resolved bonds.
        public let bondGeometry: LineGeometry?
        /// The first MODEL number when the file contains an explicit model.
        public let modelNumber: Int?
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case encoding
        case empty
        case invalidAtom(Int)
        case duplicateSerial(Int)
        case invalidConnectivity(Int)
        case invalidModel(Int)

        public var errorDescription: String? {
            switch self {
            case .fileURL: "PDB import requires a regular local file."
            case .size: "The PDB file or molecular geometry exceeds the supported size."
            case .encoding: "The PDB file needs UTF-8 text."
            case .empty: "The selected PDB model contains no atoms."
            case .invalidAtom(let line): "PDB line \(line) has an invalid atom record."
            case .duplicateSerial(let serial): "PDB atom serial \(serial) is repeated in the selected model."
            case .invalidConnectivity(let line): "PDB line \(line) has an invalid CONECT record."
            case .invalidModel(let line): "PDB line \(line) has an invalid or unterminated MODEL record."
            }
        }
    }

    private static let maximumBytes = 64 * 1_024 * 1_024
    private static let maximumBonds = 1_000_000

    public static func load(contentsOf url: URL) throws -> Asset {
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

    public static func load(_ data: Data) throws -> Asset {
        guard data.count <= maximumBytes else { throw Failure.size }
        guard let text = String(data: data, encoding: .utf8) else { throw Failure.encoding }
        return try load(text)
    }

    /// Reads the first explicit model, or all atoms in a file without MODEL records.
    /// Reciprocal and duplicate CONECT pairs are coalesced. References to unavailable
    /// atoms are skipped. Atom positions stay in the file's coordinate frame.
    public static func load(_ text: String) throws -> Asset {
        guard text.utf8.count <= maximumBytes else { throw Failure.size }
        let content = text.first == "\u{feff}" ? text.dropFirst() : text[...]
        var atoms: [Atom] = []
        var indices: [Int: Int] = [:]
        var connectivity: [Bond] = []
        var seen: Set<Bond> = []
        var firstModel: Int?
        var modelStart: Int?
        var selectedModel = false
        var start = content.startIndex
        var line = 0
        while start < content.endIndex {
            line += 1
            let end = content[start...].firstIndex(where: \.isNewline) ?? content.endIndex
            let record = content[start..<end]
            start = end < content.endIndex ? content.index(after: end) : end
            let name = record.prefix(6).trimmingCharacters(in: .whitespaces)
            switch name {
            case "MODEL":
                let row = Row(record)
                guard modelStart == nil, firstModel != nil || atoms.isEmpty,
                      let model = Int(row.field(10..<14)), model > 0 else { throw Failure.invalidModel(line) }
                modelStart = line
                selectedModel = firstModel == nil
                if selectedModel { firstModel = model }
            case "ENDMDL":
                guard modelStart != nil else { throw Failure.invalidModel(line) }
                modelStart = nil
                selectedModel = false
            case "ATOM", "HETATM":
                guard firstModel == nil || selectedModel else { continue }
                let atom = try parseAtom(Row(record), kind: name == "ATOM" ? .atom : .heteroatom, line: line)
                guard indices.updateValue(atoms.count, forKey: atom.serial) == nil else {
                    throw Failure.duplicateSerial(atom.serial)
                }
                atoms.append(atom)
            case "CONECT":
                guard modelStart == nil || selectedModel else { continue }
                let row = Row(record)
                guard let source = Int(row.field(6..<11)), source > 0 else { throw Failure.invalidConnectivity(line) }
                for offset in stride(from: 11, to: 31, by: 5) {
                    let field = row.field(offset..<(offset + 5))
                    if field.isEmpty { continue }
                    guard let target = Int(field), target >= 0 else { throw Failure.invalidConnectivity(line) }
                    if target == 0 || source == target { continue }
                    let key = Bond(first: min(source, target), second: max(source, target))
                    if seen.insert(key).inserted {
                        guard connectivity.count < maximumBonds else { throw Failure.size }
                        connectivity.append(Bond(first: source, second: target))
                    }
                }
            default: continue
            }
        }
        if let modelStart { throw Failure.invalidModel(modelStart) }
        guard !atoms.isEmpty else { throw Failure.empty }

        let positions = atoms.map(\.position)
        let points = try PointCloud(positions: positions, colors: atoms.map { color($0.element) })
        let bonds = connectivity.compactMap { serials -> Bond? in
            guard let first = indices[serials.first], let second = indices[serials.second] else { return nil }
            return Bond(first: first, second: second)
        }
        let lines = bonds.isEmpty ? nil : try LineGeometry(
            positions: bonds.flatMap { [positions[$0.first], positions[$0.second]] }, topology: .segments
        )
        return Asset(atoms: atoms, pointCloud: points, bonds: bonds, bondGeometry: lines, modelNumber: firstModel)
    }

    private struct Row {
        private let bytes: [UInt8]

        init(_ line: Substring) {
            // All supported columns are within the original 80-byte PDB record.
            bytes = Array(line.utf8.prefix(80))
        }

        var hasCoordinates: Bool { bytes.count >= 54 && bytes.allSatisfy { $0 < 128 } }

        func field(_ columns: Range<Int>) -> String {
            text(columns).trimmingCharacters(in: .whitespaces)
        }

        func text(_ columns: Range<Int>) -> String {
            guard columns.lowerBound < bytes.count else { return "" }
            return String(decoding: bytes[columns.lowerBound..<min(columns.upperBound, bytes.count)], as: UTF8.self)
        }
    }

    private static func parseAtom(_ row: Row, kind: RecordKind, line: Int) throws -> Atom {
        let name = row.field(12..<16)
        guard row.hasCoordinates, let serial = Int(row.field(6..<11)), serial > 0, !name.isEmpty,
              let x = finiteFloat(row.field(30..<38)), let y = finiteFloat(row.field(38..<46)),
              let z = finiteFloat(row.field(46..<54)) else { throw Failure.invalidAtom(line) }

        func optional(_ columns: Range<Int>) -> String? {
            let value = row.field(columns)
            return value.isEmpty ? nil : value
        }

        func number<T>(_ columns: Range<Int>, parse: (String) -> T?) throws -> T? {
            guard let value = optional(columns) else { return nil }
            guard let parsed = parse(value) else { throw Failure.invalidAtom(line) }
            return parsed
        }

        return try Atom(
            serial: serial, kind: kind, name: name,
            element: element(declared: row.field(76..<78), atom: row.text(12..<16)),
            residueName: optional(17..<20), chain: optional(21..<22),
            residueSequence: number(22..<26) { Int($0) }, insertionCode: optional(26..<27),
            alternateLocation: optional(16..<17), occupancy: number(54..<60, parse: finiteFloat),
            temperatureFactor: number(60..<66, parse: finiteFloat), position: SIMD3(x, y, z)
        )
    }

    private static func finiteFloat(_ value: String) -> Float? {
        guard let number = Float(value), number.isFinite else { return nil }
        return number
    }

    private static func element(declared: String, atom: String) -> String {
        if !declared.isEmpty { return canonicalElement(declared) }
        let letters = atom.drop(while: { !$0.isLetter })
        guard let first = letters.first else { return "X" }
        if atom.first?.isLetter == true {
            let candidate = canonicalElement(String(letters.prefix(2)))
            if cpk[candidate] != nil { return candidate }
        }
        return String(first).uppercased()
    }

    private static func canonicalElement(_ value: String) -> String {
        value.prefix(1).uppercased() + value.dropFirst().lowercased()
    }

    private static func color(_ element: String) -> SIMD4<Float> {
        let packed = cpk[element] ?? 0x909090
        func linear(_ shift: Int) -> Float {
            GLTFColorTables.sRGBToLinear(Float((packed >> shift) & 255) / 255)
        }
        return SIMD4(linear(16), linear(8), linear(0), 1)
    }

    // Three.js r187 PDBLoader CPK palette. See LICENSES/Three-CurveExtras.txt
    // for the three.js MIT notice.
    private static let cpk: [String: UInt32] = [
        "H": 0xFFFFFF, "He": 0xD9FFFF, "Li": 0xCC80FF, "Be": 0xC2FF00, "B": 0xFFB5B5,
        "C": 0x909090, "N": 0x3050F8, "O": 0xFF0D0D, "F": 0x90E050, "Ne": 0xB3E3F5,
        "Na": 0xAB5CF2, "Mg": 0x8AFF00, "Al": 0xBFA6A6, "Si": 0xF0C8A0, "P": 0xFF8000,
        "S": 0xFFFF30, "Cl": 0x1FF01F, "Ar": 0x80D1E3, "K": 0x8F40D4, "Ca": 0x3DFF00,
        "Sc": 0xE6E6E6, "Ti": 0xBFC2C7, "V": 0xA6A6AB, "Cr": 0x8A99C7, "Mn": 0x9C7AC7,
        "Fe": 0xE06633, "Co": 0xF090A0, "Ni": 0x50D050, "Cu": 0xC88033, "Zn": 0x7D80B0,
        "Ga": 0xC28F8F, "Ge": 0x668F8F, "As": 0xBD80E3, "Se": 0xFFA100, "Br": 0xA62929,
        "Kr": 0x5CB8D1, "Rb": 0x702EB0, "Sr": 0x00FF00, "Y": 0x94FFFF, "Zr": 0x94E0E0,
        "Nb": 0x73C2C9, "Mo": 0x54B5B5, "Tc": 0x3B9E9E, "Ru": 0x248F8F, "Rh": 0x0A7D8C,
        "Pd": 0x006985, "Ag": 0xC0C0C0, "Cd": 0xFFD98F, "In": 0xA67573, "Sn": 0x668080,
        "Sb": 0x9E63B5, "Te": 0xD47A00, "I": 0x940094, "Xe": 0x429EB0, "Cs": 0x57178F,
        "Ba": 0x00C900, "La": 0x70D4FF, "Ce": 0xFFFFC7, "Pr": 0xD9FFC7, "Nd": 0xC7FFC7,
        "Pm": 0xA3FFC7, "Sm": 0x8FFFC7, "Eu": 0x61FFC7, "Gd": 0x45FFC7, "Tb": 0x30FFC7,
        "Dy": 0x1FFFC7, "Ho": 0x00FF9C, "Er": 0x00E675, "Tm": 0x00D452, "Yb": 0x00BF38,
        "Lu": 0x00AB24, "Hf": 0x4DC2FF, "Ta": 0x4DA6FF, "W": 0x2194D6, "Re": 0x267DAB,
        "Os": 0x266696, "Ir": 0x175487, "Pt": 0xD0D0E0, "Au": 0xFFD123, "Hg": 0xB8B8D0,
        "Tl": 0xA6544D, "Pb": 0x575961, "Bi": 0x9E4FB5, "Po": 0xAB5C00, "At": 0x754F45,
        "Rn": 0x428296, "Fr": 0x420066, "Ra": 0x007D00, "Ac": 0x70ABFA, "Th": 0x00BAFF,
        "Pa": 0x00A1FF, "U": 0x008FFF, "Np": 0x0080FF, "Pu": 0x006BFF, "Am": 0x545CF2,
        "Cm": 0x785CE3, "Bk": 0x8A4FE3, "Cf": 0xA136D4, "Es": 0xB31FD4, "Fm": 0xB31FBA,
        "Md": 0xB30DA6, "No": 0xBD0D87, "Lr": 0xC70066, "Rf": 0xCC0059, "Db": 0xD1004F,
        "Sg": 0xD90045, "Bh": 0xE00038, "Hs": 0xE6002E, "Mt": 0xEB0026, "Ds": 0xEB0026,
        "Rg": 0xEB0026, "Cn": 0xEB0026, "Uut": 0xEB0026, "Uuq": 0xEB0026, "Uup": 0xEB0026,
        "Uuh": 0xEB0026, "Uus": 0xEB0026, "Uuo": 0xEB0026, "Nh": 0xEB0026, "Fl": 0xEB0026,
        "Mc": 0xEB0026, "Lv": 0xEB0026, "Ts": 0xEB0026, "Og": 0xEB0026, "D": 0xFFFFFF,
        "T": 0xFFFFFF,
    ]
}
