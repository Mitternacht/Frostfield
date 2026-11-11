import Foundation
import simd

/// Parses Wavefront MTL material libraries, including the PBR statements `Pr`, `Pm`,
/// `Pc`, `Pcr`, `Ke`, `norm`, `map_Pr`, `map_Pm`, and `map_Ke`.
///
/// Texture statements keep their file name and the `-s`, `-o`, `-bm`, and `-clamp`
/// options; other options are recorded as ignored. MTL texture coordinates use a
/// bottom-left origin, so `-s` and `-o` are converted to Frostfield's top-left space.
struct WavefrontMaterialLibrary {
    struct TextureReference: Equatable {
        /// The file name exactly as written, relative to the library.
        let path: String
        let transform: PhysicallyBasedMaterial.TextureTransform
        let clamps: Bool
        /// The `-bm` multiplier for bump and normal maps.
        let bumpMultiplier: Float
    }

    struct Material: Equatable {
        let name: String
        var diffuse: SIMD3<Float>?
        var dissolve: Float?
        var emissive: SIMD3<Float>?
        var specularExponent: Float?
        var indexOfRefraction: Float?
        var roughness: Float?
        var metallic: Float?
        var clearcoat: Float?
        var clearcoatRoughness: Float?
        var diffuseMap: TextureReference?
        var dissolveMap: TextureReference?
        var emissiveMap: TextureReference?
        var roughnessMap: TextureReference?
        var metallicMap: TextureReference?
        /// From `norm`, a tangent-space normal map.
        var normalMap: TextureReference?
        /// A height map from `bump` or `map_bump`.
        var bumpMap: TextureReference?
        /// Statements and texture options the importer does not translate, for warnings.
        var ignored: [String] = []
    }

    private(set) var materials: [String: Material] = [:]

    init(_ text: String) {
        var current: Material?
        func finish() {
            if let current { materials[current.name] = current }
        }
        for rawLine in text.split(omittingEmptySubsequences: true, whereSeparator: \.isNewline) {
            let line = rawLine.prefix { $0 != "#" }.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            let fields = line.split(whereSeparator: \.isWhitespace).map(String.init)
            let keyword = fields[0]
            let values = Array(fields.dropFirst())
            if keyword == "newmtl" {
                finish()
                current = Material(name: values.joined(separator: " "))
                continue
            }
            guard current != nil else { continue }
            func numbers() -> [Float] { values.compactMap { Float($0) }.filter(\.isFinite) }
            func color() -> SIMD3<Float>? {
                let parsed = numbers()
                guard values.first != "spectral", values.first != "xyz", !parsed.isEmpty else { return nil }
                return parsed.count >= 3 ? SIMD3(parsed[0], parsed[1], parsed[2]) : SIMD3(repeating: parsed[0])
            }
            func scalar() -> Float? { numbers().first }
            func map() -> TextureReference? {
                var options: [String] = []
                let reference = Self.texture(values, ignored: &options)
                current?.ignored += options
                return reference
            }
            switch keyword.lowercased() {
            case "kd": current?.diffuse = color()
            case "ke": current?.emissive = color()
            case "d": current?.dissolve = values.first == "-halo" ? nil : scalar()
            case "tr": current?.dissolve = scalar().map { 1 - $0 }
            case "ns": current?.specularExponent = scalar()
            case "ni": current?.indexOfRefraction = scalar()
            case "pr": current?.roughness = scalar()
            case "pm": current?.metallic = scalar()
            case "pc": current?.clearcoat = scalar()
            case "pcr": current?.clearcoatRoughness = scalar()
            case "map_kd": let reference = map(); current?.diffuseMap = reference
            case "map_d": let reference = map(); current?.dissolveMap = reference
            case "map_ke": let reference = map(); current?.emissiveMap = reference
            case "map_pr": let reference = map(); current?.roughnessMap = reference
            case "map_pm": let reference = map(); current?.metallicMap = reference
            case "norm": let reference = map(); current?.normalMap = reference
            case "bump", "map_bump": let reference = map(); current?.bumpMap = reference
            case "ka", "ks", "illum", "tf", "sharpness":
                // Phong ambient and specular colors and illumination models have no
                // metallic-roughness equivalent; `Ns` already informs roughness.
                break
            default:
                current?.ignored.append(keyword)
            }
        }
        finish()
    }

    /// Parses `[options] file`. File names may contain spaces after the options.
    private static func texture(_ values: [String], ignored: inout [String]) -> TextureReference? {
        let argumentCounts: [String: Int] = [
            "-blendu": 1, "-blendv": 1, "-boost": 1, "-cc": 1, "-clamp": 1, "-imfchan": 1,
            "-mm": 2, "-texres": 1, "-type": 1, "-bm": 1, "-o": 3, "-s": 3, "-t": 3,
        ]
        var offset = SIMD2<Float>.zero
        var scale = SIMD2<Float>(repeating: 1)
        var clamps = false
        var multiplier: Float = 1
        var index = 0
        while index < values.count, let count = argumentCounts[values[index].lowercased()] {
            let option = values[index].lowercased()
            index += 1
            // -o, -s, and -t take one to three numbers.
            var arguments: [String] = []
            while arguments.count < count, index < values.count,
                  count < 3 || arguments.isEmpty || Float(values[index]) != nil {
                arguments.append(values[index])
                index += 1
            }
            let numbers = arguments.compactMap(Float.init)
            switch option {
            case "-o" where !numbers.isEmpty: offset = SIMD2(numbers[0], numbers.count > 1 ? numbers[1] : 0)
            case "-s" where !numbers.isEmpty: scale = SIMD2(numbers[0], numbers.count > 1 ? numbers[1] : 1)
            case "-clamp": clamps = arguments.first?.lowercased() == "on"
            case "-bm" where !numbers.isEmpty: multiplier = numbers[0]
            default: ignored.append(option)
            }
        }
        guard index < values.count else { return nil }
        // MTL's v axis points up: v' = sv v + ov becomes top-left v' = sv v + (1 - sv - ov).
        let transform = (try? PhysicallyBasedMaterial.TextureTransform(
            offset: SIMD2(offset.x, 1 - scale.y - offset.y), rotation: 0, scale: scale
        )) ?? .identity
        return TextureReference(
            path: values[index...].joined(separator: " "), transform: transform, clamps: clamps,
            bumpMultiplier: multiplier.isFinite ? multiplier : 1
        )
    }

    /// The `mtllib` file names an OBJ declares, in order.
    static func libraries(inOBJ text: String) -> [String] {
        var names: [String] = []
        for rawLine in text.split(omittingEmptySubsequences: true, whereSeparator: \.isNewline) {
            guard rawLine.hasPrefix("mtllib") else { continue }
            let line = rawLine.prefix { $0 != "#" }
            names += line.split(whereSeparator: \.isWhitespace).dropFirst().map(String.init)
        }
        return names
    }
}
