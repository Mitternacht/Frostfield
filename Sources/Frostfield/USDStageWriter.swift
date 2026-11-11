import Foundation
import simd

/// A UsdPreviewSurface material in the exporter's neutral form. Textures refer to package images.
struct USDMaterialDescription: Hashable {
    enum Alpha: Hashable {
        case opaque
        case mask(Float)
        case blend
    }

    struct Texture: Hashable {
        /// Index into the package's image list.
        let image: Int
        let coordinates: MeshTextureCoordinateSet
        /// In Frostfield's top-left UV space.
        let transform: PhysicallyBasedMaterial.TextureTransform
        /// UsdUVTexture wrap tokens.
        let wrapS: String
        let wrapT: String
    }

    var baseColor = SIMD4<Float>(repeating: 1)
    var baseColorTexture: Texture?
    var metallic: Float = 1
    var roughness: Float = 1
    /// glTF channels: green is roughness and blue is metallic.
    var metallicRoughnessTexture: Texture?
    /// Written with the standard `(2, 2, 2, 1)` scale and `(-1, -1, -1, 0)` bias.
    var normalTexture: Texture?
    /// Red channel.
    var occlusionTexture: Texture?
    var occlusionStrength: Float = 1
    /// The emissive factor multiplied by its strength, in linear radiance units.
    var emissive = SIMD3<Float>.zero
    var emissiveTexture: Texture?
    var alpha = Alpha.opaque
    var clearcoat: Float = 0
    var clearcoatRoughness: Float = 0
    /// Red channel.
    var clearcoatTexture: Texture?
    /// Green channel.
    var clearcoatRoughnessTexture: Texture?
    var indexOfRefraction: Float = 1.5
    /// Writes the base color as emission over a black diffuse response.
    var isUnlit = false
}

/// Converts Frostfield's top-left texture transforms into UsdTransform2d parameters.
package enum USDTextureTransform {
    /// UsdTransform2d computes `result = rotate(in * scale) + translation` in USD's
    /// bottom-left `st` space with a counterclockwise rotation in degrees. Frostfield
    /// samples `u' = c sx u + s sy v + ox` and `v' = -s sx u + c sy v + oy` in top-left
    /// UV space, and `st = (u, 1 - v)`. Conjugating by that flip keeps the scale and
    /// rotation angle and moves the translation to `(ox + sy sin r, 1 - oy - sy cos r)`.
    package static func parameters(
        _ transform: PhysicallyBasedMaterial.TextureTransform
    ) -> (scale: SIMD2<Float>, rotationDegrees: Float, translation: SIMD2<Float>) {
        let sine = sin(transform.rotation), cosine = cos(transform.rotation)
        return (
            transform.scale,
            transform.rotation * 180 / .pi,
            SIMD2(transform.offset.x + transform.scale.y * sine, 1 - transform.offset.y - transform.scale.y * cosine)
        )
    }

    /// Evaluates UsdTransform2d for the given parameters, for tests and documentation.
    package static func evaluate(
        _ st: SIMD2<Float>, scale: SIMD2<Float>, rotationDegrees: Float, translation: SIMD2<Float>
    ) -> SIMD2<Float> {
        let radians = rotationDegrees * .pi / 180
        let scaled = st * scale
        return SIMD2(
            cos(radians) * scaled.x - sin(radians) * scaled.y,
            sin(radians) * scaled.x + cos(radians) * scaled.y
        ) + translation
    }
}

/// Builds indented USDA text.
struct USDAWriter {
    private(set) var text = ""
    private var depth = 0

    mutating func line(_ content: String = "") {
        if content.isEmpty {
            text += "\n"
        } else {
            text += String(repeating: "    ", count: depth) + content + "\n"
        }
    }

    /// Opens a prim or block. Metadata lines go in parentheses after the header.
    mutating func open(_ header: String, metadata: [String] = []) {
        line(header + (metadata.isEmpty ? "" : " ("))
        if !metadata.isEmpty {
            depth += 1
            metadata.forEach { line($0) }
            depth -= 1
            line(")")
        }
        line("{")
        depth += 1
    }

    mutating func close() {
        depth -= 1
        line("}")
    }

    static func layerHeader(defaultPrim: String, metersPerUnit: Double, extra: [String] = []) -> String {
        var lines = ["#usda 1.0", "("]
        lines += extra.map { "    " + $0 }
        lines += [
            "    defaultPrim = \"\(defaultPrim)\"",
            "    metersPerUnit = \(number(metersPerUnit))",
            "    upAxis = \"Y\"",
            ")",
            "",
        ]
        return lines.joined(separator: "\n") + "\n"
    }

    static func number(_ value: Float) -> String {
        value == value.rounded() && abs(value) < 1e9 ? String(Int64(value)) : "\(value)"
    }

    static func number(_ value: Double) -> String {
        value == value.rounded() && abs(value) < 1e15 ? String(Int64(value)) : "\(value)"
    }

    static func tuple(_ values: Float...) -> String {
        "(" + values.map(number).joined(separator: ", ") + ")"
    }

    static func string(_ value: String) -> String {
        var escaped = ""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": escaped += "\\\""
            case "\\": escaped += "\\\\"
            case "\n": escaped += "\\n"
            default: escaped.unicodeScalars.append(scalar)
            }
        }
        return "\"" + escaped + "\""
    }
}

/// Produces valid, unique USD prim names.
struct USDNameAllocator {
    private var used: Set<String> = []

    init(reserved: [String] = []) {
        used = Set(reserved)
    }

    /// Replaces characters outside `[A-Za-z0-9_]`, prefixes a leading digit, and appends
    /// `_1`, `_2`, … until the name is unique among this allocator's names.
    mutating func name(_ proposed: String?, fallback: String) -> String {
        var base = String((proposed ?? "").unicodeScalars.map { scalar -> Character in
            scalar.isASCII && (CharacterSet.alphanumerics.contains(scalar) || scalar == "_") ? Character(scalar) : "_"
        })
        if base.isEmpty || base.allSatisfy({ $0 == "_" }) { base = fallback }
        if let first = base.unicodeScalars.first, CharacterSet.decimalDigits.contains(first) { base = "_" + base }
        var candidate = base
        var suffix = 1
        while used.contains(candidate) {
            candidate = "\(base)_\(suffix)"
            suffix += 1
        }
        used.insert(candidate)
        return candidate
    }
}

enum USDGeometryLayer {
    /// A layer holding one triangle mesh as `/Geometry`. Points, normals, `st`, the optional
    /// `st1`, and optional tangents use vertex interpolation in the source vertex order.
    /// `st` and `st1` flip V into USD's bottom-left texture space. Tangents are `float4`
    /// with `xyz` along +s and `w` chosen so that `cross(normal, xyz) * w` points along +t,
    /// which negates Frostfield's top-left handedness.
    static func text(
        _ geometry: MeshGeometry, metersPerUnit: Double, includeSecondary: Bool, includeTangents: Bool
    ) -> String {
        var writer = USDAWriter()
        writer.open("def Mesh \"Geometry\"")
        let positions = geometry.vertices.map(\.position)
        let lower = positions.dropFirst().reduce(positions[0]) { simd_min($0, $1) }
        let upper = positions.dropFirst().reduce(positions[0]) { simd_max($0, $1) }
        writer.line("float3[] extent = [\(vector(lower)), \(vector(upper))]")
        writer.line("int[] faceVertexCounts = [" + Array(repeating: "3", count: geometry.indices.count / 3).joined(separator: ", ") + "]")
        writer.line("int[] faceVertexIndices = [" + geometry.indices.map { String($0) }.joined(separator: ", ") + "]")
        writer.line("normal3f[] normals = [" + geometry.vertices.map { vector($0.normal) }.joined(separator: ", ") + "] (")
        writer.line("    interpolation = \"vertex\"")
        writer.line(")")
        writer.line("point3f[] points = [" + positions.map(vector).joined(separator: ", ") + "]")
        writer.line("texCoord2f[] primvars:st = [" + geometry.vertices.map { flipped($0.textureCoordinate) }.joined(separator: ", ") + "] (")
        writer.line("    interpolation = \"vertex\"")
        writer.line(")")
        if includeSecondary {
            let values = geometry.vertices.map { flipped($0.secondaryTextureCoordinate ?? .zero) }
            writer.line("texCoord2f[] primvars:st1 = [" + values.joined(separator: ", ") + "] (")
            writer.line("    interpolation = \"vertex\"")
            writer.line(")")
        }
        if includeTangents {
            let values = geometry.vertices.map { vertex -> String in
                let tangent = vertex.tangent ?? SIMD4(1, 0, 0, 1)
                return USDAWriter.tuple(tangent.x, tangent.y, tangent.z, -tangent.w)
            }
            writer.line("float4[] primvars:tangents = [" + values.joined(separator: ", ") + "] (")
            writer.line("    interpolation = \"vertex\"")
            writer.line(")")
        }
        writer.line("uniform token orientation = \"rightHanded\"")
        writer.line("uniform token subdivisionScheme = \"none\"")
        writer.close()
        return USDAWriter.layerHeader(defaultPrim: "Geometry", metersPerUnit: metersPerUnit) + writer.text
    }

    private static func vector(_ value: SIMD3<Float>) -> String {
        USDAWriter.tuple(value.x, value.y, value.z)
    }

    private static func flipped(_ value: SIMD2<Float>) -> String {
        USDAWriter.tuple(value.x, 1 - value.y)
    }
}

enum USDMaterialNetwork {
    /// Writes one `Material` prim with a UsdPreviewSurface and its texture nodes.
    /// `imagePaths` maps image indices to package-relative paths.
    static func write(
        _ material: USDMaterialDescription, name: String, parentPath: String, imagePaths: [String],
        into writer: inout USDAWriter
    ) {
        let path = "\(parentPath)/\(name)"
        writer.open("def Material \"\(name)\"")
        writer.line("token outputs:surface.connect = <\(path)/PreviewSurface.outputs:surface>")
        writer.line()

        var shaders = USDAWriter()
        var readers: Set<MeshTextureCoordinateSet> = []
        // Returns the connection source for one channel output of a texture node.
        func texture(
            _ texture: USDMaterialDescription.Texture, node: String, color: Bool, outputs: [String],
            scale: SIMD4<Float>? = nil, bias: SIMD4<Float>? = nil
        ) {
            readers.insert(texture.coordinates)
            var source = "\(path)/\(readerName(texture.coordinates)).outputs:result"
            if texture.transform != .identity {
                let parameters = USDTextureTransform.parameters(texture.transform)
                shaders.line()
                shaders.open("def Shader \"\(node)Transform\"")
                shaders.line("uniform token info:id = \"UsdTransform2d\"")
                shaders.line("float2 inputs:in.connect = <\(source)>")
                shaders.line("float inputs:rotation = \(USDAWriter.number(parameters.rotationDegrees))")
                shaders.line("float2 inputs:scale = \(USDAWriter.tuple(parameters.scale.x, parameters.scale.y))")
                shaders.line("float2 inputs:translation = \(USDAWriter.tuple(parameters.translation.x, parameters.translation.y))")
                shaders.line("float2 outputs:result")
                shaders.close()
                source = "\(path)/\(node)Transform.outputs:result"
            }
            shaders.line()
            shaders.open("def Shader \"\(node)\"")
            shaders.line("uniform token info:id = \"UsdUVTexture\"")
            shaders.line("asset inputs:file = @\(imagePaths[texture.image])@")
            shaders.line("float2 inputs:st.connect = <\(source)>")
            shaders.line("token inputs:sourceColorSpace = \"\(color ? "sRGB" : "raw")\"")
            shaders.line("token inputs:wrapS = \"\(texture.wrapS)\"")
            shaders.line("token inputs:wrapT = \"\(texture.wrapT)\"")
            if let scale, scale != SIMD4(repeating: 1) {
                shaders.line("float4 inputs:scale = \(USDAWriter.tuple(scale.x, scale.y, scale.z, scale.w))")
            }
            if let bias, bias != .zero {
                shaders.line("float4 inputs:bias = \(USDAWriter.tuple(bias.x, bias.y, bias.z, bias.w))")
            }
            for output in outputs {
                shaders.line(output == "rgb" ? "float3 outputs:rgb" : "float outputs:\(output)")
            }
            shaders.close()
        }

        writer.open("def Shader \"PreviewSurface\"")
        writer.line("uniform token info:id = \"UsdPreviewSurface\"")
        let base = material.baseColor
        let usesAlpha = material.alpha != .opaque
        var baseOutputs = ["rgb"]
        if usesAlpha { baseOutputs.append("a") }
        let colorInput = material.isUnlit ? "emissiveColor" : "diffuseColor"
        if let map = material.baseColorTexture {
            writer.line("color3f inputs:\(colorInput).connect = <\(path)/BaseColorTexture.outputs:rgb>")
            texture(map, node: "BaseColorTexture", color: true, outputs: baseOutputs, scale: base)
        } else {
            writer.line("color3f inputs:\(colorInput) = \(USDAWriter.tuple(base.x, base.y, base.z))")
        }
        if material.isUnlit {
            writer.line("color3f inputs:diffuseColor = (0, 0, 0)")
            writer.line("float inputs:metallic = 0")
            writer.line("float inputs:roughness = 1")
        } else {
            if let map = material.metallicRoughnessTexture {
                writer.line("float inputs:metallic.connect = <\(path)/MetallicRoughnessTexture.outputs:b>")
                writer.line("float inputs:roughness.connect = <\(path)/MetallicRoughnessTexture.outputs:g>")
                texture(
                    map, node: "MetallicRoughnessTexture", color: false, outputs: ["g", "b"],
                    scale: SIMD4(1, material.roughness, material.metallic, 1)
                )
            } else {
                writer.line("float inputs:metallic = \(USDAWriter.number(material.metallic))")
                writer.line("float inputs:roughness = \(USDAWriter.number(material.roughness))")
            }
            if let map = material.normalTexture {
                writer.line("normal3f inputs:normal.connect = <\(path)/NormalTexture.outputs:rgb>")
                texture(
                    map, node: "NormalTexture", color: false, outputs: ["rgb"],
                    scale: SIMD4(2, 2, 2, 1), bias: SIMD4(-1, -1, -1, 0)
                )
            }
            if let map = material.occlusionTexture {
                let strength = material.occlusionStrength
                writer.line("float inputs:occlusion.connect = <\(path)/OcclusionTexture.outputs:r>")
                texture(
                    map, node: "OcclusionTexture", color: false, outputs: ["r"],
                    scale: SIMD4(strength, strength, strength, 1), bias: SIMD4(1 - strength, 1 - strength, 1 - strength, 0)
                )
            }
            let emissive = material.emissive
            if let map = material.emissiveTexture {
                writer.line("color3f inputs:emissiveColor.connect = <\(path)/EmissiveTexture.outputs:rgb>")
                texture(map, node: "EmissiveTexture", color: true, outputs: ["rgb"], scale: SIMD4(emissive, 1))
            } else if emissive != .zero {
                writer.line("color3f inputs:emissiveColor = \(USDAWriter.tuple(emissive.x, emissive.y, emissive.z))")
            }
            if material.clearcoat > 0 || material.clearcoatTexture != nil {
                if let map = material.clearcoatTexture {
                    writer.line("float inputs:clearcoat.connect = <\(path)/ClearcoatTexture.outputs:r>")
                    let factor = material.clearcoat
                    texture(map, node: "ClearcoatTexture", color: false, outputs: ["r"], scale: SIMD4(factor, factor, factor, 1))
                } else {
                    writer.line("float inputs:clearcoat = \(USDAWriter.number(material.clearcoat))")
                }
                if let map = material.clearcoatRoughnessTexture {
                    writer.line("float inputs:clearcoatRoughness.connect = <\(path)/ClearcoatRoughnessTexture.outputs:g>")
                    let factor = material.clearcoatRoughness
                    texture(
                        map, node: "ClearcoatRoughnessTexture", color: false, outputs: ["g"],
                        scale: SIMD4(factor, factor, factor, 1)
                    )
                } else {
                    writer.line("float inputs:clearcoatRoughness = \(USDAWriter.number(material.clearcoatRoughness))")
                }
            }
            if material.indexOfRefraction != 1.5 {
                writer.line("float inputs:ior = \(USDAWriter.number(material.indexOfRefraction))")
            }
        }
        if usesAlpha {
            if material.baseColorTexture != nil {
                writer.line("float inputs:opacity.connect = <\(path)/BaseColorTexture.outputs:a>")
            } else {
                writer.line("float inputs:opacity = \(USDAWriter.number(base.w))")
            }
            if case .mask(let cutoff) = material.alpha {
                writer.line("float inputs:opacityThreshold = \(USDAWriter.number(cutoff))")
            }
        }
        writer.line("int inputs:useSpecularWorkflow = 0")
        writer.line("token outputs:surface")
        writer.close()

        for set in readers.sorted(by: { $0.rawValue < $1.rawValue }) {
            writer.line()
            writer.open("def Shader \"\(readerName(set))\"")
            writer.line("uniform token info:id = \"UsdPrimvarReader_float2\"")
            writer.line("string inputs:varname = \"\(set == .primary ? "st" : "st1")\"")
            writer.line("float2 outputs:result")
            writer.close()
        }
        for line in shaders.text.split(separator: "\n", omittingEmptySubsequences: false).dropLast() {
            writer.line(String(line))
        }
        writer.close()
    }

    private static func readerName(_ set: MeshTextureCoordinateSet) -> String {
        set == .primary ? "PrimvarReader_st" : "PrimvarReader_st1"
    }
}
