import Foundation

/// Inherited presentation attributes and inline paint styling.
struct SVGElementStyle {
    private var fill = "black"
    private var color = "black"
    private var stroke = "none"
    private var strokeWidth: CGFloat = 1
    private var strokeOpacity: Float = 1
    private var strokeCap: SVGLoader.Stroke.Cap = .butt
    private var strokeJoin: SVGLoader.Stroke.Join = .miter
    private var strokeMiterLimit: CGFloat = 4
    var fillRule: SVGLoader.FillRule = .nonzero
    var fillOpacity: Float = 1
    var opacity: Float = 1
    var isVisible = true
    var displayNone = false

    var fillPaint: SVGLoader.Paint { resolvedPaint(fill) }

    var strokeStyle: SVGLoader.Stroke? {
        let paint = resolvedPaint(stroke)
        guard paint != .none else { return nil }
        return SVGLoader.Stroke(
            paint: paint, width: strokeWidth, opacity: opacity * strokeOpacity,
            cap: strokeCap, join: strokeJoin, miterLimit: strokeMiterLimit
        )
    }

    private func resolvedPaint(_ source: String) -> SVGLoader.Paint {
        Self.paint(source.lowercased() == "currentcolor" ? color : source)
    }

    static func declarations(in text: Substring) -> [String: String] {
        var properties: [String: String] = [:]
        for declaration in text.split(separator: ";") {
            guard let colon = declaration.firstIndex(of: ":") else { continue }
            let key = declaration[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = declaration[declaration.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            if !key.isEmpty, !value.isEmpty { properties[key] = value }
        }
        return properties
    }

    static func properties(_ attributes: [String: String], stylesheet: [String: String]) -> [String: String] {
        var properties = attributes.merging(stylesheet) { _, styled in styled }
        if let inline = attributes["style"] {
            properties.merge(Self.declarations(in: inline[...])) { _, styled in styled }
        }
        return properties
    }

    func applying(
        _ attributes: [String: String], stylesheet: [String: String], dotsPerInch: Double
    ) -> SVGElementStyle {
        var result = self
        let properties = Self.properties(attributes, stylesheet: stylesheet)

        func value(_ key: String) -> String? {
            guard let value = properties[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !value.isEmpty, value.lowercased() != "inherit" else { return nil }
            return value
        }
        if let paint = value("fill") { result.fill = paint }
        if let foreground = value("color"), foreground.lowercased() != "currentcolor" { result.color = foreground }
        if let rule = value("fill-rule")?.lowercased() {
            if rule == "evenodd" { result.fillRule = .evenOdd }
            if rule == "nonzero" { result.fillRule = .nonzero }
        }
        if let alpha = value("fill-opacity").flatMap(Self.opacityValue) { result.fillOpacity = alpha }
        if let paint = value("stroke") { result.stroke = paint }
        if let width = value("stroke-width"),
           let length = try? SVGLength.parse(width, dotsPerInch: dotsPerInch, name: "stroke-width"), length >= 0 {
            result.strokeWidth = length
        }
        if let alpha = value("stroke-opacity").flatMap(Self.opacityValue) { result.strokeOpacity = alpha }
        switch value("stroke-linecap")?.lowercased() {
        case "butt": result.strokeCap = .butt
        case "round": result.strokeCap = .round
        case "square": result.strokeCap = .square
        default: break
        }
        switch value("stroke-linejoin")?.lowercased() {
        case "miter": result.strokeJoin = .miter
        case "bevel": result.strokeJoin = .bevel
        case "round": result.strokeJoin = .round
        case "miter-clip": result.strokeJoin = .miterClip
        case "arcs": result.strokeJoin = .unresolved("arcs")
        default: break
        }
        if let text = value("stroke-miterlimit"), let limit = Double(text),
           limit.isFinite, (1...SVGLoader.coordinateLimit).contains(limit) { result.strokeMiterLimit = limit }
        if let alpha = value("opacity").flatMap(Self.opacityValue) { result.opacity *= alpha }
        if let visibility = value("visibility")?.lowercased() {
            if visibility == "hidden" || visibility == "collapse" { result.isVisible = false }
            if visibility == "visible" { result.isVisible = true }
        }
        if value("display")?.lowercased() == "none" { result.displayNone = true }
        return result
    }

    static func opacityValue(_ text: String) -> Float? {
        channel(text, numberScale: 1)
    }

    private static func channel(_ text: String, numberScale: Float) -> Float? {
        let percent = text.hasSuffix("%")
        guard let value = Float(percent ? String(text.dropLast()) : text), value.isFinite else { return nil }
        return min(max(value / (percent ? 100 : numberScale), 0), 1)
    }

    static func paint(_ source: String) -> SVGLoader.Paint {
        let text = source.lowercased()
        if text == "none" { return .none }
        if text == "transparent" { return .solid(.zero) }
        let rgba: SIMD4<Float>?
        if text.hasPrefix("#") {
            let digits = text.dropFirst()
            if [3, 4, 6, 8].contains(digits.count), let packed = UInt32(digits, radix: 16) {
                let short = digits.count <= 4
                let step = short ? 4 : 8
                let hasAlpha = digits.count == 4 || digits.count == 8
                let scale: Float = short ? 15 : 255
                let mask: UInt32 = short ? 15 : 255
                func value(_ component: Int) -> Float {
                    Float((packed >> (component * step)) & mask) / scale
                }
                rgba = SIMD4(value(hasAlpha ? 3 : 2), value(hasAlpha ? 2 : 1), value(hasAlpha ? 1 : 0),
                             hasAlpha ? value(0) : 1)
            } else {
                rgba = nil
            }
        } else if let color = names[text] {
            rgba = SIMD4(Float(color >> 16) / 255, Float((color >> 8) & 255) / 255, Float(color & 255) / 255, 1)
        } else {
            rgba = functionColor(text)
        }
        guard let rgba else { return .unresolved(source) }
        return .solid(SIMD4(
            GLTFColorTables.sRGBToLinear(rgba.x), GLTFColorTables.sRGBToLinear(rgba.y),
            GLTFColorTables.sRGBToLinear(rgba.z), rgba.w
        ))
    }

    private static func functionColor(_ text: String) -> SIMD4<Float>? {
        guard let open = text.firstIndex(of: "("), text.last == ")" else { return nil }
        let name = text[..<open]
        let fields = text[text.index(after: open)..<text.index(before: text.endIndex)]
            .split { $0.isWhitespace || $0 == "," || $0 == "/" }.map(String.init)
        guard fields.count == 3 || fields.count == 4 else { return nil }
        guard let alpha = fields.count == 4 ? opacityValue(fields[3]) : 1 else { return nil }
        if name == "rgb" || name == "rgba" {
            guard let r = channel(fields[0], numberScale: 255), let g = channel(fields[1], numberScale: 255),
                  let b = channel(fields[2], numberScale: 255) else { return nil }
            return SIMD4(r, g, b, alpha)
        }
        if name == "hsl" || name == "hsla", fields[1].hasSuffix("%"), fields[2].hasSuffix("%") {
            let hue = fields[0]
            let number: Float?
            let toTurns: Float
            if hue.hasSuffix("turn") { number = Float(hue.dropLast(4)); toTurns = 1 }
            else if hue.hasSuffix("grad") { number = Float(hue.dropLast(4)); toTurns = 1 / 400 }
            else if hue.hasSuffix("rad") { number = Float(hue.dropLast(3)); toTurns = 1 / (2 * .pi) }
            else { number = Float(hue.hasSuffix("deg") ? String(hue.dropLast(3)) : hue); toTurns = 1 / 360 }
            guard let number, number.isFinite,
                  let s = channel(fields[1], numberScale: 1), let l = channel(fields[2], numberScale: 1) else { return nil }
            let normalized = number * toTurns
            let h = normalized - floor(normalized)
            func component(_ offset: Float) -> Float {
                let k = (offset + 12 * h).truncatingRemainder(dividingBy: 12)
                return l - s * min(l, 1 - l) * max(-1, min(k - 3, min(9 - k, 1)))
            }
            return SIMD4(component(0), component(8), component(4), alpha)
        }
        return nil
    }

    // CSS named colors as listed by Three.js r187 under the MIT license.
    private static let names: [String: UInt32] = [
        "aliceblue": 0xF0F8FF, "antiquewhite": 0xFAEBD7, "aqua": 0x00FFFF,
        "aquamarine": 0x7FFFD4, "azure": 0xF0FFFF, "beige": 0xF5F5DC,
        "bisque": 0xFFE4C4, "black": 0x000000, "blanchedalmond": 0xFFEBCD,
        "blue": 0x0000FF, "blueviolet": 0x8A2BE2, "brown": 0xA52A2A,
        "burlywood": 0xDEB887, "cadetblue": 0x5F9EA0, "chartreuse": 0x7FFF00,
        "chocolate": 0xD2691E, "coral": 0xFF7F50, "cornflowerblue": 0x6495ED,
        "cornsilk": 0xFFF8DC, "crimson": 0xDC143C, "cyan": 0x00FFFF,
        "darkblue": 0x00008B, "darkcyan": 0x008B8B, "darkgoldenrod": 0xB8860B,
        "darkgray": 0xA9A9A9, "darkgreen": 0x006400, "darkgrey": 0xA9A9A9,
        "darkkhaki": 0xBDB76B, "darkmagenta": 0x8B008B, "darkolivegreen": 0x556B2F,
        "darkorange": 0xFF8C00, "darkorchid": 0x9932CC, "darkred": 0x8B0000,
        "darksalmon": 0xE9967A, "darkseagreen": 0x8FBC8F, "darkslateblue": 0x483D8B,
        "darkslategray": 0x2F4F4F, "darkslategrey": 0x2F4F4F, "darkturquoise": 0x00CED1,
        "darkviolet": 0x9400D3, "deeppink": 0xFF1493, "deepskyblue": 0x00BFFF,
        "dimgray": 0x696969, "dimgrey": 0x696969, "dodgerblue": 0x1E90FF,
        "firebrick": 0xB22222, "floralwhite": 0xFFFAF0, "forestgreen": 0x228B22,
        "fuchsia": 0xFF00FF, "gainsboro": 0xDCDCDC, "ghostwhite": 0xF8F8FF,
        "gold": 0xFFD700, "goldenrod": 0xDAA520, "gray": 0x808080,
        "green": 0x008000, "greenyellow": 0xADFF2F, "grey": 0x808080,
        "honeydew": 0xF0FFF0, "hotpink": 0xFF69B4, "indianred": 0xCD5C5C,
        "indigo": 0x4B0082, "ivory": 0xFFFFF0, "khaki": 0xF0E68C,
        "lavender": 0xE6E6FA, "lavenderblush": 0xFFF0F5, "lawngreen": 0x7CFC00,
        "lemonchiffon": 0xFFFACD, "lightblue": 0xADD8E6, "lightcoral": 0xF08080,
        "lightcyan": 0xE0FFFF, "lightgoldenrodyellow": 0xFAFAD2, "lightgray": 0xD3D3D3,
        "lightgreen": 0x90EE90, "lightgrey": 0xD3D3D3, "lightpink": 0xFFB6C1,
        "lightsalmon": 0xFFA07A, "lightseagreen": 0x20B2AA, "lightskyblue": 0x87CEFA,
        "lightslategray": 0x778899, "lightslategrey": 0x778899, "lightsteelblue": 0xB0C4DE,
        "lightyellow": 0xFFFFE0, "lime": 0x00FF00, "limegreen": 0x32CD32,
        "linen": 0xFAF0E6, "magenta": 0xFF00FF, "maroon": 0x800000,
        "mediumaquamarine": 0x66CDAA, "mediumblue": 0x0000CD, "mediumorchid": 0xBA55D3,
        "mediumpurple": 0x9370DB, "mediumseagreen": 0x3CB371, "mediumslateblue": 0x7B68EE,
        "mediumspringgreen": 0x00FA9A, "mediumturquoise": 0x48D1CC, "mediumvioletred": 0xC71585,
        "midnightblue": 0x191970, "mintcream": 0xF5FFFA, "mistyrose": 0xFFE4E1,
        "moccasin": 0xFFE4B5, "navajowhite": 0xFFDEAD, "navy": 0x000080,
        "oldlace": 0xFDF5E6, "olive": 0x808000, "olivedrab": 0x6B8E23,
        "orange": 0xFFA500, "orangered": 0xFF4500, "orchid": 0xDA70D6,
        "palegoldenrod": 0xEEE8AA, "palegreen": 0x98FB98, "paleturquoise": 0xAFEEEE,
        "palevioletred": 0xDB7093, "papayawhip": 0xFFEFD5, "peachpuff": 0xFFDAB9,
        "peru": 0xCD853F, "pink": 0xFFC0CB, "plum": 0xDDA0DD,
        "powderblue": 0xB0E0E6, "purple": 0x800080, "rebeccapurple": 0x663399,
        "red": 0xFF0000, "rosybrown": 0xBC8F8F, "royalblue": 0x4169E1,
        "saddlebrown": 0x8B4513, "salmon": 0xFA8072, "sandybrown": 0xF4A460,
        "seagreen": 0x2E8B57, "seashell": 0xFFF5EE, "sienna": 0xA0522D,
        "silver": 0xC0C0C0, "skyblue": 0x87CEEB, "slateblue": 0x6A5ACD,
        "slategray": 0x708090, "slategrey": 0x708090, "snow": 0xFFFAFA,
        "springgreen": 0x00FF7F, "steelblue": 0x4682B4, "tan": 0xD2B48C,
        "teal": 0x008080, "thistle": 0xD8BFD8, "tomato": 0xFF6347,
        "turquoise": 0x40E0D0, "violet": 0xEE82EE, "wheat": 0xF5DEB3,
        "white": 0xFFFFFF, "whitesmoke": 0xF5F5F5, "yellow": 0xFFFF00,
        "yellowgreen": 0x9ACD32,
    ]
}
