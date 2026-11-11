import CoreGraphics
import Foundation
import simd

/// Imports two-dimensional SVG paths in their authored coordinate system.
public enum SVGLoader {
    public enum FillRule: Equatable, Sendable {
        case nonzero, evenOdd
    }

    public enum Paint: Equatable, Sendable {
        case none
        /// Straight-alpha linear sRGB.
        case solid(SIMD4<Float>)
        case gradient(Gradient)
        /// Paint syntax retained for the caller when it cannot be resolved.
        case unresolved(String)
    }

    public struct Stroke: Equatable, Sendable {
        public enum Cap: Equatable, Sendable {
            case butt, round, square

            var native: CGLineCap {
                switch self {
                case .butt: .butt
                case .round: .round
                case .square: .square
                }
            }
        }

        public enum Join: Equatable, Sendable {
            case miter, miterClip, round, bevel
            case unresolved(String)

            var native: CGLineJoin {
                get throws {
                    switch self {
                    case .miter, .miterClip: .miter
                    case .round: .round
                    case .bevel: .bevel
                    case .unresolved(let name): throw Failure.strokeJoin(name)
                    }
                }
            }
        }

        public enum TextureCoordinates: Equatable, Sendable {
            /// Planar UVs normalized over the stroke outline, including merged overlaps.
            case planar
            /// Per-subpath UVs with U along the tessellated path and V from left to right.
            case path
        }

        public let paint: Paint
        /// Width in local SVG units, before the element's affine transform.
        public let width: CGFloat
        public let opacity: Float
        public let cap: Cap
        public let join: Join
        public let miterLimit: CGFloat
    }

    public struct Element {
        public let id: String?
        /// An immutable path in SVG document coordinates, with positive Y down.
        public let path: CGPath
        public let fill: Paint
        public let fillRule: FillRule
        /// The combined element and ancestor opacity multiplied by fill opacity.
        public let fillOpacity: Float
        public let isVisible: Bool
        public let stroke: Stroke?
        let strokeSource: (path: CGPath, transform: CGAffineTransform)?
        let gradientSource: (bounds: CGRect, transform: CGAffineTransform)?

        /// Fill regions using the element's authored rule. Open subpaths close
        /// for filling. `yUp` reflects Y around the document origin.
        public func filledShapes(tolerance: Float = 0.25, yUp: Bool = false) throws -> [PlanarShape] {
            guard isVisible, fill != .none, fillOpacity > 0 else { return [] }
            if case .solid(let rgba) = fill, rgba.w == 0 { return [] }
            return try Self.resolve(path, rule: fillRule, tolerance: tolerance, yUp: yUp)
        }

        /// Triangulated XY-plane fill. Apply the element's paint and fill opacity
        /// to its material. Empty and invisible fills return nil.
        public func fillGeometry(tolerance: Float = 0.25, yUp: Bool = false) throws -> MeshGeometry? {
            let shapes = try filledShapes(tolerance: tolerance, yUp: yUp)
            return shapes.isEmpty ? nil : try MeshGeometry.shape(shapes)
        }

        /// Outlines the stroke in local coordinates before applying the SVG transform.
        /// This preserves cap and join geometry under nonuniform scale and skew.
        public func strokedShapes(tolerance: Float = 0.25, yUp: Bool = false) throws -> [PlanarShape] {
            guard let (stroke, strokeSource) = try visibleStroke() else { return [] }
            if stroke.join == .miterClip {
                return try SVGStrokeTessellator(
                    path: strokeSource.path, transform: strokeSource.transform,
                    stroke: stroke, tolerance: tolerance, yUp: yUp
                ).outline()
            }
            let outline = try strokeSource.path.copy(
                strokingWithWidth: stroke.width, lineCap: stroke.cap.native,
                lineJoin: stroke.join.native, miterLimit: stroke.miterLimit
            )
            guard !outline.boundingBoxOfPath.isNull else { return [] }
            var transform = strokeSource.transform
            guard let path = outline.copy(using: &transform) else { throw Failure.geometry }
            guard path.isEmpty || SVGLoader.isRepresentable(path) else { throw Failure.geometry }
            return try Self.resolve(path, rule: .nonzero, tolerance: tolerance, yUp: yUp)
        }

        /// Triangulated XY-plane stroke outline. Apply the stroke paint and opacity
        /// to the material, as for a fill.
        public func strokeGeometry(tolerance: Float = 0.25, yUp: Bool = false) throws -> MeshGeometry? {
            let shapes = try strokedShapes(tolerance: tolerance, yUp: yUp)
            return shapes.isEmpty ? nil : try MeshGeometry.shape(shapes)
        }

        /// Draws the stroke with planar UVs or with UVs that follow each subpath.
        /// Path coordinates reset between subpaths and can overlap at crossings.
        public func strokeGeometry(
            textureCoordinates: Stroke.TextureCoordinates, tolerance: Float = 0.25, yUp: Bool = false
        ) throws -> MeshGeometry? {
            if textureCoordinates == .planar { return try strokeGeometry(tolerance: tolerance, yUp: yUp) }
            guard let (stroke, strokeSource) = try visibleStroke() else { return nil }
            return try SVGStrokeTessellator(
                path: strokeSource.path, transform: strokeSource.transform,
                stroke: stroke, tolerance: tolerance, yUp: yUp
            ).geometry()
        }

        private func visibleStroke() throws -> (Stroke, (path: CGPath, transform: CGAffineTransform))? {
            guard isVisible, let stroke, let strokeSource, stroke.width > 0, stroke.opacity > 0 else { return nil }
            if case .solid(let rgba) = stroke.paint, rgba.w == 0 { return nil }
            var commandsLeft = SVGLoader.maximumPathCommands
            strokeSource.path.applyWithBlock { _ in commandsLeft -= 1 }
            guard commandsLeft >= 0 else { throw Failure.size }
            return (stroke, strokeSource)
        }

        private static func resolve(
            _ path: CGPath, rule: FillRule, tolerance: Float, yUp: Bool
        ) throws -> [PlanarShape] {
            guard tolerance.isFinite, tolerance > 0 else { throw PlanarShape.Failure.invalidTolerance }
            var reflection = CGAffineTransform(scaleX: 1, y: -1)
            let input = yUp ? path.copy(using: &reflection) : path
            guard let input else { throw Failure.geometry }
            do {
                let rings = try PlanarShape.flatten(input, tolerance: Double(tolerance), maximumPoints: 4_096)
                return try PlanarShape.shapes(
                    rings: rings, fillRule: rule == .evenOdd ? .evenOdd : .winding, maximumEdges: 8_192
                )
            } catch is PlanarComplexityLimit {
                throw Failure.size
            }
        }
    }

    public struct Document {
        /// Authored viewport metadata. Paths retain source units.
        public let viewBox: CGRect?
        /// Paths in document order, including stroke-only paths.
        public let elements: [Element]
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL, size, xml, geometry, transform, invalidDPI
        case unsupportedXML
        case pathData(Int)
        case attribute(String)
        case strokeJoin(String)
        case referenceCycle
        case gradient(String)

        public var errorDescription: String? {
            switch self {
            case .fileURL: "SVG import requires a regular local file."
            case .size: "The SVG document, path or fill exceeds the supported size."
            case .xml: "The SVG document is malformed."
            case .geometry: "The SVG path cannot be represented in the supported coordinate range."
            case .transform: "The SVG transform list is malformed."
            case .invalidDPI: "SVG dots per inch must be finite and positive."
            case .unsupportedXML: "Use UTF-8 or ASCII SVG with no custom DTD or entity declarations."
            case .pathData(let offset): "SVG path data is malformed near byte \(offset)."
            case .attribute(let name): "The SVG \(name) attribute is invalid."
            case .strokeJoin(let name): "The SVG stroke join \(name) cannot be outlined."
            case .referenceCycle: "SVG elements contain a cyclic local reference."
            case .gradient(let reason): "The SVG gradient is invalid. \(reason)."
            }
        }
    }

    private static let maximumBytes = 8 * 1_024 * 1_024
    static let maximumPathCommands = 16_384
    static let coordinateLimit = 1e9

    public static func load(contentsOf url: URL, dotsPerInch: Double = 90) throws -> Document {
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
        return try load(data, dotsPerInch: dotsPerInch)
    }

    public static func load(_ data: Data, dotsPerInch: Double = 90) throws -> Document {
        guard dotsPerInch.isFinite, dotsPerInch > 0 else { throw Failure.invalidDPI }
        guard data.count <= maximumBytes else { throw Failure.size }
        let xml = try validatedXML(data)
        let delegate = SVGDocumentReader(dotsPerInch: dotsPerInch)
        let parser = XMLParser(data: xml)
        parser.delegate = delegate
        parser.shouldProcessNamespaces = true
        parser.shouldResolveExternalEntities = false
        parser.externalEntityResolvingPolicy = .never
        let parsed = parser.parse()
        if let failure = delegate.failure { throw failure }
        guard parsed else { throw Failure.xml }
        return try delegate.document()
    }

    public static func load(_ text: String, dotsPerInch: Double = 90) throws -> Document {
        guard text.utf8.count <= maximumBytes else { throw Failure.size }
        return try load(Data(text.utf8), dotsPerInch: dotsPerInch)
    }

    fileprivate static func isRepresentable(_ path: CGPath) -> Bool {
        let bounds = path.boundingBox
        return [bounds.minX, bounds.minY, bounds.maxX, bounds.maxY]
            .allSatisfy { $0.isFinite && abs($0) <= coordinateLimit }
    }

    private static func validatedXML(_ data: Data) throws -> Data {
        guard !data.contains(0) else { throw Failure.unsupportedXML }
        let content = data.starts(with: [0xef, 0xbb, 0xbf]) ? data.dropFirst(3) : data
        if content.starts(with: "<?xml".utf8), let next = content.dropFirst(5).first, SVGNumbers.isSpace(next) {
            let prefix = content.prefix(1_024)
            guard let end = prefix.range(of: Data("?>".utf8)) else { throw Failure.xml }
            let declaration = String(decoding: prefix[..<end.lowerBound], as: UTF8.self)
            if let field = declaration.range(of: "encoding") {
                let encoding = declaration[field.upperBound...].split { $0.isWhitespace || $0 == "=" }.first
                let allowed = ["\"utf-8\"", "'utf-8'", "\"us-ascii\"", "'us-ascii'"]
                guard let encoding, allowed.contains(encoding.lowercased()) else { throw Failure.unsupportedXML }
            }
        }

        // Legacy artwork can carry the SVG 1.1 public identifier. Remove that
        // exact header before passing any bytes to the XML parser.
        let keyword = Data("<!DOCTYPE".utf8)
        guard let start = data.range(of: keyword)?.lowerBound else { return data }
        let prefix = data[..<start]
        let tail = data[start...].prefix(512)
        guard prefix.count <= 1_024, let end = tail.firstIndex(of: 0x3e),
              let header = String(data: data[start...end], encoding: .ascii) else {
            throw Failure.unsupportedXML
        }
        let prolog = String(decoding: prefix, as: UTF8.self)
        let prologPattern = #"\A\uFEFF?(?:<\?xml[ \t\r\n]+[^?]*\?>)?[ \t\r\n]*\z"#
        let headerPattern = #"\A<!DOCTYPE[ \t\r\n]+svg[ \t\r\n]+PUBLIC[ \t\r\n]+(["'])-//W3C//DTD SVG 1\.1//EN\1[ \t\r\n]+(["'])https?://www\.w3\.org/Graphics/SVG/1\.1/DTD/svg11\.dtd\2[ \t\r\n]*>\z"#
        guard prolog.range(of: prologPattern, options: .regularExpression) != nil,
              header.range(of: headerPattern, options: .regularExpression) != nil else {
            throw Failure.unsupportedXML
        }
        var sanitized = data
        sanitized.removeSubrange(start...end)
        guard sanitized.range(of: keyword) == nil else { throw Failure.unsupportedXML }
        return sanitized
    }
}

private final class SVGDocumentReader: NSObject, XMLParserDelegate {
    private struct Node {
        let name: String
        let attributes: [String: String]
        let isSVG: Bool
        var children: [Int] = []
        var css: String?
    }

    private struct Frame {
        var transform = CGAffineTransform.identity
        var style = SVGElementStyle()
    }

    private let dotsPerInch: Double
    private var nodes: [Node] = []
    private var stack: [Int] = []
    private var ids: [String: Int] = [:]
    private var stylesheet = SVGStyleSheet()
    private var gradientDefinitions: [String: SVGGradientDefinition] = [:]
    private var inheritedGradients: [String: SVGGradientDefinition] = [:]
    private var gradients: [String: SVGLoader.Gradient] = [:]
    private var gradientViewport: CGRect?
    private var cssBytesLeft = 256 * 1_024
    private var visitsLeft = 65_536
    private var pathElementsLeft = 262_144
    private var elements: [SVGLoader.Element] = []
    private var viewBox: CGRect?
    private(set) var failure: Error?

    init(dotsPerInch: Double) { self.dotsPerInch = dotsPerInch }

    func parser(
        _ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
        qualifiedName: String?, attributes: [String: String]
    ) {
        do {
            guard nodes.count < 32_768, stack.count < 64 else { throw SVGLoader.Failure.size }
            let namespaceMatches = namespaceURI == nil || namespaceURI == "" || namespaceURI == "http://www.w3.org/2000/svg"
            let isSVG = namespaceMatches && (stack.last.map { nodes[$0].isSVG } ?? true)
            if stack.isEmpty {
                guard nodes.isEmpty, isSVG, name == "svg" else { throw SVGLoader.Failure.xml }
                if let text = attributes["viewBox"] {
                    let values = try Self.values(text, maximum: 4)
                    guard values.count == 4, values[2] >= 0, values[3] >= 0 else {
                        throw SVGLoader.Failure.attribute("viewBox")
                    }
                    viewBox = CGRect(x: values[0], y: values[1], width: values[2], height: values[3])
                }
            }

            let index = nodes.count
            let css = isSVG && name == "style" && (attributes["type"] ?? "text/css") == "text/css" ? "" : nil
            nodes.append(Node(name: name, attributes: attributes, isSVG: isSVG, css: css))
            if let parent = stack.last { nodes[parent].children.append(index) }
            if isSVG, let id = attributes["id"], !id.isEmpty, ids[id] == nil { ids[id] = index }
            stack.append(index)
        } catch {
            failure = error
            parser.abortParsing()
        }
    }

    func parser(_ parser: XMLParser, foundCharacters text: String) {
        guard let index = stack.last, nodes[index].css != nil else { return }
        let size = text.utf8.count
        guard size <= cssBytesLeft else {
            failure = SVGLoader.Failure.size
            parser.abortParsing()
            return
        }
        cssBytesLeft -= size
        nodes[index].css?.append(text)
    }

    func parser(_ parser: XMLParser, foundCDATA block: Data) {
        self.parser(parser, foundCharacters: String(decoding: block, as: UTF8.self))
    }

    func parser(
        _ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?
    ) {
        _ = stack.popLast()
    }

    func document() throws -> SVGLoader.Document {
        guard !nodes.isEmpty else { throw SVGLoader.Failure.xml }
        for node in nodes {
            if let css = node.css { try stylesheet.add(css) }
        }
        try readGradientDefinitions()
        gradientViewport = viewBox
        if gradientViewport == nil, let width = try? length("width", nodes[0].attributes),
           let height = try? length("height", nodes[0].attributes), width > 0, height > 0 {
            gradientViewport = CGRect(x: 0, y: 0, width: width, height: height)
        }
        var activeReferences: Set<Int> = []
        try render(0, from: Frame(), depth: 0, activeReferences: &activeReferences)
        return SVGLoader.Document(viewBox: viewBox, elements: elements)
    }

    private func render(
        _ index: Int, from parent: Frame, depth: Int,
        activeReferences: inout Set<Int>, referenceRoot: Bool = false
    ) throws {
        guard depth < 64, visitsLeft > 0 else { throw SVGLoader.Failure.size }
        visitsLeft -= 1
        let node = nodes[index]
        let name = node.name
        guard node.isSVG, !Self.definitionNames.contains(name) || (name == "symbol" && referenceRoot) else { return }

        let attributes = node.attributes
        let local = try attributes["transform"].map(Self.transform) ?? .identity
        var frame = parent
        frame.transform = local.concatenating(parent.transform)
        guard Self.isFinite(frame.transform) else { throw SVGLoader.Failure.transform }
        frame.style = frame.style.applying(
            attributes, stylesheet: stylesheet.declarations(for: attributes), dotsPerInch: dotsPerInch
        )
        guard !frame.style.displayNone else { return }

        if name == "use",
           let href = attributes["xlink:href"] ?? attributes["href"], href.first == "#",
           let target = ids[String(href.dropFirst())] {
            guard activeReferences.insert(target).inserted else { throw SVGLoader.Failure.referenceCycle }
            defer { activeReferences.remove(target) }
            let translation = try CGAffineTransform(translationX: length("x", attributes), y: length("y", attributes))
            frame.transform = translation.concatenating(frame.transform)
            guard Self.isFinite(frame.transform) else { throw SVGLoader.Failure.transform }
            try render(target, from: frame, depth: depth + 1, activeReferences: &activeReferences, referenceRoot: true)
        } else if let localPath = try makePath(name, attributes: attributes), !localPath.isEmpty {
            var transform = frame.transform
            guard let path = localPath.copy(using: &transform) else { throw SVGLoader.Failure.geometry }
            try append(path, localPath: localPath, transform: transform, attributes: attributes, style: frame.style)
        }
        for child in node.children {
            try render(child, from: frame, depth: depth + 1, activeReferences: &activeReferences)
        }
    }

    private func readGradientDefinitions() throws {
        var stopsLeft = 16_384
        for node in nodes where node.isSVG && (node.name == "linearGradient" || node.name == "radialGradient") {
            guard let id = node.attributes["id"], !id.isEmpty, gradientDefinitions[id] == nil else { continue }
            guard gradientDefinitions.count < 2_048 else { throw SVGLoader.Failure.size }
            let properties = SVGElementStyle.properties(node.attributes, stylesheet: stylesheet.declarations(for: node.attributes))
            var stops: [SVGLoader.Gradient.Stop] = []
            var lastOffset: Float = 0
            for index in node.children {
                let stop = nodes[index]
                guard stop.isSVG, stop.name == "stop" else { continue }
                guard stopsLeft > 0, stops.count < 1_024 else { throw SVGLoader.Failure.size }
                stopsLeft -= 1
                let fields = SVGElementStyle.properties(stop.attributes, stylesheet: stylesheet.declarations(for: stop.attributes))
                let offset = SVGElementStyle.opacityValue(fields["offset"] ?? "0") ?? 0
                lastOffset = max(lastOffset, offset)
                let paint = fields["stop-color"] ?? "black"
                let color = paint.lowercased() == "currentcolor" ? (fields["color"] ?? properties["color"] ?? "black") : paint
                let rgba: SIMD4<Float>
                if case .solid(let value) = SVGElementStyle.paint(color) { rgba = value }
                else { rgba = SIMD4(0, 0, 0, 1) }
                let alpha = SVGElementStyle.opacityValue(fields["stop-opacity"] ?? "1") ?? 1
                stops.append(SVGLoader.Gradient.Stop(
                    offset: lastOffset, color: SIMD4(rgba.x, rgba.y, rgba.z, rgba.w * alpha)
                ))
            }
            let href = node.attributes["xlink:href"] ?? node.attributes["href"]
            let reference = href?.first == "#" ? href.map { String($0.dropFirst()) } : nil
            gradientDefinitions[id] = SVGGradientDefinition(
                radial: node.name == "radialGradient", properties: properties, stops: stops, reference: reference
            )
        }
    }

    private func inheritedGradient(
        _ id: String, active: inout Set<String>
    ) throws -> SVGGradientDefinition? {
        if let cached = inheritedGradients[id] { return cached }
        guard var result = gradientDefinitions[id] else { return nil }
        guard active.count < 64 else { throw SVGLoader.Failure.size }
        guard active.insert(id).inserted else { throw SVGLoader.Failure.referenceCycle }
        defer { active.remove(id) }
        if let reference = result.reference, let base = try inheritedGradient(reference, active: &active) {
            result.properties.merge(base.properties) { explicit, _ in explicit }
            if result.stops.isEmpty { result.stops = base.stops }
        }
        inheritedGradients[id] = result
        return result
    }

    private func fillPaint(_ paint: SVGLoader.Paint) throws -> SVGLoader.Paint {
        guard case .unresolved(let source) = paint,
              let id = SVGGradientDefinition.identifier(in: source) else { return paint }
        if let gradient = gradients[id] { return .gradient(gradient) }
        var active: Set<String> = []
        guard let definition = try inheritedGradient(id, active: &active),
              let gradient = try definition.gradient(
                viewport: gradientViewport, dotsPerInch: dotsPerInch, parseTransform: Self.transform
              ) else { return paint }
        gradients[id] = gradient
        return .gradient(gradient)
    }

    private func append(
        _ path: CGPath, localPath: CGPath, transform: CGAffineTransform,
        attributes: [String: String], style: SVGElementStyle
    ) throws {
        guard SVGLoader.isRepresentable(path) else { throw SVGLoader.Failure.geometry }
        path.applyWithBlock { _ in pathElementsLeft -= 1 }
        guard pathElementsLeft >= 0 else { throw SVGLoader.Failure.size }
        let stroke = style.strokeStyle
        let paint = try fillPaint(style.fillPaint)
        let gradientSource: (CGRect, CGAffineTransform)?
        if case .gradient = paint { gradientSource = (localPath.boundingBoxOfPath, transform) }
        else { gradientSource = nil }
        elements.append(SVGLoader.Element(
            id: attributes["id"], path: path, fill: paint, fillRule: style.fillRule,
            fillOpacity: style.opacity * style.fillOpacity, isVisible: style.isVisible,
            stroke: stroke, strokeSource: stroke.map { _ in (localPath, transform) }, gradientSource: gradientSource
        ))
    }

    private static let definitionNames: Set<String> = [
        "defs", "symbol", "clipPath", "mask", "marker", "pattern", "linearGradient", "radialGradient",
        "script", "style", "foreignObject", "metadata",
    ]

    private func length(_ name: String, _ attributes: [String: String]) throws -> CGFloat {
        guard let text = attributes[name] else { return 0 }
        return try SVGLength.parse(text, dotsPerInch: dotsPerInch, name: name)
    }

    private func makePath(_ name: String, attributes: [String: String]) throws -> CGPath? {
        func length(_ name: String) throws -> CGFloat { try self.length(name, attributes) }

        switch name {
        case "path":
            return try attributes["d"].map(SVGPathData.parse)
        case "rect":
            let width = try length("width"), height = try length("height")
            guard width >= 0, height >= 0 else { throw SVGLoader.Failure.attribute("rect size") }
            guard width > 0, height > 0 else { return nil }
            let box = try CGRect(x: length("x"), y: length("y"), width: width, height: height)
            let rx = try length("rx"), ry = try length("ry")
            guard rx >= 0, ry >= 0 else { throw SVGLoader.Failure.attribute("rect radius") }
            let x = min(width / 2, attributes["rx"] == nil ? ry : rx)
            let y = min(height / 2, attributes["ry"] == nil ? rx : ry)
            return x > 0 && y > 0 ? CGPath(roundedRect: box, cornerWidth: x, cornerHeight: y, transform: nil)
                                  : CGPath(rect: box, transform: nil)
        case "circle", "ellipse":
            let rx = try length(name == "circle" ? "r" : "rx")
            let ry = try name == "circle" ? rx : length("ry")
            guard rx >= 0, ry >= 0 else { throw SVGLoader.Failure.attribute(name + " radius") }
            guard rx > 0, ry > 0 else { return nil }
            let box = try CGRect(x: length("cx") - rx, y: length("cy") - ry, width: 2 * rx, height: 2 * ry)
            return CGPath(ellipseIn: box, transform: nil)
        case "line":
            let path = CGMutablePath()
            path.move(to: try CGPoint(x: length("x1"), y: length("y1")))
            path.addLine(to: try CGPoint(x: length("x2"), y: length("y2")))
            return path
        case "polyline", "polygon":
            guard let points = attributes["points"] else { return nil }
            let values = try Self.values(points, maximum: SVGLoader.maximumPathCommands * 2)
            guard values.count.isMultiple(of: 2) else { throw SVGLoader.Failure.attribute("points") }
            guard values.count >= 2 else { return nil }
            let path = CGMutablePath()
            path.move(to: CGPoint(x: values[0], y: values[1]))
            for i in stride(from: 2, to: values.count, by: 2) {
                path.addLine(to: CGPoint(x: values[i], y: values[i + 1]))
            }
            if name == "polygon" { path.closeSubpath() }
            return path
        default:
            return nil
        }
    }

    private static func values(_ text: String, maximum: Int) throws -> [Double] {
        var reader = SVGNumbers(text)
        var values: [Double] = []
        while true {
            reader.skipSpace()
            guard reader.next != nil else { return values }
            guard values.count < maximum else { throw SVGLoader.Failure.size }
            values.append(try reader.number())
        }
    }

    private static func transform(_ text: String) throws -> CGAffineTransform {
        var reader = SVGNumbers(text)
        var result = CGAffineTransform.identity
        var remaining = 128
        while true {
            reader.skipSeparator()
            guard reader.next != nil else { return result }
            guard remaining > 0 else { throw SVGLoader.Failure.size }
            remaining -= 1
            let name = reader.word()
            reader.skipSpace()
            guard reader.take(40) else { throw SVGLoader.Failure.transform }
            var values: [Double] = []
            while true {
                reader.skipSpace()
                if reader.take(41) { break }
                guard values.count < 6 else { throw SVGLoader.Failure.transform }
                values.append(try reader.number())
            }
            let next: CGAffineTransform
            switch (name, values.count) {
            case ("matrix", 6):
                next = CGAffineTransform(a: values[0], b: values[1], c: values[2], d: values[3], tx: values[4], ty: values[5])
            case ("translate", 1), ("translate", 2):
                next = CGAffineTransform(translationX: values[0], y: values.count == 1 ? 0 : values[1])
            case ("scale", 1), ("scale", 2):
                next = CGAffineTransform(scaleX: values[0], y: values.count == 1 ? values[0] : values[1])
            case ("rotate", 1), ("rotate", 3):
                let rotation = CGAffineTransform(rotationAngle: values[0] * .pi / 180)
                if values.count == 3 {
                    let x = values[1], y = values[2]
                    next = CGAffineTransform(translationX: -x, y: -y).concatenating(rotation)
                        .concatenating(CGAffineTransform(translationX: x, y: y))
                } else {
                    next = rotation
                }
            case ("skewX", 1):
                next = CGAffineTransform(a: 1, b: 0, c: tan(values[0] * .pi / 180), d: 1, tx: 0, ty: 0)
            case ("skewY", 1):
                next = CGAffineTransform(a: 1, b: tan(values[0] * .pi / 180), c: 0, d: 1, tx: 0, ty: 0)
            default: throw SVGLoader.Failure.transform
            }
            result = next.concatenating(result)
            guard isFinite(result) else { throw SVGLoader.Failure.transform }
        }
    }

    private static func isFinite(_ transform: CGAffineTransform) -> Bool {
        [transform.a, transform.b, transform.c, transform.d, transform.tx, transform.ty]
            .allSatisfy { $0.isFinite && abs($0) <= SVGLoader.coordinateLimit }
    }
}
