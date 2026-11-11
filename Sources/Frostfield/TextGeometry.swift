import CoreGraphics
import CoreText
import Foundation
import simd

/// Glyph outlines of text laid out by Core Text, as planar shapes in world units.
///
/// Each line is a `CTLine`, so kerning, ligatures, contextual forms, font
/// fallback and bidirectional ordering match what Core Text draws. Outlines
/// come from `CTFontCreatePathForGlyph`, so any installed, registered or
/// bundled font with vector glyphs works without conversion. Curves are
/// flattened to `curveTolerance` and every glyph's contours are resolved
/// with the nonzero rule, which removes the overlapping contours common in
/// variable fonts and finds each counter's hole.
///
/// Coordinates are Y-up. Horizontal lines start at Y = 0 and stack down.
/// Vertical columns start at X = 0 and stack to the left. In either direction
/// the line metrics and `lineSpacing` set the distance between baselines.
public struct TextOutline: Sendable {
    /// The direction of each explicitly separated line of text.
    public enum Direction: Sendable, Equatable {
        /// Horizontal lines, with Core Text's bidirectional shaping.
        case horizontal
        /// Vertical glyphs progress toward negative Y using Core Text's
        /// vertical forms and metrics. Line separators start columns to the
        /// left. Consecutive separators reserve a blank column.
        case topToBottom
    }

    /// Placement of each line relative to the inline origin, X = 0 for
    /// horizontal text and Y = 0 for top-to-bottom text.
    public enum Alignment: Sendable, Equatable {
        /// Lines start at the inline origin.
        case leading
        /// Each line's typographic advance, without trailing whitespace, is centered on the inline origin.
        case center
        /// Lines end at the inline origin, ignoring trailing whitespace.
        case trailing
    }

    /// One glyph's filled outline.
    public struct Glyph: Sendable {
        /// The UTF-16 offset of the first character the glyph represents.
        public let characterIndex: Int
        /// The zero-based line or column that contains the glyph.
        public let line: Int
        /// The resolved regions in world units, empty for glyphs such as spaces.
        public let shapes: [PlanarShape]
    }

    /// Typographic metrics of one laid-out line or column, in world units.
    public struct Line: Sendable, Equatable {
        /// The start of the baseline after alignment.
        public let origin: SIMD2<Float>
        /// The typographic advance along the line, including trailing
        /// whitespace. For a vertical column this measures its length in Y.
        public let width: Float
        /// Distance above a horizontal baseline or right of a vertical one.
        public let ascent: Float
        /// Distance below a horizontal baseline or left of a vertical one.
        public let descent: Float
        public let leading: Float
    }

    public let glyphs: [Glyph]
    public let lines: [Line]
    public let direction: Direction

    /// Every glyph's shapes, in layout order.
    public var shapes: [PlanarShape] { glyphs.flatMap(\.shapes) }

    /// Lays out a string in one font. `size` is the world height of the
    /// font's em, so one point maps to `size / CTFontGetSize(font)` units; the
    /// font's own point size still selects its optical size. The default curve
    /// tolerance is 1/1,000 of the em.
    public init(
        _ string: String, font: CTFont, size: Float = 1, alignment: Alignment = .leading,
        lineSpacing: Float = 1, curveTolerance: Float? = nil
    ) throws {
        try self.init(
            string, font: font, size: size, direction: .horizontal, alignment: alignment,
            lineSpacing: lineSpacing, curveTolerance: curveTolerance
        )
    }

    /// Lays out a string in the chosen direction with the same size, spacing
    /// and outline tolerance conventions as horizontal text.
    public init(
        _ string: String, font: CTFont, size: Float = 1, direction: Direction,
        alignment: Alignment = .leading, lineSpacing: Float = 1, curveTolerance: Float? = nil
    ) throws {
        guard size.isFinite, size > 0 else { throw Failure.invalidSize }
        let attributes = [NSAttributedString.Key(kCTFontAttributeName as String): font]
        let text = NSAttributedString(string: string, attributes: attributes)
        let unitsPerPoint = Double(size) / Double(CTFontGetSize(font))
        try self.init(
            text, unitsPerPoint: Float(unitsPerPoint), direction: direction, alignment: alignment, lineSpacing: lineSpacing,
            curveTolerance: curveTolerance ?? size / 1_000
        )
    }

    /// Lays out an attributed string whose ranges may use different fonts and
    /// Core Text attributes such as `kCTKernAttributeName` or
    /// `kCTLigatureAttributeName`. Ranges without a font use Core Text's
    /// default. Lines break only at line separators. The default curve
    /// tolerance is 1/1,000 of the largest font size.
    public init(
        _ attributedString: NSAttributedString, unitsPerPoint: Float, alignment: Alignment = .leading,
        lineSpacing: Float = 1, curveTolerance: Float? = nil
    ) throws {
        try self.init(
            attributedString, unitsPerPoint: unitsPerPoint, direction: .horizontal,
            alignment: alignment, lineSpacing: lineSpacing, curveTolerance: curveTolerance
        )
    }

    /// Lays out attributed text in the chosen direction. Top-to-bottom
    /// layout enables vertical forms on every run in an internal copy.
    public init(
        _ attributedString: NSAttributedString, unitsPerPoint: Float, direction: Direction,
        alignment: Alignment = .leading, lineSpacing: Float = 1, curveTolerance: Float? = nil
    ) throws {
        guard unitsPerPoint.isFinite, unitsPerPoint > 0 else { throw Failure.invalidSize }
        guard lineSpacing.isFinite, lineSpacing > 0 else { throw Failure.invalidLineSpacing }
        if let curveTolerance, !(curveTolerance.isFinite && curveTolerance > 0) { throw Failure.invalidTolerance }

        let isVertical = direction == .topToBottom
        let verticalForms = NSAttributedString.Key(kCTVerticalFormsAttributeName as String)
        let layoutText: NSAttributedString
        if isVertical {
            let copy = NSMutableAttributedString(attributedString: attributedString)
            copy.addAttribute(verticalForms, value: true, range: NSRange(location: 0, length: copy.length))
            layoutText = copy
        } else {
            layoutText = attributedString
        }
        let string = layoutText.string as NSString
        var ranges: [NSRange] = []
        string.enumerateSubstrings(in: NSRange(location: 0, length: string.length), options: [.byLines, .substringNotRequired]) {
            _, range, _, _ in ranges.append(range)
        }
        if ranges.isEmpty { ranges = [NSRange(location: 0, length: 0)] }

        let scale = Double(unitsPerPoint)
        var laidOut: [(line: CTLine, range: NSRange)] = []
        var largestFontSize = 0.0
        for range in ranges {
            let line = CTLineCreateWithAttributedString(layoutText.attributedSubstring(from: range) as CFAttributedString)
            for run in CTLineGetGlyphRuns(line) as! [CTRun] {
                largestFontSize = max(largestFontSize, Double(CTFontGetSize(try Self.font(of: run))))
            }
            laidOut.append((line, range))
        }
        let tolerance = curveTolerance ?? Float(max(largestFontSize, 1) * scale / 1_000)

        var glyphs: [Glyph] = []
        var lines: [Line] = []
        var baseline = 0.0
        var previousSpacing = 0.0
        // Core Text's vertical run origins already progress down Y. Its run
        // matrix also carries a quarter turn into CTLine's horizontal metric
        // space. Undo that turn for outlines in the native drawing plane.
        let verticalRunToOutline = CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: 0)
        for (lineIndex, entry) in laidOut.enumerated() {
            var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
            let width = CTLineGetTypographicBounds(entry.line, &ascent, &descent, &leading)
            if entry.range.length == 0 {
                let font = Self.font(at: entry.range.location, in: attributedString)
                if isVertical {
                    var attributes: [NSAttributedString.Key: Any] = [verticalForms: true]
                    if let font { attributes[NSAttributedString.Key(kCTFontAttributeName as String)] = font }
                    let space = NSAttributedString(string: " ", attributes: attributes)
                    _ = CTLineGetTypographicBounds(CTLineCreateWithAttributedString(space), &ascent, &descent, &leading)
                } else if let font {
                    ascent = CTFontGetAscent(font)
                    descent = CTFontGetDescent(font)
                    leading = CTFontGetLeading(font)
                }
            }
            if lineIndex > 0 { baseline -= (previousSpacing + Double(ascent)) * Double(lineSpacing) }
            previousSpacing = Double(descent + leading)
            let visible = width - CTLineGetTrailingWhitespaceWidth(entry.line)
            let start: Double = switch alignment {
            case .leading: 0
            case .center: -visible / 2
            case .trailing: -visible
            }
            let origin = isVertical ? SIMD2(baseline, -start) : SIMD2(start, baseline)
            lines.append(Line(
                origin: SIMD2<Float>(origin * scale), width: Float(width * scale),
                ascent: Float(Double(ascent) * scale), descent: Float(Double(descent) * scale),
                leading: Float(Double(leading) * scale)
            ))

            for run in CTLineGetGlyphRuns(entry.line) as! [CTRun] {
                let runFont = try Self.font(of: run)
                let count = CTRunGetGlyphCount(run)
                guard count > 0 else { continue }
                var runGlyphs = [CGGlyph](repeating: 0, count: count)
                var positions = [CGPoint](repeating: .zero, count: count)
                var indices = [CFIndex](repeating: 0, count: count)
                CTRunGetGlyphs(run, CFRange(location: 0, length: 0), &runGlyphs)
                CTRunGetPositions(run, CFRange(location: 0, length: 0), &positions)
                CTRunGetStringIndices(run, CFRange(location: 0, length: 0), &indices)
                let runMatrix = CTRunGetTextMatrix(run)
                let drawingMatrix = isVertical ? runMatrix.concatenating(verticalRunToOutline) : runMatrix
                // Core Text already applies the font matrix when building the glyph path.
                let textMatrix = CTFontGetMatrix(runFont).inverted().concatenating(drawingMatrix)
                // Vertical origins are expressed in the run's glyph space.
                // Apply its linear transform to them as vectors. Any
                // translation is already applied to the glyph outline above.
                let originMatrix = CGAffineTransform(
                    a: drawingMatrix.a, b: drawingMatrix.b, c: drawingMatrix.c, d: drawingMatrix.d, tx: 0, ty: 0
                )
                for index in 0..<count {
                    let characterIndex = entry.range.location + indices[index]
                    let position = isVertical ? positions[index].applying(originMatrix) : positions[index]
                    var transform = textMatrix
                        .concatenating(CGAffineTransform(
                            translationX: origin.x + position.x, y: origin.y + position.y
                        ))
                        .concatenating(CGAffineTransform(scaleX: scale, y: scale))
                    var glyph = runGlyphs[index]
                    let path = CTFontCreatePathForGlyph(runFont, glyph, &transform)
                    if path == nil || path!.isEmpty {
                        var bounds = CGRect.zero
                        CTFontGetBoundingRectsForGlyphs(runFont, .default, &glyph, &bounds, 1)
                        guard bounds.isEmpty else { throw Failure.glyphWithoutOutline(characterIndex) }
                        glyphs.append(Glyph(characterIndex: characterIndex, line: lineIndex, shapes: []))
                        continue
                    }
                    let shapes: [PlanarShape]
                    do {
                        shapes = try PlanarShape.shapes(from: path!, fillRule: .winding, tolerance: tolerance)
                    } catch {
                        throw Failure.unresolvedOutline(characterIndex)
                    }
                    glyphs.append(Glyph(characterIndex: characterIndex, line: lineIndex, shapes: shapes))
                }
            }
        }
        self.glyphs = glyphs
        self.lines = lines
        self.direction = direction
    }

    private static func font(of run: CTRun) throws -> CTFont {
        let attributes = CTRunGetAttributes(run) as! [String: Any]
        guard let value = attributes[kCTFontAttributeName as String],
              CFGetTypeID(value as CFTypeRef) == CTFontGetTypeID() else {
            throw Failure.missingFont
        }
        return value as! CTFont
    }

    private static func font(at location: Int, in string: NSAttributedString) -> CTFont? {
        guard string.length > 0 else { return nil }
        let index = min(max(location - 1, 0), string.length - 1)
        guard let value = string.attribute(NSAttributedString.Key(kCTFontAttributeName as String), at: index, effectiveRange: nil),
              CFGetTypeID(value as CFTypeRef) == CTFontGetTypeID() else {
            return nil
        }
        return (value as! CTFont)
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidSize
        case invalidLineSpacing
        case invalidTolerance
        case missingFont
        case glyphWithoutOutline(Int)
        case unresolvedOutline(Int)

        public var errorDescription: String? {
            switch self {
            case .invalidSize: "The text size must be finite and positive."
            case .invalidLineSpacing: "Line spacing must be finite and positive."
            case .invalidTolerance: "The curve tolerance must be finite and positive."
            case .missingFont: "Core Text laid out a glyph run without a font."
            case .glyphWithoutOutline(let index):
                "The glyph for the character at UTF-16 offset \(index) has no vector outline, as with bitmap emoji."
            case .unresolvedOutline(let index):
                "The outline of the glyph for the character at UTF-16 offset \(index) could not be resolved."
            }
        }
    }
}

extension MeshGeometry {
    /// Text as a mesh: flat at Z = 0 when `extrusion` is nil, otherwise
    /// extruded with the shape extrusion settings. An optional UV generator is
    /// forwarded to the extrusion. See ``TextOutline`` for layout.
    /// Throws ``ShapeMeshError/noShapes`` when the text has no visible glyphs.
    public static func text(
        _ string: String, font: CTFont, size: Float = 1, extrusion: ShapeExtrusion? = nil,
        alignment: TextOutline.Alignment = .leading, lineSpacing: Float = 1, curveTolerance: Float? = nil,
        uvGenerator: ShapeExtrusionUVGenerator? = nil
    ) throws -> MeshGeometry {
        try text(
            string, font: font, size: size, direction: .horizontal, extrusion: extrusion,
            alignment: alignment, lineSpacing: lineSpacing, curveTolerance: curveTolerance, uvGenerator: uvGenerator
        )
    }

    /// Builds flat or straight-extruded text with the selected layout direction.
    public static func text(
        _ string: String, font: CTFont, size: Float = 1, direction: TextOutline.Direction,
        extrusion: ShapeExtrusion? = nil, alignment: TextOutline.Alignment = .leading,
        lineSpacing: Float = 1, curveTolerance: Float? = nil, uvGenerator: ShapeExtrusionUVGenerator? = nil
    ) throws -> MeshGeometry {
        guard extrusion != nil || uvGenerator?.hasCallbacks != true else { throw ShapeMeshError.uvGeneratorNeedsExtrusion }
        let outline = try TextOutline(
            string, font: font, size: size, direction: direction, alignment: alignment,
            lineSpacing: lineSpacing, curveTolerance: curveTolerance
        )
        if let extrusion { return try extrude(outline.shapes, extrusion, uvGenerator: uvGenerator) }
        return try shape(outline.shapes)
    }

    /// Lays out text with Core Text and sweeps its XY glyph profiles along a
    /// spatial curve. The curve's transported frame places each profile.
    /// Open sweeps have flat caps in material slot 0, with counters left open.
    /// Walls use material slot 1. Closed sweeps join the endpoints without caps.
    /// Sampling, texture mapping and optional UV callbacks follow the curve
    /// shape extrusion. Throws ``ShapeMeshError/noShapes`` for empty outlines.
    public static func text(
        _ string: String, font: CTFont, size: Float = 1, along path: Curve3D,
        steps: Int = 1, closed: Bool = false, arcLengthSubdivisions: Int = 200,
        creaseAngle: Float = .pi / 6, textureMapping: ShapeTextureMapping = .normalized,
        alignment: TextOutline.Alignment = .leading, lineSpacing: Float = 1, curveTolerance: Float? = nil,
        uvGenerator: ShapeExtrusionUVGenerator? = nil
    ) throws -> MeshGeometry {
        try text(
            string, font: font, size: size, direction: .horizontal, along: path,
            steps: steps, closed: closed, arcLengthSubdivisions: arcLengthSubdivisions,
            creaseAngle: creaseAngle, textureMapping: textureMapping, alignment: alignment,
            lineSpacing: lineSpacing, curveTolerance: curveTolerance, uvGenerator: uvGenerator
        )
    }

    /// Sweeps text laid out in the selected direction along a spatial curve.
    public static func text(
        _ string: String, font: CTFont, size: Float = 1, direction: TextOutline.Direction, along path: Curve3D,
        steps: Int = 1, closed: Bool = false, arcLengthSubdivisions: Int = 200,
        creaseAngle: Float = .pi / 6, textureMapping: ShapeTextureMapping = .normalized,
        alignment: TextOutline.Alignment = .leading, lineSpacing: Float = 1, curveTolerance: Float? = nil,
        uvGenerator: ShapeExtrusionUVGenerator? = nil
    ) throws -> MeshGeometry {
        let outline = try TextOutline(
            string, font: font, size: size, direction: direction, alignment: alignment,
            lineSpacing: lineSpacing, curveTolerance: curveTolerance
        )
        return try extrude(
            outline.shapes, along: path, steps: steps, closed: closed,
            arcLengthSubdivisions: arcLengthSubdivisions, creaseAngle: creaseAngle,
            textureMapping: textureMapping, uvGenerator: uvGenerator
        )
    }
}
