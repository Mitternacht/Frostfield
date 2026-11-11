import CoreGraphics
import CoreText
import Foundation
import Frostfield
import Metal
import Testing

struct VerticalTextGeometryTests {
    private let font = CTFontCreateWithName("Helvetica" as CFString, 40, nil)

    @Test("Top-to-bottom Latin glyphs follow native origins, advances and column spacing")
    func verticalMetrics() throws {
        try #require(CTFontCopyPostScriptName(font) as String == "Helvetica")
        let line = "OI O "
        let spacing: Float = 1.25
        let layout = try TextOutline(
            line + "\n\nO", font: font, direction: .topToBottom, alignment: .center, lineSpacing: spacing
        )
        #expect(layout.direction == .topToBottom)
        try #require(layout.lines.count == 3 && layout.glyphs.count == 6)
        #expect(layout.glyphs.map(\.characterIndex) == [0, 1, 2, 3, 4, 7])
        #expect(layout.glyphs.map(\.line) == [0, 0, 0, 0, 0, 2])
        #expect(layout.glyphs[2].shapes.isEmpty && layout.glyphs[4].shapes.isEmpty)
        #expect(layout.glyphs[0].shapes.flatMap(\.holes).count == 1)
        #expect(layout.glyphs[3].shapes.flatMap(\.holes).count == 1)
        #expect(layout.lines[1].width == 0 && layout.lines[1].origin.y == 0)
        #expect(layout.lines[0].origin.x == 0)
        for (previous, next) in zip(layout.lines, layout.lines.dropFirst()) {
            let gap = (previous.descent + previous.leading + next.ascent) * spacing
            #expect(abs(previous.origin.x - next.origin.x - gap) < 1e-5)
            #expect(gap > 0)
        }

        let characters = Array(line.utf16)
        var glyphs = [CGGlyph](repeating: 0, count: characters.count)
        try #require(CTFontGetGlyphsForCharacters(font, characters, &glyphs, characters.count))
        var advances = [CGSize](repeating: .zero, count: glyphs.count)
        var translations = [CGSize](repeating: .zero, count: glyphs.count)
        let width = CTFontGetAdvancesForGlyphs(font, .vertical, glyphs, &advances, glyphs.count)
        CTFontGetVerticalTranslationsForGlyphs(font, glyphs, &translations, glyphs.count)
        let scale = 1 / CTFontGetSize(font)
        let start = (width - (try #require(advances.last)).width) / 2
        #expect(abs(Double(layout.lines[0].width) - width * scale) < 1e-6)
        #expect(abs(Double(layout.lines[0].origin.y) - start * scale) < 1e-6)
        var distance: CGFloat = 0
        for index in glyphs.indices {
            if !layout.glyphs[index].shapes.isEmpty {
                let glyph = try #require(CTFontCreatePathForGlyph(font, glyphs[index], nil))
                let offset = translations[index]
                let expected = glyph.boundingBoxOfPath
                    .offsetBy(dx: offset.width, dy: start + offset.height - distance)
                    .applying(.init(scaleX: scale, y: scale))
                try checkBounds(layout.glyphs[index], expected)
            }
            distance += advances[index].width
        }
        let rightmost = try #require(layout.glyphs[0].shapes.first)
        let leftmost = try #require(layout.glyphs[5].shapes.first)
        #expect(rightmost.outer.allSatisfy { $0.x > layout.lines[1].origin.x })
        #expect(leftmost.outer.allSatisfy { $0.x < layout.lines[1].origin.x })

        let defaultFont = try TextOutline(
            NSAttributedString(string: "\n\nO\n"), unitsPerPoint: 1 / 40, direction: .topToBottom
        )
        try #require(defaultFont.lines.count == 3)
        #expect(defaultFont.glyphs.map(\.line) == [2])
        #expect(defaultFont.lines[0].origin.x > defaultFont.lines[1].origin.x)
        #expect(defaultFont.lines[1].origin.x > defaultFont.lines[2].origin.x)
    }

    @Test("Attributed Japanese brackets use the font's vertical glyph substitutions")
    func verticalForms() throws {
        let font = CTFontCreateWithName("HiraginoSans-W3" as CFString, 40, nil)
        let fontKey = NSAttributedString.Key(kCTFontAttributeName as String)
        let verticalKey = NSAttributedString.Key(kCTVerticalFormsAttributeName as String)
        let text = NSAttributedString(string: "「」", attributes: [fontKey: font, verticalKey: false])
        let scale: Float = 1 / 40
        let vertical = try TextOutline(text, unitsPerPoint: scale, direction: .topToBottom)
        let horizontal = try TextOutline(text, unitsPerPoint: scale)
        #expect(text.attribute(verticalKey, at: 0, effectiveRange: nil) as? Bool == false)
        #expect(horizontal.direction == .horizontal)
        try #require(vertical.glyphs.count == 2 && horizontal.glyphs.count == 2)
        for (up, across) in zip(vertical.glyphs, horizontal.glyphs) {
            let v = try glyphBounds(up), h = try glyphBounds(across)
            #expect(v.width > v.height)
            #expect(h.height > h.width)
        }
        let nativeText = NSAttributedString(string: text.string, attributes: [fontKey: font, verticalKey: true])
        let nativeLine = CTLineCreateWithAttributedString(nativeText)
        let transform = CGAffineTransform(a: 0, b: -CGFloat(scale), c: CGFloat(scale), d: 0, tx: 0, ty: 0)
        let expected = CTLineGetBoundsWithOptions(nativeLine, .useGlyphPathBounds).applying(transform)
        let combined = try glyphBounds(vertical.glyphs[0]).union(glyphBounds(vertical.glyphs[1]))
        expectBounds(combined, expected)
    }

    @Test("Vertical glyphs with a sheared font match the Core Text frame draw")
    func transformedFont() throws {
        var shear = CGAffineTransform(a: 1, b: 0, c: 0.4, d: 1, tx: 0, ty: 0)
        let font = CTFontCreateWithName("Helvetica" as CFString, 40, &shear)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTVerticalFormsAttributeName as String): true,
            NSAttributedString.Key(kCTForegroundColorFromContextAttributeName as String): true,
        ]
        let text = NSAttributedString(string: "OI", attributes: attributes)
        let outline = try TextOutline(text, unitsPerPoint: 1, direction: .topToBottom)
        try #require(outline.glyphs.count == 2)
        let setter = CTFramesetterCreateWithAttributedString(text)
        let frame = CTFramesetterCreateFrame(
            setter, CFRange(location: 0, length: 0),
            CGPath(rect: CGRect(x: 0, y: 0, width: 200, height: 180), transform: nil),
            [kCTFrameProgressionAttributeName: CTFrameProgression.rightToLeft.rawValue] as CFDictionary
        )
        try #require(CFArrayGetCount(CTFrameGetLines(frame)) == 1)
        var origin = CGPoint.zero
        CTFrameGetLineOrigins(frame, CFRange(location: 0, length: 1), &origin)
        let outlineBounds = try glyphBounds(outline.glyphs[0]).union(glyphBounds(outline.glyphs[1]))
            .offsetBy(dx: origin.x, dy: origin.y)

        let side = 240
        var mask = [UInt8](repeating: 0, count: side * side)
        try mask.withUnsafeMutableBytes { bytes in
            let context = try #require(CGContext(
                data: bytes.baseAddress, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue
            ))
            context.setShouldSmoothFonts(false)
            context.setShouldAntialias(false)
            context.setFillColor(gray: 1, alpha: 1)
            CTFrameDraw(frame, context)
        }
        var raster = CGRect.null
        for y in 0..<side {
            for x in 0..<side where mask[y * side + x] > 128 {
                raster = raster.union(CGRect(x: x, y: side - 1 - y, width: 1, height: 1))
            }
        }
        try #require(!raster.isNull)
        #expect(abs(outlineBounds.minX - raster.minX) < 1.5)
        #expect(abs(outlineBounds.minY - raster.minY) < 1.5)
        #expect(abs(outlineBounds.maxX - raster.maxX) < 1.5)
        #expect(abs(outlineBounds.maxY - raster.maxY) < 1.5)
    }

    @Test("Flat, straight and curved text use the same top-to-bottom outline")
    func meshForwarding() throws {
        let text = "O\nO", size: Float = 0.6
        let outline = try TextOutline(text, font: font, size: size, direction: .topToBottom)
        let flat = try MeshGeometry.text(text, font: font, size: size, direction: .topToBottom)
        let expectedFlat = try MeshGeometry.shape(outline.shapes)
        #expect(flat.indices == expectedFlat.indices && flat.vertices.map(\.position) == expectedFlat.vertices.map(\.position))
        let straight = try MeshGeometry.text(
            text, font: font, size: size, direction: .topToBottom, extrusion: .init(depth: 0.2)
        )
        #expect(ShapeCheck.isClosed(straight))
        #expect(straight.signedVolume > 0)
        let path = try Curve3D.cubicBezier(
            from: .zero, control1: SIMD3(0, 0, 0.34), control2: SIMD3(0.06, 0, 0.67), to: SIMD3(0.06, 0, 1)
        )
        let curved = try MeshGeometry.text(text, font: font, size: size, direction: .topToBottom, along: path, steps: 4)
        let expectedCurve = try MeshGeometry.extrude(outline.shapes, along: path, steps: 4)
        #expect(curved.indices == expectedCurve.indices && curved.vertices.map(\.position) == expectedCurve.vertices.map(\.position))
        #expect(curved.groups == expectedCurve.groups)
        #expect(curved.groups.map(\.materialIndex) == [0, 1, 0, 1])
        #expect(ShapeCheck.isClosed(curved))
    }

    @Test("Metal renders both vertical columns with clear pixels through the counters",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func render() throws {
        let text = "O\nO", size: Float = 0.6
        let outline = try TextOutline(text, font: font, size: size, direction: .topToBottom)
        let centers = outline.shapes.flatMap(\.holes).map { $0.reduce(.zero, +) / Float($0.count) }
        try #require(centers.count == 2)
        let offset = -(centers[0] + centers[1]) / 2
        let depth: Float = 1.6
        let thickness: Float = 0.12
        let mesh = try MeshGeometry.text(
            text, font: font, size: size, direction: .topToBottom, extrusion: .init(depth: thickness)
        )
        let gpu = try GeometryRenderFixture(side: 160)
        let color = SIMD3<Float>(0.25, 0.75, 0.5)
        let renderer = try gpu.unlit(mesh, material: .classic(ClassicMaterial(model: .basic, color: color)))
        let node = SceneNode(transform: .init(translation: SIMD3(offset.x, offset.y, -depth - thickness / 2)), mesh: renderer)
        let rendered = try gpu.render(SceneGraph(roots: [node], camera: gpu.camera))
        let half = gpu.side / 2
        for columns in [0..<half, half..<gpu.side] {
            let colored = (0..<gpu.side).reduce(0) { count, row in
                count + columns.filter { gpu.pixel(rendered.pixels, x: $0, y: row) == SIMD4(color, 1) }.count
            }
            #expect(colored > 100)
        }
        for center in centers {
            let point = center + offset
            let x = gpu.column(x: point.x, depth: depth), y = gpu.row(y: point.y, depth: depth)
            #expect(gpu.pixel(rendered.pixels, x: x, y: y) == .zero)
        }
    }

    private func glyphBounds(_ glyph: TextOutline.Glyph) throws -> CGRect {
        let vertices = glyph.shapes.flatMap(\.outer)
        try #require(!vertices.isEmpty)
        let bounds = vertices.reduce(CGRect.null) {
            $0.union(CGRect(x: CGFloat($1.x), y: CGFloat($1.y), width: 0, height: 0))
        }
        return bounds
    }

    private func checkBounds(_ glyph: TextOutline.Glyph, _ expected: CGRect) throws {
        expectBounds(try glyphBounds(glyph), expected)
    }

    private func expectBounds(_ bounds: CGRect, _ expected: CGRect) {
        #expect(abs(bounds.minX - expected.minX) < 0.0015)
        #expect(abs(bounds.minY - expected.minY) < 0.0015)
        #expect(abs(bounds.maxX - expected.maxX) < 0.0015)
        #expect(abs(bounds.maxY - expected.maxY) < 0.0015)
    }
}
