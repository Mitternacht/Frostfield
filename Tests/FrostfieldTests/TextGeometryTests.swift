import CoreGraphics
import CoreText
import Foundation
import Frostfield
import Metal
import Testing
import simd

struct TextGeometryTests {
    private let font = CTFontCreateWithName("Helvetica" as CFString, 72, nil)

    private func line(_ string: String, font: CTFont) -> CTLine {
        let attributes = [NSAttributedString.Key(kCTFontAttributeName as String): font]
        return CTLineCreateWithAttributedString(NSAttributedString(string: string, attributes: attributes))
    }

    @Test("Helvetica “Ag”: each glyph is a closed solid, A and g each have one counter hole, and bounds match Core Text")
    func agGlyphs() throws {
        try #require(CTFontCopyPostScriptName(font) as String == "Helvetica")
        let size: Float = 2
        let outline = try TextOutline("Ag", font: font, size: size)
        try #require(outline.glyphs.count == 2)
        let a = outline.glyphs[0], g = outline.glyphs[1]
        #expect(a.characterIndex == 0 && g.characterIndex == 1)
        #expect(a.shapes.count == 1 && a.shapes[0].holes.count == 1)
        #expect(g.shapes.count == 1 && g.shapes[0].holes.count == 1)

        let extrusion = ShapeExtrusion(depth: 0.3, bevelSize: 0.02, bevelDepth: 0.02, bevelSegments: 3)
        for glyph in outline.glyphs {
            let solid = try MeshGeometry.extrude(glyph.shapes, extrusion)
            #expect(ShapeCheck.isClosed(solid))
            #expect(ShapeCheck.tangentsFollowTextureCoordinates(solid, minimumArea: 1e-7))
            let area = glyph.shapes.reduce(0) { $0 + $1.area }
            // The bevel removes a little volume from the full prism, never adds any.
            #expect(solid.signedVolume < area * 0.3 && solid.signedVolume > area * 0.3 * 0.8)
            // The counter is open: no cap triangle covers the hole's interior.
            let hole = glyph.shapes[0].holes[0].map(SIMD2<Double>.init)
            let probe = hole.reduce(SIMD2<Double>.zero, +) / Double(hole.count)
            let flat = try MeshGeometry.shape(glyph.shapes)
            for triangle in 0..<(flat.indices.count / 3) {
                let (p, q, r) = ShapeCheck.corners(flat, triangle)
                let corners = [p, q, r].map { SIMD2($0.x, $0.y) }
                let inside = (0..<3).allSatisfy { index in
                    let from = corners[index], to = corners[(index + 1) % 3]
                    return (to.x - from.x) * (probe.y - from.y) - (to.y - from.y) * (probe.x - from.x) >= 0
                }
                #expect(!inside)
            }
        }

        // The mesh's XY bounds equal Core Text's glyph-path bounds within the flattening tolerance.
        let ctLine = line("Ag", font: font)
        let scale = Double(size) / 72
        let pathBounds = CTLineGetBoundsWithOptions(ctLine, .useGlyphPathBounds)
        let mesh = try MeshGeometry.text("Ag", font: font, size: size, extrusion: extrusion)
        #expect(ShapeCheck.isClosed(mesh))
        let bounds = ShapeCheck.bounds(mesh.vertices.map(\.position))
        let tolerance = Double(size) / 1_000 + 1e-5
        #expect(abs(Double(bounds.lower.x) - pathBounds.minX * scale) < tolerance)
        #expect(abs(Double(bounds.upper.x) - pathBounds.maxX * scale) < tolerance)
        #expect(abs(Double(bounds.lower.y) - pathBounds.minY * scale) < tolerance)
        #expect(abs(Double(bounds.upper.y) - pathBounds.maxY * scale) < tolerance)
        #expect(bounds.lower.z == -0.15 && bounds.upper.z == 0.15)

        // Typographic metrics come straight from the CTLine.
        var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
        let width = CTLineGetTypographicBounds(ctLine, &ascent, &descent, &leading)
        let metrics = outline.lines[0]
        #expect(abs(Double(metrics.width) - width * scale) < 1e-5)
        #expect(abs(Double(metrics.ascent) - ascent * scale) < 1e-5)
        #expect(abs(Double(metrics.descent) - descent * scale) < 1e-5)
        #expect(Double(bounds.lower.x) >= -1e-5 && Double(bounds.upper.x) <= width * scale + 1e-5)
        #expect(Double(bounds.lower.y) >= -descent * scale - 1e-5 && Double(bounds.upper.y) <= ascent * scale + 1e-5)
    }

    @Test("Layout follows Core Text: kerning narrows “AV”, lines stack by their metrics and alignment centers each line")
    func layout() throws {
        let kerned = try TextOutline("AV", font: font, size: 1)
        var glyphs: [CGGlyph] = [0, 0]
        var characters = Array("AV".utf16)
        #expect(CTFontGetGlyphsForCharacters(font, &characters, &glyphs, 2))
        let advances = CTFontGetAdvancesForGlyphs(font, .horizontal, &glyphs, nil, 2)
        // Helvetica kerns the pair, so the line is narrower than the sum of advances.
        #expect(Double(kerned.lines[0].width) < advances / 72 - 0.01)

        let text = try TextOutline("Hi\nWorld ", font: font, size: 1, alignment: .center, lineSpacing: 1.5)
        #expect(text.lines.count == 2)
        var a1: CGFloat = 0, d1: CGFloat = 0, l1: CGFloat = 0, a2: CGFloat = 0, d2: CGFloat = 0, l2: CGFloat = 0
        let w1 = CTLineGetTypographicBounds(line("Hi", font: font), &a1, &d1, &l1)
        let secondLine = line("World ", font: font)
        let w2 = CTLineGetTypographicBounds(secondLine, &a2, &d2, &l2)
        let trailing = CTLineGetTrailingWhitespaceWidth(secondLine)
        #expect(text.lines[0].origin == SIMD2(Float(-w1 / 2 / 72), 0))
        #expect(abs(Double(text.lines[1].origin.x) - -(w2 - trailing) / 2 / 72) < 1e-6)
        #expect(abs(Double(text.lines[1].origin.y) - -(d1 + l1 + a2) * 1.5 / 72) < 1e-6)
        // The trailing space has a glyph with no outline.
        #expect(text.glyphs.last?.shapes.isEmpty == true)
        #expect(text.glyphs.filter { $0.line == 1 }.count == 6)
    }

    @Test("A sheared font applies its transform once and matches Core Text glyph bounds")
    func fontTransform() throws {
        var shear = CGAffineTransform(a: 1, b: 0, c: 0.4, d: 1, tx: 0, ty: 0)
        let font = CTFontCreateWithName("Helvetica" as CFString, 72, &shear)
        let expected = CTLineGetBoundsWithOptions(line("HI", font: font), .useGlyphPathBounds)
        let mesh = try MeshGeometry.text("HI", font: font, size: 72)
        let bounds = ShapeCheck.bounds(mesh.vertices.map(\.position))
        #expect(abs(Double(bounds.lower.x) - expected.minX) < 1e-4)
        #expect(abs(Double(bounds.upper.x) - expected.maxX) < 1e-4)
        #expect(abs(Double(bounds.lower.y) - expected.minY) < 1e-4)
        #expect(abs(Double(bounds.upper.y) - expected.maxY) < 1e-4)
    }

    @Test("Bitmap emoji, which have no outline, are reported instead of silently dropped")
    func bitmapGlyph() {
        #expect(throws: TextOutline.Failure.glyphWithoutOutline(1)) {
            try TextOutline("A😀", font: font)
        }
        #expect(throws: TextOutline.Failure.invalidSize) {
            try TextOutline("A", font: font, size: 0)
        }
    }

    @Test("The system UI font, with variable-font outlines, extrudes to closed glyph solids")
    func systemFont() throws {
        let system = try #require(CTFontCreateUIFontForLanguage(.system, 48, nil))
        let outline = try TextOutline("Bg8&@", font: system, size: 1)
        for glyph in outline.glyphs {
            let solid = try MeshGeometry.extrude(glyph.shapes, ShapeExtrusion(depth: 0.2, bevelSize: 0.008))
            #expect(ShapeCheck.isClosed(solid))
            #expect(solid.signedVolume > 0)
        }
        // B has two counters, 8 has two, @ has one.
        #expect(outline.glyphs[0].shapes.flatMap(\.holes).count == 2)
        #expect(outline.glyphs[2].shapes.flatMap(\.holes).count == 2)
    }

    @Test("Rasterized text mesh coverage matches Core Text drawing the same line", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func renderedCoverage() throws {
        let fixture = try GeometryRenderFixture(side: 160)
        let size: Float = 1.2
        let mesh = try MeshGeometry.text("Ag", font: font, size: size, alignment: .center)
        let renderer = try fixture.unlit(mesh)
        let offsetY: Float = -0.3
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, offsetY, -1)), mesh: renderer)
        let (pixels, _) = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera))

        // Core Text draws the same line into an antialiased coverage mask with the camera's
        // mapping (at depth 1 the 90° view spans −1...1 on both axes), at five times the
        // resolution so each GPU pixel center is the center of one reference subpixel.
        let side = fixture.side, factor = 5, fine = side * factor
        var mask = [UInt8](repeating: 0, count: fine * fine)
        let pointsPerUnit = 72 / Double(size)
        var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorFromContextAttributeName as String): true,
        ]
        let ctLine = CTLineCreateWithAttributedString(NSAttributedString(string: "Ag", attributes: attributes))
        let width = CTLineGetTypographicBounds(ctLine, &ascent, &descent, &leading) - CTLineGetTrailingWhitespaceWidth(ctLine)
        mask.withUnsafeMutableBytes { bytes in
            let context = CGContext(
                data: bytes.baseAddress, width: fine, height: fine, bitsPerComponent: 8, bytesPerRow: fine,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue
            )!
            // Keep glyph origins unrounded so both images place the glyphs at the same subpixel offsets.
            context.setShouldSmoothFonts(false)
            context.setAllowsFontSubpixelPositioning(true)
            context.setShouldSubpixelPositionFonts(true)
            context.setAllowsFontSubpixelQuantization(false)
            context.setShouldSubpixelQuantizeFonts(false)
            context.setFillColor(gray: 1, alpha: 1)
            context.translateBy(x: CGFloat(fine) / 2, y: CGFloat(fine) / 2)
            context.scaleBy(x: CGFloat(fine) / 2 / pointsPerUnit, y: CGFloat(fine) / 2 / pointsPerUnit)
            context.textPosition = CGPoint(x: -width / 2, y: Double(offsetY) * pointsPerUnit)
            CTLineDraw(ctLine, context)
        }
        // The bitmap context stores its top row first, as the render target does.
        func reference(_ x: Int, _ y: Int) -> Double {
            Double(mask[(y * factor + factor / 2) * fine + x * factor + factor / 2]) / 255
        }

        // The GPU samples each pixel's center once. Wherever Core Text's coverage of the
        // subpixel around that center is decisive, at least 3/4 or at most 1/4, they must agree.
        var inside = 0, outside = 0, disagreements = 0, renderedArea = 0
        for y in 0..<side {
            for x in 0..<side {
                let gpu = fixture.pixel(pixels, x: x, y: y).w > 0.5
                if gpu { renderedArea += 1 }
                let coverage = reference(x, y)
                if coverage >= 0.75 {
                    inside += 1
                    if !gpu { disagreements += 1 }
                } else if coverage <= 0.25 {
                    outside += 1
                    if gpu { disagreements += 1 }
                }
            }
        }
        #expect(inside > 2_500 && outside > 20_000)
        #expect(disagreements == 0)
        // Total coverage agrees to within 1% of the glyph area.
        let coveredArea = mask.reduce(0.0) { $0 + Double($1) / 255 } / Double(factor * factor)
        #expect(abs(Double(renderedArea) - coveredArea) < 0.01 * coveredArea)
        // The counter of the A is open in both images.
        let outline = try TextOutline("Ag", font: font, size: size, alignment: .center)
        let counter = outline.glyphs[0].shapes[0].holes[0]
        let center = counter.reduce(SIMD2<Float>.zero, +) / Float(counter.count)
        let counterX = fixture.column(x: center.x, depth: 1), counterY = fixture.row(y: center.y + offsetY, depth: 1)
        #expect(reference(counterX, counterY) == 0)
        #expect(fixture.pixel(pixels, x: counterX, y: counterY).w == 0)
    }
}
