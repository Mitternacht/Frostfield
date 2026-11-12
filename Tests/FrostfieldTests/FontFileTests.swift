import CoreText
import Foundation
import Testing
@testable import Frostfield

struct FontFileTests {
    @Test func loadsTheFaceThatCoreTextNames() throws {
        let named = CTFontCreateWithName("Helvetica" as CFString, 72, nil)
        let url = try #require(CTFontCopyAttribute(named, kCTFontURLAttribute) as? URL)
        let data = try Data(contentsOf: url)
        let faces = try FontFile.fonts(data: data, size: 72)
        let loaded = try #require(faces.first {
            CTFontCopyPostScriptName($0) as String == "Helvetica"
        })
        #expect(abs(CTFontGetSize(loaded) - 72) < 0.001)
        #expect(abs(advance(loaded) - advance(named)) < 0.01)

        let small = try #require(FontFile.fonts(data: data, size: 36).first {
            CTFontCopyPostScriptName($0) as String == "Helvetica"
        })
        #expect(advance(small) < advance(loaded) * 0.6)

        let fromFile = try TextOutline("A", font: loaded, size: 1)
        let fromName = try TextOutline("A", font: named, size: 1)
        #expect(fromFile.glyphs.count == 1)
        #expect(fromFile.glyphs[0].shapes.count == fromName.glyphs[0].shapes.count)
        #expect(fromFile.glyphs[0].shapes[0].holes.count == 1)
        #expect(abs(fromFile.glyphs[0].shapes[0].area - fromName.glyphs[0].shapes[0].area) < 1e-4)

        #expect(throws: FontFile.Failure.size) { try FontFile.fonts(data: data, size: 0) }
        #expect(throws: FontFile.Failure.unrecognized) { try FontFile.fonts(data: Data([0, 1, 2, 3])) }
    }

    private func advance(_ font: CTFont) -> CGFloat {
        var character: UniChar = 65
        var glyph = CGGlyph()
        CTFontGetGlyphsForCharacters(font, &character, &glyph, 1)
        var advance = CGSize.zero
        CTFontGetAdvancesForGlyphs(font, .horizontal, &glyph, &advance, 1)
        return advance.width
    }
}
