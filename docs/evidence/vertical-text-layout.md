# Top-to-bottom text

Pass `direction: .topToBottom` to `TextOutline` or to the flat, straight-extruded and curve-extruded `MeshGeometry.text` entry points. Core Text selects vertical glyph forms and supplies their advances and origins, including upright Latin letters and a font's alternate punctuation forms. Attributed text retains its font and shaping attributes, with vertical forms enabled on a copy of the input. [Apple Core Text vertical forms](https://developer.apple.com/documentation/coretext/kctverticalformsattributename), [Apple vertical glyph origins](https://developer.apple.com/documentation/coretext/ctfontgetverticaltranslationsforglyphs(_:_:_:_:))

```swift
let outline = try TextOutline("OI\nO", font: font, direction: .topToBottom, alignment: .center)
let solid = try MeshGeometry.text(
    "OI\nO", font: font, direction: .topToBottom, extrusion: .init(depth: 0.12)
)
let swept = try MeshGeometry.text(
    "OI\nO", font: font, direction: .topToBottom, along: spine, steps: 12
)
```

Coordinates are Y-up, so the glyphs progress toward negative Y. The first column's baseline is X = 0 and subsequent columns lie to its left. A blank line between separators reserves a column. A terminal separator ends the last column. Column spacing uses Core Text's cross-axis metrics and the existing `lineSpacing` multiplier. Leading, centered and trailing alignment place each column's inline advance relative to Y = 0, with trailing whitespace excluded from centered and trailing alignment. `Line.width` measures the column's Y advance including trailing whitespace.

The pinned r187 Helvetiker fixture at size 40 places the I of “OI” at Y = 0 and the O at Y = 47.56 after reversing the input code points. The O still appears above the I in Y-up coordinates. A newline moves the subsequent glyphs along the same X baseline. Frostfield uses native glyph shaping and a separate column for each explicitly separated line. [Three.js r187 FontLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/FontLoader.js#L126-L169)

## Verification

- The Latin fixture checks glyph bounds against Core Text's per-glyph vertical origins and advances, counter shapes, whitespace, column spacing and alignment. The Japanese bracket fixture verifies native vertical substitutions and outline bounds.
- A sheared-font outline matches the bounds of the native Core Text vertical frame draw to raster precision.
- Native meshes preserve the same outlines through flat, straight and curve extrusion, including curve material groups. Metal readback finds both extruded columns and clear pixels through their counters.
- Xcode 27.2 passed 45 focused tests in eight suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
