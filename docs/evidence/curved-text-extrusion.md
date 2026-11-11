# Curved text extrusion

`MeshGeometry.text(_:font:size:along:steps:)` uses Core Text to lay out glyphs, resolves their vector outlines with counters, and sweeps the resulting XY profiles along a `Curve3D`. Steps and arc-length subdivisions control sampling. The transported frame sets the orientation of each profile along the curve. For attributed strings, build a `TextOutline` and pass its `shapes` to `MeshGeometry.extrude(_:along:steps:)`.

```swift
let spine = try Curve3D.cubicBezier(
    from: SIMD3(0, 0, -3), control1: SIMD3(0, 0, -2.67),
    control2: SIMD3(0.1, 0, -2.33), to: SIMD3(0.1, 0, -2)
)
let mesh = try MeshGeometry.text("O", font: font, size: 1.5, along: spine, steps: 8, alignment: .center)
```

Open sweeps use flat caps in material slot 0 and walls in slot 1. Each glyph profile keeps its counters, and a closed curve joins its matching endpoints without caps. Custom cap and wall UV callbacks use the same convention as shape extrusion. The straight text overload supports bevels. Choose a spine and text size with enough clearance for the swept glyphs.

Three.js r187 generates shapes from the selected font and forwards the path, steps and UV generator to `ExtrudeGeometry`. An upstream Helvetiker O fixture at one and six steps produced the same positions, UVs and cap/wall group ranges as direct extrusion of the font shapes. Increasing the step count from one to six increased wall callback calls from 192 to 1,152 and kept cap callback calls at 384. [Three.js r187 TextGeometry](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/geometries/TextGeometry.js)

## Verification

- A native Helvetica O–space–O fixture checks the Core Text glyph indices and counters, compares all authored UVs with the corresponding outline sweep, and verifies positive volume, closed edge incidence, tangents and separate cap/wall group ranges.
- Metal readback shows the selected green cap texel, blue wall pixels and a clear 7 × 7 region through the first glyph's counter.
- Xcode 27.2 passed 45 focused tests in eight suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
