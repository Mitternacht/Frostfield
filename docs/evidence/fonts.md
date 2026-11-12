# Font files

`FontFile.fonts(data:size:)` reads TrueType and OpenType bytes, including font
collections, and returns one Core Text font per face. The point size selects
the face's optical size. `TextOutline` and `MeshGeometry.text` still scale the
em into world units.

This replaces Three.js `TTFLoader` and `FontLoader`. Those loaders turn a font
file into typeface.js JSON because a browser has no font engine. Core Text
reads the font bytes, so Frostfield does not parse that JSON.

```swift
let faces = try FontFile.fonts(data: data, size: 72)
let outline = try TextOutline("Ag", font: faces[0], size: 1)
let solid = try MeshGeometry.text("Ag", font: faces[0], size: 1, extrusion: .init(depth: 0.2))
```

An empty or unreadable file throws `FontFile.Failure.unrecognized`. A size that
is not finite and positive throws `FontFile.Failure.size`.

`FontFileTests` passed with Metal API Validation through `xcodebuild`. The
Xcode bridge was not used. The test loads the Helvetica file Core Text names,
checks the 72-point advance against that face, and checks that the loaded face
produces the same “A” outline area and counter. macOS: `Test run with 1 test in 1 suite passed after 0.010 seconds`.
iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.038 seconds`.
