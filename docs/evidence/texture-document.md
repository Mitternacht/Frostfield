# Texture documents

`TextureDocument` is the base for a custom data texture or a block-compressed
texture, in the role of Three.js `DataTextureLoader` and
`CompressedTextureLoader`. A `TextureDocumentParser` reads the file bytes.
`TextureLoader.load(data:parser:)` checks the 64 MiB input limit, and
`upload(_:)` checks the 8,192-pixel and 16,777,216-pixel limits, then creates
a shared shader-readable texture.

RGBA8 bytes are straight and tightly packed. Block contents are ASTC 4×4 or
BC1, one byte buffer per mip level, starting at level zero. The level
dimensions halve, and each level's byte count must match its block grid.
KTX2, DDS, KTX1, PVR and HDR stay on their direct `TextureLoader` methods.

```swift
struct HeaderParser: TextureDocumentParser {
    func parse(_ data: Data) throws -> TextureDocument {
        TextureDocument(width: 2, height: 2, contents: .rgba8(data))
    }
}
let texture = try TextureLoader(device: device).load(data: pixels, parser: HeaderParser())
```

`TextureDocumentTests` passed with Metal API Validation through `xcodebuild`.
The Xcode bridge was not used. The test reads the uploaded RGBA8 pixels back,
rejects a short buffer, round-trips one ASTC 4×4 or BC1 block, and checks that
the direct image loader still rejects a three-byte input. macOS: `Test run with 1 test in 1 suite passed after 0.006 seconds`.
iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.019 seconds`.
