# DDS textures

`TextureLoader.loadDDS` accepts data or a local file and returns a Metal
2D texture or cube. The ordinary `TextureLoader.load` path also recognizes
2D DDS data. Stored cube faces are ordered +X, −X, +Y, −Y, +Z, −Z, with
each face's complete mip chain before the next face. Set `loadMipmaps: false`
to upload only the base level of each face. `mipmapped` generates a chain
when the selected source has a single uncompressed level.
[Microsoft DDS layout](https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide)

The loader accepts the r187 encodings DXT1, DXT3, DXT5, ETC1, DX10 signed
and unsigned BC6H, BGR24, and BGRA32. DXT1 samples as opaque RGB, including
black from selector 3 in a three-color block. DXT3 and DXT5 retain their
alpha. ETC1 uses compatible ETC2 RGB8 sampling on devices with Apple GPU
family 2 support. LDR defaults to sRGB, and `.linearData` selects linear
UNORM sampling. BC6H always samples as linear HDR.
[Three.js r187 DDSLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/DDSLoader.js)

Metal uses the original BC blocks when the device reports BC support.
Other devices decode DXT1, DXT3 and DXT5 to RGBA8 with the existing
Basis Universal unpackers. BC6H decodes to RGBA16Float using the BC6H-only
subset of bcdec at `80859ed3b7afb1c527a2a99d70c61457bea72d0c`.
The subset uses defined integer sign extension and preserves the upstream
half-float values. Both fallbacks crop edge blocks to the mip's dimensions.
[bcdec source](https://github.com/iOrange/bcdec/blob/80859ed3b7afb1c527a2a99d70c61457bea72d0c/bcdec.h)
[MIT license](../../LICENSES/bcdec-MIT.txt)

The parser validates the complete payload, masks, header sizes, mip count,
pitch and cube flags before texture allocation. It supports tight rows and
common DWORD-aligned or top-level declared pitches for BGR and BGRA. Limits
are 64 MiB encoded, 8,192 pixels per axis and 16,777,216 base-level texels
across all faces. DDS arrays, volumes, partial cubes, and DXGI formats beyond
the two BC6H encodings are outside this implementation.

## Verification

The native loader matched the dimensions, mip counts, face order and payload
hashes of all 14 DDS fixtures bundled with the r187 demo. Hashes use stored
blocks for compressed textures and the uploaded RGBA8 bytes for uncompressed
textures. Across 42 sampled coordinates, native BC on the M4 Max and CPU BC
fallback on the iPad Pro 13-inch M5 simulator agreed within one code for LDR
and exactly for the sampled HDR values. Both sampled ETC1 through Metal.
[Three.js r187 DDS demo](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_texture_dds.html)

The focused Swift tests cover odd block dimensions, DXT1 opacity, both DXT
alpha encodings, ETC1 texels, signed and unsigned HDR, stored mip and cube
order, base-only cube selection, and malformed input. Clang AddressSanitizer
and UndefinedBehaviorSanitizer passed 100,000 unaligned BC6H blocks in both
signed and unsigned modes.

The Xcode 27.2 MCP timed out resolving the Swift package destination, and
the Lab scheme lacks package tests. Verification used `xcodebuild` with
internal DerivedData on macOS and iPad Simulator.

The 26-test DDS, KTX2, legacy texture and cube regression run passed on both
hosts. The iPad run skipped one existing native-BC-only KTX2 case and
executed all DDS fallback cases. The integrated iPhone Simulator run at
`b6b8781` also exercised DDS with the fallback enabled.
