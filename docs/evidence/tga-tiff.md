# TGA and TIFF textures

`TextureLoader.load(data:interpretation:mipmapped:)` and its local-file
overload accept TGA and TIFF. Both use the existing RGBA8 upload and optional
Metal mip generation. `.color` and `.encodedSRGB` select sRGB sampling.
`.linearData` uses RGBA8 UNORM without a sampling transfer. The standard 64 MiB
encoded, 8,192-pixel axis, and 16,777,216-pixel limits apply.

## TGA

The r187 loader handles noninterleaved raw and RLE TGA with 8-bit indexed
pixels and 24-bit palettes, 8-bit gray, 16-bit gray with alpha, and 16-, 24-,
and 32-bit truecolor. The four TGA origins map to top-left texture rows.
Frostfield uses ImageIO for supported common layouts and a bounded native
decoder for RLE, right-origin images, offset palettes, gray with alpha,
RGB555, and color interpretation of BGRA. The native path validates palette
indices, packet lengths, image IDs, and TGA 2.0 footer offsets.
[Three.js r187 TGALoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/TGALoader.js)
[Truevision TGA 2.0 specification](https://raw.githubusercontent.com/DEAKSoftware/Truevision-TGA/master/truevision-tga.pdf)

RGB555 uses r187's five-bit expansion and its bit-15 interpretation for
declared alpha. Truecolor with zero declared alpha bits is opaque, preserving
the ImageIO behavior for BGRX. The native path treats color bytes as sRGB
and preserves straight RGB at zero alpha in the data interpretations.
`.color` clears hidden RGB at zero alpha. The native path covers the
pixel layouts above. TGA extension gamma and color-correction tables,
interleaving, and auxiliary images remain outside its scope.

## TIFF

ImageIO decodes the first TIFF image directory to RGBA8. The r187 loader also
selects the first directory and returns RGBA8. Frostfield's `.color` mode
applies TIFF orientation and converts tagged color to sRGB. The data modes
retain storage orientation. Compression, photometric conversion and TIFF
variant availability depend on ImageIO. The loader exposes one 2D texture,
with higher precision samples converted to RGBA8.
[Three.js r187 TIFFLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/TIFFLoader.js)

## Verification

The native Mac and iPad Pro 13-inch M5 simulator produced identical dimensions
and RGBA8 digests for 30 fixtures in all three interpretations. These include
both TGA crates and all three TIFF crates from the r187 demos. Before the
native TGA path, ImageIO mirrored right-origin rows, rejected offset palettes
and produced incorrect pixels for several RLE and alpha layouts.
[Three.js r187 TGA demo](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_texture_tga.html)
[Three.js r187 TIFF demo](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_texture_tiff.html)

The fixture comparison used the actual pinned loaders. TGA matched the
reference after the documented BGRX opacity and color-mode alpha rules.
TIFF LZW, big-endian PackBits, palette, and grayscale fixtures matched exactly.
The uncompressed and JPEG TIFF crates differed by at most one RGB code.
The orientation fixture rotated in color mode and matched r187's storage
order in both data modes.

Xcode 27.2 (27B52) passed 32 focused texture and ModelIO regression tests
on the M4 Max Mac and iPad simulator. The Xcode MCP timed out resolving
the Swift package destination, and the Lab scheme lacks package tests.
Both runs used `xcodebuild` with internal DerivedData.
