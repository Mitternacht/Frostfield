# Texture and image utilities

`TextureUtilities.fit` returns the native top-left `TextureTransform` for
centered cover, contain, or fill. It accepts a texture or the image's width-to-
height ratio. Cover crops the image. Contain maps the bars beyond the texture's
0...1 UV range, where the caller can mask them or choose sampler behavior. Fill
returns the identity. The fit can be applied to a base color texture or reused
for UV calculations in a graph. [Three.js r187 TextureUtils](https://github.com/mrdoob/three.js/blob/r187/src/extras/TextureUtils.js)

`TextureUtilities.byteCount` returns the tight payload of one 2D mip or slice.
It reuses the package's ordinary and BC/ETC/EAC/ASTC layouts, including odd
partial-block edges. Additional layouts cover native HDR ASTC, extended-range
packed color, and standalone depth or stencil. PVRTC1 uses its minimum
two-by-two-block footprint and requires power-of-two dimensions. Allocation
padding, Metal tile storage, other mip levels, samples, and combined
depth-stencil or video-plane layouts are outside this calculation.
[Apple Metal pixel formats](https://developer.apple.com/documentation/metal/mtlpixelformat),
[Khronos PVRTC format constraints](https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_pvrtc/)

`ImageUtilities.pngDataURL` wraps the native exporter with a base64 PNG URL.
It encodes the same source texture formats as `GLTFExporter.pngData` and takes
the same color interpretation and command queue. The sRGB transfer overloads
work on float or byte RGBA pixels and preserve straight alpha. Float conversion
keeps linear values, and byte conversion rounds to the nearest linear code.
ImageIO and Metal provide native image handling in place of browser canvas and
DOM types. [Three.js r187 ImageUtils](https://github.com/mrdoob/three.js/blob/r187/src/extras/ImageUtils.js)

`Binary16` exposes IEEE binary16 encoding and decoding using Swift `Float16`.
The default rounds to nearest, ties to even, and represents infinities and
NaNs. Finite values beyond the half range overflow to infinity as IEEE 754
specifies. The r187 JavaScript encoder instead clamps to ±65,504 and truncates
bits, as described below. Native callers can choose any pre-conversion clamp
explicitly. [Three.js r187 DataUtils](https://github.com/mrdoob/three.js/blob/r187/src/extras/DataUtils.js)

## Departures from r187

These differences are deliberate. Each except the pixel-based API shape has a test that pins the native behavior.

**`ImageUtilities.sRGBToLinear` keeps alpha.** r187
`ImageUtils.sRGBToLinear` loops over every element of the pixel array with no
stride. Its canvas path, its raw `Uint8Array` or `Uint8ClampedArray` path, and
its float path therefore all transform the alpha lane together with RGB. The
native overloads transform RGB only, because alpha is coverage and the
exporters keep it linear. The float test keeps alpha 0.25 unchanged, where
r187 would produce about 0.051. The byte test keeps alpha code 128, where r187
would produce 55.

**Byte conversion rounds to nearest.** The r187 raw-byte path computes
`Math.floor( SRGBToLinear( v / 255 ) * 255 )`, so it truncates. Its canvas path
writes the unrounded float into image data, whose clamped byte array rounds when
a browser stores it. That rounding comes from the web platform, not from the
r187 source. The native overload rounds each of the 256 codes to the nearest
linear code with a table. Truncation and rounding differ for 135 of the 256
codes by an independent double-precision computation. For example, code 7 maps
to 1 here and to 0 by truncation, and code 220 maps to 183 and 182. The byte test
compares all 256 codes with that reference and requires at least 100 codes where
the two modes disagree, so a switch to truncation fails it. No 8-bit code lands
within 0.001 of a rounding tie, so the tie rule does not matter.

**The transfer function is the same, with exact constants.** r187
[`ColorManagement.SRGBToLinear`](https://github.com/mrdoob/three.js/blob/r187/src/math/ColorManagement.js) uses `c < 0.04045`, a multiplier of 0.0773993808,
and `pow( c * 0.9478672986 + 0.0521327014, 2.4 )`. The native Float32 version
divides by 12.92 and 1.055 and uses `<=` at the boundary. A test compares 1,005
samples, including −0.5, 2 and the boundary, against the r187 form in Double
within 1e-6, absolute below 1 and relative at or above 1. Neither implementation clamps.

**The API takes pixels, not images.** r187 accepts an image element, canvas,
`ImageBitmap` or `{ data, width, height }`, returns a new canvas or object,
and warns before returning an unsupported input unchanged. The native
overloads take one `SIMD4<Float>` or `SIMD4<UInt8>` pixel. Texture-level color
handling goes through the exporter's `Interpretation`.

**`Binary16.encode` keeps IEEE overflow to infinity.** r187
`DataUtils.toHalfFloat` warns about a magnitude above 65,504, clamps the value
to ±65,504 and then converts through lookup tables that shift the mantissa
down, which truncates toward zero. The native encoder uses Swift's `Float16`
conversion. Values from 65,504 up to, but not including, 65,520 round to the
largest finite half, 65,504 (0x7bff). Values of 65,520 and above, including 1,000,000, give infinity
(0x7c00), and negative values give 0xfc00. Infinities and NaNs are preserved.
Rounding is to nearest, ties to even, so 1 + 3 × 2⁻¹¹ encodes as 0x3c02 where
truncation gives 0x3c01. A caller who wants r187's saturation clamps before
encoding. `Binary16.decode` is exact for every finite pattern and infinity, and
NaN patterns decode to NaN.

**The default PNG interpretation does not dump stored bytes.**
`pngDataURL` uses `.encodedSRGB`, as `GLTFExporter.pngData` does. That writes
sRGB-encoded 8-bit codes. Only an sRGB source format (RGBA8 or BGRA8 sRGB)
copies its stored bytes. Linear formats (RGBA8, BGRA8, R8, RG8) and float
formats are clamped to 0...1 and sRGB-encoded, so the default PNG of a linear
RGBA8 texture holds different bytes than the texture. `.linearData` writes
linear values instead, which decodes sRGB sources and copies linear byte
formats. Alpha is always linear and is copied or clamped. The r187
`ImageUtils.getDataURL` performs no color conversion. It returns an existing
`data:` URL unchanged or re-encodes what the canvas draws.

The reference fixtures compare r187 wide, tall and equal aspect ratios and
odd-size BC, EAC, ASTC and minimum PVRTC payloads. Native tests verify overflow
and unsupported formats, all 65,536 binary16 encodings, rendered Metal cover
and contain samples, and a data URL round-trip that preserves low-alpha pixels.

## Verification

Xcode 27.2 command-line `xcodebuild` ran `TextureUtilitiesTests` and
`TextureUtilitiesMetalTests` with Metal API Validation, together with the KTX2
export, KTX2 import, spatial import and cube loader suites. The macOS run
reported "Test run with 42 tests in 7 suites passed" and the iPhone 18 Pro
Simulator run reported 43 tests in 7 suites passed, with four tests skipped
for missing native cube-array or BC sampling support. Ten tests belong to
`TextureUtilitiesTests` and three to `TextureUtilitiesMetalTests`, counted from
the passing test names in the macOS log. Both utility suites passed on both
destinations.

Two tests were strengthened after an independent review. The byte sRGB test
previously used codes whose truncated and rounded results are equal, so it
could not tell rounding from truncation. It now covers all 256 codes and the
codes that separate the two. The packed-size test previously listed a subset of
the formats the layout table accepts. The new test sizes every accepted format
at four dimensions against block geometry written out in the test, covers
all three ASTC encodings for the 14 footprints and eight PVRTC1 formats at seven
power-of-two sizes against the WebGL formula, and requires every other Metal
pixel format raw value from 0 to 1,023 to throw `pixelFormat`. A Metal test now
checks that the default PNG of a linear RGBA8 texture holds sRGB-encoded codes.

The Xcode MCP bridge timed out, so command-line `xcodebuild` replaced it.

