# HDR JPEG XL import

`TextureLoader.loadHighDynamicRange` recognizes still JPEG XL
codestreams and containers from either data or a local file. The
decoder returns top-left, upright straight-alpha RGBA in extended
linear sRGB. The existing upload path writes RGBA16Float or RGBA32Float
and generates mips on request. Scene-linear samples retain their
relative values. Tagged PQ and HLG transfer functions use 203 cd/m²
reference white, with `contentHeadroom` reporting the source intensity
target divided by 203 when that ratio is at least one. The conversion
retains out-of-gamut components and skips tone mapping.
[Three.js r187 JXLLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/JXLLoader.js)
[HDR reference white](https://www.w3.org/TR/css-color-hdr-1/)

The HDR entry point has the same 64 MiB encoded-input, 8,192-pixel side,
and 16,777,216-pixel total limits as the other HDR formats. libjxl's
decoder allocator is capped at 512 MiB. The RGBA32Float output buffer
is bounded separately by the pixel limit, and embedded ICC inspection
is capped at 16 MiB. Animated images, JPEG reconstruction and
extraction of metadata boxes are outside the still-texture API.
Ordinary raster texture loading continues to use the system decoder.

## Pinned sources

The SwiftPM C++17 target `FrostfieldJPEGXL` builds vendored source
directly for Apple destinations. The libjxl subset follows the base,
decoder and CMS groups in `lib/jxl_lists.cmake`, with their transitive
headers. It disables JPEG reconstruction and metadata box extraction,
which removes the Brotli dependency. The Highway subset contains the
core headers and runtime sources with baseline SIMD selected at compile
time. SkCMS uses its baseline transform. The generated export and
version headers, private C API prefixes, and private C++ namespaces
isolate the codec symbols from clients' other libraries. The build uses
no external tool or network fetch.
[libjxl v0.12.0 at a7a9c787](https://github.com/libjxl/libjxl/tree/a7a9c787341cf703dede03c2009fa460cae5e5df)
[Highway at 457c8917](https://github.com/google/highway/tree/457c891775a7397bdb0376bb1031e6e027af1c48)
[SkCMS at 96d9171c](https://github.com/google/skcms/tree/96d9171c94b937a1b5f0293de7309ac16311b722)

Licenses and the upstream patent grant are included with the source.
[libjxl license](../../LICENSES/JPEGXL-BSD-3-Clause.txt)
[libjxl patent grant](../../LICENSES/JPEGXL-PATENTS.txt)
[Highway license](../../LICENSES/Highway-Apache-2.0.txt)
[SkCMS license](../../LICENSES/SkCMS-BSD-3-Clause.txt)

## Verification

The pinned 512 × 256 Venice panorama has a reference peak of 1859.4229
and a minimum RGB component of −0.2632083. The native decoder's 393,216
RGB samples matched the independently installed libjxl 0.11.0 reference
bit for bit on Mac. Metal readback on Mac and iPad Simulator also
matched the peak, minimum, and representative pixels. The system
ImageIO HDR decode returned peak 1.0 on Mac, including with float
decoding requested. Core Image's HDR option returned peak 1.1615.
[Three.js r187 Venice JPEG XL fixture](https://github.com/mrdoob/three.js/blob/r187/examples/textures/hdr-native/venice_sunset_512.jxl)

The focused run passed 17 test declarations on each target. The
committed fixtures check lossless Float32 RGBA, half-float sampling
and mips, grayscale and sRGB transfer, an embedded Display P3 profile
with 90° rotation, 1,000-nit PQ and HLG, and truncated input. They
were encoded with libjxl 0.11.0. The Display P3 ICC profile is from the
CC0 Compact ICC Profiles collection.
[Display P3 ICC profile](https://github.com/saucecontrol/Compact-ICC-Profiles/blob/bdd84663061bc4ae95ca70decff54f581e27f702/profiles/DisplayP3-v4.icc)

Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData and Metal API
and shader validation enabled.
