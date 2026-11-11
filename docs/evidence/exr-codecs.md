# Lossless OpenEXR scanline codecs

`EXRExporter.Compression` supports the five scanline modes exposed by Three.js
r187. Select `.rle` for one scanline per block or `.piz` for up to 32 scanlines
per block. Both options work with RGBA pixels and Metal textures, mixed named
channels, streamed files, and `PathTracer.writeStillEXR`. ZIP remains the
default. [Three.js r187 EXRExporter](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/exporters/EXRExporter.js)

RLE uses the existing ZIP byte reordering and delta predictor. PIZ rearranges
the same scanline payload into channel planes and preserves the two 16-bit
words of each float and UInt32 sample. Both codecs store the original block
when compression cannot make it smaller. PIZ also stores blocks above 64 MiB
directly to bound compression scratch space. The file keeps its chosen
compression tag, as OpenEXR readers recognize raw blocks by size. The writer
produces single-part scanline files with full-resolution channels.
[OpenEXR file layout](https://openexr.com/en/latest/OpenEXRFileLayout.html)

The private C bridge vendors only the bitmap, wavelet, and bounded Huffman
encoding routines from OpenEXR 3.5.2 at
`69b2604fc76e370615438bdc8d2cd95b9349c12e`. The extracted functions retain
upstream logic and use local linkage. The bridge regroups and validates
Frostfield's scanline data, allocates scratch space, and writes the PIZ prefix.
[Wavelet source](https://github.com/AcademySoftwareFoundation/openexr/blob/69b2604fc76e370615438bdc8d2cd95b9349c12e/src/lib/OpenEXRCore/internal_piz.c)
[Huffman source](https://github.com/AcademySoftwareFoundation/openexr/blob/69b2604fc76e370615438bdc8d2cd95b9349c12e/src/lib/OpenEXRCore/internal_huf.c)
[BSD-3-Clause license](../../LICENSES/OpenEXR-BSD-3-Clause.txt)

## Verification

Xcode 27.2 (27B52) passed all 22 focused tests on an M4 Max Mac and the iPad
Pro simulator with Metal API Validation requested. `LayeredEXRTests` checks
streamed and in-memory file equality across all five modes, provider order
and RLE fallback. `HighDynamicRangeImageTests` verifies bit-exact RGBA
decoding through ImageIO for both new modes. `PathRegionTests` compares the
PIZ path-traced beauty image and nine first-hit layers with a full-view
render across both GPU tile boundaries. The Xcode MCP timed out opening the
package, and the Lab scheme lacks package tests. The runs used
`xcodebuild` with DerivedData on the internal drive.

The official OpenEXR 3.4.4 Python reader separately decoded six fixture pairs,
each emitted by both public export paths. The 257 × 65 mixed-channel files
preserved signed zero, subnormals, infinities, NaN payloads, and full UInt32
IDs. Both codecs fell back to raw for the 1 × 1 fixture. PIZ compressed the
first two blocks and stored the final mixed row raw. A 129 × 67 fixture
compressed its three-row final block. A 513 × 33 fixture exercised the
modulo wavelet with 16,416 distinct words in its first block. All corresponding
in-memory and streamed files matched byte for byte.
[OpenEXR Python API](https://openexr.com/en/latest/python.html)

Clang AddressSanitizer and UndefinedBehaviorSanitizer passed 2,368 mixed
PIZ blocks, covering widths 1 through 37 and heights 1 through 32, and
2,368 rejected output capacities.

The integrated exporter checkpoint passed 46 macOS tests and 47 iPhone
18 Pro Simulator tests with Metal API Validation. The Lab built for macOS,
iOS Simulator, and visionOS Simulator in Xcode 27.2.
