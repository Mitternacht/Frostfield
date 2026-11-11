# Basis Universal textures in KTX2

`TextureLoader.load(data:interpretation:mipmapped:)` decodes 2D ETC1S and
UASTC LDR KTX2 textures to Metal RGBA8. UASTC levels can be raw or Zstandard
compressed. Stored mip levels and straight alpha are preserved. The default
interpretation follows the KTX2 data-format transfer field. Callers can
explicitly sample the same bytes as sRGB color or as linear material data.
`TextureLoader.loadKTX2(data:interpretation:mipmapped:)` also loads
complete ETC1S and UASTC cubemaps and 2D arrays, including a declared
singleton array. The ordinary image loader and glTF texture preparation
retain their 2D Basis contract.

The glTF importer selects valid `KHR_texture_basisu` images, including on
assets that require the extension. When an optional KTX2 source fails during
texture preparation and the asset provides a core PNG or JPEG source,
preparation uses that source instead.

The decoder-only integration pins Basis Universal 2.50 at
`9bebe16726b3a61c8c213eeee3b7cffb462ef34e`. It includes the accompanying
decoder-only Zstandard amalgamation. A private C boundary keeps C++ and
Zstandard symbols separate from other consumer dependencies. Frostfield
checks the KTX2 dimensions, metadata, level index, ETC1S codebook sizes,
uncompressed UASTC byte counts and raw block alignment before entering the
decoder. Spatial layouts check level order and byte ranges. ETC1S descriptors
validate the RGB slice and any second slice for every mip, layer, and face.
Swift reuses one slice buffer per mip and finishes that mip before advancing,
so Zstandard inflates each level once. Encoded input is capped at 64 MiB.
The base level is limited to 8,192 pixels per axis, 2,048 array layers, and
16,777,216 texels across all faces or layers. Upstream copyright notices and licenses are in `LICENSES`.
[Reference transcoder](https://github.com/BinomialLLC/basis_universal/tree/v2_50/transcoder) ·
[glTF extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)

`KTX2LoaderTests` uses test-authored RGBA quadrants encoded by the pinned
upstream CLI as ETC1S, UASTC and Zstandard UASTC. Metal samples from all
four stored levels are compared with reference texels decoded by the CLI.
Tests also cover alpha, transfer interpretation, truncated files, oversized
headers, out-of-range offsets, misalignment and invalid Zstandard data.
`GLTFExtensionPolicyTests` exercises required extension material preparation
and the fallback after a corrupt optional payload. The combined macOS and
iPhone Simulator Xcode suites passed these checks. An arm64 sanitizer
harness exercised 5,160 truncated and mutated inputs under AddressSanitizer
and UndefinedBehaviorSanitizer.

`KTX2BasisSpatialTests` covers ETC1S and Zstandard UASTC arrays with
distinct RGBA at each of four authored mips and three layers, a raw UASTC
singleton with generated mips, and a raw UASTC cube with distinct faces
and stored mips. The fixtures were encoded with Khronos KTX-Software
v4.4.0 and sampled against its RGBA8 transcode. Corrupt per-layer ranges,
missing second slices, predicted frames, and unsupported layouts are
rejected. AddressSanitizer and UndefinedBehaviorSanitizer exercised 7,976
truncated and mutated spatial inputs, valid unaligned buffers, and invalid
face and layer indices.

The Khronos Golden Gate UASTC+Zstandard cube and a Yokohama ETC1S cube
with its original lower 11 mips matched the Khronos CLI's RGBA8 output
byte for byte on Mac and iPad Pro 13-inch M5 Simulator. Yokohama's original
2,048-square base exceeds the aggregate budget. Its 1,024-square derivative
retains the lower mips, their image descriptors, and the original codebooks.
All 132 face/mip GPU samples per platform matched the CLI's texels within
0.00013 in linear sampling space. Xcode 27.2 MCP timed out resolving the
package destination, so the focused suites used `xcodebuild` with internal
DerivedData and Metal API validation.
[Khronos cube fixtures](https://github.com/KhronosGroup/KTX-Software/tree/v4.4.0/tests/testimages)

## Device-compressed storage

`loadKTX2(data:interpretation:mipmapped:basisStorage:)` accepts
`.deviceCompressed` to transcode ETC1S and UASTC LDR directly to a Metal
block format. On Apple GPU family 2 or later, UASTC selects LDR ASTC 4×4.
ETC1S selects ETC2 RGB8 for opaque input and ETC2 EAC RGBA8 when the source
has a second slice. A device with BC sampling and without the Apple block
formats selects BC7. Declared transfer and explicit sRGB or linear sampling
use the corresponding Metal format without changing the transcoded blocks.
Stored mips, cube face order, array layer order, and alpha semantics are
preserved. The opt-in reports an error if no target is supported or if the
caller requests generated mips for a single-level image larger than 1×1.
Files with an explicit native pixel format still upload their stored format.
[Three.js r187 target selection](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/KTX2Loader.js)
[Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
[Metal BC capability](https://developer.apple.com/documentation/metal/mtldevice/supportsbctexturecompression)

Metal readback matched exact block SHA-256 digests from the pinned Basis
transcoder's per-image API for alpha ETC2 and ASTC arrays, a raw UASTC
ASTC cube, and an opaque ETC2 RGB image with odd dimensions and sub-block
mips. Shader samples check mip and slice selection, source RGBA, and the
sRGB transfer. The BC7 bridge matched its reference digest and sampled on
Mac. The target selector separately checks BC7-only and unsupported
capabilities. Focused verification passed 53 tests on Mac and 54 test
declarations on iPad Simulator, with three native BC tests skipped on
the simulator. AddressSanitizer and UndefinedBehaviorSanitizer covered
9,720 truncated and mutated inputs and checked undersized output buffers
for each selected compressed target.
[Pinned Basis Universal transcoder](https://github.com/BinomialLLC/basis_universal/blob/9bebe16726b3a61c8c213eeee3b7cffb462ef34e/transcoder/basisu_transcoder.h)

The original overload, ordinary image loading, and default glTF material
preparation continue to use RGBA8 for LDR. Raw `.basis` files, animated arrays, cube
arrays, incomplete cubes, volumes, alternate orientations, and
premultiplied Basis inputs remain outside this loader.

## Scene-linear HDR

The default KTX2 loader decodes HDR Basis profiles to `.rgba16Float`.
`loadKTX2` supports complete cubes and 2D arrays, and ordinary image
loading accepts 2D HDR Basis. All interpretations sample these textures in
scene-linear space and retain radiance above one. The loader requires
linear DFD transfer. RGB remains in the source primaries and Metal
receives float samples without a gamut transform. The pinned encoder's HDR
profiles produce opaque RGB. Explicit LDR storage choices reject HDR, and
glTF `KHR_texture_basisu` remains on its LDR contract.

- UASTC HDR 4×4 uses `ASTC_4x4_SFLOAT` with DFD model 167. It accepts
  raw and Zstandard blocks.
- ASTC HDR 6×6 uses `ASTC_6x6_SFLOAT` with DFD model 162. It accepts
  raw and Zstandard blocks.
- UASTC HDR 6×6 intermediate uses undefined Vulkan format with DFD model
  168. The loader accepts scheme 4 with a per-slice profile and the older
  BasisLZ descriptor layout. It checks the complete per-mip descriptor
  table, bounds each encoded slice within its mip, and supports the
  `0xABCD` and `0xABCE` bitstream profiles.

Native compressed HDR storage, other ASTC HDR footprints, and nonlinear
HDR transfer remain outside this path.
[Basis Universal KTX2 profiles](https://github.com/BinomialLLC/basis_universal/wiki/KTX2-File-Format-Support-Technical-Details)
[Khronos KTX transcode targets](https://github.khronos.org/KTX-Software/ktxtools/ktx_transcode.html)

Metal readback on Mac and iPad Simulator matched every RGBA16Float bit in
three Khronos KTX-Software v5.0.0-rc2 Desk fixtures. The two UASTC results
also matched the Khronos CLI's RGBA16F transcode. The ASTC 6×6 result used
the direct pinned Basis decoder reference. Native shader samples matched
all 12 reference texels on each platform, including peak RGB values of
231 and 251. Authored fixtures verify all six faces and stored mips of a
raw HDR cube, distinct layers of a profile `0xABCE` array, and odd-sized
mips. Malformed transfer, footprint, block size, slice range, and profile
data are covered. AddressSanitizer and UndefinedBehaviorSanitizer passed
10,104 truncated and mutated inputs and undersized output buffers.
[Khronos HDR fixtures](https://github.com/KhronosGroup/KTX-Software/tree/v5.0.0-rc2/tests/resources/ktx2)
[KTX2 BasisLZ image descriptors](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html#basislz_global_data_structure)
[Khronos KTX-Software v4.4.0](https://github.com/KhronosGroup/KTX-Software/releases/tag/v4.4.0)


## glTF material preparation

Pass `basisTextureStorage: .preferDeviceCompressed` to
`PreparedGLTFScene` or `GLTFScene.Material.prepare` to select native
compressed storage for glTF Basis LDR maps. Preparation keeps color maps
on sRGB sampling and data maps on linear sampling. It retries RGBA8 if
the compressed upload fails or a single stored level needs generated mips.
The original overloads default to RGBA8. A texture's optional PNG or JPEG
source still applies when the KTX2 image fails decoding or its format
is unsupported.

Three.js r187 delegates the extension image to its configured KTX2
loader and marks color material slots sRGB. The Khronos extension
defines 2D ETC1S and UASTC LDR material images. Frostfield's existing
acceptance of native KTX2 payloads through this extension is importer
compatibility behavior.
[Three.js r187 GLTFLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/GLTFLoader.js)
[Khronos KHR_texture_basisu](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)

The focused scene uses ETC1S for sRGB base color and emission and
Zstandard UASTC for linear metallic-roughness. Metal sampled the actual
prepared maps. Raster color and path albedo and roughness matched the
corresponding texels through the direct texture table on Mac and the
baked table on Mac and iPad Simulator. The iPad Simulator lacks tier 2
argument buffers. The fallback tests cover generated mips, retaining
compression for a sampler without mip filtering, optional PNG fallback,
and error propagation for a damaged required Basis source. The eight-suite
run reported 49 test declarations on each platform. Xcode 27.2 MCP timed
out resolving the package destination, so verification used `xcodebuild`
with Metal API validation.
