# KTX 1.0 textures

`TextureLoader.loadKTX1` accepts data or a local file and returns a
shader-readable Metal 2D texture or complete cube. The ordinary
`TextureLoader.load` path also recognizes 2D KTX 1.0. Both byte orders are
supported. Cube storage is level-major, with faces in +X, −X, +Y, −Y, +Z,
−Z order. Set `loadMipmaps: false` to upload only the base level. A
zero-level header uses its stored base level.
[Khronos KTX 1.0 specification](https://registry.khronos.org/KTX/specs/1.0/ktxspec.v1.html)

The loader recognizes the compressed format families used by Three.js r187.
These include DXT1 RGB and RGBA, DXT3, DXT5, signed and unsigned RGTC,
signed and unsigned BC6H, BC7, ETC1, ETC2, EAC, all 14 LDR ASTC footprints,
and PVRTC1 RGB or RGBA at two or four bits per pixel. `.color` honors
a declared sRGB encoding, `.encodedSRGB` selects sRGB where the format
offers it, and `.linearData` uses linear sampling. Signed data and BC6H
retain their signed or float sample values.
[Three.js r187 KTXLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/KTXLoader.js)
[Three.js r187 WebGL format mappings](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLUtils.js)

Metal samples BC when available. The shared DDS decoder expands BC to RGBA8,
RGBA8Snorm, or RGBA16Float when the GPU lacks BC sampling. Opaque DXT1
and punch-through DXT1 retain distinct alpha behavior. ETC and ASTC
use Metal on Apple GPU family 2 and later. The same family samples square
PVRTC1 directly when every selected mip fits the minimum compression
footprint. Rectangular power-of-two images and smaller mips use the
bounded PowerVR decoder. It crops the minimum footprint and preserves
opaque RGB or RGBA semantics. Packed RG data retains its channels.
[PowerVR Native SDK](https://github.com/powervr-graphics/Native_SDK/blob/63f0098acdc11ec9a072c30000d2fa670a955bee/framework/PVRCore/texture/PVRTDecompress.cpp)
[PowerVR MIT license](../../LICENSES/PowerVR-MIT.txt)
[bcdec MIT license](../../LICENSES/bcdec-MIT.txt)

The parser validates the byte-order marker, compressed GL fields, dimensions,
metadata sizes and padding, and every stored mip and face before allocation.
It accepts top-left orientation or omitted orientation. Alternate orientations,
nonidentity swizzles, uncompressed textures, arrays, and volumes are outside
this implementation. The limits are 64 MiB encoded, 8,192 pixels per axis,
and 16,777,216 base texels across all faces.

## Verification

All nine KTX assets bundled with r187 matched the pinned JavaScript loader
in dimensions, mip counts, and stored block payloads. At 16 sampled
coordinates, the M4 Max Mac and iPad Pro 13-inch M5 Simulator agreed within
one 8-bit code. Both PVRTC1 assets matched the full reference RGBA output
from the original PowerVR decoder across all 20 mips. Native Metal base
levels matched the CPU samples at the compared coordinates.

The focused tests sample both byte orders, mip and cube order, sRGB ASTC,
all four PVRTC1 formats, rectangular and subminimum PVRTC1 fallback,
BC1 alpha modes, RGTC, BPTC, and EAC. Clang AddressSanitizer and
UndefinedBehaviorSanitizer passed 2,940 randomized, unaligned PVRTC1
surfaces and 98 rejected truncated lengths. The final five-suite run covered
34 cases on macOS and iPad Simulator with Metal API validation. The iPad
run skipped one existing KTX2 test that requires native BC sampling, and
exercised the KTX1 and DDS BC fallbacks.
[Three.js r187 KTX demo](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_texture_ktx.html)

Xcode 27.2 MCP timed out resolving the package destination. The Lab scheme
lacks package tests, so verification used `xcodebuild` with internal
DerivedData.
