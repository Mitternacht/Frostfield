# Native KTX2 cubemaps and arrays

`TextureLoader.loadKTX2(data:interpretation:mipmapped:)` loads complete
cubemaps and 2D arrays directly into shader-readable Metal textures. A
declared layer count of one returns a one-layer array. The ordinary image
loader still returns a 2D texture for a singleton layer and rejects cubes
and multilayer files.

Each stored mip is uploaded using KTX2's layer, face, depth-slice, and
block-row order. Cube faces are +X, −X, +Y, −Y, +Z, and −Z. Levels are
indexed from largest to smallest. The loader checks spatial byte ranges
for overlap, storage order, and applicable block alignment. It rounds
both dimensions up to full blocks and checks the complete byte count for
all slices before allocating the texture.
[Khronos KTX 2.0 specification](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html#_levelimages)
[Metal texture slice upload](https://developer.apple.com/documentation/metal/mtltexture/replace(region:mipmaplevel:slice:withbytes:bytesperrow:bytesperimage:))

The native formats remain BC1 RGBA, BC3, BC7, ETC2 RGB8, RGB8A1, and RGBA8,
the 14 LDR ASTC footprints, R8, RG8, RGBA8, BGRA8, and R, RG, and RGBA
half and full floats. BC uses Metal's BC capability and ETC2/ASTC require
Apple GPU family 2 or later. Native payloads can be stored or compressed
with zlib. Compressed textures keep their stored mip levels. Uncompressed
textures can generate mips under the existing loader policy. The
interpretation preserves declared sRGB in color mode and permits an
explicit linear or sRGB sampling choice where the format offers it.

Input is limited to 64 MiB encoded, 8,192 pixels per side and 16,777,216
base texels across all faces and layers. Arrays permit up to 2,048 layers.
The existing uncompressed volume path retains its 2,048-pixel axis limit.
Cube arrays, incomplete cubes, volume arrays, block-compressed volumes, and
Zstandard for native formats remain outside this import path. A cube-array
file fails with `unsupportedKTX2("cube map arrays")`. The explicit loader
restores 2D arrays only; the [native exporter](ktx2-export.md) writes both 2D
arrays and cube arrays, so a cube-array export can be inspected but not yet
reloaded as a cube array.
ETC1S and UASTC decoding have a separate [Basis import path](basis-universal.md).

The pinned Three.js r187 loader creates a compressed cube from a
recognized native Vulkan format. Its compressed array path comes through
Basis transcoding. Frostfield's new array path takes native Vulkan
formats directly.
[Three.js r187 KTX2Loader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/KTX2Loader.js)

## Verification

The focused Swift Testing suite samples all faces and mips of distinct
ASTC and BC1 cubes, a zlib-compressed odd-sized ASTC array with multiple
blocks per layer, and independently generated mips of a half-float HDR
array. It also checks singleton compatibility and malformed headers,
ranges, layer counts, and alignment.

Six Khronos KTX-Software v4.4.0 fixtures matched the metadata and
stored mip SHA-256 digests from the pinned r187 parser. These include
singleton and seven-layer RGBA8 arrays, seven-layer ASTC, ETC2, and
BC3 arrays, and a 1,024-square ASTC cube made by retaining the original
fixture's lower 11 mips. The cube also matched the r187 loader's digest
over all 66 face-mip payloads in face-first order. One GPU sample on
each face/layer and mip matched a separate 2D upload exactly. The
comparison covered 151 samples on Mac and 144 on iPad Simulator,
which reported native BC unavailable for the seven BC3 array slices.
[Khronos v4.4.0 fixture corpus](https://github.com/KhronosGroup/KTX-Software/tree/v4.4.0/tests/testimages)

Seven focused suites passed 44 tests on macOS and 45 test declarations
on iPad Pro 13-inch M5 Simulator, with two native BC tests skipped on
the simulator. The suites include existing KTX2 import and export,
DDS, KTX1, PVR, and raster texture upload regressions.

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API validation.
