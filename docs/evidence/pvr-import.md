# PVR textures

`TextureLoader.loadPVR` loads a PVRTC1 2D texture or complete cube from data
or a local file. Both PVR v2 and v3 accept RGB and RGBA at two or four bits per
pixel, the four compressed formats in Three.js r187's PVR loader. The ordinary
`TextureLoader.load` recognizes 2D PVR textures. Stored mip levels upload by
default. Set `loadMipmaps: false` on the specialized entry point to upload
only the base level.
[Three.js r187 PVRLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/PVRLoader.js)

PVR v3 accepts little- and big-endian headers. Its PVRTC1 words remain
little-endian. The v3 color-space field determines the transfer in
`.color` mode, while v2 defaults to linear. Use `.encodedSRGB` to sample
either version as sRGB or `.linearData` to sample encoded values directly.
V3 stores mips first and can declare a permutation of its six cube faces.
Cube slices use +X, −X, +Y, −Y, +Z, −Z. V2 uses a little-endian legacy header.
Explicitly flagged v2 cubes follow the PowerVR SDK's face-first layout.
Unflagged v2 files with six surfaces follow r187's mip-first cube convention.
[PowerVR header](https://docs.imgtec.com/specifications/pvr-file-format-specification/html/topics/pvr-header-format.html)
[PowerVR metadata](https://docs.imgtec.com/specifications/pvr-file-format-specification/html/topics/pvr-metadata.html)
[PowerVR texture layout](https://docs.imgtec.com/specifications/pvr-file-format-specification/html/topics/pvr-texture.html)
[Pinned PowerVR SDK legacy reader](https://github.com/powervr-graphics/Native_SDK/blob/63f0098acdc11ec9a072c30000d2fa670a955bee/framework/PVRCore/textureio/TextureReaderPVR.cpp)
[Khronos PVRTC data format](https://github.com/KhronosGroup/DataFormat/blob/main/pvrtc.txt)

PVR and KTX 1.0 share Metal's PVRTC1 path on Apple GPU family 2 and later.
The bounded PowerVR CPU decoder expands rectangular power-of-two images,
selected mip levels below the minimum native compression footprint, or
images on GPUs that lack PVRTC1 sampling. RGB formats always return alpha
one. RGBA formats retain the decoded alpha.

The limits are 64 MiB encoded, 8,192 pixels per axis, and 16,777,216 base
texels across all faces. The parser checks every declared mip and face even
for base-only uploads. PVR v1, PVRTC2, other PVR encodings, arrays, volumes,
incomplete cubes, alternate orientations, borders, premultiplied textures,
and supercompression are outside this implementation. Atlas and normal map
metadata are advisory and leave sampled texels unchanged.

## Verification

All eight Three.js r187 PVRTC demo assets matched the pinned JavaScript
loader in dimensions, encodings, face and mip counts, and exact payload
hashes. A GPU sample on each of the 74 stored surfaces and on all 18
native base-only surfaces matched between macOS and iPad Simulator. Those
samples also matched the original PowerVR decoder to float rounding
(maximum absolute difference below 0.00000004).
[Three.js r187 PVRTC demo](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_texture_pvrtc.html)
[PowerVR decoder](https://github.com/powervr-graphics/Native_SDK/blob/63f0098acdc11ec9a072c30000d2fa670a955bee/framework/PVRCore/texture/PVRTDecompress.cpp)

The focused suite covers both v3 header byte orders, sRGB, all four v2
formats, cube permutation and both v2 cube layouts, native and fallback
samples, malformed inputs, and DDS/KTX1/KTX2 regressions. Xcode 27.2 MCP
timed out resolving the package destination. Verification used
`xcodebuild` with internal DerivedData and Metal API validation.
