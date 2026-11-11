# Metal texture decompression

`TextureDecompressor` expands a native shader-readable, nonsparse 2D
block-compressed texture into private, tracked storage on the same Metal
device. `decompress` waits for completion. `encode` adds the work to a
retaining, uncommitted command buffer. Use the source's producer queue or
complete its writes before scheduling another queue. With an untracked
source, order the writes and reads explicitly.

Metal samples the source at each texel center with nearest filtering at
the original size. A positive `maximumSize` caps width and height separately,
matching the size rule in the Three.js r187 addon. Each resized level uses
bilinear sampling at that level. The default copies every allocated mip
that fits the output. Set `includeMipmaps: false` for level zero only.
Populate every requested level before copying. The output keeps the
source label and top-left row order. Metal samplers remain separate objects.
[Three.js r187 WebGPUTextureUtils](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/utils/WebGPUTextureUtils.js)

| Source on a compatible Metal device | Output |
| --- | --- |
| Linear LDR ASTC, BC1–5 unsigned, BC7, ETC2, EAC RGBA8, and PVRTC1 | RGBA8Unorm |
| sRGB ASTC, BC, ETC2, EAC RGBA8, and PVRTC1 variants | RGBA8Unorm sRGB |
| BC4/5 signed, EAC R11 or RG11, BC6H, and HDR ASTC | Linear RGBA16Float |

The source's sampled swizzle and alpha carry through the copy. An sRGB
attachment stores the sampled linear RGB with the sRGB transfer and keeps
alpha linear. Half-float output retains negative and extended-range
samples. The output can be sampled or passed to an existing export utility.
The output budget is 8,192 pixels per axis and 16,777,216 base texels.
[Metal pixel formats](https://developer.apple.com/documentation/metal/mtlpixelformat)
[Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

This API accepts native compressed 2D inputs. Cube, array, 3D, multisample,
and sparse resources are outside this path. DDS, KTX, and PVR loader CPU
fallbacks already return uncompressed textures on devices without the
corresponding native compression support.

## Verification

Six pinned r187 KTX and PVR assets ran on macOS and iPad Simulator. Across
three sample positions per stored mip, LDR output matched direct Metal
sampling exactly. Linear half-float EAC output differed by at most 0.0004.
Synthetic fixtures also verify zero-alpha RGB, distinct terminal mips,
signed EAC, PVRTC swizzles, cap scaling, and a KTX2 export and reimport.
macOS verified both BC6H variants and ASTC HDR with values above one.
[Three.js r187 KTX assets](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_texture_ktx.html)
[Three.js r187 PVRTC assets](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_texture_pvrtc.html)

Headless Chrome ran the pinned WebGPU addon on the same 512 × 512 ASTC
asset. Three opaque sRGB pixel codes matched exactly after accounting for
the browser output's `CanvasTexture.flipY` convention. The r187 128-pixel
cap selected source mip 2 at the three sampled coordinates. Frostfield
exposes that stored mip and also supports explicitly scaling the base
level for a capped output.

The final focused run passed 40 tests on macOS and 38 on iPad Simulator.
Three simulator cases were skipped for unavailable BC or HDR ASTC
sampling. Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData and Metal API validation.
