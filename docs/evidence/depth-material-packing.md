# Depth and distance material packing

`ClassicMaterial(model: .depth, depthPacking: ...)` selects an
explicit Three.js r187 depth encoding. The raw values match the
corresponding `DepthPacking` constants.

| Mode | Raw value | Output |
|---|---:|---|
| `.basic` | 3200 | One minus raster window depth in RGB, material opacity in alpha |
| `.rgba` | 3201 | Base-256 depth across RGBA, most significant digit first |
| `.rgb` | 3202 | Base-256 depth across RGB, alpha one |
| `.rg` | 3203 | Base-256 depth across RG, blue zero and alpha one |

The explicit modes use the current window depth under forward or
reversed Z. With reversed Z, Basic inverts the reversed depth, as the
r187 shader does. Omitting `depthPacking` keeps Frostfield's near-white
grayscale in either convention. [Three.js r187 depth shader](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderLib/depth.glsl.js) ·
[Packing functions](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/packing.glsl.js)

For the distance model, set `distanceOutput: .red` to put the clamped
normalized world distance in red, with green and blue zero and alpha
one. The existing default copies the distance to all three color
channels. [Three.js r187 distance shader](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderLib/distance.glsl.js)

Use a single-sample linear target with blending disabled. A linear
`rgba8Unorm` target stores the packet bytes directly. Red-only
distance can use a one-channel target, and RG depth can use a
two-channel target. Material cutouts use the existing classic
coverage path before the output is written.

SceneArchive version 13 preserves explicit output modes. Scenes that
use the defaults retain the earliest version required by their other
features. Older archives restore their original grayscale defaults.

## Verification

Metal RGBA8 readback matched the pinned r187 packing formulas byte for
byte at both endpoints, immediately below one, and for fractional depths
that exercise all four channels. Separate checks cover reversed Z,
radial distance and clamping, alpha cutouts, and rejection of
incompatible target formats and blending. Version 13 capture and
restore preserve the rendered bytes. A version 1 archive keeps its
default grayscale output.

All 75 focused tests across 11 suites passed on the M4 Max Mac and
iPad Pro M5 Simulator with Metal API Validation enabled.
