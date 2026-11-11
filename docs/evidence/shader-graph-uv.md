# Typed shader graph UV projections

- `equirectUV(direction:)` maps an explicitly supplied unit direction to a
  panorama. +X is the center longitude, +Z is one quarter-turn to its
  right, and +Y is the top pole. The seam is on −X. The explicit direction
  also works with computed directions outside a surface graph. Three.js
  can instead default to its world-space direction varying.
  [Three.js r187 equirectUV](https://github.com/mrdoob/three.js/blob/r187/src/nodes/utils/EquirectUV.js)
- `matcapUV()` uses the surface graph's world normal and viewer direction
  and the upper-left 3 × 3 of its view matrix. The explicit overload
  accepts view-space vectors wherever the graph provides them. It
  normalizes both vectors and uses the r187 0.495 disk scale. Vertical
  view directions choose a +X horizontal axis, and a zero direction
  uses +Z. [Three.js r187 matcapUV](https://github.com/mrdoob/three.js/blob/r187/src/nodes/utils/MatcapUV.js)
- `spritesheetUV(_:uv:frame:)` addresses a row-major atlas from its
  top-left tile. The zero-based frame rounds down and wraps in either
  direction. The default frame is zero and the default local UV is
  UV0. Pass explicit UVs for graphs without mesh coordinates. Finite
  grid dimensions round down to whole cells and stay within the exact
  Float range for 2²⁴ tile indices. Local UVs pass through unclamped.
  [Three.js r187 spritesheetUV](https://github.com/mrdoob/three.js/blob/r187/src/nodes/utils/SpriteSheetUV.js)

All three return top-left coordinates for native texture sampling.
Each helper emits existing typed expressions, so graph stage
validation and scene archive support follow the supplied inputs.

## Verification

The pinned r187 TSL functions were evaluated with a CPU numeric
adapter, then compared with Float32 Metal buffer readback after
converting the UV origin. Cases cover panorama cardinal directions,
poles and both sides of the seam, as well as asymmetric matcap inputs,
sprite row transitions and negative or overflowing frames.

RGBA16Float raster pixels verify a rolled camera, an off-axis
perspective ray, and a live frame selecting texels from a six-color
atlas. An orthographic path-traced frame verifies the same rolled
matcap basis with parallel rays. All 62 focused tests across nine
suites passed on the M4 Max Mac and iPad Pro M5 Simulator with
Metal API Validation enabled.
