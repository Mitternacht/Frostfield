# Classic bump and normal maps

`ClassicMaterial` accepts height and normal detail for Lambert, Phong,
toon, matcap and normal visualization. A `BumpMap` uses a linear red
channel and signed strength. It shares the raster PBR surface-gradient
function. Zero strength leaves the shading normal alone. A bound normal
map takes precedence over height, as in Three.js r187. The active map
uses its own sampler, UV set and transform. Normal and bump detail keep
the original geometry and silhouette.

Normal maps may use RGB or a packed XY encoding. Tangent-space maps
apply the two signed `normalScale` components and the frame for the
selected UV channel. Object-space normals use the inverse-transpose
transform and ignore the tangent scale. The same detail reaches the
lit material, matcap, normal visualization and the surface-attribute
pass. Classic materials use the raster renderer. Use
[`PhysicallyBasedMaterial.BumpMap`](pbr-bump-maps.md) for path tracing.

Native scene archive version 15 preserves classic height and normal
bindings and scale. Scenes with no classic detail still use their
earliest compatible format. Archives preserve an authored bump map
even when a normal map is active.

[Three.js r187 material](https://github.com/mrdoob/three.js/blob/r187/src/materials/MeshPhongMaterial.js) ·
[Three.js r187 normal selection](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/normal_fragment_maps.glsl.js) ·
[Three.js r187 bump shader](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/bumpmap_pars_fragment.glsl.js)

## Verification

All 49 focused tests passed on Mac and the iPad Pro M5 simulator with
Metal API Validation. An analytic ramp checks signed UV1 height gradients
through all five material models, mirrored UVs, zero scale, and a Phong
lighting change. A normal-map fixture verifies precedence, separate
signed axes, and inverse-transpose object normals. A version 15 archive
renders byte-identical pixels after restore and rejects a version 14
manifest containing the new fields. Existing PBR bump, packed normal,
spotlight, classic material and material override tests passed alongside
the new cases.

The Xcode workspace bridge timed out resolving the initial run
destination. The runs used Xcode 27.2 `xcodebuild` with internal
DerivedData.
