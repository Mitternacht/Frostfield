# PBR bump maps

`PhysicallyBasedMaterial.BumpMap` takes a sampleable, linear Metal 2D
texture through the existing `TextureBinding`. The red channel stores
height. UV0 or UV1, texture transforms and sampler settings apply to
all three height samples. The signed, finite scale multiplies the
height change across the local sampling footprint. Zero leaves the
shading normal alone. A bound base normal texture takes precedence,
including an object-space or packed normal texture.
[Three.js r187 normal-map precedence](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/normal_fragment_maps.glsl.js)

Raster PBR evaluates forward height differences along screen-space
UV derivatives. Path tracing applies the same surface-gradient math
along the two axes of its ray-cone footprint. Both use the footprint
for mip selection. The resulting tilt follows the sampled footprint,
so a change in camera distance, resolution or ray spread can change
its magnitude. The bump perturbs the base shading normal. The mesh
shape, silhouette, and separate clearcoat normal stay as authored.
[Three.js r187 bump shader](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/bumpmap_pars_fragment.glsl.js)

Native archive version 14 preserves the binding and scale. A material
rebuild from live factors retains the bump map. Portable glTF and
USDZ export require conversion of an active height map to a
tangent-space normal texture. This API covers the physically based
material's base layer. Classic materials, a graph-specific bump node,
and automatic importer conversion are outside this slice.

## Verification

The focused suite passed 58 tests on macOS and 58 on iPad Simulator
with Metal API and shader validation. An analytic red-channel ramp
checks raster surface normals and lit pixels, signed scale, mirrored
UV1, normal-map precedence, zero and constant heights, and a separate
UV0 clearcoat frame. First-hit normal AOVs match the same analytic
ramp on native and baked texture sampling. The Mac exercised software
and Metal ray tracing. The simulator exercised software ray tracing.
The suite also checks archive and live-factor rendering and rejects
an archive that labels bump data as an earlier schema version.

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData.
