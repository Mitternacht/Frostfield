# Directional shadows through colored glass

Set `transmissionLayers` on `DirectionalShadowMap` or `SunLight` to
capture colored glass shadows for PBR receivers. The default zero
uses the ordinary opaque depth and visibility path. A nonzero value
enables one to four layers per cascade. The extra color and scratch
depth textures have a combined 128 MiB allocation limit.

The renderer peels each layer in light-space depth, storing RGB
transmittance and the depth of the glass in an RGBA32Float array.
Each receiver multiplies the layers in front of its light-space
depth. It resolves each bilinear tap before averaging and combines
the result with opaque shadow visibility. Cascades blend in receiver
space and the authored shadow strength controls the combined effect.
The RGB light color then scales the resulting direct contribution.

This slice supports stock PBR glass on plain and skinned meshes. Base
color and transmission amount tint the straight shadow path, and
thickness with attenuation distance and color applies Beer–Lambert
absorption. The caster can also carry authored maps. The colored pass
samples the base color, the metallic channel of the metallic-roughness
map, the transmission factor, the thickness and the alpha map. Each map
keeps its own UV set, transform and sampler. sRGB base color decodes in
the sampler, and an alpha mask cutout discards fragments. Metallic
reduces the transmitted share the same way the receiver shader does.
Maps are filtered with the shadow pass's own derivatives, so a minified
map uses the sampler's mip filter at the shadow-texel footprint. No test
checks mip selection. Graph-authored casters are described below. Punctual maps,
colored transmission for classic receivers, and instanced, morphed or
vertex-colored stock glass remain open. The raster shadow pass does not bend
or focus light.

Single-map and cascaded fixtures compare open light, one and two
overlapping panes, a receiver between the panes, and an opaque
blocker against independent RGB expectations. They also compare the
opaque depth bytes, check half-strength visibility and verify that
on-demand layer textures refresh with the map. The isolated candidate
passed 43 focused tests on Mac and iPhone Simulator under Metal API
Validation. The integrated fixture also checks that changing light
color reuses the existing depth and glass layers.

[Three.js r187 shadow sampling](https://github.com/mrdoob/three.js/blob/r187/src/nodes/lighting/ShadowNode.js) ·
[Three.js r187 custom shadow color](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/NodeMaterial.js)

The textured-glass fixture compares a base color, metallic, transmission,
thickness and alpha-cutout pane against independent expectations, and then
changes the light color on an on-demand map. The tinted result scales by the
light color and the glass layers are not rendered again. After rebasing onto
the integrated sun and fog work, the focused glass, sun and cascade suites
passed on macOS arm64 and on the iPad Pro (M5) simulator under Metal API
Validation, as part of 58 tests in 12 suites. Xcode's bridge reported a full
workspace volume, so verification used `xcodebuild` and internal build caches.

## Graph-authored casters

A PBR `SurfaceShaderGraph` can author its own colored shadow with two
outputs. `castShadowColor` is a float4: straight RGB transmission and an
alpha coverage, both clamped to 0...1, and a nonfinite result casts no
shadow. `castShadowMask` is an optional scalar that requires
`castShadowColor`. Both are available on the node-array `Outputs`
initializer and in the typed `SurfaceShaderGraph(building:)` result.
They read the same material inputs as the surface: UV sets, world
position, normal, the sampled base color, uniforms, textures and the
draw's `parameters`.

With `transmissionLayers` above zero, a graph caster leaves the opaque
depth pass and joins the peel layers beside stock glass. With zero
layers, and in spot and point maps, it casts through the ordinary
opaque depth path from its coverage outputs, exactly as a graph without
cast outputs does. The generated shadow-coverage source of a graph that
authors no cast output is unchanged.

Coverage follows stock glass. The sampled base alpha, scaled by the
graph `opacity`, meets the graph `alphaTest` or the material's mask
cutoff. In opaque and mask modes a surviving fragment then covers fully,
and in material-alpha and blend modes it keeps that alpha. The authored
alpha multiplies the result. A mask-mode cutoff therefore tests the base
alpha scaled by the graph `opacity` and never the authored alpha, and
opaque mode ignores the base alpha.

`castShadowMask` replaces `mask` in every raster shadow pass:
directional, colored directional, spot and point. The color pass and
the path tracer keep reading `mask`. The same graph therefore casts the
same texels for any light type and any layer count, and a hidden mesh
can still cast. A graph without a `castShadowMask` shadows by `mask`.

The colored pass draws both faces of a graph caster, as the ordinary
depth pass does, so enabling layers does not change which faces cast.
Stock glass keeps its material's face culling.

Camera, viewport, soft-particle and directional shadow-visibility
inputs in the cast outputs fail when the graph is built or decoded
with `SurfaceShaderGraph.Failure.castShadowInput`, or
`directionalShadowCoverage` for the mask. A camera read in `opacity`,
`alphaTest` or `mask`, when no `castShadowMask` replaces it, still fails
when the caster is drawn, with `graphShadowCameraInput`, as for any graph
shadow caster. A batched or indirectly drawn caster fails with
`MeshRenderer.Failure.graphCastShadowGeometry`. While layers are on,
both errors are thrown when the caster is prepared for drawing, even on
an on-demand map that is not refreshing.

A scene archive that holds a graph with a cast output is written as
version 29. Every other archive keeps the lowest version that can store
it, and a manifest below 29 that carries the outputs is rejected. Path
tracing and raster color ignore the outputs. The glTF exporter reports
them in `surfaceGraphChannelsDropped`, and the USDZ exporter reports the
dropped surface graph.

Remaining limits for graph casters:

- The shadow geometry is the surface geometry. There is no separate
  shadow position output, and no custom shadow depth or fragment depth
  hook.
- Spot and point maps keep opaque, uncolored coverage. Colored
  transmission for them is open.
- The colored pass is tested with plain and instanced meshes. Skinned
  and morph-target casters use the mesh's own vertex stage there but
  have no colored-shadow fixture yet. Batched and indirect draws are
  rejected rather than drawn.
- A graph that reads a viewport snapshot in its surface outputs and also
  authors cast outputs has no fixture.
- A graph cannot read the camera, a viewport snapshot or the receiver's
  shadow in its cast outputs, because the pass has only light space.
- Only PBR receivers sample the layers. While layers are on, a graph
  caster leaves the opaque depth pass, as stock glass does, so a classic
  or other non-PBR receiver gets no shadow from it.
- A node's `materialParameters` scale the coverage of stock glass but
  not a graph caster's tint, which uses the base color factor and the
  instance tint. The parameters reach the graph only as an input.
- No fixture exercises graph `opacity`, graph `alphaTest`, nonfinite or
  out-of-range authored values, or texture, world position and
  `parameters` reads in the cast outputs. The colored fixtures use
  constants, uniforms and the sampled base color.
- Each crossing tints once. A closed mesh with both faces drawn uses two
  layers, and there is no automatic thickness or refraction.
- The count of layers per cascade and the memory limit stay as above.

Graph fixtures render a floor under a graph caster in single-map and
cascaded maps. They compare the authored RGB tint, two stacked graph
casters, mask precedence across directional, colored directional, spot
and point passes, alpha and mask-cutoff parity with stock glass in
opaque, mask, material-alpha and blend modes, casting through the back
face, unchanged ordinary shadow depth bytes with layers off, instanced
casters, archive round trips with the lowest-version rule, the typed
construction, decode and batched-caster errors, and the path tracer and
glTF and USDZ exporters. Temporarily reverting the face culling, the
coverage rule and the mask precedence together made the fixtures for
each of them fail on Mac. The three graph shadow suites passed 28 tests on Mac and on
iPhone Simulator under Metal API Validation, run with the command-line
`xcodebuild` because the Xcode MCP bridge was unavailable. Twenty-five
related suites passed 157 tests on each platform before two more
fixtures were added, and the full `FrostfieldTests` target passed 2100
tests in 417 suites on Mac.
