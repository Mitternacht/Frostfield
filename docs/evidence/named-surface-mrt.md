# Named PBR surface outputs

`SurfaceRenderTargets` describes a named raster layout. The `output` attachment
receives the ordinary premultiplied PBR color. Up to seven secondary names map
to straight linear base color, signed world-space shading normal, emission,
metallic, roughness or occlusion. Vector targets use RGBA16Float or RGBA32Float.
Scalar targets use R16Float or R32Float. The caller supplies the Metal
attachments and per-target clear and store actions to `makePass`.

Prepare each PBR mesh with the same layout and pass it to the standalone mesh
or scene graph encoder. The shader samples the surface once, applying material
maps and graph overrides before writing the outputs. Cutout fragments leave
all attachments and depth at their prior values. Existing single-target PBR
and explicitly indexed custom MRT pipelines use their existing entry points.

The fixtures compare named PBR color to ordinary color, read a transformed
object-space normal map with a negative component, and verify that a live
graph changes roughness and cutout coverage without rebuilding. They cover
the shadow-receiving PBR variant, mismatched names at identical formats, and
rejection of incompatible attachment types.

All 37 focused tests passed on arm64 macOS and iPad Pro 13-inch (M5)
Simulator with Xcode 27.2 and Metal API Validation enabled. The selection
included custom MRT, ordinary PBR, the surface attribute pass and graph
shadow regressions.

Three.js r187 resolves MRT outputs by the render target's texture names.
Frostfield's named pass currently exposes a fixed set of typed PBR surface
values. Arbitrary graph expressions as named outputs, per-target blending,
MSAA resolves, transmission and heterogeneous layouts remain open. The named
layout belongs to the runtime pass. SceneArchive rejects a prepared named
renderer instead of dropping its layout.

[Three.js MRTNode](https://github.com/mrdoob/three.js/blob/r187/src/nodes/core/MRTNode.js) · [Three.js PassNode](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/PassNode.js) · [Metal render pass setup](https://developer.apple.com/documentation/metal/customizing-render-pass-setup)
