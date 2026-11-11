# Raster shadow catcher

`ClassicMaterial.shadowCatcher` creates a black, transparent material with a
configurable linear tint and opacity. It uses the existing classic shadow
model. Assigned punctual maps contribute automatically. Enable
`receivesDirectionalShadows` when supplying a directional map.

The fragment multiplies the assigned maps' visibility. Its output alpha is
`opacity × (1 − visibility)`. Premultiplied source-over blending keeps the
background intact in unshadowed regions. Directional catchers sample their
assigned map with either receiver facing and at zero emitted intensity.
Basic, filtered and on-demand maps use their usual raster paths.

The default state tests depth without writing it. To use the inherited Three.js
depth-write policy, supply a `MeshRenderState` with normal blending,
`transparent: true` and `depthWrite: true`. This writes the catcher's depth
where its depth test passes, including fully transparent regions.

The Metal fixtures compare the tinted composition against an opaque background
and read stored depth in shadowed and unshadowed pixels. They exercise
directional, spot and point lights with basic and filtered sampling, including
a directional source behind the catcher with zero direct light.

All 53 focused tests passed on arm64 macOS and iPad Pro 13-inch (M5)
Simulator with Xcode 27.2 and Metal API Validation enabled. The run included
classic materials, shadow coverage, render state and scene archive checks.

Three.js r187 also has per-light shadow intensity and material fog integration.
Those are separate from the raster catcher verified here. Classic materials
remain raster-only.

[Three.js ShadowMaterial](https://github.com/mrdoob/three.js/blob/r187/src/materials/ShadowMaterial.js) · [Three.js shadow mask](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/shadowmask_pars_fragment.glsl.js) · [Three.js inherited depth state](https://github.com/mrdoob/three.js/blob/r187/src/materials/Material.js) · [Metal blending](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Render-Ctx/Render-Ctx.html)
