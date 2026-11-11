# Basic raster shadow sampling

`ShadowMapSampling.basic` opts into one nearest depth comparison per selected
shadow view. It is available on single and cascaded `DirectionalShadowMap`
initializers and on `PunctualLight.Shadow` for spot, point and rectangle lights.
Forward directional maps compare with less-or-equal. Reversed punctual maps
compare with greater-or-equal and use the selected point face or spot view.

Basic sampling keeps the configured receiver depth bias and punctual normal
bias. The angular radius, punctual filter radius and their extra footprint
biases apply to filtered maps. The default remains filtered and retains the
existing PCF and contact hardening. Cascades still blend across split bands
when a blend fraction is configured. Punctual light data written before this
option decodes as filtered.

Three.js r187 BasicShadowMap reads one depth texel and makes a hard comparison
for directional, spot and point lights. Frostfield uses its existing native
map projections and bias units. [Three.js shadow comparison](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/shadowmap_pars_fragment.glsl.js) · [Three.js shadow map setup](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLShadowMap.js) · [Metal depth comparison](https://developer.apple.com/documentation/metal/mtlsamplerdescriptor/comparefunction)

The fixtures compare lit receiver rows with and without a blocker. They
check binary visibility under basic sampling, intermediate pixels under
filtering, and the effect of explicit depth bias for directional, spot and
point lights. Graph opacity and masks are checked in both modes by reading
the shadow map depth and lit receiver pixels.

All 50 focused tests passed on arm64 macOS and iPad Pro 13-inch (M5)
Simulator using Xcode 27.2 and Metal API Validation. The selection also
covered filtered cascades and contact hardening, classic materials, named
surface targets, and scene archive regressions.
