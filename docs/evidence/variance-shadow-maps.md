# Variance shadow maps

`ShadowMapSampling.variance` is available for single and cascaded
`DirectionalShadowMap` instances. It renders the existing caster depth and
uses two Metal compute passes to box-filter the first and second moments
into reusable RG32Float arrays. `varianceBlurRadius` is measured in texels.
Directional PBR, classic materials and Lambert shading read those moments
and estimate visibility with a Chebyshev bound and bleed reduction.

The filter accepts resolutions from 1 to 2048 and radii from 0 to 16.
Variance uses a zero angular radius. The depth map remains available to the
texture viewer. The two moment arrays add 16 bytes per texel per cascade.
On-demand maps refresh depth and moments together and reuse both between
requests. Existing basic, PCF and contact-hardening modes retain their
sampling behavior.

Three.js r187 performs a separable moment filter and remaps the Chebyshev
bound to reduce light bleeding. Frostfield stores raw moments in Float32,
interpolates them explicitly and keeps the native caster selection policy.
The implementation covers directional lights, cascades and spots. Three.js
r187 excludes point lights from VSM.

Spot lights select `PunctualLight.Shadow(sampling: .variance, varianceBlurRadius: …)`.
Their moments use the punctual map's reversed perspective depth. The receiver
compares in that direction, with the configured receiver bias. A fixed
variance floor is unnecessary for reversed perspective depth and can brighten
a constant-depth blocker at long distances.

Punctual maps lazily allocate one pair of moment arrays for the spot slots.
The moment budget is 256 MiB with a maximum resolution of 2048. Basic and
filtered spots, points and rectangles can share the same depth array. Only
the updated variance slices are filtered. The on-demand lease publishes depth
and moments together, and a mode or blur-radius edit refreshes the affected
spot even when its projection is unchanged. Point and rectangle settings
reject variance sampling.

The fixtures compare actual GPU moments with a clamped CPU box filter, count
soft transition pixels against basic shadows, and check clear and fully
shadowed receiver regions. Cascaded fixtures read both moment slices and
compare stale and refreshed receiver pixels.
Spot fixtures compare both moments with a CPU filter over actual reversed
depth and read the PBR receiver and shadow catcher. A mixed on-demand fixture
checks mode and radius changes, cache reuse and unaffected light depth.

All 49 focused tests passed on arm64 macOS and iPad Pro 13-inch (M5)
Simulator using Xcode 27.2 and Metal API Validation. Both filter passes use
uniform threadgroups with bounds-checked edges.

The spot follow-up passed 51 focused tests on both platforms with Metal API
Validation. The mixed fixture kept the Basic spot, filtered point and Basic
rectangle depth bytes unchanged across spot mode, blur-radius and manual
refresh changes.

The combined build also switched the spot to automatic and rendered two
caster states in one command buffer. Each use recomputed that spot's depth
and moments. The manual spot, point and rectangle slices stayed byte-identical.

[Three.js moment filtering](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderLib/vsm.glsl.js) · [Three.js variance visibility](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/shadowmap_pars_fragment.glsl.js) · [Three.js shadow map setup](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLShadowMap.js)
