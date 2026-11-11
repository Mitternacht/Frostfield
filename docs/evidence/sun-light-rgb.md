# Sun RGB radiance

`SunLight` and `DirectionalShadowMap` accept a linear RGB tint in
0...1 and a separate nonnegative intensity. Their default tint is white.
The solar-position constructor also accepts a tint. The direction,
intensity and tint can be changed together in a validated update.

The existing raster directional receiver paths multiply the incident light
by the tint. This applies to standard and generated PBR, classic lit
materials and shadowed Lambert. Ambient, emission, environment lighting
and shadow-catcher output keep their existing inputs. The normal and
camera-relative shader bindings carry the same tint.

`update(color:)` changes only the tint. An on-demand map can reuse its
current depth and fitted cascade projections. A failed update leaves the
previous settings intact. Color changes keep the directional shadow
receiver opt-in used by the native raster renderer.

[Three.js r187 SunLight](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/lights/SunLight.js)

The focused suite passed 56 tests in 11 suites on macOS arm64 and on the
iPhone 18 Pro simulator with Metal validation. Pixel readbacks cover the
four receiver paths, neutral and tinted direct light, ambient under a
real caster, and cached cascade reuse. An invalid RGB update preserves
the prior uniform state. Xcode's bridge reported a full workspace volume,
so verification used `xcodebuild` and internal build caches.
