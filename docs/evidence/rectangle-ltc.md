# Rectangle light specular

Raster PBR rectangle lights use a linearly transformed cosine (LTC)
fit for isotropic GGX and clearcoat. The shader takes the oriented
emitter corners and the receiver's normal and view direction. It reads
the fitted transform and Fresnel coefficients at perceptual roughness
and `sqrt(1 - N·V)`, then integrates the spherical polygon with the
horizon approximation from r187. Clearcoat uses its own normal and
roughness. The lighting pass applies the same one-sided emission and
optional center-cube shadow. Rebasing applies to the emitter and view
positions in high-precision scenes.
[Three.js r187 LTC evaluator](https://github.com/mrdoob/three.js/blob/r187/src/nodes/functions/BSDF/LTC.js) ·
[Three.js r187 lighting model](https://github.com/mrdoob/three.js/blob/r187/src/nodes/functions/PhysicalLightingModel.js#L714)

Diffuse uses Frostfield's analytic horizon-clipped irradiance.
Anisotropic base specular and the small-emitter limit still use the
native representative-point approximation. The sheen and rough-diffuse
lobes retain their own approximations. Path tracing and lightmap baking
sample the actual emitter area.

## Table provenance

The 65,536-byte asset stores `LTC_HALF_1` followed by `LTC_HALF_2`
in the row and RGBA channel order returned by
`RectAreaLightTexturesLib.init()` from `three@0.187.0`. Each element is
the little-endian 16-bit representation returned by Three.js. Its
SHA-256 is `de13346ddf530f99127993fc850985997dcdadbdf5b3ea820072eb78b4c7c428`.
The half tables match the r187 path used when float32 texture filtering
is unavailable. Metal reads their 64 × 64 RGBA16Float slices bilinearly
without using an additional fragment sampler. An immutable texture is
cached on demand and retained through unretained command buffers.
[Three.js r187 table source](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/lights/RectAreaLightTexturesLib.js)

Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt,
“Real-Time Polygonal-Light Shading with Linearly Transformed Cosines,”
ACM Transactions on Graphics 35(4), SIGGRAPH 2016. The package includes
the Three.js MIT notice and the original LTC redistribution terms
beside the table bytes.
[Paper and reference implementation](https://github.com/selfshadow/ltc_code)

## Native checks

Metal API Validation ran 84 focused test declarations in 19 suites on
macOS and the iPhone 18 Pro Simulator. The Simulator skipped one
indirect-command-buffer test by its capability gate. The numeric
reference evaluated the pinned r187 LTC transform and Fresnel fit with
its original half tables. Metal pixel checks cover an asymmetric
tilted panel at two rolls, two isotropic roughness values, a grazing
view, a generated surface graph in the shadow-capable pipeline, and
clearcoat with a separate roughness and grazing Fresnel. Existing tests
also cover analytic diffuse, horizon clipping, the small-emitter limit,
one-sided emission, proxy depth and cache behavior, and shared-origin
precision. The Xcode MCP timed out resolving the package destination,
so verification used `xcodebuild`.
