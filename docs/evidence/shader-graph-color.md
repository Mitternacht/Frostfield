# Typed shader graph blend and tone operators

`blendScreen` and `blendOverlay` operate channel by channel on RGB in
the caller's working space. Overlay changes from multiply to screen at
a base of 0.5. Both preserve signed and HDR values through the formula.
[Three.js r187 blend modes](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/BlendModes.js)

`acesFilmicToneMapping` takes linear-sRGB color and a linear exposure
multiplier. The default multiplier is one. It applies the same
Stephen Hill matrices and fit as the native HDR display curve, then
clamps linear output to zero through one. [Three.js r187 tone mapping](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/ToneMappingFunctions.js)

`posterize` floors the product of a value and its step count, then
divides by that count. It accepts scalar and vector values, with a
shared or per-channel count. A zero or nonfinite divisor yields zero
through the existing graph divide contract.

`cdl` accepts straight RGBA in the caller's working space, typically
log-encoded color. It takes luma from the original RGB, applies slope
and offset, floors the intermediate RGB at zero, and raises the
positive channels to the specified powers. Saturation uses the
original luma and defaults to Rec. 709 coefficients. The final RGB
has a zero floor, and alpha passes through unchanged. [Three.js r187 color adjustment](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/ColorAdjustment.js)

The helpers use existing typed graph expressions. They work wherever
their input values are available and serialize through the current
graph archive format.

## Verification

The pinned r187 TSL bodies were evaluated with a CPU numeric adapter
using Three.js's own Matrix3 and ColorManagement. Metal Float32
buffer readback agrees for signed, SDR and HDR blends, ACES exposure
and clipping, scalar and per-channel posterization, and CDL with both
default and custom luma coefficients.

A raster surface composes all five operators, updates the exposure
without rebuilding its graph, and preserves the intended alpha.
A post-process graph matches the native display curve at the same
exposures. All 63 focused tests across nine suites passed on the
M4 Max Mac and iPad Pro M5 Simulator with Metal API Validation.
