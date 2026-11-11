# Frei–Chen edge detection

Add `PostProcessGraph.freiChen()` to a `PostProcessPass` or post-process chain.
It reads a 3 × 3 neighborhood with one-texel offsets based on the actual input
dimensions and uses the length of sampled RGB as intensity. The output is opaque
gray from zero to one. A neighborhood with zero energy returns opaque black.

The preset uses the four edge masks in the pinned [Three.js r187 FreiChenShader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/FreiChenShader.js).
Its denominator uses the equivalent squared norm of the nine input intensities.
The full mask basis is orthonormal, so that norm equals the sum of its nine
squared responses, within rounding of the pinned coefficients.

The GPU tests cover black and bright constants, a one-pixel frame, horizontal,
vertical and diagonal steps, varying alpha and HDR color. They compare the
shader against a separate CPU convolution using all nine pinned matrices,
including clamped border samples. A hue-only transition with equal RGB lengths
returns zero. The existing Sobel test runs in the same suite.

`PostProcessGraphPresetTests` passed all 12 cases on macOS and the iPhone 18 Pro
Simulator with Metal API validation. Xcode MCP reported that its external
DerivedData volume was full. Xcode command-line tests used isolated internal
DerivedData.
