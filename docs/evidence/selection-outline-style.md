# Selection outline styles

`SelectionOutline.Settings` supports pulse, glow and a screen-space pattern on
the existing visible and hidden selection. Pulse scales edge RGB between 0.25
and 1 using caller-provided `time` in seconds. Its cycle is
`2π × pulsePeriod / 10` seconds, as in the
[pinned Three.js r187 OutlinePass](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/OutlinePass.js).
The pulse scales the edge colors without changing their opacity or the hidden fill.

Set `edgeGlow` above zero to blur the same crisp visible and hidden edge field
with the native separable Gaussian kernel. `glowSigma` controls the spread in
output pixels. The glow adds light outside the selection. The sharp outline
still uses Frostfield's Euclidean pixel coverage, and the optional glow uses
three full-resolution RGBA16Float targets.

Set `patternTexture` to a tracked, float-sampleable 2D texture to draw a white
pattern from its inverse red channel. It repeats in screen space
`patternRepeat` times in each axis, six by default, and applies only inside the
selected silhouette. Occluded selected pixels receive half the pattern opacity.
The pattern composites over any hidden fill and preserves premultiplied output.

The focused GPU suite checks pulse extrema and wrap, Gaussian halo pixels
against a CPU convolution, repeating stripes across visible and hidden pixels
in both depth directions, and the existing outline coverage and mask behavior.

The outline and effect contract suites passed all nine tests on macOS and iPhone
18 Pro Simulator with Metal API validation. Xcode MCP reported that its external
DerivedData volume was full, so verification used Xcode command-line tests and
isolated internal DerivedData.
