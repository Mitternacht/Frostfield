# Tilt shift

`TiltShift` is a native `PostProcessEffect`. Pass it to
`PostProcessChain.Effect.pass` or encode it directly. The focus band uses
`rotation` in counterclockwise radians and `offset` along its normal. At zero
rotation a positive offset moves the band down. Distances use half the frame
height as one unit. The aspect ratio is applied before rotating, so diagonals
keep the same angle on rectangular images.

The sharp half-width is `max(focusArea - feather, 0)`. The renderer copies
sharp pixels directly from the source, then increases each Kawase tap's spacing
with a smoothstep up to full blur at `focusArea`. A zero feather gives a hard
transition. `KawaseBlur.Settings` controls the blur offsets and resolution scale.
Filtering operates on premultiplied linear RGBA, so blurred coverage carries
its HDR color through transparent edges.

The behavior follows the band and variable sample spacing in the
[pmndrs postprocessing v6.39.6 tilt-shift material](https://github.com/pmndrs/postprocessing/blob/v6.39.6/src/materials/glsl/convolution.tilt-shift.frag)
and its [sharp-region composite](https://github.com/pmndrs/postprocessing/blob/v6.39.6/src/effects/glsl/tilt-shift.frag).
Frostfield uses its native Kawase pipeline, including area downsampling before
the reduced-resolution blur. The screen offset follows Metal's top-left
coordinates. The pinned [Three.js r187 horizontal tilt-shift shader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/HorizontalTiltShiftShader.js)
provides the simpler fixed-horizontal focus line with Gaussian taps.

The GPU tests compare the variable four-tap blur to an independent CPU
reference. They also check an offset, rotated band on an odd-size reduced
frame, exact sharp pixels, premultiplied HDR edges, a hard band, and full-frame
equivalence to ordinary Kawase blur when the focus lies outside the image.

All 18 focused tilt-shift, native blur, lens, bokeh and effect contract tests
passed on macOS and the iPhone 18 Pro Simulator with Metal API validation.
The CPU fixture checks a fixed 0.008 bound against ideal bilinear filtering
and a model that rounds subtexel coordinates to 1/256 texel. Xcode MCP reported
a full external DerivedData volume, so the tests used Xcode command-line tools
with isolated internal DerivedData.
