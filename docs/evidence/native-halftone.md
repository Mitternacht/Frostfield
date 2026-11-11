# Native halftone

Create `HalftonePass(device:settings:)` and encode it directly or add it
with `.pass(halftone)` in a `PostProcessChain`. The pass supports dot,
ellipse, line, square and diamond, plus linear, multiply, add, lighter
and darker blends. Edit the radius, RGB rotations, scatter, blend fraction
or grayscale between encodes. Rotation follows the bottom-left,
counterclockwise convention of the reference pass. Radius sets the grid
spacing in pixels at the stage's resolution. Disable the pass for an exact
copy.

The screen follows the five shape formulas, four neighboring centers,
and nine bilinear taps per center in the
[Three.js r187 HalftoneShader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/HalftoneShader.js).
Scatter moves those centers using the reference sine hash. The pass reads
the nine taps as premultiplied values and divides their combined RGB by
their combined alpha to estimate dot density. The screen and blend operate
on the SDR portion of straight color, then restore signed HDR excess and
premultiply by the original pixel alpha. Grayscale averages the resulting
channels before premultiplication.

The tests compare all five rotated shapes against a CPU reference across a
mixed-coverage frame. A separate HDR fixture exercises all blend modes,
scattered grid positions, bilinear neighborhood sampling, grayscale, live
settings, exact bypass, and validation.

All 22 focused halftone, graph preset and post-process tests passed on macOS
and iPhone 18 Pro Simulator with Metal API validation. Xcode MCP reported a
full external DerivedData volume, so verification used Xcode command-line
tests with isolated internal DerivedData.
