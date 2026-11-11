# Radial blur

Create `RadialBlur(device:settings:)` and use `.pass(blur)` in a
`PostProcessChain`, or encode the effect directly. Edit the typed settings
between frames without rebuilding the pipeline. The center is specified in
top-left normalized coordinates. Each output pixel uses a ray toward that
center with 1–64 bilinear, edge-clamped taps.

At radius one the effect follows the [Three.js r187 radial blur](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/radialBlur.js)
for ray travel, interleaved gradient jitter, decaying weights, and additive
exposure. Larger radii extend the ray past the center. Radius, weight or exposure
zero copies the source exactly. The pass operates on linear premultiplied
RGBA16Float input. It saturates coverage to one and keeps signed HDR RGB
within the half-float range. The effect is screen-space and does not use scene depth.

The GPU tests compare every pixel of a 21 × 13 mixed-coverage HDR frame against
independent bilinear CPU references for two off-center positions, including
the maximum tap count. The single-tap fixture checks that overshooting rays
reach the expected corner, and all three bypass controls produce identical
input and output pixels. The tests also exercise validation and pipeline reuse.

All 20 focused radial blur, image blur, tilt shift and post-process contract
tests passed on macOS and iPhone 18 Pro Simulator with Metal API validation.
Xcode MCP reported a full external DerivedData volume and timed out opening
the isolated package workspace, so verification used Xcode command-line
tests with isolated internal DerivedData.
