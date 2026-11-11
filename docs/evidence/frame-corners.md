# Off-axis framing from screen corners

`CameraUtils.frameCorners` builds an explicit perspective view from a world-space
eye and three corners of a rectangular screen. Looking from the eye, supply the
bottom-left, bottom-right, and top-left in that order. The bottom and left edges
must be perpendicular, with a small allowance for Float rounding. Their cross
product points toward the eye. The fourth corner is implied.

The camera looks down local negative Z. The screen's corners map to the edges
of the Metal viewport even when the eye is off-center. The screen may lie at a
different distance than the near or far plane. The near distance is finite and
positive, and the far distance may be finite or positive infinity. Forward
depth maps near to zero and far to one. Reversed depth maps near to one and
far to zero. Invalid corners, eye locations, or unrepresentable views throw.
[Three.js r187 CameraUtils](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/utils/CameraUtils.js)

Pass the result's `clipFromWorld`, `cameraPosition`, `depthMode`, and `graphCamera`
to the explicit-view `SceneGraph.encode`. Clear depth to one for forward depth
or zero for reversed depth. Match the viewport to `aspectRatio` to preserve
the screen's proportions. Use `graphCamera` and the same depth mode to pick
from top-left normalized coordinates. Scene culling uses the exact view,
so it needs no estimated field of view.

`CameraFrameCornersTests` compares a tilted screen to the pinned r187 view
and projection after converting OpenGL depth to Metal. It also verifies all
four corners, finite and infinite clipping, invalid inputs, and real raster
culling, depth ordering, and picking for both Metal depth mappings.

The eight focused suites passed 45 tests on the Apple Silicon Mac and 45 on
the iPad Pro M5 simulator with Metal API Validation enabled.
