# Post-process depth helpers

`ShaderGraphBuilder.getViewPosition` unprojects a raw Metal depth
value and top-left screen UV through a typed inverse projection.
The view is right-handed and looks down negative Z. Use the
projection that produced the depth, including any view offset,
jitter, or reversed mapping. `inverseProjection(of:)` derives the
matrix from `ShaderGraphCamera`. A graph can also supply a matrix
built from its own columns. A point without a finite view position,
including a cleared infinite far plane, returns zero.
[Three.js r187 post-processing utilities](https://github.com/mrdoob/three.js/blob/r187/src/nodes/utils/PostProcessingUtils.js)

`linearDepth` maps distance along the view axis to 0 at the near
plane and 1 at the far plane. It accepts finite clip distances and
uses the supplied inverse projection for both perspective and
orthographic cameras. The output has this same near-to-far
direction with either depth mapping. Three.js r187 returns raw
depth for its orthographic input, which keeps the reversed mapping
in that case.
[Three.js r187 linear depth](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/ViewportDepthNode.js)

`getNormalFromDepth` uses the source `Depth32Float` texture. It
compares the second-neighbor extrapolation error on each side of
each axis and reconstructs view-space right and up differences
from the chosen adjacent texels. Queries use the covering texel's
center. Out-of-bounds taps count as empty. When a second tap is
empty, the first difference scores that side. Cleared center depth,
an axis without a usable neighbor, or degenerate differences
return zero. Pass the source depth mode so the graph identifies
the correct clear value. The existing post-process pass and
chain bind depth without a new camera context.

## Verification

The focused suite passed 49 tests on macOS and iPad Simulator with
Metal API and shader validation. Pixel readback checks position and
normalized depth for cropped perspective and orthographic cameras
in both depth modes. A two-plane fixture checks reconstructed
normals at the discontinuity and image boundaries. A separate
stencil makes the nearest first neighbor disagree with the
second-neighbor choice. Tests also cover cleared infinite depth,
missing depth, and JSON graph round trips.

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData.
