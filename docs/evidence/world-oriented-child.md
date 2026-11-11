# World-oriented scene children

Set `SceneNode.orientationMode` to `.world` to use the node's local orientation
in world space while its position and scale follow its parent. The resolved
matrix keeps the origin and signed column lengths from the composed
parent-local matrix. It replaces the orientation with the local rotation
in the same canonical mirrored form as Three.js. The world-oriented matrix
passes to the node's children. Root nodes use their local transform.
[Three.js r187 Gyroscope](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/misc/Gyroscope.js)

Coordinate conversion, scene bounds, skeleton helpers, raster, picking, and
path tracing use the same resolved pose. World-preserving attachment solves
the local scale separately for each oriented axis. Look-at and the local
gizmo use the child's world orientation. A pivot can move the node's origin
when its authored rotation changes.

glTF and USDZ scene exports report an error for world-oriented bindings.
CCD IK reports an error for chains that cross a world-oriented child.
`WorldOrientedNodeTests` compares signed and nonuniform scale with the pinned
r187 helper, exercises hierarchy edits, and verifies raster pixels, picking,
and a path-tracing transform refit.

Scene archive version 10 preserves world orientation alongside local TRS and
pivots. Archives through version 9 load with inherited orientation, and
captures with inherited nodes still use the earliest version required by
their contents. The archive fixture verifies the restored pose, identical
recapture, and rejection when a version 9 manifest contains orientation data.

The core suite passed 82 tests per platform. With archive coverage included,
the 14 focused suites passed 87 tests on the Apple Silicon Mac and 87 on the
iPad Pro M5 simulator with Metal API Validation enabled.
