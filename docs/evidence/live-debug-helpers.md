# Live box and plane helpers

`DebugHelpers.addBox(of:to:parent:color:)` attaches a line node that can
measure any scene subtree. It recomputes a world-aligned box from current
mesh, instance, batch and primitive bounds. It respects authored culling
bounds and retains the existing conservative mesh bounds for deformation.
For a parented helper, the world edges are converted to the node's local
space, keeping them world-aligned under rotation, reflection and nonuniform
scale. The helper needs an invertible world transform. A missing or empty
source keeps the helper attached and clears its draw geometry.

Nodes created by the debug-helper factory set
`SceneNode.excludesFromDebugBounds`. A box skips the entire marked subtree,
including its own geometry if it is attached beneath the measured object.
The typed marker uses archived scene metadata. Regular hidden source
geometry still contributes to the measured bounds.

`DebugHelpers.addPlane(_:to:parent:size:color:)` attaches a reusable unit
outline and double-sided translucent fill. The `Plane3D` equation and side
length use the parent's coordinates, or world coordinates for a root
helper. Changing those values updates the node's transform. The existing
line material and 20% fill opacity are preserved.

Call `attachment.update(in:)` to save the refresh in the graph. Pass
`updatingBoxes` or `updatingPlanes` to either color encode overload for a
draw-only update. Draw-only updates require a command buffer with retained
references and leave the source graph and archive bytes intact.

`LineGeometry.axes(size:colors:)` supplies one linear RGBA color per axis.
The original overload retains the default gradients.

The reference behaviors are [Three.js r187 BoxHelper](https://github.com/mrdoob/three.js/blob/r187/src/helpers/BoxHelper.js),
[PlaneHelper](https://github.com/mrdoob/three.js/blob/r187/src/helpers/PlaneHelper.js)
and [AxesHelper](https://github.com/mrdoob/three.js/blob/r187/src/helpers/AxesHelper.js).

The focused native suite passed 63 tests on macOS arm64 and 63 on an
iPhone 18 Pro simulator with Metal validation enabled. The GPU fixtures
read back the old and refreshed box frames, the transformed plane outline
and fill, and the original graph after draw-only updates. Additional checks
cover batched geometry retargeting, hidden geometry, archive metadata,
retained command buffers and invalid plane sizes.
