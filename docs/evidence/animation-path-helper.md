# Animation path helper

`DebugHelpers.animationPath(for:in:clip:options:)` prepares a line strip and
optional fixed-size key markers for the selected node's translation binding.
Use a typed `AnimationClip`, a bound glTF clip, or `AnimationClip(animation)`
for a `SceneAnimation`. The line takes evenly spaced samples across the
clip's duration using the source interpolation. Markers show the authored
keys. The defaults use 100 intervals, a green line, and red 5-pixel markers.
The helper uses the debug renderer's line width, joins, caps, and depth test.
[Three.js r187 AnimationPathHelper](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/helpers/AnimationPathHelper.js)

Add the node returned by `animationPath` at the scene root for a world-space
snapshot. Points are baked through the target parent's current world matrix,
including nonuniform scale and shear. Rebuild that snapshot to show a changed
parent pose.

Use `DebugHelpers.addAnimationPath(for:to:clip:options:)` to attach the helper
beside the target. Its line and markers stay in the parent's local space, so
evaluating an animated parent moves the existing renderers through the normal
scene graph. The helper inherits ancestor visibility and can display the path
of a hidden target. If the target changes parents, remove the helper and add
it under the new parent, or build a replacement. A failed addition leaves
the graph unchanged.

Both forms read only the selected translation binding, leaving source nodes
and shared material or shader resources alone. A rig clip can display one
joint's translation track at a time. The live helper follows the parent's
current pose while the curve still represents the selected joint's authored
translation. A trajectory that evaluates ancestor motion per sample, blends multiple
actions, or traces mesh skinning or pivot-induced movement requires a different
sampling workflow. A step
change appears as the connecting line segment between consecutive samples.

`AnimationPathHelperTests` verifies line and marker pixels, visibility,
authored key placement, clip-specific interpolation, live resource isolation,
and immutable snapshots. The linear fixture compares world positions to
the pinned Three.js r187 helper under the same sheared parent hierarchy.
The attached fixture renders an animated sheared parent at two times and
checks that both renderers retain their identity as the pixels move.

The eight focused suites passed 74 tests on the Apple Silicon Mac and 74 on
the iPad Pro M5 simulator with Metal API Validation enabled.
