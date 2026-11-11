# Scene transform gizmos

`TransformGizmo` edits a node by ID using world-space rays. Its perspective
frame places the handles at the node's origin and sizes them in drawable
pixels. `TransformGizmoOverlay` supplies colored arrows, rings, boxes and
plane patches, then raycasts the same geometry. The host can render the handles with the
scene or in a separate pass that loads the scene color.

Use a frame from the current graph on pointer-down. For an ordinary
perspective view, convert pointer coordinates to a ray with
`PerspectiveCamera.ray(at:aspectRatio:cameraTransform:)`.

```swift
let tool = try TransformGizmo(mode: .translate, snap: 0.25)
let frame = try tool.frame(for: selectedID, in: graph, drawableHeight: height)
if let frame, let handle = try overlay.hitTest(ray, in: frame) {
    drag = frame.beginDrag(handle, ray: ray)
}
```

On movement, pass the new ray to `drag.update(ray:in:)` and rebuild
the frame for drawing. Use the drag's handle to highlight the active
mesh. Keep the camera fixed during the drag so pointer rays stay in
the original view. Finish to keep the edit or cancel to restore the starting pose.
For direct object dragging, use `SceneGraph.pick` to choose the object
and start its translation frame with the screen handle.

Translation works along an axis, on the XY, YZ, and XZ planes, or in
the camera plane. Rotation accumulates signed angles around a local axis,
world axis, or the line from the pivot to the eye at pointer-down.
Scale handles change one local component, two components on a plane,
or all components together through the center handle. The drag captures
its constraint and checks for stale target or parent edits before applying
or restoring a pose.

Relative snaps measure from the starting pose. Opt in to absolute
translation snapping against the world origin projected onto the active
axes or plane, and to absolute scale snapping against local component
magnitudes. Uniform absolute scaling uses the largest component as the
reference to preserve proportions. Rotation snaps measure relative angles.
An optional parent-space translation box clips the requested displacement
segment at the first boundary, preserving the selected constraint and
movement direction. Start the drag with the node inside that box.

World translation accepts nonuniform, reflected, and sheared parents
when they are invertible. Local controls and rotation require a uniformly
scaled, orientation-preserving parent so the handles agree with the node's
translation, rotation, and scale. The gizmo uses a perspective camera.

Three.js r187 also supplies trackball rotation and browser input events.
Hosts using Frostfield supply native gesture capture, selection policy,
and hover or change notifications.
[TransformControls r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/controls/TransformControls.js) ·
[DragControls r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/controls/DragControls.js)

## Verification

Xcode 27.2 passed 14 tests on both macOS and iPhone 18 Pro Simulator
with Metal API validation enabled. The numeric fixtures cover transformed
parents, plane motion and scale, repeated and view-axis rotation,
reflection-preserving scale, relative and absolute snapping, translation
bounds, cancellation, and stale edits. The rendered fixture casts camera
rays at the pixels where Metal draws an arrow, plane patch, axis ring,
and view ring. It checks active highlighting and scene-node updates.
