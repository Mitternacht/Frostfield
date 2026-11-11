# Path-traced pose motion

Temporal path frames reproject native skin and morph poses after
`PathTracingScene.updateGeometry(for:in:replacements:)`. The motion vector
uses the current first hit's triangle corners and barycentric weights to
interpolate its position in the last rendered pose. Both software and Metal
ray tracing address the same renderer vertex IDs even when the software BVH
reorders triangles during a rebuild.

The motion history retains the actual trace's immutable instance records
and the vertex buffers that can change between corresponding poses.
Previous index buffers, BVH nodes, and acceleration structures can be
released. Several pose updates between renders still reproject to the
vertices from the last completed motion frame.

Native CPU skinning, compute skinning, and morph targets preserve
correspondence when the renderer's source vertex count and indices match.
Typed shader graph displacement also preserves correspondence across
rebuilds of the same source topology. A uniform that only feeds vertex
position can change without discarding temporal history. Editing a uniform
that reaches the traced surface still restarts history, including when it
also feeds displacement. Explicit replacements, changed instance layouts,
and material repacks restart temporal history. Pass nil
`previousFrame` after a camera cut. Ordinary progressive accumulation
continues to reject a history after any geometry rebuild.

The motion target remains previous minus current in top-left input pixels.
Pose history uses the same pinhole and planar camera contract as rigid
instance motion. This covers first-hit surface motion and the native
denoiser's history. Secondary reflection and transmission motion remain open.

## Verification

Xcode 27.2 with Metal API validation passed 27 focused tests on macOS
and 19 on iPad Pro 13-inch (M5) Simulator. The pose fixture computes the
expected vectors for a nonuniform surface displacement, two pose updates
between traces, and simultaneous camera and instance translation. It
exercises CPU skinning, compute skinning, and morph targets on both Mac
acceleration backends, and checks that the software rebuild really changes
triangle order. Progressive accumulation and unproven replacement
correspondence both reject stale history.

The native macOS MetalFX fixture retains history through a morph, reveals
the background at the previous silhouette, and matches a fresh denoiser
after an explicit replacement. The simulator verifies software-traced pose
motion and the unavailable MetalFX query.

The typed displacement fixture passes on Metal and software ray tracing on
macOS and on software ray tracing on iPad Simulator. It checks first-hit
motion across consecutive vertex-only uniform updates and verifies that a
subsequent emission edit invalidates history. The extended run passed 48
focused tests on macOS and 44 on iPad Simulator with Xcode 27.2 and Metal
API validation. The Xcode bridge timed out resolving the workspace run
destination, so verification used headless `xcodebuild`.
