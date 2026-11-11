# Raster motion vectors

`RasterMotionVectors` records the screen displacement of indexed depth-writing
meshes, including skinned, morphed and shader-displaced poses, stable instance
slots, curve flow and topology-stable dynamic meshes. Pass a new `Recording` to
`SceneGraph.encode(... motionVectors:)`, then keep its `history` for the next
accepted frame of that view. Camera transforms, off-axis projections, jitter
and camera-relative transforms come from the same preparation as raster color.

```swift
let recording = rasterMotion.recording(to: velocity, previous: viewHistory)
try scene.encode(pass: scenePass, commandBuffer: frame, motionVectors: recording)
// Advance viewHistory when the frame is accepted.
viewHistory = recording.history
```

The target is a tracked, shader-readable and renderable single-sample
`rgba16Float` texture matching the stored `depth32Float` scene attachment.
RG gives previous minus current in top-left attachment pixels, B is previous
window depth and A is one for valid history. A clear pixel is zero in all
channels. Use the same queue and a retaining command buffer. A separate
depth-equal draw uses the scene's stored depth, including mask holes and
occlusion by closer surfaces.

New meshes, replacement instance layouts and prior positions outside the
clip volume have invalid history. The history keeps the actual packed morph
weights and joint palettes of the previous accepted scene draw. Both poses
reuse the raster mesh's GPU position stages, applying morph
offsets before skinning and skinning before the instance transform. Draws
use immutable topology and forward or reversed projection depth.

For surface graph offsets, curve flow and dynamic meshes, the frame takes a
private Metal copy of the active packed vertices after the color pass. Draws
sharing that mutable vertex stream share one copy. The per-view budget is
256 MiB and dynamic capacity beyond the active vertex count does not use it.
Submit recorded frames in order on the recording queue and discard history
from abandoned or failed frames.

`CurveFlow` retains its vertex identities as placements move or change curves.
Call `MeshRenderer.updateGeometry(_:preservingVertexIdentity:)` with `true`
only when each vertex slot still represents the same vertex. The vertex
count and index sequence must match. A regular replacement starts new
motion history for that mesh, even when its index sequence happens to match.

Batches, indirect draws, depth bias, non-mesh primitives and multisampled
or logarithmic depth remain outside this pass. Depth equality alone cannot
distinguish coplanar draw winners. Use one history per view and discard it
when cutting the camera or changing attachments.

Focused tests under Metal API Validation passed on M4 Max macOS and iPad Pro
(M5) Simulator with Xcode 27.2. Readback checks object and camera motion,
projection jitter, both depth conventions, large-world transforms, foreground
occlusion, cutouts, invalid history and changed instance slots. The Xcode
bridge timed out resolving the initial destination, so the tests ran through
headless `xcodebuild`.

The pose checkpoint passed 45 focused tests on each destination. Pose
readback checks inline and buffered joint palettes, a morph returning to
its unweighted pose, and simultaneous changes to morph weights, skin depth
and instance placement. The perspective fixture checks different points on
the same mesh, including previous projected depth in forward and reversed Z.

The shader-displacement checkpoint passed 47 focused tests on Mac and iPad
Simulator. The GPU fixture queues an unrecorded pose between two accepted
poses. It verifies that two draws sharing one renderer reproject to the first
accepted buffer in both depth conventions, and that a different queue rejects
GPU-backed history.

The live-geometry checkpoint passed 44 focused tests each on Mac and iPad
Simulator with Metal API Validation. Readbacks cover CPU and Metal curve
flow and dynamic position edits across an intervening unrecorded render, in
forward and reversed depth. A topology mismatch rejects the preserving edit,
and a default replacement clears motion until a new matching frame exists.

The color and motion shaders share one explicit fused clip projection.
A translated, jittered PBR floor with varying perspective depth exposed
missing motion pixels when the shaders used independently optimized
matrix multiplies. The shared projection adds the homogeneous translation
after the fused terms. The regression verifies depth-matched output at two
visible patches in both depth conventions. The focused Mac run passed 18
tests with Metal API Validation, including both camera-relative and
world-space cascaded shadow stability, cutout, morph and skin fixtures.
[Apple Metal position consistency](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)

[Three.js r187 velocity node](https://github.com/mrdoob/three.js/blob/r187/src/nodes/accessors/VelocityNode.js)
