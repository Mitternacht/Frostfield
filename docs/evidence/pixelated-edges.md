# Pixelated depth and normal edges

Use `PixelatedEdgesPass(device:settings:surfaceAttributes:cameraRotation:)`
directly or add it with `.pass(pixelation)` in a `PostProcessChain`. The
pass samples nearest block centers from a resolved linear RGBA16Float color
frame. It clamps the grid at frame edges, including partial blocks. Set both
edge strengths to zero to use color-only pixelation.

For edge shading, pass matching stored Depth32Float in forward or reversed
mode. For normal creases, bind the RG32Uint texture and stored depth produced
by `SceneGraph.encodeSurfaceAttributes` for the same view, and set the view's
world rotation. This rotates the reference view-space normal priority into the
space of the packed normals. The surface prepass supplies opaque and masked
mesh normals. Pixels with cleared depth supply no normal.

The four-neighbor edge rules darken the nearer block
with a half-strength or full-strength step. A normal crease brightens its
preferred side when depth does not take precedence. The gain affects
premultiplied linear HDR RGB and retains the sampled color alpha. This pass
uses resolved textures at their existing size. A scene rendered directly
into reduced-resolution attachments is a separate capture workflow.
[Three.js r187 PixelationPassNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/PixelationPassNode.js)

The GPU tests compare every block in an odd-sized fixture against CPU depth
and normal references in both projection-depth modes. They exercise the
half-strength threshold, partial frame blocks, clear neighbors, a crease
whose highlight changes side as the camera rotates, coverage, signed HDR,
color-only equivalence, and resource validation.

All 36 focused pixelation, graph preset, post-process and packed surface
attribute tests passed on macOS and iPhone 18 Pro Simulator with Metal API
validation. Xcode MCP reported a full external DerivedData volume, so
verification used Xcode command-line tests with isolated internal DerivedData.
