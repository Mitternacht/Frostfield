# Software path-tracing BVH heatmap

`PathTracer.encodeBVHHeatmap` writes a linear RGBA16Float debug image using
one center ray per texel and the prepared software path-tracing acceleration
snapshot. It can count top-level or bottom-level boxes independently.
A caller supplies the camera, texture and retained command buffer. An optional
region uses the full-image camera projection. For a thin lens, the ray starts
at the center of the aperture. The existing `TextureDebugPass` can display
the result as a screen overlay.

The default ramp runs from blue through cyan, green and yellow to red, with
a configurable saturation count. Counting stops once the display saturates.
When `stopAtSurface` is set, the kernel
finds the nearest mesh surface accepted by Frostfield's face-culling and
coverage rules, then counts boxes up to that hit. This makes the cutoff
independent of tree order. Boxes touching the hit distance count. The surface
search and the box pass share one per-ray node-test budget. Magenta identifies
pixels that reach the budget with pending tests. Analytic light emitters are
outside the mesh BVH. Metal's hardware acceleration structures do not expose
their nodes, so the diagnostic requires software preparation.
[Reference Three WebGPU tracer](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/nodes/debugBounds.wgsl.js)

## Verification

Fifteen focused tests passed on Apple Silicon macOS. On the iPad Pro M5
simulator, 13 passed and two hardware-only checks were skipped. The GPU
fixtures counted the root, leaves and an interior BLAS split. They also
checked masks, live graph coverage, culling, clip and tile coordinates,
saturation, budget exhaustion, and command-buffer retention across refits.
The adjacent diagnostics and streamed path-output tests ran in the same
suites. The Xcode MCP build encountered a full external DerivedData volume,
so verification used `xcodebuild` with internal caches and the Metal debug
layer.
