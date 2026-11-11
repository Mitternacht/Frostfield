# GPU-timed progressive path preview

`PathProgressivePreview` owns one progressive view on a path tracer's queue.
The application supplies the canvas extent and a command buffer, calls
`encode`, and can sample the returned RGBA16Float texture later in that same
buffer. `latestFrame` returns the last successful image while an updated
view or render size starts accumulating. The preview enforces a configurable
maximum sample count, including a partial final batch.

The controller measures the path compute pass with `PathTracer.Timing`.
After a window of completed full batches, it estimates a new pixel budget
from the target divided by the mean GPU duration. The scale follows the
square root of that ratio. A 15% deadband avoids small oscillations, and
one window can at most double or halve the nominal pixel area. Timing is
optional on devices without pass counters, which continue at the selected
scale. An equal lower and upper scale also gives a fixed preview. Canvas
sizes and output scales are checked against the native accumulation limits.

The preview rounds the traced grid down per axis while preserving the
canvas aspect in camera rays and planar shader-graph projection matrices.
Changing the traced dimensions creates a fresh accumulation. Camera,
pose, tracer settings and lighting changes also start fresh. Call `reset()`
after mutating scene geometry, live uniforms or a texture in place.
An invalidated or discarded command cannot replace the last completed
image. The preview alternates its two display textures within each
accumulation. Its old textures survive a size change while the caller or
latest-frame slot still retains them.

The pinned WebGPU tracer has an authored full-render scale, a timed
wavefront work budget, and a low-resolution preview whose size follows
that budget. Frostfield exposes a timed native preview while callers choose
presentation filtering and any transition to a fixed-resolution render.
The existing fixed path tracer, region and still export, and fixed-size
MetalFX temporal denoiser are separate. Stereo panoramas use the fixed-size
path since their stacked eye regions require an even traced height.
[Reference WebGPU tracer](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/WebGPUPathTracer.js)

## Verification

Fourteen focused tests passed on Apple Silicon macOS and the iPad Pro M5
simulator. On Mac, two real compute pass measurements triggered a change
from 97 × 65 to 68 × 45, and the next image restarted at one sample.
The iPad simulator exercised fixed-scale fallback without pass timestamps.
The GPU fixtures also verified retired texture contents, sample caps,
invalidated and discarded commands, camera resets, and the shader graph's
projection at a rounded 2 × 1 trace size on a 5 × 3 canvas. The existing
fixed-size frame-budget and region/EXR suites passed alongside them.
The Xcode MCP build encountered a full external DerivedData volume, so
verification used `xcodebuild` with internal caches and the Metal debug layer.
