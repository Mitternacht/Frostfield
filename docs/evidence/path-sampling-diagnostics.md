# Path tracer sample density and software BVH bounds

`PathTracer.sampleDensity(_:)` combines disjoint windows using their latest
completed histories. The snapshot stores the actual uniform sample count for each
window, with zero in uncovered pixels. It rejects failed, pending, superseded or
foreign histories, mismatched image dimensions and overlapping windows. A newer
in-flight command leaves the last completed history available until it succeeds.
The snapshot has the same image size limit as an accumulation.

`SampleDensity.makeTexture(device:displayRange:)` produces an R32Float texture
with raw counts or values mapped to a caller-selected range. Use the red channel
in `TextureDebugPass` to display the normalized values. This uses the native
rendered sample budgets. The pinned tracer's density view reads per-pixel
sample counts and uses a heat ramp. In wavefront mode, individual paths
finish at different times. Its adaptive frame budget adjusts work per frame.
[Reference density view](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/WebGPUPathTracer.js#L1770-L1805)

`PathTracingScene.softwareBVHBounds` reads either the software top-level tree or
the bottom-level tree for one mesh occurrence. The snapshot keeps node depths,
tree indices, leaf counts and the scene revision. For a bottom level, its
`lineGeometry` method applies the complete instance affine transform to the
box edges for display as a scene-root `DebugHelpers` node. Depth starts at zero
and the snapshot rejects a request exceeding 4,096 nodes. Metal hardware
acceleration structures are opaque. The per-camera-ray box intersection heatmap
has its own [verification](path-bvh-heatmap.md).
[Reference bounds view](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/WebGPUPathTracer.js#L1661-L1679)

PCG and scrambled Sobol keep their existing sequences. The pinned WebGPU
sampler also offers blue dithering. It uses one shared Sobol sequence with
a fractional offset from a tiled 64 × 64 blue-noise field, indexed by image
coordinates. Matching it requires a reproducible spatial blue-noise field,
an opt-in native sampler and checks for both low-frequency power and tiled
render reproducibility.
[Reference blue dithering](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/nodes/rand/bluedither.wgsl.js)

## Verification

Focused Xcode suites passed 15 tests on Apple Silicon macOS and 14 on the
iPad Pro M5 simulator, where one hardware-only check was skipped. They covered a progressive tile's in-flight and stale
histories, disjoint mixed sample counts, exact texture readback and display
pixels, transformed TLAS and BLAS nodes, instance refits, actual wireframe
pixels, and hardware availability. Region and texture debug regressions
ran with the same suites. The Xcode MCP build encountered a full external
DerivedData volume, so verification used `xcodebuild` with internal caches
and the Metal debug layer.
