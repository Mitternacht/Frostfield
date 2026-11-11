# Blue-dithered path sampling

Set `PathTracer.Settings(sampler: .blueDither)` to use a shared Owen-scrambled
Sobol sequence with a circular shift per image pixel. The shift is the rank
divided by 4,096 from a periodic 64 × 64 spatial field. Using full-image
coordinates keeps the result consistent across regions, including streamed
stills. The seed still changes the sequence. The tracer's existing
variable-length volume walks retain their independent streams.
[Reference WebGPU sampler](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/nodes/rand/bluedither.wgsl.js)

The bundled rank field is an 8,192-byte row-major permutation of
0 through 4,095, encoded as little-endian unsigned 16-bit values. It was
generated with the pinned void-and-cluster implementation, using its default
64-pixel size, 1.5 Gaussian sigma and 0.1 starting density. To reproduce it,
set the generator's `random` to the following seeded function and write
the returned `data` in array order.

```js
let state = 0x6a09e667;
generator.random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
};
```

The field's SHA-256 is
`94af0a9f8ae046124639425e8d773d642013bbe4e1d4e54596a3edeef2c2d34c`.
The tracer loads the Metal rank buffer on the first blue-dither encode.
[Reference rank generator](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/textures/blueNoise/BlueNoiseGenerator.js)

## Verification

The Metal stripe fixture captures the primary ray's horizontal jitter as
black or white. One 64 × 64 period measured 0.02810 times independent
white-noise expectation in nonzero Fourier modes through four cycles per
period. The independent PCG render measured 1.13702 times that expectation.
The dither period repeated exactly along both image axes. Full-image renders
matched the same sample indices split across batches and a tile crossing both
repeat boundaries. Pre-change PCG and Sobol footprint hashes matched on both
platforms. The area-light fixture also checked seed changes, batch splits and
history validation for all three samplers.

Twenty focused tests passed on Apple Silicon macOS and the iPad Pro M5
simulator, including the region, streamed EXR and generated shader-graph
regressions. The Xcode MCP build encountered a full external DerivedData
volume, so verification used `xcodebuild` with internal caches and the Metal
debug layer.
