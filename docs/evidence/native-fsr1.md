# Spatial FSR 1 and RCAS

Use `FSR1Upscaler(queue:sharpening:)` to upscale a tracked, linear
RGBA16Float frame into a larger render target. It encodes on the caller's
retained command buffer and reuses its sharpening intermediate. Set
`sharpening` to nil to use only EASU. Use `RCASSharpen` directly or with
`.pass(sharpen)` in a `PostProcessChain` for sharpening at the input size.

EASU uses the reference 12-tap cross, anisotropic kernel and nearest-four
antiringing bounds. Frame boundaries clamp to the nearest source texel.
Like the Three.js port, EASU filters its input channels directly.
Frostfield uses scene-linear premultiplied HDR RGBA and keeps zero-coverage
pixels black. Equal-size RGBA16Float input and output copy exactly when
sharpening is disabled.
[Three.js r187 FSR1Node](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/FSR1Node.js)
[AMD FidelityFX FSR 1 reference](https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/ffx-fsr/ffx_fsr1.h)

RCAS uses the reference five-tap lobe and optional noise attenuation.
The `sharpness` setting counts attenuation stops. Zero is strongest,
and two gives one quarter of the sharpening weight. It sharpens
nonnegative straight linear color and restores the original pixel alpha.
A fully transparent neighbor contributes the center color for sharpening.
For a cross with HDR values above one, AMD's reversible color compression
supplies the lobe and noise estimates while reconstruction uses the original
linear radiance. This retains HDR highlights and avoids a fixed unit upper
bound. SDR values use the reference unit-range limit.
[Three.js r187 SharpenNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/SharpenNode.js)

The GPU tests compare EASU against a CPU reference across an HDR diagonal
edge and RCAS against independent SDR and HDR calculations. They also
exercise coverage, exact constant HDR, stage composition, resizing on one
command buffer, queue and texture validation, and the RCAS controls.

For a path-traced preview, write `PathTracer.encode` into a tracked
RGBA16Float texture with shader read and write usage, then pass it to
`FSR1Upscaler.encode` on the same command buffer. This spatial pass works
without motion guides or MetalFX. It can run with or without RCAS, before
display mapping. This matches the path tracer's spatial FSR route.
[Three.js path-tracer spatial upscaler](https://github.com/gkjohnson/three-gpu-pathtracer/blob/6096de4ffcd420b85de53fd64852ffc9c9c11b0d/src/webgpu/upscale/FSRUpscaler.js)

The path fixture traces a transparent HDR frame and upscales it from 17×13
to 31×25. It compares every EASU texel to the CPU reference and checks that
RCAS preserves alpha. At the center column a fast reciprocal had selected
the previous source footprint. The projection now divides precisely before
selecting taps. The path and spatial suites passed eight tests each on Mac
and iPad Simulator with Metal API Validation.

All 26 focused FSR 1, sharpening and post-process tests passed on macOS
and iPhone 18 Pro Simulator with Metal API validation. Xcode MCP reported
a full external DerivedData volume, so verification reused the internal
Xcode command-line build caches.
