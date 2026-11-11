# Raster temporal upscaling

Create a `RasterTemporalUpscaler` for one view and fixed input and output
dimensions of the same aspect ratio. Render at the input size using
`nextJitter` or another projection jitter in input pixels. Pass the resolved
RGBA16Float color, Depth32Float stored depth, and matching raster motion
attachment to `encode`. Consume `outputTexture` afterward on the same queue.
Finish each buffer before encoding the next frame. Reset after a cut or
discard an unsubmitted buffer explicitly to restart temporal history.

MetalFX uses its temporal scaler for frames declared `.opaque` when
supported by the GPU and configuration. Its input guides preserve the
motion attachment's previous-minus-current top-left pixel convention.
The descriptor enables jittered motion vectors, and the encoder supplies
the actual raster projection jitter. A reactive mask favors the current
frame at invalid motion, out-of-frame positions and mismatched previous
depth. The current depth is kept at float32 precision and supports
forward and reversed projection depth.
[Apple MetalFX temporal upscaling](https://developer.apple.com/documentation/metalfx/applying-temporal-antialiasing-and-upscaling-using-metalfx)
[Apple jittered motion vectors](https://developer.apple.com/documentation/metalfx/mtlfxtemporalscalerdescriptor/isjitteredmotionvectorsenabled)

The default premultiplied color mode uses the native motion-guided TAA
resolve followed by a linear upscale. It retains coverage. The M4 Max
MetalFX probe wrote alpha one for an input with alpha 0.5, so callers declare
opaque input to select that backend. Automatic selection also uses native
TAA when MetalFX is unavailable. Request `.metalFX` explicitly to receive
an error in that case. MetalFX configuration checks use its device-reported scale range and a
minimum 16 × 16 input. The native fallback resolves temporal detail at
input resolution before spatial expansion.

Tests check motion and disocclusion on a premultiplied HDR fixture,
opaque moving HDR pixels through MetalFX on supported hardware,
simulator backend selection, queue ownership, texture validation, and
reset or discard behavior.

All 14 focused raster upscaler and TAA tests passed on the Apple M4 Max and
the Polish iPhone 18 Pro Simulator with Metal API validation. The Mac exercised
both backends. The simulator exercised native output and unavailable MetalFX
selection. Xcode MCP reported a full external DerivedData volume, so the runs
used the existing internal Xcode command-line build caches.
