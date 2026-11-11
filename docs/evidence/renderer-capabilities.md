# Renderer capability queries

Create `RendererCapabilities(device:)` from the Metal device used by the
scene or path tracer. The value lists Frostfield's supported raster sample
counts, optional GPU paths, the maximum Metal buffer length, and the device's
per-axis threadgroup limits.

```swift
let capabilities = RendererCapabilities(device: device)
let acceleration: PathTracingScene.Options.Acceleration =
    capabilities.hasFeature(.hardwareRayTracing) ? .metal : .software
```

Feature queries reuse the path denoiser, indirect draw and stage timestamp
gates. Native device queries supply ray tracing, tier 2 argument buffers,
block compression and Float32 texture filtering. The batching query creates
one probe command buffer during capability discovery. Meshes, batched meshes
and primitives share the sample-count check.

The snapshot answers which paths a device exposes. A particular format, pass,
texture or pipeline still has its own limits and preparation errors. Use
`MTLDevice` for device queries outside the Frostfield feature list and the
prepared compute pipeline for its kernel-specific threadgroup limit.
[Apple Metal feature detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions) · [Metal device limits](https://developer.apple.com/documentation/metal/mtldevice) · [Three.js r187 Renderer](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js)

The public snapshot checks ran on the M4 Max Mac and iPhone 18 Pro Simulator
and prepared a real Metal pipeline at every reported sample count. The
focused runs passed 49 tests on Mac and 37 on iPhone Simulator, with eight
simulator skips for unavailable BC sampling and indexed indirect draws.
Xcode 27.2 `xcodebuild` ran with Metal API Validation after the Xcode bridge
timed out resolving the package destination.
