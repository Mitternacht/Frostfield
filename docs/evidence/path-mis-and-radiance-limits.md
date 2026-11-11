# Path sampling and separate radiance limits

`PathTracer.Settings.clampDirect` limits camera-visible contributions
and light reached after one scattering event. `clampIndirect` applies to
deeper paths. Both limit the mean absolute RGB per contribution and scale
channels together. Next-event estimates use the depth of the connection
to the light. A scattering ray that hits the same light uses the same
depth category. The limits suppress bright outliers and can darken the
converged result.

Zero disables the corresponding limit. A nil direct or indirect value
uses `maximumSampleRadiance` for that category. This preserves the
existing peak-channel limit for scattered contributions and the
unclamped camera emission behavior when only the legacy setting is used.
All three settings default to nil.

`multipleImportanceSampling` defaults to true. Turning it off switches
area emitters, sampled HDR environments and visible directional disks to
scattering-only estimation with unit hit weights. Analytic lights without
a scatter-visible emitter keep their light estimator. Changing the
sampling mode or a radiance limit requires fresh accumulation history.

[Pinned path contribution clamp](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/nodes/utils.wgsl.js#L209-L229) ·
[Pinned tracer settings](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/WebGPUPathTracer.js#L401-L462)

## Verification

All 41 focused tests passed on Mac and iPad Pro M5 Simulator with Metal
API Validation. Readback checks mean RGB on camera-visible emission,
one-scatter analytic light and emission reached after two mirror
reflections. Direct and indirect limits act independently, and an
explicit zero overrides the legacy fallback. The analytic point light
stays pixel-identical when MIS is off. With a sampled HDR map and an
area light, MIS and scattering-only images agree in mean energy within
6% over the same 256 samples per pixel. History rejects changes to
either limit or the sampling mode.

The Xcode workspace bridge timed out resolving the initial run destination.
The runs used Xcode 27.2 `xcodebuild` with internal DerivedData.
