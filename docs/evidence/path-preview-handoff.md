# Two-stage progressive path presentation

Use `FrostfieldPathTracingView.previewTransition(_:)` to retain a low-resolution
preview while a fixed-resolution path trace starts accumulating.

```swift
FrostfieldPathTracingView(tracer: tracer, camera: camera)
    .previewTransition(.init(
        renderDelay: .milliseconds(32),
        minimumSamples: 4,
        fadeDuration: .milliseconds(250)
    ))
```

The preview uses the view's adaptive or fixed preview configuration with a
configurable upper scale. It must complete an image before the handoff.
After the render delay, the view retains that image and starts a separate
accumulation at the fixed final scale. Large final targets are capped by the
native accumulation limits. Both stages use the canvas aspect and the
view's sample cap.

The final image remains hidden until a successful command reaches the
minimum sample count. A zero minimum still requires a completed final image.
The display pass samples the two textures independently and blends their
premultiplied linear values before tone mapping. The fade clock uses active
presentation time and pauses with the view. At the final sample cap, display
ticks continue until the fade finishes, then the host stops drawing on demand.
A zero fade duration reveals the qualified final image immediately.

Camera, lighting, scene revision, size or invalidation changes start a new
handoff. The last successfully presented image survives until a valid
replacement is drawn. A failed display encode discards its unsubmitted trace.
The preview and final histories remain separate from MetalFX and still export.

The pinned WebGPU tracer captures its low-resolution target, waits for its
full-size average sample count, and fades in the final image by elapsed
presentation time. The native path supplies uniform samples per completed
batch, so the final gate uses that completed count directly.
[Reference WebGPU tracer](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/WebGPUPathTracer.js)

## Verification

Fifteen focused tests passed on Apple Silicon macOS and the iPhone 18 Pro
simulator. Deterministic monotonic timestamps and real Metal readback
verified the preview delay, the completed final sample gate, the halfway
and final colors, holding the image across pause and resume, and completing
the fade after tracing reached its cap. The same fixture exercised resize,
explicit invalidation and rejection of a failed final display encode.
The existing HDR display and single-stage preview suites also passed.
The Xcode bridge build encountered a full external DerivedData volume,
so verification used `xcodebuild` with internal caches and Metal validation.
