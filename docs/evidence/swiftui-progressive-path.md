# SwiftUI progressive path view

`FrostfieldPathTracingView` presents a prepared `PathTracer` using the
same native Metal layer and display link as `FrostfieldSceneView`. It
accumulates on visible, active drawables, pauses at its sample cap, and can
report completed sample counts. Setting `isPaused` keeps the last image.
Window and scene lifecycle events suspend scheduling. The host has one
frame in flight, and a stale completion cannot schedule work for an
obsolete input.

Pass a new camera or lighting rig as SwiftUI input. If the prepared scene
updates its transforms or geometry, refresh the SwiftUI input to pick up
the new revision. Use a changed `invalidationID` for in-place texture or
live graph edits. Supply a replacement tracer for a new prepared scene.
Changes in drawable size, tracer or traced inputs restart accumulation.
The native overlays remain available for controls and accessibility.

The trace uses `PathProgressivePreview` for fixed or adaptive render
scales. A filtered tone-mapping pass presents the smaller linear texture
directly to the drawable. The default is neutral tone mapping into
sRGB BGRA8. `hdrDisplay` selects another supported output.
Select raster or path presentation with the same SwiftUI state that
controls the camera and graph. Returning to the path host starts a new
accumulation. To pause a trace in its existing host, use `isPaused`.

```swift
if showPathTracing {
    FrostfieldPathTracingView(tracer: tracer, camera: pathCamera, cameraTransform: graph.cameraTransform)
} else {
    FrostfieldSceneView(commandQueue: queue, graph: graph)
}
```

Prepare the raster graph for its drawable's pixel format. Use
`PathTemporalDenoiser` separately for MetalFX. The optional
[`previewTransition`](path-preview-handoff.md) hands off to a fixed-resolution
trace.

The reference is `@react-three/gpu-pathtracer` 0.2.0. Its source manages
camera and scene replacement, reset, resolution, sample stopping, and
a raster switch inside the React Three Fiber frame loop. The native view
provides declarative tracing and refinement on Apple's display loop.
[Published wrapper source](https://github.com/pmndrs/react-three-gpu-pathtracer/blob/ab3bf066502aae902c96ad8b4e65c57fb0843868/package/src/Pathtracer.tsx)

## Verification

Thirteen focused tests passed on Apple Silicon macOS and on the iPhone 18
Pro simulator. The Metal fixture verified exact endpoint colors and a
bilinear midpoint, pause and resume, sample stopping, camera and light
changes, scene revisions, tracer replacement and recovery from a failed
display encode. The existing HDR, drawable resolution and adaptive preview
suites also passed.

A temporary Lab fixture hosted the public SwiftUI view on the iPhone
simulator with Metal API Validation. It reported completed sample counts
1, 2 and 3, then stopped. Xcode's device capture showed the rendered scene
and the accessibility hierarchy reported `Path preview 3 spp`. The fixture
was removed after verification. The Xcode bridge build was blocked by a full
external DerivedData volume. The Lab build and launch used internal
`xcodebuild` and `simctl`, and Xcode's device session captured the result.
