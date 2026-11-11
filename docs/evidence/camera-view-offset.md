# Raster camera view offsets

Use `CameraViewOffset` to render a pixel window of a larger image. Its
origin is measured from the top left of the full frame. Signed offsets and
windows extending beyond the full image are useful for overscan. Render to
the window's pixel dimensions to keep the full frame's sampling density.

```swift
let window = try CameraViewOffset(
    fullWidth: 3840, fullHeight: 2160, x: 1920, y: 0, width: 1920, height: 1080
)
let projection = try camera.projection(viewOffset: window)
let tileCamera = try ShaderGraphCamera(
    viewMatrix: viewFromWorld, projectionMatrix: projection, near: camera.near, far: camera.far
)
try graph.encode(
    pass: tilePass, commandBuffer: commandBuffer,
    clipFromWorld: tileCamera.projectionMatrix * tileCamera.viewMatrix,
    cameraPosition: cameraPosition, graphCamera: tileCamera
)
```

For an existing full-frame view, use `fullCamera.cropped(to: window)`.
This works with orthographic, shifted, and reversed-depth projections and
preserves the view matrix and depth range. Supply the same depth mode to
rendering and picking, and the fixed viewing direction for orthographic
shading. Culling and `SceneGraph.pick(at:camera:depthMode:)` use the
cropped view. The path tracer has a separate tiled still API.
[Tiled path output](path-output.md)

The clip-space crop formula matched the matrices from Three.js r187 for
200 perspective and orthographic windows, including overscan, zoom, film
shift, and forward and reversed depth. The largest difference between
the double-precision reference coefficients was 7.11 × 10⁻¹⁵.
[Three.js perspective source](https://github.com/mrdoob/three.js/blob/r187/src/cameras/PerspectiveCamera.js) · [Three.js orthographic source](https://github.com/mrdoob/three.js/blob/r187/src/cameras/OrthographicCamera.js)

The Metal fixture renders a full frame and an off-center tile with the
same scene. It compares all 1,024 tile pixels against the matching full
frame crop and verifies culling and picking for both forward-depth
perspective and reversed-depth orthographic views.
The integrated fixtures passed on the M4 Max Mac, iPhone 18 Pro Simulator,
and Vision Pro Simulator in Xcode 27.2 with Metal API Validation.
