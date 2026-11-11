# Tube painting

`TubePainter` writes colored, indexed tube strokes from 3D brush samples. A move starts a stroke, a line appends a sample, and each endpoint has a rounded cap. The default brush radius is 0.01 world units with 15 radial sides. Set the radius and linear RGBA for the next sample. The first segment uses the same brush at both ends, and later segments interpolate between the recorded endpoint colors and radii.

```swift
var painter = try TubePainter()
try painter.move(to: SIMD3(-0.8, 0, -2))
try painter.line(to: SIMD3(-0.2, 0, -2))
if let initial = try painter.mesh() {
    let renderer = try MeshRenderer(
        device: device, colorPixelFormat: .rgba16Float,
        dynamicGeometry: initial, capacity: painter.capacity,
        material: .classic(ClassicMaterial(model: .basic))
    )
    // Attach the renderer to a SceneNode, then publish subsequent samples.
    try painter.setRadius(0.025)
    try painter.setColor(SIMD4(0, 1, 0, 1))
    try painter.line(to: SIMD3(0.5, 0, -2))
    if let revision = try painter.mesh() {
        try renderer.updateGeometry(revision)
    }
}
```

CPU edits append one wall ring and update the current rounded tip. Earlier stroke samples and previously returned snapshots retain their geometry. Moving starts a separate stroke with the earlier tip intact. The painter uses shortest-arc frame transport and fixed hemisphere topology as the radius changes. Duplicate samples make no edit. Capacity, reversed direction and folded surface errors leave the existing stroke unchanged.

`mesh()` becomes available after the first drawable segment. Call it when publishing the next revision to an existing dynamic renderer. The renderer manages in-flight geometry snapshots and uploads the full revision. Stroke V follows distance in world units, so earlier coordinates remain stable while painting.

The pinned r187 fixture uses 15 radial sides, four hemisphere intervals and a radius equal to 0.01 times its brush size. It produced 240 triangles for the first capped segment and 270 after an extension. Frostfield preserves these default counts using indexed rings and its native dynamic geometry API. [Three.js r187 TubePainter](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/misc/TubePainter.js)

## Verification

- Native fixtures cover indexed growth, live cap replacement, radius and color transitions, separate strokes, closed edge incidence, and transactional validation.
- A single dynamic Metal renderer displays the original red stroke, then the wider green extension after a geometry update. Readback verifies the new pixels, increased brush width and changed submitted triangle count.
- Xcode 27.2 passed 41 focused tests in seven suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
