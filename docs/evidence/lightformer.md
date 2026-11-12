# Lightformers

`Lightformer.node` builds a shaped emissive mesh for an environment capture,
like drei `<Lightformer>`. A rectangle, circle, or ring lies in the XY plane
and faces +Z. The ring's hole has radius 0.5. The box is the unit cube. Flat
forms are double-sided.

`color` is linear RGB from zero through one. `intensity` scales it, matching
`MeshBasicMaterial.color.multiplyScalar(intensity)` with tone mapping off.
The mesh emits that radiance and does not take scene lights. A `target` turns
local +Z from the target back toward the form, as `Object3D.lookAt` does.
An explicit `rotation` replaces that aim. Put the node in the scene passed to
`EnvironmentLighting.capture`.

```swift
let panel = try Lightformer.node(
    device: device, form: .rectangle, color: SIMD3(1, 0, 0), intensity: 4,
    position: SIMD3(0, 0, 2), scale: SIMD3(repeating: 4)
)
let graph = SceneGraph(
    roots: [panel], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 100)
)
let lighting = try EnvironmentLighting.capture(graph, at: .zero, queue: queue)
```

`LightformerTests` passed with Metal API Validation through `xcodebuild`. The
Xcode bridge was not used. The test checks the rectangle, the ring radii, the
circle center, the cube, and a captured +Z sample of a red panel at intensity
4. macOS: `Test run with 1 test in 1 suite passed after 0.315 seconds`.
iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 1.610 seconds`.
