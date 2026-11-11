# Planar projected shadows

`PlanarShadowProjection` projects an authored caster onto an infinite `Plane3D`. A homogeneous light with W = 0 supplies a direction toward the light. Otherwise XYZ divided by W gives the light's world position. The caster's affine world transform applies before projection. The result keeps its triangle indices, UVs, colors and groups, with normals oriented along the receiver normal.

```swift
let receiver = try Plane3D(normal: SIMD3(0, 1, 0), through: .zero)
let projection = try PlanarShadowProjection(plane: receiver, light: SIMD4(-1, 2, 1, 0))
let shadowMesh = try projection.project(casterGeometry, sourceToWorld: casterWorldTransform)

let shadows = try PlanarProjectedShadowPass(device: device, depthMode: depthMode)
try shadows.encode(
    projectedMeshes: [shadowMesh], clipFromWorld: clipFromWorld,
    color: sceneColor, depth: sceneDepth, stencil: scratchStencil,
    commandBuffer: commandBuffer
)
```

Encode after rendering the receiver and opaque occluders into stored scene attachments. The pass loads the scene color, samples scene depth, and clears separate caller-owned Stencil8 scratch storage to blend the group once per covered pixel. The default tint is black at 0.6 opacity. Use the scene's exact world-to-clip transform and forward or reversed depth mode. The pass accepts matching single-sample 2D color, shader-readable Depth32Float and Stencil8 textures.

The projection uses the native equation `dot(normal, point) + constant = 0`. The r187 `ShadowMesh` internally negates `Plane.constant`; the comparison supplies the opposite constant to the JavaScript reference. Native validation catches parallel directional light, positional light on the receiver, and vertices on or beyond the positional light's projection horizon. Projection acts on the supplied geometry snapshot. [Three.js r187 ShadowMesh](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/ShadowMesh.js), [r187 Plane](https://github.com/mrdoob/three.js/blob/r187/src/math/Plane.js)

## Verification

- Pinned r187 `ShadowMesh.update` reference points match native positions for directional and positional homogeneous lights, an offset tilted plane, and an affine caster transform with shear. Native tests verify source indices and attributes, homogeneous light rescaling and rejected singular inputs.
- Metal renders a receiver and nested transformed cube. Its projected shadow darkens the receiver exactly once even when the cube is submitted twice. Readback verifies that closer scene geometry masks the shadow and that the surrounding receiver remains unchanged in forward and reversed depth.
- Sampling stored depth keeps the pass compatible with the iPhone simulator's Metal restriction on mixing separate hardware depth and stencil formats.
- Xcode 27.2 passed 48 focused tests in eight suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
