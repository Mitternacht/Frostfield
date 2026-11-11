# Utah teapot geometry

`MeshGeometry.teapot(size:segments:bottom:lid:body:fitLid:blinn:)` tessellates the 32 original bicubic patches. The body option includes the rim, handle and spout. The lid and bottom can be drawn independently. Each patch keeps its own UV grid. Analytic derivatives provide smooth normals and a top-left UV tangent frame.

```swift
let teapot = try MeshGeometry.teapot(size: 0.75, segments: 18)
let openPot = try MeshGeometry.teapot(size: 0.75, segments: 18, lid: false)
```

`size` defaults to one world unit and sets the half-height of the complete model. `segments` defaults to ten, with a minimum of two and a combined budget of one million vertices. The lid-fit option expands the lid by 7.7% in its horizontal plane. Disabling Blinn scaling restores the original taller proportions at the same normalized height. The mesh follows the original patch seams and openings, including the gap around the lid and the inside of the spout.

The indexed controls come from the pinned r187 port of Martin Newell's Utah teapot through the Standard Procedural Databases. The source file's SHA-256 is `05dfda4e24db0b48a3cc9c10b54c37ca4017921bd123175b0e57d72403227cd1`. The Three.js MIT notice is included with the data. [Three.js r187 TeapotGeometry](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/geometries/TeapotGeometry.js), [Standard Procedural Databases](https://www.realtimerendering.com/resources/SPD/), [Third-party notice](../../LICENSES/Three-TeapotGeometry.txt)

## Verification

- At four segments the r187 reference produced 800 vertices and 992 triangles. Body-only, lid-only and bottom-only counts were 640, 240 and 112 triangles. Native fixtures compare positions, normals, UVs, index winding, bounding boxes, lid fit and non-Blinn proportions.
- A full-array comparison of five configurations produced identical indices. Maximum position and normal component differences were below 8 × 10⁻¹⁵. Converted UV differences were below 3 × 10⁻⁸.
- Collapsed top and bottom rows retain one vertex per UV sample and omit one degenerate triangle per adjacent cell. Tests check exact pole positions, outward normals, tangent frames and triangle counts. At the two-segment minimum, the mesh retains r187’s coarse triangle layout.
- A Metal fixture reads the body and outer handle and spout silhouette and verifies that removing the lid clears the knob’s pixels.
- Xcode 27.2 passed 44 focused tests in nine suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
