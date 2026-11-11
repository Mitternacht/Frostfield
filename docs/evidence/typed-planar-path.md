# Typed planar paths

`PlanarPath` stores value-semantic commands in Y-up XY coordinates. Commands move, draw lines, draw quadratic or cubic Béziers, interpolate uniform Catmull–Rom splines, add rotated elliptical arcs, and close subpaths. Angles are in radians. An arc joins the current point by default. Use `connecting: false` to start a separate subpath at the arc's start.

```swift
let path = PlanarPath([
    .move(to: SIMD2(-1, -1)),
    .line(to: SIMD2(1, -1)),
    .quadratic(to: SIMD2(1, 1), control: SIMD2(1.25, 0)),
    .line(to: SIMD2(-1, 1)),
    .close,
    .ellipticalArc(
        center: .zero, radii: SIMD2(0.35, 0.25), startAngle: 0,
        sweepAngle: -2 * .pi, connecting: false
    ),
    .close,
])
let shapes = try path.shapes(tolerance: 0.001)
let flat = try MeshGeometry.shape(shapes)
let solid = try MeshGeometry.extrude(shapes, .init(depth: 0.3))
```

`sampledSubpaths` returns an open or closed polyline for each drawable subpath. Closed subpaths omit a duplicated first vertex in their stored points and include the closing edge in their length and equal-distance samples. `makeCGPath` retains Béziers and samples elliptical arcs to the requested chord tolerance. Sampling uses the existing native Bézier flattener and Float output coordinates. Command, sampling and fill budgets bound the work.

`path.shapes()` resolves compound fills with nonzero winding by default. Even-odd parity is available explicitly. Each subpath closes implicitly for filling. `PlanarShape(outerPath:holePaths:tolerance:)` validates one authored contour per outer or hole, irrespective of its winding. Both workflows use the existing flat and extruded mesh builders.

r187 `Path` joins an ellipse to an existing curve, uses uniform Catmull–Rom controls for splines, and `CurvePath` samples by length or by constituent curve. `ShapePath` supports both fill rules, and `Shape` stores explicit hole paths. Frostfield uses tolerance-based flattening and an approximate cumulative polyline length for spaced points. [Three.js r187 Path](https://github.com/mrdoob/three.js/blob/r187/src/extras/core/Path.js), [CurvePath](https://github.com/mrdoob/three.js/blob/r187/src/extras/core/CurvePath.js), [ShapePath](https://github.com/mrdoob/three.js/blob/r187/src/extras/core/ShapePath.js), [Shape](https://github.com/mrdoob/three.js/blob/r187/src/extras/core/Shape.js)

## Verification

- Pinned r187 fixtures cover a mixed open curve, a closed 3–4–5 path, an explicit elliptic hole, and both compound fill rules. The mixed curve's length is 12.9429115. Native equal-distance points match the nine reference samples within 0.001.
- Native tests verify exact endpoint closure, value copies, separated subpaths, adaptive ellipse tolerance, invalid commands and bounded tessellation.
- An authored Bézier and ellipse contour produces a closed extruded mesh with the expected filled area and volume. Metal readback sees the rim and clear pixels through the hole.
- Xcode 27.2 passed 74 focused tests in eleven suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
