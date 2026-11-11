# Built-in three-dimensional curves

`Curve3D` includes all 14 shapes exported by the Three.js r187
CurveExtras catalog. Granny, Knot and Helix use their catalog sizes.
Heart, Viviani, Trefoil, Torus, Cinquefoil, the two polynomial knots
and the four decorated torus knots take their catalog scales by default.
Those 11 factories also accept finite signed scales. Zero collapses the
curve to the origin and negative scale reflects it through the origin.
[Three.js r187 catalog](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/curves/CurveExtras.js) ·
[MIT notice](../../LICENSES/Three-CurveExtras.txt)

All positions evaluate analytically in Double over the normalized parameter
range 0 to 1. The shared `Curve3D` API converts results to Float and reports
positions outside its representable range. It can sample by parameter or
approximate arc length for use with geometry builders.

```swift
let curve = try Curve3D.decoratedTorusKnot5a(scale: 24)
let positions = try curve.points(segments: 128, spacing: .arcLength(subdivisions: 1024))
```

`CurveExtrasTests` compares five default positions per shape and one
negative-scale position for each configurable shape with fixtures generated
by running the tagged JavaScript in Node. The tagged catalog was byte
identical to the file in the published `three@0.187.0` package. The tests
also exercise scale and parameter validation and build a tube from an
arc-length sampled helix.

Xcode 27.2 passed seven tests in `CurveExtrasTests` and
`ProceduralSweepTests` on both macOS and the iPhone 18 Pro iOS 27.2
simulator. The catalog test ran all 14 parameterized cases on each.
The Xcode MCP workspace open timed out, so these runs used
`xcodebuild` with internal DerivedData.
