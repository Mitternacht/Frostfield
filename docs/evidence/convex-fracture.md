# Convex fracture

`ConvexFracture` cuts a convex indexed mesh the way Three.js
`ConvexObjectBreaker` does. The plane is in world space,
`normal · x + constant = 0`. Each mesh edge is classified once. Points on the
positive side, the negative side, and the plane within the tolerance go to the
corresponding piece, and a crossing edge adds its intersection to both.

A side with four points or fewer is dropped. Otherwise the points are rebuilt
with `ConvexHull` around their average, including repeated edge endpoints, and
that centroid is rotated into world space. Mass is halved. Velocity and angular
velocity are copied. The piece stays breakable when twice the largest positive
offset from the centroid is greater than `minimumBreakSize`, which defaults to
1.4.

`subdivideByImpact` makes the first plane through the impact, the body origin
and the impact plus the normal, then repeats. `random` supplies the values
`Math.random` would. Zero radial and random iterations still perform that first
cut and keep both pieces.

```swift
let fracture = try ConvexFracture()
let body = ConvexFracture.Body(geometry: MeshGeometry.cube(), mass: 2)
let plane = try ConvexFracture.Plane(normal: SIMD3(0, 0, 1), through: .zero)
let pieces = try fracture.cut(body, by: plane)
```

`ConvexFractureTests` passed with Metal API Validation through `xcodebuild`.
The Xcode bridge was not used. A unit cube of mass 2 splits at z = 0 into
pieces at z = ±0.375 with mass 1, volume about one half, and the same velocity.
A plane that misses leaves one piece. An impact with a fixed random value keeps
both pieces of the first cut. macOS: `Test run with 1 test in 1 suite passed after 0.004 seconds`.
iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.006 seconds`.
