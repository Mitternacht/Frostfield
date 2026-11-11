# Curve tube geometry

`MeshGeometry.tube(along:radius:tubularSegments:radialSegments:closed:capped:arcLengthSubdivisions:)` samples a `Curve3D` at approximate equal arc lengths and transports the circular profile with the same frame construction as path extrusion. The default is an open tube with 64 axial intervals, 8 radial sides and a 200-interval arc-length table.

```swift
let spine = try Curve3D.cubicBezier(
    from: SIMD3(0, 0, 0), control1: SIMD3(0, 0, 1),
    control2: SIMD3(1, 1, 2), to: SIMD3(2, 1, 3)
)
let tube = try MeshGeometry.tube(along: spine, radius: 0.12, tubularSegments: 48, capped: true)
```

Wall U increases along the curve. Frostfield's top-left V is one minus the radial fraction. Triangle winding follows r187. Wall tangents follow increasing U and account for the distributed roll of closed paths. Positions, normals and tangents match exactly on either side of the duplicated radial seam. Closed tubes duplicate the first ring with U set to one and require matching endpoint positions and tangents. Optional open caps use exact wall rim positions with flat normals and separate disk UVs. [Three.js r187 TubeGeometry](https://github.com/mrdoob/three.js/blob/r187/src/geometries/TubeGeometry.js), [Three.js r187 Curve frames](https://github.com/mrdoob/three.js/blob/r187/src/extras/core/Curve.js)

The polyline overload retains its circumferential U, axial V and existing defaults. Curve validation covers segment counts, allocation bounds, frame continuity and local triangle winding. Choose a radius that clears nonadjacent turns.

## Verification

- The pinned r187 cubic Bézier fixture produced 54 vertices and 240 indices. Native samples match its selected positions within 0.00012 and normals within 0.0004, with analytic native curve tangents and the top-left V conversion.
- Native tests check signed volume, welded edge incidence, both UV seams, tangent handedness, cap rim positions and rejected frame and surface inputs. A Metal render reads the authored color on the lateral wall and cap and clear pixels through an uncapped bore.
- Xcode 27.2 passed 53 focused tests in ten suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
