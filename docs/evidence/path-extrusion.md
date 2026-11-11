# Path extrusion

`MeshGeometry.extrude(_:along:steps:closed:arcLengthSubdivisions:creaseAngle:textureMapping:)` sweeps planar profiles and holes along a `Curve3D`. It samples the spine at approximate equal arc lengths using one distance table. Each ring uses the transported normal and binormal. Open sweeps use flat endpoint caps in material slot 0 and walls in slot 1, with a pair of index groups for each input shape. Use `SceneNode.grouped` for separate cap and wall materials.

```swift
let spine = try Curve3D.cubicBezier(
    from: SIMD3(0, 0, 0), control1: SIMD3(0, 0, 1),
    control2: SIMD3(1, 1, 2), to: SIMD3(2, 1, 3)
)
let mesh = try MeshGeometry.extrude([profile], along: spine, steps: 24)
```

The frame seed and parallel transport follow the r187 construction, with analytic tangents where Frostfield's curve provides them. The default distance table uses 200 subdivisions. A closed sweep corrects seam twist and uses identical endpoint positions and frames, with duplicated UVs and no coincident caps. Path sweeps use flat rims. The straight overload continues to use the native bevel profile and accepts `ShapeExtrusion.steps` for the straight portion, including extruded text using those settings. [Three.js r187 ExtrudeGeometry](https://github.com/mrdoob/three.js/blob/r187/src/geometries/ExtrudeGeometry.js), [Three.js r187 Curve frames](https://github.com/mrdoob/three.js/blob/r187/src/extras/core/Curve.js#L316-L421)

Frostfield keeps its native UV conventions and crease-aware normals. Cap UVs use the local profile. Wall U follows each ring's perimeter, and V runs from the finish cap toward the start by spine distance. Both normalized and world-unit mapping are available. The sweep checks its allocation budget, frame continuity, representable positions and local triangle winding. Choose a profile narrow enough to clear nonadjacent turns. Custom UV callbacks and global self-intersection detection remain separate.

## Verification

- A pinned r187 cubic Bézier oracle with a holed profile produced 48 distinct positions, 48 cap indices and 240 wall indices. Native samples match the selected positions within 0.00012, accounting for r187's finite-difference endpoint tangents.
- Geometry fixtures check signed volume, welded edge incidence, endpoint normals, tangent/UV agreement, world texture scale, closed-loop seam consistency, straight and beveled axial intervals, and rejected stationary, reversing or locally folded sweeps.
- A grouped Metal draw shows the front cap and inner walls in different materials and preserves the open view through the hole.
- Xcode 27.2 ran 42 focused tests in seven suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
