# Curve deformation

`CurveDeformer` prepares a reusable distance and frame table from a
`Curve3D`. The source mesh's +X coordinate advances along the curve.
Mesh +Y follows the initial up direction projected into the first
cross-section, and +Z completes the right-handed frame. Parallel
transport carries that frame along the path. Open curves extend along
their endpoint tangents. Closed curves wrap signed offsets and distribute
the seam twist around the loop.

```swift
let flow = try CurveDeformer(curve: path, subdivisions: 512, boundary: .open)
let bent = try flow.deform(mesh, offset: 2, sourceXOrigin: Double(mesh.bounds.minimum.x))
let moved = try flow.deform(mesh, offset: 3, sourceXOrigin: Double(mesh.bounds.minimum.x))
```

The mesh retains its indices, material groups, both UV sets, vertex colors
and tangent handedness. The local deformation derivative carries tangent
directions and inverse-transpose normals. The bake rejects local folds and
triangles that collapse in the output precision. Tessellate the source mesh
to follow tight bends and allow room for each cross-section.

`CurveDeformer.deform(_:placements:sourceXOrigin:distanceScale:)` bakes
independent copies of the same mesh at different distances or on different
curves. It returns one mesh and the index range of each copy. A rigid
`transform(atDistance:)` can place ordinary `InstanceSet` instances without
bending them.

Preparation uses up to 4,096 intervals. A bake accepts up to 4,096 placements,
1,048,576 vertices and 6,291,456 indices. The bounds, picking, export and
render paths can use that geometry directly.

## Runtime updates

`CurveFlow` keeps a fixed number of independently placed copies. Its
renderer reuses the material pipeline, indices and vertex color buffer
across edits. It uses the updated geometry for bounds and picking and
shares the same vertices across color, depth, shadows and surface attributes.
A successful edit publishes a new snapshot, with vertex buffer leases
keeping earlier draws intact. If all three revisions are in flight, a new
encode can retry after one completes. A scene archive saves the current
mesh snapshot.

```swift
let copies = try CurveFlow(
    device: device, mesh: mesh, placements: [.init(curve: flow, offset: 2)]
)
let renderer = try MeshRenderer(curveFlow: copies, colorPixelFormat: .rgba16Float)
try copies.move(along: 0.25, at: 0)
```

The default CPU execution rebakes geometry using the prepared curve
tables and uploads packed render vertices. Changing one placement
recalculates only that copy.

Pass `execution: .metal` to keep source vertices and Hermite/transported-frame
tables on the device. A compute pass reads the current offsets and writes the
positions, normals, tangents and UVs used by the existing renderer. Draws on
the same command buffer share the pending output. Later draws can reuse
the completed revision. Offset edits still evaluate and validate the CPU
geometry for bounds, picking and export. Copies share one indexed draw
with fixed topology.

The Metal path uses the spines supplied at initialization. It checks that
the mesh cross-section can pass through every prepared interval, and bounds
the Float working range for coordinates, offsets and distance scale.
Float rounding can select a neighboring derivative at an interval or
endpoint boundary. Use CPU execution for other spines, out-of-domain
input or exact agreement with the CPU evaluator. Three.js r187 Flow
and InstancedFlow bend in the vertex shader.
[Three.js r187 CurveModifier](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/modifiers/CurveModifier.js)

## Verification

`CurveDeformerTests` checks a straight affine case, analytic quarter-circle
positions and frame directions, the normal and tangent effect of local bend
stretch, closed-loop orientation, placement ranges, and invalid inputs.
`CurveDeformerRenderTests` renders two baked curved copies in one indexed
draw and reads both positions and the original front faces with Metal.
Xcode 27.2 passed ten focused tests on macOS and ten on the iPad Pro
13-inch (M5) iOS 27.2 simulator, including the existing procedural sweep
tests. Both runs enabled Metal API Validation. The Xcode MCP workspace
open timed out, so these runs used `xcodebuild` with internal DerivedData.

`CurveFlowTests` checks offsets changing independently in a reused
renderer, culling and picking after an edit, the surface attribute pass,
and isolated vertex snapshots when draws are submitted out of order.
The five flow tests also cover bounded buffer reuse and discarding an
uncommitted draw. They passed on macOS and the same iPad simulator with
Metal API Validation. Before adding the discard case, the macOS run
passed 20 flow, deformation and instancing tests, and the iPad run
passed 10 flow and curve deformation tests.

The Metal tests also compare GPU positions, normals, tangents and UVs
against the CPU geometry on a curved path, a spatial loop, and an
open-end extension. The runtime render tests run with both execution
modes, including a draw encoded with unretained Metal references.
All 12 focused curve tests passed on macOS and the iPad simulator with
Metal API Validation enabled.
