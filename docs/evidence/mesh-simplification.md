# Indexed mesh simplification

`MeshGeometry.simplified(targetTriangleCount:options:)` and
`simplified(ratio:options:)` reduce an indexed mesh by collapsing edges.
The quadric cost combines position error with weighted normals, primary
UVs, secondary UVs when every source vertex supplies them, and linear
RGBA. The default placement can move interior vertices to reduce error.
Endpoint placement retains complete source vertices.

```swift
let reduced = try geometry.simplified(
    ratio: 0.25,
    options: .init(maximumError: 0.005, locksBoundary: true, vertexPlacement: .endpoints)
)
let weightsForReducedMesh = reduced.sourceVertexIndices.map { originalWeights[$0] }
```

Each output vertex has a source representative. Endpoint placement lets
the caller copy external channels in output order. Skinning and morph
deformation fidelity remain separate from the simplification error.

The simplifier retains separate attribute wedges at exact-position seams,
protects primary UV orientation, and preserves material group ranges and
boundaries. A target can remain unmet when the error cap, local topology,
seams, or boundaries constrain the remaining edges. The reported error is
a weighted quadric estimate. It is separate from a measured surface
distance and from an error bound projected into screen space.

## Verification

Xcode 27.2 with Metal API validation reported 19 tests passed across
`MeshSimplificationTests` and `MeshWeldingTests` on the Apple M4 Max
running macOS 27.2. The simplification suite checks the following results.

- A 512-triangle grid with affine UVs reduces to its two corner triangles
  within the test's zero-error tolerance.
- A 9,216-triangle torus reduces to within four triangles of a 922-triangle
  target. Two-sided distance sampled at vertices and triangle centroids
  stays under 0.0143 mesh units for optimal placement and 0.025 for
  endpoints. Both UV seams remain closed and UV winding stays consistent.
- Open boundaries can be locked. Material boundaries remain fixed across
  grouped and ungrouped spans, with group material indices preserved.
- Linear vertex colors and secondary UVs follow optimally placed vertices.
  Endpoint remapping preserves color, tangent, normal, and both UV sets.
- Tetrahedron edges and isolated triangle components remain intact at
  minimum targets. A no-op retains the original vertex order and indices.

The capability corresponds to the geometry reduction in Three.js
`SimplifyModifier`. Three.js r187 uses meshoptimizer with normal and
primary UV weights and copies attributes from its retained vertices.
Frostfield targets a triangle count directly and offers both endpoint
and optimal placement. [Three.js r187 source](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/modifiers/SimplifyModifier.js)


## Generated levels of detail

`MeshGeometry.levelOfDetailChain` simplifies progressively from one
full-resolution mesh and preserves the source-vertex remap for each
level. `LevelOfDetail(chain:makeRenderer:)` prepares those meshes for
the existing scene selection by camera-to-origin distance. The source
always starts at distance zero. Equal coarse thresholds retain the
most reduced result.

Pass the complete local-to-world transform and the perspective
viewport budget when generating thresholds. The origin-centered
radius uses the transformed vertices of the source and the level.
The distance adds that radius to the estimated depth needed for a
transverse displacement of the cumulative quadric error to reach the
pixel budget. The estimate assumes the camera points at the node origin.
It is separate from a screen-space error guarantee. Scene selection
uses fixed distances. Camera projection changes and world transforms
that stretch the mesh differently require new thresholds. Nonzero hysteresis can hold a coarse level inside its
nominal distance.

`MeshLevelOfDetailChainTests` checks off-origin geometry, translated,
rotated and scaled transforms, affine shear, a scaled ancestor, strict
distances, near-surface projection, and the finest and coarsest flat-grid
levels. It also prepares Metal renderers and verifies scene selection.
Xcode 27.2 reported 19 focused macOS tests with Metal API validation
and 10 on the iPad Pro 13-inch (M5) simulator with iPadOS 27.2. Three.js
`LOD` also selects prepared levels at caller-supplied distances.
[Three.js r187 source](https://github.com/mrdoob/three.js/blob/r187/src/objects/LOD.js)
