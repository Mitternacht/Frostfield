# Polygon triangulation

`PlanarShape.triangulate(outer:holes:)` returns counterclockwise triangles as `SIMD3<Int>` indices into the original contour followed by each hole in order. It accepts either winding and uses the first index of a consecutive run of duplicate points. Closing copies are omitted from the triangles. Hole offsets count every point in the original arrays, including duplicates.

```swift
let triangles = try PlanarShape.triangulate(outer: contour, holes: holes)
let points = contour + holes.flatMap { $0 }
for triangle in triangles {
    let corners = [points[triangle.x], points[triangle.y], points[triangle.z]]
    // Use the corners or the original indices to build geometry.
}
```

`shape.triangulatedIndices()` indexes the stored, normalized rings of an existing `PlanarShape`. Both entry points use the same native ear-clipping triangulator as shape meshes and extrusions. A boundary hole needs three distinct vertices enclosing an area. The public API reports `PlanarShape.Failure` for invalid rings or triangles that cannot be represented consistently. To triangulate self-intersecting or overlapping paths, first resolve them into filled shapes with `PlanarShape.shapes(rings:fillRule:)`.

r187 removes a closing copy from each input ring before indexing the compacted data. Frostfield preserves the original offsets and the source arrays. The triangle diagonals and order depend on the native triangulator. [Three.js r187 ShapeUtils](https://github.com/mrdoob/three.js/blob/r187/src/extras/ShapeUtils.js), [Three.js r187 Earcut](https://github.com/mrdoob/three.js/blob/r187/src/extras/Earcut.js)

## Verification

- Pinned r187 fixtures produced 2, 4, 16 and 5 positive-area triangles for the convex contour, concave contour, two-hole contour and shared-vertex hole. Their filled areas were 12, 16, 14 and 15.625, respectively. Native tests verify those counts and areas, the unfilled regions, boundary edge incidence and original hole offsets.
- Native tests also verify collinear boundary vertices, matching indices in the corresponding flat mesh, and errors for degenerate, nonfinite, crossing, outside and nested rings.
- Xcode 27.2 passed 66 focused tests in ten suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.

## Steiner points

In the standalone triangulation API, a hole with exactly one point refines the fill at that position. Interior points split a triangle into three. Points on shared diagonals split both incident triangles, and points on boundary edges split the one incident triangle. Points outside the fill, including inside a boundary hole, are skipped. Existing ring vertices take precedence at duplicate positions, followed by the first singleton in source order. All used indices still address the original concatenated arrays.

The refinement keeps the existing constraints and tests the exact orientation of Float coordinates when the fast determinant is uncertain. A conservative bound on triangle search work throws `PlanarShape.TriangulationFailure.steinerWorkLimit` when the requested workload exceeds the budget. [Shewchuk's adaptive orientation predicates](https://www.cs.cmu.edu/~quake/robust.html)

r187 links singleton holes as Steiner vertices. The pinned square oracle produced four triangles for an interior point or a point on the diagonal, three on the outside boundary and nine for a point on the edge of a square hole. The same oracle selected later indices for some duplicates and could change the filled area for an outside point or a point inside a hole. Frostfield keeps the validated filled area and stable original indices. [Three.js r187 Earcut](https://github.com/mrdoob/three.js/blob/r187/src/extras/lib/earcut.js#L239-L257)

Native fixtures check both sides of repeated diagonal splits, outer and hole edges, duplicate precedence, original offsets, ignored points and unchanged area. The near-cancellation fixture distinguishes an exact edge from neighboring inside and outside points across a large coordinate range. The orientation calculation also matched 40,000 binary32 samples against exact rational signs.

Xcode 27.2 passed 73 focused tests in ten suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
