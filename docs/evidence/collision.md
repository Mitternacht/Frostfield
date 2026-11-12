# Static mesh collision

`OrientedBoundingBox` supports point containment and clamping, axis-aligned
enclosures, sphere and box overlap, plane tests, and distance-limited ray hits.
An affine transform can rotate, translate, reflect or scale along the box's
axes while preserving orthogonality. `Capsule` supports translation, enclosures,
point containment, and overlap with boxes, spheres and other capsules. Both
colliders include tangency and allow zero extents.

`SceneGraph.collisionBVH` bakes a static world-space snapshot from the selected
mesh layers. It uses the current instance transforms and resolved poses. The
result uses the existing `MeshBVH` ray queries and supports triangle contacts
for spheres, capsules, axis-aligned boxes and oriented boxes. Each contact
reports a triangle index, position, separation direction and depth in the
hierarchy's coordinate space.

`MeshBVH.depenetration` can apply the deepest capsule, sphere, axis-aligned
box or oriented box contact, then query again from the displaced position.
Callers choose the tolerance and a budget of 1 to 64 iterations. The result
includes the net translation and whether it reached tolerance within that
budget. `MeshBVH.moved(_:by:)` applies a capsule displacement first, then that
push-out, so sideways travel stays while the capsule rests on a floor. There
is no swept test. A displacement that no longer overlaps the mesh is kept,
and a spine that crosses a triangle can finish on the side that needs the
smaller push. The function returns the capsule. Whether the push reached
tolerance stays on `depenetration`.

The Three.js comparison is pinned to r187. Frostfield uses its existing BVH
for static scene queries. Its public queries return ray hits, individual
triangle contacts or the net translation. Scene snapshots are rebuilt when
their geometry or transforms change.
[OBB](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/math/OBB.js) ·
[Capsule](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/math/Capsule.js) ·
[Octree](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/math/Octree.js)

## Verification

- `CollisionGeometryTests` checks separated rotated boxes, reflected
  transforms, ray clipping, and capsule caps and corners.
- `MeshCollisionTests` checks sphere and capsule contacts on faces and edges,
  verifies box query results against the existing BVH, and covers degenerate
  triangle handling.
- `SceneCollisionTests` checks collision layers, hidden meshes, mirrored
  instances and snapshot independence.
- `MeshDepenetrationTests` checks two-plane corners for all four collider types,
  opposing walls, tangent contacts, Float position precision, invalid
  iteration settings, and a capsule move that rests on a floor while keeping
  its sideways travel.

An earlier Xcode 27.2 run passed 26 macOS tests across these suites and the
existing BVH and scene bounds suites, and 13 collision tests on the iPad Pro
13-inch (M5) simulator. After the capsule move, `MeshDepenetrationTests`
passed again under Metal API Validation with `xcodebuild`: macOS `Test run
with 6 tests in 1 suite passed after 0.004 seconds`, iPad Pro (M5) simulator
`Test run with 6 tests in 1 suite passed after 0.010 seconds`.
