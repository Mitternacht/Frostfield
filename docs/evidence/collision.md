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
budget.

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
  opposing walls, tangent contacts, Float position precision and invalid
  iteration settings.

Xcode 27.2 passed 26 macOS tests across these suites and the existing BVH and
scene bounds suites. The iPad Pro 13-inch (M5) iOS 27.2 simulator passed the
13 collision tests. Both runs enabled Metal API Validation.
