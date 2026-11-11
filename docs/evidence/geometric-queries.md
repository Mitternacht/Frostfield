# World-ray and standalone geometry queries

`SceneRay.intersection(with:)` accepts an `AxisAlignedBox`, `BoundingSphere`, `Plane3D`, or `Triangle3D`. It returns the surface point and distance in world units. Box and sphere queries return the entry or, from the interior, the exit. A coplanar plane ray returns its origin. The ray interval includes both zero and `maximumDistance`. An otherwise valid hit outside the finite `Float` result range throws `ScenePickingError.unrepresentableResult`.

`Plane3D` exposes a normalized oriented equation, signed point and sphere distances, point projection, closed segment intersections, and box and sphere overlap. It keeps the supplied plane coefficients at double precision for incidence tests. `Triangle3D` provides area, centroid, optional normal and plane, projected barycentric coordinates, containment, and closest points. Collinear or repeated triangle vertices still support closest-point queries against their segments. A triangle ray hit adds weights in authored vertex order, the winding normal, and the front-face flag. The default is double-sided. Callers can select either authored face via `MeshRenderState.Side`.

Sphere queries use the perpendicular cross product to retain small offsets at large world distances and recover the smaller root without subtracting nearly equal distances. Triangle queries use double-precision sheared edge functions to keep shared edges consistent. [Three.js r187 Ray](https://github.com/mrdoob/three.js/blob/r187/src/math/Ray.js), [Plane](https://github.com/mrdoob/three.js/blob/r187/src/math/Plane.js), [Triangle](https://github.com/mrdoob/three.js/blob/r187/src/math/Triangle.js), [Woop, Benthin, and Wald](https://jcgt.org/published/0002/01/05/).

## Verification

The pinned Three.js r187 runtime passed nine matching fixtures for box entry and exit, sphere entry and exit, a coplanar plane ray, a closed plane segment, front and culled back triangle hits, and projected triangle weights.

`GeometricQueryTests` exercises inside and outside hits, touching surfaces, both ray bounds, tangency, tilted coplanar geometry, authored triangle sides and shared edges, degenerate input, and large and small coordinate ranges.

The native package passed 54 focused tests across nine suites on the Apple silicon Mac and on the iPad Pro M5 simulator with Metal validation requested. This included the existing scene picking, primitive world raycast, frustum, BVH, scene bounds, and collision suites. The Xcode bridge could not use the full external DerivedData volume, so the runs used Xcode's command-line runner with internal DerivedData.

## Closed overlap and proximity

`AxisAlignedBox` exposes containment, point clamping and distance, and
overlap with spheres, planes and triangles. Sphere queries include closed
sphere and box overlap. Standalone triangle-box overlap shares the mesh
collision separating-axis test. Face, edge and point contact count, including
degenerate triangles. [Three.js r187 Box3](https://github.com/mrdoob/three.js/blob/r187/src/math/Box3.js)

`SceneRay.closestPoint(to:)`, `distance(to:)` and
`closestApproach(toSegmentFrom:to:)` operate on the closed ray interval.
Segment proximity reports both positions and the ray and segment parameters.
Parallel overlaps select the earliest ray position. Line picking uses the
same segment math with its existing hit-distance cutoff.
[Three.js r187 Ray proximity](https://github.com/mrdoob/three.js/blob/r187/src/math/Ray.js)

The overlap and proximity follow-up passed all 62 focused tests on Mac
and iPhone 18 Pro Simulator with Metal API Validation. Fixtures check
box–sphere and triangle tangency, a diagonal separating axis, a degenerate
triangle, and bounded, parallel and unbounded segment queries.
