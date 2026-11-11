# World raycasts for scene geometry

`PointCloud.intersections(ray:localToWorld:threshold:)` and
`LineGeometry.intersections(ray:localToWorld:threshold:)` return the
geometry element index, its world position, the closest point on the ray,
and their separation. Line hits also give the parameter on the drawn
segment. All thresholds are finite, nonnegative, and inclusive. The
default is one world unit. The ray's maximum distance is inclusive and
applies to the closest point on the forward ray.

Point queries use centers and omit zero-size points. Line queries use
centerlines, including closed loop edges and degenerate drawn segments.
The math runs in world space after the affine transform, so the threshold
keeps its units through nonuniform scale, reflection, and shear. If a
parallel segment overlaps a range of equally close ray positions, the hit
uses the earliest one. Equal reported distances retain element order.
[Three.js r187 Points](https://github.com/mrdoob/three.js/blob/r187/src/objects/Points.js) ·
[Three.js r187 Line](https://github.com/mrdoob/three.js/blob/r187/src/objects/Line.js)

`SceneGraph.raycast(ray:pointThreshold:lineThreshold:layers:in:)` merges
the results with mesh triangle, instance, and batch-slot hits. An optional
layer mask replaces the graph camera's layers, and an optional group
restricts hits to its direct children. Ties preserve depth-first node order
and place mesh hits before primitive hits on the same node.

`SpriteRenderer.intersections` and the world-ray overload that takes a render
camera and viewport size intersect the actual billboard plane. They return its
top-left atlas UV and accept a ray from any origin. Tests use perspective and
orthographic cameras, an angled controller ray, and finite distance and
parallel-ray cases. The grouped input host receives the same hit as the direct
query. Viewport queries also account for drawn point sizes and line widths.
[Three.js r187 Sprite](https://github.com/mrdoob/three.js/blob/r187/src/objects/Sprite.js)

Triangle raycasts use the mesh material's front, back or double-sided mode in
both linear and BVH queries. Hits report both UV sets when present and carry
ray-facing interpolated normals and winding-based face normals in local and
world space. The world normals use the inverse transpose, preserving the
authored front side under a mirrored transform. Reusing a BVH with matching
positions and indices reports the current geometry's attributes. Raster tests
compare back-only triangles in regular, instanced, and batched draws with
nearest and all-hit raycasts.
[Three.js r187 Mesh](https://github.com/mrdoob/three.js/blob/r187/src/objects/Mesh.js)

Xcode 27.2 passed 39 tests on macOS and the iPad Pro M5 simulator with
Metal API Validation. The focused suites cover exact and invalid
thresholds, a finite ray's endpoint, reflected nonuniform transforms,
zero-length and parallel segments, drawn loop edges, and mixed scene
results with instance and batch IDs. Existing viewport picking suites
ran alongside the world-ray fixtures.

The combined package at `b6b8781` passed 1,221 macOS tests and 118 focused
iPhone 18 Pro Simulator tests with Metal API Validation. Both runs exercised
the side-aware mesh hit, BVH attribute reuse and world sprite fixtures.
