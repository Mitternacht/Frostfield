# Mesh BVH proximity

`MeshBVH.closestPoint(to:localToWorld:maximumDistance:)` returns the nearest
world-space point on the original indexed triangles. The hit includes a Double
distance, original triangle index and barycentric weights in source corner
order. The distance bound is inclusive and equal distances prefer the lowest
source triangle index. The query considers both sides of each triangle.
Collinear and repeated vertices contribute their closed segments or point.

The static CPU BVH keeps its original mesh and topology. The query transforms
visited bounds into conservative world AABBs, uses Euclidean lower bounds to
prune, and transforms only candidate triangles. Nonuniform scale, reflections,
shear and singular affine placement are supported. A nearest witness outside
the representable Float position range throws. Distance and barycentrics are
computed before conversion of the position to Float.

`MeshBVHClosestPointTests` compares 240 seeded point queries on a 180-triangle
hierarchy with an independent linear oracle under identity, nonuniform scale,
and mirrored/sheared placement. Focused fixtures cover exact distance bounds,
shared minima in separate leaves, lower-dimensional triangles and far queries.
The existing geometric query, collision, BVH ray and progressive lightmap
suites also run, including optional alpha-filtered visibility.

The reference is [three-mesh-bvh v0.9.17 closestPointToPoint](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/src/core/cast/closestPointToPoint.js).
Frostfield searches for the exact nearest witness within its inclusive limit.

## Two hierarchies

`MeshBVH.closestPoints(to:localToWorld:otherLocalToWorld:maximumDistance:)`
searches both static hierarchies using world-space AABB separation and
returns one source-indexed barycentric witness per mesh. The inclusive bound
and affine placements are the same as for the point query. A distance tie
prefers the lowest receiver index followed by the lowest argument index.
Intersecting faces, coplanar edges, and degenerate segments or points are
supported. The search treats the meshes as surfaces.

`MeshBVH.forEachOverlappingTrianglePair(with:localToWorld:otherLocalToWorld:_:)`
visits pairs with overlapping world-space triangle AABBs. The callback can
test the original triangles and stop traversal by returning true. The API
passes original indices and visits in hierarchy order.

`MeshBVHPairTests` compares a split hierarchy against exhaustive triangle
pairs under multiple placements. It also exercises exact contact, disjoint
triangles with overlapping bounds, early exit, and callback errors. A
separate reference run with Three.js r187 and three-mesh-bvh v0.9.17 matched
the native fixture distances of 0 for an interior face crossing, 0 for
coplanar edge crossings, √0.5 for disjoint faces, and 1 for skew segments.
The upstream comparison APIs are documented in
[three-mesh-bvh v0.9.17](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/API.md#closestpointtogeometry).

## Stable-topology refit

`MeshBVH.refitted(to:)` updates the bounds of a CPU hierarchy for a new
mesh with the same vertex slots and source index array. It preserves the
split topology and triangle ordering, and returns a new snapshot so existing
readers retain the old mesh and bounds. Refit recomputes every leaf from its
revised triangles and merges child bounds in reverse node order. Large
deformations can reduce pruning quality. A fresh build can choose new splits.

`MeshBVHRefitTests` compares the refit with a fresh build across bounded ray
and closest-point queries, box overlaps, and mesh-pair queries. It verifies
attribute updates, topology validation, the stability of a large collapsed
leaf, and concurrent reads of the old snapshot. A reference run with Three.js
r187 and three-mesh-bvh v0.9.17 returned the same ray distance of 1 and nearest
point of (9.75, 0.125, 2) for the translated grid fixture.
[Reference refit contract](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/API.md#refit)

## Caller-defined shape casts

`MeshBVH.shapecast` classifies the root and visited bounds in local mesh
coordinates. Callers can reject a subtree, descend to leaves, or accept a
contained subtree as one range of original triangle indices. An optional
scorer orders siblings. The range and triangle callbacks can stop traversal
or throw. The same tree keeps its node indices and triangle ranges across
refits.

`MeshBVHShapeCastTests` tests a sphere against a subdivided plane and compares
the result to exhaustive triangle checks. The independent Three.js r187
and three-mesh-bvh v0.9.17 run returned the same 80 source triangles, with an
index sum of 19,432. The focused suite also verifies containment, callback
order and failure handling, and node index stability.
[Reference shapecast contract](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/API.md#shapecast)

## Point-cloud picking

`PointCloudBVH(cloud:)` shares the immutable point cloud and orders source
indices into a local spatial hierarchy. Each query prunes against conservative
bounds in a ray-relative world frame, then tests the remaining centers with
the same predicate as the linear `PointCloud.intersections`. Both paths use
inclusive world tolerance and ray limits. Equal distances preserve source
index order, zero-size points are hidden, and an optional draw window limits
eligible source points.

`PointsRenderer(device:pickingBVH:material:)` opts scene world-ray picking
into that hierarchy. The renderer uses the same cloud for its draw and
updates the active range without rebuilding the hierarchy. The original
initializer keeps the linear path.

`PointCloudBVHTests` compares 240 seeded queries on 3,000 points with the
linear path, including mirrored nonuniform placement, shear, translated
origins, and changing draw windows. The pruning fixture checks at most 32
candidates in an 8,192-point grid. A separate Three.js r187 and
three-mesh-bvh v0.9.17 fixture matched source indices 0, 2, and 4 for a
bounded identity ray. The native affine tests use Frostfield's world-space
metric.
[Reference PointsBVH source](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/src/core/PointsBVH.js)

## Skinned and morphed scene picking

The existing scene pick, intersection, and world-ray APIs use BVHs for CPU
skin and morph poses. The renderer applies the same CPU morph-then-skin
evaluation as its linear picking path. It keys two immutable snapshots by
validated joint palette, active morph weights and resolved draw range. A
new pose with the same range refits a prior topology. Shared renderers can
serve different poses without mixing surfaces or replacing data held by a
reader. GPU shader deformation, curve flow, and dynamic geometry continue
through their existing picking paths.

`PosedMeshBVHTests` compares scene queries with a linear scan of explicitly
morphed and skinned vertices. The fixtures cover different poses of the same
renderer, mirrored placement, instance transforms, front and back hits,
both UV channels, draw range changes, and invalid poses. The pinned Three.js
r187 and three-mesh-bvh v0.9.17 reference matched the moved surface hit at
(3.25, 0.125, 1.5) and distance 2.
[Reference SkinnedMeshBVH source](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/src/core/SkinnedMeshBVH.js)

## Selective vertex refit

`MeshBVH.refitted(updatingPositions:)` applies explicit finite positions at
source vertex indices and returns a new snapshot. The first edit records
vertex-to-leaf adjacency and branch parents. Later edits on that snapshot
reuse the topology, recomputing bounds for every referenced leaf and its
ancestors. Repeated corner indices and vertices shared across leaves are
covered. Source indices, groups and the remaining vertex attributes are
retained. Empty edits and equal positions reuse the snapshot. Changing an
unreferenced vertex preserves the hierarchy bounds. The immutable vertex
and node arrays can still incur a copy on a write.

`MeshBVHSelectiveRefitTests` moves a shared apex in separate leaves, including
a repeated-index degenerate triangle. It checks bounded rays, box overlaps,
closest points, mesh pairs, and preservation of the original snapshots. It
also compares 120 queries across three sparse revisions with fresh builds.
The pinned v0.9.17 reference selected six of fourteen visited child node
IDs and produced source triangle hits 0 and 1 at distance 1.5.
[Reference selected-node refit](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/src/core/cast/refit.template.js)

## Concurrent construction

`MeshBVH.build(geometry:progress:)` runs the CPU builder off the caller's
actor in the caller's Swift task. It returns the same immutable tree as the
synchronous initializer and shares the original geometry buffers. An
optional callback receives completed bounds and indexed-triangle counts
from the build executor. Its fraction measures completed work in both
phases.

The build checks cancellation before work, at node boundaries, during
bounds preparation, binning and partitioning, and before returning the
snapshot. Long loops poll every 1,024 triangle slots. Each call runs one
CPU build, and concurrent calls have independent state.

`MeshBVHBackgroundBuildTests` compares the complete topology and query
results with the synchronous build, verifies original buffer identity
and off-main progress, and exercises cancellation before work, in both
build phases, and immediately before the result returns. Cancelled builds
leave the source usable by another build.
[Reference worker contract](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/src/workers/GenerateMeshBVHWorker.js)

`MeshRenderer.preparePickingAcceleration(progress:)` builds the active
index window and installs the result in the cache used by scene picking.
Static meshes, dynamic indexed geometry and a morphable mesh at rest
use this cache. Scene hits keep the original triangle IDs. A geometry or
draw-range edit supersedes the build, and cancellation leaves the current
cache available. Repeated preparation can reuse a hierarchy already built
by a pick. An empty range returns nil.

Xcode 27.2 passed 44 focused tests each on Mac and iPad Simulator with
Metal API Validation. The scene tests check a nonzero index window, a
geometry or range edit while a completed build is still off-actor, and
retry after cancellation. The Xcode bridge timed out resolving the
destination, so these checks used `xcodebuild`.

## Metal shader ray queries

`MeshBVH.makeShaderBuffer(device:)` prepares an immutable snapshot for
caller-written compute, vertex, or fragment shaders. The data occupies one
buffer slot and retains source triangle and vertex IDs. Pack a new snapshot
after refitting a mesh. Bind the same snapshot to each shader stage that
queries it, and retain it through execution when using an unretained command
buffer.

Append caller MSL to `MeshBVHShaderBuffer.metalSource()` and compile the
combined source with `MTLCompileOptions.mathMode = .safe`. For example,
a kernel with the binding `device const uint4* bvh [[buffer(3)]]` can call:

```metal
FrostfieldMeshBVHRay ray = {origin, direction, nearParameter, farParameter};
FrostfieldMeshBVHHit hit = frostfieldMeshBVHNearest(bvh, ray, FrostfieldMeshBVHSide::both);
if (hit.triangleIndex != frostfieldMeshBVHMiss) {
    float3 position = origin + hit.parameter * direction;
    // Use hit.vertexIndices and hit.barycentric to interpolate attributes.
}
```

In Swift, bind that buffer with `try snapshot.bind(to: encoder, index: 3)`.
The vertex and fragment overload takes a `stage` argument. Origins,
directions, and results use the mesh's coordinates. The helper leaves the
direction length as supplied so callers can preserve the ray parameter
through a world-to-mesh transform. Near and far are inclusive and far can
be positive infinity. The shader returns nearest source ID, corner indices,
barycentric weights, authored geometric normal and facing. Equal computed
parameters choose the lowest source triangle index. Inputs and intersection
arithmetic use Metal Float precision.

The shader uses packed CPU hierarchy order and escape links to skip
subtrees. The triangle math is shared with the path tracer, whose far bound
is exclusive. `MeshBVHShaderTests` compiles caller shaders, reads computed
hits and a rendered pixel, and compares a 2,048-triangle grid with the CPU
hierarchy, including shared edges. The path acceleration, generated shader,
and scene update suites also exercise the shared triangle math.
[Upstream packed shader ray query](https://github.com/gkjohnson/three-mesh-bvh/blob/v0.9.17/src/webgl/glsl/bvh_ray_functions.glsl.js)
