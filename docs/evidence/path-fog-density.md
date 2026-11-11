# Heterogeneous fog in world boxes and closed meshes

Pass an immutable `PathTracer.Fog.DensityGrid` to `PathTracer.Fog` and include the fog
in `PathTracer.Lighting`. The grid stores one linear scalar from zero to one per voxel,
with X varying fastest, followed by Y and Z. Index zero touches the minimum corner of
the fog's box, or of the mesh boundary's local bounds. Each scalar multiplies every
channel of the fog's authored extinction. The single-scattering albedo and phase
anisotropy apply throughout the region.

A grid supports 1–128 voxels per axis and uses four bytes per voxel, up to 8 MiB.
The renderer binds the read-only buffer on its owning Metal device and retains it
through command completion. Replacing the grid changes lighting identity and
requires a fresh accumulation. With no grid, the renderer uses homogeneous fog.

The renderer traverses the clipped ray segment voxel by voxel. Each constant-density
segment contributes its length times density to the optical distance. Shadow rays
apply Beer–Lambert attenuation to that integral. Camera and scattered rays draw
an exponential free path in optical distance and invert it inside the voxel that
contains the event. The local scalar cancels from the chromatic mixture weight.
Every iteration either crosses a grid plane or finishes, so the loop is bounded
by the sum of the three grid dimensions. The field is piecewise constant.

The pinned [three-gpu-pathtracer fog shader](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/shader/bsdf/fog_functions.glsl.js)
uses constant-density exponential sampling. Its [fog boundary lookup](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/shader/bvh/inside_fog_volume_function.glsl.js)
detects entry and exit through mesh faces. Frostfield provides the same idea through
`PathTracer.Fog.MeshBoundary`, described below, and keeps axis-aligned world boxes.
Existing closed transmissive and subsurface media continue to replace world fog while
a ray is inside them.

## Overlapping fog regions

Pass up to eight independent regions through `PathTracer.Lighting(fogVolumes:)`
or its scene snapshot initializer. A region is a world box or a mesh volume. Each
can use constant density or its own voxel grid, extinction, albedo and scattering
anisotropy. Extinction adds where regions overlap. A new region or grid changes the
lighting identity and requires a fresh accumulation. The original single-box
initializer uses the same single-box estimator.

The path sampler picks one spectral channel, samples a free path independently
for every intersected region in that channel, and chooses the nearest event.
The event's probability uses that region's local extinction and the combined
transmittance to its depth. This weights all channels and retains the selected
region's albedo and phase function. Shadow rays use the product of transmittance
through each traversed region. [Reference on volumetric transport](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators)

Camera rays and every scattered ray, whether a surface or a fog event scattered it,
take the same branch. Only one world box without a mesh uses the single-box estimator.
A single mesh volume uses the multi-region sampler, so its camera and scattered rays
sample it exactly like a box. An earlier revision skipped that branch: a lone mesh
region then attenuated shadow rays but neither camera nor scattered rays.
`singleMeshCameraRays` and `scatteredRayThroughMesh` guard the fix.

## Mesh boundaries

`PathTracer.Fog.MeshBoundary(device:geometry:)` builds an invisible, immutable
boundary from a triangle mesh. It never draws or occludes anything. The renderer
packs it into the shader BVH snapshot that `MeshBVH.makeShaderBuffer` produces and
shares it between fog snapshots on its device. `PathTracer.Fog(extinction:albedo:anisotropy:meshBoundary:transform:densityGrid:)`
places it with a `SceneTransform`. The fog fills the interior, and an optional density
grid spans the boundary's local bounds and moves with the mesh.

A shader integrates one ray against one region as follows.

1. It carries the world ray into local coordinates without normalizing the direction.
   The ray parameter then equals world distance, so extinction stays per world unit
   under any scale, including non-uniform and mirrored scale.
2. It clips the ray to the region's local box and asks the BVH for the next face.
   Faces wind counterclockwise outward. A front face enters the volume and a back face
   leaves it, so a first back face means the ray already starts inside.
3. It integrates each inside interval separately, with the box estimator for constant
   density or the voxel walk for a grid, and ends at a free-path event or the last interval.
   Hollow shapes and separate shells therefore contribute only where fog exists.

A ray that passes through an edge or vertex hits every face that shares it at nearly one
parameter. Faces on the same side count once. A hit with faces on both sides is a
touch of a convex edge or a slide through a reflex one, and the ray keeps its side.
Faces within 16 ulps of the hit parameter form one group, and a fan of groups is
bounded.

### Accepted shapes

Outer shells wind outward. Interior cavity shells wind toward the empty cavity, and an
island shell inside a cavity winds outward again. Concave shapes and disjoint shells work.

Initialization welds vertices closer than a millionth of the mesh's largest dimension, so
duplicated face corners and exporter rounding close the surface. It then rejects, with
`PathTracer.Failure.invalidFogBoundary`:

- open or non-manifold surfaces, and edges used with mixed orientation;
- zero-area faces and shells that enclose no volume;
- inverted shells, and shells nested inside another with the same orientation;
- shells that overlap another shell at the checked point.

Each connected shell is classified by its signed volume. At one probe point on each
shell, the winding number contributed by all other shells must be zero for an outward
shell and one for a cavity. The probe uses double-precision solid angles. The check
cannot prove that no two shells intersect anywhere. A boundary accepts up to 1,024
shells, and shells times triangles may not exceed 40 million.

### Placement

The placement is a scale, rotation and translation. `PathTracer.Fog` builds the inverse from
those factors in double precision, not by inverting a rounded matrix, and throws
`PathTracer.Failure.invalidFogTransform` for non-finite values, a rotation that is not a
unit quaternion, a scale magnitude below 10⁻⁶ or a largest-to-smallest scale ratio above
10⁶. World bounds beyond ±10¹² units throw `PathTracer.Failure.invalidFog`.

## Limits and gaps

- A ray tracks up to 256 boundary crossings in one region, which is 128 thin slabs. After
  that the region stops contributing along the ray and the remainder counts as outside it.
  A deterministic shadow ray pins the exact count in `crossingLimit`. Each crossing costs one
  BVH query and every event costs two integrals, so very detailed boundaries along a long ray
  are slow.
- Features closer than 16 ulps along a ray merge. That is about 3·10⁻⁵ at a distance of 26.
- A boundary may contain at most 1,024 shells, and a scene may hold at most eight regions.
- Albedo and anisotropy are constant inside a region. Spatially variable albedo,
  anisotropy and emission remain open.
- Regions do not nest or replace one another. Extinction of overlapping regions adds,
  and one boundary rejects nested shells of the same orientation. Priority-ordered or
  nested media with different indices of refraction remain open.
- Inside a closed transmissive or subsurface volume, that volume's own medium
  still replaces world fog.
- Temporal path denoising still requires a scene without world fog.
- The static `PathTracer.metal` library declares `#pragma METAL fp math_mode(safe)`.
  Boundary traversal compares against infinite limits and steps by `nextafter`, which
  fast math may fold away. The runtime-generated surface libraries already compile with safe
  math. A `metal` command-line check confirmed that the pragma keeps those operations
  under `-ffast-math`. No test was run without the pragma, so no test demonstrates the failure.
- The runtime surface library splices the mesh BVH source into its template and throws when the
  include is missing. `spliceRequiresInclude` covers the failure.

## Verification

- `PathFogDensityTests` compares four independently varying voxel columns to
  analytic chromatic attenuation, in both traversal directions. A clear channel
  and a zero-density column retain their expected transmittance.
- Directional and finite point-light shadows match the integral through the
  traversed cells. The point light ends inside the second cell. Both the box
  and an equivalent mesh volume with the same grid pass.
- Free paths traverse empty foreground voxels to reach a dense rear slab. The
  rendered open and occluded volumes match the same slab authored as homogeneous fog.
- A hollow mesh clips separate voxel intervals around its cavity and classifies rays that
  start inside, in both traversal directions. A camera inside the volume stops before a
  visible emitter that sits within it.
- A single mesh volume matches analytic attenuation for a narrow and a wide camera view from
  both sides. A mirror turns the camera ray toward a mesh volume it never entered, and the
  reflected radiance matches the clear scene scaled by the analytic transmittance.
- A mesh volume with a density grid and a world box multiply their transmittance, in
  either order and direction.
- Three placements, with non-uniform, rotated and mirrored scale, match chords that an
  independent slab test computes from the transform's float inverse.
- Single scattering in a mesh volume rotated 45° about the view axis matches the analytic
  radiance integral and coverage. Its sun shadow ray leaves through a corner of the diamond.
- Eight-bounce scattering in a mesh cube matches the same cube authored as a world box.
- A shadow ray through 200 thin slabs tracks exactly 256 crossings.
- CPU checks cover cavities and islands, nested, inverted, mixed and overlapping shells,
  welding of rounding noise, singular and near-singular placements, the inverse of a general
  placement and the missing shader include.
- The suite also checks malformed inputs, the Swift/Metal media record layout (160 bytes per
  region) and rejection of stale lighting history after grid replacement.
- `PathMediaTests` retains the existing homogeneous fog, nested-volume,
  subsurface and phase-function checks.

Verification used Xcode 27.2 command-line `xcodebuild` with Metal API validation
(`MTL_DEBUG_LAYER=1`), because the Xcode MCP bridge timed out and its DerivedData volume
was full.

- macOS, the full `Frostfield-Package` scheme: 2100 tests in 416 suites passed after
  805.8 seconds, plus 6 tests in 2 suites in the SwiftUI target. Seven optional benchmark and
  capture tests were skipped. The fog fixtures ran against both the software BVH and Metal
  ray tracing, and `PathFogDensityTests` passed after 3.5 seconds.
- iPad Pro 13-inch (M5) Simulator, software traversal, the seven suites
  `PathFogDensityTests`, `PathMediaTests`, `PathEmissiveSamplingTests`,
  `PathTemporalDenoiserTests`, `SurfaceShaderRenderingTests`, `PathAccelerationTests` and
  `PathSceneEnvironmentTests`: 54 tests passed after 103.1 seconds. Four tests that need Metal
  ray tracing or direct mip sampling were skipped.

An earlier variant of the crossing limit kept the ray's last inside or outside
classification once the limit was reached. Its camera-ray fixture measured 0.569 where
the tracked slabs predict 0.599, so the shader now treats the remainder as outside the
region, and the deterministic shadow-ray fixture pins the count. The cause of the earlier
measurement was not isolated.
