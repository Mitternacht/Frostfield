# Marching cubes

`MarchingCubes` extracts an indexed `MeshGeometry` from a regular scalar
grid. Supply world bounds and the sample counts, including both endpoints
of each axis. Load stored values in X-fastest order or sample a Swift
function. Positive-side samples have values strictly greater than the
isovalue.

```swift
let bounds = try AxisAlignedBox(minimum: SIMD3(repeating: -1), maximum: SIMD3(repeating: 1))
var field = try MarchingCubes(resolution: SIMD3(repeating: 48), bounds: bounds)
try field.addMetaball(center: SIMD3(-0.18, 0, 0), strength: 0.04, subtract: 0.1, color: SIMD3(1, 0, 0))
try field.addMetaball(center: SIMD3(0.18, 0, 0), strength: 0.04, subtract: 0.1, color: SIMD3(0, 0, 1))
let surface = try field.mesh(isovalue: 1, boundary: .requireClosed)
```

Metaballs and axis-aligned planes use the r187 compact inverse-square
potentials in world units. Negative strength subtracts a contribution.
The field supports replacement and reset, and potential updates keep the
previous samples if a result would exceed Float range.

Each colored metaball contributes linear RGB using the r187 quintic
falloff over its support radius. Active contributions accumulate at the
grid samples, interpolate across shared surface edges, and become
opaque mesh vertex colors. Colors are explicit per ball. Replacing or
resetting the scalar field clears its palette.

The extractor visits every cell. It interpolates crossings once per grid
edge and welds crossings that snap to the same exact-level sample. Winding
faces lower field values. Normals use central or boundary one-sided
differences, and UVs use the grid's XZ projection. Checkerboard faces use
the bilinear asymptotic decider in a shared face order. A saddle tie makes
the same deterministic choice from both sides.

Each cell contour loop is filled as a disk. Interior trilinear ambiguities
that connect loops through a cell are outside this method's guarantees.
The extractor checks edge incidence and vertex fans and rejects singular
topology, including coincident sheets on an internal exact-level face.
Open mode clips at the grid bounds. Closed mode requires exterior
boundary samples and a closed mesh. No-area output returns nil. Resolution,
storage and output triangle budgets are validated before publishing a mesh.

Three.js r187 uses the classic fixed edge and triangle tables attributed
there to Paul Bourke and Cory Gene Bloyd. Frostfield computes face
connections directly and uses independent indexed output. The Swift field
and mesh APIs cover authored scalar grids, metaballs and planes. Each
extraction creates a mesh snapshot. `MeshRenderer.updateGeometry` can
publish the next snapshot on one prepared raster renderer with bounded
buffers that preserve earlier frames. [Dynamic geometry](dynamic-geometry.md) ·
[Three.js r187 MarchingCubes](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/MarchingCubes.js) ·
[Three.js MIT license](https://github.com/mrdoob/three.js/blob/r187/LICENSE) ·
[Nielson and Hamann's asymptotic decider](https://doi.org/10.1109/visual.1991.175782) ·
[Lewiner et al. on interior ambiguity](https://thomas.lewiner.org/pub/marching_cubes_jgt.html)

The marching cubes, mesh and material suites passed 21 tests each on
macOS and iPad Pro M5 Simulator with Metal API Validation. Fixtures cover
every nonuniform cube mask, both shared saddle decisions and the exact
tie, an aligned open plane, coincident sheet rejection, a closed indexed
sphere and merged metaballs, and GPU pixels showing the sphere's normals.

The palette follow-up passed 23 tests on each destination. A Node run
against revision 187 confirmed the red and blue half-radius sum. The
Swift fixtures verify the exact-level color, interpolation between
samples, and the resulting mixed color in a raster pixel.
