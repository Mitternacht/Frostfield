# Loop and Catmull–Clark subdivision

`PolygonMesh` describes shared positions, polygon faces, face-varying
UV0, UV1 and linear RGBA, material face ranges, creases, and fixed
corners. Loop refinement accepts triangular faces. Catmull–Clark turns
each n-sided face into n quads and retains that polygon topology until
the caller triangulates it.

```swift
let cage = try PolygonMesh(geometry, hardEdgesAsCreases: true)
let refined = try cage.subdividedCatmullClark(
    iterations: 2,
    options: .init(textureCoordinates: .smooth, projectsToLimit: true)
)
let renderGeometry = try refined.triangulated().generatingTangents()
```

`MeshGeometry.subdividedLoop` and `subdividedCatmullClark` provide direct
paths for triangle inputs. Conversion welds exact positions and discards
zero-area input faces before discovering adjacent faces and creases. If
referenced vertices supply secondary UVs, all referenced vertices must
supply them. Vertices without colors use opaque white when the source
has colors. Refined normals come from the new triangles. Regenerate
tangents for normal mapping. Skin weights and morph deltas live outside
the control mesh and require caller-authored transfer.

UVs and colors can refine linearly or smoothly. Smooth face-varying
interpolation respects attribute seams. Infinite creases remain sharp,
finite creases blend and decay across refinement levels, and boundary
corners can stay fixed. Material ranges follow their child faces.
Options include a triangle budget, up to eight uniform refinement
levels and projection of the final vertices to the limit surface.

## Verification

Xcode 27.2 reported 21 focused tests with Metal API validation on the
Apple M4 Max, and 11 on the iPad Pro 13-inch (M5) simulator.
`MeshSubdivisionTests` verifies these results.

- A Catmull–Clark cube and Loop icosahedron approach their closed-form
  vertex limits. The Loop limit mesh stays closed.
- Boundary rules round an open patch unless its corners are pinned.
  Infinite creases keep a box sharp and finite crease strengths blend.
- Linear and smooth UV refinement preserve face-varying seams.
- A concave quad triangulates with the intended winding and signed area,
  even when the other diagonal is shorter.
- Both subdivision schemes discard welded repeated-position and
  collinear triangles while preserving material assignments, UV1, and
  linear RGBA on the surviving faces.

Loop is the triangle subdivision scheme used by the Three.js ecosystem
modifier. Frostfield also exposes polygon cages for Catmull–Clark.
The verification covers uniform refinement and its authored crease
rules. Adaptive subdivision remains separate.
[Three Subdivide source](https://github.com/stevinz/three-subdivide)
