# Loft geometry

`MeshGeometry.loft(sections:closed:capStart:capEnd:)` connects corresponding
points in two or more 3D cross sections. Their point counts must match.
Profiles can change shape, size, position and orientation. The default
connects each profile into a ring. An open profile makes a strip.

Wall triangles follow the profile winding. Smooth normals are accumulated
from their triangle areas. Each closed ring duplicates its seam for
texture interpolation and shares the smoothed seam normal. U uses
cumulative mean distance between corresponding points on neighboring
sections. V uses distance along each section, running from one to zero
in Frostfield's top-left UV convention.

Start and end caps are independent. Each projects its endpoint onto a
Newell plane for UV mapping and concave triangulation. The output keeps
authored positions and uses separate cap vertices so the cap can have a
hard edge. A planar cap has a flat normal. An open profile can have a cap
that closes its endpoint polygon while its side remains open.

The API rejects invalid point counts, nonfinite or repeated profile
points, collapsed wall triangles, undefined normals, and cap projections
that do not form a simple polygon. It accepts up to one million output
vertices and up to 4,096 points on a cap. Author corresponding sample
order explicitly, including where the closed seam should lie.
[Three.js r187 LoftGeometry](https://raw.githubusercontent.com/mrdoob/three.js/r187/examples/jsm/geometries/LoftGeometry.js)

## Verification

Xcode 27.2 passed 30 focused tests across six suites on arm64 macOS and on
iPad Pro M5 Simulator. `LoftGeometryTests` checks exact strip indices and
arc-length UVs, matching closed seam normals, concave cap area and outward
winding, watertight edge pairing after joining coincident vertices, and an
open nonplanar cap. It also checks invalid counts, nonfinite points,
repeated endpoints, collapsed walls, and self-crossing caps. The existing
parametric, sweep, extrusion, polygon triangulation and shape fixtures
passed in the same runs.

The Xcode package workspace open timed out, and the Lab workspace could
not create its build arena on the full external volume. Verification used
`xcodebuild` with DerivedData on the internal volume.
