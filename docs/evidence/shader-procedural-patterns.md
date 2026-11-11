# Procedural patterns in shader graphs

`ShaderGraphBuilder.simplexNoise` evaluates the polynomial-gradient 3D
simplex field used by Three.js r187's curl helper. `curlNoise` takes
central differences of that field along three axes and returns a
three-component vector. This field is separate from the seeded
[CPU simplex generator](cpu-procedural-noise.md).

`animatedVoronoi(_:time:)` returns the squared distance to the nearest
animated cell point. Its 2D overload visits nine neighbors and its 3D
overload visits 27. Supply a scalar time value, such as the graph clock,
to move the points. The same graph nodes work in compute and material
graphs. Native graph archives preserve their operands.

The Metal buffer fixture compared seven positions and times with a
scalar float32 evaluation of the r187 expressions, including negative
cells and equal components at lattice boundaries. Simplex agreed within
0.00004, curl within 0.001, and the Voronoi distances within 0.002.
Two separately authored and archived material graphs used the patterns
as emissive color. Both rendered the expected pixels in raster and the
path tracer, including different times at the same spatial coordinate.

All 17 focused tests passed on the M4 Max Mac and the iPad Pro M5
Simulator with Metal API Validation. The Xcode bridge timed out
resolving the package destination. Verification used Xcode 27.2
`xcodebuild` with internal DerivedData.

[Three.js r187 simplex and curl](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/math/curlNoise.js) ·
[Three.js r187 animated Voronoi](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/math/voronoiNoise.js) ·
[Three.js MIT notice](../../LICENSES/Three-CurveExtras.txt)
