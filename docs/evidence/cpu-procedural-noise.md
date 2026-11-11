# CPU Perlin and simplex noise

Use `ShaderNoise.improvedPerlin` for signed 3D improved Perlin noise.
Its fixed lattice repeats every 256 cells along each axis. Use
`SimplexNoise(seed:)` for repeatable signed 2D, 3D and 4D samples,
or `init(table:)` to reproduce an authored 256-value lattice.
The simplex cell rank selects a consistent corner order at ties.
Nonfinite coordinates return NaN.

The focused Mac suite passed 12 tests across three suites with
Metal API Validation. A Node run against Three.js r187 produced
36 reference values for four algorithms, including negative cells,
exact ties, wrapping and large coordinates. The native results matched
within 1e-12. Tests also check seed repeatability, invalid inputs
and imported tables. The Xcode bridge timed out resolving the package
destination, so the run used `xcodebuild` with internal DerivedData.

[Three.js r187 ImprovedNoise](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/math/ImprovedNoise.js)
· [Three.js r187 SimplexNoise](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/math/SimplexNoise.js)
· [Three.js MIT notice](../../LICENSES/Three-CurveExtras.txt)
