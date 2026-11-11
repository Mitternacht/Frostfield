# XYZ point clouds

`XYZLoader.load` imports local files, data, or text into a `PointCloud`.
It supports the whitespace-separated XYZ and XYZRGB records used by
Three.js r187. RGB code values in 0...255 become linear RGB, and mixed
files fill XYZ-only rows with opaque white. Blank lines and comments
beginning with `#` are accepted. The loader checks finite coordinates,
record width, and color range and caps input at 64 MiB and 2²² points.
[Three.js r187 XYZLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/XYZLoader.js)

The native loader matched all 201 positions of the r187 helix fixture
bit for bit on macOS and iPad Simulator. The focused suite tests both
record forms, mixed colors, malformed input, and a Metal point-renderer
readback. The fixture run reported 20 tests on each target.
[Three.js r187 helix fixture](https://github.com/mrdoob/three.js/blob/r187/examples/models/xyz/helix_201.xyz)

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API validation.
