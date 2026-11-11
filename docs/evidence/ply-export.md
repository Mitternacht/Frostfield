# PLY geometry export

`PLYExporter.data(from:options:)` writes Stanford PLY 1.0 as ASCII or
binary in either byte order. It accepts `MeshGeometry`, `PointCloud`,
and `SceneGraph`. Positions use Float32. Meshes can include normals,
bottom-left primary UVs as `s` and `t`, and triangle faces. Vertex colors
are clamped, encoded to sRGB and quantized to RGB bytes. Missing colors
use white when another vertex supplies color.

The scene overload writes mesh instances and point clouds in resolved
node order. Mesh vertices include the current skin and morph pose. The
export applies each node's world transform and each mesh instance placement.
Mirrored instances retain front-face winding. A scene containing triangles
and points keeps its faces and leaves the point vertices unreferenced.
Set `includeFaces: false` to write only vertex records. A point-only file
omits the face element automatically.

Pass ordered `ScalarProperty` values for named Float32 columns, or
`TypedScalarProperty` values through `Options.typedScalarProperties` for
signed and unsigned 8-, 16-, and 32-bit integers, Float32, or Float64. The
typed columns follow the Float32 columns from `Options.scalarProperties`.
Both use unique ASCII identifiers and one value per output vertex.
Floating-point values must be finite. Numeric precision and the requested
binary byte order are preserved. A point cloud's size multipliers can be
exported as a Float32 column. Mesh and point materials, alpha, textures,
tangents, and secondary UVs are outside this geometry format's supported
export contract.

The standard field names and custom scalar types match Three.js r187.
Its exporter drops all faces when it sees a Points child. Frostfield keeps
mesh faces in mixed files unless face export is disabled.
[Three.js r187 PLY exporter](https://raw.githubusercontent.com/mrdoob/three.js/r187/examples/jsm/exporters/PLYExporter.js)

Model I/O reads triangle positions, normals and winding in all three
encodings, and reads a face-free point file as point topology. It exposes
normalized, sRGB-encoded RGB floats. The export tests verify UVs and custom
columns against the file itself because Model I/O omits those columns.

## Verification

Xcode 27.2 passed 22 tests across `PLYExporterTests` and
`ModelIOSceneLoaderTests` on arm64 macOS. All seven `PLYExporterTests`
passed on iPad Pro M5 Simulator. The fixtures check mesh imports in all
three encodings, face-free point import, and mixed world-space geometry.
They also check exact ASCII and both binary byte orders for all eight
scalar types, including integer values beyond exact Float32 precision
and the next representable Float64 after one. Legacy Float32 columns,
property validation, instance winding, and face offsets remain covered.

The Xcode package workspace open timed out. The Lab workspace could not
create a build arena on the full external volume, so verification used
`xcodebuild` with DerivedData on the internal volume.
