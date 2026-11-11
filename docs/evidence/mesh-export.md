# OBJ, PLY and STL mesh export

The three geometry exporters accept `MeshGeometry` or a `SceneGraph`
snapshot. The scene overloads capture each resolved world transform,
selected LOD and current pose, and expand mesh instances. Mirrored
transforms reverse the index winding. Hidden geometry is included.
Scene export is synchronous and can wait for vertex deformation on Metal.

`OBJExporter.data(from:)` writes triangles, polylines and points. It
converts primary UVs to bottom-left coordinates and emits sRGB vertex
RGB. Material groups become named OBJ groups. The direct
`OBJMeshLoader` reads normalized sRGB RGB or RGBA into linear vertex
colors. Material definitions and texture files are separate assets.
[Three.js r187 OBJ exporter](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/exporters/OBJExporter.js)

The material-reference overloads write `usemtl` names and an optional
`mtllib` file token. Scene callers supply a default name and can override
it by node UUID. Instances share their node's name. The fixture resolves a
single-token name through a separate MTL file in Model I/O. The OBJ output
and the standalone MTL loader retain multiword names, which the tagged
Three.js OBJ parser also reads.

`STLExporter.data(from:format:name:)` writes one solid in ASCII or
binary. Both use geometric facet normals and preserve the source units.
Binary uses little-endian Float32, an unsigned 32-bit facet count and
50-byte facet records. Degenerate facets get a zero normal.
[Three.js r187 STL exporter](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/exporters/STLExporter.js)

`PLYExporter` supports mesh and point geometry, both binary byte
orders, optional normals, UVs and colors, and named float columns.
[PLY evidence](ply-export.md)

`MeshFileExporterTests` round-trips OBJ UVs and color and both STL
formats, including native Model I/O reimport. A mirrored parent scene
mixes a point cloud, posed instances and a line loop. The fixture
verifies each world-space facet and separate OBJ position, normal and UV
indices. The exporters also reject scenes without representable
geometry. The focused exporter suite passed on macOS and iPhone 18 Pro
Simulator in Xcode 27.2 with Metal API Validation.

These are geometry snapshots. Use glTF or USDZ when an asset needs
textures, materials or a scene hierarchy.
