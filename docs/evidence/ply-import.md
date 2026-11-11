# Native PLY import

`PLYLoader.load(_:options:)` accepts PLY 1.0 `Data` in ASCII, binary
little-endian, and binary big-endian form. `load(contentsOf:options:)` reads a
local file. Indexed triangle meshes use `MeshGeometry`. Face-free files use
`PointCloud`, with authored normals and UVs returned as separate arrays.
The loader consumes element records in header order and skips geometry
construction for other elements. [Stanford PLY specification](https://gamma-web.iacs.umd.edu/POWERPLANT/papers/ply.pdf)

The loader recognizes the position, normal, UV, RGB, and index aliases from
Three.js r187. Quads split into 0–1–2 and 0–2–3 triangles. It decodes sRGB to
linear Float RGB, converts bottom-left UVs to Frostfield's top-left origin,
and uses an alpha of 1. Authored mesh normals keep their values. Missing or
zero mesh normals use area-weighted incident triangles, with an up vector
when no nonzero sum exists. Per-face RGB and corner UVs expand vertices
and provide a source vertex remap. An empty face UV list uses vertex UVs
for that face. [Three.js r187 PLYLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/PLYLoader.js)

`propertyNameMapping` renames columns for recognition. `customVertexProperties`
selects individual scalar columns, in the requested order. Those columns keep
their signedness, width, and values in `PLYExporter.TypedScalarProperty` arrays.
Face expansion applies the same remap to custom columns. The arrays are CPU
data and can be written back with `PLYExporter`. Imported points render positions
and RGB through `PointCloud`; point normals, UVs, and custom scalars remain
available to the caller. Standard geometry attributes use native Float32 storage.
Faces with more than four corners are rejected.

The parser checks header and body lengths, scalar ranges, integer list counts,
truncation, and face indices. Files are bounded to 256 MiB and 2²² total element
records. The header is bounded to 1 MiB, with up to 256 elements and 256
properties per element. Lists can contain up to 65,536 values, and selected
custom column storage is bounded to 128 MiB. Face expansion is bounded to 2²²
output vertices.

## Verification

Xcode 27.2 (27B52) passed all 12 tests in `PLYLoaderTests` and `PLYExporterTests`
on the M4 Max Mac and iPad Pro 13-inch M5 simulator. Tests cover byte-for-byte
export round trips in all three formats, including all eight scalar types,
the quad split and face attributes, position aliases, normals and UVs on points,
and malformed headers, lists, indices, and truncated bodies. Xcode MCP timed out
resolving the Swift package destination, and the Lab scheme lacks the package
test target. Both runs used `xcodebuild` with internal DerivedData.

Frostfield and the pinned Three.js loader produced equal position bits and
triangle indices for the five PLY assets bundled with r187. The ASCII dolphins,
colored ASCII dolphins, and both binary byte orders each yielded 855 vertices
and 1,689 triangles. The binary Lucy model yielded 50,002 vertices and 100,000
triangles. [Three.js r187 sample assets](https://github.com/mrdoob/three.js/tree/r187/examples/models/ply)
