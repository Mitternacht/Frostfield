# XML VTK surface import

`VTKLoader.load` detects XML VTK PolyData alongside the legacy surface
reader. A single serial PolyData piece can store its arrays as ASCII,
inline base64, or appended base64. Binary arrays accept UInt32 and
UInt64 headers, either byte order, and uncompressed or
`vtkZLibDataCompressor` payloads. Appended offsets are measured from the
byte after the underscore in the original XML. Each zlib block must
consume exactly its declared compressed bytes and yield exactly its
declared decoded bytes.
[VTK XML format](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html)
[Three.js r187 VTKLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/VTKLoader.js)

The loader reads Float32 and Float64 positions and active point normals,
with signed or unsigned 32-bit and 64-bit connectivity and offsets.
Strips and polygon fans use the shared mesh builder. Vertex and line
cells are validated, with triangle surfaces returned as native
`MeshGeometry`. Other XML point and cell data are outside this reader's
attribute scope. The reader accepts UTF-8 and US-ASCII XML, with nesting
limited to 64 levels, 4,096 arrays, and 65,536 zlib blocks per array.
The shared VTK input and geometry limits also apply. Raw appended bytes, other
compressors, multiple or parallel pieces, and XML DTDs are outside the
supported input.

## Verification

The three historical Three.js cube fixtures cover ASCII, uncompressed
inline binary, and zlib-compressed inline binary. Two additional
fixtures store the same binary arrays in the appended base64 layout.
All five matched the r187 loader's 24 positions, 12 triangles, and
24 authored normals bit for bit on macOS and iPad Simulator. The
focused run passed nine test declarations on each target. The
committed tests cover a big-endian UInt64 multi-block stream, CRLF
appended offsets, and invalid data. A rendered fixture samples the
imported point normals and verifies polygon and strip draw coverage.
[Three.js r183 VTK fixtures](https://github.com/mrdoob/three.js/tree/r183/examples/models/vtk)

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API and shader
validation enabled.
