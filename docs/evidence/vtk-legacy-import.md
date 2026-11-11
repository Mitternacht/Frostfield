# Legacy VTK surface import

`VTKLoader.load` reads a local file or data into indexed `MeshGeometry`.
It accepts legacy ASCII and big-endian binary POLYDATA, with source
points, polygon fans and alternating triangle-strip winding. If both
surface sections are present, both contribute triangles in file order.
Vertex and line sections are validated and contribute to cell numbering.
Only polygon and strip triangles become geometry.
[VTK legacy format](https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html)
[VTK cell numbering](https://github.com/Kitware/VTK/blob/v9.6.0/Common/DataModel/vtkPolyData.cxx)

The loader preserves authored normals and generates missing point
normals. RGB and RGBA color scalars are interpreted as sRGB with linear
alpha. Cell normals or colors expand vertices so that every triangle
uses its source cell's attributes. A point color is converted once, as
is a cell color. Unused numeric arrays are skipped by their declared
shape and width. Scalar lookup-table shading, modern split legacy
OFFSETS/CONNECTIVITY arrays, other dataset types, and robust concave
polygon triangulation are outside this slice.
[XML PolyData import](vtk-xml-import.md)

File size is limited to 128 MiB, source points and cells to 2,097,152
each, connectivity entries to 8,388,608, and the output to 6,291,456
indices and 2,097,152 vertices. The loader checks data bounds, tuple
counts, finite positions, and indices before constructing the mesh.

## Verification

On macOS and iPad Simulator, the ASCII bunny and binary liver from
the last Three.js VTK example fixtures matched the r187 parser.
The 69,897 source positions and 136,603 triangles matched bit for
bit, including the liver's 33,950 authored point normals. The fixture
run passed five test declarations on each target. A synthetic
big-endian strip reaches the Metal color pass with its expected color
and coverage. Other focused tests cover polygon and strip winding,
wrapped ASCII, point RGBA, cell RGB, and malformed counts and records.
[Three.js r187 VTKLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/VTKLoader.js)
[Three.js r183 VTK fixtures](https://github.com/mrdoob/three.js/tree/r183/examples/models/vtk)

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API and shader
validation enabled.
