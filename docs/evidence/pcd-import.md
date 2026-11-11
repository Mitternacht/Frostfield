# PCD point clouds

`PCDLoader.load` reads local files or data into a `PointCloud` that can be
passed to `PointsRenderer`. It retains the organized width and height,
and returns normals, intensity, and integer labels in source point order
when present. Packed RGB and RGBA bytes become linear RGB with straight
alpha. RGB alone is opaque. The loader reads data fields in their declared
order, size, and type and skips unknown fields. `VIEWPOINT` metadata leaves
the authored coordinates in place.

ASCII, binary, and LZF `binary_compressed` are supported. Binary point
fields default to little-endian and callers can select big-endian. The
compressed size prefix is little-endian. LZF output is field-major, with
each field's full `COUNT` values kept together per point. Compressed
layouts omit the PCL padding field `_`.
[Point Cloud Library PCD format](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)
[PCL binary reader and writer](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/io/src/pcd_io.cpp)

The reader limits the header to 64 KiB, the encoded and expanded payloads
to 256 MiB each, and the grid to 2²² points. It checks field widths, point
counts, complete records, decoded lengths, and LZF back-reference ranges.
Returned position, normal, and intensity components must be finite. Extra
complete binary records and zero padding after a declared compressed
stream are accepted for compatibility with the pinned demo files.

## Verification

The focused tests cover organized ASCII, both binary byte orders, mixed
field widths, packed float RGB and integer RGBA, field-major compression,
overlapping LZF references, and malformed input. A real Metal readback
checks the positions, linear colors, and pixel coverage of two imported
points drawn through the scene graph.

All four PCD files from the Three.js r187 demo matched the pinned
JavaScript loader on Mac and iPad Simulator. The 132,622 loaded positions
matched bit for bit. Both colored files matched within 0.00000018 in
linear RGB. The binary Zaghetto fixture carries 327 unused complete
records. The compressed logo carries 3,902 unused zero bytes after the
LZF payload. The focused fixture run reported 22 test declarations on
each target.
[Three.js r187 PCDLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/PCDLoader.js)
[Three.js r187 PCD fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/pcd)

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API validation.
