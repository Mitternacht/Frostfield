# Native STL import

`STLLoader.load(_:)` accepts ASCII and little-endian binary STL data.
`load(contentsOf:)` reads a local file. Each facet uses three independent
vertices in file order, preserving hard edges, winding, and authored
normals. Zero normals use the geometric facet normal, with an up vector
for degenerate triangles. The loader preserves original units.

ASCII files return solid names and contiguous index ranges in file order.
Nonempty solids map to `MeshGeometryGroup` material slots by their ordinal.
Empty solids retain their names and empty ranges in the asset. The binary
byte count takes precedence over a header starting with `solid`.
[Three.js r187 STLLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/STLLoader.js)

Binary files with a Magics `COLOR=` marker decode the four default color
bytes. A set high bit in the facet attribute uses default RGB. A clear
high bit uses that facet's five-bit red, green, and blue values. Both
sources convert sRGB to linear RGB. The header alpha applies to each
vertex color. Files without the marker use geometry alone. VisCAM/SolidView
colors and Magics material extensions are outside this decoder's contract.
[Three.js r187 color handling](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/STLLoader.js)

The loader bounds the file to 256 MiB and the geometry to 2²⁰ facets.
It requires the declared binary length, or complete ASCII facet and
solid records, and rejects nonfinite coordinates or normals. The existing
`ModelIOSceneLoader` entry point continues to use ModelIO.

## Verification

Xcode 27.2 (27B52) passed all 25 tests in `STLLoaderTests`,
`MeshFileExporterTests`, and `ModelIOSceneLoaderTests` on an M4 Max
Mac and the iPad Pro 13-inch M5 simulator. These runs covered both native
export round trips, named and empty ASCII solid ranges, zero normals,
binary color precedence, and malformed records. The Xcode MCP timed out
resolving the Swift package destination, and the Lab scheme lacks package
tests. Both runs used `xcodebuild` with internal DerivedData.

Frostfield and the pinned Three.js loader agreed on triangle counts,
position bits, authored normals, and ASCII solid names and ranges across
all six STL files bundled with r187. This included the 2,156-triangle
colored binary fixture, with linear RGB agreement within 5.97 × 10⁻⁸
and matching header alpha. [Three.js r187 sample assets](https://github.com/mrdoob/three.js/tree/r187/examples/models/stl)
