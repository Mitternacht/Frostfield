# Vertex frames and UV layout debugging

`LineGeometry.vertexNormals` and `vertexTangents` emit one independent
segment per authored vertex, including unused vertices and vertices
outside the active draw range. Both helpers start at the world-transformed
vertex and normalize the direction before applying the world-space size.
Normals use the inverse transpose. Tangents use the linear transform,
with the same direction for either handedness. The defaults are red
normals and cyan tangents, each one world unit long. Use
`MeshGeometry.generatingTangents` to prepare tangent data when needed.
[Three.js VertexNormalsHelper r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/helpers/VertexNormalsHelper.js) ·
[Three.js VertexTangentsHelper r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/helpers/VertexTangentsHelper.js)

The geometry overload accepts an affine object-to-world matrix. The
scene overload resolves an ordinary mesh through its hierarchy, including
nonuniform parents and reflections. It creates a snapshot that can be
rebuilt when the object moves. Add the result at the scene root with
`DebugHelpers.node`. Use the geometry overload with caller-resolved
vertices for a posed, custom or instanced draw.

```swift
let normals = try LineGeometry.vertexNormals(of: mesh, worldFromObject: model)
let helper = try debugHelpers.node(normals)
let uv = try UVLayoutDebug(geometry: mesh, coordinateSet: .secondary)
```

`UVLayoutDebug` accepts UV0, UV1, or raw UV coordinates with optional
indices. Raw nonindexed coordinates form consecutive triples. It emits
all triangle edges at `(u, 1 - v, 0)`, with world +Y up and native UV V
pointing down. The layout keeps authored coordinates outside the unit
square, so a wider view can show tiled charts. Face annotations give the
triangle index at its centroid. Corner annotations give the corner
letter and vertex index halfway between that centroid and the corner.
Their positions are normalized top-left UV coordinates.
[Three.js UVsDebug r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/utils/UVsDebug.js)

Render the contours with the line helper. For a labeled image, prepare a
`TextureDebugPass.Label` from each visible annotation's text. Multiply its
anchor by the target dimensions and center the glyph mask there. Composite
it with the annotation's linear color on the same command buffer. Callers
choose how to clip or wrap labels at the layout boundary.

All 45 focused tests in six suites passed on arm64 macOS and the iPhone
18 Pro Simulator with Metal validation. The fixtures checked world-space
directions and fixed lengths under a mirrored, nonuniform hierarchy,
sequential and indexed streams, UV1 outside the unit square, and invalid
input. Metal readback verified both colored vertex directions, UV
contour orientation, and a composited Core Text face label. The
Xcode bridge reported a full DerivedData volume, so the tests used
`xcodebuild` with internal DerivedData.
