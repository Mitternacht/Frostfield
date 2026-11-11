# Mesh normals, transforms and centering

`MeshGeometry.generatingVertexNormals()` sums the area-weighted normals
of indexed faces. Separate indices preserve authored seams. Vertices with
no net contribution retain the direction of their previous normals.
The result keeps index order, material groups, colors and both UV sets.
Regenerate tangent frames after changing normals.

```swift
let processed = try mesh.generatingVertexNormals().generatingTangents().centered()
```

`transformed(by:)` bakes a finite affine matrix into positions, inverse-
transpose normals and transformed tangents. Mirrors reverse triangle
winding and tangent handedness so the prepared mesh retains the same
front face as an equivalent mirrored node. The mesh file exporters share
the same transform implementation. `centered()` moves the axis-aligned
bounding box center to the origin.
[Three.js r187 BufferGeometry](https://github.com/mrdoob/three.js/blob/r187/src/core/BufferGeometry.js) · [MikkTSpace tangents](tangent-generation.md)

The pinned r187 implementation produced the same area-weighted and
inverse-transpose normals for the test fixture, and the same centered
bounds. The Metal fixture compares the baked and live mirror's full
color images. All 39 focused tests passed on the M4 Max Mac and iPhone
18 Pro Simulator with Metal API Validation, covering processing,
merging, tangent generation, mesh file export and draw ranges. Xcode
27.2 `xcodebuild` ran after the bridge timed out resolving the destination.
