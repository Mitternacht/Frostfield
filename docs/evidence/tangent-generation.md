# MikkTSpace tangent generation

`MeshGeometry.generatingTangents(textureCoordinateSet:convention:)` computes
MikkTSpace tangents for indexed triangle meshes using either UV set. Use
`.openGL` for the green-channel direction used by glTF and Blender normal
maps. Use `.directX` for normal maps authored with green pointing down the
image. A missing secondary UV on a referenced vertex produces an error.

Mirrored UVs can require a second tangent frame at the same source vertex.
The first frame keeps the original vertex index and additional frames
append copies. The operation keeps color, UVs, and index-range material
groups. For separate per-vertex arrays, use
`generatingTangentsWithRemap(textureCoordinateSet:convention:)` to carry the
original vertex index for every output vertex. For example, a caller can
rebuild skin influences after adding tangents to the bind-pose geometry.

```swift
let result = try geometry.generatingTangentsWithRemap()
let skin = try MeshSkin(
    jointCount: originalSkin.jointCount,
    influences: result.sourceVertexIndices.map { originalSkin.influences[$0] }
)
```

The tests compare generated tangent directions and signs with recorded
Blender reference frames on a mirrored saddle and a sphere in both
conventions. A Metal fixture compares rendered brightness against the
expected normal-map response. Additional checks cover secondary UVs,
source-vertex remapping, material groups, vertex colors, and very small
and very large finite normals.
[Blender MikkTSpace source](https://github.com/blender/blender/tree/main/intern/mikktspace)
· [Three.js r187 geometry utilities](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/utils/BufferGeometryUtils.js)

This API operates on triangles. A non-planar quad baked with different
triangle diagonals can yield different frames. Degenerate UV regions get a
perpendicular unit tangent where the reference leaves the frame undefined.
Callers choose when to generate tangents for imported meshes.
