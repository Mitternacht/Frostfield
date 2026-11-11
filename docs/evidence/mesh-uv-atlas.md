# Automatic mesh UV atlases

`MeshGeometry.unwrappedUVAtlas(resolution:padding:)` cuts connected charts,
flattens them, and packs one square UV1 page. It preserves UV0 and each
indexed corner's original position, normal, tangent, and vertex color.
Triangle order and material ranges stay the same. Seams can duplicate
vertices and the result drops unused vertices. Copy external per-vertex
data by indexing it with `atlas.sourceVertexIndices`.

```swift
let atlas = try mesh.unwrappedUVAtlas(resolution: 512, padding: 4)
let baked = try MeshTextureBaker(
    device: device, geometry: atlas.geometry,
    width: atlas.resolution, height: atlas.resolution,
    textureCoordinateSet: .secondary, padding: atlas.padding
)
```

Use the same geometry, dimensions, UV set, and padding with
`MeshLightmapBaker`. Authored UV0 tangents remain usable with their original
textures. The atlas reserves room between islands for bake padding and
bilinear sampling at the chosen dimensions. It does not generate mips or
UV1 tangents.

## Implementation and verification

The Swift API uses the pinned MIT-licensed xatlas implementation through
a private C bridge. Two upstream files supply the chart solver and packer.
The build renames the C++ namespace and runs each invocation serially
with local state. The packer stops before allocating a second page and
guards the integer conversion of pathological chart dimensions. The
bridge can retry packing at a lower texel density without computing charts
again. License and attribution files are in `LICENSES`.
[xatlas revision](https://github.com/jpcy/xatlas/tree/f700c7790aaa030e794b52ba7791a05c085faf0c)

The atlas checks original corner correspondence, then verifies coverage
with the bakers' fixed-point rasterizer. Tests cover a subdivided plane
remaining one chart, repeatable seam splits on a closed torus, material
ranges and morph remapping, and reserved gutters between packed charts.
The new atlas directly feeds both the attribute and directional lightmap
bakers. Xcode 27.2 ran 14 focused tests on macOS and the iPhone 18 Pro
Simulator with Metal API validation enabled.

The synchronous generator accepts one indexed triangle mesh and a square
resolution from 16 through 4,096. It rejects degenerate input, invalid
parameterizations, and charts that cannot fit with usable sample coverage.
Individual small triangles can fall between texel centers. Determinism
applies to the same ordered mesh, options, and toolchain. Multi-page
atlases and packing multiple instances into a shared texture are outside
this API.
