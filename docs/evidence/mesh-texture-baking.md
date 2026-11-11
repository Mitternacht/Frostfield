# UV-space mesh attribute baking

`MeshTextureBaker` converts authored vertex colors and normals into
Metal textures using the mesh's primary or secondary UV atlas. Baking
runs on the CPU. The result is ready for shader reads or export when
the call returns.

```swift
let baked = try MeshTextureBaker(
    device: device,
    geometry: geometry,
    width: 1024,
    height: 1024,
    textureCoordinateSet: .secondary,
    padding: 8
)
let material = try PhysicallyBasedMaterial(
    baseColorTexture: baked.vertexColorTexture,
    alphaMode: .blend,
    baseColorTextureCoordinateSet: baked.textureCoordinateSet
)
```

The color map stores straight linear RGBA in `rgba16Float`. For a mesh
whose color should come solely from the baked map, render the same
geometry with its vertex colors cleared. The normal map encodes
normalized object-space or world-space normals as `0.5 × n + 0.5`.
Use its directional data in the corresponding space. Coverage is
a separate `r8Unorm` texture so transparent interior samples remain
distinguishable from empty texels.

## Atlas rules

- The atlas uses the top-left UV origin and one base level. Both UV
  components must be within 0...1 on each referenced vertex.
- Texel centers have deterministic edge ownership. The rasterizer uses
  12 fractional bits per texel and the top-left edge rule. Either UV
  winding works. Degenerate or sub-texel charts can leave no samples.
- If UV triangles cover the same texel, the caller can reject the bake
  or select the first or last triangle in index order.
- Padding copies the nearest covered texel in four-neighbor steps.
  Equal-distance ties use row-major order. The coverage bytes are 255
  for interior, 128 for padding, and zero for empty.

## Verification

Xcode 27.2 reported 12 focused macOS tests in
`MeshTextureBakerTests` and `VertexColorTests` with Metal API validation,
and six baker tests on the iPad Pro 13-inch (M5) simulator. The baker tests check exact linear RGBA samples across shared edges,
inverse-transpose world normals under a mirrored nonuniform transform,
overlap and padding policies, and input validation. In the material
fixture, an independently drawn mesh with the UV1-baked texture matches
the original vertex-colored mesh within 0.0015 per premultiplied
RGBA channel across the tested interior.

This workflow bakes authored attributes. Illumination, automatic UV
unwrapping, tangent-space transfer, and mip-chain generation are
separate asset operations.
