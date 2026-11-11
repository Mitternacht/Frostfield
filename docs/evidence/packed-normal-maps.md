# Packed tangent-space normal maps

Use `.xyUNorm` or `.xySNorm` on a `PhysicallyBasedMaterial.TextureBinding`
for two-channel normals. `.rgb` is the default and decodes all three channels
from `[0, 1]`. Stock PBR base and clearcoat maps select their encodings
independently.

```swift
let texture = try TextureLoader(device: device).loadKTX1(data: data, interpretation: .linearData)
let normal = PhysicallyBasedMaterial.TextureBinding(texture: texture, normalEncoding: .xySNorm)
let material = try PhysicallyBasedMaterial(normalTexture: normal)
```

Raster and path shading share the decode. UNORM remaps XY to `[-1, 1]`.
SNORM samples are already signed. Both reconstruct Z as
`sqrt(max(0, 1 - dot(xy, xy)))` before applying the slot's normal scale and
the tangent frame of the selected, transformed UV set. A packed map can
represent the positive tangent-space hemisphere. Use RGB for normals that
need a negative Z component.

KTX 1.0 BC5/RGTC and EAC RG11 have signed and unsigned variants. Metal
samples native blocks where supported. The BC fallback expands to RGBA8
UNORM or SNORM, and the binding still describes how to interpret the two
channels. Packed modes accept linear normalized two-channel textures and
compatible RGBA or half-float textures. A normalized texture whose signedness
conflicts with the selected encoding throws during renderer preparation.
[Khronos KTX 1.0 specification](https://registry.khronos.org/KTX/specs/1.0/ktxspec.v1.html)
[Apple BC5 RG SNORM](https://developer.apple.com/documentation/metal/mtlpixelformat/bc5_rgsnorm)
[Apple EAC RG11 SNORM](https://developer.apple.com/documentation/metal/mtlpixelformat/eac_rg11snorm)

Live UV transform edits preserve the encoding. Scene archives store it in
version 4 along with the stored texture format. Omitted encodings decode as
RGB, and captures without packed bindings retain the prior version selection.
An archived native compressed texture needs sampling support for its stored
format on the destination device. Convert packed normals to RGB before glTF
export or when using a base normal map in USDZ.

## Verification

The raster tests compare GPU surface normals from signed and unsigned BC5
and EAC against RGB references, using a rotated UV1 frame and normal scale.
The path tracer tests compare first-hit normal and lit color pixels under
full-resolution and baked sampling. A directional-light fixture checks
that a packed clearcoat normal produces the same highlight as the RGB map
in both renderers. Archive tests compare the restored texture format,
encoding, and raster pixels. A live-factor fixture uses RGBA8 SNORM and
checks the restored transform and pixels.

The five-suite run passed 32 tests on the Apple M4 Max Mac, with Metal API
validation. On the iPad Pro 13-inch (M5) Simulator, all packed-map cases
passed, including BC5 software fallback and native EAC. The broader run
reported 31 passes and a raster specular-mask failure in
`TextureTransformTests.layerMapTransform`. The same two assertions failed
on the unmodified base revision `b3e6615` on that destination.
The subsequent Tier 1 argument-buffer fix restored both specular-mask
assertions. [Raster binding evidence](raster-layer-argument-buffers.md)

Xcode 27.2 MCP timed out opening the package. The runs used `xcodebuild`
with isolated DerivedData.
