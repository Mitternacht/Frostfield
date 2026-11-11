# Object-space normal maps

Select `normalSpace: .objectSpace` on a `PhysicallyBasedMaterial.TextureBinding`
for a normal map authored in mesh coordinates. Base and clearcoat normal
maps select their spaces independently. Existing bindings default to
`.tangentSpace`.

```swift
let normal = PhysicallyBasedMaterial.TextureBinding(
    texture: linearNormalTexture, normalSpace: .objectSpace
)
let material = try PhysicallyBasedMaterial(normalTexture: normal)
```

Raster and path shading decode the RGB vector and transform it by the
inverse transpose of the node and instance transform. Back faces flip the
normal. The map samples the selected, transformed UV set. A valid tangent
frame is unnecessary, and the material's normal scale applies only to
tangent-space maps. The same binding can select a packed XY encoding when
the normals stay in the positive object-space Z hemisphere.
[Three.js r187 normal-map shader](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/normal_fragment_maps.glsl.js)

The map uses the mesh object frame for skinned and morphed geometry.
Deformation changes the mesh shape while the mapped normal stays in that
frame. Raster shading applies the authored object-space vector directly.
The path tracer keeps the geometric or vertex normal if the mapped normal
would face through the geometry or away from the incident ray. It traces
back faces of double-sided materials. Its render-state validation rejects
back-face-only materials.

Live UV updates preserve normal space. Scene archives store it in version 5.
Older bindings load as tangent-space maps, and scenes without object-space
bindings keep their earlier archive versions. The glTF and USDZ exporters
report an error for object-space maps. Convert the maps to tangent space
before using these exporters.
[Khronos glTF normal texture](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materialnormaltexture)
[OpenUSD Preview Surface](https://openusd.org/dev/spec_usdpreviewsurface.html)

## Verification

The GPU tests compare surface-attribute normals and lit pixels against
an independently transformed vertex-normal reference. They cover
nonuniform node and instance transforms, mirrored instances, back faces,
packed XY reconstruction, and a transformed UV1 map on geometry with no
usable tangent frame.
The clearcoat fixture checks a separate highlight and unchanged base
normal. A mixed-space fixture checks that an object-space base map leaves
the tangent-space clearcoat frame intact. The archive fixture compares the
pixels after a live UV edit and round trip, then verifies the interchange
errors.

The M4 Max Mac run passed 48 tests with Metal API validation, including
instancing, skin and morph regressions. The iPad Pro 13-inch (M5) Simulator
passed 33 focused tests with Metal API validation. Both ran the new raster
and software path-tracer pixel cases. Xcode 27.2 MCP timed out resolving the
package run destination. Verification used `xcodebuild` and isolated
DerivedData.
