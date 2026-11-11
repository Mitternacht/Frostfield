# Classic environment reflection and refraction

Basic, Lambert and Phong materials share `ClassicMaterial.EnvironmentMap`.
Bind a shader-readable HDR cube, such as `EnvironmentLighting.skyCube`.
The default `.reflection` samples the mirrored view ray. Select
`.refraction(ratio:)` to sample a refracted view ray, with air IOR divided
by material IOR. The ratio accepts zero through one and defaults to
`0.98` when refraction is selected. The lookup uses the shaded world
normal, the surface-to-eye direction for perspective views, and the
constant view direction for explicit orthographic views.
[Three.js classic material r187](https://github.com/mrdoob/three.js/blob/r187/src/materials/MeshBasicMaterial.js#L145-L172) ·
[Three.js environment shader r187](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/envmap_fragment.glsl.js)

Both modes use the existing cubemap yaw and intensity. Multiply blends the
shaded RGB toward its product with the environment, mix blends directly
toward the environment, and add adds the environment times reflectivity.
The lookup uses the cubemap’s RGB channels.

Scene archive v24 stores the optional refraction ratio. Older archives
decode as reflection, and reflection-only captures retain their previous
archive version. Frostfield uses scalar reflectivity and yaw.
Three.js also supports a specular texture as a per-texel weight and a full
Euler environment rotation.

All 38 focused tests in five suites passed with Metal validation on
arm64 macOS and the iPhone 18 Pro Simulator. GPU readback distinguished
the reflection and refraction cube faces at an oblique angle, checked
yaw and parallel orthographic rays, and compared all three combines on
Basic, Lambert and Phong materials against their linear RGB equations.
The tests also checked exact pixels at zero reflectivity and after
loading v1 reflection and v24 refraction archives. The test used
`xcodebuild` with internal DerivedData after the Xcode bridge reported
a full volume.
