# Raster scene backgrounds

`EnvironmentBackground(device:source:destinationPixelFormat:)` prepares a
color, flat texture, cube, equirectangular panorama, or filtered environment
for the raster scene's background. Encode it before the scene's color pass
and give that pass the same attachment with a `.load` action. The background
pass uses no depth attachment. An optional whole-pixel viewport preserves
initialized pixels outside it and uses its own aspect ratio for the sky.

Flat images fill the viewport with top-left UVs. A texture transform and
sampler control placement, repetition, and filtering. Metal decodes sRGB
sources before applying intensity. Straight-alpha input becomes premultiplied
linear scene color. Set `premultipliedAlpha: true` when using a scene render
as the source. The prepared pass can write RGBA16Float for HDR composition,
or linear or sRGB RGBA8 and BGRA8. Display exposure and tone mapping belong to
the caller's display pass.

Cube and panorama orientation is an independent world-space quaternion.
The sky follows camera rotation, and camera translation leaves it fixed.
A panorama uses a 2:1 image with +Y at the top and +X on the center column.
Horizontal sampling wraps at the seam. A raw cube or panorama uses
trilinear source mips when blurriness is above zero. Supply populated mips
for this mode. A filtered environment uses the sharp sky at zero and its
existing GGX roughness levels as blurriness rises to one. The older
`EnvironmentBackground(lighting:)` initializer and yaw-based RGBA16Float
encoding remain available.

Use `encode(destination:view:options:viewport:commandBuffer:)` with the same
`ShaderGraphCamera` as an explicit raster pass, such as an imported glTF view.
An asymmetric perspective matrix sets the actual ray directions in the
viewport. An orthographic view samples its constant forward direction across
the viewport. Clip depth conventions and camera position leave sky direction
unchanged. The explicit overload accepts a rigid view and a rectilinear
projection without skew.

Three.js r187 likewise uses a viewport-filling flat texture with its own UV
transform and uses orientation and blur for directional textures. Its WebGL
renderer prepares PMREM automatically for blurred skies. Frostfield accepts
a caller-prepared `EnvironmentLighting` or the raw texture's mip chain.
Backgrounds are composed explicitly before the raster pass.
[Three.js r187 background renderer](https://raw.githubusercontent.com/mrdoob/three.js/r187/src/renderers/webgl/WebGLBackground.js)

## Verification

Xcode 27.2 passed 14 tests across `EnvironmentBackgroundTests`,
`CubeCameraTests`, `EnvironmentLightingTests`, and `LightRigTests` on arm64
macOS with Metal API Validation. Seven tests across `EnvironmentBackgroundTests`
and `CubeCameraTests` passed on iPhone 18 Pro Simulator. The background fixtures
check viewport borders, UV mirroring, sRGB input and output, straight and
premultiplied alpha, cube pitch, panorama roll and seam wrapping, fractional
raw mip levels, and a softened HDR hotspot from the filtered sky. They reject
texture feedback, invalid panorama dimensions, an invalid quaternion, and
blur on a raw texture with a single mip.
The explicit view fixture checks analytic asymmetric perspective sampling with
reversed infinite depth, parallel orthographic sampling, and pixel agreement
between the legacy camera and a matching explicit view in an inset viewport.

The Xcode package workspace open timed out, and the Lab workspace could not
create its build arena on the full external volume. These runs used
`xcodebuild` with DerivedData on the internal volume.
