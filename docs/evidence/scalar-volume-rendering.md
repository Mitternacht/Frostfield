# Scalar volume rendering

`ScalarVolumeRenderer` renders NRRD data or a shader-readable R16Float
or R32Float 3D texture. The NRRD initializer converts scalars to Float32
and keeps the original voxel-to-space affine. Scene placement composes
with that affine. The pass intersects each camera ray with the physical
volume, including oblique axes and half-voxel boundaries. Rays use the
same projection matrices as `ShaderGraphCamera` and support orthographic,
finite perspective, and infinite perspective views.

Maximum-intensity projection searches the clipped ray and refines its
strongest sample. Values below the scalar window leave the target
untouched. Isosurface mode finds the first threshold crossing and
refines the bracket, using physical-space gradients for headlight
shading. The scalar window selects colors from an existing
`ScalarColorLookupTable`. The caller can choose nearest or trilinear
reconstruction and bound the coarse step count. Trilinear reconstruction
uses explicit voxel reads so R32Float works on devices without hardware
Float32 filtering.
[Three.js r187 VolumeShader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/VolumeShader.js)
[Three.js r187 NRRD volume example](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_loader_nrrd.html)

The pass writes premultiplied source-over color on the caller's command
buffer. It loads existing color or clears it when the caller supplies a
clear color. A same-size shader-readable Depth32Float image can shorten
each ray in forward or reversed depth. The pass leaves opaque depth
untouched. Separate volume passes composite in caller order. Per-sample
absorption and emission, cloud scattering, and order-independent
composition are outside this pass.

## Verification

Metal readback checks NRRD-backed MIP and isosurfaces, anisotropic and
sheared bounds, placement, nearest and trilinear reconstruction, finite
and infinite projection, loaded-target blending, scalar windows,
threshold clipping, and world-space gradient lighting. Opaque-depth
cases distinguish a visible lower maximum from a brighter hidden voxel
and an isosurface hidden before its first crossing.

The focused run reported 30 tests in five suites on macOS and iPad
Simulator, with one existing native BC test skipped on the simulator.
All six volume-rendering tests passed on both targets.

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API validation.
