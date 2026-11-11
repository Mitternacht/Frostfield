# Ground-projected skybox

Create an inward-facing skybox from a 2:1 equirectangular panorama. The
image uses Frostfield's top-left orientation, with +X at the center
and +Y along the top. The source format determines sRGB decoding, and
linear HDR values remain available when rendering to RGBA16Float.

```swift
let sky = try GroundedSkybox.makeNode(panorama: panorama, height: 3, radius: 100)
graph.roots.append(sky)
```

`height` is the camera height used to capture the panorama. The node's
default translation places its projected floor at world Y = 0. To
place it elsewhere, move the node so its local origin is `height`
above that floor. Rotate the node about Y to turn the panorama. Use
a radius that keeps the viewing camera inside, with a far clip that
includes the visible sky. The floor has flat triangles where the lower
rings reach the flat branch of the projection.

`GroundedSkybox.geometry` exposes the projected mesh separately. The
mesh applies r187's radial contraction with a curved transition into
the flat floor, while retaining each vertex's panorama UV. Native
tessellation uses inward normals and one UV pole vertex per wedge.
With the default draw order, the scene helper depth-tests after ordinary
opaque geometry and leaves depth unchanged. It draws an opaque raster
background and casts no shadows.
[Three.js r187 GroundedSkybox](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/GroundedSkybox.js)

The numeric fixture compares upper, curved and flat rings against the
pinned r187 constructor. Metal readback checks HDR orientation and yaw
against the native panorama background, and places a marker above and
below the floor with both forward and reversed depth. All 18 focused
tests passed on macOS and iPhone 18 Pro Simulator with Metal API
Validation enabled.

The pinned source file's SHA-256 is
`86e13051e05cb5beed632a377abc9eff13fd7cae10f1f5fb4d8cc41ce1cf2681`.
[Three.js MIT notice](../../LICENSES/Three-GroundedSkybox.txt)
