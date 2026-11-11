# Box-projected environment reflections

Bind a prepared environment to a world-aligned room proxy through a
PBR material. Capture the source at the same world position used for
the projection.

```swift
let room = try AxisAlignedBox(minimum: SIMD3(4, 1, -12), maximum: SIMD3(12, 7, -2))
let projection = try EnvironmentBoxProjection(bounds: room, capturePosition: SIMD3(7, 4, -7))
let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting, boxProjection: projection)
let metal = try PhysicallyBasedMaterial(metallic: 1, roughness: 0.15, environment: environment)
```

For a reflected ray starting inside the room, the lookup direction
runs from the capture point to the ray's first exit. This follows the
r187 helper for a centered capture and supports a capture point offset
within the proxy. Parallel rays skip their parallel axes. Fragments
outside the proxy use the ordinary direction. The environment's yaw
applies after the world-space correction.
[Three.js r187 parallax helper](https://github.com/mrdoob/three.js/blob/r187/src/nodes/functions/material/getParallaxCorrectNormal.js) ·
[Three.js r187 box projection example](https://github.com/mrdoob/three.js/blob/r187/examples/webgpu_materials_envmaps_bpcem.html)

The correction applies to raster PBR specular and clearcoat. Diffuse
irradiance keeps its normal-based lookup. The default environment
binding retains the original pixels. The projection is a fixed
material binding and can be shared across receiving meshes. Replace
the binding when the room or capture point moves. Treat the prepared
lighting as read-only and keep the material and its textures alive
until completion for unretained command buffers.

The path tracer and screen-space fallback use their existing environment
inputs. SceneArchive rejects the box-projected binding during capture so
the proxy settings stay explicit in the application.

A Metal fixture captures red and blue wall patches from an off-center
position in a translated room. The ordinary mirror samples blue and
the projected mirror samples the red wall it faces. Rendering at the
capture origin, outside the box, and with the default binding confirms
the corresponding ordinary pixels. The numeric tests cover translated
bounds, parallel rays, and a small capture offset in a wide box.

The 31-test focused run passed on macOS and iPhone 18 Pro Simulator with
Metal API Validation enabled. The simulator skipped the hardware
ray-tracing regression according to its capability check.
