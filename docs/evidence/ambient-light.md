# Scene ambient light

Attach `AmbientLight` to a scene node for direction-independent diffuse
irradiance. The color is linear RGB from zero to one. Intensity is
nonnegative and can scale the light above display white. The defaults
are white and an intensity of one.

```swift
let fill = try AmbientLight(color: SIMD3(0.7, 0.8, 1), intensity: 0.35)
graph.roots.append(SceneNode(ambientLight: fill))
```

The renderer sums color times intensity across visible ambient nodes
whose layers intersect the active camera's layers. Node position and
rotation have no effect. The sum feeds the diffuse response of raster
PBR, native Lambert, and classic Lambert, Phong, and toon materials.
Shadowed materials retain the same ambient contribution. PBR and lit
classic materials add it to the evaluated hemisphere and other
spherical-harmonic probes. Probe reconstruction is clamped independently
before adding the ambient term. Environment lighting keeps its own
intensity and rotation. [Three.js r187 AmbientLight](https://github.com/mrdoob/three.js/blob/r187/src/lights/AmbientLight.js) ·
[r187 light accumulation](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLLights.js) ·
[r187 diffuse irradiance](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/lights_fragment_begin.glsl.js)

SceneArchive uses version 12 when a node carries an ambient light and
preserves its visibility and layers. Captures without ambient nodes
continue to use the earliest version required by their other features.
Ambient nodes are a raster lighting control. Path tracing takes its
surrounding radiance from the authored environment.

## Verification

All 71 focused tests across 14 suites passed on the M4 Max Mac and
iPad Pro M5 Simulator with Metal API Validation. RGBA16Float fixtures
check summed lights, rotated receivers, inherited visibility and camera
layers. The PBR fixtures measure the dielectric diffuse response, the
combined environment and hemisphere contribution, and a signed probe
that leaves ambient irradiance intact. Unlit and purely metallic
fixtures keep their baseline pixels. The archive tests preserve the
light values and rendered pixels and reject ambient records under a
legacy header.
