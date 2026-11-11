# Spherical-harmonic light probes

`LightProbe` stores nine linear RGB radiance coefficients in the
real L0–L2 basis used by Three.js `SphericalHarmonics3`. Author the
coefficients directly or project a mip of a native Metal cube.
`radiance(in:)` reconstructs radiance, and `irradiance(at:)` applies
the cosine convolution for diffuse lighting. Both normalize the
supplied direction and apply the probe's intensity. Reconstruction
can contain signed values at sharp transitions.

```swift
let probe = try await LightProbe(cube: capture.target, commandQueue: queue, mipLevel: 0)
let upwardIrradiance = try probe.irradiance(at: SIMD3(0, 1, 0))
```

Projection reads six faces in Metal's +X, −X, +Y, −Y, +Z, −Z order,
with top-left texel axes. It accepts linear RGBA8, sRGB RGBA8, RGBA16Float
and RGBA32Float. RGB stays linear and HDR values survive the readback.
The projection ignores alpha. It weights each texel center by its
approximate solid angle and normalizes the sum to 4π, following r187.
Submit cube writes on the same queue first and keep the source unchanged
while its faces are read. For a cube over 1,024 texels per side, select
a lower mip.

## Raster binding

Set `graph.lightProbes = [probe]` to add distant diffuse light to PBR
and classic Lambert, Phong and toon materials. Multiple probes sum
their intensity-scaled coefficients in world space. The renderer
evaluates diffuse irradiance at the fragment's shading normal, clamps
negative reconstruction lobes, and applies the material's diffuse
response. The probe term is independent of cube IBL and its scale
and yaw. It contributes diffuse energy and leaves reflections to
the existing environment maps.

## Scene archive

`SceneArchive` stores the ordered probe list, including signed
coefficients and intensity, in version 6 or later. Graphs without probes
continue to use the earliest version their other resources require.
Older archives restore with an empty probe list.

## Verification

The probe, projection, cube camera, texture transfer and environment
suites passed 19 tests each on macOS and iPad Pro M5 Simulator with
Metal API Validation. A private HDR cube captured by `CubeCamera`
reconstructs constant radiance and π times that radiance as diffuse
irradiance. All nine coefficients of an asymmetric cube and its
evaluated directions match the r187 generator after translating
WebGPU face axes. Separate fixtures check linear and sRGB SDR input.

The raster and existing shading suites passed 40 tests on each
destination with Metal API Validation. Rendered PBR and classic
surfaces respond to world-space normals, two half-strength probes
match one full-strength probe, and shadowed PBR receives additive
probe light alongside its cubemap IBL.

The archive suites passed 21 tests on each destination. A version 6
round trip preserves probe order, signed coefficients, intensities and
rendered pixels. The checks also load an older archive and reject a
legacy header with probe records, malformed coefficients and an
unrepresentable sum.

[Three.js r187 SphericalHarmonics3](https://github.com/mrdoob/three.js/blob/r187/src/math/SphericalHarmonics3.js) ·
[LightProbeGenerator](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/lights/LightProbeGenerator.js) ·
[Ramamoorthi and Hanrahan](https://graphics.stanford.edu/papers/envmap/envmap.pdf)

## Hemisphere lighting

Use `LightProbe.hemisphere` to add diffuse sky and ground fill to raster PBR
and lit classic materials. Supply linear RGB irradiance at each endpoint,
an intensity, and the world-space direction toward the sky. The default
axis is +Y. A rotated rig can pass its current world direction and replace
the probe without preparing any mesh again.

```swift
graph.lightProbes.append(try LightProbe.hemisphere(
    skyColor: SIMD3(0.8, 0.95, 1.2), groundColor: SIMD3(0.3, 0.16, 0.08), intensity: 0.5
))
```

The constant and linear spherical-harmonic bands reproduce the sky-to-ground
irradiance blend of Three.js r187. The light adds no shadow or specular
term. SceneArchive preserves the resulting coefficients and intensity.
[Three.js r187 hemisphere light](https://github.com/mrdoob/three.js/blob/r187/src/lights/HemisphereLight.js) ·
[Three.js r187 irradiance calculation](https://github.com/mrdoob/three.js/blob/r187/src/nodes/lighting/HemisphereLightNode.js)

CPU checks cover the sky, ground and horizon for a nonunit, non-axis-aligned
direction. Metal tests compare rotated PBR and Lambert surfaces to the
independent sky/ground blend for three directions. The four focused suites
passed all 13 tests on the M4 Max Mac and iPhone 18 Pro Simulator with
Metal API Validation enabled.
