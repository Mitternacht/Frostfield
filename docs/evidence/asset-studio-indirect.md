# Asset Studio constant-proxy bake baseline

This page records the earlier constant-proxy capture. The current indirect
variant reads authored color and emissive textures.
[Current Asset Studio evidence](asset-studio.md)

The macOS capture route can render the rangefinder with a matched direct
reference and a diffuse bounce baked into the sweep. Both commands use the
same camera, floor geometry, UV0 atlas, six directional lights, and opaque
camera geometry.

```sh
FrostfieldLab --capture-asset-studio direct.png --studio-bake-reference
FrostfieldLab --capture-asset-studio indirect.png --studio-bake-indirect
```

The reference publishes the zero-sample snapshot from
`ProgressiveLightmapBaker`. The indirect variant adds 48 paths per covered
texel in three batches of 16, with one diffuse reflection and a fixed seed.
The 384×256 atlas limits each batch to at most 1,572,864 paths. The sweep
samples the resulting irradiance and converts it to reflected radiance
using its existing linear albedo divided by π.

The transport uses the actual placed opaque camera meshes with a constant matte
linear RGB proxy of (0.18, 0.12, 0.075). The rendered camera uses its
original textured PBR finishes, including the transmissive lens. The
sweep acts as the receiver, and the six directional lights supply all
energy for the bake. The measured addition comes from one diffuse
reflection off the camera proxy.
[Asset Studio fixture](asset-studio.md) · [Progressive baker](static-lightmap.md)

## Verification

Xcode 27.2 built the Lab for macOS and iPad Pro 13-inch (M5) Simulator.
Both 1440×960 macOS captures completed with Metal API validation. The
default capture matched the previously verified Asset Studio PNG byte
for byte.

The direct and indirect textures from the same baker differed at
15,056 of 66,816 covered floor texels. In scene-linear RGB radiance the
mean addition over the measured floor was (0.0000521, 0.0000570,
0.0000416), and the componentwise peak was (0.00366, 0.00337, 0.00269).
The two PNGs differ at 7,050 pixels near the subject and contact shadow.
The indirect capture took 36.4 seconds in the Debug build on the Mac used
for verification. The integrated build reproduced the texel counts, radiance
measurements, and indirect PNG byte for byte.

The Xcode 27.2 build at `b41d51a` repeated the indirect capture with Metal
API Validation and matched the previous integrated PNG byte for byte.
The covered texel counts and scene-linear additions also matched.
