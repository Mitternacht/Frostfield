# Path-traced media and translucent materials

`PhysicallyBasedMaterial.DiffuseTransmission` transmits diffuse light from
the back of a thin sheet, or enters a closed mesh volume. It takes its share
from diffuse reflection. Its textures scale the factor by alpha and tint it
by RGB. `DiffuseRoughness` changes diffuse reflection from Lambertian to the
energy-preserving Oren–Nayar model. Both renderers evaluate that model for
direct light. The raster environment approximation uses its directional
albedo, exact under uniform illumination.
[EON paper](https://jcgt.org/published/0014/01/06/)

`VolumeScatter` adds RGB scattering and Henyey–Greenstein anisotropy to
a transmissive volume. Its extinction comes from the existing attenuation
color and distance. `Subsurface` uses a color, an RGB mean free path
and anisotropy for a random walk inside a closed outward-facing mesh.
The path tracer samples free paths per channel and weights them by their
mixture probability. Raster uses wrapped diffuse light and a thickness
approximation. A zero subsurface radius uses the local diffuse lobe.
Path tracing also samples clearcoat normal maps in the coat's own tangent
frame, independently of the base normal.

`PathTracer.Fog` defines a homogeneous axis-aligned world box with RGB
extinction and single-scattering albedo. Direct light and shadow rays
account for extinction in the box. The innermost closed material volume
takes precedence over the box. Changing fog invalidates accumulation
history, as a lighting change does.

## Verification

Xcode 27.2 on a Mac with Apple M4 Max and an iPad Pro 13-inch (M5) simulator
running iPadOS 27.2.

- `PathMediaTests` checks white-furnace conservation, Chandrasekhar
  plane albedo, slab reflectance and transmittance against an independent
  Monte Carlo reference, and Henyey–Greenstein phase sampling. It also checks
  Beer–Lambert attenuation through nested index-matched volumes and fog.
  Single scattering and occlusion agree with analytic radiance within 3%
  in the light-shaft fixture.
- `TranslucencyTests` compares thin diffuse transmission and rough diffuse
  reflection with analytic values in both renderers, verifies the raster
  volume approximation, and traces the clearcoat normal with both
  full-resolution and baked material maps.
- `GLTFTranslucencyTests` verifies material extension preparation, texture
  coordinates and transforms, round-trip export of imported and prepared
  materials in both glTF and GLB, and loss reporting for native subsurface.
  [Diffuse transmission extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_diffuse_transmission/README.md)
- The iPad simulator passed the 15 focused tests in these three suites
  before the export round-trip follow-up. The initial Mac run passed
  25 focused tests across media, glTF, AOVs, camera projection and analytic lights.
  The Mac export follow-up passed 14 tests across translucency and existing
  glTF exporter and transmission suites with Metal API validation. The iPad
  simulator then passed all eight path-media tests with nested-volume shadows.

## Raster fog presets

`PostProcessGraph.exponentialDepthFog` applies exponential-squared fog
by perspective view depth. `exponentialHeightFog` scales that depth by
the distance below an authored world Y threshold, reconstructed from
perspective depth and the camera pose. Both preserve premultiplied alpha
and support forward and reversed depth. The presets use the formulas
from [Three.js r187 fog nodes](https://github.com/mrdoob/three.js/blob/r187/src/nodes/fog/Fog.js).
The depth source is the opaque depth buffer, so blended surfaces use the
depth of the surfaces behind them. The existing range-fog preset retains
its opaque fog color blend.

`PostProcessGraphTests` passed 15 tests on macOS 27.2 with Metal API
validation. The new check compares GPU output with an inverse-projection
reference using a translated and rotated camera, near and far samples,
and opaque, translucent and transparent source pixels. The iPad simulator
passed 19 tests across post-process graphs and glTF translucency.

## Path pipeline preparation

Guide and AOV variants prepare on first use. The Lab prepares its guides
during fixture setup through `PathTracer.prepareGuides()`. PathTracer specializes
the shadow kernel against the material snapshot's index-matched boundary flag. The renderer
selects the cached variant on every encode so live material changes can switch
it. Fog and subsurface settings still use the ordinary per-encode inputs.

The M4 Max comparison used Xcode 27.2 and Metal API Validation. Each run used
the same coarse-mipmap graph fixture with three texture addressing modes.
Temporary entry-point names separated the experiment's Metal pipeline cache
keys. Times cover the test, with the package build excluded.

| Configuration | Cold color pipeline per graph | Test elapsed | One-minute host load, start to end |
| --- | ---: | ---: | ---: |
| Eager guide and dynamic boundary flag | 15.22–16.70 s | 106.78 s | 10.03–10.45 |
| Guide on first use | 13.51–13.68 s | 43.18 s | 7.58–7.00 |
| Guide on first use and boundary specialization pilot | 7.72–8.40 s | 27.05 s | 8.31–8.77 |

The eager run spent 53.68 s preparing three guide pipelines the fixture did
not use. Instruments attributed 99.7% of the sampled compiler-service time
to backend code generation. An outline experiment reduced compile time
further but increased the small direct-light fixture from 0.66 ms to 1.18 ms
median GPU dispatch time. Boundary specialization gave 0.67 ms in that
fixture and 38.99 ms in the tiny-radius subsurface walk versus 40.04 ms at
baseline. These measurements describe the fixture workloads on this Mac.

`PathMediaTests.liveShadowBoundaryVariant` toggles the eligibility of a
transmitting surface and uses one tracer throughout. The color pipeline and
the guide-plus-AOV pipeline both switch between a blocked ray and the
analytically attenuated ray. Xcode reported 21 focused tests on macOS and
16 on iPhone Simulator with Metal API Validation. Xcode MCP could not open
the package and the Lab build arena was on a full drive, so verification used
xcodebuild with internal DerivedData.

## Limits

Random walks use up to four nested closed media and stop after 256 internal
events per walk. Direct light sampling inside a closed volume requires an
index-matched boundary. Thickness in the path tracer comes from intersected
geometry. Raster thickness remains an author-supplied approximation. Fog
supports a single homogeneous box. Raster volume ray marching, per-material fog,
heterogeneous density, and per-pixel subsurface radius remain open.
The glTF writer preserves the supported draft diffuse roughness and
volume scatter records and reports loss when exporting native subsurface.
First-hit render layers describe geometric surfaces, including through fog.
Physical-device qualification and matched performance measurements remain
separate.
