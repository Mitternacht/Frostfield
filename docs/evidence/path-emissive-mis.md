# Emissive triangle sampling

Prepare a `PathTracingScene` with `.init(sampleEmissiveMeshes: true)` to add explicit mesh
lights to the path estimator. Check `sampledEmissiveTriangleCount` for the number of sampled
world-space triangles, including instances. Preparation defaults to the existing path-hit
estimator. Metal specializes the trace shader to omit emitter bindings and sampling when
the scene has no eligible emitters.

The distribution weights each eligible opaque PBR triangle by its transformed area and the linear
Rec. 709 luminance of its authored emissive factor. The sampler chooses a triangle, then draws a
uniform point on it. The density is expressed in solid angle at the scattering point.
Explicit light rays and BSDF or phase-function rays that hit the same triangle use
complementary power-heuristic weights. Primary hits keep their full authored emission.
Visibility uses the ordinary path shadow traversal, with a finite segment to the sample.
The renderer's one-sided and two-sided behavior also governs its emission.

Mapped emitters sample their emissive texture with authored UV set, transform,
wrapping, filtering and mips. Direct samples and scattering rays that hit the light
use the same outgoing ray footprint. The prepared scene reuses the existing direct
or baked material texture bindings. Texture luminance does not change selection
probability within or between triangles.

The immutable table uses the current traversal's instance and triangle IDs. It is rebuilt
on transform, geometry and live material updates. An encoded command keeps the previous
table and acceleration resources. A successful update invalidates existing accumulation
through the scene revision.

A scene samples up to 262,144 triangles in prepared instance and traversal order. Rows
with zero area or zero probability on the sampler's 24-bit grid are omitted. Surface
graphs, masked or stochastic coverage, written material alpha and unlit emission
retain their ordinary path-hit contribution, as do emissive boundaries of closed-volume
random walks. Texture importance within a triangle is outside this slice.

The pinned [three-gpu-pathtracer v0.0.28 README](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/README.md)
lists emissive materials without MIS. This capability extends that pinned behavior.

## Verification

- GPU tests compare two mirrored, nonuniformly scaled instances to equivalent analytic
  panels and independent BSDF-only tracing. They also check first-sample lighting,
  front and back faces, blockers, two-sided emission, and unchanged primary hits.
- GPU tests change instance area, replace geometry and edit live emissive strength.
  They compare a retained pre-edit command to the old scene, reject stale accumulation,
  and compare updated lighting to a fresh preparation.
- Two colored textures are compared to independent analytic panels and BSDF-only
  tracing across both texture paths and ray-tracing backends on Mac. An authored mip
  chain checks that direct lighting selects the scattering ray's footprint.
- Masked and subsurface surfaces retain exact path-hit pixels with the feature enabled.
- All 30 focused tests passed on Mac and iPad Pro (M5) Simulator with Metal API Validation.

The focused package run covers `PathEmissiveSamplingTests`, `PathSamplingTests`,
`PathGeneratedEnvironmentTests` and `PathBVHHeatmapTests` with Metal API validation.
Mac tests exercise both software traversal and Metal hardware ray tracing. The iPhone
18 Pro Simulator exercises software traversal and skips the hardware-only BVH check.
Xcode MCP's external DerivedData volume reported no free space, so Xcode command-line
tests used an isolated internal DerivedData path.
