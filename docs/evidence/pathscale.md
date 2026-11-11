# Path tracing texture detail and acceleration

The path tracer samples eligible material and shader graph textures directly at
their source resolution. The texture table keeps each map's mip chain and
format, and material bindings describe the sampler. Anisotropic ray cones
select the sampling footprint at each hit. `PathTracingScene.Options` controls
the texture mode, memory budget, and maximum anisotropy. The explicit
`PathTracingScene(queue:graph:textureSize:)` initializer still bakes every map
at the requested size. Use `PathTracingScene(queue:graph:)` for full-resolution
sampling with a baked fallback for maps that the GPU cannot sample directly.

The scene retains the directly sampled textures and sampler states. Prepare
their contents before tracing, or order their writes on the tracer's queue
with hazard tracking. Pass nil history after changing a texture's contents so
that the accumulation starts with the new material data. Preparation checks
the resident texture budget against the source allocations and the actual
baked mip array, including Metal's allocation padding. When Metal omits
allocation bytes, the baked array reports its logical RGBA16Float mip size.
When the device gives no recommended working set, the default budget is
64 MiB. A caller-specified budget takes precedence.

With `Options.Acceleration.automatic`, the scene selects Metal acceleration
structures when the device reports ray-tracing support, and the watertight
software BVH otherwise. The Metal backend compacts bottom levels and reuses
them across instances. It builds or refits the top level on a dedicated queue
and publishes the new snapshot after a successful build. A previously encoded
trace keeps its original resources until GPU completion. The scene's revision
changes only when an update commits. Asynchronous preparation can build
bottom levels off the caller's executor and reports progress and cancellation.

## Verification

Xcode 27.2 (27B52) with Metal API Validation on November 10, 2026.
Xcode MCP opened the Lab project, but its DerivedData volume was full and
opening the standalone package timed out. The test runs used `xcodebuild`
with an internal DerivedData path.

| Destination | Result |
| --- | --- |
| Apple M4 Max, macOS 27.2 (26B123) | 464 tests in 101 suites passed on the acceleration and texture compatibility changes. Automatic selection used Metal. |
| Apple M4 Max, macOS 27.2 (26B123) | Six full-resolution texture tests passed after the texture budget change, including one-byte-short and exact-budget checks. |
| iPhone 18 Pro Simulator, iOS 27.2 | Four acceleration tests passed. Automatic selection used the software BVH, and forcing Metal returned `rayTracingUnsupported`. |
| iPhone 18 Pro Simulator, iOS 27.2 | The device reported Tier 1 argument buffers and no working-set recommendation. The default bake matched the explicit baked render. The exact-budget and one-byte-short checks passed. Six direct-sampling tests require Tier 2 and ran on the Mac. |

- `PathFullResolutionTextureTests` compares magnified 2,048² sRGB glTF
  textures with raster at a maximum linear channel difference below 1/255.
  It checks grazing views of a 4,096² mipmapped checker against a supersampled
  reference and exercises all 4,096 texture-table entries.
- `PathAccelerationTests` traces 1,048,576 unique triangles with each backend
  and refits a Metal scene with 1,048,576 instances. A matched 96 × 64 studio
  image at 32 samples per pixel had a mean maximum channel difference of
  0.0000871 between the two backends. 6,001 of 6,144 pixels were within 0.001.
- The same suite covers watertight shared edges, subnormal BVH bounds, updates
  after a trace was enqueued, release of replaced buffers, empty instance sets,
  and rejection of changes to an instance set's structure.

The opt-in `PathScaleBenchmarkTests` fixture traces a procedural terrain at
1280 × 720 with roughly 100,000, 1,000,000, and 5,000,000 triangles. Run it
in Release and read the `PATHSCALE` output for separate preparation and GPU
trace times.

```sh
TEST_RUNNER_FROSTFIELD_PATH_BENCHMARK=1 xcodebuild test -configuration Release \
  -scheme Frostfield-Package -destination 'platform=macOS,arch=arm64' \
  -only-testing:FrostfieldTests/PathScaleBenchmarkTests
```

Sustained performance on physical iPhone, iPad, and Apple Vision Pro and a
matched comparison against Three.js remain open.
