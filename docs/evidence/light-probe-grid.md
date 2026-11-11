# World-space light probe grids

`LightProbeGrid` holds a `LightProbe` at the center of each cell in an
axis-aligned world volume. Author probes in X, Z, Y storage order or
capture them from a prepared scene. `irradiance(at:normal:)` returns
linear diffuse irradiance before the Lambertian BRDF. It normalizes the
world-space normal, moves the lookup half a cell along that normal, and
clamps the lookup to the outer probe centers. It blends all nine signed
coefficients trilinearly, then clamps reconstructed irradiance to zero
and applies grid intensity. The cell positions and lookup follow the
standalone Three.js r187 irradiance helper.
[LightProbeGrid](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/lighting/LightProbeGrid.js) ·
[LightProbeGridNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/lighting/LightProbeGridNode.js)

```swift
let bounds = try AxisAlignedBox(minimum: SIMD3(-4, 0, -4), maximum: SIMD3(4, 3, 4))
let grid = try await LightProbeGrid.capture(
    graph, bounds: bounds, resolution: SIMD3(6, 3, 6), commandQueue: queue, faceSize: 16
)
let floorIrradiance = try grid.irradiance(at: SIMD3(1, 0, 2), normal: SIMD3(0, 1, 0))
```

Capture uses the scene's authored materials and reuses one linear
RGBA16Float cube across all positions. Prepare drawable meshes for
single-sample HDR and forward depth as for `CubeCamera`. The same capture
layers, excluded subtrees, background and supplied shadow maps apply to
all cells. Each completed cube is read back and projected to SH before
the next capture. Capture limits are 4,096 probes and 16,777,216 total
cube texels. Authored grids can hold up to 65,536 probes.

With `ProgressiveLightmapBaker.accumulate(samples:indirectIrradiance:)`,
use the grid as the indirect term alongside the baker's direct lights.
Choose the capture scene so the grid represents the lighting intended
for that term.

```swift
try baker.accumulate(samples: 1) { sample in
    try grid.irradiance(at: SIMD3<Float>(sample.worldPosition),
                        normal: SIMD3<Float>(sample.worldNormal))
}
```

## Raster volume lighting

Prepare the volumes once on the render device and attach them to the
scene graph. Lit classic and PBR materials, including generated PBR
surfaces, sample local irradiance at each fragment's world position and
shading normal. This adds diffuse light alongside the graph's distant
probes, ambient lights and environment. The highest-priority containing
volume on a camera-visible layer wins. Ties go to the later entry in
the array. Containment includes the half-cell normal offset and a
one-percent cell margin used by r187.
[Three.js r187 WebGL registration and priority](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLRenderStates.js)

```swift
graph.lightProbeGridLighting = try LightProbeGridLighting(device: queue.device, volumes: [
    .init(roomGrid), .init(displayGrid, priority: 1)
])
```

The prepared set retains an immutable Float32 Metal buffer with signed
SH coefficients and evaluates trilinear interpolation in the fragment
shader. It supports up to 32 volumes with 65,536 probes in total.
Its CPU `irradiance(at:normal:cameraLayers:)` uses the same selection
rules. It returns zero outside the volumes. Prepare a new set when
the grids or their registration settings change.

Grid lighting belongs to raster scene graphs. Remove the binding before
archive capture and restore it with the prepared grids. Standalone mesh
draws and path tracing have separate lighting inputs.

## Ranged and iterative baking

`capture` renders one pass. `bake` adds up to eight diffuse feedback
passes. Each pass renders the original scene with the preceding completed
grid temporarily appended at the chosen priority and capture layers.
The new result replaces the preceding pass. The initial pass uses the
base scene's lights and volumes, which remain registered in later passes.
The caller supplies the
base scene without the destination grid already registered. The capture
count and cube texel budgets apply to the sum across passes.

```swift
let bounced = try await LightProbeGrid.bake(
    graph, bounds: bounds, resolution: SIMD3(6, 3, 6), commandQueue: queue,
    bounces: 2, bouncePriority: 1, faceSize: 16
)
```

To budget the work in smaller ranges, replace part of a grid with
`updating(probes:from:commandQueue:)`. Use the same completed previous
grid as `bounceSource` for every range in an indirect pass. Capture and
publication use separate immutable values, so a range completed earlier
does not change the source for a later range.

```swift
let previousPass = LightProbeGridLighting.Volume(grid, priority: 1)
let middle = grid.probes.count / 2
let first = try await grid.updating(
    probes: 0..<middle, from: graph, commandQueue: queue, bounceSource: previousPass
)
let nextPass = try await first.updating(
    probes: middle..<grid.probes.count, from: graph, commandQueue: queue, bounceSource: previousPass
)
```

The feedback uses diffuse raster materials and the existing native cube
projection. Each range keeps its probe order and retains cells outside
the requested range. The native API publishes a range only after its
captures complete. [Three.js r187 probe baking and bounce snapshots](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/lighting/LightProbeGrid.js)

## Verification

The numeric fixture uses eight cells with unequal world extents and
nonzero terms for all nine coefficients. The expected cell centers and
irradiance come from `three@0.187.0` `LightProbeGrid.getProbePosition`
and `SphericalHarmonics3.getIrradianceAt` with independently interpolated
coefficients. Additional fixtures cover the normal offset, singleton
axes, clamping outside the bounds, signed cancellation and invalid input.
The Metal fixture captures red and blue HDR surfaces from two positions,
then checks irradiance above 1, the distinct spatial response and the
midpoint blend. The seven focused suites passed all 22 tests on macOS
and iPhone 18 Pro Simulator with Metal API Validation enabled.

The raster checks compare CPU and rendered pixels across an eight-cell
grid, overlapping priorities, camera layers and positions outside the
volumes. The shadowed generated PBR fixture also samples a baked UV light
map in the same draw. The ten combined suites passed all 32 tests on
macOS and iPhone 18 Pro Simulator with Metal API Validation enabled.

A closed Lambertian chamber with albedo 0.5 produces radiance of 1,
1.5 and 1.75 across the initial pass and two bounces. Two ranged updates
using one prior snapshot exactly match the complete first bounce. The
fixtures also cover priority, capture layers, untouched cells and total
work limits. All 35 focused tests passed on macOS and iPhone 18 Pro
Simulator with Metal API Validation enabled.
