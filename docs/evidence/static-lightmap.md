# Static and progressive lightmap baking

`MeshLightmapBaker` bakes linear irradiance into a receiver's authored
UV0 or UV1 atlas. It reconstructs each covered world-space surface
point and normal and accumulates positive cosine-weighted directional
irradiance. Opaque caster geometry goes into a mesh BVH. An unblocked
shadow ray contributes the light, and a blocked ray contributes zero.
The receiver can cast onto itself. Both the receiver and casters
accept affine world transforms, including mirrors.

The bake returns an RGBA16Float irradiance texture and an R8 coverage
texture. It shares texel ownership, overlap policies, and edge padding
with the attribute baker. [UV atlas contract](mesh-texture-baking.md)

## Using the result

For diffuse radiance from a Lambertian material, multiply irradiance by
linear albedo divided by π. This example renders baked diffuse light with
a constant albedo using the PBR emissive channel.

```swift
let lamp = try MeshLightmapBaker.DirectionalLight(
    directionToLight: SIMD3(0, 1, 1),
    irradiance: SIMD3(2.5, 2, 1.5)
)
let map = try MeshLightmapBaker(
    device: device,
    geometry: receiver,
    width: 512,
    height: 512,
    lights: [lamp],
    casters: [.init(caster, transform: casterToWorld)],
    padding: 4
)
let material = try PhysicallyBasedMaterial(
    lightIntensity: 0,
    ambientIntensity: 0,
    baseColorFactor: SIMD4(0, 0, 0, 1),
    emissiveTexture: .init(
        texture: map.irradianceTexture,
        textureCoordinateSet: map.textureCoordinateSet
    ),
    emissiveFactor: albedo / .pi,
    alphaMode: .opaque
)
```

## Progressive diffuse transport

`ProgressiveLightmapBaker` prepares a receiver atlas and a static mesh BVH.
It traces cosine-weighted paths from covered texels and accumulates
linear irradiance. It evaluates shadowed directional, point, spot and
rectangle lights at receivers and reflected surfaces, and samples constant
sky radiance and front-facing emissive geometry. Opaque geometry occludes
paths from both sides. Bounce surfaces use their geometric normals and
Lambertian albedo.

```swift
let bake = try ProgressiveLightmapBaker(
    device: device,
    receiver: .init(geometry: receiver),
    width: 256,
    height: 256,
    lights: [lamp],
    surfaces: [
        .init(geometry: wall, worldFromObject: wallToWorld, diffuseAlbedo: wallAlbedo)
    ],
    maximumBounces: 2,
    padding: 4
)
try bake.accumulate(samples: 16)
let preview = try bake.snapshot()
let matte = try ClassicMaterial(model: .basic, color: albedo, lightMap: preview.lightMapBinding())
try bake.accumulate(samples: 16)
let refined = try bake.snapshot()
```

A transport surface can use independent diffuse and emissive maps. A
`ProgressiveLightmapBaker.TextureSnapshot` captures a
`PhysicallyBasedMaterial.TextureBinding` at level zero and waits for Metal
to read it. Supply the producer command queue or finish any GPU writes
before capture. Reuse a snapshot across surfaces that share its texture and
UV settings. The snapshot keeps linear RGBA, including sRGB-decoded color
and HDR emission, with up to 8,388,608 distinct base-level texels per bake.
[Metal texture color conversion](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Mem-Obj/Mem-Obj.html)

At a hit, the baker interpolates the geometry's UV0 or UV1 in top-left
coordinates. Each map applies its own transform, addressing modes and
nearest or bilinear magnification filter. Diffuse and emission samples
multiply their supplied linear factors. Diffuse reflectance clamps to one
and emission must stay within the supported HDR range. Back faces
occlude paths at covered hits.

Set `Surface.usesVertexColors` to multiply diffuse reflectance by linearly
interpolated vertex RGB as well. Vertices without a color contribute white,
and the final reflectance stays in 0...1. This opt-in applies at bounce hits.
Alpha coverage still uses `AlphaMask`, and receiver irradiance stays separate
from receiver albedo. The vertex-color fixture matches an equivalent
bilinear texture gradient and checks multiplication by an independent map.
It passed with the bake and BVH suites on Mac and iPad Simulator, 64 tests
each under Metal API Validation.

### Alpha-tested transport

`ProgressiveLightmapBaker.Surface` accepts an optional `AlphaMask`. It multiplies
the authored opacity, interpolated vertex alpha and sampled texture alpha,
then keeps samples at or above the cutoff. Each mask can share the diffuse
snapshot or use its own UV set, transform and sampler. For glTF-style
base-color cutouts, pass the base-color factor's alpha and the same texture
snapshot used for diffuse color.

```swift
let map = try ProgressiveLightmapBaker.TextureSnapshot(baseColorBinding)
let leaves = try ProgressiveLightmapBaker.Surface(
    geometry: leafMesh, worldFromObject: leafPlacement, diffuseAlbedo: leafTint,
    diffuseTexture: map, alphaMask: .init(cutoff: 0.5, texture: map)
)
```

Directional, point, spot and rectangle shadow rays continue through holes.
Diffuse paths can reach the next covered surface or the sky. The receiver
rasterizer uses the original authored coordinates even when the display mesh's
UV1 is repacked. It excludes cutouts from interior coverage before gutter
padding. The textured Asset Studio bake passes PBR mask parameters to this
API. Translucent blending and transmission require a different transport
model. [Three.js r187 alpha test](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/alphatest_fragment.glsl.js)

Snapshots contain fresh RGBA16Float irradiance textures and share a read-only
coverage texture. The binding uses the bake's UV set and clamps atlas edges.
Sample sequences use the absolute sample index and texel index, so splitting
the same work across batches preserves the result. Before the first batch,
a snapshot contains the direct lighting term.

Point, spot and rectangle lights use the native `PunctualLight` color and
intensity, with a world pose supplied to `ProgressiveLightmapBaker.PositionedLight`.
A light resolved from a scene node can be passed directly to that initializer.
Use the `punctualLights` list with either atlas initializer. It shares the
16-light budget with the directional list.

The CPU bake uses the raster model's inverse-square attenuation, quartic
range window and squared spot cone ramp. Falloff uses the surface position.
Visibility uses a finite ray from the biased origin to just before the light. Every source uses the bake's own shadow bias. For a positioned
light, the offset is limited to half the distance to the light.

Point and spot IES profiles multiply intensity by the angular sample of
the same normalized half-float table used in the renderers. Spot cookies
multiply the light by linear RGB transmittance. A resolved light keeps its
node roll for both effects. Cookies use the renderer's top-left orientation
and bilinear filtering with clamped edges, including resampling to the
light list's largest cookie dimensions and RGBA16Float texel conversion.
The baker retains the immutable CPU inputs and samples those texels on
demand. The native light constructors still define supported kinds and
resources. IES profiles require supported LM-63 data. `TILT=INCLUDE` and a supplied tilt document are kept on `lampTilt` and do not scale the measured candela.
and cookies require a spot with an outer angle of at most 89°.

Rectangles emit uniformly from their front along local −Z. The bake keeps
resolved aim and roll and uses authored width and height in world units.
Luminance is weighted by sampled area, both surface cosines and inverse
squared distance. The same calculation applies to the initial receiver
and later diffuse hits. Each uses a deterministic 8×8 grid of cell centers
with up to 64 finite shadow rays per rectangle. More progressive
samples refine the bounce paths while the rectangle grid stays fixed.
Very close surfaces and thin blockers can expose quadrature error. Light
rectangles are analytic emitters. Supply a separate opaque surface for
the fixture body if it should block rays.

### Relighting a prepared atlas

Call `replaceLighting(lights:punctualLights:skyRadiance:)` to change the
lighting over fixed prepared geometry. Each call supplies the complete
new light lists and sky. Empty lists and black sky are the defaults.
The baker checks the inputs and computes the new direct term before
installing it. A successful call reuses the atlas, receiver bindings,
coverage and BVH, and restarts diffuse accumulation at sample zero with
the original seed. A failed call leaves the current result and sample
sequence available. Previously published textures retain their pixels.

### Moving transport surfaces

Pass the complete list of extra surfaces to `replaceSurfaces(_:)` after
moving a caster, changing its mesh or material, or adding a new one. The
baker retains the receiver atlas and coverage. It refits the transport BVH
when the merged vertex count and index sequence match, and otherwise
builds a new hierarchy. Face normals, material maps and cutouts follow the
updated surfaces. Pass an empty array to remove all extra geometry.

Each successful edit recalculates direct lighting and restarts the
progressive sequence. A rejected edit preserves the previous direct term,
progressive samples and published textures.

Xcode 27.2 with Metal API Validation passed 61 focused tests on Mac and
iPad Simulator. The moved caster and emitter matched fresh bakes on both
authored and packed atlases, including split sample batches. Tests verified
the old snapshot and the shared coverage texture, and checked atomic
rejection of invalid geometry, excessive surfaces and a move exposing
irradiance beyond the output range. The Xcode bridge timed out resolving
the destination, so the tests used `xcodebuild` with internal caches.

### Moving receivers

Pass one affine transform per receiver to `replaceReceiverPlacements(_:)`.
The baker reuses local geometry, the atlas and UV charts, including chart
splits. It reconstructs the receiver points and transport geometry in their
new world placements. The returned bindings carry the new transforms.
Previous bindings and texture snapshots retain their values.

Xcode 27.2 passed 67 focused tests on Mac and iPad Simulator with Metal
API Validation. The fixture translates and mirrors both an authored and a
packed receiver. Their direct shadows and progressive results match fresh
bakes with the same charts. Invalid placement count and singular transforms
leave the previous bake and sample count intact.

### Rolling previews

Call `updatePreview(samples:blendWindow:)` after updating the light rig or
extra surfaces. Each update estimates direct and diffuse light, then blends
it into the previous preview with weight `1 / blendWindow`. The first update
uses the current estimate at full weight. A window of one shows each update
immediately. `resetPreview()` clears this history and its sample sequence.

This is the camera-independent counterpart to the Three.js progressive
lightmap's rolling atlas. Frostfield evaluates world-space visibility at
each UV receiver point. The preview and the cumulative bake have independent
sample sequences, so a moving preview can persist through a light or caster
edit while `accumulate` restarts for the new scene. Returned textures are
immutable snapshots sharing the receiver coverage.
[Three.js r187 progressive lightmap](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/misc/ProgressiveLightMap.js)

The rolling preview passed 63 focused tests on Mac and iPad Simulator with
Metal API Validation. The fixtures verify fractional color and shadow changes,
immediate updates, independent cumulative history and coverage, and a
deterministic two-update average against a cumulative bake. Invalid preview
settings preserve the sample sequence.

### Caller-defined irradiance

Use `accumulate(samples:sampling:indirectIrradiance:)` when the indirect light comes
from caller data such as a CPU probe grid. The closure receives the world
position, unit shading normal and absolute sample index. Return linear
RGB irradiance. The bake combines this estimate with its own direct lights,
without applying receiver color or exposure. Choose `.texelCenter` to reuse
the prepared coverage or `.halton` to re-evaluate covered world positions
at a deterministic subtexel offset for each sample. The Halton raster uses
the inverse of the reference projected-geometry shift in top-left UV space.
Each texel divides by its own number of covered samples. Jittered snapshots
publish their own coverage and padding, which can grow as edge samples arrive.
Call `reset()` before switching between centered and jittered accumulation.
The receiver atlas must cover at least one texel center at preparation.
The closure evaluates on the CPU. For Metal-authored sources, pass a
`ShaderBufferKernel` to `accumulate(samples:sampling:irradianceKernel:uniforms:textures:commandQueue:)`.
The graph receives position and sample index in field 0, world normal in
field 1, and writes irradiance to field 2. It can sample named textures and
use live uniforms. The baker submits one compute batch per sample, reads
back the irradiance, and normalizes and pads it through the same atlas path.
Use the texture producer's queue so tracked writes precede the compute read.
[Three.js r187 LightMapper](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/lighting/LightMapper.js)

An analytic fixture adds directional light to a position-, normal- and
sample-dependent source. Split and single batches match. Throwing from the
source or producing invalid or unrepresentable irradiance leaves all prior
samples and snapshots intact. Mac and iPad Simulator each passed 65 focused
tests with Metal API Validation.

The Halton fixture compares the first three reference positions on a two-texel
chart. It verifies distinct sample counts, a padded texel becoming covered,
split-batch identity, rollback and old coverage lifetime. Mac and iPad
Simulator each passed 66 focused tests with Metal API Validation.

The Metal fixture compares a shader that reads position, normal, sample
index, a texture and a uniform against the CPU result in both sampling
modes. Wrong layouts, missing textures and nonfinite shader outputs leave
the bake unchanged. Mac and iPad Simulator each passed 83 focused tests
with Metal API Validation. The bridge timed out resolving the workspace
destination, so the tests used `xcodebuild`.

### Shared receiver atlas

Pass multiple receiver surfaces to pack their world-space charts into
one square UV1 atlas. Each receiver casts shadows, reflects and emits
light in the common scene. Extra surfaces contribute to transport
without taking up atlas space.

```swift
let shared = try ProgressiveLightmapBaker(
    device: device, receivers: [first, second], resolution: 128,
    lights: [lamp], maximumBounces: 2, padding: 4
)
try shared.accumulate(samples: 16)
let lightMap = try shared.snapshot().lightMapBinding()
let firstMesh = shared.receiverBindings[0].geometry
let secondMesh = shared.receiverBindings[1].geometry
```

Each returned mesh uses the shared binding and its own original placement.
Packed meshes preserve UV0, triangle order, winding, and material groups.
Use the source-vertex indices to remap external per-vertex attributes across
chart splits. The transport BVH uses the input mesh coordinates, including
authored UV1 material maps. The returned display mesh uses UV1 for the new
lightmap, so a display material with an existing UV1 map needs its own
available coordinate set or a remapped texture. The shared atlas supports
square resolutions from 16 to 512 texels per side.
[Three.js r187 progressive lightmap](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/misc/ProgressiveLightMap.js)

The CPU bake accepts up to 262,144 texels, 1,024 surfaces including the
receivers, 262,144 triangles, and 524,288 vertices. Each batch is limited to
256 samples per covered texel and 4,194,304 starting paths. A history can
accumulate up to 65,536 samples per texel with up to eight diffuse reflections
per path. Texture LOD, shader graphs, metallic-roughness
and normal maps, specular and transmissive transport, and environment maps
remain open. [Diffuse transport reference](https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection)

## Verification

Xcode 27.2 reported 20 focused macOS tests with Metal API validation in
`MeshLightmapBakerTests`, `MeshTextureBakerTests` and `MeshBVHTests`,
and 10 baker tests on the iPad Pro 13-inch (M5) simulator. The lightmap tests compare known directional irradiance in lit,
blocked, and caster-free texels. They cover self-shadows, multiple
directions, light behind a receiver, and a nonuniform reflected world
transform. The material fixture compares the unoccluded baked UV1 map
to direct Lambertian rendering within 0.001 per channel and checks
the baked shadow with the caster absent from the rendered scene.

`ProgressiveLightmapBakerTests` compares split and single batches bit for
bit, verifies retained snapshots and padded coverage, and checks red
color transfer from white directional light with and without a blocker.
It exercises a shadowed fixture that receives light only after two
reflections, analytically checks constant sky and front-facing emission,
and renders the baked UV1 lightmap through `ClassicMaterial` with Metal.
Xcode 27.2 passed 27 macOS tests across the two lightmap suites, the texture
baker and the BVH, and 11 lightmap tests on the iPad Pro 13-inch (M5) iOS 27.2
simulator. Both runs enabled Metal API Validation.

The textured bake, existing lightmap, BVH and scene picking suites passed 30
tests each on macOS and the iPad Pro M5 simulator with Metal API Validation.
The fixtures verify sRGB color transfer, spatial UV1 hits, repeat and mirror
addressing, level-zero HDR emission, and independent transforms for maps
that share a private texture written on the supplied queue. The same texels
matched constant analytic references and were stable across batch boundaries.

The shared-atlas and existing bake suites passed 38 focused tests each on
macOS and the iPad Pro M5 simulator with Metal API Validation. Two facing
receivers with independently transformed meshes and authored UV1 albedo
maps occupy disjoint interior texels. They block opposing direct lights
throughout the covered atlas, exchange the expected red and blue indirect
light, and render through the same lightmap binding. A separate fixture
checks local vertex attributes, groups and winding after chart splits.

The punctual bake, existing bake and mesh-picking suites, native punctual
lighting and light rigs passed 58 tests each on macOS and the iPad Pro M5
simulator with Metal API Validation. The punctual fixtures check inverse
square and cosine at known positions, range cutoff, spot penumbra and aim,
blockers before and beyond a light, and a spotlight that reaches the
receiver after a diffuse reflection.

The relighting and existing bake suites passed 45 tests each on macOS
and the iPad Pro M5 simulator with Metal API Validation. The relit and
fresh shared atlases match bit for bit before and after sampling, including
split sample batches. Tests also verify clearing the lights, retaining
published texture pixels, and continuing the same sequence after rejected
light counts, sky values or an out-of-range direct term.

The photometric bake, light rig, renderer and archive suites passed 48
tests each on macOS and the iPad Pro M5 simulator with Metal API Validation.
The bake fixtures check IES angles and seam interpolation, resolved roll,
cookie quadrants and mixed-size filtering, and exact profile-times-cookie
color scaling on a diffuse bounce after relighting.

The rectangle bake and existing bake and native lighting suites passed 47
tests each on macOS and the iPad Pro M5 simulator with Metal API Validation.
The rectangle fixtures check irradiance against a closed-form integral,
including changes in size, distance and roll, and verify one-sided emission.
A symmetric partial blocker halves irradiance. Opaque geometry beyond the
emitter leaves it unchanged. The diffuse bounce fixture checks the panel's
reflected color and verifies that relighting matches a fresh bake across
split batches, including after a rejected over-range replacement.

Mac and iPad Pro M5 Simulator each passed 51 focused tests under Metal API
Validation for alpha-tested transport and existing baker and BVH behavior.
Fixtures confirm all four light kinds respect an sRGB texture's independent
alpha, a missed cutout reveals an emissive surface, and packed receiver
coverage uses authored UV1, a repeat transform and vertex alpha. Xcode MCP
timed out resolving the package destination, so the tests ran through
headless `xcodebuild` with internal DerivedData.
