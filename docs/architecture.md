# Frostfield architecture

The `Frostfield` target owns scene data and Metal rendering. The optional
`FrostfieldSwiftUI` target wraps it in a native Metal view.

## Boundaries

Frostfield's renderer will turn application-supplied scene intent into Metal work. The
core stays independent of AppKit, UIKit, SwiftUI, and application view update
frequency. Each FrostfieldLab host supplies the drawable and render pass to the
same package renderer and shader.

| Platform | Lab presentation |
|---|---|
| macOS | AppKit Metal layer in a SwiftUI window |
| iOS and iPadOS | UIKit Metal layer in a UIKit window scene |
| visionOS | UIKit Metal layer in a native SwiftUI window, with a separate immersive Compositor Services fixture |

The immersive fixture uses compositor frame timing, dedicated eye textures,
external projections, and reversed depth. The renderer encodes a separate
pass for each view the compositor provides. The Simulator supplies one view.
Windowed stereo and physical headset qualification remain separate work.
[Apple's immersive Metal guide](https://developer.apple.com/documentation/compositorservices/drawing-fully-immersive-content-using-metal)

The future structure has three responsibilities:

| Responsibility | Owns | Does not own |
|---|---|---|
| Scene description | Geometry, transforms, cameras, materials, lighting and effect parameters, stable identities | Native views or Metal submission timing |
| Rendering implementation | Device capabilities, prepared pipelines, resource pools, render passes, scheduling, GPU lifetime | Native control semantics or application navigation |
| Presentation adapters | Drawable size, visibility/lifecycle signals, native layout and input integration | A separate simulation/render loop or per-frame scene reconstruction |

The SwiftUI target depends on the core. It retains a coordinator per prepared
renderer or scene command queue, sends updated scene values to it, and draws
on demand. The core also accepts a caller-owned Metal command buffer for
offscreen use.

## Rendering approach

Start with a conventional `MTLCommandQueue` triangle and a bounded number of frames in flight. Use the same fixture for a Metal 4 spike. Metal 4 is a backend candidate, not a user-facing promise of superior performance. Apple documents incremental adoption and independent Metal 4 command types alongside shared resource types. Runtime OS availability and GPU capability checks must agree before using a path. [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api), [feature detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions).

Metal 4 supports A14/M1 and later hardware, and optional features have their own limits.
Check support for each required GPU capability independently of the package deployment
targets. Record simulator and physical-device results separately. [Apple's Metal 4 overview](https://developer.apple.com/videos/play/wwdc2025/205/), [feature tables](https://developer.apple.com/metal/capabilities/)

Compare Satin with the minimal implementation in an isolated, bounded experiment. Examine shader escape hatches, resource ownership, dependency cost, licenses, and same-scene timings. The experimental common interface must not freeze the public architecture before either implementation has proven useful. The original [Hi-Rez/Satin](https://github.com/Hi-Rez/Satin) repository was archived on April 15, 2025; [Fabric-Project/Satin](https://github.com/Fabric-Project/Satin) is a candidate to revalidate, not a dependency already selected. Its inspected manifest uses tools 6.2, Swift 5 language mode, C++17, and no declared external package dependencies; its MIT license retains Hi-Rez copyright. Test integration with a Swift 6 consumer and record the exact evaluated revision. See [research spikes](research-spikes.md).

The first renderer needs ordered passes, not a general-purpose render-graph compiler. Add offscreen scene output and one post-process when required by a fixture. Consider more elaborate scheduling only when measured dependencies and reuse warrant it.

## State, threading, and lifetime

Native hosts follow the main actor's UI rules. Rendering receives a stable state snapshot or an explicit update batch; its exact representation is established by the transform/material fixtures. The UI should not publish thousands of particle positions or drive individual GPU updates through SwiftUI state.

Every native host derives size, display scale, lifecycle, and placement from its own
view and scene. On macOS, use the AppKit view's backing coordinates. On UIKit hosts,
use local bounds and `traitCollection.displayScale`. Respect safe areas and live
resizing. Give each window its own host and drawable state so two scenes can resize,
move between displays, pause, and resume independently.

A render owner serializes mutation of its device-specific caches and frame resources. GPU resource safety is separate from Swift actor isolation: a CPU task finishing does not mean the GPU has stopped reading its buffer. Every reusable resource needs a bounded ownership path from acquisition through submission to completion. Do not mark Metal resources broadly `Sendable` simply to silence compiler diagnostics.

Prepare shader libraries and common pipeline variants before interaction. Parameter changes update existing buffers or bindings. Resource pools are bounded, keyed by compatible descriptors, and account for textures still held by in-flight consumers. Resize creates a new usable generation while the old generation remains alive until its readers finish.

The native window hosts receive drawables from `CAMetalDisplayLink` on the
main run loop. The window owner pauses the link while idle or inactive, and
keeps one command buffer in flight. It holds a size-matched depth texture and
lets the committed buffer retain any previous attachment through completion.
[CAMetalDisplayLink](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink)

## Creative API progression

The implemented mesh path uses right-handed coordinates and column vectors.
The camera looks down local negative Z. The model matrix is `T * R * S` and
clip coordinates use projection times inverse camera transform times model.
The camera takes vertical field of view in radians and finite positive clip
distances with `near < far`. The projection maps that interval to Metal depth
`[0, 1]`.

Mesh vertices upload as consecutive `float4` position, normal, paired UV,
and tangent slots. The stride is 64 bytes with offsets 0, 16, 32, and 48.
UV0 uses XY and UV1 uses ZW. Tangent W carries handedness. Indices are UInt32.
Vertex buffer 0 holds the packed vertices. Buffers 1, 2, and 3 hold the clip
matrix, the inverse transpose for normals, and the world matrix respectively.
Custom fragments may read the normal, top-left UV, and world position varyings
as `float3 [[user(frostfield_normal)]]`, `float2 [[user(frostfield_uv)]]`,
and `float3 [[user(frostfield_world_position)]]`. The renderer validates
one read-only `float4` at fragment buffer 0 when it prepares a custom pipeline.
Custom materials return premultiplied linear RGBA.

Offscreen frames use linear sRGB and premultiplied RGBA16Float. A frame reports
readiness, dimensions, device, and a monotonic ID. The pool submits producer and
consumer work on its own queue and keeps the texture leased through GPU
completion. Retained frames continue to occupy a slot.

The first composition path is a camera plus transformed indexed geometry and a prepared material. Expose a documented shader binding contract so app authors can invent materials without waiting for a catalogue of effects. Add one directional Lambert light, one bounded particle emitter, and one full-screen post-process as separate slices.

Define matrix order, coordinate handedness, depth conventions, units, alpha handling, and color conversion once and test them with deterministic fixtures. Invalid geometry, shader functions, or incompatible bindings should produce actionable errors before unsafe work is encoded. Avoid a public type for every internal implementation detail.

Scene nodes compose world transforms as parent times local. Traversal preserves
the supplied sibling order and rejects duplicate IDs and invalid transforms.
Multiple nodes can share one prepared mesh. The graph validates each draw before
encoding the color and depth pass. A prepared custom Metal fragment can write
two to eight color attachments with independent formats, in pipeline index
order. Every visible mesh selected for that color pass uses the same format
layout and a shared depth texture. The caller owns the attachments and pass
actions. A named PBR pass can instead write regular color and selected surface
values, including the signed shading normal and graph-modified roughness. All
outputs share the surface's cutout coverage. Primitives use a separate pass.

Viewport picking unprojects a top-left normalized point through the camera's
clip planes. World rays can use any origin and direction. Static meshes and
batch geometry share local BVHs, and posed meshes test their current triangles.
Mesh rays honor the raster material side and report UVs and normals. Mixed
queries merge meshes, points, lines and sprites by world distance. Ties
preserve node and element order. Native hosts supply viewport coordinates or
controller rays and own their input subscriptions.

Texture loading distinguishes sRGB color from linear data maps. The PBR path
supports base color, metallic-roughness, normal, occlusion, and emissive maps,
plus per-map UV sets, sampler filtering and wrap modes. Authored tangent
frames carry handedness through skinning and world transforms. For meshes
without tangents, the normal shader derives the frame from world-position
and selected UV derivatives. Opaque materials write full
coverage. Masked materials discard pixels below the cutoff in color and
shadow passes. Blended draws sort by projected origin and test depth with
writes disabled.

OBJ and glTF loaders produce validated indexed geometry before Metal
allocation. Both expose top-left UV coordinates. glTF scene import can also
load node TRS animation, skin joint indices, and inverse bind matrices.
Adjacent buffers and images must be regular files, with no directory separators
or symlink traversal. [Khronos glTF specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)

Keyframe tracks target scene node IDs and interpolate translation, rotation,
and scale. Applying a clip returns a graph snapshot, leaving the source
available for the next sample. Imported clips support STEP, LINEAR, and
CUBICSPLINE channels. A prepared glTF scene reuses renderers and derives
per-instance joint palettes from the sampled graph. Skinning uses four
influences per vertex in both color and shadow passes. Picking deforms the
selected mesh's CPU query geometry with the same palette.

A directional shadow map uses a depth pass on the caller's queue before
the shared color pass. Lambert and metallic-roughness materials can use the
map's directional light and visibility. A finite light angle estimates
blocker separation and widens the filter away from contact. Additional
punctual lights come from scene nodes and add direct PBR illumination. The
per-graph list is bounded to 16 lights. Point and spot lights use inverse
square attenuation and optional range, with cone falloff for spots. Directional
and punctual maps can select basic sampling for one nearest depth comparison
with their configured receiver bias.
Directional and spot variance maps filter the first and second depth moments
with two compute passes. PBR, classic materials, and directional Lambert
shading use the stored moments for soft visibility.
On-demand maps reuse completed depth across caster edits until requested.
Punctual map sets preserve eligible light slots across traversal changes and
can refresh individual scene light nodes. A changed projection or slot
assignment refreshes that light's views.
`ClassicMaterial.shadowCatcher` uses those same maps to blend a tinted shadow
over previously drawn color. Its default state tests depth and preserves the
background depth buffer.

The procedural geometry builders produce ordinary indexed `MeshGeometry`.
Revolved profiles can carry hard rings or authored normals. Rounded boxes
preserve smooth normals across their UV charts. Simple concave contours
produce capped extrusions, and open 3D paths produce tubes with frames
transported along the path. Parametric grids can wrap either UV axis and
derive a smooth tangent basis from adjacent positions. The generated meshes
use the same material, shadow, and picking paths as imported triangles.

`ProceduralSurface` generates mipmapped PBR maps from a seeded preset or a
Swift function sampled at UV texel centers. It converts linear color to sRGB
storage, packs occlusion, roughness, and metallic channels as linear data,
and derives tangent-space normals from a wrapping height field. Preparation
finishes the Metal mip chain before publishing the textures. Build once and
reuse the returned material or textures between draws.

The optional path tracer prepares an immutable scene snapshot from static PBR
nodes. It flattens the hierarchy into world-space triangles and constructs a
bounded BVH. Texture maps become a mipmapped linear array. A conventional
Metal compute kernel traverses the BVH for camera, bounce, and visibility rays.
Rectangular emitters use next-event estimation and multiple importance
sampling. The integrator evaluates diffuse and GGX lobes, clearcoat, and
rough dielectric transmission. Closed dielectric boundaries determine path
length for absorption. [PBRT path integration](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer)

The caller owns a pair of tracked float32 mean textures through an
accumulation object. History is valid for one scene, camera, and set of
sampling parameters and advances only after successful GPU completion.
Camera changes start a new history. The optional denoiser uses first-hit
normals, distance, albedo, and roughness to guide a bounded spatial filter.
The HDR image then uses the existing tone mapper. Raster lights, shadow maps,
and baked AO are separate from the traced lighting.
[Edge-aware denoising reference](https://diglib.cgv.tugraz.at/items/89ef8071-636b-407c-b56c-2e2ab391bf0a)

`PathTracer.Lighting` is a value that can override the default rig for a
dispatch. The encoder copies the light data into the command buffer. The
accumulation validates the effective rig alongside the view and sample
settings so it cannot mix different lighting histories. Aimed rectangular
emitters accept radiance or emitted RGB power. The power initializer uses
panel area to convert to radiance.

`SurfaceShaderGraph` validates a bounded, typed DAG and emits one inspectable
Metal function for its connected material channels. Raster PBR calls the
function after sampling the authored material. Path tracing deduplicates
graphs in its scene snapshot and generates a dispatch to the same functions
at ray hits. Both use the material's primary UV, world position, shading
normal, and per-node parameters. Ordinary PBR pipelines use the bundled
precompiled shaders. Generated variants compile during preparation.
[Apple shader library APIs](https://developer.apple.com/documentation/metal/shader-library-and-archive-creation)

Environment lighting prepares a sharp radiance cube, cosine-convolved
diffuse light, roughness-dependent GGX reflections, and a split-sum BRDF
lookup. Materials can share those resources. The PBR shader samples them
in world space. A dielectric clearcoat can add a second reflection lobe with
its own roughness and normal while attenuating the underlying material. The
background pass uses the same yaw and an opaque HDR color target. Optional
contact occlusion and depth of field read stored perspective depth. HDRDisplay
maps exposed scene color to a display-referred range for Metal's sRGB conversion.
[Filament image-based lighting](https://google.github.io/filament/Filament.html#lighting/imagebasedlights)

Optical transmission splits a graph's HDR color pass around a snapshot of the
opaque and masked draws. The caller owns the tracked mipmapped texture and
can reuse it after the command buffer completes. The renderer copies opaque
color and generates its mip levels, then loads opaque depth for the overlay
pass. The material refracts using its index of refraction and an exit point
estimated from mesh-space thickness. Roughness selects a backdrop mip and
travel distance controls Beer attenuation. The snapshot covers visible opaque
geometry, with no recursive layer refraction or refractive shadows. Alpha
coverage remains separate. [Volume extension](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_volume)

An imported camera is attached to a glTF node. Resolving it from a prepared
graph preserves the node's animated location and rotation while discarding
scale from the view basis. The explicit view supports finite and infinite
perspective and positive orthographic extents. Orthographic shading uses a
constant view direction. Forward and reversed projections map into Metal's
normalized depth range. [glTF cameras](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#cameras)

Material graphs validate and compile typed expressions to Metal fragments
using the custom shader contract. The surface sampler chooses triangles
proportional to area and interpolates vertex attributes. Flow particles use
those samples as emission seeds and evolve on one GPU queue. Screen effects
read an offscreen RGBA16Float frame, apply half-resolution threshold bloom,
and combine it with chromatic aberration and seeded grain in the final pass.

The first usable v0.1 covers camera and transforms, indexed geometry, unlit and
custom materials, and native sample hosts with recorded qualification. Checkpoint C
requires clean-consumer builds and physical-device evidence for the configurations
advertised. Public texture exchange and GlassWeave interoperability follow in
separate milestones. Keep offscreen operations in the Satin comparison internal
until a consumer demonstrates the needed public contract.

## Shader graphs, light rigs, and scene tools

The [Three.js parity checklist](parity/threejs.md) orders this work and lists
what remains.

**Graph language.** `MaterialGraph.Node` is the typed vocabulary for every graph
kind: scalar and vector2–4 values, built-in inputs per stage, vector math,
conversions, noise, declared uniforms, declared textures, and caller-written
Metal functions. `ShaderProgram` validates types, declarations, and stage
availability, then emits one statement per reachable node with a `// node N`
marker. `ShaderGraphBuilder` describes the same graphs in Swift with typed
`ShaderValue<T>` handles. `SurfaceShaderGraph`, `PostProcessGraph`,
`TextureShaderGraph`, and `BufferShaderGraph` wrap the program for the surface,
vertex, post-process, and compute stages. Generated functions call renderer
hooks for uniforms, textures, and scene samples, so one graph compiles into
raster PBR, the path tracer, deformation kernels, and post passes.
`ShaderLibraryCache` compiles each generated source once per device.

**Live values.** `ShaderUniforms` holds a graph's uniform values and a revision.
Draws, dispatches, and post passes copy the current values each time, so edits
never touch pipelines. The path tracer records the revisions in its history and
rejects stale accumulation.

**Vertex deformation.** A surface graph's position offset runs in a compute pass
before the frame's render passes. It writes a deformed copy of the packed vertex
buffer and rebuilds normals from incident faces, welding vertices that share a
position and authored normal. Color, directional and punctual shadow passes,
picking, and path-tracing snapshots read the deformed vertices.

**Lighting.** `PunctualShadowMaps` renders spot, point, and rectangle-light
shadows into one depth array with reversed infinite-far views. Rectangle lights
use Lambertian irradiance with analytic horizon clipping and a linearly
transformed cosine fit for isotropic GGX and clearcoat. The path tracer samples
HDR environment maps by luminance with MIS and
adds analytic point, spot, and directional lights that match raster falloff.
`LightRig` authors lights by power or photometric intensity, size, color
temperature, and aim, and converts to raster scene nodes and `PathTracer.Lighting`.
`SceneGraph.environment` supplies a prepared image-based light to raster PBR
surfaces without a material environment. Compatible scene source replacements
reuse the same pipelines. Intensity and yaw are per-pass values.

**Animation and tools.** `AnimationMixer` blends `AnimationClip`s from Swift or
glTF with weights, cross-fades, loop modes, and additive layers.
`PreparedGLTFScene.posing(_:)` rebuilds joint palettes for a mixed graph.
`MeshMorphTargets` imports and applies glTF morph targets on the CPU.
`OrbitCameraController` and `FlyCameraController` take abstract input deltas.
`MeshBVH` accelerates ray queries, and `ViewFrustum` classifies bounds.
`GLTFExporter` writes glTF and GLB from imported scenes or scene graphs.

**Materials.** `PhysicallyBasedMaterial` implements the glTF material extensions
in both renderers. Every map has a `TextureTransform`, and normal-map tangent
frames follow the transformed coordinates. Specular, sheen, anisotropy,
iridescence, emissive strength and unlit shading share one header,
`FrostfieldMaterialExtensions.h`, so raster and path tracing use the same
formulas. Raster reads the layer maps through an argument buffer at fragment
buffer 22. The path tracer appends a 592-byte extension block to each material
and traces dispersion with per-wavelength refraction.

**Assets.** `GLTFMeshLoader` dequantizes `KHR_mesh_quantization` attributes and
decodes meshopt-compressed buffer views on load in pure Swift. `TextureLoader`
uploads KTX2 block formats and stored mip levels directly, and decodes WebP and
AVIF through ImageIO. A KTX2 texture keeps its PNG or JPEG source for GPUs
without the block format. `GLTFMeshLoader.supportedExtensions` lists every
honored extension.

**Scale.** `InstanceSet` draws many copies of one mesh in a single draw in the
color and shadow passes, with a ring of three upload buffers so edits never race
the GPU. Morph weights blend per draw in the vertex stage, before skinning.
`SceneGraph.encode` culls color draws by the camera frustum and returns
`SceneDrawStatistics`; shadow passes cull by each light's own view.
`LevelOfDetail` picks a mesh by distance with hysteresis, and every pass and the
path-tracing snapshot use the same choice.

**Dynamic path tracing.** `PathTracingScene` builds one bottom level per
distinct mesh and a top level over instances. The device's ray-tracing
capability selects Metal acceleration structures or a software BVH.
`updateTransforms(_:)` refits the top level, and
`updateGeometry(for:in:replacements:)` rebuilds chosen bottom levels for
skinned, morphed, or deformed meshes. Metal top-level builds run on a dedicated
queue so a refit can complete while a trace remains enqueued on the caller's
queue. Updates publish new resources after a successful build, so frames in
flight keep reading the old snapshot and the scene's `revision` invalidates
prior history. Material maps use their original textures where the device
supports direct sampling. Sampling defaults to Owen-scrambled Sobol points.
[Texture and acceleration evidence](evidence/pathscale.md)

## GlassWeave interoperability

GlassWeave is a separate framework for native and custom glass surfaces. Neither library initially depends on the other. An example app may depend on both and connect a Frostfield offscreen output to a GlassWeave background input.

The exchange contract should be small and explicit. Its exact Swift spelling is intentionally deferred:

| Contract field | Required meaning |
|---|---|
| Texture/device | A readable texture on the same `MTLDevice` as the consumer; reject unsupported cross-device transfer |
| Frame identity | Monotonic identity and producer time; define reuse, skipped, and stale-frame behavior |
| Dimensions/region | Pixel dimensions and valid content rectangle, independent of view points and display scale |
| Coordinates | Explicit origin, normalized UV mapping, crop/transform, and edge-sampling policy |
| Format/color | Pixel format, linear versus encoded values, color primaries, alpha convention, and conversion responsibility |
| Readiness | Producer writes are ordered before consumer reads, using the agreed queue/command order or an explicit supported synchronization primitive |
| Lifetime | A lease remains alive through all consumer GPU reads; reuse/destruction occurs only after their completion |
| Backpressure | A bounded number of leases; skip or reuse the last complete frame when no slot is available, according to documented policy |

For the first interoperable fixture, prefer one agreed linear working color space and a single negotiated format, with premultiplied alpha and an explicit SDR display conversion. A possible internal working format is linear sRGB in an `rgba16Float` texture. GlassWeave v0.1 instead proposes an sRGB BGRA8 input: the bridge must export that supported format with a real conversion, or explicitly extend and validate GlassWeave's accepted formats. Metadata alone cannot convert pixel data. Verify format/usage support and both implementations before fixing the public contract. Wide-gamut/HDR and arbitrary format conversion are later capabilities. A color-chart fixture must detect accidental double encoding, incorrect premultiplication, and inverted UVs.

Begin with a coordinated same-queue producer/consumer path if both implementations can share it cleanly. Cross-queue transfer requires an explicit event/wait contract; cross-device import is out of first-release scope. In Metal 4, allocator memory must not be reset before all command buffers using it finish. [Allocator reuse](https://developer.apple.com/documentation/metal/mtl4commandallocator/reset()).

Frostfield output supplies controlled background pixels. It does not grant GlassWeave access to arbitrary app views, or make custom materials part of Apple's Liquid Glass renderer. SwiftUI's `layerEffect` applies to its own rasterized content and documents limitations for UIKit/AppKit-backed views. [layerEffect](https://developer.apple.com/documentation/swiftui/view/layereffect(_:maxsampleoffset:isenabled:)).

## When to share GPU infrastructure

Extract an optional low-level product only when both working frameworks need the same device/capability handling, pipeline preparation, resource ownership, or texture-exchange primitives. The extraction must remove measured duplication without importing scenes, meshes, loaders, lighting, or UI frameworks into GlassWeave's native-only path.

Before extraction, record two concrete consumers, an acyclic dependency graph, ownership and versioning, standalone build coverage, binary/linkage impact, and the migration cost. A product hosted in the Frostfield repository may avoid linking a scene engine while still creating a package/repository dependency; document that distinction. A third repository is not assumed necessary.

## Scope of native coexistence

Apple's documented `UIGlassEffect` controls material configuration and interactivity. `UIGlassContainerEffect` combines views using that effect. These interfaces do not document injecting Frostfield or GlassWeave shaders into the native glass renderer. Custom merging requires custom-rendered participants; native controls can remain above them. [UIGlassEffect](https://developer.apple.com/documentation/uikit/uiglasseffect), [UIGlassContainerEffect](https://developer.apple.com/documentation/uikit/uiglasscontainereffect).

The integration target is a coherent native app with programmable graphics and custom surfaces alongside system glass, with explicit cost and lifecycle behavior.

## Language decision

Use Swift for CPU implementation and native APIs, MSL for custom GPU work, and a
small verified C-compatible data-layout boundary when needed. The
manifest baseline is Swift tools 6.2 with Swift 6 language mode. See the
[language strategy](language-strategy.md)
for toolchain distinctions, layout tests, ownership rules, foreign-language
admission gates, and conditional backend alternatives.
