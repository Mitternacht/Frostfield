# Frostfield implementation handoff

The shared Swift and Metal package powers programmable scenes and effects in
FrostfieldLab. The lab builds for native macOS, iOS/iPadOS, and visionOS, with
windowed hosts and a visionOS immersive fixture.
[Scene and effects evidence](docs/evidence/programmable-scene.md)

## Read in order

1. [AGENTS.md](AGENTS.md) — repository and Apple-tool requirements.
2. [Product and delivery plan](tasks/plan.md).
3. [Architecture and ownership](docs/architecture.md).
4. [Backend experiments](docs/research-spikes.md).
5. [Verification and performance protocol](docs/validation.md).
6. [Ordered implementation tasks](tasks/todo.md) — the only completion checklist.

Before adding GPU code, read the [language strategy](docs/language-strategy.md) for
Swift CPU code, MSL shaders, CPU/GPU data layout, and the criteria for adding C/C++
or Rust.

## Product intent

Build a Swift and Metal graphics framework with the creative breadth of Three.js.
Authors should be able to compose scene graphs, geometry, cameras, materials, and
effects, then extend them with their own shaders. Keep the graphics API independent
of AppKit, UIKit, and SwiftUI. Native hosts supply drawables and lifecycle signals.

The companion [GlassWeave](https://github.com/Mitternacht/GlassWeave) repository
owns glass surfaces and interaction. Its integration proof is the first priority
for the combined effort. Neither repository initially depends on the other.
Frostfield work can run independently when a separate implementer is available.

## Start here

Use the [Three.js checklist](docs/parity/threejs.md) to choose the next small
slice. Matching Three.js capabilities is the minimum target, alongside
photoreal asset creation. Track core, official add-ons, and ecosystem entries
separately. Keep each open until a public API and its verification cover the
listed behavior or a documented native platform equivalent.
The current integration includes glTF material tracing, full-resolution texture
sampling, Metal ray tracing with a software fallback, physical cameras and
photometric lights, raster primitives and materials, and live animation bindings.
The Lab exercises the same renderer. [Texture and acceleration
evidence](docs/evidence/pathscale.md) · [Camera and light evidence](docs/evidence/pathcamera.md)

The current verified package also has dynamic animation groups, independent
skinned clones and additive retargeting, PBR directional contact shadows,
camera-fitted sun cascades, editable helpers, textured sprite flares, and
planar refraction and water. Sun color and the remaining water variants
are tracked separately in the parity checklist.
[Animation evidence](docs/evidence/animtools.md) ·
[Lighting evidence](docs/evidence/rasterlight.md) ·
[Water evidence](docs/evidence/planar-water.md)

Opaque textured emitters participate in path light sampling. Raster adds
per-light shadow strength and HDR water with fog and shadowed sun scatter.
Supported graph-displaced meshes reuse or prewarm a posed picking hierarchy.
[Mesh light evidence](docs/evidence/path-emissive-mis.md) ·
[Graph picking evidence](docs/evidence/graph-pose-picking-bvh.md)

The cascaded sun takes a linear RGB tint, and `transmissionLayers` casts
layered colored shadows from uniform PBR glass onto PBR receivers.
`PlanarWater.flowingViewport` refracts a caller-supplied opaque snapshot with a
depth-safe fallback. Path fog accepts up to eight overlapping regions, each a
world box or an invisible closed mesh volume. A PBR surface graph can author its
own colored shadow with `castShadowColor` and `castShadowMask`, stored in scene
archive version 29. `ArrayCamera` encodes ordered views as one pass each.
`KTX2Exporter` writes native cube arrays, and `TextureUtilities`,
`ImageUtilities` and `Binary16` cover the r187 texture and image helpers.
`HDRDisplay.Settings.dithering` applies the r187 dithering shift to 8-bit output.
`SceneNode.onBeforeEncode` and `onAfterEncode` bracket each encoded draw, and
`SceneGraph.encodeInspector` observes scene encodes. `setRenderObjectFunction`
stays unimplemented by design.
[Sun RGB evidence](docs/evidence/sun-light-rgb.md) ·
[Glass shadow evidence](docs/evidence/directional-transmitted-shadows.md) ·
[Water evidence](docs/evidence/planar-water.md) ·
[Fog density evidence](docs/evidence/path-fog-density.md) ·
[Array camera evidence](docs/evidence/array-camera.md) ·
[KTX2 export evidence](docs/evidence/ktx2-export.md) ·
[Texture utility evidence](docs/evidence/texture-utilities.md) ·
[Dithering evidence](docs/evidence/dithering.md) ·
[Render hook evidence](docs/evidence/render-hooks.md)

The checklist records the recovered modeling tools, scene archives, video
textures, spatial audio, cascaded shadows, screen-space reflections, and
translucent and participating media. The asset tools now include mesh LODs,
subdivision, Draco import and export, Basis KTX2 decoding and native KTX2
export, automatic UV1 unwrapping, OBJ/PLY/STL export and progressive diffuse
lightmap baking with movable casters and receivers, rolling previews and
Halton-sampled irradiance from Swift or a Metal graph. The path tracer
streams EXR stills and uses MetalFX for
planar views with camera, rigid instance and topology-stable skin, morph and
typed shader displacement on supported devices. An FSR 1 pass can upscale
the path frame on the trace command buffer. The Asset Studio Lab bakes a
generated UV1 atlas and captures one textured diffuse reflection. The
shared-atlas variant packs the studio floor and backdrop into one lightmap.
A separate capture uses Metal-authored bounce irradiance and samples the
atlas through a native PBR material. Both renderers support that binding.
[Asset Studio evidence](docs/evidence/asset-studio.md)
[PBR light maps](docs/evidence/pbr-light-map.md)
The raster scene supports six-face cube image import, HDR cube capture,
filtered probes, and solid, flat-texture and directional backgrounds.
Authored or captured probe volumes add local diffuse light to classic and
PBR materials. Bake multiple bounces and refresh ranges from an immutable
prior grid. [Probe volume evidence](docs/evidence/light-probe-grid.md)
Room Environment prepares a reusable HDR light rig, and Grounded Skybox
projects a panorama across an inward sky and floor. Projected contact
shadows add a finite tinted receiver to a raster scene. Both PBR renderers
support scalar retroreflection.
[Room environment](docs/evidence/room-environment.md) · [Grounded skybox](docs/evidence/grounded-skybox.md) ·
[Contact shadows](docs/evidence/contact-shadows.md) · [Retroreflection](docs/evidence/pbr-retroreflection.md)
Classic materials reflect or refract an HDR cube, and PBR reflection and
clearcoat can project a captured environment onto room bounds. The native
outline effect draws styled hulls after scene color. Procedural clips animate
rotation, scale, shake, pulse, visibility and material RGB.
[Classic environment](docs/evidence/classic-environment-mapping.md) · [Box projection](docs/evidence/box-projected-environment.md) ·
[Outlines](docs/evidence/toon-outline-pass.md) · [Animation](docs/evidence/animtools.md)
Individual passes accept viewport and scissor controls. Rigid, skinned,
morphed, shader-displaced, curve-flow and topology-stable dynamic meshes can
drive HDR blur, temporal AA and temporal upscaling through raster motion.
Material stencil state supports posed, instanced, live indexed, curve-flow
and graph-displaced draws. It can resolve per-sample color on a combined
MSAA attachment. Encoded graph-displaced draws keep their exact geometry
through submission, including when commands run out of encode order. Motion blur can
jitter its taps with a caller-controlled blue-noise seed.
[Motion AA](docs/evidence/reference-traa.md) · [Motion blur](docs/evidence/raster-motion-blur.md) ·
[Temporal upscaling](docs/evidence/raster-temporal-upscaling.md) · [FSR 1](docs/evidence/native-fsr1.md) ·
[Stencil state](docs/evidence/material-stencil.md)
[Deformation ordering](docs/evidence/mesh-deformation-order.md) · [Shader BVH queries](docs/evidence/bvh-proximity.md#metal-shader-ray-queries)
Scene captures select their own layers, materials and resolution for
post-processing. Insert and remove prepared composer stages between frames.
Stages support geometry or texture masks, HDR save targets and full-screen
texture compositing. Scaled screen-space GI and reflections use depth-aware
resolves. Planar mirrors use an oblique reflected camera and can store color
mips and depth. [Scene pass](docs/evidence/post-process-scene-pass.md) ·
[Composer masks](docs/evidence/post-mask-save.md) · [Screen-space GI](docs/evidence/screen-space-gi.md) ·
[Screen-space reflections](docs/evidence/screen-space-reflections.md) · [Planar reflection](docs/evidence/planar-reflection.md)
Raster surface graphs can read stored scene color and depth. Scene-wide
image-based lighting can supply raster PBR and a path lighting snapshot.
[Viewport inputs](docs/evidence/surface-viewport-snapshot.md) ·
[Scene environment](docs/evidence/scene-environment.md)
Automatic opaque capture stores color mips and depth before transparent
draws. Planar mirrors can resolve multisampled sky and geometry and
compose bounded captures from distinct virtual cameras. Camera autofocus
reads a rendered depth texel, and tilt shift supplies a rotated focus band.
[Reflection](docs/evidence/planar-reflection.md) ·
[Autofocus](docs/evidence/camera-autofocus.md) · [Tilt shift](docs/evidence/tilt-shift.md)
Native points, sprites, graph surfaces and particle emitters can fade against
stored scene depth. Scene environment lighting supports three-axis orientation.
Mirror bounces can attach shadow and transmission passes and filter each
reflected view by depth. A pixelation pass renders a reduced grid and
expands it into nearest-neighbor blocks.
[Scene pixelation](docs/evidence/pixelated-scene.md)
[Scene environment](docs/evidence/scene-environment.md) ·
[Planar reflection](docs/evidence/planar-reflection.md)
Raster rectangle lights use LTC specular and clearcoat, and the path tracer
can sample constant opaque emissive triangles with reciprocal MIS. Path
settings separate direct and indirect radiance limits and let callers
disable MIS. [Soft particles](docs/evidence/soft-particles.md) ·
[LTC](docs/evidence/rectangle-ltc.md) · [Path sampling](docs/evidence/path-mis-and-radiance-limits.md) ·
[Mesh emitters](docs/evidence/path-emissive-mis.md)
Still exports can include selected first-hit layers alongside beauty. Scene
snapshots can query capsule, sphere and box collision through the mesh BVH.
Ray-driven transform gizmos move, rotate and scale scene nodes. Model curves
and surfaces include Catmull–Rom, NURBS and sampled lofts. Live curve flow
updates independent copies through CPU uploads or a Metal compute pass.
Native PLY imports meshes and point clouds with typed scalar data. Native STL
imports named solids and binary Magics colors. Scene archives reconnect typed
animation bindings and restore shared IES profiles and light cookies.
Box selection and mixed raycasts support perspective and orthographic views,
including controller rays that hit camera-facing sprites. Multi-geometry
batches have editable palettes, custom per-pass ordering and optional Metal
command bundles. The normal color pass can report GPU occlusion by node
and measure native stage durations. Graph and custom Metal passes and full
post-process chains also measure GPU time. A frame batch resolves render and
compute totals asynchronously across command buffers. Scene queries report
submitted primitive counts and prepared resource allocations. Static receivers can
share one atlas and relight with directional, point, spot and rectangle
sources, including IES and cookies where supported. Texture import covers DDS, KTX1 and PVR mip
chains and cubemaps. Copy or read selected texture subresources, decompress
block textures, and inspect live textures and shadow maps with scene helpers
or labeled overlays. Ordinary raster uploads can flip decoded rows and associate
alpha. Both renderers sample tangent-space or object-space base and clearcoat
normals, including packed XY, and can use seeded flake maps. Color grading
accepts image strips as 3D LUTs, and scalar color maps can generate legends.
Native cube mips project to nine spherical-harmonic coefficients for diffuse
raster probes that scene archives preserve. KTX2 loads native and Basis cubes
and arrays, with optional device-compressed LDR Basis and scene-linear HDR
profiles. Surface graphs control opacity and cutouts in raster color, surface
attributes, raster shadow depth, and path intersections. Marching cubes turns
sampled fields into indexed meshes that can update a prepared raster renderer.
Custom Metal fragments and named PBR outputs can fill multiple attachments.
Raster views can render and pick through perspective or orthographic
sub-frusta. Nodes can use authored pivots in transforms, animation and
interchange. Hemisphere lights use the raster diffuse probe path. glTF
material preparation can prefer device-compressed Basis textures.
Basic sampling compares one nearest shadow texel. Load BVH joint motion
into playable clips and PCD into colored point clouds. Query device
capabilities and upload selected CPU texture regions or array layers.
Indexed draw ranges apply to raster and path snapshots. NRRD loads typed
scalar volumes with physical coordinates. Animation paths follow their
scene parents, and raster maps can reuse completed depth for shadow catchers.
Scalar volumes support raymarched maximum intensity and isosurfaces. Process
mesh normals and bake affine transforms, archive world-oriented children,
and refresh individual lights in on-demand punctual shadow sets.
Point and line ranges drive draws and picking. Expand instanced or batched
draws into editable children and load XYZ point clouds or PDB molecules.
One indexed range can span material groups. Variance moments filter
directional and spot depth, with independent punctual update policies.
Frame off-axis camera views from three rectangular screen corners.
Load static MagicaVoxel scenes, add raster ambient lights, and query boxes,
spheres, planes and triangles with bounded world rays. Typed graphs support
refraction and face-forward.
Render into volume planes, array layers and cube faces. Import legacy and
XML VTK surfaces. Use panorama, matcap and atlas coordinates in graphs
and select cardinal orbit views in a Metal axis overlay.
Test closed shape overlap and ray proximity. Label the axis overlay and
turn at a fixed angular speed. Blend and tone-map colors in typed graphs.
Clear selected framebuffer attachments and sort visible meshes and primitives
per raster pass. Pack raster depth and distance, load still HDR JPEG XL, and
draw world camera frusta, posed skeletons and hemisphere light helpers.
Refresh posed skeletons at encode time. Choose a SwiftUI drawable pixel
ratio, import filled SVG paths, and compose large-world raster transforms
in Double through the graph camera. Bind raw Metal vertex and fragment
functions, outline SVG strokes, and trace elliptical disk emitters.
Import SVG gradients, local references, styled path UVs and standard SVG 1.1
public headers. Use reversed depth in SwiftUI, named perspective passes,
visibility queries, timings and particle passes. Present wide-gamut HDR in
SwiftUI. Trace finite-aperture spots with blue-dithered Sobol sampling and
inspect completed sample counts and per-ray software acceleration heatmaps.
Use linear PBR, Wavefront and typed graph bump maps in either renderer.
Sample the r187 simplex, curl and animated Voronoi fields in shader graphs.
Adapt the path preview to completed GPU timings and retain its last image
through a resolution change. Preserve local precision in single directional
and cascaded shadows at shared world origins.
[PBR bump maps](docs/evidence/pbr-bump-maps.md) · [Graph bump](docs/evidence/shader-texture-bump-maps.md) ·
[Wavefront bump](docs/evidence/mtl-bump-maps.md) · [CPU noise](docs/evidence/cpu-procedural-noise.md) ·
[GPU patterns](docs/evidence/shader-procedural-patterns.md) · [Adaptive preview](docs/evidence/path-adaptive-preview.md)
Present the tracer in a native SwiftUI view. Reconstruct post-process depth
and surface normals from the view's projection. Inspect meshes with a
pass-specific material and keep spotlight shadows aligned at large origins.
[SwiftUI path view](docs/evidence/swiftui-progressive-path.md) · [Depth helpers](docs/evidence/post-process-depth-helpers.md) ·
[Material override](docs/evidence/scene-material-override.md)
Blend independent scene renders with a mask and draw depth-tested toon
hulls. Fade from an adaptive path preview to a fixed-size trace. Sample
classic height and normal detail and preserve point and rectangle shadow
coordinates at shared origins.
[Transitions](docs/evidence/scene-transition-pass.md) · [Toon outline](docs/evidence/toon-outline-pass.md) ·
[Preview handoff](docs/evidence/path-preview-handoff.md) · [Classic maps](docs/evidence/classic-bump-and-normal-maps.md)
Override points, lines and sprites independently. Denoise AO with depth
and normal guidance. Trace generated HDR panoramas and the spectral sun.
Keep batched model-view transforms and instanced shadows precise at large origins.
[Overrides](docs/evidence/scene-material-override.md) · [AO denoise](docs/evidence/ao-denoise.md) ·
[Environment maps](docs/evidence/path-generated-environments.md)
[Wide-gamut output](docs/evidence/wide-gamut-output.md) · [SwiftUI HDR](docs/evidence/swiftui-hdr-presentation.md) ·
[Reversed depth](docs/evidence/reversed-depth.md) · [Finite spotlights](docs/evidence/path-finite-spot.md) ·
[SVG gradients](docs/evidence/svg-gradient-import.md) · [Path diagnostics](docs/evidence/path-sampling-diagnostics.md)
[Raw Metal programs](docs/evidence/raw-mesh-program.md) · [SVG strokes](docs/evidence/svg-stroke-import.md) ·
[Path area emitters](docs/evidence/path-shaped-emitters.md)
[Drawable resolution](docs/evidence/drawable-resolution.md) · [SVG paths](docs/evidence/svg-path-import.md) · [High-precision raster](docs/evidence/high-precision-model-view.md)
[Framebuffer clears](docs/evidence/framebuffer-clear.md) · [Scene ordering](docs/evidence/scene-render-ordering.md) ·
[Depth packing](docs/evidence/depth-material-packing.md) · [HDR JPEG XL](docs/evidence/jpeg-xl-import.md) ·
[Scene helpers](docs/evidence/scene-debug-helpers.md)
[Curve flow evidence](docs/evidence/curve-deformation.md) ·
[PLY evidence](docs/evidence/ply-import.md) ·
[STL evidence](docs/evidence/stl-import.md) ·
[PCD point clouds](docs/evidence/pcd-import.md) · [BVH motion capture](docs/evidence/bvh-motion-capture.md) ·
[Basic shadow sampling](docs/evidence/basic-shadow-map.md) · [Renderer capabilities](docs/evidence/renderer-capabilities.md) ·
[On-demand shadows](docs/evidence/on-demand-shadow-updates.md) · [Shadow catcher](docs/evidence/shadow-catcher.md) ·
[Animation paths](docs/evidence/animation-path-helper.md) · [NRRD volumes](docs/evidence/nrrd-import.md) · [Mesh draw ranges](docs/evidence/mesh-draw-range.md) ·
[Scalar volume rendering](docs/evidence/scalar-volume-rendering.md) · [Mesh processing](docs/evidence/mesh-processing.md) · [World-oriented children](docs/evidence/world-oriented-child.md) ·
[Scene utilities](docs/evidence/scene-utils.md) · [XYZ point clouds](docs/evidence/xyz-import.md) · [Variance shadows](docs/evidence/variance-shadow-maps.md) ·
[PDB molecules](docs/evidence/pdb-import.md) · [Corner framing](docs/evidence/frame-corners.md) ·
[MagicaVoxel scenes](docs/evidence/vox-import.md) · [Ambient light](docs/evidence/ambient-light.md) ·
[Geometric queries](docs/evidence/geometric-queries.md) · [Vector math](docs/evidence/graphlang.md#vector-directions) ·
[Layered targets](docs/evidence/layered-render-targets.md) · [VTK surfaces](docs/evidence/vtk-legacy-import.md) ·
[Graph UV projections](docs/evidence/shader-graph-uv.md) · [Axis view](docs/evidence/axis-view-helper.md) ·
[Graph color operators](docs/evidence/shader-graph-color.md) ·
[Scene archives](docs/evidence/scenekit.md) ·
[Selection and input](docs/evidence/animation-and-tools.md#selection-and-input) ·
[Batches](docs/evidence/batched-mesh.md) · [Occlusion queries](docs/evidence/scene-occlusion.md) ·
[GPU timing](docs/evidence/scene-gpu-timing.md) ·
[Draw and resource statistics](docs/evidence/scene-render-statistics.md) ·
[Graph coverage](docs/evidence/graph-coverage.md) · [Marching cubes](docs/evidence/marching-cubes.md) ·
[Dynamic geometry](docs/evidence/dynamic-geometry.md) · [Multiple render targets](docs/evidence/custom-mesh-mrt.md) ·
[Named PBR outputs](docs/evidence/named-surface-mrt.md) · [Scene pivots](docs/evidence/scene-pivots.md) ·
[Camera view offsets](docs/evidence/camera-view-offset.md) ·
[KTX2 spatial import](docs/evidence/ktx2-spatial-import.md) ·
[Texture transfers](docs/evidence/texture-transfers.md) · [Decompression](docs/evidence/texture-decompression.md) ·
[Texture previews](docs/evidence/texture-debug-pass.md) · [Raster uploads](docs/evidence/raster-texture-upload.md) ·
[Packed normals](docs/evidence/packed-normal-maps.md) · [Object-space normals](docs/evidence/object-space-normal-maps.md) ·
[Flake normals](docs/evidence/flake-normal-textures.md) · [Image LUTs](docs/evidence/lut-image-import.md) ·
[Scalar color maps](docs/evidence/scalar-colormap.md) · [Spherical probes](docs/evidence/spherical-light-probe.md) ·
[DDS](docs/evidence/dds-import.md) · [KTX1](docs/evidence/ktx1-import.md) · [PVR](docs/evidence/pvr-import.md) ·
[TGA and TIFF](docs/evidence/tga-tiff.md) ·
[Probe evidence](docs/evidence/rasterlight.md#hdr-scene-cube-capture) ·
[Still output evidence](docs/evidence/path-output.md) · [Collision evidence](docs/evidence/collision.md) ·
[Mesh export evidence](docs/evidence/mesh-export.md) · [Gizmo evidence](docs/evidence/transform-gizmo.md)
Open areas include broader archives, dynamic and nondiffuse bake transport,
and temporal reprojection of arbitrary deformations. Keep simulator checks,
physical-device qualification, and matched Three.js benchmarks separate.

Continue qualification from the task list. The native Mac lifecycle and the
programmable scene have local evidence. Run the Mac lab for development and
record simulator and physical-device evidence separately. The conventional
Metal renderer is the current path after the bounded backend probes.
[Lifecycle results](docs/evidence/render-lifecycle.md)

The first checkpoint is a render host that behaves correctly through resize,
inactivity, and teardown. Use that fixture for the backend comparisons. The
developer preview then adds a composed mesh scene and custom shader.

The limited standalone v0.1 candidate is the programmable core through FF-08
plus Checkpoint C's clean-consumer and physical-device qualification. Later
effects and GlassWeave interoperability do not block that independent release.

## Decisions and latitude

| Fixed direction | Implementation freedom requiring evidence |
| --- | --- |
| Independent public repository; MIT license | Internal file names and target decomposition |
| Swift-first API, Metal rendering, native presentation | Metal 4 adoption and compatibility policy |
| Native AppKit and UIKit Metal hosts, with SwiftUI windows on macOS and visionOS | Custom renderer versus a bounded Satin-backed approach |
| macOS, iOS, iPadOS, and visionOS 27.0 or later | Which configurations qualify for release and which quality tiers ship |
| No GlassWeave scene dependency; no premature shared core | Extraction of a small Metal product after two real consumers exist |

Use Xcode 27 with the macOS, iOS, and visionOS 27 SDKs. The visionOS window has a
single raster view. Compositor Services supplies views to the immersive
fixture. Qualify stereo, tracking, and comfort on physical Vision Pro.

## Working contract

For Apple work, load the matching installed Xcode 27 skill and use the native
Xcode MCP server for documentation, project edits, builds, runs, diagnostics,
tests, and previews. Verify a working bridge before relying on build results.
Do not replace that workflow silently with terminal builds. A missing package
creation operation can be handled as a documented bootstrap limitation; regular
project editing and verification still use native Xcode tools.

Use short, reviewable slices. Test observable behavior, resource lifetimes, and
failure paths. Add focused CI test jobs with the first testable behavior. Keep
API examples tied to code that actually compiles. Do not add release tags or
performance claims before their acceptance gates pass.

At each checkpoint record the commit, Xcode/SDK versions, destination, build and
test results, visual evidence, and any missing physical-device evidence. Keep
failed checks visible. A simulator cannot satisfy a physical-device performance
gate. Update the task list only after its stated verification passes.

## Suggested implementation kickoff

> Read AGENTS.md and HANDOFF.md, then the linked plan, architecture, experiments,
> validation protocol, and task list. Implement Frostfield in dependency order,
> starting with the first open acceptance item in the task list. Use the required
> Apple skills and native Xcode MCP tools. Keep interfaces experimental until
> working fixtures establish them. Record evidence and update the task list as
> each slice passes. Preserve GlassWeave's independence and avoid expanding the
> first milestone into a complete engine.
