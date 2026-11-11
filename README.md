# Frostfield

An Apple-native creative graphics framework for macOS, iOS, iPadOS, and visionOS,
built with Swift and Metal.

Frostfield takes Three.js as a reference for the kinds of scenes and effects authors
can build. The rendering core is shared across platforms and independent of the
frameworks that present it.

## Status

**Programmable scene preview.** The `Frostfield` product composes procedural
and imported meshes, perspective cameras, animation, and picking in a scene
graph. glTF 2.0 assets can supply cameras, node animation, skinning, PBR maps,
texture samplers, alpha modes, and punctual lights. The PBR renderer supports
HDR environment reflections, directional shadows with contact hardening,
clearcoat, optical transmission and volume absorption, authored tangents,
and secondary UVs. Use typed material graphs
or your own Metal vertex and fragment functions with explicit buffers,
textures and samplers. [Raw Metal programs](docs/evidence/raw-mesh-program.md)
The package includes
screen-space ambient occlusion, depth of field, HDR tone mapping,
surface-seeded flow particles, bloom, chromatic
aberration, grain, and completion-aware offscreen textures. The optional
`FrostfieldSwiftUI` product displays a prepared mesh or scene graph in a
native Metal view. Choose its drawable pixel ratio independently of SwiftUI
layout and observe the completed pixel dimensions. Present HDR in
Display P3 or extended linear Rec. 2020.
[Drawable resolution](docs/evidence/drawable-resolution.md) · [SwiftUI HDR](docs/evidence/swiftui-hdr-presentation.md)
[Run FrostfieldLab](Examples/README.md) · [Verification](docs/evidence/programmable-scene.md)

Build assets directly in Swift with revolved profiles, beveled boxes,
extruded contours, swept tubes, and parametric surfaces. Shape meshes support
vector text, decals, convex hulls, and Boolean solids. Geometry tools merge
indexed parts, weld vertices, split creases, tessellate long edges, simplify
meshes into LOD chains, subdivide surfaces, and generate MikkTSpace tangents. `ProceduralSurface`
creates color, normal, and roughness maps from presets or a Swift function.
Extract indexed isosurfaces from scalar fields and interpolate metaball colors.
Update the resulting geometry on a prepared raster renderer.
[Marching cubes evidence](docs/evidence/marching-cubes.md) · [Dynamic geometry](docs/evidence/dynamic-geometry.md)
Sample Catmull–Rom, Bézier and NURBS curves by parameter or distance, and
tessellate NURBS patches or lofts through sampled profiles. Animate independently
placed mesh copies along prepared curves with `CurveFlow`.
[Curve and surface evidence](docs/evidence/curves.md) · [Curve flow evidence](docs/evidence/curve-deformation.md)
Author cap and wall UVs for beveled solids, curve extrusions, and text.
[Custom extrusion UVs](docs/evidence/extrusion-custom-uv.md)
Sweep tubes along sampled curves and triangulate original polygon rings.
[Curve tubes](docs/evidence/curve-tube-geometry.md) ·
[Polygon triangulation](docs/evidence/polygon-triangulation.md)
FrostfieldLab's rangefinder builds its geometry, finishes, lettering, and
studio lighting at runtime. [Code and renders](docs/evidence/procedural-camera.md) ·
[Shape evidence](docs/evidence/geoshapes.md)

The Metal path tracer progressively renders PBR scenes with full-resolution
material maps, indirect lighting, and dielectric refraction. It selects Metal
ray tracing or a software BVH based on the GPU. It supports physical aperture
controls, panoramic projections, photometric lights, rectangular and elliptical disk emitters,
finite-radius spots, subsurface scattering, diffuse transmission, and
participating fog. Choose scrambled or blue-dithered Sobol sampling.
Voxel density grids give world-box fog spatially varying extinction.
[Fog density](docs/evidence/path-fog-density.md)
Toggle multiple importance sampling, set separate direct and indirect
radiance limits, and opt in to light sampling of constant and textured
opaque emissive triangles. [Path sampling](docs/evidence/path-mis-and-radiance-limits.md) ·
[Mesh emitters](docs/evidence/path-emissive-mis.md)
Inspect completed sample counts and visualize software BVH bounds with a per-ray heatmap.
Linear PBR and typed graph height maps perturb the shading normal in both renderers.
The progressive preview can adapt its resolution to a GPU timing budget.
Generate gradient or procedural HDR environments and trace the spectral sun
through camera rays and reflections. [Environment maps](docs/evidence/path-generated-environments.md)
[PBR bump maps](docs/evidence/pbr-bump-maps.md) · [Graph bump](docs/evidence/shader-texture-bump-maps.md) ·
[Adaptive preview](docs/evidence/path-adaptive-preview.md)
Present a progressive path trace directly in SwiftUI with camera updates,
pause and sample limits. Fade from an adaptive preview after the fixed-size
trace reaches an authored sample count. [SwiftUI path view](docs/evidence/swiftui-progressive-path.md) ·
[Preview handoff](docs/evidence/path-preview-handoff.md)
[Finite spotlights](docs/evidence/path-finite-spot.md) · [Path diagnostics](docs/evidence/path-sampling-diagnostics.md) [Path-traced renders
and usage](docs/evidence/path-tracing.md) · [Texture and acceleration
evidence](docs/evidence/pathscale.md) · [Camera and light evidence](docs/evidence/pathcamera.md)

Trace still images in deterministic tiles and stream HDR beauty and selected
first-hit layers together to OpenEXR. Use native MetalFX denoising for
planar views with camera, rigid instance and topology-stable pose motion on supported devices. [Path output
evidence](docs/evidence/path-output.md) · [Temporal evidence](docs/evidence/path-temporal-inputs.md)

Import Draco-compressed glTF triangles and ETC1S or UASTC KTX2 material maps.
Imported glTF materials can prefer device-compressed Basis storage while
retaining RGBA8 and optional image fallbacks.
Load stored native and Basis KTX2 cubes and arrays. Transcode LDR Basis to
device-compressed storage or decode supported HDR profiles to linear half float.
Load DDS, compressed KTX1, and PVR textures and cubemaps with stored mips,
raw and RLE TGA textures, and the first image of TIFF files. Decode still HDR
JPEG XL textures to linear half or full float. [JPEG XL evidence](docs/evidence/jpeg-xl-import.md)
Flip decoded raster rows or associate alpha before generating mipmaps.
Expand compressed textures for editing or export, and upload, copy or read
selected texture regions and mips. Inspect live textures in the scene or a screen
overlay, including labeled native shadow maps.
Use tangent-space or object-space base and clearcoat normal maps, including
packed BC5 and EAC. Generate seeded normal flakes for metallic paint.
Load 3D color lookup tables from text files or image strips, or visualize
scalar data with reusable color ramps and legends.
Load PLY meshes and point clouds with per-corner UVs and typed scalar columns,
and STL meshes with ASCII solid groups or binary Magics colors.
Load organized PCD point clouds in ASCII, binary or LZF-compressed form,
XYZ point clouds with optional RGB, and PDB atoms and explicit bonds.
Load MagicaVoxel models and static scenes as colored meshes or occupancy textures.
Load legacy and XML VTK PolyData as indexed meshes. Import SVG paths,
basic shapes, filled contours with holes, local references and CSS styles.
Prepare gradient fills and textured outlines with clipped joins.
[SVG gradient fills](docs/evidence/svg-gradient-import.md)
[SVG path import](docs/evidence/svg-path-import.md) · [SVG strokes](docs/evidence/svg-stroke-import.md)
Load typed NRRD scalar volumes with physical voxel orientation. Raymarch
their native Metal textures with maximum-intensity projection or threshold isosurfaces.
Generate vertex normals and tangent frames, center mesh bounds, and bake affine transforms.
Export Draco meshes and point clouds and uncompressed KTX2 textures with mips.
Bake mesh colors, normals and static directional irradiance into authored
or automatically unwrapped UV atlases. Accumulate indirect diffuse light
from static base-color and emissive maps across multiple receivers in one
shared UV1 atlas. Move receivers without repacking and supply irradiance
from a Swift closure or a Metal buffer graph. Native PBR materials sample
the baked map in both renderers.
[Draco](docs/evidence/draco.md) · [DDS](docs/evidence/dds-import.md) ·
[KTX1](docs/evidence/ktx1-import.md) · [PVR](docs/evidence/pvr-import.md) ·
[TGA and TIFF](docs/evidence/tga-tiff.md) ·
[Basis Universal](docs/evidence/basis-universal.md) · [PLY](docs/evidence/ply-import.md) ·
[STL](docs/evidence/stl-import.md) · [UV atlases](docs/evidence/mesh-uv-atlas.md) ·
[Texture and light-map baking](docs/evidence/static-lightmap.md) ·
[PBR light maps](docs/evidence/pbr-light-map.md) ·
[Packed normals](docs/evidence/packed-normal-maps.md) · [Object-space normals](docs/evidence/object-space-normal-maps.md) ·
[Paint flakes](docs/evidence/flake-normal-textures.md) · [Image LUTs](docs/evidence/lut-image-import.md) · [Scalar color maps](docs/evidence/scalar-colormap.md) ·
[Texture transfers](docs/evidence/texture-transfers.md) · [Decompression](docs/evidence/texture-decompression.md) ·
[Texture previews](docs/evidence/texture-debug-pass.md) · [Raster uploads](docs/evidence/raster-texture-upload.md)
[KTX2 cubes and arrays](docs/evidence/ktx2-spatial-import.md) · [PCD point clouds](docs/evidence/pcd-import.md) · [XYZ point clouds](docs/evidence/xyz-import.md) · [PDB molecules](docs/evidence/pdb-import.md) · [MagicaVoxel scenes](docs/evidence/vox-import.md) · [NRRD volumes](docs/evidence/nrrd-import.md) · [Scalar volume rendering](docs/evidence/scalar-volume-rendering.md) · [Mesh processing](docs/evidence/mesh-processing.md)
[Legacy VTK](docs/evidence/vtk-legacy-import.md) · [XML VTK](docs/evidence/vtk-xml-import.md)

`SurfaceShaderGraph` generates Metal for base color, metallic, roughness,
emission, and a transmission mask in both renderers. A prepared path tracer
accepts a new light rig per dispatch. FrostfieldLab exposes power, size, angle, tint, and exposure controls
for its studio. [Generated shaders and live relighting](docs/evidence/lights-and-shaders.md)

Shader graphs are typed and can be written as Swift expressions. They sample
textures, call your own Metal functions, expose uniforms that edit live without
recompiling, set normals and occlusion, and offset vertices. The same graph
shades raster PBR and the path tracer. Post-process graphs form bypassable
chains, and compute graphs generate textures, PBR maps, and buffer updates.
Capture a scene for post-processing with its own layers, materials and
resolution. Apply Sobel or Frei–Chen edge detection to its color.
[Scene capture](docs/evidence/post-process-scene-pass.md) ·
[Frei–Chen](docs/evidence/frei-chen.md)
Surface graphs can sample stored or automatically captured opaque color and
depth, including explicit color mips. Post effects can composite textured
lens dirt from HDR bloom, style selection outlines, and add tilt shift.
[Viewport inputs](docs/evidence/surface-viewport-snapshot.md) ·
[Lens dirt](docs/evidence/lens-dirt.md) · [Outline styling](docs/evidence/selection-outline-style.md) ·
[Tilt shift](docs/evidence/tilt-shift.md)
Read GPU execution times for graph and custom Metal passes and full post-process
chains. Resolve separate render and compute totals across a frame's command buffers.
Inspect submitted geometry and the allocations held by a prepared scene.
Edit indexed mesh ranges across material groups and rebuild their path
snapshots. Point and line draw ranges apply to raster and picking.
Query supported render paths and sample counts on the device.
Custom Metal fragments can write multiple color attachments in one raster pass.
Named PBR passes can write lit color, world normals and material values together.
Render into selected volume planes, array layers and cube faces at any prepared mip.
Clear color, depth and stencil independently, and choose the SwiftUI drawable clear color.
Sort opaque and transparent scene draws with per-pass comparators.
Use camera-relative Double transforms for large-world perspective,
orthographic and off-axis raster draws. Reversed depth applies to
owned perspective views, including SwiftUI, and both particle renderers.
Single-sample mesh passes can use logarithmic depth for large clip ratios.
Planar mirrors can capture reflected skies, resolved color and depth, and
multiple bounces with view-specific shadows and transmission.
[Logarithmic depth](docs/evidence/logarithmic-depth.md) ·
[Planar reflection](docs/evidence/planar-reflection.md)

Scene image-based lighting supports three-axis orientation in raster and path.
[Scene environment](docs/evidence/scene-environment.md)
Points, sprites, surface graphs and particle emitters can fade intersections against
stored scene depth.
[Soft particles](docs/evidence/soft-particles.md)
Sample CPU improved Perlin and seeded simplex noise, or shader simplex,
curl and animated Voronoi fields. Rebase directional shadows at large
shared world origins, including cascades and split transitions.
[CPU noise](docs/evidence/cpu-procedural-noise.md) · [GPU patterns](docs/evidence/shader-procedural-patterns.md)
Inspect meshes, points, lines and sprites with alternate raster materials.
Reconstruct view positions and normals from stored depth and denoise AO in
post-process graphs. [AO denoise](docs/evidence/ao-denoise.md)
[Material override](docs/evidence/scene-material-override.md) · [Depth helpers](docs/evidence/post-process-depth-helpers.md)
Draw inverted hulls around toon meshes, composite two scene renders, and
sample classic bump or normal maps.
[Toon outline](docs/evidence/toon-outline-pass.md) · [Transitions](docs/evidence/scene-transition-pass.md) ·
[Classic maps](docs/evidence/classic-bump-and-normal-maps.md)
Refresh posed skeleton helpers while encoding the scene.
[High-precision raster](docs/evidence/high-precision-model-view.md) · [Scene helpers](docs/evidence/scene-debug-helpers.md)
Write raster depth and normalized distance to packed linear render targets.
[Framebuffer clears](docs/evidence/framebuffer-clear.md) · [Scene ordering](docs/evidence/scene-render-ordering.md) · [Depth packing](docs/evidence/depth-material-packing.md)
Perspective and orthographic raster views can render and pick through pixel
windows of a larger frame.
Frame an off-axis camera from three screen corners and a world-space eye.
Use refraction, face-forward, equirectangular, matcap and sprite atlas operations in typed shader graphs.
Compose screen and overlay blends, ACES, posterization, and color decision lists in graphs.
[Shader Lab and verification](docs/evidence/shader-graphs.md) · [GPU timing](docs/evidence/scene-gpu-timing.md)
[Draw and resource statistics](docs/evidence/scene-render-statistics.md) · [Graph coverage](docs/evidence/graph-coverage.md)
[Multiple render targets](docs/evidence/custom-mesh-mrt.md) · [Named PBR outputs](docs/evidence/named-surface-mrt.md) · [Camera view offsets](docs/evidence/camera-view-offset.md) · [Corner framing](docs/evidence/frame-corners.md)
[Renderer capabilities](docs/evidence/renderer-capabilities.md) · [Mesh draw ranges](docs/evidence/mesh-draw-range.md) · [Vector math](docs/evidence/graphlang.md#vector-directions)
[Layered render targets](docs/evidence/layered-render-targets.md) · [Graph UV projections](docs/evidence/shader-graph-uv.md)
[Graph color operators](docs/evidence/shader-graph-color.md)

`LightRig` authors softboxes, spots, bulbs, and suns by power, size, color
temperature, and aim for both renderers. Raster spot, point, and rectangle
lights cast shadows, and the path tracer samples HDR environment maps.
A shared scene environment can light raster PBR and supply the path tracer's
importance-sampled sky. [Scene environment](docs/evidence/scene-environment.md)
Raster PBR uses LTC tables for isotropic rectangle highlights and clearcoat.
[Rectangle lighting](docs/evidence/rectangle-ltc.md)
Use nearest-depth shadow sampling for hard directional, spot or point shadows.
Use variance sampling for filtered directional, cascaded and spot shadows.
Mix automatic and on-demand policies in a punctual shadow set, refresh one
light at a time, and composite a tinted shadow catcher over a background.
[Basic shadow sampling](docs/evidence/basic-shadow-map.md) · [Variance shadows](docs/evidence/variance-shadow-maps.md) · [On-demand shadows](docs/evidence/on-demand-shadow-updates.md) · [Shadow catcher](docs/evidence/shadow-catcher.md)
Project native cubemaps into nine spherical-harmonic coefficients and use the
probes as diffuse lights in raster scenes. Author the same diffuse lighting
with sky and ground colors and a hemisphere direction. Add ambient nodes
for direction-independent diffuse light.
[Light probe evidence](docs/evidence/spherical-light-probe.md) · [Ambient lights](docs/evidence/ambient-light.md)
Raster lighting also supports directional shadow cascades, bent-normal
occlusion, scaled screen-space GI and reflections, and HDR cube captures for
reflection probes. Load six SDR or HDR faces into a cube for raster skies and image-based
lighting. Raster backgrounds support flat images, directional textures, and
filtered skies with independent orientation, blur and intensity.
[Raster lighting](docs/evidence/rasterlight.md) · [Screen-space GI](docs/evidence/screen-space-gi.md) ·
[Screen-space reflections](docs/evidence/screen-space-reflections.md) · [Backgrounds](docs/evidence/environment-background.md) ·
[Cube textures](docs/evidence/formats.md#six-face-cube-textures)

Edit scene hierarchies with world-oriented children and save native scene
archives with supported renderers, textures, and animation. Load assets through a shared async
cache. Use AVPlayer textures in materials and play spatial audio from scene
nodes. [Scene tools and media](docs/evidence/scenekit.md) ·
[BVH motion capture](docs/evidence/bvh-motion-capture.md) · [Animation path helpers](docs/evidence/animation-path-helper.md) ·
[World-oriented children](docs/evidence/world-oriented-child.md)

Expand instance and batch draws into editable children, sort instance
records, and reduce visible world-space vertices. Query boxes, spheres, planes
and triangles with a bounded world ray and measure proximity to points and segments.
Traverse mesh BVHs with custom shape tests and selectively refit changed
vertex positions. Scene picking refits skinned and morphed meshes. Point
renderers can opt into BVH picking with the same world-space tolerance.
[Scene utilities](docs/evidence/scene-utils.md) ·
[Geometric queries](docs/evidence/geometric-queries.md) ·
[BVH queries](docs/evidence/bvh-proximity.md) ·
[Animation mixing, camera controls, picking, and glTF export](docs/evidence/animation-and-tools.md) ·
[Three.js parity checklist](docs/parity/threejs.md)

Move, rotate and scale scene nodes with pickable transform gizmos. Export
posed meshes and instanced scene geometry as OBJ, PLY or STL.
Label a six-axis camera overlay and align an orbit view with a fixed-speed turn.
[Gizmos](docs/evidence/transform-gizmo.md) · [Axis view](docs/evidence/axis-view-helper.md) · [Mesh export](docs/evidence/mesh-export.md)

FrostfieldLab builds a closed iceberg, open ocean, and sky from code. The
iceberg's surface graph varies roughness and transmission across the same
mesh. [Path-traced iceberg](docs/evidence/iceberg-ocean.md)

FrostfieldLab's Cyber Ocean reference scene runs the glTF dolphin's original
swim cycle through a procedural tunnel, GPU flow particles, and a screen
effects chain. [Native Cyber Ocean captures](docs/evidence/cyber-ocean.md)

The PBR Studio fixture renders a textured Blender-exported antique camera
under an HDR environment and a studio light rig. The same fixture accepts
local glTF and GLB assets. [Mac renders and fixture instructions](docs/evidence/pbr-studio.md)

Asset Studio uses an automatic UV1 color bake on the procedural rangefinder's
metal grip. Its capture variants bake one progressive diffuse reflection
against a direct-light reference, with the floor and backdrop available
as separate receivers in a shared lightmap. Texture cutouts can transmit
baked shadow and bounce rays and define receiver coverage.
[Launch and capture instructions](docs/evidence/asset-studio.md) ·
[Indirect lighting evidence](docs/evidence/asset-studio-indirect.md)

## Direction

The preview uses conventional Metal. FrostfieldLab has native window hosts on
macOS, iOS, iPadOS, and visionOS, with an immersive Compositor Services fixture
on visionOS. Release qualification tracks physical devices and presentation
performance separately. Draco point clouds, additional material extensions, and windowed stereo
remain open.
[Metal backend](docs/decisions/metal-backend.md) · [Satin comparison](docs/decisions/renderer-foundation.md)

## Start here

- [Implementation handoff](HANDOFF.md)
- [Implementation plan](tasks/plan.md)
- [Ordered tasks and acceptance criteria](tasks/todo.md)
- [Architecture](docs/architecture.md)
- [Language strategy](docs/language-strategy.md)
- [Validation and performance protocol](docs/validation.md)
- [Contributing](CONTRIBUTING.md)

## Development baseline

| Area | Initial choice |
| --- | --- |
| Language | Swift 6 mode; Swift tools version 6.2 |
| Deployment targets | macOS, iOS, iPadOS, and visionOS 27.0 or later |
| Development toolchain | Xcode 27 with the macOS, iOS, and visionOS 27 SDKs |
| Local agent workflow | Xcode 27 skills and the native Xcode MCP server |
| CI checks | None; builds and tests run locally with Xcode 27.2 |
| Dependencies | None |

Use the native macOS lab for local rendering work. The release gates track
physical device coverage and resource budgets. [Implementation handoff](HANDOFF.md)

Each backend will check Metal capabilities at runtime independently of the OS
deployment target.

## Companion project

[GlassWeave](https://github.com/Mitternacht/GlassWeave) is the independent companion
framework for native and custom glass surfaces. Neither package currently depends on
the other. Shared low-level Metal utilities will be extracted only after both
implementations establish a concrete need; glass must not require a scene engine.

## License

[MIT](LICENSE).
