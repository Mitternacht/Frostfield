# What is left, and how to continue

Written on 2026-11-11 and updated on 2026-11-12, when shader work on `codex/matrix-uniforms` stopped with a clean tree.
`main` is at the merge of pull request 2 (`6667b8c`). Read [AGENTS.md](../AGENTS.md) first, then this page, then the ledger.

## Where things stand

- The ledger [`docs/parity/threejs.md`](parity/threejs.md) tracks 524 capabilities against Three.js r187: **354 done, 114 partial, 56 missing**.
  Its section "Priorities: done and remaining" is the ordered queue and is current. Do not trust a row until its evidence page and tests agree.
- The last full run passed on macOS (2,183 tests in 426 suites), the iPad Pro (M5) simulator and the Vision Pro simulator (345 tests in 46 suites each).
  Frostfield Lab built for all three, and three Asset Studio captures were byte-identical to the previous ones. All of it ran under Metal API Validation.
- Nothing has run on physical iPhone, iPad or Vision Pro hardware, and nothing has been compared with Three.js on the same hardware and scene.
  Performance and capability parity are separate questions. Keep them separate.
- The owner does not want CI on this repository. Do not add a workflow.
- Photoreal asset authoring and rendering is part of the goal but is not one of the 524 rows. See
  [`docs/evidence/asset-studio.md`](evidence/asset-studio.md) for what exists. Ask the owner what "done" means there before starting.

## How to verify (these commands worked on the owner's Mac)

Use the Xcode MCP first, as AGENTS.md says. It timed out earlier when the disk was full, so `xcodebuild` is the fallback. Say which you used.
Run one Xcode job at a time, for example with `lockf /private/tmp/frostfield-xcode.lock xcodebuild ...`.

```bash
# Package tests. Add -only-testing:FrostfieldTests/<Suite> to focus a run.
MTL_DEBUG_LAYER=1 xcodebuild test -scheme Frostfield-Package -configuration Debug \
  -destination 'platform=macOS' -derivedDataPath <dir> -jobs 3 -parallel-testing-enabled NO
# Simulators used so far: iPad Pro (M5) C131684E-215A-4D00-9C92-C877CE2581C7,
# Vision Pro E9553A5D-1872-435C-993D-DDF211392FCB, iPhone 18 Pro C030382C-A112-43E8-A9C0-F2CEC6DA130D
#   -destination "platform=iOS Simulator,id=<UDID>"   or   "platform=visionOS Simulator,id=<UDID>"

# Lab build. The checkout folder must be named "Frostfield" or the Lab project cannot find the package.
xcodebuild build -project Examples/FrostfieldLab/FrostfieldLab.xcodeproj -scheme FrostfieldLab \
  -configuration Debug -destination platform=macOS -derivedDataPath <dir> CODE_SIGNING_ALLOWED=NO

# Asset Studio captures (compare the PNGs byte for byte with the previous run)
FrostfieldLab --capture-asset-studio out.png --studio-bake-shared-shader      # also --studio-bake-shared-indirect, --studio-bake-indirect
```

The Mac run takes about 3 to 13 minutes, the focused simulator runs about 1 to 5. Builds need several GB of free disk. Some tests skip on purpose
(benchmarks and captures need an environment variable; some GPU features are missing on simulators).

## How the work was done, and what to keep doing

- One lane per capability: a builder implements it with tests, and a separate reviewer who did not write it reads the diff and the logs and tries to break it.
  Reviewers found a real blocker or major problem in every lane so far, so do not skip that step.
- Verify claims against logs. Quote test counts only from a `Test run with N tests` line you read.
- Close a ledger row only with verified public behavior or a documented native equivalent. Otherwise mark it partial and write what is open.
  `setRenderObjectFunction` is left missing on purpose. The reasons are in [`docs/evidence/render-hooks.md`](evidence/render-hooks.md).
- Every new behavior needs a test that fails when the behavior is broken. Several lanes got extra tests after review showed their first ones could not fail.
- Default paths must stay byte-identical when a new option is off. Prove it with stored digests or a before-and-after comparison.
- Update the evidence page, then the ledger row, then recount with `python3 docs/parity/recount-parity.py docs/parity/threejs.md`. It rewrites the summary table. Edit the line "307 of 524 capabilities ..." by hand so it matches.

## Open items left by the lanes finished on 2026-11-11

Each item is also in the ledger row and the evidence page named in brackets.

- **Mesh-shaped fog** ([path-fog-density](evidence/path-fog-density.md)): albedo and anisotropy are constant inside a region; no nested or priority media; no emissive volumes.
  A ray tracks 256 boundary crossings per region. Mesh boundaries always use the software BVH. Temporal path denoising needs a scene without world fog.
  One camera-ray fixture measured 0.569 against 0.599 expected near the crossing limit and the cause was never found; it was replaced by a shadow-ray fixture.
- **Colored glass shadows** ([directional-transmitted-shadows](evidence/directional-transmitted-shadows.md)): spot and point maps, classic receivers, instanced, morphed or vertex-colored stock glass,
  and skinned, morphed, batched or indirect graph casters are open. No fixture covers graph `opacity`, `alphaTest`, nonfinite values or texture reads in `castShadowColor`.
- **Array camera** ([array-camera](evidence/array-camera.md)): single-pass vertex amplification, choosing among overlapping views when picking, and per-view screen-space occlusion,
  indirect draws, toon outlines and motion vectors are open. A later-view failure that depends on state from earlier views cannot be rolled back.
- **KTX2** ([ktx2-export](evidence/ktx2-export.md)): importing cube arrays and Basis cube arrays is open. The native cube-array tests skip on simulators. The cube-array files were validated with a release candidate of the Khronos tools.
- **Dithering** ([dithering](evidence/dithering.md)): it is per output stage, not per material. Meshes drawn straight into an 8-bit target, RGBA16Float destinations and final stages other than tone mapping are not covered.
  The golden digests are tied to an Apple M4 Max.
- **Render hooks** ([render-hooks](evidence/render-hooks.md)): no shadow-pass hooks, scene or material callbacks. The inspector reports no compute or copy events. Toon outline, motion vector and depth-only passes do not call hooks.
- **FABRIK** ([fabrik-chain](evidence/fabrik-chain.md)): one fixed-base chain only. Pole targets, joint constraints and connected chains are open.

## Known bugs and ideas from earlier lanes (check each one first, some may already be fixed)

- A path-traced node whose instance set is empty at build time makes every `updateTransforms` throw `structureChanged`.
- Bloom radius 10 turns a one-pixel highlight into a 5 by 5 dot grid (single 5-tap pass).
- Path-traced ground textures looked blurred or moiré in the Scale Lab capture (512 bake, no anisotropic filtering).
- The glTF exporter does not write morph targets (it only warns). Imported `EXT_mesh_gpu_instancing` expands to per-instance nodes instead of an instance set.
- The path tracer rejects clearcoat normal maps.

## Stopped on 2026-11-12

Branch `codex/matrix-uniforms`. The working tree was clean. Nothing from this stage was pushed. Builds and tests used `xcodebuild` with Metal API Validation, because the Xcode bridge was not available. Counts below come from `Test run with N tests` lines. There has been no new full-package run since the 2026-11-11 baseline.

This stage, newest commit first:

- Declared functions can take and return a graph struct. The body uses the struct's Swift name. Struct uniforms are still unavailable. macOS `ShaderPatternTests`: 18 tests, 0.226 seconds. iPad Pro (M5) simulator: 18 tests, 1.787 seconds.
- An analytic line graph replaces the material width in the material's pixel or world units. A later uniform edit does not build a new pipeline. Hairlines stay one pixel wide. macOS `LineRenderingTests`: 13 tests, 0.443 seconds. iPad simulator, that suite plus `PointsAndSpritesTests`: 26 tests, 4.162 seconds.
- A point graph replaces the material size the same way. macOS `PointsAndSpritesTests`: 13 tests, 0.446 seconds. iPad simulator, that suite plus `LineRenderingTests`: 25 tests, 3.347 seconds.
- The ledger now records the height bump raster and path tracing already evaluate. The height is the texture's red channel.
- Compute graphs sample a 3D texture at normalized coordinates, mip 0. macOS `ShaderPatternTests`: 17 tests, 0.487 seconds. iPad simulator: 17 tests, 3.741 seconds.
- Compute graphs sample a cube map by direction, mip 0. macOS `ShaderPatternTests`: 16 tests, 0.324 seconds. iPad simulator: 16 tests, 2.634 seconds.
- Graphs sample a texture from explicit UV derivatives. macOS pattern suite plus one raster, post, and deformation compile: 18 tests, 3.027 seconds. iPad simulator: 18 tests, 10.060 seconds.
- Graphs sample a texture at an explicit mip. macOS, same four suites: 17 tests, 2.353 seconds. iPad simulator: 17 tests, 9.472 seconds.

Raster, post, and path tracing still bind 2D graph textures. A cube or 3D declaration is rejected there.

Leave these alone unless the owner says otherwise: skeletons and bones, property binding paths, physical-device qualification, and Three.js speed comparisons. The animation lane in the other clone is still the owner of those two rows if it has not merged.

## Continued on 2026-11-12

Still `codex/matrix-uniforms`, still `xcodebuild` with Metal API Validation. The Xcode bridge was not used. These rows moved to complete: orthographic scene camera, shared camera controls, raster dispersion, `EXT_materials_bump`, primitive indirect draws, and device allocation totals. The ledger is **354 complete, 114 partial, and 56 missing**.

macOS, the focused suites: `Test run with 11 tests in 7 suites passed after 0.237 seconds`. iPad Pro (M5) simulator: `Test run with 11 tests in 7 suites passed after 0.771 seconds`. That simulator does not report support for primitive indirect draws, so `PrimitiveIndirectDrawTests` was skipped there. macOS ran those two tests. After picking and parallel shading were wired to the orthographic camera, `OrthographicSceneCameraTests` passed again: macOS `Test run with 2 tests in 1 suite passed after 0.034 seconds`, iPad simulator `Test run with 2 tests in 1 suite passed after 0.087 seconds`. `OrbitCameraController.look(at:from:)`, `transitionDuration`, and frame padding then closed the smooth-transition row. `CameraControllerTests`: macOS `Test run with 16 tests in 1 suite passed after 0.006 seconds`, iPad simulator `Test run with 16 tests in 1 suite passed after 0.013 seconds`. `MeshRenderer.updateVertices(_:startingAt:)` then closed dynamic buffer ranges. `DynamicGeometryTests`: macOS `Test run with 7 tests in 1 suite passed after 0.091 seconds`, iPad simulator `Test run with 7 tests in 1 suite passed after 2.075 seconds`. `SceneNode.receivesShadow` then lets one node stay lit. `ClassicMaterialTests/nodeReceivesShadow`: macOS `Test run with 1 test in 1 suite passed after 0.028 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 0.075 seconds`. A material can now skip the display curve. `ToneMappingTests`: macOS `Test run with 12 tests in 1 suite passed after 0.528 seconds`, iPad simulator `Test run with 12 tests in 1 suite passed after 3.206 seconds`. XYZ Euler angles and a manually held matrix then closed object transforms. `SceneMathTests`: macOS `Test run with 7 tests in 1 suite passed after 0.003 seconds`, iPad simulator `Test run with 7 tests in 1 suite passed after 0.016 seconds`. Separate vertex channels can now be interleaved. `MeshWeldingTests`: macOS `Test run with 11 tests in 1 suite passed after 0.058 seconds`, iPad simulator `Test run with 11 tests in 1 suite passed after 0.086 seconds`. Classic and physically based materials can now draw their edges or shade with the face normal. `WireframeShadingTests`: macOS `Test run with 3 tests in 1 suite passed after 0.212 seconds`, iPad simulator `Test run with 3 tests in 1 suite passed after 0.718 seconds`. A material can now stay hidden, or stay unfogged while the image fogs. `MaterialBaseStateTests`: macOS `Test run with 2 tests in 1 suite passed after 0.193 seconds`, iPad simulator `Test run with 2 tests in 1 suite passed after 0.996 seconds`. CPU noise was already public, and seeded random now matches the Three.js Park-Miller sequence. `SeededRandomTests`: macOS `Test run with 1 test in 1 suite passed after 0.001 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 0.044 seconds`. A blended surface can now fog at its own depth. `FogDepthCaptureTests`: macOS `Test run with 1 test in 1 suite passed after 0.018 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 1.397 seconds`. OBJ files can now keep their polylines and points. `OBJMeshLoaderTests`: macOS `Test run with 7 tests in 1 suite passed after 0.004 seconds`, iPad simulator `Test run with 7 tests in 1 suite passed after 0.019 seconds`. A shader graph can now draw a circle. `ShaderPatternTests`: macOS `Test run with 19 tests in 1 suite passed after 1.954 seconds`. The new circle test on the iPad simulator: `Test run with 1 test in 1 suite passed after 0.778 seconds`. Raster shadow maps and path-traced shadow rays can now cast one side. `ShadowSideTests`: macOS `Test run with 1 test in 1 suite passed after 0.027 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 0.162 seconds`. `PathEmissiveSamplingTests` shadow side: macOS `Test run with 1 test in 1 suite passed after 8.199 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 13.321 seconds`. A material can now dither an 8-bit target. `MaterialDitheringTests`: macOS `Test run with 1 test in 1 suite passed after 0.193 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 0.740 seconds`. A node can now cast with a different mesh than the one it draws. `CustomDepthMeshTests`: macOS `Test run with 1 test in 1 suite passed after 0.021 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 1.021 seconds`. An orbit target can now stay inside a box, and the camera can stay a radius away from mesh colliders. `CameraControllerTests` and `AxisViewHelperTests`: macOS `Test run with 28 tests in 2 suites passed after 0.048 seconds`, iPad Pro (M5) simulator `Test run with 28 tests in 2 suites passed after 1.040 seconds`. The new constraint test is inside that camera suite. An earlier camera-only run on macOS was `Test run with 17 tests in 1 suite passed after 0.005 seconds`, and on the iPad simulator `Test run with 17 tests in 1 suite passed after 0.014 seconds`. A capsule can now move through a mesh and rest on a floor while keeping its slide. `MeshDepenetrationTests`: macOS `Test run with 6 tests in 1 suite passed after 0.004 seconds`, iPad Pro (M5) simulator `Test run with 6 tests in 1 suite passed after 0.010 seconds`. The floor move passed inside that suite after 0.001 seconds on macOS. A directional shadow map now holds eight cascades. `CascadedShadowSplitTests`, `CascadedShadowMapTests`, and `SunLightTests`: macOS `Test run with 15 tests in 3 suites passed after 1.994 seconds`, iPad Pro (M5) simulator `Test run with 15 tests in 3 suites passed after 5.588 seconds`. The eight-cascade fade is one case of the last-cascade test in that run. An IES file can now include a lamp-tilt curve, or take one from a separate document, without scaling the measured candela. `IESProfileTests`: macOS `Test run with 7 tests in 1 suite passed after 0.954 seconds`, iPad Pro (M5) simulator `Test run with 7 tests in 1 suite passed after 3.823 seconds`. The tilt test passed inside that suite after 0.051 seconds on macOS. A tagged image can now be converted into linear Display P3 or Rec. 2020. The default load stays an sRGB texture. `TextureLoaderTests`: macOS `Test run with 8 tests in 1 suite passed after 0.115 seconds`, iPad Pro (M5) simulator `Test run with 8 tests in 1 suite passed after 3.321 seconds`. A mesh can now grow an outline shell, and a moving point can leave a ribbon. `GeometryHelperTests`: macOS `Test run with 2 tests in 1 suite passed after 0.001 seconds`, iPad Pro (M5) simulator `Test run with 2 tests in 1 suite passed after 0.006 seconds`. Daylight, sunset and night panoramas prepare lighting, and one call captures a scene cube into a probe. `EnvironmentPresetTests`: macOS `Test run with 2 tests in 1 suite passed after 0.024 seconds`, iPad Pro (M5) simulator `Test run with 2 tests in 1 suite passed after 1.018 seconds`. A uniform set can now edit itself when a frame or an object is drawn. `ShaderUniformUpdateTests`: macOS `Test run with 1 test in 1 suite passed after 0.001 seconds`, iPad Pro (M5) simulator `Test run with 1 test in 1 suite passed after 0.004 seconds`. Wood can be shaded per UV, and a surface graph can evaluate the ring term. `ProceduralSurfaceTests`: macOS `Test run with 6 tests in 1 suite passed after 0.082 seconds`, iPad Pro (M5) simulator `Test run with 6 tests in 1 suite passed after 0.185 seconds`. The new graph test: macOS `Test run with 1 test in 1 suite passed after 0.098 seconds`, iPad simulator `Test run with 1 test in 1 suite passed after 1.483 seconds`. The scene view can keep drawing while it is active. `RenderScheduleTests`: macOS `Test run with 5 tests in 1 suite passed after 0.001 seconds`, iPad Pro (M5) simulator `Test run with 5 tests in 1 suite passed after 0.005 seconds`. The SwiftUI host compiled in the same runs: macOS `Test run with 3 tests in 1 suite passed after 0.003 seconds`, iPad simulator `Test run with 3 tests in 1 suite passed after 0.015 seconds`. A surface graph can now read authored material values by name. `SurfaceShaderGraphTests` and the named-reference render: macOS `Test run with 6 tests in 2 suites passed after 0.082 seconds`. iPad Pro (M5) simulator, including the archive version check: `Test run with 7 tests in 3 suites passed after 17.809 seconds`. A scalar height can now bump a surface the way a red-channel height texture does. `BumpMapTests/scalarHeight()`: macOS `Test run with 1 test in 1 suite passed after 8.128 seconds`. iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 20.445 seconds`. The raster scene view can now run a post-process chain before display. `DrawableResolutionTests/postChain()`: macOS `Test run with 1 test in 1 suite passed after 0.406 seconds`. The iPad Pro (M5) simulator suite: `Test run with 4 tests in 1 suite passed after 1.295 seconds`. A static node can now keep a captured world matrix. `StaticNodeTests`: macOS `Test run with 3 tests in 1 suite passed after 0.017 seconds`. iPad Pro (M5) simulator: `Test run with 3 tests in 1 suite passed after 0.081 seconds`. Finite float keys can now be sorted on the CPU and the GPU. `SortTests`: macOS `Test run with 2 tests in 1 suite passed after 0.022 seconds`. iPad Pro (M5) simulator: `Test run with 2 tests in 1 suite passed after 0.887 seconds`. Frame times can now step a quality factor and a pixel scale. `PerformanceMonitorTests`: macOS `Test run with 2 tests in 1 suite passed after 0.001 seconds`. iPad Pro (M5) simulator: `Test run with 2 tests in 1 suite passed after 0.005 seconds`. The r187 toon, hatching, and dotted fragment shaders can now be evaluated. `ToonShaderTests`: macOS `Test run with 1 test in 1 suite passed after 0.077 seconds`. iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.758 seconds`. Anaglyph, parallax-barrier, and side-by-side views now follow the r187 stereo effects. `StereoPresentationTests`: macOS `Test run with 1 test in 1 suite passed after 0.020 seconds`. iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.026 seconds`. A TrueType or OpenType file can now supply the faces used by text outlines. `FontFileTests`: macOS `Test run with 1 test in 1 suite passed after 0.010 seconds`. iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.038 seconds`. A buffer graph can now dispatch from threadgroup counts written earlier in the same command buffer. `IndirectDispatchTests` and `BufferShaderGraphTests`: macOS `Test run with 5 tests in 2 suites passed after 0.291 seconds`. iPad Pro (M5) simulator: `Test run with 5 tests in 2 suites passed after 2.132 seconds`. Creased normals for a loaded mesh match the 3D Tiles plugin's `toCreasedNormals` call. `MeshWeldingTests`: macOS `Test run with 11 tests in 1 suite passed after 0.058 seconds`. iPad Pro (M5) simulator: `Test run with 11 tests in 1 suite passed after 0.233 seconds`. A rectangle, circle, ring, or cube can now emit scaled color into an environment capture. `LightformerTests`: macOS `Test run with 1 test in 1 suite passed after 0.315 seconds`. iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 1.610 seconds`. A parser can now supply RGBA8 or ASTC 4×4 and BC1 blocks for upload. `TextureDocumentTests`: macOS `Test run with 1 test in 1 suite passed after 0.006 seconds`. iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.019 seconds`. A convex mesh can now split on a world plane or around an impact. `ConvexFractureTests`: macOS `Test run with 1 test in 1 suite passed after 0.004 seconds`. iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.006 seconds`. Path-traced graphs can now read the camera pixel, and a display clock can step uniform time. `ShaderGraphInputTests` screen and display-clock cases: macOS `Test run with 2 tests in 1 suite passed after 0.073 seconds`. iPad Pro (M5) simulator, with the indirect dispatch test: `Test run with 3 tests in 2 suites passed after 18.852 seconds`.

## What is left

The ledger is **354 complete, 114 partial, and 56 missing**. Its section "Priorities: done and remaining" is the ordered queue. The next open shader rows, top to bottom:

- Typed values: struct uniforms. Integer, boolean, matrix, and struct uniforms are not animation tracks.
- Node materials: sprite size and the remaining classic and primitive slots.
- Custom shaders: built-in fog, light, and clipping bindings.
- Texture sampling: cube and 3D samples in raster, post, and the path tracer.
- Matrices and integers: no `uint`, integer vectors, or `mat2`.
- Built-in PBR: final-color and layer outputs.
- Vertex position: no clip-space vertex node, and graph offset is not combined with morph targets.
- Attributes and varyings: missing. Geometry accessors still lack custom attributes, instance index, and front-facing.
- Screen and time: screen inputs are raster-only, and the renderer does not advance the graph clock.
- Storage buffers, compute atomics and subgroups, texture feedback, asynchronous pipeline compile, and node isolation.
- Physical material slots driven by a graph: missing. Path-traced glTF shading still has the gaps in its row.

Then P2 through P5 in the ledger, then the known bugs and the 2026-11-11 lane gaps above. Do not close a row until its tests and evidence page agree with the new behavior.

## In progress when the 2026-11-11 note was written

An animation lane (a `Skeleton` and `Bone` API, and r187 property-path strings such as `.bones[name].position`) was running in a separate clone. If it is not merged, treat
those two animation rows as open and do not duplicate the work without checking with the owner.

## Suggested order

1. Pick from the ledger's "Priorities: done and remaining" section, top to bottom. Shader authoring (P1) has the most partial rows.
2. Hardware: run the Lab and the capture chain on a real iPhone, iPad and Vision Pro, then measure against Three.js with the same scene, resolution and settings. Record it in a new evidence page.
3. The biggest missing features are in the appendix below. Clipping planes, Gaussian splats, order-independent transparency and FBX import are core or add-on rows that users will notice first.

## Appendix: missing rows by area

Generated from `docs/parity/threejs.md` on 2026-11-11. The ledger stays the source of truth. A note is shown only where the row has one.

**renderer** (6)
- Clipping planes and clipping groups
- Per-object render function override: Not implemented by design. A draw is a prepared immutable `MeshRenderer`, and a replacement would need private per-draw buffers and a tested...
- Direct single-pass output: The mesh pass never tone maps; display output is always a separate `HDRDisplay` or post-process pass
- Gaussian splat rendering
- Order-independent transparency: Per-node back-to-front sort only
- Streamed Gaussian splats mixed with meshes

**cameras-controls** (1)
- Controls base class: `OrbitCameraController` and `FlyCameraController` are separate value types with no shared protocol, enabled flag or event binding

**geometry** (1)
- SDF generation, voxelization and sculpting

**materials-textures** (1)
- MaterialX shading

**shader-authoring** (9)
- Physical slots: clearcoat, sheen, iridescence, anisotropy, thickness, IOR, specular, dispersion: Layers keep their own factors and maps beside a graph; graphs cannot drive them
- Lighting and environment slots
- Backdrop slots: The transmission backdrop is not exposed to surface graphs; only post-process graphs read the frame
- Attribute and varying nodes: Fixed built-in inputs only
- Custom lighting models: BRDF helpers are internal MSL
- GLSL and ShaderToy transpiler
- Visual graph editor: Graphs are authored in Swift; the Lab's uniform sliders and generated-Metal view are Lab-only
- BRDF and sampling helpers: Internal MSL only
- Layer-based composable materials

**lights-shadows** (3)
- Tiled shadow maps
- Clustered and batched many-light shading: Raster loops over at most 16 punctual lights per fragment; the path tracer takes 16 area and 16 analytic lights with one shadow ray per analytic...
- Accumulative soft shadows: PCSS approximates soft sun shadows; the path tracer converges true soft shadows from area lights and sun disks

**environment-ibl** (1)
- Custom background and environment nodes

**animation** (2)
- Physics integration
- Visual keyframe authoring

**loaders-assets** (9)
- EXT_materials_bump
- Gaussian splats (KHR_gaussian_splatting, .ply, .splat, .ksplat, .spz)
- FBX import
- Other DCC and CAD importers: The listed Three.js formats are missing. Frostfield separately imports Alembic meshes through ModelIO. `ModelIOSceneLoaderTests`. [Format...
- MaterialX import
- glTF document SDK: The glTF loader and exporter read and write supported extensions. There is no general-purpose glTF document editing API.
- Texture compression pipeline: KTX2 block formats load but nothing encodes them
- Design-tool scene import
- Progressive mesh and texture LOD streaming: `LevelOfDetail` switches preloaded meshes by distance; nothing streams

**post-processing** (2)
- Render-to-texture nodes: No in-graph node; whole graphs render to textures through `ShaderTextureGenerator` or `PostProcessPass`
- Temporally stable SSAO with denoise

**effects-misc** (6)
- Volumetric god rays
- Volumetric clouds
- Volumetric spotlights and caustics: No volumetric lights; path-traced caustics are tested only off a mirror (`PathSamplingTests`), not through transmission
- Text rendering (SDF and 3D)
- Physics engines
- Navmesh and crowds

**path-tracing** (1)
- Voxel cone-traced GI

**interaction-tools** (3)
- Sculpting
- Interactive color-grading tools
- Editor gizmos, grids and helpers

**performance-scale** (7)
- Pluggable drawing pipeline
- Vertex attribute quantization: glTF `KHR_mesh_quantization` attributes dequantize to Float on load; vertex buffers stay 32-bit
- Automatic batching
- Meshlet pipeline: Metal mesh shaders are unused
- Visibility buffer
- glTF optimization pipeline: The loader consumes compressed, quantized and WebP output. Mesh simplification is available as a separate operation. An end-to-end glTF optimizer...
- 3D Tiles streaming

**xr-platform** (6)
- Tracked controller models
- Tracked hand models
- AR plane detection visualization
- AR light estimation
- XR text labels
- Engine with components and baked lighting

