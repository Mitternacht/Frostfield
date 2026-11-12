# What is left, and how to continue

Written on 2026-11-11 and updated on 2026-11-12, when shader work on `codex/matrix-uniforms` stopped with a clean tree.
`main` is at the merge of pull request 2 (`6667b8c`). Read [AGENTS.md](../AGENTS.md) first, then this page, then the ledger.

## Where things stand

- The ledger [`docs/parity/threejs.md`](parity/threejs.md) tracks 524 capabilities against Three.js r187: **315 done, 141 partial, 68 missing**.
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

Still `codex/matrix-uniforms`, still `xcodebuild` with Metal API Validation. The Xcode bridge was not used. These rows moved to complete: orthographic scene camera, shared camera controls, raster dispersion, `EXT_materials_bump`, primitive indirect draws, and device allocation totals. The ledger is **315 complete, 141 partial, and 68 missing**.

macOS, the focused suites: `Test run with 11 tests in 7 suites passed after 0.237 seconds`. iPad Pro (M5) simulator: `Test run with 11 tests in 7 suites passed after 0.771 seconds`. That simulator does not report support for primitive indirect draws, so `PrimitiveIndirectDrawTests` was skipped there. macOS ran those two tests. After picking and parallel shading were wired to the orthographic camera, `OrthographicSceneCameraTests` passed again: macOS `Test run with 2 tests in 1 suite passed after 0.034 seconds`, iPad simulator `Test run with 2 tests in 1 suite passed after 0.087 seconds`. `OrbitCameraController.look(at:from:)`, `transitionDuration`, and frame padding then closed the smooth-transition row. `CameraControllerTests`: macOS `Test run with 16 tests in 1 suite passed after 0.006 seconds`, iPad simulator `Test run with 16 tests in 1 suite passed after 0.013 seconds`.

## What is left

The ledger is **315 complete, 141 partial, and 68 missing**. Its section "Priorities: done and remaining" is the ordered queue. The next open shader rows, top to bottom:

- Typed values: struct uniforms. Integer, boolean, matrix, and struct uniforms are not animation tracks.
- Node materials: sprite size and the remaining classic and primitive slots.
- Custom shaders: built-in fog, light, and clipping bindings.
- Texture sampling: cube and 3D samples in raster, post, and the path tracer.
- Uniforms: per-frame and per-object update hooks.
- Normals: a bump from an arbitrary scalar, beyond the texture red channel.
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

**materials-textures** (2)
- Toon and cel shaders
- MaterialX shading

**shader-authoring** (10)
- Physical slots: clearcoat, sheen, iridescence, anisotropy, thickness, IOR, specular, dispersion: Layers keep their own factors and maps beside a graph; graphs cannot drive them
- Lighting and environment slots
- Backdrop slots: The transmission backdrop is not exposed to surface graphs; only post-process graphs read the frame
- Attribute and varying nodes: Fixed built-in inputs only
- Custom lighting models: BRDF helpers are internal MSL
- Indirect compute dispatch
- GLSL and ShaderToy transpiler
- Visual graph editor: Graphs are authored in Swift; the Lab's uniform sliders and generated-Metal view are Lab-only
- BRDF and sampling helpers: Internal MSL only
- Layer-based composable materials

**lights-shadows** (3)
- Tiled shadow maps
- Clustered and batched many-light shading: Raster loops over at most 16 punctual lights per fragment; the path tracer takes 16 area and 16 analytic lights with one shadow ray per analytic...
- Accumulative soft shadows: PCSS approximates soft sun shadows; the path tracer converges true soft shadows from area lights and sun disks

**environment-ibl** (2)
- Custom background and environment nodes
- Lightformers

**animation** (2)
- Physics integration
- Visual keyframe authoring

**loaders-assets** (11)
- Data and compressed texture loader bases: `TextureLoader` loads KTX2, DDS, KTX1 and PVR directly. An extensible loader base remains open.
- EXT_materials_bump
- Gaussian splats (KHR_gaussian_splatting, .ply, .splat, .ksplat, .spz)
- FBX import
- Other DCC and CAD importers: The listed Three.js formats are missing. Frostfield separately imports Alembic meshes through ModelIO. `ModelIOSceneLoaderTests`. [Format...
- Fonts
- MaterialX import
- glTF document SDK: The glTF loader and exporter read and write supported extensions. There is no general-purpose glTF document editing API.
- Texture compression pipeline: KTX2 block formats load but nothing encodes them
- Design-tool scene import
- Progressive mesh and texture LOD streaming: `LevelOfDetail` switches preloaded meshes by distance; nothing streams

**post-processing** (3)
- Render-to-texture nodes: No in-graph node; whole graphs render to textures through `ShaderTextureGenerator` or `PostProcessPass`
- Declarative effect chains: `FrostfieldSceneView` has no post chain; `PostProcessChain` is configured in code
- Temporally stable SSAO with denoise

**effects-misc** (8)
- Stereoscopic effects
- Convex object fracture
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

**performance-scale** (11)
- Static object hint
- Pluggable drawing pipeline
- Vertex attribute quantization: glTF `KHR_mesh_quantization` attributes dequantize to Float on load; vertex buffers stay 32-bit
- Automatic batching
- Meshlet pipeline: Metal mesh shaders are unused
- Visibility buffer
- CPU and GPU sorting
- 3D Tiles creased normals
- Adaptive quality
- glTF optimization pipeline: The loader consumes compressed, quantized and WebP output. Mesh simplification is available as a separate operation. An end-to-end glTF optimizer...
- 3D Tiles streaming

**xr-platform** (6)
- Tracked controller models
- Tracked hand models
- AR plane detection visualization
- AR light estimation
- XR text labels
- Engine with components and baked lighting

