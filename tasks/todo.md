# Frostfield implementation tasks

Follow [the plan](plan.md) and [validation requirements](../docs/validation.md). Use the native Xcode MCP workflow for project edits, builds, runs, diagnostics, and tests. Proposed paths are provisional; adjust them when a working slice reveals better boundaries. If a slice grows beyond five files or one focused session, split it before implementation.

## FF-01 — Create the package and build baseline

The package foundation is recorded in [build baseline evidence](../docs/evidence/build-baseline.md).

**Acceptance criteria**
- [x] `Package.swift` declares tools version 6.2, Swift 6 language mode, iOS, macOS, and visionOS 27.0, and one `Frostfield` library product with no GlassWeave dependency.
- [x] The target was bootstrapped as a buildable module without speculative public APIs, fake renderer behavior, or placeholder tests.
- [x] The chosen Xcode/SDK and successful package builds are recorded; a hosted CI workflow was configured for compilation checks and later removed because the project does not use CI.

**Verification:** Use Xcode MCP to open/build the package on an available destination, inspect diagnostics, and record toolchain/version. Verify manifest platform parsing with that toolchain. Record compiler version, SDK, Swift language mode, default actor isolation, and the selected MSL dialect separately; follow `docs/language-strategy.md`. CI must choose an Xcode version supporting the declared SDKs, fail clearly when unavailable, and avoid claiming device performance. Add tests when real behavior exists.

**Dependencies:** None. **Scope:** M, four files.
**Likely files:** `Package.swift`, `Sources/Frostfield/Frostfield.swift`, `.github/workflows/ci.yml`, `docs/evidence/build-baseline.md`.

## FF-02 — Render one triangle across the native hosts

Display the same Metal output in native macOS, iOS/iPadOS, and visionOS
windows. The lab started with `MTKView` and now uses `CAMetalDisplayLink`.

See [triangle fixture evidence](../docs/evidence/triangle-fixture.md).

**Acceptance criteria**
- [x] A colored triangle appears in the native macOS lab and on the iPhone and Apple Vision Pro simulators.
- [ ] A colored triangle appears on a physical iPhone or iPad and on Apple Vision Pro using the package's renderer. Deferred to platform qualification.
- [x] A missing Metal device or pipeline error produces an explicit failure state instead of a crash or blank success.
- [x] Shader source is packaged reproducibly and the example resolves the correct library.

**Verification:** Xcode build and run. Inspect the triangle and failure states and retain the destination, OS, build record, and capture. Record Mac, simulator, and mobile hardware checks separately.

**Dependencies:** FF-01. **Scope:** M, five files.
**Likely files:** `Examples/FrostfieldLab/FrostfieldLab.xcodeproj/project.pbxproj`, `Examples/FrostfieldLab/App.swift`, `Examples/FrostfieldLab/SceneViewController.swift`, `Sources/Frostfield/Renderer.swift`, `Sources/Frostfield/Shaders/Triangle.metal`.

## FF-03 — Make the host obey its lifecycle

Add resize, inactive/hidden pause, and explicit requested-versus-continuous drawing to the triangle fixture.

See [lifecycle evidence](../docs/evidence/render-lifecycle.md).

**Acceptance criteria**
- [x] The Mac lab redraws after resize, pauses when minimized, and keeps drawable size and submission state independent for two windows.
- [x] Swift Testing covers invalidation, pause and resume, a bounded missing drawable retry, and the single frame slot.
- [ ] Each host updates drawable dimensions using its own view and window context. Verify Mac resize and backing scale changes, UIKit rotation and safe areas, and iPadOS and visionOS window resizing.
- [ ] Inactive or hidden content submits no recurring work; a still scene redraws only after invalidation.
- [x] The windowed hosts take drawables from the display link callback and keep
  one command buffer in flight. They skip drawables with stale dimensions.

**Verification:** Xcode run through resize, background/foreground, and hide/show sequences; inspect submission counters and GPU trace; run focused policy tests for pause/resume and backpressure transitions. With two scenes, resize and deactivate one while confirming the other's scale, resources, and rendering continue independently; exercise display changes where hardware permits.

**Dependencies:** FF-02 native Mac render check. **Scope:** M, five files.
**Likely files:** `Sources/Frostfield/Renderer.swift`, `Sources/Frostfield/RenderSchedule.swift`, `Examples/FrostfieldLab/FrostfieldLab/TriangleViewController.swift`, `Tests/FrostfieldTests/RenderScheduleTests.swift`, `Package.swift`.

### Checkpoint A — Host foundation

- [ ] FF-02–FF-03 evidence is recorded, the example builds, and no lifecycle defect is carried into backend comparisons.

## FF-04 — Compare the Metal 4 command path

Timebox a Metal 4 version of the same triangle to one focused spike. Keep it experimental until the support and lifetime costs are understood.

[Metal backend decision and reproducible probe](../docs/decisions/metal-backend.md)

**Acceptance criteria**
- [x] The report records API/GPU requirements and a runtime capability decision, including unsupported devices.
- [x] The offscreen comparison matches the conventional renderer byte for byte on the M4 Max.
- [ ] Identical visuals are compared with conventional Metal on the same physical device and settings.
- [ ] The report explains allocator reuse, residency, synchronization, complexity, and the chosen next step; missing measurements remain explicitly missing.

**Verification:** Xcode build/run plus Metal validation and a trace for each available backend; compare screenshots and counters. If suitable hardware is unavailable, retain the conventional path and leave performance conclusions open.

**Dependencies:** FF-03. **Scope:** M, three files.
**Likely files:** `Examples/Experiments/Metal4Triangle.swift`, `Examples/FrostfieldLab/SceneViewController.swift`, `docs/decisions/metal-backend.md`.

## FF-05 — Compare Satin with the minimal renderer

Timebox a Satin-backed version of the same small scene to one focused spike behind a local experimental interface. Investigate a dependency without making it part of the package's public contract. The original Hi-Rez repository was archived on April 15, 2025; evaluate the Fabric-Project candidate instead and recheck its maintenance at implementation time. Its currently inspected manifest uses Swift tools 6.2, Swift 5 language mode, and C++17, so explicitly test the Swift 6 consumer/concurrency boundary. Follow [the research spike scorecard](../docs/research-spikes.md).

[Satin feasibility and adoption decision](../docs/decisions/renderer-foundation.md)

**Acceptance criteria**
- [ ] Both candidates render one transformed indexed mesh using a custom MSL shader, export the resulting offscreen texture, and present it through the same UIKit host at equivalent settings; these are experimental operations, not production APIs.
- [ ] The comparison verifies the exported texture's readiness/lifetime and a Swift 6 clean-consumer build, and records packaging, license obligations, dependency footprint, maintenance, and measured cost.
- [x] Satin remains an isolated experiment and the documented adoption decision keeps it out of the package.

**Verification:** Xcode build/run of the isolated experiment in a clean Swift 6 consumer; inspect the custom-shader output and exported texture using a fixed color/geometry fixture, including read-before-write/reuse hazards. Inspect the resolved dependency revision and upstream license; collect a matched trace or mark measurement unavailable. If the one-session feasibility pass cannot establish these operations, record the blocker and repair scope; do not pass the dependency-adoption gate based on a built-in triangle alone.

**Dependencies:** FF-03; use FF-04's findings when choosing the production path. **Scope:** M, four files.
**Likely files:** `Examples/Experiments/SatinComparison/Package.swift`, `Examples/Experiments/SatinComparison/Comparison.swift`, `Examples/Experiments/SatinComparison/Package.resolved`, `docs/decisions/renderer-foundation.md`.

### Checkpoint B — Backend choice

- [ ] FF-04–FF-05 leave a reasoned implementation choice and no speculative public abstraction. Baseline device coverage and the first benchmark fixture are named.

## FF-06 — Move a primitive through a camera

Add only the transform and camera data needed to position the existing primitive. Establish a consistent coordinate convention before a larger scene graph.

[Programmable scene evidence](../docs/evidence/programmable-scene.md)

**Acceptance criteria**
- [x] One primitive can translate, rotate, and scale under a perspective camera using app-supplied state.
- [x] Coordinate handedness, matrix order, depth range, and near/far validation are documented.
- [x] Camera and transform updates reuse the prepared mesh and pipeline.

**Verification:** Xcode build/run of deterministic poses; focused Swift Testing cases for transform composition and invalid projection inputs; compare known screen positions. Verify the first uploaded transform record with host size/alignment/offset/stride assertions and a distinctive-value GPU round-trip; split this into a prerequisite subtask if packaging the shared layout exceeds this slice.

**Dependencies:** Checkpoint B. **Scope:** M, five files.
**Likely files:** `Sources/Frostfield/Transform.swift`, `Sources/Frostfield/Camera.swift`, `Sources/Frostfield/Renderer.swift`, `Tests/FrostfieldTests/TransformTests.swift`, `Examples/FrostfieldLab/CameraFixture.swift`.

## FF-07 — Render indexed procedural geometry

Introduce one indexed mesh representation and a procedural cube fixture. Reuse the camera and transform path rather than adding an asset loader.

**Acceptance criteria**
- [x] A cube with consistent winding and normals renders correctly under the documented coordinate convention.
- [x] Invalid indices and nonfinite vertex data fail before GPU encoding. The typed vertex API fixes the attribute layout.
- [x] Repeated transform-only updates reuse the mesh buffers.

**Verification:** Xcode build/run with wireframe or diagnostic coloring; focused geometry tests for index bounds and normals; inspect buffer-allocation counters during animation.

**Dependencies:** FF-06. **Scope:** M, four files.
**Likely files:** `Sources/Frostfield/Geometry.swift`, `Sources/Frostfield/Renderer.swift`, `Tests/FrostfieldTests/GeometryTests.swift`, `Examples/FrostfieldLab/GeometryFixture.swift`.

## FF-08 — Supply a custom material shader

Create the first material binding contract from a working built-in unlit material and a caller-supplied shader fixture.

**Acceptance criteria**
- [x] An app supplies a shader function and documented parameters to shade the existing mesh.
- [x] Missing functions and incompatible bindings produce actionable errors. Parameter updates reuse the prepared pipeline.
- [x] The custom and built-in material paths use the same encoding and lifetime rules.

**Verification:** Xcode build/run with a time-varying color material; exercise invalid-function and binding cases; confirm no pipeline creation during warmed-up parameter animation.

**Dependencies:** FF-07. **Scope:** M, five files.
**Likely files:** `Sources/Frostfield/Material.swift`, `Sources/Frostfield/PipelineCache.swift`, `Sources/Frostfield/Renderer.swift`, `Sources/Frostfield/Shaders/Unlit.metal`, `Examples/FrostfieldLab/MaterialFixture.swift`.

### Checkpoint C — Standalone v0.1 qualification

- [x] FF-01–FF-08 render a deterministic programmable mesh scene with camera/transforms and unlit/custom shaders, pass focused tests and GPU validation, and expose only API justified by the fixture.
- [ ] A clean consuming app resolves only Frostfield at the candidate revision, loads packaged shaders, and runs native macOS, iOS/iPadOS, and visionOS windowed fixtures without sibling repositories, development-only paths, or GlassWeave. Record installation, error and lifecycle behavior, and the scoped supported-device matrix.
- [ ] Run the physical-device recipe in `docs/validation.md` on this core fixture, establish and meet named-device CPU/GPU/memory/idle budgets, and complete that document's release checklist. Save scoped evidence as `docs/evidence/v0.1-core-qualification.md`; record any unavailable measurements as incomplete gates.

This is the first usable release boundary. A limited v0.1 may ship here independently of GlassWeave and FF-09–FF-15. Its scope is camera/transforms, indexed geometry, unlit/custom material shaders, and native windowed sample hosts. The later effects and integrations have separate qualification gates. The experimental export in FF-05 validates a candidate backend. This qualification needs its own measured baseline. Split fixes discovered by the qualification into focused tasks before accepting the gate.

The following tasks are optional later milestones and do not block the limited v0.1.

## FF-09 — Light one mesh

Add a single directional Lambert light as the first composable lighting slice.

**Acceptance criteria**
- [x] The prepared Lambert material renders a directional-light cube in the native Mac fixture.
- [x] GPU readback verifies linear light intensity for front and back lighting.
- [x] The rotated, non-uniformly scaled cube matches the expected transformed normal.
- [x] The API documents the directional Lambert model and its parameters.

**Verification:** Xcode build/run at known light directions and scales; capture reference images with fixed camera, exposure, and shader parameters.

**Dependencies:** FF-08. **Scope:** M, three files.
**Likely files:** `Sources/Frostfield/DirectionalLight.swift`, `Sources/Frostfield/Shaders/Lambert.metal`, `Examples/FrostfieldLab/LightingFixture.swift`.

## FF-10 — Animate one bounded particle emitter

Implement one deterministic emitter with a fixed maximum particle capacity, reusing existing scheduling and material conventions.

**Acceptance criteria**
- [x] The seeded, fixed-capacity GPU emitter renders in the expanded lab.
- [x] A seeded emitter animates a fixed-capacity set of particles through a compute update and draw sequence.
- [x] The emitter uses one queue and a persistent state buffer. GPU readback tests verify equal seeds at equal time steps.
- [x] Zero delta skips the update and large deltas are capped. The lab pauses its simulation with its draw schedule.

**Verification:** Xcode build/run plus GPU validation; compare equal-seed/equal-time reference frames; inspect dispatch ordering and memory during a sustained run.

**Dependencies:** FF-08 and Checkpoint C's render-pass contract. **Scope:** M, four files.
**Likely files:** `Sources/Frostfield/ParticleEmitter.swift`, `Sources/Frostfield/Shaders/Particles.metal`, `Sources/Frostfield/Renderer.swift`, `Examples/FrostfieldLab/ParticleFixture.swift`.

### Checkpoint D — Lighting and animation

- [ ] FF-09–FF-10 remain optional scene features, preserve idle scheduling, and introduce no unexplained memory growth or GPU validation errors.

## FF-11 — Apply one post-process

Add a single full-screen color-grade pass to an offscreen scene render. This establishes pass composition without designing a general render-graph system.

**Acceptance criteria**
- [x] The grade can be enabled, bypassed, and parameterized without changing geometry or materials.
- [x] The lab reuses a separate offscreen texture until drawable size changes.
- [x] GPU readback verifies the bypass colors and premultiplied alpha.

**Verification:** Xcode build/run with color-chart and moving-scene fixtures; inspect pass ordering and allocations; compare bypass output against the direct path.

**Dependencies:** FF-08. **Scope:** M, four files.
**Likely files:** `Sources/Frostfield/PostProcess.swift`, `Sources/Frostfield/Shaders/ColorGrade.metal`, `Sources/Frostfield/Renderer.swift`, `Examples/FrostfieldLab/PostProcessFixture.swift`.

## FF-12 — Export a texture with a lifetime contract

Expose a narrow offscreen-output path suitable for an external consumer such as GlassWeave, independently of a UI host.

**Acceptance criteria**
- [x] Output specifies device, dimensions, format, color and alpha convention, UV mapping, and frame identity.
- [x] Delayed producer and consumer GPU tests verify that leases prevent reuse until completion.
- [x] The pool handles saturation and resize. Retained old frames remain valid, and foreign output frames are rejected.

**Verification:** Xcode build/run a delayed-consumer fixture with GPU validation; focused lifetime-policy tests; use known color quadrants and frame stamps to check orientation, color, and ordering.

**Dependencies:** FF-08; use FF-11's offscreen path if available. **Scope:** M, five files.
**Likely files:** `Sources/Frostfield/TextureOutput.swift`, `Sources/Frostfield/Renderer.swift`, `Tests/FrostfieldTests/TextureLifetimeTests.swift`, `Examples/FrostfieldLab/TextureConsumerFixture.swift`, `docs/architecture.md`.

## FF-13 — Package an optional SwiftUI host

Offer a thin reusable adapter around the proven native views while retaining each host's renderer and scheduler across ordinary SwiftUI updates.

**Acceptance criteria**
- [x] A native button changes scene state through the retained renderer and on-demand SwiftUI host.
- [ ] Identity changes, resizing, and teardown create and release the expected number of renderer instances.
- [x] The native overlay responds to iPhone touch and Mac accessibility actions. SwiftUI lives in its own package target.

**Verification:** Load matching Xcode SwiftUI guidance; Xcode build/run a state-update and navigation fixture; inspect instance/submission counters and native overlay interaction.

**Dependencies:** FF-03, FF-08. **Scope:** M, four files.
**Likely files:** `Package.swift`, `Sources/FrostfieldUIKit/FrostfieldView.swift`, `Sources/FrostfieldSwiftUI/FrostfieldSceneView.swift`, `Examples/FrostfieldLab/SwiftUIFixture.swift`.

### Checkpoint E — App integration

- [ ] FF-11–FF-13 preserve color/lifetime contracts and the core can render independently of both UI frameworks. Proposed target extraction is verified by package dependency inspection.

## FF-14 — Establish a sustained resource budget

Instrument and profile the expanded reference fixture for a later release, then fix only the largest evidenced resource or scheduling problem within this slice. Split additional fixes into new tasks. This supplements the scoped v0.1 measurements at Checkpoint C; it does not postpone or replace them.

The [windowed metrics recipe](../docs/evidence/render-metrics.md) records
an inactive-window check. A [ten-minute Mac offscreen run](../docs/evidence/offscreen-profile.md)
measured the combined effects workload without a drawable. Presented frame
intervals and the full release matrix remain open.

**Acceptance criteria**
- [ ] The record contains the required physical-device timing, memory, thermal, idle, and interaction measurements.
- [ ] Frame-rate/quality choices and numeric acceptance budgets are set from evidence for named devices; unmet budgets remain open.
- [ ] Warmed-up rendering has no unexplained allocation growth or shader/pipeline compilation; any justified recurring work is documented.

**Verification:** Execute the reproducible recipe in `docs/validation.md`; capture before/after traces for any fix; rerun only affected correctness fixtures.

**Dependencies:** FF-09–FF-13. **Scope:** M, four files.
**Likely files:** `Sources/Frostfield/RenderMetrics.swift`, `Sources/Frostfield/Renderer.swift`, `Examples/FrostfieldLab/BenchmarkFixture.swift`, `docs/evidence/initial-device-profile.md`.

## FF-15 — Feed a Frostfield frame to GlassWeave

Use an example app with local package dependencies to connect the two real implementations without adding a library dependency cycle.

The sibling GlassWeave checkout at `181c846` contains planning documents.
The custom-surface integration and external texture consumer remain dependencies.

**Acceptance criteria**
- [ ] A Frostfield-rendered moving texture drives a custom GlassWeave surface while native controls remain responsive.
- [ ] The fixture verifies color, UV, readiness, and completion against FF-12; it exercises resize and delayed consumption.
- [ ] The report identifies actual duplicated utilities and either justifies a future extraction or records that none is needed.

**Verification:** Xcode build/run the combined fixture with GPU validation; compare known color/frame-stamp cases and record a physical-device trace. Keep cross-repository changes in a separate coordinated task if necessary.

**Dependencies:** FF-12, FF-14, and GlassWeave's custom-surface integration proof. **Scope:** M, four files in this repository.
**Likely files:** `Examples/Interoperability/Interoperability.xcodeproj/project.pbxproj`, `Examples/Interoperability/App.swift`, `Examples/Interoperability/TextureBridge.swift`, `docs/evidence/glassweave-interoperability.md`.

### Checkpoint F — Evidence

- [ ] FF-14–FF-15 are measured on physical devices, limitations are public, and no shared infrastructure is extracted solely for anticipated reuse.

### Checkpoint G — Expanded release qualification

- [ ] A clean checkout resolves the documented products and runs the example; package/CI checks and the physical-device release matrix pass.
- [ ] README and release notes separate working features from the roadmap, link evidence, and state supported OS/device/backend combinations precisely.
- [ ] Dependency license obligations, public API docs, failure behavior, and known limitations match the shipped contents. Complete the release checklist in `docs/validation.md` before tagging.

Run this checkpoint when a later release advertises the expanded effects/integration scope. Checkpoint C already permits a qualified, standalone limited v0.1; GlassWeave interoperability never blocks a Frostfield-only release that does not advertise it. Split any corrective implementation work into additional focused tasks. Every published capability needs working code and linked evidence; unimplemented roadmap work stays unchecked.

## FF-16 — Compose and pick a scene

- [x] Nested nodes inherit parent transforms and preserve their identities.
- [x] Prepared meshes share one render pass and depth test. The picker agrees with the rendered overlap test.
- [x] The lab highlights the picked cube. Mac click and simulator touch both select a mesh.
- [x] Draw points, lines and sprites in the scene color pass, with ordering,
  matching multisample attachments, and screen-space primitive picking.
  [Drawable verification](../docs/evidence/pointslines.md)
- [x] Select geometry centers and instance origins with a screen rectangle.
- [x] Query every mesh hit and route native pointer events to direct group children.
- [x] Return ordered mesh, point, line and sprite hits in a shared viewport query.
  [Selection and input verification](../docs/evidence/animation-and-tools.md#selection-and-input)
- [x] Raycast all drawables in world space and report material-side-aware mesh normals and UVs.
  [World raycast verification](../docs/evidence/primitive-world-raycasts.md)

[Scene graph and picking evidence](../docs/evidence/programmable-scene.md#scene-graph-and-picking)

## FF-17 — Load and animate textured assets

- [x] Image loading produces oriented sRGB textures with a defined UV and
  alpha convention.
- [x] OBJ and glTF import return indexed triangle geometry. Both assets
  render with textures in the Mac lab.
- [x] Static glTF import resolves the scene's hierarchy, geometry, and optional
  opaque base-color and metallic-roughness material data.
- [x] Procedural planes and UV spheres have tested indices, winding, and UVs.
- [x] Transform tracks target stable scene identities and interpolate a
  looping graph animation in the asset fixture.

[Asset fixture evidence](../docs/evidence/programmable-scene.md#asset-and-material-fixtures)

## FF-18 — Add directional shadows and physically based shading

- [x] The lab renders a scene with a directional depth map and verifies
  actual caster-to-receiver visibility in GPU readback.
- [x] Textured metallic-roughness shading renders on the Mac. GPU readback
  verifies material parameter response, premultiplied alpha, and shadow
  reception.

[Material fixture evidence](../docs/evidence/programmable-scene.md#asset-and-material-fixtures)

## FF-19 — Render a visionOS immersive scene

- [x] The renderer accepts a caller's projection, eye position, and viewport
  and supports reversed depth. A GPU test verifies occlusion and viewport bounds.
- [x] The opt-in Compositor Services fixture presents a view on the Apple
  Vision Pro Simulator.
- [ ] Verify native launcher interaction, dismissal, physical stereo, predicted
  head tracking, and comfort on Vision Pro.

[Immersive fixture evidence](../docs/evidence/programmable-scene.md#visionos-immersive-preview)

## FF-20 — Import and render animated glTF assets

- [x] Import node TRS channels, inverse bind matrices, and vertex influences.
- [x] Reuse prepared meshes while sampling clips and updating joint palettes.
- [x] Deform color and shadow draws on the GPU and pick the posed geometry.
- [x] Prepare core PBR maps, sampler modes, and alpha coverage on a Metal device.

## FF-21 — Compose programmable materials, surface flow, and screen effects

- [x] Compile validated typed material graphs into Metal fragments.
- [x] Sample mesh surfaces by area and drive deterministic GPU flow particles.
- [x] Apply bloom, chromatic aberration, and film grain to offscreen scene color.

## FF-22 — Run the Cyber Ocean reference scene

- [x] Render the imported dolphin's swim cycle, procedural tunnel, particle
  flow, and screen effects together on the Mac GPU.
- [x] Verify native play, pause, and orbit on iPhone Simulator, and render
  the scene on iPad and Vision Pro simulators.
- [ ] Qualify sustained presentation and interaction on the release hardware.

[Cyber Ocean render evidence](../docs/evidence/cyber-ocean.md)

## FF-23 — Render a Blender asset under HDR lighting

- [x] Generate diffuse irradiance, prefiltered glossy reflections, and a visible
  HDR environment from the same linear panorama.
- [x] Preserve HDR highlights through contact-hardening shadows, lens blur, and
  tone mapping. Verify shadow and lens behavior with Mac GPU readback.
- [x] Render the Blender-exported camera on the Mac and in iPhone, iPad,
  and Vision Pro simulators with Metal API Validation.
- [ ] Qualify sustained orbit and resource use on physical iPhone, iPad, and
  Vision Pro hardware.

## FF-24 — Retain authored lighting and texture coordinates

- [x] Apply glTF tangents and either UV set to the relevant material maps, including
  alpha-masked shadow casters.
- [x] Import and render a light rig with photometric punctual lights, including
  spotlight cones and distance attenuation.

[Blender asset studio evidence](../docs/evidence/pbr-studio.md)

## FF-25 — Render layered finishes and authored camera views

- [x] Import clearcoat with independent factor, roughness, and normal maps.
  Verify direct and environment lighting, and compare the coated asset to its base.
- [x] Preserve glTF perspective and orthographic framing, with camera poses
  resolved from the current graph and coverage for both Metal depth modes.
- [x] Offer optional contact occlusion with GPU tests for both depth modes.
- [x] Preview a local glTF or GLB subject using the same Mac studio fixture.

## FF-26 — Render imported optical transmission and volume

- [x] Parse and prepare transmission, volume, and index of refraction while
  preserving independent map channels, samplers, UV sets, and alpha coverage.
- [x] Preserve opaque scene depth and snapshot HDR color for roughness-filtered
  refraction and thickness-based absorption. Verify the snapshot with GPU readback.
- [x] Render an authored glass model on Mac and the iPhone, iPad, and Vision Pro
  simulators under Metal API Validation.
- [ ] Profile refractive scenes on physical iPhone, iPad, and Vision Pro hardware.

[Clearcoat and glass evidence](../docs/evidence/pbr-studio.md#layered-finishes-and-glass)

## FF-27 — Generate meshes and physical surfaces from code

- [x] Build lathes, rounded boxes, concave extrusions, swept tubes, and
  parametric surfaces with reusable indexed geometry, normals, UVs, and tangents.
- [x] Generate mipmapped PBR maps from seeded presets and custom UV functions.
- [x] Verify mesh winding, caps, seams, and material channel contents.
- [x] Build vector text, filled shapes, decals, convex hulls and closed Boolean
  solids. Merge indexed parts and generate tangent frames with source remapping.
  [Shape verification](../docs/evidence/geoshapes.md) · [Tangent verification](../docs/evidence/tangent-generation.md)

## FF-28 — Render a fully procedural product scene

- [x] Build the rangefinder's body, optics, controls, lettering, and finishes
  and light it with a generated studio environment.
- [x] Compare native Mac captures with the bundled sample assets removed.
- [x] Render the same procedural scene in iPhone, iPad, and Vision Pro simulators.
- [ ] Qualify procedural generation and orbit performance on physical mobile
  and headset hardware.

[Procedural camera source and evidence](../docs/evidence/procedural-camera.md)

## FF-29 — Trace light through a static PBR scene

- [x] Prepare world-space geometry, textures, materials, and bounded BVH buffers.
- [x] Sample rectangular emitters, diffuse and specular bounces, and dielectric
  boundaries in linear HDR. Verify an off-camera reflection across two bounces.
- [x] Compare direct and indirect lighting under the same camera and emitter.

## FF-30 — Accumulate and present a photoreal still or preview

- [x] Reuse successful sample histories and reset on view changes. Verify batch
  equivalence and enforce accumulation ownership.
- [x] Filter first-hit guides and capture the product scene on the Mac GPU.
- [x] Render the progressive preview in iPhone, iPad, and Vision Pro simulators.
- [x] Trace large-image regions with the same camera rays, guides, and samples
  as a full-frame render. [Region verification](../docs/evidence/path-output.md)
- [x] Write first-hit textures and export mixed half, float, and integer layers
  in one OpenEXR image. [Output verification](../docs/evidence/path-output.md)
- [x] Recommend samples per dispatch from completed GPU timings within caller
  bounds. [Frame-budget verification](../docs/evidence/path-output.md)
- [ ] Qualify sustained path tracing and resource use on physical mobile and headset hardware.

[Path tracing evidence and reproduction commands](../docs/evidence/path-tracing.md)

## FF-31 — Relight a prepared path tracing scene

- [x] Supply a per-dispatch light rig, including aimed rectangular emitters
  with a power or radiance input.
- [x] Reject stale accumulation history after a lighting change and verify
  that a reset renders the new lighting with the same prepared scene.
- [x] Verify the native studio controls and deterministic Reset in the simulator.

## FF-32 — Generate PBR shader code from typed graphs

- [x] Emit inspectable Metal for selected material channels with bounded graph
  validation and procedural math nodes.
- [x] Use the same generated functions in raster PBR and path tracing, and
  verify matching material output across the renderers.
- [x] Capture a generated finish under two editable studio light rigs.

[Lighting and shader evidence](../docs/evidence/lights-and-shaders.md)

## FF-33 — Render procedural frost and clear ice on one iceberg

- [x] Let generated surface graphs scale transmission and verify clear and
  opaque regions on a closed mesh in the Metal path tracer.
- [x] Generate an iceberg, ocean, and cloudy sky from code and capture the
  path-traced scene on a native Mac.

[Iceberg source and capture](../docs/evidence/iceberg-ocean.md)

## FF-34 — Author shaders as typed graphs in both renderers

Ordered by the [Three.js parity checklist](../docs/parity/threejs.md).

- [x] Validate typed scalar and vector graphs with stage-checked inputs, uniforms,
  textures, and caller-written Metal functions, and describe them with a Swift builder.
- [x] Drive raster PBR and the path tracer from the same graph, including
  uniforms, textures, tangent-space normals, and occlusion; verify matching
  output and that uniform edits reuse compiled libraries.
- [x] Report compiler errors with the generated line and emit inspectable Metal.
- [x] Evaluate branches, bounded loops, matrices, and camera inputs in both
  renderers. Evaluate screen derivatives in fragment stages.
  [Graph verification](../docs/evidence/graphlang.md)

## FF-35 — Deform meshes, post-process frames, and compute with graphs

- [x] Offset vertices from a surface graph on the GPU with rebuilt normals in
  color, shadow, picking, and path-tracing snapshots.
- [x] Compose bypassable post-process graph passes with the existing effects.
- [x] Generate textures, PBR maps, and buffer updates from compute graphs.

## FF-36 — Author light rigs and shadow local lights

- [x] Shadow spot, point, and rectangle lights in raster PBR and render
  rectangular area lights with exact diffuse irradiance.
- [x] Importance-sample HDR environment maps and add analytic point, spot, and
  directional lights to the path tracer, matching raster falloff.
- [x] Convert one `LightRig` with power, size, color temperature, and aim to both
  renderers and verify matching radiance; keep light, geometry, and exposure edits distinct.

## FF-37 — Mix animation, control cameras, pick quickly, and export

- [x] Blend, cross-fade, and layer clips from Swift and glTF; import morph target data.
- [x] Provide orbit and fly camera controllers, BVH ray queries, bounds, and frustum tests.
- [x] Export imported scenes and scene graphs as glTF and GLB with loss reporting.
- [x] Bind animation to material and node properties, import supported glTF
  animation pointers, and use compute skinning for path-tracing updates.
  [Animation verification](../docs/evidence/animtools.md)

## FF-39 — Import extended materials and compressed assets

- [x] Render texture transforms, emissive strength, specular, sheen, anisotropy,
  iridescence, dispersion, and unlit materials in raster PBR and the path tracer,
  and select glTF material variants.
- [x] Load quantized and meshopt-compressed meshes, KTX2 block-compressed
  textures, WebP, AVIF, and glTF GPU instancing data.
- [x] Export USDZ, import Model I/O assets, and read and write HDR image data
  with the limits recorded in [format evidence](../docs/evidence/formats.md).
- [x] Decode Draco triangle meshes and decode ETC1S and UASTC KTX2 to RGBA8,
  including required glTF extensions. [Draco evidence](../docs/evidence/draco.md) ·
  [Basis evidence](../docs/evidence/basis-universal.md)
- [ ] Decode Draco point clouds and triangle strips and transcode Basis textures
  to device-native compressed formats.

[Materials and assets](../docs/evidence/materials-and-assets.md)

## FF-40 — Draw large scenes and update the path tracer live

- [x] Draw instanced meshes with per-instance tints, blend morph targets on the
  GPU, cull by frustum, and select distance LODs in every pass.
- [x] Draw different indexed geometries in shared Metal batches with per-slot visibility and depth order.
  [Batch verification](../docs/evidence/batched-mesh.md)
- [x] Edit batch geometry palettes and supply validated custom order for each pass.
  [Batch verification](../docs/evidence/batched-mesh.md)
- [x] Reuse Metal command bundles for eligible batch pipelines and consume GPU-written indexed arguments in standalone draws.
  [Batch bundles](../docs/evidence/batched-mesh.md) · [Indirect arguments](../docs/evidence/mesh-indirect-draws.md)
- [x] Query per-node fragment visibility from the normal color pass.
  [Occlusion verification](../docs/evidence/scene-occlusion.md)
- [x] Consume GPU-written indexed arguments in graph passes and validate signed base vertices on the GPU.
  [Indirect arguments](../docs/evidence/mesh-indirect-draws.md)
- [x] Measure native vertex and fragment timestamps in raster color and shadow passes.
  [Raster timing evidence](../docs/evidence/scene-gpu-timing.md)
- [x] Measure GPU timestamps for shader graph and custom Metal compute dispatches.
  [Compute timing evidence](../docs/evidence/scene-gpu-timing.md#compute-timing)
- [x] Measure render stages for custom Metal and post-process graphs, and whole chains asynchronously.
  [Render timing evidence](../docs/evidence/scene-gpu-timing.md#render-passes-and-post-process-chains)
- [x] Resolve separate render and compute GPU totals across a frame's command buffers.
  [Frame timing evidence](../docs/evidence/scene-gpu-timing.md#frame-timing-batches)
- [x] Count submitted mesh and sprite triangles, points, line segments, and shadow triangles.
- [x] Count shared prepared draw resources and their reported allocation sizes once per scene.
  [Draw and resource evidence](../docs/evidence/scene-render-statistics.md)
- [x] Refit path tracer transforms without rebuilding meshes, rebuild skinned and
  morphed geometry, sample with scrambled Sobol points, and time the GPU pass.
- [x] Sample full-resolution material textures and select Metal ray tracing or
  the software BVH based on the device. Test million-triangle scenes and a
  million-instance Metal scene. [Verification](../docs/evidence/pathscale.md)
- [ ] Cull and compact instances on the GPU, and refit bottom levels on the GPU.

[Scale and dynamics](../docs/evidence/scale-and-dynamics.md)

## FF-41 — Process geometry, preserve scenes, and light translucent assets

- [x] Weld vertices, split creases, and tessellate long edges while preserving
  material ranges, vertex channels and source remaps.
  [Geometry evidence](../docs/evidence/mesh-topology.md)
- [x] Edit hierarchies and camera layers, load assets through a shared cache,
  save supported scene data, and verify video textures and positional audio.
  [Scene evidence](../docs/evidence/scenekit.md)
- [x] Restore typed animation tracks, live material factors, and IES and cookie light resources in scene archives.
  [Scene archive evidence](../docs/evidence/scenekit.md#archive-data-and-limits)
- [x] Fit cascaded shadows and render bent-normal occlusion and screen-space
  reflections within the stated limits. [Raster evidence](../docs/evidence/rasterlight.md)
- [x] Trace subsurface and participating media, verify the raster material
  approximation, and round-trip supported glTF material extensions.
  [Media evidence](../docs/evidence/path-media.md)
- [x] Simplify and subdivide geometry with tested material and attribute handling.
  [Simplification and LOD](../docs/evidence/mesh-simplification.md) ·
  [Subdivision](../docs/evidence/mesh-subdivision.md)
- [x] Stream HDR stills and bake UV colors, normals and static directional
  irradiance. [Still output](../docs/evidence/path-output.md) ·
  [Light-map evidence](../docs/evidence/static-lightmap.md)
- [x] Denoise and upscale static planar path frames with MetalFX where supported.
  [Temporal evidence](../docs/evidence/path-temporal-inputs.md)
- [x] Exercise authored UV color and static lightmap baking on a rendered
  rangefinder studio set. [Asset Studio evidence](../docs/evidence/asset-studio.md)
- [x] Capture six raster views into HDR environment probes and apply
  attachment-space scissors to scene passes.
  [Probes](../docs/evidence/rasterlight.md#hdr-scene-cube-capture) ·
  [Scissor](../docs/evidence/primitives.md#raster-scissor)
- [x] Stream selected first-hit layers alongside beauty in tiled stills.
  [Still output evidence](../docs/evidence/path-output.md#streamed-exr-output)
- [x] Load six SDR or HDR faces into one texture for skies and filtered probes.
  [Format evidence](../docs/evidence/formats.md#six-face-cube-textures)
- [x] Generate a UV1 atlas from an indexed mesh and bake it in Asset Studio.
  [UV atlas evidence](../docs/evidence/mesh-uv-atlas.md) ·
  [Asset Studio evidence](../docs/evidence/asset-studio.md)
- [x] Reproject rigid instance motion in path-traced MetalFX frames.
  [Temporal evidence](../docs/evidence/path-temporal-inputs.md)
- [x] Query oriented boxes, capsules and spheres against static scene snapshots.
  [Collision evidence](../docs/evidence/collision.md)
- [x] Render solid, textured and directional backgrounds for explicit views.
  [Background evidence](../docs/evidence/environment-background.md)
- [x] Move, rotate and scale scene nodes using rendered, ray-picked handles.
  [Gizmo evidence](../docs/evidence/transform-gizmo.md)
- [x] Export posed scene geometry as OBJ, PLY and STL.
  [Mesh export evidence](../docs/evidence/mesh-export.md)
- [x] Load standalone MTL libraries and preserve normalized PLY mesh colors.
  [Format evidence](../docs/evidence/formats.md#modelio-import)
- [x] Accumulate diffuse bounce lighting in authored UV space and bind the HDR result.
  [Lightmap evidence](../docs/evidence/static-lightmap.md)
- [x] Bake static base-color and emissive textures using the hit UV, and exercise the result in Asset Studio.
  [Asset Studio evidence](../docs/evidence/asset-studio.md)
- [x] Pack several receivers into one lightmap atlas and bake their cross-object visibility and diffuse bounce.
  [Shared atlas verification](../docs/evidence/static-lightmap.md#shared-receiver-atlas)
- [x] Bake point and spot lights and replace lighting transactionally on a prepared atlas.
  [Progressive lightmap evidence](../docs/evidence/static-lightmap.md)
- [x] Apply IES angular profiles and spot-cookie color to direct and bounced lightmap samples.
  [Photometric baking evidence](../docs/evidence/static-lightmap.md)
- [x] Bake one-sided rectangle illumination and finite soft shadows into progressive lightmaps.
  [Rectangle baking evidence](../docs/evidence/static-lightmap.md)
- [x] Export Draco meshes and points and native KTX2 textures with stored mips.
  [Draco evidence](../docs/evidence/draco.md) · [KTX2 evidence](../docs/evidence/ktx2-export.md)
- [x] Load native and Basis KTX2 cube faces and 2D array layers with stored mips.
  [Spatial KTX2 evidence](../docs/evidence/ktx2-spatial-import.md) · [Basis evidence](../docs/evidence/basis-universal.md)
- [x] Transcode LDR Basis KTX2 to device-compressed storage and decode linear HDR Basis profiles.
  [Basis evidence](../docs/evidence/basis-universal.md)
- [x] Prepare compressed glTF Basis material maps while preserving decode and optional image fallbacks.
  [glTF Basis evidence](../docs/evidence/basis-universal.md#gltf-material-preparation)
- [x] Sample Catmull–Rom, Bézier and NURBS curves and tessellate NURBS patches and lofts.
  [Curve evidence](../docs/evidence/curves.md)
- [x] Load BVH motion capture into a scene hierarchy and playable native clips.
  [Motion capture evidence](../docs/evidence/bvh-motion-capture.md)
- [x] Visualize a translation track and keyframes under the current or animated parent pose.
  [Animation path evidence](../docs/evidence/animation-path-helper.md)
- [x] Bend meshes along sampled curves and render multiple baked placements together.
- [x] Animate individual curve placements with shared raster snapshots and optional Metal compute.
  [Curve deformation evidence](../docs/evidence/curve-deformation.md)
- [x] Preserve OBJ material references and typed PLY properties, and write RLE and PIZ scanline EXR.
  [Mesh export evidence](../docs/evidence/mesh-export.md) · [EXR evidence](../docs/evidence/exr-codecs.md)
- [x] Import typed PLY meshes and point clouds, including binary big-endian and per-corner UVs.
  [PLY import verification](../docs/evidence/ply-import.md)
- [x] Import and render ASCII, binary and LZF-compressed PCD point clouds.
  [PCD import evidence](../docs/evidence/pcd-import.md)
- [x] Import XYZ point clouds with optional sRGB colors.
  [XYZ evidence](../docs/evidence/xyz-import.md)
- [x] Import PDB atoms, colors, metadata and explicit bonds into renderable primitives.
  [PDB evidence](../docs/evidence/pdb-import.md)
- [x] Import MagicaVoxel models and first-frame hierarchies as colored meshes and occupancy textures.
  [VOX evidence](../docs/evidence/vox-import.md)
- [x] Import legacy and XML VTK PolyData with triangulated polygons and strips.
  [Legacy VTK evidence](../docs/evidence/vtk-legacy-import.md) · [XML VTK evidence](../docs/evidence/vtk-xml-import.md)
- [x] Import embedded typed NRRD scalar volumes and upload their original voxel order to Metal.
  [NRRD evidence](../docs/evidence/nrrd-import.md)
- [x] Raymarch scalar volumes using maximum intensity or an isosurface with opaque depth clipping.
  [Scalar volume evidence](../docs/evidence/scalar-volume-rendering.md)
- [x] Import ASCII STL solid groups and binary Magics colors and alpha.
  [STL import verification](../docs/evidence/stl-import.md)
- [x] Decode raw and RLE TGA textures and verify TIFF through ImageIO.
  [TGA and TIFF verification](../docs/evidence/tga-tiff.md)
- [x] Load DDS mip chains and cubemaps with native BC sampling or CPU fallback.
  [DDS verification](../docs/evidence/dds-import.md)
- [x] Load compressed KTX1 and PVR mip chains and cubemaps with native or CPU PVRTC sampling.
  [KTX1 verification](../docs/evidence/ktx1-import.md) · [PVR verification](../docs/evidence/pvr-import.md)
- [x] Copy and read back Metal texture regions, levels and slices, and expand compressed 2D textures.
  [Texture transfers](../docs/evidence/texture-transfers.md) · [Decompression](../docs/evidence/texture-decompression.md)
- [x] Stage partial CPU texture uploads for selected regions, mips and layers.
  [Texture uploads](../docs/evidence/texture-transfers.md#partial-cpu-uploads)
- [x] Inspect live textures with scene helpers and show labeled native shadow maps in a render overlay.
  [Texture and shadow preview evidence](../docs/evidence/texture-debug-pass.md)
- [x] Flip decoded raster rows and premultiply alpha before texture upload and mip generation.
  [Raster upload evidence](../docs/evidence/raster-texture-upload.md)
- [x] Decode signed and unsigned two-channel base and clearcoat normals in both renderers.
  [Packed normal map verification](../docs/evidence/packed-normal-maps.md)
- [x] Shade object-space base and clearcoat normals with node and instance transforms.
  [Object-space normal evidence](../docs/evidence/object-space-normal-maps.md)
- [x] Generate seeded flake normals for paint and clearcoat, with raster and path-traced verification.
  [Flake texture evidence](../docs/evidence/flake-normal-textures.md)
- [x] Read horizontal and vertical image strips as 3D color lookup tables.
  [LUT import verification](../docs/evidence/lut-image-import.md)
- [x] Apply scalar color ramps and generate lookup and legend textures.
  [Scalar color map evidence](../docs/evidence/scalar-colormap.md)
- [x] Project native HDR cube captures to nine spherical-harmonic light-probe coefficients.
  [Spherical probe evidence](../docs/evidence/spherical-light-probe.md)
- [x] Light raster PBR and classic surfaces with diffuse probes and preserve them in scene archives.
  [Raster and archive evidence](../docs/evidence/spherical-light-probe.md)
- [x] Author sky and ground hemisphere irradiance with a world-space direction.
  [Hemisphere evidence](../docs/evidence/spherical-light-probe.md#hemisphere-lighting)
- [x] Apply surface graph opacity, cutoff and masks in raster color and attributes and in path intersections.
  [Graph coverage evidence](../docs/evidence/graph-coverage.md)
- [x] Apply graph cutouts in directional, spot and point raster shadow maps.
  [Graph shadow coverage](../docs/evidence/graph-coverage.md#raster-shadow-coverage)
- [x] Offer nearest-depth sampling for directional, spot and point shadow maps.
  [Basic shadow evidence](../docs/evidence/basic-shadow-map.md)
- [x] Filter depth moments for variance sampling in directional shadow maps and cascades.
  [Variance shadow evidence](../docs/evidence/variance-shadow-maps.md)
- [x] Filter spot variance moments while preserving other lights' depth and cache state.
  [Spot variance evidence](../docs/evidence/variance-shadow-maps.md)
- [x] Reuse completed shadow depth until requested or invalidated by the light-space layout.
  [On-demand shadow evidence](../docs/evidence/on-demand-shadow-updates.md)
- [x] Refresh only the affected spot slice or point-light faces in an on-demand punctual set.
  [Selective refresh evidence](../docs/evidence/on-demand-shadow-updates.md)
- [x] Support independent automatic and on-demand policies for lights in the same punctual set.
  [Mixed shadow policy evidence](../docs/evidence/on-demand-shadow-updates.md#mixed-punctual-policies)
- [x] Composite tinted directional and punctual raster shadows on a transparent catcher.
  [Shadow catcher evidence](../docs/evidence/shadow-catcher.md)
- [x] Extract indexed isosurfaces from scalar fields, metaballs and plane fields.
  [Marching cubes evidence](../docs/evidence/marching-cubes.md)
- [x] Replace bounded raster geometry on a prepared mesh while earlier GPU draws retain their own snapshots.
  [Dynamic geometry evidence](../docs/evidence/dynamic-geometry.md)
- [x] Select indexed draw ranges in raster, picking, path snapshots and native archives.
  [Mesh draw range evidence](../docs/evidence/mesh-draw-range.md)
- [x] Select point and line draw ranges in raster, both picking paths and native archives.
  [Primitive draw range evidence](../docs/evidence/mesh-draw-range.md#point-and-line-draw-ranges)
- [x] Intersect one source draw range across material groups in raster, picking and path tracing.
  [Grouped draw range evidence](../docs/evidence/mesh-draw-range.md#material-groups)
- [x] Generate mesh normals and tangents, center bounds, and bake affine transforms.
  [Mesh processing evidence](../docs/evidence/mesh-processing.md)
- [x] Resolve world-oriented children consistently across renderers, picking and native archives.
  [World-oriented child evidence](../docs/evidence/world-oriented-child.md)
- [x] Expose standalone planes and triangles and bounded ray hits against basic shapes.
  [Geometric query evidence](../docs/evidence/geometric-queries.md)
- [x] Test closed shape overlap and bounded ray proximity to points and segments.
  [Overlap and proximity evidence](../docs/evidence/geometric-queries.md#closed-overlap-and-proximity)
- [x] Author raster ambient nodes that add diffuse irradiance alongside probes and image lighting.
  [Ambient light evidence](../docs/evidence/ambient-light.md)
- [x] Use refraction, face-forward and two-component cross products in typed graphs.
  [Vector math evidence](../docs/evidence/graphlang.md#vector-directions)
- [x] Generate equirectangular, matcap and wrapped sprite atlas coordinates in typed graphs.
  [Graph UV evidence](../docs/evidence/shader-graph-uv.md)
- [x] Compose screen and overlay blends, ACES, posterization and color decision lists in typed graphs.
  [Graph color evidence](../docs/evidence/shader-graph-color.md)
- [x] Expand instanced and batched draws, reorder instance records, and reduce visible world vertices.
  [Scene utilities evidence](../docs/evidence/scene-utils.md)
- [x] Render custom fragments into multiple color attachments in one raster pass.
  [Multiple render target evidence](../docs/evidence/custom-mesh-mrt.md)
- [x] Render selected array layers, volume planes and cube faces with mip-aware coordinates and transmission.
  [Layered target evidence](../docs/evidence/layered-render-targets.md)
- [x] Write lit PBR color and named surface values with shared depth and cutout coverage.
  [Named output evidence](../docs/evidence/named-surface-mrt.md)
- [x] Render and pick through perspective and orthographic raster view offsets.
  [Camera view offset evidence](../docs/evidence/camera-view-offset.md)
- [x] Frame off-axis raster views from a world-space eye and three rectangular screen corners.
  [Corner framing evidence](../docs/evidence/frame-corners.md)
- [x] Composite a pickable six-axis overlay and align an orbit camera to a selected endpoint.
  [Axis view evidence](../docs/evidence/axis-view-helper.md)
- [x] Style axis labels and animate optional constant-speed orbit alignment.
  [Axis transition evidence](../docs/evidence/axis-view-helper.md)
- [x] Clear individual framebuffer attachments and set the drawable's clear color.
  [Framebuffer clear evidence](../docs/evidence/framebuffer-clear.md)
- [x] Customize opaque, transparent and batched raster draw ordering.
  [Scene ordering evidence](../docs/evidence/scene-render-ordering.md)
- [x] Render packed window depth and normalized distance to linear targets.
  [Depth packing evidence](../docs/evidence/depth-material-packing.md)
- [x] Decode still HDR JPEG XL textures without clipping extended radiance.
  [JPEG XL evidence](../docs/evidence/jpeg-xl-import.md)
- [x] Draw world camera frusta, posed skeletons and hemisphere light helpers.
  [Scene helper evidence](../docs/evidence/scene-debug-helpers.md)
- [x] Refresh posed skeleton helpers in both scene color encode paths.
  [Scene helper evidence](../docs/evidence/scene-debug-helpers.md)
- [x] Set the SwiftUI drawable pixel ratio and observe its completed dimensions.
  [Drawable resolution evidence](../docs/evidence/drawable-resolution.md)
- [x] Import and rasterize SVG paths and filled contours with holes.
  [SVG path evidence](../docs/evidence/svg-path-import.md)
- [x] Keep small mesh and primitive offsets at large shared origins with the graph's perspective camera.
  [High-precision raster evidence](../docs/evidence/high-precision-model-view.md)
- [x] Prepare and draw raw Metal mesh vertex and fragment stages with reflected resources.
  [Raw Metal evidence](../docs/evidence/raw-mesh-program.md)
- [x] Outline SVG strokes with authored caps and joins.
  [SVG stroke evidence](../docs/evidence/svg-stroke-import.md)
- [x] Trace circular and elliptical emitters with matching sampled and ray-hit PDFs.
  [Path area light evidence](../docs/evidence/path-shaped-emitters.md)
- [x] Present Display P3 and Rec. 2020 HDR output through a native SwiftUI drawable.
  [SwiftUI HDR evidence](../docs/evidence/swiftui-hdr-presentation.md)
- [x] Render explicit off-axis and orthographic cameras at large shared origins.
  [High-precision raster evidence](../docs/evidence/high-precision-model-view.md)
- [x] Depth-test both particle renderers against reversed perspective scenes.
  [Reversed depth evidence](../docs/evidence/reversed-depth.md)
- [x] Resolve SVG styles, local references, gradients, clipped joins and textured strokes.
  [SVG gradient evidence](../docs/evidence/svg-gradient-import.md) · [SVG stroke evidence](../docs/evidence/svg-stroke-import.md)
- [x] Trace finite spot apertures with cone, IES, cookie and blocker sampling.
  [Finite spot evidence](../docs/evidence/path-finite-spot.md)
- [x] Read path sample density and software BVH bounds, including per-ray heatmaps, and choose blue-dithered Sobol.
  [Path diagnostic evidence](../docs/evidence/path-sampling-diagnostics.md)
- [x] Carry reversed depth through SwiftUI graph draws, named targets, visibility queries and timing.
  [Reversed depth evidence](../docs/evidence/reversed-depth.md)
- [x] Read the standard SVG 1.1 public header on the r187 tiger without resolving entities.
  [SVG path evidence](../docs/evidence/svg-path-import.md)
- [x] Match r187 CPU improved Perlin and simplex samples and the shader Bayer RGB phase.
  [CPU noise evidence](../docs/evidence/cpu-procedural-noise.md) · [Graph evidence](../docs/evidence/graphlang.md)
- [x] Shade native PBR height maps in raster and path tracing and preserve them in archives.
  [PBR bump evidence](../docs/evidence/pbr-bump-maps.md)
- [x] Import scalar Wavefront bump maps and evaluate typed graph height in raster and path tracing.
  [Wavefront bump](../docs/evidence/mtl-bump-maps.md) · [Graph bump](../docs/evidence/shader-texture-bump-maps.md)
- [x] Match the pinned r187 simplex, curl and animated Voronoi helpers in typed shader graphs.
  [GPU patterns](../docs/evidence/shader-procedural-patterns.md)
- [x] Rebase single and cascaded directional shadows at large shared world origins.
  [High-precision evidence](../docs/evidence/high-precision-model-view.md)
- [x] Adapt path preview resolution from completed GPU timings and retain the last completed image.
  [Adaptive preview](../docs/evidence/path-adaptive-preview.md)
- [x] Host progressive path tracing in SwiftUI and stop at the sample cap.
  [SwiftUI path evidence](../docs/evidence/swiftui-progressive-path.md)
- [x] Reconstruct post-process view positions, surface normals and linear depth.
  [Depth helper evidence](../docs/evidence/post-process-depth-helpers.md)
- [x] Override indexed mesh raster materials and preserve authored geometry and shadow coverage.
  [Material override evidence](../docs/evidence/scene-material-override.md)
- [x] Keep spotlight lighting, caster depth and variance moments precise at large shared origins.
  [High-precision evidence](../docs/evidence/high-precision-model-view.md)
- [x] Preserve point and rectangle receiver lighting and shadow cube faces at large origins.
  [High-precision evidence](../docs/evidence/high-precision-model-view.md)
- [x] Blend independent scene renders through feathered masks and hard cuts.
  [Transition evidence](../docs/evidence/scene-transition-pass.md)
- [x] Draw r187-style inverted hulls for classic toon meshes.
  [Toon outline evidence](../docs/evidence/toon-outline-pass.md)
- [x] Gate and fade from adaptive preview to a fixed-resolution path render.
  [Preview handoff](../docs/evidence/path-preview-handoff.md)
- [x] Apply classic bump and normal detail with native archive round trips.
  [Classic map evidence](../docs/evidence/classic-bump-and-normal-maps.md)
- [x] Override raster point, line and sprite materials while preserving live ranges.
  [Override evidence](../docs/evidence/scene-material-override.md)
- [x] Denoise AO with depth and normal guidance in the post-process graph.
  [Denoising evidence](../docs/evidence/ao-denoise.md)
- [x] Generate HDR panoramas with importance sampling and trace the spectral solar disk.
  [Environment evidence](../docs/evidence/path-generated-environments.md)
- [x] Set independent direct and indirect path radiance limits and opt in to light sampling of constant opaque emissive triangles.
  [Sampling evidence](../docs/evidence/path-mis-and-radiance-limits.md) · [Mesh emitter evidence](../docs/evidence/path-emissive-mis.md)
- [x] Fade both native particle renderers against stored scene depth and use LTC specular for raster PBR rectangle lights.
  [Particle evidence](../docs/evidence/soft-particles.md) · [LTC evidence](../docs/evidence/rectangle-ltc.md)
- [x] Capture per-pass scene layers and material overrides for post-processing and apply Frei–Chen edge detection.
  [Scene capture evidence](../docs/evidence/post-process-scene-pass.md) · [Edge detection evidence](../docs/evidence/frei-chen.md)
- [x] Render obliquely clipped planar mirrors and single-sample meshes with logarithmic depth.
  [Reflection evidence](../docs/evidence/planar-reflection.md) · [Depth evidence](../docs/evidence/logarithmic-depth.md)
- [x] Sweep holed shapes along spatial curves and trace light through voxel fog density grids.
  [Path extrusion evidence](../docs/evidence/path-extrusion.md) · [Fog density evidence](../docs/evidence/path-fog-density.md)
- [x] Sample stored viewport color and depth in raster surface graphs and share scene environment lighting between raster and path workflows.
  [Viewport evidence](../docs/evidence/surface-viewport-snapshot.md) · [Environment evidence](../docs/evidence/scene-environment.md)
- [x] Author custom extrusion UVs and style post effects with lens dirt, outline pulse, glow and patterns.
  [Extrusion evidence](../docs/evidence/extrusion-custom-uv.md) · [Lens dirt evidence](../docs/evidence/lens-dirt.md) · [Outline evidence](../docs/evidence/selection-outline-style.md)
- [x] Capture opaque scene color mips and depth and render bounded view-specific mirror bounces with reflected skies.
  [Viewport evidence](../docs/evidence/surface-viewport-snapshot.md) · [Reflection evidence](../docs/evidence/planar-reflection.md)
- [x] Add playing animation groups, independent skinned clones and additive retargeting.
  [Animation evidence](../docs/evidence/animtools.md)
- [x] Apply directional contact visibility in PBR, fit sun cascades to the active camera and expose directional receiver bias and strength.
  [Contact shadow evidence](../docs/evidence/screen-space-shadows.md) · [Raster lighting evidence](../docs/evidence/rasterlight.md)
- [x] Add editable geometry helpers, textured sprite flares, oblique planar refraction and the two baseline animated water models.
  [Helper evidence](../docs/evidence/live-debug-helpers.md) · [Flare evidence](../docs/evidence/sprite-lens-flare.md) · [Water evidence](../docs/evidence/planar-water.md)
- [x] Sample textured opaque path emitters and reuse prepared picking hierarchies for graph-displaced meshes.
  [Mesh light evidence](../docs/evidence/path-emissive-mis.md) · [Graph picking evidence](../docs/evidence/graph-pose-picking-bvh.md)
- [x] Apply independent punctual shadow strength and retain HDR radiance through planar water with fog and custom shading.
  [Raster lighting evidence](../docs/evidence/rasterlight.md) · [Water evidence](../docs/evidence/planar-water.md)
- [x] Tint the cascaded sun in RGB and cast layered colored shadows from uniform directional glass.
  [Sun RGB evidence](../docs/evidence/sun-light-rgb.md) · [Glass shadow evidence](../docs/evidence/directional-transmitted-shadows.md)
- [x] Refract planar water through a depth-safe viewport snapshot and trace up to eight overlapping fog boxes.
  [Water evidence](../docs/evidence/planar-water.md) · [Fog density evidence](../docs/evidence/path-fog-density.md)
- [x] Trace closed mesh volumes beside fog boxes, author colored shadows from surface graphs and encode ordered array-camera views.
  [Fog density evidence](../docs/evidence/path-fog-density.md) · [Glass shadow evidence](../docs/evidence/directional-transmitted-shadows.md) · [Array camera evidence](../docs/evidence/array-camera.md)
- [x] Export native KTX2 texture and cube arrays and add the texture, image and binary16 utilities.
  [KTX2 export evidence](../docs/evidence/ktx2-export.md) · [Texture utility evidence](../docs/evidence/texture-utilities.md)
- [x] Add opt-in r187 dithering to the HDR display pass and the final tone-mapping stage.
  [Dithering evidence](../docs/evidence/dithering.md)
- [x] Add per-node encode hooks and an opt-in scene encode inspector.
  [Render hook evidence](../docs/evidence/render-hooks.md)
- [x] Sweep curve tubes, triangulate standalone polygons and add tilt shift and rendered-depth autofocus.
  [Tube evidence](../docs/evidence/curve-tube-geometry.md) · [Triangulation evidence](../docs/evidence/polygon-triangulation.md) · [Tilt-shift evidence](../docs/evidence/tilt-shift.md) · [Autofocus evidence](../docs/evidence/camera-autofocus.md)
- [x] Shape horizontal and vertical text for 3D extrusion, including curve sweeps, and insert standalone triangulation Steiner points.
  [Curved text](../docs/evidence/curved-text-extrusion.md) · [Vertical layout](../docs/evidence/vertical-text-layout.md) · [Triangulation](../docs/evidence/polygon-triangulation.md)
- [x] Apply three-axis scene environment orientation, graph and primitive soft intersections, and shadow and transmission attachments per planar bounce.
  [Scene environment](../docs/evidence/scene-environment.md) · [Soft particles](../docs/evidence/soft-particles.md) · [Planar reflection](../docs/evidence/planar-reflection.md)
- [x] Add texture-driven glitch, screen-space CRT, weighted radial blur and five RGB halftone shapes.
  [Glitch](../docs/evidence/texture-glitch.md) · [CRT](../docs/evidence/retro-crt.md) · [Radial blur](../docs/evidence/radial-blur.md) · [Halftone](../docs/evidence/native-halftone.md)
- [x] Rebase batch clipping, culling and sort depth and preserve unskinned instance shadows.
  [High-precision evidence](../docs/evidence/high-precision-model-view.md)
- [x] Query renderer GPU features and raster sample counts on the selected device.
  [Capability evidence](../docs/evidence/renderer-capabilities.md)
- [x] Apply authored local pivots through hierarchy, animation, picking and rendering, and preserve poses through interchange.
  [Scene pivot evidence](../docs/evidence/scene-pivots.md)
- [x] Reproject topology-stable skin and morph poses in temporal path frames.
  [Pose motion evidence](../docs/evidence/path-deformation-motion.md)
- [x] Capture direct and indirect floor lighting in Asset Studio.
  [Indirect lighting evidence](../docs/evidence/asset-studio-indirect.md)
- [x] Pack the Asset Studio floor and backdrop into one UV1 atlas and verify diffuse contributions on both receivers.
  [Shared atlas evidence](../docs/evidence/asset-studio.md#shared-floor-and-backdrop-atlas)
- [x] Draw typed planar paths, tube painting strokes and the pinned Utah teapot.
  [Planar paths](../docs/evidence/typed-planar-path.md) · [Tube painter](../docs/evidence/tube-painter.md) · [Teapot](../docs/evidence/teapot-geometry.md)
- [x] Feed rigid raster motion into HDR blur and temporal anti-aliasing and compose depth-filtered planar captures.
  [Motion blur](../docs/evidence/raster-motion-blur.md) · [Motion AA](../docs/evidence/raster-motion-taa.md) · [Planar reflection](../docs/evidence/planar-reflection.md)
- [x] Apply texture and vertex alpha cutouts to lightmap receiver coverage, direct shadows and diffuse bounce rays.
  [Cutout baking evidence](../docs/evidence/static-lightmap.md#alpha-tested-transport)
- [x] Render depth-filtered automatic mirrors and reduced-resolution scene pixelation.
  [Planar reflections](../docs/evidence/planar-reflection.md) · [Pixelation](../docs/evidence/pixelated-scene.md)
- [x] Replay native skin and morph poses in raster motion, and apply r187-style temporal resolve and native upscaling.
  [Raster motion](../docs/evidence/raster-motion-vectors.md) · [TRAA](../docs/evidence/reference-traa.md) · [Upscaling](../docs/evidence/raster-temporal-upscaling.md)
- [x] Project caster shadows onto planes and gather visible diffuse bounce in a native screen-space pass.
  [Projected shadows](../docs/evidence/planar-projected-shadows.md) · [Screen-space GI](../docs/evidence/screen-space-gi.md)
- [x] Retain typed shader displacement motion in raster and path frames, and keep raster depth replay consistent with color.
  [Raster motion](../docs/evidence/raster-motion-vectors.md) · [Path motion](../docs/evidence/path-deformation-motion.md)
- [x] Scale and reproject screen-space GI, add selective and rough screen-space reflections, and upscale HDR with spatial FSR 1 and RCAS.
  [Screen-space GI](../docs/evidence/screen-space-gi.md) · [Screen-space reflections](../docs/evidence/screen-space-reflections.md) · [FSR 1](../docs/evidence/native-fsr1.md)
- [x] Mask post-processing stages, save HDR intermediates and composite a full-screen texture.
  [Mask and save stages](../docs/evidence/post-mask-save.md) · [Texture pass](../docs/evidence/post-texture-pass.md)
- [x] Query mesh BVH proximity, traverse pairs and shapes, refit stable topology, and accelerate point-cloud picking.
  [BVH queries](../docs/evidence/bvh-proximity.md)
- [x] Track curve-flow and dynamic mesh poses in raster motion, and reproject prior screen-reflection radiance across camera moves.
  [Live raster motion](../docs/evidence/raster-motion-vectors.md) · [Reflections](../docs/evidence/screen-space-reflections.md)
- [x] Insert and remove prepared composer stages and verify spatial FSR on a transparent path-traced frame.
  [Composer stages](../docs/evidence/post-mask-save.md) · [Path FSR](../docs/evidence/native-fsr1.md)
- [x] Apply and archive material stencil state in regular mesh draws and refit posed or selectively edited mesh hierarchies.
  [Stencil](../docs/evidence/material-stencil.md) · [BVH queries](../docs/evidence/bvh-proximity.md)
- [x] Move bake casters without repacking the receiver atlas and blend live light or caster updates over a rolling window.
  [Lightmap evidence](../docs/evidence/static-lightmap.md)
- [x] Add seeded motion-blur jitter and cancellable background BVH builds.
  [Motion blur](../docs/evidence/raster-motion-blur.md) · [BVH queries](../docs/evidence/bvh-proximity.md)
- [x] Extend stencil state to posed, instanced and live indexed meshes and verify per-sample color resolves.
  [Stencil evidence](../docs/evidence/material-stencil.md)
- [x] Evaluate custom irradiance with Halton UV samples and bake vertex colors into diffuse bounces.
  [Lightmap evidence](../docs/evidence/static-lightmap.md)
- [x] Query mesh BVHs from caller-written Metal shaders and add reduced, temporally reprojected GTAO.
  [Shader BVH queries](../docs/evidence/bvh-proximity.md#metal-shader-ray-queries) · [GTAO evidence](../docs/evidence/rasterlight.md)
- [x] Render stencil for curve and graph deformation and bind one exact pose to each encoded draw.
  [Stencil evidence](../docs/evidence/material-stencil.md) · [Deformation ordering](../docs/evidence/mesh-deformation-order.md)
- [x] Move packed receivers during a progressive bake, accumulate Metal-authored irradiance, and sample the atlas in raster and path PBR.
  [Baking](../docs/evidence/static-lightmap.md) · [PBR light maps](../docs/evidence/pbr-light-map.md) · [Asset Studio](../docs/evidence/asset-studio.md)
- [x] Bake and refresh local irradiance probe volumes with immutable bounce sources and prioritized raster sampling.
  [Probe grids](../docs/evidence/light-probe-grid.md)
- [x] Optimize redundant clip keys while preserving spline, additive and glTF morph behavior.
  [Animation](../docs/evidence/animtools.md)
- [x] Prepare HDR room lighting and a ground-projected panorama, and render a depth-tested contact-shadow receiver.
  [Room environment](../docs/evidence/room-environment.md) · [Grounded skybox](../docs/evidence/grounded-skybox.md) · [Contact shadows](../docs/evidence/contact-shadows.md)
- [x] Match GTAO receiver bounds and tap spacing, and sample a retroreflective GGX lobe in raster and path tracing.
  [GTAO](../docs/evidence/rasterlight.md) · [Retroreflection](../docs/evidence/pbr-retroreflection.md)
- [x] Add classic environment refraction, local PBR environment projection and post-scene mesh outlines.
  [Classic environment](../docs/evidence/classic-environment-mapping.md) · [Box projection](../docs/evidence/box-projected-environment.md) · [Outlines](../docs/evidence/toon-outline-pass.md)
- [x] Play and archive procedural visibility and RGB clips with independent alpha.
  [Animation](../docs/evidence/animtools.md)
- [ ] Add arbitrary deformation motion and extend diffuse bake transport to translucent and nondiffuse materials.
- [ ] Finish archive coverage for typed animation bindings and photometric lights.

## FF-38 — Exercise the new systems in the Lab

- [x] Add Shader, Light, and Animation scenes with generated inspectors, Mac
  captures, and offscreen benchmarks, and run them in iPhone, iPad, and Vision Pro simulators.
- [ ] Qualify these scenes on physical iPhone, iPad, and Vision Pro hardware.
- [ ] Compare against Three.js on the same hardware, scene, resolution, and quality settings.

[Shader graphs](../docs/evidence/shader-graphs.md) ·
[Light rigs](../docs/evidence/light-rig.md) ·
[Animation and tools](../docs/evidence/animation-and-tools.md) ·
[Benchmarks](../docs/evidence/lab-benchmarks.md)
