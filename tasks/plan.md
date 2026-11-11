# Frostfield implementation plan

The Swift package and the native triangle hosts are in place. See the
[build baseline](../docs/evidence/build-baseline.md),
[triangle evidence](../docs/evidence/triangle-fixture.md), and [task list](todo.md).

## Product goal

Frostfield is a Swift and Metal creative graphics framework for macOS, iOS,
iPadOS, and visionOS. Three.js is the reference for the breadth of scenes and
effects authors should be able to build. The first release should let an app
author compose a small scene and supply a custom shader. AppKit, UIKit, and
SwiftUI handle native presentation around the shared core.

The long-term scope includes scene graphs with hierarchical transforms and
cameras, procedural and indexed geometry, programmable materials, textures and
asset loaders, lights and shadows, animation, render targets and
post-processing, and picking. The task list stages the early pieces. Later
features need their own fixtures and qualification before they are advertised.

GlassWeave is a separate, independently released glass framework. Its first integration proof has priority across the two projects: custom refractive glass over an explicit moving background, native controls above it, and genuine Apple glass beside it. Frostfield can establish its own rendering fixtures in parallel. Neither package initially depends on the other.

## Decisions for the first implementation

- Use Swift 6 language mode and the Xcode 27 SDKs. Target macOS, iOS, iPadOS, and visionOS 27.0 or later.
- Use AppKit on macOS, UIKit on iOS and iPadOS, and a UIKit view in a SwiftUI window on visionOS. The current hosts receive drawables through `CAMetalDisplayLink`. Start with a conventional Metal command queue, then compare Metal 4 using the same output. Check API availability and device capabilities independently.
- Keep the first visionOS fixture windowed with a single raster view. Treat stereo windows and immersive Compositor Services as separate presentation milestones.
- Keep application intent separate from render execution. UI code submits stable state; the renderer owns scheduling, resources, and GPU completion tracking. Main-thread isolation does not establish GPU synchronization.
- Compare a minimal custom renderer with a bounded Satin prototype. Use a narrow experimental interface only inside the comparison. Record dependency, licensing, shader, packaging, and profiling findings before deciding; do not ship the experimental abstraction as public API.
- Build vertical slices that render something inspectable. Introduce public API only after a working fixture establishes its shape. Keep shader compilation and pipeline creation out of steady-state interaction.
- Start with internal GPU utilities. Extract shared infrastructure only after Frostfield and GlassWeave have independently demonstrated the same requirements. GlassWeave must never need Frostfield's scene graph merely to render a glass control.
- Make future texture exchange explicit about format, color, coordinates, ownership, device, completion, and frame identity. Transparent views do not imply access to application content behind them.

See [architecture](../docs/architecture.md) for boundaries and [validation](../docs/validation.md) for evidence requirements.

## Delivery sequence

The actionable checklist lives in [todo.md](todo.md); do not maintain a second completion checklist here. Each task is a small or medium slice with at most five likely implementation files. File names are planning aids, not promised public symbols. Split a task further when its actual implementation exceeds that scope.

| Stage | Tasks | Result |
|---|---|---|
| Package foundation | FF-01 | Swift package and build baseline |
| Rendering evidence | FF-02–FF-05 | Native triangle fixtures, lifecycle behavior, and documented backend choices |
| First usable release | FF-06–FF-08, then Checkpoint C | Camera/transform, indexed geometry, a custom material shader, and measured standalone v0.1 qualification |
| Optional later effects | FF-09–FF-11 | One light model, one particle emitter, and one post-process |
| Optional later integration | FF-12–FF-13 | Explicit offscreen texture output and a reusable SwiftUI adapter |
| Expanded evidence | FF-14–FF-15, then Checkpoint G | Full-fixture profiling and GlassWeave interoperability for later releases |
| Creative scenes | FF-16–FF-19 | Hierarchies, picking, textures, asset import, PBR, shadows, and a visionOS immersive fixture |
| Animated reference scene | FF-20–FF-22 | glTF skinning, material graphs, surface flow, and a native Cyber Ocean fixture |
| Photoreal asset pipeline | FF-23–FF-24 | HDR environments, soft shadows, lens and display effects, and higher-fidelity glTF assets |
| Physical materials and authored framing | FF-25–FF-26 | Layered finishes, camera projections, and screen-space optical transmission |
| Procedural asset authoring | FF-27–FF-28 | Code-built solid and parametric geometry, generated PBR maps, and a complete rangefinder fixture |
| Photoreal light transport | FF-29–FF-30 | Static PBR path tracing, progressive previews, and high-sample product captures |
| Lighting and shader authoring | FF-31–FF-32 | Editable area-light rigs and shared generated PBR functions |

The core follows FF-01 → FF-02 → FF-03 → FF-04/FF-05 → FF-06 → FF-07 → FF-08 →
Checkpoint C. Checkpoint C qualifies a standalone v0.1 with physical measurements and
clean-consumer builds. After the core, lighting, particles, post-processing, and
texture output can proceed independently once their shared render-pass contracts are
stable. FF-13 follows the native host and creative core. FF-14 measures the expanded
fixture for a later release. FF-15 also requires GlassWeave's custom-surface
integration proof.

The planned v0.1 covers a programmable mesh scene with a camera, transforms,
and unlit or custom materials. Qualify native lifecycle behavior, error
reporting, deterministic tests, and clean consuming apps. Record physical-device
evidence for the advertised configurations. A prototype before Checkpoint C is a
developer preview. Lighting, particles, post-processing, public texture export,
the reusable SwiftUI adapter, and GlassWeave interoperability follow in later slices.

## Three.js parity

The [Three.js r187 checklist](../docs/parity/threejs.md) defines the capability
target across core, official add-ons, and the tracked ecosystem. Work through
the recovered renderer, geometry, and asset pipeline changes first. Integrate
scene primitives and serialization before tools that depend on their scene
contracts. Continue path-traced media, still output, and asset baking in
parallel with the raster and scene work.

Close a row after verifying the advertised behavior through its public API,
including relevant failure paths. Record native platform equivalents in the
checklist. Measure physical devices and run matched Three.js comparisons
after the relevant behavior is available in both engines.

## Risks and mitigation

| Risk | Consequence | Planned response |
|---|---|---|
| Building an engine before testing app integration | Large framework with poor native behavior | Native hosts and lifecycle fixtures precede scene systems |
| Metal 4 selected by branding rather than evidence | Extra synchronization and residency complexity | Same-scene comparison; preserve capability-based selection and document supported configurations |
| A high-level dependency restricts custom shaders or packaging | Public API becomes coupled to its implementation | Bounded Satin comparison before API commitments; license and maintenance review |
| Invisible render loops or per-frame compilation | Energy cost and interaction stalls | Idle/background gates, pipeline warm-up, and physical-device traces |
| Unsafe texture reuse or mismatched color spaces | Flicker, corruption, or visual disagreement with GlassWeave | Explicit exchange contract and deterministic color/motion fixtures |
| Shared-core extraction too early | Coordinated releases and broad dependency surface | Keep utilities private until two actual consumers justify extraction |
| Simulator-only confidence | Unsupported device assumptions and misleading timing | Physical-device gates; record hardware, OS, thermal state, and fixture settings |
| Overstated scaffold capabilities | Contributors mistake packaging for a working renderer | Keep README and release notes tied to verified behavior and evidence |

## Decisions still to establish through work

The first physical test-device models, absolute CPU/GPU/memory budgets, supported quality levels, and backend choice remain open until FF-02–FF-05 produce evidence. These do not block preparation. Record decisions with measured rationale in the architecture document or a focused ADR when implementation begins. Do not infer hardware coverage merely from an OS deployment target.

## Collaboration

One agent owns renderer lifetime and shared resource changes at a time. Another can maintain deterministic fixtures and profiling records; a third can work on an adapter or independent effect after the input/output contract is established. Use file ownership and focused diffs. Build, render, inspect, profile, and revise each slice through the project's Apple-tool workflow before checking it off. Documentation, research, and fixture design may proceed while the GlassWeave implementation is being validated.

## Language decision

Use Swift for CPU implementation and native APIs, MSL for custom GPU work, and a
small verified C-compatible data-layout boundary when needed. The
manifest baseline is Swift tools 6.2 with Swift 6 language mode. See the
[language strategy](../docs/language-strategy.md)
for toolchain distinctions, layout tests, ownership rules, foreign-language
admission gates, and conditional backend alternatives.
