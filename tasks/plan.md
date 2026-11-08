# Frostfield implementation plan

Prepared 2026-11-08. Status: documentation and handoff preparation only. No package manifest, source, tests, CI implementation, renderer, scene API, or supported visual effects are implemented. Claude will start implementation from this plan.

## Product goal

Frostfield is an Apple-native creative graphics framework: programmable scenes, procedural geometry, materials, lighting, particles, animation, and post-processing through Swift and Metal. UIKit and SwiftUI are presentation adapters; neither defines the rendering architecture. The first release should let an app author compose a small scene and supply a custom shader without adopting an application engine.

GlassWeave is a separate, independently released glass framework. Its first integration proof has priority across the two projects: custom refractive glass over an explicit moving background, native controls above it, and genuine Apple glass beside it. Frostfield can establish its own rendering fixtures in parallel. Neither package initially depends on the other.

## Decisions for the first implementation

- Use Swift 6 language mode. Target iOS/iPadOS 26 first. The macOS 26 package declaration permits core build checks; it is not a promise of a macOS presentation adapter or validated Mac GPU behavior.
- Start with `MTKView` in a UIKit sample. Implement a conventional Metal command-queue fixture first; compare Metal 4 using the same output before choosing a production backend. Check both API availability and device capabilities.
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
| Package foundation | FF-01 | A real package and verified build baseline created during implementation |
| Rendering evidence | FF-02–FF-05 | UIKit triangle fixture, lifecycle behavior, and documented backend choices |
| First usable release | FF-06–FF-08, then Checkpoint C | Camera/transform, indexed geometry, a custom material shader, and measured standalone v0.1 qualification |
| Optional later effects | FF-09–FF-11 | One light model, one particle emitter, and one post-process |
| Optional later integration | FF-12–FF-13 | Explicit offscreen texture output and an optional SwiftUI host |
| Expanded evidence | FF-14–FF-15, then Checkpoint G | Full-fixture profiling and GlassWeave interoperability for later releases |

Dependency spine: FF-01 → FF-02 → FF-03 → FF-04/FF-05 → FF-06 → FF-07 → FF-08 → Checkpoint C. Checkpoint C qualifies a standalone v0.1 with its own physical measurements and clean-consumer build. It does not depend on GlassWeave or FF-09–FF-15. After the core, lighting, particles, post-processing, and texture output can proceed independently if their shared render-pass contracts are frozen first. FF-13 follows the UIKit host and creative core. FF-14 measures the expanded fixture for a later release. FF-15 additionally requires GlassWeave's custom-surface integration proof; that dependency applies only to the optional interoperability example.

The planned v0.1 is intentionally limited to a programmable mesh scene with camera/transforms, unlit/custom material shaders, and the proven UIKit sample host. Native lifecycle behavior, error reporting, deterministic tests, a clean consuming app, and scoped physical-device qualification are required. Lighting, particles, post-processing, public texture export, the SwiftUI adapter, and GlassWeave interoperability are later opt-in milestones. A prototype before Checkpoint C is a developer preview; neither a preview nor v0.1 may advertise those later capabilities as shipped.

## Deliberately later

Typed/node-based material graphs; a visual editor; Three.js feature parity; scene serialization; glTF or other asset-loader suites; physically based rendering; skeletal animation; ray tracing; physics; networking; general-purpose render graphs; cross-platform backends; and additional Apple UI adapters. These can become separate plans after the first programmable rendering path is validated.

## Risks and mitigation

| Risk | Consequence | Planned response |
|---|---|---|
| Building an engine before testing app integration | Large framework with poor native behavior | UIKit host and lifecycle fixture precede scene systems; GlassWeave proof leads the joint program |
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
