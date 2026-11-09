# Frostfield validation

Status: validation plan, 2026-11-08. No package build, GPU performance, rendered-image correctness, device compatibility, or energy claim has been established by this planning work. A future empty package build will validate packaging only.

## Evidence levels

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Manifest/core compilation | Package syntax, module layout, Swift compilation | Rendering correctness or supported GPU/OS combinations |
| Focused Swift Testing cases | Deterministic math, input validation, scheduling/lifetime policy | Actual GPU ordering, visual appearance, or energy use |
| Simulator run | App wiring and a supplementary visual smoke check | Target-device GPU features or physical-device performance |
| Physical-device render | Observed behavior for the recorded hardware/OS/fixture | Universal device coverage |
| Sustained device profile | Timing, memory, thermal, and interaction behavior under recorded conditions | A permanent performance guarantee under arbitrary app workloads |

Simulator translates graphics work to the host Mac and can differ from the selected simulated device; never substitute its timings for a phone or tablet benchmark. [Apple's Simulator guidance](https://developer.apple.com/documentation/metal/developing-metal-apps-that-run-in-simulator).

## Tool workflow

Load the matching installed Xcode 27 skill before Apple-platform work. Use the native Xcode MCP tools for project edits, build/run/test, previews where applicable, documentation lookup, and diagnostics. Begin by discovering the workspace, scheme, and destinations; do not assume their names. Use a physical destination for rendering/performance evidence. Record any unavailable device, unsupported tool operation, or incomplete check explicitly.

Tests are added with behavior that warrants them, not to make a scaffold look complete. Prioritize transform/projection boundaries, invalid geometry, render scheduling, and resource-lifetime state transitions. Shader fixtures should test invariants and meaningful reference outputs; avoid tests that merely repeat implementation constants.

## Required rendering fixtures

For the limited standalone v0.1 at Checkpoint C, require the triangle, fixed camera/indexed cube, custom material, and native-host lifecycle/interaction fixtures. Use them for the physical-device recipe below and record a scoped baseline before release. Light, particles, post-processing, exported-texture, SwiftUI-adapter, and GlassWeave fixtures apply only when those later capabilities are advertised. FF-14 profiles the expanded set; it is not a prerequisite or substitute for core v0.1 qualification.

| Fixture | Detects |
|---|---|
| Triangle and diagnostic colors | Shader packaging, drawable path, orientation, channel/format errors |
| Fixed camera poses and indexed cube | Matrix order, handedness, depth, winding, normals |
| Custom material with animated uniforms | Binding contract, shader escape hatch, pipeline churn |
| Directional light at known angles | Normal transforms and linear-light behavior |
| Seeded bounded particles | Determinism, update/draw ordering, capacity and time-step behavior |
| Bypassed/enabled color grade | Intermediate-texture hazards, alpha and conversion errors |
| Frame-stamped offscreen texture | Stale frames, reuse before completion, UV/crop/scale mismatch |
| Native app with scrolling and overlay controls | Event routing, lifecycle, surrounding UI responsiveness |

Capture fixed camera, seed, time, resolution, quality, and color settings. Compare identical inputs on the same hardware/backend first. Cross-device image comparisons need documented tolerances for floating-point/shader variation. Preserve reference images only after human inspection; do not automatically replace a reference to make a regression pass.

## Provisional physical-device recipe

The initial target is a sustained 60 Hz reference scene, not a current performance claim. The nominal 16.67 ms frame interval is the whole app's display budget; Frostfield must leave room for native UI and other work. Set a smaller numeric CPU/GPU budget after FF-02–FF-05 establish device-specific baselines. A later 120 Hz target requires separate evidence.

1. Name the oldest intended physical test device and one recent device. Record model/chip, OS, display refresh mode, Xcode/SDK, commit, backend, build configuration, dimensions, scale, quality, particle count, and content seed.
2. Use a Release-equivalent build and a controlled environment. Record battery/power state, Low Power Mode, initial thermal state, brightness, and whether a cable, screen recording, or profiling attachment changes the workload.
3. Run a native-only baseline, then the triangle, then the full reference fixture at matching resolution and duration. Keep scrolling and control-interaction scripts reproducible. Compare backends at identical visual quality.
4. Warm up for 30 seconds, then collect a 10-minute run. Record CPU frame/submission time and GPU duration distributions (p50/p95/p99), actual presented frame intervals, missed-frame counts, resident/allocated memory, peak in-flight resources, thermal transitions, and any available energy indicators.
5. Separately inspect hide/show, background/foreground, window resize/rotation, safe-area changes, missing drawable, saturated consumer, and shader failure. With two scenes, resize/deactivate one and confirm the other stays correct; test display-scale changes where hardware permits. Capture an idle period: a still or hidden scene must not submit recurring GPU work.
6. Repeat the measured run at least three times per configuration before claiming a comparison. Record spread, instrumentation overhead, interruptions, and whether the device began warm. Use a clean run when the profiling tool meaningfully changes timing.

Prefer Instruments and Metal tooling appropriate to the active Xcode version. First query available native capabilities. Keep capture instructions reproducible and distinguish direct measurement from proxy values: a thermal state is not a power measurement, and GPU command duration is not input-to-photon latency.

## Initial gates

Before an implementation milestone is accepted:

- The focused tests and relevant Xcode build pass without unresolved diagnostics introduced by the slice.
- GPU validation reports no unexplained resource/synchronization failures, and the deterministic fixture looks correct.
- Steady-state interaction does not compile shaders or create pipeline variants. Any recurring allocations are explained and bounded.
- Memory does not show unexplained growth across repeated scene teardown, resize, or the sustained run; in-flight textures are released after completion.
- Hidden/inactive scenes stop recurring submissions and resume without a large stale time step.
- Physical measurements are attached when the milestone makes a performance or compatibility claim; otherwise the result remains unverified.

The first profiling report must turn provisional budgets into named-device acceptance thresholds. Record a failed threshold as a failed gate; do not silently lower quality or relax the target to mark the task complete. Changes to a target require a stated product tradeoff and a comparable rerun.

## Texture interoperability checks

Use a four-quadrant color chart, transparent edge pattern, known pixel coordinate marks, and a monotonically increasing frame stamp. Verify the negotiated format/color/alpha convention, UV mapping, native display scale, and resize behavior. Delay the consumer deliberately and saturate the lease pool; the producer must not overwrite a texture still being read. Exercise the documented skip/reuse policy and inspect GPU readiness/completion ordering.

The combined GlassWeave example must preserve native control interaction and compare custom glass beside genuine Apple glass over controlled content. It does not validate arbitrary view capture or optical merging with system glass.

## Evidence record template

Store a concise report in `docs/evidence/` when real evidence exists; do not create passing-looking empty reports.

```text
Date / commit / task:
Device and OS / toolchain:
Fixture and exact settings:
Backend / build configuration:
Environment and starting thermal/power state:
Procedure and repetitions:
CPU p50/p95/p99 / GPU p50/p95/p99:
Presented intervals / missed frames:
Memory and in-flight-resource observations:
Idle/lifecycle/interaction observations:
Validation errors and image differences:
Capture locations and instrumentation caveats:
Acceptance thresholds / pass-fail / remaining unknowns:
```

Keep traces or large media in an appropriate artifact location and link them from the report. Do not commit private application content, credentials, signing identities, or unrelated device logs into the public repository.

## First release checklist

Apply this checklist to the limited v0.1 after FF-08, with the scoped Checkpoint C evidence. No GlassWeave package, example, release, or device result is required for a Frostfield-only release. Do not include later effects or interoperability in v0.1's advertised scope.

- [ ] A clean consuming app resolves Frostfield at the candidate revision and runs the documented example, including shader resources, without sibling repositories or development-only paths.
- [ ] The supported OS/device/backend matrix contains actual observations and clearly distinguishes untested combinations.
- [ ] Deterministic tests, scoped visual fixtures, lifecycle checks, and named-device physical thresholds pass for the core advertised feature set; `docs/evidence/v0.1-core-qualification.md` contains measurements and remaining limits.
- [ ] README/API docs describe actual capabilities, failure behavior, thread/lifetime rules, color conventions, and performance limitations.
- [ ] Dependency versions and license obligations are recorded; experimental dependencies do not leak into shipping products.
- [ ] Release notes link the evidence and retain unimplemented features as roadmap items.

Status (2026-11-09): no item is complete. The
[v0.1 core qualification record](evidence/v0.1-core-qualification.md) lists what the Mac
and the iOS Simulator established and every physical-device measurement still missing.

## Language and data-layout evidence

Follow [the language strategy](language-strategy.md). With the first actual GPU
records, verify host size/alignment/offsets/stride and a shader round-trip with
distinctive values. Record allocations, copied bytes where measurable, and
retain/release hotspots during state updates, resize, and steady rendering.
For foreign kernels, include conversion and FFI cost and a clean Swift consumer
build; do not infer a whole-app benefit from an isolated microbenchmark.
