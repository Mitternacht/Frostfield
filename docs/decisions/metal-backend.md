# Decision: Metal command backend (FF-04)

Date: 2026-11-08 · Task: FF-04 · Status: **conventional Metal stays the only
production path.** Physical iPhone and iPad measurements are missing.

## Question

Does the Metal 4 command path (`MTL4CommandQueue`, `MTL4CommandAllocator`,
`MTL4CommandBuffer`, argument tables, residency sets) improve Frostfield's
workload enough to justify its resource, synchronization, and compatibility costs?

## Decision

Keep the conventional `MTLCommandQueue` renderer for v0.1 and the work after it
(FF-06 to FF-08). Do not ship a Metal 4 backend, a second backend behind a flag, or
a public multi-backend protocol. The Metal 4 variant stays an experiment under
[`Examples/Experiments/Metal4Comparison`](../../Examples/Experiments/Metal4Comparison/README.md).

Reasons:

1. On the only GPU measured (Apple M5, a Mac), the two paths produced
   byte-identical images. Their CPU encode and GPU times fell within each other's
   run-to-run spread (see below). This trivial fixture showed no gain.
2. Metal 4 moves allocator memory, residency, resource lifetime, and pass hazards
   from Metal to the app. The Metal 4 renderer has about 1.7 times the code of the
   conventional mirror (156 against 94 non-comment lines). Its lifetime rule cannot
   be expressed safely through Swift 6's `Sendable` checking without a new ownership
   design (see "Lifetime").
3. Any shipping Metal 4 path would still need the conventional path: the iOS
   Simulator SDK has no Metal 4 API, and devices that run iOS 26 without the Metal 4
   GPU family must keep working. Two full backends would double the verification
   matrix, and nothing yet requires a Metal 4-only feature.

## Requirements and runtime capability decision

| Requirement | Finding |
| --- | --- |
| API availability | `MTL4*` types and `MTLGPUFamily.metal4` are declared `API_AVAILABLE(macos(26.0), ios(26.0))` in the macOS 27.2 and iOS 27.2 device SDKs. |
| GPU family | Apple documents Metal 4 for A14/M1 and later GPUs ([WWDC25 205](https://developer.apple.com/videos/play/wwdc2025/205/)). The M5 reports `supportsFamily(.metal4) == true` and Apple family 10. |
| iOS Simulator | The iOS Simulator 27.2 SDK ships empty `MTL4*` headers (for example `MTL4CommandQueue.h` holds only imports) and does not declare `MTLGPUFamily.metal4`. Code that names a Metal 4 type does not compile for the simulator. It needs `#if !targetEnvironment(simulator)`, and simulator builds always use conventional Metal. |
| Drawables | `CAMetalLayer.residencySet` exists in both iOS SDKs. A Metal 4 drawable path would add that set to the queue and use `waitForDrawable`/`signalDrawable`. The experiment does not cover this path (it renders off-screen only). |

**Runtime rule (as implemented in the experiment's `TriangleBackend.select`):** use
Metal 4 only when the build contains the Metal 4 API (not a simulator build) **and**
the rendering device reports `MTLGPUFamily.metal4`. Otherwise use conventional Metal.
The OS version never decides on its own. Frostfield's `RenderCapabilities.supportsMetal4`
already encodes the same rule, and a test checks that the two agree.

**Unsupported devices:** every iOS Simulator destination, and any device that runs
iOS or iPadOS 26 with a GPU older than A14 (it lacks the Metal 4 family). On those,
the conventional path is the only path. A Metal 4 path would therefore always be an
addition, never a replacement, while the deployment floor stays at iOS 26.

## Fixture and environment

| Item | Value |
| --- | --- |
| Fixture | Frostfield's diagnostic triangle (`Triangle.metal`, the same source compiled into the experiment), one draw, clear and store, `bgra8Unorm` |
| Paths | Frostfield's shipping `TriangleRenderer`; an experiment mirror of it (`ConventionalTriangle`, which also reports GPU time); `Metal4Triangle` |
| Policy (all paths) | Two frame slots, never wait for a slot, one uniform buffer per slot, one render target per slot |
| Machine | Mac with Apple M5 GPU, macOS 27.2 (26B123), Xcode 27.2 (27B52), thermal state nominal, Low Power Mode off |
| Timing build | `Metal4Benchmark` command-line tool, Release, Metal API validation and GPU capture off (the scheme's launch action; the tool prints the device type `AGXG17GDevice` to confirm) |
| Correctness build | `Metal4ComparisonTests`, Debug, `MTL_DEBUG_LAYER=1` (the test confirms the `MTLDebugDevice` wrapper) |

## Correctness

Xcode MCP `RunAllTests`, scheme `Metal4ComparisonTests`, destination My Mac, with
API validation on. 9 test functions (16 cases) passed:

- `Metal 4 output matches conventional Metal byte for byte`: 64×64 at 0 rad,
  200×100 at π/3, 90×160 at 2 rad, and 1206×2622 at 0.7 rad. Every byte was equal
  (0 differing pixels), and each image contains saturated red, green, and blue.
- `Conventional mirror matches the shipping renderer`: the same four cases. The
  experiment's conventional path equals Frostfield's `TriangleRenderer` byte for
  byte, so the timing comparison uses the shipping renderer's behavior.
- `Allocators are reused only after their frames complete`: 300 back-to-back
  requests with two slots, two allocators, and two targets. The in-flight count
  never exceeded 2, every frame completed without a commit-feedback error, and
  each allocator was reset only when its slot came back from completion
  (`slotReuses == submitted - 2`).
- `A frame requested while every slot is busy is skipped without waiting`,
  `A missing target releases its slot and allocator unused`, and
  `A resized target renders with the new aspect after the old one is evicted`.
- `Metal 4 is chosen only when the device reports the family` (probe with and
  without the family) and `The selection agrees with Frostfield's capability survey`.

The same scheme on the **iPhone 17 simulator** (iOS 27.2): the Metal 4 renderer and
its lifetime tests are compiled out. The selection returns conventional for both
probes, the conventional mirror still matches the shipping renderer in all four
cases, and the Metal 4 equivalence test is skipped. The xcresult reports 5 test
functions: 4 passed, 1 skipped. This is the unsupported path actually running, not
a mock.

## Measurements (Mac only, not iPhone evidence)

Three separate launches of `Metal4Benchmark` (Xcode MCP `RunProject`, console read
with `GetConsoleOutput`). Each launch ran three rounds, with path order alternating
between rounds. Each round measured each path for 100 warm-up plus 1,000 frames at
1206 × 2622. That gives 9 blocks per path. The table shows each statistic's range
across the 9 blocks and, in brackets, the median of the 9.

| Statistic (µs unless noted) | Shipping `TriangleRenderer` | Conventional mirror | Metal 4 |
| --- | --- | --- | --- |
| Paced CPU encode+commit p50 | 6.4–9.6 [7.9] | 7.5–8.4 [7.7] | 7.5–7.9 [7.7] |
| Paced CPU p95 | 8.6–25.8 | 8.2–11.5 | 8.2–12.3 |
| Paced CPU p99 | 10.8–37.2 | 9.0–20.6 | 9.7–50.8 |
| Paced GPU p50 | not reported by the renderer | 58.5–76.1 [58.9] | 58.5–75.8 [58.8] |
| Paced GPU p95 | — | 59.3–125.2 | 59.6–98.7 |
| Paced GPU p99 | — | 949–1419 | 790–1301 |
| Two-in-flight CPU p50 | 6.3–7.7 [7.6] | 7.4–8.3 [7.5] | 6.4–7.5 [7.4] |
| Two-in-flight frames/s | 11,141–12,299 [12,144] | 9,291–11,852 [11,660] | 10,495–11,978 [11,558] |
| Renderer creation (ms) | 0.33–1.49 | 0.12–0.48 | 0.19–0.38 |
| Command allocator memory | — | not exposed | 4,690 KiB for two allocators (`allocatedSize()`), every block |

Definitions: "paced" submits one frame and awaits its completion before the next,
like requested drawing. GPU time is `gpuEndTime − gpuStartTime` from the command
buffer (conventional) or the `MTL4CommitFeedback` (Metal 4). "Two-in-flight" keeps
both slots busy and measures the CPU time of submitted frames; frames/s is the
submit-and-complete loop rate, not a presentation rate.

Reading: CPU and GPU medians differ by less than 0.2 µs, and every range overlaps.
The paced GPU p99 of about 1 ms appears on both paths and probably reflects the GPU
idling between paced frames; it was not investigated. Renderer creation ran with
Metal's shader cache already warm from the test runs, so **cold pipeline-compilation
time was not measured**. `MTLDevice.currentAllocatedSize` did not change when a
renderer was created; the allocator memory above does not appear in that counter.

## Findings by concern

**Allocator reuse.** Each frame slot owns one `MTL4CommandAllocator`.
`MTL4CommandAllocator.reset()` requires every command buffer that used its memory to
be complete. The experiment resets an allocator only when the slot pool hands its
slot out again, and the pool frees a slot only from the commit-feedback handler.
One `MTL4CommandBuffer` is reused for every frame, which Metal permits right after
commit. Apple's "Drawing a triangle with Metal 4" sample instead blocks the encoding
thread on a shared event (with a 10 ms timeout) before reusing an allocator.
Frostfield's never-block rule rejects that, so the experiment skips the frame.

**Residency.** Nothing is resident implicitly. The uniform buffers live in a
queue-level `MTLResidencySet`, and render targets must be added to a second set
before first use (the renderer checks this with a precondition on every frame).
Resizing means new targets. Old targets may leave residency only after the frames
that used them complete; the experiment requires an idle renderer to evict them. A
production path would need per-generation tracking, which is the same
completion-aware lease that FF-12 must design anyway.

**Synchronization.** Completion and GPU timing come from
`MTL4CommitOptions.addFeedbackHandler`, and there is no CPU wait. Metal 4 does not
track hazards between passes. The experiment gives each in-flight frame its own
target. A real frame graph would need explicit queue barriers between dependent
passes. The conventional path gets both for free.

**Lifetime.** `MTL4CommandBuffer` does not retain the allocator, buffers, or targets
it uses. The conventional completed handler needs nothing extra, because a
conventional command buffer retains its resources. Under Swift 6, the Metal 4
feedback handler is `@Sendable`, and `MTLBuffer`, `MTLTexture`,
`MTL4CommandAllocator`, and `MTLResidencySet` are not `Sendable`. So the handler
cannot hold them alive without `@unchecked Sendable` or `nonisolated(unsafe)`, both
of which the project rules forbid. The experiment therefore traps in `deinit` if it
is released with frames in flight, and its hosts drain first. A shipping Metal 4
backend would need an isolation-domain owner (an actor, or a `Mutex`-owned resource
set that also owns the targets) or a deferred-release queue. That is a real design
cost, not boilerplate.

**Compiler.** Pipelines come from an explicit `MTL4Compiler` with
`MTL4LibraryFunctionDescriptor`s. A missing function surfaces as a generic compiler
error, so the experiment checks function names first to keep the typed
`shaderFunctionMissing` failure.

**Complexity.** Metal4Triangle.swift has 156 non-comment lines and
ConventionalTriangle.swift has 94. The extra code is residency, allocators, the
argument table, and compile-time exclusion. Testing needs two destinations, because
Metal 4 code is compiled out of simulator builds.

## Missing evidence (explicitly not established)

- Any iPhone or iPad measurement, image comparison, or validation run. Physical
  devices were out of scope for this session. The M5 numbers are Mac
  characterization only.
- The on-screen drawable path (`MTKView`/`CAMetalLayer` with Metal 4 residency and
  drawable wait/signal). The comparison is off-screen only.
- GPU traces or Instruments captures for either path.
- Cold pipeline-compilation time (the shader cache was warm), peak resident memory
  beyond allocator size, thermal and energy behavior, and a workload with many draws
  per frame, where Metal 4's encoding model is meant to help.

## Revisit when

- A planned feature needs Metal 4 (for example, machine-learning passes in a command
  buffer, placement sparse resources, or argument-table bindless at scale), or
- physical-device traces of the FF-08 or later fixture show CPU encode or submission
  as a limiting cost, in which case rerun this harness with a many-draw fixture on
  named devices before deciding. Until then, FF-06 and later build on the
  conventional queue.

## Reproduce

Open `Examples/Frostfield.xcworkspace` in Xcode.

- Correctness: scheme `Metal4ComparisonTests`, destination My Mac (or an iOS
  Simulator for the unsupported path), Test.
- Timing: scheme `Metal4Benchmark`, destination My Mac, Run. Its launch action is
  Release with API validation off. Read the `RESULT` lines in the console.
