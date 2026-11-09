# Frostfield: implementation handoff for Claude

Prepared 2026-11-08, when this repository contained a plan, not an
implementation. The user deliberately reserved implementation for Claude.

**Status (2026-11-09):** the repository now contains an experimental,
unreleased Swift package, the `FrostfieldLab` example app, and a CI workflow for
FF-01–FF-08. [The task list](tasks/todo.md) records which acceptance criteria
have evidence; nothing has run on a physical iPhone or iPad, the CI workflow has
not run on GitHub, and no release exists. Everything after FF-08 is planned.

## Pick up here (2026-11-09)

The work lives on the branch `implement/milestone-1`. It is not merged into
`main`. FF-01–FF-08 are implemented, and an independent review's findings were
fixed on the branch. The last full runs:

- Package scheme on My Mac (Apple M5), with Metal API validation required:
  116 test functions, 208 cases, all passed.
- `FrostfieldLab` UI tests: 13 of 13 passed on the iPhone 17 simulator
  (iOS 27.2); a 6-test subset passed on the iPad Air 11-inch (M4) simulator.

Open work, in the order it matters:

1. **Physical-device evidence.** This includes the FF-02 triangle on a device,
   the FF-04 and FF-05 device comparisons, and the Checkpoint C
   recipe and budgets in `docs/validation.md`. Checkpoints A, B, and C stay
   unchecked until it exists.
2. **FF-03 criterion 3.** The host still gets drawables through
   `MTKView.currentDrawable`, which can wait on the main thread when every
   drawable is in flight. The bounded retry does not change that.
3. **Owner decisions** recorded in the task list:
   - whether `RenderCapabilities` stays public (FF-01 criterion 2);
   - which oldest and recent devices form the baseline (Checkpoint B);
   - whether to keep or restate the five-file slice rule.
4. **CI on GitHub.** `.github/workflows/ci.yml` runs on pull requests and on
   pushes to `main`, and has not run yet.
5. **Optional later milestones.** FF-09 onward has not started. FF-09 (one
   directional light) is the next task in order.

Tooling notes from this round:

- In Xcode MCP, open `Examples/Frostfield.xcworkspace` (package, lab, and
  shared package scheme). Do not open the package folder on its own first. The
  headless bridge can then keep a stale package root, and lab builds fail with
  "Missing package product".
- After switching schemes, set the run destination explicitly: My Mac for the
  package's GPU tests, an iPhone simulator for the lab. The destination
  otherwise tends to jump to "My Mac (Designed for iPad)", which needs signing.
- The package scheme sets `MTL_DEBUG_LAYER=1` and
  `FROSTFIELD_REQUIRE_API_VALIDATION=1`. The validation check fails, rather
  than skips, when validation is missing.
- Lab UI tests must launch through `makeApp()`, which adds
  `-FrostfieldLabUITesting`. `RenderSchedule.frameDidEnd(_:framesInFlight:)`
  needs the in-flight count of every renderer whose completions it receives.
- This is a public repository. Commit with the account's GitHub noreply
  address, and keep personal paths, team IDs, and device identifiers out of
  tracked files.

## Read in order

1. [AGENTS.md](AGENTS.md) — repository and Apple-tool requirements.
2. [Product and delivery plan](tasks/plan.md).
3. [Architecture and ownership](docs/architecture.md).
4. [Backend experiments](docs/research-spikes.md).
5. [Verification and performance protocol](docs/validation.md).
6. [Ordered implementation tasks](tasks/todo.md) — the only completion checklist.

Read the [language strategy](docs/language-strategy.md) before creating the package.
It specifies Swift CPU code, MSL shaders, a verified data-layout boundary, and gates
for adding C/C++ or Rust.

## Product intent

Build a Swift and Metal creative-graphics framework with programmable scenes,
procedural geometry, custom materials, lights, particles, animation, and
post-processing. The eventual user should be able to invent a visual, not only
select a preset. Keep the graphics API independent of SwiftUI and UIKit.
Native UI adapters host it without owning its render loop.

The companion [GlassWeave](https://github.com/Mitternacht/GlassWeave) repository
owns glass surfaces and interaction. Its integration proof is the first priority
for the combined effort. Neither repository initially depends on the other.
Frostfield work can run independently when a separate implementer is available.

## Start here

FF-01–FF-08 are done; continue from "Pick up here" above. The original
guidance for a fresh start follows.

Start with **FF-01** in the task list to establish the package and build baseline.
Then use **FF-02–FF-03** for the smallest runnable UIKit/Metal fixture and its
lifecycle, following each task's evidence requirements. Avoid creating the
entire proposed module tree, a universal renderer interface, or scene DSL first.
Split implementation into additional subtasks if a task exceeds a focused session.

The first useful Frostfield checkpoint is a real render host that behaves
correctly through resize, inactivity, and teardown, followed by recorded backend
choices. A full engine is not the first checkpoint. The later developer preview
must prove a composed scene and custom shader, not just an animated triangle.

The limited standalone v0.1 candidate is the programmable core through FF-08
plus Checkpoint C's clean-consumer and physical-device qualification. Later
effects and GlassWeave interoperability do not block that independent release.

## Decisions and latitude

| Fixed direction | Implementation freedom requiring evidence |
| --- | --- |
| Independent public repository; MIT license | Internal file names and target decomposition |
| Swift-first API, Metal rendering, native presentation | Metal 4 adoption and compatibility policy |
| UIKit/MTKView first, optional SwiftUI adapter | Custom renderer versus a bounded Satin-backed approach |
| Proposed first release targets iOS/iPadOS 26 | Which physical devices qualify and which quality tiers ship |
| No GlassWeave scene dependency; no premature shared core | Extraction of a small Metal product after two real consumers exist |

Platform and toolchain choices are recorded defaults, not a claim that they were
explicitly chosen by the user or already tested. Revisit a default with a written
reason if source evidence or measurements invalidate it. Keep Swift language
mode separate from the minimum compiler/toolchain version.

## Working contract

For Apple work, load the matching installed Xcode 27 skill and use the native
Xcode MCP server for documentation, project edits, builds, runs, diagnostics,
tests, and previews. Verify a working bridge before relying on build results.
Do not replace that workflow silently with terminal builds. A missing package
creation operation can be handled as a documented bootstrap limitation; regular
project editing and verification still use native Xcode tools.

Use short, reviewable slices. Test observable behavior, resource lifetimes, and
failure paths; do not add a suite that merely asserts a scaffold exists. Add CI
when there is code to build and meaningful tests to run. Keep API examples tied
to code that actually compiles. Do not add release tags or performance claims
before their acceptance gates pass.

At each checkpoint record the commit, Xcode/SDK versions, destination, build and
test results, visual evidence, and any missing physical-device evidence. Keep
failed checks visible. A simulator cannot satisfy a physical-device performance
gate. Update the task list only after its stated verification passes.

## Suggested Claude kickoff

> Read AGENTS.md and HANDOFF.md, then the linked plan, architecture, experiments,
> validation protocol, and task list. Implement Frostfield in dependency order,
> continuing from the first open item under "Pick up here". Use the required
> Apple skills and native Xcode MCP tools. Keep interfaces experimental until
> working fixtures establish them. Record evidence and update the task list as
> each slice passes. Preserve GlassWeave's independence and avoid expanding the
> first milestone into a complete engine.
