# Frostfield

An Apple-native creative graphics framework built around Swift and Metal.

Frostfield is planned as a programmable graphics toolkit for scenes, procedural
geometry, materials, lighting, particles, animation, and post-processing. Its rendering
core will be independent of SwiftUI, with native presentation adapters added where useful.

## Status

**Early implementation; no release.** Everything below is experimental API in an
unreleased package. It has been verified with GPU unit tests on a Mac and with the
[`Examples/`](Examples/README.md) lab app on the iOS Simulator, **not on a physical iPhone
or iPad**. The limited v0.1 candidate is not qualified: the physical-device measurements
and budgets it requires have not been taken
([v0.1 core qualification record](docs/evidence/v0.1-core-qualification.md)). Progress and
evidence are tracked in [the task list](tasks/todo.md).

## What works today

- A UIKit `MTKView` sample host with requested and continuous drawing, per-scene pause
  while hidden or in the background, and drawable size from each view's own bounds and
  display scale (`RenderSchedule`).
- Conventional Metal rendering with a bounded number of frames in flight that skips
  instead of waiting, typed construction errors, and shaders packaged in the
  framework's resource bundle (`TriangleRenderer`, `MeshRenderer`).
- `Transform`, a perspective `Camera`, and validated `PerspectiveProjection` under one
  documented [coordinate convention](docs/coordinates.md).
- Validated indexed geometry (`IndexedGeometry`), a procedural cube, depth testing, and
  back-face culling.
- Materials: a built-in unlit material and fragment shaders an app supplies from its
  own Metal library, under a documented [binding contract](docs/materials.md), with
  typed errors before anything is drawn and no pipeline creation while parameters
  animate.
- A runtime GPU capability survey (`RenderCapabilities`).

Failure behavior, thread and lifetime rules, and color handling are described in
[materials](docs/materials.md), [coordinates](docs/coordinates.md), and the
[architecture](docs/architecture.md). Physical-device performance is unknown.

## Roadmap (not implemented)

- Physical-device qualification of the v0.1 core: named devices, budgets, and traces.
- One directional light, one bounded particle emitter, and one post-process.
- A public offscreen texture-output contract and an optional SwiftUI adapter.
- Interoperability with GlassWeave (see [Companion project](#companion-project)).
- A Metal 4 backend, if measurements on devices justify it
  ([decision](docs/decisions/metal-backend.md)).

## Direction

- Swift scene and material APIs with a direct Metal shader escape hatch.
- A Metal renderer with explicit resource lifetimes and measured frame scheduling.
- A UIKit `MTKView` host first, followed by an optional SwiftUI adapter.
- Small reference scenes and physical-device measurements before broad feature expansion.

Across the two projects, GlassWeave's custom-glass integration proof has priority.
Backend choices are recorded with evidence: conventional Metal and a minimal renderer,
without Satin ([renderer foundation](docs/decisions/renderer-foundation.md)).

## Start here

- [Claude implementation handoff](HANDOFF.md)
- [Implementation plan](tasks/plan.md)
- [Ordered tasks and acceptance criteria](tasks/todo.md)
- [Architecture](docs/architecture.md)
- [Language strategy](docs/language-strategy.md)
- [Validation and performance protocol](docs/validation.md)
- [Contributing](CONTRIBUTING.md)

## Development baseline

| Area | Choice |
| --- | --- |
| Language | Swift 6 mode; Swift tools version 6.2; shaders in MSL |
| First UI target | iOS and iPadOS 26 or later (tested so far only on the iOS 27.2 Simulator) |
| macOS | macOS 26 core builds and GPU unit tests; a native macOS UI adapter is future work |
| Rendering backend | Conventional Metal command queue; Metal 4 is an experiment, not shipped |
| Local agent workflow | Xcode 27 skills and the native Xcode MCP server |
| CI | `macos-26` runner: package tests on macOS, lab build for the iOS Simulator (written; not yet run on GitHub) |
| Dependencies | None |

Recorded build, test, and qualification results live in [`docs/evidence/`](docs/evidence/).
The OS baseline is a project choice; GPU features are checked on the actual device at
run time, not inferred from the OS version.

## Companion project

[GlassWeave](https://github.com/Mitternacht/GlassWeave) is the independent companion
framework for native and custom glass surfaces. Neither package currently depends on
the other. Shared low-level Metal utilities will be extracted only after both
implementations establish a concrete need; glass must not require a scene engine.

## License

[MIT](LICENSE).
