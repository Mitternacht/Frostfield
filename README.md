# Frostfield

An Apple-native creative graphics framework built around Swift and Metal.

Frostfield is planned as a programmable graphics toolkit for scenes, procedural
geometry, materials, lighting, particles, animation, and post-processing. Its rendering
core will be independent of SwiftUI, with native presentation adapters added where useful.

## Status

**Early implementation; no release.** The package currently contains a GPU capability
survey and an experimental diagnostic-triangle renderer with focused tests, and
[`Examples/`](Examples/README.md) holds a UIKit `MTKView` lab app. These are rendering
fixtures, not a scene API; physical-device behavior has not been verified. All framework
capabilities described below remain planned work. Progress and evidence are tracked in
[the task list](tasks/todo.md).

## Direction

- Swift scene and material APIs with a direct Metal shader escape hatch.
- A Metal renderer with explicit resource lifetimes and measured frame scheduling.
- A UIKit `MTKView` host first, followed by an optional SwiftUI adapter.
- Small reference scenes and physical-device measurements before broad feature expansion.

The first graphics milestone is a programmable scene rendered in a native host. Across
the two projects, GlassWeave's custom-glass integration proof has priority. Backend
choices, including Metal 4 and a bounded Satin comparison, remain evidence-driven
experiments rather than selected project dependencies.

## Start here

- [Claude implementation handoff](HANDOFF.md)
- [Implementation plan](tasks/plan.md)
- [Ordered tasks and acceptance criteria](tasks/todo.md)
- [Architecture](docs/architecture.md)
- [Language strategy](docs/language-strategy.md)
- [Validation and performance protocol](docs/validation.md)
- [Contributing](CONTRIBUTING.md)

## Proposed development baseline

| Area | Initial choice |
| --- | --- |
| Language | Swift 6 mode; Swift tools version 6.2 |
| First UI target | iOS and iPadOS 26 or later |
| macOS | macOS 26 core build checks; a native macOS UI adapter is future work |
| Local agent workflow | Xcode 27 skills and the native Xcode MCP server |
| Future CI | macOS 26 runner; macOS Debug/Release and iOS Simulator Release builds |
| Dependencies | None selected; keep the first implementation independent |

Start implementation from [HANDOFF.md](HANDOFF.md). The first implementation task
will create the Swift package and establish native Xcode builds. Add focused tests
with testable behavior and a runnable reference example with the rendering milestone.
Recorded build and test results live in [`docs/evidence/`](docs/evidence/).

The OS baseline is a project choice. Metal 4 capability will be checked independently
of OS availability when a backend is implemented.

## Companion project

[GlassWeave](https://github.com/Mitternacht/GlassWeave) is the independent companion
framework for native and custom glass surfaces. Neither package currently depends on
the other. Shared low-level Metal utilities will be extracted only after both
implementations establish a concrete need; glass must not require a scene engine.

## License

[MIT](LICENSE).
